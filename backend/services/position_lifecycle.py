"""Pure transition validator for the position/order lifecycle contract.

Spec: `docs/specs/2026-09-26-position-lifecycle-contract.md`.

Two separate state machines — an order is an instruction with a terminal outcome,
a position is exposure that exists until something closes it. A terminal order
state is *evidence* the position machine consumes, never a position state itself.
Collapsing them yields the claim that a rejected exit means no position exists,
when the position is still fully held and still at risk.

No database, no clock, no network. This module answers only whether a transition
is legal and adequately evidenced — never what to *do* in a state. Its tests
therefore demonstrate that the rules hold as written and are **not** evidence of
end-to-end execution correctness (spec §9).
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Mapping, Optional

# Identifiers that look like values but carry no information. Accepting any of
# these is the `order_id="unknown"` defect the contract exists to eliminate.
_PLACEHOLDER_IDS = (None, "", "unknown", "none", "null")


class IllegalTransition(Exception):
    """The transition is not permitted from this state, regardless of evidence."""


class InsufficientEvidence(Exception):
    """The transition is permitted but the evidence required for it is absent."""


class OrderState(Enum):
    INTENT_CREATED = "INTENT_CREATED"
    SUBMITTING = "SUBMITTING"
    ACCEPTED = "ACCEPTED"
    WORKING_PARTIAL = "WORKING_PARTIAL"
    FILLED = "FILLED"
    SETTLED_PARTIAL = "SETTLED_PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL_ORDER_STATES


_TERMINAL_ORDER_STATES = frozenset(
    {
        OrderState.FILLED,
        OrderState.SETTLED_PARTIAL,
        OrderState.CANCELLED,
        OrderState.REJECTED,
    }
)


class PositionState(Enum):
    FLAT = "FLAT"
    OPENING = "OPENING"
    OPEN = "OPEN"
    OPEN_WITH_RESIDUAL = "OPEN_WITH_RESIDUAL"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


_HELD_STATES = frozenset({PositionState.OPEN, PositionState.OPEN_WITH_RESIDUAL})

# Evidence required to create an intent, per spec §4. Exchange identifiers are
# deliberately absent: they are nullable until observed (spec I8).
_INTENT_FIELDS = (
    "client_order_id",
    "product_id",
    "side",
    "intended_size",
    "execution_mode",
    "run_id",
    "config_version",
    "model_hash",
    "created_at",
)


def _require(evidence: Mapping[str, Any], *fields: str) -> None:
    missing = [f for f in fields if f not in evidence or evidence[f] is None]
    if missing:
        raise InsufficientEvidence(f"missing required evidence: {', '.join(missing)}")


def _require_identifier(evidence: Mapping[str, Any], field: str) -> None:
    value = evidence.get(field)
    if isinstance(value, str):
        value = value.strip().lower()
    if value in _PLACEHOLDER_IDS:
        raise InsufficientEvidence(
            f"{field} must identify the order; {evidence.get(field)!r} is a placeholder"
        )


def _require_decimal(evidence: Mapping[str, Any], field: str) -> Decimal:
    _require(evidence, field)
    value = evidence[field]
    if not isinstance(value, Decimal):
        raise InsufficientEvidence(
            f"{field} must be a Decimal for increment-exact comparison, got {type(value).__name__}"
        )
    return value


# ── Order machine ─────────────────────────────────────────────────────────────


def next_order_state(
    current: Optional[OrderState],
    event: str,
    evidence: Mapping[str, Any],
) -> OrderState:
    """Next order state, or raise.

    `current` is None only for `intent_created`. Terminal states accept no
    events: an order that has finished cannot un-finish.
    """
    if current is not None and current.is_terminal:
        raise IllegalTransition(f"{current.value} is terminal; refusing event {event!r}")

    if event == "state_unknown":
        # Always reachable: not knowing is a fact about us, not about the order.
        return OrderState.UNKNOWN

    if event == "intent_created":
        if current is not None:
            raise IllegalTransition(f"intent_created is only valid from nothing, not {current}")
        _require(evidence, *_INTENT_FIELDS)
        _require_identifier(evidence, "client_order_id")
        return OrderState.INTENT_CREATED

    if current is None:
        raise IllegalTransition(f"event {event!r} requires an existing order")

    if event == "submitted":
        if current is not OrderState.INTENT_CREATED:
            raise IllegalTransition(f"submitted is only valid from INTENT_CREATED, not {current}")
        _require(evidence, "submitted_at")
        return OrderState.SUBMITTING

    if event == "accepted":
        if current is not OrderState.SUBMITTING:
            raise IllegalTransition(f"accepted is only valid from SUBMITTING, not {current}")
        _require(evidence, "accepted_at")
        _require_identifier(evidence, "order_id")
        return OrderState.ACCEPTED

    if event == "cancel_acknowledged":
        # Spec §4: an acknowledgement is not a terminal confirmation. It changes
        # nothing about exposure and grants no permission to replace the order.
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"cancel_acknowledged is not valid from {current}")
        return current

    if event == "rejected":
        if current not in (OrderState.SUBMITTING, OrderState.ACCEPTED):
            raise IllegalTransition(f"rejected is not valid from {current}")
        _require(evidence, "terminal_status", "reason", "observed_at")
        return OrderState.REJECTED

    if event == "cancel_confirmed":
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"cancel_confirmed is not valid from {current}")
        _require(evidence, "terminal_status", "observed_at")
        filled_size = _require_decimal(evidence, "filled_size")
        filled_value = _require_decimal(evidence, "filled_value")
        if filled_size > 0 or filled_value > 0:
            raise IllegalTransition(
                "CANCELLED requires zero fills; a cancel with fills is SETTLED_PARTIAL"
            )
        return OrderState.CANCELLED

    if event == "fill_observed":
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"fill_observed is not valid from {current}")
        _require(evidence, "avg_fill_price", "fill_ids", "observed_at")
        _require_identifier(evidence, "order_id")
        filled_size = _require_decimal(evidence, "filled_size")
        intended = _require_decimal(evidence, "intended_size")
        if filled_size >= intended:
            _require(evidence, "terminal_status")
            return OrderState.FILLED
        # Partial. Non-terminal only while the remainder is provably live —
        # requiring a terminal status here would contradict WORKING_PARTIAL's
        # own definition (spec §4).
        _require(evidence, "remainder_live", "remaining_size")
        if evidence["remainder_live"]:
            return OrderState.WORKING_PARTIAL
        return OrderState.SETTLED_PARTIAL

    if event == "remainder_terminal":
        if current is not OrderState.WORKING_PARTIAL:
            raise IllegalTransition("remainder_terminal is only valid from WORKING_PARTIAL")
        _require(evidence, "terminal_status", "fill_ids", "observed_at")
        _require_decimal(evidence, "filled_size")
        if evidence.get("remainder_live"):
            raise InsufficientEvidence("remainder_terminal requires the remainder to be gone")
        return OrderState.SETTLED_PARTIAL

    raise IllegalTransition(f"unknown order event {event!r}")


# ── Position machine ──────────────────────────────────────────────────────────


def _is_flat(evidence: Mapping[str, Any]) -> bool:
    """Flat means zero exposure at the product's increment, not float-zero."""
    size = _require_decimal(evidence, "position_size")
    increment = _require_decimal(evidence, "size_increment")
    return abs(size) < increment


def next_position_state(
    current: PositionState,
    event: str,
    evidence: Mapping[str, Any],
) -> PositionState:
    """Next position state, or raise.

    Exposure is the subject here. A failed *order* never means absent
    *exposure* — see `exit_failed`.
    """
    if event == "disagreement_detected":
        # Always reachable from every state (spec §3).
        _require(evidence, "local", "exchange")
        return PositionState.RECONCILIATION_REQUIRED

    if current is PositionState.RECONCILIATION_REQUIRED:
        # Spec I5: only the reconciler clears this, never a retry.
        if event == "reconciler_confirmed":
            if not evidence.get("exchange_terminal"):
                raise InsufficientEvidence(
                    "deterministic confirmation requires terminal exchange evidence; "
                    "an ambiguous case needs an authorised override"
                )
            return PositionState.CLOSED if _is_flat(evidence) else PositionState.OPEN_WITH_RESIDUAL
        if event == "override":
            _require(evidence, "authorised_by", "authorisation_reason")
            return PositionState.CLOSED if _is_flat(evidence) else PositionState.OPEN_WITH_RESIDUAL
        raise IllegalTransition(
            f"{event!r} cannot clear RECONCILIATION_REQUIRED; only the reconciler can"
        )

    if event == "entry_submitted":
        if current is not PositionState.FLAT:
            raise IllegalTransition(f"entry_submitted is only valid from FLAT, not {current}")
        return PositionState.OPENING

    if event == "entry_filled":
        if current is not PositionState.OPENING:
            raise IllegalTransition(f"entry_filled is only valid from OPENING, not {current}")
        size = _require_decimal(evidence, "position_size")
        intended = _require_decimal(evidence, "intended_size")
        _require_decimal(evidence, "size_increment")
        if _is_flat(evidence):
            return PositionState.FLAT
        return PositionState.OPEN if size >= intended else PositionState.OPEN_WITH_RESIDUAL

    if event == "entry_failed":
        if current is not PositionState.OPENING:
            raise IllegalTransition(f"entry_failed is only valid from OPENING, not {current}")
        _require(evidence, "reason")
        return PositionState.FLAT

    if event == "exit_submitted":
        if current not in _HELD_STATES:
            raise IllegalTransition(f"exit_submitted requires held exposure, not {current}")
        return PositionState.CLOSING

    if event == "exit_filled":
        if current is not PositionState.CLOSING:
            raise IllegalTransition(f"exit_filled is only valid from CLOSING, not {current}")
        _require_decimal(evidence, "position_size")
        _require_decimal(evidence, "size_increment")
        return PositionState.CLOSED if _is_flat(evidence) else PositionState.OPEN_WITH_RESIDUAL

    if event == "exit_failed":
        # Finding 3, and the most important rule here: the position is STILL
        # HELD. A failed exit order never means absent exposure.
        if current is not PositionState.CLOSING:
            raise IllegalTransition(f"exit_failed is only valid from CLOSING, not {current}")
        _require(evidence, "reason")
        prior = evidence.get("prior_state")
        if prior is PositionState.OPEN_WITH_RESIDUAL:
            return PositionState.OPEN_WITH_RESIDUAL
        return PositionState.OPEN

    raise IllegalTransition(f"unknown position event {event!r}")


def describe() -> Dict[str, Any]:
    """Machine-readable summary of the contract, for diagnostics and docs."""
    return {
        "order_states": [s.value for s in OrderState],
        "terminal_order_states": sorted(s.value for s in _TERMINAL_ORDER_STATES),
        "position_states": [s.value for s in PositionState],
        "held_position_states": sorted(s.value for s in _HELD_STATES),
        "spec": "docs/specs/2026-09-26-position-lifecycle-contract.md",
    }
