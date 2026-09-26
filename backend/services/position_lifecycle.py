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

## Validity, not presence

The first implementation (3bb8aa1) checked that evidence keys were PRESENT and
accepted whatever they contained. Driven adversarially it accepted a cancellation
with `filled_size=-1`, an `order_id` of `123`, a "settled" partial with a 0.6
remainder still outstanding, an entry failure that erased a known position, an
"infinite" size increment that made every position look flat, and the string
`"false"` as a boolean. That is the accepted-versus-confirmed mistake this
contract exists to prevent, committed inside the module written to prevent it.

Every helper below therefore validates the *content*: identifiers must be
non-empty non-placeholder strings, quantities must be finite Decimals with the
right sign, booleans must be actual booleans rather than truthy values, terminal
statuses must be drawn from an allowed set, and quantities must be mutually
consistent. Absence of proof is never proof.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional

# Strings that look like identifiers but carry no information. Accepting any of
# these is the `order_id="unknown"` defect the contract exists to eliminate.
_PLACEHOLDER_IDS = frozenset({"", "unknown", "none", "null", "nil", "n/a", "-"})

# Terminal statuses that constitute proof an order left the book without fills.
_CANCEL_TERMINALS = frozenset({"CANCELLED", "CANCELED", "EXPIRED"})
_FILL_TERMINALS = frozenset({"FILLED", "DONE", "CLOSED"})
_REJECT_TERMINALS = frozenset({"REJECTED", "FAILED"})
_ALL_TERMINALS = _CANCEL_TERMINALS | _FILL_TERMINALS | _REJECT_TERMINALS

# Failure reasons that mean "we do not know", not "nothing happened". A timeout
# is the canonical case: the order may well exist.
_AMBIGUOUS_REASONS = frozenset(
    {"timeout", "unknown", "network_error", "no_response", "connection_error"}
)


class IllegalTransition(Exception):
    """The transition is not permitted from this state, regardless of evidence."""


class InsufficientEvidence(Exception):
    """The transition is permitted but the evidence required for it is absent
    or does not prove what it claims."""


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


# ── Evidence validators ───────────────────────────────────────────────────────


def _require(evidence: Mapping[str, Any], *fields: str) -> None:
    missing = [f for f in fields if f not in evidence or evidence[f] is None]
    if missing:
        raise InsufficientEvidence(f"missing required evidence: {', '.join(missing)}")


def _require_str_id(evidence: Mapping[str, Any], field: str) -> str:
    value = evidence.get(field)
    # bool is a subclass of int, and neither is an identifier.
    if not isinstance(value, str):
        raise InsufficientEvidence(
            f"{field} must be a string identifier, got {type(value).__name__}"
        )
    cleaned = value.strip()
    if cleaned.lower() in _PLACEHOLDER_IDS:
        raise InsufficientEvidence(
            f"{field} must identify the order; {value!r} is blank or a placeholder"
        )
    return cleaned


def _require_nonempty_str(evidence: Mapping[str, Any], field: str) -> str:
    value = evidence.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InsufficientEvidence(f"{field} must be a non-empty string")
    return value.strip()


def _require_id_list(evidence: Mapping[str, Any], field: str) -> list:
    value = evidence.get(field)
    if not isinstance(value, (list, tuple)) or not value:
        raise InsufficientEvidence(f"{field} must be a non-empty list of identifiers")
    out = []
    for item in value:
        if not isinstance(item, str) or item.strip().lower() in _PLACEHOLDER_IDS:
            raise InsufficientEvidence(f"{field} contains a blank or placeholder identifier")
        out.append(item.strip())
    return out


def _require_finite_decimal(evidence: Mapping[str, Any], field: str) -> Decimal:
    _require(evidence, field)
    value = evidence[field]
    if not isinstance(value, Decimal):
        raise InsufficientEvidence(
            f"{field} must be a Decimal for increment-exact comparison, got {type(value).__name__}"
        )
    if not value.is_finite():
        raise InsufficientEvidence(f"{field} must be finite, got {value}")
    return value


def _require_nonneg(evidence: Mapping[str, Any], field: str) -> Decimal:
    value = _require_finite_decimal(evidence, field)
    if value < 0:
        raise InsufficientEvidence(f"{field} must not be negative, got {value}")
    return value


def _require_positive(evidence: Mapping[str, Any], field: str) -> Decimal:
    value = _require_finite_decimal(evidence, field)
    if value <= 0:
        raise InsufficientEvidence(f"{field} must be strictly positive, got {value}")
    return value


def _require_exact_zero(evidence: Mapping[str, Any], field: str) -> Decimal:
    value = _require_finite_decimal(evidence, field)
    if value != 0:
        raise InsufficientEvidence(f"{field} must be exactly zero, got {value}")
    return value


def _require_strict_bool(evidence: Mapping[str, Any], field: str) -> bool:
    value = evidence.get(field)
    if type(value) is not bool:  # noqa: E721 — truthiness is the bug being excluded
        raise InsufficientEvidence(
            f"{field} must be a bool, not {type(value).__name__} {value!r}; "
            f"truthiness is not evidence"
        )
    return value


def _require_terminal_status(evidence: Mapping[str, Any], allowed: Iterable[str]) -> str:
    status = _require_nonempty_str(evidence, "terminal_status").upper()
    allowed = frozenset(allowed)
    if status not in allowed:
        raise InsufficientEvidence(f"terminal_status {status!r} is not one of {sorted(allowed)}")
    return status


def _require_linked_order(evidence: Mapping[str, Any]) -> None:
    """A position transition must cite the order and fills that caused it.

    Position size alone is not evidence: it says what we believe, not why.
    """
    _require_str_id(evidence, "order_id")
    _require_id_list(evidence, "fill_ids")


# ── Order machine ─────────────────────────────────────────────────────────────


def _classify_fill(current: OrderState, evidence: Mapping[str, Any]) -> OrderState:
    """Shared by fill_observed and the UNKNOWN resolution path."""
    _require(evidence, "observed_at")
    _require_str_id(evidence, "order_id")
    _require_id_list(evidence, "fill_ids")
    _require_positive(evidence, "avg_fill_price")
    filled = _require_positive(evidence, "filled_size")  # a zero fill is not a fill
    intended = _require_positive(evidence, "intended_size")

    if filled > intended:
        raise IllegalTransition(
            f"filled_size {filled} exceeds intended_size {intended}; "
            f"an overfill is a reconciliation case, not a clean fill"
        )

    if filled == intended:
        _require_terminal_status(evidence, _FILL_TERMINALS)
        return OrderState.FILLED

    # Partial.
    remainder_live = _require_strict_bool(evidence, "remainder_live")
    remaining = _require_nonneg(evidence, "remaining_size")
    if filled + remaining > intended:
        raise IllegalTransition(
            f"filled_size {filled} plus remaining_size {remaining} exceeds intended_size {intended}"
        )

    if remainder_live:
        if remaining <= 0:
            raise InsufficientEvidence("a live remainder must have a positive remaining_size")
        return OrderState.WORKING_PARTIAL

    # Not live: that claim needs terminal proof, and nothing may still be working.
    _require_terminal_status(evidence, _CANCEL_TERMINALS | _FILL_TERMINALS)
    _require_exact_zero(evidence, "remaining_size")
    return OrderState.SETTLED_PARTIAL


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

    if current is OrderState.UNKNOWN:
        # The only way out of UNKNOWN is affirmative terminal evidence. An
        # ordinary event cannot resolve it, because we do not know what happened.
        if event != "state_resolved":
            raise IllegalTransition(
                f"UNKNOWN is only cleared by state_resolved with terminal evidence, "
                f"not by {event!r}"
            )
        status = _require_terminal_status(evidence, _ALL_TERMINALS)
        if status in _REJECT_TERMINALS:
            _require(evidence, "reason", "observed_at")
            return OrderState.REJECTED
        if status in _FILL_TERMINALS:
            return _classify_fill(current, evidence)
        # Cancel family: zero fills means CANCELLED, otherwise it settled partial.
        _require(evidence, "observed_at")
        _require_str_id(evidence, "order_id")
        filled = _require_nonneg(evidence, "filled_size")
        if filled == 0:
            _require_exact_zero(evidence, "filled_value")
            return OrderState.CANCELLED
        return _classify_fill(current, evidence)

    if event == "intent_created":
        if current is not None:
            raise IllegalTransition(f"intent_created is only valid from nothing, not {current}")
        _require(evidence, *_INTENT_FIELDS)
        _require_str_id(evidence, "client_order_id")
        _require_positive(evidence, "intended_size")
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
        _require_str_id(evidence, "order_id")
        return OrderState.ACCEPTED

    if event == "cancel_acknowledged":
        # An acknowledgement is not a terminal confirmation. It changes nothing
        # about exposure and grants no permission to replace the order (spec §4).
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"cancel_acknowledged is not valid from {current}")
        return current

    if event == "rejected":
        if current not in (OrderState.SUBMITTING, OrderState.ACCEPTED):
            raise IllegalTransition(f"rejected is not valid from {current}")
        _require(evidence, "reason", "observed_at")
        _require_terminal_status(evidence, _REJECT_TERMINALS)
        return OrderState.REJECTED

    if event == "cancel_confirmed":
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"cancel_confirmed is not valid from {current}")
        _require(evidence, "observed_at")
        _require_str_id(evidence, "order_id")
        _require_terminal_status(evidence, _CANCEL_TERMINALS)
        # Exactly zero, not merely "not positive": negative is not evidence of
        # no fills, it is evidence of a corrupt record.
        _require_exact_zero(evidence, "filled_size")
        _require_exact_zero(evidence, "filled_value")
        return OrderState.CANCELLED

    if event == "fill_observed":
        if current not in (OrderState.ACCEPTED, OrderState.WORKING_PARTIAL):
            raise IllegalTransition(f"fill_observed is not valid from {current}")
        return _classify_fill(current, evidence)

    if event == "remainder_terminal":
        if current is not OrderState.WORKING_PARTIAL:
            raise IllegalTransition("remainder_terminal is only valid from WORKING_PARTIAL")
        _require(evidence, "observed_at")
        _require_str_id(evidence, "order_id")
        _require_id_list(evidence, "fill_ids")
        _require_terminal_status(evidence, _CANCEL_TERMINALS | _FILL_TERMINALS)
        _require_nonneg(evidence, "filled_size")
        _require_exact_zero(evidence, "remaining_size")
        if _require_strict_bool(evidence, "remainder_live"):
            raise InsufficientEvidence("remainder_terminal requires the remainder to be gone")
        return OrderState.SETTLED_PARTIAL

    raise IllegalTransition(f"unknown order event {event!r}")


# ── Position machine ──────────────────────────────────────────────────────────


def _is_flat(evidence: Mapping[str, Any]) -> bool:
    """Flat means zero exposure at the product's increment, not float-zero.

    The increment must be positive and finite: an "infinite" increment made every
    position compare as flat in the first implementation.
    """
    size = _require_nonneg(evidence, "position_size")
    increment = _require_positive(evidence, "size_increment")
    return size < increment


def _settled_position(evidence: Mapping[str, Any]) -> PositionState:
    return PositionState.CLOSED if _is_flat(evidence) else PositionState.OPEN_WITH_RESIDUAL


def next_position_state(
    current: PositionState,
    event: str,
    evidence: Mapping[str, Any],
) -> PositionState:
    """Next position state, or raise.

    Exposure is the subject here. A failed *order* never means absent
    *exposure* — see `entry_failed` and `exit_failed`.
    """
    if event == "disagreement_detected":
        # Always reachable from every state (spec §3).
        _require(evidence, "local", "exchange")
        return PositionState.RECONCILIATION_REQUIRED

    if current is PositionState.RECONCILIATION_REQUIRED:
        # Spec I5: only the reconciler clears this, never a retry.
        if event == "reconciler_confirmed":
            if _require_strict_bool(evidence, "exchange_terminal") is not True:
                raise InsufficientEvidence(
                    "deterministic confirmation requires exchange_terminal True; "
                    "an ambiguous case needs an authorised override"
                )
            _require_linked_order(evidence)
            return _settled_position(evidence)
        if event == "override":
            _require_nonempty_str(evidence, "authorised_by")
            _require_nonempty_str(evidence, "authorisation_reason")
            _require_linked_order(evidence)
            return _settled_position(evidence)
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
        _require_linked_order(evidence)
        size = _require_nonneg(evidence, "position_size")
        intended = _require_positive(evidence, "intended_size")
        if _is_flat(evidence):
            return PositionState.FLAT
        return PositionState.OPEN if size >= intended else PositionState.OPEN_WITH_RESIDUAL

    if event == "entry_failed":
        # A failed entry may still have created exposure. Reaching FLAT requires
        # affirmative proof of zero fill AND zero position; anything else, and
        # every ambiguous reason such as a timeout, is a reconciliation case.
        if current is not PositionState.OPENING:
            raise IllegalTransition(f"entry_failed is only valid from OPENING, not {current}")
        reason = _require_nonempty_str(evidence, "reason")
        if reason.strip().lower() in _AMBIGUOUS_REASONS:
            return PositionState.RECONCILIATION_REQUIRED
        try:
            _require_terminal_status(evidence, _REJECT_TERMINALS | _CANCEL_TERMINALS)
            _require_str_id(evidence, "order_id")
            _require_exact_zero(evidence, "filled_size")
            if not _is_flat(evidence):
                return PositionState.RECONCILIATION_REQUIRED
        except InsufficientEvidence:
            # No affirmative proof that nothing was filled. Do not erase exposure.
            return PositionState.RECONCILIATION_REQUIRED
        return PositionState.FLAT

    if event == "exit_submitted":
        if current not in _HELD_STATES:
            raise IllegalTransition(f"exit_submitted requires held exposure, not {current}")
        return PositionState.CLOSING

    if event == "exit_filled":
        if current is not PositionState.CLOSING:
            raise IllegalTransition(f"exit_filled is only valid from CLOSING, not {current}")
        _require_linked_order(evidence)
        return _settled_position(evidence)

    if event == "exit_failed":
        # Finding 3, and the most important rule here: the position is STILL
        # HELD. A failed exit order never means absent exposure. An ambiguous
        # failure is worse than a rejection, because the exit may have partially
        # filled — that is a reconciliation case, not a return to OPEN.
        if current is not PositionState.CLOSING:
            raise IllegalTransition(f"exit_failed is only valid from CLOSING, not {current}")
        reason = _require_nonempty_str(evidence, "reason")
        if reason.strip().lower() in _AMBIGUOUS_REASONS:
            return PositionState.RECONCILIATION_REQUIRED
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
        "cancel_terminals": sorted(_CANCEL_TERMINALS),
        "fill_terminals": sorted(_FILL_TERMINALS),
        "reject_terminals": sorted(_REJECT_TERMINALS),
        "ambiguous_reasons": sorted(_AMBIGUOUS_REASONS),
        "spec": "docs/specs/2026-09-26-position-lifecycle-contract.md",
    }
