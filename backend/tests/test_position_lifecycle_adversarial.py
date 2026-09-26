"""Adversarial suite for the lifecycle validator.

Every case here was REPRODUCED against the first implementation (3bb8aa1) and
**accepted** by it. The root cause was uniform: the validator checked that
evidence was PRESENT rather than that it PROVED anything — the
accepted-versus-confirmed mistake the contract exists to prevent, committed
inside the module written to prevent it.

Found by the parallel Codex session by importing the module and driving it
directly with hostile inputs. Kept as its own named suite so the class of defect
stays visible rather than being absorbed into the happy-path file.

Spec: docs/specs/2026-09-26-position-lifecycle-contract.md
"""

import os
import sys
from decimal import Decimal

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import pytest  # noqa: E402

from services.position_lifecycle import (  # noqa: E402
    IllegalTransition,
    InsufficientEvidence,
    OrderState,
    PositionState,
    next_order_state,
    next_position_state,
)

_REJECTED = (IllegalTransition, InsufficientEvidence)
_INF = Decimal("Infinity")
_NAN = Decimal("NaN")

_INTENT = {
    "client_order_id": "cid-1",
    "product_id": "BTC-USD",
    "side": "BUY",
    "intended_size": Decimal("1.0"),
    "execution_mode": "live",
    "run_id": "run-1",
    "config_version": "cfg-1",
    "model_hash": "abc123",
    "created_at": 1_800_000_000,
}
_ACCEPTED = {**_INTENT, "submitted_at": 1, "order_id": "ex-1", "accepted_at": 2}
_FILL = {
    **_ACCEPTED,
    "terminal_status": "FILLED",
    "filled_size": Decimal("1.0"),
    "avg_fill_price": Decimal("100.0"),
    "fee": Decimal("0.6"),
    "fill_ids": ["f-1"],
    "observed_at": 3,
}


def _pos(size, intended=Decimal("1.0"), increment=Decimal("0.00000001")):
    return {
        "position_size": size,
        "intended_size": intended,
        "size_increment": increment,
        "order_id": "ex-1",
        "fill_ids": ["f-1"],
        "observed_at": 1,
    }


# ── Blocker 1: cancel_confirmed took a non-terminal status and negatives ─────


def test_cancel_confirmed_rejects_a_non_cancel_terminal_status():
    ev = {
        **_ACCEPTED,
        "terminal_status": "OPEN",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev)


def test_cancel_confirmed_rejects_negative_amounts():
    """Negative is not zero. A `> 0` guard let -1 through as "no fills"."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("-1"),
        "filled_value": Decimal("-10"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev)


@pytest.mark.parametrize("bad", [_INF, -_INF, _NAN])
def test_cancel_confirmed_rejects_nonfinite_amounts(bad):
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": bad,
        "filled_value": Decimal("0"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev)


# ── Blocker 2: identifiers were not required to be strings ───────────────────


@pytest.mark.parametrize("bad", [123, 0, 1.5, b"ex-1", ["ex-1"], {"id": "ex-1"}, True])
def test_accepted_rejects_non_string_identifiers(bad):
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.SUBMITTING, "accepted", {**_ACCEPTED, "order_id": bad})


@pytest.mark.parametrize("bad", ["   ", "\t", "UNKNOWN", "None", "null", "unknown"])
def test_accepted_rejects_blank_and_placeholder_strings(bad):
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.SUBMITTING, "accepted", {**_ACCEPTED, "order_id": bad})


# ── Blocker 3: partial fills ─────────────────────────────────────────────────


def test_partial_fill_rejects_empty_fill_ids():
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": [],
        "remainder_live": True,
        "remaining_size": Decimal("0.6"),
        "observed_at": 1,
    }
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_settled_partial_requires_terminal_proof_not_just_a_false_flag():
    """`remainder_live=False` alone is not evidence the remainder is gone."""
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": False,
        "remaining_size": Decimal("0.6"),
        "observed_at": 1,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_partial_fill_rejects_inconsistent_remaining_quantity():
    """filled + remaining may never exceed intended."""
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": True,
        "remaining_size": Decimal("5"),
        "observed_at": 1,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_working_partial_requires_a_positive_remaining_size():
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": True,
        "remaining_size": Decimal("0"),
        "observed_at": 1,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


@pytest.mark.parametrize("bad", ["false", "true", 0, 1, "", None])
def test_remainder_live_must_be_a_strict_bool(bad):
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": bad,
        "remaining_size": Decimal("0.6"),
        "observed_at": 1,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_overfill_is_not_a_clean_fill():
    with pytest.raises(_REJECTED):
        next_order_state(
            OrderState.ACCEPTED, "fill_observed", {**_FILL, "filled_size": Decimal("2.0")}
        )


def test_zero_fill_is_not_a_fill():
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": True,
        "remaining_size": Decimal("1"),
        "observed_at": 1,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_filled_requires_an_allowed_terminal_status():
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", {**_FILL, "terminal_status": "OPEN"})


@pytest.mark.parametrize("bad", [_INF, _NAN, Decimal("-1")])
def test_fill_rejects_nonfinite_or_negative_price(bad):
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", {**_FILL, "avg_fill_price": bad})


# ── Blocker 4: entry_failed erased exposure that may exist ───────────────────


def test_entry_failed_cannot_erase_known_exposure():
    """The defect I guarded on the exit side and left open on the entry side."""
    ev = {"reason": "timeout", "position_size": Decimal("1")}
    assert (
        next_position_state(PositionState.OPENING, "entry_failed", ev)
        is PositionState.RECONCILIATION_REQUIRED
    )


def test_entry_failed_on_a_timeout_is_reconciliation_not_flat():
    assert (
        next_position_state(PositionState.OPENING, "entry_failed", {"reason": "timeout"})
        is PositionState.RECONCILIATION_REQUIRED
    )


def test_entry_failed_reaches_flat_only_with_affirmative_zero_proof():
    ev = {
        "reason": "REJECTED",
        "terminal_status": "REJECTED",
        "filled_size": Decimal("0"),
        "position_size": Decimal("0"),
        "size_increment": Decimal("0.00000001"),
        "order_id": "ex-1",
    }
    assert next_position_state(PositionState.OPENING, "entry_failed", ev) is PositionState.FLAT


# ── Blocker 5: non-finite sizes and increments ───────────────────────────────


def test_exit_filled_rejects_an_infinite_increment():
    """`abs(100) < Infinity` made a 100-unit position "flat"."""
    with pytest.raises(_REJECTED):
        next_position_state(
            PositionState.CLOSING, "exit_filled", {**_pos(Decimal("100")), "size_increment": _INF}
        )


@pytest.mark.parametrize("bad", [Decimal("0"), Decimal("-1"), _NAN, _INF])
def test_exit_filled_rejects_nonpositive_or_nonfinite_increment(bad):
    with pytest.raises(_REJECTED):
        next_position_state(
            PositionState.CLOSING, "exit_filled", {**_pos(Decimal("0")), "size_increment": bad}
        )


@pytest.mark.parametrize("bad", [_INF, _NAN, Decimal("-5")])
def test_exit_filled_rejects_nonfinite_or_negative_position_size(bad):
    with pytest.raises(_REJECTED):
        next_position_state(
            PositionState.CLOSING, "exit_filled", {**_pos(Decimal("0")), "position_size": bad}
        )


# ── Blocker 6: truthiness accepted where a boolean was meant ─────────────────


@pytest.mark.parametrize("bad", ["false", "true", "yes", 1, 0, "", None, "True"])
def test_reconciler_confirmed_requires_a_strict_bool(bad):
    with pytest.raises(InsufficientEvidence):
        next_position_state(
            PositionState.RECONCILIATION_REQUIRED,
            "reconciler_confirmed",
            {**_pos(Decimal("0")), "exchange_terminal": bad},
        )


def test_reconciler_confirmed_requires_linked_order_evidence():
    """Position size alone is not evidence; it must be tied to an order and fills."""
    ev = {
        "position_size": Decimal("0"),
        "size_increment": Decimal("0.01"),
        "exchange_terminal": True,
    }
    with pytest.raises(InsufficientEvidence):
        next_position_state(PositionState.RECONCILIATION_REQUIRED, "reconciler_confirmed", ev)


@pytest.mark.parametrize("field", ["authorised_by", "authorisation_reason"])
def test_override_rejects_blank_authorisation(field):
    ev = {
        **_pos(Decimal("0")),
        "authorised_by": "operator",
        "authorisation_reason": "manual audit",
        field: "   ",
    }
    with pytest.raises(InsufficientEvidence):
        next_position_state(PositionState.RECONCILIATION_REQUIRED, "override", ev)


# ── UNKNOWN had no recovery path at all ──────────────────────────────────────


def test_unknown_can_be_resolved_with_terminal_evidence():
    assert next_order_state(OrderState.UNKNOWN, "state_resolved", _FILL) is OrderState.FILLED


def test_unknown_cannot_be_resolved_without_terminal_evidence():
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.UNKNOWN, "state_resolved", {"terminal_status": "OPEN"})


def test_unknown_is_not_cleared_by_an_ordinary_event():
    with pytest.raises(IllegalTransition):
        next_order_state(OrderState.UNKNOWN, "accepted", _ACCEPTED)
