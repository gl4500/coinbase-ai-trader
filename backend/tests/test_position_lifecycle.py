"""Pure transition validator for the position/order lifecycle contract.

Spec: docs/specs/2026-09-26-position-lifecycle-contract.md

Scope, stated up front because it matters: these tests exercise no exchange, no
database, no concurrency and no clock. They demonstrate that the transition rules
hold as written. They are NOT evidence of end-to-end execution correctness, and a
green run here must never be read as "execution is safe" — that would repeat the
mistake the contract exists to correct (spec §9).
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

# Evidence fixtures — minimal complete sets per spec §4.
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
_SUBMITTED = {**_INTENT, "submitted_at": 1_800_000_001}
_ACCEPTED = {**_SUBMITTED, "order_id": "ex-1", "accepted_at": 1_800_000_002}
_FILL = {
    **_ACCEPTED,
    "terminal_status": "FILLED",
    "filled_size": Decimal("1.0"),
    "avg_fill_price": Decimal("100.0"),
    "fee": Decimal("0.6"),
    "fill_ids": ["f-1"],
    "observed_at": 1_800_000_100,
}


# ── Order machine: legal paths ────────────────────────────────────────────────


def test_intent_is_reachable_from_nothing():
    assert next_order_state(None, "intent_created", _INTENT) is OrderState.INTENT_CREATED


def test_happy_path_to_filled():
    s = next_order_state(None, "intent_created", _INTENT)
    s = next_order_state(s, "submitted", _SUBMITTED)
    assert s is OrderState.SUBMITTING
    s = next_order_state(s, "accepted", _ACCEPTED)
    assert s is OrderState.ACCEPTED
    s = next_order_state(s, "fill_observed", _FILL)
    assert s is OrderState.FILLED


def test_intent_requires_a_client_order_id():
    """Spec §2: the client_order_id is what makes "did you ever see this?"
    answerable after a crash."""
    evidence = {k: v for k, v in _INTENT.items() if k != "client_order_id"}
    with pytest.raises(InsufficientEvidence):
        next_order_state(None, "intent_created", evidence)


def test_intent_must_not_require_exchange_ids():
    """Spec I8: order_id and fill_id are nullable until observed. An intent row
    cannot carry an identifier the exchange has not issued."""
    assert "order_id" not in _INTENT
    assert next_order_state(None, "intent_created", _INTENT) is OrderState.INTENT_CREATED


def test_accepted_requires_an_exchange_order_id():
    evidence = {k: v for k, v in _ACCEPTED.items() if k != "order_id"}
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.SUBMITTING, "accepted", evidence)


def test_a_placeholder_order_id_is_rejected():
    """Spec §4: "unknown" is an error path, never an identifier."""
    for bad in ("unknown", "", None):
        with pytest.raises(InsufficientEvidence):
            next_order_state(OrderState.SUBMITTING, "accepted", {**_ACCEPTED, "order_id": bad})


# ── Order machine: the three distinctions the old code did not make ──────────


def test_cancellation_acknowledged_is_not_cancelled():
    """Spec §4: an acknowledgement is not a terminal confirmation. It must not
    move the order to a terminal state."""
    s = next_order_state(OrderState.ACCEPTED, "cancel_acknowledged", _ACCEPTED)
    assert s is OrderState.ACCEPTED
    assert not OrderState.ACCEPTED.is_terminal


def test_cancelled_requires_terminal_status_and_explicit_zero_fill():
    ok = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
        "observed_at": 1_800_000_050,
    }
    assert next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ok) is OrderState.CANCELLED

    # Missing the explicit zeros is not evidence of a clean cancel.
    for missing in ("filled_size", "filled_value"):
        bad = {k: v for k, v in ok.items() if k != missing}
        with pytest.raises(InsufficientEvidence):
            next_order_state(OrderState.ACCEPTED, "cancel_confirmed", bad)


def test_cancellation_with_fills_is_not_cancelled():
    partial = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "filled_size": Decimal("0.4"),
        "filled_value": Decimal("40"),
        "observed_at": 1_800_000_050,
    }
    with pytest.raises(IllegalTransition):
        next_order_state(OrderState.ACCEPTED, "cancel_confirmed", partial)


def test_rejected_is_distinct_from_cancelled():
    ev = {
        **_SUBMITTED,
        "order_id": "ex-1",
        "terminal_status": "REJECTED",
        "reason": "INSUFFICIENT_FUNDS",
        "observed_at": 1_800_000_003,
    }
    assert next_order_state(OrderState.SUBMITTING, "rejected", ev) is OrderState.REJECTED
    assert OrderState.REJECTED is not OrderState.CANCELLED


def test_working_partial_needs_a_live_remainder_not_a_terminal_status():
    """Spec §4: requiring a terminal status here contradicted the state's own
    non-terminality."""
    ev = {
        **_ACCEPTED,
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": True,
        "remaining_size": Decimal("0.6"),
        "observed_at": 1_800_000_050,
    }
    s = next_order_state(OrderState.ACCEPTED, "fill_observed", ev)
    assert s is OrderState.WORKING_PARTIAL
    assert not s.is_terminal


def test_settled_partial_needs_terminal_confirmation_of_the_remainder():
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "filled_size": Decimal("0.4"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remainder_live": False,
        "remaining_size": Decimal("0"),
        "observed_at": 1_800_000_060,
    }
    s = next_order_state(OrderState.WORKING_PARTIAL, "remainder_terminal", ev)
    assert s is OrderState.SETTLED_PARTIAL
    assert s.is_terminal


def test_unknown_is_reachable_and_is_not_terminal():
    s = next_order_state(OrderState.SUBMITTING, "state_unknown", {"reason": "timeout"})
    assert s is OrderState.UNKNOWN
    assert not s.is_terminal


# ── Order machine: illegal paths ─────────────────────────────────────────────


def test_a_terminal_order_cannot_transition():
    for terminal in (
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.SETTLED_PARTIAL,
    ):
        with pytest.raises(IllegalTransition):
            next_order_state(terminal, "fill_observed", _FILL)


def test_cannot_skip_submission():
    with pytest.raises(IllegalTransition):
        next_order_state(OrderState.INTENT_CREATED, "accepted", _ACCEPTED)


def test_unknown_event_raises():
    with pytest.raises(IllegalTransition):
        next_order_state(OrderState.ACCEPTED, "teleported", _ACCEPTED)


# ── Position machine ─────────────────────────────────────────────────────────


def _pos(size, intended=Decimal("1.0"), increment=Decimal("0.00000001")):
    return {
        "position_size": size,
        "intended_size": intended,
        "size_increment": increment,
        "order_id": "ex-1",
        "fill_ids": ["f-1"],
        "observed_at": 1,
    }


def test_entry_opens_a_position():
    s = next_position_state(PositionState.FLAT, "entry_submitted", {})
    assert s is PositionState.OPENING
    s = next_position_state(s, "entry_filled", _pos(Decimal("1.0")))
    assert s is PositionState.OPEN


def test_partial_entry_is_open_with_residual():
    s = next_position_state(PositionState.OPENING, "entry_filled", _pos(Decimal("0.4")))
    assert s is PositionState.OPEN_WITH_RESIDUAL


def test_rejected_entry_returns_to_flat():
    s = next_position_state(PositionState.OPENING, "entry_failed", {"reason": "REJECTED"})
    assert s is PositionState.FLAT


def test_rejected_exit_returns_the_position_to_open_never_closed():
    """Spec §3 and finding 3: the position is still held. This is the single most
    important rule in the position machine."""
    s = next_position_state(PositionState.CLOSING, "exit_failed", {"reason": "REJECTED"})
    assert s is PositionState.OPEN
    assert s is not PositionState.CLOSED


def test_exit_rejection_from_a_residual_position_returns_to_residual():
    s = next_position_state(
        PositionState.CLOSING,
        "exit_failed",
        {"reason": "REJECTED", "prior_state": PositionState.OPEN_WITH_RESIDUAL},
    )
    assert s is PositionState.OPEN_WITH_RESIDUAL


def test_closed_requires_zero_exposure_at_the_product_increment():
    """Spec I7: quantities compare as Decimal at the product increment."""
    s = next_position_state(PositionState.CLOSING, "exit_filled", _pos(Decimal("0")))
    assert s is PositionState.CLOSED

    # Dust below the increment counts as flat.
    dust = _pos(Decimal("0.000000001"), increment=Decimal("0.00000001"))
    assert next_position_state(PositionState.CLOSING, "exit_filled", dust) is PositionState.CLOSED

    # Residual at or above the increment is NOT flat.
    residual = _pos(Decimal("0.5"), increment=Decimal("0.00000001"))
    assert (
        next_position_state(PositionState.CLOSING, "exit_filled", residual)
        is PositionState.OPEN_WITH_RESIDUAL
    )


def test_partial_exit_leaves_residual_exposure_not_closed():
    s = next_position_state(PositionState.CLOSING, "exit_filled", _pos(Decimal("0.6")))
    assert s is PositionState.OPEN_WITH_RESIDUAL
    assert s is not PositionState.CLOSED


def test_reconciliation_required_is_reachable_from_every_state():
    for state in PositionState:
        if state is PositionState.RECONCILIATION_REQUIRED:
            continue
        assert (
            next_position_state(state, "disagreement_detected", {"local": "x", "exchange": "y"})
            is PositionState.RECONCILIATION_REQUIRED
        )


def test_reconciliation_clears_only_with_deterministic_evidence():
    """Spec §7: mechanical confirmation is allowed; ambiguity needs authorisation."""
    determined = {**_pos(Decimal("0")), "exchange_terminal": True}
    assert (
        next_position_state(
            PositionState.RECONCILIATION_REQUIRED, "reconciler_confirmed", determined
        )
        is PositionState.CLOSED
    )

    with pytest.raises(InsufficientEvidence):
        next_position_state(
            PositionState.RECONCILIATION_REQUIRED,
            "reconciler_confirmed",
            {**_pos(Decimal("0")), "exchange_terminal": False},
        )


def test_discretionary_override_requires_recorded_authorisation():
    ambiguous = {**_pos(Decimal("0")), "exchange_terminal": False}
    with pytest.raises(InsufficientEvidence):
        next_position_state(PositionState.RECONCILIATION_REQUIRED, "override", ambiguous)
    authorised = {**ambiguous, "authorised_by": "operator", "authorisation_reason": "manual audit"}
    assert (
        next_position_state(PositionState.RECONCILIATION_REQUIRED, "override", authorised)
        is PositionState.CLOSED
    )


def test_a_retry_cannot_clear_reconciliation():
    """Spec I5."""
    with pytest.raises(IllegalTransition):
        next_position_state(PositionState.RECONCILIATION_REQUIRED, "entry_submitted", {})


def test_closing_is_only_reachable_from_held_exposure():
    for held in (PositionState.OPEN, PositionState.OPEN_WITH_RESIDUAL):
        assert next_position_state(held, "exit_submitted", {}) is PositionState.CLOSING
    for not_held in (PositionState.FLAT, PositionState.CLOSED):
        with pytest.raises(IllegalTransition):
            next_position_state(not_held, "exit_submitted", {})


def test_float_sizes_are_rejected():
    """Spec I7: never float equality on quantities."""
    with pytest.raises(InsufficientEvidence):
        next_position_state(
            PositionState.CLOSING,
            "exit_filled",
            {**_pos(Decimal("0")), "position_size": 0.0},
        )
