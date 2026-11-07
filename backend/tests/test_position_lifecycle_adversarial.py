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


# ─────────────────────────────────────────────────────────────────────────────
# Review round 2 (Codex). A DIFFERENT root cause from the block above.
#
# Round 1 was presence-versus-validity: evidence keys existed but their contents
# were never checked. These four are STATE-VERSUS-EVIDENCE CONSISTENCY: each
# transition validated its evidence in isolation while ignoring what the CURRENT
# STATE already proves. WORKING_PARTIAL witnesses a positive fill, so a claim of
# zero cumulative fill is not merely unproven — it is contradicted, and the
# validator accepted it, erasing a known fill.
#
# Contradictory evidence must fail closed, whichever branch would otherwise win.
# ─────────────────────────────────────────────────────────────────────────────


# ── 7. a rejection must prove identity and zero fills ────────────────────────


def test_rejected_requires_identity():
    ev = {
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.SUBMITTING, "rejected", ev)


def test_rejected_with_fills_is_not_a_rejection():
    """A "rejection" carrying a fill is a reconciliation case, not a rejection."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("1"),
        "filled_value": Decimal("100"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "rejected", ev)


def test_resolving_unknown_as_rejected_requires_identity_and_zero_fills():
    bare = {
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("1"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.UNKNOWN, "state_resolved", bare)

    no_identity = {
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    with pytest.raises(InsufficientEvidence):
        next_order_state(OrderState.UNKNOWN, "state_resolved", no_identity)


def test_resolving_unknown_as_rejected_succeeds_with_full_proof():
    ev = {
        **_ACCEPTED,
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    assert next_order_state(OrderState.UNKNOWN, "state_resolved", ev) is OrderState.REJECTED


# ── 8. a cancel may never erase a fill the state already witnesses ───────────


def test_cancel_confirmed_from_working_partial_cannot_claim_zero_fill():
    """WORKING_PARTIAL means a positive fill was observed. Zero cumulative fill
    contradicts the state rather than merely lacking proof."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.WORKING_PARTIAL, "cancel_confirmed", ev)


def test_cancel_confirmed_from_working_partial_settles_the_partial():
    """The legitimate outcome: the remainder is cancelled, the fill is preserved."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0.4"),
        "filled_value": Decimal("40"),
        "avg_fill_price": Decimal("100"),
        "fill_ids": ["f-1"],
        "remaining_size": Decimal("0"),
        "remainder_live": False,
    }
    assert (
        next_order_state(OrderState.WORKING_PARTIAL, "cancel_confirmed", ev)
        is OrderState.SETTLED_PARTIAL
    )


def test_cancel_confirmed_from_accepted_with_zero_fill_is_still_cancelled():
    """Unchanged: from ACCEPTED, nothing has been witnessed, so zero is coherent."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
    }
    assert next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev) is OrderState.CANCELLED


# ── 9. remainder_terminal must reuse the positive-fill classifier ────────────


def test_remainder_terminal_requires_a_positive_cumulative_fill():
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "remaining_size": Decimal("0"),
        "remainder_live": False,
        "fill_ids": ["f-1"],
        "avg_fill_price": Decimal("100"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.WORKING_PARTIAL, "remainder_terminal", ev)


def test_remainder_that_fully_filled_classifies_as_filled_not_settled_partial():
    """Without an intended_size comparison a fully-filled remainder was
    mislabelled SETTLED_PARTIAL."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "FILLED",
        "observed_at": 1,
        "filled_size": Decimal("1.0"),
        "intended_size": Decimal("1.0"),
        "remaining_size": Decimal("0"),
        "remainder_live": False,
        "fill_ids": ["f-1", "f-2"],
        "avg_fill_price": Decimal("100"),
    }
    assert (
        next_order_state(OrderState.WORKING_PARTIAL, "remainder_terminal", ev) is OrderState.FILLED
    )


def test_remainder_terminal_partial_still_settles_partial():
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0.4"),
        "intended_size": Decimal("1.0"),
        "remaining_size": Decimal("0"),
        "remainder_live": False,
        "fill_ids": ["f-1"],
        "avg_fill_price": Decimal("100"),
    }
    assert (
        next_order_state(OrderState.WORKING_PARTIAL, "remainder_terminal", ev)
        is OrderState.SETTLED_PARTIAL
    )


# ── 10. contradictory evidence must fail closed on every branch ──────────────


def test_full_fill_with_a_live_remainder_fails_closed():
    """filled == intended cannot coexist with a live 0.5 remainder. The FILLED
    branch returned before ever reading the remainder fields."""
    ev = {**_FILL, "remainder_live": True, "remaining_size": Decimal("0.5")}
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_full_fill_with_a_positive_remaining_size_fails_closed():
    ev = {**_FILL, "remaining_size": Decimal("0.5")}
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "fill_observed", ev)


def test_full_fill_with_an_explicitly_dead_zero_remainder_is_fine():
    """Consistent evidence still passes: nothing left, and it is not live."""
    ev = {**_FILL, "remainder_live": False, "remaining_size": Decimal("0")}
    assert next_order_state(OrderState.ACCEPTED, "fill_observed", ev) is OrderState.FILLED


# ─────────────────────────────────────────────────────────────────────────────
# Review round 3 (Codex): CLASS closure, not another example.
#
# Round 2's fixes were written branch by branch and promptly diverged: a
# hand-written `is True` bypassed the strict-bool validator (an int 1 and the
# string "true" both passed), the partial-cancel branch omitted the overfill
# comparison, and remainder_terminal disagreed with fill_observed about whether a
# complete fill reported CANCELLED is FILLED.
#
# Codex's instruction was to centralise rather than patch, so these tests are
# PARAMETERISED ACROSS EVENTS. Each invariant is asserted once against every
# event that reasons about fills, so a future divergence between branches fails
# here rather than being found by probing again.
# ─────────────────────────────────────────────────────────────────────────────

_BASE = {
    "client_order_id": "cid-1",
    "order_id": "ex-1",
    "observed_at": 1,
    "avg_fill_price": Decimal("100"),
    "fill_ids": ["f-1"],
    "product_id": "BTC-USD",
    "side": "BUY",
    "execution_mode": "live",
    "run_id": "r",
    "config_version": "c",
    "model_hash": "m",
    "created_at": 1,
    "submitted_at": 1,
    "accepted_at": 1,
}


def _drive(event, extra, state=None):
    """Apply `event` with _BASE + extra from the state that event is valid in."""
    states = {
        "fill_observed": OrderState.ACCEPTED,
        "remainder_terminal": OrderState.WORKING_PARTIAL,
        "cancel_confirmed": OrderState.WORKING_PARTIAL,
        "state_resolved": OrderState.UNKNOWN,
    }
    return next_order_state(state or states[event], event, {**_BASE, **extra})


_FILL_EVENTS = ["fill_observed", "remainder_terminal", "cancel_confirmed", "state_resolved"]


def _terminal_for(event):
    """A terminal status each event will accept, so the parameterised cases
    isolate the invariant under test rather than tripping on status policy."""
    return "CANCELLED" if event == "cancel_confirmed" else "FILLED"


# ── Invariant: a non-boolean remainder_live is never accepted, on any event ───


@pytest.mark.parametrize("event", _FILL_EVENTS)
@pytest.mark.parametrize("truthy", [1, 0, "true", "false", "True", [], [1], 1.0])
def test_no_event_accepts_a_non_boolean_remainder_live(event, truthy):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("40"),
                "remainder_live": truthy,
            },
        )


# ── Invariant: overfill is refused on every event ────────────────────────────


@pytest.mark.parametrize("event", _FILL_EVENTS)
def test_no_event_accepts_an_overfill(event):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("2"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("200"),
                "remainder_live": False,
            },
        )


# ── Invariant: filled + remaining may not exceed intended, on every event ────


@pytest.mark.parametrize("event", _FILL_EVENTS)
def test_no_event_accepts_inconsistent_remaining(event):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("5"),
                "filled_value": Decimal("40"),
                "remainder_live": True,
            },
        )


# ── Invariant: a non-finite or negative quantity is refused on every event ───


@pytest.mark.parametrize("event", _FILL_EVENTS)
@pytest.mark.parametrize("bad", [_INF, _NAN, Decimal("-1")])
def test_no_event_accepts_a_nonfinite_or_negative_fill(event, bad):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": bad,
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("0"),
                "remainder_live": False,
            },
        )


# ── Invariant: identity is required on every event ────────────────────────────


@pytest.mark.parametrize("event", _FILL_EVENTS)
def test_no_event_accepts_missing_identity(event):
    evidence = {k: v for k, v in _BASE.items() if k not in ("order_id", "client_order_id")}
    with pytest.raises(InsufficientEvidence):
        next_order_state(
            {
                "fill_observed": OrderState.ACCEPTED,
                "remainder_terminal": OrderState.WORKING_PARTIAL,
                "cancel_confirmed": OrderState.WORKING_PARTIAL,
                "state_resolved": OrderState.UNKNOWN,
            }[event],
            event,
            {
                **evidence,
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("40"),
                "remainder_live": False,
            },
        )


# ── Invariant: empty fill ids are refused on every event ─────────────────────


@pytest.mark.parametrize("event", _FILL_EVENTS)
def test_no_event_accepts_empty_fill_ids(event):
    with pytest.raises(InsufficientEvidence):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("40"),
                "remainder_live": False,
                "fill_ids": [],
            },
        )


# ── ONE classification policy: quantity decides, across every event ──────────


@pytest.mark.parametrize("event", ["fill_observed", "remainder_terminal", "cancel_confirmed"])
def test_a_complete_fill_classifies_as_filled_on_every_event(event):
    """The round-2 inconsistency: remainder_terminal said FILLED while
    fill_observed raised, for identical evidence. Quantity decides now."""
    assert (
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("1"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("100"),
                "remainder_live": False,
            },
        )
        is OrderState.FILLED
    )


@pytest.mark.parametrize("event", ["fill_observed", "remainder_terminal", "cancel_confirmed"])
def test_a_partial_settles_partial_on_every_event(event):
    assert (
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("40"),
                "remainder_live": False,
            },
        )
        is OrderState.SETTLED_PARTIAL
    )


def test_a_complete_fill_reported_cancelled_is_filled_not_cancelled():
    """The cancel lost the race. The fill happened, so the order is FILLED — and
    this must not depend on which event carried the evidence."""
    for event in ("fill_observed", "remainder_terminal"):
        assert (
            _drive(
                event,
                {
                    "terminal_status": "CANCELLED",
                    "filled_size": Decimal("1"),
                    "intended_size": Decimal("1"),
                    "remaining_size": Decimal("0"),
                    "filled_value": Decimal("100"),
                    "remainder_live": False,
                },
            )
            is OrderState.FILLED
        )


# ── A non-terminal status is never terminal evidence, on every event ─────────


@pytest.mark.parametrize("event", _FILL_EVENTS)
@pytest.mark.parametrize("bad_status", ["OPEN", "PENDING", "WORKING", "", "   "])
def test_no_event_treats_a_nonterminal_status_as_terminal(event, bad_status):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": bad_status,
                "filled_size": Decimal("1"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("100"),
                "remainder_live": False,
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# Review round 4 (Codex): terminality is shared evidence too.
#
# Round 3 centralised the QUANTITY rules into `_validated_fill` and fixed the
# three bypasses — but centralising is only half the job if the other half of
# the evidence stays per-branch. Two defects came directly out of that gap, and
# reproducing them exposed two more Codex had not named:
#
#   1. `_ALL_TERMINALS` was substituted for `_FILL_TERMINALS` while unifying the
#      classification policy, which let the REJECT family stand as proof of a
#      fill — so a REJECTED status carrying a complete fill returned FILLED,
#      contradicting the rejection zero-fill contract two branches away.
#      (Also true of `remainder_terminal`, which Codex did not list.)
#   2. No branch asked whether a TERMINAL outcome still had something working.
#      A confirmed cancellation was accepted with `remainder_live=True` and a
#      positive `remaining_size` — including the zero-fill cancellation paths on
#      both `cancel_confirmed` and UNKNOWN resolution, which is precisely where
#      an unreconciled live order would be silently forgotten.
#
# Terminality therefore gets the same treatment quantities got: one validator,
# applied at every terminal exit, parameterised across events here.
# ─────────────────────────────────────────────────────────────────────────────

_TERMINAL_EVENTS = ["fill_observed", "remainder_terminal", "cancel_confirmed", "state_resolved"]


# ── The reject family is never proof of a fill ────────────────────────────────


@pytest.mark.parametrize("event", ["fill_observed", "remainder_terminal", "state_resolved"])
@pytest.mark.parametrize("status", ["REJECTED", "FAILED"])
def test_a_reject_status_is_never_proof_of_a_fill(event, status):
    """A rejection carrying fills is a reconciliation case. `_ALL_TERMINALS`
    admitted the reject family into the fill path and returned FILLED."""
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": status,
                "reason": "x",
                "filled_size": Decimal("1"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0"),
                "filled_value": Decimal("100"),
                "remainder_live": False,
            },
        )


# ── A terminal outcome cannot leave anything working ─────────────────────────


@pytest.mark.parametrize("event", _TERMINAL_EVENTS)
def test_no_terminal_event_accepts_a_live_remainder(event):
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0.6"),
                "filled_value": Decimal("40"),
                "remainder_live": True,
            },
        )


@pytest.mark.parametrize("event", _TERMINAL_EVENTS)
def test_no_terminal_event_accepts_a_positive_working_remainder(event):
    """Even without a remainder_live claim, a positive remaining_size contradicts
    terminality."""
    with pytest.raises(_REJECTED):
        _drive(
            event,
            {
                "terminal_status": _terminal_for(event),
                "filled_size": Decimal("0.4"),
                "intended_size": Decimal("1"),
                "remaining_size": Decimal("0.6"),
                "filled_value": Decimal("40"),
            },
        )


# ── Zero-fill cancellation is terminal too ───────────────────────────────────


def test_a_zero_fill_cancellation_cannot_leave_a_live_remainder_from_accepted():
    """The most dangerous shape: nothing filled, the cancel is recorded as
    confirmed, and a live remainder is still working at the exchange with no
    record that anything is outstanding."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
        "remaining_size": Decimal("1"),
        "remainder_live": True,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev)


def test_a_zero_fill_cancellation_cannot_leave_a_live_remainder_from_unknown():
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
        "remaining_size": Decimal("1"),
        "remainder_live": True,
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.UNKNOWN, "state_resolved", ev)


def test_a_clean_zero_fill_cancellation_still_passes():
    """The honest version of the same shape stays accepted, so the rule above is
    a contradiction check and not a blanket refusal."""
    ev = {
        **_ACCEPTED,
        "terminal_status": "CANCELLED",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
        "remaining_size": Decimal("0"),
        "remainder_live": False,
    }
    assert next_order_state(OrderState.ACCEPTED, "cancel_confirmed", ev) is OrderState.CANCELLED
    assert next_order_state(OrderState.UNKNOWN, "state_resolved", ev) is OrderState.CANCELLED


def test_a_rejection_cannot_leave_a_live_remainder():
    ev = {
        **_ACCEPTED,
        "terminal_status": "REJECTED",
        "reason": "x",
        "observed_at": 1,
        "filled_size": Decimal("0"),
        "filled_value": Decimal("0"),
        "remainder_live": True,
        "remaining_size": Decimal("1"),
    }
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.ACCEPTED, "rejected", ev)
    with pytest.raises(_REJECTED):
        next_order_state(OrderState.UNKNOWN, "state_resolved", ev)


# ── Equivalent evidence must produce equivalent verdicts ─────────────────────


@pytest.mark.parametrize(
    "filled,expected",
    [
        (Decimal("1"), OrderState.FILLED),
        (Decimal("0.4"), OrderState.SETTLED_PARTIAL),
    ],
)
def test_identical_evidence_agrees_across_every_event_that_can_carry_it(filled, expected):
    """The round-3 inconsistency generalised: one evidence dict, every event that
    accepts it, one verdict. This is the regression that catches a future branch
    drifting away from the shared validators."""
    evidence = {
        "terminal_status": "CANCELLED",
        "filled_size": filled,
        "intended_size": Decimal("1"),
        "remaining_size": Decimal("0"),
        "filled_value": filled * Decimal("100"),
        "remainder_live": False,
    }
    verdicts = {event: _drive(event, evidence) for event in _TERMINAL_EVENTS}
    assert set(verdicts.values()) == {expected}, verdicts
