"""False-success placement and cancellation in OrderExecutor.

Three call sites report success for outcomes the exchange never confirmed. This
is the same defect the position-lifecycle contract exists to prevent — **an
acknowledgement is treated as a confirmation** — committed in the module that
actually places money at risk.

  1. `execute_market_order`: `resp.get("success_response", resp)` falls back to
     the WHOLE response, so a failure body yields `order_id="unknown"`, persists
     `status="live"`, and returns `{"success": True}`. `resp.get("success")` is
     never read at all.
  2. `execute_signal`: same fallback and same `"unknown"` default. It does
     compute `status` as live-or-failed, so the database records the failure —
     and then it returns `{"success": True}` anyway. The row and the caller
     disagree about whether an order exists.
  3. `cancel_order`: persists `"canceled"` on the mere absence of an exception,
     never inspecting the per-order results, so a REFUSED cancel is recorded as
     a completed one.
  4. `execute_signal`'s retry loop resubmits up to three times on exception. An
     exception after the request was sent does not mean the order was not
     placed, so a blind retry can double real exposure.

The required semantics (agreed with the parallel Codex session, which owns the
maker-path fixes this builds on):

  * `success is True` **and** a usable non-placeholder string id → accepted.
  * `success is False` → a rejection. Nothing was placed; safe to report failure.
  * anything else — a missing or malformed `success`, or `success is True` with
    no usable id — is **ambiguous**, not a rejection: an order may exist. It must
    return `reconciliation_required` and must never be retried.
  * a cancellation needs an affirmative per-order acknowledgement **and** a
    follow-up terminal snapshot. A partial fill or an unknown state stays
    `reconciliation_required` and must never persist `"canceled"`; a settled
    partial is not a zero-fill cancellation.

Spec: docs/specs/2026-09-26-position-lifecycle-contract.md §4 (identification is
necessary but not sufficient) and the three-way distinction in its table:
submission accepted / cancellation acknowledged / cancellation terminally
confirmed.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _live_executor():
    from agents.order_executor import OrderExecutor

    ex = OrderExecutor(dry_run=False)
    # Drawdown and preflight have their own coverage; mock them so these
    # assertions isolate placement/cancellation evidence handling.
    ex._check_drawdown = AsyncMock(return_value=None)
    ex._preflight = AsyncMock(return_value=None)
    return ex


@pytest.fixture
def signal_buy():
    return {
        "product_id": "BTC-USD",
        "side": "BUY",
        "price": 100.0,
        "quote_size": 50.0,
        "signal_type": "TEST",
    }


def _saved_statuses(db) -> list:
    return [call.args[0].get("status") for call in db.save_order.call_args_list]


def _saved_order_ids(db) -> list:
    return [call.args[0].get("order_id") for call in db.save_order.call_args_list]


# ── Responses that must never be read as a confirmed placement ────────────────

_REJECTION = {"success": False, "error_response": {"error": "INSUFFICIENT_FUND"}}
_AMBIGUOUS_RESPONSES = [
    pytest.param({}, id="empty-body"),
    pytest.param({"error_response": {"error": "boom"}}, id="error-body-no-success-key"),
    pytest.param({"success_response": {"order_id": "ex-1"}}, id="id-but-no-success-key"),
    pytest.param({"success": True}, id="success-but-no-success_response"),
    pytest.param({"success": True, "success_response": {}}, id="success-but-no-id"),
    pytest.param(
        {"success": True, "success_response": {"order_id": "unknown"}}, id="placeholder-id"
    ),
    pytest.param({"success": True, "success_response": {"order_id": ""}}, id="empty-id"),
    pytest.param({"success": True, "success_response": {"order_id": "   "}}, id="blank-id"),
    pytest.param({"success": True, "success_response": {"order_id": 12345}}, id="non-str-id"),
    pytest.param({"success": True, "success_response": {"order_id": None}}, id="none-id"),
    pytest.param(
        {"success": "true", "success_response": {"order_id": "ex-1"}}, id="string-true-not-bool"
    ),
    pytest.param({"success": 1, "success_response": {"order_id": "ex-1"}}, id="int-one-not-bool"),
]

_ACCEPTED = {"success": True, "success_response": {"order_id": "ex-1"}}


# ── 1. execute_market_order ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_market_order_rejection_is_not_success():
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_market_order = AsyncMock(return_value=_REJECTION)
        db.save_order = AsyncMock()
        result = await ex.execute_market_order("BTC-USD", "BUY", 50.0)

    assert result["success"] is False
    # A rejection is unambiguous: nothing was placed, so nothing to reconcile.
    assert result.get("reconciliation_required") is not True
    assert "live" not in _saved_statuses(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", _AMBIGUOUS_RESPONSES)
async def test_market_order_ambiguous_response_requires_reconciliation(resp):
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_market_order = AsyncMock(return_value=resp)
        db.save_order = AsyncMock()
        result = await ex.execute_market_order("BTC-USD", "BUY", 50.0)

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert "live" not in _saved_statuses(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [_REJECTION, *_AMBIGUOUS_RESPONSES])
async def test_market_order_never_persists_a_placeholder_identifier(resp):
    """`"unknown"` is an error path, not an identifier (spec §4)."""
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_market_order = AsyncMock(return_value=resp)
        db.save_order = AsyncMock()
        await ex.execute_market_order("BTC-USD", "BUY", 50.0)

    assert "unknown" not in _saved_order_ids(db)


@pytest.mark.asyncio
async def test_market_order_accepted_placement_still_succeeds():
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_market_order = AsyncMock(return_value=_ACCEPTED)
        db.save_order = AsyncMock()
        result = await ex.execute_market_order("BTC-USD", "BUY", 50.0)

    assert result["success"] is True
    assert result["order_id"] == "ex-1"
    assert _saved_statuses(db) == ["live"]
    assert _saved_order_ids(db) == ["ex-1"]


# ── 2. execute_signal ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_signal_rejection_is_not_success(signal_buy):
    """The row said "failed" while the caller was told "success"."""
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=_REJECTION)
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        result = await ex.execute_signal(signal_buy)

    assert result["success"] is False
    assert "live" not in _saved_statuses(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", _AMBIGUOUS_RESPONSES)
async def test_signal_ambiguous_response_requires_reconciliation(resp, signal_buy):
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=resp)
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        result = await ex.execute_signal(signal_buy)

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert "live" not in _saved_statuses(db)


@pytest.mark.asyncio
@pytest.mark.parametrize("resp", [_REJECTION, *_AMBIGUOUS_RESPONSES])
async def test_signal_never_resubmits_after_an_unconfirmed_placement(resp, signal_buy):
    """The retry loop must not fire on a response-level failure: an order may
    already exist, and resubmitting doubles real exposure."""
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=resp)
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        await ex.execute_signal(signal_buy)

    assert cb.place_limit_order.await_count == 1


@pytest.mark.asyncio
async def test_signal_does_not_retry_after_an_exception_mid_submission(signal_buy):
    """An exception after the request was sent does not prove the order was not
    placed. Retrying is how exposure doubles (spec I4)."""
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(side_effect=TimeoutError("no response"))
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        result = await ex.execute_signal(signal_buy)

    assert cb.place_limit_order.await_count == 1
    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert "live" not in _saved_statuses(db)


@pytest.mark.asyncio
async def test_signal_accepted_placement_still_succeeds(signal_buy):
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=_ACCEPTED)
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        result = await ex.execute_signal(signal_buy)

    assert result["success"] is True
    assert result["order_id"] == "ex-1"
    assert _saved_statuses(db) == ["live"]


# ── 3. cancel_order ─────────────────────────────────────────────────────────


_CANCEL_ACK = {"results": [{"success": True, "order_id": "ex-1"}]}
_TERMINAL_CANCELLED = {
    "order_id": "ex-1",
    "status": "CANCELLED",
    "pending_cancel": False,
    "filled_size": "0",
    "filled_value": "0",
}


def _cancel_mocks(cb, db, *, cancel_resp, snapshot=None, snapshot_exc=None):
    cb.cancel_orders = AsyncMock(return_value=cancel_resp)
    if snapshot_exc is not None:
        cb.get_order = AsyncMock(side_effect=snapshot_exc)
    else:
        cb.get_order = AsyncMock(return_value=snapshot or {})
    db.update_order_status = AsyncMock()


def _persisted_cancel(db) -> bool:
    return any(
        "cancel" in str(call.args[1]).lower() for call in db.update_order_status.call_args_list
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_resp",
    [
        pytest.param({}, id="no-results-key"),
        pytest.param({"results": []}, id="empty-results"),
        pytest.param({"results": [{"success": True, "order_id": "OTHER"}]}, id="different-order"),
        pytest.param(
            {
                "results": [
                    {
                        "success": False,
                        "failure_reason": "UNKNOWN_CANCEL_FAILURE_REASON",
                        "order_id": "ex-1",
                    }
                ]
            },
            id="per-order-refusal",
        ),
        pytest.param({"results": [{"order_id": "ex-1"}]}, id="no-success-field"),
        pytest.param({"results": [{"success": "true", "order_id": "ex-1"}]}, id="string-true"),
        pytest.param(
            {
                "results": [
                    {"success": True, "order_id": "ex-1"},
                    {"success": True, "order_id": "ex-1"},
                ]
            },
            id="duplicate-results",
        ),
    ],
)
async def test_cancel_without_an_affirmative_acknowledgement_is_not_canceled(cancel_resp):
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=cancel_resp)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert _persisted_cancel(db) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        pytest.param({}, id="empty-snapshot"),
        pytest.param({"order_id": "OTHER", "status": "CANCELLED"}, id="wrong-order"),
        pytest.param({"order_id": "ex-1", "status": "OPEN"}, id="still-open"),
        pytest.param(
            {
                "order_id": "ex-1",
                "status": "CANCELLED",
                "pending_cancel": True,
                "filled_size": "0",
                "filled_value": "0",
            },
            id="pending-cancel",
        ),
        pytest.param(
            {
                "order_id": "ex-1",
                "status": "CANCELLED",
                "pending_cancel": False,
                "filled_size": "0",
            },
            id="missing-filled_value",
        ),
    ],
)
async def test_cancel_acknowledged_but_not_terminally_confirmed_requires_reconciliation(snapshot):
    """An acknowledgement is not a terminal state. Only the snapshot decides."""
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot=snapshot)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert _persisted_cancel(db) is False


@pytest.mark.asyncio
async def test_cancel_of_a_partially_filled_order_is_not_a_cancellation():
    """A settled partial is not a zero-fill cancel. The fill must survive in the
    result, and the row must not claim the order was cancelled."""
    ex = _live_executor()
    partial = {
        "order_id": "ex-1",
        "status": "CANCELLED",
        "pending_cancel": False,
        "filled_size": "0.4",
        "filled_value": "40",
    }
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot=partial)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert result.get("order_id") == "ex-1"
    assert str(result.get("filled_size")) == "0.4"
    assert str(result.get("filled_value")) == "40"
    assert _persisted_cancel(db) is False


@pytest.mark.asyncio
async def test_cancel_that_lost_the_race_to_a_fill_is_reported_as_filled():
    ex = _live_executor()
    filled = {
        "order_id": "ex-1",
        "status": "FILLED",
        "pending_cancel": False,
        "filled_size": "1",
        "filled_value": "100",
    }
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot=filled)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert _persisted_cancel(db) is False
    assert str(result.get("status", "")).upper() == "FILLED"


@pytest.mark.asyncio
async def test_cancel_snapshot_failure_requires_reconciliation():
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot_exc=RuntimeError("no status"))
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert _persisted_cancel(db) is False


@pytest.mark.asyncio
async def test_a_terminally_confirmed_zero_fill_cancel_still_succeeds():
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot=_TERMINAL_CANCELLED)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is True
    assert _persisted_cancel(db) is True
    db.update_order_status.assert_awaited_once_with("ex-1", "canceled")


@pytest.mark.asyncio
async def test_dry_run_cancel_is_unchanged():
    from agents.order_executor import OrderExecutor

    ex = OrderExecutor(dry_run=True)
    with patch("agents.order_executor.database") as db:
        db.update_order_status = AsyncMock()
        result = await ex.cancel_order("DRY_1")

    assert result["success"] is True
    assert result.get("dry_run") is True
    db.update_order_status.assert_awaited_once_with("DRY_1", "canceled")


# ── 4. the shared validator itself ──────────────────────────────────────────


def test_placement_validation_is_shared_not_duplicated():
    """Rounds 3 and 4 of the lifecycle-validator review established that
    per-call-site copies of the same rule diverge. All three placement paths must
    route through one helper, so this asserts the helper exists and is used."""
    import inspect

    from agents import order_executor

    assert hasattr(order_executor, "_accepted_placement")
    for name in ("execute_signal", "execute_market_order", "execute_maker_signal"):
        src = inspect.getsource(getattr(order_executor.OrderExecutor, name))
        assert "_accepted_placement" in src, f"{name} does not use the shared validator"


@pytest.mark.parametrize(
    "resp,expected",
    [
        (_ACCEPTED, "ex-1"),
        ({"success": True, "success_response": {"order_id": " ex-2 "}}, "ex-2"),
    ],
)
def test_accepted_placement_returns_the_usable_identifier(resp, expected):
    from agents.order_executor import _accepted_placement

    assert _accepted_placement(resp) == expected


@pytest.mark.parametrize("resp", [_REJECTION, *_AMBIGUOUS_RESPONSES])
def test_accepted_placement_returns_none_for_anything_unconfirmed(resp):
    from agents.order_executor import _accepted_placement

    assert _accepted_placement(resp) is None


@pytest.mark.parametrize(
    "resp,rejected",
    [
        (_REJECTION, True),
        ({"success": False}, True),
        ({}, False),
        ({"success": True, "success_response": {}}, False),
        ({"success": "false"}, False),
        ({"success": 0}, False),
    ],
)
def test_explicit_rejection_is_distinguished_from_ambiguity(resp, rejected):
    """`success is False` means nothing was placed. Anything else that fails
    validation might have placed something, and the two must not be conflated —
    that conflation is what this whole branch exists to remove."""
    from agents.order_executor import _explicit_rejection

    assert _explicit_rejection(resp) is rejected


def test_non_dict_responses_are_never_accepted():
    from agents.order_executor import _accepted_placement, _explicit_rejection

    for junk in (None, [], "ok", 0, MagicMock()):
        assert _accepted_placement(junk) is None
        assert _explicit_rejection(junk) is False


# ─────────────────────────────────────────────────────────────────────────────
# Review round 1 on this branch (Codex): a persistence failure must not discard
# an accepted order id.
#
# Once `_accepted_placement` returns an id, a REAL ORDER EXISTS at the exchange.
# The persistence that follows — `save_order`, then `mark_signal_acted` — was
# left unguarded, so an exception there propagated out of the method and the
# accepted identifier went with it: no row names the order, and the caller
# receives an exception rather than a result carrying the id.
#
# The old code was worse in a different way (persistence sat inside the retry
# loop, so a failed write triggered another placement), but "no longer retries"
# is not the same as "does not lose the order". An accepted placement whose id
# is discarded is unreconcilable exposure, which is the defect class this branch
# exists to remove.
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["save_order", "mark_signal_acted"])
async def test_signal_persistence_failure_preserves_the_accepted_order_id(failing, signal_buy):
    signal_buy["id"] = 77  # so mark_signal_acted is reached
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=_ACCEPTED)
        db.save_order = AsyncMock()
        db.mark_signal_acted = AsyncMock()
        getattr(db, failing).side_effect = RuntimeError("db is down")
        result = await ex.execute_signal(signal_buy)

    assert result["success"] is False
    assert result["order_id"] == "ex-1", "the accepted id must survive a persistence failure"
    assert result.get("reconciliation_required") is True
    # The order exists; nothing may be placed again.
    assert cb.place_limit_order.await_count == 1


@pytest.mark.asyncio
async def test_market_order_persistence_failure_preserves_the_accepted_order_id():
    ex = _live_executor()
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_market_order = AsyncMock(return_value=_ACCEPTED)
        db.save_order = AsyncMock(side_effect=RuntimeError("db is down"))
        result = await ex.execute_market_order("BTC-USD", "BUY", 50.0)

    assert result["success"] is False
    assert result["order_id"] == "ex-1"
    assert result.get("reconciliation_required") is True
    assert cb.place_market_order.await_count == 1


@pytest.mark.asyncio
async def test_maker_persistence_failure_preserves_the_accepted_order_id():
    ex = _live_executor()
    signal = {
        "product_id": "BTC-USD",
        "side": "BUY",
        "price": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "quote_size": 50.0,
        "signal_type": "TEST",
    }
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        cb.place_limit_order = AsyncMock(return_value=_ACCEPTED)
        # Set explicitly: on a bare MagicMock, `await_count` is itself a mock, so
        # comparing it to 0 asserts nothing at all.
        cb.place_market_order = AsyncMock(return_value=_ACCEPTED)
        cb.cancel_orders = AsyncMock(return_value=_CANCEL_ACK)
        db.save_order = AsyncMock(side_effect=RuntimeError("db is down"))
        result = await ex.execute_maker_signal(signal, timeout_secs=0.01)

    assert result["success"] is False
    assert result["order_id"] == "ex-1"
    assert result.get("reconciliation_required") is True
    # Critically: no market fallback may be attempted for an order we cannot
    # record, and no cancellation either — that is how one signal becomes two.
    assert cb.place_market_order.await_count == 0
    assert cb.cancel_orders.await_count == 0


@pytest.mark.asyncio
async def test_cancel_snapshot_for_a_different_order_contributes_no_evidence():
    """Found by Codex probing the ordering assumption in this method.

    The identity check lived inside the FILLED branch, so a snapshot naming a
    DIFFERENT order fell through to `_confirmed_unfilled_cancel` — which rejects
    it on identity, correctly refusing the DB write — and then the result was
    built with that other order's status and fills attached to the order we asked
    about. Refusing to persist is not enough: a reconciliation record citing
    another order's 100 units is worse than one citing none, because it looks
    like evidence. Identity must be established before any field is trusted.
    """
    ex = _live_executor()
    other = {
        "order_id": "different-2",
        "status": "FILLED",
        "pending_cancel": False,
        "filled_size": "100",
        "filled_value": "9000",
    }
    with (
        patch("agents.order_executor.coinbase_client") as cb,
        patch("agents.order_executor.database") as db,
    ):
        _cancel_mocks(cb, db, cancel_resp=_CANCEL_ACK, snapshot=other)
        result = await ex.cancel_order("ex-1")

    assert result["success"] is False
    assert result.get("reconciliation_required") is True
    assert result.get("order_id") == "ex-1"
    assert _persisted_cancel(db) is False
    # None of the other order's evidence may be attributed to this one.
    assert result.get("filled_size") is None
    assert result.get("filled_value") is None
    assert result.get("status") is None
    assert "100" not in str(result.get("reason", ""))
