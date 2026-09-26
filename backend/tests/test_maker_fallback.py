"""Maker timeout fallback must never place a market order on an unconfirmed cancel.

Fixes execution finding 4 from docs/audits/2026-09-26-strategy-audit-report.md
section 5; analysis in docs/handoffs/2026-09-26-execution-findings.md.

Before this change, `execute_maker_signal` caught the cancel exception, logged it,
and fell through to `place_market_order` unconditionally. It also never inspected
the cancel response body, and treated a partial fill as no fill — so the account
could end up holding the resting limit AND a full-size market order.

The rule now: after a timeout, re-query the order's actual state and place a
market order ONLY for a confirmed-cancelled remainder. A missed entry is cheap;
double exposure is not.

Nothing here reaches the network — every exchange call is stubbed.
"""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import pytest  # noqa: E402

from agents import order_executor as oe  # noqa: E402

_SIGNAL = {
    "product_id": "BTC-USD",
    "side": "BUY",
    "bid": 100.0,
    "ask": 100.1,
    "quote_size": 50.0,
}


@pytest.fixture
def maker(monkeypatch):
    """Executor stubbed just far enough to reach the timeout-fallback branch."""
    executor = oe.OrderExecutor(dry_run=False)

    async def _no_drawdown(self):
        return None

    async def _no_preflight(self, quote_size):
        return None

    async def _never_fills(self, pid, order_id, timeout_secs):
        return False

    async def _place_limit(*a, **k):
        return {"success_response": {"order_id": "limit-1"}}

    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(oe.OrderExecutor, "_check_drawdown", _no_drawdown)
    monkeypatch.setattr(oe.OrderExecutor, "_preflight", _no_preflight)
    monkeypatch.setattr(oe.OrderExecutor, "_wait_for_fill", _never_fills)
    monkeypatch.setattr(oe.coinbase_client, "place_limit_order", _place_limit)
    monkeypatch.setattr(oe.database, "save_order", _noop)
    monkeypatch.setattr(oe.database, "update_order_status", _noop)
    return executor


@pytest.fixture
def market_calls(monkeypatch):
    """Records every market order the code attempts."""
    calls = []

    async def _place_market(pid, side, quote_size):
        calls.append({"pid": pid, "side": side, "quote_size": quote_size})
        return {"success_response": {"order_id": "market-1"}}

    monkeypatch.setattr(oe.coinbase_client, "place_market_order", _place_market)
    return calls


def _stub_orders(monkeypatch, order):
    """Make get_orders report one order state (or none, when order is None)."""

    async def _get_orders(product_id=None, order_status=None, limit=100):
        return [] if order is None else [order]

    monkeypatch.setattr(oe.coinbase_client, "get_orders", _get_orders)


def _stub_cancel(monkeypatch, *, raises=False, body=None):
    async def _cancel(order_ids):
        if raises:
            raise RuntimeError("cancel failed")
        return body if body is not None else {"results": [{"success": True}]}

    monkeypatch.setattr(oe.coinbase_client, "cancel_orders", _cancel)


# ── The core fix: no market order on an unconfirmed cancel ────────────────────


async def test_cancel_raises_places_no_market_order(maker, market_calls, monkeypatch):
    _stub_cancel(monkeypatch, raises=True)
    _stub_orders(monkeypatch, {"order_id": "limit-1", "status": "OPEN", "filled_size": "0"})

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == [], "must not add exposure while the limit may still be live"
    assert result["success"] is False
    assert "cancel" in result["reason"].lower()


async def test_cancel_reports_failure_in_body_places_no_market_order(
    maker, market_calls, monkeypatch
):
    """A cancel that fails without raising must be caught too."""
    _stub_cancel(
        monkeypatch,
        body={"results": [{"success": False, "failure_reason": "UNKNOWN_CANCEL_ORDER"}]},
    )
    _stub_orders(monkeypatch, {"order_id": "limit-1", "status": "OPEN", "filled_size": "0"})

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["success"] is False


async def test_order_state_unknown_places_no_market_order(maker, market_calls, monkeypatch):
    """If the order cannot be found after cancelling, its state is unknown."""
    _stub_cancel(monkeypatch)
    _stub_orders(monkeypatch, None)

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["success"] is False


async def test_order_query_failure_places_no_market_order(maker, market_calls, monkeypatch):
    _stub_cancel(monkeypatch)

    async def _boom(product_id=None, order_status=None, limit=100):
        raise RuntimeError("exchange unreachable")

    monkeypatch.setattr(oe.coinbase_client, "get_orders", _boom)

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["success"] is False


# ── The race: the limit filled while we were cancelling ───────────────────────


async def test_order_filled_during_the_race_places_no_market_order(
    maker, market_calls, monkeypatch
):
    """Cancel legitimately fails when the order just filled. That is a MAKER fill,
    not a reason to buy again."""
    _stub_cancel(monkeypatch, raises=True)
    _stub_orders(
        monkeypatch,
        {
            "order_id": "limit-1",
            "status": "FILLED",
            "filled_size": "0.5",
            "average_filled_price": "100.0",
        },
    )

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["success"] is True
    assert result["fill_mode"] == "MAKER"


# ── Confirmed cancel: the fallback is allowed, sized to the remainder ─────────


async def test_confirmed_cancel_with_no_fill_uses_the_full_quote_size(
    maker, market_calls, monkeypatch
):
    _stub_cancel(monkeypatch)
    _stub_orders(monkeypatch, {"order_id": "limit-1", "status": "CANCELLED", "filled_size": "0"})

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert len(market_calls) == 1
    assert market_calls[0]["quote_size"] == 50.0
    assert result["success"] is True
    assert result["fill_mode"] == "TAKER_FALLBACK"


async def test_partial_fill_only_tops_up_the_remainder(maker, market_calls, monkeypatch):
    """0.2 filled at 100.0 = $20 of a $50 order, so only $30 may be bought."""
    _stub_cancel(monkeypatch)
    _stub_orders(
        monkeypatch,
        {
            "order_id": "limit-1",
            "status": "CANCELLED",
            "filled_size": "0.2",
            "average_filled_price": "100.0",
        },
    )

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert len(market_calls) == 1
    assert market_calls[0]["quote_size"] == pytest.approx(30.0)
    assert result["fill_mode"] == "TAKER_FALLBACK"


async def test_partial_fill_remainder_below_minimum_places_nothing(
    maker, market_calls, monkeypatch
):
    """$49.60 of a $50 order filled: the $0.40 remainder is below the $1 floor."""
    _stub_cancel(monkeypatch)
    _stub_orders(
        monkeypatch,
        {
            "order_id": "limit-1",
            "status": "CANCELLED",
            "filled_size": "0.496",
            "average_filled_price": "100.0",
        },
    )

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["success"] is True
    assert result["fill_mode"] == "MAKER_PARTIAL"


async def test_canceled_american_spelling_is_also_accepted(maker, market_calls, monkeypatch):
    """The exchange has used both spellings; neither may block the fallback."""
    _stub_cancel(monkeypatch)
    _stub_orders(monkeypatch, {"order_id": "limit-1", "status": "CANCELED", "filled_size": "0"})

    await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert len(market_calls) == 1


# ── Unchanged behaviour ───────────────────────────────────────────────────────


async def test_a_clean_maker_fill_never_reaches_the_fallback(maker, market_calls, monkeypatch):
    async def _fills(self, pid, order_id, timeout_secs):
        return True

    monkeypatch.setattr(oe.OrderExecutor, "_wait_for_fill", _fills)
    _stub_cancel(monkeypatch, raises=True)  # would explode if reached
    _stub_orders(monkeypatch, None)

    result = await maker.execute_maker_signal(_SIGNAL, timeout_secs=0.01)

    assert market_calls == []
    assert result["fill_mode"] == "MAKER"
    assert result["success"] is True
