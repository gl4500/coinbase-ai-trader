"""TDD tests for #119 — maker (LIMIT post-only) order path with timeout fallback.

Adds OrderExecutor.execute_maker_signal(signal, timeout_secs) which:
  - Places a post_only=True LIMIT at signal.bid (BUY) / signal.ask (SELL)
  - Polls order status until FILLED or timeout
  - On timeout: cancels the limit and places a market order

Cuts fees from 1.20% RT (taker both legs) to 0.80% RT (maker entry, taker exit)
or 0.50% on volume tier 2. The new method is purely additive — no existing
caller is migrated until the user opts in, so blast radius is zero.
"""
import os
import sys

import pytest
from unittest.mock import AsyncMock, patch

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
def signal_buy():
    return {
        "product_id": "BTC-USD",
        "side": "BUY",
        "price": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "quote_size": 50.0,
        "signal_type": "TEST",
    }


@pytest.fixture
def signal_sell():
    return {
        "product_id": "BTC-USD",
        "side": "SELL",
        "price": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "quote_size": 50.0,
        "signal_type": "TEST",
    }


def _make_live_executor():
    from agents.order_executor import OrderExecutor
    ex = OrderExecutor(dry_run=False)
    # The maker-path tests aren't exercising drawdown / preflight; those have
    # their own coverage via test_cnn_agent. Mock them out so the assertions
    # focus on order construction and fill polling.
    ex._check_drawdown = AsyncMock(return_value=None)
    ex._preflight = AsyncMock(return_value=None)
    return ex


class TestExecuteMakerSignal:

    @pytest.mark.asyncio
    async def test_buy_places_post_only_limit_at_bid(self, signal_buy):
        ex = _make_live_executor()
        with patch("agents.order_executor.coinbase_client") as cb, \
             patch("agents.order_executor.database") as db:
            cb.place_limit_order = AsyncMock(return_value={
                "success": True, "success_response": {"order_id": "ord-1"},
            })
            cb.get_orders = AsyncMock(return_value=[
                {"order_id": "ord-1", "status": "FILLED"},
            ])
            db.save_order = AsyncMock()
            db.mark_signal_acted = AsyncMock()

            result = await ex.execute_maker_signal(signal_buy, timeout_secs=1.0)

        assert result["success"] is True
        assert result["fill_mode"] == "MAKER"
        cb.place_limit_order.assert_called_once()
        # signature: place_limit_order(pid, side, base_size, limit_price, post_only=...)
        call = cb.place_limit_order.call_args
        assert call.args[0] == "BTC-USD"
        assert call.args[1] == "BUY"
        assert call.args[3] == 99.5  # maker BUY price = bid
        assert call.kwargs.get("post_only") is True

    @pytest.mark.asyncio
    async def test_sell_places_post_only_limit_at_ask(self, signal_sell):
        ex = _make_live_executor()
        with patch("agents.order_executor.coinbase_client") as cb, \
             patch("agents.order_executor.database") as db:
            cb.place_limit_order = AsyncMock(return_value={
                "success": True, "success_response": {"order_id": "ord-2"},
            })
            cb.get_orders = AsyncMock(return_value=[
                {"order_id": "ord-2", "status": "FILLED"},
            ])
            db.save_order = AsyncMock()
            db.mark_signal_acted = AsyncMock()

            await ex.execute_maker_signal(signal_sell, timeout_secs=1.0)

        call = cb.place_limit_order.call_args
        assert call.args[1] == "SELL"
        assert call.args[3] == 100.5  # maker SELL price = ask
        assert call.kwargs.get("post_only") is True

    @pytest.mark.asyncio
    async def test_timeout_cancels_and_falls_back_to_market(self, signal_buy):
        ex = _make_live_executor()
        with patch("agents.order_executor.coinbase_client") as cb, \
             patch("agents.order_executor.database") as db:
            cb.place_limit_order = AsyncMock(return_value={
                "success": True, "success_response": {"order_id": "ord-3"},
            })
            # Order never fills — always returns OPEN
            cb.get_orders = AsyncMock(return_value=[
                {"order_id": "ord-3", "status": "OPEN"},
            ])
            cb.cancel_orders = AsyncMock(return_value={"success": True})
            cb.place_market_order = AsyncMock(return_value={
                "success": True, "success_response": {"order_id": "mkt-3"},
            })
            db.save_order = AsyncMock()
            db.mark_signal_acted = AsyncMock()
            db.update_order_status = AsyncMock()

            result = await ex.execute_maker_signal(signal_buy, timeout_secs=0.1)

        assert result["success"] is True
        assert result["fill_mode"] == "TAKER_FALLBACK"
        cb.cancel_orders.assert_called_once()
        cb.place_market_order.assert_called_once()
        # market_market_ioc takes quote_size, not base_size
        mkt_call = cb.place_market_order.call_args
        assert mkt_call.args[0] == "BTC-USD"
        assert mkt_call.args[1] == "BUY"
        assert mkt_call.args[2] == 50.0

    @pytest.mark.asyncio
    async def test_dry_run_short_circuits_no_live_calls(self, signal_buy):
        from agents.order_executor import OrderExecutor
        ex = OrderExecutor(dry_run=True)
        ex._check_drawdown = AsyncMock(return_value=None)
        ex._preflight = AsyncMock(return_value=None)

        with patch("agents.order_executor.coinbase_client") as cb, \
             patch("agents.order_executor.database") as db:
            cb.place_limit_order = AsyncMock()
            cb.place_market_order = AsyncMock()
            cb.get_orders = AsyncMock()
            cb.cancel_orders = AsyncMock()
            db.save_order = AsyncMock()
            db.mark_signal_acted = AsyncMock()

            result = await ex.execute_maker_signal(signal_buy, timeout_secs=1.0)

        assert result["success"] is True
        assert result["dry_run"] is True
        cb.place_limit_order.assert_not_called()
        cb.place_market_order.assert_not_called()
        cb.get_orders.assert_not_called()
        cb.cancel_orders.assert_not_called()


class TestMakerPriceHelper:
    """Pure helper: bid for BUY, ask for SELL."""

    def test_buy_returns_bid(self):
        from agents.order_executor import _maker_price
        assert _maker_price("BUY", bid=99.5, ask=100.5) == 99.5

    def test_sell_returns_ask(self):
        from agents.order_executor import _maker_price
        assert _maker_price("SELL", bid=99.5, ask=100.5) == 100.5

    def test_case_insensitive_side(self):
        from agents.order_executor import _maker_price
        assert _maker_price("buy", 10.0, 11.0) == 10.0
        assert _maker_price("sell", 10.0, 11.0) == 11.0
