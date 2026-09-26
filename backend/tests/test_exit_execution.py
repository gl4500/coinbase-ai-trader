"""Tests for agents/exit_execution.py — maker/taker routing for live exits.

Mock-only: no live network, no real DB. Safe to write during 8001 trading;
only the pytest invocation itself is gated by feedback_no_pytest_during_trading.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import agents.exit_execution as exit_mod
from agents.exit_execution import execute_live_exit, is_maker_exit


def _live_executor():
    ex = AsyncMock()
    ex.dry_run = False
    ex.execute_maker_signal.return_value = {"success": True, "fill_mode": "MAKER"}
    ex.execute_signal.return_value = {"success": True, "fill_mode": "TAKER"}
    return ex


class TestClassification:
    def test_maker_triggers(self):
        for t in ("TRAIL_STOP", "WS_TRAIL_STOP", "MODEL_DOWN", "WS_MODEL_DOWN"):
            assert is_maker_exit(t) is True

    def test_taker_triggers(self):
        for t in ("STOP_LOSS", "WS_STOP_LOSS", "MAX_HOLD", "LEGACY_EXIT"):
            assert is_maker_exit(t) is False


class TestGate:
    @pytest.mark.asyncio
    async def test_none_executor_noops(self):
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True):
            result = await execute_live_exit(
                None, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP"
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_flag_off_noops(self):
        ex = _live_executor()
        with (
            patch.object(exit_mod.config, "use_maker_execution", False, create=True),
            patch.object(exit_mod.coinbase_client, "get_best_bid_ask", new=AsyncMock()) as mock_bba,
        ):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP"
            )
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()
        mock_bba.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_executor_noops(self):
        ex = _live_executor()
        ex.dry_run = True
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP"
            )
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()


class TestMakerRouting:
    @pytest.mark.asyncio
    async def test_maker_trigger_routes_with_quotes_and_sizing(self):
        ex = _live_executor()
        quotes = {"BTC-USD": {"bid": 99.0, "ask": 101.0, "price": 100.0}}
        with (
            patch.object(exit_mod.config, "use_maker_execution", True, create=True),
            patch.object(
                exit_mod.coinbase_client, "get_best_bid_ask", new=AsyncMock(return_value=quotes)
            ) as mock_bba,
        ):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=2.0, trigger="TRAIL_STOP"
            )

        mock_bba.assert_awaited_once_with(["BTC-USD"])
        ex.execute_signal.assert_not_called()
        ex.execute_maker_signal.assert_awaited_once()
        sig = ex.execute_maker_signal.call_args.args[0]
        assert sig["side"] == "SELL"
        assert sig["product_id"] == "BTC-USD"
        assert sig["signal_type"] == "TRAIL_STOP"
        assert sig["bid"] == 99.0 and sig["ask"] == 101.0
        assert sig["quote_size"] == round(2.0 * 101.0, 2)  # size * ask
        assert "atr" not in sig
        assert result == {"success": True, "fill_mode": "MAKER"}

    @pytest.mark.asyncio
    async def test_maker_trigger_missing_quotes_noops(self):
        ex = _live_executor()
        quotes = {"BTC-USD": {"bid": 0.0, "ask": 0.0}}
        with (
            patch.object(exit_mod.config, "use_maker_execution", True, create=True),
            patch.object(
                exit_mod.coinbase_client, "get_best_bid_ask", new=AsyncMock(return_value=quotes)
            ),
        ):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="MODEL_DOWN"
            )
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()


class TestTakerRouting:
    @pytest.mark.asyncio
    async def test_taker_trigger_routes_without_quote_fetch(self):
        ex = _live_executor()
        with (
            patch.object(exit_mod.config, "use_maker_execution", True, create=True),
            patch.object(exit_mod.coinbase_client, "get_best_bid_ask", new=AsyncMock()) as mock_bba,
        ):
            result = await execute_live_exit(
                ex, pid="ETH-USD", price=50.0, size=3.0, trigger="STOP_LOSS"
            )

        mock_bba.assert_not_called()
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_awaited_once()
        sig = ex.execute_signal.call_args.args[0]
        assert sig["side"] == "SELL"
        assert sig["product_id"] == "ETH-USD"
        assert sig["signal_type"] == "STOP_LOSS"
        assert sig["quote_size"] == round(3.0 * 50.0, 2)  # size * price
        assert "bid" not in sig and "ask" not in sig and "atr" not in sig
        assert result == {"success": True, "fill_mode": "TAKER"}
