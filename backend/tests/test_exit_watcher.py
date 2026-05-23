"""Tests for agents/exit_watcher.py — WS-driven trail-stop / stop-loss exits.

Mock-only: no live WS, no real DB. Safe to write while 8001 is trading;
only the pytest invocation itself is gated by feedback_no_pytest_during_trading.
"""
import asyncio
import logging
import os
import sys
from unittest.mock import AsyncMock

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")


class _FakeBook:
    """In-memory stand-in for _CNNBook. Tests only need .positions + .sell."""

    def __init__(self):
        self.positions: dict = {}
        self.sell = AsyncMock(return_value=0.0)


def _make_pos(*, avg=100.0, peak=100.0, trail=0.06):
    return {"size": 1.0, "avg_price": avg, "peak_price": peak, "trail_pct": trail}


class TestOnPriceTick:

    @pytest.mark.asyncio
    async def test_no_position_returns_immediately(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        await on_price_tick("BTC-USD", 100.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_trail_stop_fires_when_price_below_threshold(self):
        # peak=100, trail=6% → trail threshold = 94. price=93 triggers.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 93.0, book)
        book.sell.assert_called_once_with("BTC-USD", 93.0, trigger="WS_TRAIL_STOP")

    @pytest.mark.asyncio
    async def test_stop_loss_fires_when_price_below_threshold(self):
        # avg=100, _CNN_STOP_LOSS_PCT=8% → stop threshold = 92. price=91 triggers.
        # peak=100 so trail_threshold=94; both would fire but stop-loss has priority.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 91.0, book)
        book.sell.assert_called_once_with("BTC-USD", 91.0, trigger="WS_STOP_LOSS")

    @pytest.mark.asyncio
    async def test_stop_loss_priority_over_trail(self):
        # Both triggered at price=85: stop-loss must fire, trail must NOT.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 85.0, book)
        assert book.sell.call_count == 1
        book.sell.assert_called_with("BTC-USD", 85.0, trigger="WS_STOP_LOSS")

    @pytest.mark.asyncio
    async def test_peak_ratchets_on_new_high(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 105.0, book)
        assert book.positions["BTC-USD"]["peak_price"] == 105.0
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_swallows_exceptions(self, caplog):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        book.sell.side_effect = RuntimeError("simulated DB failure")
        with caplog.at_level(logging.ERROR, logger="agents.exit_watcher"):
            # Must NOT re-raise — invariant #18
            await on_price_tick("BTC-USD", 91.0, book)
        assert any(r.levelno == logging.ERROR for r in caplog.records), (
            "exit_watcher must log at ERROR when sell() raises"
        )


class TestAttach:
    """End-to-end: attach() registers the handler with CoinbaseWSSubscriber,
    and a ticker message routed through ws._handle() reaches book.sell().
    Verifies wiring without opening a real socket.
    """

    @pytest.mark.asyncio
    async def test_attach_registers_handler_and_dispatches_ticks(self):
        from services.ws_subscriber import CoinbaseWSSubscriber
        from agents.exit_watcher import attach

        ws = CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)

        attach(ws, book)

        fake_msg = {
            "channel": "ticker",
            "events": [{
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price":      "93.0",
                    "best_bid":   "92.99",
                    "best_ask":   "93.01",
                }],
            }],
        }
        await ws._handle(fake_msg)

        # _price_handlers fire via asyncio.create_task — yield once so the
        # spawned task runs before we assert.
        await asyncio.sleep(0.05)

        book.sell.assert_called_once_with("BTC-USD", 93.0, trigger="WS_TRAIL_STOP")
