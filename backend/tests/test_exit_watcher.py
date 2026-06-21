"""Tests for agents/exit_watcher.py — WS-driven trail-stop / stop-loss exits.

Mock-only: no live WS, no real DB. Safe to write while 8001 is trading;
only the pytest invocation itself is gated by feedback_no_pytest_during_trading.
"""
import asyncio
import logging
import os
import sys
import time
from unittest.mock import AsyncMock, patch

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
    """In-memory stand-in for _CNNBook. Tests need .positions + .sell + .balance
    (balance was added for task #46 PnL-anchored trail; default $1000 keeps
    pre-#46 tests passing)."""

    def __init__(self):
        self.positions: dict = {}
        self.sell = AsyncMock(return_value=0.0)
        self.balance: float = 1000.0   # default for pre-#46 tests


def _make_pos(*, avg=100.0, peak=100.0, trail=0.06,
              peak_pnl_pct=None, position_dollars=None, size=1.0):
    """Build a position dict. Defaults match pre-#46 behavior; pass
    peak_pnl_pct/position_dollars explicitly for #46 tests."""
    if peak_pnl_pct is None:
        peak_pnl_pct = ((peak - avg) / avg) if avg > 0 else 0.0
    if position_dollars is None:
        position_dollars = float(size * avg)
    return {
        "size": size, "avg_price": avg, "peak_price": peak, "trail_pct": trail,
        "peak_pnl_pct": peak_pnl_pct, "position_dollars": position_dollars,
    }


class TestOnPriceTick:

    @pytest.mark.asyncio
    async def test_no_position_returns_immediately(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        await on_price_tick("BTC-USD", 100.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_trail_stop_fires_when_price_below_threshold(self):
        # B2: peak=110, avg=100 → peak_pnl 10% → threshold = 10 - 1.2 = 8.8%.
        # price=107 → pnl=7% < 8.8% threshold → fire.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=110.0, trail=0.06)
        await on_price_tick("BTC-USD", 107.0, book)
        book.sell.assert_called_once_with("BTC-USD", 107.0, trigger="WS_TRAIL_STOP")

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
        # B2: peak above entry so the new gain-proportional trail engages
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=110.0, trail=0.06)

        attach(ws, book)

        fake_msg = {
            "channel": "ticker",
            "events": [{
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price":      "107.0",
                    "best_bid":   "106.99",
                    "best_ask":   "107.01",
                }],
            }],
        }
        await ws._handle(fake_msg)

        # _price_handlers fire via asyncio.create_task — yield once so the
        # spawned task runs before we assert.
        await asyncio.sleep(0.05)

        book.sell.assert_called_once_with("BTC-USD", 107.0, trigger="WS_TRAIL_STOP")


# ═══════════════════════════════════════════════════════════════════════════
# Task #46 — PnL-anchored trail behavior
# ═══════════════════════════════════════════════════════════════════════════

class TestOnPriceTickPnLAnchored:
    """on_price_tick uses _compute_exit_threshold from exit_thresholds and
    fires WS_TRAIL_STOP / WS_STOP_LOSS per the new threshold."""

    @pytest.mark.asyncio
    async def test_on_price_tick_fires_ws_trail_when_pnl_below_threshold(self):
        """DASH-like: peak_pnl=7.8%, atr=6%, pos $58, capital $1k.
        Layer 3 baseline (0.078 - 0.06 = 0.018) wins over break-even 0.012.
        Price $43.30 = +0.77% pnl < 1.8% threshold → WS_TRAIL_STOP fires."""
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06, size=1.34,
        )
        await on_price_tick("DASH-USD", 43.30, book)   # pnl ≈ +0.77%
        book.sell.assert_called_once()
        _args, kwargs = book.sell.call_args
        assert kwargs.get("trigger") == "WS_TRAIL_STOP"

    @pytest.mark.asyncio
    async def test_on_price_tick_does_not_fire_above_threshold(self):
        """Same DASH setup, price $44.04 = +2.5% pnl > 1.8% threshold → no fire."""
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06, size=1.34,
        )
        await on_price_tick("DASH-USD", 46.20, book)   # +7.5% > 6.6% threshold
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_price_tick_fires_ws_model_down_when_p_down_above_threshold(self):
        """B2 MODEL_DOWN: WS handler fires WS_MODEL_DOWN when cached p_down > threshold."""
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05   # 0.60
        book.positions["ABC-USD"]["p_down_ts_ms"] = int(time.time() * 1000)   # fresh (task #80)
        await on_price_tick("ABC-USD", 101.0, book)   # +1% PnL — trail wouldn't fire
        book.sell.assert_called_once()
        _args, kwargs = book.sell.call_args
        assert kwargs.get("trigger") == "WS_MODEL_DOWN"

    @pytest.mark.asyncio
    async def test_on_price_tick_does_not_fire_ws_model_down_below_threshold(self):
        """B2 MODEL_DOWN: p_down below threshold → no model-driven WS exit."""
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=110.0, trail=0.06)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD - 0.05   # 0.50
        book.positions["ABC-USD"]["p_down_ts_ms"] = int(time.time() * 1000)   # fresh (task #80)
        # Price near peak — trail also won't fire
        await on_price_tick("ABC-USD", 109.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_price_tick_does_not_fire_ws_model_down_when_p_down_stale(self):
        """Task #80: cached p_down > threshold but ts > _P_DOWN_STALE_MS old → no WS_MODEL_DOWN.
        WS path is most exposed: ticks arrive ~5-10/sec; cached p_down can age 60s between
        scans while the market reverses. Staleness gate prevents spurious exits."""
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD, _P_DOWN_STALE_MS
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=110.0, trail=0.06)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05   # would fire if fresh
        book.positions["ABC-USD"]["p_down_ts_ms"] = int(time.time() * 1000) - _P_DOWN_STALE_MS - 1000
        # Price near peak — trail wouldn't fire either
        await on_price_tick("ABC-USD", 109.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_price_tick_does_not_fire_ws_model_down_when_ts_missing(self):
        """Task #80: legacy position has p_down but no p_down_ts_ms → treated as stale, no fire."""
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=110.0, trail=0.06)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05   # no ts_ms set
        await on_price_tick("ABC-USD", 109.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_price_tick_ratchets_peak_pnl_pct(self):
        """Price prints new high → peak_pnl_pct ratchets up to match."""
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06, size=1.34,
        )
        await on_price_tick("DASH-USD", 48.0, book)
        expected_pnl = (48.0 - 42.97) / 42.97
        assert book.positions["DASH-USD"]["peak_pnl_pct"] == pytest.approx(expected_pnl)


# ═══════════════════════════════════════════════════════════════════════════
# Maker-execution exit leg — WS live routing
# ═══════════════════════════════════════════════════════════════════════════

class TestOnPriceTickLiveRouting:
    """on_price_tick places a live exit after the paper close, routed
    maker/taker by trigger, only when USE_MAKER_EXECUTION is on + executor live."""

    def _live_executor(self):
        ex = AsyncMock()
        ex.dry_run = False
        ex.execute_maker_signal.return_value = {"success": True, "fill_mode": "MAKER"}
        ex.execute_signal.return_value = {"success": True, "fill_mode": "TAKER"}
        return ex

    @pytest.mark.asyncio
    async def test_ws_trail_stop_routes_maker(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            await on_price_tick("BTC-USD", 105.0, book, ex)   # +5% << trail thr
        book.sell.assert_called_once_with("BTC-USD", 105.0, trigger="WS_TRAIL_STOP")
        ex.execute_maker_signal.assert_awaited_once()
        sig = ex.execute_maker_signal.call_args.args[0]
        assert sig["side"] == "SELL" and sig["signal_type"] == "WS_TRAIL_STOP"
        assert sig["quote_size"] == round(2.0 * 106.0, 2)

    @pytest.mark.asyncio
    async def test_ws_stop_loss_routes_taker(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["ETH-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await on_price_tick("ETH-USD", 91.0, book, ex)    # -9% -> WS_STOP_LOSS
        book.sell.assert_called_once_with("ETH-USD", 91.0, trigger="WS_STOP_LOSS")
        mock_bba.assert_not_called()
        ex.execute_signal.assert_awaited_once()
        ex.execute_maker_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_ws_model_down_routes_maker(self):
        import time as _t
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06, size=2.0)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05
        book.positions["ABC-USD"]["p_down_ts_ms"] = int(_t.time() * 1000)
        ex = self._live_executor()
        quotes = {"ABC-USD": {"bid": 100.5, "ask": 101.5}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            await on_price_tick("ABC-USD", 101.0, book, ex)   # +1% -> WS_MODEL_DOWN
        book.sell.assert_called_once_with("ABC-USD", 101.0, trigger="WS_MODEL_DOWN")
        ex.execute_maker_signal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flag_off_stays_paper_only(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", False, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await on_price_tick("BTC-USD", 105.0, book, ex)
        book.sell.assert_called_once()
        mock_bba.assert_not_called()
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_forwards_order_executor(self):
        import agents.exit_execution as exit_mod
        from services.ws_subscriber import CoinbaseWSSubscriber
        from agents.exit_watcher import attach
        ws = CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            attach(ws, book, ex)
            await ws._handle({
                "channel": "ticker",
                "events": [{"tickers": [{"product_id": "BTC-USD", "price": "105.0"}]}],
            })
            await asyncio.sleep(0.05)
        ex.execute_maker_signal.assert_awaited_once()
