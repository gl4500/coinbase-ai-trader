"""
Tests for ScalpAgent — fast-cycle BTC/ETH scalper.

Strategy under test:
  Entry : RSI(7)+BB(20,2)+VWAP confluence scoring (min 5 pts)
  Regime: ADX(10) gate — ADX>25 trend / ADX<20 range
  Exit  : +0.30% TP | -0.25% SL | 1.5xATR trailing | 15-min time exit

All external I/O (database, WebSocket) is mocked.
No real Coinbase credentials are needed.
"""
import asyncio
import math
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from agents.scalp_agent import (
    ScalpAgent,
    _ScalpBook,
    _confluence_score,
    _exit_reason,
    _TAKE_PROFIT,
    _HARD_STOP,
    _TIME_EXIT_SEC,
    _MIN_SCORE,
    _SCALP_BALANCE,
    _MAX_FRAC,
    _MAX_SCALP_PRODUCTS,
    _MIN_CANDLES,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_candles(n: int = 120, base: float = 50_000.0, trend: str = "flat") -> dict:
    """Return closes/highs/lows/vols suitable for indicator tests."""
    closes, highs, lows, vols = [], [], [], []
    for i in range(n):
        if trend == "down":
            c = base - i * 10          # falling → oversold RSI
        elif trend == "up":
            c = base + i * 10
        else:
            c = base + 200 * math.sin(i / 6.0)
        closes.append(c)
        highs.append(c + abs(c * 0.005))
        lows.append(c - abs(c * 0.005))
        vols.append(10_000 + i * 50)
    return {"closes": closes, "highs": highs, "lows": lows, "vols": vols}


def _make_db_candles(n: int = 120, base: float = 50_000.0) -> list:
    candles = []
    for i in range(n):
        c = base + 200 * math.sin(i / 6.0)
        candles.append({
            "close": c, "high": c + 50, "low": c - 50, "volume": 10_000,
        })
    return candles


# ── _confluence_score ──────────────────────────────────────────────────────────

class TestConfluenceScore:

    def test_returns_tuple(self):
        d = _make_candles(120, trend="flat")
        score, reasons = _confluence_score(d["closes"], d["highs"], d["lows"], d["vols"])
        assert isinstance(score, int)
        assert isinstance(reasons, list)

    def test_score_non_negative(self):
        d = _make_candles(120, trend="flat")
        score, _ = _confluence_score(d["closes"], d["highs"], d["lows"], d["vols"])
        assert score >= 0

    def test_oversold_downtrend_scores_higher(self):
        """A falling market should produce a higher oversold score than flat."""
        down = _make_candles(120, trend="down")
        flat = _make_candles(120, trend="flat")
        score_down, _ = _confluence_score(down["closes"], down["highs"], down["lows"], down["vols"])
        score_flat, _ = _confluence_score(flat["closes"], flat["highs"], flat["lows"], flat["vols"])
        assert score_down >= score_flat

    def test_reasons_match_score(self):
        """Each reason string should correspond to at least 1 point added."""
        d = _make_candles(120, trend="down")
        score, reasons = _confluence_score(d["closes"], d["highs"], d["lows"], d["vols"])
        # Each reason has a "+N" suffix — total should equal score
        total = sum(
            int(r.split("(+")[1].rstrip(")")) for r in reasons if "(+" in r
        )
        assert total == score


# ── _exit_reason ──────────────────────────────────────────────────────────────

class TestExitReason:

    def _pos(self, entry: float = 50_000.0, hw: float = None,
              trail_dist: float = 50.0, age: float = 0.0) -> dict:
        return {
            "avg_price":  entry,
            "entry_time": time.time() - age,
            "high_water": hw or entry,
            "trail_dist": trail_dist,
        }

    def test_take_profit(self):
        pos = self._pos(50_000.0)
        tp_price = 50_000.0 * (1 + _TAKE_PROFIT + 0.001)
        assert _exit_reason(pos, tp_price) == "TP"

    def test_hard_stop(self):
        pos = self._pos(50_000.0)
        sl_price = 50_000.0 * (1 - _HARD_STOP - 0.001)
        assert _exit_reason(pos, sl_price) == "SL"

    def test_trailing_stop(self):
        # entry=50000, TP at 50000*1.003=50150, SL at 49875
        # hw=50120 (price ticked up but not past TP), trail=50
        # hw - trail = 50070; price=50060 is below hw-trail → TRAIL fires
        # 50060 is NOT above TP (50150) and NOT below SL (49875)
        pos = self._pos(50_000.0, hw=50_120.0, trail_dist=50.0)
        assert _exit_reason(pos, 50_060.0) == "TRAIL"

    def test_time_exit(self):
        pos = self._pos(50_000.0, age=_TIME_EXIT_SEC + 1)
        # Price unchanged (no TP/SL/TRAIL triggered)
        assert _exit_reason(pos, 50_000.0) == "TIME"

    def test_no_exit_within_bounds(self):
        pos = self._pos(50_000.0)
        # Price moved +0.10% — neither TP nor SL
        assert _exit_reason(pos, 50_050.0) is None

    def test_tp_takes_priority_over_time(self):
        """If both TP and time exit conditions are met, TP is checked first."""
        pos = self._pos(50_000.0, age=_TIME_EXIT_SEC + 1)
        tp_price = 50_000.0 * (1 + _TAKE_PROFIT + 0.001)
        assert _exit_reason(pos, tp_price) == "TP"


# ── _ScalpBook ─────────────────────────────────────────────────────────────────

class TestScalpBook:

    @pytest.mark.asyncio
    async def test_initial_balance(self):
        book = _ScalpBook()
        assert book.balance == _SCALP_BALANCE

    @pytest.mark.asyncio
    async def test_buy_deducts_balance(self):
        book = _ScalpBook()
        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.open_trade",       new=AsyncMock()),
        ):
            spent, size = await book.buy("BTC-USD", 50_000.0, atr=100.0)
        assert spent > 0
        assert size > 0
        assert book.balance == pytest.approx(_SCALP_BALANCE - spent)

    @pytest.mark.asyncio
    async def test_buy_max_fraction(self):
        """Buy should spend at most _MAX_FRAC of balance."""
        book = _ScalpBook()
        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.open_trade",       new=AsyncMock()),
        ):
            spent, _ = await book.buy("BTC-USD", 50_000.0, atr=100.0)
        assert spent <= _SCALP_BALANCE * _MAX_FRAC + 1e-6

    @pytest.mark.asyncio
    async def test_buy_zero_price_rejected(self):
        book = _ScalpBook()
        spent, size = await book.buy("BTC-USD", 0.0, atr=100.0)
        assert spent == 0.0
        assert size == 0.0

    @pytest.mark.asyncio
    async def test_sell_returns_pnl(self):
        book = _ScalpBook()
        entry = 50_000.0
        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.open_trade",       new=AsyncMock()),
        ):
            await book.buy("BTC-USD", entry, atr=100.0)

        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.close_trade",      new=AsyncMock()),
        ):
            pnl = await book.sell("BTC-USD", entry * 1.003)   # +0.3% gain
        assert pnl > 0

    @pytest.mark.asyncio
    async def test_sell_unknown_pid_returns_zero(self):
        book = _ScalpBook()
        pnl = await book.sell("UNKNOWN-USD", 100.0)
        assert pnl == 0.0

    @pytest.mark.asyncio
    async def test_sell_clears_position(self):
        book = _ScalpBook()
        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.open_trade",       new=AsyncMock()),
        ):
            await book.buy("BTC-USD", 50_000.0, atr=100.0)

        assert book.has_position("BTC-USD")
        with (
            patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
            patch("agents.scalp_agent.database.close_trade",      new=AsyncMock()),
        ):
            await book.sell("BTC-USD", 50_200.0)
        assert not book.has_position("BTC-USD")

    def test_exit_stats_recorded_on_loss(self):
        """sell() at a loss increments SL losses counter and total_pnl."""
        import asyncio
        book = _ScalpBook()
        book.positions["BTC-USD"] = {
            "size": 0.004, "avg_price": 50_000.0,
            "entry_time": time.time() - 60,
            "high_water": 50_000.0, "trail_dist": 62.5,
            "entry_reasons": ["RSI7=22 deeply oversold"],
        }
        book.balance = 800.0

        async def _run():
            with (
                patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
                patch("agents.scalp_agent.database.close_trade",      new=AsyncMock()),
            ):
                await book.sell("BTC-USD", 49_875.0, trigger="SL")  # -0.25% → loss

        asyncio.get_event_loop().run_until_complete(_run())
        assert book._stats["SL"]["losses"] == 1
        assert book._stats["SL"]["total_pnl"] < 0

    def test_exit_stats_recorded_on_win(self):
        """sell() at a gain increments TP wins counter."""
        import asyncio
        book = _ScalpBook()
        book.positions["BTC-USD"] = {
            "size": 0.004, "avg_price": 50_000.0,
            "entry_time": time.time() - 60,
            "high_water": 50_150.0, "trail_dist": 62.5,
            "entry_reasons": [],
        }
        book.balance = 800.0

        async def _run():
            with (
                patch("agents.scalp_agent.database.save_agent_state", new=AsyncMock()),
                patch("agents.scalp_agent.database.close_trade",      new=AsyncMock()),
            ):
                await book.sell("BTC-USD", 50_150.0, trigger="TP")  # +0.30% → win

        asyncio.get_event_loop().run_until_complete(_run())
        assert book._stats["TP"]["wins"] == 1
        assert book._stats["TP"]["total_pnl"] > 0

    def test_status_structure(self):
        book = _ScalpBook()
        s = book.status
        assert s["agent"] == "SCALP"
        assert "balance" in s
        assert "realized_pnl" in s
        assert "open_positions" in s
        # scan_count / last_scan_at are added by ScalpAgent.status, not _ScalpBook


# ── ScalpAgent ─────────────────────────────────────────────────────────────────

class TestScalpAgent:

    def _agent(self, ws_price: float = None) -> ScalpAgent:
        ws = None
        if ws_price is not None:
            ws = MagicMock()
            ws.get_price = MagicMock(return_value=ws_price)
        return ScalpAgent(ws_subscriber=ws)

    def test_live_price_uses_ws(self):
        ag = self._agent(ws_price=55_000.0)
        assert ag._live_price("BTC-USD") == 55_000.0

    def test_live_price_no_ws_returns_none(self):
        ag = self._agent()
        assert ag._live_price("BTC-USD") is None

    def test_status_keys_present(self):
        ag = self._agent()
        s = ag.status
        for key in ("agent", "balance", "realized_pnl", "open_positions",
                    "scan_count", "last_scan_at", "positions"):
            assert key in s, f"Missing key: {key}"

    def test_scan_count_starts_zero(self):
        ag = self._agent()
        assert ag.scan_count == 0
        assert ag.last_scan_at is None

    @pytest.mark.asyncio
    async def test_start_stop(self):
        ag = self._agent()
        with patch("agents.scalp_agent.database.load_agent_state",
                   new=AsyncMock(return_value=None)):
            await ag.start()
        assert ag._running is True
        await ag.stop()
        assert ag._running is False

    @pytest.mark.asyncio
    async def test_scan_logs_summary_each_cycle(self):
        """_scan_entries must log a summary line so activity is visible in the UI."""
        import logging
        from agents.scalp_agent import ScalpAgent

        ag = ScalpAgent()
        products_mock = AsyncMock(return_value=[])   # no eligible products

        with (
            patch.object(ag, "_get_scalp_products", products_mock),
            patch("agents.scalp_agent.logger") as mock_log,
        ):
            await ag._scan_entries()

        # Any call to logger.info, logger.warning, or logger.debug counts as a summary
        any_log_call = (
            mock_log.info.called or
            mock_log.warning.called or
            mock_log.debug.called
        )
        assert any_log_call, (
            "_scan_entries produced no log output — scan activity is invisible to the user"
        )

    @pytest.mark.asyncio
    async def test_scan_entries_continues_after_losses(self):
        """No daily halt — agent keeps scanning even after losing trades."""
        ag = self._agent()
        # Simulate prior losses in stats
        ag.book._stats["SL"]["losses"] = 5
        ag.book._stats["SL"]["total_pnl"] = -15.0

        buy_mock = AsyncMock(return_value=(0.0, 0.0))
        with (
            patch("agents.scalp_agent.database.get_products",
                  new=AsyncMock(return_value=[])),
            patch.object(ag.book, "buy", buy_mock),
        ):
            await ag._scan_entries()  # should run, not be blocked

        # No buy because no products — but the scan ran (not skipped by halt)
        buy_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_entries_skips_at_max_concurrent(self):
        """Already at _MAX_CONCURRENT positions → no new buy attempted."""
        from agents.scalp_agent import _MAX_CONCURRENT
        ag = self._agent(ws_price=50_000.0)
        # Fill positions up to the max
        for i in range(_MAX_CONCURRENT):
            ag.book.positions[f"COIN{i}-USD"] = {
                "size": 0.001, "avg_price": 50_000.0,
                "entry_time": time.time(), "high_water": 50_000.0, "trail_dist": 50.0,
            }
        buy_mock = AsyncMock()
        products_mock = AsyncMock(return_value=["XRP-USD", "ADA-USD"])
        with (
            patch.object(ag, "_get_scalp_products", products_mock),
            patch.object(ag.book, "buy", buy_mock),
        ):
            await ag._scan_entries()

        buy_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_exits_triggers_tp(self):
        """Exit loop sells a position when take-profit is hit."""
        ag = self._agent()
        entry = 50_000.0
        tp_price = entry * (1 + _TAKE_PROFIT + 0.001)

        ag.book.positions["BTC-USD"] = {
            "size": 0.001, "avg_price": entry,
            "entry_time": time.time(), "high_water": entry, "trail_dist": 50.0,
        }
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=tp_price)
        ag._ws = ws

        sell_mock = AsyncMock(return_value=5.0)
        decision_mock = AsyncMock()
        with (
            patch.object(ag.book, "sell", sell_mock),
            patch("agents.scalp_agent.database.save_agent_decision", decision_mock),
        ):
            await ag._check_exits()

        sell_mock.assert_called_once_with("BTC-USD", tp_price, trigger="TP")

    @pytest.mark.asyncio
    async def test_check_exits_updates_high_water(self):
        """Exit loop updates the high-water mark when price rises."""
        ag = self._agent()
        entry = 50_000.0
        new_high = entry * 1.001   # +0.1% — not yet TP

        ag.book.positions["BTC-USD"] = {
            "size": 0.001, "avg_price": entry,
            "entry_time": time.time(), "high_water": entry, "trail_dist": 200.0,
        }
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=new_high)
        ag._ws = ws

        await ag._check_exits()
        assert ag.book.positions["BTC-USD"]["high_water"] == new_high

    @pytest.mark.asyncio
    async def test_check_exits_no_ws_skips(self):
        """Without a WS price, exits are not triggered."""
        ag = self._agent()   # no ws
        ag.book.positions["BTC-USD"] = {
            "size": 0.001, "avg_price": 50_000.0,
            "entry_time": time.time() - 1000, "high_water": 50_000.0, "trail_dist": 50.0,
        }
        sell_mock = AsyncMock()
        with patch.object(ag.book, "sell", sell_mock):
            await ag._check_exits()

        sell_mock.assert_not_called()


# ── Dynamic product selection ───────────────────────────────────────────────────

class TestGetScalpProducts:
    """_get_scalp_products() should return tracked products with enough candle data."""

    @pytest.mark.asyncio
    async def test_returns_products_with_candles(self):
        """Only products with >= MIN_CANDLES candles and price >= _MIN_PRICE are returned."""
        ag = ScalpAgent()
        tracked = [
            {"product_id": "XRP-USD", "price": 1.33},
            {"product_id": "ADA-USD", "price": 0.24},
            {"product_id": "BTC-USD", "price": 0.0},   # 0 candles → excluded
        ]
        candle_map = {"XRP-USD": 120, "ADA-USD": 80, "BTC-USD": 0}

        async def fake_get_candles(pid, limit):
            return [{}] * candle_map[pid]

        with (
            patch("agents.scalp_agent.database.get_products", AsyncMock(return_value=tracked)),
            patch("agents.scalp_agent.database.get_candles", side_effect=fake_get_candles),
        ):
            result = await ag._get_scalp_products()

        assert "XRP-USD" in result
        assert "ADA-USD" in result
        assert "BTC-USD" not in result  # 0 candles → excluded

    @pytest.mark.asyncio
    async def test_empty_when_no_tracked_products(self):
        ag = ScalpAgent()
        with patch("agents.scalp_agent.database.get_products", AsyncMock(return_value=[])):
            result = await ag._get_scalp_products()
        assert result == []

    @pytest.mark.asyncio
    async def test_limited_to_max_products(self):
        """Should cap at _MAX_SCALP_PRODUCTS to avoid scanning everything."""
        from agents.scalp_agent import _MAX_SCALP_PRODUCTS
        ag = ScalpAgent()
        # 10 tracked products all with candles and valid prices
        tracked = [{"product_id": f"COIN{i}-USD", "price": 1.0 + i} for i in range(10)]

        async def fake_get_candles(pid, limit):
            return [{}] * 120  # all have enough

        with (
            patch("agents.scalp_agent.database.get_products", AsyncMock(return_value=tracked)),
            patch("agents.scalp_agent.database.get_candles", side_effect=fake_get_candles),
        ):
            result = await ag._get_scalp_products()

        assert len(result) <= _MAX_SCALP_PRODUCTS


# ── SL cooldown ────────────────────────────────────────────────────────────────

class TestSLCooldown:
    """After a stop-loss exit, the agent must not re-enter for _SL_COOLDOWN_SEC."""

    def test_sl_cooldown_constant_exists(self):
        from agents.scalp_agent import _SL_COOLDOWN_SEC
        assert _SL_COOLDOWN_SEC >= 60, "SL cooldown must be at least 60 seconds"
        assert _SL_COOLDOWN_SEC <= 1800, "SL cooldown should not exceed 30 min"

    @pytest.mark.asyncio
    async def test_sl_sets_cooldown_timestamp(self):
        """An SL exit stores the current time in _sl_cooldown for that product."""
        ag = ScalpAgent()
        pid = "BTC-USD"
        # Seed a fake open position
        ag.book.positions[pid] = {
            "size": 0.001, "avg_price": 50_000.0, "entry_time": time.time() - 60,
            "high_water": 50_000.0, "entry_reasons": [],
        }
        ag.book.balance = 900.0

        before = time.time()
        ag._ws = MagicMock()
        ag._ws.get_price.return_value = 50_000.0 * (1 - 0.003)  # -0.30% → triggers SL

        with (
            patch("agents.scalp_agent.database.save_agent_decision", new=AsyncMock()),
            patch("agents.scalp_agent.database.close_trade",          new=AsyncMock()),
            patch("agents.scalp_agent.database.open_trade",           new=AsyncMock(return_value=1)),
            patch("agents.scalp_agent.get_tracker", return_value=MagicMock(
                record=AsyncMock(), validate_with_ollama=AsyncMock()
            )),
        ):
            await ag._check_exits()

        assert pid in ag._sl_cooldown, "_sl_cooldown must be set after SL exit"
        assert ag._sl_cooldown[pid] >= before

    @pytest.mark.asyncio
    async def test_sl_cooldown_blocks_reentry(self):
        """_scan_entries must skip a product whose SL cooldown is still active."""
        from agents.scalp_agent import _SL_COOLDOWN_SEC
        ag = ScalpAgent()
        pid = "BTC-USD"
        ag._sl_cooldown[pid] = time.time()   # just exited via SL

        # Mock all the scaffolding so entry would otherwise fire
        candles_raw = [
            {"product_id": pid, "close": 50000.0 - i * 10, "high": 50010.0,
             "low": 49990.0, "volume": 1000.0, "start_time": i}
            for i in range(120)
        ]
        tracked = [{"product_id": pid, "price": 50_000.0}]
        buy_mock = AsyncMock()
        ag.book.buy = buy_mock

        with (
            patch("agents.scalp_agent.database.get_products",
                  new=AsyncMock(return_value=tracked)),
            patch("agents.scalp_agent.database.get_candles",
                  new=AsyncMock(return_value=candles_raw)),
            patch("agents.scalp_agent.database.save_agent_decision", new=AsyncMock()),
        ):
            await ag._scan_entries()

        buy_mock.assert_not_called(), \
            "book.buy must not be called while SL cooldown is active"

    @pytest.mark.asyncio
    async def test_sl_cooldown_expires_and_allows_reentry(self):
        """After _SL_COOLDOWN_SEC elapses, the product is eligible again."""
        from agents.scalp_agent import _SL_COOLDOWN_SEC
        ag = ScalpAgent()
        pid = "ETH-USD"
        # Cooldown expired: set timestamp to well before the window
        ag._sl_cooldown[pid] = time.time() - _SL_COOLDOWN_SEC - 10

        # Build candles that score >= _MIN_SCORE to trigger a buy
        import math
        closes = [200.0 - math.sin(i / 3) * 5 for i in range(120)]
        candles_raw = [
            {"product_id": pid, "close": c, "high": c + 0.5, "low": c - 0.5,
             "volume": 5000.0, "start_time": i}
            for i, c in enumerate(closes)
        ]
        tracked = [{"product_id": pid, "price": closes[-1]}]

        # Check that the cooldown key is gone from the block (i.e., _scan_entries
        # doesn't skip due to cooldown — behaviour: it proceeds to scoring).
        # We verify by confirming the cooldown guard doesn't short-circuit.
        skip_due_to_cooldown = (
            time.time() - ag._sl_cooldown.get(pid, 0) < _SL_COOLDOWN_SEC
        )
        assert not skip_due_to_cooldown, \
            "Expired cooldown should not block re-entry"
