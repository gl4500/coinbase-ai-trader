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

    def test_daily_halt_triggers_at_3pct(self):
        book = _ScalpBook()
        book._start_balance = 1_000.0
        book.balance = 969.9   # 3.01% drawdown
        book._day_str = time.strftime("%Y-%m-%d")
        assert book.daily_halt() is True

    def test_daily_halt_not_triggered_below_threshold(self):
        book = _ScalpBook()
        book._start_balance = 1_000.0
        book.balance = 975.0   # 2.5% drawdown
        book._day_str = time.strftime("%Y-%m-%d")
        assert book.daily_halt() is False

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
    async def test_scan_entries_skips_when_halted(self):
        ag = self._agent()
        ag.book.balance = 0.0          # 100% drawdown → daily halt
        ag.book._start_balance = 1000.0
        ag.book._day_str = time.strftime("%Y-%m-%d")

        buy_mock = AsyncMock()
        with patch.object(ag.book, "buy", buy_mock):
            await ag._scan_entries()

        buy_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_entries_skips_at_max_concurrent(self):
        """Already at _MAX_CONCURRENT positions → no new buy attempted."""
        ag = self._agent(ws_price=50_000.0)
        # Manually populate two positions
        ag.book.positions = {
            "BTC-USD": {"size": 0.001, "avg_price": 50_000.0,
                        "entry_time": time.time(), "high_water": 50_000.0, "trail_dist": 50.0},
            "ETH-USD": {"size": 0.01, "avg_price": 3_000.0,
                        "entry_time": time.time(), "high_water": 3_000.0, "trail_dist": 5.0},
        }
        buy_mock = AsyncMock()
        with patch.object(ag.book, "buy", buy_mock):
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
