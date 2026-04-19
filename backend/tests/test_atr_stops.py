"""
Tests for ATR-based trailing stop in TechAgent and MomentumAgent.

Old behavior: fixed _HARD_STOP_LOSS = 0.05 (5% below entry for all assets)
New behavior: stop = ATR(14) × ATR_MULTIPLIER / entry_price (adapts to volatility)

For BTC at $100k with ATR $500 (0.5%), stop = 3 × 500 / 100000 = 1.5%
For ETH at $3k  with ATR $90  (3.0%), stop = 3 × 90  / 3000   = 9.0%
"""
import math
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from agents.tech_agent_cb import TechAgentCB, _ATR_MULTIPLIER, _ATR_STOP_MIN, _ATR_STOP_MAX


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_candles(n: int = 80, base: float = 50_000.0, trend: str = "flat") -> list:
    candles = []
    for i in range(n):
        if trend == "down":
            c = base - i * 20
        elif trend == "up":
            c = base + i * 20
        else:
            c = base + 300 * math.sin(i / 8.0)
        candles.append({
            "close": c, "high": c + 50, "low": c - 50,
            "volume": 10_000, "start": 1_700_000_000 + i * 3600,
        })
    return candles


# ── ATR constants ──────────────────────────────────────────────────────────────

class TestATRStopConstants:

    def test_atr_multiplier_is_reasonable(self):
        """ATR multiplier should be 2–4× (common trading practice)."""
        assert 2.0 <= _ATR_MULTIPLIER <= 4.0, (
            f"_ATR_MULTIPLIER={_ATR_MULTIPLIER} outside [2, 4]"
        )

    def test_atr_stop_min_is_reasonable(self):
        """Minimum ATR stop should be at least 1% to prevent stop-hunting."""
        assert 0.01 <= _ATR_STOP_MIN <= 0.05

    def test_atr_stop_max_is_reasonable(self):
        """Maximum ATR stop should cap at ≤ 15% to limit drawdown."""
        assert _ATR_STOP_MAX <= 0.15

    def test_min_less_than_max(self):
        assert _ATR_STOP_MIN < _ATR_STOP_MAX


# ── _compute_atr_stop ──────────────────────────────────────────────────────────

class TestComputeATRStop:

    def test_returns_float(self):
        ag = TechAgentCB()
        candles = _make_candles(80)
        stop = ag._compute_atr_stop(candles, entry_price=50_000.0)
        assert isinstance(stop, float)

    def test_stop_is_positive(self):
        ag = TechAgentCB()
        candles = _make_candles(80)
        stop = ag._compute_atr_stop(candles, entry_price=50_000.0)
        assert stop > 0.0

    def test_stop_clamped_to_bounds(self):
        ag = TechAgentCB()
        candles = _make_candles(80)
        stop = ag._compute_atr_stop(candles, entry_price=50_000.0)
        assert _ATR_STOP_MIN <= stop <= _ATR_STOP_MAX

    def test_volatile_asset_larger_stop(self):
        """High-volatility candles → bigger ATR → larger stop percentage."""
        ag = TechAgentCB()
        # Low volatility: tight range ±1
        calm_candles = [{"close": 100.0, "high": 101.0, "low": 99.0, "volume": 1000,
                          "start": 1_700_000_000 + i * 3600} for i in range(80)]
        # High volatility: wide range ±10
        vol_candles  = [{"close": 100.0, "high": 110.0, "low": 90.0, "volume": 1000,
                          "start": 1_700_000_000 + i * 3600} for i in range(80)]

        stop_calm = ag._compute_atr_stop(calm_candles,  entry_price=100.0)
        stop_vol  = ag._compute_atr_stop(vol_candles,   entry_price=100.0)
        assert stop_vol > stop_calm, (
            f"Volatile stop {stop_vol:.4f} should exceed calm stop {stop_calm:.4f}"
        )

    def test_insufficient_candles_returns_default(self):
        """Fewer than period+1 candles → return _ATR_STOP_MIN as safe default."""
        ag = TechAgentCB()
        candles = _make_candles(5)
        stop = ag._compute_atr_stop(candles, entry_price=50_000.0)
        assert stop >= _ATR_STOP_MIN   # must return a safe minimum, not 0 or negative


# ── ATR stop triggers in on_price_tick ────────────────────────────────────────

class TestATRStopTrigger:

    @pytest.mark.asyncio
    async def test_tick_triggers_atr_stop_on_large_drop(self):
        """
        If price drops by more than the ATR-based stop from entry,
        on_price_tick should sell the position.
        """
        ag = TechAgentCB()

        entry_price = 50_000.0
        atr_stop    = 0.03    # 3 % ATR stop stored with position
        trigger_price = entry_price * (1 - atr_stop - 0.005)  # just past stop

        ag.book.positions["BTC-USD"] = {
            "size": 0.01,
            "avg_price": entry_price,
            "atr_stop": atr_stop,   # new: ATR stop stored per position
        }

        with (
            patch("agents.tech_agent_cb.database.close_trade",    new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", trigger_price)

        assert "BTC-USD" not in ag.book.positions, (
            "Position should be closed when price drops past ATR stop"
        )

    @pytest.mark.asyncio
    async def test_tick_does_not_stop_small_drop(self):
        """
        A small drop (half the ATR stop) should NOT trigger an exit.
        """
        ag = TechAgentCB()

        entry_price = 50_000.0
        atr_stop    = 0.05  # 5% ATR stop
        current_price = entry_price * (1 - atr_stop / 2)  # only 2.5% below

        ag.book.positions["BTC-USD"] = {
            "size": 0.01,
            "avg_price": entry_price,
            "atr_stop": atr_stop,
        }
        # No score cache → no signal exit
        ag._score_cache.pop("BTC-USD", None)

        await ag.on_price_tick("BTC-USD", current_price)

        assert "BTC-USD" in ag.book.positions, (
            "Position should NOT be closed for a drop smaller than the ATR stop"
        )
