"""
TDD tests for Momentum Agent entry filter improvements.

Current win rate: 34.1% (need ~41%+ to break even).
Fixes:
  1. Raise buy threshold 0.30 → 0.45 (stronger signal required)
  2. Add RSI < 65 gate (avoid overbought entries)
  3. Add ADX > 20 gate (require confirmed trend, not noise)

Written before implementation — tests will fail until code is updated.
"""
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

from agents.momentum_agent_cb import MomentumAgentCB, _BUY_THRESHOLD


# ── Threshold constant tests ───────────────────────────────────────────────────

class TestMomentumThresholds:

    def test_buy_threshold_raised_to_0_45(self):
        """_BUY_THRESHOLD must be 0.45 to reduce low-conviction entries."""
        assert _BUY_THRESHOLD == 0.45, (
            f"_BUY_THRESHOLD={_BUY_THRESHOLD} — expected 0.45. "
            "Raising from 0.30 eliminates weak entries that drove 34% win rate."
        )

    def test_score_cache_includes_rsi(self):
        """_compute_scores() must return 'rsi' key for the entry filter."""
        import math
        agent = MomentumAgentCB()
        # Build enough fake candles for RSI(14)
        candles = []
        for i in range(80):
            c = 100.0 + math.sin(i / 3.0) * 5
            candles.append({"open": c - 0.5, "high": c + 1, "low": c - 1,
                             "close": c, "volume": 1000 + i * 10,
                             "start_time": i})
        closes  = [c["close"]  for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]
        scores = agent._score(closes, volumes, highs, lows)
        assert scores is not None
        assert "rsi" in scores, (
            "Score cache must include 'rsi' key so entry filter can gate on it. "
            f"Keys present: {list(scores.keys())}"
        )


# ── Entry filter: RSI gate ─────────────────────────────────────────────────────

class TestMomentumRSIFilter:

    def _make_agent_with_cache(self, pid: str, buy_score: float,
                                rsi: float, adx: float,
                                vw_mom: float = 0.01) -> MomentumAgentCB:
        agent = MomentumAgentCB()
        agent._score_cache[pid] = {
            "buy_score":   buy_score,
            "sell_score":  0.0,
            "buy_reasons": ["5d mom=5.0%"],
            "sell_reasons": [],
            "vw_mom":      vw_mom,
            "mom_s":       0.05,
            "mom_m":       0.03,
            "consistency": 0.70,
            "adx":         adx,
            "mfi":         55.0,
            "rsi":         rsi,
        }
        return agent

    @pytest.mark.asyncio
    async def test_buy_blocked_when_rsi_overbought(self):
        """RSI=70 (overbought) must block the buy even with strong score."""
        agent = self._make_agent_with_cache(
            "BTC-USD", buy_score=0.55, rsi=70.0, adx=30.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(0.0, 0.0))

        with patch.object(agent.book, "buy", buy_mock):
            await agent.on_price_tick("BTC-USD", 50_000.0)

        buy_mock.assert_not_called(), \
            "RSI=70 is overbought — buy must be blocked"

    @pytest.mark.asyncio
    async def test_buy_allowed_when_rsi_normal(self):
        """RSI=55 with strong score and trending market → buy executes."""
        agent = self._make_agent_with_cache(
            "ETH-USD", buy_score=0.55, rsi=55.0, adx=30.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(50.0, 0.01))

        with (
            patch.object(agent.book, "buy", buy_mock),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            await agent.on_price_tick("ETH-USD", 3_000.0)

        buy_mock.assert_called_once(), \
            "RSI=55 + score=0.55 + ADX=30 should trigger a buy"

    @pytest.mark.asyncio
    async def test_buy_blocked_when_score_below_new_threshold(self):
        """Score=0.35 is above old threshold (0.30) but below new (0.45) → blocked."""
        agent = self._make_agent_with_cache(
            "SOL-USD", buy_score=0.35, rsi=50.0, adx=30.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(0.0, 0.0))

        with patch.object(agent.book, "buy", buy_mock):
            await agent.on_price_tick("SOL-USD", 100.0)

        buy_mock.assert_not_called(), \
            "Score=0.35 is below new threshold 0.45 — buy must be blocked"

    @pytest.mark.asyncio
    async def test_buy_blocked_when_adx_too_low(self):
        """ADX=15 (weak/no trend) must block buy even with good score and RSI."""
        agent = self._make_agent_with_cache(
            "AVAX-USD", buy_score=0.55, rsi=50.0, adx=15.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(0.0, 0.0))

        with patch.object(agent.book, "buy", buy_mock):
            await agent.on_price_tick("AVAX-USD", 25.0)

        buy_mock.assert_not_called(), \
            "ADX=15 has no confirmed trend — buy must be blocked"

    @pytest.mark.asyncio
    async def test_buy_allowed_at_rsi_boundary_64(self):
        """RSI=64 is just under 65 threshold → buy should be allowed."""
        agent = self._make_agent_with_cache(
            "XRP-USD", buy_score=0.55, rsi=64.0, adx=25.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(50.0, 10.0))

        with (
            patch.object(agent.book, "buy", buy_mock),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            await agent.on_price_tick("XRP-USD", 0.5)

        buy_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_buy_blocked_at_rsi_boundary_65(self):
        """RSI=65 is exactly at the overbought boundary → buy must be blocked."""
        agent = self._make_agent_with_cache(
            "DOGE-USD", buy_score=0.55, rsi=65.0, adx=25.0
        )
        agent.book.balance = 1000.0
        buy_mock = AsyncMock(return_value=(0.0, 0.0))

        with patch.object(agent.book, "buy", buy_mock):
            await agent.on_price_tick("DOGE-USD", 0.08)

        buy_mock.assert_not_called()
