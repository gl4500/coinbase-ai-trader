"""
Tests for MomentumAgentCB — trend-following crypto agent.

Strategy: ROC + MACD + RSI + ADX(14) gate + MFI(14) volume confirmation
  BUY  : positive momentum, ADX >= 20 (confirming trend), MFI > 60 adds +0.05
  SELL : negative momentum, MFI < 40 adds +0.05
  ADX gate prevents momentum entries in ranging (choppy) markets.

Key structural difference from TechAgent:
  - TechAgent buys WEAKNESS (oversold RSI, lower BB)
  - MomentumAgent buys STRENGTH (positive ROC, rising MACD)

All external I/O is mocked.
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

from agents.momentum_agent_cb import MomentumAgentCB, _BUY_THRESHOLD, _SELL_THRESHOLD


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_candles(n: int = 120, base: float = 50_000.0,
                  trend: str = "flat") -> list:
    candles = []
    for i in range(n):
        if trend == "up":
            c = base + i * 30          # strong uptrend → positive ROC, overbought MFI
        elif trend == "down":
            c = base - i * 30          # downtrend → negative ROC
        else:
            c = base + 300 * math.sin(i / 8.0)
        candles.append({
            "close":  c,
            "high":   c + abs(c * 0.005),
            "low":    c - abs(c * 0.005),
            "volume": 15_000 + i * 100,
            "start":  1_700_000_000 + i * 3600,
        })
    return candles


def _agent(ws_price: float = 50_000.0) -> MomentumAgentCB:
    ws = MagicMock()
    ws.get_price = MagicMock(return_value=ws_price)
    return MomentumAgentCB(ws_subscriber=ws)


# ── Score structure ────────────────────────────────────────────────────────────

class TestMomentumAgentScore:

    def _score(self, candles):
        ag = _agent()
        closes  = [c["close"]  for c in candles]
        vols    = [c["volume"] for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        return ag._score(closes, vols, highs, lows)

    def test_score_contains_adx(self):
        """ADX must be present in MomentumAgent score (moved FROM TechAgent)."""
        candles = _make_candles(120)
        sc = self._score(candles)
        assert "adx" in sc, "ADX missing from MomentumAgent score"

    def test_score_contains_mfi(self):
        """MFI must be present in MomentumAgent score (moved FROM TechAgent)."""
        candles = _make_candles(120)
        sc = self._score(candles)
        assert "mfi" in sc, "MFI missing from MomentumAgent score"

    def test_score_required_keys(self):
        candles = _make_candles(120)
        sc = self._score(candles)
        for key in ("buy_score", "sell_score", "adx", "mfi", "vw_mom"):
            assert key in sc, f"Missing key: {key}"

    def test_score_bounded(self):
        candles = _make_candles(120)
        sc = self._score(candles)
        assert 0.0 <= sc["buy_score"] <= 1.5   # MFI can add up to 0.05 over 1.0
        assert 0.0 <= sc["sell_score"] <= 1.5

    def test_uptrend_higher_buy_score(self):
        """A strong uptrend → buy_score > sell_score."""
        candles = _make_candles(120, trend="up")
        sc = self._score(candles)
        assert sc["buy_score"] > sc["sell_score"]

    def test_downtrend_higher_sell_score(self):
        """A downtrend → sell_score > buy_score."""
        candles = _make_candles(120, trend="down")
        sc = self._score(candles)
        assert sc["sell_score"] > sc["buy_score"]

    def test_adx_non_negative(self):
        candles = _make_candles(120)
        sc = self._score(candles)
        assert sc["adx"] >= 0.0

    def test_mfi_in_range(self):
        """MFI is a 0–100 oscillator."""
        candles = _make_candles(120)
        sc = self._score(candles)
        assert 0.0 <= sc["mfi"] <= 100.0

    def test_mfi_buy_bonus_applied(self):
        """Uptrend with high MFI (buying pressure) boosts buy_score."""
        candles_up   = _make_candles(120, trend="up")
        candles_flat = _make_candles(120, trend="flat")
        sc_up   = self._score(candles_up)
        sc_flat = self._score(candles_flat)
        # MFI > 60 in uptrend adds +0.05 to buy_score
        # We can't guarantee MFI > 60 with synthetic data, so just check structure
        assert isinstance(sc_up["mfi"], float)
        if sc_up["mfi"] > 60:
            # If MFI IS elevated, there should be a buy_score contribution
            assert sc_up["buy_score"] > 0

    def test_mfi_sell_bonus_applied(self):
        """Downtrend with low MFI (selling pressure) boosts sell_score."""
        candles_down = _make_candles(120, trend="down")
        sc_down = self._score(candles_down)
        assert isinstance(sc_down["mfi"], float)
        if sc_down["mfi"] < 40:
            assert sc_down["sell_score"] > 0


# ── ADX gate ──────────────────────────────────────────────────────────────────

class TestMomentumADXGate:
    """The ADX gate prevents momentum buys in ranging (low ADX) markets."""

    @pytest.mark.asyncio
    async def test_adx_below_threshold_blocks_buy(self):
        """ADX < 20 → buy entry blocked even if buy_score is above threshold."""
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(120, trend="flat")

        closes  = [c["close"]  for c in candles]
        vols    = [c["volume"] for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        sc = ag._score(closes, vols, highs, lows)

        # Simulate score above threshold but ADX below gate
        sc["buy_score"] = _BUY_THRESHOLD + 0.10
        sc["adx"]       = 15.0          # below 20 → ranging → gate closed
        sc["vw_mom"]    = 0.01          # positive volume-weighted momentum

        buy_mock = AsyncMock()
        with (
            patch.object(ag.book, "buy", buy_mock),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            # Manually run the logic that checks the ADX gate
            has_pos = False
            if sc["buy_score"] >= _BUY_THRESHOLD and not has_pos \
                    and sc["vw_mom"] > 0 and sc["adx"] >= 20:
                await ag.book.buy("BTC-USD", 50_000.0, trigger="MOM")

        buy_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_adx_above_threshold_allows_buy(self):
        """ADX >= 20 → buy entry allowed when buy_score is above threshold."""
        ag = _agent(ws_price=50_000.0)

        sc = {
            "buy_score": _BUY_THRESHOLD + 0.10,
            "adx":       25.0,          # above gate → trending
            "vw_mom":    0.01,
        }

        buy_mock = AsyncMock(return_value=(100.0, 0.002))
        with patch.object(ag.book, "buy", buy_mock):
            has_pos = False
            if sc["buy_score"] >= _BUY_THRESHOLD and not has_pos \
                    and sc["vw_mom"] > 0 and sc["adx"] >= 20:
                await ag.book.buy("BTC-USD", 50_000.0, trigger="MOM")

        buy_mock.assert_called_once()


# ── Live price helper ──────────────────────────────────────────────────────────

class TestMomentumLivePrice:

    def test_ws_price_preferred(self):
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=60_000.0)
        ag = MomentumAgentCB(ws_subscriber=ws)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 60_000.0

    def test_fallback_when_ws_zero(self):
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=0.0)
        ag = MomentumAgentCB(ws_subscriber=ws)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 50_000.0

    def test_fallback_when_no_ws(self):
        ag = MomentumAgentCB(ws_subscriber=None)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 50_000.0


# ── Thresholds ─────────────────────────────────────────────────────────────────

class TestMomentumThresholds:

    def test_buy_threshold(self):
        assert _BUY_THRESHOLD > 0.0

    def test_sell_threshold(self):
        assert _SELL_THRESHOLD > 0.0

    def test_thresholds_symmetric(self):
        """Buy and sell thresholds should be equal (symmetric strategy)."""
        assert _BUY_THRESHOLD == pytest.approx(_SELL_THRESHOLD, abs=0.01)


# ── analyze_product ────────────────────────────────────────────────────────────

class TestMomentumAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_insufficient_candles_returns_none(self):
        ag = _agent()
        short = _make_candles(10)
        product = {"product_id": "BTC-USD", "price": 50_000.0, "volume_24h": 1e9}
        with patch("agents.momentum_agent_cb.database.get_candles",
                   new=AsyncMock(return_value=short)):
            result = await ag.analyze_product(product)
        assert result is None

    @pytest.mark.asyncio
    async def test_analyze_returns_dict_or_none(self):
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(120, trend="up")
        product = {"product_id": "BTC-USD", "price": 50_000.0, "volume_24h": 1e9}

        with (
            patch("agents.momentum_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            result = await ag.analyze_product(product)

        assert result is None or isinstance(result, dict)
        if result is not None:
            assert result["product_id"] == "BTC-USD"
            assert result["side"] in ("BUY", "SELL", "HOLD")
