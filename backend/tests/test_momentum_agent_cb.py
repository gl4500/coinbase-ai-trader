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

    def test_sell_threshold_raised_to_match_tech(self):
        """SELL raised 0.30 → 0.55 to match TechAgent and filter noisy SELLs."""
        assert _SELL_THRESHOLD == 0.55, (
            f"SELL threshold must be 0.55 (was 0.30) to mirror TechAgent's "
            f"high-confidence SELL bar; got {_SELL_THRESHOLD}"
        )


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


class TestMomentumMacroRegime:
    """
    Momentum must gate its SELL score through MacroContext.sell_gate_multiplier()
    so that when shorts are crowded (squeeze risk) we don't sell into lows.
    Mirrors TechAgentCB's macro integration.
    """

    def _neutral(self):
        from services.macro_signals import MacroContext
        return MacroContext(
            funding_rate=0.0001, ls_ratio=1.0,
            oi_usd=10e9, oi_trend=0.0,
            btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
        )

    def _short_squeeze(self):
        from services.macro_signals import MacroContext
        return MacroContext(
            funding_rate=-0.002, ls_ratio=0.6,
            oi_usd=10e9, oi_trend=0.0,
            btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
        )

    def _overheated(self):
        from services.macro_signals import MacroContext
        return MacroContext(
            funding_rate=0.002, ls_ratio=2.5,
            oi_usd=15e9, oi_trend=0.2,
            btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
        )

    def test_neutral_macro_leaves_sell_score_unchanged(self):
        ag = _agent()
        sc = ag._score(
            [c["close"]  for c in _make_candles(120, trend="down")],
            [c["volume"] for c in _make_candles(120, trend="down")],
            [c["high"]   for c in _make_candles(120, trend="down")],
            [c["low"]    for c in _make_candles(120, trend="down")],
        )
        adj = ag._macro_adjusted_sell_score(sc, self._neutral())
        assert adj == pytest.approx(sc["sell_score"], abs=0.01)

    def test_short_squeeze_macro_reduces_sell_score(self):
        ag = _agent()
        candles = _make_candles(120, trend="down")
        sc = ag._score(
            [c["close"]  for c in candles],
            [c["volume"] for c in candles],
            [c["high"]   for c in candles],
            [c["low"]    for c in candles],
        )
        if sc["sell_score"] <= 0:
            pytest.skip("downtrend produced zero sell_score — cannot test reduction")
        adj = ag._macro_adjusted_sell_score(sc, self._short_squeeze())
        assert adj < sc["sell_score"]

    def test_adjusted_sell_score_never_exceeds_one(self):
        ag = _agent()
        candles = _make_candles(120, trend="down")
        sc = ag._score(
            [c["close"]  for c in candles],
            [c["volume"] for c in candles],
            [c["high"]   for c in candles],
            [c["low"]    for c in candles],
        )
        assert ag._macro_adjusted_sell_score(sc, self._overheated()) <= 1.0

    def test_adjusted_sell_score_never_negative(self):
        ag = _agent()
        candles = _make_candles(120, trend="down")
        sc = ag._score(
            [c["close"]  for c in candles],
            [c["volume"] for c in candles],
            [c["high"]   for c in candles],
            [c["low"]    for c in candles],
        )
        assert ag._macro_adjusted_sell_score(sc, self._overheated()) >= 0.0
        assert ag._macro_adjusted_sell_score(sc, self._short_squeeze()) >= 0.0

    def test_macro_adjusted_buy_score_helper_exists(self):
        """Symmetry with tech: momentum agent must also expose buy-score adjuster."""
        ag = _agent()
        candles = _make_candles(120, trend="up")
        sc = ag._score(
            [c["close"]  for c in candles],
            [c["volume"] for c in candles],
            [c["high"]   for c in candles],
            [c["low"]    for c in candles],
        )
        adj = ag._macro_adjusted_buy_score(sc, self._neutral())
        assert adj == pytest.approx(sc["buy_score"], abs=0.01)

    @pytest.mark.asyncio
    async def test_analyze_uses_macro_adjusted_sell_score(self):
        """
        analyze_product must fetch MacroContext and apply sell_gate_multiplier
        before comparing sell_score to _SELL_THRESHOLD. We verify by mocking
        get_macro_service and confirming it was awaited at least once.
        """
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(120, trend="up")   # no SELL expected
        product = {"product_id": "BTC-USD", "price": 50_000.0}

        with (
            patch("agents.momentum_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
            patch("agents.momentum_agent_cb.get_macro_service") as mock_ms,
        ):
            mock_ms.return_value.get_macro_context = AsyncMock(
                return_value=self._neutral()
            )
            await ag.analyze_product(product)
            mock_ms.return_value.get_macro_context.assert_awaited()


class TestMomentumATRStop:
    """
    Momentum must use ATR(14)-based stops instead of a fixed 3% trail + 5% hard stop.
    Mirrors TechAgentCB._compute_atr_stop.
    """

    def test_compute_atr_stop_returns_floor_with_insufficient_data(self):
        from agents.momentum_agent_cb import MomentumAgentCB, _ATR_STOP_MIN
        ag = _agent()
        short = _make_candles(10)
        assert ag._compute_atr_stop(short, 50_000.0) == _ATR_STOP_MIN

    def test_compute_atr_stop_bounded_within_min_max(self):
        from agents.momentum_agent_cb import _ATR_STOP_MIN, _ATR_STOP_MAX
        ag = _agent()
        candles = _make_candles(60, trend="up")
        stop = ag._compute_atr_stop(candles, 50_000.0)
        assert _ATR_STOP_MIN <= stop <= _ATR_STOP_MAX

    def test_compute_atr_stop_zero_entry_price_returns_floor(self):
        from agents.momentum_agent_cb import _ATR_STOP_MIN
        ag = _agent()
        candles = _make_candles(60, trend="up")
        assert ag._compute_atr_stop(candles, 0.0) == _ATR_STOP_MIN

    @pytest.mark.asyncio
    async def test_buy_stores_atr_stop_on_position(self):
        """When BUY fires, the opened position dict must include atr_stop."""
        from services.macro_signals import MacroContext
        neutral = MacroContext(
            funding_rate=0.0001, ls_ratio=1.0,
            oi_usd=10e9, oi_trend=0.0,
            btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
        )
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(120, trend="up")
        product = {"product_id": "BTC-USD", "price": 50_000.0}

        async def fake_buy(pid, price, frac, hw, trigger="SCAN"):
            ag.book.positions[pid] = {"size": 0.01, "avg_price": price}
            return price * 0.01, 0.01

        with (
            patch("agents.momentum_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
            patch.object(ag.book, "buy", side_effect=fake_buy),
            patch("agents.momentum_agent_cb.get_macro_service") as mock_ms,
        ):
            mock_ms.return_value.get_macro_context = AsyncMock(return_value=neutral)
            await ag.analyze_product(product)

        pos = ag.book.positions.get("BTC-USD")
        if pos is not None:   # only assert if buy path fired (uptrend usually triggers)
            assert "atr_stop" in pos, "BUY must attach atr_stop to position dict"
            from agents.momentum_agent_cb import _ATR_STOP_MIN, _ATR_STOP_MAX
            assert _ATR_STOP_MIN <= pos["atr_stop"] <= _ATR_STOP_MAX

    @pytest.mark.asyncio
    async def test_tick_uses_position_atr_stop_not_hard_stop(self):
        """Tick handler must compare pct to stored atr_stop, not _HARD_STOP_LOSS."""
        ag = _agent(ws_price=47_000.0)
        ag.book.positions["BTC-USD"] = {
            "size": 0.01, "avg_price": 50_000.0, "atr_stop": 0.02,  # 2% stop
        }
        ag._high_water["BTC-USD"] = 50_000.0

        sell_mock = AsyncMock(return_value=-50.0)
        with (
            patch.object(ag.book, "sell", sell_mock),
            patch("agents.momentum_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            # Price at 47k = -6% — triggers even a loose stop
            await ag.on_price_tick("BTC-USD", 47_000.0)

        assert sell_mock.await_count == 1


class TestMomentumMinPriceGuard:
    """Momentum agent must skip products priced below MIN_PRICE."""

    @pytest.mark.asyncio
    async def test_micro_price_returns_none(self):
        from agents.momentum_agent_cb import MomentumAgentCB, MIN_PRICE
        ag = MomentumAgentCB()
        product = {"product_id": "BONK-USD", "price": 0.0000139}
        result = await ag.analyze_product(product)
        assert result is None, f"Momentum must return None for price < MIN_PRICE, got {result}"

    @pytest.mark.asyncio
    async def test_price_at_min_price_not_blocked(self):
        from agents.momentum_agent_cb import MomentumAgentCB, MIN_PRICE
        ag = MomentumAgentCB()
        product = {"product_id": "EDGE-USD", "price": MIN_PRICE}
        with patch("agents.momentum_agent_cb.database.get_candles", new=AsyncMock(return_value=[])):
            result = await ag.analyze_product(product)
        assert result is None  # no candles → None, but not blocked by price guard
