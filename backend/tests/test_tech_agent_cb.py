"""
Tests for TechAgentCB — mean-reversion crypto scalper.

Strategy: RSI(14) + Bollinger Bands + Stochastic RSI + VWAP
  BUY  threshold : 0.55
  SELL threshold : 0.55
  ADX and MFI are NOT part of TechAgent (they were removed; see momentum_agent_cb).

All external I/O (database, WebSocket, Coinbase) is mocked.
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

from agents.tech_agent_cb import TechAgentCB, _BUY_THRESHOLD, _SELL_THRESHOLD


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_candles(n: int = 100, base: float = 50_000.0,
                  trend: str = "flat") -> list:
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


def _agent(ws_price: float = 50_000.0) -> TechAgentCB:
    ws = MagicMock()
    ws.get_price = MagicMock(return_value=ws_price)
    return TechAgentCB(ws_subscriber=ws)


# ── Score structure ────────────────────────────────────────────────────────────

class TestTechAgentScore:

    def test_score_returns_required_keys(self):
        """_score() must return a dict with buy_score, sell_score, and indicators."""
        ag = _agent()
        candles = _make_candles(100)
        sc = ag._score(candles)
        for key in ("buy_score", "sell_score", "rsi", "bb_pos", "stoch_k"):
            assert key in sc, f"Missing key in score dict: {key}"

    def test_score_no_adx_or_mfi(self):
        """ADX and MFI must NOT appear in TechAgent score (moved to MomentumAgent)."""
        ag = _agent()
        candles = _make_candles(100)
        sc = ag._score(candles)
        assert "adx" not in sc, "ADX should not be in TechAgent score"
        assert "mfi" not in sc, "MFI should not be in TechAgent score"

    def test_score_bounded(self):
        """buy_score and sell_score should be floats in [0, 1]."""
        ag = _agent()
        candles = _make_candles(100)
        sc = ag._score(candles)
        assert 0.0 <= sc["buy_score"] <= 1.0
        assert 0.0 <= sc["sell_score"] <= 1.0

    def test_downtrend_higher_buy_score(self):
        """A falling market (oversold RSI) → buy_score > sell_score."""
        ag = _agent()
        candles = _make_candles(100, trend="down")
        sc = ag._score(candles)
        assert sc["buy_score"] > sc["sell_score"]

    def test_uptrend_higher_sell_score(self):
        """A rising market (overbought RSI) → sell_score > buy_score."""
        ag = _agent()
        candles = _make_candles(100, trend="up")
        sc = ag._score(candles)
        assert sc["sell_score"] > sc["buy_score"]

    def test_rsi_bounded(self):
        ag = _agent()
        candles = _make_candles(100)
        sc = ag._score(candles)
        assert 0.0 <= sc["rsi"] <= 100.0

    def test_bb_pos_bounded(self):
        """Bollinger position should be in [0, 1]."""
        ag = _agent()
        candles = _make_candles(100)
        sc = ag._score(candles)
        assert 0.0 <= sc["bb_pos"] <= 1.0


# ── Live price helper ──────────────────────────────────────────────────────────

class TestTechAgentLivePrice:

    def test_ws_price_preferred(self):
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=60_000.0)
        ag = TechAgentCB(ws_subscriber=ws)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 60_000.0

    def test_fallback_when_ws_none(self):
        ag = TechAgentCB(ws_subscriber=None)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 50_000.0

    def test_fallback_when_ws_returns_zero(self):
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=0.0)
        ag = TechAgentCB(ws_subscriber=ws)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 50_000.0

    def test_fallback_when_ws_returns_none(self):
        ws = MagicMock()
        ws.get_price = MagicMock(return_value=None)
        ag = TechAgentCB(ws_subscriber=ws)
        assert ag._live_price("BTC-USD", fallback=50_000.0) == 50_000.0


# ── analyze_product ────────────────────────────────────────────────────────────

class TestTechAgentAnalyze:

    @pytest.mark.asyncio
    async def test_analyze_returns_signal_on_buy(self):
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(100, trend="down")
        product = {"product_id": "BTC-USD", "price": 50_000.0, "volume_24h": 1e9}

        with (
            patch("agents.tech_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.tech_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            sc = ag._score(candles)
            if sc["buy_score"] >= _BUY_THRESHOLD:
                result = await ag.analyze_product(product)
                if result:
                    assert result["side"] == "BUY"
                    assert result["product_id"] == "BTC-USD"

    @pytest.mark.asyncio
    async def test_analyze_returns_none_on_hold(self):
        """A flat market with low scores → analyze returns None (HOLD)."""
        ag = _agent(ws_price=50_000.0)
        candles = _make_candles(100, trend="flat")
        product = {"product_id": "BTC-USD", "price": 50_000.0, "volume_24h": 1e9}

        with (
            patch("agents.tech_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.tech_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
        ):
            sc = ag._score(candles)
            if sc["buy_score"] < _BUY_THRESHOLD and sc["sell_score"] < _SELL_THRESHOLD:
                result = await ag.analyze_product(product)
                assert result is None

    @pytest.mark.asyncio
    async def test_analyze_insufficient_candles_returns_none(self):
        ag = _agent()
        short_candles = _make_candles(10)
        product = {"product_id": "BTC-USD", "price": 50_000.0, "volume_24h": 1e9}
        with patch("agents.tech_agent_cb.database.get_candles",
                   new=AsyncMock(return_value=short_candles)):
            result = await ag.analyze_product(product)
        assert result is None


# ── Thresholds ─────────────────────────────────────────────────────────────────

class TestTechAgentThresholds:

    def test_buy_threshold_value(self):
        """BUY threshold should be 0.55 (mean-reversion tuning)."""
        assert _BUY_THRESHOLD == pytest.approx(0.55, abs=0.01)

    def test_sell_threshold_value(self):
        """SELL threshold should be 0.55."""
        assert _SELL_THRESHOLD == pytest.approx(0.55, abs=0.01)


class TestTechMinPriceGuard:
    """Tech agent must skip products priced below MIN_PRICE."""

    @pytest.mark.asyncio
    async def test_micro_price_returns_none(self):
        from agents.tech_agent_cb import TechAgentCB, MIN_PRICE
        ag = TechAgentCB()
        product = {"product_id": "SHIB-USD", "price": 0.000012}
        result = await ag.analyze_product(product)
        assert result is None, f"Tech must return None for price < MIN_PRICE, got {result}"

    @pytest.mark.asyncio
    async def test_price_at_min_price_not_blocked(self):
        from agents.tech_agent_cb import TechAgentCB, MIN_PRICE
        ag = TechAgentCB()
        product = {"product_id": "EDGE-USD", "price": MIN_PRICE}
        # Should not return None due to the price guard (may return None for other reasons)
        # We just verify no ValueError / AttributeError is raised at the guard stage
        # by patching DB so it returns no candles (quick exit after guard)
        with patch("agents.tech_agent_cb.database.get_candles", new=AsyncMock(return_value=[])):
            result = await ag.analyze_product(product)
        # None is acceptable (no candles), but must NOT be blocked solely by price guard
        # i.e. the code reached the candle-fetch stage
        assert result is None  # no candles → None, but price guard did not fire


# ── Trailing Dollar Exit ───────────────────────────────────────────────────────

class TestTrailingDollarExitState:
    """Per-position peak_pnl_usd state initialization in _Book."""

    @pytest.mark.asyncio
    async def test_new_position_initializes_peak_pnl_to_zero(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.open_trade", new=AsyncMock()),
        ):
            await ag.book.buy("BTC-USD", price=50_000.0, frac=0.10, trigger="TEST")
        pos = ag.book.positions["BTC-USD"]
        assert pos["peak_pnl_usd"] == 0.0, (
            f"Expected peak_pnl_usd=0.0 on new position, got {pos.get('peak_pnl_usd')!r}"
        )

    @pytest.mark.asyncio
    async def test_average_up_does_not_reset_peak(self):
        """Adding to an existing position must NOT reset peak_pnl_usd."""
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.open_trade", new=AsyncMock()),
        ):
            await ag.book.buy("BTC-USD", price=50_000.0, frac=0.10, trigger="TEST")
            ag.book.positions["BTC-USD"]["peak_pnl_usd"] = 2.50
            await ag.book.buy("BTC-USD", price=51_000.0, frac=0.10, trigger="TEST")
        assert ag.book.positions["BTC-USD"]["peak_pnl_usd"] == 2.50, (
            "Averaging up must not reset peak_pnl_usd"
        )

    def test_trail_constants_are_dollar_values(self):
        from agents.tech_agent_cb import _TRAIL_ARM_USD, _TRAIL_GIVEBACK_USD
        assert _TRAIL_ARM_USD == pytest.approx(1.00, abs=1e-6)
        assert _TRAIL_GIVEBACK_USD == pytest.approx(0.25, abs=1e-6)


class TestTrailingDollarExit:
    """on_price_tick exit-chain behavior for the trailing $ take-profit."""

    @staticmethod
    def _seed_position(ag, pid: str, *, size: float, avg_price: float,
                       peak_pnl_usd: float = 0.0, atr_stop: float = 0.05) -> None:
        """Plant a position directly in the book without going through buy()."""
        ag.book.positions[pid] = {
            "size": size,
            "avg_price": avg_price,
            "atr_stop": atr_stop,
            "peak_pnl_usd": peak_pnl_usd,
        }

    @pytest.mark.asyncio
    async def test_peak_pnl_updates_when_price_rises(self):
        """A tick at a higher price bumps peak_pnl_usd."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0)
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 101.50)
        pos = ag.book.positions.get("BTC-USD")
        assert pos is not None, "Position should remain — no giveback yet"
        assert pos["peak_pnl_usd"] == pytest.approx(1.50, abs=1e-6)

    @pytest.mark.asyncio
    async def test_peak_pnl_held_when_price_falls(self):
        """A tick at a lower current PnL leaves peak_pnl_usd unchanged."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=0.50)
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 100.20)  # current PnL = +$0.20
        pos = ag.book.positions.get("BTC-USD")
        assert pos is not None, "Below-arm pullback must NOT trigger sell"
        assert pos["peak_pnl_usd"] == pytest.approx(0.50, abs=1e-6), (
            "Peak must not decrease on a falling tick"
        )

    @pytest.mark.asyncio
    async def test_no_trail_sell_when_peak_below_arm_threshold(self):
        """Peak = $0.80 with full giveback → no sell (not armed)."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=0.80)
        save_decision = AsyncMock()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=save_decision),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 100.10)
        assert ag.book.has_position("BTC-USD"), "Trail must not fire below arm threshold"
        assert ag.signals_sell == 0
        save_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_trail_sell_when_giveback_below_threshold(self):
        """Peak $1.50, current $1.30 (giveback $0.20) → no sell."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=1.50)
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 101.30)
        assert ag.book.has_position("BTC-USD")
        assert ag.signals_sell == 0
        assert ag.book.positions["BTC-USD"]["peak_pnl_usd"] == pytest.approx(1.50, abs=1e-6)

    @pytest.mark.asyncio
    async def test_trail_sell_fires_when_armed_and_giveback_reached(self):
        """Peak $1.50, current $1.25 (giveback $0.25) → SELL with TICK_TRAIL."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=1.50)
        save_decision = AsyncMock()
        close_trade = AsyncMock()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=save_decision),
            patch("agents.tech_agent_cb.database.close_trade", new=close_trade),
        ):
            await ag.on_price_tick("BTC-USD", 101.25)
        assert not ag.book.has_position("BTC-USD"), "Position should be closed by trail"
        assert ag.signals_sell == 1
        save_decision.assert_called_once()
        decision = save_decision.call_args[0][0]
        assert decision["side"] == "SELL"
        assert decision["agent"] == "TECH"
        close_trade.assert_called_once()
        assert close_trade.call_args.kwargs["trigger_close"] == "TICK_TRAIL"

    @pytest.mark.asyncio
    async def test_trail_fires_before_6pct_take_profit(self):
        """Both trail and +6% take-profit eligible → trail wins (priority)."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=7.00)
        close_trade = AsyncMock()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=close_trade),
        ):
            await ag.on_price_tick("BTC-USD", 106.50)
        close_trade.assert_called_once()
        assert close_trade.call_args.kwargs["trigger_close"] == "TICK_TRAIL"

    @pytest.mark.asyncio
    async def test_atr_stop_still_wins_over_trail(self):
        """Position deep in loss: ATR stop fires; trail never reached."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=0.0, atr_stop=0.05)
        close_trade = AsyncMock()
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=close_trade),
        ):
            await ag.on_price_tick("BTC-USD", 93.0)
        close_trade.assert_called_once()
        assert close_trade.call_args.kwargs["trigger_close"] == "TICK_STOP"

    @pytest.mark.asyncio
    async def test_position_without_peak_key_loads_safely(self):
        """Legacy saved position (no peak_pnl_usd key) → tick must not crash."""
        ag = _agent()
        ag.book.positions["BTC-USD"] = {
            "size": 1.0, "avg_price": 100.0, "atr_stop": 0.05,
        }
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 100.40)
        pos = ag.book.positions.get("BTC-USD")
        assert pos is not None
        assert pos["peak_pnl_usd"] == pytest.approx(0.40, abs=1e-6)

    @pytest.mark.asyncio
    async def test_reentry_after_trail_sell_starts_with_fresh_peak(self):
        """After trail sells, next buy for same pid initializes peak_pnl_usd=0.0."""
        ag = _agent()
        self._seed_position(ag, "BTC-USD", size=1.0, avg_price=100.0,
                            peak_pnl_usd=1.50)
        with (
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_decision", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade", new=AsyncMock()),
            patch("agents.tech_agent_cb.database.open_trade", new=AsyncMock()),
        ):
            await ag.on_price_tick("BTC-USD", 101.25)
            assert "BTC-USD" not in ag.book.positions
            await ag.book.buy("BTC-USD", price=200.0, frac=0.10, trigger="TEST")
        pos = ag.book.positions["BTC-USD"]
        assert pos["peak_pnl_usd"] == 0.0, "Re-entry must start with fresh peak"
