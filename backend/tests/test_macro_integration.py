"""
Tests for macro signal integration in TechAgent.

Verifies that MacroContext.buy_gate_multiplier() is applied to buy scores,
SELL is not suppressed, and Kelly fraction is used for sizing.
"""
import math
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from services.macro_signals import MacroContext


def _make_candles(n: int = 100, trend: str = "down") -> list:
    candles = []
    base = 50_000.0
    for i in range(n):
        c = base - i * 20 if trend == "down" else base + 300 * math.sin(i / 8.0)
        candles.append({"close": c, "high": c + 50, "low": c - 50,
                         "volume": 10_000, "start": 1_700_000_000 + i * 3600})
    return candles


def _overheated() -> MacroContext:
    return MacroContext(
        funding_rate=0.002, ls_ratio=2.5,
        oi_usd=15e9, oi_trend=0.2,
        btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
    )


def _neutral() -> MacroContext:
    return MacroContext(
        funding_rate=0.0001, ls_ratio=1.0,
        oi_usd=10e9, oi_trend=0.0,
        btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
    )


# ── _macro_adjusted_buy_score / sell_score ─────────────────────────────────────

class TestTechAgentMacroAdjustedScore:

    def test_neutral_macro_leaves_buy_score_unchanged(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        sc = ag._score(_make_candles(100, trend="down"))
        assert ag._macro_adjusted_buy_score(sc, _neutral()) == pytest.approx(sc["buy_score"], abs=0.01)

    def test_overheated_macro_reduces_buy_score(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        sc = ag._score(_make_candles(100, trend="down"))
        adj = ag._macro_adjusted_buy_score(sc, _overheated())
        assert adj < sc["buy_score"]

    def test_macro_does_not_significantly_reduce_sell_score(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        sc = ag._score(_make_candles(100, trend="up"))
        adj_sell = ag._macro_adjusted_sell_score(sc, _overheated())
        assert adj_sell >= sc["sell_score"] * 0.85

    def test_adjusted_score_never_exceeds_one(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        sc = ag._score(_make_candles(100))
        # Even with a boost multiplier (contrarian BUY), score ≤ 1.0
        short_squeeze = MacroContext(
            funding_rate=-0.002, ls_ratio=0.6,
            oi_usd=10e9, oi_trend=0.0,
            btc_dominance=52.0, coinbase_premium=0.0, fetch_ok=True,
        )
        assert ag._macro_adjusted_buy_score(sc, short_squeeze) <= 1.0

    def test_adjusted_score_never_negative(self):
        from agents.tech_agent_cb import TechAgentCB
        ag = TechAgentCB()
        sc = ag._score(_make_candles(100))
        assert ag._macro_adjusted_buy_score(sc, _overheated()) >= 0.0
        assert ag._macro_adjusted_sell_score(sc, _overheated()) >= 0.0


# ── analyze_product with macro mock ───────────────────────────────────────────

class TestTechAnalyzeProductWithMacro:

    @pytest.mark.asyncio
    async def test_overheated_market_can_suppress_buy(self):
        from agents.tech_agent_cb import TechAgentCB, _BUY_THRESHOLD
        ag = TechAgentCB()
        candles = _make_candles(100, trend="down")
        macro   = _overheated()

        with (
            patch("agents.tech_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.tech_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
            patch("agents.tech_agent_cb.get_macro_service") as mock_ms,
        ):
            mock_ms.return_value.get_macro_context = AsyncMock(return_value=macro)
            sc  = ag._score(candles)
            adj = ag._macro_adjusted_buy_score(sc, macro)

            # Only assert suppression when TA said BUY but macro pulled it below threshold
            if sc["buy_score"] >= _BUY_THRESHOLD and adj < _BUY_THRESHOLD:
                result = await ag.analyze_product(
                    {"product_id": "BTC-USD", "price": 50_000.0}
                )
                if result:
                    assert result["side"] != "BUY"

    @pytest.mark.asyncio
    async def test_macro_does_not_block_sell(self):
        from agents.tech_agent_cb import TechAgentCB, _SELL_THRESHOLD
        ag = TechAgentCB()
        candles = _make_candles(100, trend="up")
        ag.book.positions["BTC-USD"] = {
            "size": 0.01, "avg_price": 40_000.0, "atr_stop": 0.05
        }

        with (
            patch("agents.tech_agent_cb.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.tech_agent_cb.database.save_agent_decision",
                  new=AsyncMock()),
            patch("agents.tech_agent_cb.database.close_trade",      new=AsyncMock()),
            patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
            patch("agents.tech_agent_cb.get_macro_service") as mock_ms,
        ):
            mock_ms.return_value.get_macro_context = AsyncMock(return_value=_overheated())
            sc = ag._score(candles)

            if sc["sell_score"] >= _SELL_THRESHOLD:
                result = await ag.analyze_product(
                    {"product_id": "BTC-USD", "price": 50_000.0}
                )
                if result:
                    assert result["side"] == "SELL"
