"""
Tests for CoinbaseCNNAgent and FeatureBuilder.

These tests run without real credentials, PyTorch, or a live Ollama server.
All external calls are patched.
"""
import asyncio
import math
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# ── Path ──────────────────────────────────────────────────────────────────────
BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "qwen2.5:7b")

from agents.cnn_agent import (
    CoinbaseCNNAgent,
    FeatureBuilder,
    N_CHANNELS,
    SEQ_LEN,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_candles(n: int = 80, start: float = 50000.0) -> list:
    """Sinusoidal OHLCV candles for testing."""
    candles = []
    for i in range(n):
        c = start + 500 * math.sin(i / 5.0)
        candles.append({
            "open":   c - 50,
            "high":   c + 100,
            "low":    c - 100,
            "close":  c,
            "volume": 10_000 + i * 100,
            "start":  1_700_000_000 + i * 3600,
        })
    return candles


# ── FeatureBuilder tests ───────────────────────────────────────────────────────

class TestFeatureBuilder:
    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_output_shape(self):
        candles  = _make_candles(80)
        channels = self.fb.build(candles, {})
        assert len(channels) == N_CHANNELS
        for ch in channels:
            assert len(ch) == SEQ_LEN, f"channel length {len(ch)} != {SEQ_LEN}"

    def test_short_history_padded(self):
        """Fewer than SEQ_LEN bars should be left-padded."""
        candles  = _make_candles(10)
        channels = self.fb.build(candles, {})
        assert len(channels[0]) == SEQ_LEN

    def test_empty_candles_returns_zeros(self):
        channels = self.fb.build([], {})
        assert len(channels) == N_CHANNELS
        assert all(v == 0.0 for v in channels[0])

    def test_norm_close_bounded(self):
        """Channel 0 (normalised close) should be in [0, 1]."""
        candles  = _make_candles(80)
        channels = self.fb.build(candles, {})
        assert all(0.0 <= v <= 1.0 for v in channels[0]), "Ch0 norm_close out of [0,1]"

    def test_ob_channels_broadcast(self):
        """OB bid/ask channels should be uniform scalars (broadcast)."""
        candles = _make_candles(60)
        ob = {"bid_depth": 50_000, "ask_depth": 30_000}
        channels = self.fb.build(candles, ob)
        assert len(set(channels[10])) == 1, "bid_ch should be uniform"
        assert len(set(channels[11])) == 1, "ask_ch should be uniform"

    def test_ob_empty_gives_zero_channels(self):
        """Empty orderbook should give 0 in bid/ask channels."""
        candles  = _make_candles(60)
        channels = self.fb.build(candles, {})
        assert channels[10][0] == 0.0
        assert channels[11][0] == 0.0

    def test_rising_prices_positive_change(self):
        """Steadily rising closes → positive 1-bar change at the end (ch 9)."""
        candles = []
        for i in range(70):
            p = 100.0 + i
            candles.append({"open": p - 0.5, "high": p + 1, "low": p - 1,
                             "close": p, "volume": 1000, "start": i})
        channels = self.fb.build(candles, {})
        assert channels[9][-1] > 0, "Rising prices → positive 1-bar change"

    def test_candle_body_direction(self):
        """Bullish candle (close > open) → positive body channel value."""
        candle = [{"open": 100.0, "high": 110.0, "low": 95.0,
                   "close": 108.0, "volume": 500, "start": 0}]
        channels = self.fb.build(candle * 60, {})
        # body = (close - open) / open > 0 for bullish candles
        assert channels[3][-1] > 0, "Bullish candle body should be positive"


# ── CoinbaseCNNAgent tests ─────────────────────────────────────────────────────

@pytest.fixture
def agent():
    """CNN agent with PyTorch disabled (linear fallback)."""
    with patch("agents.cnn_agent._TORCH", False):
        a = CoinbaseCNNAgent()
    return a


@pytest.fixture
def product():
    return {
        "product_id":   "BTC-USD",
        "base_currency": "BTC",
        "price":        95_000.0,
        "volume_24h":   500_000_000,
    }


class TestCoinbaseCNNAgent:

    @pytest.mark.asyncio
    async def test_generate_signal_buy(self, agent, product):
        """High model_prob → BUY signal returned."""
        candles = _make_candles(80)
        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.82)),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(return_value=1)),
        ):
            sig = await agent.generate_signal(product)

        assert sig is not None
        assert sig["side"] == "BUY"
        assert sig["signal_type"] == "CNN_LONG"
        assert sig["product_id"] == "BTC-USD"
        assert sig["strength"] > 0

    @pytest.mark.asyncio
    async def test_generate_signal_sell(self, agent, product):
        """Low model_prob → SELL signal returned."""
        candles = _make_candles(80)
        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.15)),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(return_value=2)),
        ):
            sig = await agent.generate_signal(product)

        assert sig is not None
        assert sig["side"] == "SELL"
        assert sig["signal_type"] == "CNN_SHORT"

    @pytest.mark.asyncio
    async def test_generate_signal_no_conviction(self, agent, product):
        """model_prob near 0.5 → no signal (returns None)."""
        candles = _make_candles(80)
        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.50)),
        ):
            sig = await agent.generate_signal(product)

        assert sig is None

    @pytest.mark.asyncio
    async def test_generate_signal_no_price(self, agent):
        """Product with no price → returns None immediately."""
        p = {"product_id": "ETH-USD", "base_currency": "ETH", "price": 0}
        sig = await agent.generate_signal(p)
        assert sig is None

    @pytest.mark.asyncio
    async def test_generate_signal_insufficient_candles(self, agent, product):
        """Fewer than 30 candles → returns None."""
        with patch("agents.cnn_agent.database.get_candles",
                   new=AsyncMock(return_value=_make_candles(5))):
            sig = await agent.generate_signal(product)
        assert sig is None

    @pytest.mark.asyncio
    async def test_cache_skips_fetch(self, agent, product):
        """Second call within TTL reuses cached CNN prob — no candle fetch."""
        import time
        agent._cache["BTC-USD"] = (0.75, time.time())

        mock_get_candles = AsyncMock(return_value=_make_candles(80))
        with (
            patch("agents.cnn_agent.database.get_candles", mock_get_candles),
            patch("agents.cnn_agent._ollama_prob", new=AsyncMock(return_value=0.82)),
            patch("agents.cnn_agent.database.save_signal", new=AsyncMock(return_value=3)),
        ):
            await agent.generate_signal(product)

        mock_get_candles.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_all_empty(self, agent):
        """scan_all with no tracked products returns empty list."""
        with patch("agents.cnn_agent.database.get_products",
                   new=AsyncMock(return_value=[])):
            result = await agent.scan_all()
        assert result == []

    @pytest.mark.asyncio
    async def test_scan_all_sorted_by_strength(self, agent):
        """Signals from scan_all are sorted by strength descending."""
        products = [
            {"product_id": "BTC-USD", "base_currency": "BTC", "price": 95000.0},
            {"product_id": "ETH-USD", "base_currency": "ETH", "price": 3000.0},
        ]
        fake_signals = [
            {"product_id": "BTC-USD", "side": "BUY", "strength": 0.30,
             "signal_type": "CNN_LONG", "price": 95000.0, "quote_size": 30.0},
            {"product_id": "ETH-USD", "side": "BUY", "strength": 0.60,
             "signal_type": "CNN_LONG", "price": 3000.0, "quote_size": 60.0},
        ]
        idx = [0]
        async def _fake_generate(p, execute=False, order_executor=None):
            s = fake_signals[idx[0]]
            idx[0] += 1
            return s

        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch.object(agent, "generate_signal", side_effect=_fake_generate),
        ):
            results = await agent.scan_all()

        assert len(results) == 2
        assert results[0]["strength"] >= results[1]["strength"]

    def test_linear_fallback_high_prob(self, agent):
        """Linear fallback returns > 0.5 when normalised close (ch0) is high."""
        channels = [[0.9] * SEQ_LEN] * N_CHANNELS
        p = agent._linear(channels)
        assert p > 0.5

    def test_linear_fallback_low_prob(self, agent):
        """Linear fallback returns < 0.5 when normalised close (ch0) is low."""
        channels = [[0.1] * SEQ_LEN] + [[0.9] * SEQ_LEN] * (N_CHANNELS - 1)
        p = agent._linear(channels)
        assert p < 0.5
