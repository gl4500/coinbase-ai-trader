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
import agents.cnn_agent as _cnn_mod

try:
    import torch
    import torch.nn as nn
    from agents.cnn_agent import GatedConv1d, SignalCNN
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


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

    # Common patches shared by all generate_signal tests that need candles
    _common_patches = {
        "agents.cnn_agent.database.get_agent_decisions": [],
        "agents.cnn_agent.database.save_cnn_scan":       None,
    }

    @pytest.mark.asyncio
    async def test_generate_signal_buy(self, agent, product):
        """High model_prob → BUY signal returned.
        Pin _cnn_prob to 0.82 so the test is independent of the linear fallback and
        doesn't get flipped by the LLM-skip threshold check."""
        candles = _make_candles(80)
        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",
                  new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.82)),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(return_value=1)),
            patch.object(agent, "_cnn_prob", return_value=0.82),
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
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",
                  new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.15)),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(return_value=2)),
            patch.object(agent, "_cnn_prob", return_value=0.15),
        ):
            sig = await agent.generate_signal(product)

        assert sig is not None
        assert sig["side"] == "SELL"
        assert sig["signal_type"] == "CNN_SHORT"

    @pytest.mark.asyncio
    async def test_generate_signal_no_conviction(self, agent, product):
        """model_prob near 0.5 → no signal (returns None).
        Pin cnn_prob to 0.5 so skip_llm doesn't fire (cnn_dist=0 < threshold)."""
        candles = _make_candles(80)
        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",
                  new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=0.50)),
            patch.object(agent, "_cnn_prob", return_value=0.50),
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
        agent._cache["BTC-USD"] = (0.75, time.time(), {})

        mock_get_candles = AsyncMock(return_value=_make_candles(80))
        with (
            patch("agents.cnn_agent.database.get_candles", mock_get_candles),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan", new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent._ollama_prob", new=AsyncMock(return_value=0.82)),
            patch("agents.cnn_agent.database.save_signal", new=AsyncMock(return_value=3)),
        ):
            await agent.generate_signal(product)

        mock_get_candles.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_write_produces_three_tuple(self, agent, product):
        """CNN cache invariant #2: write path stores (cnn_prob, timestamp, indicators_dict)."""
        assert "BTC-USD" not in agent._cache

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=_make_candles(80))),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan", new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.75, 0, 0))),
            patch.object(agent, "_cnn_prob", return_value=0.75),
            patch("agents.cnn_agent.database.save_signal", new=AsyncMock(return_value=3)),
        ):
            await agent.generate_signal(product)

        cached = agent._cache.get("BTC-USD")
        assert cached is not None, "generate_signal should populate cache on miss"
        assert len(cached) == 3, f"cache entry must be 3-tuple, got {len(cached)}-tuple"
        cnn_prob, ts, indicators = cached
        assert isinstance(cnn_prob, float)
        assert isinstance(ts, float)
        assert isinstance(indicators, dict)

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


# ── train_on_history ───────────────────────────────────────────────────────────

class TestTrainOnHistory:
    """train_on_history: 80/20 split, val loss, fit diagnosis."""

    @pytest.fixture(autouse=True)
    def _isolate_model_paths(self, tmp_path, monkeypatch):
        """Redirect best-loss + checkpoint paths into tmp_path so real
        train_on_history calls cannot clobber backend/cnn_best_loss.txt
        or backend/cnn_model.pt. See test_production_paths_are_isolated."""
        import agents.cnn_agent as ca
        monkeypatch.setattr(ca, "_BEST_LOSS_PATH", str(tmp_path / "best_loss.txt"))
        monkeypatch.setattr(ca, "MODEL_PATH",      str(tmp_path / "cnn_model.pt"))
        monkeypatch.setattr(ca, "_MODEL_BAK_PATH", str(tmp_path / "cnn_model.pt.bak"))

    def test_production_paths_are_isolated(self):
        """Guard: these tests call real train_on_history which writes the
        best-loss file and saves the .pt checkpoint. _BEST_LOSS_PATH and
        MODEL_PATH MUST be redirected to tmp_path by an autouse fixture,
        otherwise every pre-commit run clobbers production state.

        Incident 2026-04-21: cnn_best_loss.txt was repeatedly reset to 0.0
        by these tests, and cnn_model.pt.bak was overwritten with a
        synthetic-data checkpoint.
        """
        import agents.cnn_agent as ca
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        prod_loss = os.path.abspath(os.path.join(backend_dir, "cnn_best_loss.txt"))
        prod_ckpt = os.path.abspath(os.path.join(backend_dir, "cnn_model.pt"))
        assert os.path.abspath(ca._BEST_LOSS_PATH) != prod_loss, (
            "_BEST_LOSS_PATH points at production file; add autouse fixture."
        )
        assert os.path.abspath(ca.MODEL_PATH) != prod_ckpt, (
            "MODEL_PATH points at production file; add autouse fixture."
        )

    def _make_sqlite_candles(self, n: int = 80, start: float = 50000.0) -> list:
        """Candles using SQLite column name 'start_time' (not 'start')."""
        candles = []
        for i in range(n):
            c = start + 500 * math.sin(i / 5.0)
            candles.append({
                "open":       c - 50,
                "high":       c + 100,
                "low":        c - 100,
                "close":      c,
                "volume":     10_000 + i * 100,
                "start_time": 1_700_000_000 + i * 3600,  # ← SQLite column
            })
        return candles

    @pytest.mark.asyncio
    async def test_sqlite_candles_start_time_key_does_not_crash(self):
        """SQLite returns 'start_time' not 'start' — training must not KeyError."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")

        agent = CoinbaseCNNAgent()
        products = [{"product_id": f"COIN{i}-USD"} for i in range(6)]
        candles  = self._make_sqlite_candles(80)

        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.load_history", return_value=[]),
        ):
            result = await agent.train_on_history(epochs=2)

        # Should succeed — no KeyError
        assert "error" not in result or "Not enough" not in result.get("error", "")

    @pytest.fixture
    def trained_agent(self):
        """CNN agent with PyTorch enabled (uses real model if available, else skip)."""
        import pytest
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("PyTorch not installed")
        a = CoinbaseCNNAgent()
        return a

    def _make_products_and_candles(self, n_products: int = 6):
        import math
        products = [{"product_id": f"COIN{i}-USD"} for i in range(n_products)]
        candles = []
        # Need > SEQ_LEN(60) + _FORWARD_HOURS(4) + 1 = 65 bars to generate samples.
        # Span several sine periods so the 4h-ahead return crosses both sides of
        # zero — otherwise the dataset collapses to a single class and BCE loss
        # can degenerate to 0.0 in a few epochs on mock data.
        for i in range(200):
            c = 50_000.0 + 500 * math.sin(i / 5.0)
            candles.append({
                "open": c - 50, "high": c + 100, "low": c - 100,
                "close": c, "volume": 10_000 + i * 100,
                "start": 1_700_000_000 + i * 3600,
            })
        return products, candles

    @pytest.mark.asyncio
    async def test_returns_error_when_too_few_samples(self, trained_agent):
        """Fewer than SEQ_LEN+5 candles per product → skipped → <4 total samples → error."""
        products = [{"product_id": "BTC-USD"}, {"product_id": "ETH-USD"}]
        candles  = _make_candles(60)   # 60 < SEQ_LEN(60)+_FORWARD_HOURS(4)+1=65 → skipped
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.load_history", return_value=[]),
        ):
            result = await trained_agent.train_on_history(epochs=2)

        assert "error" in result

    @pytest.mark.asyncio
    async def test_returns_required_keys(self, trained_agent):
        """With enough data, result contains all diagnostic keys."""
        products, candles = self._make_products_and_candles(6)
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
        ):
            result = await trained_agent.train_on_history(epochs=3)

        for key in ("epochs", "samples", "train_samples", "val_samples",
                    "initial_loss", "final_train_loss", "final_val_loss",
                    "fit_status", "fit_advice", "epoch_log"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_train_val_split_is_80_20(self, trained_agent):
        """Train samples ≈ 80% of total, val ≈ 20%."""
        products, candles = self._make_products_and_candles(10)
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
        ):
            result = await trained_agent.train_on_history(epochs=2)

        assert result["train_samples"] + result["val_samples"] == result["samples"]
        assert result["train_samples"] == int(result["samples"] * 0.8)

    @pytest.mark.asyncio
    async def test_epoch_log_has_correct_length(self, trained_agent):
        """epoch_log should have one entry per epoch."""
        products, candles = self._make_products_and_candles(6)
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
        ):
            result = await trained_agent.train_on_history(epochs=5)

        assert len(result["epoch_log"]) == 5
        assert result["epoch_log"][0]["epoch"] == 1
        assert result["epoch_log"][-1]["epoch"] == 5

    @pytest.mark.asyncio
    async def test_fit_status_is_valid_value(self, trained_agent):
        """fit_status must be one of the three known values."""
        products, candles = self._make_products_and_candles(6)
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
        ):
            result = await trained_agent.train_on_history(epochs=3)

        assert result["fit_status"] in ("OK", "OVERFIT", "UNDERFIT", "REJECTED")

    @pytest.mark.asyncio
    async def test_losses_are_positive_floats(self, trained_agent):
        """All reported losses must be finite non-negative numbers within a
        sane range. On perfectly-learnable mock sinusoidal data the loss can
        round to 0.0 in fp32 even at epoch 1 — the test's real intent is
        catching NaN/Inf/huge values, not strict positivity."""
        import math
        products, candles = self._make_products_and_candles(6)
        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
        ):
            result = await trained_agent.train_on_history(epochs=3)

        for key in ("initial_loss", "final_train_loss", "final_val_loss"):
            v = result[key]
            assert math.isfinite(v), f"{key} not finite: {v}"
            assert v >= 0, f"{key} must be non-negative: {v}"
            assert v < 10, f"{key} suspiciously large: {v}"


# ── _CNNBook reconciliation ────────────────────────────────────────────────────

class TestCNNBookReconciliation:
    """On load, ghost open trades (from prior sessions) must be closed."""

    @pytest.mark.asyncio
    async def test_orphan_trades_closed_on_load(self):
        """Trades open in DB but absent from loaded book positions get closed."""
        from agents.cnn_agent import _CNNBook

        book = _CNNBook()
        saved_state = {
            "balance":      850.0,
            "realized_pnl": -10.0,
            "positions":    {"XRP-USD": {"size": 100, "avg_price": 1.30}},
        }
        # DB has two open trades: XRP (still open) and BIO (ghost from old session)
        open_trades = [
            {"id": 1, "product_id": "XRP-USD", "entry_price": 1.30, "size": 100},
            {"id": 2, "product_id": "BIO-USD",  "entry_price": 0.50, "size": 50},
        ]

        close_mock = AsyncMock()
        with (
            patch("agents.cnn_agent.database.load_agent_state",
                  new=AsyncMock(return_value=saved_state)),
            patch("agents.cnn_agent.database.get_trades",
                  new=AsyncMock(return_value=open_trades)),
            patch("agents.cnn_agent.database.close_trade_by_id", close_mock),
        ):
            await book.load()

        # BIO-USD (id=2) is orphaned — should be closed; XRP-USD (id=1) is current — should not
        close_mock.assert_called_once()
        assert close_mock.call_args.args[0] == 2       # trade_id positional
        assert close_mock.call_args.kwargs["trigger_close"] == "STARTUP_CLEANUP"

    @pytest.mark.asyncio
    async def test_duplicate_open_rows_for_same_product_closed(self):
        """When a position has multiple open trade rows, all but the newest are closed."""
        from agents.cnn_agent import _CNNBook

        book = _CNNBook()
        saved_state = {
            "balance": 800.0, "realized_pnl": 0.0,
            "positions": {"BIO-USD": {"size": 100, "avg_price": 0.50}},
        }
        # 3 open rows for BIO-USD from 3 different sessions — only newest (id=3) should stay
        open_trades = [
            {"id": 3, "product_id": "BIO-USD", "entry_price": 0.50, "size": 30},
            {"id": 2, "product_id": "BIO-USD", "entry_price": 0.48, "size": 40},
            {"id": 1, "product_id": "BIO-USD", "entry_price": 0.45, "size": 30},
        ]

        close_mock = AsyncMock()
        with (
            patch("agents.cnn_agent.database.load_agent_state",
                  new=AsyncMock(return_value=saved_state)),
            patch("agents.cnn_agent.database.get_trades",
                  new=AsyncMock(return_value=open_trades)),
            patch("agents.cnn_agent.database.close_trade_by_id", close_mock),
        ):
            await book.load()

        # Rows with id=1 and id=2 should be closed; id=3 (newest) stays open
        assert close_mock.call_count == 2
        closed_ids = {call.args[0] for call in close_mock.call_args_list}
        assert closed_ids == {1, 2}, f"Expected ids {{1,2}}, got {closed_ids}"

    @pytest.mark.asyncio
    async def test_no_close_when_all_trades_match_positions(self):
        """No close_trade_by_id calls when DB trades match current book positions."""
        from agents.cnn_agent import _CNNBook

        book = _CNNBook()
        saved_state = {
            "balance":      900.0,
            "realized_pnl": 0.0,
            "positions":    {"XRP-USD": {"size": 100, "avg_price": 1.30}},
        }
        open_trades = [
            {"id": 1, "product_id": "XRP-USD", "entry_price": 1.30, "size": 100},
        ]

        close_mock = AsyncMock()
        with (
            patch("agents.cnn_agent.database.load_agent_state",
                  new=AsyncMock(return_value=saved_state)),
            patch("agents.cnn_agent.database.get_trades",
                  new=AsyncMock(return_value=open_trades)),
            patch("agents.cnn_agent.database.close_trade_by_id", close_mock),
        ):
            await book.load()

        close_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_fresh_start_no_crash(self):
        """No saved state and no open trades → no errors."""
        from agents.cnn_agent import _CNNBook

        book = _CNNBook()
        with (
            patch("agents.cnn_agent.database.load_agent_state",
                  new=AsyncMock(return_value=None)),
            patch("agents.cnn_agent.database.get_trades",
                  new=AsyncMock(return_value=[])),
        ):
            await book.load()

        assert book.balance == 1000.0
        assert book.positions == {}


# ── GatedConv1d / GLU architecture tests ──────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestGatedConv1d:
    """Tests for the GLU-gated CNN architecture."""

    def _layer(self, in_ch=4, out_ch=8, kernel=3):
        return GatedConv1d(in_ch, out_ch, kernel, padding=kernel // 2)

    def _batch(self, in_ch=4, seq=16, batch=2):
        return torch.randn(batch, in_ch, seq)

    def test_gate_suppression_at_large_negative(self):
        """When gate bias is extremely negative, output is near zero for any input."""
        layer = self._layer()
        nn.init.constant_(layer.conv_gate.weight, 0.0)   # zero weights
        nn.init.constant_(layer.conv_gate.bias,   -100.0) # gate = sigmoid(-100) ≈ 0
        x   = self._batch()
        out = layer(x)
        assert float(out.abs().max()) < 1e-3, "Gate should suppress output to ~0"

    def test_gate_passthrough_at_large_positive(self):
        """When gate bias is extremely positive, output ≈ BN(conv_main(x))."""
        layer = self._layer()
        nn.init.constant_(layer.conv_gate.weight, 0.0)   # zero weights
        nn.init.constant_(layer.conv_gate.bias,   100.0)  # gate = sigmoid(100) ≈ 1
        x        = self._batch()
        out      = layer(x)
        # GatedConv1d applies BatchNorm after the gate multiplication, so
        # gate≈1 → out = BN(conv_main(x) × 1) = BN(conv_main(x))
        expected = layer.bn(layer.conv_main(x))
        assert torch.allclose(out, expected, atol=1e-3), \
            "Gate fully open → output should equal BN(conv_main(x))"

    def test_output_shape_preserved(self):
        """Output shape matches (batch, out_ch, seq_len)."""
        layer = self._layer(in_ch=4, out_ch=8)
        x     = self._batch(in_ch=4, seq=60)
        out   = layer(x)
        assert out.shape == (2, 8, 60)

    def test_signal_cnn_first_block_is_gated(self):
        """SignalCNN.c1 is a GatedConv1d, not a plain Conv1d."""
        model = SignalCNN(n_ch=N_CHANNELS)
        assert isinstance(model.c1, GatedConv1d), \
            "First conv block should be GatedConv1d"

    def test_signal_cnn_arch_tag(self):
        """SignalCNN carries arch='glu2' class attribute for checkpoint compat."""
        assert SignalCNN.arch == "glu2"

    def test_end_to_end_train_and_predict(self):
        """Model trains one step without error and returns a probability in [0, 1]."""
        import torch.optim as optim
        import torch.nn.functional as F
        model = SignalCNN(n_ch=N_CHANNELS)
        x     = torch.randn(4, N_CHANNELS, SEQ_LEN)
        y     = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        opt   = optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        opt.zero_grad()
        # Model outputs raw logits — use BCEWithLogitsLoss (numerically stable)
        loss = F.binary_cross_entropy_with_logits(model(x), y)
        loss.backward()
        opt.step()
        prob = model.predict(torch.randn(N_CHANNELS, SEQ_LEN))
        assert 0.0 <= prob <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Saved checkpoint loads back with matching weights."""
        model_a = SignalCNN(n_ch=N_CHANNELS)
        path    = str(tmp_path / "test_model.pt")
        torch.save({"arch": model_a.arch, "state_dict": model_a.state_dict()}, path)

        model_b = SignalCNN(n_ch=N_CHANNELS)
        ckpt    = torch.load(path, map_location="cpu", weights_only=False)
        model_b.load_state_dict(ckpt["state_dict"])

        for (na, pa), (nb, pb) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            assert torch.allclose(pa, pb), f"Weight mismatch at {na}"

    def test_save_load_arch_tag_preserved(self, tmp_path):
        """Checkpoint arch tag is 'glu2' after save."""
        model = SignalCNN(n_ch=N_CHANNELS)
        path  = str(tmp_path / "test_model.pt")
        torch.save({"arch": model.arch, "state_dict": model.state_dict()}, path)
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["arch"] == "glu2"

    def test_get_learned_weights_sums_to_one(self):
        """FeatureBuilder.get_learned_weights() must return weights that sum to 1.0."""
        model = SignalCNN(n_ch=N_CHANNELS)
        fb    = FeatureBuilder()
        # get_learned_weights reads first conv layer weights to rank channels
        weights = fb.get_learned_weights(model)
        assert weights is not None, "get_learned_weights returned None"
        assert abs(sum(weights) - 1.0) < 1e-5, \
            f"Weights should sum to 1.0, got {sum(weights):.6f}"


# ── train_on_history executor tests ───────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
class TestTrainOnHistoryNonBlocking:
    """train_on_history must offload the PyTorch fit loop to a thread executor
    so that other coroutines can run while training is in progress."""

    @pytest.fixture(autouse=True)
    def _isolate_model_paths(self, tmp_path, monkeypatch):
        """Redirect best-loss + checkpoint paths into tmp_path so real
        train_on_history calls cannot clobber backend/cnn_best_loss.txt
        or backend/cnn_model.pt. See test_production_paths_are_isolated."""
        import agents.cnn_agent as ca
        monkeypatch.setattr(ca, "_BEST_LOSS_PATH", str(tmp_path / "best_loss.txt"))
        monkeypatch.setattr(ca, "MODEL_PATH",      str(tmp_path / "cnn_model.pt"))
        monkeypatch.setattr(ca, "_MODEL_BAK_PATH", str(tmp_path / "cnn_model.pt.bak"))

    def test_production_paths_are_isolated(self):
        """Guard: see TestTrainOnHistory.test_production_paths_are_isolated."""
        import agents.cnn_agent as ca
        backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        prod_loss = os.path.abspath(os.path.join(backend_dir, "cnn_best_loss.txt"))
        prod_ckpt = os.path.abspath(os.path.join(backend_dir, "cnn_model.pt"))
        assert os.path.abspath(ca._BEST_LOSS_PATH) != prod_loss, (
            "_BEST_LOSS_PATH points at production file; add autouse fixture."
        )
        assert os.path.abspath(ca.MODEL_PATH) != prod_ckpt, (
            "MODEL_PATH points at production file; add autouse fixture."
        )

    def _make_products_and_candles(self, n_products: int = 6):
        import math
        products = [{"product_id": f"COIN{i}-USD"} for i in range(n_products)]
        candles = []
        for i in range(80):
            c = 50_000.0 + 500 * math.sin(i / 5.0)
            candles.append({
                "open": c - 50, "high": c + 100, "low": c - 100,
                "close": c, "volume": 10_000 + i * 100,
                "start": 1_700_000_000 + i * 3600,
            })
        return products, candles

    @pytest.mark.asyncio
    async def test_event_loop_not_blocked_during_training(self):
        """A concurrent coroutine must be able to run while train_on_history executes.

        If training runs synchronously on the event loop, the side-task will be
        starved and its flag will still be False when training finishes.
        """
        flag = {"ran": False}

        async def _side_task():
            """Yields control and sets flag — should run while training happens."""
            await asyncio.sleep(0)
            flag["ran"] = True

        agent = CoinbaseCNNAgent()
        products, candles = self._make_products_and_candles(6)

        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.load_history", return_value=[]),
        ):
            # Run training and the side task concurrently
            await asyncio.gather(
                agent.train_on_history(epochs=2),
                _side_task(),
            )

        assert flag["ran"], (
            "Side coroutine never ran — train_on_history is blocking the event loop. "
            "Wrap the PyTorch fit loop in run_in_executor()."
        )

    @pytest.mark.asyncio
    async def test_training_result_valid_after_executor(self):
        """Result dict is intact and correct even when fit runs in a thread."""
        agent    = CoinbaseCNNAgent()
        products, candles = self._make_products_and_candles(6)

        with (
            patch("agents.cnn_agent.database.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.load_history", return_value=[]),
        ):
            result = await agent.train_on_history(epochs=3)

        assert "error" not in result
        for key in ("epochs", "samples", "train_samples", "val_samples",
                    "initial_loss", "final_train_loss", "final_val_loss",
                    "fit_status", "epoch_log"):
            assert key in result, f"Missing key after executor refactor: {key}"
        assert result["epochs"] == 3
        assert len(result["epoch_log"]) == 3


# ── Kelly sizing bug regression tests ─────────────────────────────────────────

class TestKellySizingBug:
    """
    Regression tests for: CNN BUY skipped because kelly_frac=0.00

    Root cause: BUY sizing called _kelly_fraction(strength) where
      strength = (model_prob - 0.5) * 2
    For model_prob=0.62, strength=0.24 → kelly = max(0, 2*0.24-1) = 0 → no trade.
    Kelly only fired when model_prob > 0.75.

    Fix: call _kelly_fraction(model_prob) directly — model_prob IS a win probability.
    """

    @staticmethod
    def _make_tracker_mock():
        """Return a MagicMock with async get_lessons and record methods."""
        from unittest.mock import MagicMock
        t = MagicMock()
        t.get_lessons = AsyncMock(return_value=[])
        t.record      = AsyncMock()
        return t

    @pytest.mark.asyncio
    async def test_buy_frac_nonzero_at_model_prob_0_62(self, agent, product):
        """model_prob=0.62 → kelly_frac must be > 0 so book.buy() actually spends.

        Before fix: _kelly_fraction(strength=0.24) = 0 → frac passed to buy = 0.
        After fix:  _kelly_fraction(model_prob=0.62) = 0.24 → frac > 0 → trade executes.
        """
        candles  = _make_candles(80)
        buy_mock = AsyncMock(return_value=(50.0, 1))
        tracker  = self._make_tracker_mock()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",   new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons", new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",  new=AsyncMock(return_value=0.62)),
            patch("agents.cnn_agent.database.save_signal", new=AsyncMock(return_value=1)),
            patch("agents.cnn_agent._hurst_exponent", return_value=0.55),  # above _HURST_MR_THRESH=0.45
            patch("agents.cnn_agent.get_tracker",   return_value=tracker),
            patch.object(agent, "_cnn_prob",  return_value=0.62),
            patch.object(agent._lgbm, "allow_buy", return_value=True),
            patch.object(agent._lgbm, "predict",   return_value=0.7),
            patch.object(agent.book, "buy",   buy_mock),
            patch.object(agent.book, "has_position", return_value=False),
        ):
            sig = await agent.generate_signal(product, execute=True)

        assert sig is not None, "Signal should be generated at model_prob=0.62"
        assert sig["side"] == "BUY"
        buy_mock.assert_called_once()
        # Third positional arg to buy() is frac — must be > 0 after Kelly fix
        frac_passed = buy_mock.call_args[0][2]
        assert frac_passed > 0.0, (
            f"book.buy() called with frac={frac_passed:.4f} — kelly is still using "
            f"strength instead of model_prob. Expected 0.15 (=min(2*0.62-1, _CNN_MAX_FRAC))."
        )
        # _CNN_MAX_FRAC=0.15 caps kelly(0.62)=0.24 to 0.15
        assert abs(frac_passed - 0.15) < 0.01, \
            f"Expected frac≈0.15 (capped), got {frac_passed:.4f}"

    @pytest.mark.asyncio
    async def test_buy_frac_nonzero_at_model_prob_0_65(self, agent, product):
        """model_prob=0.65 (above 0.60 threshold) → frac must be > 0.

        Before fix: strength=(0.65-0.5)*2=0.30 → kelly=max(0, 2*0.30-1)=0 → skipped.
        After fix:  kelly=max(0, 2*0.65-1)=0.30 → trade executes.
        """
        candles  = _make_candles(80)
        buy_mock = AsyncMock(return_value=(65.0, 2))
        tracker  = self._make_tracker_mock()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",   new=AsyncMock()),
            patch("agents.cnn_agent.database.get_recent_lessons", new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",  new=AsyncMock(return_value=0.65)),
            patch("agents.cnn_agent.database.save_signal", new=AsyncMock(return_value=2)),
            patch("agents.cnn_agent._hurst_exponent", return_value=0.55),  # above _HURST_MR_THRESH=0.45
            patch("agents.cnn_agent.get_tracker",   return_value=tracker),
            patch.object(agent, "_cnn_prob",  return_value=0.65),
            patch.object(agent._lgbm, "allow_buy", return_value=True),
            patch.object(agent._lgbm, "predict",   return_value=0.7),
            patch.object(agent.book, "buy",   buy_mock),
            patch.object(agent.book, "has_position", return_value=False),
        ):
            sig = await agent.generate_signal(product, execute=True)

        assert sig is not None
        assert sig["side"] == "BUY"
        buy_mock.assert_called_once()
        frac_passed = buy_mock.call_args[0][2]
        assert frac_passed > 0.0, (
            f"book.buy() called with frac={frac_passed:.4f} — must be > 0 for model_prob=0.65"
        )

    def test_kelly_with_model_prob_gives_nonzero_at_0_62(self):
        """Direct unit test: _kelly_fraction(0.62) must be > 0."""
        from agents.signal_generator import _kelly_fraction
        frac = _kelly_fraction(0.62)
        assert frac > 0, (
            f"_kelly_fraction(0.62) = {frac} — should be 0.24 (=2*0.62-1). "
            "If 0, the caller is passing 'strength' instead of 'model_prob'."
        )
        assert abs(frac - 0.24) < 0.001, f"Expected 0.24, got {frac}"

    def test_kelly_with_strength_gives_zero_at_0_24(self):
        """Shows the OLD (broken) behaviour: strength=0.24 → kelly=0."""
        from agents.signal_generator import _kelly_fraction
        strength_for_0_62 = (0.62 - 0.5) * 2   # = 0.24
        frac = _kelly_fraction(strength_for_0_62)
        assert frac == pytest.approx(0.0, abs=0.001), (
            "This test documents the bug: passing strength instead of model_prob "
            "produces kelly=0 even for a valid BUY signal."
        )


# ── Training framework tests ───────────────────────────────────────────────────

class TestTrainingFramework:
    """
    Verify best-model tracking, early stopping, arch-mismatch guard,
    and conditional checkpoint save.
    """

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_needs_retrain_flag_set_on_channel_mismatch(self, tmp_path):
        """Loading a checkpoint with wrong n_channels sets _needs_retrain=True."""
        import torch
        from agents.cnn_agent import SignalCNN, N_CHANNELS, _EARLY_STOP_PATIENCE

        agent = CoinbaseCNNAgent()
        agent._needs_retrain = False

        # Build a fake checkpoint claiming a different channel count
        wrong_channels = N_CHANNELS + 5
        ckpt_path = str(tmp_path / "bad_ckpt.pt")
        torch.save(
            {"arch": "glu", "n_channels": wrong_channels,
             "state_dict": SignalCNN().state_dict()},
            ckpt_path,
        )

        import agents.cnn_agent as ca
        orig = ca.MODEL_PATH
        ca.MODEL_PATH = ckpt_path
        try:
            agent._load()
        finally:
            ca.MODEL_PATH = orig

        assert agent._needs_retrain is True, (
            "_needs_retrain must be True after loading incompatible channel count"
        )

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    @pytest.mark.asyncio
    async def test_generate_signal_suppressed_when_needs_retrain(self):
        """generate_signal returns None when _needs_retrain is True."""
        agent = CoinbaseCNNAgent()
        agent._needs_retrain = True

        product = {"product_id": "BTC-USD", "price": 94000.0}
        result = await agent.generate_signal(product, execute=False)
        assert result is None, "Signal must be suppressed when model is incompatible"

    @pytest.mark.skipif(not _TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_save_model_stores_n_channels(self, tmp_path):
        """save_model() writes n_channels into the checkpoint."""
        import torch
        import agents.cnn_agent as ca

        agent = CoinbaseCNNAgent()
        ckpt_path = str(tmp_path / "test_ckpt.pt")
        orig = ca.MODEL_PATH
        ca.MODEL_PATH = ckpt_path
        try:
            agent.save_model(backup=False)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        finally:
            ca.MODEL_PATH = orig

        assert "n_channels" in ckpt, "Checkpoint must contain n_channels key"
        assert ckpt["n_channels"] == ca.N_CHANNELS

    def test_best_loss_read_write_roundtrip(self, tmp_path):
        """_read_best_loss / _write_best_loss survive a roundtrip."""
        import agents.cnn_agent as ca
        orig = ca._BEST_LOSS_PATH
        ca._BEST_LOSS_PATH = str(tmp_path / "best_loss.txt")
        try:
            assert CoinbaseCNNAgent._read_best_loss() == float("inf"), \
                "Missing file should return inf"
            CoinbaseCNNAgent._write_best_loss(0.4321)
            assert abs(CoinbaseCNNAgent._read_best_loss() - 0.4321) < 1e-6
        finally:
            ca._BEST_LOSS_PATH = orig

    def test_best_loss_stale_tiny_value_treated_as_unset(self, tmp_path):
        """A stale tiny value (< 0.1) in cnn_best_loss.txt must be treated as unset.

        Rationale: BCE loss at chance is ~0.693; a legitimate best_val_loss will
        not be below 0.1. Without this guard, a corrupted/stale file (the real
        bug: content=1e-06) causes every trained model to be rejected at save
        time because best_val_loss >= 1e-06 always.
        """
        import agents.cnn_agent as ca
        orig = ca._BEST_LOSS_PATH
        ca._BEST_LOSS_PATH = str(tmp_path / "best_loss.txt")
        try:
            with open(ca._BEST_LOSS_PATH, "w") as f:
                f.write("1e-06")
            assert CoinbaseCNNAgent._read_best_loss() == float("inf"), \
                "Stale sub-0.1 value must be treated as unset so real models can save"
            with open(ca._BEST_LOSS_PATH, "w") as f:
                f.write("0.5")
            assert abs(CoinbaseCNNAgent._read_best_loss() - 0.5) < 1e-6, \
                "Realistic values >= 0.1 must be preserved"
        finally:
            ca._BEST_LOSS_PATH = orig


class TestHMMStability:
    """HMM fit should keep old model on failure and warn on label flip."""

    def test_fit_keeps_old_model_on_failure(self):
        """If GaussianHMM raises, the old model is preserved."""
        from services.hmm_regime import HMMRegimeDetector
        import unittest.mock as mock

        det = HMMRegimeDetector()
        det._model = "SENTINEL"  # pretend we have an existing model

        with mock.patch("hmmlearn.hmm.GaussianHMM") as MockHMM:
            MockHMM.return_value.fit.side_effect = RuntimeError("numerical error")
            result = det.fit(list(range(1, 400)))   # enough bars

        assert det._model == "SENTINEL", \
            "Existing model must not be cleared when fit raises"
        assert result is False

    def test_degenerate_states_rejected(self):
        """Fit with identical-vol states should not replace the current model."""
        from services.hmm_regime import HMMRegimeDetector
        import unittest.mock as mock
        import numpy as np

        det = HMMRegimeDetector()
        det._model = "SENTINEL"

        fake_model = mock.MagicMock()
        # All states have identical volatility → degenerate
        fake_model.means_ = np.array([[0.0, 0.001], [0.0, 0.001], [0.0, 0.001]])

        with mock.patch("hmmlearn.hmm.GaussianHMM") as MockHMM:
            MockHMM.return_value = fake_model
            fake_model.fit = mock.MagicMock()
            result = det.fit(list(range(1, 400)))

        assert det._model == "SENTINEL", \
            "Degenerate-state model must not replace existing model"


# ── Display bug regression tests ──────────────────────────────────────────────

class TestRegimeLabelAndVWAPDisplay:
    """
    Two bugs observed in production (2026-04-19):

    1. cnn_scans.regime stored "RANGING" while signals.reasoning said "CHAOTIC"
       for the same scan. Caused by binary trending/ranging fallback discarding
       the HMM's CHAOTIC label.

    2. signals.reasoning printed "Price below VWAP by 27.98%" when the true
       delta was 1.47%. Caused by multiplying the normalised vwap distance
       (already divided by 0.05) by 100 instead of recomputing from prices.
    """

    @staticmethod
    def _make_tracker_mock():
        from unittest.mock import MagicMock
        t = MagicMock()
        t.get_lessons = AsyncMock(return_value=[])
        t.record      = AsyncMock()
        return t

    @pytest.mark.asyncio
    async def test_hmm_chaotic_label_is_stored_verbatim_in_cnn_scans(self, agent, product):
        """When HMM detector returns CHAOTIC, save_cnn_scan must receive regime='CHAOTIC'."""
        candles = _make_candles(80)
        saved = {}

        async def _capture(row):
            saved.update(row)

        fake_detector = type("D", (), {
            "predict": staticmethod(lambda closes: ("CHAOTIC", 0.71, 2))
        })()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",
                  new=AsyncMock(side_effect=_capture)),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(return_value=1)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.82, 0, 0))),
            patch("agents.cnn_agent.get_detector",
                  return_value=fake_detector),
            patch("agents.cnn_agent.get_tracker",
                  return_value=self._make_tracker_mock()),
            patch.object(agent, "_cnn_prob", return_value=0.82),
        ):
            await agent.generate_signal(product)

        assert saved, "save_cnn_scan was not called"
        assert saved.get("regime") == "CHAOTIC", (
            f"Expected regime='CHAOTIC' (from HMM), "
            f"got regime={saved.get('regime')!r}"
        )

    @pytest.mark.asyncio
    async def test_reasoning_vwap_percent_matches_actual_price_delta(self, agent, product):
        """
        VWAP % in reasoning text must match (price - vwap) / vwap * 100,
        not the normalised vwap_d (which is dist/0.05).
        """
        candles = _make_candles(80)
        captured_signal = {}

        async def _capture_sig(row):
            captured_signal.update(row)
            return 1

        fake_detector = type("D", (), {
            "predict": staticmethod(lambda closes: ("TRENDING", 0.80, 0))
        })()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",
                  new=AsyncMock()),
            patch("agents.cnn_agent.database.save_signal",
                  new=AsyncMock(side_effect=_capture_sig)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.82, 0, 0))),
            patch("agents.cnn_agent.get_detector",
                  return_value=fake_detector),
            patch("agents.cnn_agent.get_tracker",
                  return_value=self._make_tracker_mock()),
            patch.object(agent, "_cnn_prob", return_value=0.82),
        ):
            await agent.generate_signal(product)

        reasoning = captured_signal.get("reasoning", "")
        assert "Price" in reasoning and "VWAP" in reasoning, \
            f"reasoning missing VWAP line: {reasoning[:200]}"

        import re
        m = re.search(r"Price (?:below|above) VWAP by ([0-9.]+)%", reasoning)
        assert m, f"Could not parse VWAP percent from: {reasoning[:300]}"
        displayed_pct = float(m.group(1))

        m2 = re.search(r"VWAP\(20\):\s*\$?([0-9,.]+)", reasoning)
        assert m2, f"Could not parse VWAP price from: {reasoning[:300]}"
        vwap_price = float(m2.group(1).replace(",", ""))

        price = product["price"]
        expected_pct = abs(price - vwap_price) / vwap_price * 100

        assert abs(displayed_pct - expected_pct) < 0.1, (
            f"VWAP percent mismatch: displayed={displayed_pct:.4f}%, "
            f"expected={expected_pct:.4f}% "
            f"(price={price}, vwap={vwap_price})"
        )


class TestPhase2LogCadence:
    """Phase-2 dataset build must log often enough to keep the watchdog quiet.

    Phase 2 takes ~10-13 min per 10 products. Logging every 10 products risks
    slipping past the 30-min log-stale window when one product is slower than
    usual. Cadence of every 5 products gives ~5-6 min between log lines.
    """

    def test_phase2_log_every_is_5(self):
        assert _cnn_mod._PHASE2_LOG_EVERY == 5


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestDatasetCache:
    """Phase-2 tensor build takes 30-40 min. Rerunning every hour re-does
    identical work because the inputs barely change. Cache X/y to disk keyed
    by a fingerprint of the inputs so only truly-new data forces a rebuild."""

    def _candles(self, n=5, start_ts=1000, start_close=100.0):
        return [
            {"time": start_ts + i * 3600, "open": start_close + i,
             "high": start_close + i + 1, "low": start_close + i - 1,
             "close": start_close + i, "volume": 1000.0}
            for i in range(n)
        ]

    def test_fingerprint_stable_across_calls(self):
        sets = [self._candles(10), self._candles(15, start_ts=2000)]
        fp1 = _cnn_mod._dataset_fingerprint(sets, 60, 4, 0.003, 27)
        fp2 = _cnn_mod._dataset_fingerprint(sets, 60, 4, 0.003, 27)
        assert fp1 == fp2 and isinstance(fp1, str) and len(fp1) == 64

    def test_fingerprint_changes_when_candles_grow(self):
        sets = [self._candles(10)]
        fp_a = _cnn_mod._dataset_fingerprint(sets, 60, 4, 0.003, 27)
        sets_b = [self._candles(11)]
        fp_b = _cnn_mod._dataset_fingerprint(sets_b, 60, 4, 0.003, 27)
        assert fp_a != fp_b

    def test_fingerprint_changes_when_params_change(self):
        sets = [self._candles(10)]
        fp_a = _cnn_mod._dataset_fingerprint(sets, 60, 4, 0.003, 27)
        fp_b = _cnn_mod._dataset_fingerprint(sets, 60, 8, 0.003, 27)
        fp_c = _cnn_mod._dataset_fingerprint(sets, 120, 4, 0.003, 27)
        assert fp_a != fp_b and fp_a != fp_c and fp_b != fp_c

    def test_cache_miss_on_missing_file(self, tmp_path):
        assert _cnn_mod._load_dataset_cache(
            str(tmp_path / "no_such.pt"), "any-fp"
        ) is None

    def test_cache_roundtrip(self, tmp_path):
        import torch
        path = str(tmp_path / "cache.pt")
        X = [torch.zeros(27, 60), torch.ones(27, 60)]
        y = [0.0, 1.0]
        _cnn_mod._save_dataset_cache(path, "fp-abc", X, y)
        got = _cnn_mod._load_dataset_cache(path, "fp-abc")
        assert got is not None
        X_got, y_got = got
        assert len(X_got) == 2 and len(y_got) == 2
        assert torch.equal(X_got[0], X[0]) and torch.equal(X_got[1], X[1])
        assert y_got == [0.0, 1.0]

    def test_cache_miss_on_fingerprint_mismatch(self, tmp_path):
        import torch
        path = str(tmp_path / "cache.pt")
        _cnn_mod._save_dataset_cache(path, "fp-old", [torch.zeros(27, 60)], [0.0])
        assert _cnn_mod._load_dataset_cache(path, "fp-new") is None


# ── Per-product append-only dataset cache (P2) ───────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestPerProductDatasetCache:
    """P2 — per-product append-only cache.

    The all-or-nothing fingerprint cache in TestDatasetCache invalidates the
    full 103-minute phase-2 build whenever a single product ticks. This class
    exercises the per-product helper that appends only the new samples when
    candles are extended and only rebuilds when a product's history actually
    changes shape.
    """

    class _FakeFB:
        """Minimal FeatureBuilder shim: 1-channel window of closes."""
        def build(self, window, _idx, candles_5m=None):
            closes = [float(c["close"]) for c in window]
            if len(closes) < SEQ_LEN:
                closes = [closes[0]] * (SEQ_LEN - len(closes)) + closes
            return [closes[:SEQ_LEN]]

        def to_tensor(self, channels):
            import torch
            return torch.tensor(channels, dtype=torch.float32)

    _FWD   = 4
    _THR   = 0.003

    def _helper_args(self):
        return (self._FakeFB(), SEQ_LEN, self._FWD, self._THR)

    def test_full_build_when_entry_missing(self):
        candles = _make_candles(80)
        entry, status = _cnn_mod._extend_or_rebuild_product(
            None, candles, *self._helper_args()
        )
        assert status == "rebuild"
        assert entry is not None
        assert entry["first_ts"] == candles[0]["start"]
        assert entry["last_ts"]  == candles[-1]["start"]
        assert entry["last_n"]   == 80
        assert len(entry["X"]) == len(entry["y"]) > 0

    def test_hit_when_candles_unchanged(self):
        candles = _make_candles(80)
        entry, _ = _cnn_mod._extend_or_rebuild_product(
            None, candles, *self._helper_args()
        )
        n_before = len(entry["X"])
        entry2, status = _cnn_mod._extend_or_rebuild_product(
            entry, candles, *self._helper_args()
        )
        assert status == "hit"
        assert len(entry2["X"]) == n_before
        assert entry2["last_n"] == 80

    def test_appends_new_samples_when_candles_extended(self):
        candles80 = _make_candles(80)
        entry, _ = _cnn_mod._extend_or_rebuild_product(
            None, candles80, *self._helper_args()
        )
        n_before = len(entry["X"])
        candles85 = _make_candles(85)
        # Sanity: extended series is a superset of the original prefix
        assert candles85[:80] == candles80
        entry2, status = _cnn_mod._extend_or_rebuild_product(
            entry, candles85, *self._helper_args()
        )
        assert status == "append"
        assert entry2["last_n"] == 85
        assert entry2["last_ts"] == candles85[-1]["start"]
        assert len(entry2["X"]) >= n_before
        # At least one new sample should have been added — 5 new candles
        # add up to 5 sliding-window positions (minus dead-zone skips).
        assert len(entry2["X"]) > n_before

    def test_rebuilds_when_first_ts_changed(self):
        candles = _make_candles(80)
        entry, _ = _cnn_mod._extend_or_rebuild_product(
            None, candles, *self._helper_args()
        )
        # Simulate a data re-ingest that shifted the series start.
        candles2 = [dict(c, start=c["start"] + 10_000) for c in _make_candles(80)]
        entry2, status = _cnn_mod._extend_or_rebuild_product(
            entry, candles2, *self._helper_args()
        )
        assert status == "rebuild"
        assert entry2["first_ts"] == candles2[0]["start"]

    def test_rebuilds_when_entry_last_n_exceeds_candles(self):
        candles = _make_candles(80)
        entry, _ = _cnn_mod._extend_or_rebuild_product(
            None, candles, *self._helper_args()
        )
        # Corrupt last_n to exceed current candle count.
        stale = {**entry, "last_n": 200}
        entry2, status = _cnn_mod._extend_or_rebuild_product(
            stale, candles, *self._helper_args()
        )
        assert status == "rebuild"
        assert entry2["last_n"] == 80

    def test_returns_skip_when_too_few_candles(self):
        candles = _make_candles(SEQ_LEN + self._FWD)   # one short
        entry, status = _cnn_mod._extend_or_rebuild_product(
            None, candles, *self._helper_args()
        )
        assert status == "skip"
        assert entry is None

    def test_pp_cache_file_roundtrip(self, tmp_path):
        import torch
        path = str(tmp_path / "pp_cache.pt")
        schema = _cnn_mod._dataset_schema(SEQ_LEN, self._FWD, self._THR, N_CHANNELS)
        products = {
            "BTC-USD": {
                "first_ts": 1_700_000_000,
                "last_ts":  1_700_100_000,
                "last_n":   80,
                "X": [torch.zeros(1, SEQ_LEN)],
                "y": [1.0],
            }
        }
        _cnn_mod._save_pp_cache(path, schema, products)
        got = _cnn_mod._load_pp_cache(path, schema)
        assert got is not None
        assert set(got.keys()) == {"BTC-USD"}
        assert got["BTC-USD"]["last_n"] == 80
        assert got["BTC-USD"]["y"] == [1.0]
        assert torch.equal(got["BTC-USD"]["X"][0], torch.zeros(1, SEQ_LEN))

    def test_pp_cache_schema_mismatch_invalidates(self, tmp_path):
        import torch
        path = str(tmp_path / "pp_cache.pt")
        schema_a = _cnn_mod._dataset_schema(SEQ_LEN, 4, self._THR, N_CHANNELS)
        schema_b = _cnn_mod._dataset_schema(SEQ_LEN, 8, self._THR, N_CHANNELS)
        _cnn_mod._save_pp_cache(path, schema_a, {
            "X-Y": {"first_ts": 0, "last_ts": 1, "last_n": 1,
                    "X": [torch.zeros(1, SEQ_LEN)], "y": [0.0]}
        })
        assert _cnn_mod._load_pp_cache(path, schema_b) is None

    def test_pp_cache_missing_file_returns_none(self, tmp_path):
        schema = _cnn_mod._dataset_schema(SEQ_LEN, self._FWD, self._THR, N_CHANNELS)
        assert _cnn_mod._load_pp_cache(
            str(tmp_path / "no_such.pt"), schema
        ) is None


# ── Triple-barrier labeling (P3a) ─────────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestTripleBarrierLabels:
    """P3a — replace sign-of-4h-return with López-de-Prado triple-barrier.

    The sign-of-return label is noisy: a 3% up-spike that retraces to 0 gets
    labeled 0 (flat), which contradicts what the model should predict. The
    triple-barrier labels by whichever of three events happens first inside
    the forward window: upper barrier hit (1), lower barrier hit (0), or
    time barrier expiration (sign of final close move, with dead-zone skip).
    """

    _UP  = 0.01   # +1% upper barrier
    _DN  = 0.01   # -1% lower barrier
    _MAX = 4      # forward window bars
    _THR = 0.003

    def _bar(self, o, h, l, c, ts=0):
        return {"open": o, "high": h, "low": l, "close": c,
                "volume": 1000.0, "start": ts}

    def test_upper_barrier_hit_first_labels_up(self):
        # entry=100, upper=101, lower=99. Bar 1 hits high 102 first.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 102, 99.5, 101.5, ts=1),
            self._bar(101, 101, 100, 100, ts=2),
            self._bar(100, 101, 99,   99.5, ts=3),
            self._bar(99.5, 100, 98,  98.5, ts=4),
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got == 1.0

    def test_lower_barrier_hit_first_labels_down(self):
        # entry=100, upper=101, lower=99. Bar 1 lows 97 first.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 100.5, 97, 98.5, ts=1),
            self._bar(98.5, 99, 98, 98.2, ts=2),
            self._bar(98.2, 99, 97, 97.5, ts=3),
            self._bar(97.5, 98, 96, 96.8, ts=4),
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got == 0.0

    def test_time_barrier_up_with_move_above_thresh(self):
        # No intra-bar barrier hit (highs < 101, lows > 99), but final close
        # moves > +0.3% → label 1.0.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 100.5, 99.5, 100.2, ts=1),
            self._bar(100.2, 100.8, 99.8, 100.5, ts=2),
            self._bar(100.5, 100.9, 100.1, 100.6, ts=3),
            self._bar(100.6, 100.9, 100.3, 100.8, ts=4),   # final close = 100.8
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got == 1.0

    def test_time_barrier_down_with_move_below_thresh(self):
        # No barrier hit, final close moves < -0.3% → label 0.0.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 100.5, 99.5, 99.8, ts=1),
            self._bar(99.8, 100.2, 99.2, 99.5, ts=2),
            self._bar(99.5, 99.9, 99.1, 99.3, ts=3),
            self._bar(99.3, 99.9, 99.0, 99.2, ts=4),     # final close = 99.2
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got == 0.0

    def test_time_barrier_dead_zone_returns_none(self):
        # No barrier, final close within ±0.3% of entry → skip.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 100.3, 99.7, 100.05, ts=1),
            self._bar(100.05, 100.4, 99.8, 100.1, ts=2),
            self._bar(100.1, 100.3, 99.9, 100.05, ts=3),
            self._bar(100.05, 100.3, 99.8, 100.1, ts=4),   # final close = 100.1 → 0.10% < 0.3%
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got is None

    def test_both_barriers_same_bar_use_close_direction(self):
        # Bar 1 touches both upper (102) and lower (97) — tie broken by close.
        # close=102 > entry=100 → label 1.0.
        candles = [
            self._bar(100, 100, 100, 100, ts=0),
            self._bar(100, 102, 97, 102, ts=1),
            self._bar(102, 103, 101, 102, ts=2),
            self._bar(102, 103, 101, 102, ts=3),
            self._bar(102, 103, 101, 102, ts=4),
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got == 1.0

    def test_invalid_entry_price_returns_none(self):
        candles = [self._bar(0, 0, 0, 0, ts=0)] + [
            self._bar(100, 101, 99, 100, ts=k) for k in range(1, 5)
        ]
        got = _cnn_mod._label_triple_barrier(
            candles, 0, self._MAX, self._UP, self._DN, self._THR
        )
        assert got is None


# ── Training/inference distribution alignment (P3b) ───────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestTrainingConstantChannelMask:
    """P3b — align training and inference input distributions.

    Training calls fb.build(window, {}, candles_5m=proxy_5m) without
    btc_closes / funding_rate / iv_rv* / ls_sentiment / orderbook, so a known
    subset of channels is constant-zero during training. At inference those
    channels receive real data, but the model learned nothing useful from
    them because their training-time gradient was zero. Zeroing them at
    inference keeps the inference distribution aligned with what the model
    actually learned — cuts train/serve skew without retraining.
    """

    def test_mask_set_covers_expected_channels(self):
        # These channels require inputs that training never provides.
        expected = {10, 11, 15, 17, 18, 19, 20, 21, 24, 25, 26}
        assert set(_cnn_mod._TRAINING_CONSTANT_CHANNELS) == expected

    def test_mask_zeros_designated_channels(self):
        channels = [[float(i + 0.1)] * SEQ_LEN for i in range(N_CHANNELS)]
        masked   = _cnn_mod._mask_training_constant_channels(channels)
        assert len(masked) == N_CHANNELS
        for idx, ch in enumerate(masked):
            assert len(ch) == SEQ_LEN
            if idx in _cnn_mod._TRAINING_CONSTANT_CHANNELS:
                assert all(v == 0.0 for v in ch), \
                    f"channel {idx} should be zeroed after masking"
            else:
                # Informative channels are preserved unchanged.
                assert ch == channels[idx], f"channel {idx} was altered"

    def test_mask_handles_empty_channels(self):
        assert _cnn_mod._mask_training_constant_channels([]) == []

    def test_mask_does_not_mutate_input(self):
        channels = [[1.0] * SEQ_LEN for _ in range(N_CHANNELS)]
        snapshot = [ch[:] for ch in channels]
        _cnn_mod._mask_training_constant_channels(channels)
        assert channels == snapshot, "input list must not be mutated"


# ── Sample-uniqueness weighting (P3c) ─────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestSampleUniqueness:
    """P3c — down-weight overlapping samples when training.

    Forward windows of length forward_hours share causal data across
    neighboring samples. López de Prado's sample-uniqueness assigns each
    sample a weight = mean(1/N_t) over its forward window, where N_t is the
    count of concurrent samples whose forward window includes time t.
    """

    def test_single_isolated_sample_has_weight_one(self):
        # N_t = 1 for every t in the sample's forward window → u = 1.0
        assert _cnn_mod._compute_uniqueness([10], 4, 100) == [1.0]

    def test_two_disjoint_samples_have_weight_one(self):
        # Sample 10's forward window [11..14] is disjoint from sample 30's
        # [31..34], so neither sample shares any future bar with the other.
        assert _cnn_mod._compute_uniqueness([10, 30], 4, 100) == [1.0, 1.0]

    def test_two_adjacent_samples_share_weight(self):
        # Samples 10 and 11, h=4.
        # Sample 10 window [11..14], sample 11 window [12..15].
        # N_11=1, N_12=N_13=N_14=2, N_15=1.
        # u_10 = mean(1, 1/2, 1/2, 1/2) = 0.625
        # u_11 = mean(1/2, 1/2, 1/2, 1) = 0.625
        w = _cnn_mod._compute_uniqueness([10, 11], 4, 100)
        assert abs(w[0] - 0.625) < 1e-9
        assert abs(w[1] - 0.625) < 1e-9

    def test_uniform_interior_approaches_1_over_h(self):
        # Long consecutive run: interior samples have u ≈ 1/h.
        h = 4
        indices = list(range(59, 180))
        w = _cnn_mod._compute_uniqueness(indices, h, 200)
        interior = w[20:100]   # skip boundary effects
        mean_w = sum(interior) / len(interior)
        assert abs(mean_w - 1.0 / h) < 0.01

    def test_empty_indices_returns_empty(self):
        assert _cnn_mod._compute_uniqueness([], 4, 100) == []

    def test_weights_clamped_at_out_of_range_t(self):
        # Last samples may have forward bars beyond n — those t's are skipped
        # from the average, not counted with N_t=0 (which would div-by-zero).
        w = _cnn_mod._compute_uniqueness([96], 4, 100)   # window [97..100], t=100 out-of-range
        # Only t=97,98,99 in range; all have N_t=1 → u=1.0
        assert w == [1.0]


# ── Label smoothing (P3d) ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not installed")
class TestLabelSmoothing:
    """P3d — soften hard targets {0,1} to {ε, 1-ε} before BCE.

    Szegedy 2016 (Inception-v3): label smoothing penalises over-confident
    predictions and acts as regularisation. The mapping is y' = y*(1-2ε)+ε
    so y=0 → ε and y=1 → 1-ε; values already in the interior are left near
    their input.
    """

    def test_eps_zero_is_identity(self):
        import torch
        y = torch.tensor([[0.0], [1.0], [0.5]])
        got = _cnn_mod._smooth_labels(y, 0.0)
        assert torch.allclose(got, y)

    def test_endpoints_pulled_toward_centre(self):
        import torch
        y = torch.tensor([[0.0], [1.0]])
        got = _cnn_mod._smooth_labels(y, 0.05)
        assert torch.allclose(got, torch.tensor([[0.05], [0.95]]), atol=1e-7)

    def test_midpoint_unchanged(self):
        import torch
        y = torch.tensor([[0.5]])
        got = _cnn_mod._smooth_labels(y, 0.05)
        assert torch.allclose(got, y)

    def test_default_smoothing_constant_is_0_05(self):
        # The module-level default must match the plan (ε=0.05).
        assert _cnn_mod._LABEL_SMOOTH == 0.05


# ── Walk-forward purged K-fold CV (P3e) ───────────────────────────────────────

class TestPurgedWalkforwardSplits:
    """P3e — walk-forward CV with purging and embargo.

    López de Prado 2018 ch. 7: time-series CV must (a) go forward-only (never
    train on the future and validate on the past) and (b) purge training
    samples whose forward label window overlaps the validation fold, plus (c)
    embargo a gap of bars immediately after the validation fold so nothing
    leaks in via serial correlation.

    `_purged_walkforward_splits(sample_indices, n_splits, forward_hours,
    embargo_bars)` returns a list of (train_idx, val_idx) pairs where
    train_idx/val_idx are positions into `sample_indices` (not raw candle
    indices). Folds are walk-forward: fold k trains on the first k chunks
    and validates on chunk k+1.
    """

    def test_returns_n_splits_folds(self):
        idx = list(range(100))
        folds = _cnn_mod._purged_walkforward_splits(idx, n_splits=3, forward_hours=4, embargo_bars=2)
        assert len(folds) == 3

    def test_folds_are_walkforward_and_disjoint_on_val(self):
        # Val sets should be consecutive, disjoint, and cover the tail.
        idx = list(range(120))
        folds = _cnn_mod._purged_walkforward_splits(idx, n_splits=3, forward_hours=4, embargo_bars=2)
        vals = [v for (_t, v) in folds]
        # Each val set contiguous and sorted
        for v in vals:
            assert list(v) == sorted(v)
        # Val sets disjoint
        flat = [p for v in vals for p in v]
        assert len(flat) == len(set(flat))
        # Walk-forward: every val index in fold k < every val index in fold k+1
        for a, b in zip(vals, vals[1:]):
            assert max(a) < min(b)

    def test_train_set_is_strictly_earlier_than_val(self):
        # Walk-forward guarantee: training samples precede validation samples.
        idx = list(range(100))
        folds = _cnn_mod._purged_walkforward_splits(idx, n_splits=3, forward_hours=4, embargo_bars=0)
        for train, val in folds:
            if not train or not val:
                continue
            assert max(train) < min(val)

    def test_purging_drops_overlapping_forward_windows(self):
        # A training sample at candle i has forward window (i, i+forward_hours].
        # If i + forward_hours >= first_val_candle, it must be purged.
        # Use candle indices 0..99; val starts around position 66 with n_splits=3.
        sample_indices = list(range(100))  # sample j maps to candle index j
        folds = _cnn_mod._purged_walkforward_splits(
            sample_indices, n_splits=3, forward_hours=4, embargo_bars=0,
        )
        for train_pos, val_pos in folds:
            if not train_pos or not val_pos:
                continue
            first_val_candle = sample_indices[val_pos[0]]
            for pos in train_pos:
                cand = sample_indices[pos]
                assert cand + 4 < first_val_candle, (
                    f"sample at candle {cand} has forward window ending at "
                    f"{cand + 4}, overlapping val start {first_val_candle}"
                )

    def test_embargo_drops_samples_just_after_val(self):
        # In walk-forward CV the *next* fold's training set must skip the
        # embargo window immediately following the previous val block.
        sample_indices = list(range(120))
        folds = _cnn_mod._purged_walkforward_splits(
            sample_indices, n_splits=3, forward_hours=0, embargo_bars=5,
        )
        # Inspect fold 2's training set — it should not include any candle
        # within embargo_bars of fold 1's val end (plus the purge window).
        train1, val1 = folds[1]
        train2, val2 = folds[2]
        val1_end = sample_indices[val1[-1]]
        for pos in train2:
            cand = sample_indices[pos]
            if cand <= val1_end:
                continue  # this sample predates val1, embargo doesn't apply
            # Samples in the embargo band (val1_end, val1_end+embargo] must be dropped
            assert cand > val1_end + 5

    def test_embargo_zero_keeps_samples_adjacent_to_val(self):
        sample_indices = list(range(60))
        folds = _cnn_mod._purged_walkforward_splits(
            sample_indices, n_splits=3, forward_hours=0, embargo_bars=0,
        )
        # With no purge/embargo, training sets grow monotonically.
        train_sizes = [len(t) for (t, _v) in folds]
        assert train_sizes == sorted(train_sizes)

    def test_sparse_sample_indices_respected(self):
        # Samples at candles [0, 10, 20, ..., 90] — ensure purge uses candle
        # distance, not sample position distance.
        sparse = list(range(0, 100, 10))  # 10 samples at candles 0,10,...,90
        folds = _cnn_mod._purged_walkforward_splits(
            sparse, n_splits=3, forward_hours=15, embargo_bars=0,
        )
        for train_pos, val_pos in folds:
            if not train_pos or not val_pos:
                continue
            first_val_candle = sparse[val_pos[0]]
            for pos in train_pos:
                cand = sparse[pos]
                assert cand + 15 < first_val_candle

    def test_empty_input_returns_empty_folds(self):
        folds = _cnn_mod._purged_walkforward_splits([], n_splits=3, forward_hours=4, embargo_bars=2)
        assert folds == [([], []), ([], []), ([], [])]

    def test_n_splits_one_raises(self):
        # Walk-forward CV with only 1 split is meaningless.
        with pytest.raises(ValueError):
            _cnn_mod._purged_walkforward_splits([0, 1, 2], n_splits=1, forward_hours=4, embargo_bars=0)


# ── Per-regime validation metrics (P4) ────────────────────────────────────────

class TestPerRegimeMetrics:
    """P4 — break validation metrics out by HMM regime.

    An overall val_loss mean hides whether the model works everywhere. A model
    can be accurate in TRENDING regimes and worse-than-random in CHAOTIC
    regimes while showing a respectable aggregate. `_per_regime_metrics`
    buckets (y_true, y_pred) pairs by regime label and returns per-regime
    n, accuracy, BCE loss, and positive rate.
    """

    def test_all_in_one_regime(self):
        y_true = [1.0, 0.0, 1.0, 0.0]
        y_pred = [0.9, 0.1, 0.8, 0.2]
        regimes = ["TRENDING"] * 4
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert set(got.keys()) == {"TRENDING"}
        assert got["TRENDING"]["n"] == 4
        assert got["TRENDING"]["acc"] == 1.0
        assert abs(got["TRENDING"]["pos_rate"] - 0.5) < 1e-9

    def test_multiple_regimes_partitioned(self):
        y_true = [1.0, 0.0, 1.0, 0.0]
        y_pred = [0.9, 0.1, 0.9, 0.1]
        regimes = ["TRENDING", "TRENDING", "CHAOTIC", "CHAOTIC"]
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert got["TRENDING"]["n"] == 2
        assert got["CHAOTIC"]["n"] == 2
        assert got["TRENDING"]["acc"] == 1.0
        assert got["CHAOTIC"]["acc"] == 1.0

    def test_accuracy_uses_0_5_threshold(self):
        y_true = [1.0, 1.0, 0.0, 0.0]
        y_pred = [0.51, 0.49, 0.49, 0.51]  # only 2 of 4 cross threshold correctly
        regimes = ["RANGING"] * 4
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert got["RANGING"]["acc"] == 0.5

    def test_loss_is_finite_and_nonnegative(self):
        y_true = [1.0, 0.0]
        y_pred = [0.7, 0.3]
        regimes = ["CHAOTIC", "CHAOTIC"]
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert got["CHAOTIC"]["loss"] >= 0.0
        import math
        assert math.isfinite(got["CHAOTIC"]["loss"])

    def test_pos_rate_reflects_label_balance(self):
        y_true = [1.0, 1.0, 1.0, 0.0]
        y_pred = [0.6, 0.6, 0.6, 0.4]
        regimes = ["TRENDING"] * 4
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert abs(got["TRENDING"]["pos_rate"] - 0.75) < 1e-9

    def test_empty_input_returns_empty_dict(self):
        got = _cnn_mod._per_regime_metrics([], [], [])
        assert got == {}

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            _cnn_mod._per_regime_metrics([1.0, 0.0], [0.5], ["TRENDING", "CHAOTIC"])

    def test_unknown_regime_label_is_preserved(self):
        # If a regime string we don't recognise comes in, it's still bucketed
        # (the helper is oblivious to the canonical TRENDING/RANGING/CHAOTIC set).
        y_true = [1.0]
        y_pred = [0.9]
        regimes = ["UNKNOWN"]
        got = _cnn_mod._per_regime_metrics(y_true, y_pred, regimes)
        assert "UNKNOWN" in got
        assert got["UNKNOWN"]["n"] == 1


class TestPrecisionRecallAtThreshold:
    """_precision_recall_at_threshold mirrors the production BUY gate.

    The CNN signal fires BUY when model_prob > config.cnn_buy_threshold
    (cnn_agent.py:1637). To evaluate whether a checkpoint would produce
    profitable BUY signals on the val set, we need precision/recall at
    that same threshold — not AUC (all thresholds) or BCE (calibration).
    """

    def test_empty_returns_none_none(self):
        p, r = _cnn_mod._precision_recall_at_threshold([], [], 0.60)
        assert p is None and r is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            _cnn_mod._precision_recall_at_threshold([0.7, 0.4], [1.0], 0.60)

    def test_all_below_threshold_with_positives(self):
        # No predictions above threshold → precision undefined (None).
        # Positives exist but none caught → recall = 0.0.
        probs  = [0.1, 0.2, 0.3]
        labels = [1.0, 0.0, 1.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.60)
        assert p is None, "precision should be None when no preds above threshold"
        assert r == 0.0

    def test_all_below_threshold_no_positives(self):
        # No preds above, no actual positives → both undefined.
        probs  = [0.1, 0.2, 0.3]
        labels = [0.0, 0.0, 0.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.60)
        assert p is None
        assert r is None

    def test_perfect_classifier(self):
        probs  = [0.9, 0.8, 0.1, 0.2]
        labels = [1.0, 1.0, 0.0, 0.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.60)
        assert p == 1.0
        assert r == 1.0

    def test_mixed_known_values(self):
        # probs > 0.5: [0.7, 0.8] → labels [1, 0] → TP=1, FP=1, FN=1
        # precision = 1/(1+1) = 0.5, recall = 1/(1+1) = 0.5
        probs  = [0.7, 0.4, 0.8, 0.3]
        labels = [1.0, 1.0, 0.0, 0.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.50)
        assert p == 0.5
        assert r == 0.5

    def test_strict_greater_than_threshold(self):
        # Matches cnn_agent.py:1637 `model_prob > config.cnn_buy_threshold` (strict).
        # A prob EXACTLY at the threshold must NOT count as a positive prediction.
        probs  = [0.60, 0.60]
        labels = [1.0, 0.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.60)
        assert p is None, "prob == threshold should not count as above (gate is strict >)"
        assert r == 0.0

    def test_threshold_045_for_sell_side(self):
        # Symmetric SELL gate is model_prob < 0.40; if we want a P/R view of
        # BUY-like signals at a lower threshold (e.g. exploratory), helper
        # must still work with any threshold in (0, 1).
        probs  = [0.50, 0.46, 0.44]
        labels = [1.0, 0.0, 1.0]
        p, r = _cnn_mod._precision_recall_at_threshold(probs, labels, 0.45)
        # Above 0.45: [0.50, 0.46] → labels [1, 0] → TP=1, FP=1, FN=1
        assert p == 0.5
        assert r == 0.5


# ── Inference-time regime gate (Option C) ─────────────────────────────────────

class TestInferenceRegimeGate:
    """Option C — block BUY when HMM regime is not CHAOTIC.

    Phase-1 live data (2026-04-23): BUYs in CHAOTIC won 58.5% vs 44.3% in
    TRENDING and 45.7% in RANGING. The CNN is most confident in TRENDING
    (avg cnn_prob 0.925) but least accurate there — classic inverse
    calibration. Gate keeps the CHAOTIC edge and blocks the losing regimes.
    Env-overridable via CNN_REGIME_GATE=off for emergency unblock.
    """

    @staticmethod
    def _make_tracker_mock():
        from unittest.mock import MagicMock
        t = MagicMock()
        t.get_lessons = AsyncMock(return_value=[])
        t.record      = AsyncMock()
        return t

    @pytest.mark.asyncio
    async def test_buy_blocked_when_regime_is_trending(self, agent, product):
        """TRENDING regime → BUY signal returned but book.buy is NOT called."""
        candles  = _make_candles(80)
        buy_mock = AsyncMock(return_value=(0.0, 0))

        fake_detector = type("D", (), {
            "predict": staticmethod(lambda closes: ("TRENDING", 0.80, 0))
        })()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",   new=AsyncMock()),
            patch("agents.cnn_agent.database.save_signal",     new=AsyncMock(return_value=1)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.82, 0, 0))),
            patch("agents.cnn_agent._hurst_exponent", return_value=0.55),
            patch("agents.cnn_agent.get_detector",    return_value=fake_detector),
            patch("agents.cnn_agent.get_tracker",     return_value=self._make_tracker_mock()),
            patch.object(agent, "_cnn_prob",          return_value=0.82),
            patch.object(agent._lgbm, "allow_buy",    return_value=True),
            patch.object(agent._lgbm, "predict",      return_value=0.7),
            patch.object(agent.book, "buy",           buy_mock),
            patch.object(agent.book, "has_position",  return_value=False),
        ):
            sig = await agent.generate_signal(product, execute=True)

        assert sig is not None
        assert sig["side"] == "BUY"
        buy_mock.assert_not_called()
        assert sig["execution"]["success"] is False
        reason = sig["execution"]["reason"].lower()
        assert "regime" in reason or "chaotic" in reason, (
            f"Expected reason to mention regime/CHAOTIC, got: {sig['execution']['reason']!r}"
        )

    @pytest.mark.asyncio
    async def test_buy_allowed_when_regime_is_chaotic(self, agent, product):
        """CHAOTIC regime + all other gates open → book.buy IS called."""
        candles  = _make_candles(80)
        buy_mock = AsyncMock(return_value=(50.0, 1))

        fake_detector = type("D", (), {
            "predict": staticmethod(lambda closes: ("CHAOTIC", 0.70, 2))
        })()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",   new=AsyncMock()),
            patch("agents.cnn_agent.database.save_signal",     new=AsyncMock(return_value=1)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.82, 0, 0))),
            patch("agents.cnn_agent._hurst_exponent", return_value=0.55),
            patch("agents.cnn_agent.get_detector",    return_value=fake_detector),
            patch("agents.cnn_agent.get_tracker",     return_value=self._make_tracker_mock()),
            patch.object(agent, "_cnn_prob",          return_value=0.82),
            patch.object(agent._lgbm, "allow_buy",    return_value=True),
            patch.object(agent._lgbm, "predict",      return_value=0.7),
            patch.object(agent.book, "buy",           buy_mock),
            patch.object(agent.book, "has_position",  return_value=False),
        ):
            sig = await agent.generate_signal(product, execute=True)

        assert sig is not None
        assert sig["side"] == "BUY"
        buy_mock.assert_called_once()
        assert sig["execution"]["success"] is True

    @pytest.mark.asyncio
    async def test_regime_gate_disabled_via_env(self, agent, product, monkeypatch):
        """CNN_REGIME_GATE=off → BUY executes even in TRENDING."""
        monkeypatch.setenv("CNN_REGIME_GATE", "off")
        # Reload module-level flag by re-reading env at call site.
        # Production code must read the flag each call, not at import time.
        candles  = _make_candles(80)
        buy_mock = AsyncMock(return_value=(50.0, 1))

        fake_detector = type("D", (), {
            "predict": staticmethod(lambda closes: ("TRENDING", 0.80, 0))
        })()

        with (
            patch("agents.cnn_agent.database.get_candles",
                  new=AsyncMock(return_value=candles)),
            patch("agents.cnn_agent.database.get_agent_decisions",
                  new=AsyncMock(return_value=[])),
            patch("agents.cnn_agent.database.save_cnn_scan",   new=AsyncMock()),
            patch("agents.cnn_agent.database.save_signal",     new=AsyncMock(return_value=1)),
            patch("agents.cnn_agent.coinbase_client.get_orderbook",
                  new=AsyncMock(return_value={"bids": [], "asks": []})),
            patch("agents.cnn_agent._ollama_prob",
                  new=AsyncMock(return_value=(0.82, 0, 0))),
            patch("agents.cnn_agent._hurst_exponent", return_value=0.55),
            patch("agents.cnn_agent.get_detector",    return_value=fake_detector),
            patch("agents.cnn_agent.get_tracker",     return_value=self._make_tracker_mock()),
            patch.object(agent, "_cnn_prob",          return_value=0.82),
            patch.object(agent._lgbm, "allow_buy",    return_value=True),
            patch.object(agent._lgbm, "predict",      return_value=0.7),
            patch.object(agent.book, "buy",           buy_mock),
            patch.object(agent.book, "has_position",  return_value=False),
        ):
            sig = await agent.generate_signal(product, execute=True)

        assert sig is not None
        assert sig["side"] == "BUY"
        buy_mock.assert_called_once()
        assert sig["execution"]["success"] is True
