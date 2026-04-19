"""
Regression tests for BSM integration features added 2026-04-17:
  - _realized_vol        (signal_generator)
  - _shannon_entropy     (signal_generator)
  - deribit_iv           (services/deribit_iv)
  - binance_sentiment    (services/binance_sentiment)
  - hmm_regime           (services/hmm_regime)
  - FeatureBuilder       (cnn_agent — 27 channels)
  - LGBMFilter           (data/lgbm_filter — unseen label fix)
"""
import math
import os
import sys
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _prices(n=100, start=100.0, drift=0.001, noise=0.01):
    """Generate synthetic price series with mild upward drift."""
    import random
    random.seed(42)
    p = [start]
    for _ in range(n - 1):
        p.append(p[-1] * (1 + drift + random.gauss(0, noise)))
    return p


def _candle(close=100.0, open_=None, high=None, low=None, vol=1000.0, ts="2025-01-01T10:00:00Z"):
    return {
        "open": open_ or close * 0.999,
        "high": high or close * 1.002,
        "low":  low  or close * 0.997,
        "close": close,
        "volume": vol,
        "start": ts,
    }


# ── _realized_vol ─────────────────────────────────────────────────────────────

class TestRealizedVol:
    def setup_method(self):
        from agents.signal_generator import _realized_vol
        self.rv = _realized_vol

    def test_returns_zero_on_insufficient_data(self):
        assert self.rv([100, 101], window=20) == 0.0

    def test_returns_positive_for_valid_series(self):
        prices = _prices(60)
        result = self.rv(prices, window=20)
        assert result > 0.0

    def test_crypto_annualisation_higher_than_equity(self):
        prices = _prices(60)
        rv_crypto = self.rv(prices, window=20, annualize_days=365)
        rv_equity = self.rv(prices, window=20, annualize_days=252)
        assert rv_crypto > rv_equity

    def test_flat_prices_near_zero_vol(self):
        flat = [100.0] * 30
        result = self.rv(flat, window=20)
        assert result == 0.0 or result < 1e-6

    def test_window_60_uses_more_bars(self):
        prices = _prices(100)
        rv20 = self.rv(prices, window=20)
        rv60 = self.rv(prices, window=60)
        # Both should be positive; they won't necessarily be equal
        assert rv20 > 0 and rv60 > 0

    def test_high_volatility_series(self):
        import random
        random.seed(1)
        volatile = [100 * (1 + random.gauss(0, 0.05)) for _ in range(60)]
        result = self.rv(volatile, window=20)
        assert result > 0.5   # should be clearly > 50% annualised


# ── _shannon_entropy ──────────────────────────────────────────────────────────

class TestShannonEntropy:
    def setup_method(self):
        from agents.signal_generator import _shannon_entropy
        self.ent = _shannon_entropy

    def test_returns_neutral_on_insufficient_data(self):
        assert self.ent([100, 101, 102], window=20) == 0.5

    def test_flat_series_near_zero_entropy(self):
        flat = [100.0] * 30
        result = self.ent(flat, window=20)
        assert result == 0.0

    def test_result_bounded_0_to_1(self):
        prices = _prices(60)
        result = self.ent(prices, window=20)
        assert 0.0 <= result <= 1.0

    def test_random_walk_high_entropy(self):
        import random
        random.seed(7)
        random_walk = [100 + sum(random.gauss(0, 1) for _ in range(i)) for i in range(60)]
        result = self.ent(random_walk, window=30)
        assert result > 0.5   # random walk should have high entropy

    def test_flat_has_lower_entropy_than_noisy(self):
        # flat prices → zero returns → entropy near 0
        flat = [100.0] * 60
        # noisy prices → spread returns → entropy > 0
        import random; random.seed(9)
        noisy = [100 + random.gauss(0, 2) for _ in range(60)]
        e_flat  = self.ent(flat,  window=20)
        e_noisy = self.ent(noisy, window=20)
        assert e_flat < e_noisy


# ── deribit_iv service ────────────────────────────────────────────────────────

class TestDeribitIV:
    def test_compute_iv_rv_spreads_positive(self):
        from services.deribit_iv import compute_iv_rv_spreads
        result = compute_iv_rv_spreads(iv=0.80, rv20=0.50, rv60=0.45)
        assert result["iv_rv20_spread"] == pytest.approx(0.30, abs=0.01)
        assert result["iv_rv60_spread"] == pytest.approx(0.35, abs=0.01)

    def test_compute_iv_rv_spreads_clipped_at_1(self):
        from services.deribit_iv import compute_iv_rv_spreads
        result = compute_iv_rv_spreads(iv=2.0, rv20=0.5, rv60=0.5)
        assert result["iv_rv20_spread"] == 1.0
        assert result["iv_rv60_spread"] == 1.0

    def test_compute_iv_rv_spreads_clipped_at_minus1(self):
        from services.deribit_iv import compute_iv_rv_spreads
        result = compute_iv_rv_spreads(iv=0.1, rv20=2.0, rv60=2.0)
        assert result["iv_rv20_spread"] == -1.0
        assert result["iv_rv60_spread"] == -1.0

    def test_none_product_returns_none(self):
        """Products without Deribit options return None without making network calls."""
        import asyncio
        from services.deribit_iv import get_iv
        result = asyncio.get_event_loop().run_until_complete(
            get_iv("DOGE-USD", 0.15)
        )
        assert result is None


# ── binance_sentiment service ─────────────────────────────────────────────────

class TestBinanceSentiment:
    def test_unknown_product_returns_none(self):
        import asyncio
        from services.binance_sentiment import get_ls_sentiment
        result = asyncio.get_event_loop().run_until_complete(
            get_ls_sentiment("FARTCOIN-USD")
        )
        assert result is None

    def test_known_products_are_mapped(self):
        from services.binance_sentiment import _PRODUCT_TO_BN
        for pid in ["BTC-USD", "ETH-USD", "SOL-USD"]:
            assert pid in _PRODUCT_TO_BN


# ── HMM regime detector ───────────────────────────────────────────────────────

class TestHMMRegime:
    def test_unknown_before_fit(self):
        from services.hmm_regime import HMMRegimeDetector
        d = HMMRegimeDetector.__new__(HMMRegimeDetector)
        d._model = None
        d._state_labels = {}
        regime, conf, state = d.predict(_prices(100))
        assert regime == "UNKNOWN"
        assert state == -1

    def test_fit_returns_true_on_sufficient_data(self):
        from services.hmm_regime import HMMRegimeDetector
        d = HMMRegimeDetector.__new__(HMMRegimeDetector)
        d._model = None
        d._state_labels = {}
        result = d.fit(_prices(200))
        assert result is True
        assert d.is_ready()

    def test_predict_after_fit_returns_valid_regime(self):
        from services.hmm_regime import HMMRegimeDetector, REGIME_NAMES
        d = HMMRegimeDetector.__new__(HMMRegimeDetector)
        d._model = None
        d._state_labels = {}
        d.fit(_prices(200))
        regime, conf, state = d.predict(_prices(100))
        assert regime in REGIME_NAMES.values()
        assert 0.0 <= conf <= 1.0

    def test_regime_blend_trending(self):
        from services.hmm_regime import regime_blend
        cnn_w, llm_w = regime_blend("TRENDING", 1.0)
        assert cnn_w == pytest.approx(0.75)
        assert llm_w == pytest.approx(0.25)

    def test_regime_blend_chaotic(self):
        from services.hmm_regime import regime_blend
        cnn_w, llm_w = regime_blend("CHAOTIC", 1.0)
        assert cnn_w == pytest.approx(0.40)
        assert llm_w == pytest.approx(0.60)

    def test_regime_blend_low_confidence_toward_50_50(self):
        from services.hmm_regime import regime_blend
        cnn_w, llm_w = regime_blend("TRENDING", 0.0)
        assert cnn_w == pytest.approx(0.5, abs=0.01)
        assert llm_w == pytest.approx(0.5, abs=0.01)

    def test_weights_always_sum_to_1(self):
        from services.hmm_regime import regime_blend
        for regime in ["TRENDING", "RANGING", "CHAOTIC", "UNKNOWN"]:
            for conf in [0.0, 0.5, 1.0]:
                cnn_w, llm_w = regime_blend(regime, conf)
                assert cnn_w + llm_w == pytest.approx(1.0, abs=0.01)


# ── FeatureBuilder — 27 channels ──────────────────────────────────────────────

class TestFeatureBuilder27Channels:
    def setup_method(self):
        from agents.cnn_agent import FeatureBuilder, N_CHANNELS, SEQ_LEN
        self.fb = FeatureBuilder()
        self.N  = N_CHANNELS
        self.T  = SEQ_LEN

    def _dummy_candles(self, n=70, price=100.0):
        return [_candle(close=price + i * 0.1) for i in range(n)]

    def test_channel_count_is_27(self):
        assert self.N == 27

    def test_build_returns_27_channels(self):
        ch = self.fb.build(self._dummy_candles(), {})
        assert len(ch) == 27

    def test_each_channel_has_seq_len_timesteps(self):
        ch = self.fb.build(self._dummy_candles(), {})
        for i, c in enumerate(ch):
            assert len(c) == self.T, f"Ch {i} has {len(c)} timesteps, expected {self.T}"

    def test_iv_rv_channels_default_to_zero_for_non_btc(self):
        ch = self.fb.build(self._dummy_candles(), {})
        # Ch 24 and 25 should be 0.0 when no IV/RV data provided
        assert all(v == 0.0 for v in ch[24])
        assert all(v == 0.0 for v in ch[25])

    def test_iv_rv_channels_populated_when_provided(self):
        ch = self.fb.build(
            self._dummy_candles(), {},
            iv_rv20_spread=0.25, iv_rv60_spread=0.15
        )
        assert all(v == pytest.approx(0.25) for v in ch[24])
        assert all(v == pytest.approx(0.15) for v in ch[25])

    def test_ls_sentiment_channel_default_zero(self):
        ch = self.fb.build(self._dummy_candles(), {})
        assert all(v == 0.0 for v in ch[26])

    def test_ls_sentiment_channel_populated(self):
        ch = self.fb.build(self._dummy_candles(), {}, ls_sentiment=-0.3)
        assert all(v == pytest.approx(-0.3) for v in ch[26])

    def test_empty_candles_returns_27_zero_channels(self):
        ch = self.fb.build([], {})
        assert len(ch) == 27
        assert all(len(c) == self.T for c in ch)


# ── LGBMFilter — unseen label fix ─────────────────────────────────────────────

class TestLGBMFilterUnseenLabel:
    def _rows(self, n_wins, n_losses):
        rows = []
        for i in range(n_wins):
            rows.append({k: 0.5 for k in
                ["cnn_prob","rsi","adx","strength","macd",
                 "mfi","stoch_k","hour_of_day","day_of_week","usd_open",
                 "pnl"]})
            rows[-1]["pnl"] = 1.0   # win
        for i in range(n_losses):
            row = {k: 0.5 for k in
                ["cnn_prob","rsi","adx","strength","macd",
                 "mfi","stoch_k","hour_of_day","day_of_week","usd_open",
                 "pnl"]}
            row["pnl"] = -1.0   # loss
            rows.append(row)
        return rows

    def test_single_class_train_returns_none(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        result = f.train(self._rows(n_wins=60, n_losses=0))
        assert result is None

    def test_both_classes_in_val_does_not_crash(self):
        """Regression: val split with unseen label must not throw."""
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        # 50 wins + 50 losses — 80% train may have imbalance, 20% val may differ
        rows = self._rows(n_wins=50, n_losses=50)
        try:
            result = f.train(rows)
            # Either trains successfully or returns None — must not raise
        except Exception as e:
            pytest.fail(f"LGBMFilter.train raised unexpectedly: {e}")

    def test_not_ready_before_training(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert not f.is_ready()

    def test_allow_buy_true_when_not_ready(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert f.allow_buy({"cnn_prob": 0.5}) is True

    def test_predict_returns_05_when_not_ready(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert f.predict({"cnn_prob": 0.5}) == 0.5
