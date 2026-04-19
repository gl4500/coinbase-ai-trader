"""
TDD tests for LightGBM entry filter.

Purpose: train on closed CNN trades (features from cnn_scans) and gate
future BUY decisions — only execute when predicted win probability >= threshold.

Features used: cnn_prob, rsi, adx, strength, macd, mfi, stoch_k,
               hour_of_day, day_of_week, usd_open
Target: pnl > 0  (binary win/loss)

Written before implementation.
"""
import os
import sys
import math
import tempfile
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_sample_rows(n: int, win: bool = True):
    """Generate n fake feature dicts matching real cnn_scans fields."""
    rows = []
    for i in range(n):
        rows.append({
            "cnn_prob":    0.80 if win else 0.45,
            "rsi":         45.0 if win else 72.0,
            "adx":         300.0,
            "strength":    0.60 if win else 0.20,
            "macd":        0.5  if win else -0.5,
            "mfi":         55.0,
            "stoch_k":     40.0,
            "hour_of_day": 10,
            "day_of_week": 1,
            "usd_open":    200.0,
            "pnl":         1.5  if win else -1.5,
        })
    return rows


# ── Model construction ────────────────────────────────────────────────────────

class TestLGBMFilterConstruction:

    def test_import(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert f is not None

    def test_not_ready_before_training(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert not f.is_ready()

    def test_ready_after_training(self):
        from data.lgbm_filter import LGBMFilter
        rows = _make_sample_rows(30, win=True) + _make_sample_rows(30, win=False)
        f = LGBMFilter()
        f.train(rows)
        assert f.is_ready()

    def test_train_returns_metrics(self):
        from data.lgbm_filter import LGBMFilter
        rows = _make_sample_rows(30, win=True) + _make_sample_rows(30, win=False)
        f = LGBMFilter()
        metrics = f.train(rows)
        assert "n_samples" in metrics
        assert "n_wins"    in metrics
        assert "auc"       in metrics
        assert metrics["n_samples"] == 60

    def test_train_requires_min_samples(self):
        """Fewer than MIN_SAMPLES rows → train() returns None, is_ready() stays False."""
        from data.lgbm_filter import LGBMFilter, MIN_SAMPLES
        rows = _make_sample_rows(MIN_SAMPLES - 1, win=True)
        f = LGBMFilter()
        result = f.train(rows)
        assert result is None
        assert not f.is_ready()


# ── Prediction ────────────────────────────────────────────────────────────────

class TestLGBMFilterPredict:

    def _trained(self):
        from data.lgbm_filter import LGBMFilter
        rows = _make_sample_rows(40, win=True) + _make_sample_rows(40, win=False)
        f = LGBMFilter()
        f.train(rows)
        return f

    def test_predict_returns_float(self):
        f = self._trained()
        prob = f.predict({
            "cnn_prob": 0.80, "rsi": 45.0, "adx": 300.0,
            "strength": 0.60, "macd": 0.5, "mfi": 55.0,
            "stoch_k": 40.0, "hour_of_day": 10, "day_of_week": 1, "usd_open": 200.0,
        })
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_predict_win_higher_than_loss_features(self):
        """Win-pattern features should score higher than loss-pattern features."""
        f = self._trained()
        win_prob  = f.predict({
            "cnn_prob": 0.85, "rsi": 40.0, "adx": 300.0,
            "strength": 0.70, "macd": 0.8, "mfi": 50.0,
            "stoch_k": 35.0, "hour_of_day": 10, "day_of_week": 1, "usd_open": 200.0,
        })
        loss_prob = f.predict({
            "cnn_prob": 0.42, "rsi": 75.0, "adx": 300.0,
            "strength": 0.15, "macd": -0.8, "mfi": 50.0,
            "stoch_k": 40.0, "hour_of_day": 10, "day_of_week": 1, "usd_open": 200.0,
        })
        assert win_prob > loss_prob

    def test_predict_returns_05_when_not_ready(self):
        """Untrained model returns neutral 0.5 so it never blocks buys."""
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        prob = f.predict({"cnn_prob": 0.8, "rsi": 50.0, "adx": 200.0,
                          "strength": 0.5, "macd": 0.0, "mfi": 50.0,
                          "stoch_k": 50.0, "hour_of_day": 10, "day_of_week": 1,
                          "usd_open": 200.0})
        assert prob == 0.5


# ── Gate threshold ────────────────────────────────────────────────────────────

class TestLGBMGateThreshold:

    def test_threshold_constant_exists(self):
        from data.lgbm_filter import LGBM_GATE_THRESHOLD
        assert 0.40 <= LGBM_GATE_THRESHOLD <= 0.70

    def test_allow_buy_returns_true_when_not_ready(self):
        """Untrained model must ALLOW all buys — no false blocks before training."""
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        assert f.allow_buy({"cnn_prob": 0.5, "rsi": 50.0, "adx": 200.0,
                             "strength": 0.3, "macd": 0.0, "mfi": 50.0,
                             "stoch_k": 50.0, "hour_of_day": 10, "day_of_week": 1,
                             "usd_open": 200.0}) is True

    def test_allow_buy_blocks_low_prob(self):
        from data.lgbm_filter import LGBMFilter, LGBM_GATE_THRESHOLD
        rows = _make_sample_rows(40, win=True) + _make_sample_rows(40, win=False)
        f = LGBMFilter()
        f.train(rows)
        # Force a predict that returns below threshold by patching
        f._model_predict = lambda X: [LGBM_GATE_THRESHOLD - 0.05]
        result = f.allow_buy({"cnn_prob": 0.4, "rsi": 75.0, "adx": 100.0,
                               "strength": 0.1, "macd": -1.0, "mfi": 80.0,
                               "stoch_k": 90.0, "hour_of_day": 2, "day_of_week": 0,
                               "usd_open": 50.0})
        assert result is False

    def test_allow_buy_passes_high_prob(self):
        from data.lgbm_filter import LGBMFilter, LGBM_GATE_THRESHOLD
        rows = _make_sample_rows(40, win=True) + _make_sample_rows(40, win=False)
        f = LGBMFilter()
        f.train(rows)
        f._model_predict = lambda X: [LGBM_GATE_THRESHOLD + 0.05]
        result = f.allow_buy({"cnn_prob": 0.85, "rsi": 40.0, "adx": 400.0,
                               "strength": 0.7, "macd": 1.0, "mfi": 45.0,
                               "stoch_k": 20.0, "hour_of_day": 10, "day_of_week": 2,
                               "usd_open": 300.0})
        assert result is True


# ── Save / load ───────────────────────────────────────────────────────────────

class TestLGBMFilterPersistence:

    def test_save_and_load_roundtrip(self):
        from data.lgbm_filter import LGBMFilter
        rows = _make_sample_rows(40, win=True) + _make_sample_rows(40, win=False)
        f = LGBMFilter()
        f.train(rows)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lgbm.pkl")
            f.save(path)
            assert os.path.exists(path)

            f2 = LGBMFilter()
            f2.load(path)
            assert f2.is_ready()

            # Predictions should be identical after reload
            feat = {"cnn_prob": 0.75, "rsi": 50.0, "adx": 250.0,
                    "strength": 0.5, "macd": 0.2, "mfi": 55.0,
                    "stoch_k": 45.0, "hour_of_day": 9, "day_of_week": 2,
                    "usd_open": 250.0}
            assert abs(f.predict(feat) - f2.predict(feat)) < 1e-6

    def test_load_missing_file_leaves_not_ready(self):
        from data.lgbm_filter import LGBMFilter
        f = LGBMFilter()
        f.load("/nonexistent/path/lgbm.pkl")
        assert not f.is_ready()
