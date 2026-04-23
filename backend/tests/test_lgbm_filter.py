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
        # Lower bound 0.10 covers the Session 34 env-override design intent:
        # operators can set LGBM_GATE_THRESHOLD as low as 0.10 to unblock the CNN
        # when binary-label training collapses probabilities (see Task #39/43).
        from data.lgbm_filter import LGBM_GATE_THRESHOLD
        assert 0.10 <= LGBM_GATE_THRESHOLD <= 0.70

    def test_threshold_default_when_env_unset(self, monkeypatch):
        """Default stays 0.52 when env var is not set."""
        import importlib, data.lgbm_filter as lf
        monkeypatch.delenv("LGBM_GATE_THRESHOLD", raising=False)
        importlib.reload(lf)
        assert lf.LGBM_GATE_THRESHOLD == 0.52

    def test_threshold_reads_env_override(self, monkeypatch):
        """Setting LGBM_GATE_THRESHOLD=0.35 in env overrides the 0.52 default."""
        import importlib, data.lgbm_filter as lf
        monkeypatch.setenv("LGBM_GATE_THRESHOLD", "0.35")
        importlib.reload(lf)
        assert lf.LGBM_GATE_THRESHOLD == 0.35
        # restore default for other tests
        monkeypatch.delenv("LGBM_GATE_THRESHOLD", raising=False)
        importlib.reload(lf)

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


# ── Pnl-weighted labels (Task #43) ────────────────────────────────────────────

class TestLGBMFilterPnlWeighting:
    """
    Root cause (Session 2026-04-23): binary pnl>0 labels trained on 23% win-rate
    history score every live CNN BUY at p(win)=0.15–0.17 — 100% gate block.
    Fix: weight training samples by |pnl| so large winners/losers dominate.
    """

    def test_sample_weights_equal_abs_pnl(self):
        """_sample_weights returns |pnl| per row — large-magnitude trades weigh more."""
        from data.lgbm_filter import LGBMFilter
        rows = [
            {"pnl":  0.10, "cnn_prob": 0.8, "rsi": 50, "adx": 200, "strength": 0.5,
             "macd": 0, "mfi": 50, "stoch_k": 50, "hour_of_day": 10, "day_of_week": 1,
             "usd_open": 100},
            {"pnl": -5.00, "cnn_prob": 0.8, "rsi": 50, "adx": 200, "strength": 0.5,
             "macd": 0, "mfi": 50, "stoch_k": 50, "hour_of_day": 10, "day_of_week": 1,
             "usd_open": 100},
            {"pnl":  2.50, "cnn_prob": 0.8, "rsi": 50, "adx": 200, "strength": 0.5,
             "macd": 0, "mfi": 50, "stoch_k": 50, "hour_of_day": 10, "day_of_week": 1,
             "usd_open": 100},
        ]
        f = LGBMFilter()
        w = f._sample_weights(rows)
        assert w[0] == pytest.approx(0.10)
        assert w[1] == pytest.approx(5.00)
        assert w[2] == pytest.approx(2.50)

    def test_sample_weights_floor_on_zero_pnl(self):
        """Zero-pnl rows must get a small positive weight, never 0 (LightGBM drops 0-weight rows)."""
        from data.lgbm_filter import LGBMFilter
        rows = [{"pnl": 0.0, "cnn_prob": 0.8, "rsi": 50, "adx": 200, "strength": 0.5,
                 "macd": 0, "mfi": 50, "stoch_k": 50, "hour_of_day": 10, "day_of_week": 1,
                 "usd_open": 100}]
        f = LGBMFilter()
        w = f._sample_weights(rows)
        assert w[0] > 0.0

    def test_train_passes_sample_weight_to_fit(self, monkeypatch):
        """train() must forward sample_weight=|pnl| to LGBMClassifier.fit."""
        import lightgbm as lgb
        from data.lgbm_filter import LGBMFilter
        captured = {}
        orig_fit = lgb.LGBMClassifier.fit

        def spy_fit(self, X, y, *args, **kwargs):
            captured["sample_weight"] = kwargs.get("sample_weight")
            return orig_fit(self, X, y, *args, **kwargs)

        monkeypatch.setattr(lgb.LGBMClassifier, "fit", spy_fit)

        rows = _make_sample_rows(30, win=True) + _make_sample_rows(30, win=False)
        # _make_sample_rows sets pnl=±1.5 → |pnl|=1.5
        f = LGBMFilter()
        f.train(rows)

        import numpy as np
        sw = captured.get("sample_weight")
        assert sw is not None, "sample_weight must be passed to fit() after Task #43"
        # Training split is 80% — weights slice should match
        split = max(1, int(60 * 0.8))
        assert len(sw) == split
        assert np.all(np.asarray(sw) == pytest.approx(1.5))
