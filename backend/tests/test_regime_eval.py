"""TDD tests for regime-stratified walk-forward eval (#165).

Bins samples by realised-volatility terciles and reports per-regime AUC
inside the existing purged walk-forward folds. Pure-numpy first pass.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestClassifyRegimes:

    def test_terciles_split_evenly(self):
        from tools.regime_eval import _classify_regimes
        vols = np.arange(300, dtype=np.float64)
        labels = _classify_regimes(vols)
        assert (labels == "low").sum() == 100
        assert (labels == "mid").sum() == 100
        assert (labels == "high").sum() == 100

    def test_low_assigned_to_smallest(self):
        from tools.regime_eval import _classify_regimes
        vols = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        labels = _classify_regimes(vols)
        assert labels[0] == "low"
        assert labels[-1] == "high"

    def test_constant_vol_does_not_crash(self):
        from tools.regime_eval import _classify_regimes
        vols = np.full(50, 0.5, dtype=np.float64)
        labels = _classify_regimes(vols)
        assert len(labels) == 50
        assert set(labels).issubset({"low", "mid", "high"})


class TestPerRegimeMetrics:

    def test_returns_one_row_per_regime(self):
        from tools.regime_eval import _per_regime_metrics
        rng = np.random.default_rng(42)
        n = 300
        y_true = rng.integers(0, 2, n)
        y_score = rng.uniform(0, 1, n)
        regimes = np.array(["low"] * 100 + ["mid"] * 100 + ["high"] * 100)
        out = _per_regime_metrics(y_true, y_score, regimes)
        assert set(out.keys()) == {"low", "mid", "high"}
        for r in ("low", "mid", "high"):
            assert "n" in out[r]
            assert "base_rate" in out[r]
            assert "auc" in out[r]
            assert out[r]["n"] == 100

    def test_perfect_separation_gives_auc_one(self):
        from tools.regime_eval import _per_regime_metrics
        n = 100
        y_true = np.array([0] * 50 + [1] * 50)
        y_score = np.linspace(0.0, 1.0, n)
        regimes = np.array(["low"] * n)
        out = _per_regime_metrics(y_true, y_score, regimes)
        assert out["low"]["auc"] == pytest.approx(1.0, abs=1e-6)
        assert out["low"]["base_rate"] == pytest.approx(0.5)

    def test_single_class_regime_returns_none_auc(self):
        from tools.regime_eval import _per_regime_metrics
        # All-zero labels in the "low" bucket → AUC undefined
        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1])
        y_score = np.linspace(0, 1, 8)
        regimes = np.array(["low"] * 4 + ["mid"] * 4)
        out = _per_regime_metrics(y_true, y_score, regimes)
        assert out["low"]["auc"] is None
        # mid has both classes
        assert out["mid"]["auc"] is not None
