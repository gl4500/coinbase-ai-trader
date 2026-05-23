"""Unit tests for backend/tools/v4_horizon_compare.py."""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestEvaluateOnHoldout:
    def test_returns_metrics_dict(self):
        from tools.v4_horizon_compare import _evaluate_on_holdout

        class _StubBooster:
            def predict(self, dmat):
                # Return calibrated-ish probs aligned with labels
                return np.array([0.2, 0.7, 0.3, 0.8])

        class _IdentityCal:
            def transform(self, x):
                return x

        X = np.zeros((4, 150), dtype=np.float64)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        names = [f"col{i}" for i in range(150)]
        out = _evaluate_on_holdout(_StubBooster(), _IdentityCal(), X, y, names)
        assert "auc" in out
        assert "logloss" in out
        assert "pos_frac" in out
        assert "n_samples" in out
        assert out["n_samples"] == 4
        assert out["pos_frac"] == 0.5
        # AUC for perfectly aligned probs: 1.0
        assert out["auc"] == pytest.approx(1.0)

    def test_returns_nan_auc_for_single_class(self):
        from tools.v4_horizon_compare import _evaluate_on_holdout

        class _StubBooster:
            def predict(self, dmat):
                return np.array([0.5, 0.5, 0.5])

        class _IdentityCal:
            def transform(self, x):
                return x

        X = np.zeros((3, 150), dtype=np.float64)
        y = np.array([1, 1, 1], dtype=np.int8)
        names = [f"col{i}" for i in range(150)]
        out = _evaluate_on_holdout(_StubBooster(), _IdentityCal(), X, y, names)
        assert np.isnan(out["auc"])


class TestRenderHtmlReport:
    def test_writes_html_file(self, tmp_path):
        from tools.v4_horizon_compare import _render_html_report

        metrics = {
            4:   {"auc": 0.512, "logloss": 0.69, "pos_frac": 0.48, "n_samples": 1000},
            24:  {"auc": 0.534, "logloss": 0.68, "pos_frac": 0.45, "n_samples": 800},
            72:  {"auc": 0.561, "logloss": 0.67, "pos_frac": 0.40, "n_samples": 600},
            168: {"auc": 0.589, "logloss": 0.66, "pos_frac": 0.35, "n_samples": 400},
        }
        out_path = str(tmp_path / "report.html")
        _render_html_report(metrics, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        # Sanity checks on content
        assert "0.589" in html or "0.59" in html   # h168 AUC visible
        assert "h168" in html or "168" in html     # horizon visible
        assert "auc" in html.lower()
