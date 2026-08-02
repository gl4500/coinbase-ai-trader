"""Unit tests for backend/tools/v4_5_horizon_compare.py."""

from __future__ import annotations

import os
import sys

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestEvaluateOnHoldout3Class:
    def test_returns_metrics_dict(self):
        from tools.v4_5_horizon_compare import _evaluate_on_holdout_3class

        class _StubBooster:
            def predict(self, dmat):
                # 4 samples, 3 classes — peaked at correct labels
                return np.array(
                    [
                        [0.7, 0.2, 0.1],  # DOWN
                        [0.1, 0.2, 0.7],  # UP
                        [0.2, 0.7, 0.1],  # NEUTRAL
                        [0.1, 0.1, 0.8],  # UP
                    ]
                )

        X = np.zeros((4, 210), dtype=np.float64)
        y = np.array([0, 2, 1, 2], dtype=np.int8)
        names = [f"col{i}" for i in range(210)]
        out = _evaluate_on_holdout_3class(_StubBooster(), X, y, names)
        for k in (
            "auc_down",
            "auc_neutral",
            "auc_up",
            "auc_macro",
            "logloss",
            "n_samples",
            "pos_frac_down",
            "pos_frac_neutral",
            "pos_frac_up",
        ):
            assert k in out
        assert out["n_samples"] == 4
        assert out["pos_frac_down"] == 0.25
        assert out["pos_frac_up"] == 0.5

    def test_single_class_returns_nan_macro(self):
        from tools.v4_5_horizon_compare import _evaluate_on_holdout_3class

        class _StubBooster:
            def predict(self, dmat):
                return np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])

        X = np.zeros((2, 210), dtype=np.float64)
        y = np.array([0, 0], dtype=np.int8)  # only DOWN
        names = [f"col{i}" for i in range(210)]
        out = _evaluate_on_holdout_3class(_StubBooster(), X, y, names)
        # No UP or NEUTRAL samples -> their AUCs nan
        assert np.isnan(out["auc_up"])
        assert np.isnan(out["auc_neutral"])


class TestDecisionRules:
    def test_argmax_margin_buy(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules

        # Sample 0: p_up dominant with > 10pt margin over p_down -> BUY
        # Sample 1: p_down dominant with > 10pt margin -> SELL
        # Sample 2: p_neutral high, no margin -> HOLD
        probs = np.array(
            [
                [0.1, 0.2, 0.7],  # UP (margin 0.6)
                [0.7, 0.2, 0.1],  # DOWN (margin 0.6)
                [0.3, 0.4, 0.3],  # NEUTRAL
            ]
        )
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "argmax_margin" in out
        # Should fire 1 BUY (correct) and 1 SELL (correct)
        rule = out["argmax_margin"]
        for k in (
            "buy_precision",
            "buy_recall",
            "buy_f1",
            "sell_precision",
            "sell_recall",
            "sell_f1",
            "trade_rate",
            "hold_rate",
        ):
            assert k in rule

    def test_indep_thresholds(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules

        probs = np.array(
            [
                [0.10, 0.30, 0.60],
                [0.60, 0.30, 0.10],
                [0.30, 0.40, 0.30],
            ]
        )
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "indep_thresholds" in out

    def test_net_direction(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules

        probs = np.array(
            [
                [0.10, 0.20, 0.70],  # net +0.6 -> BUY
                [0.70, 0.20, 0.10],  # net -0.6 -> SELL
                [0.40, 0.30, 0.30],  # net -0.1 -> HOLD (below 0.20 threshold)
            ]
        )
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "net_direction" in out


class TestRenderHtmlReport:
    def test_writes_html_with_horizons_and_rules(self, tmp_path):
        from tools.v4_5_horizon_compare import _render_html_report

        metrics = {
            24: {
                "auc_macro": 0.55,
                "auc_down": 0.54,
                "auc_neutral": 0.50,
                "auc_up": 0.61,
                "logloss": 1.0,
                "n_samples": 1000,
                "pos_frac_down": 0.3,
                "pos_frac_neutral": 0.4,
                "pos_frac_up": 0.3,
            },
            72: {
                "auc_macro": 0.57,
                "auc_down": 0.56,
                "auc_neutral": 0.51,
                "auc_up": 0.64,
                "logloss": 0.98,
                "n_samples": 800,
                "pos_frac_down": 0.32,
                "pos_frac_neutral": 0.38,
                "pos_frac_up": 0.30,
            },
            168: {
                "auc_macro": 0.53,
                "auc_down": 0.52,
                "auc_neutral": 0.50,
                "auc_up": 0.57,
                "logloss": 1.05,
                "n_samples": 500,
                "pos_frac_down": 0.35,
                "pos_frac_neutral": 0.30,
                "pos_frac_up": 0.35,
            },
        }
        rules = {
            24: {
                "argmax_margin": {
                    "buy_f1": 0.30,
                    "sell_f1": 0.25,
                    "buy_precision": 0.5,
                    "buy_recall": 0.2,
                    "sell_precision": 0.4,
                    "sell_recall": 0.2,
                    "trade_rate": 0.4,
                    "hold_rate": 0.6,
                },
                "indep_thresholds": {
                    "buy_f1": 0.28,
                    "sell_f1": 0.22,
                    "buy_precision": 0.45,
                    "buy_recall": 0.20,
                    "sell_precision": 0.40,
                    "sell_recall": 0.15,
                    "trade_rate": 0.45,
                    "hold_rate": 0.55,
                },
                "net_direction": {
                    "buy_f1": 0.32,
                    "sell_f1": 0.28,
                    "buy_precision": 0.50,
                    "buy_recall": 0.24,
                    "sell_precision": 0.42,
                    "sell_recall": 0.21,
                    "trade_rate": 0.42,
                    "hold_rate": 0.58,
                },
            },
            72: {
                "argmax_margin": {
                    "buy_f1": 0.35,
                    "sell_f1": 0.30,
                    "buy_precision": 0.6,
                    "buy_recall": 0.25,
                    "sell_precision": 0.5,
                    "sell_recall": 0.21,
                    "trade_rate": 0.4,
                    "hold_rate": 0.6,
                },
                "indep_thresholds": {
                    "buy_f1": 0.33,
                    "sell_f1": 0.28,
                    "buy_precision": 0.55,
                    "buy_recall": 0.23,
                    "sell_precision": 0.45,
                    "sell_recall": 0.20,
                    "trade_rate": 0.43,
                    "hold_rate": 0.57,
                },
                "net_direction": {
                    "buy_f1": 0.36,
                    "sell_f1": 0.31,
                    "buy_precision": 0.58,
                    "buy_recall": 0.26,
                    "sell_precision": 0.48,
                    "sell_recall": 0.23,
                    "trade_rate": 0.41,
                    "hold_rate": 0.59,
                },
            },
            168: {
                "argmax_margin": {
                    "buy_f1": 0.20,
                    "sell_f1": 0.18,
                    "buy_precision": 0.4,
                    "buy_recall": 0.13,
                    "sell_precision": 0.35,
                    "sell_recall": 0.12,
                    "trade_rate": 0.3,
                    "hold_rate": 0.7,
                },
                "indep_thresholds": {
                    "buy_f1": 0.19,
                    "sell_f1": 0.17,
                    "buy_precision": 0.38,
                    "buy_recall": 0.13,
                    "sell_precision": 0.33,
                    "sell_recall": 0.11,
                    "trade_rate": 0.32,
                    "hold_rate": 0.68,
                },
                "net_direction": {
                    "buy_f1": 0.21,
                    "sell_f1": 0.19,
                    "buy_precision": 0.41,
                    "buy_recall": 0.14,
                    "sell_precision": 0.36,
                    "sell_recall": 0.13,
                    "trade_rate": 0.30,
                    "hold_rate": 0.70,
                },
            },
        }
        out_path = str(tmp_path / "report.html")
        _render_html_report(metrics, rules, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        # Sanity: each horizon + each rule appears somewhere
        assert "h24" in html or "24" in html
        assert "h72" in html or "72" in html
        assert "h168" in html or "168" in html
        assert "argmax_margin" in html
        assert "indep_thresholds" in html
        assert "net_direction" in html
