"""TDD tests for tools/rsi_rank_probe.py — cross-sectional RSI rank signal.

Helper computes, for each (timestamp, product) pair, the percentile rank of
that product's RSI vs all other products at the same timestamp. The probe
swaps that signal in for one of the marginal channels and runs the standard
walk-forward Δ-AUC harness.

Decision rule (#162): Δ ≥ +0.01 → integrate as a real channel.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestCrossSectionalRank:

    def test_single_product_returns_neutral(self):
        from tools.rsi_rank_probe import _cross_sectional_rank
        out = _cross_sectional_rank({"BTC-USD": 0.7})
        assert out == {"BTC-USD": 0.5}

    def test_three_products_distinct(self):
        from tools.rsi_rank_probe import _cross_sectional_rank
        out = _cross_sectional_rank({"A": 0.2, "B": 0.5, "C": 0.8})
        assert out["A"] == pytest.approx(0.0)
        assert out["B"] == pytest.approx(0.5)
        assert out["C"] == pytest.approx(1.0)

    def test_ties_get_average_rank(self):
        from tools.rsi_rank_probe import _cross_sectional_rank
        out = _cross_sectional_rank({"A": 0.5, "B": 0.5, "C": 0.9})
        # A and B tie at rank 0.5 (avg of 0 and 1), normalized to 0.25 (0.5/2)
        assert out["A"] == pytest.approx(out["B"])
        assert out["C"] > out["A"]

    def test_empty_dict_returns_empty(self):
        from tools.rsi_rank_probe import _cross_sectional_rank
        assert _cross_sectional_rank({}) == {}

    def test_nan_values_skipped(self):
        from tools.rsi_rank_probe import _cross_sectional_rank
        out = _cross_sectional_rank({"A": 0.2, "B": float("nan"), "C": 0.8})
        # B with NaN should not participate; A and C ranked among themselves
        assert "B" not in out or np.isnan(out["B"])
        assert out["A"] == pytest.approx(0.0)
        assert out["C"] == pytest.approx(1.0)


class TestBuildRankSignal:

    def test_shape_matches_n_t(self):
        from tools.rsi_rank_probe import build_rank_signal
        # Synthetic: 2 products, 3 samples each, 60-bar windows, 1h spacing
        rsi_by_pid = {
            "A": (np.linspace(0.1, 0.9, 3 + 59).astype(np.float32),  # full RSI series
                  (np.arange(3 + 59) * 3600).astype(np.int64)),       # bar timestamps
            "B": (np.linspace(0.9, 0.1, 3 + 59).astype(np.float32),
                  (np.arange(3 + 59) * 3600).astype(np.int64)),
        }
        # Sample end-times for product A: bars 59, 60, 61
        sample_end_ts = (np.array([59, 60, 61]) * 3600).astype(np.int64)
        sig = build_rank_signal("A", sample_end_ts, rsi_by_pid)
        assert sig.shape == (3, 60)

    def test_rank_against_self_is_neutral(self):
        from tools.rsi_rank_probe import build_rank_signal
        # Only one product → rank always 0.5
        rsi = np.linspace(0.1, 0.9, 60).astype(np.float32)
        ts = (np.arange(60) * 3600).astype(np.int64)
        rsi_by_pid = {"A": (rsi, ts)}
        sample_end_ts = np.array([ts[-1]], dtype=np.int64)
        sig = build_rank_signal("A", sample_end_ts, rsi_by_pid)
        assert sig.shape == (1, 60)
        assert np.allclose(sig, 0.5)

    def test_higher_rsi_gets_higher_rank(self):
        from tools.rsi_rank_probe import build_rank_signal
        # A always has higher RSI than B → A's rank should be 1.0
        n_bars = 70
        ts = (np.arange(n_bars) * 3600).astype(np.int64)
        rsi_by_pid = {
            "A": (np.full(n_bars, 0.9, dtype=np.float32), ts),
            "B": (np.full(n_bars, 0.1, dtype=np.float32), ts),
        }
        sample_end_ts = np.array([ts[-1]], dtype=np.int64)
        sig_a = build_rank_signal("A", sample_end_ts, rsi_by_pid)
        sig_b = build_rank_signal("B", sample_end_ts, rsi_by_pid)
        assert np.allclose(sig_a, 1.0)
        assert np.allclose(sig_b, 0.0)

    def test_missing_timestamp_gives_neutral(self):
        from tools.rsi_rank_probe import build_rank_signal
        # Product A has data only for ts=0..59; sample at ts=200 (no data)
        ts = (np.arange(60) * 3600).astype(np.int64)
        rsi_by_pid = {
            "A": (np.full(60, 0.5, dtype=np.float32), ts),
            "B": (np.full(60, 0.7, dtype=np.float32), ts),
        }
        sample_end_ts = np.array([200 * 3600], dtype=np.int64)
        sig = build_rank_signal("A", sample_end_ts, rsi_by_pid)
        # No coverage → all 0.5 neutral
        assert np.allclose(sig, 0.5)
