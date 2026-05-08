"""TDD tests for the heuristic stationarity audit (#164).

Pure-numpy first pass — formal ADF deferred (would require statsmodels).
The audit reports drift_z, var_ratio, lag1 autocorr, and a flag per
channel. These tests cover the `_stationarity_metrics` helper on
canonical synthetic series.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestStationarityMetrics:

    def test_constant_series_is_stationary(self):
        from tools.stationarity_audit import _stationarity_metrics
        series = np.full(1000, 0.5, dtype=np.float64)
        m = _stationarity_metrics(series)
        assert m["drift_z"] == pytest.approx(0.0, abs=1e-9)
        assert m["var_ratio"] == pytest.approx(1.0, abs=1e-9)
        assert m["flag"] == "stationary"

    def test_white_noise_is_stationary(self):
        from tools.stationarity_audit import _stationarity_metrics
        rng = np.random.default_rng(42)
        series = rng.normal(0, 1, 5000)
        m = _stationarity_metrics(series)
        assert m["drift_z"] < 0.5
        assert 0.5 < m["var_ratio"] < 1.5
        assert abs(m["lag1"]) < 0.5
        assert m["flag"] == "stationary"

    def test_random_walk_is_suspect(self):
        from tools.stationarity_audit import _stationarity_metrics
        rng = np.random.default_rng(0)
        # Random walk: cumulative sum of N(0,1). Lag-1 autocorr → 1.
        series = np.cumsum(rng.normal(0, 1, 5000))
        m = _stationarity_metrics(series)
        assert m["flag"] == "suspect"
        assert m["lag1"] > 0.95

    def test_trending_series_is_suspect(self):
        from tools.stationarity_audit import _stationarity_metrics
        # Linear trend: drift between halves dominates
        series = np.linspace(0.0, 100.0, 5000)
        m = _stationarity_metrics(series)
        assert m["flag"] == "suspect"
        assert m["drift_z"] > 0.5

    def test_variance_blowout_is_suspect(self):
        from tools.stationarity_audit import _stationarity_metrics
        rng = np.random.default_rng(7)
        h = 2500
        # First half N(0,1); second half N(0,5) — same mean, 5x variance
        s1 = rng.normal(0, 1, h)
        s2 = rng.normal(0, 5, h)
        series = np.concatenate([s1, s2])
        m = _stationarity_metrics(series)
        assert m["flag"] == "suspect"
        assert m["var_ratio"] > 2.0

    def test_short_series_does_not_crash(self):
        from tools.stationarity_audit import _stationarity_metrics
        series = np.array([1.0, 2.0], dtype=np.float64)
        m = _stationarity_metrics(series)
        # Pathological — just don't crash; flag is allowed to be either.
        assert "flag" in m
        assert "lag1" in m


class TestAuditChannels:

    def test_returns_one_row_per_channel(self):
        import torch
        from tools.stationarity_audit import _audit_channels
        # 5 samples × 3 channels × 60 timesteps
        n_samples, n_channels, seq_len = 5, 3, 60
        prods = {
            "P": {
                "first_ts": 0,
                "last_ts": (n_samples - 1) * 3600,
                "last_n": n_samples,
                "X": [torch.zeros(n_channels, seq_len) for _ in range(n_samples)],
                "y": np.zeros(n_samples, dtype=np.float32),
                "indices": np.arange(n_samples, dtype=np.int64),
            }
        }
        rows = _audit_channels(prods, n_pids=1, snapshot_ts=None)
        assert len(rows) == n_channels
        assert {r["channel"] for r in rows} == {0, 1, 2}
        # All-zero terminals → drift_z=0, var_ratio=1, flag=stationary
        for r in rows:
            assert r["flag"] == "stationary"
