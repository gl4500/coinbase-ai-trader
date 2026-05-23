"""TDD tests for per-channel distribution drift monitor (#170).

PSI-based proxy. Splits each channel's terminal-value series
chronologically into halves and reports Population Stability Index.
Thresholds: <0.1 stable, 0.1-0.25 minor drift, >=0.25 significant.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestBinEdges:

    def test_returns_n_plus_1_edges_for_n_bins(self):
        from tools.drift_monitor import _bin_edges
        values = np.linspace(0, 1, 1000)
        edges = _bin_edges(values, n_bins=10)
        assert len(edges) == 11
        assert edges[0] <= 0.0
        assert edges[-1] >= 1.0

    def test_constant_input_does_not_crash(self):
        from tools.drift_monitor import _bin_edges
        values = np.full(50, 0.5, dtype=np.float64)
        edges = _bin_edges(values, n_bins=10)
        assert len(edges) == 11
        # Edges may collapse and outer edges may be -inf/+inf — that's allowed.
        assert not np.any(np.isnan(edges))


class TestPSI:

    def test_identical_distributions_return_zero(self):
        from tools.drift_monitor import _psi
        p = np.array([0.1, 0.2, 0.3, 0.2, 0.2])
        psi = _psi(p, p)
        assert psi == pytest.approx(0.0, abs=1e-9)

    def test_significant_shift_exceeds_threshold(self):
        from tools.drift_monitor import _psi
        # Concentrated in first bin vs last bin → big PSI
        p = np.array([0.9, 0.05, 0.025, 0.015, 0.01])
        q = np.array([0.01, 0.015, 0.025, 0.05, 0.9])
        psi = _psi(p, q)
        assert psi > 0.25

    def test_handles_zero_bin_via_epsilon(self):
        from tools.drift_monitor import _psi
        # Empty bin in q would explode log(0); epsilon must regularise
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.5, 0.5, 0.0])
        psi = _psi(p, q)
        assert np.isfinite(psi)


class TestPerChannelDrift:

    def test_stable_channel_low_psi(self):
        from tools.drift_monitor import _channel_drift
        rng = np.random.default_rng(42)
        # Same distribution throughout → PSI near 0
        series = rng.normal(0, 1, 4000)
        out = _channel_drift(series)
        assert out["psi"] < 0.1
        assert out["flag"] == "stable"

    def test_distribution_shift_flagged(self):
        from tools.drift_monitor import _channel_drift
        rng = np.random.default_rng(7)
        # First half N(0,1), second half N(3,1) — clear shift
        s1 = rng.normal(0, 1, 2000)
        s2 = rng.normal(3, 1, 2000)
        series = np.concatenate([s1, s2])
        out = _channel_drift(series)
        assert out["psi"] >= 0.25
        assert out["flag"] == "significant"

    def test_minor_shift_flagged(self):
        from tools.drift_monitor import _channel_drift
        rng = np.random.default_rng(11)
        # First half N(0,1), second half N(0.5,1) — minor shift
        s1 = rng.normal(0, 1, 2000)
        s2 = rng.normal(0.5, 1, 2000)
        series = np.concatenate([s1, s2])
        out = _channel_drift(series)
        assert 0.05 < out["psi"] < 0.5
        assert out["flag"] in ("minor", "significant")

    def test_short_series_does_not_crash(self):
        from tools.drift_monitor import _channel_drift
        series = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = _channel_drift(series)
        assert "psi" in out
        assert "flag" in out
