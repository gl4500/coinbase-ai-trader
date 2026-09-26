"""TDD tests for per-channel drift diagnostic (#208/#209).

Follow-up to #170. The drift monitor flagged Ch 5 with PSI=0.198 (minor).
This module decomposes that scalar into actionable pieces for any channel:

  - decompose_psi: per-bin contribution breakdown (which bins moved?)
  - summary_stats: mean/var/skew/min/max for a numeric vector
  - per_product_drift: PSI per pid, sorted desc (is drift concentrated?)
  - bin_count_sensitivity: PSI as a function of n_bins (normalization probe)

No I/O, pure numpy. CLI lives in tools/channel_drift_diagnose.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestDecomposePSI:
    def test_total_matches_sum_of_per_bin_contributions(self):
        from tools.channel_drift_diagnose import decompose_psi

        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0.2, 1.1, 500)
        out = decompose_psi(a, b, n_bins=10)
        contribs = [r["contribution"] for r in out["per_bin"]]
        assert pytest.approx(out["total_psi"], abs=1e-9) == sum(contribs)
        assert len(out["per_bin"]) == 10

    def test_identical_halves_give_near_zero_psi(self):
        from tools.channel_drift_diagnose import decompose_psi

        rng = np.random.default_rng(1)
        a = rng.normal(0, 1, 1000)
        out = decompose_psi(a, a.copy(), n_bins=10)
        assert abs(out["total_psi"]) < 1e-6

    def test_per_bin_records_have_required_keys(self):
        from tools.channel_drift_diagnose import decompose_psi

        a = np.linspace(-1, 1, 200)
        b = np.linspace(-0.5, 1.5, 200)
        out = decompose_psi(a, b, n_bins=5)
        for r in out["per_bin"]:
            for key in ("bin_idx", "lo", "hi", "p", "q", "contribution"):
                assert key in r, r

    def test_concentrated_shift_concentrated_contribution(self):
        """A shift that moves mass between just two specific bins should
        produce per-bin contributions concentrated in those two bins.
        Uniform [0, 10) reference; second half drops bin 0 entirely and
        doubles the mass in bin 9 (the rest remains uniform). Bins 0
        and 9 should dominate the contribution sum."""
        from tools.channel_drift_diagnose import decompose_psi

        a = np.linspace(0.0, 10.0, 1000, endpoint=False)
        # b drops values that fell in [0, 1) (200 of them) and adds them
        # to [9, 10): the result has 800 values in [1, 9) and 200 in
        # [9, 10), totaling 1000.
        b = np.concatenate(
            [
                np.linspace(1.0, 9.0, 800, endpoint=False),
                np.linspace(9.0, 10.0, 200, endpoint=False),
            ]
        )
        out = decompose_psi(a, b, n_bins=10)
        abs_contribs = sorted(
            (abs(r["contribution"]) for r in out["per_bin"]),
            reverse=True,
        )
        top2 = abs_contribs[0] + abs_contribs[1]
        total = sum(abs_contribs)
        assert top2 / max(total, 1e-12) > 0.9, (top2, total)


class TestSummaryStats:
    def test_known_mean_and_var(self):
        from tools.channel_drift_diagnose import summary_stats

        s = summary_stats(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert s["n"] == 5
        assert pytest.approx(s["mean"], abs=1e-9) == 3.0
        # population variance: ((1-3)^2 + ... + (5-3)^2) / 5 = 10/5 = 2.0
        assert pytest.approx(s["var"], abs=1e-9) == 2.0
        assert s["min"] == 1.0
        assert s["max"] == 5.0

    def test_includes_skew_for_asymmetric(self):
        """A right-skewed distribution should produce positive skew."""
        from tools.channel_drift_diagnose import summary_stats

        rng = np.random.default_rng(7)
        # exponential is right-skewed
        s = summary_stats(rng.exponential(1.0, 5000))
        assert s["skew"] > 0.5

    def test_handles_empty_safely(self):
        from tools.channel_drift_diagnose import summary_stats

        s = summary_stats(np.array([]))
        assert s["n"] == 0
        # Empty stats are NaN/0 — caller decides how to render. Just don't crash.


class TestPerProductDrift:
    def test_returns_sorted_by_psi_desc(self):
        from tools.channel_drift_diagnose import per_product_drift

        rng = np.random.default_rng(2)
        # Three pids: pid_A drifts hard, pid_B moderate, pid_C stable
        n = 400
        ts = np.arange(n)
        prods = {
            "PID-A": {
                "channel": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(2.5, 1, n // 2)]),
                "ts": ts,
            },
            "PID-B": {
                "channel": np.concatenate([rng.normal(0, 1, n // 2), rng.normal(0.4, 1, n // 2)]),
                "ts": ts,
            },
            "PID-C": {
                "channel": rng.normal(0, 1, n),
                "ts": ts,
            },
        }
        out = per_product_drift(prods, n_bins=10)
        psis = [r["psi"] for r in out]
        assert psis == sorted(psis, reverse=True)
        assert out[0]["pid"] == "PID-A"
        assert out[-1]["pid"] == "PID-C"
        for r in out:
            for key in ("pid", "n", "psi", "flag"):
                assert key in r

    def test_skips_short_series_safely(self):
        from tools.channel_drift_diagnose import per_product_drift

        prods = {
            "TINY": {"channel": np.array([0.1, 0.2]), "ts": np.array([0, 1])},
        }
        out = per_product_drift(prods, n_bins=10)
        assert len(out) == 1
        assert out[0]["psi"] == 0.0


class TestBinCountSensitivity:
    def test_returns_psi_per_n_bins(self):
        from tools.channel_drift_diagnose import bin_count_sensitivity

        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 500)
        b = rng.normal(0.3, 1, 500)
        result = bin_count_sensitivity(a, b, n_bins_list=(5, 10, 20))
        assert set(result.keys()) == {5, 10, 20}
        for v in result.values():
            assert isinstance(v, float)
            assert v >= 0.0

    def test_stable_for_identical_halves(self):
        from tools.channel_drift_diagnose import bin_count_sensitivity

        a = np.linspace(-1, 1, 800)
        result = bin_count_sensitivity(a, a.copy(), n_bins_list=(4, 10, 20))
        for v in result.values():
            assert v < 1e-6
