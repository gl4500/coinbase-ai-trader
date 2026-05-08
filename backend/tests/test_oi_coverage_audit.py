"""TDD tests for OKX OI coverage audit (#210).

Follow-up to #209's Ch 27 finding: 19/20 pids reported per-pid PSI=0.0000
on the OI channel, meaning the channel was constant (typically zero) in
at least one half of each pid's time series. Before doing further
OI-feature work we need to know: for each pid, how much of the OI
history is missing/constant, and is the missingness clustered at the
start (legitimate backfill) or scattered (cache-build defaulting bug).

Pure numpy primitives:
  - coverage_stats(arr): n, n_zero, n_nonzero, frac_zero, first_nonzero_idx
  - per_pid_coverage(prods): list[{pid, n, frac_zero, first_nonzero_idx, ...}]
  - leading_zero_run(arr): length of contiguous zero prefix

The CLI lives in tools/oi_coverage_audit.py and reports against the
live cache.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestCoverageStats:

    def test_all_zero_series(self):
        from tools.oi_coverage_audit import coverage_stats
        s = coverage_stats(np.zeros(100))
        assert s["n"] == 100
        assert s["n_zero"] == 100
        assert s["n_nonzero"] == 0
        assert s["frac_zero"] == 1.0
        assert s["first_nonzero_idx"] is None

    def test_all_nonzero_series(self):
        from tools.oi_coverage_audit import coverage_stats
        s = coverage_stats(np.full(50, 0.3))
        assert s["n"] == 50
        assert s["n_zero"] == 0
        assert s["n_nonzero"] == 50
        assert s["frac_zero"] == 0.0
        assert s["first_nonzero_idx"] == 0

    def test_zero_prefix_then_nonzero(self):
        from tools.oi_coverage_audit import coverage_stats
        arr = np.concatenate([np.zeros(60), np.full(40, 0.5)])
        s = coverage_stats(arr)
        assert s["n"] == 100
        assert s["n_zero"] == 60
        assert s["n_nonzero"] == 40
        assert s["frac_zero"] == pytest.approx(0.6)
        assert s["first_nonzero_idx"] == 60

    def test_empty_series_safe(self):
        from tools.oi_coverage_audit import coverage_stats
        s = coverage_stats(np.array([]))
        assert s["n"] == 0
        assert s["frac_zero"] == 0.0
        assert s["first_nonzero_idx"] is None


class TestLeadingZeroRun:

    def test_no_leading_zeros(self):
        from tools.oi_coverage_audit import leading_zero_run
        assert leading_zero_run(np.array([0.1, 0.2, 0.0, 0.3])) == 0

    def test_all_zeros(self):
        from tools.oi_coverage_audit import leading_zero_run
        assert leading_zero_run(np.zeros(20)) == 20

    def test_partial_leading_zeros(self):
        from tools.oi_coverage_audit import leading_zero_run
        arr = np.concatenate([np.zeros(15), [0.1, 0.0, 0.2]])
        assert leading_zero_run(arr) == 15

    def test_empty(self):
        from tools.oi_coverage_audit import leading_zero_run
        assert leading_zero_run(np.array([])) == 0


class TestPerPidCoverage:

    def test_returns_one_row_per_pid_sorted_by_frac_zero_desc(self):
        from tools.oi_coverage_audit import per_pid_coverage
        prods = {
            "PID-A": {"channel": np.zeros(100), "ts": np.arange(100)},  # all zero
            "PID-B": {"channel": np.concatenate([np.zeros(50),
                                                  np.full(50, 0.3)]),
                       "ts": np.arange(100)},  # half zero
            "PID-C": {"channel": np.full(80, 0.5), "ts": np.arange(80)},  # all real
        }
        rows = per_pid_coverage(prods)
        assert len(rows) == 3
        # most-zero first
        assert rows[0]["pid"] == "PID-A"
        assert rows[0]["frac_zero"] == 1.0
        assert rows[1]["pid"] == "PID-B"
        assert rows[1]["frac_zero"] == pytest.approx(0.5)
        assert rows[2]["pid"] == "PID-C"
        assert rows[2]["frac_zero"] == 0.0
        for r in rows:
            for k in ("pid", "n", "n_zero", "n_nonzero",
                      "frac_zero", "first_nonzero_idx", "leading_zeros"):
                assert k in r, r

    def test_classifies_leading_block_vs_scattered(self):
        """If all the zeros are at the start, leading_zeros == n_zero ⇒
        legitimate backfill gap. If zeros are scattered, leading_zeros
        will be smaller than n_zero ⇒ defaulting bug suspect."""
        from tools.oi_coverage_audit import per_pid_coverage
        prods = {
            "BACKFILL": {  # all zeros are at start
                "channel": np.concatenate([np.zeros(40), np.full(60, 0.3)]),
                "ts": np.arange(100),
            },
            "SCATTERED": {  # zeros mixed throughout
                "channel": np.where(
                    np.arange(100) % 3 == 0, 0.0, 0.3,
                ),
                "ts": np.arange(100),
            },
        }
        rows = {r["pid"]: r for r in per_pid_coverage(prods)}
        # BACKFILL: leading_zeros == n_zero (all zeros at start)
        assert rows["BACKFILL"]["leading_zeros"] == rows["BACKFILL"]["n_zero"]
        # SCATTERED: leading_zeros (1 leading zero) < n_zero (~33)
        assert rows["SCATTERED"]["leading_zeros"] < rows["SCATTERED"]["n_zero"]
