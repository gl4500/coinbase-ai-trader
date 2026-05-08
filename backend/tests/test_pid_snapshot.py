"""TDD tests for tools/pid_snapshot.py — survivorship-aware top-N selection.

Existing probes select top-N pids by total `len(entry["X"])`, which is
post-hoc: products that grew the most data dominate. The fix exposes
a `snapshot_ts` parameter so selection is based on samples-existing-at-time.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


_BAR_SECS = 3600


def _entry(first_ts: int, n: int, indices_step: int = 1) -> dict:
    return {
        "first_ts": int(first_ts),
        "last_ts": int(first_ts + (n - 1) * _BAR_SECS),
        "last_n": n,
        "X": [np.zeros((28, 60), dtype=np.float32) for _ in range(n)],
        "y": np.zeros(n, dtype=np.float32),
        "indices": np.arange(0, n * indices_step, indices_step, dtype=np.int64),
    }


class TestSurvivorshipAwareTopN:

    def test_no_snapshot_matches_legacy_sort(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        prods = {
            "A": _entry(0, 10),
            "B": _entry(0, 30),
            "C": _entry(0, 20),
        }
        out = survivorship_aware_top_n(prods, n=2, snapshot_ts=None)
        assert out == ["B", "C"]

    def test_snapshot_excludes_data_after_cutoff(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        # A has 10 samples ending at ts=10*3600
        # B starts AFTER cutoff (ts=20*3600), all samples post-cutoff
        # C has 5 samples ending at ts=5*3600 (below cutoff)
        prods = {
            "A": _entry(first_ts=0, n=10),
            "B": _entry(first_ts=20 * _BAR_SECS, n=30),
            "C": _entry(first_ts=0, n=5),
        }
        cutoff = 15 * _BAR_SECS
        out = survivorship_aware_top_n(prods, n=2, snapshot_ts=cutoff)
        # Only A (10) and C (5) are visible at cutoff. B is excluded entirely.
        assert out == ["A", "C"]

    def test_snapshot_partial_truncates_count(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        # A: 100 samples, all before cutoff
        # B: 100 samples but only 5 before cutoff
        prods = {
            "A": _entry(first_ts=0, n=100),
            "B": _entry(first_ts=95 * _BAR_SECS, n=100),
        }
        cutoff = 100 * _BAR_SECS
        out = survivorship_aware_top_n(prods, n=2, snapshot_ts=cutoff)
        # A has 100 ≤ cutoff samples; B has only 5 ≤ cutoff. A first.
        assert out[0] == "A"
        assert out[1] == "B"

    def test_returns_at_most_n(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        prods = {f"P{i}": _entry(0, i + 1) for i in range(5)}
        out = survivorship_aware_top_n(prods, n=3, snapshot_ts=None)
        assert len(out) == 3
        assert out == ["P4", "P3", "P2"]

    def test_empty_prods_returns_empty(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        assert survivorship_aware_top_n({}, n=5, snapshot_ts=None) == []

    def test_skips_empty_X_entries(self):
        from tools.pid_snapshot import survivorship_aware_top_n
        prods = {
            "A": _entry(0, 5),
            "B": {"first_ts": 0, "last_ts": 0, "last_n": 0,
                  "X": [], "y": np.array([], dtype=np.float32),
                  "indices": np.array([], dtype=np.int64)},
            "C": _entry(0, 3),
        }
        out = survivorship_aware_top_n(prods, n=5, snapshot_ts=None)
        assert "B" not in out
        assert out == ["A", "C"]

    def test_default_snapshot_ts_recommendation(self):
        """`recommended_snapshot_ts` returns the median first_ts across products
        — a sensible cutoff that excludes products that joined recently."""
        from tools.pid_snapshot import recommended_snapshot_ts
        prods = {
            "A": _entry(first_ts=0, n=5),
            "B": _entry(first_ts=10 * _BAR_SECS, n=5),
            "C": _entry(first_ts=20 * _BAR_SECS, n=5),
            "D": _entry(first_ts=30 * _BAR_SECS, n=5),
            "E": _entry(first_ts=40 * _BAR_SECS, n=5),
        }
        ts = recommended_snapshot_ts(prods)
        # median first_ts of A/B/C/D/E = 20*3600
        assert ts == 20 * _BAR_SECS
