"""TDD tests for feature_set_compare survivorship-aware migration (#163).

Covers the new `snapshot_ts` plumbing through `_pooled_top_n` and the
`_parse_snapshot_ts` CLI helper. The behaviour of the underlying
`survivorship_aware_top_n` is covered separately in test_pid_snapshot.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_BAR_SECS = 3600


def _entry(first_ts: int, n: int) -> dict:
    return {
        "first_ts": int(first_ts),
        "last_ts": int(first_ts + (n - 1) * _BAR_SECS),
        "last_n": n,
        "X": [torch.zeros(27, 60) for _ in range(n)],
        "y": np.zeros(n, dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }


class TestParseSnapshotTs:
    def test_none_returns_none(self):
        from tools.feature_set_compare import _parse_snapshot_ts

        assert _parse_snapshot_ts(None, {}) is None

    def test_explicit_int_string_parsed(self):
        from tools.feature_set_compare import _parse_snapshot_ts

        assert _parse_snapshot_ts("1700000000", {}) == 1700000000

    def test_auto_resolves_to_recommended(self):
        from tools.feature_set_compare import _parse_snapshot_ts

        prods = {
            "A": _entry(first_ts=1000 * _BAR_SECS, n=5),
            "B": _entry(first_ts=2000 * _BAR_SECS, n=5),
            "C": _entry(first_ts=3000 * _BAR_SECS, n=5),
        }
        assert _parse_snapshot_ts("auto", prods) == 2000 * _BAR_SECS

    def test_auto_falls_back_to_none_on_empty(self):
        from tools.feature_set_compare import _parse_snapshot_ts

        # No products with non-empty X → recommended_snapshot_ts returns 0
        assert _parse_snapshot_ts("auto", {}) is None


class TestPooledTopNSnapshotPlumbing:
    def test_legacy_passthrough_when_snapshot_none(self):
        from tools.feature_set_compare import _pooled_top_n

        prods = {
            "OLD": _entry(first_ts=0, n=5),
            "OLDER": _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        # Legacy: NEWCOMER (100) wins, then OLDER (10), then OLD (5)
        X, y, ts = _pooled_top_n(prods, n=2, snapshot_ts=None)
        # 100 + 10 = 110 samples in legacy mode
        assert len(y) == 110

    def test_snapshot_excludes_newcomers(self):
        from tools.feature_set_compare import _pooled_top_n

        prods = {
            "OLD": _entry(first_ts=0, n=5),
            "OLDER": _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        # Cutoff at 15h → NEWCOMER's first sample (20h) is post-cutoff,
        # only OLD (5) + OLDER (10 truncated to ≤15h) remain.
        X, y, ts = _pooled_top_n(prods, n=2, snapshot_ts=15 * _BAR_SECS)
        assert len(y) < 110
        # No timestamps should exceed the cutoff
        assert ts.max() <= 15 * _BAR_SECS
