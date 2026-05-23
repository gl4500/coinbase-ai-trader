"""TDD tests for _timescale_sanity survivorship-aware migration (#163).

Smaller surface area than the other consumers: this is a one-shot
diagnostic with no public API beyond `main()`. We extract `_pick_pids`
as the only testable seam and verify it delegates to
`survivorship_aware_top_n`.
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


class TestPickPids:

    def test_legacy_passthrough_when_snapshot_none(self):
        from tools._timescale_sanity import _pick_pids
        prods = {
            "OLD":      _entry(first_ts=0, n=5),
            "OLDER":    _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        # Legacy: top-2 by len(X) → NEWCOMER (100) + OLDER (10)
        pids = _pick_pids(prods, n=2, snapshot_ts=None)
        assert "NEWCOMER" in pids
        assert "OLDER" in pids

    def test_snapshot_excludes_newcomers(self):
        from tools._timescale_sanity import _pick_pids
        prods = {
            "OLD":      _entry(first_ts=0, n=5),
            "OLDER":    _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        pids = _pick_pids(prods, n=2, snapshot_ts=15 * _BAR_SECS)
        assert "NEWCOMER" not in pids
