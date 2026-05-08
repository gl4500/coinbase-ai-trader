"""TDD tests for hour_of_day_probe survivorship-aware migration (#163).

Mirrors the feature_set_compare snapshot tests: covers `snapshot_ts`
plumbing through `_load_pooled` and the `_parse_snapshot_ts` CLI helper.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
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
        from tools.hour_of_day_probe import _parse_snapshot_ts
        assert _parse_snapshot_ts(None, {}) is None

    def test_explicit_int_string_parsed(self):
        from tools.hour_of_day_probe import _parse_snapshot_ts
        assert _parse_snapshot_ts("1700000000", {}) == 1700000000

    def test_auto_resolves_to_recommended(self):
        from tools.hour_of_day_probe import _parse_snapshot_ts
        prods = {
            "A": _entry(first_ts=1000 * _BAR_SECS, n=5),
            "B": _entry(first_ts=2000 * _BAR_SECS, n=5),
            "C": _entry(first_ts=3000 * _BAR_SECS, n=5),
        }
        assert _parse_snapshot_ts("auto", prods) == 2000 * _BAR_SECS

    def test_auto_falls_back_to_none_on_empty(self):
        from tools.hour_of_day_probe import _parse_snapshot_ts
        assert _parse_snapshot_ts("auto", {}) is None


class TestLoadPooledSnapshotPlumbing:

    def test_legacy_passthrough_when_snapshot_none(self, tmp_path, monkeypatch):
        from tools import hour_of_day_probe
        prods = {
            "OLD":      _entry(first_ts=0, n=5),
            "OLDER":    _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        cache = tmp_path / "cache.pt"
        torch.save({"products": prods}, cache)
        monkeypatch.setattr(hour_of_day_probe, "_CACHE_PATH", str(cache))
        # Legacy: top-2 by len(X) → NEWCOMER (100) + OLDER (10) = 110 samples
        X, y, ts = hour_of_day_probe._load_pooled(n=2, snapshot_ts=None)
        assert len(y) == 110

    def test_snapshot_excludes_newcomers(self, tmp_path, monkeypatch):
        from tools import hour_of_day_probe
        prods = {
            "OLD":      _entry(first_ts=0, n=5),
            "OLDER":    _entry(first_ts=0, n=10),
            "NEWCOMER": _entry(first_ts=20 * _BAR_SECS, n=100),
        }
        cache = tmp_path / "cache.pt"
        torch.save({"products": prods}, cache)
        monkeypatch.setattr(hour_of_day_probe, "_CACHE_PATH", str(cache))
        X, y, ts = hour_of_day_probe._load_pooled(n=2, snapshot_ts=15 * _BAR_SECS)
        assert len(y) < 110
        assert ts.max() <= 15 * _BAR_SECS
