"""Unit tests for backend/tools/train_xgb_v4_5.py helpers.

Tests pure helpers (_triple_barrier_label_3class, _build_samples_for_pid,
_walk_forward_split) on synthetic candles. Orchestrator main() is
exercised by operator-run smoke test post-commit.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_candles(n: int, base_close: float = 100.0,
                  drift: float = 0.0) -> List[Dict[str, float]]:
    """Synthetic OHLCV with linear drift."""
    candles = []
    for i in range(n):
        c = base_close + drift * i
        candles.append({
            "start":  1700000000 + i * 3600,
            "open":   c - 0.1,
            "high":   c + 0.5,
            "low":    c - 0.5,
            "close":  c,
            "volume": 100.0 + i,
        })
    return candles


class TestTripleBarrierLabel3Class:
    def test_up_breach_returns_2(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # close[start]=100, threshold=0.01, forward 4 bars
        # close[start+1]=101.5 -> +1.5% > 1% -> UP breach (returns 2)
        closes = np.array([100.0, 101.5, 100.0, 99.0, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 2

    def test_down_breach_returns_0(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        closes = np.array([100.0, 98.5, 99.0, 100.0, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 0

    def test_no_breach_returns_1_neutral(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # No bar exceeds +/-1%
        closes = np.array([100.0, 100.5, 99.5, 100.5, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 1

    def test_tie_up_wins(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # First bar after start hits +exact threshold; subsequent bar would hit -
        # UP barrier checked before DOWN inside the loop -> UP wins
        closes = np.array([100.0, 101.0, 99.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=2, label_thresh=0.01,
        ) == 2

    def test_truncated_returns_none(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        closes = np.array([100.0, 101.0])  # only 1 forward bar, need 4
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) is None


class TestBuildSamplesForPid:
    def test_empty_candles_returns_empty_arrays(self):
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        X, y, ts = _build_samples_for_pid(
            [], label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 210)
        assert y.shape == (0,)
        assert ts.shape == (0,)

    def test_too_few_candles_returns_empty(self):
        """Need at least macro + BB_PREFIX + forward_hours candles."""
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(100)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 210)

    def test_returns_correct_feature_width(self):
        """500+ candles with drift -> some samples produced."""
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(500, drift=0.05)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape[1] == 210
        assert X.shape[0] == y.shape[0] == ts.shape[0]
        assert X.shape[0] > 0

    def test_labels_in_valid_set(self):
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(500, drift=0.05)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert set(np.unique(y).tolist()).issubset({0, 1, 2})


class TestWalkForwardSplit:
    def test_splits_into_three_chronological_groups(self):
        from tools.train_xgb_v4_5 import _walk_forward_split
        n = 1000
        X = np.random.rand(n, 210)
        y = np.random.randint(0, 3, n)
        ts = np.arange(n, dtype=np.int64) + 1700000000

        (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
            X, y, ts, embargo_bars=24, val_frac=0.15, cal_frac=0.15,
        )
        assert X_tr.shape[0] > 0
        assert X_va.shape[0] > 0
        assert X_ca.shape[0] > 0
        total = X_tr.shape[0] + X_va.shape[0] + X_ca.shape[0]
        assert total <= n
