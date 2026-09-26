"""Unit tests for backend/tools/train_xgb_v4.py helpers.

We test the pure helpers (_build_samples_for_pid, _walk_forward_split,
_triple_barrier_label) on synthetic candles. The orchestrator main() is
exercised end-to-end by operator-run smoke-test after the commit lands.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_candles(n: int, base_close: float = 100.0, drift: float = 0.0) -> List[Dict[str, float]]:
    """Synthetic OHLCV: linear drift, volume ramping."""
    candles = []
    for i in range(n):
        c = base_close + drift * i
        candles.append(
            {
                "start": 1700000000 + i * 3600,
                "open": c - 0.1,
                "high": c + 0.5,
                "low": c - 0.5,
                "close": c,
                "volume": 100.0 + i,
            }
        )
    return candles


class TestTripleBarrierLabel:
    def test_up_breach_returns_1(self):
        from tools.train_xgb_v4 import _triple_barrier_label

        # close[start]=100, threshold=0.01, forward 4 bars
        # close[start+1]=101.5 -> +1.5% > 1% -> UP = 1
        closes = np.array([100.0, 101.5, 100.0, 99.0, 100.0])
        assert _triple_barrier_label(closes, start=0, forward_hours=4, label_thresh=0.01) == 1

    def test_down_breach_returns_0(self):
        from tools.train_xgb_v4 import _triple_barrier_label

        closes = np.array([100.0, 98.5, 99.0, 100.0, 100.0])
        assert _triple_barrier_label(closes, start=0, forward_hours=4, label_thresh=0.01) == 0

    def test_no_breach_returns_0(self):
        from tools.train_xgb_v4 import _triple_barrier_label

        closes = np.array([100.0, 100.5, 99.5, 100.5, 100.0])
        # no bar exceeds +/-1%
        assert _triple_barrier_label(closes, start=0, forward_hours=4, label_thresh=0.01) == 0

    def test_returns_none_if_window_truncated(self):
        from tools.train_xgb_v4 import _triple_barrier_label

        closes = np.array([100.0, 101.0])
        # forward_hours=4 but only 1 forward bar available
        assert _triple_barrier_label(closes, start=0, forward_hours=4, label_thresh=0.01) is None


class TestBuildSamplesForPid:
    def test_empty_candles_returns_empty_arrays(self):
        from tools.train_xgb_v4 import _build_samples_for_pid

        X, y, ts = _build_samples_for_pid(
            [],
            label_thresh=0.003,
            forward_hours=4,
            micro=60,
            meso=168,
            macro=336,
        )
        assert X.shape == (0, 150)
        assert y.shape == (0,)
        assert ts.shape == (0,)

    def test_too_few_candles_returns_empty(self):
        """Need at least macro + forward_hours candles to produce any sample."""
        from tools.train_xgb_v4 import _build_samples_for_pid

        candles = _make_candles(100)  # < 336 macro
        X, y, ts = _build_samples_for_pid(
            candles,
            label_thresh=0.003,
            forward_hours=4,
            micro=60,
            meso=168,
            macro=336,
        )
        assert X.shape == (0, 150)

    def test_returns_correct_feature_width(self):
        from tools.train_xgb_v4 import _build_samples_for_pid

        # 500 candles, drift up -> some samples produced
        candles = _make_candles(500, drift=0.01)
        X, y, ts = _build_samples_for_pid(
            candles,
            label_thresh=0.003,
            forward_hours=4,
            micro=60,
            meso=168,
            macro=336,
        )
        assert X.shape[1] == 150
        assert X.shape[0] == y.shape[0] == ts.shape[0]
        assert X.shape[0] > 0


class TestWalkForwardSplit:
    def test_splits_into_three_chronological_groups(self):
        from tools.train_xgb_v4 import _walk_forward_split

        n = 1000
        X = np.random.rand(n, 150)
        y = np.random.randint(0, 2, n)
        ts = np.arange(n, dtype=np.int64) + 1700000000

        (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
            X,
            y,
            ts,
            embargo_bars=4,
            val_frac=0.15,
            cal_frac=0.15,
        )
        # Train < Val < Cal chronologically; no overlap
        assert X_tr.shape[0] > 0
        assert X_va.shape[0] > 0
        assert X_ca.shape[0] > 0
        # Total should be <= n (embargo gaps removed)
        total = X_tr.shape[0] + X_va.shape[0] + X_ca.shape[0]
        assert total <= n
        # Embargo creates a gap
        assert total <= n - 2 * 4  # at most 2 embargo gaps of 4 bars
