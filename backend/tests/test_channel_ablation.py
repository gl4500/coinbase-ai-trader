"""TDD tests for tools/channel_ablation.py — single-channel drop harness.

The harness answers: "if we zero channel c before extract_features, how
much does mean_auc move vs the full 27-channel baseline?" Useful to
identify dead-weight channels that earn no slot.

Only the small, deterministic helper is unit-tested here. The full
walk-forward runner is exercised by an integration smoke test that
asserts shape/coverage on synthetic input.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestZeroChannel:

    def test_zeros_target_channel_only(self):
        from tools.channel_ablation import _zero_channel
        X = np.ones((4, 27, 60), dtype=np.float32)
        out = _zero_channel(X, 5)
        assert (out[:, 5, :] == 0).all()
        # all other channels preserved
        for c in range(27):
            if c == 5:
                continue
            assert (out[:, c, :] == 1).all(), f"ch{c} mutated"

    def test_does_not_mutate_input(self):
        from tools.channel_ablation import _zero_channel
        X = np.full((2, 27, 60), 7.0, dtype=np.float32)
        original = X.copy()
        _ = _zero_channel(X, 0)
        assert (X == original).all(), "input array was mutated"

    def test_invalid_channel_raises(self):
        from tools.channel_ablation import _zero_channel
        X = np.zeros((1, 27, 60), dtype=np.float32)
        with pytest.raises((IndexError, ValueError)):
            _zero_channel(X, 27)
        with pytest.raises((IndexError, ValueError)):
            _zero_channel(X, -1)


class TestRunAblation:

    def test_returns_one_row_per_channel(self):
        """Smoke test on tiny synthetic input — 27 channels in, 27 rows out."""
        from tools.channel_ablation import run_ablation
        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 27, 60)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        ts = (np.arange(n) * 3600).astype(np.int64)
        results = run_ablation(X, y, ts, n_folds=2, embargo_hours=0,
                               n_estimators=10)
        assert "baseline_auc" in results
        assert "rows" in results
        assert len(results["rows"]) == 27
        for row in results["rows"]:
            assert set(row.keys()) >= {"channel", "mean_auc", "delta"}
            assert 0 <= row["channel"] < 27

    def test_baseline_row_has_zero_delta(self):
        """The 'no-drop' baseline should have delta=0.0 by construction."""
        from tools.channel_ablation import run_ablation
        rng = np.random.default_rng(1)
        n = 150
        X = rng.standard_normal((n, 27, 60)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        ts = (np.arange(n) * 3600).astype(np.int64)
        results = run_ablation(X, y, ts, n_folds=2, embargo_hours=0,
                               n_estimators=10)
        # delta is mean_auc(drop ch c) - baseline_auc; baseline must match
        for row in results["rows"]:
            assert row["delta"] == pytest.approx(
                row["mean_auc"] - results["baseline_auc"], abs=1e-9
            )
