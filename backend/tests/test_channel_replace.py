"""TDD tests for tools/channel_replace.py — single-channel replacement probe.

The harness answers: "if I substitute channel c's data with a candidate
signal (OI, hour-of-day, etc.), how much does mean_auc move vs the no-swap
baseline?" Used to evaluate new feature candidates against the existing
28-channel stack before paying the integration cost.

Pairs with channel_ablation.py — replace == ablate-then-add in one step.
"""

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestReplaceChannel:
    def test_substitutes_target_channel(self):
        from tools.channel_replace import _replace_channel

        X = np.zeros((3, 28, 60), dtype=np.float32)
        sig = np.full((3, 60), 7.0, dtype=np.float32)
        out = _replace_channel(X, 4, sig)
        assert (out[:, 4, :] == 7.0).all()

    def test_leaves_other_channels_intact(self):
        from tools.channel_replace import _replace_channel

        X = np.full((2, 28, 60), 3.0, dtype=np.float32)
        sig = np.zeros((2, 60), dtype=np.float32)
        out = _replace_channel(X, 10, sig)
        for c in range(28):
            if c == 10:
                continue
            assert (out[:, c, :] == 3.0).all(), f"ch{c} mutated"

    def test_does_not_mutate_input(self):
        from tools.channel_replace import _replace_channel

        X = np.ones((1, 28, 60), dtype=np.float32)
        original = X.copy()
        sig = np.zeros((1, 60), dtype=np.float32)
        _ = _replace_channel(X, 0, sig)
        assert (X == original).all()

    def test_shape_mismatch_raises(self):
        from tools.channel_replace import _replace_channel

        X = np.zeros((4, 28, 60), dtype=np.float32)
        bad = np.zeros((3, 60), dtype=np.float32)  # wrong N
        with pytest.raises((ValueError, AssertionError)):
            _replace_channel(X, 0, bad)
        bad2 = np.zeros((4, 30), dtype=np.float32)  # wrong T
        with pytest.raises((ValueError, AssertionError)):
            _replace_channel(X, 0, bad2)

    def test_invalid_channel_raises(self):
        from tools.channel_replace import _replace_channel

        X = np.zeros((1, 28, 60), dtype=np.float32)
        sig = np.zeros((1, 60), dtype=np.float32)
        with pytest.raises((IndexError, ValueError)):
            _replace_channel(X, 28, sig)
        with pytest.raises((IndexError, ValueError)):
            _replace_channel(X, -1, sig)


class TestRunReplace:
    def test_returns_mean_auc_and_delta(self):
        """Smoke test on synthetic input — runs CV, returns expected keys."""
        from tools.channel_replace import run_replace

        rng = np.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 28, 60)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        ts = (np.arange(n) * 3600).astype(np.int64)
        sig = rng.standard_normal((n, 60)).astype(np.float32)
        result = run_replace(
            X, y, ts, channel_idx=15, replacement=sig, n_folds=2, embargo_hours=0, n_estimators=10
        )
        assert set(result.keys()) >= {"channel", "baseline_auc", "replaced_auc", "delta"}
        assert result["channel"] == 15
        assert result["delta"] == pytest.approx(
            result["replaced_auc"] - result["baseline_auc"], abs=1e-9
        )

    def test_replacing_with_same_data_yields_zero_delta(self):
        """Replacing channel c with X[:, c, :] is a no-op → delta ≈ 0."""
        from tools.channel_replace import run_replace

        rng = np.random.default_rng(2)
        n = 250
        X = rng.standard_normal((n, 28, 60)).astype(np.float32)
        y = rng.integers(0, 2, size=n).astype(np.float32)
        ts = (np.arange(n) * 3600).astype(np.int64)
        sig = X[:, 5, :].copy()
        result = run_replace(
            X, y, ts, channel_idx=5, replacement=sig, n_folds=2, embargo_hours=0, n_estimators=10
        )
        assert result["delta"] == pytest.approx(0.0, abs=1e-9)
