"""TDD tests for tools/walk_forward.py — purged walk-forward CV splitter.

Phase 2 of the CNN -> XGBoost transition. The CNN's 80/20 cut at
cnn_agent.py:2565 ignores the 4-hour forward-window embargo and concatenates
products without global time-sorting (Phase 0 finding). This module fixes
both:

  - global time-sort by absolute timestamp
  - 4-hour embargo: drop train samples whose forward window enters val region
  - K time-ordered folds with per-fold visibility (AUC reported per fold)
"""

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _hourly_ts(n: int, start: int = 1_700_000_000) -> np.ndarray:
    """N timestamps, one per hour."""
    return np.arange(start, start + n * 3600, 3600, dtype=np.int64)


# ── Signature & basic shape ─────────────────────────────────────────────


class TestSignature:
    def test_function_exists(self):
        from tools.walk_forward import purged_walk_forward_splits

        assert callable(purged_walk_forward_splits)

    def test_yields_iterator_of_tuples(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(200)
        splits = list(purged_walk_forward_splits(ts, n_folds=4, embargo_hours=4))
        assert len(splits) == 4
        for s in splits:
            assert isinstance(s, tuple) and len(s) == 2
            train_idx, val_idx = s
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(val_idx, np.ndarray)


# ── Disjoint train/val ──────────────────────────────────────────────────


class TestDisjoint:
    def test_train_and_val_indices_dont_overlap(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(500)
        for train_idx, val_idx in purged_walk_forward_splits(ts, n_folds=5, embargo_hours=4):
            assert len(np.intersect1d(train_idx, val_idx)) == 0

    def test_val_regions_dont_overlap_across_folds(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(500)
        all_val = []
        for _, val_idx in purged_walk_forward_splits(ts, n_folds=5, embargo_hours=4):
            all_val.append(val_idx)
        cat = np.concatenate(all_val)
        assert len(cat) == len(np.unique(cat)), (
            "Val folds must partition (no sample in two val sets)"
        )


# ── Embargo enforcement ─────────────────────────────────────────────────


class TestEmbargo:
    def test_no_train_sample_within_embargo_of_val_start(self):
        """A train sample at time t with forward_hours=4 looks at t+4h.
        If val starts at t_val, no train sample at t_val - 4h or later
        should appear in train (its forward window enters val)."""
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(500)
        embargo = 4
        for train_idx, val_idx in purged_walk_forward_splits(ts, n_folds=5, embargo_hours=embargo):
            if len(val_idx) == 0 or len(train_idx) == 0:
                continue
            val_ts = ts[val_idx]
            train_ts = ts[train_idx]
            t_val_start = val_ts.min()
            t_val_end = val_ts.max()
            # Train samples before val: forward window must not reach val_start
            before = train_ts[train_ts < t_val_start]
            if len(before) > 0:
                assert before.max() + embargo * 3600 <= t_val_start, (
                    "Train sample's forward window enters val region — "
                    f"latest train before val: {before.max()}, val_start: {t_val_start}"
                )
            # Train samples after val: val's forward window must not reach them
            after = train_ts[train_ts > t_val_end]
            if len(after) > 0:
                assert after.min() >= t_val_end + embargo * 3600, (
                    "Val sample's forward window reaches train sample — "
                    f"earliest train after val: {after.min()}, val_end: {t_val_end}"
                )


# ── Time-ordered folds ──────────────────────────────────────────────────


class TestTimeOrder:
    def test_val_regions_appear_in_chronological_order(self):
        """Fold k's val region should start later than fold k-1's val region."""
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(500)
        prev_max = -1
        for _, val_idx in purged_walk_forward_splits(ts, n_folds=5, embargo_hours=4):
            v_min = ts[val_idx].min()
            assert v_min > prev_max, "Val regions must advance in time across folds"
            prev_max = ts[val_idx].max()


# ── Coverage ────────────────────────────────────────────────────────────


class TestCoverage:
    def test_each_sample_appears_in_exactly_one_val_fold(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(500)
        all_val = np.concatenate(
            [v for _, v in purged_walk_forward_splits(ts, n_folds=5, embargo_hours=4)]
        )
        assert sorted(all_val.tolist()) == list(range(len(ts))), (
            "Val folds must partition the dataset exactly"
        )


# ── Unsorted-input safety ───────────────────────────────────────────────


class TestUnsortedInput:
    def test_unsorted_timestamps_are_handled(self):
        """Real X_list is concatenated by product, so timestamps arrive
        unsorted. The splitter must internally sort by time before slicing."""
        from tools.walk_forward import purged_walk_forward_splits

        ts_sorted = _hourly_ts(200)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(ts_sorted))
        ts_unsorted = ts_sorted[perm]

        for train_idx, val_idx in purged_walk_forward_splits(
            ts_unsorted, n_folds=4, embargo_hours=4
        ):
            train_ts = ts_unsorted[train_idx]
            val_ts = ts_unsorted[val_idx]
            if len(train_ts) and len(val_ts):
                # Most train ts should be before val_ts (or after with embargo);
                # allow no train ts inside [val_min - embargo, val_max + embargo)
                window_lo = val_ts.min() - 4 * 3600
                window_hi = val_ts.max() + 4 * 3600
                in_window = (train_ts >= window_lo) & (train_ts < window_hi)
                assert not in_window.any(), (
                    "Unsorted input: embargo not enforced after internal sort"
                )


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_too_few_samples_raises(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(3)
        with pytest.raises(ValueError):
            list(purged_walk_forward_splits(ts, n_folds=5, embargo_hours=4))

    def test_n_folds_minimum_is_two(self):
        from tools.walk_forward import purged_walk_forward_splits

        ts = _hourly_ts(100)
        with pytest.raises(ValueError):
            list(purged_walk_forward_splits(ts, n_folds=1, embargo_hours=4))
