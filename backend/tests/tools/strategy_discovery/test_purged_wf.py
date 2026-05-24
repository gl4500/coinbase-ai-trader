"""Tests for tools.strategy_discovery.purged_wf (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from tools.strategy_discovery.purged_wf import inner_folds, outer_folds


def test_5_folds_cover_all_rows_disjointly():
    n = 1000
    folds = outer_folds(n, n_folds=5, embargo_bars=0)
    assert len(folds) == 5
    test_union = np.concatenate([test_idx for _, test_idx in folds])
    assert len(np.unique(test_union)) == len(test_union), "test sets overlap"
    assert set(test_union.tolist()) == set(range(n)), "test sets don't cover all rows"
    expected_size = n // 5
    for _, test_idx in folds:
        assert abs(len(test_idx) - expected_size) <= 1


def test_embargo_drops_train_rows_within_horizon_of_test_start():
    n = 100
    folds = outer_folds(n, n_folds=5, embargo_bars=10)
    train_idx, test_idx = folds[2]
    assert test_idx.tolist() == list(range(40, 60))
    assert set(range(0, 30)).issubset(set(train_idx.tolist()))
    assert set(range(60, 100)).issubset(set(train_idx.tolist()))
    for embargoed in range(30, 40):
        assert embargoed not in train_idx.tolist()
