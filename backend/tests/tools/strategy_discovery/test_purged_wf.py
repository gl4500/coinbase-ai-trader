"""Tests for tools.strategy_discovery.purged_wf (Phase 3)."""

from __future__ import annotations

import numpy as np

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


def test_nested_inner_cv_uses_only_outer_train():
    n = 600
    outer = outer_folds(n, n_folds=5, embargo_bars=0)
    outer_train, outer_test = outer[2]
    inner = inner_folds(outer_train, n_folds=3, embargo_bars=0)
    assert len(inner) == 3
    outer_train_set = set(outer_train.tolist())
    outer_test_set = set(outer_test.tolist())
    for inner_train, inner_test in inner:
        for idx in inner_train.tolist() + inner_test.tolist():
            assert idx in outer_train_set, f"inner idx {idx} not in outer train"
            assert idx not in outer_test_set, f"inner idx {idx} leaked from outer test"
