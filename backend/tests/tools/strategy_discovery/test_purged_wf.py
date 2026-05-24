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
