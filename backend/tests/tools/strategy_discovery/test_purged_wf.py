"""Chronological validation must never train on the future."""

import numpy as np
import pytest

from tools.strategy_discovery.purged_wf import inner_folds, outer_folds


def test_five_disjoint_tests_follow_initial_warmup():
    folds = outer_folds(1000, n_folds=5, embargo_bars=0)
    assert len(folds) == 5
    tests = np.concatenate([test for _, test in folds])
    np.testing.assert_array_equal(tests, np.arange(170, 1000))
    for train, test in folds:
        np.testing.assert_array_equal(train, np.arange(test[0]))
        assert len(test) == 166


def test_embargo_excludes_label_overlap_and_all_future_rows():
    for train, test in outer_folds(100, n_folds=5, embargo_bars=10):
        np.testing.assert_array_equal(train, np.arange(test[0] - 10))
        assert train[-1] + 10 < test[0]


def test_nested_folds_are_chronological_subsets_with_gap():
    for outer_train, outer_test in outer_folds(1000, embargo_bars=24):
        for train, test in inner_folds(outer_train, embargo_bars=24):
            assert set(train) | set(test) <= set(outer_train)
            assert train[-1] + 24 < test[0] < outer_test[0]
            np.testing.assert_array_equal(train, np.arange(len(train)))


@pytest.mark.parametrize("n,gap", [(0, 0), (3, 0), (100, 100)])
def test_insufficient_history_returns_no_empty_folds(n, gap):
    assert outer_folds(n, embargo_bars=gap) == []


def test_only_usable_folds_returned_when_gap_consumes_early_history():
    folds = outer_folds(60, n_folds=5, embargo_bars=25)
    assert len(folds) == 3
    assert [test[0] for _, test in folds] == [30, 40, 50]
    assert all(len(train) and len(test) for train, test in folds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_rows": -1},
        {"n_rows": 10, "n_folds": 0},
        {"n_rows": 10, "embargo_bars": -1},
        {"n_rows": 10.5},
    ],
)
def test_invalid_split_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        outer_folds(**kwargs)


@pytest.mark.parametrize(
    "indices",
    [
        np.array([2, 1]),
        np.array([1, 1]),
        np.array([0.5, 1.5]),
        np.array([[0, 1]]),
        np.array([-1, 0]),
    ],
)
def test_invalid_inner_indices_rejected(indices):
    with pytest.raises(ValueError):
        inner_folds(indices)


def test_inner_maps_sparse_indices_without_reordering():
    indices = np.arange(0, 200, 2)
    for train, test in inner_folds(indices, embargo_bars=10):
        assert set(train) | set(test) <= set(indices)
        assert train[-1] + 10 < test[0]
