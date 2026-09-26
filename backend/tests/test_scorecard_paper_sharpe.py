import numpy as np
import pytest

from tools._scorecard._paper_sharpe import paper_sharpe_per_fold


def test_paper_sharpe_constant_per_signal_return():
    """If every fold has identical net returns, std across folds = 0."""
    np.random.default_rng(0)
    n = 200
    scores = np.full(n, 0.9)
    returns = np.full(n, 0.01)  # constant => std=0 per fold => Sharpe inf
    fold_ids = np.tile(np.arange(5), n // 5)
    fold_spans_days = {i: 30.0 for i in range(5)}

    # constant returns => per-signal std = 0 => guard returns NaN sharpe
    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == n
    assert np.isnan(mean_s)
    assert np.isnan(std_s)


def test_paper_sharpe_positive_with_noise():
    """Positive mean and finite std => positive annualized Sharpe."""
    rng = np.random.default_rng(42)
    n_per_fold = 100
    n_folds = 5
    n = n_per_fold * n_folds
    scores = np.full(n, 0.9)
    returns = rng.normal(loc=0.005, scale=0.01, size=n)
    fold_ids = np.repeat(np.arange(n_folds), n_per_fold)
    fold_spans_days = {i: 30.0 for i in range(n_folds)}

    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == n
    assert mean_s > 0
    assert std_s >= 0
    # Sanity: per-signal Sharpe ~0.5, annualization factor sqrt(100*365/30) ~ 34.9
    # => mean_s should be in single-digits not 0.5
    assert 1.0 < mean_s < 50.0


def test_paper_sharpe_fold_with_no_fires_excluded():
    """Folds with zero signals should not contribute to mean/std."""
    n = 200
    scores = np.zeros(n)
    scores[:50] = 0.9  # only fold 0 has fires
    returns = np.full(n, 0.005)
    fold_ids = np.repeat(np.arange(5), 40)[:n]  # ensure fold 0 has at least 1
    fold_ids = np.array([0] * 50 + [1] * 50 + [2] * 50 + [3] * 25 + [4] * 25)
    fold_spans_days = {i: 30.0 for i in range(5)}

    # std-Sharpe per fold needs >=2 fires for sample std; fold 0 with constant
    # returns gives NaN; folds 1-4 have 0 fires. Result should be NaN.
    mean_s, std_s, n_fired = paper_sharpe_per_fold(
        scores, returns, fold_ids, fold_spans_days, tau=0.5, fee=0.0
    )
    assert n_fired == 50
    # All folds either NaN (fold 0, constant returns) or no fires (1-4) => NaN mean
    assert np.isnan(mean_s)


def test_paper_sharpe_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        paper_sharpe_per_fold(
            np.array([0.9, 0.5]),
            np.array([0.01]),
            np.array([0, 1]),
            {0: 30.0, 1: 30.0},
            tau=0.5,
            fee=0.0,
        )


def test_paper_sharpe_missing_fold_span():
    """fold_ids referring to unknown fold should fail loud."""
    with pytest.raises(KeyError):
        paper_sharpe_per_fold(
            np.array([0.9, 0.9]),
            np.array([0.01, 0.02]),
            np.array([0, 7]),  # fold 7 not in spans dict
            {0: 30.0},
            tau=0.5,
            fee=0.0,
        )
