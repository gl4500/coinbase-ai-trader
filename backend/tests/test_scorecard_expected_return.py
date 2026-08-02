import numpy as np
import pytest

from tools._scorecard._expected_return import expected_return_at_tau


def test_expected_return_basic_with_fee():
    """Fired samples return mean(r) - 2*fee."""
    scores = np.array([0.9, 0.7, 0.4])
    returns = np.array([0.02, 0.01, 0.05])  # 0.05 doesn't fire
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.5, fee=0.006)
    expected = np.mean([0.02, 0.01]) - 2 * 0.006
    assert n_fired == 2
    assert er == pytest.approx(expected)


def test_expected_return_no_fires_returns_nan():
    scores = np.array([0.1, 0.2])
    returns = np.array([0.05, 0.05])
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.9, fee=0.006)
    assert n_fired == 0
    assert np.isnan(er)


def test_expected_return_fee_zero():
    """fee=0 gives raw mean return on fired samples."""
    scores = np.array([0.9, 0.8])
    returns = np.array([0.03, -0.01])
    er, n_fired = expected_return_at_tau(scores, returns, tau=0.5, fee=0.0)
    assert n_fired == 2
    assert er == pytest.approx(0.01)


def test_expected_return_rejects_negative_fee():
    with pytest.raises(ValueError, match="non-negative"):
        expected_return_at_tau(np.array([0.9]), np.array([0.02]), tau=0.5, fee=-0.001)


def test_expected_return_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        expected_return_at_tau(np.array([0.9, 0.5]), np.array([0.02]), tau=0.5, fee=0.0)
