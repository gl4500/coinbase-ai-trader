import numpy as np
import pytest
from tools._scorecard._ece import expected_calibration_error


def test_ece_perfectly_calibrated():
    """Scores match empirical hit rates per bin => ECE near 0."""
    rng = np.random.default_rng(0)
    n = 10000
    scores = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < scores).astype(int)
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece < 0.02


def test_ece_completely_miscalibrated():
    """Model says 0.9 but truth is 0.1 hit rate => ECE high."""
    n = 1000
    scores = np.full(n, 0.9)
    labels = np.zeros(n, dtype=int)
    labels[:100] = 1
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece == pytest.approx(0.8, abs=0.01)


def test_ece_empty_bins_skipped():
    """Bins with zero samples should not contribute or NaN out the result."""
    scores = np.array([0.05, 0.95, 0.05, 0.95])
    labels = np.array([0, 1, 0, 1])
    ece = expected_calibration_error(scores, labels, n_bins=10)
    assert ece == pytest.approx(0.0, abs=0.06)


def test_ece_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        expected_calibration_error(np.array([0.5, 0.5]), np.array([1]), n_bins=10)


def test_ece_nonbinary_labels():
    with pytest.raises(ValueError, match="binary"):
        expected_calibration_error(np.array([0.5]), np.array([2]), n_bins=10)


def test_ece_invalid_n_bins():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(np.array([0.5]), np.array([1]), n_bins=0)
