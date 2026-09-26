import numpy as np
import pytest

from tools._scorecard._precision import precision_at_tau


def test_precision_at_tau_basic():
    """3 fire at tau=0.5, 2 of 3 are positive => precision = 2/3."""
    scores = np.array([0.9, 0.7, 0.6, 0.4, 0.3])
    labels = np.array([1, 1, 0, 0, 1])  # last positive doesn't fire
    p, n_fired = precision_at_tau(scores, labels, tau=0.5)
    assert n_fired == 3
    assert p == pytest.approx(2 / 3)


def test_precision_at_tau_no_fires_returns_nan():
    """No samples above threshold => precision undefined (NaN), n_fired=0."""
    scores = np.array([0.1, 0.2, 0.3])
    labels = np.array([1, 1, 0])
    p, n_fired = precision_at_tau(scores, labels, tau=0.9)
    assert n_fired == 0
    assert np.isnan(p)


def test_precision_at_tau_all_fire():
    """tau=0 => all fire => precision = pos_rate."""
    scores = np.array([0.1, 0.5, 0.9])
    labels = np.array([0, 1, 1])
    p, n_fired = precision_at_tau(scores, labels, tau=0.0)
    assert n_fired == 3
    assert p == pytest.approx(2 / 3)


def test_precision_at_tau_strict_gt():
    """Threshold is strict > tau, not >=. Sample at exactly tau does NOT fire."""
    scores = np.array([0.5, 0.5, 0.51])
    labels = np.array([1, 1, 1])
    p, n_fired = precision_at_tau(scores, labels, tau=0.5)
    assert n_fired == 1
    assert p == pytest.approx(1.0)


def test_precision_at_tau_rejects_nonbinary_labels():
    """Labels must be 0/1 — fail loud on other values."""
    with pytest.raises(ValueError, match="binary"):
        precision_at_tau(np.array([0.9]), np.array([2]), tau=0.5)


def test_precision_at_tau_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        precision_at_tau(np.array([0.9, 0.5]), np.array([1]), tau=0.5)
