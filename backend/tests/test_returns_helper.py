import numpy as np

from tools._returns import realized_log_returns_per_sample


def test_realized_log_returns_basic():
    """Given entry close and forward close, return ln(forward/entry)."""
    entry_closes = np.array([100.0, 100.0, 100.0])
    forward_closes = np.array([101.0, 99.0, 100.5])
    result = realized_log_returns_per_sample(entry_closes, forward_closes)
    expected = np.log(np.array([1.01, 0.99, 1.005]))
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_realized_log_returns_zero_when_equal():
    """ln(1) = 0 when forward equals entry."""
    e = np.array([50.0, 75.0])
    f = np.array([50.0, 75.0])
    result = realized_log_returns_per_sample(e, f)
    np.testing.assert_allclose(result, np.zeros(2), atol=1e-12)


def test_realized_log_returns_raises_on_nonpositive():
    """Non-positive prices indicate data corruption — fail loud."""
    import pytest

    with pytest.raises(ValueError, match="non-positive"):
        realized_log_returns_per_sample(np.array([100.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="non-positive"):
        realized_log_returns_per_sample(np.array([-1.0]), np.array([100.0]))


def test_realized_log_returns_shape_mismatch():
    """Mismatched array lengths should fail immediately."""
    import pytest

    with pytest.raises(ValueError, match="shape"):
        realized_log_returns_per_sample(np.array([100.0, 101.0]), np.array([102.0]))
