"""Expected log-return per fired signal, net of round-trip fee."""
from __future__ import annotations

import numpy as np


def expected_return_at_tau(
    scores: np.ndarray,
    returns: np.ndarray,
    tau: float,
    fee: float,
) -> tuple[float, int]:
    """Mean realized log-return on samples with score > tau, minus 2 * fee.

    Args:
        scores: shape (N,).
        returns: shape (N,) realized log-returns per sample.
        tau: strict-greater-than threshold.
        fee: per-side fee (e.g., 0.006 for 0.6%). Round-trip cost = 2 * fee.

    Returns:
        (expected_return, n_fired). NaN if no signals fire.

    Raises:
        ValueError: on shape mismatch or negative fee.
    """
    if scores.shape != returns.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs returns {returns.shape}")
    if fee < 0:
        raise ValueError(f"fee must be non-negative, got {fee}")

    fired = scores > tau
    n_fired = int(fired.sum())
    if n_fired == 0:
        return float("nan"), 0
    return float(returns[fired].mean() - 2 * fee), n_fired
