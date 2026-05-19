"""Pure helper: realized log-returns per sample from entry and forward closes.

Consumer: the deployment-aligned scorecard (tools/scorecard.py, Task 3+).
"""
from __future__ import annotations

import numpy as np


def realized_log_returns_per_sample(
    entry_closes: np.ndarray,
    forward_closes: np.ndarray,
) -> np.ndarray:
    """Return ln(forward_close / entry_close) elementwise.

    Args:
        entry_closes: shape (N,), close price at sample entry bar.
        forward_closes: shape (N,), close price at the bar the triple-barrier
            resolved on (TP hit, SL hit, or timeout).

    Returns:
        shape (N,) log-returns.

    Raises:
        ValueError: if shapes mismatch or any price is <= 0.
    """
    if entry_closes.shape != forward_closes.shape:
        raise ValueError(
            f"shape mismatch: entry {entry_closes.shape} vs forward {forward_closes.shape}"
        )
    if (entry_closes <= 0).any() or (forward_closes <= 0).any():
        raise ValueError("non-positive price found in entry_closes or forward_closes")
    return np.log(forward_closes / entry_closes)
