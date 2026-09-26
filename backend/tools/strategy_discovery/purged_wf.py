"""Expanding chronological walk-forward splits with a purged label horizon.

Only earlier rows may train a fold. Callers must supply chronologically ordered
rows at least one bar apart; removing rows makes the positional gap conservative.
This is deliberately not purged k-fold or combinatorial cross-validation.
"""

from __future__ import annotations

from numbers import Integral
from typing import List, Tuple

import numpy as np


def outer_folds(
    n_rows: int, n_folds: int = 5, embargo_bars: int = 168
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return up to n_folds expanding-prefix train/test pairs.

    Reserve one block plus division remainder for initial training. Remaining
    test blocks have equal size. Purge embargo_bars immediately before each
    test; omit folds without training history. Never borrow future rows.
    """
    for name, value, minimum in (
        ("n_rows", n_rows, 0),
        ("n_folds", n_folds, 1),
        ("embargo_bars", embargo_bars, 0),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    size = n_rows // (n_folds + 1)
    if size == 0:
        return []
    warmup = n_rows - n_folds * size
    out = []
    for test_start in range(warmup, n_rows, size):
        train_end = test_start - embargo_bars
        if train_end <= 0:
            continue
        out.append(
            (
                np.arange(train_end, dtype=np.int64),
                np.arange(test_start, test_start + size, dtype=np.int64),
            )
        )
    return out


def inner_folds(
    train_idx: np.ndarray, n_folds: int = 3, embargo_bars: int = 168
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Split ordered outer-training indices; the positional gap is conservative."""
    train_idx = np.asarray(train_idx)
    if (
        train_idx.ndim != 1
        or not np.issubdtype(train_idx.dtype, np.integer)
        or np.any(train_idx < 0)
        or np.any(train_idx[1:] <= train_idx[:-1])
    ):
        raise ValueError("train_idx must be strictly increasing nonnegative integer indices")
    return [
        (train_idx[train], train_idx[test])
        for train, test in outer_folds(len(train_idx), n_folds, embargo_bars)
    ]
