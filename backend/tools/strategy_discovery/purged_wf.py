"""Purged Walk-Forward CV + nested inner CV for Phase 3 mining.

Pure numpy index math. No tensors, no trees, no I/O.

Embargo rule: any train row whose ts falls within [test_start - horizon, test_start)
is dropped from the train set to prevent label leakage (a train row's
label_h{horizon} could span into the test fold).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def outer_folds(
    n_rows: int,
    n_folds: int = 5,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Chronological k-fold WF with embargo. Returns [(train_idx, test_idx), ...].

    Fold k's test set = rows [k * size, (k+1) * size) (last fold absorbs remainder).
    Train set = all rows OUTSIDE [embargo_lo, test_end) where embargo_lo = test_start - embargo_bars.
    """
    base = n_rows // n_folds
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_start = k * base
        test_end   = (k + 1) * base if k < n_folds - 1 else n_rows
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        embargo_lo = max(0, test_start - embargo_bars)
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[embargo_lo:test_end] = False
        train_idx = np.where(train_mask)[0]
        out.append((train_idx, test_idx))
    return out


def inner_folds(
    train_idx: np.ndarray,
    n_folds: int = 3,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Nested CV on the outer-train subset.

    Embargo computed on positions WITHIN train_idx, not on global row ids.
    """
    n = len(train_idx)
    base = n // n_folds
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_start = k * base
        test_end   = (k + 1) * base if k < n_folds - 1 else n
        embargo_lo = max(0, test_start - embargo_bars)
        mask = np.ones(n, dtype=bool)
        mask[embargo_lo:test_end] = False
        inner_train = train_idx[mask]
        inner_test  = train_idx[test_start:test_end]
        out.append((inner_train, inner_test))
    return out
