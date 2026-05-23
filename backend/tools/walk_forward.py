"""Purged walk-forward CV splitter with embargo.

Phase 2 of the CNN -> XGBoost transition. Replaces the broken 80/20 cut at
agents/cnn_agent.py:2565 (Phase 0 finding: concatenates products without
global time-sorting, ignores _WALKFORWARD_EMBARGO).

Critical for both CNN and XGBoost training paths: any train sample whose
forward window overlaps the val region peeks at the same future. The
embargo drops those samples so AUC reports reflect honest generalization.
"""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


def purged_walk_forward_splits(
    timestamps: np.ndarray,
    n_folds: int,
    embargo_hours: int = 4,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) tuples for K-fold time-ordered CV.

    Indices reference rows in `timestamps` as supplied (unsorted is fine).
    Val regions partition the dataset chronologically. Train indices for
    fold k are all rows whose timestamp falls strictly outside
    [val_min - embargo, val_max + embargo].

    Raises:
        ValueError: if n_folds < 2 or fewer than 2*n_folds samples.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    timestamps = np.asarray(timestamps, dtype=np.int64)
    n = len(timestamps)
    if n < n_folds * 2:
        raise ValueError(
            f"need at least {n_folds * 2} samples for {n_folds} folds, got {n}"
        )

    order = np.argsort(timestamps, kind="stable")
    ts_sorted = timestamps[order]

    fold_sizes = np.full(n_folds, n // n_folds, dtype=np.int64)
    fold_sizes[: n % n_folds] += 1
    boundaries = np.concatenate([[0], np.cumsum(fold_sizes)])

    embargo_secs = int(embargo_hours) * 3600

    for k in range(n_folds):
        v_lo, v_hi = int(boundaries[k]), int(boundaries[k + 1])
        val_idx = order[v_lo:v_hi]
        val_min = int(ts_sorted[v_lo])
        val_max = int(ts_sorted[v_hi - 1])

        keep = (timestamps < val_min - embargo_secs) | (
            timestamps > val_max + embargo_secs
        )
        train_idx = np.where(keep)[0]

        yield train_idx, val_idx
