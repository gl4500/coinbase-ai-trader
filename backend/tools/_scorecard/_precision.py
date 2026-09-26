"""Precision at a score threshold: of signals that fire, what fraction win?"""

from __future__ import annotations

import numpy as np


def precision_at_tau(
    scores: np.ndarray,
    labels: np.ndarray,
    tau: float,
) -> tuple[float, int]:
    """Compute precision among samples with score > tau.

    Args:
        scores: shape (N,) model output probabilities.
        labels: shape (N,) binary 0/1 ground truth.
        tau: strict-greater-than threshold.

    Returns:
        (precision, n_fired). precision = NaN if n_fired == 0.

    Raises:
        ValueError: on shape mismatch or non-binary labels.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")
    uniq = np.unique(labels)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"labels must be binary 0/1, got {uniq}")

    fired = scores > tau
    n_fired = int(fired.sum())
    if n_fired == 0:
        return float("nan"), 0
    n_tp = int(labels[fired].sum())
    return n_tp / n_fired, n_fired
