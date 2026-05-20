"""Expected calibration error with equal-width bins on [0, 1]."""
from __future__ import annotations

import numpy as np


def expected_calibration_error(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute ECE: weighted mean of |empirical_acc - mean_score| per bin.

    Bins are equal-width on [0, 1]. Per O4 resolution in design spec, decile
    binning is safe at 167k+ samples; revisit BBQ-style adaptive binning only
    for smaller subsets.

    Args:
        scores: shape (N,) in [0, 1].
        labels: shape (N,) binary 0/1.
        n_bins: number of equal-width bins.

    Returns:
        ECE in [0, 1]; 0 = perfect calibration.

    Raises:
        ValueError: on shape mismatch, non-binary labels, or n_bins <= 0.
    """
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: scores {scores.shape} vs labels {labels.shape}")
    uniq = np.unique(labels)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError(f"labels must be binary 0/1, got {uniq}")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be > 0, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges, right=False) - 1, 0, n_bins - 1)

    n = scores.shape[0]
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        acc_b = float(labels[mask].mean())
        conf_b = float(scores[mask].mean())
        ece += (n_b / n) * abs(acc_b - conf_b)
    return ece
