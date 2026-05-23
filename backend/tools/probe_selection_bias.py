"""Selection-bias meta-analysis of the XGB feature-search probe history.

Deflates the recorded probe results for the size of the search: given N trials
and a per-trial noise scale, computes the best result a true-null search would
still produce, and a Deflated-Sharpe-style probability that the observed edge
exceeds that floor. See 2026-05-22-probe-selection-bias-design.md.
"""
from __future__ import annotations

import math
from statistics import NormalDist

_NORM = NormalDist()  # standard normal
_EULER_GAMMA = 0.5772156649015329


def iid_auc_se(n_pos: int, n_neg: int) -> float:
    """SE of an AUC under H0 (true AUC = 0.5), treating samples as iid.

    The Mann-Whitney null variance: (n_pos + n_neg + 1) / (12 * n_pos * n_neg).
    Optimistic for overlapping financial labels — reported only for contrast.

    Raises:
        ValueError: if either count is not positive.
    """
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError(f"counts must be positive, got n_pos={n_pos} n_neg={n_neg}")
    return math.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))


def fold_level_se(fold_aucs: list[float]) -> float:
    """Empirical SE from a list of per-fold AUCs — the sample standard deviation.

    The honest noise unit for non-iid overlapping labels.

    Raises:
        ValueError: if fewer than 2 folds are given.
    """
    n = len(fold_aucs)
    if n < 2:
        raise ValueError(f"need >= 2 folds for an empirical SE, got {n}")
    mean = sum(fold_aucs) / n
    var = sum((a - mean) ** 2 for a in fold_aucs) / (n - 1)
    return math.sqrt(var)
