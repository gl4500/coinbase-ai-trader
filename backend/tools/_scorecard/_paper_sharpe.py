"""Annualized paper-trade Sharpe, computed per-fold then aggregated.

Per O3 resolution in design spec: each fold gets its own per-signal Sharpe and
annualization factor sqrt(N_f) where N_f = signals/year in fold f. Aggregate
across folds as mean +/- std for an honest variance estimate.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def paper_sharpe_per_fold(
    scores: np.ndarray,
    returns: np.ndarray,
    fold_ids: np.ndarray,
    fold_spans_days: Mapping[int, float],
    tau: float,
    fee: float,
) -> tuple[float, float, int]:
    """Per-fold annualized Sharpe, then mean and std across folds.

    Args:
        scores: shape (N,).
        returns: shape (N,) realized log-returns.
        fold_ids: shape (N,) integer fold index per sample.
        fold_spans_days: fold_id -> span of that fold in days.
        tau: strict-greater-than firing threshold.
        fee: per-side fee; round-trip cost = 2*fee subtracted from each return.

    Returns:
        (mean_annual_sharpe, std_annual_sharpe, total_n_fired).
        mean/std are NaN if no fold has >=2 fires (sample std undefined).

    Raises:
        ValueError: on shape mismatch.
        KeyError: if a fold_id in the data isn't in fold_spans_days.
    """
    if not (scores.shape == returns.shape == fold_ids.shape):
        raise ValueError(
            f"shape mismatch: scores {scores.shape}, returns {returns.shape}, fold_ids {fold_ids.shape}"
        )

    fired = scores > tau
    total_n_fired = int(fired.sum())

    fold_sharpes: list[float] = []
    unique_folds = np.unique(fold_ids[fired]) if total_n_fired > 0 else np.array([], dtype=int)

    for f in unique_folds:
        f_int = int(f)
        span_days = fold_spans_days[f_int]  # KeyError surfaces loudly
        mask = fired & (fold_ids == f)
        f_returns = returns[mask] - 2 * fee
        n_f = int(mask.sum())
        if n_f < 2:
            continue  # sample std undefined with <2 obs
        mu = float(f_returns.mean())
        sigma = float(f_returns.std(ddof=1))
        if sigma == 0:
            continue  # degenerate fold, skip
        per_signal_sharpe = mu / sigma
        n_per_year = n_f * 365.0 / span_days
        annualized = per_signal_sharpe * np.sqrt(n_per_year)
        fold_sharpes.append(annualized)

    if len(fold_sharpes) == 0:
        return float("nan"), float("nan"), total_n_fired
    if len(fold_sharpes) == 1:
        return float(fold_sharpes[0]), float("nan"), total_n_fired
    arr = np.array(fold_sharpes)
    return float(arr.mean()), float(arr.std(ddof=1)), total_n_fired
