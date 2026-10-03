"""Drawdown, complete-week returns and a PAIRED moving-block bootstrap (exploratory, not a gate)."""

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series, initial: float) -> float:
    path = np.r_[initial, equity.to_numpy(dtype=float)]
    peak = np.maximum.accumulate(path)
    return float((1 - path / peak).max())


def _sundays(equity: pd.Series) -> pd.Series:
    return equity[equity.index.weekday == 6]


def weekly_returns(equity: pd.Series) -> pd.Series:
    """Sunday close to Sunday close: complete Mon-Sun weeks only."""
    return _sundays(equity).pct_change().dropna()


def boundary_returns(equity: pd.Series, initial: float) -> dict:
    s = _sundays(equity)
    if s.empty:
        return {"head": float(equity.iloc[-1] / initial - 1), "tail": None}
    return {"head": float(s.iloc[0] / initial - 1), "tail": float(equity.iloc[-1] / s.iloc[-1] - 1)}


def paired_block_ci(a: pd.Series, b: pd.Series, block: int, n: int, seed: int) -> dict:
    if not a.index.equals(b.index):
        raise ValueError("paired series must be aligned (identical ordered indexes)")
    d = a.to_numpy(dtype=float) - b.to_numpy(dtype=float)
    if not np.isfinite(d).all():
        raise ValueError("paired series must be finite")
    m = len(d)
    if m < block:
        return {"mean_excess": float(d.mean()) if m else None, "lo": None, "hi": None, "n_weeks": m}
    rng = np.random.default_rng(seed)
    k = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :m]
    means = d[idx].mean(axis=1)
    return {
        "mean_excess": float(d.mean()),
        "lo": float(np.percentile(means, 2.5)),
        "hi": float(np.percentile(means, 97.5)),
        "n_weeks": m,
    }
