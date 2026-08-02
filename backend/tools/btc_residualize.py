"""Pure-array helpers for BTC β-residualization (#253).

Why these helpers exist:
    Empirical history showed every prior BTC-based feature added BTC's move
    INTO the channel — Ch 21 btc_corr_20 (#149 dropped as anti-signal),
    BTC-dominance (#156), and BTC lead-lag at 5 horizons (#246-#248) all
    failed the +0.01 AUC gate. The diagnosis: those features encode the
    market-wide common factor, which the model already gets from price
    motion itself, so the channel adds variance without adding edge.

    β-residualization inverts that. Decompose:
        r_alt[t] = β[t] · r_btc[t] + ε[t]   →   ε[t] = r_alt[t] - β[t] · r_btc[t]
    The residual ε is the alt-specific component AFTER stripping the
    BTC-correlated part. If that's where the alpha lives, replacing Ch 9
    (1-bar price change) with ε is the test.

Causality:
    β[t] is computed on the strictly-prior window [t-W, t-1]. It does NOT
    see r_alt[t] or r_btc[t]. This matches the per-bar causality discipline
    that #157 imposed on Ch 15 ADX.

Timeframe-agnostic:
    Helpers operate on raw return arrays — no timestamps, no dicts. The
    same code runs on 1m, 5m, hourly. Probe (#253c) calls with 5m returns
    and W=288 (24h).
"""

from __future__ import annotations

import numpy as np


def _validate(alt_returns: np.ndarray, btc_returns: np.ndarray, window: int) -> None:
    if alt_returns.shape != btc_returns.shape:
        raise ValueError(
            f"alt_returns and btc_returns must have the same shape; "
            f"got {alt_returns.shape} vs {btc_returns.shape}"
        )
    if window <= 0:
        raise ValueError(f"window must be > 0; got {window}")


def compute_rolling_beta(
    alt_returns: np.ndarray,
    btc_returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """β[t] = Cov(alt, btc) / Var(btc) on [t-window, t-1] strictly.

    Returns an array shape == input. First `window` entries are NaN
    (warm-up — strict causality means t cannot use t itself, so t < window
    has no fully-prior window). Bars where `Var(btc) ≈ 0` fall back to 0.0
    so downstream consumers always see finite values.
    """
    _validate(alt_returns, btc_returns, window)
    n = alt_returns.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= window:
        return out
    a64 = np.asarray(alt_returns, dtype=np.float64)
    b64 = np.asarray(btc_returns, dtype=np.float64)
    for t in range(window, n):
        a_win = a64[t - window : t]
        b_win = b64[t - window : t]
        b_mean = b_win.mean()
        b_dev = b_win - b_mean
        var_b = float(np.dot(b_dev, b_dev) / window)
        if var_b < 1e-20:
            out[t] = 0.0
            continue
        a_dev = a_win - a_win.mean()
        cov = float(np.dot(a_dev, b_dev) / window)
        out[t] = cov / var_b
    return out


def residualize_returns(
    alt_returns: np.ndarray,
    btc_returns: np.ndarray,
    window: int,
) -> np.ndarray:
    """ε[t] = alt_returns[t] - β[t] · btc_returns[t]; β[t] from [t-W, t-1].

    Shape preserved. First `window` entries are NaN. When the prior window
    has zero btc variance, β=0 and ε[t] = alt_returns[t] (residual is the
    raw alt return — the most informative fallback).
    """
    _validate(alt_returns, btc_returns, window)
    beta = compute_rolling_beta(alt_returns, btc_returns, window)
    a64 = np.asarray(alt_returns, dtype=np.float64)
    b64 = np.asarray(btc_returns, dtype=np.float64)
    eps = a64 - beta * b64
    return eps
