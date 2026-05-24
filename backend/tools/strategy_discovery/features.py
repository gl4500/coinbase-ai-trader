"""Trend feature compute for the strategy-discovery rebuild (Phase 2).

Adds 7 trend columns to a 1h OHLCV DataFrame:
  - price_over_ema20 / 50 / 200 (close / EMA, scale-free ratio)
  - ret_1h_sign / ret_24h_sign / ret_7d_sign (numpy.sign of close-vs-past-close)
  - atr14_pct (Wilder ATR-14 divided by close)

Pure functions on pandas DataFrames. No I/O. No tokenomics. No labels.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

_TREND_COLUMNS: Tuple[str, ...] = (
    "price_over_ema20",
    "price_over_ema50",
    "price_over_ema200",
    "ret_1h_sign",
    "ret_24h_sign",
    "ret_7d_sign",
    "atr14_pct",
)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_trend_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add the 7 trend feature columns to a 1h OHLCV DataFrame.

    Requires columns: ts, open, high, low, close. Returns a copy with 7 added
    columns. Rows inside the warm-up region (< 200 bars from the start) will
    have NaN in EMA200-dependent ratios; use first_valid_index() to skip them.
    """
    out = df_ohlcv.copy()
    close = out["close"]
    out["price_over_ema20"] = close / _ema(close, 20)
    return out


def first_valid_index(df: pd.DataFrame, min_warmup: int = 200) -> int:
    """First row index where all trend feature columns are finite."""
    cols = [c for c in _TREND_COLUMNS if c in df.columns]
    if not cols:
        return min(min_warmup, len(df))
    finite = df[cols].notna().all(axis=1) & np.isfinite(df[cols]).all(axis=1)
    n = len(df)
    for i in range(min_warmup, n):
        if bool(finite.iloc[i]):
            return i
    return n
