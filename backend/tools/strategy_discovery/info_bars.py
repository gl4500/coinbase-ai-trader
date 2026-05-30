"""Aggregate a 1h OHLCV DataFrame into matched-count dollar bars.

A dollar bar closes when the cumulative dollar value (volume x (H+L+C)/3) of
consecutive 1h rows crosses a threshold equal to total_dollar_value / n_1h_rows.
This makes the emitted bar count approximately equal to the source 1h-bar count,
holding sample size fixed and isolating the sampling clock as the only variable
that changes vs the 1h baseline.

Mirrors the accumulation contract of `tools.build_dollar_bars.dollar_bars_from_candles`,
fed 1h rows instead of 1m candles. Pure function, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OUT_COLUMNS = (
    "start", "end", "open", "high", "low", "close",
    "volume", "dollar_value", "n_1h",
)


def aggregate_dollar_bars(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time-ordered 1h OHLCV rows into matched-count dollar bars.

    Input: DataFrame with columns ``start`` (epoch seconds), ``open``, ``high``,
    ``low``, ``close``, ``volume``. Rows are assumed time-sorted.

    Output: DataFrame with columns ``start`` (first merged row's epoch s),
    ``end`` (closing merged row's epoch s), ``open`` (first merged row's open),
    ``high`` / ``low`` (max / min over merged rows), ``close`` (last merged row's
    close), ``volume`` / ``dollar_value`` (sums), ``n_1h`` (merged row count).

    The trailing sub-threshold residual is dropped. Returns an empty frame with
    the full schema when input is empty or total dollar value is non-positive.
    """
    if len(df_1h) == 0:
        return pd.DataFrame({c: [] for c in _OUT_COLUMNS})

    typical = (df_1h["high"] + df_1h["low"] + df_1h["close"]) / 3.0
    dv = (df_1h["volume"] * typical).to_numpy(dtype="float64")
    total = float(dv.sum())
    n_rows = len(df_1h)
    if total <= 0.0:
        return pd.DataFrame({c: [] for c in _OUT_COLUMNS})

    threshold = total / n_rows

    starts = df_1h["start"].to_numpy(dtype="int64")
    opens  = df_1h["open"].to_numpy(dtype="float64")
    highs  = df_1h["high"].to_numpy(dtype="float64")
    lows   = df_1h["low"].to_numpy(dtype="float64")
    closes = df_1h["close"].to_numpy(dtype="float64")
    vols   = df_1h["volume"].to_numpy(dtype="float64")

    bars: list[dict] = []
    acc_dv = 0.0
    acc_vol = 0.0
    bar_start = None
    bar_open = None
    bar_high = None
    bar_low = None
    n = 0

    for i in range(n_rows):
        if bar_start is None:
            bar_start = int(starts[i])
            bar_open = float(opens[i])
            bar_high = float(highs[i])
            bar_low = float(lows[i])
        else:
            if highs[i] > bar_high:
                bar_high = float(highs[i])
            if lows[i] < bar_low:
                bar_low = float(lows[i])
        acc_dv += float(dv[i])
        acc_vol += float(vols[i])
        n += 1

        if acc_dv >= threshold:
            bars.append({
                "start":        bar_start,
                "end":          int(starts[i]),
                "open":         bar_open,
                "high":         bar_high,
                "low":          bar_low,
                "close":        float(closes[i]),
                "volume":       acc_vol,
                "dollar_value": acc_dv,
                "n_1h":         n,
            })
            acc_dv = 0.0
            acc_vol = 0.0
            bar_start = None
            bar_open = None
            bar_high = None
            bar_low = None
            n = 0

    return pd.DataFrame(bars, columns=list(_OUT_COLUMNS))
