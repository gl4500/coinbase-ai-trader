"""The frozen rule: long iff completed close > SMA(n) including that close; equality is flat."""

import numpy as np
import pandas as pd


def decisions(close: pd.Series, n: int) -> pd.Series:
    sma = close.rolling(n, min_periods=n).mean()
    out = pd.Series(np.where(close > sma, 1.0, 0.0), index=close.index)
    return out.where(sma.notna() & close.notna())


def desired_state(dec: pd.Series) -> pd.Series:
    """Hold the current state through NaN decisions; the slice starts FLAT.

    Callers slice `dec` to the period BEFORE calling this, so no pre-period state carries in.
    """
    return dec.ffill().fillna(0.0).astype(bool)
