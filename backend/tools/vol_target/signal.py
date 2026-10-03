"""Weekly volatility-target signal. Pure: no I/O, no clock."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tools.vol_target import prereg as P


def annualised_sigma(close: pd.Series) -> pd.Series:
    """`close` must be on a full daily calendar (missing days are NaN rows). A window with any
    missing, non-finite or non-positive close yields NaN: no filled or compressed windows."""
    c = close.astype(float).where(close > 0)
    r = np.log(c).diff()
    n = P.VOL_RETURNS
    return r.rolling(n, min_periods=n).std(ddof=1) * math.sqrt(P.ANNUALISATION)


def target_weight(sigma: pd.Series) -> pd.Series:
    s = sigma.astype(float)
    w = pd.Series(np.nan, index=s.index)
    finite = np.isfinite(s)
    w[finite & (s == 0)] = 1.0
    pos = finite & (s > 0)
    w[pos] = np.minimum(1.0, P.SIGMA_TARGET / s[pos])
    return w


def schedule(close: pd.Series, start: str, end: str, delay: int) -> pd.DataFrame:
    """One row per scheduled weekly decision whose execution day (Sunday + 1 + delay) lies in
    [start, end] and whose Sunday is >= start - 1 day. Uses only history up to that Sunday."""
    w = target_weight(annualised_sigma(close))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for d in close.index[close.index.weekday == P.DECISION_WEEKDAY]:
        ex = d + pd.Timedelta(days=1 + delay)
        if d >= lo - pd.Timedelta(days=1) and lo <= ex <= hi:
            rows.append((d, ex, float(w.loc[d]), float(close.loc[d])))
    out = pd.DataFrame(rows, columns=["decision", "execute", "target", "decision_close"])
    return out.set_index("decision")
