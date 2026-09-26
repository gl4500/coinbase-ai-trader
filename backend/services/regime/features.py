"""Derive the macro-regime scalar features from daily series.

corr_spx_90d: BTC-SPX daily-log-return Pearson over the last 90 pairs.
macro_risk_raw: equity trend + dollar direction + real-yield direction,
each standardized to ~[-1,1], averaged over the available sub-signals.
Callers pass already-date-aligned recent series (oldest..newest).
"""

from __future__ import annotations

import math
from typing import List, Optional


def _log_returns(closes: List[float]) -> List[float]:
    out = []
    for a, b in zip(closes, closes[1:], strict=False):  # pairwise slide: lengths differ by 1
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def corr_spx_90d(
    btc_closes: List[float], spx_closes: List[float], window: int = 90
) -> Optional[float]:
    n = min(len(btc_closes), len(spx_closes))
    if n < 31:
        return None
    rb = _log_returns(btc_closes[-(window + 1) :])
    rs = _log_returns(spx_closes[-(window + 1) :])
    m = min(len(rb), len(rs))
    if m < 30:
        return None
    rb, rs = rb[-m:], rs[-m:]
    mb, ms = sum(rb) / m, sum(rs) / m
    num = sum((a - mb) * (b - ms) for a, b in zip(rb, rs, strict=True))
    db = math.sqrt(sum((a - mb) ** 2 for a in rb))
    ds = math.sqrt(sum((b - ms) ** 2 for b in rs))
    if db == 0 or ds == 0:
        return None
    return max(-1.0, min(1.0, num / (db * ds)))


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def macro_risk_raw(
    spx_closes: List[float], dxy_closes: List[float], real_yield: List[float]
) -> Optional[float]:
    subs: List[float] = []
    # Equity trend: % of last close above/below its 50d MA, +-5% -> +-1.
    if len(spx_closes) >= 50 and spx_closes[-1] > 0:
        sma = sum(spx_closes[-50:]) / 50.0
        if sma > 0:
            subs.append(_clip((spx_closes[-1] / sma - 1.0) / 0.05))
    # Dollar direction: DXY 20d change; FALLING dollar = risk-on (+). +-2% -> +-1.
    if len(dxy_closes) >= 21 and dxy_closes[-21] > 0:
        subs.append(_clip(-(dxy_closes[-1] / dxy_closes[-21] - 1.0) / 0.02))
    # Real-yield direction: 20d change in pct-points; RISING yield = risk-off (-).
    if len(real_yield) >= 21:
        subs.append(_clip(-(real_yield[-1] - real_yield[-21]) / 0.25))
    if not subs:
        return None
    return _clip(sum(subs) / len(subs))
