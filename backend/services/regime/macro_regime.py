"""RegimeEvaluator — the macro-regime exposure formula (pure, no I/O).

exposure_scalar = clip(mvrv_prior * macro_mult, *EXPOSURE_CLAMP), applied to
ENTRY sizing only. Any missing/stale input degrades its factor to 1.0; all
missing -> 1.0 (neutral). Never raises. See the design spec for rationale.
"""
from __future__ import annotations

import math
from typing import Optional

from services.regime.state import (
    RegimeState, MVRV_ANCHORS, MACRO_K, EXPOSURE_CLAMP, REGIME_STALE_DAYS,
)


def _finite(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return x if math.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def mvrv_prior(mvrv: Optional[float]) -> float:
    mvrv = _finite(mvrv)
    if mvrv is None:
        return 1.0
    lo_m, lo_p = MVRV_ANCHORS[0]
    if mvrv <= lo_m:
        return lo_p
    hi_m, hi_p = MVRV_ANCHORS[-1]
    if mvrv >= hi_m:
        return hi_p
    for (m0, p0), (m1, p1) in zip(MVRV_ANCHORS, MVRV_ANCHORS[1:]):
        if m0 <= mvrv <= m1:
            frac = (mvrv - m0) / (m1 - m0)
            return p0 + frac * (p1 - p0)
    return 1.0


def macro_mult(corr_spx_90d: Optional[float], macro_risk_raw: Optional[float],
               k: float = MACRO_K) -> float:
    corr = _finite(corr_spx_90d)
    risk = _finite(macro_risk_raw)
    if corr is None or risk is None:
        return 1.0
    w = max(0.0, corr)
    return 1.0 + k * w * risk


def evaluate(*, date: str, mvrv: Optional[float], corr_spx_90d: Optional[float],
             macro_risk_raw: Optional[float], mvrv_age_days: int = 0,
             macro_age_days: int = 0) -> RegimeState:
    # Staleness: too-old inputs are treated as missing.
    mvrv_in = _finite(mvrv) if mvrv_age_days <= REGIME_STALE_DAYS else None
    if macro_age_days <= REGIME_STALE_DAYS:
        corr_in, risk_in = _finite(corr_spx_90d), _finite(macro_risk_raw)
    else:
        corr_in, risk_in = None, None

    prior = mvrv_prior(mvrv_in)
    mult = macro_mult(corr_in, risk_in)
    lo, hi = EXPOSURE_CLAMP
    scalar = max(lo, min(hi, prior * mult))

    mvrv_fresh = 1.0 if mvrv_in is not None else 0.0
    macro_fresh = 1.0 if (corr_in is not None and risk_in is not None) else 0.0
    confidence = (mvrv_fresh + macro_fresh) / 2.0

    return RegimeState(
        date=date, mvrv=mvrv_in, mvrv_prior=prior, corr_spx_90d=corr_in,
        macro_risk_raw=risk_in, macro_mult=mult, exposure_scalar=scalar,
        confidence=confidence,
        components={"mvrv_prior": prior, "macro_mult": mult},
    )
