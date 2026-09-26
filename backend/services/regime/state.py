"""RegimeState value object + macro-regime formula constants.

All constants are config-tunable; the Phase-1 backtest is the judge of their
values (see docs/superpowers/specs/2026-07-05-macro-regime-layer-design.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# (mvrv, prior) anchors; linear interpolation between, flat outside the ends.
MVRV_ANCHORS: List[Tuple[float, float]] = [(0.8, 1.25), (1.5, 1.05), (3.0, 0.95), (3.5, 0.85)]
MACRO_K: float = 0.3
EXPOSURE_CLAMP: Tuple[float, float] = (0.4, 1.25)
REGIME_STALE_DAYS: int = 3


@dataclass
class RegimeState:
    date: str
    mvrv: Optional[float]
    mvrv_prior: float
    corr_spx_90d: Optional[float]
    macro_risk_raw: Optional[float]
    macro_mult: float
    exposure_scalar: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "date": self.date,
            "mvrv": self.mvrv,
            "mvrv_prior": self.mvrv_prior,
            "corr_spx_90d": self.corr_spx_90d,
            "macro_risk_raw": self.macro_risk_raw,
            "macro_mult": self.macro_mult,
            "exposure_scalar": self.exposure_scalar,
            "confidence": self.confidence,
            "components": json.dumps(self.components),
        }

    @classmethod
    def from_row(cls, row: dict) -> "RegimeState":
        comp = row.get("components")
        return cls(
            date=row["date"],
            mvrv=row.get("mvrv"),
            mvrv_prior=row["mvrv_prior"],
            corr_spx_90d=row.get("corr_spx_90d"),
            macro_risk_raw=row.get("macro_risk_raw"),
            macro_mult=row["macro_mult"],
            exposure_scalar=row["exposure_scalar"],
            confidence=row.get("confidence", 0.0),
            components=json.loads(comp) if isinstance(comp, str) else (comp or {}),
        )
