"""Per-field merge of two MarketcapRow objects (CG + CMC) into FundamentalsRow.

Pure function. Per-field rules:
  circulating_supply, total_supply, price, price_chg_24h_pct → median
  volume_24h                                                 → average
  market_cap                                                 → price * circ
  fdv                                                        → price * total
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Literal

from services.coingecko_marketcap import MarketcapRow
from services.leader_ranker import FundamentalsRow

MergeQuality = Literal["BOTH", "CG_ONLY", "CMC_ONLY"]
_NUMERIC_FIELDS = ("price", "circ_supply", "total_supply", "volume_24h", "price_chg_24h_pct")


@dataclass(frozen=True)
class MergeOutcome:
    row: FundamentalsRow
    merge_quality: MergeQuality
    cg_market_cap: float | None
    cmc_market_cap: float | None
    cg_circ_supply: float | None
    cmc_circ_supply: float | None
    cg_total_supply: float | None
    cmc_total_supply: float | None
    cg_volume_24h: float | None
    cmc_volume_24h: float | None
    audit_entries: list[dict[str, Any]] = field(default_factory=list)


def _delta_pct(a: float, b: float) -> float:
    m = median([a, b])
    return abs(a - b) / abs(m) if m != 0 else 0.0


def merge_marketcap_rows(
    pid: str,
    cg: MarketcapRow | None,
    cmc: MarketcapRow | None,
    *,
    divergence_threshold: float,
) -> MergeOutcome:
    if cg is None and cmc is None:
        raise ValueError(f"{pid}: both providers missing")

    audit: list[dict[str, Any]] = []
    flags: list[str] = []

    if cg is not None and cmc is not None:
        quality: MergeQuality = "BOTH"
        for fname in _NUMERIC_FIELDS:
            cgv = getattr(cg, fname)
            cmcv = getattr(cmc, fname)
            d = _delta_pct(cgv, cmcv)
            flagged = d > divergence_threshold
            if flagged:
                flags.append(fname)
            audit.append({"field": fname, "cg": cgv, "cmc": cmcv, "delta_pct": d, "flagged": flagged})
        price = float(median([cg.price, cmc.price]))
        circ = float(median([cg.circ_supply, cmc.circ_supply]))
        total = float(median([cg.total_supply, cmc.total_supply]))
        chg = float(median([cg.price_chg_24h_pct, cmc.price_chg_24h_pct]))
        vol = (cg.volume_24h + cmc.volume_24h) / 2.0
    elif cg is not None:
        quality = "CG_ONLY"
        price, circ, total, chg, vol = (
            cg.price, cg.circ_supply, cg.total_supply, cg.price_chg_24h_pct, cg.volume_24h)
    else:
        assert cmc is not None
        quality = "CMC_ONLY"
        price, circ, total, chg, vol = (
            cmc.price, cmc.circ_supply, cmc.total_supply, cmc.price_chg_24h_pct, cmc.volume_24h)

    market_cap = price * circ
    fdv = price * total

    row = FundamentalsRow(
        product_id=pid, price=price, market_cap=market_cap, fdv=fdv,
        volume_24h=vol, circulating_supply=circ, total_supply=total,
        price_chg_24h_pct=chg, divergence_flags=list(flags),
    )

    return MergeOutcome(
        row=row, merge_quality=quality,
        cg_market_cap=cg.market_cap if cg else None,
        cmc_market_cap=cmc.market_cap if cmc else None,
        cg_circ_supply=cg.circ_supply if cg else None,
        cmc_circ_supply=cmc.circ_supply if cmc else None,
        cg_total_supply=cg.total_supply if cg else None,
        cmc_total_supply=cmc.total_supply if cmc else None,
        cg_volume_24h=cg.volume_24h if cg else None,
        cmc_volume_24h=cmc.volume_24h if cmc else None,
        audit_entries=audit,
    )
