"""Pure-function ranker for the leader filter (#leader-filter).

Given a snapshot of merged fundamentals + a config, produces:
  - top-N pids ranked by volume-confirmed momentum × supply-quality multiplier
  - per-pid ScoreRow with exclusion reason (PASS / FLOOR_* / DIVERGENCE / RANKED_OUT)

No I/O. No side effects. Same input → same output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Reason = Literal["PASS", "FLOOR_MCAP", "FLOOR_VOL", "FLOOR_FDV", "DIVERGENCE", "RANKED_OUT"]


@dataclass(frozen=True)
class FundamentalsRow:
    product_id: str
    price: float
    market_cap: float
    fdv: float
    volume_24h: float
    circulating_supply: float
    total_supply: float
    price_chg_24h_pct: float
    divergence_flags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScoreRow:
    product_id: str
    raw_score: float
    multiplier: float
    final_score: float
    reason: Reason
    leader_rank: int | None
    fdv_mcap_ratio: float
    circ_total_ratio: float

    @property
    def was_leader(self) -> bool:
        return self.leader_rank is not None


@dataclass(frozen=True)
class RankerResult:
    top_pids: list[str]
    scored: dict[str, ScoreRow]
    snapshot_ts: datetime


@dataclass(frozen=True)
class LeaderRankerConfig:
    min_market_cap_usd: float
    min_volume_24h_usd: float
    max_fdv_mcap_ratio: float
    supply_divergence_threshold: float
    top_n: int


def _f(key: str, default: float) -> float:
    v = os.environ.get(key)
    return float(v) if v is not None else default


def _i(key: str, default: int) -> int:
    v = os.environ.get(key)
    return int(v) if v is not None else default


def load_config_from_env() -> LeaderRankerConfig:
    return LeaderRankerConfig(
        min_market_cap_usd=_f("LEADER_MIN_MARKET_CAP_USD", 50_000_000),
        min_volume_24h_usd=_f("LEADER_MIN_VOLUME_24H_USD", 5_000_000),
        max_fdv_mcap_ratio=_f("LEADER_MAX_FDV_MCAP_RATIO", 5.0),
        supply_divergence_threshold=_f("LEADER_SUPPLY_DIVERGENCE_THRESHOLD", 0.10),
        top_n=_i("LEADER_TOP_N", 3),
    )


def supply_multiplier(circ_over_total: float) -> float:
    if circ_over_total >= 0.90:
        return 1.00
    if circ_over_total >= 0.50:
        return 0.85
    if circ_over_total >= 0.20:
        return 0.70
    return 0.50


def _floor_reason(row: FundamentalsRow, cfg: LeaderRankerConfig) -> Reason | None:
    if row.market_cap < cfg.min_market_cap_usd:
        return "FLOOR_MCAP"
    if row.volume_24h < cfg.min_volume_24h_usd:
        return "FLOOR_VOL"
    if row.market_cap > 0 and (row.fdv / row.market_cap) > cfg.max_fdv_mcap_ratio:
        return "FLOOR_FDV"
    return None


def rank_leaders(
    snapshot: dict[str, FundamentalsRow],
    cfg: LeaderRankerConfig,
    *,
    snapshot_ts: datetime,
) -> RankerResult:
    scored: dict[str, ScoreRow] = {}
    candidates: list[tuple[str, float]] = []

    for pid, row in snapshot.items():
        fdv_mcap = (row.fdv / row.market_cap) if row.market_cap > 0 else float("inf")
        circ_total = (row.circulating_supply / row.total_supply) if row.total_supply > 0 else 0.0

        if row.divergence_flags:
            scored[pid] = ScoreRow(pid, 0.0, 0.0, 0.0, "DIVERGENCE", None, fdv_mcap, circ_total)
            continue

        floor = _floor_reason(row, cfg)
        if floor is not None:
            scored[pid] = ScoreRow(pid, 0.0, 0.0, 0.0, floor, None, fdv_mcap, circ_total)
            continue

        raw = abs(row.price_chg_24h_pct) * (row.volume_24h / row.market_cap)
        mult = supply_multiplier(circ_total)
        final = raw * mult

        scored[pid] = ScoreRow(pid, raw, mult, final, "PASS", None, fdv_mcap, circ_total)
        candidates.append((pid, final))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = candidates[: cfg.top_n]
    top_pids = [pid for pid, _ in top]

    finalized: dict[str, ScoreRow] = {}
    for pid, sr in scored.items():
        if sr.reason != "PASS":
            finalized[pid] = sr
            continue
        if pid in top_pids:
            rank = top_pids.index(pid) + 1
            finalized[pid] = ScoreRow(
                sr.product_id, sr.raw_score, sr.multiplier, sr.final_score,
                "PASS", rank, sr.fdv_mcap_ratio, sr.circ_total_ratio,
            )
        else:
            finalized[pid] = ScoreRow(
                sr.product_id, sr.raw_score, sr.multiplier, sr.final_score,
                "RANKED_OUT", None, sr.fdv_mcap_ratio, sr.circ_total_ratio,
            )

    return RankerResult(top_pids=top_pids, scored=finalized, snapshot_ts=snapshot_ts)
