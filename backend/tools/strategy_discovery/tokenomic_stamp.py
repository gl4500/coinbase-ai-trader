"""T+1 tokenomic stamping for the strategy-discovery rebuild (Phase 2).

Aligns a daily CoinPaprika snapshot (one row per UTC day at 00:00:00 UTC)
onto an hourly OHLCV grid using the T+1 causality rule from the spec:
the daily row dated UTC-day D applies only to candidate hours in
UTC-day D+1 and later.

Adds 6 tokenomic columns to the input DataFrame:
  market_cap, fdv, fdv_over_mc, circ_over_total, vol_24h, vol_over_mc

Pure functions. No I/O. Imports only stdlib + pandas + numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_DAY_MS = 86_400_000

_TOKENOMIC_COLUMNS: Tuple[str, ...] = (
    "market_cap",
    "fdv",
    "fdv_over_mc",
    "circ_over_total",
    "vol_24h",
    "vol_over_mc",
)


@dataclass(frozen=True)
class SupplySnapshot:
    """Point-in-time supply reading from CoinPaprika /v1/tickers/{cp_id}."""

    pid: str
    circulating: float
    total: float
    max_supply: Optional[float]


def _stamp_daily_to_hourly(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    *,
    value_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Forward-merge daily values onto hourly rows with a +1 day shift.

    df_daily['ts'] is UTC-day midnight in ms. Each row applies to candidate
    hours starting (ts + 1 day) — the T+1 rule.
    """
    daily = df_daily[["ts", *value_cols]].copy()
    daily["ts"] = daily["ts"].astype("int64") + _DAY_MS
    daily = daily.sort_values("ts").reset_index(drop=True)
    hourly = df_hourly[["ts"]].sort_values("ts").reset_index(drop=True)
    return pd.merge_asof(hourly, daily, on="ts", direction="backward")


def stamp_tokenomic(
    df_hourly: pd.DataFrame,
    df_daily_marketcap: pd.DataFrame,
    supply_row: SupplySnapshot,
    drop_on_missing_volume: bool = True,
) -> pd.DataFrame:
    """Add 6 tokenomic columns to df_hourly via T+1 stamping.

    df_hourly must contain: ts (epoch ms), close.
    df_daily_marketcap must contain: ts (UTC-day midnight epoch ms),
                                     market_cap, volume_24h.
    Returns a copy with 6 added columns. When drop_on_missing_volume=True,
    rows whose stamped vol_24h is NaN are dropped (per spec missing-data
    policy: volume staleness is unsafe to forward-fill).
    """
    stamped = _stamp_daily_to_hourly(
        df_hourly[["ts"]],
        df_daily_marketcap.rename(columns={"volume_24h": "vol_24h"}),
        value_cols=("market_cap", "vol_24h"),
    )
    out = df_hourly.copy().sort_values("ts").reset_index(drop=True)
    out["market_cap"] = stamped["market_cap"].to_numpy()
    out["vol_24h"] = stamped["vol_24h"].to_numpy()
    out["fdv"] = out["close"].astype("float64") * float(supply_row.total)
    out["fdv_over_mc"] = out["fdv"] / np.maximum(out["market_cap"], 1e-12)
    out["circ_over_total"] = float(supply_row.circulating) / max(float(supply_row.total), 1e-12)
    out["vol_over_mc"] = out["vol_24h"] / np.maximum(out["market_cap"], 1e-12)
    if drop_on_missing_volume:
        out = out[out["vol_24h"].notna()].reset_index(drop=True)
    return out
