"""Tier-aware hourly candle fetcher for XGB feature_set v3.

Synchronous by design — see spec docs/superpowers/specs/2026-05-16-xgb-mixed-lookback-design.md
section 6.1. Lives outside the async aiosqlite stack so xgb_signal.xgb_prob
(sync) can call it without bubbling await through _cnn_prob.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Literal, Optional

import pandas as pd

_TIER_WINDOWS: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}

_DEFAULT_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PARQUET_DIR = os.path.join(_DEFAULT_BACKEND_DIR, "data", "history")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_BACKEND_DIR, "coinbase.db")


def _slice_tail(candles: List[Dict], n: int) -> List[Dict]:
    """Return last-n in ascending order, or [] if fewer than n available."""
    if len(candles) < n:
        return []
    return candles[-n:]


def _candle_dict(row) -> Dict:
    return {
        "start":  float(row["start"]),
        "open":   float(row["open"]),
        "high":   float(row["high"]),
        "low":    float(row["low"]),
        "close":  float(row["close"]),
        "volume": float(row["volume"]),
    }


def _read_parquet(pid: str, parquet_dir: str, now_ts: Optional[float]) -> List[Dict]:
    path = os.path.join(parquet_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    df = df.sort_values("start", kind="mergesort")
    if now_ts is not None:
        df = df[df["start"] < now_ts]
    return [_candle_dict(r) for _, r in df.iterrows()]


def _read_sqlite(pid: str, db_path: str, now_ts: Optional[float],
                 limit: int = 400) -> List[Dict]:
    if not os.path.exists(db_path):
        return []
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT start, open, high, low, close, volume FROM candles "
            "WHERE product_id = ?"
        )
        args: list = [pid]
        if now_ts is not None:
            sql += " AND start < ?"
            args.append(now_ts)
        sql += " ORDER BY start ASC"
        rows = c.execute(sql, args).fetchall()
    finally:
        c.close()
    rows = rows[-limit:] if limit and len(rows) > limit else rows
    return [_candle_dict(r) for r in rows]


def fetch_tiered(
    pid: str,
    source: Literal["parquet", "live"] = "live",
    now_ts: Optional[float] = None,
    parquet_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Return {"micro", "meso", "macro"} candle slices for `pid`.

    Each tier slice is the LAST N bars (n=60/168/336) in ascending order.
    Returns [] for any tier whose underlying series has fewer than N bars.
    """
    if source not in ("parquet", "live"):
        raise ValueError(f"unknown source={source!r}; expected 'parquet' or 'live'")

    pdir = parquet_dir or _DEFAULT_PARQUET_DIR
    dpath = db_path or _DEFAULT_DB_PATH

    if source == "parquet":
        all_candles = _read_parquet(pid, pdir, now_ts)
    else:  # live
        all_candles = _read_sqlite(pid, dpath, now_ts, limit=max(_TIER_WINDOWS.values()))
        if len(all_candles) < _TIER_WINDOWS["macro"]:
            parquet_prefix = _read_parquet(pid, pdir, now_ts)
            if parquet_prefix:
                seen = {c["start"] for c in all_candles}
                merged = parquet_prefix + [c for c in all_candles if c["start"] not in seen]
                merged.sort(key=lambda c: c["start"])
                all_candles = merged[-_TIER_WINDOWS["macro"]:]

    return {
        "micro": _slice_tail(all_candles, _TIER_WINDOWS["micro"]),
        "meso":  _slice_tail(all_candles, _TIER_WINDOWS["meso"]),
        "macro": _slice_tail(all_candles, _TIER_WINDOWS["macro"]),
    }
