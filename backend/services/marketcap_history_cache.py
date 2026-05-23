"""Parquet-backed bronze cache for CoinGecko marketcap history (#284).

Wraps `services.coingecko_marketcap.fetch_marketcap_history` so probe re-runs
hit a local parquet first and only fall back to the API for misses / stale
windows. Schema mirrors `tools/build_marketcap_parquet._SCHEMA` and carries PIT
columns (`ingest_ts`, `schema_version`) per #164b.

Cache hit when ALL hold:
  * <parquet_dir>/<pid>.parquet exists with >= 1 row.
  * Newest `ingest_ts` is within `refresh_secs` of wall-clock.
  * Cached rows cover [start_ms, end_ms] (newest cached `start * 1000` >=
    end_ms - 3600_000 — within one bar of the requested end).

Cache miss / stale -> call `fetch_marketcap_history`, merge with cached rows,
re-stamp `ingest_ts = int(time.time())`, write parquet, return merged rows
filtered to the requested [start_ms, end_ms] window as `(ts_ms, market_cap)`
tuples sorted ascending.
"""
from __future__ import annotations

import os
import time
from typing import List, Tuple

from services.coingecko_marketcap import fetch_marketcap_history
from tools.build_marketcap_parquet import (
    _load_marketcap_history,
    _save_marketcap_history,
    _SCHEMA_VERSION,
)

_BAR_SECS = 3600


def _parquet_path(parquet_dir: str, product_id: str) -> str:
    safe = product_id.replace("/", "_")
    return os.path.join(parquet_dir, f"{safe}.parquet")


def _is_fresh(rows: List[dict], now_ts: int, refresh_secs: int) -> bool:
    if not rows:
        return False
    newest = max(int(r.get("ingest_ts", 0)) for r in rows)
    return (now_ts - newest) < refresh_secs


def _covers_range(rows: List[dict], end_ms: int) -> bool:
    if not rows:
        return False
    newest_start_ms = max(int(r["start"]) for r in rows) * 1000
    return newest_start_ms >= (end_ms - _BAR_SECS * 1000)


def _schema_is_stale(rows: List[dict]) -> bool:
    """Returns True if any cached row has schema_version < _SCHEMA_VERSION.

    Triggers full refetch even if ingest_ts is fresh — v1 parquets lack the
    volume_24h column required by Step A's downstream consumers.
    """
    if not rows:
        return True
    return any(int(r.get("schema_version", 0)) < _SCHEMA_VERSION for r in rows)


async def fetch_marketcap_history_cached(
    product_id: str,
    start_ms: int,
    end_ms: int,
    parquet_dir: str,
    refresh_secs: int = 86400,
) -> List[Tuple[int, float, float]]:
    """Return marketcap history for `product_id` over [start_ms, end_ms].

    Hits parquet bronze cache before the upstream CoinGecko fetcher. On miss
    or stale window the fresh rows are merged into the parquet file with
    `ingest_ts = int(time.time())` and `schema_version = _SCHEMA_VERSION`.
    """
    path = _parquet_path(parquet_dir, product_id)
    cached = _load_marketcap_history(path)
    now_ts = int(time.time())

    use_cache = (
        cached
        and not _schema_is_stale(cached)
        and _is_fresh(cached, now_ts, refresh_secs)
        and _covers_range(cached, end_ms)
    )

    if not use_cache:
        fresh = await fetch_marketcap_history(product_id, start_ms, end_ms)
        merged_by_start: dict = {}
        for r in cached:
            merged_by_start[int(r["start"])] = {
                "start": int(r["start"]),
                "market_cap": float(r["market_cap"]),
                "fdv": float(r.get("fdv", r["market_cap"])),
                "volume_24h": float(r.get("volume_24h", 0.0)),
                "ingest_ts": int(r.get("ingest_ts", now_ts)),
                "schema_version": int(r.get("schema_version", _SCHEMA_VERSION)),
            }
        for ts_ms, mc, vol in fresh:
            start = (int(ts_ms) // 1000 // _BAR_SECS) * _BAR_SECS
            merged_by_start[start] = {
                "start": start,
                "market_cap": float(mc),
                "fdv": float(mc),
                "volume_24h": float(vol),
                "ingest_ts": now_ts,
                "schema_version": _SCHEMA_VERSION,
            }
        if merged_by_start:
            _save_marketcap_history(
                path, list(merged_by_start.values()), now_ts=now_ts
            )
            cached = sorted(merged_by_start.values(), key=lambda r: r["start"])

    out: List[Tuple[int, float, float]] = []
    for r in cached:
        ts_ms = int(r["start"]) * 1000
        if start_ms <= ts_ms <= end_ms:
            out.append((ts_ms, float(r["market_cap"]), float(r.get("volume_24h", 0.0))))
    out.sort(key=lambda x: x[0])
    return out
