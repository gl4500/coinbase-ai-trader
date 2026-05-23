"""
CoinGecko historical marketcap + FDV fetcher (#260).

Sibling of services/okx_oi_history (#86 / #88) and services/okx_long_short_history
(#235). Adds marketcap / fully-diluted-valuation as a candidate exogenous input
after seven sequential BTC-flavored probes failed the +0.01 mean-AUC gate (see
project memory xgb_feature_optimization_findings.md).

Public API (mirrors funding/OI fetchers):
    await fetch_marketcap_snapshot(pids: list[str])
        -> dict[pid -> MarketcapRow]
    await fetch_marketcap_history(pid: str, start_ms: int, end_ms: int)
        -> list[(ts_ms, market_cap)] sorted ascending
    align_to_hourly(rows, hour_grid_ms, lag_secs=86400)
        -> dict[ts_ms -> value]   (forward-filled with strict-causal lag)

CoinGecko endpoints (free tier, no key required):
  - GET /api/v3/coins/markets?vs_currency=usd&ids=bitcoin,ethereum,...
      one row per coin, current snapshot of market_cap, fully_diluted_valuation,
      circulating_supply, total_supply.
  - GET /api/v3/coins/{id}/market_chart/range?vs_currency=usd&from=&to=
      three parallel timeseries (prices, market_caps, total_volumes). Granularity
      auto-selected by range: <2d -> 5m, 2-90d -> hourly, >90d -> daily.

Strict causality: align_to_hourly applies a default 1-day lag because CoinGecko
marks daily marketcap at end-of-day UTC; using same-day data at sample t would
leak future close into a feature read at t.

Kill switch: env COINGECKO_DISABLED=1 short-circuits without an HTTP call.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.coingecko.com/api/v3"
_MARKETS_URL = f"{_BASE}/coins/markets"
_HISTORY_URL = f"{_BASE}/coins/{{cg_id}}/market_chart/range"

_SCHEMA_VERSION = 1


# ── Coinbase product → CoinGecko id mapping ─────────────────────────────────
# Slugs verified live on CoinGecko 2026-05-09. CoinGecko occasionally renames
# slugs; if a probe fails for a single pid, re-check the slug at
# https://www.coingecko.com/en/coins/<slug>.
_PRODUCT_TO_CG_ID: Dict[str, str] = {
    "BTC-USD":     "bitcoin",
    "ETH-USD":     "ethereum",
    "SOL-USD":     "solana",
    "XRP-USD":     "ripple",
    "BNB-USD":     "binancecoin",
    "ADA-USD":     "cardano",
    "AVAX-USD":    "avalanche-2",
    "LINK-USD":    "chainlink",
    "DOT-USD":     "polkadot",
    "DOGE-USD":    "dogecoin",
    "ARB-USD":     "arbitrum",
    "ALGO-USD":    "algorand",
    "ONDO-USD":    "ondo-finance",
    "FET-USD":     "fetch-ai",
    "PEPE-USD":    "pepe",
    "BONK-USD":    "bonk",
    "POPCAT-USD":  "popcat",
    "JTO-USD":     "jito-governance-token",
    "PENGU-USD":   "pudgy-penguins",
    "ZK-USD":      "zksync",
    "TRU-USD":     "truefi",
    "SKL-USD":     "skale",
    "JASMY-USD":   "jasmycoin",
    "NKN-USD":     "nkn",
    "AIOZ-USD":    "aioz-network",
    "MOODENG-USD": "moo-deng",
    "LRDS-USD":    "lordlabs",
    "XCN-USD":     "onyxcoin",
}


def _coinbase_to_cg_id(product_id: str) -> Optional[str]:
    """Coinbase product_id -> CoinGecko coin slug, or None if unmapped."""
    if not product_id:
        return None
    return _PRODUCT_TO_CG_ID.get(product_id)


def _is_disabled() -> bool:
    """True when COINGECKO_DISABLED env is set to a truthy value."""
    return os.environ.get("COINGECKO_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _demo_key_headers() -> Dict[str, str]:
    """Demo-tier auth header dict, or {} when COINGECKO_API_KEY is unset.

    The free CoinGecko Demo plan (10k req/month, includes historical
    market_chart) authenticates with the `x-cg-demo-api-key` header. Pro plan
    keys use a different header (`x-cg-pro-api-key`); set COINGECKO_API_PRO=1
    to opt into that variant.
    """
    key = os.environ.get("COINGECKO_API_KEY", "").strip()
    if not key:
        return {}
    if os.environ.get("COINGECKO_API_PRO", "").strip().lower() in {"1", "true", "yes", "on"}:
        return {"x-cg-pro-api-key": key}
    return {"x-cg-demo-api-key": key}


@dataclass
class MarketcapRow:
    market_cap: float
    fdv: float
    circ_supply: float
    total_supply: float
    ingest_ts: int           # PIT column per #164b
    schema_version: int      # PIT column per #164b


# ── Current snapshot ────────────────────────────────────────────────────────

async def fetch_marketcap_snapshot(pids: Iterable[str]) -> Dict[str, MarketcapRow]:
    """One-shot snapshot of marketcap + FDV for the given Coinbase pids.

    Returns a dict mapping pid -> MarketcapRow. Pids without a CoinGecko id
    or pids missing from the response are silently omitted. Returns an empty
    dict when COINGECKO_DISABLED=1 or on any HTTP failure.
    """
    if _is_disabled():
        return {}

    pid_list = list(pids)
    cg_ids = [(pid, _coinbase_to_cg_id(pid)) for pid in pid_list]
    cg_ids = [(pid, cid) for pid, cid in cg_ids if cid]
    if not cg_ids:
        return {}

    params = {
        "vs_currency": "usd",
        "ids": ",".join(cid for _, cid in cg_ids),
        "per_page": str(len(cg_ids)),
        "page": "1",
        "sparkline": "false",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                _MARKETS_URL, params=params, headers=_demo_key_headers()
            )
    except Exception as e:
        logger.warning("coingecko_marketcap snapshot HTTP error: %r", e)
        return {}

    if resp.status_code != 200:
        logger.warning(
            "coingecko_marketcap snapshot non-200: status=%d", resp.status_code
        )
        return {}

    try:
        body = resp.json()
    except Exception as e:
        logger.warning("coingecko_marketcap snapshot json decode failed: %r", e)
        return {}

    if not isinstance(body, list):
        return {}

    cg_to_pid = {cid: pid for pid, cid in cg_ids}
    now_ts = int(time.time())
    out: Dict[str, MarketcapRow] = {}
    for entry in body:
        cid = entry.get("id")
        pid = cg_to_pid.get(cid)
        if pid is None:
            continue
        market_cap = entry.get("market_cap")
        fdv = entry.get("fully_diluted_valuation")
        if market_cap is None:
            continue
        if fdv is None:
            fdv = market_cap
        circ = entry.get("circulating_supply") or 0.0
        total = entry.get("total_supply") or 0.0
        out[pid] = MarketcapRow(
            market_cap=float(market_cap),
            fdv=float(fdv),
            circ_supply=float(circ),
            total_supply=float(total),
            ingest_ts=now_ts,
            schema_version=_SCHEMA_VERSION,
        )
    return out


# ── Historical timeseries ───────────────────────────────────────────────────

async def fetch_marketcap_history(
    product_id: str, start_ms: int, end_ms: int
) -> List[Tuple[int, float, float]]:
    """Historical marketcap timeseries for a single Coinbase pid.

    Returns list of (ts_ms, market_cap, volume_24h) sorted ascending. Empty
    list when:
      - pid is unmapped
      - COINGECKO_DISABLED=1
      - HTTP non-200 or transport error
      - response shape unexpected

    Step A (2026-05-16): return tuple shape extended to carry volume_24h
    parsed from the response's `total_volumes` parallel array. Missing or
    empty total_volumes -> warning + 0.0 fill.
    """
    if _is_disabled():
        return []

    cg_id = _coinbase_to_cg_id(product_id)
    if cg_id is None:
        return []

    url = _HISTORY_URL.format(cg_id=cg_id)
    params = {
        "vs_currency": "usd",
        "from": str(int(start_ms / 1000)),
        "to":   str(int(end_ms / 1000)),
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(
                url, params=params, headers=_demo_key_headers()
            )
    except Exception as e:
        logger.warning(
            "coingecko_marketcap history HTTP error pid=%s: %r", product_id, e
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            "coingecko_marketcap history non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return []

    try:
        body = resp.json()
    except Exception:
        return []

    if not isinstance(body, dict):
        return []
    mc_raw  = body.get("market_caps")
    vol_raw = body.get("total_volumes")
    if not isinstance(mc_raw, list) or not mc_raw:
        return []

    if not isinstance(vol_raw, list) or not vol_raw:
        logger.warning(
            "coingecko_marketcap: total_volumes missing/empty pid=%s — volume_24h=0.0",
            product_id,
        )
        vol_raw = []

    # Index volume by timestamp for safe lookup (CG arrays usually align but
    # we don't depend on equal lengths).
    vol_by_ts: Dict[int, float] = {}
    for entry in vol_raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            vol_by_ts[int(entry[0])] = float(entry[1])
        except (TypeError, ValueError):
            continue

    rows: List[Tuple[int, float, float]] = []
    for entry in mc_raw:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        try:
            ts = int(entry[0])
            mc = float(entry[1])
        except (TypeError, ValueError):
            continue
        vol = vol_by_ts.get(ts, 0.0)
        rows.append((ts, mc, vol))
    rows.sort(key=lambda r: r[0])
    return rows


# ── Forward-fill alignment with strict causality ────────────────────────────

def align_to_hourly(
    rows: List[Tuple[int, float]],
    hour_grid_ms: Iterable[int],
    lag_secs: int = 86400,
) -> Dict[int, float]:
    """Forward-fill `rows` onto `hour_grid_ms` with strict-causal lag.

    For each grid timestamp `t`, output uses the latest row with
    `row.ts <= t - lag_secs*1000`. Grid points where no qualifying row
    exists are omitted from the output (caller treats them as missing).

    Default lag is 1 day to compensate for CoinGecko's end-of-day UTC
    daily marketcap stamping.
    """
    if not rows:
        return {}
    sorted_rows = sorted(rows, key=lambda r: r[0])
    lag_ms = int(lag_secs) * 1000
    out: Dict[int, float] = {}
    n = len(sorted_rows)
    i = 0
    last_val: Optional[float] = None
    grid = sorted(set(int(t) for t in hour_grid_ms))
    for ts in grid:
        cutoff = ts - lag_ms
        while i < n and sorted_rows[i][0] <= cutoff:
            last_val = sorted_rows[i][1]
            i += 1
        if last_val is not None:
            out[ts] = last_val
    return out
