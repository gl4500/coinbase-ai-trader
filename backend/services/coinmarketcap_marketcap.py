"""
CoinMarketCap snapshot marketcap + FDV fetcher (#280).

Sibling of services/coingecko_marketcap (#260). Mirrors the same shape so
both providers can be A/B-compared via the marketcap probe harness.

Free-tier limitation:
    CoinMarketCap historical endpoints (`/v1/cryptocurrency/quotes/historical`,
    `/v2/cryptocurrency/ohlcv/historical`) require Hobbyist plan or higher
    ($29/mo+). On the Basic free tier only snapshot endpoints are available,
    so `fetch_marketcap_history` is a deliberate no-op that returns []. Probes
    that need history must continue using `services.coingecko_marketcap`.

Public API (mirrors services.coingecko_marketcap):
    await fetch_marketcap_snapshot(pids: Iterable[str])
        -> dict[pid -> MarketcapRow]
    await fetch_marketcap_history(pid, start_ms, end_ms)
        -> []   (free-tier no-op; warns once)

Endpoint:
    GET /v1/cryptocurrency/quotes/latest?id=1,1027,...&convert=USD
        Auth: header X-CMC_PRO_API_KEY: <key>
        Response: { data: { "<id>": { id, symbol, slug,
                                       circulating_supply, total_supply,
                                       max_supply,
                                       quote: { USD: { market_cap,
                                                       fully_diluted_market_cap,
                                                       ... }}}}}

Auth source: env COINMARKETCAP_API_KEY.
Kill switch: env COINMARKETCAP_DISABLED=1 short-circuits without an HTTP call.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://pro-api.coinmarketcap.com"
_QUOTES_LATEST_URL = f"{_BASE}/v1/cryptocurrency/quotes/latest"

_SCHEMA_VERSION = 1

# Suppress repeated free-tier warnings to avoid log spam during probe runs.
_history_warned = False


# ── Coinbase product → CoinMarketCap numeric id mapping ────────────────────
# CMC IDs are numeric and stable (unlike CoinGecko slugs which can rename).
# Verified live on CoinMarketCap 2026-05-09 via /v1/cryptocurrency/map. If a
# new pid is added, look up the id at https://coinmarketcap.com/currencies/<slug>/
# (the URL bar doesn't show id; use the API map endpoint or the CMC docs).
_PRODUCT_TO_CMC_ID: Dict[str, int] = {
    "BTC-USD":     1,
    "ETH-USD":     1027,
    "SOL-USD":     5426,
    "XRP-USD":     52,
    "BNB-USD":     1839,
    "ADA-USD":     2010,
    "AVAX-USD":    5805,
    "LINK-USD":    1975,
    "DOT-USD":     6636,
    "DOGE-USD":    74,
    "ARB-USD":     11841,
    "ALGO-USD":    4030,
    "ONDO-USD":    21159,
    "FET-USD":     3773,
    "PEPE-USD":    24478,
    "BONK-USD":    23095,
    "POPCAT-USD":  28782,
    "JTO-USD":     29210,
    "PENGU-USD":   34466,
    "ZK-USD":      24091,
    "TRU-USD":     7725,
    "SKL-USD":     5691,
    "JASMY-USD":   8425,
    "NKN-USD":     2780,
    "AIOZ-USD":    9265,
    "MOODENG-USD": 33093,
    "XCN-USD":     10474,
}


def _coinbase_to_cmc_id(product_id: Optional[str]) -> Optional[int]:
    """Coinbase product_id -> CoinMarketCap numeric id, or None if unmapped."""
    if not product_id:
        return None
    return _PRODUCT_TO_CMC_ID.get(product_id)


def _is_disabled() -> bool:
    """True when COINMARKETCAP_DISABLED env is set to a truthy value."""
    return os.environ.get("COINMARKETCAP_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _api_key() -> str:
    """Returns CMC API key from env, or empty string when unset."""
    return os.environ.get("COINMARKETCAP_API_KEY", "").strip()


@dataclass
class MarketcapRow:
    """Identical shape to services.coingecko_marketcap.MarketcapRow so the
    probe harness can swap providers without changing downstream code."""
    market_cap: float
    fdv: float
    circ_supply: float
    total_supply: float
    ingest_ts: int           # PIT column per #164b
    schema_version: int      # PIT column per #164b


# ── Current snapshot ────────────────────────────────────────────────────────

async def fetch_marketcap_snapshot(pids: Iterable[str]) -> Dict[str, MarketcapRow]:
    """One-shot snapshot of marketcap + FDV for the given Coinbase pids.

    Returns a dict mapping pid -> MarketcapRow. Pids without a CMC id, pids
    missing from the response, or any of these conditions return an empty
    dict (no exception):
      - COINMARKETCAP_DISABLED=1
      - COINMARKETCAP_API_KEY unset
      - HTTP transport error
      - non-200 response (401 bad key, 429 rate-limit, etc.)
      - JSON decode failure or unexpected shape
    """
    if _is_disabled():
        return {}

    api_key = _api_key()
    if not api_key:
        logger.warning(
            "coinmarketcap_marketcap snapshot: COINMARKETCAP_API_KEY not set; "
            "skipping (set in .env to enable)."
        )
        return {}

    pid_list = list(pids)
    cmc_ids = [(pid, _coinbase_to_cmc_id(pid)) for pid in pid_list]
    cmc_ids = [(pid, cid) for pid, cid in cmc_ids if cid is not None]
    if not cmc_ids:
        return {}

    params = {
        "id":      ",".join(str(cid) for _, cid in cmc_ids),
        "convert": "USD",
    }
    headers = {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept":            "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                _QUOTES_LATEST_URL, params=params, headers=headers
            )
    except Exception as e:
        logger.warning("coinmarketcap_marketcap snapshot HTTP error: %r", e)
        return {}

    if resp.status_code != 200:
        logger.warning(
            "coinmarketcap_marketcap snapshot non-200: status=%d",
            resp.status_code,
        )
        return {}

    try:
        body = resp.json()
    except Exception as e:
        logger.warning("coinmarketcap_marketcap snapshot json decode failed: %r", e)
        return {}

    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        return {}

    cmc_to_pid = {cid: pid for pid, cid in cmc_ids}
    now_ts = int(time.time())
    out: Dict[str, MarketcapRow] = {}
    for cid_key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        try:
            cid_int = int(cid_key)
        except (TypeError, ValueError):
            continue
        pid = cmc_to_pid.get(cid_int)
        if pid is None:
            continue
        usd = (entry.get("quote") or {}).get("USD") or {}
        market_cap = usd.get("market_cap")
        if market_cap is None:
            continue
        fdv = usd.get("fully_diluted_market_cap")
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


# ── Historical timeseries (free-tier no-op) ─────────────────────────────────

async def fetch_marketcap_history(
    product_id: str, start_ms: int, end_ms: int
) -> List[Tuple[int, float]]:
    """Free-tier no-op. CMC historical endpoints (`quotes/historical`,
    `ohlcv/historical`) require Hobbyist plan ($29/mo) or higher. We
    deliberately return [] without making the HTTP call so the probe harness
    can fall back to services.coingecko_marketcap cleanly.

    Emits one warning per process to surface the limitation without spamming
    the log during probe runs.
    """
    global _history_warned
    if _is_disabled():
        return []
    if not _history_warned:
        logger.warning(
            "coinmarketcap_marketcap.fetch_marketcap_history: free-tier no-op. "
            "CMC historical endpoints require Hobbyist plan ($29/mo) or higher. "
            "Falling back to services.coingecko_marketcap for history."
        )
        _history_warned = True
    return []
