"""CoinPaprika historical marketcap fetcher (#293) — free-tier sibling of
services/coingecko_marketcap (#260) that does NOT require an API key.

Why a second source: CoinGecko's free tier began returning 401 on its
historical market_chart/range endpoint (probe #260d/e/f, 2026-05-09).
CoinPaprika offers the same shape of data (one daily row per coin with a
market_cap field) at 25,000 requests/day with no key.

Public API mirrors coingecko_marketcap so the probe harness can swap providers
without changing downstream code:

    await fetch_marketcap_history(pid: str, start_ms: int, end_ms: int)
        -> list[(ts_ms, market_cap)] sorted ascending

    _coinbase_to_cp_id(pid)   -> Optional[str]    ("BTC-USD" -> "btc-bitcoin")

Endpoint (free tier — `coins/{id}/ohlcv/historical` is paywalled, this one
isn't):
    GET https://api.coinpaprika.com/v1/tickers/{cp_id}/historical
        ?start=YYYY-MM-DD&interval=1d
    Returns one row per UTC day with: timestamp, price, volume_24h, market_cap.
    Free tier window is rolling ~12 months; older starts return HTTP 402.

Strict causality: same EOD-UTC stamping as CoinGecko, so the same 1-day lag
applies at align time. This module returns raw rows; alignment lives in the
probe / parquet writer.

Kill switch: env COINPAPRIKA_DISABLED=1 short-circuits without an HTTP call.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.coinpaprika.com/v1"
_HISTORY_URL = f"{_BASE}/tickers/{{cp_id}}/historical"
_TICKER_URL  = f"{_BASE}/tickers/{{cp_id}}"

_SCHEMA_VERSION = 1


# ── Coinbase product → CoinPaprika id mapping ───────────────────────────────
# CoinPaprika ids follow the pattern <ticker>-<slug>. Verified live 2026-05-10.
_PRODUCT_TO_CP_ID = {
    "BTC-USD":     "btc-bitcoin",
    "ETH-USD":     "eth-ethereum",
    "SOL-USD":     "sol-solana",
    "XRP-USD":     "xrp-xrp",
    "BNB-USD":     "bnb-binance-coin",
    "ADA-USD":     "ada-cardano",
    "AVAX-USD":    "avax-avalanche",
    "LINK-USD":    "link-chainlink",
    "DOT-USD":     "dot-polkadot",
    "DOGE-USD":    "doge-dogecoin",
    "ARB-USD":     "arb-arbitrum",
    "ALGO-USD":    "algo-algorand",
    "ONDO-USD":    "ondo-ondo",
    "FET-USD":     "fet-fetch-ai",
    "PEPE-USD":    "pepe-pepe",
    "BONK-USD":    "bonk-bonk1",
    "POPCAT-USD":  "popcat-popcat",
    "JTO-USD":     "jto-jito",
    "PENGU-USD":   "pengu-pudgy-penguins",
    "ZK-USD":      "zk-zksync",
    "TRU-USD":     "tru-truefi-token",
    "SKL-USD":     "skl-skale-network",
    "JASMY-USD":   "jasmy-jasmycoin",
    "NKN-USD":     "nkn-nkn",
    "AIOZ-USD":    "aioz-aioz-network",
    "MOODENG-USD": "moodeng-moo-deng",
    "LRDS-USD":    "lrds-lordlabs",
    "XCN-USD":     "xcn-onyxcoin",
}


def _coinbase_to_cp_id(product_id: Optional[str]) -> Optional[str]:
    """Coinbase product_id -> CoinPaprika coin id, or None if unmapped."""
    if not product_id:
        return None
    return _PRODUCT_TO_CP_ID.get(product_id)


def _is_disabled() -> bool:
    return os.environ.get("COINPAPRIKA_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _ms_to_iso_date(ms: int) -> str:
    """Epoch ms -> YYYY-MM-DD UTC."""
    d = _dt.datetime.fromtimestamp(int(ms) / 1000, tz=_dt.timezone.utc)
    return d.strftime("%Y-%m-%d")


def _iso_to_ms(iso: str) -> Optional[int]:
    """CoinPaprika time_open like '2025-01-01T00:00:00Z' -> epoch ms.
    Returns None if unparseable."""
    if not iso:
        return None
    s = iso.replace("Z", "+00:00")
    try:
        return int(_dt.datetime.fromisoformat(s).timestamp() * 1000)
    except ValueError:
        return None


async def fetch_marketcap_history(
    product_id: str, start_ms: int, end_ms: int
) -> List[Tuple[int, float, float]]:
    """Daily marketcap timeseries for one Coinbase pid.

    Returns list of (ts_ms, market_cap, volume_24h) sorted ascending. Skips
    rows with a missing or non-positive market_cap. Empty list when:
      - pid is unmapped
      - COINPAPRIKA_DISABLED=1
      - HTTP non-200 or transport error
      - response shape unexpected

    Step A (2026-05-16): return tuple shape extended to carry volume_24h
    parsed from the response's `volume_24h` field. Missing -> 0.0 fill.
    """
    if _is_disabled():
        return []

    cp_id = _coinbase_to_cp_id(product_id)
    if cp_id is None:
        return []

    params = {
        "start":    _ms_to_iso_date(start_ms),
        "interval": "1d",
    }
    url = _HISTORY_URL.format(cp_id=cp_id)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url, params=params)
    except Exception as e:
        logger.warning(
            "coinpaprika_marketcap history HTTP error pid=%s: %r", product_id, e
        )
        return []

    if resp.status_code != 200:
        logger.warning(
            "coinpaprika_marketcap history non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return []

    try:
        body = resp.json()
    except Exception:
        return []

    if not isinstance(body, list):
        return []

    rows: List[Tuple[int, float, float]] = []
    for entry in body:
        if not isinstance(entry, dict):
            continue
        mc = entry.get("market_cap")
        if mc is None:
            continue
        try:
            mc_f = float(mc)
        except (TypeError, ValueError):
            continue
        if mc_f <= 0.0:
            continue
        ts_ms = _iso_to_ms(entry.get("timestamp"))
        if ts_ms is None:
            continue
        try:
            vol_f = float(entry.get("volume_24h") or 0.0)
        except (TypeError, ValueError):
            vol_f = 0.0
        rows.append((ts_ms, mc_f, vol_f))
    rows.sort(key=lambda r: r[0])
    return rows


async def fetch_supply_snapshot(
    product_id: str,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """Current-ticker supply snapshot for one Coinbase pid.

    Returns (circulating_supply, total_supply, max_supply_or_None) or None on
    any failure (unmapped pid, disabled, non-200, malformed body, missing
    circulating/total).

    max_supply is None for tokens with no fixed cap (e.g. ETH) — the
    distinction matters because it changes how FDV is interpreted downstream.

    Endpoint: GET /v1/tickers/{cp_id}  (no key, free tier).
    """
    if _is_disabled():
        return None

    cp_id = _coinbase_to_cp_id(product_id)
    if cp_id is None:
        return None

    url = _TICKER_URL.format(cp_id=cp_id)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
    except Exception as e:
        logger.warning(
            "coinpaprika_marketcap ticker HTTP error pid=%s: %r", product_id, e
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "coinpaprika_marketcap ticker non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return None

    try:
        body = resp.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    try:
        circ  = float(body["circulating_supply"])
        total = float(body["total_supply"])
    except (KeyError, TypeError, ValueError):
        return None

    max_raw = body.get("max_supply")
    max_supply: Optional[float]
    if max_raw is None:
        max_supply = None
    else:
        try:
            max_supply = float(max_raw)
        except (TypeError, ValueError):
            max_supply = None

    return (circ, total, max_supply)
