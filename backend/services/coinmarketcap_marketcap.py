"""CoinMarketCap /cryptocurrency/quotes/latest fetcher (#leader-filter).

Mirror of services/coingecko_marketcap (#260) — same shape, same error
handling, same kill-switch convention. Returns `dict[pid -> MarketcapRow]`
using the shared MarketcapRow type from coingecko_marketcap.

Free Basic tier: requires COINMARKETCAP_API_KEY (sign up free at
coinmarketcap.com/api). 10k calls/mo cap is well within hourly fetch budget
for ~28 pids.

Kill switch: env COINMARKETCAP_DISABLED=1 short-circuits without HTTP call.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, Iterable, Mapping, Optional

import httpx

from services.coingecko_marketcap import MarketcapRow

logger = logging.getLogger(__name__)

_BASE = "https://pro-api.coinmarketcap.com/v1"
_QUOTES_URL = f"{_BASE}/cryptocurrency/quotes/latest"

_SCHEMA_VERSION = 1


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
    "JTO-USD":     28541,
    "PENGU-USD":   34466,
    "ZK-USD":      24091,
    "TRU-USD":     7725,
    "SKL-USD":     5691,
    "JASMY-USD":   8425,
    "NKN-USD":     2780,
    "AIOZ-USD":    9265,
    "MOODENG-USD": 33861,
    "LRDS-USD":    33970,
    "XCN-USD":     8546,
}


def _coinbase_to_cmc_id(product_id: str) -> Optional[int]:
    if not product_id:
        return None
    return _PRODUCT_TO_CMC_ID.get(product_id)


def _is_disabled() -> bool:
    return os.environ.get("COINMARKETCAP_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def parse_quotes_response(
    payload: dict, cmc_to_pid: Mapping[int, str],
) -> Dict[str, MarketcapRow]:
    data = payload.get("data") or {}
    out: Dict[str, MarketcapRow] = {}
    now_ts = int(time.time())
    for key, item in data.items():
        try:
            cmc_id = int(key)
        except (TypeError, ValueError):
            continue
        pid = cmc_to_pid.get(cmc_id)
        if pid is None:
            continue

        usd = ((item or {}).get("quote") or {}).get("USD") or {}

        total_supply = item.get("total_supply") or item.get("max_supply")
        if total_supply is None:
            continue

        price = float(usd.get("price") or 0.0)
        circ = float(item.get("circulating_supply") or 0.0)

        market_cap = float(usd.get("market_cap") or 0.0)
        if market_cap == 0 and circ > 0:
            market_cap = price * circ

        fdv_raw = usd.get("fully_diluted_valuation")
        if fdv_raw is None:
            fdv = price * float(total_supply)
        else:
            fdv = float(fdv_raw)

        out[pid] = MarketcapRow(
            market_cap=market_cap,
            fdv=fdv,
            circ_supply=circ,
            total_supply=float(total_supply),
            ingest_ts=now_ts,
            schema_version=_SCHEMA_VERSION,
            price=price,
            volume_24h=float(usd.get("volume_24h") or 0.0),
            price_chg_24h_pct=float(usd.get("percent_change_24h") or 0.0),
        )
    return out


async def fetch_marketcap_snapshot(pids: Iterable[str]) -> Dict[str, MarketcapRow]:
    """One-shot snapshot. Empty dict on missing key, kill switch, or HTTP failure."""
    if _is_disabled():
        return {}

    api_key = os.environ.get("COINMARKETCAP_API_KEY")
    if not api_key:
        logger.warning("coinmarketcap_marketcap: COINMARKETCAP_API_KEY not set")
        return {}

    pid_list = list(pids)
    pairs = [(pid, _coinbase_to_cmc_id(pid)) for pid in pid_list]
    pairs = [(pid, cid) for pid, cid in pairs if cid is not None]
    if not pairs:
        return {}

    cmc_to_pid = {cid: pid for pid, cid in pairs}
    params = {
        "id": ",".join(str(cid) for _, cid in pairs),
        "convert": "USD",
    }
    headers = {
        "X-CMC_PRO_API_KEY": api_key,
        "Accept": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(_QUOTES_URL, params=params, headers=headers)
    except Exception as e:
        logger.warning("coinmarketcap_marketcap snapshot HTTP error: %r", e)
        return {}

    if resp.status_code != 200:
        logger.warning(
            "coinmarketcap_marketcap snapshot non-200: status=%d", resp.status_code
        )
        return {}

    try:
        body = resp.json()
    except Exception as e:
        logger.warning("coinmarketcap_marketcap json decode failed: %r", e)
        return {}

    return parse_quotes_response(body, cmc_to_pid)
