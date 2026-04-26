"""
Binance Futures historical funding-rate fetcher.

Used by CNN training to populate Ch 20 (funding rate) with the actual rate
that was active at each candle bar's timestamp, instead of zero.

Public API:
    await fetch_funding_history(product_id, start_ms, end_ms)
        -> list[(funding_time_ms, funding_rate)] sorted ascending,
           empty list if symbol not supported on Binance futures or fetch fails.

Funding events occur every 8h. Binance returns up to 1000 records per call,
which spans ~333 days — sufficient for the typical training window.

Kill switch (#81): set env BINANCE_FUNDING_DISABLED=1 to short-circuit and
return [] without an HTTP call. fapi.binance is geo-blocked from the US
(HTTP 451), so the call always fails on the production host — skip it.
"""
import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def _is_disabled() -> bool:
    """True when BINANCE_FUNDING_DISABLED env is set to a truthy value."""
    return os.environ.get("BINANCE_FUNDING_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }

_PRODUCT_TO_BN = {
    "BTC-USD":   "BTCUSDT",
    "ETH-USD":   "ETHUSDT",
    "SOL-USD":   "SOLUSDT",
    "XRP-USD":   "XRPUSDT",
    "BNB-USD":   "BNBUSDT",
    "ADA-USD":   "ADAUSDT",
    "AVAX-USD":  "AVAXUSDT",
    "LINK-USD":  "LINKUSDT",
    "DOT-USD":   "DOTUSDT",
    "MATIC-USD": "MATICUSDT",
    "DOGE-USD":  "DOGEUSDT",
    "LTC-USD":   "LTCUSDT",
    "ATOM-USD":  "ATOMUSDT",
    "FIL-USD":   "FILUSDT",
    "NEAR-USD":  "NEARUSDT",
    "APT-USD":   "APTUSDT",
    "INJ-USD":   "INJUSDT",
    "ARB-USD":   "ARBUSDT",
    "OP-USD":    "OPUSDT",
    "TIA-USD":   "TIAUSDT",
    "SEI-USD":   "SEIUSDT",
    "SUI-USD":   "SUIUSDT",
    "RNDR-USD":  "RNDRUSDT",
    "FET-USD":   "FETUSDT",
    "AAVE-USD":  "AAVEUSDT",
    "UNI-USD":   "UNIUSDT",
}


def _coinbase_to_binance(product_id: str) -> Optional[str]:
    return _PRODUCT_TO_BN.get(product_id)


async def fetch_funding_history(
    product_id: str,
    start_ms: int,
    end_ms: int,
) -> List[Tuple[int, float]]:
    if _is_disabled():
        return []
    bn_sym = _coinbase_to_binance(product_id)
    if not bn_sym:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _URL,
                params={
                    "symbol":    bn_sym,
                    "startTime": int(start_ms),
                    "endTime":   int(end_ms),
                    "limit":     1000,
                },
            )
        if resp.status_code != 200:
            return []
        payload = resp.json()
        if not isinstance(payload, list):
            return []
        out: List[Tuple[int, float]] = []
        for row in payload:
            try:
                t = int(row["fundingTime"])
                r = float(row["fundingRate"])
                out.append((t, r))
            except (KeyError, TypeError, ValueError):
                continue
        out.sort(key=lambda tr: tr[0])
        return out
    except Exception as e:
        logger.debug("Funding history unavailable for %s: %s", product_id, e)
        return []
