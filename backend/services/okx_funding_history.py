"""
OKX historical funding-rate fetcher.

Drop-in replacement for `services/binance_funding_history` after Binance's
fapi.binance was geo-blocked from the US (#80/#81). OKX's public funding-rate
endpoint is reachable from this region.

Public API (matches Binance fetcher exactly):
    await fetch_funding_history(product_id, start_ms, end_ms)
        -> list[(funding_time_ms, funding_rate)] sorted ascending,
           empty list if symbol not on OKX or fetch fails.

OKX details:
  - URL: https://www.okx.com/api/v5/public/funding-rate-history
  - Response: {"code": "0", "msg": "", "data": [...]} — `code != "0"` means
    OKX rejected the call (treat like non-200).
  - Each row's `fundingTime` and `fundingRate` are STRINGS.
  - `limit` caps at 100 per call (vs Binance 1000) — long windows require
    pagination via `after=<ts_ms>`, which returns records older than ts.
  - Symbol convention: USDT-margined SWAP, e.g. `BTC-USDT-SWAP`.

Kill switch: set env OKX_FUNDING_DISABLED=1 to short-circuit and return []
without an HTTP call.
"""

import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_URL = "https://www.okx.com/api/v5/public/funding-rate-history"

# OKX caps `limit` at 100 records per call. Each page spans ~33 days at the
# 8-hour funding cadence, so a 12-month window needs ~11 calls.
_PAGE_SIZE = 100

# Hard ceiling on pagination calls per fetch — defends against runaway loops
# if OKX returns rows but never crosses start_ms (shouldn't happen, but safe).
_MAX_PAGES = 60


def _is_disabled() -> bool:
    """True when OKX_FUNDING_DISABLED env is set to a truthy value."""
    return os.environ.get("OKX_FUNDING_DISABLED", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Coinbase product → OKX SWAP instId. Only entries listed here trigger a
# network call; everything else returns [] without a request, mirroring
# Binance fetcher behaviour and avoiding wasted calls for products OKX
# doesn't list (e.g. VVV-USD as of 2026-04-27).
_PRODUCT_TO_OKX = {
    "BTC-USD": "BTC-USDT-SWAP",
    "ETH-USD": "ETH-USDT-SWAP",
    "SOL-USD": "SOL-USDT-SWAP",
    "XRP-USD": "XRP-USDT-SWAP",
    "BNB-USD": "BNB-USDT-SWAP",
    "ADA-USD": "ADA-USDT-SWAP",
    "AVAX-USD": "AVAX-USDT-SWAP",
    "LINK-USD": "LINK-USDT-SWAP",
    "DOT-USD": "DOT-USDT-SWAP",
    "DOGE-USD": "DOGE-USDT-SWAP",
    "LTC-USD": "LTC-USDT-SWAP",
    "ATOM-USD": "ATOM-USDT-SWAP",
    "FIL-USD": "FIL-USDT-SWAP",
    "NEAR-USD": "NEAR-USDT-SWAP",
    "APT-USD": "APT-USDT-SWAP",
    "INJ-USD": "INJ-USDT-SWAP",
    "ARB-USD": "ARB-USDT-SWAP",
    "OP-USD": "OP-USDT-SWAP",
    "TIA-USD": "TIA-USDT-SWAP",
    "SEI-USD": "SEI-USDT-SWAP",
    "SUI-USD": "SUI-USDT-SWAP",
    "AAVE-USD": "AAVE-USDT-SWAP",
    "UNI-USD": "UNI-USDT-SWAP",
    "HYPE-USD": "HYPE-USDT-SWAP",
    "ICP-USD": "ICP-USDT-SWAP",
    "TAO-USD": "TAO-USDT-SWAP",
    "BCH-USD": "BCH-USDT-SWAP",
    "ZEC-USD": "ZEC-USDT-SWAP",
    "SHIB-USD": "SHIB-USDT-SWAP",
    "TRX-USD": "TRX-USDT-SWAP",
    # #211: alts/memes that #210 audit found all-zero in the OI cache. Same
    # set is added here so both fetchers share one supported-symbol set.
    # Pids in the zero set that aren't listed on OKX (NKN, AIOZ, JASMY,
    # TRU, SKL, FET, XCN, LRDS) are intentionally omitted.
    "PENGU-USD": "PENGU-USDT-SWAP",
    "JTO-USD": "JTO-USDT-SWAP",
    "POPCAT-USD": "POPCAT-USDT-SWAP",
    "BONK-USD": "BONK-USDT-SWAP",
    "ZK-USD": "ZK-USDT-SWAP",
    "PEPE-USD": "PEPE-USDT-SWAP",
    "MOODENG-USD": "MOODENG-USDT-SWAP",
    "ONDO-USD": "ONDO-USDT-SWAP",
    "ALGO-USD": "ALGO-USDT-SWAP",
    "ZORA-USD": "ZORA-USDT-SWAP",
    "WIF-USD": "WIF-USDT-SWAP",
    "RENDER-USD": "RENDER-USDT-SWAP",
    "FLOKI-USD": "FLOKI-USDT-SWAP",
    "WLD-USD": "WLD-USDT-SWAP",
    "BERA-USD": "BERA-USDT-SWAP",
    "ENA-USD": "ENA-USDT-SWAP",
    "STRK-USD": "STRK-USDT-SWAP",
    "TON-USD": "TON-USDT-SWAP",
    "JUP-USD": "JUP-USDT-SWAP",
}


def _coinbase_to_okx(product_id: str) -> Optional[str]:
    return _PRODUCT_TO_OKX.get(product_id)


def _parse_rows(rows) -> List[Tuple[int, float]]:
    """Decode OKX response rows; skip any malformed entries silently."""
    out: List[Tuple[int, float]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        try:
            t = int(row["fundingTime"])
            r = float(row["fundingRate"])
            out.append((t, r))
        except (KeyError, TypeError, ValueError):
            continue
    return out


async def fetch_funding_history(
    product_id: str,
    start_ms: int,
    end_ms: int,
) -> List[Tuple[int, float]]:
    if _is_disabled():
        return []
    inst_id = _coinbase_to_okx(product_id)
    if not inst_id:
        return []

    collected: List[Tuple[int, float]] = []
    cursor = int(end_ms)  # OKX `after=` returns records OLDER than this ts

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(_MAX_PAGES):
                resp = await client.get(
                    _URL,
                    params={
                        "instId": inst_id,
                        "after": cursor,
                        "limit": _PAGE_SIZE,
                    },
                )
                if resp.status_code != 200:
                    break
                payload = resp.json()
                if not isinstance(payload, dict) or payload.get("code") != "0":
                    break
                rows = _parse_rows(payload.get("data", []))
                if not rows:
                    break
                collected.extend(rows)
                # Walk back: next page anchored at the oldest ts we just saw.
                oldest = min(t for t, _ in rows)
                if oldest <= int(start_ms):
                    break
                cursor = oldest
    except Exception as e:
        logger.debug("OKX funding history unavailable for %s: %s", product_id, e)
        return []

    # Filter to requested window and sort ascending.
    in_window = [(t, r) for t, r in collected if int(start_ms) <= t <= int(end_ms)]
    in_window.sort(key=lambda tr: tr[0])
    return in_window
