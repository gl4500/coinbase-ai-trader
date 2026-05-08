"""
OKX historical open-interest fetcher.

Sibling of `services/okx_funding_history` (#86 / #88). Adds open-interest as a
new CNN/XGB input channel after the XGB hyperparameter sweep on the existing
27 channels capped at AUC ≈ 0.528 (gate 0.55) — see project memory
`xgb_feature_optimization_findings.md`.

Public API (mirrors funding fetcher):
    await fetch_oi_history(product_id, start_ms, end_ms, bar="1H")
        -> list[(ts_ms, oi_contracts)] sorted ascending,
           empty list if symbol not on OKX or fetch fails.

OKX details:
  - URL: https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history
  - Response: {"code": "0", "msg": "", "data": [...]} — `code != "0"` means
    OKX rejected the call (treat like non-200).
  - Each row's `ts` and `oi` are STRINGS (`oiCcy` is also present but unused).
  - `limit` caps at 100 per call — long windows require pagination via
    `after=<ts_ms>`, which returns records older than ts.
  - `period` controls bar size (1H, 4H, 1D, etc.).
  - Symbol convention: USDT-margined SWAP, e.g. `BTC-USDT-SWAP`.

Kill switch: set env OKX_OI_DISABLED=1 to short-circuit and return []
without an HTTP call.
"""
import logging
import os
from typing import List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_URL = "https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history"

_PAGE_SIZE = 100
_MAX_PAGES = 60


def _is_disabled() -> bool:
    """True when OKX_OI_DISABLED env is set to a truthy value."""
    return os.environ.get("OKX_OI_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


# Coinbase product → OKX SWAP instId. Mirrors okx_funding_history exactly so
# both fetchers honour the same supported-symbol set.
_PRODUCT_TO_OKX = {
    "BTC-USD":   "BTC-USDT-SWAP",
    "ETH-USD":   "ETH-USDT-SWAP",
    "SOL-USD":   "SOL-USDT-SWAP",
    "XRP-USD":   "XRP-USDT-SWAP",
    "BNB-USD":   "BNB-USDT-SWAP",
    "ADA-USD":   "ADA-USDT-SWAP",
    "AVAX-USD":  "AVAX-USDT-SWAP",
    "LINK-USD":  "LINK-USDT-SWAP",
    "DOT-USD":   "DOT-USDT-SWAP",
    "DOGE-USD":  "DOGE-USDT-SWAP",
    "LTC-USD":   "LTC-USDT-SWAP",
    "ATOM-USD":  "ATOM-USDT-SWAP",
    "FIL-USD":   "FIL-USDT-SWAP",
    "NEAR-USD":  "NEAR-USDT-SWAP",
    "APT-USD":   "APT-USDT-SWAP",
    "INJ-USD":   "INJ-USDT-SWAP",
    "ARB-USD":   "ARB-USDT-SWAP",
    "OP-USD":    "OP-USDT-SWAP",
    "TIA-USD":   "TIA-USDT-SWAP",
    "SEI-USD":   "SEI-USDT-SWAP",
    "SUI-USD":   "SUI-USDT-SWAP",
    "AAVE-USD":  "AAVE-USDT-SWAP",
    "UNI-USD":   "UNI-USDT-SWAP",
    "HYPE-USD":  "HYPE-USDT-SWAP",
    "ICP-USD":   "ICP-USDT-SWAP",
    "TAO-USD":   "TAO-USDT-SWAP",
    "BCH-USD":   "BCH-USDT-SWAP",
    "ZEC-USD":   "ZEC-USDT-SWAP",
    "SHIB-USD":  "SHIB-USDT-SWAP",
    "TRX-USD":   "TRX-USDT-SWAP",
    # #211: alts/memes that #210 audit found all-zero in the cache. Live OKX
    # SWAP probe (probe_okx_swap_listings.py, 2026-05-08) confirmed each one
    # has a `<TICKER>-USDT-SWAP` instrument. Pids in the #210 zero set that
    # are NOT listed on OKX (NKN, AIOZ, JASMY, TRU, SKL, FET, XCN, LRDS) are
    # intentionally omitted so they keep returning [] without an HTTP call.
    "PENGU-USD":   "PENGU-USDT-SWAP",
    "JTO-USD":     "JTO-USDT-SWAP",
    "POPCAT-USD":  "POPCAT-USDT-SWAP",
    "BONK-USD":    "BONK-USDT-SWAP",
    "ZK-USD":      "ZK-USDT-SWAP",
    "PEPE-USD":    "PEPE-USDT-SWAP",
    "MOODENG-USD": "MOODENG-USDT-SWAP",
    "ONDO-USD":    "ONDO-USDT-SWAP",
    "ALGO-USD":    "ALGO-USDT-SWAP",
    "ZORA-USD":    "ZORA-USDT-SWAP",
    "WIF-USD":     "WIF-USDT-SWAP",
    "RENDER-USD":  "RENDER-USDT-SWAP",
    "FLOKI-USD":   "FLOKI-USDT-SWAP",
    "WLD-USD":     "WLD-USDT-SWAP",
    "BERA-USD":    "BERA-USDT-SWAP",
    "ENA-USD":     "ENA-USDT-SWAP",
    "STRK-USD":    "STRK-USDT-SWAP",
    "TON-USD":     "TON-USDT-SWAP",
    "JUP-USD":     "JUP-USDT-SWAP",
}


def _coinbase_to_okx(product_id: str) -> Optional[str]:
    return _PRODUCT_TO_OKX.get(product_id)


def _parse_rows(rows) -> List[Tuple[int, float]]:
    """Decode OKX response rows; skip any malformed entries silently.

    Live OKX returns rows as positional arrays: [ts_ms, oi_contracts, oiCcy,
    oiUsd] (all strings). Dict shape is also accepted so the original test
    fixtures keep working — both paths land at the same (ts_ms, oi) tuple.
    """
    out: List[Tuple[int, float]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        try:
            if isinstance(row, dict):
                t = int(row["ts"])
                v = float(row["oi"])
            else:
                t = int(row[0])
                v = float(row[1])
            out.append((t, v))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


async def fetch_oi_history(
    product_id: str,
    start_ms: int,
    end_ms: int,
    bar: str = "1H",
) -> List[Tuple[int, float]]:
    if _is_disabled():
        return []
    inst_id = _coinbase_to_okx(product_id)
    if not inst_id:
        return []

    collected: List[Tuple[int, float]] = []
    cursor = int(end_ms)   # OKX `after=` returns records OLDER than this ts

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            for _ in range(_MAX_PAGES):
                resp = await client.get(
                    _URL,
                    params={
                        "instId": inst_id,
                        "period": bar,
                        "after":  cursor,
                        "limit":  _PAGE_SIZE,
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
                oldest = min(t for t, _ in rows)
                if oldest <= int(start_ms):
                    break
                cursor = oldest
    except Exception as e:
        logger.debug("OKX OI history unavailable for %s: %s", product_id, e)
        return []

    in_window = [(t, v) for t, v in collected if int(start_ms) <= t <= int(end_ms)]
    in_window.sort(key=lambda tv: tv[0])
    return in_window
