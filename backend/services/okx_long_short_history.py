"""
OKX historical long/short *account* ratio fetcher.

Sibling of `services/okx_oi_history` (#141 / #142). Adds long/short ratio as a
candidate single-add channel after OKX OI alone (Ch 27) failed to lift the
XGB pooled-top-20 AUC past the 0.55 hard gate — see project memory
`xgb_feature_optimization_findings.md` (`2026-05-08 #156 ... abandon` and the
"four remaining moves" punchlist).

Public API (mirrors OI fetcher):
    await fetch_long_short_ratio_history(product_id, start_ms, end_ms,
                                         bar="1H")
        -> list[(ts_ms, ratio_float)] sorted ascending,
           empty list if symbol not on OKX or fetch fails.

OKX details:
  - URL: https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio
  - Response: {"code": "0", "msg": "", "data": [...]} — `code != "0"` means
    OKX rejected the call (treat like non-200).
  - Each row is a positional array [ts_ms, ratio] (both strings on the live
    API). Dict shape {"ts": ..., "ratio": ...} also accepted.
  - `limit` caps at 100 per call — long windows require pagination via
    `after=<ts_ms>`, which returns records older than ts.
  - `period` controls bar size (1H, 4H, 1D, etc.).
  - Symbol convention: USDT-margined SWAP, e.g. `BTC-USDT-SWAP` — same map
    as okx_oi_history (re-exported to keep them in sync).

Kill switch: set env OKX_LS_DISABLED=1 to short-circuit and return []
without an HTTP call.
"""
import logging
import os
from typing import List, Optional, Tuple

import httpx

from services.okx_oi_history import _PRODUCT_TO_OKX  # single source of truth

logger = logging.getLogger(__name__)

_URL = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"

_PAGE_SIZE = 100
_MAX_PAGES = 60


def _is_disabled() -> bool:
    """True when OKX_LS_DISABLED env is set to a truthy value."""
    return os.environ.get("OKX_LS_DISABLED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def _coinbase_to_okx(product_id: str) -> Optional[str]:
    return _PRODUCT_TO_OKX.get(product_id)


def _parse_rows(rows) -> List[Tuple[int, float]]:
    """Decode OKX response rows; skip any malformed entries silently.

    Live OKX returns rows as positional arrays: [ts_ms, ratio] (both strings).
    Dict shape {"ts": ..., "ratio": ...} is also accepted so test fixtures
    stay readable — both paths land at the same (ts_ms, ratio) tuple.
    """
    out: List[Tuple[int, float]] = []
    if not isinstance(rows, list):
        return out
    for row in rows:
        try:
            if isinstance(row, dict):
                t = int(row["ts"])
                v = float(row["ratio"])
            else:
                t = int(row[0])
                v = float(row[1])
            out.append((t, v))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out


async def fetch_long_short_ratio_history(
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
        logger.debug("OKX L/S history unavailable for %s: %s", product_id, e)
        return []

    in_window = [(t, v) for t, v in collected if int(start_ms) <= t <= int(end_ms)]
    in_window.sort(key=lambda tv: tv[0])
    return in_window
