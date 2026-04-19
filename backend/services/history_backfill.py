"""
Historical candle backfill service.

Fetches hourly OHLCV from Coinbase (max 300 bars/request) paginating
backwards in time, and persists each product to a parquet file:

    backend/data/history/{product_id}.parquet
    columns: start (int64), open, high, low, close, volume (float64)

Usage:
    backfill = HistoryBackfill()
    result   = await backfill.run(days=365)   # fetch ~1 year per product
    result   = await backfill.run(days=90, product_ids=["BTC-USD"])

Incremental: if a parquet file already exists, only fetches bars newer
than the last stored timestamp — safe to re-run at any time.
"""
import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Set

import pyarrow as pa
import pyarrow.parquet as pq

import database
from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)

_HISTORY_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "history")
_GRANULARITY  = "ONE_HOUR"
_BAR_SECS     = 3600          # seconds per hourly bar
_MAX_PER_REQ  = 300           # Coinbase max bars per request
_REQ_DELAY    = 0.35          # seconds between requests (rate-limit friendly)

_SCHEMA = pa.schema([
    pa.field("start",  pa.int64()),
    pa.field("open",   pa.float64()),
    pa.field("high",   pa.float64()),
    pa.field("low",    pa.float64()),
    pa.field("close",  pa.float64()),
    pa.field("volume", pa.float64()),
])


def _parquet_path(product_id: str) -> str:
    safe = product_id.replace("/", "_")
    return os.path.join(_HISTORY_DIR, f"{safe}.parquet")


def load_history(product_id: str) -> List[Dict]:
    """
    Load all stored candles for product_id from parquet.
    Returns list of dicts (start, open, high, low, close, volume), oldest first.
    Returns [] if no file exists.
    """
    path = _parquet_path(product_id)
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    rows  = table.to_pydict()
    n     = len(rows["start"])
    candles = [
        {
            "start":  rows["start"][i],
            "open":   rows["open"][i],
            "high":   rows["high"][i],
            "low":    rows["low"][i],
            "close":  rows["close"][i],
            "volume": rows["volume"][i],
        }
        for i in range(n)
    ]
    return sorted(candles, key=lambda c: c["start"])


def _save_history(product_id: str, candles: List[Dict]) -> None:
    """Write full deduplicated candle list to parquet (overwrites)."""
    os.makedirs(_HISTORY_DIR, exist_ok=True)
    # Deduplicate by start timestamp, keep latest value
    seen: Dict[int, Dict] = {}
    for c in candles:
        seen[c["start"]] = c
    ordered = sorted(seen.values(), key=lambda c: c["start"])
    table = pa.table(
        {
            "start":  [c["start"]  for c in ordered],
            "open":   [c["open"]   for c in ordered],
            "high":   [c["high"]   for c in ordered],
            "low":    [c["low"]    for c in ordered],
            "close":  [c["close"]  for c in ordered],
            "volume": [c["volume"] for c in ordered],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, _parquet_path(product_id), compression="snappy")


async def _fetch_range(
    product_id: str,
    start_ts: int,
    end_ts: int,
) -> List[Dict]:
    """Fetch one page (≤300 bars) between start_ts and end_ts from Coinbase."""
    try:
        import httpx
        from config import config
        from clients.coinbase_client import _auth_headers, _BASE  # noqa: PLC2701

        url    = f"{_BASE}/products/{product_id}/candles"
        params = {
            "start":       str(start_ts),
            "end":         str(end_ts),
            "granularity": _GRANULARITY,
        }
        hdrs = _auth_headers("GET", f"/api/v3/brokerage/products/{product_id}/candles") \
               if config.has_credentials else {}
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url, headers=hdrs, params=params)
            resp.raise_for_status()
            data = resp.json()
        candles = []
        for c in data.get("candles", []):
            candles.append({
                "start":  int(c["start"]),
                "open":   float(c["open"]),
                "high":   float(c["high"]),
                "low":    float(c["low"]),
                "close":  float(c["close"]),
                "volume": float(c["volume"]),
            })
        return candles
    except Exception as e:
        logger.warning(f"history fetch {product_id} [{start_ts}→{end_ts}]: {e}")
        return []


async def backfill_product(
    product_id: str,
    days: int = 365,
) -> Dict:
    """
    Backfill one product.  Fetches at most `days` of hourly history.
    Incremental: skips bars already stored.
    Returns {"product_id", "new_bars", "total_bars", "oldest_ts"}.
    """
    existing  = load_history(product_id)
    known_set = {c["start"] for c in existing}

    # Determine fetch window
    now        = int(time.time())
    target_start = now - days * 86400
    # If we already have data, only fetch from the last known bar onwards
    fetch_end  = now
    fetch_start = target_start if not known_set else max(target_start, max(known_set))

    all_new: List[Dict] = []
    window_end = fetch_end

    while window_end > fetch_start:
        window_start = max(fetch_start, window_end - _MAX_PER_REQ * _BAR_SECS)
        page = await _fetch_range(product_id, window_start, window_end)
        new  = [c for c in page if c["start"] not in known_set]
        all_new.extend(new)
        known_set.update(c["start"] for c in new)

        if not page:
            break
        window_end = window_start
        await asyncio.sleep(_REQ_DELAY)

    if all_new:
        merged = existing + all_new
        _save_history(product_id, merged)
        total  = len({c["start"] for c in merged})
    else:
        total = len(existing)

    oldest = min(known_set) if known_set else None
    logger.info(
        f"Backfill {product_id}: +{len(all_new)} new bars | "
        f"{total} total | oldest={oldest}"
    )
    return {
        "product_id": product_id,
        "new_bars":   len(all_new),
        "total_bars": total,
        "oldest_ts":  oldest,
    }


class HistoryBackfill:
    """
    Async backfill runner.  Call run() from the API endpoint or on startup.
    """

    async def get_all_coinbase_usd_pairs(self) -> List[str]:
        """Fetch the full list of online USD spot pairs from Coinbase."""
        try:
            all_products = await coinbase_client.get_products()
            pids = [
                p["product_id"] for p in all_products
                if p.get("quote_currency_id") == "USD"
                and p.get("status") == "online"
                and not p.get("trading_disabled", False)
            ]
            logger.info(f"Backfill: discovered {len(pids)} Coinbase USD pairs")
            return pids
        except Exception as e:
            logger.error(f"Backfill: failed to fetch Coinbase product list: {e}")
            return []

    async def run(
        self,
        days:          int                  = 365,
        product_ids:   Optional[List[str]]  = None,
        all_coinbase:  bool                 = False,
    ) -> Dict:
        """
        Backfill products and write parquet files.
        - all_coinbase=True  → all online USD spot pairs on Coinbase
        - product_ids list   → specific subset
        - neither            → tracked products only
        Returns summary {"products": [...results...], "total_new_bars": int, "total_products": int}.
        """
        if all_coinbase:
            product_ids = await self.get_all_coinbase_usd_pairs()
        elif product_ids is None:
            products    = await database.get_products(tracked_only=True, limit=500)
            product_ids = [p["product_id"] for p in products]

        results   = []
        total_new = 0
        for i, pid in enumerate(product_ids, 1):
            logger.info(f"Backfill [{i}/{len(product_ids)}] {pid}")
            result = await backfill_product(pid, days=days)
            results.append(result)
            total_new += result["new_bars"]

        logger.info(
            f"Backfill complete: {len(product_ids)} products | "
            f"+{total_new} total new bars"
        )
        return {
            "products":       results,
            "total_new_bars": total_new,
            "total_products": len(product_ids),
        }

    def _products_without_history(self, product_ids: List[str]) -> List[str]:
        """Return the subset of product_ids that have no parquet file yet."""
        return [pid for pid in product_ids if not os.path.exists(_parquet_path(pid))]

    async def run_loop(self) -> None:
        """
        Background loop that keeps parquet history fresh automatically.

        On every cycle (default every 24 h):
          1. New-coin sweep — any tracked product without a parquet file gets a
             full BACKFILL_NEW_PRODUCT_DAYS backfill immediately.
          2. Incremental top-up — all other tracked products get the bars since
             their newest stored timestamp fetched and appended.

        Interval is controlled by BACKFILL_INTERVAL_HOURS in .env (0 = disabled).
        """
        interval_h = config.backfill_interval_hours
        if interval_h <= 0:
            logger.info("Periodic backfill disabled (BACKFILL_INTERVAL_HOURS=0)")
            return

        interval_s = interval_h * 3600
        # Stagger first run by 5 minutes so startup I/O settles first
        await asyncio.sleep(300)
        logger.info(
            f"Periodic backfill loop started — every {interval_h}h | "
            f"new-product history: {config.backfill_new_product_days}d"
        )

        while True:
            try:
                products    = await database.get_products(tracked_only=True, limit=500)
                all_tracked = [p["product_id"] for p in products]

                # ── Phase 1: full backfill for products with no history yet ────
                new_products = self._products_without_history(all_tracked)
                if new_products:
                    logger.info(
                        f"Backfill: {len(new_products)} new product(s) detected — "
                        f"fetching {config.backfill_new_product_days}d history: {new_products}"
                    )
                    for pid in new_products:
                        try:
                            result = await backfill_product(pid, days=config.backfill_new_product_days)
                            logger.info(
                                f"New-product backfill {pid}: "
                                f"+{result['new_bars']} bars | {result['total_bars']} total"
                            )
                        except Exception as e:
                            logger.warning(f"New-product backfill failed for {pid}: {e}")

                # ── Phase 2: incremental top-up for existing history ───────────
                existing = [pid for pid in all_tracked if pid not in new_products]
                if existing:
                    logger.info(f"Backfill: incremental top-up for {len(existing)} products")
                    total_new = 0
                    for pid in existing:
                        try:
                            result  = await backfill_product(pid, days=config.backfill_new_product_days)
                            total_new += result["new_bars"]
                        except Exception as e:
                            logger.warning(f"Incremental backfill failed for {pid}: {e}")
                    logger.info(f"Backfill top-up complete: +{total_new} new bars across {len(existing)} products")

            except asyncio.CancelledError:
                logger.info("Backfill loop cancelled")
                return
            except Exception as e:
                logger.error(f"Backfill loop error: {e}")

            await asyncio.sleep(interval_s)


# Module-level singleton
_instance: Optional[HistoryBackfill] = None


def get_backfill() -> HistoryBackfill:
    global _instance
    if _instance is None:
        _instance = HistoryBackfill()
    return _instance
