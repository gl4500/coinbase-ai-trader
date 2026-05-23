"""Operator CLI: backfill 1-minute candles for the scorecard's top-20 products.

Each product's depth is computed from its existing 1h parquet so the 1m
history covers the same calendar span. Long-running and network-heavy — run
offline; it makes only read-only Coinbase candle requests (it does not touch
the database or the live backend).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time

from services.history_backfill import backfill_product_1m, load_history


def _days_to_cover(pid: str, now_ts: int) -> int:
    """Days back needed for 1m history to reach the 1h parquet's first bar.

    Returns 0 when the product has no 1h parquet (nothing to calibrate against).
    """
    one_h = load_history(pid)
    if not one_h:
        return 0
    first_ts = int(one_h[0]["start"])
    return max(1, math.ceil((now_ts - first_ts) / 86400))


def _resolve_pids(cache_path: str, pids_arg: str | None) -> list[str]:
    """Explicit --pids list, else the survivorship-aware top-20 from the cache."""
    if pids_arg:
        return [p.strip() for p in pids_arg.split(",") if p.strip()]
    from tools._scorecard._cv_harness import top_n_pids_from_cache
    return list(top_n_pids_from_cache(cache_path))


async def _run(pids: list[str]) -> None:
    now_ts = int(time.time())
    for i, pid in enumerate(pids, 1):
        days = _days_to_cover(pid, now_ts)
        if days == 0:
            print(f"[{i}/{len(pids)}] {pid}: no 1h parquet — skip", flush=True)
            continue
        print(f"[{i}/{len(pids)}] {pid}: backfilling 1m, {days}d ...", flush=True)
        result = await backfill_product_1m(pid, days=days)
        print(f"    +{result['new_bars']} new | {result['total_bars']} total",
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill 1-minute candles for the top-20 products"
    )
    parser.add_argument("--cache", default="cnn_dataset_cache.pt",
                        help="cache for the survivorship-aware top-20 ranking")
    parser.add_argument("--pids", default=None,
                        help="comma-separated product ids (overrides --cache)")
    args = parser.parse_args()
    pids = _resolve_pids(args.cache, args.pids)
    print(f"1m backfill: {len(pids)} products", flush=True)
    asyncio.run(_run(pids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
