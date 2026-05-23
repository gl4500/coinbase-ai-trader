"""Bulk marketcap historical backfill for the curated 50-pid universe.

Reads the universe JSON from Task 5, identifies pids whose existing marketcap
parquets don't cover at least `--min-days` of history, and dispatches
`fetch_marketcap_history` for each missing pid. Honors the data-first
directive — never re-fetches a pid that already has sufficient coverage.

Run:
    cd backend && python -m tools.strategy_discovery.build_universe_marketcap \\
        --start 2025-05-23 --end 2026-05-23
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
from typing import Dict, List

import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services.coinpaprika_marketcap import fetch_marketcap_history  # noqa: E402
from tools.build_marketcap_parquet import (  # noqa: E402
    _save_marketcap_history,
    rows_from_history,
)

_MARKETCAP_DIR = os.path.join(BACKEND, "data", "marketcap")


def universe_pids_from_curation(curation_json_path: str) -> List[str]:
    """Flatten {cohort: [pids]} into a single deduplicated sorted list."""
    with open(curation_json_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def pids_needing_backfill(*, pids: List[str], marketcap_dir: str, min_days: int) -> List[str]:
    """Return the subset of pids whose existing marketcap parquet covers < min_days."""
    out: List[str] = []
    for pid in pids:
        path = os.path.join(marketcap_dir, f"{pid}.parquet")
        if not os.path.exists(path):
            out.append(pid)
            continue
        table = pq.read_table(path, columns=["start"])
        starts = table.column("start").to_pylist()
        if not starts:
            out.append(pid)
            continue
        days = (int(max(starts)) - int(min(starts))) // 86400
        if days < min_days:
            out.append(pid)
    return out


def _date_to_ms(s: str) -> int:
    d = _dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


async def backfill_one(
    pid: str, *, start_ms: int, end_ms: int, marketcap_dir: str
) -> int:
    """Fetch + save one pid. Returns rows written (0 on failure)."""
    history = await fetch_marketcap_history(pid, start_ms, end_ms)
    if not history:
        return 0
    rows = rows_from_history(history)
    path = os.path.join(marketcap_dir, f"{pid}.parquet")
    _save_marketcap_history(path, rows)
    return len(rows)


async def backfill_universe(
    *,
    pids: List[str],
    start_ms: int,
    end_ms: int,
    marketcap_dir: str = _MARKETCAP_DIR,
    sleep_secs: float = 0.5,
) -> Dict[str, int]:
    results: Dict[str, int] = {}
    for i, pid in enumerate(pids):
        n = await backfill_one(
            pid, start_ms=start_ms, end_ms=end_ms, marketcap_dir=marketcap_dir
        )
        results[pid] = n
        print(f"  [{i+1}/{len(pids)}] {pid}: {n} rows", flush=True)
        if i + 1 < len(pids) and sleep_secs > 0.0:
            await asyncio.sleep(sleep_secs)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe-json",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--marketcap-dir", default=_MARKETCAP_DIR)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--end",   required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--min-days", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even for pids with sufficient coverage")
    args = parser.parse_args()

    pids = universe_pids_from_curation(args.universe_json)
    print(f"  universe: {len(pids)} pids", flush=True)

    if args.force:
        needs = pids
    else:
        needs = pids_needing_backfill(
            pids=pids, marketcap_dir=args.marketcap_dir, min_days=args.min_days
        )
    print(f"  needing backfill: {len(needs)} pids", flush=True)

    if not needs:
        print("  nothing to do — all pids already covered", flush=True)
        return 0

    start_ms = _date_to_ms(args.start)
    end_ms   = _date_to_ms(args.end)
    print(f"  range: {args.start} -> {args.end}", flush=True)

    results = asyncio.run(backfill_universe(
        pids=needs,
        start_ms=start_ms,
        end_ms=end_ms,
        marketcap_dir=args.marketcap_dir,
        sleep_secs=args.sleep,
    ))

    succeeded = [p for p, n in results.items() if n > 0]
    failed    = [p for p, n in results.items() if n == 0]
    print(f"\n  succeeded: {len(succeeded)}", flush=True)
    print(f"  failed:    {len(failed)}  -> {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
