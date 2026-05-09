"""Leader filter CLI (#leader-filter).

Modes:
  forward — fetch latest fundamentals for tracked pids, merge, rank, persist
  inspect — read-only: print last N days of rows for one pid

Run:
  cd backend && python tools/leader_filter_runner.py --mode=forward
  cd backend && python tools/leader_filter_runner.py --mode=inspect --pid=BTC-USD --days=7
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import services.coingecko_marketcap as _cg_mod  # noqa: E402
import services.coinmarketcap_marketcap as _cmc_mod  # noqa: E402
from services.coingecko_marketcap import _PRODUCT_TO_CG_ID  # noqa: E402
from services.coinmarketcap_marketcap import _PRODUCT_TO_CMC_ID  # noqa: E402
from services.leader_fundamentals_merge import merge_marketcap_rows  # noqa: E402
from services.leader_fundamentals_db import FundamentalsDB, PersistRecord  # noqa: E402
from services.leader_ranker import (  # noqa: E402
    load_config_from_env, rank_leaders,
)


_DEFAULT_DB = os.path.join(BACKEND, "coinbase.db")
_DEFAULT_DATA_DIR = os.path.join(BACKEND, "data", "leader_filter")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="leader_filter_runner")
    p.add_argument("--mode", choices=["forward", "inspect"], required=True)
    p.add_argument("--pid", default=None, help="for --mode=inspect")
    p.add_argument("--days", type=int, default=7, help="for --mode=inspect")
    p.add_argument("--db", default=_DEFAULT_DB)
    p.add_argument("--data-dir", default=_DEFAULT_DATA_DIR)
    return p.parse_args(argv)


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    print(f"[{ts}] {msg}")


def _default_tracked_pids() -> list[str]:
    cg_set = set(_PRODUCT_TO_CG_ID.keys())
    cmc_set = set(_PRODUCT_TO_CMC_ID.keys())
    return sorted(cg_set & cmc_set)


async def run_forward(
    db_path: str, data_dir: str, *, tracked_pids: Iterable[str] | None = None
) -> None:
    cfg = load_config_from_env()

    pids = list(tracked_pids) if tracked_pids is not None else _default_tracked_pids()
    if not pids:
        _log("no tracked pids — nothing to do")
        return

    cg_task = _cg_mod.fetch_marketcap_snapshot(pids)
    cmc_task = _cmc_mod.fetch_marketcap_snapshot(pids)
    cg_rows, cmc_rows = await asyncio.gather(cg_task, cmc_task)
    _log(f"fetched: CG={len(cg_rows)} pids, CMC={len(cmc_rows)} pids")

    snapshot_dt = datetime.now(timezone.utc)
    snapshot_ts = int(snapshot_dt.timestamp())
    now_ts = int(time.time())

    persist_records: list[PersistRecord] = []
    merged_snapshot = {}

    for pid in pids:
        cg_row = cg_rows.get(pid)
        cmc_row = cmc_rows.get(pid)
        if cg_row is None and cmc_row is None:
            _log(f"skip {pid}: no provider returned data")
            continue
        try:
            outcome = merge_marketcap_rows(
                pid, cg_row, cmc_row,
                divergence_threshold=cfg.supply_divergence_threshold,
            )
        except ValueError as e:
            _log(f"skip {pid}: {e}")
            continue
        merged_snapshot[pid] = outcome.row
        persist_records.append(PersistRecord(
            snapshot_ts=snapshot_ts, product_id=pid, row=outcome.row,
            merge_quality=outcome.merge_quality,
            cg_market_cap=outcome.cg_market_cap, cmc_market_cap=outcome.cmc_market_cap,
            cg_circ_supply=outcome.cg_circ_supply, cmc_circ_supply=outcome.cmc_circ_supply,
            cg_total_supply=outcome.cg_total_supply, cmc_total_supply=outcome.cmc_total_supply,
            cg_volume_24h=outcome.cg_volume_24h, cmc_volume_24h=outcome.cmc_volume_24h,
            source="forward", ingest_ts=now_ts, schema_version=1,
        ))
        _append_divergence_log(data_dir, pid, snapshot_ts, outcome.audit_entries)

    db = FundamentalsDB(db_path)
    await db.ensure_schema()
    await db.insert(persist_records)
    _log(f"inserted {len(persist_records)} rows → {db_path}")

    result = rank_leaders(merged_snapshot, cfg, snapshot_ts=snapshot_dt)

    _write_top_leaders_json(data_dir, snapshot_dt, snapshot_ts, cfg, result)
    _print_top_leaders(result, merged_snapshot)


def _append_divergence_log(data_dir: str, pid: str, ts: int, entries: list[dict]) -> None:
    path = Path(data_dir) / "divergence_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps({"ts": ts, "pid": pid, **e}) + "\n")


def _write_top_leaders_json(data_dir, snapshot_dt, snapshot_ts, cfg, result) -> None:
    path = Path(data_dir) / "top_leaders.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    exclusions: dict[str, list[str]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    for pid, sr in result.scored.items():
        counts[sr.reason] += 1
        if sr.reason != "PASS":
            exclusions[sr.reason].append(pid)

    top_leaders = []
    for rank, pid in enumerate(result.top_pids, start=1):
        sr = result.scored[pid]
        top_leaders.append({
            "rank": rank, "pid": pid,
            "final_score": sr.final_score, "raw_score": sr.raw_score,
            "multiplier": sr.multiplier,
            "fdv_mcap_ratio": sr.fdv_mcap_ratio,
            "circ_total_ratio": sr.circ_total_ratio,
        })

    payload = {
        "snapshot_ts": snapshot_ts,
        "snapshot_iso": snapshot_dt.isoformat(),
        "thresholds_used": {
            "min_market_cap_usd": cfg.min_market_cap_usd,
            "min_volume_24h_usd": cfg.min_volume_24h_usd,
            "max_fdv_mcap_ratio": cfg.max_fdv_mcap_ratio,
            "supply_divergence_threshold": cfg.supply_divergence_threshold,
            "top_n": cfg.top_n,
        },
        "top_leaders": top_leaders,
        "exclusions": dict(exclusions),
        "exclusion_counts": dict(counts),
    }
    path.write_text(json.dumps(payload, indent=2))
    _log(f"top_leaders.json updated → {path}")


def _print_top_leaders(result, snapshot) -> None:
    if not result.top_pids:
        _log("no leaders this run (all excluded)")
        return
    _log(f"current top-{len(result.top_pids)} leaders:")
    for rank, pid in enumerate(result.top_pids, start=1):
        sr = result.scored[pid]
        row = snapshot[pid]
        vol_pct = (row.volume_24h / row.market_cap * 100) if row.market_cap > 0 else 0
        print(f"             {rank}. {pid:11}  score {sr.final_score:.4f}  "
              f"({row.price_chg_24h_pct:+.1f}%, vol/cap {vol_pct:.1f}%, mult {sr.multiplier:.2f})")


async def run_inspect(db_path: str, pid: str, days: int) -> None:
    db = FundamentalsDB(db_path)
    await db.ensure_schema()
    rows = await db.query_by_pid(pid, limit=days * 24)
    print(f"{pid}: {len(rows)} rows in last {days} days")
    for r in rows:
        ts = datetime.fromtimestamp(r.snapshot_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"  {ts}  price={r.row.price:.4f}  mcap={r.row.market_cap:.0f}  "
              f"vol24h={r.row.volume_24h:.0f}  chg24h={r.row.price_chg_24h_pct:+.2f}%")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    if args.mode == "forward":
        if not os.environ.get("COINMARKETCAP_API_KEY"):
            print(
                "ERROR: COINMARKETCAP_API_KEY is not set in environment. "
                "Sign up free at coinmarketcap.com/api and add the key to .env.",
                file=sys.stderr,
            )
            return 2
        asyncio.run(run_forward(args.db, args.data_dir))
        return 0

    if args.mode == "inspect":
        if not args.pid:
            print("ERROR: --mode=inspect requires --pid", file=sys.stderr)
            return 2
        asyncio.run(run_inspect(args.db, args.pid, args.days))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
