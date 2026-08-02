"""Bronze marketcap parquet writer (#295).

Mirrors the bronze schema convention from services/history_backfill.py (#168):
hourly bar-aligned `start` (epoch seconds), domain columns (market_cap, fdv),
and PIT columns (ingest_ts, schema_version).

Per-pid path: backend/data/marketcap/<pid>.parquet

Two providers supported via --source flag (CLI):
    coingecko   — services/coingecko_marketcap (needs COINGECKO_API_KEY for free
                  Demo tier, otherwise paywalled on /market_chart/range).
    coinpaprika — services/coinpaprika_marketcap (no key, 25k req/day).

Both providers expose `fetch_marketcap_history(pid, start_ms, end_ms) ->
list[(ts_ms, market_cap)]`. CoinPaprika has no FDV; rows from coinpaprika
default `fdv = market_cap`.

Run:
    cd backend && python -m tools.build_marketcap_parquet \\
        --source coinpaprika --pids BTC-USD,ETH-USD,SOL-USD \\
        --start 2025-01-01 --end 2026-05-10
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import logging
import os
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logger = logging.getLogger(__name__)

_BAR_SECS = 3600
_MARKETCAP_DIR = os.path.join(BACKEND, "data", "marketcap")

_SCHEMA_VERSION = 2  # was 1; v2 adds volume_24h (Step A, 2026-05-16)
_SCHEMA = pa.schema(
    [
        pa.field("start", pa.int64()),
        pa.field("market_cap", pa.float64()),
        pa.field("fdv", pa.float64()),
        pa.field("volume_24h", pa.float64()),
        pa.field("ingest_ts", pa.int64()),
        pa.field("schema_version", pa.int32()),
    ]
)


def _parquet_path(product_id: str) -> str:
    safe = product_id.replace("/", "_")
    return os.path.join(_MARKETCAP_DIR, f"{safe}.parquet")


def _load_marketcap_history(path: str) -> List[Dict]:
    """Read a marketcap parquet file by absolute path; [] if missing."""
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    rows = table.to_pydict()
    n = len(rows["start"])
    out: List[Dict] = []
    has_ingest = "ingest_ts" in rows
    has_sv = "schema_version" in rows
    for i in range(n):
        r = {
            "start": int(rows["start"][i]),
            "market_cap": float(rows["market_cap"][i]),
            "fdv": float(rows["fdv"][i]),
        }
        # volume_24h is v2-only — old v1 parquets lack the column entirely.
        if "volume_24h" in rows and rows["volume_24h"][i] is not None:
            r["volume_24h"] = float(rows["volume_24h"][i])
        if has_ingest and rows["ingest_ts"][i] is not None:
            r["ingest_ts"] = int(rows["ingest_ts"][i])
        if has_sv and rows["schema_version"][i] is not None:
            r["schema_version"] = int(rows["schema_version"][i])
        out.append(r)
    return sorted(out, key=lambda r: r["start"])


def _save_marketcap_history(
    path: str,
    rows: List[Dict],
    *,
    now_ts: Optional[int] = None,
) -> None:
    """Write deduplicated marketcap rows to an absolute parquet path.

    PIT semantics mirror history_backfill (#168):
      - Rows without `ingest_ts` are stamped with `now_ts` (or wall-clock).
      - Rows already carrying `ingest_ts` keep it across rewrites.
      - On `start`-collision dedup, the LAST row in input wins (caller can
        sort to express priority).
      - `fdv` defaults to `market_cap` when missing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if now_ts is None:
        now_ts = int(time.time())
    seen: Dict[int, Dict] = {}
    for r in rows:
        start = int(r["start"])
        mc = float(r["market_cap"])
        fdv = float(r.get("fdv", mc))
        merged: Dict = {
            "start": start,
            "market_cap": mc,
            "fdv": fdv,
            "volume_24h": float(r.get("volume_24h", 0.0)),
        }
        if "ingest_ts" in r:
            merged["ingest_ts"] = int(r["ingest_ts"])
        if "schema_version" in r:
            merged["schema_version"] = int(r["schema_version"])
        seen[start] = merged
    ordered = sorted(seen.values(), key=lambda r: r["start"])
    table = pa.table(
        {
            "start": [r["start"] for r in ordered],
            "market_cap": [r["market_cap"] for r in ordered],
            "fdv": [r["fdv"] for r in ordered],
            "volume_24h": [float(r.get("volume_24h", 0.0)) for r in ordered],
            "ingest_ts": [int(r.get("ingest_ts", now_ts)) for r in ordered],
            "schema_version": [int(r.get("schema_version", _SCHEMA_VERSION)) for r in ordered],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def rows_from_history(
    history: Iterable[Tuple[int, float, float]],
    fdv_history: Optional[Iterable[Tuple[int, float]]] = None,
) -> List[Dict]:
    """Convert raw `(ts_ms, market_cap, volume_24h)` rows to bar-aligned save dicts.

    `history` is the output of services/<provider>.fetch_marketcap_history()
    which now returns 3-tuples (Step A, 2026-05-16). Each ts_ms is floored to
    the hour bar (epoch_secs // 3600 * 3600). Optional `fdv_history` aligns
    by ts_ms; when not provided, fdv == market_cap.
    """
    fdv_lookup: Dict[int, float] = {}
    if fdv_history is not None:
        for ts_ms, fdv in fdv_history:
            fdv_lookup[int(ts_ms)] = float(fdv)

    out: List[Dict] = []
    for entry in history:
        # Accept (ts_ms, mc) for back-compat AND (ts_ms, mc, vol) for v2 fetchers.
        if len(entry) >= 3:
            ts_ms, mc, vol = entry[0], entry[1], entry[2]
        else:
            ts_ms, mc = entry[0], entry[1]
            vol = 0.0
        start = (int(ts_ms) // 1000 // _BAR_SECS) * _BAR_SECS
        fdv = fdv_lookup.get(int(ts_ms), float(mc))
        out.append(
            {
                "start": start,
                "market_cap": float(mc),
                "fdv": fdv,
                "volume_24h": float(vol),
            }
        )
    return out


# ── CLI runner ──────────────────────────────────────────────────────────────


async def _fetch_one(source: str, pid: str, start_ms: int, end_ms: int):
    if source == "coingecko":
        from services.coingecko_marketcap import fetch_marketcap_history
    elif source == "coinpaprika":
        from services.coinpaprika_marketcap import fetch_marketcap_history
    else:
        raise ValueError(f"unknown source: {source}")
    return await fetch_marketcap_history(pid, start_ms, end_ms)


def _date_to_ms(s: str) -> int:
    d = _dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["coingecko", "coinpaprika"])
    parser.add_argument("--pids", required=True, help="comma-separated, e.g. BTC-USD,ETH-USD")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--out-dir", default=_MARKETCAP_DIR)
    args = parser.parse_args()

    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    start_ms = _date_to_ms(args.start)
    end_ms = _date_to_ms(args.end)

    print(f"source={args.source}  pids={pids}  range={args.start}..{args.end}", flush=True)

    written = 0
    skipped: List[str] = []
    for pid in pids:
        print(f"  fetching {pid}...", flush=True)
        history = asyncio.run(_fetch_one(args.source, pid, start_ms, end_ms))
        if not history:
            skipped.append(pid)
            print("    no rows — skipping", flush=True)
            continue
        rows = rows_from_history(history)
        path = os.path.join(args.out_dir, f"{pid.replace('/', '_')}.parquet")
        _save_marketcap_history(path, rows)
        print(f"    {len(rows):,} rows -> {path}", flush=True)
        written += 1

    print(f"\ndone: wrote {written}/{len(pids)} pids; skipped={skipped}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
