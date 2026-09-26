"""Supply-snapshot parquet writer + CLI for the strategy-discovery rebuild.

Hits CoinPaprika's /v1/tickers/{cp_id} endpoint (via
services.coinpaprika_marketcap.fetch_supply_snapshot) and persists one row per
pid to backend/data/supply/snapshot.parquet.

Schema:
  pid          : string   (Coinbase product id, e.g. "BTC-USD")
  circulating  : float64  (tokens in market)
  total        : float64  (tokens that will eventually exist; max may exceed)
  max_supply   : float64? (None for uncapped tokens like ETH; stored as null)
  ingest_ts    : int64    (wall-clock epoch seconds at write time)
  schema_version : int32  (currently 1)

On `pid`-collision dedup, the LAST row in input wins (mirrors
build_marketcap_parquet semantics).
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Ensure backend/ is importable when this file is run as a script.
BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services.coinpaprika_marketcap import fetch_supply_snapshot  # noqa: E402

_SCHEMA_VERSION = 1
_SCHEMA = pa.schema(
    [
        pa.field("pid", pa.string()),
        pa.field("circulating", pa.float64()),
        pa.field("total", pa.float64()),
        pa.field("max_supply", pa.float64()),
        pa.field("ingest_ts", pa.int64()),
        pa.field("schema_version", pa.int32()),
    ]
)

_DEFAULT_PATH = os.path.join(BACKEND, "data", "supply", "snapshot.parquet")


def _now_ts() -> int:
    return int(time.time())


def save_snapshot(path: str, rows: List[Dict]) -> None:
    """Write deduplicated supply-snapshot rows to a parquet path.

    Dedup: on pid-collision, the last row in input wins.
    `max_supply=None` is stored as a parquet null (pa.float64() is nullable).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen: Dict[str, Dict] = {}
    for r in rows:
        pid = str(r["pid"])
        seen[pid] = {
            "pid": pid,
            "circulating": float(r["circulating"]),
            "total": float(r["total"]),
            "max_supply": None if r.get("max_supply") is None else float(r["max_supply"]),
            "ingest_ts": int(r.get("ingest_ts", _now_ts())),
            "schema_version": int(r.get("schema_version", _SCHEMA_VERSION)),
        }
    ordered = sorted(seen.values(), key=lambda r: r["pid"])
    table = pa.table(
        {
            "pid": [r["pid"] for r in ordered],
            "circulating": [r["circulating"] for r in ordered],
            "total": [r["total"] for r in ordered],
            "max_supply": [r["max_supply"] for r in ordered],
            "ingest_ts": [r["ingest_ts"] for r in ordered],
            "schema_version": [r["schema_version"] for r in ordered],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def load_snapshot(path: str) -> List[Dict]:
    """Read supply-snapshot rows from a parquet path. Missing -> []."""
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = len(cols["pid"])
    out: List[Dict] = []
    for i in range(n):
        out.append(
            {
                "pid": str(cols["pid"][i]),
                "circulating": float(cols["circulating"][i]),
                "total": float(cols["total"][i]),
                "max_supply": None
                if cols["max_supply"][i] is None
                else float(cols["max_supply"][i]),
                "ingest_ts": int(cols["ingest_ts"][i]),
                "schema_version": int(cols["schema_version"][i]),
            }
        )
    return out


async def fetch_and_persist(
    pids: List[str],
    *,
    parquet_path: str = _DEFAULT_PATH,
    sleep_secs: float = 0.5,
) -> Dict[str, Optional[tuple]]:
    """Fetch supply snapshots for each pid and merge into the parquet at parquet_path.

    Returns {pid: (circ, total, max_or_None) | None} for caller diagnostics.
    Existing rows for pids not in `pids` are preserved.
    A `sleep_secs` delay between fetches keeps us friendly with the free tier
    (default 0.5s ~ 7,200 req/hour, well under CoinPaprika's 25k/day). Tests
    can pass sleep_secs=0.0 to skip the delay.
    """
    existing = load_snapshot(parquet_path)
    merged: Dict[str, Dict] = {r["pid"]: r for r in existing}
    results: Dict[str, Optional[tuple]] = {}
    now = _now_ts()

    for pid in pids:
        snap = await fetch_supply_snapshot(pid)
        results[pid] = snap
        if snap is None:
            continue
        circ, total, max_supply = snap
        merged[pid] = {
            "pid": pid,
            "circulating": circ,
            "total": total,
            "max_supply": max_supply,
            "ingest_ts": now,
            "schema_version": _SCHEMA_VERSION,
        }
        if sleep_secs > 0.0:
            await asyncio.sleep(sleep_secs)

    save_snapshot(parquet_path, list(merged.values()))
    return results


def main() -> int:
    """CLI: fetch supply snapshots for a comma-separated pid list.

    Run:
        cd backend && python -m tools.strategy_discovery.build_supply_snapshot \\
            --pids BTC-USD,ETH-USD,SOL-USD
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", required=True, help="comma-separated, e.g. BTC-USD,ETH-USD")
    parser.add_argument("--out", default=_DEFAULT_PATH)
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    print(f"  fetching supply snapshots for {len(pids)} pids -> {args.out}", flush=True)

    results = asyncio.run(fetch_and_persist(pids, parquet_path=args.out, sleep_secs=args.sleep))

    ok = [p for p, v in results.items() if v is not None]
    failed = [p for p, v in results.items() if v is None]
    print(f"  ok:     {len(ok)}", flush=True)
    print(f"  failed: {len(failed)}  -> {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
