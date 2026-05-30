"""CLI: aggregate 1h history into matched-count dollar bars per universe pid.

Reads the universe JSON, loads each pid's 1h history parquet, runs
aggregate_dollar_bars, and writes data/history/dollar_1h/<pid>.parquet
in the same history schema (the existing Phase 2 loader reads it unchanged).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.info_bars import aggregate_dollar_bars  # noqa: E402

_DEFAULT_HISTORY_DIR = Path(BACKEND) / "data" / "history"
_DEFAULT_OUTPUT_DIR  = Path(BACKEND) / "data" / "history" / "dollar_1h"
_DEFAULT_UNIVERSE    = (Path(BACKEND).parent / "docs" / "superpowers"
                        / "specs" / "2026-05-23-universe-50.json")


def _pids_from_universe_json(universe_path: Path) -> List[str]:
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def build_info_bars_for_pid(pid: str, history_dir: Path, output_dir: Path) -> Dict:
    """Build matched-count dollar bars for one pid; write parquet on success."""
    src = Path(history_dir) / f"{pid}.parquet"
    if not src.exists():
        return {"pid": pid, "n_bars": 0, "n_1h": 0, "error": "missing history"}
    df = pq.read_table(src).to_pandas().sort_values("start").reset_index(drop=True)
    bars = aggregate_dollar_bars(df)
    if len(bars) == 0:
        return {"pid": pid, "n_bars": 0, "n_1h": int(len(df)),
                "error": "empty aggregation"}
    out_path = Path(output_dir) / f"{pid}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out_path, compression="snappy", index=False)
    return {"pid": pid, "n_bars": int(len(bars)), "n_1h": int(len(df)), "error": None}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build matched-count dollar bars for a universe.")
    parser.add_argument("--universe",   default=str(_DEFAULT_UNIVERSE))
    parser.add_argument("--history-dir", default=str(_DEFAULT_HISTORY_DIR))
    parser.add_argument("--output-dir",  default=str(_DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    pids = _pids_from_universe_json(Path(args.universe))
    n_ok = 0
    n_err = 0
    for i, pid in enumerate(pids, 1):
        r = build_info_bars_for_pid(pid, Path(args.history_dir), Path(args.output_dir))
        if r["error"]:
            n_err += 1
            print(f"  [{i:3d}/{len(pids)}] {pid:14s}  ERROR: {r['error']}", flush=True)
        else:
            n_ok += 1
            print(f"  [{i:3d}/{len(pids)}] {pid:14s}  {r['n_bars']:6d} bars "
              f"(from {r['n_1h']} 1h rows)", flush=True)
    print(f"  ok: {n_ok}  error: {n_err}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
