"""Phase 2 orchestrator — universe → per-token feature+label parquet.

Loads inputs (curated universe JSON, per-pid hourly OHLCV parquet, per-pid
daily CoinPaprika marketcap parquet, single supply-snapshot parquet),
applies features → tokenomic stamping → labels, and writes one parquet
per pid to backend/data/phase2/.

Run:
    cd backend && python -m tools.strategy_discovery.build_phase2 \\
        --universe ../docs/superpowers/specs/2026-05-23-universe-50.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.features import (  # noqa: E402
    _TREND_COLUMNS,
    add_trend_features,
    first_valid_index,
)
from tools.strategy_discovery.labels import (  # noqa: E402
    _DEFAULT_HORIZONS,
    simulate_dynamic_exit_labels,
)
from tools.strategy_discovery.tokenomic_stamp import (  # noqa: E402
    SupplySnapshot,
    _TOKENOMIC_COLUMNS,
    stamp_tokenomic,
)

_SCHEMA_VERSION = 1
_DEFAULT_HISTORY_DIR  = Path(BACKEND) / "data" / "history"
_DEFAULT_MARKETCAP_DIR = Path(BACKEND) / "data" / "marketcap"
_DEFAULT_SUPPLY_PATH  = Path(BACKEND) / "data" / "supply" / "snapshot.parquet"
_DEFAULT_OUTPUT_DIR   = Path(BACKEND) / "data" / "phase2"


@dataclass
class BuildResult:
    pid: str
    rows_written: int = 0
    rows_dropped_missing_volume: int = 0
    nan_label_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


def _load_supply_snapshot(supply_path: Path, pid: str) -> Optional[SupplySnapshot]:
    """Read one pid's row from the supply snapshot parquet. Returns None if absent."""
    if not supply_path.exists():
        return None
    tbl = pq.read_table(supply_path).to_pandas()
    row = tbl[tbl["pid"] == pid]
    if row.empty:
        return None
    r = row.iloc[0]
    max_supply = None if pd.isna(r["max_supply"]) else float(r["max_supply"])
    return SupplySnapshot(
        pid=pid,
        circulating=float(r["circulating"]),
        total=float(r["total"]),
        max_supply=max_supply,
    )


def _load_history_parquet(path: Path) -> pd.DataFrame:
    """Read history parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def _load_marketcap_parquet(path: Path) -> pd.DataFrame:
    """Read marketcap parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def build_phase2_for_pid(
    pid: str,
    history_dir: Path,
    marketcap_dir: Path,
    supply_path: Path,
    output_dir: Path,
) -> BuildResult:
    """End-to-end Phase 2 build for one pid. Writes output_dir/{pid}.parquet."""
    history_path   = Path(history_dir)   / f"{pid}.parquet"
    marketcap_path = Path(marketcap_dir) / f"{pid}.parquet"
    if not history_path.exists():
        return BuildResult(pid=pid, error=f"missing history: {history_path}")
    if not marketcap_path.exists():
        return BuildResult(pid=pid, error=f"missing marketcap: {marketcap_path}")
    supply = _load_supply_snapshot(Path(supply_path), pid)
    if supply is None:
        return BuildResult(pid=pid, error=f"missing supply: {pid}")

    df_hourly = _load_history_parquet(history_path)
    if len(df_hourly) < 200:
        return BuildResult(pid=pid, error=f"history too short ({len(df_hourly)} < 200 bars)")
    df_daily = _load_marketcap_parquet(marketcap_path)

    # features → drop warmup → stamp → labels
    df_feat = add_trend_features(df_hourly)
    cut = first_valid_index(df_feat, min_warmup=200)
    df_feat = df_feat.iloc[cut:].reset_index(drop=True)
    rows_pre_drop = len(df_feat)
    df_stamped = stamp_tokenomic(df_feat, df_daily, supply, drop_on_missing_volume=True)
    rows_dropped = rows_pre_drop - len(df_stamped)
    df_labeled = simulate_dynamic_exit_labels(df_stamped, horizons=list(_DEFAULT_HORIZONS))

    df_labeled["pid"] = pid
    df_labeled["schema_version"] = _SCHEMA_VERSION
    nan_counts = {
        f"label_h{h}": int(df_labeled[f"label_h{h}"].isna().sum())
        for h in _DEFAULT_HORIZONS
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{pid}.parquet"
    df_labeled.to_parquet(out_path, compression="snappy", index=False)
    return BuildResult(
        pid=pid,
        rows_written=len(df_labeled),
        rows_dropped_missing_volume=rows_dropped,
        nan_label_counts=nan_counts,
    )


def _pids_from_universe_json(universe_path: Path) -> List[str]:
    """Flatten {cohort: [pids]} into a deduplicated sorted pid list."""
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def build_phase2_for_universe(
    universe_path: Path,
    history_dir: Optional[Path] = None,
    marketcap_dir: Optional[Path] = None,
    supply_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[BuildResult]:
    """Iterate every pid in the universe JSON; collect per-pid BuildResults."""
    history_dir   = Path(history_dir)   if history_dir   else _DEFAULT_HISTORY_DIR
    marketcap_dir = Path(marketcap_dir) if marketcap_dir else _DEFAULT_MARKETCAP_DIR
    supply_path   = Path(supply_path)   if supply_path   else _DEFAULT_SUPPLY_PATH
    output_dir    = Path(output_dir)    if output_dir    else _DEFAULT_OUTPUT_DIR
    pids = _pids_from_universe_json(Path(universe_path))
    results: List[BuildResult] = []
    for pid in pids:
        results.append(build_phase2_for_pid(
            pid, history_dir, marketcap_dir, supply_path, output_dir,
        ))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — build Phase 2 parquet for an entire universe."""
    import argparse
    parser = argparse.ArgumentParser(description="Build Phase 2 features+labels for a universe.")
    parser.add_argument(
        "--universe",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--history-dir",   default=None)
    parser.add_argument("--marketcap-dir", default=None)
    parser.add_argument("--supply",        default=None)
    parser.add_argument("--output-dir",    default=None)
    args = parser.parse_args(argv)

    results = build_phase2_for_universe(
        Path(args.universe),
        history_dir   = Path(args.history_dir)   if args.history_dir   else None,
        marketcap_dir = Path(args.marketcap_dir) if args.marketcap_dir else None,
        supply_path   = Path(args.supply)        if args.supply        else None,
        output_dir    = Path(args.output_dir)    if args.output_dir    else None,
    )
    n_ok  = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    print(f"  ok:    {n_ok}", flush=True)
    print(f"  error: {n_err}", flush=True)
    for r in results:
        if r.error:
            print(f"    [ERR] {r.pid}: {r.error}", flush=True)
        else:
            print(f"    {r.pid}: {r.rows_written:,} rows "
                  f"(dropped {r.rows_dropped_missing_volume:,} missing-vol)", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
