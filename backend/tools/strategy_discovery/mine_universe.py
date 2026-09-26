"""Phase 3 universe driver — iterate (pid, horizon) pairs, write outputs.

Loads phase2 parquets, dispatches mine_profiles per (pid, horizon), and
writes profiles_h{h}.parquet + rule_paths_h{h}.json + mining_summary.md.

The ONLY module in Phase 3 that touches the filesystem.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.mine_profiles import LeafProfile  # noqa: E402

_DEFAULT_HORIZONS = (1, 4, 24, 72, 168)
_SCHEMA_VERSION = 1
_DEFAULT_PHASE2_DIR = Path(BACKEND) / "data" / "phase2"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase3"


def pids_from_universe_json(universe_path: Path) -> List[str]:
    """Flatten {cohort: [pids]} into a deduplicated sorted pid list."""
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def write_profile_parquet(
    profiles: List[LeafProfile],
    *,
    pid: str,
    horizon: int,
    output_path: Path,
) -> None:
    """Write a list of LeafProfile rows to parquet, adding pid + horizon + schema_version.

    Appends to the existing file if it exists (multiple pids land in same per-horizon file).
    """
    rows = []
    for p in profiles:
        d = asdict(p)
        d["pid"] = pid
        d["horizon"] = int(horizon)
        d["schema_version"] = _SCHEMA_VERSION
        rows.append(d)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pq.read_table(output_path).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(output_path, compression="snappy", index=False)


from tools.strategy_discovery.mine_profiles import mine_profiles_for_pid_horizon  # noqa: E402


def mine_universe(
    *,
    universe_path: Path,
    phase2_dir: Path = _DEFAULT_PHASE2_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    horizons=_DEFAULT_HORIZONS,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[int, List[LeafProfile]]:
    """Iterate (pid, horizon) cross-product; collect profiles per horizon."""
    import time

    pids = pids_from_universe_json(Path(universe_path))
    all_profiles: Dict[int, List[LeafProfile]] = {int(h): [] for h in horizons}
    rule_paths_per_horizon: Dict[int, Dict[str, str]] = {int(h): {} for h in horizons}
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks: List[tuple] = []
    for pid in pids:
        parquet_path = phase2_dir / f"{pid}.parquet"
        if not parquet_path.exists():
            continue
        for h in horizons:
            tasks.append((pid, int(h), parquet_path))

    print(f"  total tasks: {len(tasks)}  device: {device}", flush=True)
    start_time = time.time()

    for idx, (pid, h, parquet_path) in enumerate(tasks, start=1):
        t0 = time.time()
        profiles = mine_profiles_for_pid_horizon(
            pid=pid,
            horizon=h,
            parquet_path=parquet_path,
            device=device,
            seed=seed,
        )
        task_secs = time.time() - t0
        all_profiles[h].extend(profiles)
        if profiles:
            write_profile_parquet(
                profiles,
                pid=pid,
                horizon=h,
                output_path=output_dir / f"profiles_h{h}.parquet",
            )
            for p in profiles:
                rule_paths_per_horizon[h][f"{pid}__{p.leaf_id}"] = p.rule_path_summary
        elapsed = time.time() - start_time
        rate = idx / elapsed if elapsed > 0 else 0.0
        eta = (len(tasks) - idx) / rate if rate > 0 else 0.0
        print(
            f"  [{idx:3d}/{len(tasks)}] {pid:14s} h{h:3d}: "
            f"{len(profiles):3d} profiles  task={task_secs:5.0f}s  "
            f"elapsed={elapsed:6.0f}s  ETA={eta:6.0f}s",
            flush=True,
        )

    for h, paths in rule_paths_per_horizon.items():
        if paths:
            with open(output_dir / f"rule_paths_h{int(h)}.json", "w", encoding="utf-8") as f:
                json.dump(paths, f, indent=2)
    return all_profiles


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Mine Phase 3 profiles for a universe.")
    parser.add_argument(
        "--universe",
        default=str(
            Path(BACKEND).parent / "docs" / "superpowers" / "specs" / "2026-05-23-universe-50.json"
        ),
    )
    parser.add_argument("--phase2-dir", default=str(_DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    profiles = mine_universe(
        universe_path=Path(args.universe),
        phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
    )
    for h, plist in sorted(profiles.items()):
        print(f"  h{h}: {len(plist)} profiles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
