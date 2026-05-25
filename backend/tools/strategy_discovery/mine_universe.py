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

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

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
    workers: int = 4,
) -> Dict[int, List[LeafProfile]]:
    """Iterate (pid, horizon) cross-product; collect profiles per horizon.

    Parallelism: `workers` threads each call mine_profiles_for_pid_horizon
    concurrently. PyTorch CUDA ops release the GIL, so kernel-launch overhead
    is amortized across threads even on CPython. Default workers=4 fits the
    RTX 2060 6GB budget (each tree-fit holds ~200MB of GPU tensors).

    Tree fits are independent per (pid, horizon) — no shared state. Parquet
    writes happen once-per-horizon at the end (single thread) to avoid the
    read-then-rewrite race that per-pid streaming writes would create.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pids = pids_from_universe_json(Path(universe_path))
    all_profiles: Dict[int, List[LeafProfile]] = {int(h): [] for h in horizons}
    rule_paths_per_horizon: Dict[int, Dict[str, str]] = {int(h): {} for h in horizons}
    # Group profiles per horizon so we can write each output parquet once at the end.
    profiles_per_horizon: Dict[int, List[tuple]] = {int(h): [] for h in horizons}
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for pid in pids:
        parquet_path = phase2_dir / f"{pid}.parquet"
        if not parquet_path.exists():
            continue
        for h in horizons:
            tasks.append((pid, int(h), parquet_path))

    def _run_one(task_tuple):
        pid, h, parquet_path = task_tuple
        profiles = mine_profiles_for_pid_horizon(
            pid=pid, horizon=h, parquet_path=parquet_path,
            device=device, seed=seed,
        )
        return pid, h, profiles

    n_workers = max(1, int(workers))
    print(f"  total tasks: {len(tasks)}  workers: {n_workers}  device: {device}",
          flush=True)

    start_time = time.time()
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                pid, h, profiles = fut.result()
            except Exception as e:
                src = futures[fut]
                print(f"  [ERR] {src[0]} h{src[1]}: {e!r}", flush=True)
                continue
            completed += 1
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            eta = (len(tasks) - completed) / rate if rate > 0 else 0.0
            all_profiles[h].extend(profiles)
            if profiles:
                profiles_per_horizon[h].append((pid, profiles))
                for p in profiles:
                    rule_paths_per_horizon[h][f"{pid}__{p.leaf_id}"] = p.rule_path_summary
            print(f"  [{completed:3d}/{len(tasks)}] {pid:14s} h{h:3d}: "
                  f"{len(profiles):3d} profiles  elapsed={elapsed:6.0f}s ETA={eta:6.0f}s",
                  flush=True)

    # Write per-horizon parquets in a single sequential pass (no race).
    for h, pid_profile_pairs in profiles_per_horizon.items():
        if not pid_profile_pairs:
            continue
        out_path = output_dir / f"profiles_h{int(h)}.parquet"
        for pid, profiles in pid_profile_pairs:
            write_profile_parquet(
                profiles, pid=pid, horizon=int(h), output_path=out_path,
            )

    # Write rule_paths sidecars.
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
        default=str(Path(BACKEND).parent / "docs" / "superpowers" / "specs" / "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--phase2-dir",  default=str(_DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir",  default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel (pid, horizon) workers (default 4)")
    args = parser.parse_args(argv)
    profiles = mine_universe(
        universe_path=Path(args.universe),
        phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
        workers=args.workers,
    )
    for h, plist in sorted(profiles.items()):
        print(f"  h{h}: {len(plist)} profiles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
