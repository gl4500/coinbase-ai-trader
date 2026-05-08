"""Sanity check: at horizon=4h, does the fresh relabel match the cache's
pre-baked y for the same samples?

If the labels match, the +0.12 AUC gap is in the pipeline.
If they differ, the gap is in the labeling — possibly leakage or stale parquet.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from services.history_backfill import load_history  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.pid_snapshot import survivorship_aware_top_n  # noqa: E402
from tools.timescale_sweep import _relabel_at_horizon  # noqa: E402

CACHE = os.path.join(BACKEND, "cnn_dataset_cache.pt")
HORIZON = 4


def _pick_pids(prods: dict, n: int, snapshot_ts: Optional[int] = None) -> List[str]:
    """Top-n pid selection. `snapshot_ts=None` reproduces legacy
    `len(entry["X"])` ranking; pass an int to opt into survivorship-aware
    selection per #163."""
    return survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)


def main():
    blob = torch.load(CACHE, map_location="cpu", weights_only=False)
    prods = blob["products"]
    pids = _pick_pids(prods, n=5, snapshot_ts=None)  # smaller sample for sanity

    total_n = total_match = total_both_valid = 0
    print(f"Comparing fresh relabel vs cache y at horizon={HORIZON}h", flush=True)
    print(f"{'pid':<14} {'n':>8} {'fresh_valid':>12} {'cache_y_pos':>12} "
          f"{'fresh_y_pos':>12} {'agree':>8} {'agree%':>8}", flush=True)
    for pid in pids:
        entry = prods[pid]
        X, y_cache, ts = _entry_to_arrays(entry)
        candles = load_history(pid) or []
        if not candles:
            continue
        y_fresh, mask = _relabel_at_horizon(candles, ts, HORIZON)

        n = len(y_cache)
        valid = int(mask.sum())
        agree = int((y_cache[mask] == y_fresh[mask]).sum())

        total_n += n
        total_match += agree
        total_both_valid += valid

        print(f"{pid:<14} {n:>8,} {valid:>12,} "
              f"{y_cache.mean():>12.3f} "
              f"{(y_fresh[mask].mean() if valid else 0):>12.3f} "
              f"{agree:>8,} "
              f"{(100.0*agree/valid if valid else 0):>7.2f}%", flush=True)

    print(f"\nTOTAL: cached samples={total_n:,}  "
          f"fresh-valid={total_both_valid:,}  "
          f"matches={total_match:,}  "
          f"agree={100.0*total_match/total_both_valid:.2f}%", flush=True)


if __name__ == "__main__":
    main()
