"""Sanity check: at horizon=4h, does the fresh relabel match the cache's
pre-baked y for the same samples?

If the labels match, the +0.12 AUC gap is in the pipeline.
If they differ, the gap is in the labeling — possibly leakage or stale parquet.
"""
from __future__ import annotations

import os
import sys
import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from services.history_backfill import load_history  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.timescale_sweep import _relabel_at_horizon  # noqa: E402

CACHE = os.path.join(BACKEND, "cnn_dataset_cache.pt")
HORIZON = 4


def main():
    blob = torch.load(CACHE, map_location="cpu", weights_only=False)
    prods = blob["products"]
    sized = sorted(
        ((pid, len(e.get("X", []))) for pid, e in prods.items()),
        key=lambda x: -x[1],
    )[:5]  # smaller sample for sanity
    pids = [pid for pid, _ in sized]

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
