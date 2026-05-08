"""OKX OI coverage audit (#210).

Follow-up to #209's Ch 27 finding: 19/20 pids reported per-pid PSI=0.0000
on the OI channel, meaning the OI series was constant (typically zero)
for at least one half of each pid's window. Before doing further OI
feature work we need to know: per pid, what fraction of OI rows are
zero, and is the missingness clustered at the start (legitimate
backfill gap) or scattered (cache-build defaulting bug).

Pure numpy. CLI hydrates the live cnn_dataset_cache and runs the audit
against the chronologically-sorted Ch 27 terminal-value series for the
top-N survivorship-aware pids.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_OI_CHANNEL = 27
_ZERO_TOL = 1e-12


def coverage_stats(values: np.ndarray) -> dict:
    """Per-series coverage: how much of `values` is zero, where the
    first nonzero is, etc. Treats |x| < _ZERO_TOL as zero.

    Returns a dict with keys: n, n_zero, n_nonzero, frac_zero,
    first_nonzero_idx (None if all zeros / empty).
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0, "n_zero": 0, "n_nonzero": 0,
            "frac_zero": 0.0, "first_nonzero_idx": None,
        }
    is_zero = np.abs(arr) < _ZERO_TOL
    n_zero = int(is_zero.sum())
    n_nonzero = n - n_zero
    nonzero_idxs = np.flatnonzero(~is_zero)
    first_nonzero_idx = int(nonzero_idxs[0]) if nonzero_idxs.size else None
    return {
        "n": n,
        "n_zero": n_zero,
        "n_nonzero": n_nonzero,
        "frac_zero": n_zero / n,
        "first_nonzero_idx": first_nonzero_idx,
    }


def leading_zero_run(values: np.ndarray) -> int:
    """Length of the contiguous zero prefix at the start of `values`.

    Returns 0 if the first value is nonzero, n if all zero.
    """
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0
    is_zero = np.abs(arr) < _ZERO_TOL
    if not is_zero[0]:
        return 0
    nonzero_idxs = np.flatnonzero(~is_zero)
    if nonzero_idxs.size == 0:
        return int(arr.size)
    return int(nonzero_idxs[0])


def per_pid_coverage(prods: Dict[str, dict]) -> List[dict]:
    """One coverage row per pid, sorted by frac_zero desc.

    Each `prods[pid]` must carry `channel` (1-D values, chronological)
    and `ts` (used only for length sanity).
    """
    out: List[dict] = []
    for pid, entry in prods.items():
        ch = np.asarray(entry["channel"], dtype=np.float64).ravel()
        cs = coverage_stats(ch)
        lz = leading_zero_run(ch)
        out.append({
            "pid": pid,
            "n": cs["n"],
            "n_zero": cs["n_zero"],
            "n_nonzero": cs["n_nonzero"],
            "frac_zero": cs["frac_zero"],
            "first_nonzero_idx": cs["first_nonzero_idx"],
            "leading_zeros": lz,
        })
    out.sort(key=lambda r: r["frac_zero"], reverse=True)
    return out


def _load_cache_and_extract_oi() -> Dict[str, dict]:
    """Load cnn_dataset_cache.pt and extract Ch 27 (OI) per-pid series
    sorted chronologically. CLI-only path."""
    import torch  # local import — heavy
    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n

    cache_path = os.path.join(BACKEND, "cnn_dataset_cache.pt")
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    prods = blob["products"]
    snap = recommended_snapshot_ts(prods)
    pids = survivorship_aware_top_n(prods, n=20, snapshot_ts=snap)
    pid_data: Dict[str, dict] = {}
    for pid in pids:
        X, _y, ts = _entry_to_arrays(prods[pid])
        order = np.argsort(ts, kind="stable")
        pid_data[pid] = {
            "channel": X[order, _OI_CHANNEL, -1],
            "ts": ts[order],
        }
    return pid_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    pid_data = _load_cache_and_extract_oi()

    print(f"\nCh {_OI_CHANNEL} (okx_oi) coverage audit", flush=True)
    print("=" * 72, flush=True)
    rows = per_pid_coverage(pid_data)
    print(
        f"  {'pid':<14} {'n':>8} {'n_zero':>8} {'frac_z':>7} "
        f"{'1st_nz':>8} {'lead_z':>8} {'verdict':<22}",
        flush=True,
    )
    for r in rows:
        if r["frac_zero"] >= 0.99:
            verdict = "all-zero"
        elif r["leading_zeros"] == r["n_zero"] and r["n_zero"] > 0:
            verdict = "leading-block (backfill)"
        elif r["n_zero"] == 0:
            verdict = "fully covered"
        else:
            verdict = "scattered (suspect)"
        first_nz = (
            r["first_nonzero_idx"] if r["first_nonzero_idx"] is not None
            else "—"
        )
        print(
            f"  {r['pid']:<14} {r['n']:>8,} {r['n_zero']:>8,} "
            f"{r['frac_zero']:>7.3f} {first_nz!s:>8} "
            f"{r['leading_zeros']:>8,} {verdict:<22}",
            flush=True,
        )

    n_total = sum(r["n"] for r in rows)
    n_zero_total = sum(r["n_zero"] for r in rows)
    print("-" * 72, flush=True)
    print(
        f"  total samples = {n_total:,} | total zero = {n_zero_total:,} "
        f"| frac_zero = {n_zero_total / max(n_total, 1):.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
