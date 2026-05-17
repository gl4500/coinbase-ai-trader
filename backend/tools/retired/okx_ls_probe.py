"""Single-add probe (#235): replace ch13 (obv_slope, the most marginal noise
channel per #146 ablation) with the per-pid OKX long/short *account* ratio
(z-scored), then measure Δ mean_auc against the +0.01 gate.

Why long/short ratio:
    OKX OI alone (Ch 27, #143–#145) didn't lift the pooled-top-20 AUC past
    0.55. L/S account ratio is a different OKX endpoint that measures retail
    positioning skew — ratio > 1 means more accounts are net-long than
    net-short. Hypothesis: extreme positioning crowding precedes mean
    reversion, which the per-product 60-bar window can't reconstruct from
    price/volume alone.

Decision rule: Δ ≥ +0.01 mean AUC → bundle with #162 RSI-rank into a single
retrain cycle. Δ < +0.01 → document as the third L/S-style probe failure
and stop (alongside MFI-rank, log10-vol-rank).

Run:
    cd backend && python tools/okx_ls_probe.py
    cd backend && python tools/okx_ls_probe.py --snapshot-ts auto
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_SEQ_LEN = 60
_TARGET_CHANNEL = 13   # obv_slope — most marginal per #146


# ---------------------------------------------------------------------------
# Pure functions (tested in tests/test_okx_ls_probe.py)
# ---------------------------------------------------------------------------

def ls_history_to_bar_grid(
    rows: Iterable[Tuple[int, float]],
) -> Dict[int, float]:
    """Convert OKX (ts_ms, ratio) rows to a {bar_ts_secs: ratio} dict.

    Off-grid timestamps (not exact `_BAR_SECS` boundaries) are dropped — OKX
    occasionally returns partial-bar stamps that don't align with our hourly
    bar grid.
    """
    out: Dict[int, float] = {}
    for ts_ms, ratio in rows:
        secs = int(ts_ms) // 1000
        if secs % _BAR_SECS != 0:
            continue
        out[secs] = float(ratio)
    return out


def build_ls_signal(
    sample_end_ts: np.ndarray,
    ls_history: Mapping[int, float],
    seq_len: int = _SEQ_LEN,
) -> np.ndarray:
    """Build [N, seq_len] z-scored L/S ratio signal aligned to per-sample
    bar-end timestamps for one product.

    Mirrors `btc_dominance_probe.build_btc_dom_signal` semantics:
      - Forward-fill missing hours from prior known value
      - Bars earlier than the first known hour use the mean (z=0)
      - The full known series is z-scored once → channel has unit variance
      - Empty / single-point / zero-std history → all-zero output
    """
    n = len(sample_end_ts)
    out = np.zeros((n, seq_len), dtype=np.float32)
    if not ls_history or len(ls_history) < 2:
        return out

    sorted_hours = sorted(ls_history.keys())
    vals = np.array([ls_history[h] for h in sorted_hours], dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma == 0.0:
        return out

    first_h = sorted_hours[0]
    last_h = sorted_hours[-1]
    filled: Dict[int, float] = {}
    last_val = mu
    for h in range(first_h, last_h + _BAR_SECS, _BAR_SECS):
        if h in ls_history:
            last_val = ls_history[h]
        filled[h] = last_val

    for i, end_ts in enumerate(sample_end_ts):
        for j in range(seq_len):
            h = int(end_ts) - (seq_len - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            if h < first_h:
                v = mu
            else:
                v = filled.get(h, mu)
            out[i, j] = (v - mu) / sigma

    return out


# ---------------------------------------------------------------------------
# IO + probe runner
# ---------------------------------------------------------------------------

async def _fetch_ls_for_pids(
    pids: List[str],
    start_ms: int,
    end_ms: int,
) -> Dict[str, Dict[int, float]]:
    """Fetch per-pid L/S history sequentially (avoids hammering OKX rate limit).

    Returns pid -> {bar_ts_secs: ratio}.
    """
    from services.okx_long_short_history import fetch_long_short_ratio_history
    out: Dict[str, Dict[int, float]] = {}
    for pid in pids:
        rows = await fetch_long_short_ratio_history(pid, start_ms, end_ms)
        out[pid] = ls_history_to_bar_grid(rows)
    return out


def _load_pooled_with_ls(
    n: int = 20,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import torch
    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import survivorship_aware_top_n
    """Load pooled top-N samples + aligned per-pid L/S ratio signal.

    Unlike btc_dominance (single market-wide series), L/S is per-product so
    each pid gets its own z-scored signal block.
    """
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
    sized = [(pid, len(prods[pid].get("X", []))) for pid in top_pids]
    mode = "legacy" if snapshot_ts is None else f"snapshot_ts={snapshot_ts}"
    print(f"  pooled top-{n} ({mode}): {top_pids}", flush=True)

    # Determine aggregate fetch window across all top pids
    all_ts = np.concatenate([
        _entry_to_arrays(prods[pid])[2] for pid, _ in sized
    ])
    start_ms = int(all_ts.min() - _SEQ_LEN * _BAR_SECS) * 1000
    end_ms = int(all_ts.max() + _BAR_SECS) * 1000
    print(f"  fetch window: {start_ms} ms .. {end_ms} ms", flush=True)

    print("Fetching OKX L/S history per pid...", flush=True)
    t0 = time.time()
    ls_by_pid = asyncio.run(_fetch_ls_for_pids(top_pids, start_ms, end_ms))
    elapsed = time.time() - t0
    cov_pids = sum(1 for h in ls_by_pid.values() if h)
    print(f"  L/S coverage: {cov_pids}/{len(top_pids)} pids in {elapsed:.1f}s",
          flush=True)
    for pid in top_pids:
        h = ls_by_pid.get(pid, {})
        if h:
            vals = np.array(list(h.values()))
            print(f"    {pid}: {len(h):,} hours, ratio mean={vals.mean():.3f} "
                  f"std={vals.std():.3f}", flush=True)
        else:
            print(f"    {pid}: no L/S data", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        sig = build_ls_signal(ts, ls_by_pid.get(pid, {}), seq_len=_SEQ_LEN)
        Xs.append(X)
        ys.append(y)
        tss.append(ts)
        sigs.append(sig)
        coverage = float((sig != 0).any(axis=1).mean())
        products_used.append(f"{pid}({coverage:.0%})")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    sig_all = np.concatenate(sigs, axis=0)

    order = np.argsort(ts_all, kind="stable")
    return (
        X_all[order],
        y_all[order],
        ts_all[order],
        sig_all[order],
        products_used,
    )


def main():
    import torch
    from tools.channel_replace import run_replace
    from tools.pid_snapshot import recommended_snapshot_ts

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-ts", type=str, default=None,
                        help="Survivorship-aware top-N selection cutoff: "
                             "'auto' (median first_ts), an integer epoch "
                             "seconds, or omit for legacy behavior (#163).")
    args = parser.parse_args()

    snapshot_ts: int = None
    if args.snapshot_ts is not None:
        if args.snapshot_ts == "auto":
            blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
            snapshot_ts = recommended_snapshot_ts(blob["products"])
            print(f"snapshot_ts=auto -> {snapshot_ts} "
                  f"(median first_ts across {len(blob['products'])} products)",
                  flush=True)
        else:
            snapshot_ts = int(args.snapshot_ts)
            print(f"snapshot_ts={snapshot_ts}", flush=True)

    X, y, ts, sig, used = _load_pooled_with_ls(n=20, snapshot_ts=snapshot_ts)
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"products & L/S coverage: {used}", flush=True)
    coverage_pct = float((sig != 0).any(axis=1).mean())
    print(f"per-sample non-zero L/S coverage: {coverage_pct:.1%}", flush=True)

    print(f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with per-pid L/S ratio "
          f"z-score; running 5-fold purged CV (4h embargo)...", flush=True)
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"\n=== single-add probe: OKX L/S -> ch{_TARGET_CHANNEL} ===")
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")


if __name__ == "__main__":
    main()
