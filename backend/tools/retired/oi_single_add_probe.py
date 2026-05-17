"""Single-add probe: replace ch13 (obv_slope, the most marginal noise channel
per #146 ablation) with hourly OKX OI for each pooled-top-20 product, then
measure Δ mean_auc.

Products without OKX SWAP coverage (PENGU/ZK/ONDO/BONK/ALGO/ZORA/PEPE/
DRIFT/XCN/JASMY/HBAR/LRC/STRK/SKL) get zeros for that channel — same as
inference would see in production. This is a faithful test of "would adding
OI as a feature lift AUC at the deployment scale?"

Decision rule: Δ ≥ +0.01 → integrate OI as a real channel (#143-#145).
Otherwise abandon the OKX OI path.

Run:
    cd backend && python tools/oi_single_add_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from services.okx_oi_history import (  # noqa: E402
    _PRODUCT_TO_OKX, fetch_oi_history,
)
from tools.channel_replace import run_replace  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_SEQ_LEN = 60
_TARGET_CHANNEL = 13  # obv_slope — most marginal per #146 (Δ -0.0002)


async def _fetch_oi_aligned(
    product_id: str,
    sample_ts_secs: np.ndarray,
) -> np.ndarray:
    """Fetch OI for product, return aligned [N, 60] z-scored array.

    sample_ts_secs: per-sample bar-end timestamps in seconds.
    Returns zeros if product not on OKX or fetch fails.
    """
    n = len(sample_ts_secs)
    out = np.zeros((n, _SEQ_LEN), dtype=np.float32)
    if product_id not in _PRODUCT_TO_OKX:
        return out

    # Window: 60 bars before earliest sample, through latest sample
    earliest = int(sample_ts_secs.min()) - _SEQ_LEN * _BAR_SECS
    latest = int(sample_ts_secs.max())

    rows = await fetch_oi_history(
        product_id,
        start_ms=earliest * 1000,
        end_ms=latest * 1000,
        bar="1H",
    )
    if not rows:
        return out

    # Build hourly dict: ts_secs -> oi_val
    oi_by_hour: Dict[int, float] = {}
    for ts_ms, oi in rows:
        # Round to hour bucket
        h = (int(ts_ms) // 1000 // _BAR_SECS) * _BAR_SECS
        oi_by_hour[h] = float(oi)

    if not oi_by_hour:
        return out

    # Z-score normalize on the full product series
    vals = np.array(list(oi_by_hour.values()), dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std()) or 1.0

    # Forward-fill: build a sorted hour list, fill gaps
    sorted_hours = sorted(oi_by_hour.keys())
    last_val = mu  # default if no prior data
    filled: Dict[int, float] = {}
    h_idx = 0
    if sorted_hours:
        first_h = sorted_hours[0]
        last_h = sorted_hours[-1]
        for h in range(first_h, last_h + _BAR_SECS, _BAR_SECS):
            if h in oi_by_hour:
                last_val = oi_by_hour[h]
            filled[h] = last_val

    # Build per-sample windows
    for i, ts in enumerate(sample_ts_secs):
        for j in range(_SEQ_LEN):
            h = int(ts) - (_SEQ_LEN - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            v = filled.get(h, mu)
            out[i, j] = (v - mu) / sigma

    return out


def _load_pooled_with_oi(
    n: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load pooled top-N samples plus aligned OI signal. Returns (X, y, ts,
    oi_signal, products_used_for_oi)."""
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    sized = sorted(
        ((pid, len(e.get("X", []))) for pid, e in prods.items()),
        key=lambda x: -x[1],
    )[:n]
    print(f"  pooled top-{n} products: {[pid for pid,_ in sized]}", flush=True)

    Xs, ys, tss, ois = [], [], [], []
    oi_supported: List[str] = []

    async def gather_all():
        tasks = []
        for pid, _ in sized:
            X, y, ts = _entry_to_arrays(prods[pid])
            Xs.append(X)
            ys.append(y)
            tss.append(ts)
            tasks.append(_fetch_oi_aligned(pid, ts))
        return await asyncio.gather(*tasks)

    print("Fetching OI per product (only OKX-supported pairs hit network)...", flush=True)
    t0 = time.time()
    oi_arrays = asyncio.run(gather_all())
    elapsed = time.time() - t0
    print(f"  OI fetch elapsed: {elapsed:.1f}s", flush=True)

    for (pid, _), oi_arr in zip(sized, oi_arrays):
        ois.append(oi_arr)
        coverage = float((oi_arr != 0).any(axis=1).mean())
        if coverage > 0:
            oi_supported.append(f"{pid}({coverage:.0%})")

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    oi_all = np.concatenate(ois, axis=0)

    order = np.argsort(ts_all, kind="stable")
    return (
        X_all[order],
        y_all[order],
        ts_all[order],
        oi_all[order],
        oi_supported,
    )


def main():
    X, y, ts, oi_sig, oi_supported = _load_pooled_with_oi(n=20)
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"OI-covered products: {oi_supported}", flush=True)
    coverage_pct = float((oi_sig != 0).any(axis=1).mean())
    print(f"per-sample OI coverage (any nonzero): {coverage_pct:.1%}", flush=True)

    print(f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with OI z-score; "
          f"running 5-fold purged CV (4h embargo)...", flush=True)
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=oi_sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"\n=== single-add probe: OI -> ch{_TARGET_CHANNEL} ===")
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")


if __name__ == "__main__":
    main()
