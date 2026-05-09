"""Single-add probe (#260): replace ch13 (obv_slope, lowest-importance per
#146 ablation) with per-pid log(market_cap) z-scored, sourced from CoinGecko.

Why marketcap:
    Seven sequential BTC-flavored exogenous-input probes (#156, #235, #243,
    #246, #248, #253, plus the L/S coverage-gap retry #235g) all missed the
    +0.01 mean-AUC gate. Marketcap is the first market-structure probe that
    isn't BTC-derived: it captures mc-tier (large-cap vs micro-cap) which
    plausibly correlates with realized volatility, mean-reversion strength,
    and liquidity premium — none of which the existing 28 channels read
    explicitly.

Strict causality:
    CoinGecko stamps daily marketcap at end-of-day UTC. Reading day-D's
    marketcap at sample time t = day-D-anything would leak future close into
    the feature. align_to_hourly applies a default 1-day lag so a row stamped
    at day D is only visible at samples >= day D+1.

Decision rule: Δ ≥ +0.01 mean AUC vs ch13 baseline → integration via
FeatureBuilder + cache version bump (separate task). Δ < +0.01 → log as the
8th sequential exogenous-input failure and consider the existing 28-channel
representation feature-complete for the +0.01 gate.

Run:
    cd backend && python tools/marketcap_probe.py --snapshot-ts auto
"""
from __future__ import annotations

import argparse
import asyncio
import math
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
_TARGET_CHANNEL = 13       # obv_slope — lowest-importance per #146 ablation
_LAG_SECS = 86400          # 1-day strict-causal lag (CoinGecko EOD-UTC)


# ---------------------------------------------------------------------------
# Pure helpers (tested in tests/test_marketcap_probe.py)
# ---------------------------------------------------------------------------

def marketcap_rows_to_log_grid(
    rows: Iterable[Tuple[int, float]],
    hour_grid_ms: Iterable[int],
    lag_secs: int = _LAG_SECS,
) -> Dict[int, float]:
    """Convert raw CoinGecko (ts_ms, market_cap) rows to {bar_ts_secs: log_mc}.

    Pipeline:
      1. Filter out non-positive market_cap (log undefined; CoinGecko sometimes
         returns 0 for pre-launch coins or stale rows).
      2. Forward-fill onto `hour_grid_ms` via services.coingecko_marketcap
         .align_to_hourly with the requested lag (default 1 day).
      3. Apply log-transform to every aligned value.

    Output keys are bar-aligned epoch *seconds* (ms // 1000) to match the
    hourly bar convention used by build_marketcap_signal.
    """
    from services.coingecko_marketcap import align_to_hourly

    valid_rows = [
        (int(ts), float(mc)) for ts, mc in rows
        if mc is not None and float(mc) > 0.0
    ]
    if not valid_rows:
        return {}

    aligned_ms = align_to_hourly(valid_rows, hour_grid_ms, lag_secs=lag_secs)

    return {
        int(ts_ms) // 1000: float(math.log(v))
        for ts_ms, v in aligned_ms.items()
    }


def build_marketcap_signal(
    sample_end_ts: np.ndarray,
    mc_history: Mapping[int, float],
    seq_len: int = _SEQ_LEN,
) -> np.ndarray:
    """Build [N, seq_len] z-scored log-marketcap signal aligned to per-sample
    bar-end timestamps for one product.

    Mirrors okx_ls_probe.build_ls_signal contract:
      - Forward-fill missing hours from prior known value
      - Bars earlier than the first known hour use the mean (z=0)
      - The full known series is z-scored once → channel has unit variance
      - Empty / single-point / zero-std history → all-zero output
    """
    n = len(sample_end_ts)
    out = np.zeros((n, seq_len), dtype=np.float32)
    if not mc_history or len(mc_history) < 2:
        return out

    sorted_hours = sorted(mc_history.keys())
    vals = np.array([mc_history[h] for h in sorted_hours], dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma == 0.0:
        return out

    first_h = sorted_hours[0]
    last_h = sorted_hours[-1]
    filled: Dict[int, float] = {}
    last_val = mu
    for h in range(first_h, last_h + _BAR_SECS, _BAR_SECS):
        if h in mc_history:
            last_val = mc_history[h]
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

async def _fetch_marketcap_for_pids(
    pids: List[str],
    start_ms: int,
    end_ms: int,
    hour_grid_ms: List[int],
) -> Dict[str, Dict[int, float]]:
    """Fetch per-pid marketcap history sequentially (CoinGecko free tier
    rate-limit: ~30 req/min; 20 pids takes ~40s).

    Returns pid -> {bar_ts_secs: log_marketcap}. Pids with no data (unmapped
    slugs, COINGECKO_DISABLED=1, transport errors) get an empty dict, which
    build_marketcap_signal converts to an all-zero feature block.
    """
    from services.coingecko_marketcap import fetch_marketcap_history

    out: Dict[str, Dict[int, float]] = {}
    for pid in pids:
        rows = await fetch_marketcap_history(pid, start_ms, end_ms)
        out[pid] = marketcap_rows_to_log_grid(
            rows, hour_grid_ms=hour_grid_ms, lag_secs=_LAG_SECS,
        )
    return out


def _load_pooled_with_marketcap(
    n: int = 20,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import torch
    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import survivorship_aware_top_n

    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
    sized = [(pid, len(prods[pid].get("X", []))) for pid in top_pids]
    mode = "legacy" if snapshot_ts is None else f"snapshot_ts={snapshot_ts}"
    print(f"  pooled top-{n} ({mode}): {top_pids}", flush=True)

    all_ts = np.concatenate([
        _entry_to_arrays(prods[pid])[2] for pid, _ in sized
    ])
    start_ms = int(all_ts.min() - _SEQ_LEN * _BAR_SECS - _LAG_SECS) * 1000
    end_ms = int(all_ts.max() + _BAR_SECS) * 1000
    print(f"  fetch window: {start_ms} ms .. {end_ms} ms", flush=True)

    grid_secs = sorted({
        ((int(t) // _BAR_SECS) * _BAR_SECS) for t in all_ts
    })
    grid_ms = [t * 1000 for t in grid_secs]

    print("Fetching CoinGecko marketcap history per pid...", flush=True)
    t0 = time.time()
    mc_by_pid = asyncio.run(
        _fetch_marketcap_for_pids(top_pids, start_ms, end_ms, grid_ms)
    )
    elapsed = time.time() - t0
    cov_pids = sum(1 for h in mc_by_pid.values() if h)
    print(
        f"  marketcap coverage: {cov_pids}/{len(top_pids)} pids in "
        f"{elapsed:.1f}s",
        flush=True,
    )
    for pid in top_pids:
        h = mc_by_pid.get(pid, {})
        if h:
            vals = np.array(list(h.values()))
            print(
                f"    {pid}: {len(h):,} hours, log_mc mean={vals.mean():.2f} "
                f"std={vals.std():.3f}",
                flush=True,
            )
        else:
            print(f"    {pid}: no marketcap data", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        sig = build_marketcap_signal(
            ts, mc_by_pid.get(pid, {}), seq_len=_SEQ_LEN,
        )
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
    parser.add_argument(
        "--snapshot-ts", type=str, default=None,
        help="Survivorship-aware top-N selection cutoff: 'auto' (median "
             "first_ts), an integer epoch seconds, or omit for legacy "
             "behavior (#163).",
    )
    args = parser.parse_args()

    snapshot_ts: int = None
    if args.snapshot_ts is not None:
        if args.snapshot_ts == "auto":
            blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
            snapshot_ts = recommended_snapshot_ts(blob["products"])
            print(
                f"snapshot_ts=auto -> {snapshot_ts} (median first_ts across "
                f"{len(blob['products'])} products)",
                flush=True,
            )
        else:
            snapshot_ts = int(args.snapshot_ts)
            print(f"snapshot_ts={snapshot_ts}", flush=True)

    X, y, ts, sig, used = _load_pooled_with_marketcap(
        n=20, snapshot_ts=snapshot_ts,
    )
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"products & marketcap coverage: {used}", flush=True)
    coverage_pct = float((sig != 0).any(axis=1).mean())
    print(f"per-sample non-zero marketcap coverage: {coverage_pct:.1%}", flush=True)

    print(
        f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with per-pid log_marketcap "
        f"z-score (1-day lag); running 5-fold purged CV (4h embargo)...",
        flush=True,
    )
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(
        f"\n=== single-add probe: log_marketcap -> ch{_TARGET_CHANNEL} ==="
    )
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result["delta"] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")


if __name__ == "__main__":
    main()
