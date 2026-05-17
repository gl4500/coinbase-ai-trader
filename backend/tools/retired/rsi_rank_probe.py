"""Single-add probe (#162): replace ch13 (obv_slope, marginal per #146) with
the cross-sectional rank of a chosen source channel — at each per-bar
timestamp, the percentile rank of this product's value vs all other
products' values at the same hour.

Default source channel is RSI (Ch 4). Override with `--source-channel N`
to scout other candidates (e.g. Ch 1 volume = BTC-dominance proxy for #156,
Ch 12 MFI, Ch 22 funding sin component, etc.) — a channel-agnostic
cross-section harness.

Hypothesis: cross-sectional position carries information that no single
product's 60-bar window can reconstruct. Integration gate: Δ ≥ +0.01.

Run:
    cd backend && python tools/rsi_rank_probe.py
    cd backend && python tools/rsi_rank_probe.py --source-channel 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.channel_replace import run_replace  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.pid_snapshot import (  # noqa: E402
    recommended_snapshot_ts,
    survivorship_aware_top_n,
)

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_SEQ_LEN = 60
_DEFAULT_SOURCE_CHANNEL = 4    # Ch 4 = RSI(14) / 100 (default — #162 winner)
_TARGET_CHANNEL = 13           # obv_slope — most marginal per #146 (Δ -0.0002)


def _cross_sectional_rank(rsi_by_pid: Dict[str, float]) -> Dict[str, float]:
    """Percentile-rank RSI values across products at one timestamp.

    Returns rank in [0, 1]. NaN inputs are dropped from both numerator and
    denominator. Single product → 0.5 (neutral). Ties get the average rank.
    """
    if not rsi_by_pid:
        return {}
    valid = {p: v for p, v in rsi_by_pid.items() if not np.isnan(v)}
    n = len(valid)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(valid)): 0.5}
    # Average rank, normalized to [0, 1]: rank in [1, n] → (avg_rank - 1) / (n - 1)
    pids = list(valid.keys())
    vals = np.array([valid[p] for p in pids], dtype=np.float64)
    # average ranks
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0  # 0-indexed average of tied positions
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    norm = ranks / (n - 1)
    return {pids[k]: float(norm[k]) for k in range(n)}


def build_rank_signal(
    target_pid: str,
    sample_end_ts: np.ndarray,
    rsi_by_pid: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Build [N, 60] rank signal for `target_pid`.

    rsi_by_pid: pid -> (rsi_series, ts_series), where ts_series is the
        bar-end timestamps in seconds matching rsi_series 1:1.
    sample_end_ts: per-sample bar-end timestamps in seconds, length N.
    Returns float32 array shape (N, _SEQ_LEN), values in [0, 1] with 0.5
    fallback when no cross-sectional data is available at a given hour.
    """
    n = len(sample_end_ts)
    out = np.full((n, _SEQ_LEN), 0.5, dtype=np.float32)
    if target_pid not in rsi_by_pid:
        return out

    # Pre-build hour-bucket lookups: pid -> dict[ts_hour -> rsi_val]
    by_hour: Dict[str, Dict[int, float]] = {}
    for pid, (rsi_series, ts_series) in rsi_by_pid.items():
        d: Dict[int, float] = {}
        for r, t in zip(rsi_series.tolist(), ts_series.tolist()):
            h = (int(t) // _BAR_SECS) * _BAR_SECS
            d[h] = float(r)
        by_hour[pid] = d

    for i, ts in enumerate(sample_end_ts):
        for j in range(_SEQ_LEN):
            h = int(ts) - (_SEQ_LEN - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            snap: Dict[str, float] = {}
            for pid, d in by_hour.items():
                v = d.get(h)
                if v is not None and not np.isnan(v):
                    snap[pid] = v
            ranks = _cross_sectional_rank(snap)
            r = ranks.get(target_pid)
            if r is not None:
                out[i, j] = float(r)
    return out


def _load_pooled_with_ranks(
    n: int = 20,
    source_channel: int = _DEFAULT_SOURCE_CHANNEL,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load pooled top-N samples and build aligned cross-sectional rank signal
    from `source_channel` via a vectorized (T, P) matrix. Returns (X, y, ts,
    rank_sig, products_used).

    `snapshot_ts=None` (default) preserves the legacy `len(entry["X"])`
    ordering. Pass an integer cutoff (or sentinel from
    `recommended_snapshot_ts`) to engage survivorship-aware selection.
    """
    from scipy.stats import rankdata

    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    print(f"Building (T, P) matrix for source ch={source_channel}...", flush=True)
    t0 = time.time()
    pid_list: List[str] = []
    rsi_per_pid: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_hours: set = set()
    for pid, e in prods.items():
        X_p, _, ts_p = _entry_to_arrays(e)
        if X_p.shape[0] == 0:
            continue
        rsi_chan = X_p[:, source_channel, :].astype(np.float32)  # [n, 60]
        offsets = (np.arange(_SEQ_LEN)[::-1] * _BAR_SECS).astype(np.int64)
        bar_ts = (ts_p[:, None] - offsets[None, :]).astype(np.int64)  # [n, 60]
        flat_ts = bar_ts.reshape(-1)
        flat_rsi = rsi_chan.reshape(-1)
        # Dedupe per pid: keep first occurrence per hour
        order = np.argsort(flat_ts, kind="stable")
        ts_s = flat_ts[order]
        rsi_s = flat_rsi[order]
        uniq = np.concatenate(([True], np.diff(ts_s) != 0))
        ts_u = ts_s[uniq]
        rsi_u = rsi_s[uniq]
        rsi_per_pid[pid] = (ts_u, rsi_u)
        all_hours.update(ts_u.tolist())
        pid_list.append(pid)
    pid_idx = {p: j for j, p in enumerate(pid_list)}
    sorted_hours = np.array(sorted(all_hours), dtype=np.int64)
    h_idx_map = {int(h): i for i, h in enumerate(sorted_hours.tolist())}
    T = len(sorted_hours)
    P = len(pid_list)
    print(f"  unique hours T={T:,}  products P={P}  build={time.time()-t0:.1f}s",
          flush=True)

    # Step 2: fill matrix
    M = np.full((T, P), np.nan, dtype=np.float32)
    t0 = time.time()
    for pid, (ts_u, rsi_u) in rsi_per_pid.items():
        j = pid_idx[pid]
        idxs = np.searchsorted(sorted_hours, ts_u)
        M[idxs, j] = rsi_u
    print(f"  matrix fill: {time.time()-t0:.1f}s", flush=True)

    # Step 3: per-row rank, normalized to [0, 1] with 0.5 fallback
    t0 = time.time()
    R = np.full_like(M, 0.5)
    valid = ~np.isnan(M)
    n_valid = valid.sum(axis=1)
    multi = n_valid > 1
    rows = np.where(multi)[0]
    for i in rows:
        v = valid[i]
        vals = M[i, v]
        ranks = rankdata(vals, method="average")
        nv = len(vals)
        normed = (ranks - 1) / (nv - 1)
        R[i, v] = normed.astype(np.float32)
    print(f"  per-row ranking: {time.time()-t0:.1f}s ({len(rows):,} rows)",
          flush=True)

    # Step 4: build per-sample rank windows for top-N products
    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
    sized = [(pid, len(prods[pid].get("X", []))) for pid in top_pids]
    mode = "legacy" if snapshot_ts is None else f"snapshot_ts={snapshot_ts}"
    print(f"  pooled top-{n} ({mode}): {top_pids}", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    t0 = time.time()
    for pid, _ in sized:
        X_p, y_p, ts_p = _entry_to_arrays(prods[pid])
        if pid not in pid_idx:
            sig = np.full((len(ts_p), _SEQ_LEN), 0.5, dtype=np.float32)
        else:
            j = pid_idx[pid]
            offsets = (np.arange(_SEQ_LEN)[::-1] * _BAR_SECS).astype(np.int64)
            bar_ts = (ts_p[:, None] - offsets[None, :]).astype(np.int64)
            idxs = np.searchsorted(sorted_hours, bar_ts.reshape(-1))
            # Mask out-of-range
            in_range = (idxs < T) & (sorted_hours[np.clip(idxs, 0, T - 1)]
                                     == bar_ts.reshape(-1))
            flat = np.full(bar_ts.size, 0.5, dtype=np.float32)
            flat[in_range] = R[idxs[in_range], j]
            sig = flat.reshape(bar_ts.shape)
        Xs.append(X_p)
        ys.append(y_p)
        tss.append(ts_p)
        sigs.append(sig)
        coverage = float((sig != 0.5).any(axis=1).mean())
        products_used.append(f"{pid}({coverage:.0%})")
    print(f"  rank signal build: {time.time()-t0:.1f}s", flush=True)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-channel", type=int,
                        default=_DEFAULT_SOURCE_CHANNEL,
                        help="Source channel to rank cross-sectionally "
                             "(default 4 = RSI; e.g. 1 = log volume).")
    parser.add_argument("--snapshot-ts", type=str, default=None,
                        help="Survivorship-aware top-N selection cutoff: "
                             "'auto' (median first_ts), an integer epoch "
                             "seconds, or omit for legacy behavior (#163).")
    args = parser.parse_args()
    src = int(args.source_channel)

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

    X, y, ts, rank_sig, used = _load_pooled_with_ranks(
        n=20, source_channel=src, snapshot_ts=snapshot_ts)
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"products & rank coverage: {used}", flush=True)
    coverage_pct = float((rank_sig != 0.5).any(axis=1).mean())
    print(f"per-sample non-neutral rank coverage: {coverage_pct:.1%}", flush=True)

    print(f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with cross-sectional "
          f"rank of ch{src}; running 5-fold purged CV (4h embargo)...", flush=True)
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=rank_sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"\n=== single-add probe: ch{src}-rank -> ch{_TARGET_CHANNEL} ===")
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")


if __name__ == "__main__":
    main()
