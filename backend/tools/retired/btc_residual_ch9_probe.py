"""Single-add probe (#253c): replace Ch 9 (1-bar price change) with the
β-residualized version.

Hypothesis (#253):
    Six prior probes added BTC signal *into* a channel and all missed the
    +0.01 AUC gate (#156, #235, #243, #246-#248). β-residualization
    inverts the polarity:
        r_alt[t] = β[t] · r_btc[t] + ε[t]
        ε[t] = r_alt[t] - β[t] · r_btc[t]   ← alt-specific component
    Replacing Ch 9 (the rawest price-change channel) with ε tests whether
    the BTC-stripped alt return carries the alpha that the BTC-explained
    component does not.

Spec deviations (documented):
    Spec said `β window = 288 bars (24h on 5m candles)`. The probe harness
    consumes hourly parquet history (per #246-#248 convention via
    btc_leadlag_probe._BAR_SECS=3600), so 24h on this grid maps to W=24.
    Same intent — same calendar context — different bar size.

Targets only Ch 9 (1-bar price change). Cache version is NOT bumped:
this is a probe-only diagnostic, not a feature integration. Decision rule
remains the +0.01 AUC gate. If it passes, integration into FeatureBuilder
+ cache rebuild is a separate task (#253d-followup).

Run:
    cd backend && python tools/btc_residual_ch9_probe.py --snapshot-ts auto
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Mapping, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.retired.btc_leadlag_probe import (  # noqa: E402
    _BAR_SECS,
    _CACHE_PATH,
    _HISTORY_DIR,
    _SEQ_LEN,
    _load_pid_closes,
    align_pair,
    build_leadlag_signal,
    log_returns,
)
from tools.btc_residualize import residualize_returns  # noqa: E402


_TARGET_CHANNEL = 9      # Ch 9 = 1-bar price change (per #253a spec)
_BETA_WINDOW = 24        # 24h on hourly grid (spec said 24h; harness is hourly)
_BTC_PID = "BTC-USD"


def _build_residual_signal_for_pid(
    pid: str,
    alt_closes: Mapping[int, float],
    btc_closes: Mapping[int, float],
    window: int = _BETA_WINDOW,
) -> Dict[int, float]:
    """{ts: ε_t} where ε_t = alt_ret_t − β_t · btc_ret_t.

    β_t is fit by `tools.btc_residualize.residualize_returns` on the
    strictly-prior window [t-window, t-1]. NaN warm-up entries are filtered
    out so the dict carries only finite, eligible bars.

    BTC-USD vs itself returns {} (residual is identically zero — no
    information; emitting it would skew downstream z-scoring with a
    flat-zero channel).
    """
    if pid == _BTC_PID:
        return {}

    btc_ret = log_returns(btc_closes)
    alt_ret = log_returns(alt_closes)
    ts, alt_arr, btc_arr = align_pair(alt_ret, btc_ret)
    if len(ts) <= window:
        return {}

    eps = residualize_returns(alt_arr, btc_arr, window=window)
    out: Dict[int, float] = {}
    for t, v in zip(ts, eps):
        if np.isfinite(v):
            out[int(t)] = float(v)
    return out


def _load_pooled_with_residual(
    n: int = 20,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Mirror of btc_leadlag_probe._load_pooled_with_leadlag, but the
    per-pid candidate is the β-residualized 1-bar return."""
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

    print(f"Loading parquet history for top-{n} pids + BTC...", flush=True)
    t0 = time.time()
    btc_closes = _load_pid_closes(_BTC_PID)
    closes_by_pid: Dict[str, Dict[int, float]] = {}
    for pid, _ in sized:
        closes_by_pid[pid] = _load_pid_closes(pid)
    elapsed = time.time() - t0
    cov_pids = sum(1 for c in closes_by_pid.values() if c)
    btc_label = f"{len(btc_closes):,} hours" if btc_closes else "MISSING"
    print(
        f"  parquet coverage: {cov_pids}/{len(top_pids)} pids in "
        f"{elapsed:.1f}s (BTC: {btc_label})",
        flush=True,
    )
    if not btc_closes:
        raise RuntimeError("BTC-USD parquet missing — cannot build residual")

    print(
        f"Computing beta-residual (W={_BETA_WINDOW}h) per pid...",
        flush=True,
    )
    t0 = time.time()
    candidate_by_pid: Dict[str, Dict[int, float]] = {
        pid: _build_residual_signal_for_pid(pid, closes, btc_closes)
        for pid, closes in closes_by_pid.items()
    }
    elapsed = time.time() - t0
    cov_pids = sum(1 for c in candidate_by_pid.values() if c)
    print(
        f"  candidate coverage: {cov_pids}/{len(top_pids)} pids in "
        f"{elapsed:.1f}s (BTC pid skipped by design)",
        flush=True,
    )
    for pid in top_pids:
        c = candidate_by_pid.get(pid, {})
        if c:
            vals = np.array(list(c.values()))
            print(
                f"    {pid}: {len(c):,} ts, mean={vals.mean():+.5f} "
                f"std={vals.std():.5f}",
                flush=True,
            )
        else:
            tag = "skipped (BTC)" if pid == _BTC_PID else "no candidate data"
            print(f"    {pid}: {tag}", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        sig = build_leadlag_signal(
            ts, candidate_by_pid.get(pid, {}), seq_len=_SEQ_LEN,
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


def _run_probe(snapshot_ts: int) -> Dict[str, float]:
    from tools.channel_replace import run_replace

    X, y, ts, sig, used = _load_pooled_with_residual(
        n=20, snapshot_ts=snapshot_ts,
    )
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"products & coverage: {used}", flush=True)
    coverage_pct = float((sig != 0).any(axis=1).mean())
    print(f"per-sample non-zero coverage: {coverage_pct:.1%}", flush=True)

    print(
        f"\nReplacing ch{_TARGET_CHANNEL} (1-bar price change) with beta-residual "
        f"(W={_BETA_WINDOW}h) z-score; running 5-fold purged CV (4h embargo)...",
        flush=True,
    )
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(
        f"\n=== single-add probe: btc_residual_ret -> ch{_TARGET_CHANNEL} ==="
    )
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result["delta"] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")
    return result


def main():
    import torch
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

    _run_probe(snapshot_ts)


if __name__ == "__main__":
    main()
