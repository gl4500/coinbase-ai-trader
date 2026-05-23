"""Single-add probe (#243-#245): replace ch13 (obv_slope) with a long-horizon
trend feature, then measure Δ mean_auc against the +0.01 gate.

Why long-horizon trend (#243):
    The 28-channel feature set has no multi-week trend feature. Longest MA
    distance is `ema21_dist` (21 bars on 1h ≈ 0.9 days). Daily/weekly trend
    drives crypto regime; the existing channels can't see it. This probe
    tests whether SMA50/SMA200 distance or a golden-cross binary lifts AUC
    past the existing 0.55 ceiling.

Candidates (selected via --candidate):
    sma50_1h        SMA50 on hourly bars (~2 days)
    sma200_1h       SMA200 on hourly bars (~8.3 days)
    sma50_d1        SMA50 on daily-resampled closes (~50 days)
    sma200_d1       SMA200 on daily-resampled closes (~200 days)
    golden_cross_d1 sign(SMA50_d1 - SMA200_d1) ∈ {-1, 0, +1}

Decision rule: best candidate Δ ≥ +0.01 → integrate as a real channel
(rebuild cache + retrain). Otherwise document as 5th probe failure and stop.

Run:
    cd backend && python tools/long_trend_probe.py --snapshot-ts auto
    cd backend && python tools/long_trend_probe.py --snapshot-ts auto \\
                    --candidate golden_cross_d1
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_BAR_SECS = 3600
_DAY_SECS = 86400
_SEQ_LEN = 60
_TARGET_CHANNEL = 13   # obv_slope — most marginal per #146

_CANDIDATES = (
    "sma50_1h",
    "sma200_1h",
    "sma50_d1",
    "sma200_d1",
    "golden_cross_d1",
)


# ---------------------------------------------------------------------------
# Pure functions (tested in tests/test_long_trend_probe.py)
# ---------------------------------------------------------------------------

def compute_sma_dist_series(
    closes_by_hour: Mapping[int, float],
    period_bars: int,
) -> Dict[int, float]:
    """Per-bar (close - SMA_period) / SMA_period over the bars in
    `closes_by_hour`, computed in chronological order across the input keys.

    Bars in the warm-up region (fewer than `period_bars` priors available)
    are NOT included in the output. SMA is computed over the most recent
    `period_bars` known closes, irrespective of calendar gaps — this matches
    how SMA is normally read from a price chart with non-trading hours
    collapsed.
    """
    if not closes_by_hour or period_bars <= 0:
        return {}

    keys = sorted(closes_by_hour.keys())
    vals = [float(closes_by_hour[k]) for k in keys]
    out: Dict[int, float] = {}

    for i in range(period_bars - 1, len(keys)):
        window = vals[i - period_bars + 1: i + 1]
        sma = sum(window) / period_bars
        if sma == 0.0:
            continue
        out[int(keys[i])] = (vals[i] - sma) / sma
    return out


def daily_resample(closes_by_hour: Mapping[int, float]) -> Dict[int, float]:
    """Pick the last (highest-timestamp) close within each UTC day.

    Day boundary uses unix epoch (epoch second 0 == midnight UTC). For each
    day, the entry is keyed by the ACTUAL observation timestamp (the highest
    in-day hour available), not the day-start. Keying at day-start would let
    later builders forward-fill a same-day SMA from 23:00 onto a 14:00
    sample → ~9 hours of lookahead leak.
    """
    if not closes_by_hour:
        return {}
    by_day: Dict[int, Tuple[int, float]] = {}
    for ts, close in closes_by_hour.items():
        ts_i = int(ts)
        day = (ts_i // _DAY_SECS) * _DAY_SECS
        prev = by_day.get(day)
        if prev is None or ts_i > prev[0]:
            by_day[day] = (ts_i, float(close))
    return {v[0]: v[1] for v in by_day.values()}


def golden_cross_signal(
    closes_by_ts: Mapping[int, float],
    fast: int = 50,
    slow: int = 200,
) -> Dict[int, int]:
    """sign(SMA_fast - SMA_slow) at each timestamp where both SMAs are warm.

    Output values are -1, 0, or +1. Spacing of input timestamps is preserved
    in the output — pass daily-resampled closes for a true golden-cross.
    Warm-up: needs `slow` bars accumulated → first eligible index is `slow-1`.
    """
    if not closes_by_ts or slow <= 0 or fast <= 0:
        return {}
    keys = sorted(closes_by_ts.keys())
    vals = [float(closes_by_ts[k]) for k in keys]
    if len(vals) < slow:
        return {}
    out: Dict[int, int] = {}
    for i in range(slow - 1, len(vals)):
        sma_fast = sum(vals[i - fast + 1: i + 1]) / fast
        sma_slow = sum(vals[i - slow + 1: i + 1]) / slow
        diff = sma_fast - sma_slow
        if diff > 0:
            out[int(keys[i])] = 1
        elif diff < 0:
            out[int(keys[i])] = -1
        else:
            out[int(keys[i])] = 0
    return out


def build_trend_signal(
    sample_end_ts: np.ndarray,
    value_by_ts: Mapping[int, float],
    seq_len: int = _SEQ_LEN,
) -> np.ndarray:
    """Build [N, seq_len] z-scored trend signal aligned to per-sample bar-end
    timestamps.

    Mirrors `btc_dominance_probe.build_btc_dom_signal` semantics:
      - Forward-fill missing hours from prior known value
      - Bars earlier than the first known timestamp use the mean (z=0)
      - Constant or empty inputs → all-zero output (no fake variance)
      - Z-score over the full known series so the channel has unit variance

    Inputs may be daily-spaced (e.g. golden_cross_d1) — the per-hour scan
    inside the loop forward-fills from the last known daily value at the
    bar's hour grid.
    """
    n = len(sample_end_ts)
    out = np.zeros((n, seq_len), dtype=np.float32)
    if not value_by_ts or len(value_by_ts) < 2:
        return out

    sorted_ts = sorted(value_by_ts.keys())
    vals = np.array([value_by_ts[t] for t in sorted_ts], dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std())
    # Use tolerance, not strict equality: float reductions on a "constant"
    # series can produce sigma ~1e-17 instead of 0, which would otherwise
    # blow up to ±1.0 z-scores. Constant input must yield all-zero output.
    if sigma <= 1e-12:
        return out

    first_ts = sorted_ts[0]
    last_ts = sorted_ts[-1]
    filled: Dict[int, float] = {}
    last_val = mu
    # Walk the hour grid spanning the known series so we can forward-fill
    # daily-spaced inputs onto hourly alignment.
    for h in range(first_ts, last_ts + _BAR_SECS, _BAR_SECS):
        if h in value_by_ts:
            last_val = value_by_ts[h]
        filled[h] = last_val

    for i, end_ts in enumerate(sample_end_ts):
        for j in range(seq_len):
            h = int(end_ts) - (seq_len - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            if h < first_ts:
                v = mu                  # pre-history → neutral (z=0)
            else:
                v = filled.get(h, last_val)
            out[i, j] = (v - mu) / sigma
    return out


# ---------------------------------------------------------------------------
# IO + probe runner
# ---------------------------------------------------------------------------

def _load_pid_closes(
    pid: str,
    history_dir: str = _HISTORY_DIR,
) -> Dict[int, float]:
    """Read hourly parquet for `pid` and return {ts_secs: close}.
    Returns {} if the parquet is missing or unreadable."""
    path = os.path.join(history_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return {}
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(path)
    except Exception:
        return {}
    return {int(t): float(c) for t, c in zip(df["start"], df["close"])}


def _build_candidate_for_pid(
    closes_by_hour: Mapping[int, float],
    candidate: str,
) -> Dict[int, float]:
    """Dispatch on candidate name; returns the per-ts value dict to feed into
    `build_trend_signal`."""
    if candidate == "sma50_1h":
        return compute_sma_dist_series(closes_by_hour, period_bars=50)
    if candidate == "sma200_1h":
        return compute_sma_dist_series(closes_by_hour, period_bars=200)
    if candidate == "sma50_d1":
        daily = daily_resample(closes_by_hour)
        return compute_sma_dist_series(daily, period_bars=50)
    if candidate == "sma200_d1":
        daily = daily_resample(closes_by_hour)
        return compute_sma_dist_series(daily, period_bars=200)
    if candidate == "golden_cross_d1":
        daily = daily_resample(closes_by_hour)
        signal = golden_cross_signal(daily, fast=50, slow=200)
        # Cast int → float for the z-score path.
        return {ts: float(v) for ts, v in signal.items()}
    raise ValueError(f"unknown candidate: {candidate!r}; choose from {_CANDIDATES}")


def _load_pooled_with_trend(
    candidate: str,
    n: int = 20,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import torch
    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import survivorship_aware_top_n
    """Load pooled top-N samples + aligned per-pid trend signal."""
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
    sized = [(pid, len(prods[pid].get("X", []))) for pid in top_pids]
    mode = "legacy" if snapshot_ts is None else f"snapshot_ts={snapshot_ts}"
    print(f"  pooled top-{n} ({mode}): {top_pids}", flush=True)

    print(f"Loading parquet history for top-{n} pids...", flush=True)
    t0 = time.time()
    closes_by_pid: Dict[str, Dict[int, float]] = {}
    for pid, _ in sized:
        closes_by_pid[pid] = _load_pid_closes(pid)
    elapsed = time.time() - t0
    cov_pids = sum(1 for c in closes_by_pid.values() if c)
    print(f"  parquet coverage: {cov_pids}/{len(top_pids)} pids in {elapsed:.1f}s",
          flush=True)

    print(f"Computing candidate '{candidate}' per pid...", flush=True)
    t0 = time.time()
    candidate_by_pid: Dict[str, Dict[int, float]] = {
        pid: _build_candidate_for_pid(closes, candidate)
        for pid, closes in closes_by_pid.items()
    }
    elapsed = time.time() - t0
    cov_pids = sum(1 for c in candidate_by_pid.values() if c)
    print(f"  candidate coverage: {cov_pids}/{len(top_pids)} pids in {elapsed:.1f}s",
          flush=True)
    for pid in top_pids:
        c = candidate_by_pid.get(pid, {})
        if c:
            vals = np.array(list(c.values()))
            print(f"    {pid}: {len(c):,} ts, mean={vals.mean():+.4f} "
                  f"std={vals.std():.4f}", flush=True)
        else:
            print(f"    {pid}: no candidate data", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        sig = build_trend_signal(ts, candidate_by_pid.get(pid, {}),
                                 seq_len=_SEQ_LEN)
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


def _run_one(candidate: str, snapshot_ts: int) -> Dict[str, float]:
    from tools.channel_replace import run_replace
    X, y, ts, sig, used = _load_pooled_with_trend(
        candidate=candidate, n=20, snapshot_ts=snapshot_ts,
    )
    print(f"\npooled samples: n={len(y):,}", flush=True)
    print(f"products & coverage: {used}", flush=True)
    coverage_pct = float((sig != 0).any(axis=1).mean())
    print(f"per-sample non-zero coverage: {coverage_pct:.1%}", flush=True)

    print(f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with '{candidate}' "
          f"z-score; running 5-fold purged CV (4h embargo)...", flush=True)
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"\n=== single-add probe: {candidate} -> ch{_TARGET_CHANNEL} ===")
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")
    return result


def main():
    import torch
    from tools.pid_snapshot import recommended_snapshot_ts

    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-ts", type=str, default=None,
                        help="Survivorship-aware top-N selection cutoff: "
                             "'auto' (median first_ts), an integer epoch "
                             "seconds, or omit for legacy behavior (#163).")
    parser.add_argument("--candidate", type=str, default="all",
                        choices=("all",) + _CANDIDATES,
                        help="Which trend feature to test. 'all' sweeps each "
                             "candidate sequentially.")
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

    candidates = _CANDIDATES if args.candidate == "all" else (args.candidate,)
    results: Dict[str, float] = {}
    for c in candidates:
        print(f"\n{'=' * 70}\nCANDIDATE: {c}\n{'=' * 70}", flush=True)
        try:
            r = _run_one(c, snapshot_ts)
            results[c] = r["delta"]
        except Exception as exc:  # pragma: no cover
            print(f"  ERROR: {exc}", flush=True)
            results[c] = float("nan")

    if len(candidates) > 1:
        # ASCII-only output: Windows cp1252 console can't encode 'Δ' (#153).
        print(f"\n{'=' * 70}\nSUMMARY (delta AUC vs ch13 baseline)\n{'=' * 70}",
              flush=True)
        for c in candidates:
            d = results[c]
            gate = "PASS" if d >= 0.01 else "FAIL"
            print(f"  {c:<22} delta={d:+.4f}  {gate}")
        best = max(results, key=lambda k: (results[k]
                                           if not np.isnan(results[k])
                                           else -np.inf))
        print(f"\nBest: {best}  delta={results[best]:+.4f}")


if __name__ == "__main__":
    main()
