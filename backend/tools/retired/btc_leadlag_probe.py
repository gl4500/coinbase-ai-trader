"""Single-add probe (#246-#248): replace ch13 (obv_slope) with a BTC→altcoin
lead-lag feature, then measure Δ mean_auc against the +0.01 gate.

Why lead-lag (#246):
    Existing Ch 21 `btc_corr_20` is *contemporaneous* — rolling correlation
    of alt and BTC returns at the same t. Lead-lag is *temporal*: does
    BTC's move at t-k predict alt's move at t? β-residual is the
    idiosyncratic alt component once the current-bar BTC influence is
    stripped. Both are structurally novel relative to the 28-channel set
    and the 4 failed positioning probes.

Candidates (selected via --candidate):
    btc_ret_lag_1     BTC log-return at t-1 (1h ago)
    btc_ret_lag_4     BTC log-return at t-4 (4h ago)
    btc_ret_lag_12    BTC log-return at t-12 (12h ago)
    btc_beta_60       Rolling 60-bar β of alt_ret on btc_ret
    btc_beta_residual_60  alt_ret minus β·btc_ret (60-bar window)

Notes:
    Skips BTC-USD itself (β=1, residual=0 trivially) — its lag-of-self is
    autocorrelation, not lead-lag. So the probe runs on ≤19 of the top-20
    survivorship pids when BTC happens to be in the snapshot, and 20 when
    it is not.

Decision rule: best candidate Δ ≥ +0.01 → integrate. Otherwise this is
the 6th probe failure → flag whether the 0.55 gate is reachable on
price/orderflow features alone.

Run:
    cd backend && python tools/btc_leadlag_probe.py --snapshot-ts auto
    cd backend && python tools/btc_leadlag_probe.py --snapshot-ts auto \\
                    --candidate btc_beta_residual_60
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
_SEQ_LEN = 60
_TARGET_CHANNEL = 13   # obv_slope — most marginal per #146

_BTC_PID = "BTC-USD"
_BETA_WINDOW = 60

_CANDIDATES = (
    "btc_ret_lag_1",
    "btc_ret_lag_4",
    "btc_ret_lag_12",
    "btc_beta_60",
    "btc_beta_residual_60",
)


# ---------------------------------------------------------------------------
# Pure functions (tested in tests/test_btc_leadlag_probe.py)
# ---------------------------------------------------------------------------

def log_returns(closes_by_ts: Mapping[int, float]) -> Dict[int, float]:
    """{ts: log(close_t / close_{t-1})} computed in chronological order over
    available timestamps. The first available ts is excluded (no prior).
    Gaps don't synthesize bars — return is computed against the previous
    AVAILABLE close, not a forward-filled one.
    """
    if not closes_by_ts:
        return {}
    keys = sorted(closes_by_ts.keys())
    out: Dict[int, float] = {}
    prev = float(closes_by_ts[keys[0]])
    for k in keys[1:]:
        cur = float(closes_by_ts[k])
        if prev > 0.0 and cur > 0.0:
            out[int(k)] = float(np.log(cur / prev))
        prev = cur
    return out


def lag_dict(value_by_ts: Mapping[int, float], lag_bars: int) -> Dict[int, float]:
    """Shift the (ts, v) series forward by `lag_bars` hourly bars: the value
    that was at t now appears at t + lag_bars * _BAR_SECS. Use to express
    "BTC return from k bars ago" without leaking forward.
    """
    if lag_bars == 0:
        return {int(k): float(v) for k, v in value_by_ts.items()}
    shift = int(lag_bars) * _BAR_SECS
    return {int(k) + shift: float(v) for k, v in value_by_ts.items()}


def align_pair(
    a_by_ts: Mapping[int, float],
    b_by_ts: Mapping[int, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (ts, a_arr, b_arr) for the timestamps present in BOTH dicts,
    sorted ascending. Empty intersection → three empty arrays."""
    common = sorted(set(a_by_ts.keys()) & set(b_by_ts.keys()))
    if not common:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))
    ts = np.array(common, dtype=np.int64)
    a = np.array([a_by_ts[t] for t in common], dtype=np.float64)
    b = np.array([b_by_ts[t] for t in common], dtype=np.float64)
    return ts, a, b


def _rolling_window_indices(n: int, window: int) -> Iterable[Tuple[int, int]]:
    """Yield (start, end_inclusive) index pairs for each rolling window of
    size `window`. First eligible end-index is `window - 1`."""
    if n < window or window <= 0:
        return
    for end in range(window - 1, n):
        yield (end - window + 1, end)


def rolling_beta(
    ts: np.ndarray,
    alt_ret: np.ndarray,
    btc_ret: np.ndarray,
    window: int = _BETA_WINDOW,
) -> Dict[int, float]:
    """Per-bar rolling OLS β of alt_ret on btc_ret over the last `window`
    aligned bars.

    Returns {ts: beta} only for bars with a full warm window. If btc_ret
    variance is 0 in a window, β is undefined → output gets the previous
    finite β (carried) or 0.0 if none yet (so all output values are finite).
    """
    n = len(ts)
    out: Dict[int, float] = {}
    last_finite = 0.0
    for s, e in _rolling_window_indices(n, window):
        bw = btc_ret[s:e + 1]
        aw = alt_ret[s:e + 1]
        b_mean = float(bw.mean())
        a_mean = float(aw.mean())
        b_centered = bw - b_mean
        a_centered = aw - a_mean
        denom = float((b_centered * b_centered).sum())
        if denom <= 1e-18:
            beta = last_finite
        else:
            beta = float((b_centered * a_centered).sum() / denom)
            last_finite = beta
        out[int(ts[e])] = beta
    return out


def beta_residual(
    ts: np.ndarray,
    alt_ret: np.ndarray,
    btc_ret: np.ndarray,
    window: int = _BETA_WINDOW,
) -> Dict[int, float]:
    """Per-bar idiosyncratic alt return: alt_ret_t - β_t · btc_ret_t, where
    β_t comes from `rolling_beta`. Same warm-up rule as rolling_beta."""
    betas = rolling_beta(ts, alt_ret, btc_ret, window=window)
    out: Dict[int, float] = {}
    ts_to_idx = {int(t): i for i, t in enumerate(ts)}
    for t, b in betas.items():
        i = ts_to_idx[t]
        out[int(t)] = float(alt_ret[i]) - b * float(btc_ret[i])
    return out


def build_leadlag_signal(
    sample_end_ts: np.ndarray,
    value_by_ts: Mapping[int, float],
    seq_len: int = _SEQ_LEN,
) -> np.ndarray:
    """[N, seq_len] z-scored signal aligned to per-sample bar-end timestamps.

    Same alignment contract as build_btc_dom_signal / build_trend_signal:
    forward-fill missing hours, neutral z=0 for pre-history bars, all-zero
    output on empty/constant inputs.
    """
    n = len(sample_end_ts)
    out = np.zeros((n, seq_len), dtype=np.float32)
    if not value_by_ts or len(value_by_ts) < 2:
        return out

    sorted_ts = sorted(value_by_ts.keys())
    vals = np.array([value_by_ts[t] for t in sorted_ts], dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std())
    if sigma <= 1e-12:
        return out

    first_ts = sorted_ts[0]
    last_ts = sorted_ts[-1]
    filled: Dict[int, float] = {}
    last_val = mu
    for h in range(first_ts, last_ts + _BAR_SECS, _BAR_SECS):
        if h in value_by_ts:
            last_val = value_by_ts[h]
        filled[h] = last_val

    for i, end_ts in enumerate(sample_end_ts):
        for j in range(seq_len):
            h = int(end_ts) - (seq_len - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            if h < first_ts:
                v = mu
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
    pid: str,
    alt_closes: Mapping[int, float],
    btc_closes: Mapping[int, float],
    candidate: str,
) -> Dict[int, float]:
    if pid == _BTC_PID:
        return {}   # skip BTC vs itself (autocorrelation, not lead-lag)
    btc_ret = log_returns(btc_closes)
    alt_ret = log_returns(alt_closes)
    if candidate == "btc_ret_lag_1":
        return lag_dict(btc_ret, 1)
    if candidate == "btc_ret_lag_4":
        return lag_dict(btc_ret, 4)
    if candidate == "btc_ret_lag_12":
        return lag_dict(btc_ret, 12)
    ts, alt_arr, btc_arr = align_pair(alt_ret, btc_ret)
    if candidate == "btc_beta_60":
        return rolling_beta(ts, alt_arr, btc_arr, window=_BETA_WINDOW)
    if candidate == "btc_beta_residual_60":
        return beta_residual(ts, alt_arr, btc_arr, window=_BETA_WINDOW)
    raise ValueError(f"unknown candidate: {candidate!r}; choose from {_CANDIDATES}")


def _load_pooled_with_leadlag(
    candidate: str,
    n: int = 20,
    snapshot_ts: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import torch
    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import survivorship_aware_top_n
    """Load pooled top-N samples + aligned per-pid BTC lead-lag signal."""
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
    print(f"  parquet coverage: {cov_pids}/{len(top_pids)} pids in {elapsed:.1f}s "
          f"(BTC: {btc_label})", flush=True)
    if not btc_closes:
        raise RuntimeError("BTC-USD parquet missing — cannot build lead-lag")

    print(f"Computing candidate '{candidate}' per pid...", flush=True)
    t0 = time.time()
    candidate_by_pid: Dict[str, Dict[int, float]] = {
        pid: _build_candidate_for_pid(pid, closes, btc_closes, candidate)
        for pid, closes in closes_by_pid.items()
    }
    elapsed = time.time() - t0
    cov_pids = sum(1 for c in candidate_by_pid.values() if c)
    print(f"  candidate coverage: {cov_pids}/{len(top_pids)} pids in {elapsed:.1f}s "
          f"(BTC pids skipped by design)", flush=True)
    for pid in top_pids:
        c = candidate_by_pid.get(pid, {})
        if c:
            vals = np.array(list(c.values()))
            print(f"    {pid}: {len(c):,} ts, mean={vals.mean():+.4f} "
                  f"std={vals.std():.4f}", flush=True)
        else:
            tag = "skipped (BTC)" if pid == _BTC_PID else "no candidate data"
            print(f"    {pid}: {tag}", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    products_used: List[str] = []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        sig = build_leadlag_signal(ts, candidate_by_pid.get(pid, {}),
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
    X, y, ts, sig, used = _load_pooled_with_leadlag(
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
                        help="Which lead-lag feature to test. 'all' sweeps "
                             "all candidates sequentially.")
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
            print(f"  {c:<24} delta={d:+.4f}  {gate}")
        best = max(results, key=lambda k: (results[k]
                                           if not np.isnan(results[k])
                                           else -np.inf))
        print(f"\nBest: {best}  delta={results[best]:+.4f}")


if __name__ == "__main__":
    main()
