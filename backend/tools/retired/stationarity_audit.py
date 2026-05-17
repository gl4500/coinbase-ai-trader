"""Heuristic stationarity audit for the 28-channel CNN feature stack (#164).

Pure-numpy first-pass audit. Per-channel metrics:
- `drift_z`: |mean(first_half) - mean(second_half)| / std(overall) — flags
  level shifts indicative of non-stationarity in mean.
- `var_ratio`: std(second_half) / std(first_half) — flags volatility
  regime changes.
- `lag1`: lag-1 autocorrelation — values near 1 indicate unit-root /
  random-walk behaviour.
- `flag`: 'stationary' if all three metrics fall inside reasonable
  tolerances, else 'suspect'.

This is a heuristic, not formal ADF. Adding statsmodels would let us run
true Augmented Dickey-Fuller; deferred until a finding warrants the dep.

Run:
    python tools/stationarity_audit.py [--snapshot-ts auto|<int>]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_DRIFT_Z_THRESH = 0.5
_VAR_RATIO_TOL = 0.5    # |1 - var_ratio| > this → suspect
_LAG1_THRESH = 0.95


def _stationarity_metrics(series: np.ndarray) -> dict:
    """Return drift_z, var_ratio, lag1 autocorr, and a stationary/suspect flag.

    No formal ADF — pure-numpy proxies. Pathological inputs (constant,
    very short) return safe values without crashing.
    """
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)
    overall_std = float(series.std())
    eps = 1e-12

    if n < 4:
        return {
            "n": n, "mean": float(series.mean()) if n else 0.0,
            "std": overall_std, "drift_z": 0.0, "var_ratio": 1.0,
            "lag1": 0.0, "flag": "stationary",
        }

    h = n // 2
    s1, s2 = series[:h], series[h:]
    mean1, mean2 = float(s1.mean()), float(s2.mean())
    std1, std2 = float(s1.std()), float(s2.std())
    drift_z = abs(mean1 - mean2) / (overall_std + eps)
    var_ratio = (std2 + eps) / (std1 + eps)

    if overall_std < eps:
        lag1 = 0.0
    else:
        a = series[:-1] - series[:-1].mean()
        b = series[1:] - series[1:].mean()
        denom = (a.std() * b.std()) + eps
        lag1 = float((a * b).mean() / denom)

    flag = "stationary"
    if (
        drift_z > _DRIFT_Z_THRESH
        or abs(1.0 - var_ratio) > _VAR_RATIO_TOL
        or abs(lag1) > _LAG1_THRESH
    ):
        flag = "suspect"

    return {
        "n": n, "mean": float(series.mean()), "std": overall_std,
        "drift_z": drift_z, "var_ratio": var_ratio,
        "lag1": lag1, "flag": flag,
    }


def _parse_snapshot_ts(arg: Optional[str], prods: dict) -> Optional[int]:
    if arg is None:
        return None
    if arg == "auto":
        ts = recommended_snapshot_ts(prods)
        return ts if ts > 0 else None
    return int(arg)


def _audit_channels(
    prods: dict, n_pids: int = 20, snapshot_ts: Optional[int] = None,
) -> list:
    """Run per-channel stationarity audit on the terminal value of each
    sample's window, ordered chronologically. Returns a list of metric
    dicts (one per channel) with `channel` index added."""
    pids = survivorship_aware_top_n(prods, n=n_pids, snapshot_ts=snapshot_ts)
    Xs, tss = [], []
    for pid in pids:
        X, _y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        tss.append(ts)
    X_all = np.concatenate(Xs, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    X_sorted = X_all[order]

    n_channels = X_sorted.shape[1]
    out = []
    for ch in range(n_channels):
        terminal = X_sorted[:, ch, -1]
        m = _stationarity_metrics(terminal)
        m["channel"] = ch
        out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-ts",
        default=None,
        help="Survivorship-aware top-N cutoff (#163). Pass 'auto' for the "
             "median-first_ts default, or an explicit Unix-seconds int.",
    )
    parser.add_argument("--n-pids", type=int, default=20)
    args = parser.parse_args()

    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    snapshot_ts = _parse_snapshot_ts(args.snapshot_ts, prods)
    if snapshot_ts is not None:
        print(f"  snapshot_ts (survivorship-aware): {snapshot_ts}", flush=True)

    rows = _audit_channels(prods, n_pids=args.n_pids, snapshot_ts=snapshot_ts)

    print(
        f"\n{'ch':>3} {'n':>8} {'mean':>10} {'std':>10} "
        f"{'drift_z':>8} {'var_ratio':>10} {'lag1':>7} flag",
        flush=True,
    )
    print("-" * 72, flush=True)
    for r in rows:
        print(
            f"{r['channel']:>3} {r['n']:>8,} {r['mean']:>10.4f} {r['std']:>10.4f} "
            f"{r['drift_z']:>8.3f} {r['var_ratio']:>10.3f} {r['lag1']:>7.3f} "
            f"{r['flag']}",
            flush=True,
        )

    suspects = [r["channel"] for r in rows if r["flag"] == "suspect"]
    print(f"\nsuspect channels ({len(suspects)}/{len(rows)}): {suspects}", flush=True)


if __name__ == "__main__":
    main()
