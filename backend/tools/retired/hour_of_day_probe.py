"""Single-add probe: replace ch13 (obv_slope, marginal noise per #146) with a
sin/cos hour-of-day signal across the pooled top-20.

Zero-fetch — `(ts // 3600) % 24` is computed locally, full coverage on every
sample. Tests whether intraday seasonality has signal value beyond what the
existing 27 channels capture.

Runner emits sin and cos as two separate probes (each replaces ch13), since
a single channel can hold only one of them. Decision rule mirrors #147:
Δ ≥ +0.01 → integrate; otherwise drop.

Run:
    python tools/hour_of_day_probe.py
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.channel_replace import run_replace  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_SEQ_LEN = 60
_TARGET_CHANNEL = 13  # obv_slope — most marginal per #146 (Δ -0.0002)


def _parse_snapshot_ts(arg: Optional[str], prods: dict) -> Optional[int]:
    if arg is None:
        return None
    if arg == "auto":
        ts = recommended_snapshot_ts(prods)
        return ts if ts > 0 else None
    return int(arg)


def _load_pooled(
    n: int = 20, snapshot_ts: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled top-N concat. `snapshot_ts=None` reproduces legacy ranking; pass
    an int (or use `recommended_snapshot_ts`) for survivorship-aware selection
    per #163."""
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
    print(f"  pooled top-{n}: {top_pids}", flush=True)

    Xs, ys, tss = [], [], []
    for pid in top_pids:
        X, y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        ys.append(y)
        tss.append(ts)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    return X_all[order], y_all[order], ts_all[order]


def _hour_window(ts_secs: np.ndarray, kind: str) -> np.ndarray:
    """For each sample with bar-end ts (sec), build a [N, 60] window of
    sin(2π·h/24) or cos(2π·h/24) for the trailing 60 hourly bars."""
    n = len(ts_secs)
    out = np.zeros((n, _SEQ_LEN), dtype=np.float32)
    for j in range(_SEQ_LEN):
        offset = (_SEQ_LEN - 1 - j) * _BAR_SECS
        hours = ((ts_secs - offset) // _BAR_SECS) % 24
        angle = 2.0 * np.pi * hours.astype(np.float64) / 24.0
        out[:, j] = (np.sin(angle) if kind == "sin" else np.cos(angle)).astype(np.float32)
    return out


def _run_one(label: str, X, y, ts, sig):
    print(f"\n--- probe: {label} -> ch{_TARGET_CHANNEL} ---", flush=True)
    res = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"  baseline mean_auc = {res['baseline_auc']:.4f}", flush=True)
    print(f"  replaced mean_auc = {res['replaced_auc']:.4f}", flush=True)
    print(f"  delta             = {res['delta']:+.4f}", flush=True)
    gate = "PASS" if res['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-ts",
        default=None,
        help="Survivorship-aware top-N cutoff (#163). Pass 'auto' for the "
             "median-first_ts default, or an explicit Unix-seconds int. "
             "Omit to keep the legacy len(X) ranking.",
    )
    args = parser.parse_args()

    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    snapshot_ts = _parse_snapshot_ts(args.snapshot_ts, blob["products"])
    if snapshot_ts is not None:
        print(f"  snapshot_ts (survivorship-aware): {snapshot_ts}", flush=True)

    X, y, ts = _load_pooled(n=20, snapshot_ts=snapshot_ts)
    print(f"\npooled samples: n={len(y):,}", flush=True)

    sin_sig = _hour_window(ts, kind="sin")
    cos_sig = _hour_window(ts, kind="cos")

    res_sin = _run_one("hour-of-day sin", X, y, ts, sin_sig)
    res_cos = _run_one("hour-of-day cos", X, y, ts, cos_sig)

    print("\n=== summary ===", flush=True)
    print(f"  sin delta = {res_sin['delta']:+.4f}", flush=True)
    print(f"  cos delta = {res_cos['delta']:+.4f}", flush=True)
    best = max(res_sin['delta'], res_cos['delta'])
    print(f"  best      = {best:+.4f}  ({'PASS' if best >= 0.01 else 'FAIL'})", flush=True)


if __name__ == "__main__":
    main()
