"""Compare XGBoost feature sets v1 vs v2 on the live cnn_dataset_cache.pt.

Cells:
    BTC-USD only      x  v1  (baseline)
    BTC-USD only      x  v2  (current 270 + 10 cross-channel addons)
    pooled top-N      x  v1
    pooled top-N      x  v2

Output: 4-cell mean_auc grid + gate pass/fail. Drives the decision on
whether to promote v2 to default and whether to retrain BTC-only or pooled.
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

from tools.calibration_probe import calibration_probe  # noqa: E402
from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600  # cache built on 1h candles


def _load_cache() -> dict:
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    return blob["products"]


def _entry_to_arrays(entry: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-product cache entry -> (samples [N,27,60], labels [N], ts [N])."""
    X = np.stack([
        x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        for x in entry["X"]
    ]).astype(np.float32)
    y = np.asarray(entry["y"], dtype=np.float32)
    first_ts = int(entry["first_ts"])
    idx = np.asarray(entry["indices"], dtype=np.int64)
    ts = first_ts + idx * _BAR_SECS
    return X, y, ts


def _btc_only(prods: dict):
    return _entry_to_arrays(prods["BTC-USD"])


def _pooled_top_n(prods: dict, n: int = 20, snapshot_ts: Optional[int] = None):
    """Concat samples from the top-N products, sorted by ts.

    `snapshot_ts=None` reproduces the legacy `len(entry["X"])` ranking so prior
    results stay reproducible. Pass an int (or use `recommended_snapshot_ts`)
    to opt into survivorship-aware selection per #163.
    """
    top_pids = survivorship_aware_top_n(prods, n=n, snapshot_ts=snapshot_ts)
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


def _run_cell(name: str, X, y, ts, feature_set: str):
    print(f"-- {name} | feature_set={feature_set} | n={len(y):,} ", flush=True)
    res = calibration_probe(
        X, y, ts,
        n_folds=5, embargo_hours=4, n_buckets=10,
        params={"max_depth": 4, "min_child_weight": 1, "subsample": 1.0},
        n_estimators=200, learning_rate=0.05, seed=0,
        feature_set=feature_set,
    )
    fold_str = ", ".join(f"{a:.3f}" for a in res["fold_aucs"])
    print(
        f"   mean_auc={res['mean_auc']:.4f}  "
        f"folds=[{fold_str}]  "
        f"monotonic={res['is_monotonic']}  "
        f"gate={res['gate_passed']}",
        flush=True,
    )
    return res


def _parse_snapshot_ts(arg: Optional[str], prods: dict) -> Optional[int]:
    if arg is None:
        return None
    if arg == "auto":
        ts = recommended_snapshot_ts(prods)
        return ts if ts > 0 else None
    return int(arg)


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

    print(f"Loading cache: {_CACHE_PATH}")
    prods = _load_cache()
    print(f"  products: {len(prods)}")
    snapshot_ts = _parse_snapshot_ts(args.snapshot_ts, prods)
    if snapshot_ts is not None:
        print(f"  snapshot_ts (survivorship-aware): {snapshot_ts}")
    print()

    Xb, yb, tb = _btc_only(prods)
    Xp, yp, tp = _pooled_top_n(prods, n=20, snapshot_ts=snapshot_ts)

    cells = {
        ("BTC-USD only", "v1"): _run_cell("BTC-USD only", Xb, yb, tb, "v1"),
        ("BTC-USD only", "v2"): _run_cell("BTC-USD only", Xb, yb, tb, "v2"),
        ("pooled top-20", "v1"): _run_cell("pooled top-20", Xp, yp, tp, "v1"),
        ("pooled top-20", "v2"): _run_cell("pooled top-20", Xp, yp, tp, "v2"),
    }

    print("\n=== summary mean_auc ===")
    print(f"{'cell':<18} {'v1':>8} {'v2':>8} {'delta':>8}")
    for src in ("BTC-USD only", "pooled top-20"):
        a = cells[(src, "v1")]["mean_auc"]
        b = cells[(src, "v2")]["mean_auc"]
        print(f"{src:<18} {a:>8.4f} {b:>8.4f} {b - a:+8.4f}")

    print("\n=== gate ===")
    for (src, fs), r in cells.items():
        print(f"  {src:<18} {fs}  passed={r['gate_passed']}  reason={r['gate_reason']}")


if __name__ == "__main__":
    main()
