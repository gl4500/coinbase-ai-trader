"""Production XGB booster retrain on the live cnn_dataset_cache.pt.

Trains the booster used by agents/xgb_signal at inference. Pooled top-20
products, 5-fold purged walk-forward CV, fixed best_params from the
May 3 production train (max_depth=4, mcw=1, subsample=0.7).

Why fixed params: the only change is Ch 27 (OI) being added; full grid
search is unlikely to shift the optimum and costs ~5x runtime. If
val_auc drops materially from the 0.5224 May 3 baseline, re-run
xgb_top40_tune.py to re-search.

Outputs:
    backend/xgb_model.json     — booster (280 features)
    backend/xgb_features.json  — feature_names + best_params

Run:
    cd backend && python -m tools.train_xgb_prod
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.train_xgb import train_xgb  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_TOP_N = 20
_BEST_PARAMS = {"max_depth": 4, "min_child_weight": 1, "subsample": 0.7}


def _entry_to_arrays(entry: dict):
    X = np.stack([
        x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        for x in entry["X"]
    ]).astype(np.float32)
    y = np.asarray(entry["y"], dtype=np.float32)
    first_ts = int(entry["first_ts"])
    idx = np.asarray(entry["indices"], dtype=np.int64)
    ts = first_ts + idx * _BAR_SECS
    return X, y, ts


def _pooled_top_n(prods: dict, n: int):
    sized = sorted(
        ((pid, len(e.get("X", []))) for pid, e in prods.items()),
        key=lambda x: -x[1],
    )[:n]
    print(f"  pooling top {n} products by sample count:")
    for pid, count in sized:
        print(f"    {pid:<14} {count:,}")

    Xs, ys, tss = [], [], []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        ys.append(y)
        tss.append(ts)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    return X_all[order], y_all[order], ts_all[order]


def main():
    t0 = time.time()
    print(f"Loading cache: {_CACHE_PATH}")
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    schema = blob.get("schema", {})
    prods = blob["products"]
    print(
        f"  schema: version={schema.get('version')} "
        f"n_channels={schema.get('n_channels')} seq_len={schema.get('seq_len')} "
        f"forward_hours={schema.get('forward_hours')} "
        f"label_thresh={schema.get('label_thresh')}"
    )
    print(f"  products: {len(prods)}")

    X, y, ts = _pooled_top_n(prods, _TOP_N)
    print(
        f"\nPooled dataset: X={X.shape} y={y.shape} ts=[{ts[0]} .. {ts[-1]}]"
    )
    print(f"  positives: {int(y.sum()):,} / {len(y):,} = {y.mean():.4f}")

    print(f"\nTraining XGB booster with fixed params: {_BEST_PARAMS}")
    res = train_xgb(
        X, y, ts,
        n_folds=5,
        embargo_hours=4,
        grid=[_BEST_PARAMS],
        out_dir=BACKEND,
        n_estimators=200,
        learning_rate=0.05,
        seed=0,
    )

    fold_str = ", ".join(f"{a:.4f}" for a in res["fold_aucs"])
    print(
        f"\nResult: mean_auc={res['mean_auc']:.4f} "
        f"folds=[{fold_str}]"
    )
    print(f"  feature count: {len(res['feature_names'])}")
    print(f"  saved: {res['model_path']}")
    print(f"  saved: {res['features_path']}")
    print(f"  elapsed: {time.time() - t0:.1f}s")


def main_v3() -> None:
    """CLI entry for the v3 mixed-lookback trainer (#311e).

    Auto-discovers pids from backend/data/history/*.parquet.
    Delegates to tools.train_xgb.train_xgb_v3.
    """
    import argparse
    import json as _json
    from pathlib import Path as _Path
    from tools.train_xgb import train_xgb_v3

    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="backend/data/history")
    p.add_argument("--out-dir", default="backend")
    p.add_argument(
        "--pids", nargs="*", default=None,
        help="restrict to these pids; defaults to all parquet files",
    )
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.05)
    args = p.parse_args()

    if args.pids is None:
        pids = [f.stem for f in _Path(args.parquet_dir).glob("*.parquet")
                if not f.stem.startswith("__")]
    else:
        pids = args.pids

    result = train_xgb_v3(
        pids, args.parquet_dir, args.out_dir,
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
    )
    print(_json.dumps(result, indent=2))


if __name__ == "__main__":
    # Route to v3 trainer if `--feature-set v3` is in argv, else legacy
    if len(sys.argv) > 1 and "--feature-set" in sys.argv and "v3" in sys.argv:
        # consume the flag pair
        i = sys.argv.index("--feature-set")
        del sys.argv[i:i + 2]
        main_v3()
    else:
        main()
