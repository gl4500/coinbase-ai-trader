"""Hyperparameter tune XGB on the top-40-by-gain feature subset.

Top-40 lifted pooled AUC to 0.5273 (+0.015 over 270-feature baseline)
with default params (depth=4, lr=0.05, n=200). This sweeps depth,
min_child_weight, subsample, colsample_bytree, learning_rate, and
n_estimators to see whether the 0.55 gate is reachable on these features.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.feature_set_compare import _pooled_top_n  # noqa: E402
from tools.walk_forward import purged_walk_forward_splits  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")


def _eval(masked, y, ts, names, splits, params, n_est):
    aucs = []
    for tr, va in splits:
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(0.5)
            continue
        d_tr = xgb.DMatrix(masked[tr], label=y[tr], feature_names=names)
        d_va = xgb.DMatrix(masked[va], label=y[va], feature_names=names)
        b = xgb.train(params, d_tr, num_boost_round=n_est)
        aucs.append(float(roc_auc_score(y[va], b.predict(d_va))))
    return float(np.mean(aucs)), aucs


def main():
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    X, y, ts = _pooled_top_n(prods, n=20)
    feats, names = extract_features(X, feature_set="v1")
    splits = list(purged_walk_forward_splits(ts, 5, 4))
    print(f"pooled samples: n={len(y):,}")

    # Rank features by gain on full 270.
    base_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 1.0,
        "seed": 0,
        "verbosity": 0,
    }
    full = xgb.train(
        base_params,
        xgb.DMatrix(feats, label=y, feature_names=names),
        num_boost_round=200,
    )
    gain = full.get_score(importance_type="gain")
    sorted_gain = sorted(gain.items(), key=lambda x: -x[1])
    keep = set(n for n, _ in sorted_gain[:40])
    keep_idx = [i for i, nm in enumerate(names) if nm in keep]
    masked = np.zeros_like(feats)
    masked[:, keep_idx] = feats[:, keep_idx]

    best = (0.0, None)

    # Grid 1: depth × n_est × lr at default reg
    grid = []
    for depth in (3, 4, 5, 6):
        for lr in (0.03, 0.05, 0.1):
            for n_est in (200, 400, 600):
                grid.append(
                    {
                        "max_depth": depth,
                        "learning_rate": lr,
                        "n_est": n_est,
                        "min_child_weight": 1,
                        "subsample": 1.0,
                        "colsample_bytree": 1.0,
                        "reg_lambda": 1.0,
                        "reg_alpha": 0.0,
                    }
                )

    # Grid 2: regularization at depth=4
    for mcw in (1, 5, 10, 20):
        for sub in (0.6, 0.8, 1.0):
            for cs in (0.6, 0.8, 1.0):
                grid.append(
                    {
                        "max_depth": 4,
                        "learning_rate": 0.05,
                        "n_est": 400,
                        "min_child_weight": mcw,
                        "subsample": sub,
                        "colsample_bytree": cs,
                        "reg_lambda": 1.0,
                        "reg_alpha": 0.0,
                    }
                )

    # Grid 3: stronger L1/L2
    for la in (1.0, 5.0, 20.0):
        for al in (0.0, 1.0, 5.0):
            grid.append(
                {
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "n_est": 400,
                    "min_child_weight": 5,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": la,
                    "reg_alpha": al,
                }
            )

    print(f"\nrunning {len(grid)} configs on top-40 ({len(keep)} features)\n")
    print(
        f"{'depth':>5} {'lr':>5} {'n':>4} {'mcw':>4} {'sub':>4} {'cs':>4} "
        f"{'la':>5} {'al':>4}  {'mean_auc':>9}"
    )
    print("-" * 72)
    for cfg in grid:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": cfg["learning_rate"],
            "max_depth": cfg["max_depth"],
            "min_child_weight": cfg["min_child_weight"],
            "subsample": cfg["subsample"],
            "colsample_bytree": cfg["colsample_bytree"],
            "reg_lambda": cfg["reg_lambda"],
            "reg_alpha": cfg["reg_alpha"],
            "seed": 0,
            "verbosity": 0,
        }
        ma, _ = _eval(masked, y, ts, names, splits, params, cfg["n_est"])
        print(
            f"{cfg['max_depth']:>5} {cfg['learning_rate']:>5.2f} "
            f"{cfg['n_est']:>4} {cfg['min_child_weight']:>4} "
            f"{cfg['subsample']:>4.1f} {cfg['colsample_bytree']:>4.1f} "
            f"{cfg['reg_lambda']:>5.1f} {cfg['reg_alpha']:>4.1f}  {ma:>9.4f}"
        )
        if ma > best[0]:
            best = (ma, cfg)

    print(f"\nBEST: mean_auc={best[0]:.4f}  cfg={best[1]}")
    print(f"vs baseline pooled v1 = 0.5127  delta = {best[0] - 0.5127:+.4f}")
    print(f"vs gate 0.55  delta = {best[0] - 0.55:+.4f}")


if __name__ == "__main__":
    main()
