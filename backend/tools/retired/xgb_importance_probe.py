"""One-shot XGB feature importance on pooled top-20 cache.

Trains a single XGBoost on all pooled samples (no CV — we want the
feature-usage profile, not generalization). Dumps gain-based importance
sorted descending. Identifies bottom-N features XGB never splits on.
Then re-runs calibration_probe with those features zeroed to test
whether pruning lifts AUC vs the 0.5127 v1 baseline.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import xgboost as xgb

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.calibration_probe import calibration_probe  # noqa: E402
from tools.feature_set_compare import _pooled_top_n  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402


_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")


def _load() -> dict:
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    return blob["products"]


def main():
    print(f"Loading cache: {_CACHE_PATH}")
    prods = _load()
    X, y, ts = _pooled_top_n(prods, n=20)
    print(f"  pooled samples: n={len(y):,}")

    feats, names = extract_features(X, feature_set="v1")
    print(f"  features: {feats.shape}, names={len(names)}")

    dmat = xgb.DMatrix(feats, label=y, feature_names=names)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 1,
            "subsample": 1.0,
            "seed": 0,
            "verbosity": 0,
        },
        dmat,
        num_boost_round=200,
    )

    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    used = set(gain.keys())
    unused = [n for n in names if n not in used]

    print(f"\n=== usage ===")
    print(f"used features: {len(used)} / {len(names)}")
    print(f"unused (XGB never split on): {len(unused)}")

    by_channel: Counter = Counter()
    for n in unused:
        if n.startswith("ch") and "_" in n:
            ch = int(n.split("_")[0][2:])
            by_channel[ch] += 1
    print(f"unused per channel (top): {by_channel.most_common(10)}")

    sorted_gain = sorted(gain.items(), key=lambda x: -x[1])
    print(f"\n=== top 20 by gain ===")
    for n, g in sorted_gain[:20]:
        w = weight.get(n, 0)
        print(f"  {n:<28} gain={g:>10.2f}  splits={int(w)}")

    print(f"\n=== bottom 20 (used) by gain ===")
    for n, g in sorted_gain[-20:]:
        w = weight.get(n, 0)
        print(f"  {n:<28} gain={g:>10.2f}  splits={int(w)}")

    print(f"\n=== first 20 unused names ===")
    for n in unused[:20]:
        print(f"  {n}")

    print(f"\n=== prune test: zero out {len(unused)} unused features and re-probe ===")
    pruned = feats.copy()
    unused_idx = [i for i, n in enumerate(names) if n in set(unused)]
    pruned[:, unused_idx] = 0.0

    from sklearn.metrics import roc_auc_score
    from tools.walk_forward import purged_walk_forward_splits

    splits = list(purged_walk_forward_splits(ts, 5, 4))
    aucs = []
    for tr, va in splits:
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(0.5)
            continue
        d_tr = xgb.DMatrix(pruned[tr], label=y[tr], feature_names=names)
        d_va = xgb.DMatrix(pruned[va], label=y[va], feature_names=names)
        b = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "learning_rate": 0.05, "max_depth": 4,
                "min_child_weight": 1, "subsample": 1.0,
                "seed": 0, "verbosity": 0,
            },
            d_tr, num_boost_round=200,
        )
        aucs.append(float(roc_auc_score(y[va], b.predict(d_va))))
    mean_auc = float(np.mean(aucs))
    print(f"   pruned mean_auc = {mean_auc:.4f}  folds={[f'{a:.3f}' for a in aucs]}")
    print(f"   baseline v1 pooled = 0.5127  delta = {mean_auc - 0.5127:+.4f}")

    print(f"\n=== top-50 only test: keep top-50 by gain, zero rest ===")
    keep = {n for n, _ in sorted_gain[:50]}
    keep_idx = [i for i, n in enumerate(names) if n in keep]
    masked = np.zeros_like(feats)
    masked[:, keep_idx] = feats[:, keep_idx]

    aucs2 = []
    for tr, va in splits:
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs2.append(0.5)
            continue
        d_tr = xgb.DMatrix(masked[tr], label=y[tr], feature_names=names)
        d_va = xgb.DMatrix(masked[va], label=y[va], feature_names=names)
        b = xgb.train(
            {
                "objective": "binary:logistic", "eval_metric": "auc",
                "learning_rate": 0.05, "max_depth": 4,
                "min_child_weight": 1, "subsample": 1.0,
                "seed": 0, "verbosity": 0,
            },
            d_tr, num_boost_round=200,
        )
        aucs2.append(float(roc_auc_score(y[va], b.predict(d_va))))
    mean_auc2 = float(np.mean(aucs2))
    print(f"   top-50 mean_auc = {mean_auc2:.4f}  folds={[f'{a:.3f}' for a in aucs2]}")
    print(f"   delta vs baseline = {mean_auc2 - 0.5127:+.4f}")


if __name__ == "__main__":
    main()
