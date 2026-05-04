"""Sweep top-N feature retention to find the AUC peak.

Top-50 lifted pooled AUC by +0.014 over the 270-feature baseline.
This sweeps N in {20, 30, 40, 50, 60, 75, 100, 150, 210} to find the
inflection. Also dumps which channels dominate the top-N at each cut.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

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
_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "auc",
    "learning_rate": 0.05, "max_depth": 4,
    "min_child_weight": 1, "subsample": 1.0,
    "seed": 0, "verbosity": 0,
}


def _channel_of(name: str) -> int | None:
    if not name.startswith("ch") or "_" not in name:
        return None
    try:
        return int(name.split("_")[0][2:])
    except ValueError:
        return None


def main():
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    X, y, ts = _pooled_top_n(prods, n=20)
    print(f"pooled samples: n={len(y):,}")

    feats, names = extract_features(X, feature_set="v1")
    splits = list(purged_walk_forward_splits(ts, 5, 4))

    full = xgb.train(
        _PARAMS,
        xgb.DMatrix(feats, label=y, feature_names=names),
        num_boost_round=200,
    )
    gain = full.get_score(importance_type="gain")
    sorted_gain = sorted(gain.items(), key=lambda x: -x[1])
    ranked = [n for n, _ in sorted_gain]

    print(f"\nbaseline 270-feature pooled v1 = 0.5127 (from prior run)")
    print(f"\n{'N':>4} {'mean_auc':>10} {'delta':>8}  top channels")
    print("-" * 70)

    grid = [20, 30, 40, 50, 60, 75, 100, 150, 210, 270]
    for n in grid:
        keep = set(ranked[:n]) if n < 270 else set(names)
        keep_idx = [i for i, nm in enumerate(names) if nm in keep]
        masked = np.zeros_like(feats)
        masked[:, keep_idx] = feats[:, keep_idx]

        aucs = []
        for tr, va in splits:
            if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
                aucs.append(0.5)
                continue
            d_tr = xgb.DMatrix(masked[tr], label=y[tr], feature_names=names)
            d_va = xgb.DMatrix(masked[va], label=y[va], feature_names=names)
            b = xgb.train(_PARAMS, d_tr, num_boost_round=200)
            aucs.append(float(roc_auc_score(y[va], b.predict(d_va))))
        ma = float(np.mean(aucs))

        chans: Counter = Counter()
        for nm in (ranked[:n] if n < 270 else ranked):
            c = _channel_of(nm)
            if c is not None:
                chans[c] += 1
        top_ch = ", ".join(f"ch{c}({k})" for c, k in chans.most_common(6))
        print(f"{n:>4} {ma:>10.4f} {ma-0.5127:>+8.4f}  {top_ch}")


if __name__ == "__main__":
    main()
