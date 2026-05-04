"""XGBoost training with purged walk-forward CV and grid search.

Phase 3 of the CNN -> XGBoost transition. Uses tools/xgb_features for
tabular feature extraction and tools/walk_forward for time-ordered CV with
4-hour embargo (the gap the CNN training path was missing — see Phase 0
findings).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from tools.xgb_features import extract_features
from tools.walk_forward import purged_walk_forward_splits


_DEFAULT_GRID: list = [
    {"max_depth": d, "min_child_weight": w, "subsample": s}
    for d in (3, 4, 6)
    for w in (1, 5, 20)
    for s in (0.7, 1.0)
]


def _xgb_params(base: dict, lr: float, seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": lr,
        "seed": seed,
        "verbosity": 0,
        **base,
    }


def train_xgb(
    samples: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    n_folds: int = 5,
    embargo_hours: int = 4,
    grid: Optional[Sequence[dict]] = None,
    out_dir: Union[str, os.PathLike] = ".",
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict:
    """Run grid search via purged walk-forward CV; train final model on all data.

    Returns:
        dict with best_params, fold_aucs (best config), mean_auc,
        feature_names, model_path (xgb_model.json),
        features_path (xgb_features.json).
    """
    grid_list = list(grid) if grid is not None else _DEFAULT_GRID
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, feature_names = extract_features(samples)
    labels = np.asarray(labels, dtype=np.float32)

    splits = list(
        purged_walk_forward_splits(timestamps, n_folds, embargo_hours)
    )

    best_score = -1.0
    best_params: Optional[dict] = None
    best_fold_aucs: list = []

    for params in grid_list:
        fold_aucs: list = []
        for train_idx, val_idx in splits:
            x_tr, y_tr = features[train_idx], labels[train_idx]
            x_va, y_va = features[val_idx], labels[val_idx]
            if len(np.unique(y_va)) < 2:
                # single-class fold — AUC undefined, fall back to 0.5
                fold_aucs.append(0.5)
                continue
            d_tr = xgb.DMatrix(x_tr, label=y_tr, feature_names=feature_names)
            d_va = xgb.DMatrix(x_va, label=y_va, feature_names=feature_names)
            booster = xgb.train(
                _xgb_params(params, learning_rate, seed),
                d_tr,
                num_boost_round=n_estimators,
                evals=[(d_va, "val")],
                verbose_eval=False,
            )
            preds = booster.predict(d_va)
            fold_aucs.append(float(roc_auc_score(y_va, preds)))

        score = float(np.mean(fold_aucs))
        if score > best_score:
            best_score = score
            best_params = dict(params)
            best_fold_aucs = list(fold_aucs)

    if best_params is None:
        raise RuntimeError("Grid search produced no valid configs")

    d_all = xgb.DMatrix(features, label=labels, feature_names=feature_names)
    final = xgb.train(
        _xgb_params(best_params, learning_rate, seed),
        d_all,
        num_boost_round=n_estimators,
    )

    model_path = str(out_dir / "xgb_model.json")
    features_path = str(out_dir / "xgb_features.json")
    final.save_model(model_path)
    with open(features_path, "w") as f:
        json.dump(
            {"feature_names": feature_names, "best_params": best_params},
            f,
            indent=2,
        )

    return {
        "best_params": best_params,
        "fold_aucs": best_fold_aucs,
        "mean_auc": best_score,
        "feature_names": feature_names,
        "model_path": model_path,
        "features_path": features_path,
    }
