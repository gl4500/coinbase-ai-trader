"""Calibration probe — Phase 4 HARD GATE for the CNN -> XGBoost transition.

Runs purged walk-forward CV with a fixed XGBoost config, collects out-of-fold
predictions, buckets them by predicted probability, and reports realized
win rate per bucket. Produces a single boolean gate decision:

    gate_passed = (mean_auc >= 0.55) AND is_monotonic

This is the check the CNN training path never had — recording val_auc=1.0
without verifying that confidence buckets actually correlate with realized
outcomes is the bug that produced the 47.5% live WR (Session 58 audit).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

from tools.xgb_features import extract_features
from tools.walk_forward import purged_walk_forward_splits


_DEFAULT_PARAMS = {"max_depth": 4, "min_child_weight": 1, "subsample": 1.0}
_GATE_AUC = 0.55


def _xgb_params(base: dict, lr: float, seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": lr,
        "seed": seed,
        "verbosity": 0,
        **base,
    }


def _bucket_calibration(
    preds: np.ndarray, actuals: np.ndarray, n_buckets: int
) -> list:
    """Quantile-bucket preds, return list of {bucket, n, mean_pred, mean_actual}."""
    if len(preds) == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, n_buckets + 1)
    edges = np.quantile(preds, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    table: list = []
    for i in range(n_buckets):
        lo, hi = edges[i], edges[i + 1]
        # last bucket inclusive on both sides; others left-inclusive
        if i == n_buckets - 1:
            mask = (preds >= lo) & (preds <= hi)
        else:
            mask = (preds >= lo) & (preds < hi)
        if not mask.any():
            table.append(
                {"bucket": i, "n": 0, "mean_pred": float("nan"),
                 "mean_actual": float("nan")}
            )
            continue
        table.append({
            "bucket": i,
            "n": int(mask.sum()),
            "mean_pred": float(preds[mask].mean()),
            "mean_actual": float(actuals[mask].mean()),
        })
    return table


def _is_monotonic_increasing(table: list) -> bool:
    """Pearson correlation of mean_pred vs mean_actual >= 0 across buckets.

    Stricter than 'every step rises' — sampling noise can flip one adjacent
    pair. We require the overall trend to point the right way.
    """
    valid = [r for r in table if r["n"] > 0
             and not np.isnan(r["mean_pred"])
             and not np.isnan(r["mean_actual"])]
    if len(valid) < 3:
        return False
    preds = np.array([r["mean_pred"] for r in valid])
    actuals = np.array([r["mean_actual"] for r in valid])
    if preds.std() == 0 or actuals.std() == 0:
        return False
    corr = float(np.corrcoef(preds, actuals)[0, 1])
    return corr > 0.0


def calibration_probe(
    samples: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    params: Optional[dict] = None,
    n_folds: int = 5,
    embargo_hours: int = 4,
    n_buckets: int = 10,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
    feature_set: str = "v1",
) -> dict:
    """Run walk-forward CV, bucket OOF preds, return gate decision.

    Returns dict with:
        fold_aucs: per-fold AUC list
        mean_auc: mean across folds
        calibration_table: [{bucket, n, mean_pred, mean_actual}, ...]
        is_monotonic: bool — actuals trend up with preds
        gate_passed: bool — mean_auc >= 0.55 AND is_monotonic
        gate_reason: str — human-readable explanation
    """
    cfg = dict(params) if params is not None else dict(_DEFAULT_PARAMS)
    features, feature_names = extract_features(samples, feature_set=feature_set)
    labels = np.asarray(labels, dtype=np.float32)

    splits = list(
        purged_walk_forward_splits(timestamps, n_folds, embargo_hours)
    )

    fold_aucs: list = []
    oof_pred: list = []
    oof_actual: list = []

    for train_idx, val_idx in splits:
        x_tr, y_tr = features[train_idx], labels[train_idx]
        x_va, y_va = features[val_idx], labels[val_idx]
        if len(np.unique(y_va)) < 2 or len(np.unique(y_tr)) < 2:
            fold_aucs.append(0.5)
            continue
        d_tr = xgb.DMatrix(x_tr, label=y_tr, feature_names=feature_names)
        d_va = xgb.DMatrix(x_va, label=y_va, feature_names=feature_names)
        booster = xgb.train(
            _xgb_params(cfg, learning_rate, seed),
            d_tr,
            num_boost_round=n_estimators,
            evals=[(d_va, "val")],
            verbose_eval=False,
        )
        preds = booster.predict(d_va)
        fold_aucs.append(float(roc_auc_score(y_va, preds)))
        oof_pred.append(preds)
        oof_actual.append(y_va)

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5

    if oof_pred:
        all_pred = np.concatenate(oof_pred)
        all_actual = np.concatenate(oof_actual)
    else:
        all_pred = np.array([])
        all_actual = np.array([])

    table = _bucket_calibration(all_pred, all_actual, n_buckets)
    is_monotonic = _is_monotonic_increasing(table)

    gate_passed = (mean_auc >= _GATE_AUC) and is_monotonic
    if gate_passed:
        gate_reason = (
            f"PASS: mean_auc={mean_auc:.3f} >= {_GATE_AUC}, "
            f"calibration is monotonic"
        )
    else:
        reasons = []
        if mean_auc < _GATE_AUC:
            reasons.append(f"mean_auc={mean_auc:.3f} < {_GATE_AUC}")
        if not is_monotonic:
            reasons.append("calibration is non-monotonic")
        gate_reason = "FAIL: " + "; ".join(reasons)

    return {
        "fold_aucs": fold_aucs,
        "mean_auc": mean_auc,
        "calibration_table": table,
        "is_monotonic": is_monotonic,
        "gate_passed": gate_passed,
        "gate_reason": gate_reason,
        "feature_names": feature_names,
    }
