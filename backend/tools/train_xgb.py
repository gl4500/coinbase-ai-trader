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

from tools.walk_forward import purged_walk_forward_splits
from tools.xgb_features import extract_features

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

    splits = list(purged_walk_forward_splits(timestamps, n_folds, embargo_hours))

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


def train_xgb_v3(
    pids: Sequence[str],
    parquet_dir: str,
    out_dir: Union[str, os.PathLike],
    sample_step: int = 24,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict:
    """Train XGBoost booster with feature_set='v3' (mixed-lookback). (#311e)

    For each pid:
      - Loads parquet via services.tiered_history.fetch_tiered(source='parquet').
      - Skips pids with < 336 bars (macro window unsatisfiable).
      - Rolls one sample every `sample_step` bars; each sample is now_ts-truncated
        so future bars stay invisible.
      - Builds per-tier slices, extracts v3 features, label = 1 if close[t+4] > close[t].

    Writes xgb_model.json + xgb_features.json atomically (tmp + rename).
    Returns {"n_samples", "skipped_pids", "feature_set", "model_path",
             "features_path"}.
    """
    import logging as _log
    import shutil
    import time as _time

    import pandas as pd

    from tools.xgb_features import (
        _v3_feature_names,
        extract_features,
        feature_weights_v3,
    )

    _trainer_log = _log.getLogger("train_xgb_v3")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    skipped: list = []
    X_list: list = []
    y_list: list = []

    _t_start = _time.time()
    for pid_idx, pid in enumerate(pids):
        path = os.path.join(parquet_dir, f"{pid}.parquet")
        if not os.path.exists(path):
            skipped.append(pid)
            continue
        df = pd.read_parquet(path).sort_values("start")
        if len(df) < 336:
            skipped.append(pid)
            continue

        # Single parquet read per pid; slice the in-memory record list per
        # sample. 500x faster than calling tiered_history.fetch_tiered per
        # sample, which would re-read the same parquet from disk every time.
        records = df.to_dict("records")
        starts = df["start"].to_numpy()
        closes = df["close"].to_numpy()
        n_samples_this_pid = 0
        for t in range(336, len(starts) - 4, sample_step):
            tiers = {
                "micro": records[t - 60 : t],
                "meso": records[t - 168 : t] if t >= 168 else [],
                "macro": records[t - 336 : t] if t >= 336 else [],
            }
            feats, _ = extract_features(tiers, feature_set="v3")
            label = 1 if closes[t + 4] > closes[t] else 0
            X_list.append(feats[0])
            y_list.append(label)
            n_samples_this_pid += 1
        _trainer_log.info(
            "v3 features built: pid=%s (%d/%d) samples=%d elapsed=%.1fs",
            pid,
            pid_idx + 1,
            len(pids),
            n_samples_this_pid,
            _time.time() - _t_start,
        )
        # Also print so background nohup logs see progress (logging may be
        # silenced when called as a module under nohup with empty config).
        print(
            f"v3 features built: pid={pid} ({pid_idx + 1}/{len(pids)}) "
            f"samples={n_samples_this_pid} elapsed={_time.time() - _t_start:.1f}s",
            flush=True,
        )

    if not X_list:
        raise RuntimeError("no training samples produced — all pids skipped")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)
    names = _v3_feature_names()
    weights = feature_weights_v3()

    dtrain = xgb.DMatrix(X, label=y, feature_names=names)
    # feature_weights bias XGBoost's column subsampling toward macro/meso tiers
    # (set on DMatrix per xgboost API; xgb.train does NOT accept it directly).
    dtrain.set_info(feature_weights=weights)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": learning_rate,
        "seed": seed,
        "verbosity": 0,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,  # required for feature_weights to take effect
    }
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

    # Temp filenames keep ".json" as the LAST extension because xgboost picks
    # serialization format from the trailing extension. "xgb_model.json.tmp"
    # writes UBJSON (binary) and then the rename to ".json" leaves a binary
    # file that load_model parses as JSON and rejects.
    tmp_model = out_path / "xgb_model.tmp.json"
    tmp_feats = out_path / "xgb_features.tmp.json"
    booster.save_model(str(tmp_model))
    with open(tmp_feats, "w") as f:
        json.dump(
            {
                "feature_names": names,
                "feature_set": "v3",
                "best_params": {"max_depth": 4, "min_child_weight": 1, "subsample": 0.7},
                "feature_weights": weights.tolist(),
            },
            f,
        )
    shutil.move(str(tmp_model), str(out_path / "xgb_model.json"))
    shutil.move(str(tmp_feats), str(out_path / "xgb_features.json"))

    return {
        "n_samples": int(X.shape[0]),
        "skipped_pids": skipped,
        "feature_set": "v3",
        "model_path": str(out_path / "xgb_model.json"),
        "features_path": str(out_path / "xgb_features.json"),
    }
