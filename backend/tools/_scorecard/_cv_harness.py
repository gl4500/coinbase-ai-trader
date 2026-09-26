"""v3 driver scorecard: sample building + per-fold OOF prediction.

The v3 booster is NOT trained on cnn_dataset_cache.pt. It is trained by
tools.train_xgb.train_xgb_v3, which reads per-pid OHLCV parquets, builds
tiered candle slices (micro 60 / meso 168 / macro 336 bars), extracts v3
features via extract_features(feature_set='v3'), and labels each sample
`close[t+4] > close[t]` (a naive 4-bar-ahead direction — NOT the triple
barrier; that belongs to the CNN cache).

This harness mirrors that pipeline so the scorecard measures the v3 model as
actually trained and deployed. Realized return per sample is therefore the
plain 4-bar forward log-return ln(close[t+4]/close[t]).

Per-fold retraining (not the deployed booster) gives honest OOF predictions:
xgb_model.json was fit on the full dataset, so scoring its own training rows
would be in-sample. Booster config mirrors train_xgb_v3 (feature_weights_v3
on the DMatrix per CLAUDE.md invariant #13; subsample=0.7; colsample=0.8).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

_MIN_BARS = 336  # macro tier window
_HORIZON = 4  # v3 label/return horizon, bars
_MICRO, _MESO, _MACRO = 60, 168, 336


@dataclass
class V3Dataset:
    """Pooled, time-sorted v3 samples with pre-extracted features."""

    X: np.ndarray  # (N, 350) v3 features
    y: np.ndarray  # (N,) binary 0/1 = close[t+4] > close[t]
    entry_ts: np.ndarray  # (N,) int64 epoch seconds at the entry bar
    entry_close: np.ndarray  # (N,) close[t]
    exit_close: np.ndarray  # (N,) close[t+4]
    pid: np.ndarray  # (N,) object, source product id


def top_n_pids_from_cache(cache_path: str, n: int = 20, snapshot_ts=None) -> list[str]:
    """Survivorship-aware top-N product ids (#163), for comparability with the
    AUC probes. The cache is read only for this ranking — v3 samples are built
    from parquets, not the cache."""
    import torch

    from tools.pid_snapshot import survivorship_aware_top_n

    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    return list(survivorship_aware_top_n(blob["products"], n=n, snapshot_ts=snapshot_ts))


def build_v3_samples(
    pids: list[str],
    parquet_dir: str,
    sample_step: int = 24,
) -> V3Dataset:
    """Build pooled v3 samples from per-pid parquets — mirrors train_xgb_v3.

    For each pid: read OHLCV parquet, roll one sample every `sample_step` bars
    from bar 336 onward, build tiered slices, extract v3 features, label
    `close[t+4] > close[t]`. Records entry/exit close for realized returns.
    """
    import pandas as pd

    from tools.xgb_features import extract_features

    X_list, y_list, ts_list, ec_list, xc_list, pid_list = [], [], [], [], [], []
    for pid in pids:
        path = os.path.join(parquet_dir, f"{pid}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path).sort_values("start")
        if len(df) < _MIN_BARS:
            continue
        records = df.to_dict("records")
        starts = df["start"].to_numpy()
        closes = df["close"].to_numpy()
        for t in range(_MIN_BARS, len(starts) - _HORIZON, sample_step):
            tiers = {
                "micro": records[t - _MICRO : t],
                "meso": records[t - _MESO : t],
                "macro": records[t - _MACRO : t],
            }
            feats, _ = extract_features(tiers, feature_set="v3")
            X_list.append(feats[0])
            y_list.append(1 if closes[t + _HORIZON] > closes[t] else 0)
            ts_list.append(int(starts[t]))
            ec_list.append(float(closes[t]))
            xc_list.append(float(closes[t + _HORIZON]))
            pid_list.append(pid)

    if not X_list:
        raise RuntimeError(f"no v3 samples produced — check parquet_dir={parquet_dir!r}")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)
    ts = np.array(ts_list, dtype=np.int64)
    ec = np.array(ec_list, dtype=np.float64)
    xc = np.array(xc_list, dtype=np.float64)
    pid_arr = np.array(pid_list, dtype=object)
    order = np.argsort(ts, kind="stable")
    return V3Dataset(X[order], y[order], ts[order], ec[order], xc[order], pid_arr[order])


def train_fold_v3(X_tr: np.ndarray, y_tr: np.ndarray, X_va: np.ndarray) -> np.ndarray:
    """Train a fresh v3 booster on the fold's train rows, predict on val rows.

    X_tr / X_va are pre-extracted (N, 350) v3 feature matrices. Params and
    feature weights mirror tools.train_xgb.train_xgb_v3.
    """
    import xgboost as xgb

    from tools.xgb_features import _v3_feature_names, feature_weights_v3

    if len(np.unique(y_tr)) < 2:
        return np.full(len(X_va), 0.5, dtype=np.float64)

    names = _v3_feature_names()
    d_tr = xgb.DMatrix(X_tr, label=y_tr, feature_names=names)
    d_tr.set_info(feature_weights=feature_weights_v3())
    d_va = xgb.DMatrix(X_va, feature_names=names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": 0.05,
        "seed": 0,
        "verbosity": 0,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
    }
    booster = xgb.train(params, d_tr, num_boost_round=200)
    return np.asarray(booster.predict(d_va), dtype=np.float64)


def oof_predict_v3(
    ds: V3Dataset,
    n_folds: int = 5,
    embargo_hours: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[int, float]]:
    """Out-of-fold v3 predictions via purged walk-forward CV.

    Returns (scores, fold_ids, fold_spans_days): scores[i] the OOF prediction,
    fold_ids[i] the val fold, fold_spans_days the val-region span per fold.
    """
    from tools.walk_forward import purged_walk_forward_splits

    n = len(ds.y)
    scores = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_spans_days: dict[int, float] = {}

    splits = list(purged_walk_forward_splits(ds.entry_ts, n_folds, embargo_hours))
    for f_idx, (train_idx, val_idx) in enumerate(splits):
        scores[val_idx] = train_fold_v3(ds.X[train_idx], ds.y[train_idx], ds.X[val_idx])
        fold_ids[val_idx] = f_idx
        span = (int(ds.entry_ts[val_idx].max()) - int(ds.entry_ts[val_idx].min())) / 86400.0
        fold_spans_days[f_idx] = float(span) if span > 0 else 1.0
    return scores, fold_ids, fold_spans_days
