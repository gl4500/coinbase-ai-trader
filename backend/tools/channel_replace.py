"""Single-channel replacement probe — measure Δ AUC of swapping one channel
with a candidate signal.

Pairs with channel_ablation.py (#146). Where ablation tells you what's
dead-weight, this answers: "if I substitute that channel with OI / hour-of-day
/ on-chain flow / etc., how much does it lift mean_auc?" Decision rule: if
Δ ≥ +0.01 the candidate is worth integrating into FeatureBuilder + cache
rebuild. Otherwise don't pay the integration cost.

Run examples (from main()):
    cd backend && python tools/channel_replace.py
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.walk_forward import purged_walk_forward_splits  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402

_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1,
    "subsample": 1.0,
    "seed": 0,
    "verbosity": 0,
}


def _replace_channel(
    X: np.ndarray,
    channel: int,
    replacement: np.ndarray,
) -> np.ndarray:
    """Return a copy of X with channel `channel` swapped for `replacement`.

    X shape: [N, C, T]. replacement shape must be [N, T].
    """
    if not (0 <= channel < X.shape[1]):
        raise IndexError(f"channel {channel} out of range [0, {X.shape[1]})")
    if replacement.shape != (X.shape[0], X.shape[2]):
        raise ValueError(
            f"replacement shape {replacement.shape} does not match (N, T) = "
            f"({X.shape[0]}, {X.shape[2]})"
        )
    out = X.copy()
    out[:, channel, :] = replacement
    return out


def _cv_mean_auc(
    feats: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    names: List[str],
    n_folds: int,
    embargo_hours: int,
    n_estimators: int,
) -> float:
    splits = list(purged_walk_forward_splits(ts, n_folds, embargo_hours))
    aucs: List[float] = []
    for tr, va in splits:
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            aucs.append(0.5)
            continue
        d_tr = xgb.DMatrix(feats[tr], label=y[tr], feature_names=names)
        d_va = xgb.DMatrix(feats[va], label=y[va], feature_names=names)
        b = xgb.train(_PARAMS, d_tr, num_boost_round=n_estimators)
        aucs.append(float(roc_auc_score(y[va], b.predict(d_va))))
    return float(np.mean(aucs))


def run_replace(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    channel_idx: int,
    replacement: np.ndarray,
    n_folds: int = 5,
    embargo_hours: int = 4,
    n_estimators: int = 200,
) -> Dict:
    """Compare baseline vs channel-replaced mean_auc under purged CV."""
    feats0, names = extract_features(X, feature_set="v1")
    baseline = _cv_mean_auc(feats0, y, ts, names, n_folds, embargo_hours, n_estimators)

    Xr = _replace_channel(X, channel_idx, replacement)
    feats_r, _ = extract_features(Xr, feature_set="v1")
    replaced = _cv_mean_auc(feats_r, y, ts, names, n_folds, embargo_hours, n_estimators)

    return {
        "channel": channel_idx,
        "baseline_auc": baseline,
        "replaced_auc": replaced,
        "delta": replaced - baseline,
    }


if __name__ == "__main__":
    print(
        "channel_replace.py is a library — wire a candidate signal in a "
        "runner (e.g., tools/oi_single_add_probe.py) and call run_replace()."
    )
