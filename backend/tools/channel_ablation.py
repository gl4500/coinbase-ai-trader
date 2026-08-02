"""Single-channel ablation harness for the 27-channel CNN feature stack.

For each channel c in [0, 27): zero c across all timesteps in X, run
purged 5-fold walk-forward with the v1 feature extractor + the same XGB
config as feature_set_compare, and report Δ mean_auc vs the no-drop
baseline. Channels with delta ≈ 0 are dead weight; channels with strongly
negative delta are load-bearing.

Channels {17, 18, 19} are already inference-masked (`MASKED_CHANNELS` in
xgb_features.py) — extract_features zeros their stats either way, so
their ablation row should report ~0 delta and acts as a sanity check.

Run:
    cd backend && python tools/channel_ablation.py
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

import torch  # noqa: E402

from tools.feature_set_compare import _pooled_top_n  # noqa: E402
from tools.walk_forward import purged_walk_forward_splits  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
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


def _zero_channel(X: np.ndarray, channel: int) -> np.ndarray:
    """Return a copy of X with channel `channel` zeroed across all timesteps."""
    if not (0 <= channel < X.shape[1]):
        raise IndexError(f"channel {channel} out of range [0, {X.shape[1]})")
    out = X.copy()
    out[:, channel, :] = 0
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


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    n_folds: int = 5,
    embargo_hours: int = 4,
    n_estimators: int = 200,
) -> Dict:
    """Run single-channel ablation. Returns baseline_auc + per-channel rows."""
    n_channels = X.shape[1]

    feats0, names = extract_features(X, feature_set="v1")
    baseline = _cv_mean_auc(feats0, y, ts, names, n_folds, embargo_hours, n_estimators)

    rows: List[Dict] = []
    for c in range(n_channels):
        Xc = _zero_channel(X, c)
        feats_c, _ = extract_features(Xc, feature_set="v1")
        m = _cv_mean_auc(feats_c, y, ts, names, n_folds, embargo_hours, n_estimators)
        rows.append({"channel": c, "mean_auc": m, "delta": m - baseline})

    return {"baseline_auc": baseline, "rows": rows}


def _channel_label(c: int) -> str:
    """Map channel index to short human-readable name."""
    labels = {
        0: "log_ret_1h",
        1: "log_ret_4h",
        2: "log_ret_24h",
        3: "vol_z",
        4: "atr_pct",
        5: "rsi14",
        6: "rsi_macro",
        7: "macd_hist",
        8: "bb_width",
        9: "ema_cross",
        10: "L1_bid",
        11: "L1_ask",
        12: "spread_bps",
        13: "vwap_dev",
        14: "obv",
        15: "adx",
        16: "stoch_k",
        17: "MASKED_iv",
        18: "MASKED_rv_iv",
        19: "MASKED_skew",
        20: "funding",
        21: "btc_corr",
        22: "rv20",
        23: "rv60",
        24: "MASKED?",
        25: "MASKED?",
        26: "MASKED?",
    }
    return labels.get(c, f"ch{c}")


def main():
    print(f"Loading cache: {_CACHE_PATH}")
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    X, y, ts = _pooled_top_n(prods, n=20)
    print(f"  pooled samples: n={len(y):,}\n")

    results = run_ablation(X, y, ts, n_folds=5, embargo_hours=4, n_estimators=200)
    baseline = results["baseline_auc"]
    rows = sorted(results["rows"], key=lambda r: r["delta"])

    print(f"baseline (no drop) mean_auc = {baseline:.4f}\n")
    print(f"{'rank':>4} {'ch':>4} {'name':<14} {'mean_auc':>10} {'delta':>9}")
    print("-" * 50)
    for rank, r in enumerate(rows, 1):
        c = r["channel"]
        d = r["delta"]
        m = r["mean_auc"]
        marker = "  *load-bearing*" if d <= -0.005 else ("  dead?" if abs(d) < 0.001 else "")
        print(f"{rank:>4} {c:>4} {_channel_label(c):<14} {m:>10.4f} {d:>+9.4f}{marker}")


if __name__ == "__main__":
    main()
