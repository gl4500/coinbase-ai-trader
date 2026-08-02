"""XGB v4.5 3-class trainer (#xgb-v4.5 / Step B.1.5).

Reads OHLCV per pid from backend/data/history/<pid>.parquet. Builds 3-class
triple-barrier labels (DOWN=0, NEUTRAL=1, UP=2) at CLI-specified
--forward-hours / --label-thresh. Walk-forward splits chronologically.
Trains v4.5 booster (multi:softprob, num_class=3) on 7-channel features
(OHLCV + bb_pos + bb_width = 210 cols). Writes horizon-suffixed artifacts
at backend/xgb_*_v4_5_h<HOURS>.* paths. No calibrator in v4.5 (raw softmax).

Per feedback_python_clean_functions: main() delegates to small
single-responsibility helpers, each pure data-in/data-out.

Run (horizon sweep — operator runs 3 times, then v4_5_horizon_compare):
    cd backend && python -m tools.train_xgb_v4_5 \
      --pids BTC-USD,ETH-USD,... --forward-hours 24 --label-thresh 0.015
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_5_features import (  # noqa: E402
    BB_PERIOD,
    N_FEATURES_V45,
    TIER_WINDOWS_V45,
    extract_v4_5,
    feature_names_v4_5,
    feature_weights_v4_5,
)

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_DIR = BACKEND
# No defaults for --forward-hours / --label-thresh; operator MUST specify per
# the horizon sweep workflow. See spec "Architecture decisions".
_VAL_FRAC = 0.15
_CAL_FRAC = 0.15


# ── Pure helpers ──────────────────────────────────────────────────────────


def _load_candles_for_pid(pid: str, history_dir: str) -> List[Dict[str, float]]:
    """Read OHLCV candles for one pid from parquet. [] if file missing."""
    import pyarrow.parquet as pq

    path = os.path.join(history_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    rows = table.to_pydict()
    n = len(rows["start"])
    out: List[Dict[str, float]] = []
    for i in range(n):
        out.append(
            {
                "start": int(rows["start"][i]),
                "open": float(rows["open"][i]),
                "high": float(rows["high"][i]),
                "low": float(rows["low"][i]),
                "close": float(rows["close"][i]),
                "volume": float(rows["volume"][i]),
            }
        )
    out.sort(key=lambda r: r["start"])
    return out


def _triple_barrier_label_3class(
    closes: np.ndarray,
    start: int,
    forward_hours: int,
    label_thresh: float,
) -> Optional[int]:
    """3-class triple-barrier label.

    Returns:
        2 (UP)      if any forward close >= entry * (1 + label_thresh) hit first
        0 (DOWN)    if any forward close <= entry * (1 - label_thresh) hit first
        1 (NEUTRAL) if neither barrier hit within window (vertical timeout)
        None        if window truncated (start + forward_hours >= len(closes))

    Tie-break: UP barrier is checked before DOWN within each bar, so a bar
    that simultaneously crosses both gets UP (favors the actionable signal).
    """
    n = closes.size
    if start + forward_hours >= n:
        return None
    entry = closes[start]
    up_thr = entry * (1.0 + label_thresh)
    dn_thr = entry * (1.0 - label_thresh)
    for i in range(start + 1, start + forward_hours + 1):
        c = closes[i]
        if c >= up_thr:
            return 2
        if c <= dn_thr:
            return 0
    return 1


def _build_samples_for_pid(
    candles: List[Dict[str, float]],
    *,
    label_thresh: float,
    forward_hours: int,
    micro: int,
    meso: int,
    macro: int,
    bb_prefix: int = BB_PERIOD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each valid bar i where i >= macro+bb_prefix AND a label can be
    computed, produce one (features [210], int8 label, int64 timestamp).

    Returns:
        features:   (N, 210) float64
        labels:     (N,) int8 (0=DOWN, 1=NEUTRAL, 2=UP)
        timestamps: (N,) int64 (epoch seconds at sample bar)
    """
    n = len(candles)
    min_needed = macro + bb_prefix + forward_hours + 1
    if n < min_needed:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    feats_list: List[np.ndarray] = []
    labels_list: List[int] = []
    ts_list: List[int] = []
    for i in range(macro + bb_prefix, n):
        label = _triple_barrier_label_3class(
            closes,
            i,
            forward_hours,
            label_thresh,
        )
        if label is None:
            continue
        tier_slices = {
            # Include bb_prefix bars BEFORE each tier slice so bb_position
            # can be computed at every bar in the slice (the prefix bars
            # are used only for BB calculation, not for stats — _compute_stats
            # ignores prefix because it sees only the trailing tier_window).
            "micro": candles[i - micro - bb_prefix : i],
            "meso": candles[i - meso - bb_prefix : i],
            "macro": candles[i - macro - bb_prefix : i],
        }
        feats, _ = extract_v4_5(tier_slices)
        feats_list.append(feats[0])
        labels_list.append(label)
        ts_list.append(candles[i]["start"])
    if not feats_list:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    X = np.stack(feats_list, axis=0)
    y = np.array(labels_list, dtype=np.int8)
    ts = np.array(ts_list, dtype=np.int64)
    return X, y, ts


def _walk_forward_split(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    embargo_bars: int,
    val_frac: float = _VAL_FRAC,
    cal_frac: float = _CAL_FRAC,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Chronological split (train, val, cal) with embargo gaps."""
    n = features.shape[0]
    cal_n = int(n * cal_frac)
    val_n = int(n * val_frac)
    train_end = n - val_n - cal_n - 2 * embargo_bars
    if train_end < 1:
        train_end = max(1, n - val_n - cal_n)
        embargo_bars = 0
    val_start = train_end + embargo_bars
    val_end = val_start + val_n
    cal_start = val_end + embargo_bars
    cal_end = cal_start + cal_n
    X_tr = features[:train_end]
    y_tr = labels[:train_end]
    X_va = features[val_start:val_end]
    y_va = labels[val_start:val_end]
    X_ca = features[cal_start:cal_end]
    y_ca = labels[cal_start:cal_end]
    return (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca)


def _train_booster_3class(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    feature_weights: np.ndarray,
):
    """Train one 3-class xgb.Booster (multi:softprob). Returns booster +
    val mlogloss."""
    import xgboost as xgb

    d_tr = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    d_tr.set_info(feature_weights=feature_weights)
    d_va = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    d_va.set_info(feature_weights=feature_weights)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "seed": 0,
    }
    booster = xgb.train(
        params,
        d_tr,
        num_boost_round=200,
        evals=[(d_va, "val")],
        verbose_eval=False,
    )
    val_pred = booster.predict(d_va)  # shape (N, 3)
    # mlogloss as quick sanity metric — full AUC per class in compare
    from sklearn.metrics import log_loss

    if len(set(y_val.tolist())) >= 2:
        val_mlogloss = float(log_loss(y_val, val_pred, labels=[0, 1, 2]))
    else:
        val_mlogloss = float("nan")
    return booster, val_mlogloss


def _save_artifacts(
    booster,
    feature_names: List[str],
    out_dir: str,
    *,
    forward_hours: int,
) -> Dict[str, str]:
    """Atomic write of model.json + features.json with horizon suffix.
    No calibrator file in v4.5 (raw softmax used)."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_h{forward_hours}"
    model_path = os.path.join(out_dir, f"xgb_model_v4_5{suffix}.json")
    feat_path = os.path.join(out_dir, f"xgb_features_v4_5{suffix}.json")

    # tmp paths: model tmp MUST end in .json (xgboost format auto-detection)
    model_tmp = os.path.join(out_dir, f"xgb_model_v4_5{suffix}.tmp.json")
    feat_tmp = feat_path + ".tmp"

    booster.save_model(model_tmp)
    os.replace(model_tmp, model_path)

    with open(feat_tmp, "w") as f:
        json.dump({"feature_names": feature_names, "feature_set": "v4_5"}, f)
    os.replace(feat_tmp, feat_path)

    return {"model": model_path, "features": feat_path}


# ── Orchestrator ──────────────────────────────────────────────────────────


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pids", required=True, help="comma-separated, e.g. BTC-USD,ETH-USD")
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument(
        "--forward-hours",
        type=int,
        required=True,
        help="label horizon in bars (24, 72, 168 per sweep)",
    )
    p.add_argument(
        "--label-thresh",
        type=float,
        required=True,
        help="triple-barrier threshold (e.g. 0.015, 0.03, 0.06)",
    )
    p.add_argument("--embargo-bars", type=int, default=0, help="defaults to forward_hours if 0")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    micro = TIER_WINDOWS_V45["micro"]
    meso = TIER_WINDOWS_V45["meso"]
    macro = TIER_WINDOWS_V45["macro"]
    embargo = args.embargo_bars if args.embargo_bars > 0 else args.forward_hours

    t0 = time.time()
    print(
        f"v4.5 train: pids={pids} forward_hours={args.forward_hours} "
        f"label_thresh={args.label_thresh} embargo_bars={embargo} "
        f"-> xgb_*_v4_5_h{args.forward_hours}.*",
        flush=True,
    )

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_t: List[np.ndarray] = []
    skipped: List[str] = []
    for pid in pids:
        candles = _load_candles_for_pid(pid, args.history_dir)
        if not candles:
            skipped.append(pid)
            print(f"  {pid}: no parquet — skip", flush=True)
            continue
        X, y, ts = _build_samples_for_pid(
            candles,
            label_thresh=args.label_thresh,
            forward_hours=args.forward_hours,
            micro=micro,
            meso=meso,
            macro=macro,
        )
        if X.shape[0] == 0:
            skipped.append(pid)
            print(f"  {pid}: too few candles ({len(candles)}) — skip", flush=True)
            continue
        all_X.append(X)
        all_y.append(y)
        all_t.append(ts)
        # Distribution check — for 3-class want all 3 classes represented
        cls_counts = {c: int((y == c).sum()) for c in (0, 1, 2)}
        print(f"  {pid}: {X.shape[0]:,} samples, class counts={cls_counts}", flush=True)

    if not all_X:
        print("ERROR: no usable pids", flush=True)
        return 1

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    order = np.argsort(t, kind="stable")
    X = X[order]
    y = y[order]
    t = t[order]
    cls_counts = {c: int((y == c).sum()) for c in (0, 1, 2)}
    print(f"\nPooled: X={X.shape} class counts={cls_counts}", flush=True)

    (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
        X,
        y,
        t,
        embargo_bars=embargo,
    )
    print(f"Split: train={X_tr.shape} val={X_va.shape} cal={X_ca.shape}", flush=True)

    names = feature_names_v4_5()
    weights = feature_weights_v4_5()
    booster, val_mlogloss = _train_booster_3class(
        X_tr,
        y_tr,
        X_va,
        y_va,
        names,
        weights,
    )
    print(f"Train done: val_mlogloss={val_mlogloss:.4f}", flush=True)

    paths = _save_artifacts(
        booster,
        names,
        args.out_dir,
        forward_hours=args.forward_hours,
    )
    print(f"Wrote: {paths}", flush=True)
    print(f"Skipped pids: {skipped}", flush=True)
    print(f"Total wall: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
