"""Sweep forward_hours to identify the horizon where signal is strongest.

Five candidates fail at the current 4h horizon (post-#149: AUC 0.5164, gate
0.55). This probe pivots the question: is the horizon, not the channel set,
the binding constraint?

For each h in {1, 4, 12, 24, 72} we keep the existing 27-channel feature
stack (horizon-agnostic), reload raw candles per pid, and recompute
triple-barrier labels at horizon h. Then run the same 5-fold purged
walk-forward XGB used in #149 / #146 / hour_of_day_probe.

No cache rebuild — features stay frozen. Only y changes.

Run:
    python tools/timescale_sweep.py
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.cnn_agent import _label_triple_barrier  # noqa: E402
from services.history_backfill import load_history  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.walk_forward import purged_walk_forward_splits  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_BAR_SECS = 3600
_HORIZONS = (1, 4, 12, 24, 72)
_UP_MULT = 0.01
_DN_MULT = 0.01
_LABEL_THRESH = 0.003

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


def _relabel_at_horizon(
    candles: List[dict],
    sample_ts: np.ndarray,
    forward_hours: int,
    up_mult: float = _UP_MULT,
    dn_mult: float = _DN_MULT,
    label_thresh: float = _LABEL_THRESH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map sample timestamps to candle indices, recompute triple-barrier label.

    Returns (y, mask). mask[i] = False when ts is unmappable or labeler
    returns None (insufficient lookahead, dead-zone return, invalid entry).
    """
    start_to_idx = {int(c["start"]): k for k, c in enumerate(candles)}
    n = len(sample_ts)
    y = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        ts = int(sample_ts[i])
        idx = start_to_idx.get(ts)
        if idx is None:
            continue
        lab = _label_triple_barrier(
            candles, idx, forward_hours, up_mult, dn_mult, label_thresh,
        )
        if lab is None:
            continue
        y[i] = float(lab)
        mask[i] = True
    return y, mask


def _load_pooled_with_pids(n: int = 20):
    """Pooled top-n cache load, retaining per-sample pid for relabeling."""
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    sized = sorted(
        ((pid, len(e.get("X", []))) for pid, e in prods.items()),
        key=lambda x: -x[1],
    )[:n]
    pids = [pid for pid, _ in sized]
    print(f"  pooled top-{n}: {pids}", flush=True)

    Xs, tss, pid_idx = [], [], []
    for pi, pid in enumerate(pids):
        X, _y_old, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        tss.append(ts)
        pid_idx.append(np.full(len(ts), pi, dtype=np.int32))
    X_all = np.concatenate(Xs, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    pid_all = np.concatenate(pid_idx, axis=0)
    return pids, X_all, ts_all, pid_all


def _cv_mean_auc(
    feats: np.ndarray,
    y: np.ndarray,
    ts: np.ndarray,
    names: List[str],
    n_folds: int,
    embargo_hours: int,
    n_estimators: int,
) -> Tuple[float, List[float]]:
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
    return float(np.mean(aucs)), aucs


def _run_horizon(
    h: int,
    pids: List[str],
    candles_by_pid: Dict[str, list],
    feats: np.ndarray,
    names: List[str],
    ts: np.ndarray,
    pid_idx: np.ndarray,
) -> Dict:
    print(f"\n--- horizon: {h}h ---", flush=True)
    n = len(ts)
    y = np.zeros(n, dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for pi, pid in enumerate(pids):
        sel = pid_idx == pi
        if not sel.any():
            continue
        candles = candles_by_pid.get(pid)
        if not candles:
            continue
        y_pid, m_pid = _relabel_at_horizon(candles, ts[sel], h)
        y[sel] = y_pid
        mask[sel] = m_pid
    n_ok = int(mask.sum())
    if n_ok == 0:
        print("  (no labeled samples — skipping)", flush=True)
        return {"horizon": h, "n": 0, "mean_auc": float("nan"), "fold_aucs": []}
    pos_rate = float(y[mask].mean())
    print(f"  labeled samples: {n_ok:,} / {n:,}  pos_rate={pos_rate:.3f}", flush=True)

    mean_auc, fold_aucs = _cv_mean_auc(
        feats[mask], y[mask], ts[mask], names,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    fold_str = ", ".join(f"{a:.3f}" for a in fold_aucs)
    print(f"  mean_auc={mean_auc:.4f}  folds=[{fold_str}]", flush=True)
    return {
        "horizon": h, "n": n_ok, "pos_rate": pos_rate,
        "mean_auc": mean_auc, "fold_aucs": fold_aucs,
    }


def main():
    pids, X, ts, pid_idx = _load_pooled_with_pids(n=20)
    print(f"\npooled samples: n={len(ts):,}", flush=True)

    print("Loading raw candles per pid...", flush=True)
    candles_by_pid: Dict[str, list] = {}
    for pid in pids:
        candles_by_pid[pid] = load_history(pid) or []
        print(f"  {pid}: {len(candles_by_pid[pid])} hourly bars", flush=True)

    print("\nExtracting v1 features (once, horizon-agnostic)...", flush=True)
    feats, names = extract_features(X, feature_set="v1")
    print(f"  feats shape: {feats.shape}", flush=True)

    results = []
    for h in _HORIZONS:
        results.append(_run_horizon(h, pids, candles_by_pid, feats, names, ts, pid_idx))

    print("\n=== summary ===", flush=True)
    print(f"{'horizon':>10} {'n':>10} {'pos_rate':>10} {'mean_auc':>10}", flush=True)
    for r in results:
        if r["n"] == 0:
            print(f"{r['horizon']:>9}h {'-':>10} {'-':>10} {'-':>10}", flush=True)
        else:
            print(
                f"{r['horizon']:>9}h {r['n']:>10,} "
                f"{r['pos_rate']:>10.3f} {r['mean_auc']:>10.4f}",
                flush=True,
            )

    valid = [r for r in results if r["n"] > 0]
    if valid:
        best = max(valid, key=lambda r: r["mean_auc"])
        print(
            f"\nbest horizon: {best['horizon']}h  "
            f"mean_auc={best['mean_auc']:.4f}  "
            f"(vs current 4h baseline)",
            flush=True,
        )


if __name__ == "__main__":
    main()
