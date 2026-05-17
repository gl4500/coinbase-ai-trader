"""CNN vs XGB delta probe — Phase 1 retrospective on labeled cache.

Runs the active CNN checkpoint and the production XGB booster (+isotonic
calibration) on the same chronological val split (last 20% of
cnn_dataset_cache.pt) and reports:
  - probability distributions per model
  - Pearson correlation between cnn_prob and xgb_prob
  - decision-agreement matrix at the live threshold pair (0.8 BUY / 0.2 SELL)
  - per-model AUC and accuracy on the labeled targets
  - sample of disagreement cases

Read-only — touches no DB rows, no model checkpoints, no live process.

Run:
    cd C:/Users/gl450/polymarket_app/backend
    ../.venv/Scripts/python.exe -m tools.cnn_xgb_delta_probe
"""
from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import torch

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)


def _cnn_probs_batched(model, X: torch.Tensor, batch: int = 256) -> np.ndarray:
    """Run the CNN forward pass batched, return [N] sigmoid probs as float64."""
    out = np.empty(X.shape[0], dtype=np.float64)
    for s in range(0, X.shape[0], batch):
        e = min(s + batch, X.shape[0])
        with torch.no_grad():
            logits = model(X[s:e])
            probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        out[s:e] = probs.astype(np.float64)
    return out


def _xgb_probs(X: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """Run XGB booster + isotonic calibration on the same val tensor.

    Mirrors agents/xgb_signal.xgb_prob but vectorised for the whole batch.
    Returns (raw, calibrated) so callers can isolate calibrator collapse.
    """
    import json
    import pickle
    import xgboost as xgb
    from tools.xgb_features import extract_features

    model_path = os.path.join(_BACKEND_DIR, "xgb_model.json")
    feat_path = os.path.join(_BACKEND_DIR, "xgb_features.json")
    cal_path = os.path.join(_BACKEND_DIR, "xgb_calibration.pkl")

    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(feat_path, "r") as f:
        meta = json.load(f)
    feature_names = list(meta.get("feature_names", []))
    feature_set = "v2" if len(feature_names) > 270 else "v1"

    arr = X.cpu().numpy().astype(np.float64)
    features, _ = extract_features(arr, feature_set=feature_set)
    dmat = xgb.DMatrix(features, feature_names=feature_names)
    raw = booster.predict(dmat).astype(np.float64)

    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)
        cal = calibrator.transform(raw)
    else:
        cal = raw.copy()

    return raw, np.clip(cal, 0.01, 0.99)


def _load_val_split_with_products() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """Mirror permutation_importance._load_val_split but track pid per sample.

    Returns (X_val, y_val, w_val, products_val) — products_val is a list of
    str pid labels parallel to X_val rows.
    """
    from agents.cnn_agent import (
        _DATASET_CACHE_PATH, _dataset_schema, _load_pp_cache,
        _compute_uniqueness, _TRAINING_CONSTANT_CHANNELS,
        SEQ_LEN, N_CHANNELS, _DEVICE,
    )
    schema = _dataset_schema(SEQ_LEN, 4, 0.003, N_CHANNELS)
    cache = _load_pp_cache(_DATASET_CACHE_PATH, schema)
    if not cache:
        raise RuntimeError(f"No matching dataset cache at {_DATASET_CACHE_PATH}.")

    X_list, y_list, w_list, p_list = [], [], [], []
    for pid in sorted(cache.keys()):
        entry = cache[pid]
        if not entry:
            continue
        xs = entry.get("X", [])
        ys = entry.get("y", [])
        ws = _compute_uniqueness(entry.get("indices", []), 4, int(entry["last_n"]))
        if len(ws) != len(xs):
            ws = [1.0] * len(xs)
        X_list.extend(xs)
        y_list.extend(ys)
        w_list.extend(ws)
        p_list.extend([pid] * len(xs))

    X_all = torch.stack([
        x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        for x in X_list
    ]).to(_DEVICE)
    for ch in _TRAINING_CONSTANT_CHANNELS:
        X_all[:, ch, :] = 0.0
    y_all = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)
    w_all = torch.tensor(w_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)

    split = max(1, int(len(X_list) * 0.8))
    return X_all[split:], y_all[split:], w_all[split:], p_list[split:]


def _summary(name: str, probs: np.ndarray) -> None:
    pct = np.percentile(probs, [5, 25, 50, 75, 95])
    print(
        f"  {name:5s}  n={len(probs):,d}  "
        f"mean={probs.mean():.4f}  std={probs.std():.4f}  "
        f"p5={pct[0]:.4f}  p50={pct[2]:.4f}  p95={pct[4]:.4f}"
    )


def _decisions(probs: np.ndarray, buy: float = 0.8, sell: float = 0.2) -> np.ndarray:
    """Map probs to integer side labels: 1=BUY, -1=SELL, 0=HOLD."""
    out = np.zeros(probs.shape, dtype=np.int8)
    out[probs > buy] = 1
    out[probs < sell] = -1
    return out


def _agreement_matrix(cnn_d: np.ndarray, xgb_d: np.ndarray) -> None:
    """Print 3x3 cross-tab of CNN side vs XGB side."""
    sides = [(-1, "SELL"), (0, "HOLD"), (1, "BUY")]
    print("\n  CNN \\ XGB    SELL    HOLD     BUY    total")
    for cs, cname in sides:
        row = [int(((cnn_d == cs) & (xgb_d == xs)).sum()) for xs, _ in sides]
        total = sum(row)
        print(f"  {cname:10s} {row[0]:7d} {row[1]:7d} {row[2]:7d} {total:8d}")
    col_totals = [int((xgb_d == xs).sum()) for xs, _ in sides]
    print(f"  {'total':10s} {col_totals[0]:7d} {col_totals[1]:7d} {col_totals[2]:7d} {sum(col_totals):8d}")
    agree = int((cnn_d == xgb_d).sum())
    n = len(cnn_d)
    print(f"\n  agreement: {agree}/{n} = {agree/n*100:.2f}%")


def _auc(probs: np.ndarray, y: np.ndarray) -> float:
    """ROC-AUC by Mann-Whitney U; numerically stable, no sklearn dep."""
    pos = probs[y == 1]
    neg = probs[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    pos_rank_sum = ranks[: len(pos)].sum()
    u = pos_rank_sum - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def _per_group_auc(
    group_labels: np.ndarray, cnn: np.ndarray, xgb_raw: np.ndarray,
    xgb_cal: np.ndarray, y: np.ndarray, label_name: str = "group",
    min_n: int = 50,
) -> None:
    """Pretty-print per-group AUC for CNN, XGB-raw, XGB-calibrated.

    Skips groups with fewer than min_n samples or no label variance.
    """
    uniq = sorted(set(group_labels.tolist()))
    rows = []
    for g in uniq:
        mask = group_labels == g
        n = int(mask.sum())
        if n < min_n:
            continue
        yg = y[mask]
        if yg.sum() == 0 or yg.sum() == n:
            continue
        rows.append((
            g, n, int(yg.sum()),
            _auc(cnn[mask], yg),
            _auc(xgb_raw[mask], yg),
            _auc(xgb_cal[mask], yg),
        ))
    if not rows:
        print(f"  (no {label_name} with n>={min_n} and label variance)")
        return
    rows.sort(key=lambda r: r[4])  # sort by xgb-RAW AUC ascending (signal-bearing)
    header = f"  {label_name:<14s} {'n':>7s} {'pos':>6s}  {'CNN':>7s}  {'XGBraw':>7s}  {'XGBcal':>7s}  {'d(CNN-raw)':>10s}"
    print(header)
    for g, n, pos, a_c, a_r, a_x in rows:
        delta = a_c - a_r
        print(f"  {str(g):<14s} {n:>7d} {pos:>6d}  {a_c:>7.4f}  {a_r:>7.4f}  {a_x:>7.4f}  {delta:>+10.4f}")


def main() -> None:
    print("[delta] loading val split + CNN model ...", flush=True)
    from tools.permutation_importance import _load_model

    X_val, y_val, _w, products_val = _load_val_split_with_products()
    model, arch = _load_model()
    n = X_val.shape[0]
    print(f"[delta] arch={arch}  n_val={n:,}  channels={X_val.shape[1]}  "
          f"distinct_products={len(set(products_val))}", flush=True)

    cnn = _cnn_probs_batched(model, X_val)
    print("[delta] CNN done; running XGB ...", flush=True)
    xgb_raw, xgb_cal = _xgb_probs(X_val)
    y = y_val.squeeze(-1).cpu().numpy().astype(np.int64)
    products_arr = np.array(products_val, dtype=object)

    print("\n=== probability distributions ===")
    _summary("CNN", cnn)
    _summary("XGBraw", xgb_raw)
    _summary("XGBcal", xgb_cal)

    if cnn.std() > 0 and xgb_cal.std() > 0:
        corr_cal = float(np.corrcoef(cnn, xgb_cal)[0, 1])
    else:
        corr_cal = float("nan")
    if cnn.std() > 0 and xgb_raw.std() > 0:
        corr_raw = float(np.corrcoef(cnn, xgb_raw)[0, 1])
    else:
        corr_raw = float("nan")
    print(f"\n  Pearson r(CNN, XGBraw) = {corr_raw:+.4f}")
    print(f"  Pearson r(CNN, XGBcal) = {corr_cal:+.4f}")

    print("\n=== overall AUC (whole val split) ===")
    print(f"  label balance: pos={(y == 1).sum()}/{n} = {(y == 1).mean()*100:.2f}%")
    print(f"  CNN     AUC = {_auc(cnn, y):.4f}")
    print(f"  XGBraw  AUC = {_auc(xgb_raw, y):.4f}    (booster output, no calibrator)")
    print(f"  XGBcal  AUC = {_auc(xgb_cal, y):.4f}    (after isotonic regression)")

    # ------------------------------------------------------------------
    # Per-product breakdown (each pid that has >=50 val samples)
    # ------------------------------------------------------------------
    print("\n=== per-product AUC (n>=50, label variance present) ===")
    _per_group_auc(products_arr, cnn, xgb_raw, xgb_cal, y,
                   label_name="product", min_n=50)

    # ------------------------------------------------------------------
    # Per-regime breakdown via Ch 15 (adx14)
    # ------------------------------------------------------------------
    print("\n=== per-regime AUC (split by ADX-14 median over T axis) ===")
    adx_mean = X_val[:, 15, :].mean(dim=1).cpu().numpy()
    cutoff = float(np.median(adx_mean))
    regime = np.where(adx_mean >= cutoff, "trending", "ranging")
    print(f"  ADX-14 median cutoff = {cutoff:+.4f}  (channel 15, mean over 60 timesteps)")
    _per_group_auc(regime, cnn, xgb_raw, xgb_cal, y, label_name="regime", min_n=50)

    # ------------------------------------------------------------------
    # Time-slice (4 chronological quintiles of the val split itself)
    # ------------------------------------------------------------------
    print("\n=== time-slice AUC (val split cut into 4 chronological quintiles) ===")
    edges = np.linspace(0, n, 5).astype(int)
    slices = np.empty(n, dtype=object)
    for q in range(4):
        slices[edges[q]:edges[q + 1]] = f"Q{q+1}"
    _per_group_auc(slices, cnn, xgb_raw, xgb_cal, y, label_name="slice", min_n=50)

    # ------------------------------------------------------------------
    # Decision agreement (live thresholds)
    # ------------------------------------------------------------------
    print("\n=== decision agreement at live thresholds (BUY > 0.8, SELL < 0.2) ===")
    cnn_d = _decisions(cnn)
    xgb_d = _decisions(xgb_cal)
    _agreement_matrix(cnn_d, xgb_d)


if __name__ == "__main__":
    main()
