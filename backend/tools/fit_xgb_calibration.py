"""Fit isotonic calibrator for the XGB booster (XGB-Step2 #180, refit #187).

Two source modes:

* `shadow` (legacy, #180) — live (raw_prob, win_label) pairs from the
  signal_outcomes table since the Phase 6 deploy. Small, biased toward the
  high-prob tail (resolved CNN BUYs only). When the sample is too small
  (~few hundred rows) isotonic creates one wide plateau covering ~95% of
  the booster's [0.4, 0.6] output range and live XGB collapses to a
  near-constant ~0.4346 (cnn_xgb_delta_probe 2026-05-04 finding — XGBcal
  AUC dropped from 0.60 raw to 0.05 calibrated).

* `cache` (added #187) — booster predictions + 4h ±0.3% labels over the
  chronological 20% val split of cnn_dataset_cache.pt (~83k samples,
  balanced labels). Same calibration-set principle as
  sklearn's CalibratedClassifierCV: fit on a held-out slice of the same
  distribution the booster was trained on. Produces a non-collapsed
  monotone curve that preserves ranking AUC within tolerance.

Usage (from backend/):
    # default — legacy shadow path (kept for backward compat)
    ../.venv/Scripts/python.exe -m tools.fit_xgb_calibration

    # new — refit on cache val split (recommended; #187)
    ../.venv/Scripts/python.exe -m tools.fit_xgb_calibration --source cache
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sqlite3
import sys
from typing import Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB = os.path.join(_BACKEND_DIR, "coinbase.db")
_DEFAULT_OUT = os.path.join(_BACKEND_DIR, "xgb_calibration.pkl")
_DEFAULT_SHADOW_START = "2026-05-03 19:15:15"
_DEFAULT_CACHE_PATH = os.path.join(_BACKEND_DIR, "cnn_dataset_cache.pt")
_DEFAULT_MODEL_PATH = os.path.join(_BACKEND_DIR, "xgb_model.json")
_DEFAULT_FEATURES_PATH = os.path.join(_BACKEND_DIR, "xgb_features.json")
_MIN_SAMPLES = 200
_MIN_CACHE_SAMPLES = 5_000


def _load_shadow_pairs(db_path: str, shadow_start: str) -> tuple[np.ndarray, np.ndarray]:
    """Pull (predicted_prob, win_label) pairs from resolved shadow BUYs."""
    c = sqlite3.connect(db_path)
    rows = c.execute(
        """
        SELECT confidence, outcome
        FROM signal_outcomes
        WHERE source='CNN'
          AND side='BUY'
          AND created_at >= ?
          AND checked_at IS NOT NULL
          AND outcome IN ('WIN', 'LOSS')
          AND confidence IS NOT NULL
        """,
        (shadow_start,),
    ).fetchall()
    c.close()
    if not rows:
        return np.array([]), np.array([])
    probs = np.array([float(r[0]) for r in rows], dtype=np.float64)
    wins = np.array([1.0 if r[1] == "WIN" else 0.0 for r in rows], dtype=np.float64)
    return probs, wins


def _detect_feature_set(feature_names) -> str:
    """v1/v2/v3 detection by name infix/prefix. v3 introduces the _mWWW_
    infix (m060/m168/m336); v2 introduces the xt_ prefix; v1 is the default.
    (#311f)"""
    names = [str(n) for n in feature_names]
    if any(("_m060_" in n) or ("_m168_" in n) or ("_m336_" in n) for n in names):
        return "v3"
    if any(n.startswith("xt_") for n in names):
        return "v2"
    return "v1"


def _save_calibrator(calibrator, out_path: str, feature_set: str = "v1") -> None:
    """Pickle calibrator with feature_set metadata for v3-aware loading. (#311f)

    Writes dict shape {"calibrator", "feature_set"} so xgb_signal._try_load
    can detect mismatches between calibrator and booster feature_sets and
    skip calibration when they don't agree (avoids v1-fit-on-v3 mapping).
    """
    with open(out_path, "wb") as f:
        pickle.dump({"calibrator": calibrator, "feature_set": feature_set}, f)


def _detect_calibration_target_feature_set() -> str:
    """Inspect backend/xgb_features.json to decide which feature_set tag to
    write into the calibrator pickle. Defaults to 'v1' if metadata missing
    or unreadable. (#311f)"""
    import json
    try:
        meta = json.load(open(_DEFAULT_FEATURES_PATH))
        if isinstance(meta, dict):
            tag = meta.get("feature_set")
            if tag:
                return str(tag)
            names = meta.get("feature_names")
            if names:
                return _detect_feature_set(names)
    except Exception:
        pass
    return "v1"


def _load_cache_pairs(
    cache_path: str = _DEFAULT_CACHE_PATH,
    model_path: str = _DEFAULT_MODEL_PATH,
    features_path: str = _DEFAULT_FEATURES_PATH,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the booster over the chronological val 20% of the dataset cache.

    Mirrors tools.permutation_importance._load_val_split: sorted-by-pid
    concatenation, masking _TRAINING_CONSTANT_CHANNELS, 80/20 split. Then
    extracts XGB tabular features and runs booster.predict to produce raw
    probs aligned with cache labels (4h ±0.3% triple-barrier).

    Returns (raw_probs, labels) as float64 numpy arrays.
    """
    import json
    import torch
    import xgboost as xgb
    from agents.cnn_agent import (
        _DATASET_CACHE_PATH,
        _dataset_schema,
        _load_pp_cache,
        _TRAINING_CONSTANT_CHANNELS,
        SEQ_LEN,
        N_CHANNELS,
    )
    from tools.xgb_features import extract_features

    _FORWARD_HOURS = 4
    _LABEL_THRESH = 0.003

    schema = _dataset_schema(SEQ_LEN, _FORWARD_HOURS, _LABEL_THRESH, N_CHANNELS)
    cache_to_load = cache_path if cache_path else _DATASET_CACHE_PATH
    cache = _load_pp_cache(cache_to_load, schema)
    if not cache:
        raise RuntimeError(
            f"No matching dataset cache at {cache_to_load} (schema "
            f"version {schema['version']})"
        )

    X_list: list = []
    y_list: list = []
    for pid in sorted(cache.keys()):
        entry = cache[pid]
        if not entry:
            continue
        X_list.extend(entry.get("X", []))
        y_list.extend(entry.get("y", []))

    if len(X_list) < _MIN_CACHE_SAMPLES:
        raise RuntimeError(
            f"refusing to fit calibrator: only {len(X_list)} cache samples "
            f"(need >= {_MIN_CACHE_SAMPLES})"
        )

    X_all = torch.stack([
        x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
        for x in X_list
    ])
    for ch in _TRAINING_CONSTANT_CHANNELS:
        X_all[:, ch, :] = 0.0
    y_all = np.array(y_list, dtype=np.float64)

    split = max(1, int(len(X_list) * 0.8))
    X_val = X_all[split:].numpy()
    y_val = y_all[split:]

    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_names = list(meta.get("feature_names", []))
    feature_set = _detect_feature_set(feature_names)
    features, _ = extract_features(X_val, feature_set=feature_set)
    dmat = xgb.DMatrix(features, feature_names=feature_names)
    raw = booster.predict(dmat).astype(np.float64)
    return raw, y_val


def _bucket_table(probs: np.ndarray, wins: np.ndarray, edges: np.ndarray) -> list:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append((lo, hi, 0, float("nan")))
        else:
            rows.append((lo, hi, n, float(wins[mask].mean())))
    return rows


def fit_calibration(
    db_path: str = _DEFAULT_DB,
    out_path: str = _DEFAULT_OUT,
    shadow_start: str = _DEFAULT_SHADOW_START,
    source: str = "shadow",
    cache_path: str = _DEFAULT_CACHE_PATH,
    model_path: str = _DEFAULT_MODEL_PATH,
    features_path: str = _DEFAULT_FEATURES_PATH,
) -> dict:
    """Fit and persist an isotonic calibrator. Returns summary dict.

    source:
        "shadow" (default, legacy #180): pull (raw, label) from the
            signal_outcomes table since the Phase 6 deploy.
        "cache" (#187): run booster on the chronological 20% val split
            of cnn_dataset_cache.pt and use the cache's own 4h ±0.3% labels.
    """
    if source == "shadow":
        probs, wins = _load_shadow_pairs(db_path, shadow_start)
        min_required = _MIN_SAMPLES
        src_desc = f"signal_outcomes since {shadow_start}"
    elif source == "cache":
        probs, wins = _load_cache_pairs(cache_path, model_path, features_path)
        min_required = _MIN_CACHE_SAMPLES
        src_desc = f"cache val split ({cache_path})"
    else:
        raise ValueError(f"unknown source={source!r}; expected 'shadow' or 'cache'")
    n = len(probs)
    if n < min_required:
        raise RuntimeError(
            f"refusing to fit calibrator: only {n} samples from {source} "
            f"(need >= {min_required})"
        )
    log.info("loaded %d (raw_prob, label) pairs from %s", n, src_desc)

    edges = np.linspace(0.0, 1.0, 11)
    pre_table = _bucket_table(probs, wins, edges)
    log.info("PRE-calibration win rate by raw bucket:")
    for lo, hi, k, wr in pre_table:
        wr_str = f"{100*wr:5.1f}%" if not np.isnan(wr) else "  n/a"
        log.info("  [%.2f, %.2f)  n=%4d  win=%s", lo, hi, k, wr_str)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(probs, wins)
    calibrated = iso.transform(probs)

    post_table = _bucket_table(calibrated, wins, edges)
    log.info("POST-calibration win rate by calibrated bucket:")
    for lo, hi, k, wr in post_table:
        wr_str = f"{100*wr:5.1f}%" if not np.isnan(wr) else "  n/a"
        log.info("  [%.2f, %.2f)  n=%4d  win=%s", lo, hi, k, wr_str)

    grid = np.linspace(0.0, 1.0, 11)
    grid_cal = iso.transform(grid)
    log.info("Calibration grid (raw -> calibrated):")
    for r, c in zip(grid, grid_cal):
        log.info("  %.2f -> %.4f", r, c)

    feature_set = _detect_calibration_target_feature_set()
    _save_calibrator(iso, out_path, feature_set=feature_set)
    log.info("saved isotonic calibrator (feature_set=%s) -> %s", feature_set, out_path)

    return {
        "n_samples": n,
        "pre_table": pre_table,
        "post_table": post_table,
        "grid": list(zip(grid.tolist(), grid_cal.tolist())),
        "out_path": out_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="shadow", choices=["shadow", "cache"])
    ap.add_argument("--db", default=_DEFAULT_DB)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--shadow-start", default=_DEFAULT_SHADOW_START)
    ap.add_argument("--cache-path", default=_DEFAULT_CACHE_PATH)
    ap.add_argument("--model-path", default=_DEFAULT_MODEL_PATH)
    ap.add_argument("--features-path", default=_DEFAULT_FEATURES_PATH)
    args = ap.parse_args()
    try:
        fit_calibration(
            db_path=args.db,
            out_path=args.out,
            shadow_start=args.shadow_start,
            source=args.source,
            cache_path=args.cache_path,
            model_path=args.model_path,
            features_path=args.features_path,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
