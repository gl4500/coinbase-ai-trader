"""Fit isotonic calibrator from live XGB shadow data — XGB-Step2 (#180).

The Phase 4 calibration_probe walk-forward CV passed monotonicity, but live
4-day shadow win-rate-by-bucket is U-shaped:

    0.2-0.3: 30.4% (n=138)
    0.3-0.4: 27.8% (n=108)
    0.4-0.5: 13.7% (n=95)
    0.5-0.6: 11.5% (n=191)  <- trough
    0.6-0.7: 13.7% (n=168)
    0.7-0.8: 18.2% (n=181)
    0.8-0.9: 30.0% (n=120)  <- peak

The booster overconfidently calls ranging-market borderline buys at 0.5-0.6
and they lose. Walk-forward CV missed this because it pooled across regimes.
Post-hoc isotonic regression on live (raw_prob, win_label) pairs flattens
the U into a monotonic curve without retraining the booster — the booster's
ranking is fine, just its absolute-prob calibration was off.

Usage (from backend/):
    ../.venv/Scripts/python.exe -m tools.fit_xgb_calibration
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sqlite3
import sys

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
_MIN_SAMPLES = 200


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
) -> dict:
    """Fit and persist an isotonic calibrator. Returns summary dict."""
    probs, wins = _load_shadow_pairs(db_path, shadow_start)
    n = len(probs)
    if n < _MIN_SAMPLES:
        raise RuntimeError(
            f"refusing to fit calibrator: only {n} resolved shadow BUYs "
            f"(need >= {_MIN_SAMPLES})"
        )
    log.info("loaded %d (raw_prob, win_label) pairs since %s", n, shadow_start)

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

    with open(out_path, "wb") as f:
        pickle.dump(iso, f)
    log.info("saved isotonic calibrator -> %s", out_path)

    return {
        "n_samples": n,
        "pre_table": pre_table,
        "post_table": post_table,
        "grid": list(zip(grid.tolist(), grid_cal.tolist())),
        "out_path": out_path,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=_DEFAULT_DB)
    ap.add_argument("--out", default=_DEFAULT_OUT)
    ap.add_argument("--shadow-start", default=_DEFAULT_SHADOW_START)
    args = ap.parse_args()
    try:
        fit_calibration(
            db_path=args.db, out_path=args.out, shadow_start=args.shadow_start
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
