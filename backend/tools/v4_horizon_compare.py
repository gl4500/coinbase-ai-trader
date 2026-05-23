"""XGB v4 horizon sweep comparison report (#xgb-v4 / Step B.1).

For each horizon (4, 24, 72, 168), load that horizon's artifacts (booster +
calibrator + feature_names), build a held-out test set at that horizon,
compute AUC + logloss + n_samples + pos_frac, render side-by-side HTML
report.

Per feedback_python_clean_functions: pure-function helpers, main()
orchestrator only.

Run (after all 4 horizons have been trained via train_xgb_v4.py):
    cd backend && python -m tools.v4_horizon_compare \
      --horizons 4,24,72,168 \
      --pids BTC-USD,ETH-USD,SOL-USD,...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = BACKEND
_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_PATH = os.path.join(
    BACKEND, "tools", "xgb_v4_horizon_compare.html"
)
# Map horizon -> label_thresh (must match what train_xgb_v4 was run with)
_HORIZON_THRESHOLDS: Dict[int, float] = {
    4: 0.003, 24: 0.01, 72: 0.02, 168: 0.05,
}


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_horizon_artifacts(horizon: int, base_dir: str) -> Dict[str, object]:
    """Load booster + calibrator + feature_names for one horizon.

    Expected files:
      base_dir/xgb_model_v4_h<H>.json
      base_dir/xgb_features_v4_h<H>.json
      base_dir/xgb_calibration_v4_h<H>.pkl
    """
    import xgboost as xgb

    model_path = os.path.join(base_dir, f"xgb_model_v4_h{horizon}.json")
    feat_path  = os.path.join(base_dir, f"xgb_features_v4_h{horizon}.json")
    cal_path   = os.path.join(base_dir, f"xgb_calibration_v4_h{horizon}.pkl")
    for p in (model_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"horizon h{horizon} artifact missing: {p}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)["feature_names"]
    calibrator: Optional[object] = None
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "calibrator" in obj:
            calibrator = obj["calibrator"]
    return {
        "booster": booster,
        "calibrator": calibrator,
        "feature_names": feature_names,
    }


def _evaluate_on_holdout(
    booster, calibrator, X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute AUC + logloss + pos_frac on a held-out set.

    Returns dict with keys 'auc', 'logloss', 'pos_frac', 'n_samples'.
    AUC is nan when y has a single class.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss

    n = X.shape[0]
    pos_frac = float(y.mean()) if n > 0 else 0.0
    if n == 0:
        return {"auc": float("nan"), "logloss": float("nan"),
                "pos_frac": pos_frac, "n_samples": 0}
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    raw = booster.predict(dmat)
    if calibrator is not None:
        raw = calibrator.transform(raw)
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, raw)) if len(set(y)) == 2 else float("nan")
    ll = float(log_loss(y, raw)) if len(set(y)) == 2 else float("nan")
    return {"auc": auc, "logloss": ll,
            "pos_frac": pos_frac, "n_samples": n}


def _build_holdout_dataset(
    pids: List[str], horizon: int, label_thresh: float,
    history_dir: str, holdout_frac: float = 0.15,
):
    """Build a held-out (X, y) test set per pid using the LAST holdout_frac
    of each pid's history (chronologically AFTER what train_xgb_v4 used).

    Uses _build_samples_for_pid from train_xgb_v4 for consistency.
    """
    from tools.train_xgb_v4 import _build_samples_for_pid, _load_candles_for_pid
    from tools.xgb_v4_features import TIER_WINDOWS_V4, N_FEATURES_V4

    micro = TIER_WINDOWS_V4["micro"]
    meso  = TIER_WINDOWS_V4["meso"]
    macro = TIER_WINDOWS_V4["macro"]

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for pid in pids:
        candles = _load_candles_for_pid(pid, history_dir)
        if not candles:
            continue
        X, y, _ts = _build_samples_for_pid(
            candles, label_thresh=label_thresh, forward_hours=horizon,
            micro=micro, meso=meso, macro=macro,
        )
        if X.shape[0] == 0:
            continue
        # Take last holdout_frac of samples per pid
        n_hold = max(1, int(X.shape[0] * holdout_frac))
        all_X.append(X[-n_hold:])
        all_y.append(y[-n_hold:])
    if not all_X:
        return np.zeros((0, N_FEATURES_V4), dtype=np.float64), np.zeros(0, dtype=np.int8)
    return np.vstack(all_X), np.concatenate(all_y)


def _render_html_report(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    out_path: str,
) -> None:
    """Side-by-side HTML report (dark mode, matches xgb_v3_channel_options.html style)."""
    # Determine winner by AUC (highest, ignoring NaN)
    valid = {h: m for h, m in metrics_by_horizon.items()
             if not np.isnan(m.get("auc", float("nan")))}
    winner = max(valid, key=lambda h: valid[h]["auc"]) if valid else None

    rows = []
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        cls = "winner" if h == winner else ""
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>h{h}</td>"
            f"<td class='num'>{m['auc']:.4f}</td>"
            f"<td class='num'>{m['logloss']:.4f}</td>"
            f"<td class='num'>{m['pos_frac']:.4f}</td>"
            f"<td class='num'>{m['n_samples']:,}</td>"
            f"</tr>"
        )

    winner_banner = (
        f"<div class='banner'>Winner: <strong>h{winner}</strong> "
        f"(AUC {valid[winner]['auc']:.4f})</div>"
    ) if winner is not None else "<div class='banner'>No valid AUC computed.</div>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>XGB v4 horizon comparison</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,sans-serif;
          padding:32px; max-width:900px; margin:auto; }}
  h1 {{ color:#fff; }}
  .banner {{ background:#1f3a1f; border:1px solid #1f6b33; color:#56d364;
             padding:14px 20px; border-radius:6px; margin:20px 0; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; color:#8b949e; padding:8px; border-bottom:1px solid #30363d; }}
  td {{ padding:8px; border-bottom:1px solid #21262d; font-family:ui-monospace,monospace; }}
  tr.winner td {{ background:#0d1c11; color:#56d364; font-weight:600; }}
  .num {{ text-align:right; }}
</style></head><body>
<h1>XGB v4 horizon comparison</h1>
{winner_banner}
<table>
  <tr><th>horizon</th><th>auc</th><th>logloss</th><th>pos_frac</th><th>n_samples</th></tr>
  {''.join(rows)}
</table>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", required=True,
                   help="comma-separated, e.g. 4,24,72,168")
    p.add_argument("--pids", required=True,
                   help="comma-separated pid list (same as train_xgb_v4)")
    p.add_argument("--base-dir", default=_DEFAULT_BASE_DIR,
                   help="directory containing xgb_*_v4_h<N>.* artifacts")
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-path", default=_DEFAULT_OUT_PATH)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]

    metrics: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        thresh = _HORIZON_THRESHOLDS.get(h)
        if thresh is None:
            print(f"  h{h}: no default threshold known — skipping", flush=True)
            continue
        print(f"  h{h}: loading artifacts...", flush=True)
        try:
            artifacts = _load_horizon_artifacts(h, args.base_dir)
        except FileNotFoundError as exc:
            print(f"  h{h}: {exc} — skip", flush=True)
            continue
        print(f"  h{h}: building holdout dataset...", flush=True)
        X, y = _build_holdout_dataset(pids, h, thresh, args.history_dir)
        print(f"  h{h}: evaluating on {X.shape[0]} samples...", flush=True)
        metrics[h] = _evaluate_on_holdout(
            artifacts["booster"], artifacts["calibrator"],
            X, y, artifacts["feature_names"],
        )
        m = metrics[h]
        print(f"  h{h}: auc={m['auc']:.4f} logloss={m['logloss']:.4f} "
              f"n={m['n_samples']} pos_frac={m['pos_frac']:.4f}", flush=True)

    _render_html_report(metrics, args.out_path)
    print(f"\nHTML report: {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
