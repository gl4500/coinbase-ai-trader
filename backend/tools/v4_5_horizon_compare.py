"""XGB v4.5 horizon + decision-rule comparison report (#xgb-v4.5 / Step B.1.5).

For each horizon (24/72/168): load 3-class artifacts, build per-pid last-15%
holdout, predict (N, 3) softmax probs, compute per-class AUC + macro-AUC +
logloss + class distribution. Then evaluate 3 decision rules
(argmax_margin / indep_thresholds / net_direction) on the same holdout —
precision/recall/F1 of BUY signal (labels==UP) and SELL signal (labels==DOWN).

Render side-by-side HTML report at backend/tools/xgb_v4_5_horizon_compare.html
with the (horizon, rule) combo highlighted by best buy_f1 + sell_f1 composite.

Per feedback_python_clean_functions: pure-function helpers, main()
orchestrator only.

Run (after all 3 horizons trained via train_xgb_v4_5.py):
    cd backend && python -m tools.v4_5_horizon_compare \
      --horizons 24,72,168 --pids BTC-USD,ETH-USD,...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = BACKEND
_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_PATH = os.path.join(BACKEND, "tools", "xgb_v4_5_horizon_compare.html")
_HORIZON_THRESHOLDS: Dict[int, float] = {24: 0.015, 72: 0.03, 168: 0.06}


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_horizon_artifacts(horizon: int, base_dir: str) -> Dict[str, object]:
    """Load v4.5 booster + feature_names for one horizon.

    Expected files:
      base_dir/xgb_model_v4_5_h<H>.json
      base_dir/xgb_features_v4_5_h<H>.json
    """
    import xgboost as xgb

    model_path = os.path.join(base_dir, f"xgb_model_v4_5_h{horizon}.json")
    feat_path  = os.path.join(base_dir, f"xgb_features_v4_5_h{horizon}.json")
    for p in (model_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"v4.5 horizon h{horizon} artifact missing: {p}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)["feature_names"]
    return {"booster": booster, "feature_names": feature_names}


def _evaluate_on_holdout_3class(
    booster, X: np.ndarray, y: np.ndarray, feature_names: List[str],
) -> Dict[str, float]:
    """Compute per-class AUC + macro-AUC + logloss + class distribution.

    Returns dict with: auc_down, auc_neutral, auc_up, auc_macro, logloss,
    n_samples, pos_frac_down, pos_frac_neutral, pos_frac_up. AUC for a
    class is NaN if no positive examples of that class in holdout.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss

    n = X.shape[0]
    out: Dict[str, float] = {
        "n_samples": n,
        "pos_frac_down":    float((y == 0).mean()) if n > 0 else 0.0,
        "pos_frac_neutral": float((y == 1).mean()) if n > 0 else 0.0,
        "pos_frac_up":      float((y == 2).mean()) if n > 0 else 0.0,
    }
    if n == 0:
        return {**out, "auc_down": float("nan"), "auc_neutral": float("nan"),
                "auc_up": float("nan"), "auc_macro": float("nan"),
                "logloss": float("nan")}
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    probs = booster.predict(dmat)  # (N, 3)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    aucs: List[float] = []
    for cls in (0, 1, 2):
        if (y == cls).sum() == 0 or (y != cls).sum() == 0:
            aucs.append(float("nan"))
            continue
        try:
            aucs.append(float(roc_auc_score((y == cls).astype(np.int8),
                                              probs[:, cls])))
        except ValueError:
            aucs.append(float("nan"))
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    out["auc_down"]    = aucs[0]
    out["auc_neutral"] = aucs[1]
    out["auc_up"]      = aucs[2]
    out["auc_macro"]   = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    if len(set(y.tolist())) >= 2:
        out["logloss"] = float(log_loss(y, probs, labels=[0, 1, 2]))
    else:
        out["logloss"] = float("nan")
    return out


def _evaluate_decision_rules(
    probs: np.ndarray,    # shape (N, 3)
    labels: np.ndarray,   # shape (N,) — 0/1/2
) -> Dict[str, Dict[str, float]]:
    """Per-rule scorecard.

    Each rule produces BUY/SELL/HOLD decisions. We score BUY signals against
    labels==2 (UP) and SELL signals against labels==0 (DOWN). Precision/recall
    of each (signal_class).

    Returns dict keyed by rule name, each value containing:
      buy_precision, buy_recall, buy_f1,
      sell_precision, sell_recall, sell_f1,
      trade_rate (BUY + SELL fraction), hold_rate.
    """
    n = probs.shape[0]
    p_down, p_neutral, p_up = probs[:, 0], probs[:, 1], probs[:, 2]
    argmax = probs.argmax(axis=1)

    rules_buy_sell: Dict[str, tuple] = {
        "argmax_margin": (
            (argmax == 2) & ((p_up - p_down) > 0.10),
            (argmax == 0) & ((p_down - p_up) > 0.10),
        ),
        "indep_thresholds": (
            (p_up > 0.50) & (p_up >= p_down),
            (p_down > 0.50) & (p_down > p_up),
        ),
        "net_direction": (
            (p_up - p_down) > 0.20,
            (p_down - p_up) > 0.20,
        ),
    }

    out: Dict[str, Dict[str, float]] = {}
    label_up = (labels == 2)
    label_dn = (labels == 0)

    def _prf(signal: np.ndarray, truth: np.ndarray) -> tuple:
        if signal.sum() == 0:
            precision = float("nan")
        else:
            precision = float((signal & truth).sum()) / float(signal.sum())
        if truth.sum() == 0:
            recall = float("nan")
        else:
            recall = float((signal & truth).sum()) / float(truth.sum())
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    for name, (buy, sell) in rules_buy_sell.items():
        bp, br, bf1 = _prf(buy, label_up)
        sp, sr, sf1 = _prf(sell, label_dn)
        trade_rate = float((buy | sell).mean()) if n > 0 else 0.0
        out[name] = {
            "buy_precision": bp, "buy_recall": br, "buy_f1": bf1,
            "sell_precision": sp, "sell_recall": sr, "sell_f1": sf1,
            "trade_rate": trade_rate, "hold_rate": 1.0 - trade_rate,
        }
    return out


def _build_holdout_dataset(
    pids: List[str], horizon: int, label_thresh: float,
    history_dir: str, holdout_frac: float = 0.15,
):
    """Build held-out (X, y) test set per pid using the LAST holdout_frac
    of each pid's history. Uses _build_samples_for_pid from train_xgb_v4_5."""
    from tools.train_xgb_v4_5 import (
        _build_samples_for_pid, _load_candles_for_pid,
    )
    from tools.xgb_v4_5_features import TIER_WINDOWS_V45, N_FEATURES_V45

    micro = TIER_WINDOWS_V45["micro"]
    meso  = TIER_WINDOWS_V45["meso"]
    macro = TIER_WINDOWS_V45["macro"]

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
        n_hold = max(1, int(X.shape[0] * holdout_frac))
        all_X.append(X[-n_hold:])
        all_y.append(y[-n_hold:])
    if not all_X:
        return (np.zeros((0, N_FEATURES_V45), dtype=np.float64),
                np.zeros(0, dtype=np.int8))
    return np.vstack(all_X), np.concatenate(all_y)


def _render_html_report(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    rules_by_horizon: Dict[int, Dict[str, Dict[str, float]]],
    out_path: str,
) -> None:
    """Side-by-side HTML report with per-horizon AUC + per-rule scorecard.
    Highlights winning horizon (best auc_macro) and winning rule per horizon
    (best composite buy_f1 + sell_f1)."""
    # Winning horizon by macro AUC (ignore NaN)
    valid_h = {h: m for h, m in metrics_by_horizon.items()
               if not np.isnan(m.get("auc_macro", float("nan")))}
    winner_h = max(valid_h, key=lambda h: valid_h[h]["auc_macro"]) if valid_h else None

    # Winning rule per horizon by buy_f1+sell_f1
    winner_rule_by_h: Dict[int, str] = {}
    for h, rules in rules_by_horizon.items():
        scored = {r: rules[r]["buy_f1"] + rules[r]["sell_f1"] for r in rules}
        winner_rule_by_h[h] = max(scored, key=lambda r: scored[r]) if scored else ""

    horizons_rows: List[str] = []
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        cls = "winner" if h == winner_h else ""
        horizons_rows.append(
            f"<tr class='{cls}'>"
            f"<td>h{h}</td>"
            f"<td class='num'>{m['auc_macro']:.4f}</td>"
            f"<td class='num'>{m['auc_down']:.4f}</td>"
            f"<td class='num'>{m['auc_neutral']:.4f}</td>"
            f"<td class='num'>{m['auc_up']:.4f}</td>"
            f"<td class='num'>{m['logloss']:.4f}</td>"
            f"<td class='num'>{m['n_samples']:,}</td>"
            f"<td class='num'>{m['pos_frac_down']:.2f}/{m['pos_frac_neutral']:.2f}/{m['pos_frac_up']:.2f}</td>"
            f"</tr>"
        )

    rule_blocks: List[str] = []
    for h in sorted(rules_by_horizon.keys()):
        rules = rules_by_horizon[h]
        w = winner_rule_by_h.get(h, "")
        rule_rows = []
        for r_name, r in rules.items():
            cls = "winner" if r_name == w else ""
            rule_rows.append(
                f"<tr class='{cls}'><td>{r_name}</td>"
                f"<td class='num'>{r['buy_precision']:.3f}</td>"
                f"<td class='num'>{r['buy_recall']:.3f}</td>"
                f"<td class='num'>{r['buy_f1']:.3f}</td>"
                f"<td class='num'>{r['sell_precision']:.3f}</td>"
                f"<td class='num'>{r['sell_recall']:.3f}</td>"
                f"<td class='num'>{r['sell_f1']:.3f}</td>"
                f"<td class='num'>{r['trade_rate']:.3f}</td>"
                f"</tr>"
            )
        rule_blocks.append(
            f"<h3>h{h} decision rules</h3><table>"
            "<tr><th>rule</th><th>buy_p</th><th>buy_r</th><th>buy_f1</th>"
            "<th>sell_p</th><th>sell_r</th><th>sell_f1</th><th>trade_rate</th></tr>"
            + "".join(rule_rows) + "</table>"
        )

    winner_banner = (
        f"<div class='banner'>Winning horizon: <strong>h{winner_h}</strong> "
        f"(auc_macro={valid_h[winner_h]['auc_macro']:.4f}) · "
        f"Winning rule: <strong>{winner_rule_by_h.get(winner_h, 'n/a')}</strong></div>"
    ) if winner_h is not None else "<div class='banner'>No valid metrics.</div>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>XGB v4.5 horizon + rule comparison</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,sans-serif;
          padding:32px; max-width:1100px; margin:auto; }}
  h1 {{ color:#fff; }}
  h3 {{ color:#79c0ff; margin-top:24px; }}
  .banner {{ background:#1f3a1f; border:1px solid #1f6b33; color:#56d364;
             padding:14px 20px; border-radius:6px; margin:20px 0; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:16px; }}
  th {{ text-align:left; color:#8b949e; padding:8px; border-bottom:1px solid #30363d; }}
  td {{ padding:8px; border-bottom:1px solid #21262d; font-family:ui-monospace,monospace; }}
  tr.winner td {{ background:#0d1c11; color:#56d364; font-weight:600; }}
  .num {{ text-align:right; }}
</style></head><body>
<h1>XGB v4.5 horizon + rule comparison</h1>
{winner_banner}
<h3>Per-horizon metrics</h3>
<table>
  <tr><th>horizon</th><th>auc_macro</th><th>auc_down</th><th>auc_neutral</th>
      <th>auc_up</th><th>logloss</th><th>n</th><th>class_fracs (D/N/U)</th></tr>
  {''.join(horizons_rows)}
</table>
{''.join(rule_blocks)}
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", required=True,
                   help="comma-separated, e.g. 24,72,168")
    p.add_argument("--pids", required=True,
                   help="comma-separated pid list")
    p.add_argument("--base-dir", default=_DEFAULT_BASE_DIR)
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-path", default=_DEFAULT_OUT_PATH)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]

    metrics: Dict[int, Dict[str, float]] = {}
    rules: Dict[int, Dict[str, Dict[str, float]]] = {}
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
        metrics[h] = _evaluate_on_holdout_3class(
            artifacts["booster"], X, y, artifacts["feature_names"],
        )
        # Re-predict for decision-rule eval (same probs)
        import xgboost as xgb
        if X.shape[0] > 0:
            dmat = xgb.DMatrix(X, feature_names=artifacts["feature_names"])
            probs = artifacts["booster"].predict(dmat)
            rules[h] = _evaluate_decision_rules(probs, y)
        else:
            rules[h] = {}
        m = metrics[h]
        print(f"  h{h}: auc_macro={m['auc_macro']:.4f} logloss={m['logloss']:.4f} "
              f"n={m['n_samples']} class_fracs="
              f"{m['pos_frac_down']:.2f}/{m['pos_frac_neutral']:.2f}/{m['pos_frac_up']:.2f}",
              flush=True)

    _render_html_report(metrics, rules, args.out_path)
    print(f"\nHTML report: {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
