"""XGB deployment-aligned scorecard orchestrator.

Composes per-metric computers from tools._scorecard.* into a single
ScorecardReport for a model variant, plus a CLI runner for the v3 driver.
"""

from __future__ import annotations

import os
from typing import Mapping, Sequence

import numpy as np

from tools._scorecard._ece import expected_calibration_error
from tools._scorecard._expected_return import expected_return_at_tau
from tools._scorecard._paper_sharpe import paper_sharpe_per_fold
from tools._scorecard._precision import precision_at_tau
from tools._scorecard._report import ScorecardReport

FEE_TIERS: Mapping[str, float] = {"retail": 0.006, "mid": 0.0025, "pro": 0.0005}
DEFAULT_TAU_GRID: tuple[float, ...] = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
)
N_FIRED_FLOOR = 100


def compute_scorecard(
    scores: np.ndarray,
    labels: np.ndarray,
    returns: np.ndarray,
    fold_ids: np.ndarray,
    fold_spans_days: Mapping[int, float],
    *,
    fee_tiers: Mapping[str, float] = FEE_TIERS,
    gate_tier: str = "retail",
    tau_grid: Sequence[float] = DEFAULT_TAU_GRID,
    n_ece_bins: int = 10,
) -> ScorecardReport:
    """Compute the full deployment-aligned scorecard for one model variant.

    Args:
        scores: (N,) score-class probabilities (e.g., p_up).
        labels: (N,) binary 0/1 ground truth (UP barrier hit).
        returns: (N,) realized log-return per sample.
        fold_ids: (N,) integer fold index 0..K-1.
        fold_spans_days: fold_id -> span in days.
        fee_tiers: name -> per-side fee. Default retail/mid/pro.
        gate_tier: which tier's metrics evaluate the hard gates.
        tau_grid: thresholds to sweep.
        n_ece_bins: equal-width bins for ECE.

    Returns:
        ScorecardReport.

    Raises:
        ValueError: if gate_tier not in fee_tiers.
    """
    if gate_tier not in fee_tiers:
        raise ValueError(f"gate_tier {gate_tier!r} not in fee_tiers {list(fee_tiers)}")

    pos_rate = float(labels.mean())
    per_tau_rows: list[dict] = []

    for tau in tau_grid:
        prec, n_fired = precision_at_tau(scores, labels, tau)
        row: dict = {"tau": float(tau), "precision": prec, "n_fired": n_fired}
        for tier_name, fee in fee_tiers.items():
            er, _ = expected_return_at_tau(scores, returns, tau, fee)
            sh_mean, sh_std, _ = paper_sharpe_per_fold(
                scores, returns, fold_ids, fold_spans_days, tau, fee
            )
            row[f"e_return_{tier_name}"] = er
            row[f"sharpe_mean_{tier_name}"] = sh_mean
            row[f"sharpe_std_{tier_name}"] = sh_std
        per_tau_rows.append(row)

    ece = expected_calibration_error(scores, labels, n_bins=n_ece_bins)

    rec_tau: float = float("nan")
    eligible = [
        row
        for row in per_tau_rows
        if row["n_fired"] >= N_FIRED_FLOOR
        and not np.isnan(row[f"e_return_{gate_tier}"])
        and row[f"e_return_{gate_tier}"] > 0
        and not np.isnan(row["precision"])
    ]
    if eligible:
        best = max(eligible, key=lambda r: r["precision"])
        rec_tau = best["tau"]

    if np.isnan(rec_tau):
        gates = {
            "precision": False,
            "expected_return": False,
            "paper_sharpe": False,
            "ece": ece < 0.05,
        }
    else:
        op = next(r for r in per_tau_rows if r["tau"] == rec_tau)
        gates = {
            "precision": op["precision"] >= pos_rate + 0.03,
            "expected_return": op[f"e_return_{gate_tier}"] > 0,
            "paper_sharpe": (
                not np.isnan(op[f"sharpe_mean_{gate_tier}"]) and op[f"sharpe_mean_{gate_tier}"] > 0
            ),
            "ece": ece < 0.05,
        }

    return ScorecardReport(
        per_tau_rows=per_tau_rows,
        ece=ece,
        recommended_operating_tau=rec_tau,
        gates_passed=gates,
        pos_rate=pos_rate,
        gate_tier=gate_tier,
    )


# ----------------------------------------------------------------------
# CLI — v3 driver track only (v4/v4.5 use a different data path; see plan
# tasks 7b/7c). Run from the backend/ cwd.
# ----------------------------------------------------------------------

_DEFAULT_CACHE = "cnn_dataset_cache.pt"
_DEFAULT_PARQUET_DIR = os.path.join("data", "history")
_DEFAULT_SAMPLE_STEP = 24


def _load_v3_track(cache_path: str, parquet_dir: str, sample_step: int) -> dict:
    """Build OOF v3 predictions, labels, realized returns, and fold metadata.

    Mirrors train_xgb_v3: builds tiered v3 samples from per-pid OHLCV parquets
    (label close[t+4] > close[t]). Realized return per sample is the plain
    4-bar forward log-return ln(close[t+4]/close[t]) — v3 has no barriers.
    """
    from tools._returns import realized_log_returns_per_sample
    from tools._scorecard._cv_harness import (
        build_v3_samples,
        oof_predict_v3,
        top_n_pids_from_cache,
    )

    pids = top_n_pids_from_cache(cache_path)
    ds = build_v3_samples(pids, parquet_dir, sample_step)
    scores, fold_ids, fold_spans_days = oof_predict_v3(ds)

    valid = ~np.isnan(scores) & (fold_ids >= 0)
    returns = realized_log_returns_per_sample(ds.entry_close[valid], ds.exit_close[valid])
    return {
        "scores": scores[valid],
        "labels": ds.y[valid].astype(int),
        "returns": returns,
        "fold_ids": fold_ids[valid],
        "fold_spans_days": fold_spans_days,
        "n_total": int(len(scores)),
        "n_valid": int(valid.sum()),
    }


def _format_report(report: ScorecardReport, track: str, extra: dict) -> str:
    lines = [
        f"=== Scorecard for track={track} (gate_tier={report.gate_tier}) ===",
        f"samples: {extra['n_valid']} scored / {extra['n_total']} pooled",
        f"OOF mean AUC: {extra['auc']:.4f}  (sanity anchor — v3 should land near 0.528)",
        f"pos_rate: {report.pos_rate:.4f}",
        f"ECE: {report.ece:.4f}  "
        f"(gate <0.05 => {'PASS' if report.gates_passed['ece'] else 'FAIL'})",
        f"Recommended operating tau: {report.recommended_operating_tau}",
        "",
        "Per-tau table (E[r] and Sharpe shown for retail / mid / pro tiers):",
        f"{'tau':>6}  {'prec':>6}  {'n_fired':>8}  "
        f"{'E[r]_R':>9}  {'E[r]_M':>9}  {'E[r]_P':>9}  "
        f"{'Sh_R':>7}  {'Sh_M':>7}  {'Sh_P':>7}",
    ]
    for row in report.per_tau_rows:
        lines.append(
            f"{row['tau']:>6.2f}  "
            f"{row['precision']:>6.3f}  {row['n_fired']:>8d}  "
            f"{row['e_return_retail']:>9.5f}  {row['e_return_mid']:>9.5f}  "
            f"{row['e_return_pro']:>9.5f}  "
            f"{row['sharpe_mean_retail']:>7.3f}  {row['sharpe_mean_mid']:>7.3f}  "
            f"{row['sharpe_mean_pro']:>7.3f}"
        )
    lines.append("")
    lines.append("Gates passed: " + ", ".join(f"{k}={v}" for k, v in report.gates_passed.items()))
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="XGB deployment-aligned scorecard runner (v3 driver track)"
    )
    parser.add_argument(
        "--track",
        required=True,
        choices=["v3", "v4", "v4.5"],
        help="Which XGB head to score (v1 implements v3 only)",
    )
    parser.add_argument(
        "--cache",
        default=_DEFAULT_CACHE,
        help="cnn_dataset_cache.pt — used only for the survivorship-aware top-20 pid ranking",
    )
    parser.add_argument(
        "--parquet-dir",
        default=_DEFAULT_PARQUET_DIR,
        help="OHLCV parquet directory (default: ./data/history)",
    )
    parser.add_argument(
        "--sample-step",
        type=int,
        default=_DEFAULT_SAMPLE_STEP,
        help="Roll one sample every N bars (default: 24, matches "
        "train_xgb_v3; lower = denser evaluation, slower)",
    )
    parser.add_argument(
        "--gate-tier",
        default="retail",
        choices=list(FEE_TIERS),
        help="Fee tier used to evaluate hard gates (default: retail)",
    )
    args = parser.parse_args()

    if args.track != "v3":
        raise NotImplementedError(
            f"--track {args.track} is not implemented in scorecard v1. v4 and "
            "v4.5 read OHLCV parquets with different extractors (extract_v4 / "
            "extract_v4_5) and a separate per-fold harness — see plan tasks "
            "7b/7c in docs/superpowers/plans/2026-05-18-xgb-deployment-scorecard.md."
        )

    data = _load_v3_track(args.cache, args.parquet_dir, args.sample_step)
    report = compute_scorecard(
        data["scores"],
        data["labels"],
        data["returns"],
        data["fold_ids"],
        data["fold_spans_days"],
        gate_tier=args.gate_tier,
    )
    try:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(data["labels"], data["scores"]))
    except Exception:
        auc = float("nan")
    print(
        _format_report(
            report,
            args.track,
            {"auc": auc, "n_total": data["n_total"], "n_valid": data["n_valid"]},
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
