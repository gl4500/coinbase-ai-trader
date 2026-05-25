"""Per-(pid, horizon) mining orchestrator for Phase 3.

Composes profit_tree + purged_wf + the deflation factor + Q0 gates + bootstrap CI.
Pure functions on torch.Tensor inputs (caller loads the parquet). No filesystem.

This task (Task 4) lands the pure-function HELPERS. The end-to-end
mine_profiles_for_pid_horizon body lands in Task 6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

_DEPTH_GRID    = (3, 5, 7)
_MIN_LEAF_GRID = (20, 50, 100)
_RETAIL_FEE    = 0.012
_Q0_AVG_WIN    = 0.05    # >= +5%
_Q0_AVG_LOSS   = -0.10   # avg_loss must be >= -0.10 to pass
_Q0_MAX_DD     = 0.30
_Q0_MIN_FOLDS  = 4
_BOOTSTRAP_N   = 1000


@dataclass
class LeafProfile:
    leaf_id: int
    rule_path_summary: str
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd: float
    cumulative_profit_raw: float
    cumulative_profit_deflated: float
    deflation_pp: float
    n_combos_searched: int
    inner_cv_se: float
    sortino: float
    n_folds_passed_q0: int
    bootstrap_triggered: bool
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    chosen_depth: int = 0
    chosen_min_leaf: int = 0


def apply_deflation(raw: float, inner_cv_se: float, n_combos: int) -> Tuple[float, float]:
    """Apply max-of-N inflation correction to a search-best profit estimate.

    Returns (deflated_profit, inflation). inflation = σ × √(2 × ln N).
    """
    inflation = float(inner_cv_se) * math.sqrt(2.0 * math.log(max(int(n_combos), 1)))
    return raw - inflation, inflation


def long_shot_band(avg_win: float, avg_loss: float, win_rate: float) -> bool:
    """Per spec: avg_win >= 15% AND |avg_loss| <= 7% AND win_rate >= 70%."""
    return avg_win >= 0.15 and abs(avg_loss) <= 0.07 and win_rate >= 0.70


def leaf_metrics(trades_net: np.ndarray) -> dict:
    """Compute trade-list metrics from a leaf's net-PnL trade sequence."""
    n = int(trades_net.shape[0])
    if n == 0:
        return {
            "trade_count": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_dd": 0.0, "cumulative_profit_raw": 0.0, "sortino": 0.0,
        }
    wins = trades_net[trades_net > 0]
    losses = trades_net[trades_net < 0]
    avg_win  = float(wins.mean())  if wins.size  > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    cum = float(trades_net.sum())
    equity = np.concatenate([[0.0], np.cumsum(trades_net)])
    running_max = np.maximum.accumulate(equity)
    drawdown    = running_max - equity
    max_dd = float(drawdown.max())
    mean_trade = float(trades_net.mean())
    if losses.size > 0:
        downside_dev = float(np.sqrt(np.mean(losses ** 2)))
    else:
        downside_dev = 0.0
    sortino = mean_trade / downside_dev if downside_dev > 0 else 0.0
    return {
        "trade_count": n,
        "win_rate": float((trades_net > 0).mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "cumulative_profit_raw": cum,
        "sortino": sortino,
    }


def leaf_qualifies(fold_metric: dict) -> bool:
    """True if the leaf passes all three Q0 hard gates on this fold."""
    if fold_metric["avg_win"] < _Q0_AVG_WIN:
        return False
    if fold_metric["avg_loss"] < _Q0_AVG_LOSS:
        return False
    if fold_metric["max_dd"] > _Q0_MAX_DD:
        return False
    return True


def pick_best_hyperparams(inner_scores: dict) -> Tuple[int, int, float, float]:
    """Pick argmax of inner-CV mean-profit table.

    Returns (chosen_depth, chosen_min_leaf, raw_max_profit, inner_cv_se).
    inner_cv_se = std (ddof=1) of inner mean profits — drives deflation.
    """
    best_combo, raw_max = max(inner_scores.items(), key=lambda kv: kv[1])
    chosen_depth, chosen_min_leaf = best_combo
    values = np.array(list(inner_scores.values()), dtype="float64")
    inner_cv_se = float(values.std(ddof=1))
    return int(chosen_depth), int(chosen_min_leaf), float(raw_max), inner_cv_se


def bootstrap_ci(
    trades_net: np.ndarray,
    n_iter: int = _BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Returns (95% lower, 95% upper) on cumulative profit via bootstrap resampling."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(trades_net)
    if n == 0:
        return 0.0, 0.0
    samples = rng.choice(trades_net, size=(int(n_iter), n), replace=True)
    cum = samples.sum(axis=1)
    return float(np.percentile(cum, 2.5)), float(np.percentile(cum, 97.5))
