"""Beam-search knapsack over profile subsets for Phase 4.

For each candidate subset of size `cap`, calls portfolio_sim and keeps the
top-BEAM_WIDTH subsets at each step. Applies σ × √(2 ln K) deflation to
the winner's cumulative profit using K = total portfolio_sim calls.

Pure Python orchestration; depends only on portfolio_sim + profile_loader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.strategy_discovery.portfolio_sim import (
    PortfolioMetrics,
    TelemetryRow,
    simulate_portfolio,
)
from tools.strategy_discovery.profile_loader import LoadedProfile

_DEFAULT_BEAM_WIDTH = 20
_DEFAULT_POOL_SIZE = 100
_DEFAULT_BOOTSTRAP_N = 1000


@dataclass
class KnapsackResult:
    best_subset: List[LoadedProfile]
    best_metrics: PortfolioMetrics
    best_telemetry: List[TelemetryRow]
    k_evaluated: int
    inflation: float
    beam_history: List[List[float]] = field(default_factory=list)


def _bootstrap_portfolio_std(
    trade_pnls: List[float],
    n_iter: int = _DEFAULT_BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Bootstrap SE of cumulative profit. Resample the trade list with replacement."""
    if rng is None:
        rng = np.random.default_rng()
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype="float64")
    n = arr.shape[0]
    samples = rng.choice(arr, size=(int(n_iter), n), replace=True)
    cums = samples.sum(axis=1)
    return float(cums.std(ddof=1))


def beam_search_knapsack(
    all_qualifying: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
    beam_width: int = _DEFAULT_BEAM_WIDTH,
    pool_size: int = _DEFAULT_POOL_SIZE,
    bootstrap_iter: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
) -> KnapsackResult:
    """Beam search over subsets of size `cap`; return best by deflated profit."""
    # Cap the candidate pool
    pool = sorted(all_qualifying, key=lambda p: -p.cumulative_profit_deflated)[: int(pool_size)]
    if not pool:
        return KnapsackResult(
            best_subset=[],
            best_metrics=PortfolioMetrics(),
            best_telemetry=[],
            k_evaluated=0,
            inflation=0.0,
            beam_history=[],
        )

    # Beam = list of (subset, metrics, telemetry, trade_pnls)
    beam: List[Tuple[List[LoadedProfile], PortfolioMetrics, List[TelemetryRow], List[float]]] = [
        ([], PortfolioMetrics(), [], [])
    ]
    k_evaluated = 0
    beam_history: List[List[float]] = []

    for _step in range(int(cap)):
        candidates: List[
            Tuple[List[LoadedProfile], PortfolioMetrics, List[TelemetryRow], List[float]]
        ] = []
        for subset, _m, _t, _p in beam:
            occupied_pids = {p.pid for p in subset}
            for cand in pool:
                if cand.pid in occupied_pids or cand in subset:
                    continue
                new_subset = subset + [cand]
                metrics, telemetry = simulate_portfolio(
                    new_subset, cap=int(cap), pid_features=pid_features
                )
                trade_pnls = [t.realized_pnl for t in telemetry if t.realized_pnl is not None]
                k_evaluated += 1
                candidates.append((new_subset, metrics, telemetry, trade_pnls))
        if not candidates:
            break
        candidates.sort(key=lambda c: -c[1].cumulative_profit_raw)
        beam = candidates[: int(beam_width)]
        beam_history.append([c[1].cumulative_profit_raw for c in beam])

    # Pick the best from final beam
    best_subset, best_metrics, best_telemetry, best_trades = (
        beam[0] if beam else ([], PortfolioMetrics(), [], [])
    )
    rng = np.random.default_rng(seed)
    sigma = _bootstrap_portfolio_std(best_trades, n_iter=bootstrap_iter, rng=rng)
    inflation = sigma * math.sqrt(2.0 * math.log(max(k_evaluated, 1)))
    best_metrics.cumulative_profit_deflated = best_metrics.cumulative_profit_raw - inflation
    return KnapsackResult(
        best_subset=best_subset,
        best_metrics=best_metrics,
        best_telemetry=best_telemetry,
        k_evaluated=k_evaluated,
        inflation=inflation,
        beam_history=beam_history,
    )
