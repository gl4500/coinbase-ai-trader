"""Tests for tools.strategy_discovery.knapsack_search (Phase 4)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.knapsack_search import (
    KnapsackResult,
    beam_search_knapsack,
)
from tools.strategy_discovery.profile_loader import LoadedProfile


def _make_profile(pid: str, leaf_id: int, horizon: int, deflated: float) -> LoadedProfile:
    return LoadedProfile(
        pid=pid, horizon=horizon, leaf_id=leaf_id,
        rule_path="price_over_ema20 > 1.0",
        cumulative_profit_raw=deflated + 0.02,
        cumulative_profit_deflated=deflated,
        deflation_pp=0.02,
        win_rate=0.6, avg_win=0.08, avg_loss=-0.04,
        max_dd=0.20, sortino=1.2, trade_count=30,
        n_folds_passed_q0=5, chosen_depth=5, chosen_min_leaf=50,
    )


def _make_pid_features(pid: str, n: int):
    return pd.DataFrame({
        "ts": (np.arange(n, dtype="int64") * 3_600_000).tolist(),
        "close": [1.0] * n,
        "price_over_ema20": [1.5] * n,
        "vol_over_mc": [0.01] * n,
        "label_h1": [0.10] * n,
    })


def test_returns_k_evaluated_for_deflation():
    profiles = [_make_profile(f"P{i}-USD", 0, 1, deflated=0.05 + i * 0.01) for i in range(5)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=3, pool_size=5, bootstrap_iter=100, seed=42,
    )
    # K_evaluated > 0 and matches: step 1 (1 × 5) + step 2 (3 × 4) = 5 + 12 = 17
    assert result.k_evaluated > 0
    assert result.k_evaluated == 1 * 5 + 3 * 4
    # inflation > 0 because σ × √(2 ln 17) > 0 for any non-zero σ
    assert result.inflation >= 0
    # best subset size == cap
    assert len(result.best_subset) == 2


def test_beam_search_finds_known_optimal_on_toy_3_profile_pool():
    # 3 profiles with clearly ordered deflated profits. Best cap=2 subset = top-2.
    profiles = [_make_profile(f"P{i}-USD", 0, 1, deflated=0.05 + i * 0.05) for i in range(3)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=10, pool_size=3, bootstrap_iter=100, seed=42,
    )
    # The 2 best by deflated profit are P2 (0.15) and P1 (0.10)
    chosen_pids = sorted(p.pid for p in result.best_subset)
    assert chosen_pids == ["P1-USD", "P2-USD"]


def test_beam_width_caps_branching():
    # 5 profiles, beam_width=2, cap=2.
    # Step 1: 1 × 5 = 5 candidates → beam keeps 2.
    # Step 2: 2 × 4 = 8 candidates → beam keeps 2.
    # Total k_evaluated = 5 + 8 = 13.
    profiles = [_make_profile(f"Q{i}-USD", 0, 1, deflated=0.05 + i * 0.01) for i in range(5)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=2, pool_size=5, bootstrap_iter=100, seed=42,
    )
    assert result.k_evaluated == 5 + 8
    # Beam history captures 2 steps
    assert len(result.beam_history) == 2
    # Beam at each step ≤ 2
    for step in result.beam_history:
        assert len(step) <= 2
