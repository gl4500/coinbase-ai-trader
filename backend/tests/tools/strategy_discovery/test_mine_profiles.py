"""Tests for tools.strategy_discovery.mine_profiles (Phase 3)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.strategy_discovery.mine_profiles import (
    _assign_leaves,
    apply_deflation,
    bootstrap_ci,
    leaf_metrics,
    leaf_qualifies,
    long_shot_band,
    pick_best_hyperparams,
)
from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves


def test_deflation_factor_applied_to_reported_profit():
    deflated, infl = apply_deflation(raw=0.072, inner_cv_se=0.015, n_combos=9)
    assert infl == pytest.approx(0.015 * math.sqrt(2 * math.log(9)), rel=1e-9)
    assert deflated == pytest.approx(0.072 - infl, rel=1e-9)


def test_long_shot_band_triggers_bootstrap():
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.75) is True
    assert long_shot_band(avg_win=0.14, avg_loss=-0.05, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.08, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.69) is False


def test_leaf_metrics_computes_win_loss_dd_sortino():
    trades = np.array([0.08, 0.08, -0.05, 0.08, 0.08, -0.05])
    m = leaf_metrics(trades)
    assert m["trade_count"] == 6
    assert m["win_rate"] == pytest.approx(4 / 6)
    assert m["avg_win"]  == pytest.approx(0.08)
    assert m["avg_loss"] == pytest.approx(-0.05)
    assert m["cumulative_profit_raw"] == pytest.approx(0.22, abs=1e-9)
    assert m["max_dd"] == pytest.approx(0.05, abs=1e-9)
    assert m["sortino"] == pytest.approx(0.22 / 6 / 0.05, rel=1e-6)


def test_q0_gates_applied_to_deflated_profit():
    fold_metrics = [
        {"avg_win": 0.08, "avg_loss": -0.08, "max_dd": 0.20, "deflated_profit": 0.05},  # PASS
        {"avg_win": 0.06, "avg_loss": -0.09, "max_dd": 0.25, "deflated_profit": 0.03},  # PASS
        {"avg_win": 0.07, "avg_loss": -0.11, "max_dd": 0.22, "deflated_profit": 0.04},  # FAIL (avg_loss < -0.10)
        {"avg_win": 0.09, "avg_loss": -0.07, "max_dd": 0.18, "deflated_profit": 0.06},  # PASS
        {"avg_win": 0.10, "avg_loss": -0.06, "max_dd": 0.15, "deflated_profit": 0.07},  # PASS
    ]
    n_pass = sum(leaf_qualifies(m) for m in fold_metrics)
    assert n_pass == 4
    from tools.strategy_discovery.mine_profiles import _Q0_MIN_FOLDS
    assert n_pass >= _Q0_MIN_FOLDS


def test_pick_best_hyperparams_picks_max_inner_cv():
    inner_scores = {
        (3, 20):  0.012, (3, 50):  0.018, (3, 100): 0.005,
        (5, 20):  0.022, (5, 50):  0.019, (5, 100): 0.011,
        (7, 20):  0.025, (7, 50):  0.020, (7, 100): 0.015,
    }
    chosen_depth, chosen_min_leaf, raw_max, inner_se = pick_best_hyperparams(inner_scores)
    assert chosen_depth == 7
    assert chosen_min_leaf == 20
    assert raw_max == pytest.approx(0.025, rel=1e-9)
    expected_se = float(np.std(list(inner_scores.values()), ddof=1))
    assert inner_se == pytest.approx(expected_se, rel=1e-9)


def test_bootstrap_ci_returns_95pct_band_on_resampled_trades():
    trades = np.full(50, 0.01)
    rng = np.random.default_rng(7)
    lower, upper = bootstrap_ci(trades, n_iter=500, rng=rng)
    assert lower == pytest.approx(0.50, abs=1e-6)
    assert upper == pytest.approx(0.50, abs=1e-6)
    mixed = np.concatenate([np.full(30, 0.10), np.full(20, -0.05)])
    lo2, hi2 = bootstrap_ci(mixed, n_iter=500, rng=rng)
    assert lo2 < 2.0 < hi2
    assert (hi2 - lo2) > 0.1


def test_mine_profiles_for_pid_horizon_returns_qualifying_leaves_only(tmp_path):
    """Integration: build a synthetic Phase 2 parquet where a clear cohort exists,
    run mine_profiles_for_pid_horizon end-to-end (on CPU), and assert at least
    one qualifying profile comes back."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pandas as pd
    from tools.strategy_discovery.mine_profiles import mine_profiles_for_pid_horizon

    rng = np.random.default_rng(101)
    n = 1000
    ts_ms = np.arange(n, dtype="int64") * 3_600_000
    feat_0 = rng.uniform(0.0, 1.0, size=n)
    feat_1 = rng.uniform(0.0, 1.0, size=n)
    labels = np.where(feat_0 > 0.5, 0.08, -0.02).astype("float64")
    df = pd.DataFrame({
        "ts":               ts_ms,
        "open":             np.full(n, 1.0),
        "high":             np.full(n, 1.0),
        "low":              np.full(n, 1.0),
        "close":            np.full(n, 1.0),
        "market_cap":       np.full(n, 1e9),
        "fdv":              np.full(n, 2e9),
        "fdv_over_mc":      np.full(n, 2.0),
        "circ_over_total":  np.full(n, 0.5),
        "vol_24h":          np.full(n, 1e7),
        "vol_over_mc":      np.full(n, 0.01),
        "price_over_ema20": feat_0,
        "price_over_ema50": np.full(n, 1.0),
        "price_over_ema200":np.full(n, 1.0),
        "ret_1h_sign":      np.full(n, 0.0),
        "ret_24h_sign":     feat_1,
        "ret_7d_sign":      np.full(n, 0.0),
        "atr14_pct":        np.full(n, 0.02),
        "label_h1":   labels,
        "label_h4":   labels,
        "label_h24":  labels,
        "label_h72":  labels,
        "label_h168": labels,
    })
    parquet_path = tmp_path / "FOO-USD.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    profiles = mine_profiles_for_pid_horizon(
        pid="FOO-USD", horizon=24, parquet_path=parquet_path, device="cpu", seed=42,
    )
    assert len(profiles) >= 1
    winners = [p for p in profiles if p.avg_win >= 0.05 and p.cumulative_profit_deflated > 0]
    assert len(winners) >= 1, f"no winners; got profiles: {[(p.avg_win, p.cumulative_profit_deflated) for p in profiles]}"


def _make_test_tree():
    """Tree: x[0] <= 0.5 ? (x[1] <= 0.2 ? L0 : L1) : L2  (DFS leaf order = L0, L1, L2)."""
    leaf0 = TreeNode(indices=torch.tensor([], dtype=torch.int64), cumulative_pnl=0.0)
    leaf1 = TreeNode(indices=torch.tensor([], dtype=torch.int64), cumulative_pnl=0.0)
    leaf2 = TreeNode(indices=torch.tensor([], dtype=torch.int64), cumulative_pnl=0.0)
    left  = TreeNode(feature=1, threshold=0.2, left=leaf0, right=leaf1)
    root  = TreeNode(feature=0, threshold=0.5, left=left,  right=leaf2)
    return root, [leaf0, leaf1, leaf2]


def test_assign_leaves_matches_scalar_on_random_rows():
    root, leaves = _make_test_tree()
    g = torch.Generator().manual_seed(0)
    rows = torch.randn(50, 2, generator=g, dtype=torch.float64)
    out = _assign_leaves(root, rows)
    assert len(out) == 50
    for i, r in enumerate(rows):
        node = root
        while not node.is_leaf:
            v = float(r[node.feature].item())
            node = node.left if v <= node.threshold else node.right
        expected = next(idx for idx, lf in enumerate(leaves) if lf is node)
        assert out[i] == expected


def test_assign_leaves_handles_single_leaf_tree():
    leaf = TreeNode(indices=torch.tensor([], dtype=torch.int64), cumulative_pnl=0.0)
    rows = torch.zeros(7, 3, dtype=torch.float64)
    assert _assign_leaves(leaf, rows) == [0] * 7


def test_assign_leaves_handles_empty_features():
    root, _ = _make_test_tree()
    rows = torch.zeros(0, 2, dtype=torch.float64)
    assert _assign_leaves(root, rows) == []
