"""Tests for tools.strategy_discovery.profit_tree (Phase 3)."""

from __future__ import annotations

import numpy as np
import torch

from tools.strategy_discovery.profit_split import build_next_eligible
from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves, fit_tree


def _max_depth(node: TreeNode, depth: int = 0) -> int:
    if node.is_leaf:
        return depth
    return max(_max_depth(node.left, depth + 1), _max_depth(node.right, depth + 1))


def _synthetic_data(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    ts = torch.arange(n, dtype=torch.int64) * 3_600_000
    features = torch.from_numpy(rng.normal(0.0, 1.0, size=(n, 4)).astype("float64"))
    labels = torch.from_numpy(rng.normal(0.0, 0.05, size=n).astype("float64"))
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    return features, labels, next_eligible


def test_fit_respects_max_depth():
    features, labels, next_eligible = _synthetic_data()
    root = fit_tree(features, labels, next_eligible, max_depth=2, min_leaf=10)
    assert _max_depth(root) <= 2


def test_fit_respects_min_samples_per_leaf():
    features, labels, next_eligible = _synthetic_data(n=200)
    root = fit_tree(features, labels, next_eligible, max_depth=10, min_leaf=40)
    leaves = collect_leaves(root)
    for leaf in leaves:
        assert leaf.indices is not None
        assert int(leaf.indices.shape[0]) >= 40, f"leaf with {leaf.indices.shape[0]} rows"


def test_fit_stops_on_unprofitable_split():
    n = 200
    ts = torch.arange(n, dtype=torch.int64) * 3_600_000
    features = torch.from_numpy(
        np.random.default_rng(1).normal(0.0, 1.0, size=(n, 4)).astype("float64")
    )
    labels = torch.full((n,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    root = fit_tree(features, labels, next_eligible, max_depth=5, min_leaf=20)
    assert root.is_leaf
    assert root.indices is not None
    assert int(root.indices.shape[0]) == n
    assert root.cumulative_pnl < 0


def test_collect_leaves_returns_all_leaves_in_dfs_order():
    leaf_a = TreeNode(indices=torch.tensor([0, 1, 2], dtype=torch.int64), cumulative_pnl=1.0)
    leaf_b = TreeNode(indices=torch.tensor([3, 4], dtype=torch.int64), cumulative_pnl=2.0)
    leaf_c = TreeNode(indices=torch.tensor([5, 6, 7], dtype=torch.int64), cumulative_pnl=3.0)
    right_subtree = TreeNode(feature=0, threshold=0.5, left=leaf_b, right=leaf_c)
    root = TreeNode(feature=1, threshold=0.0, left=leaf_a, right=right_subtree)
    leaves = collect_leaves(root)
    assert [leaf.cumulative_pnl for leaf in leaves] == [1.0, 2.0, 3.0]
