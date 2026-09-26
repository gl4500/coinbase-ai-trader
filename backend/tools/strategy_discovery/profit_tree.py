"""Recursive tree fitter for Phase 3 mining.

Builds a binary tree by repeatedly calling profit_split.best_split on the
current node's row subset until depth/leaf-size constraints stop the recursion.

Pure functions on torch.Tensor + Python dataclasses. No I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from tools.strategy_discovery.profit_split import best_split


@dataclass
class TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    indices: Optional[torch.Tensor] = None  # populated on leaves
    cumulative_pnl: float = 0.0  # populated on leaves

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


def fit_tree(
    features: torch.Tensor,
    labels: torch.Tensor,
    next_eligible: torch.Tensor,
    max_depth: int,
    min_leaf: int,
    n_thresholds: int = 256,
) -> TreeNode:
    """Build a profit-maximizing decision tree on the full pid row set."""
    N = features.shape[0]
    all_indices = torch.arange(N, dtype=torch.int64, device=features.device)
    return _fit_recursive(
        features,
        all_indices,
        labels,
        next_eligible,
        depth=0,
        max_depth=max_depth,
        min_leaf=min_leaf,
        n_thresholds=n_thresholds,
    )


def _fit_recursive(
    features: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    next_eligible: torch.Tensor,
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    n_thresholds: int,
) -> TreeNode:
    node_rows = features.index_select(0, indices)
    n = indices.shape[0]
    if depth >= max_depth or n < 2 * min_leaf:
        return _leaf(indices, labels, next_eligible)
    split = best_split(node_rows, indices, labels, next_eligible, n_thresholds=n_thresholds)
    if split is None:
        return _leaf(indices, labels, next_eligible)
    left_idx = indices[split.left_mask]
    right_idx = indices[~split.left_mask]
    if int(left_idx.shape[0]) < min_leaf or int(right_idx.shape[0]) < min_leaf:
        return _leaf(indices, labels, next_eligible)
    return TreeNode(
        feature=split.feature,
        threshold=split.threshold,
        left=_fit_recursive(
            features,
            left_idx,
            labels,
            next_eligible,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            n_thresholds=n_thresholds,
        ),
        right=_fit_recursive(
            features,
            right_idx,
            labels,
            next_eligible,
            depth=depth + 1,
            max_depth=max_depth,
            min_leaf=min_leaf,
            n_thresholds=n_thresholds,
        ),
    )


def _leaf(indices: torch.Tensor, labels: torch.Tensor, next_eligible: torch.Tensor) -> TreeNode:
    from tools.strategy_discovery.profit_split import walk_and_sum

    sorted_idx, _ = torch.sort(indices)
    pad = sorted_idx.unsqueeze(0)
    cum = float(walk_and_sum(pad, next_eligible, labels)[0].item())
    return TreeNode(indices=sorted_idx, cumulative_pnl=cum)


def collect_leaves(root: TreeNode) -> List[TreeNode]:
    out: List[TreeNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            out.append(n)
        else:
            stack.append(n.right)
            stack.append(n.left)
    return out
