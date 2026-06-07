"""Tests for tools.strategy_discovery.profit_split (Phase 3)."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from tools.strategy_discovery.profit_split import (
    SplitResult,
    best_split,
    build_next_eligible,
    walk_and_sum,
)


def _naive_walk_and_sum_py(
    indices: List[int],
    next_eligible: List[int],
    labels: List[float],
) -> float:
    """Reference: walk indices in order, only enter if not already in a trade."""
    open_until = -1   # exclusive
    total = 0.0
    for i in indices:
        if i < open_until:
            continue
        total += labels[i]
        open_until = next_eligible[i]
    return total


def test_walk_and_sum_matches_naive_python_reference():
    rng = np.random.default_rng(13)
    N = 500
    labels = rng.normal(0.0, 0.05, size=N).astype("float64")
    horizon_bars = 24
    next_eligible = np.minimum(np.arange(N) + horizon_bars, N).astype("int64")
    B = 7
    subsets = []
    for _ in range(B):
        size = rng.integers(50, 200)
        chosen = sorted(rng.choice(N, size=size, replace=False).tolist())
        subsets.append(chosen)
    max_k = max(len(s) for s in subsets)
    subset_idx = torch.full((B, max_k), -1, dtype=torch.int64)
    for b, s in enumerate(subsets):
        subset_idx[b, : len(s)] = torch.tensor(s, dtype=torch.int64)
    out = walk_and_sum(
        subset_idx,
        torch.from_numpy(next_eligible),
        torch.from_numpy(labels),
    )
    expected = [_naive_walk_and_sum_py(s, next_eligible.tolist(), labels.tolist()) for s in subsets]
    np.testing.assert_allclose(out.cpu().numpy(), np.array(expected), rtol=1e-9, atol=1e-12)


def test_concurrency_max_1_skips_overlapping_entry():
    labels = torch.tensor([1.0, 10.0, 100.0, 2.0, 50.0], dtype=torch.float64)
    next_eligible = build_next_eligible(5, horizon_bars=3)
    assert next_eligible.tolist() == [3, 4, 5, 5, 5]
    subset = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    total = walk_and_sum(subset, next_eligible, labels)
    assert total.item() == pytest.approx(3.0, abs=1e-12)


def test_split_metric_picks_higher_pnl_subgroup():
    N = 100
    horizon_bars = 1
    features = torch.zeros((N, 1), dtype=torch.float64)
    features[50:, 0] = 1.0
    labels = torch.zeros(N, dtype=torch.float64)
    labels[:50] = -0.02
    labels[50:] = 0.10
    next_eligible = build_next_eligible(N, horizon_bars=horizon_bars)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is not None
    assert result.feature == 0
    assert 0.0 < result.threshold < 1.0
    assert result.score == pytest.approx(5.0, abs=1e-9)


def test_no_profitable_split_returns_none():
    N = 30
    features = torch.linspace(0.0, 1.0, N, dtype=torch.float64).unsqueeze(1)
    labels = torch.full((N,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(N, horizon_bars=1)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is None


def _synthetic_inputs(n=400, f=12, h=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(n, f, generator=g, dtype=torch.float64)
    labels = torch.randn(n, generator=g, dtype=torch.float64) * 0.05
    indices = torch.arange(n, dtype=torch.int64)
    next_eligible = build_next_eligible(n, horizon_bars=h)
    return features, indices, labels, next_eligible


def test_batched_best_split_score_matches_reference_implementation():
    """Score from best_split matches a brute-force walk_and_sum on the chosen subset."""
    features, indices, labels, next_eligible = _synthetic_inputs(seed=1)
    out = best_split(features, indices, labels, next_eligible, n_thresholds=64)
    assert out is not None
    chosen_idx = indices[out.left_mask].unsqueeze(0)
    ref_score = float(walk_and_sum(chosen_idx, next_eligible, labels)[0].item())
    assert abs(ref_score - out.score) < 1e-9


def test_batched_best_split_handles_right_side_winner():
    """A feature where positive labels sit on col>threshold side yields a valid split."""
    n, f = 200, 3
    features = torch.zeros(n, f, dtype=torch.float64)
    features[:, 0] = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    labels = (features[:, 0] > 0.3).double() * 0.1
    indices = torch.arange(n, dtype=torch.int64)
    next_eligible = build_next_eligible(n, horizon_bars=2)
    out = best_split(features, indices, labels, next_eligible, n_thresholds=32)
    assert out is not None
    assert out.feature == 0
    assert 0 < int(out.left_mask.sum().item()) < n


def test_batched_best_split_returns_none_for_zero_labels():
    """All-zero labels produce no positive-score split → returns None."""
    features, indices, _labels, next_eligible = _synthetic_inputs(seed=2)
    labels = torch.zeros_like(_labels)
    out = best_split(features, indices, labels, next_eligible, n_thresholds=64)
    assert out is None


def test_batched_best_split_handles_few_unique_feature_values():
    """A feature with only 2 unique values must not crash; output is None or SplitResult."""
    n = 200
    features = torch.zeros(n, 4, dtype=torch.float64)
    features[:, 0] = torch.where(torch.arange(n) < n // 2,
                                 torch.tensor(0.0), torch.tensor(1.0)).double()
    g = torch.Generator().manual_seed(3)
    features[:, 1:] = torch.randn(n, 3, generator=g, dtype=torch.float64)
    g2 = torch.Generator().manual_seed(4)
    labels = torch.randn(n, generator=g2, dtype=torch.float64) * 0.05
    indices = torch.arange(n, dtype=torch.int64)
    next_eligible = build_next_eligible(n, horizon_bars=2)
    out = best_split(features, indices, labels, next_eligible, n_thresholds=64)
    assert out is None or isinstance(out.score, float)
