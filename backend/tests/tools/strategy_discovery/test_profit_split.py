"""Tests for tools.strategy_discovery.profit_split (Phase 3)."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from tools.strategy_discovery.profit_split import (
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
    open_until = -1  # exclusive
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
    ts = torch.arange(5, dtype=torch.int64) * 3_600_000
    labels = torch.tensor([1.0, 10.0, 100.0, 2.0, 50.0], dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=3)
    assert next_eligible.tolist() == [3, 4, 5, 5, 5]
    subset = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    total = walk_and_sum(subset, next_eligible, labels)
    assert total.item() == pytest.approx(3.0, abs=1e-12)


def test_split_metric_picks_higher_pnl_subgroup():
    N = 100
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    horizon_bars = 1
    features = torch.zeros((N, 1), dtype=torch.float64)
    features[50:, 0] = 1.0
    labels = torch.zeros(N, dtype=torch.float64)
    labels[:50] = -0.02
    labels[50:] = 0.10
    next_eligible = build_next_eligible(ts, horizon_bars=horizon_bars)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is not None
    assert result.feature == 0
    assert 0.0 < result.threshold < 1.0
    assert result.score == pytest.approx(5.0, abs=1e-9)


def test_no_profitable_split_returns_none():
    N = 30
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    features = torch.linspace(0.0, 1.0, N, dtype=torch.float64).unsqueeze(1)
    labels = torch.full((N,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is None
