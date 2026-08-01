"""GPU-vectorized custom split criterion for Phase 3 mining.

For a candidate split `(feature_j, threshold_t)` on a row subset, computes the
concurrency-capped cumulative PnL of each side (max 1 open position per token).
The split metric is `max(cum_pnl_left, cum_pnl_right)`.

Pure functions on torch.Tensor. No I/O, no tree state, no filesystem.
Mirrors backend/tools/xgb_v4_5_features_batch.py conventions for device handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

_MS_PER_BAR: int = 3_600_000  # one 1h bar in epoch milliseconds


@dataclass
class SplitResult:
    """Best-split outcome from one node's candidate scan.

    Treat as immutable — not frozen because the `left_mask` torch.Tensor is
    unhashable, so `frozen=True` would create a misleading hashability contract.
    """

    feature: int
    threshold: float
    left_mask: torch.Tensor  # (n,) bool — True for rows going to left subtree
    score: float  # the split_metric value (cum_pnl of better side)


def build_next_eligible(ts_ms: torch.Tensor, horizon_bars: int) -> torch.Tensor:
    """Returns `next_eligible[i] = min { j : ts[j] >= ts[i] + horizon_bars * 1h }`.

    Returns `len(ts)` for any row whose horizon would extend past the end.
    """
    n = ts_ms.shape[0]
    horizon_ms = int(horizon_bars) * _MS_PER_BAR
    target = ts_ms + horizon_ms
    out = torch.searchsorted(ts_ms, target, right=False)
    return out.clamp_max(n)


def walk_and_sum(
    subset_indices: torch.Tensor,
    next_eligible: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Vectorized concurrency-capped (max-1) cumulative PnL across B candidate subsets.

    For each row in B, scan its K rows in order; if a row's index is below the
    current open_until, skip; else add its label and advance open_until to its
    next_eligible. Padding (-1) is treated as "no row".
    """
    B, K = subset_indices.shape
    labels.shape[0]
    valid = subset_indices >= 0
    safe_idx = subset_indices.clamp_min(0)
    row_labels = labels.gather(0, safe_idx.reshape(-1)).reshape(B, K)
    row_next = next_eligible.gather(0, safe_idx.reshape(-1)).reshape(B, K)
    open_until = torch.full((B,), -1, dtype=torch.int64, device=subset_indices.device)
    total = torch.zeros((B,), dtype=labels.dtype, device=subset_indices.device)
    for k in range(K):
        col_idx = safe_idx[:, k]
        col_lab = row_labels[:, k]
        col_next = row_next[:, k]
        col_valid = valid[:, k]
        fire = col_valid & (col_idx >= open_until)
        total = total + torch.where(fire, col_lab, torch.zeros_like(col_lab))
        open_until = torch.where(fire, col_next, open_until)
    return total


def _quantile_thresholds(values: torch.Tensor, n: int) -> torch.Tensor:
    """Returns up to n candidate split thresholds as midpoints of adjacent unique values.

    Standard CART approach: thresholds are strictly between consecutive distinct
    feature values, so the left/right split is always non-trivial.
    """
    if values.numel() == 0:
        return values.new_empty((0,))
    unique_vals = torch.unique(values)
    if unique_vals.numel() < 2:
        return values.new_empty((0,))
    midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
    if midpoints.numel() <= n:
        return midpoints
    # Sub-sample evenly when more midpoints than budget
    indices = torch.linspace(0, midpoints.numel() - 1, n, dtype=torch.float64, device=values.device)
    indices = indices.long()
    return midpoints[indices]


def best_split(
    features: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    next_eligible: torch.Tensor,
    n_thresholds: int = 256,
) -> Optional[SplitResult]:
    """Scan all (feature, threshold) candidates; return the best SplitResult or None.

    `None` means no candidate produced strictly positive cum_pnl in either side.
    """
    n, F = features.shape
    if n < 2:
        return None
    best: Optional[SplitResult] = None
    for f in range(F):
        col = features[:, f]
        thresholds = _quantile_thresholds(col, n_thresholds)
        if thresholds.numel() == 0:
            continue
        left_mask = col.unsqueeze(0) <= thresholds.unsqueeze(1)
        T = thresholds.numel()
        for side_is_left in (True, False):
            mask = left_mask if side_is_left else ~left_mask
            counts = mask.sum(dim=1)
            if counts.max().item() == 0:
                continue
            max_k = int(counts.max().item())
            subset_idx = torch.full((T, max_k), -1, dtype=torch.int64, device=features.device)
            row_pos = mask.cumsum(dim=1) - 1
            abs_idx_broadcast = indices.unsqueeze(0).expand(T, n)
            r_t, r_n = torch.where(mask)
            subset_idx[r_t, row_pos[r_t, r_n]] = abs_idx_broadcast[r_t, r_n]
            scores = walk_and_sum(subset_idx, next_eligible, labels)
            best_t = int(scores.argmax().item())
            best_score = float(scores[best_t].item())
            if best_score > 0.0 and (best is None or best_score > best.score):
                best = SplitResult(
                    feature=f,
                    threshold=float(thresholds[best_t].item()),
                    left_mask=(col <= thresholds[best_t]),
                    score=best_score,
                )
    return best
