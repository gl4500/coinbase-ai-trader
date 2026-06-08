"""GPU-vectorized custom split criterion for Phase 3 mining.

For a candidate split `(feature_j, threshold_t)` on a row subset, computes the
concurrency-capped cumulative PnL of each side (max 1 open position per token).
The split metric is `max(cum_pnl_left, cum_pnl_right)`.

Pure functions on torch.Tensor. No I/O, no tree state, no filesystem.
Mirrors backend/tools/xgb_v4_5_features_batch.py conventions for device handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

# _MS_PER_BAR removed — bars are no longer assumed to be fixed-width in time.


@dataclass
class SplitResult:
    """Best-split outcome from one node's candidate scan.

    Treat as immutable — not frozen because the `left_mask` torch.Tensor is
    unhashable, so `frozen=True` would create a misleading hashability contract.
    """
    feature: int
    threshold: float
    left_mask: torch.Tensor   # (n,) bool — True for rows going to left subtree
    score: float              # the split_metric value (cum_pnl of better side)


def build_next_eligible(n_rows: int, horizon_bars: int) -> torch.Tensor:
    """Returns `next_eligible[i] = min(i + horizon_bars, n_rows)` on bar indices.

    Bar-index concurrency: an entry at row i opens a position whose minimum
    next-eligible entry is row i + horizon_bars (clamped to n). No timestamp
    arithmetic, no fixed-width-bar assumption.

    Returns a CPU tensor; callers using CUDA-resident labels/features must
    move it onto the target device before passing to walk_and_sum / best_split.
    """
    n = int(n_rows)
    h = int(horizon_bars)
    return (torch.arange(n) + h).clamp_max(n)


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
    N = labels.shape[0]
    valid = subset_indices >= 0
    safe_idx = subset_indices.clamp_min(0)
    row_labels = labels.gather(0, safe_idx.reshape(-1)).reshape(B, K)
    row_next   = next_eligible.gather(0, safe_idx.reshape(-1)).reshape(B, K)
    open_until = torch.full((B,), -1, dtype=torch.int64, device=subset_indices.device)
    total      = torch.zeros((B,), dtype=labels.dtype, device=subset_indices.device)
    for k in range(K):
        col_idx   = safe_idx[:, k]
        col_lab   = row_labels[:, k]
        col_next  = row_next[:, k]
        col_valid = valid[:, k]
        fire = col_valid & (col_idx >= open_until)
        total      = total + torch.where(fire, col_lab, torch.zeros_like(col_lab))
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
    """Scan all (feature, threshold, side) candidates in one batched walk_and_sum.

    Same semantics as a per-(feature, side) loop: returns the SplitResult with
    the highest strictly-positive cum_pnl, or None if no candidate is positive.
    Differences vs the prior loop appear only on exact-score ties, where this
    implementation deterministically picks the lowest flat batch index.
    """
    n, F = features.shape
    if n < 2:
        return None
    dev = features.device
    T = int(n_thresholds)

    n_valid_per_f: List[int] = []
    thr_list: List[torch.Tensor] = []
    for f in range(F):
        col = features[:, f]
        thr = _quantile_thresholds(col, T)
        n_valid_per_f.append(int(thr.numel()))
        if thr.numel() < T:
            pad = torch.full((T - thr.numel(),), float("inf"),
                             device=dev, dtype=thr.dtype)
            thr = torch.cat([thr, pad])
        thr_list.append(thr)
    if not thr_list:
        return None
    thresholds_all = torch.stack(thr_list, dim=0)                          # (F, T)
    n_valid_t = torch.tensor(n_valid_per_f, device=dev)
    t_range = torch.arange(T, device=dev)
    valid_thr = t_range.unsqueeze(0) < n_valid_t.unsqueeze(1)              # (F, T)

    cols = features.t().unsqueeze(1)                                       # (F, 1, n)
    left_mask = cols <= thresholds_all.unsqueeze(-1)                       # (F, T, n)
    masks = torch.stack([left_mask, ~left_mask], dim=1)                    # (F, 2, T, n)
    F_, S, T_, N = masks.shape
    B = F_ * S * T_
    masks = masks.view(B, N)

    counts = masks.sum(dim=1)                                              # (B,)
    subset_idx = torch.full((B, N), -1, dtype=torch.int64, device=dev)
    row_pos = masks.cumsum(dim=1, dtype=torch.int64) - 1
    r_b, r_n = torch.where(masks)
    abs_idx_broadcast = indices.unsqueeze(0).expand(B, n)
    subset_idx[r_b, row_pos[r_b, r_n]] = abs_idx_broadcast[r_b, r_n]

    scores = walk_and_sum(subset_idx, next_eligible, labels)               # (B,)

    valid_flat = valid_thr.unsqueeze(1).expand(-1, S, -1).reshape(B)
    nonempty = counts > 0
    scores = scores.where(valid_flat & nonempty,
                          torch.tensor(-float("inf"), device=dev, dtype=scores.dtype))

    best_idx = int(scores.argmax().item())
    best_score = float(scores[best_idx].item())
    if not (best_score > 0.0):
        return None

    f = best_idx // (S * T_)
    t = best_idx % T_
    chosen_thr_val = float(thresholds_all[f, t].item())
    chosen_col = features[:, f]
    # left_mask is structural: always (col <= threshold). The winning side info
    # is carried by `score` (cum_pnl of the better side). Negating left_mask on
    # right-side winners would swap left/right children at every right-winning
    # split, changing tree structure and downstream rule_path_summary keys.
    return SplitResult(
        feature=int(f),
        threshold=chosen_thr_val,
        left_mask=(chosen_col <= chosen_thr_val),
        score=best_score,
    )
