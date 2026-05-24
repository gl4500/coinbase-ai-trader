"""XGB v4.5 batched-tensor feature extractor.

Vectorizes feature extraction across all sample bars per PID using PyTorch
tensor ops. Either CPU- or GPU-resident depending on `device` arg. Used by
train_xgb_v4_5.py when --device != cpu; existing per-bar extract_v4_5 in
xgb_v4_5_features.py remains canonical for inference + CPU default training.

Numerical contract: max abs diff < 1e-4 vs per-bar numpy reference (see
test_xgb_v4_5_features_batch.py).

Per feedback_python_clean_functions: type hints, pure data-in/data-out helpers,
derived constants, no in-place buffer mutation outside slot writes.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_5_features import (  # noqa: E402
    N_CHANNELS_V45, N_TIERS_V45, N_STATS_V45, N_FEATURES_V45,
    TIER_WINDOWS_V45, BB_PERIOD, BB_MULT,
)

_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")


__all__ = [
    "batch_build_samples_for_pid",
]


# ═══════════════════════════════════════════════════════════════════════════
# Private helpers (pure functions, one responsibility each)
# ═══════════════════════════════════════════════════════════════════════════

def _build_pid_ohlcv(
    candles: List[Dict[str, float]],
    device: str,
) -> torch.Tensor:
    """Stack OHLCV from candle list into (n_candles, 5) float64 tensor.

    Column order: open, high, low, close, volume (matches _OHLCV_FIELDS in
    xgb_v4_5_features.py).

    Empty input -> (0, 5) tensor on the requested device.
    """
    if not candles:
        return torch.zeros((0, 5), dtype=torch.float64, device=device)
    arr = np.empty((len(candles), 5), dtype=np.float64)
    for idx, c in enumerate(candles):
        arr[idx, 0] = c["open"]
        arr[idx, 1] = c["high"]
        arr[idx, 2] = c["low"]
        arr[idx, 3] = c["close"]
        arr[idx, 4] = c["volume"]
    return torch.tensor(arr, dtype=torch.float64, device=device)


def _local_bb_in_windows(
    closes_windows: torch.Tensor,
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-bar Bollinger position + width inside each tier window.

    For each sample bar's window-of-W closes, compute (bb_position, bb_width)
    at every bar j in [0, W). Bars j < period-1 get fallback (0.5, 0.0)
    because they don't have `period` prior bars WITHIN the local window.

    Matches xgb_v4_5_features._compute_bb_position + _compute_bb_width
    semantics (local-per-slice, NOT global). Returns float64 tensors on the
    same device as closes_windows.

    Shapes:
        closes_windows: (n_samples, W)
        return: (bb_position, bb_width), each (n_samples, W)
    """
    device = closes_windows.device
    dtype = closes_windows.dtype
    n_samples, W = closes_windows.shape

    bb_pos = torch.full((n_samples, W), 0.5, dtype=dtype, device=device)
    bb_wid = torch.zeros((n_samples, W), dtype=dtype, device=device)

    if W < period:
        return bb_pos, bb_wid

    # Inner strided windows: for each bar j in [period-1, W), use closes[j-period+1 .. j+1]
    inner = closes_windows.unfold(1, period, 1)        # (n_samples, W - period + 1, period)
    mean = inner.mean(dim=-1)                          # (n_samples, W - period + 1)
    std = inner.std(dim=-1, correction=0)              # ddof=0 to match numpy default

    upper = mean + mult * std
    lower = mean - mult * std
    bw = upper - lower

    # bb_position with clamp [0, 1], fallback 0.5 where bw <= 0
    inner_closes = closes_windows[:, period - 1 : W]   # (n_samples, W - period + 1)
    pos = (inner_closes - lower) / bw.clamp_min(1e-12)
    pos = pos.clamp(0.0, 1.0)
    valid_pos = bw > 0
    pos = torch.where(valid_pos, pos, torch.full_like(pos, 0.5))

    # bb_width = (2 * mult * std) / mean; fallback 0 where mean == 0
    wid = (2.0 * mult * std) / mean.clamp_min(1e-12)
    valid_wid = mean != 0
    wid = torch.where(valid_wid, wid, torch.zeros_like(wid))

    # Place into bb_pos/bb_wid at indices [period-1, W). Bars before that stay fallback.
    bb_pos[:, period - 1 :] = pos
    bb_wid[:, period - 1 :] = wid
    return bb_pos, bb_wid


def _batch_stats(
    windows: torch.Tensor,
) -> torch.Tensor:
    """Compute all 10 stats per (sample, channel) in one batched call.

    Stat order matches xgb_v4_5_features._STAT_NAMES_V45:
        0=last 1=mean 2=std 3=slope 4=min 5=max
        6=pct_rank 7=dlt5 8=dlt10 9=dlt30

    Shapes:
        windows: (n_samples, window_len, n_channels)
        return:  (n_samples, n_channels, 10)

    Empty window dim -> returns zeros (matches _compute_stats fallback).
    """
    n_samples, W, C = windows.shape
    device = windows.device
    dtype = windows.dtype

    if W == 0:
        return torch.zeros((n_samples, C, 10), dtype=dtype, device=device)

    # 0 last
    last = windows[:, -1, :]                                         # (n_samples, C)
    # 1 mean
    mean = windows.mean(dim=1)
    # 2 std (ddof=0 to match numpy .std() default)
    if W < 2:
        std = torch.zeros_like(mean)
    else:
        std = windows.std(dim=1, correction=0)
    # 3 slope (OLS over index 0..W-1)
    if W < 2:
        slope = torch.zeros_like(mean)
    else:
        x = torch.arange(W, dtype=dtype, device=device)
        x_mean = x.mean()
        x_dev = x - x_mean                                           # (W,)
        x_dev_sq = (x_dev * x_dev).sum()                             # scalar
        y_dev = windows - mean.unsqueeze(1)                          # (n_samples, W, C)
        num = (x_dev.unsqueeze(0).unsqueeze(-1) * y_dev).sum(dim=1)  # (n_samples, C)
        if float(x_dev_sq) == 0.0:
            slope = torch.zeros_like(mean)
        else:
            slope = num / x_dev_sq
    # 4 min, 5 max
    mn = windows.min(dim=1).values
    mx = windows.max(dim=1).values
    # 6 pct_rank — (sum(values < last) + 0.5 * sum(values == last)) / n
    if W < 2:
        pct_rank = torch.zeros_like(mean)
    else:
        last_b = last.unsqueeze(1)                                   # (n_samples, 1, C)
        below = (windows < last_b).sum(dim=1).to(dtype)              # (n_samples, C)
        equal = (windows == last_b).sum(dim=1).to(dtype)
        pct_rank = (below + 0.5 * equal) / float(W)

    # 7-9 dlt5/10/30 — values[-1] - values[-1-lookback]; zero if W too small
    def _dlt(lookback: int) -> torch.Tensor:
        if W < lookback + 1:
            return torch.zeros_like(mean)
        return windows[:, -1, :] - windows[:, -1 - lookback, :]
    dlt5 = _dlt(5)
    dlt10 = _dlt(10)
    dlt30 = _dlt(30)

    out = torch.stack(
        [last, mean, std, slope, mn, mx, pct_rank, dlt5, dlt10, dlt30],
        dim=-1,
    )
    return out


def _batch_triple_barrier(
    closes: torch.Tensor,
    sample_indices: torch.Tensor,
    forward_hours: int,
    label_thresh: float,
) -> torch.Tensor:
    """Vectorized 3-class triple-barrier labeling.

    For each sample bar i: examine closes[i+1 .. i+forward_hours]. Return:
        2 (UP)      if any forward close >= entry * (1 + label_thresh) hit first
        0 (DOWN)    if any forward close <= entry * (1 - label_thresh) hit first
        1 (NEUTRAL) if neither barrier hit within window
        -1          if i + forward_hours >= n_candles (right-edge, truncated)

    Tie-break: UP barrier is checked first within each bar (matches
    train_xgb_v4_5._triple_barrier_label_3class).

    Shapes:
        closes:         (n_candles,) float64
        sample_indices: (n_samples,) int64
        return:         (n_samples,) int8
    """
    n_candles = closes.shape[0]
    n_samples = sample_indices.shape[0]
    device = closes.device

    out = torch.full((n_samples,), 1, dtype=torch.int8, device=device)  # default NEUTRAL

    # Mark truncated samples (right edge): i + forward_hours >= n_candles
    truncated = (sample_indices + forward_hours) >= n_candles
    out[truncated] = -1

    entry = closes[sample_indices]                                  # (n_samples,)
    up_thr = entry * (1.0 + label_thresh)
    dn_thr = entry * (1.0 - label_thresh)

    offsets = torch.arange(1, forward_hours + 1, device=device, dtype=torch.int64)
    forward_indices = sample_indices.unsqueeze(1) + offsets.unsqueeze(0)       # (n_samples, fh)
    forward_indices_safe = forward_indices.clamp(max=n_candles - 1)
    forward_closes = closes[forward_indices_safe]                              # (n_samples, fh)

    up_hit = forward_closes >= up_thr.unsqueeze(1)
    dn_hit = forward_closes <= dn_thr.unsqueeze(1)

    big = forward_hours + 1
    up_idx = torch.where(
        up_hit, offsets.unsqueeze(0).expand_as(up_hit),
        torch.full_like(up_hit, big, dtype=torch.int64),
    ).min(dim=1).values                                             # (n_samples,)
    dn_idx = torch.where(
        dn_hit, offsets.unsqueeze(0).expand_as(dn_hit),
        torch.full_like(dn_hit, big, dtype=torch.int64),
    ).min(dim=1).values

    up_first = (up_idx <= dn_idx) & (up_idx <= forward_hours)
    dn_first = (~up_first) & (dn_idx <= forward_hours)

    labels = torch.full((n_samples,), 1, dtype=torch.int8, device=device)
    labels[up_first] = 2
    labels[dn_first] = 0

    out = torch.where(truncated, out, labels)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def batch_build_samples_for_pid(
    candles: List[Dict[str, float]],
    *,
    forward_hours: int,
    label_thresh: float,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched-tensor equivalent of train_xgb_v4_5._build_samples_for_pid.

    For one PID, vectorize feature extraction across all valid sample bars.
    Returns numpy arrays (host memory) so the caller can hand them straight
    to numpy / xgboost just like _build_samples_for_pid.

    Args:
        candles: chronologically-sorted OHLCV list (same format as per-bar path).
        forward_hours: triple-barrier horizon (e.g. 72).
        label_thresh: triple-barrier threshold (e.g. 0.03).
        device: "cpu" or "cuda". On "cuda", tensors are created on GPU and
            results are moved back to host as numpy arrays.

    Returns:
        (X, y, ts) — same shape and dtype as _build_samples_for_pid:
            X:  (N, 210) float64
            y:  (N,)     int8 (0=DOWN, 1=NEUTRAL, 2=UP)
            ts: (N,)     int64 epoch seconds

    Empty/insufficient input -> three empty arrays.
    """
    micro = TIER_WINDOWS_V45["micro"]                       # 60
    meso  = TIER_WINDOWS_V45["meso"]                        # 168
    macro = TIER_WINDOWS_V45["macro"]                       # 336
    bb_prefix = BB_PERIOD                                   # 20
    sample_offset = macro + bb_prefix                       # 356

    n_candles = len(candles)
    min_needed = sample_offset + forward_hours + 1
    if n_candles < min_needed:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )

    ohlcv = _build_pid_ohlcv(candles, device=device)        # (n_candles, 5)
    closes = ohlcv[:, 3]                                    # (n_candles,)

    # Sample bars: i in [sample_offset, n_candles - forward_hours). Triple-barrier
    # at i needs closes[i+1 .. i+forward_hours] all in range — last valid i is
    # n_candles - forward_hours - 1 (so that i + forward_hours == n_candles - 1).
    last_valid_i = n_candles - forward_hours - 1
    n_samples = last_valid_i - sample_offset + 1
    if n_samples <= 0:
        return (
            np.zeros((0, N_FEATURES_V45), dtype=np.float64),
            np.zeros(0, dtype=np.int8),
            np.zeros(0, dtype=np.int64),
        )
    sample_indices = torch.arange(
        sample_offset, sample_offset + n_samples, device=device, dtype=torch.int64,
    )

    # Per-tier feature blocks
    tier_blocks: List[torch.Tensor] = []
    for tier_name in _TIER_ORDER:
        tier_window = TIER_WINDOWS_V45[tier_name]
        window_len = tier_window + bb_prefix

        # tensor.unfold(0, window_len, 1)[k] corresponds to bars [k .. k+window_len-1].
        # Per-bar reference uses candles[i-window_len:i] (= bars [i-W .. i-1]).
        # For sample bar i, we want unfold index k = i - W.
        ohlcv_unfold = ohlcv.unfold(0, window_len, 1)       # (n_unfold, 5, window_len)
        ohlcv_unfold = ohlcv_unfold.permute(0, 2, 1)        # (n_unfold, window_len, 5)
        first_sample_unfold_idx = sample_offset - window_len
        ohlcv_windows = ohlcv_unfold[
            first_sample_unfold_idx : first_sample_unfold_idx + n_samples
        ].contiguous()                                      # (n_samples, window_len, 5)

        # Per-window local BB (channels 5, 6)
        closes_windows = ohlcv_windows[:, :, 3].contiguous()
        bb_pos, bb_wid = _local_bb_in_windows(closes_windows)
        bb_windows = torch.stack([bb_pos, bb_wid], dim=-1)  # (n_samples, window_len, 2)

        # 7-channel windows
        ch_windows = torch.cat([ohlcv_windows, bb_windows], dim=-1)  # (n_samples, W, 7)

        # 10 stats per channel
        tier_stats = _batch_stats(ch_windows)               # (n_samples, 7, 10)
        tier_blocks.append(tier_stats)

    # Stack tiers and reshape to feature-major layout matching feature_names_v4_5().
    # feature_names_v4_5 layout: channel-major -> tier-major -> stat-major
    #   slot index = c * (N_TIERS * N_STATS) + tier * N_STATS + stat
    # tier_blocks[t] has shape (n_samples, 7, 10).
    # Stack to (n_samples, 7, n_tiers, 10), then reshape to (n_samples, 210).
    all_stats = torch.stack(tier_blocks, dim=2)             # (n_samples, 7, 3, 10)
    X = all_stats.reshape(n_samples, N_FEATURES_V45)        # (n_samples, 210)

    # Labels
    y = _batch_triple_barrier(closes, sample_indices, forward_hours, label_thresh)

    # Timestamps from the "start" field of each sample bar
    starts_np = np.empty(n_samples, dtype=np.int64)
    for s, i in enumerate(range(sample_offset, sample_offset + n_samples)):
        starts_np[s] = int(candles[i]["start"])

    # Defensive filter for any -1 labels (shouldn't happen given our n_samples bound,
    # but keeps the contract explicit).
    y_host = y.cpu().numpy().astype(np.int8)
    X_host = X.cpu().numpy().astype(np.float64)
    valid = y_host >= 0
    return X_host[valid], y_host[valid], starts_np[valid]
