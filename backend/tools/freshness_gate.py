"""Inference-time feature-freshness gate (#169).

A live (n_channels, seq_len) window is "fresh enough" only if no
required channel has been frozen at a single value for more than
`max_flat_bars` trailing bars. Frozen tails indicate paused feeds,
geo-blocks, broker hiccups, or stale caches — exactly the failure
modes that the stationarity audit (#164) and drift monitor (#170)
caught offline. This is the runtime counterpart: cheap, allocation-
light, no I/O — call it before scoring with CNN/XGB.

Caller (cnn_agent / xgb_signal) decides what to do with a stale
verdict: skip the bar, warn-and-score, or fall back to neutral.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

_DEFAULT_MAX_FLAT = 5


def _trailing_flat_bars(channel: np.ndarray) -> int:
    """Number of trailing identical-value bars (steps without change).

    A channel ending in [..., 5, 5, 5] returns 2: two transitions
    from the same value. A channel ending in [..., 4, 5] returns 0.
    Length 0 / 1 → 0.
    """
    arr = np.asarray(channel, dtype=np.float64).ravel()
    if arr.size < 2:
        return 0
    diffs = np.diff(arr)
    flat_mask = diffs == 0
    n = 0
    for v in flat_mask[::-1]:
        if v:
            n += 1
        else:
            break
    return int(n)


def evaluate_freshness(
    window: np.ndarray,
    max_flat_bars: int = _DEFAULT_MAX_FLAT,
    per_channel_max: Optional[dict] = None,
    ignore_channels: Optional[Iterable[int]] = None,
) -> dict:
    """Assess per-channel staleness of a (n_channels, seq_len) window.

    Returns:
      {
        "fresh": bool,                       # all required channels OK
        "stale_channels": list[int],         # channels exceeding their budget
        "channel_flat_bars": list[int],      # per-channel trailing-flat count
        "max_flat_bars": int,                # default budget echoed back
      }

    `per_channel_max` overrides the default for specific channels (e.g. a
    1h-cadence feed at 5m bar interval will legitimately repeat ~11 bars).
    `ignore_channels` are excluded from the freshness verdict — useful for
    permanently-zero geo-blocked channels.
    """
    arr = np.asarray(window, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"window must be 2D (n_channels, seq_len), got {arr.shape}")
    n_channels = arr.shape[0]
    overrides = dict(per_channel_max or {})
    skip = set(ignore_channels or [])

    flat_counts = [_trailing_flat_bars(arr[c]) for c in range(n_channels)]
    stale = []
    for c, flat in enumerate(flat_counts):
        if c in skip:
            continue
        budget = overrides.get(c, max_flat_bars)
        if flat > budget:
            stale.append(c)

    return {
        "fresh": len(stale) == 0,
        "stale_channels": stale,
        "channel_flat_bars": flat_counts,
        "max_flat_bars": int(max_flat_bars),
    }
