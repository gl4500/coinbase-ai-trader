"""Survivorship-aware top-N pid selection (#163).

Existing probes select top-N pids by `len(entry["X"])` — total cache sample
count. This is post-hoc: products that grew the most data dominate, including
products that joined the tracked set recently. The result is biased toward
recent winners.

`survivorship_aware_top_n(prods, n, snapshot_ts)` returns the top-N pids by
sample count where `entry sample timestamps ≤ snapshot_ts`. With
`snapshot_ts=None` it falls back to the legacy "all-time sample count"
behavior so callers can opt in incrementally.

Pair with `recommended_snapshot_ts(prods)` (median of all `first_ts` values)
for a sensible default cutoff that excludes products joining after the
median entry date.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

_BAR_SECS = 3600


def _samples_at_or_before(entry: dict, snapshot_ts: int) -> int:
    """Count samples in `entry` whose bar timestamp is ≤ `snapshot_ts`."""
    n = len(entry.get("X", []))
    if n == 0:
        return 0
    first_ts = int(entry["first_ts"])
    if first_ts > snapshot_ts:
        return 0
    indices = np.asarray(entry.get("indices", np.arange(n)), dtype=np.int64)
    if len(indices) != n:
        indices = np.arange(n, dtype=np.int64)
    sample_ts = first_ts + indices * _BAR_SECS
    return int(np.sum(sample_ts <= snapshot_ts))


def survivorship_aware_top_n(
    prods: Dict[str, dict],
    n: int,
    snapshot_ts: Optional[int] = None,
) -> List[str]:
    """Return up to `n` pids ranked by sample count visible at `snapshot_ts`.

    `snapshot_ts=None` reproduces the legacy `len(entry["X"])` ranking so
    existing call sites are unchanged until they opt in.
    """
    if not prods:
        return []
    sized: List[tuple] = []
    for pid, entry in prods.items():
        if snapshot_ts is None:
            count = len(entry.get("X", []))
        else:
            count = _samples_at_or_before(entry, int(snapshot_ts))
        if count <= 0:
            continue
        sized.append((pid, count))
    sized.sort(key=lambda x: (-x[1], x[0]))
    return [pid for pid, _ in sized[:n]]


def recommended_snapshot_ts(prods: Dict[str, dict]) -> int:
    """Median of `first_ts` across all products with non-empty X.

    Pids whose first sample arrives after this cutoff are excluded from the
    survivorship-aware top-N — they're "newcomers" relative to the corpus.
    """
    first_tss = [
        int(entry["first_ts"]) for entry in prods.values()
        if len(entry.get("X", [])) > 0
    ]
    if not first_tss:
        return 0
    return int(np.median(np.asarray(first_tss, dtype=np.int64)))
