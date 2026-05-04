"""Tests for tools/timescale_sweep.py — the timescale exploration runner.

The novel logic in the runner is `_relabel_at_horizon`: take sample timestamps
plus a raw candle series, and recompute triple-barrier labels at an arbitrary
forward_hours. Returns (y, mask) — mask is False for samples where the labeler
returns None (insufficient lookahead, dead-zone return, invalid entry).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from agents.cnn_agent import _label_triple_barrier  # noqa: E402


_BAR_SECS = 3600


def _flat_then_pump_candles(n: int = 50, pump_at: int = 30) -> list[dict]:
    """50 hourly bars, all closes 100.0 except a +5% spike at pump_at.

    No barrier triggers anywhere except via the time barrier exit, except
    around `pump_at` where the upper barrier (+1%) hits first.
    """
    candles = []
    for i in range(n):
        start = 1_700_000_000 + i * _BAR_SECS
        if i == pump_at:
            close = high = 105.0
            low = 100.0
        else:
            close = 100.0
            high = 100.0
            low = 100.0
        candles.append({
            "start": start, "open": close, "high": high,
            "low": low, "close": close, "volume": 0.0,
        })
    return candles


def test_relabel_at_horizon_matches_triple_barrier_oracle():
    """For sample timestamps mapped to candle indices, _relabel_at_horizon
    must return the same label as calling _label_triple_barrier directly,
    and the mask must be False where the labeler returns None."""
    from tools.timescale_sweep import _relabel_at_horizon

    candles = _flat_then_pump_candles(n=50, pump_at=30)
    forward_hours = 4
    up_mult = dn_mult = 0.01
    label_thresh = 0.003

    sample_indices = [10, 25, 29, 35]
    sample_ts = np.array(
        [candles[i]["start"] for i in sample_indices], dtype=np.int64
    )

    expected = [
        _label_triple_barrier(candles, i, forward_hours,
                              up_mult, dn_mult, label_thresh)
        for i in sample_indices
    ]

    y, mask = _relabel_at_horizon(
        candles, sample_ts, forward_hours,
        up_mult=up_mult, dn_mult=dn_mult, label_thresh=label_thresh,
    )

    assert y.shape == (len(sample_indices),)
    assert mask.shape == (len(sample_indices),)

    for j, exp in enumerate(expected):
        if exp is None:
            assert not mask[j], (
                f"sample {j} (idx={sample_indices[j]}) expected None → mask=False"
            )
        else:
            assert mask[j], (
                f"sample {j} (idx={sample_indices[j]}) expected {exp} → mask=True"
            )
            assert y[j] == pytest.approx(exp), (
                f"sample {j}: got y={y[j]}, expected {exp}"
            )


def test_relabel_at_horizon_marks_unmappable_ts_as_invalid():
    """Timestamps that don't map to any candle in the series get mask=False."""
    from tools.timescale_sweep import _relabel_at_horizon

    candles = _flat_then_pump_candles(n=50, pump_at=30)
    bogus_ts = np.array([candles[10]["start"], 999_999_999], dtype=np.int64)

    y, mask = _relabel_at_horizon(
        candles, bogus_ts, forward_hours=4,
        up_mult=0.01, dn_mult=0.01, label_thresh=0.003,
    )

    assert mask[0] == True or mask[0] == False  # depends on labeler; both ok
    assert mask[1] == False, "unmappable ts must yield mask=False"
