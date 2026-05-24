"""Tests for tools.strategy_discovery.labels (Phase 2)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.labels import (
    _DEFAULT_ATR_TRAIL_FLOOR,
    _DEFAULT_ROUND_TRIP_FEE,
    _DEFAULT_STOP_LOSS_PCT,
    simulate_dynamic_exit_labels,
)


def _frame(closes, highs=None, lows=None, atrs=None):
    n = len(closes)
    if highs is None:
        highs = list(closes)
    if lows is None:
        lows = list(closes)
    if atrs is None:
        atrs = [0.02] * n
    return pd.DataFrame({
        "ts":         np.arange(n, dtype="int64") * 3_600_000,
        "open":       np.array(closes, dtype="float64"),
        "high":       np.array(highs,  dtype="float64"),
        "low":        np.array(lows,   dtype="float64"),
        "close":      np.array(closes, dtype="float64"),
        "atr14_pct":  np.array(atrs,   dtype="float64"),
    })


def test_stop_loss_fires_at_8pct_drawdown():
    # Entry at close=100; bar 1 dips to low=91 (-9% from entry, beats SL=-8%).
    # Expected exit price = 100 * (1 - 0.08) = 92; label = (92/100 - 1) - 0.012 = -0.092.
    df = _frame(
        closes=[100.0, 95.0, 95.0, 95.0],
        highs=[100.0, 95.0, 95.0, 95.0],
        lows=[100.0, 91.0, 95.0, 95.0],
        atrs=[0.02, 0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[3])
    assert out.loc[0, "label_h3"] == pytest.approx(-0.092, abs=1e-9)
