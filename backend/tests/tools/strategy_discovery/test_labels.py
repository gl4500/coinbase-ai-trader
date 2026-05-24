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


def test_trail_stop_fires_at_atr_floor():
    # Entry at close=100. Bar 1: high=110 (new peak). Bar 2: low=103.4
    # → drawdown from peak = 103.4/110 - 1 = -0.06 (exactly the 6% floor).
    # ATR provided is 0.03 (below floor) → effective trail = 6% floor.
    # Trail exit price = peak * (1 - 0.06) = 110 * 0.94 = 103.4.
    # Net label = (103.4/100 - 1) - 0.012 = 0.034 - 0.012 = 0.022.
    df = _frame(
        closes=[100.0, 110.0, 103.4, 103.4],
        highs=[100.0, 110.0, 103.4, 103.4],
        lows=[100.0, 100.0, 103.4, 103.4],
        atrs=[0.03, 0.03, 0.03, 0.03],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[3])
    assert out.loc[0, "label_h3"] == pytest.approx(0.022, abs=1e-9)


def test_max_hold_cap_at_168_for_h168():
    # Construct a 200-bar series where price drifts up linearly (no SL, no
    # trail triggers given small ATR). horizon=168 should exit at index 168
    # post-entry, NOT at any later bar.
    n = 200
    closes = list(np.linspace(100.0, 200.0, n))
    df = _frame(
        closes=closes,
        highs=[c + 0.01 for c in closes],          # tiny range → trail never fires
        lows= [c - 0.01 for c in closes],
        atrs=[0.0001] * n,                          # ATR floor (6%) governs; never fires
    )
    out = simulate_dynamic_exit_labels(df, horizons=[168])
    # Entry at index 0, horizon_cap = min(168, 168) = 168, exit at index 168.
    entry_close = closes[0]
    exit_close  = closes[168]
    expected = (exit_close / entry_close - 1.0) - _DEFAULT_ROUND_TRIP_FEE
    assert out.loc[0, "label_h168"] == pytest.approx(expected, abs=1e-9)
