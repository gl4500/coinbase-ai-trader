"""Tests for tools.strategy_discovery.labels (Phase 2)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.labels import (
    _DEFAULT_ROUND_TRIP_FEE,
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
    return pd.DataFrame(
        {
            "ts": np.arange(n, dtype="int64") * 3_600_000,
            "open": np.array(closes, dtype="float64"),
            "high": np.array(highs, dtype="float64"),
            "low": np.array(lows, dtype="float64"),
            "close": np.array(closes, dtype="float64"),
            "atr14_pct": np.array(atrs, dtype="float64"),
        }
    )


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
        highs=[c + 0.01 for c in closes],  # tiny range → trail never fires
        lows=[c - 0.01 for c in closes],
        atrs=[0.0001] * n,  # ATR floor (6%) governs; never fires
    )
    out = simulate_dynamic_exit_labels(df, horizons=[168])
    # Entry at index 0, horizon_cap = min(168, 168) = 168, exit at index 168.
    entry_close = closes[0]
    exit_close = closes[168]
    expected = (exit_close / entry_close - 1.0) - _DEFAULT_ROUND_TRIP_FEE
    assert out.loc[0, "label_h168"] == pytest.approx(expected, abs=1e-9)


def test_horizon_reached_without_trigger_uses_close():
    # Entry at close=100. Bars 1..4 stay flat. horizon=4 reaches without
    # SL or trail triggers — exit at closes[4] = 100.
    # Net label = (100/100 - 1) - 0.012 = -0.012.
    df = _frame(
        closes=[100.0] * 6,
        highs=[100.0] * 6,
        lows=[100.0] * 6,
        atrs=[0.02] * 6,
    )
    out = simulate_dynamic_exit_labels(df, horizons=[4])
    assert out.loc[0, "label_h4"] == pytest.approx(-0.012, abs=1e-9)


def test_stop_loss_priority_over_trail():
    # Entry at close=100. Bar 1: high=120 (new peak), low=91.
    # - SL trigger: low/entry - 1 = 91/100 - 1 = -0.09 <= -0.08 → SL fires.
    # - Trail trigger: low/peak - 1 = 91/120 - 1 = -0.2417 <= -max(atr, 0.06)=-0.06 → also fires.
    # Both trigger in the same bar — SL must win.
    # SL exit price = 100 * 0.92 = 92; label = -0.08 - 0.012 = -0.092.
    df = _frame(
        closes=[100.0, 95.0, 95.0],
        highs=[100.0, 120.0, 95.0],
        lows=[100.0, 91.0, 95.0],
        atrs=[0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[2])
    assert out.loc[0, "label_h2"] == pytest.approx(-0.092, abs=1e-9)


def test_fee_subtracted_from_label():
    # Entry at 100, flat 5 bars → exit at close[5] = 100. Default fee = 0.012.
    df = _frame(
        closes=[100.0] * 6,
        highs=[100.0] * 6,
        lows=[100.0] * 6,
        atrs=[0.02] * 6,
    )
    # With default fee
    out_default = simulate_dynamic_exit_labels(df, horizons=[5])
    assert out_default.loc[0, "label_h5"] == pytest.approx(-_DEFAULT_ROUND_TRIP_FEE, abs=1e-9)
    # With zero fee — gross PnL should be zero
    out_zero = simulate_dynamic_exit_labels(df, horizons=[5], round_trip_fee=0.0)
    assert out_zero.loc[0, "label_h5"] == pytest.approx(0.0, abs=1e-9)


def test_insufficient_forward_bars_returns_nan():
    # Only 3 rows total; horizon=5 requires 5 forward bars → entry at index 0
    # has only 2 forward bars → NaN label. Horizons that fit (h=2) must NOT be NaN.
    df = _frame(
        closes=[100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0],
        atrs=[0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[2, 5])
    # h=2 fits at index 0 (entry+2 = 2 is in-bounds)
    assert not math.isnan(out.loc[0, "label_h2"])
    # h=5 does NOT fit (entry+5 = 5 out of bounds)
    assert math.isnan(out.loc[0, "label_h5"])
    # h=2 does NOT fit at index 2 (entry+2 = 4 out of bounds) — NaN expected
    assert math.isnan(out.loc[2, "label_h2"])
