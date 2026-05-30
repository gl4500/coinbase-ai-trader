"""Tests for tools.strategy_discovery.info_bars (matched-count dollar bars)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.info_bars import aggregate_dollar_bars


def _mk_1h(starts, opens, highs, lows, closes, vols):
    return pd.DataFrame({
        "start":  np.asarray(starts,  dtype="int64"),
        "open":   np.asarray(opens,   dtype="float64"),
        "high":   np.asarray(highs,   dtype="float64"),
        "low":    np.asarray(lows,    dtype="float64"),
        "close":  np.asarray(closes,  dtype="float64"),
        "volume": np.asarray(vols,    dtype="float64"),
    })


def test_empty_input_returns_empty_with_full_schema():
    out = aggregate_dollar_bars(_mk_1h([], [], [], [], [], []))
    assert list(out.columns) == [
        "start", "end", "open", "high", "low", "close",
        "volume", "dollar_value", "n_1h",
    ]
    assert len(out) == 0


def test_zero_total_dollar_value_returns_empty():
    df = _mk_1h([1, 2, 3], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10], [0, 0, 0])
    out = aggregate_dollar_bars(df)
    assert len(out) == 0


def test_emits_n1h_bars_when_dollar_value_is_flat():
    # Every 1h row carries identical dollar value → threshold = mean → 1 bar per row.
    df = _mk_1h(
        starts=[100, 200, 300, 400],
        opens=[1.0] * 4, highs=[1.0] * 4, lows=[1.0] * 4, closes=[1.0] * 4,
        vols=[10.0] * 4,
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 4
    assert out["start"].tolist() == [100, 200, 300, 400]
    assert out["end"].tolist() == [100, 200, 300, 400]
    assert out["n_1h"].tolist() == [1, 1, 1, 1]


def test_ohlc_integrity_when_two_rows_merge_into_one_bar():
    # Row 0 dollar_value=1, row 1 dollar_value=5 → threshold = 3.0; bar 1 closes on row 1.
    df = _mk_1h(
        starts=[10, 20],
        opens=[100.0, 105.0],
        highs=[110.0, 115.0],
        lows=[95.0, 100.0],
        closes=[105.0, 112.0],
        vols=[(1.0 / ((110 + 95 + 105) / 3.0)), (5.0 / ((115 + 100 + 112) / 3.0))],
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["start"] == 10
    assert row["end"] == 20
    assert row["open"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(112.0)
    assert row["high"] == pytest.approx(115.0)   # max over rows 0, 1
    assert row["low"] == pytest.approx(95.0)     # min over rows 0, 1
    assert row["n_1h"] == 2
    assert row["dollar_value"] == pytest.approx(6.0, abs=1e-9)


def test_residual_below_threshold_is_dropped():
    # Three rows, threshold = mean = (1 + 5 + 0.5) / 3 ≈ 2.167. Row 0 alone (1.0) < threshold;
    # rows 0+1 cumulative 6.0 ≥ threshold → bar closes on row 1. Row 2 (0.5) < threshold → dropped.
    df = _mk_1h(
        starts=[10, 20, 30],
        opens=[100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0],
        vols=[1.0 / 100.0, 5.0 / 100.0, 0.5 / 100.0],
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 1
    assert out.iloc[0]["end"] == 20
    assert out.iloc[0]["n_1h"] == 2


def test_volume_and_dollar_value_are_sums_over_merged_rows():
    df = _mk_1h(
        starts=[1, 2, 3],
        opens=[1.0] * 3, highs=[1.0] * 3, lows=[1.0] * 3, closes=[1.0] * 3,
        vols=[10.0, 20.0, 30.0],
    )
    out = aggregate_dollar_bars(df)
    # threshold = mean dollar_value = 20.0; each row's dv = its volume × 1.0.
    # Row 0 (10) < 20; rows 0+1 (30) ≥ 20 → bar 1 closes at row 1, sums vol=30, dv=30.
    # Row 2 (30) ≥ 20 → bar 2 closes at row 2, sums vol=30, dv=30.
    assert len(out) == 2
    assert out["volume"].tolist() == [pytest.approx(30.0), pytest.approx(30.0)]
    assert out["dollar_value"].tolist() == [pytest.approx(30.0), pytest.approx(30.0)]
    assert out["n_1h"].tolist() == [2, 1]


def test_start_field_is_monotonic_nondecreasing():
    rng = np.random.default_rng(7)
    n = 200
    starts = np.arange(n, dtype="int64") * 3600
    vols = rng.uniform(0.1, 10.0, size=n)
    df = _mk_1h(starts, [1.0] * n, [1.0] * n, [1.0] * n, [1.0] * n, vols)
    out = aggregate_dollar_bars(df)
    assert len(out) > 0
    arr = out["start"].to_numpy()
    assert np.all(np.diff(arr) >= 0)
