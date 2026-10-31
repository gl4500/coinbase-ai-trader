import numpy as np
import pandas as pd

from tools.slow_trend.rule import decisions, desired_state

IDX = pd.date_range("2024-01-01", periods=6, freq="D")


def test_long_above_flat_below():
    d = decisions(pd.Series([1, 2, 3, 2, 2, 5], index=IDX, dtype=float), 3)
    assert np.isnan(d.iloc[0]) and np.isnan(d.iloc[1])
    assert d.iloc[2] == 1.0 and d.iloc[3] == 0.0


def test_equality_is_flat():
    d = decisions(pd.Series([2.0] * 6, index=IDX), 3)
    assert (d.dropna() == 0.0).all()


def test_missing_day_holds_state():
    d = decisions(pd.Series([1, 2, 3, np.nan, 9, 10], index=IDX, dtype=float), 3)
    assert d.iloc[3:6].isna().all()
    s = desired_state(d)
    assert s.iloc[2] and s.iloc[3] and s.iloc[5]


def test_state_starts_flat_on_any_slice():
    d = pd.Series([1.0, np.nan, np.nan], index=IDX[:3])
    assert list(desired_state(d.iloc[1:])) == [False, False]
