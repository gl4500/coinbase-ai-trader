import math

import numpy as np
import pandas as pd
import pytest

from tools.vol_target import prereg as P
from tools.vol_target.signal import annualised_sigma, schedule, target_weight

IDX = pd.date_range("2024-01-01", periods=60, freq="D")  # 2024-01-01 is a Monday


def _close(vals=None):
    vals = vals if vals is not None else 100 * np.exp(0.01 * np.sin(np.arange(60)))
    return pd.Series(vals, index=IDX[: len(vals)], dtype=float)


def test_sigma_is_sample_std_of_20_log_returns_annualised():
    c = _close()
    r = np.log(c.to_numpy())[1:] - np.log(c.to_numpy())[:-1]
    expected = np.std(r[-20:], ddof=1) * math.sqrt(365)
    assert annualised_sigma(c).iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_sigma_needs_21_consecutive_valid_closes():
    s = annualised_sigma(_close())
    assert s.iloc[:20].isna().all() and np.isfinite(s.iloc[20])


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
def test_any_invalid_close_in_the_window_invalidates_sigma(bad):
    v = _close().to_numpy().copy()
    v[40] = bad
    s = annualised_sigma(_close(v))
    assert s.iloc[40:61].isna().all()  # every window containing day 40 (through day 60)
    assert np.isfinite(s.iloc[39])


def test_target_weight_caps_at_one_and_scales_down():
    w = target_weight(pd.Series([0.25, 0.5, 1.0, 2.0]))
    assert list(w) == [1.0, 1.0, 0.5, 0.25]


def test_zero_sigma_is_full_weight_and_invalid_sigma_is_no_target():
    w = target_weight(pd.Series([0.0, np.nan, np.inf, -0.1]))
    assert w.iloc[0] == 1.0 and w.iloc[1:].isna().all()


def test_schedule_tuesday_start_waits_for_first_monday():
    c = _close()
    sch = schedule(c, "2024-01-30", "2024-02-28", delay=0)  # 2024-01-30 is a Tuesday
    assert sch["execute"].iloc[0] == pd.Timestamp("2024-02-05")  # Monday
    assert sch.index[0] == pd.Timestamp("2024-02-04")  # its Sunday decision


def test_schedule_monday_start_uses_the_preceding_sunday():
    sch = schedule(_close(), "2024-01-29", "2024-02-28", delay=0)  # Monday start
    assert sch.index[0] == pd.Timestamp("2024-01-28")
    assert sch["execute"].iloc[0] == pd.Timestamp("2024-01-29")


def test_schedule_d1_executes_tuesday_with_the_sunday_target():
    c = _close()
    p0 = schedule(c, "2024-01-29", "2024-02-28", delay=0)
    d1 = schedule(c, "2024-01-29", "2024-02-28", delay=1)
    assert d1["execute"].iloc[0] == pd.Timestamp("2024-01-30")
    assert d1["target"].iloc[0] == p0["target"].iloc[0]  # frozen at Sunday, not re-decided


def test_schedule_excludes_executions_after_the_period_end():
    sch = schedule(_close(), "2024-01-29", "2024-02-25", delay=0)  # ends on a Sunday
    assert sch["execute"].max() <= pd.Timestamp("2024-02-25")


def test_prereg_freezes_the_agreed_values():
    assert (P.VOL_RETURNS, P.ANNUALISATION, P.SIGMA_TARGET, P.DEADBAND) == (20, 365, 0.5, 0.10)
    assert (P.DEV_START, P.DEV_END, P.BLOCK_START, P.BLOCK_END) == (
        "2016-08-30",
        "2025-04-13",
        "2025-04-14",
        "2026-10-02",
    )
    assert P.MIN_BLOCK_VALID_DECISIONS == 4
    assert P.SOURCE_SNAPSHOT_LOCK.startswith("sha256:5215b520ba2d25b7")
