import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.metrics import boundary_returns, max_drawdown, paired_block_ci, weekly_returns
from tools.slow_trend.sim import Costs, Product, run_sleeve


def test_max_drawdown_is_seeded_with_initial():
    eq = pd.Series([95, 120, 90, 130, 65], dtype=float)
    assert max_drawdown(eq, 100.0) == pytest.approx(0.5)
    assert max_drawdown(pd.Series([90.0, 80.0]), 100.0) == pytest.approx(0.2)


def test_constant_price_buy_hold_drawdown_is_entry_fee():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    bars = pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)
    r = run_sleeve(
        bars, pd.Series(True, index=idx), 500.0, Costs(0.009, 0.009, 0.0), Product(1e-8, 1e-8, 1.0)
    )
    assert max_drawdown(r.equity, 500.0) == pytest.approx(r.exec_fees / 500.0, rel=1e-6)


def test_weekly_returns_use_complete_weeks_only():
    idx = pd.date_range("2024-01-03", "2024-01-19", freq="D")  # Wed .. Fri
    eq = pd.Series(100.0, index=idx)
    eq.loc["2024-01-14":] = 110.0
    w = weekly_returns(eq)
    assert list(w.index.strftime("%Y-%m-%d")) == ["2024-01-14"]
    assert w.iloc[0] == pytest.approx(0.10)


def test_boundary_returns_report_partial_head_and_tail():
    idx = pd.date_range("2024-01-03", "2024-01-19", freq="D")
    eq = pd.Series(100.0, index=idx)
    eq.iloc[-1] = 121.0
    b = boundary_returns(eq, 100.0)
    assert b["head"] == pytest.approx(0.0) and b["tail"] == pytest.approx(0.21)


def test_identical_series_zero_excess():
    s = pd.Series(np.random.default_rng(1).normal(0, 0.05, 60))
    ci = paired_block_ci(s, s.copy(), block=8, n=500, seed=7)
    assert ci["lo"] == ci["hi"] == ci["mean_excess"] == 0.0


def test_ci_is_deterministic_for_seed():
    rng = np.random.default_rng(2)
    a, b = pd.Series(rng.normal(0.01, 0.05, 80)), pd.Series(rng.normal(0, 0.05, 80))
    assert paired_block_ci(a, b, 8, 300, 11) == paired_block_ci(a, b, 8, 300, 11)


def test_short_series_reports_no_ci_instead_of_crashing():
    ci = paired_block_ci(pd.Series([0.01] * 5), pd.Series([0.0] * 5), 13, 100, 0)
    assert ci["lo"] is None and ci["hi"] is None and ci["n_weeks"] == 5


def test_ci_rejects_misaligned_indexes():
    a = pd.Series([0.1, 0.2], index=[0, 1])
    with pytest.raises(ValueError, match="aligned"):
        paired_block_ci(a, pd.Series([0.1, 0.2], index=[1, 2]), 1, 10, 0)


def test_ci_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        paired_block_ci(pd.Series([0.1, np.nan]), pd.Series([0.1, 0.2]), 1, 10, 0)
