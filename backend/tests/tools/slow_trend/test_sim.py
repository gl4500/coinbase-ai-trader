import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

IDX = pd.date_range("2024-01-01", periods=5, freq="D")
FREE = Costs(0.0, 0.0, 0.0)
FINE = Product(base_increment=1e-8, base_min=1e-8, quote_min=1.0)


def _bars(opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)], dtype=float)


def _t(*vals):
    return pd.Series(list(vals), index=IDX[: len(vals)])


def test_trade_never_executes_on_decision_day():
    state = _t(True, True, True, True, True)
    assert list(trend_target(state, 0)[:2]) == [False, True]
    assert list(trend_target(state, 1)[:3]) == [False, False, True]


def test_round_trip_cost_identity():
    r = run_sleeve(
        _bars([100] * 4, [100] * 4),
        _t(True, True, False, False),
        500.0,
        Costs(0.009, 0.009, 0.0),
        FINE,
    )
    assert r.terminal_value == pytest.approx(500.0 * (1 - 0.009) / (1 + 0.009), rel=1e-6)
    assert (r.entries, r.exits, r.round_trips, r.terminal_fee) == (1, 1, 1, 0.0)


def test_slip_is_adverse_on_both_legs():
    r = run_sleeve(
        _bars([100] * 3, [100] * 3), _t(True, False, False), 500.0, Costs(0.0, 0.0, 0.0025), FINE
    )
    assert r.terminal_value == pytest.approx(500.0 * 0.9975 / 1.0025, rel=1e-6)
    units = 500.0 / 100.25
    assert r.slippage_cost == pytest.approx(2 * units * 100 * 0.0025, rel=1e-6)
    assert r.terminal_value == pytest.approx(500.0 - r.slippage_cost, rel=1e-6)


def test_sell_below_quote_min_is_retained_and_endpoint_unliquidatable():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(
        _bars([2.0, 0.5, 0.5], [2.0, 0.5, 0.5]), _t(True, False, False), 2.5, FREE, coarse
    )
    assert r.entries == 1 and r.exits == 0 and r.skipped_exits == 2
    assert r.unliquidatable is True
    assert r.terminal_value == pytest.approx(0.5)  # cash only: a LOWER BOUND, not a value
    assert (r.residual_units, r.residual_marked_value) == (1.0, 0.5)
    assert r.terminal_fee == 0.0


def test_open_position_marked_in_equity_and_liquidated_in_terminal_value():
    r = run_sleeve(
        _bars([100, 100], [100, 110]), _t(True, True), 500.0, Costs(0.0, 0.009, 0.0), FINE
    )
    assert r.equity.iloc[-1] == pytest.approx(550.0)
    assert r.terminal_value == pytest.approx(550.0 * (1 - 0.009))
    assert r.terminal_fee == pytest.approx(550.0 * 0.009) and r.round_trips == 0


def test_costs_reconcile_with_terminal_value():
    c = Costs(0.009, 0.009, 0.0)
    r = run_sleeve(_bars([100] * 3, [100] * 3), _t(True, True, True), 500.0, c, FINE)
    assert r.terminal_value == pytest.approx(500.0 - r.exec_fees - r.terminal_fee, rel=1e-9)
    assert r.traded_notional == pytest.approx(500.0 / 1.009, rel=1e-6)


def test_terminal_close_missing_raises():
    with pytest.raises(ValueError, match="terminal close missing"):
        run_sleeve(_bars([100, 100], [100, np.nan]), _t(True, True), 500.0, FREE, FINE)


def test_missing_open_defers_and_cash_marks_through_gap():
    r = run_sleeve(
        _bars([100, np.nan, 100], [100, np.nan, 100]), _t(False, True, True), 500.0, FREE, FINE
    )
    assert r.entries == 1 and r.equity.iloc[1] == pytest.approx(500.0)
    assert r.stale_mark_days == 1 and r.max_stale_run == 1


def test_missed_target_is_superseded_not_queued():
    r = run_sleeve(
        _bars([100, np.nan, 100], [100, 100, 100]), _t(False, True, False), 500.0, FREE, FINE
    )
    assert r.entries == 0 and r.exits == 0


def test_rejected_buy_retried_at_next_open():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(_bars([600, 400], [600, 400]), _t(True, True), 500.0, FREE, coarse)
    assert r.skipped == 1 and r.entries == 1


def test_below_quote_min_is_skipped():
    r = run_sleeve(_bars([100], [100]), _t(True), 0.5, FREE, FINE)
    assert r.skipped == 1 and r.entries == 0


def test_units_floor_to_increment():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(_bars([300, 300], [300, 300]), _t(True, True), 500.0, FREE, coarse)
    assert r.equity.iloc[-1] == pytest.approx(500.0)


def test_dca_tranches_are_fee_inclusive_and_mondays_only():
    idx = pd.date_range("2024-01-01", periods=15, freq="D")  # 2024-01-01 is a Monday
    bars = pd.DataFrame({"open": 10.0, "close": 10.0}, index=idx)
    r = run_dca(bars, 520.0, 52, Costs(0.009, 0.009, 0.0), FINE)
    assert r.entries == 3
    spent = 520.0 - (r.equity.iloc[-1] - r.traded_notional)
    assert spent == pytest.approx(3 * 10.0, rel=1e-6)  # 3 tranches of 520/52, fee inside
