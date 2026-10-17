import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.sim import Costs, Product
from tools.vol_target.fsim import run_weight_sleeve, solve_units

FREE = Costs(0.0, 0.0, 0.0)
TAKER = Costs(0.009, 0.009, 0.0)
FINE = Product(base_increment=1e-8, base_min=1e-8, quote_min=1.0)
IDX = pd.date_range("2024-01-07", periods=10, freq="D")  # Sunday 01-07 .. Tuesday 01-16


def _bars(opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)], dtype=float)


def _sched(*rows, close=100.0):  # (decision_day_index, execute_day_index, target)
    return pd.DataFrame(
        [(IDX[d], IDX[e], t, close) for d, e, t in rows],
        columns=["decision", "execute", "target", "decision_close"],
    ).set_index("decision")


def _marked_weight(cash, units, px):
    return units * px / (cash + units * px)


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_solve_units_hits_target_weight_net_of_fees_at_the_open(w):
    cash, units, px = 300.0, 2.0, 100.0
    u = solve_units(cash, units, px, w, TAKER)
    du = u - units
    fee = abs(du) * px * 0.009
    new_cash = cash - du * px - fee
    assert new_cash >= -1e-9
    assert _marked_weight(new_cash, u, px) == pytest.approx(w, abs=1e-9)


def test_full_weight_spends_all_cash_zero_weight_sells_all():
    assert solve_units(500.0, 0.0, 100.0, 1.0, TAKER) == pytest.approx(500 / (100 * 1.009))
    assert solve_units(0.0, 3.0, 100.0, 0.0, TAKER) == 0.0


def test_deadband_equality_does_not_trade():
    # Sunday decision: held 0 (cash), target exactly 0.10 -> |0.10 - 0| is NOT > 0.10
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 0.10)), 500.0, FREE, FINE, 0.10
    )
    assert (r.buys, r.within_deadband, r.executed) == (0, 1, 0)


def test_just_above_deadband_trades():
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 0.1001)), 500.0, FREE, FINE, 0.10
    )
    assert r.buys == 1 and r.executed_weights[0] == pytest.approx(0.1001, abs=1e-6)


def test_overnight_gap_solves_quantity_at_the_monday_open():
    # Sunday close 100, Monday open 200: target 0.5 must be met at the 200 open
    r = run_weight_sleeve(
        _bars([100, 200, 200], [100, 200, 200]), _sched((0, 1, 0.5)), 500.0, FREE, FINE, 0.10
    )
    assert r.executed_weights[0] == pytest.approx(0.5, abs=1e-6)
    assert r.equity.iloc[1] == pytest.approx(500.0, rel=1e-6)  # no gain: bought at 200


def test_d1_executes_the_frozen_sunday_target_on_tuesday():
    r = run_weight_sleeve(_bars([100] * 4, [100] * 4), _sched((0, 2, 0.6)), 500.0, FREE, FINE, 0.10)
    assert r.equity.index[2] == IDX[2] and r.buys == 1
    assert r.decision_log[0]["target"] == 0.6 and r.decision_log[0]["execution"] == "executed"
    assert r.executed_weights[0] == pytest.approx(0.6, abs=1e-6)


def test_missing_execution_open_expires_without_retry():
    r = run_weight_sleeve(
        _bars([100, np.nan, 100, 100], [100] * 4), _sched((0, 1, 0.8)), 500.0, FREE, FINE, 0.10
    )
    assert (r.missed_open, r.buys) == (1, 0)
    assert r.terminal_value == pytest.approx(500.0)


def test_invalid_decision_holds_units_and_cash():
    s = _sched((0, 1, 0.8), (7, 8, np.nan))
    r = run_weight_sleeve(_bars([100] * 10, [100] * 7 + [150] * 3), s, 500.0, FREE, FINE, 0.10)
    assert (r.buys, r.sells, r.invalid_decisions, r.valid_decisions) == (1, 0, 1, 1)


def test_size_rejected_rebalance_is_skipped_and_holdings_unchanged():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 0.15)), 500.0, FREE, coarse, 0.10
    )
    assert (r.size_skipped, r.buys) == (1, 0)  # 0.75 units rounds down to 0
    assert r.terminal_value == pytest.approx(500.0)


def test_never_overspends_with_fees_and_rounding():
    coarse = Product(base_increment=0.01, base_min=0.01, quote_min=1.0)
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 1.0)), 500.0, TAKER, coarse, 0.10
    )
    assert r.equity.min() > 0 and r.exec_fees > 0
    assert r.executed_weights[0] <= 1.0


def test_weights_are_recomputed_from_executed_rounded_holdings():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 0.55)), 1000.0, FREE, coarse, 0.10
    )
    assert r.decision_log[0]["target"] == 0.55 and r.executed_weights == [pytest.approx(0.5)]


def test_terminal_dust_is_unliquidatable_and_never_cash():
    tiny = Product(base_increment=1e-8, base_min=1e-8, quote_min=50.0)
    r = run_weight_sleeve(
        _bars([100] * 3, [100] * 3), _sched((0, 1, 0.12)), 500.0, FREE, tiny, 0.10
    )
    # 0.12*500 = 60 bought (>= quote_min); price unchanged so proceeds 60 >= 50 -> liquidatable
    assert not r.unliquidatable
    crash = run_weight_sleeve(
        _bars([100, 100, 10], [100, 100, 10]), _sched((0, 1, 0.12)), 500.0, FREE, tiny, 0.10
    )
    assert crash.unliquidatable and crash.residual_units > 0
    assert crash.terminal_value == pytest.approx(500.0 - 60.0)  # residual excluded


def test_exposure_is_time_average_marked_asset_fraction():
    r = run_weight_sleeve(_bars([100] * 4, [100] * 4), _sched((0, 1, 0.5)), 500.0, FREE, FINE, 0.10)
    assert r.exposure == pytest.approx((0 + 0.5 + 0.5 + 0.5) / 4, abs=1e-6)


def test_self_financing_round_trip_cost_identity():
    s = _sched((0, 1, 1.0), (7, 8, 0.0))
    r = run_weight_sleeve(_bars([100] * 10, [100] * 10), s, 500.0, TAKER, FINE, 0.10)
    assert r.terminal_value == pytest.approx(500.0 * (1 - 0.009) / (1 + 0.009), rel=1e-6)
    assert (r.buys, r.sells) == (1, 1)


def test_partial_sell_residual_below_base_min_is_unliquidatable_even_above_quote_min():
    # buy to 0.5 (2.5 units), then cut to 0.02 (0.1 units): residual < base_min 1.0 but its
    # proceeds (10) clear quote_min (1). The full size contract makes it unsellable.
    lot = Product(base_increment=0.01, base_min=1.0, quote_min=1.0)
    s = _sched((0, 1, 0.5), (7, 8, 0.02))
    r = run_weight_sleeve(_bars([100] * 10, [100] * 10), s, 500.0, FREE, lot, 0.10)
    assert (r.buys, r.sells) == (1, 1)
    assert r.unliquidatable and r.residual_units == pytest.approx(0.1)
    assert r.terminal_value == pytest.approx(490.0)


# -- Codex plan-review N4-N6 -----------------------------------------------------


@pytest.mark.parametrize(
    "target,held,expected",
    [
        (0.4, 0.3, False),
        (0.3, 0.4, False),
        (0.6, 0.5, False),
        (0.5, 0.6, False),
        (0.41, 0.3, True),
        (0.19, 0.3, True),
    ],
)
def test_deadband_boundary_is_frozen_at_12_decimals_in_both_directions(target, held, expected):
    from tools.vol_target.fsim import eligible

    assert 0.4 - 0.3 > 0.10  # the binary-float trap this rule exists for
    assert eligible(target, held, 0.10) is expected


def test_missing_initialisation_sunday_close_suppresses_even_a_fixed_target():
    # pre-period Sunday (index 0 is before bars start at index 1) with a NaN close
    bars = _bars([100] * 3, [100] * 3).iloc[1:]
    r = run_weight_sleeve(bars, _sched((0, 1, 0.5), close=float("nan")), 500.0, FREE, FINE, 0.10)
    assert (r.buys, r.invalid_decisions) == (0, 1)
    ok = run_weight_sleeve(bars, _sched((0, 1, 0.5)), 500.0, FREE, FINE, 0.10)
    assert ok.buys == 1


def test_decision_log_records_every_decision_at_decision_time():
    s = _sched((0, 1, 0.05), (7, 8, 0.9))
    r = run_weight_sleeve(_bars([100] * 8 + [np.nan, 100], [100] * 10), s, 500.0, FREE, FINE, 0.10)
    log = r.decision_log
    assert [e["outcome"] for e in log] == ["within_deadband", "eligible"]
    assert log[0]["target"] == 0.05 and log[0]["held"] == 0.0
    assert log[1]["execution"] == "missed_open" and r.executed_weights == []


def test_sell_side_fee_and_slip_hit_the_target_weight():
    c = Costs(0.009, 0.009, 0.0025)
    cash, units, px, w = 100.0, 4.0, 100.0, 0.3
    u = solve_units(cash, units, px, w, c)
    sold = units - u
    new_cash = cash + sold * px * (1 - 0.0025) * (1 - 0.009)
    assert sold > 0 and _marked_weight(new_cash, u, px) == pytest.approx(w, abs=1e-9)


def test_overnight_gap_reverses_the_trade_direction_from_existing_holdings():
    # hold 0.5 bought at 100; Sunday close 100 asks for 0.7 (a buy), but Monday opens at 300:
    # marked at the open the sleeve is already 0.75, so the frozen 0.7 target is a SELL
    s = _sched((0, 1, 0.5), (7, 8, 0.7))
    r = run_weight_sleeve(
        _bars([100] * 8 + [300, 300], [100] * 8 + [300, 300]), s, 500.0, FREE, FINE, 0.10
    )
    assert (r.buys, r.sells) == (1, 1)
    assert r.executed_weights[-1] == pytest.approx(0.7, abs=1e-6)
