import pytest

from tools.alerts.carry import CarryParams, evaluate_dated, fut_fee

P = CarryParams(
    capital=1000.0,
    spot_maker_fee=0.005,
    fut_fee_rate=0.0002,
    fut_fee_min=0.15,
    cash_rate=0.04,
    reserve_rally=0.30,
    min_excess_annual=0.01,
    min_days=7,
)


def test_futures_fee_has_a_per_contract_minimum():
    assert fut_fee(100.0, P) == 0.15  # 0.02% of $100 = $0.02 -> minimum applies
    assert fut_fee(10_000.0, P) == pytest.approx(2.0)


def test_capital_uses_spot_notional_for_spot_and_futures_notional_for_margin():
    r = evaluate_dated(
        spot_ask=100.0,
        fut_bid=110.0,
        contract_size=1.0,
        short_margin=0.25,
        days_to_expiry=30.0,
        p=P,
    )
    expected = 100.0 * 1.005 + 0.25 * 110.0 + 0.30 * 110.0 * 1.25 + 2 * 0.15
    assert r["capital_per_contract"] == pytest.approx(expected)


def test_excess_is_premium_minus_costs_minus_forgone_cash_on_all_capital():
    r = evaluate_dated(
        spot_ask=100.0,
        fut_bid=102.0,
        contract_size=1.0,
        short_margin=0.25,
        days_to_expiry=30.0,
        p=P,
    )
    cap = r["capital_per_contract"]
    costs = 2 * 0.005 * 100.0 + 2 * 0.15
    forgone = cap * 0.04 * 30.0 / 365
    assert r["premium"] == pytest.approx(2.0)
    assert r["excess"] == pytest.approx(2.0 - costs - forgone)
    assert r["excess_annual"] == pytest.approx((2.0 - costs - forgone) / cap * 365 / 30.0)


def test_alerts_only_when_affordable_and_above_threshold():
    rich = evaluate_dated(
        spot_ask=100.0,
        fut_bid=103.0,
        contract_size=1.0,
        short_margin=0.25,
        days_to_expiry=30.0,
        p=P,
    )
    assert rich["contracts_affordable"] >= 1 and rich["alert"] is True
    thin = evaluate_dated(
        spot_ask=100.0,
        fut_bid=100.2,
        contract_size=1.0,
        short_margin=0.25,
        days_to_expiry=30.0,
        p=P,
    )
    assert thin["alert"] is False
    too_big = evaluate_dated(
        spot_ask=100.0,
        fut_bid=103.0,
        contract_size=20.0,
        short_margin=0.25,
        days_to_expiry=30.0,
        p=P,
    )
    assert too_big["contracts_affordable"] == 0 and too_big["alert"] is False


def test_no_alert_too_close_to_expiry_or_on_bad_quotes():
    near = evaluate_dated(
        spot_ask=100.0, fut_bid=103.0, contract_size=1.0, short_margin=0.25, days_to_expiry=3.0, p=P
    )
    assert near["alert"] is False
    for bad in (dict(spot_ask=0.0, fut_bid=103.0), dict(spot_ask=100.0, fut_bid=float("nan"))):
        r = evaluate_dated(contract_size=1.0, short_margin=0.25, days_to_expiry=30.0, p=P, **bad)
        assert r["alert"] is False and r["valid"] is False
