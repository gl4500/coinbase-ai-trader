from tools.slow_trend.gates import affected, passes, verdict

OK = {"G1": True, "G2": True, "pass": True, "affected": False}
BAD = {"G1": False, "G2": True, "pass": False, "affected": False}
OK_AFFECTED = dict(OK, affected=True)
BAD_AFFECTED = dict(BAD, affected=True)


def _side(*flags):
    return {"unliquidatable": list(flags)}


def test_affected_by_buy_hold_only_or_trend_only():
    assert affected(_side(False, False), _side(False, True)) is True  # buy-and-hold only
    assert affected(_side(True, False), _side(False, False)) is True  # trend only
    assert affected(_side(False, False), _side(False, False)) is False


def test_buy_hold_write_down_cannot_manufacture_a_pass():
    # near-boundary: trend would fail the return comparison but for a buy-and-hold write-down
    r = verdict(True, _res(block=OK_AFFECTED), 10)
    assert r == {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}


def test_affected_failure_is_not_a_kill():
    assert verdict(True, _res(dev=BAD_AFFECTED), 10) == {
        "verdict": "INCONCLUSIVE",
        "reason": "terminal_size",
    }


def test_unaffected_failure_kills_even_if_other_period_affected():
    assert verdict(True, _res(dev=BAD, block=OK_AFFECTED), 10)["verdict"] == "KILL"


def test_affected_gating_sensitivity_is_terminal_size():
    assert verdict(True, _res(dev_D1=OK_AFFECTED), 10)["reason"] == "terminal_size"


def _res(dev=OK, block=OK, **over):
    base = {s: OK for s in ("P0", "S10", "S25", "D1", "SM")}
    r = {"dev": dict(base, P0=dev), "block": dict(base, P0=block)}
    for key, val in over.items():
        period, scen = key.split("_")
        r[period][scen] = val
    return r


def test_g1_strictly_above_initial():
    t = {"terminal_value": 1000.0, "net_return": 0.0, "max_drawdown": 0.1}
    assert passes(t, t, 1000.0, 2 / 3)["G1"] is False


def test_g2_return_or_drawdown():
    bh = {"terminal_value": 1500.0, "net_return": 0.5, "max_drawdown": 0.6}
    low_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.39}
    high_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.41}
    assert passes(low_dd, bh, 1000.0, 2 / 3)["G2"] is True
    assert passes(high_dd, bh, 1000.0, 2 / 3)["G2"] is False


def test_data_inadequate_first():
    assert verdict(False, None, 0) == {"verdict": "INCONCLUSIVE", "reason": "data"}


def test_block_failure_kills():
    assert verdict(True, _res(block=BAD), 10)["verdict"] == "KILL"


def test_fragile_when_gating_sensitivity_fails():
    assert verdict(True, _res(block_S25=BAD), 10) == {
        "verdict": "INCONCLUSIVE",
        "reason": "fragile",
    }


def test_maker_sensitivity_never_consulted():
    assert verdict(True, _res(dev_SM=BAD), 10)["verdict"] == "PASS_TO_FORWARD"
    assert verdict(True, _res(dev=BAD, dev_SM=OK), 10)["verdict"] == "KILL"


def test_insufficient_transitions():
    assert verdict(True, _res(), 3) == {
        "verdict": "INCONCLUSIVE",
        "reason": "insufficient_transitions",
    }


def test_pass():
    assert verdict(True, _res(), 4) == {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
