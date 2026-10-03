import pytest

from tools.slow_trend import gates as G
from tools.vol_target.verdict import verdict

OK = {"pass": True, "affected": False}
FAIL = {"pass": False, "affected": False}
AFF = {"pass": False, "affected": True}
SC = ("P0", "S10", "S25", "D1", "SM")
COVER = {"BTC-USD": 50, "ETH-USD": 50}


def _res(**over):
    r = {p: {s: dict(OK) for s in SC} for p in ("dev", "block")}
    for key, v in over.items():
        p, s = key.split("_")
        r[p][s] = v
    return r


@pytest.mark.parametrize(
    "res,expected",
    [
        (_res(dev_P0=FAIL), ("KILL", "primary_failed")),
        (_res(block_P0=AFF), ("INCONCLUSIVE", "terminal_size")),
        (_res(block_S25=AFF), ("INCONCLUSIVE", "terminal_size")),
        (_res(dev_D1=FAIL), ("INCONCLUSIVE", "fragile")),
        (_res(block_SM=FAIL), ("PASS_TO_FORWARD", "all_gates")),  # SM never gates
        (_res(), ("PASS_TO_FORWARD", "all_gates")),
    ],
)
def test_shared_branches_match_the_slow_trend_order(res, expected):
    v = verdict(True, res, COVER)
    assert (v["verdict"], v["reason"]) == expected
    assert v == G.verdict(True, res, 10**6)  # identical wherever coverage is met


def test_data_failure_first():
    assert verdict(False, None, {})["reason"] == "data"


def test_coverage_is_per_sleeve_and_last():
    v = verdict(True, _res(), {"BTC-USD": 50, "ETH-USD": 3})
    assert (v["verdict"], v["reason"]) == ("INCONCLUSIVE", "insufficient_coverage")
    assert verdict(True, _res(dev_P0=FAIL), {"BTC-USD": 0, "ETH-USD": 0})["verdict"] == "KILL"
