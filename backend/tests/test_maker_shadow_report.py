import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.maker_shadow_report import summarize


def _r(status, touched=False, ttc=None, spread=None, markout=None, gap=False, late=0.0):
    return {
        "status": status,
        "touched": touched,
        "time_to_cross_s": ttc,
        "spread_bps": spread,
        "markout_bps": markout,
        "feed_gap": gap,
        "finalised_late_s": late,
    }


def test_rates_are_conditional_on_clean_measured_intents():
    rows = [
        _r("crossed", True, 4.0, 20.0, -10.0),
        _r("not_crossed", True, None, 30.0),
        _r("not_crossed", False, None, 40.0),
        _r("crossed", True, 2.0, 50.0, 5.0, gap=True),  # excluded: feed gap
        _r("no_quote", gap=None, late=None),
        _r("duplicate", gap=None, late=None),
    ]
    s = summarize(rows)
    assert s["n_total"] == 6
    assert s["n_measured"] == 3
    assert s["n_feed_gap"] == 1
    assert s["n_no_quote"] == 1 and s["n_duplicate"] == 1
    assert s["coverage"] == pytest.approx(3 / 6)
    assert s["cross_rate"] == pytest.approx(1 / 3)
    assert s["touch_rate"] == pytest.approx(2 / 3)
    assert s["median_time_to_cross_s"] == 4.0
    assert s["median_spread_bps"] == 30.0
    assert s["median_markout_bps_crossed"] == -10.0


def test_late_finalisation_is_counted():
    rows = [_r("not_crossed", late=0.5), _r("not_crossed", late=12.0)]
    assert summarize(rows)["n_late"] == 1


def test_unknown_feed_state_is_not_treated_as_clean():
    s = summarize([_r("crossed", True, 1.0, 10.0, 0.0, gap=None)])
    assert s["n_measured"] == 0 and s["n_feed_unknown"] == 1


def test_empty_is_none_not_zero():
    s = summarize([_r("no_quote", gap=None, late=None)])
    assert s["n_measured"] == 0
    assert s["cross_rate"] is None and s["median_spread_bps"] is None
    assert s["coverage"] == 0.0


def test_no_rows_has_no_coverage():
    assert summarize([])["coverage"] is None
