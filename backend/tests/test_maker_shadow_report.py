import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.maker_shadow_report import summarize


def _r(status, touched=False, ttf=None, spread=None, drift=None):
    return {
        "status": status,
        "touched": touched,
        "time_to_fill_s": ttf,
        "spread_bps": spread,
        "drift_bps": drift,
    }


def test_rates_use_measured_denominator_only():
    rows = [
        _r("filled", True, 4.0, 20.0, -10.0),
        _r("unfilled", True, None, 30.0),
        _r("unfilled", False, None, 40.0),
        _r("no_quote"),
        _r("duplicate"),
    ]
    s = summarize(rows)
    assert s["n_total"] == 5 and s["n_measured"] == 3
    assert s["n_no_quote"] == 1 and s["n_duplicate"] == 1
    assert s["fill_rate"] == pytest.approx(1 / 3)
    assert s["touch_rate"] == pytest.approx(2 / 3)
    assert s["median_time_to_fill_s"] == 4.0
    assert s["median_spread_bps"] == 30.0
    assert s["median_drift_bps_filled"] == -10.0


def test_empty_is_none_not_zero():
    s = summarize([_r("no_quote")])
    assert s["n_measured"] == 0
    assert s["fill_rate"] is None and s["median_spread_bps"] is None
