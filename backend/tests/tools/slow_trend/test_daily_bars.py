import asyncio

import numpy as np
import pandas as pd
import pytest

from tools.slow_trend import daily_bars as D

DAY = 86400
T0 = 1_700_006_400  # 2023-11-15 00:00 UTC


def _candles(starts):
    return [
        {"start": str(s), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}
        for s in starts
    ]


def _raw(starts, **cols):
    n = len(starts)
    base = {
        "start": starts,
        "open": [1.0] * n,
        "high": [1.0] * n,
        "low": [1.0] * n,
        "close": [1.0] * n,
        "volume": [1.0] * n,
        "page": [0] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def test_fetch_keeps_raw_rows_with_page_numbers():
    async def getter(path, params):
        s, e = int(params["start"]), int(params["end"])
        return {"candles": _candles(range(s - s % DAY, e, DAY))}

    raw = asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 700 * DAY, getter, page_days=300))
    assert len(raw) == 700 and set(raw["page"]) == {0, 1, 2}


def test_fetch_rejects_malformed_response():
    async def getter(path, params):
        return {"candles": None}

    with pytest.raises(ValueError, match="malformed"):
        asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 10 * DAY, getter))


def test_fetch_rejects_candle_missing_fields():
    async def getter(path, params):
        return {"candles": [{"start": str(T0), "open": "1"}]}

    with pytest.raises(ValueError, match="missing"):
        asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 10 * DAY, getter))


def test_audit_counts_missing_and_misaligned():
    raw = _raw([T0, T0 + 2 * DAY + 3600])
    a = D.audit(raw, "2023-11-15", "2023-11-17", 3)
    assert a["misaligned"] == 1 and a["missing_days"] == ["2023-11-16", "2023-11-17"]
    assert a["adequate"] is False


def test_audit_allows_up_to_three_missing():
    starts = [T0 + i * DAY for i in range(10) if i not in (2, 5, 7)]
    a = D.audit(_raw(starts), "2023-11-15", "2023-11-24", 3)
    assert len(a["missing_days"]) == 3 and a["adequate"] is True


def test_audit_flags_conflicting_duplicate():
    raw = _raw([T0, T0], close=[1.0, 2.0], high=[1.0, 2.0])
    a = D.audit(raw, "2023-11-15", "2023-11-15", 3)
    assert a["conflicting_duplicates"] == 1 and a["adequate"] is False


def test_audit_allows_identical_page_boundary_copy():
    raw = _raw([T0, T0], page=[0, 1])
    a = D.audit(raw, "2023-11-15", "2023-11-15", 3)
    assert a["identical_copies"] == 1 and a["conflicting_duplicates"] == 0
    assert a["adequate"] is True


def test_audit_flags_invalid_prices():
    raw = _raw(
        [T0, T0 + DAY, T0 + 2 * DAY, T0 + 3 * DAY],
        close=[np.nan, 1.0, 1.0, 1.0],
        open=[1.0, 0.0, 1.0, 1.0],
        high=[1.0, 1.0, 0.5, np.inf],
    )
    a = D.audit(raw, "2023-11-15", "2023-11-18", 3)
    assert a["invalid_rows"] == 4 and a["adequate"] is False


def test_normalise_collapses_identical_and_rejects_conflict():
    assert len(D.normalise(_raw([T0, T0], page=[0, 1]))) == 1
    with pytest.raises(ValueError, match="conflicting"):
        D.normalise(_raw([T0, T0], close=[1.0, 2.0], high=[1.0, 2.0]))


def test_to_calendar_marks_missing_as_nan():
    cal = D.to_calendar(D.normalise(_raw([T0, T0 + 2 * DAY])), "2023-11-15", "2023-11-17")
    assert list(cal.index.strftime("%Y-%m-%d")) == ["2023-11-15", "2023-11-16", "2023-11-17"]
    assert cal.loc["2023-11-16"].isna().all()


def test_first_day_ignores_misaligned_rows():
    assert D.first_day(_raw([T0 - 3600, T0 + DAY])) == "2023-11-16"


def test_first_day_none_without_aligned_rows():
    assert D.first_day(_raw([T0 + 3600])) is None
    assert D.first_day(_raw([])) is None


def test_product_constraints_require_every_field():
    ok = {"base_increment": "0.00000001", "base_min_size": "0.00000001", "quote_min_size": "1"}
    assert D.product_constraints(ok) == {"base_increment": 1e-8, "base_min": 1e-8, "quote_min": 1.0}
    with pytest.raises(ValueError, match="quote_min_size"):
        D.product_constraints({k: v for k, v in ok.items() if k != "quote_min_size"})
    with pytest.raises(ValueError, match="base_increment"):
        D.product_constraints(dict(ok, base_increment="0"))
    with pytest.raises(ValueError, match="product"):
        D.product_constraints(None)


def test_hourly_overlap_uses_only_complete_days():
    hours = [T0 + h * 3600 for h in range(24 + 5)]
    hourly = pd.DataFrame({"start": hours, "close": [float(h) for h in range(29)]})
    cal = pd.DataFrame(
        {"open": [0.0, 0.0], "close": [23.0, 99.0]},
        index=pd.to_datetime(["2023-11-15", "2023-11-16"]),
    )
    o = D.hourly_overlap(hourly, cal)
    assert o["days_compared"] == 1 and o["max_abs_rel_diff"] == pytest.approx(0.0)
