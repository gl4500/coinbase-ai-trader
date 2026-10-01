"""Tests for services.maker_shadow — paper post-only fill measurement."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.maker_shadow import MakerShadow


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _make(window_s=30.0):
    rows = []

    async def sink(row):
        rows.append(row)

    clock = _Clock()
    return MakerShadow(sink=sink, clock=clock, window_s=window_s), clock, rows


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_trade_through_inside_window_is_filled():
    shadow, clock, rows = _make()
    assert shadow.register("ABC-USD", bid=10.0, ask=10.1) == "open"
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 10.05))
    assert len(rows) == 1
    r = rows[0]
    assert r["status"] == "filled" and r["touched"] is True
    assert r["time_to_fill_s"] == pytest.approx(5.0)
    assert r["last_price"] == pytest.approx(10.05)
    assert r["drift_bps"] == pytest.approx((10.05 - 10.0) / 10.0 * 1e4)
    assert r["spread_bps"] == pytest.approx(0.1 / 10.05 * 1e4)


def test_touch_is_not_a_fill():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 3
    _run(shadow.on_tick("ABC-USD", 10.0))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 10.02))
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["touched"] is True
    assert rows[0]["fill_ts"] is None


def test_trade_through_after_window_does_not_count():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 9.5))
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["touched"] is False


def test_intent_finalised_by_other_products_tick():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 40
    _run(shadow.on_tick("XYZ-USD", 1.0))
    assert len(rows) == 1 and rows[0]["product_id"] == "ABC-USD"
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["last_price"] is None
    assert rows[0]["drift_bps"] is None
    assert shadow.open_count() == 0


@pytest.mark.parametrize(
    "bid,ask,detail",
    [
        (None, 10.1, "missing quote"),
        (10.0, None, "missing quote"),
        (0.0, 10.1, "non-positive bid"),
        (10.2, 10.1, "crossed quote"),
    ],
)
def test_bad_quotes_recorded_as_no_quote(bid, ask, detail):
    shadow, clock, rows = _make()
    assert shadow.register("ABC-USD", bid=bid, ask=ask) == "no_quote"
    _run(shadow.on_tick("ABC-USD", 10.0))
    assert rows[0]["status"] == "no_quote"
    assert rows[0]["detail"] == detail
    assert shadow.open_count() == 0


def test_second_buy_while_open_is_duplicate_and_keeps_first():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    assert shadow.register("ABC-USD", bid=11.0, ask=11.1) == "duplicate"
    clock.t += 1
    _run(shadow.on_tick("ABC-USD", 9.9))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 9.9))
    statuses = sorted(r["status"] for r in rows)
    assert statuses == ["duplicate", "filled"]
    filled = next(r for r in rows if r["status"] == "filled")
    assert filled["limit_price"] == 10.0


def test_sink_failure_is_swallowed():
    async def bad_sink(row):
        raise RuntimeError("database is locked")

    clock = _Clock()
    shadow = MakerShadow(sink=bad_sink, clock=clock)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 10.0))  # must not raise
    assert shadow.open_count() == 0
