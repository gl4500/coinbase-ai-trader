"""Tests for services.maker_shadow — paper post-only crossing measurement."""

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


class _Epoch:
    def __init__(self):
        self.n = 0

    def __call__(self):
        return self.n


def _make(window_s=30.0, markout_s=60.0):
    rows = []

    async def sink(row):
        rows.append(row)

    clock = _Clock()
    epoch = _Epoch()
    shadow = MakerShadow(
        sink=sink, clock=clock, window_s=window_s, markout_s=markout_s, feed_epoch=epoch
    )
    return shadow, clock, rows, epoch


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ── crossing (renamed from "fill": a price-path proxy, not a fill) ─────────────


def test_trade_through_inside_window_is_crossed():
    shadow, clock, rows, _ = _make()
    assert shadow.register("ABC-USD", bid=10.0, ask=10.1) == "open"
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    clock.t += 70
    _run(shadow.sweep())
    r = rows[0]
    assert r["status"] == "crossed" and r["touched"] is True
    assert r["time_to_cross_s"] == pytest.approx(5.0)
    assert r["spread_bps"] == pytest.approx(0.1 / 10.05 * 1e4)


def test_touch_is_not_a_crossing():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 3
    _run(shadow.on_tick("ABC-USD", 10.0))
    clock.t += 30
    _run(shadow.sweep())
    assert rows[0]["status"] == "not_crossed"
    assert rows[0]["touched"] is True
    assert rows[0]["cross_ts"] is None
    assert rows[0]["markout_bps"] is None


def test_trade_through_after_window_does_not_count():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 9.5))
    assert rows[0]["status"] == "not_crossed"
    assert rows[0]["touched"] is False
    assert rows[0]["window_close_price"] is None


# ── fixed-horizon own-product markout (Codex P1 #1) ────────────────────────────


def test_markout_is_last_own_trade_at_fixed_horizon_from_cross():
    shadow, clock, rows, _ = _make(markout_s=60.0)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))  # cross at t=1005
    clock.t += 40
    _run(shadow.on_tick("ABC-USD", 9.90))  # t=1045, inside horizon (<=1065)
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 11.0))  # t=1075, after horizon: ignored
    r = rows[0]
    assert r["markout_s"] == 60.0
    assert r["markout_bps"] == pytest.approx((9.90 - 10.0) / 10.0 * 1e4)
    assert r["mark_age_s"] == pytest.approx(1065.0 - 1045.0)


def test_other_products_tick_never_sets_the_mark():
    shadow, clock, rows, _ = _make(markout_s=60.0)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    clock.t += 100
    _run(shadow.on_tick("XYZ-USD", 123.0))
    r = rows[0]
    assert r["markout_bps"] == pytest.approx((9.99 - 10.0) / 10.0 * 1e4)
    assert r["mark_age_s"] == pytest.approx(60.0)


def test_post_deadline_own_tick_does_not_overwrite_window_close_price():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 10
    _run(shadow.on_tick("ABC-USD", 10.05))
    clock.t += 25
    _run(shadow.on_tick("ABC-USD", 12.0))  # after the 30 s window
    assert rows[0]["window_close_price"] == pytest.approx(10.05)


# ── deadline-driven finalisation (Codex P1 #2) ─────────────────────────────────


def test_sweep_finalises_without_any_tick():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.sweep())
    assert len(rows) == 1 and rows[0]["status"] == "not_crossed"
    assert rows[0]["finalised_late_s"] == pytest.approx(1.0)
    assert shadow.open_count() == 0


def test_crossed_intent_stays_open_until_markout_horizon():
    shadow, clock, rows, _ = _make(markout_s=60.0)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    clock.t += 40  # past the 30 s window, before cross+60
    _run(shadow.sweep())
    assert rows == [] and shadow.open_count() == 1
    clock.t += 21
    _run(shadow.sweep())
    assert len(rows) == 1


def test_expired_intent_is_not_counted_as_duplicate():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31  # no tick, no sweep
    assert shadow.register("ABC-USD", bid=11.0, ask=11.1) == "open"
    _run(shadow.sweep())
    assert [r["status"] for r in rows] == ["not_crossed"]
    assert shadow.open_count() == 1


def test_second_buy_while_open_is_duplicate_and_keeps_first():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    assert shadow.register("ABC-USD", bid=11.0, ask=11.1) == "duplicate"
    clock.t += 1
    _run(shadow.on_tick("ABC-USD", 9.9))
    clock.t += 70
    _run(shadow.sweep())
    statuses = sorted(r["status"] for r in rows)
    assert statuses == ["crossed", "duplicate"]
    crossed = next(r for r in rows if r["status"] == "crossed")
    assert crossed["limit_price"] == 10.0


# ── feed coverage (Codex P1 #5) ────────────────────────────────────────────────


def test_reconnect_during_observation_sets_feed_gap():
    shadow, clock, rows, epoch = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    epoch.n += 1
    clock.t += 31
    _run(shadow.sweep())
    assert rows[0]["feed_gap"] is True


def test_no_reconnect_means_no_feed_gap():
    shadow, clock, rows, _ = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.sweep())
    assert rows[0]["feed_gap"] is False


# ── unmeasurable intents and isolation ─────────────────────────────────────────


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
    shadow, clock, rows, _ = _make()
    assert shadow.register("ABC-USD", bid=bid, ask=ask) == "no_quote"
    _run(shadow.sweep())
    assert rows[0]["status"] == "no_quote"
    assert rows[0]["detail"] == detail
    assert shadow.open_count() == 0


def test_sink_failure_is_swallowed():
    async def bad_sink(row):
        raise RuntimeError("database is locked")

    clock = _Clock()
    shadow = MakerShadow(sink=bad_sink, clock=clock)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 10.0))  # must not raise
    assert shadow.open_count() == 0


def test_feed_epoch_failure_is_swallowed():
    def bad_epoch():
        raise RuntimeError("ws gone")

    rows = []

    async def sink(row):
        rows.append(row)

    clock = _Clock()
    shadow = MakerShadow(sink=sink, clock=clock, feed_epoch=bad_epoch)
    assert shadow.register("ABC-USD", bid=10.0, ask=10.1) == "open"
    clock.t += 31
    _run(shadow.sweep())  # must not raise
    assert rows[0]["feed_gap"] is None
