import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def db(tmp_path):
    import database

    importlib.reload(database)
    database.DB_PATH = str(tmp_path / "t.db")
    asyncio.new_event_loop().run_until_complete(database.init_db())
    return database


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _row(**kw):
    base = {
        "product_id": "ABC-USD",
        "status": "crossed",
        "touched": True,
        "limit_price": 10.0,
        "ask": 10.1,
        "spread_bps": 99.5,
        "created_ts": 1000.0,
        "cross_ts": 1005.0,
        "time_to_cross_s": 5.0,
        "window_close_price": 10.02,
        "markout_s": 60.0,
        "markout_bps": -10.0,
        "mark_age_s": 20.0,
        "finalised_late_s": 0.5,
        "feed_gap": False,
        "window_s": 30.0,
        "detail": None,
    }
    base.update(kw)
    return base


def test_round_trip_preserves_every_field(db):
    _run(db.save_maker_shadow(_row()))
    rows = _run(db.get_maker_shadow_rows())
    assert len(rows) == 1
    for k, v in _row().items():
        assert rows[0][k] == v, k


@pytest.mark.parametrize("gap", [None, True, False])
def test_feed_gap_three_states_round_trip(db, gap):
    _run(db.save_maker_shadow(_row(feed_gap=gap)))
    assert _run(db.get_maker_shadow_rows())[0]["feed_gap"] is gap


def test_no_quote_row_with_nulls_round_trips(db):
    _run(
        db.save_maker_shadow(
            _row(
                status="no_quote",
                touched=False,
                limit_price=None,
                ask=None,
                spread_bps=None,
                cross_ts=None,
                time_to_cross_s=None,
                window_close_price=None,
                markout_bps=None,
                mark_age_s=None,
                finalised_late_s=None,
                feed_gap=None,
                detail="missing quote",
            )
        )
    )
    r = _run(db.get_maker_shadow_rows())[0]
    assert r["status"] == "no_quote" and r["touched"] is False and r["limit_price"] is None


def test_since_filter(db):
    _run(db.save_maker_shadow(_row(created_ts=100.0)))
    _run(db.save_maker_shadow(_row(created_ts=200.0)))
    assert [r["created_ts"] for r in _run(db.get_maker_shadow_rows(since_ts=150.0))] == [200.0]


def test_shadow_rows_round_trip_through_database(db):
    """End to end: a real MakerShadow writing through save_maker_shadow."""
    from services.maker_shadow import MakerShadow

    t = [1000.0]
    shadow = MakerShadow(sink=db.save_maker_shadow, clock=lambda: t[0])
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    t[0] += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    t[0] += 70
    _run(shadow.sweep())
    rows = _run(db.get_maker_shadow_rows())
    assert [r["status"] for r in rows] == ["crossed"]
    assert rows[0]["feed_gap"] is False
