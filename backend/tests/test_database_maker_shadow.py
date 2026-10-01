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
        "status": "filled",
        "touched": True,
        "limit_price": 10.0,
        "ask": 10.1,
        "spread_bps": 99.5,
        "created_ts": 1000.0,
        "fill_ts": 1005.0,
        "time_to_fill_s": 5.0,
        "last_price": 10.05,
        "drift_bps": 50.0,
        "window_s": 30.0,
        "detail": None,
    }
    base.update(kw)
    return base


def test_round_trip_preserves_every_field(db):
    _run(db.save_maker_shadow(_row()))
    rows = _run(db.get_maker_shadow_rows())
    assert len(rows) == 1
    r = rows[0]
    for k, v in _row().items():
        assert r[k] == v, k


def test_no_quote_row_with_nulls_round_trips(db):
    _run(
        db.save_maker_shadow(
            _row(
                status="no_quote",
                touched=False,
                limit_price=None,
                ask=None,
                spread_bps=None,
                fill_ts=None,
                time_to_fill_s=None,
                last_price=None,
                drift_bps=None,
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
