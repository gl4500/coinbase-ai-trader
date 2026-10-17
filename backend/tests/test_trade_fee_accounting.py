"""Net-of-fee paper accounting on the trades table.

Paper fills never paid fees, so gross `pnl` overstates results. New closes record the assumed
per-side rate, the dollar fee and `pnl_net`; gross `pnl` and `balance_after` are unchanged, so no
trading behaviour moves. Historical rows stay NULL (no retroactive attribution).
"""

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


def _open_and_close(db, pnl=10.0, **close_kw):
    # entry 100 x 1.0 = $100 open; exit 110 x 1.0 = $110 close; gross pnl +$10
    _run(db.open_trade("CNN", "ABC-USD", 100.0, 1.0, 100.0, "SCAN", 900.0))
    _run(db.close_trade("CNN", "ABC-USD", 110.0, 1.0, pnl, "TICK_TRAIL", 1010.0, **close_kw))
    return _run(db.get_trades(agent="CNN", limit=10))[0]


def test_close_records_taker_fee_and_net_pnl_by_default(db):
    row = _open_and_close(db)
    assert row["fee_rate"] == 0.009
    assert row["fee_usd"] == pytest.approx(0.90 + 0.99)
    assert row["pnl_net"] == pytest.approx(10.0 - 1.89)


def test_gross_pnl_and_balance_are_unchanged(db):
    row = _open_and_close(db)
    assert row["pnl"] == pytest.approx(10.0)
    assert row["balance_after"] == pytest.approx(1010.0)


def test_explicit_rate_is_recorded(db):
    row = _open_and_close(db, fee_rate=0.005)
    assert row["fee_rate"] == 0.005
    assert row["fee_usd"] == pytest.approx(0.50 + 0.55)


def test_fee_uses_the_stored_open_notional_not_the_close(db):
    # A losing trade: open $100, close $80 -> fee 0.90 + 0.72
    _run(db.open_trade("CNN", "ABC-USD", 100.0, 1.0, 100.0, "SCAN", 900.0))
    _run(db.close_trade("CNN", "ABC-USD", 80.0, 1.0, -20.0, "TICK_STOP", 880.0))
    row = _run(db.get_trades(agent="CNN", limit=1))[0]
    assert row["fee_usd"] == pytest.approx(1.62)
    assert row["pnl_net"] == pytest.approx(-21.62)


def test_close_without_open_row_charges_both_sides_on_close_notional(db):
    # Restart path inserts a synthetic closed row with usd_open == usd_close.
    _run(db.close_trade("CNN", "XYZ-USD", 50.0, 2.0, 0.0, "RECONCILE", 1000.0))
    row = _run(db.get_trades(agent="CNN", limit=1))[0]
    assert row["fee_usd"] == pytest.approx(2 * 100.0 * 0.009)
    assert row["pnl_net"] == pytest.approx(-1.8)


def test_historical_rows_without_fee_data_stay_null(db):
    # Simulate a pre-migration closed trade: written without the new columns.
    async def legacy():
        async with db._db() as conn:
            await conn.execute(
                """INSERT INTO trades (agent, product_id, entry_price, exit_price, size,
                   usd_open, usd_close, pnl, pct_pnl, trigger_open, trigger_close,
                   balance_after, opened_at, closed_at)
                   VALUES ('CNN','OLD-USD',1,1,1,1,1,0.5,50,'SCAN','X',1,
                           '2026-01-01T00:00:00+00:00','2026-01-01T01:00:00+00:00')"""
            )
            await conn.commit()

    _run(legacy())
    row = _run(db.get_trades(agent="CNN", limit=1))[0]
    assert row["fee_rate"] is None and row["fee_usd"] is None and row["pnl_net"] is None


def test_fee_summary_reports_net_only_over_fee_bearing_rows(db):
    _open_and_close(db)  # gross +10, fee 1.89, net 8.11

    async def legacy():
        async with db._db() as conn:
            await conn.execute(
                """INSERT INTO trades (agent, product_id, entry_price, exit_price, size,
                   usd_open, usd_close, pnl, pct_pnl, trigger_open, trigger_close,
                   balance_after, opened_at, closed_at)
                   VALUES ('CNN','OLD-USD',1,1,1,1,1,5.0,0,'SCAN','X',1,
                           '2026-01-01T00:00:00+00:00','2026-01-01T01:00:00+00:00')"""
            )
            await conn.commit()

    _run(legacy())
    s = _run(db.get_fee_summary())
    assert s["closed_trades"] == 2
    assert s["fee_bearing_trades"] == 1
    assert s["gross_pnl_fee_bearing"] == pytest.approx(10.0)
    assert s["fee_usd"] == pytest.approx(1.89)
    assert s["net_pnl"] == pytest.approx(8.11)


def test_fee_summary_since_filters_by_close_time(db):
    _open_and_close(db)
    s = _run(db.get_fee_summary(since="2999-01-01T00:00:00+00:00"))
    assert s["closed_trades"] == 0
    assert s["net_pnl"] == 0.0


def test_close_rejects_impossible_rate(db):
    _run(db.open_trade("CNN", "ABC-USD", 100.0, 1.0, 100.0, "SCAN", 900.0))
    with pytest.raises(ValueError):
        _run(db.close_trade("CNN", "ABC-USD", 110.0, 1.0, 10.0, "T", 1010.0, fee_rate=-0.01))


@pytest.mark.asyncio
async def test_performance_endpoint_reports_net_beside_unchanged_gross(tmp_path, monkeypatch):
    import database
    import main

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "perf.db"))
    await database.init_db()
    await database.open_trade("CNN", "ABC-USD", 100.0, 1.0, 100.0, "SCAN", 900.0)
    await database.close_trade("CNN", "ABC-USD", 110.0, 1.0, 10.0, "TICK_TRAIL", 1010.0)

    perf = await main.get_performance()

    assert perf["overall"]["total_pnl"] == pytest.approx(10.0)
    assert perf["overall"]["net_of_fees"]["net_pnl"] == pytest.approx(8.11)
    assert perf["rolling_30d"]["net_of_fees"]["fee_usd"] == pytest.approx(1.89)
