"""Tests for ViewMaterializer — keeps materialized_* tables up to date from events."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.view_materializer import ViewMaterializer


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


@pytest.mark.asyncio
async def test_price_tick_updates_latest_price(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            payload = et.PriceTickPayload(
                pid="BTC-USD", price=95000.0, bid=94999.0, ask=95001.0,
                volume_24h=100.0, source="ws",
            )
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1_700_000_000_000, payload=payload)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT price, bid, ask FROM materialized_latest_price WHERE pid = ?",
                ("BTC-USD",),
            )
            row = await cur.fetchone()
        assert row == (95000.0, 94999.0, 95001.0)
    finally:
        await mat.stop()


@pytest.mark.asyncio
async def test_trade_decided_updates_positions_for_correct_model(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={"p_up": 0.6}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(
                conn, producer="model_v3", ts_ms=1, payload=sig,
            )
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000.0, actual_entry_price=95000.0,
                fee_paid=0.57, trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_1",
            )
            await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=2, payload=td)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT size, avg_price FROM materialized_positions_v3 WHERE pid = ?",
                ("BTC-USD",),
            )
            row = await cur.fetchone()
        assert row == (0.001, 95000.0)
    finally:
        await mat.stop()


@pytest.mark.asyncio
async def test_trade_closed_clears_position(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(
                conn, producer="model_v3", ts_ms=1, payload=sig,
            )
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000.0, actual_entry_price=95000.0,
                fee_paid=0.57, trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_1",
            )
            dec_id = await event_writer.write_trade_decided(
                conn, producer="model_v3", ts_ms=2, payload=td,
            )
            tc = et.TradeClosedPayload(
                pid="BTC-USD", trade_uid="v3_BTC-USD_1", exit_price=96000.0,
                exit_size=0.001, pnl=1.0, pct_pnl=1.05, hold_secs=3600,
                trigger_close="WS_TRAIL_STOP", decision_event_id=dec_id,
                exit_signal_event_id=None,
            )
            await event_writer.write_trade_closed(conn, producer="model_v3", ts_ms=3, payload=tc)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM materialized_positions_v3 WHERE pid = ?",
                ("BTC-USD",),
            )
            (n,) = await cur.fetchone()
        assert n == 0
    finally:
        await mat.stop()
