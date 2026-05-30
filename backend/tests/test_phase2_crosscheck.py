"""Phase 2 cross-check: ensure model_service's signal_scored output for a
deterministic synthetic feed is what we expect.

This is the unit-level analog of the operator-run Phase 2 cross-check that
compares model_service vs monolith on real events for 48+ hours.
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService


@pytest.mark.asyncio
async def test_no_signal_with_under_60_candles(tmp_path):
    db = str(tmp_path / "events.db")
    async with aiosqlite.connect(db) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3"])
        for i in range(10):
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=1, high=2, low=0.5, close=1.5, volume=10,
                bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=cc.bar_ts_ms, payload=cc)
    svc = ModelService(db_path=db, model_name="v3")
    await svc.start()
    try:
        while True:
            n = await svc.tick()
            if n == 0:
                break
    finally:
        await svc.stop()
    async with aiosqlite.connect(db) as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='signal_scored'"
        )
        (n,) = await cur.fetchone()
    assert n == 0
