"""Tests for typed event INSERT helpers."""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema


@pytest.fixture
async def db(tmp_path):
    path = str(tmp_path / "test.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        yield conn


@pytest.mark.asyncio
async def test_write_price_tick_inserts_row(db):
    payload = et.PriceTickPayload(
        pid="BTC-USD", price=100.0, bid=99.9, ask=100.1,
        volume_24h=1.0, source="ws",
    )
    eid = await event_writer.write_price_tick(
        db, producer="ingest", ts_ms=1700000000000, payload=payload,
    )
    cur = await db.execute(
        "SELECT event_type, pid, payload_json, producer FROM events WHERE id = ?",
        (eid,),
    )
    row = await cur.fetchone()
    assert row[0] == "price_tick"
    assert row[1] == "BTC-USD"
    assert json.loads(row[2])["price"] == 100.0
    assert row[3] == "ingest"


@pytest.mark.asyncio
async def test_write_signal_scored_routes_to_correct_event_type(db):
    payload = et.SignalScoredPayload(
        pid="ETH-USD", model="v4_5", model_version="h168_2026-05-23",
        feature_hash="x", scores={"p_up": 0.5, "p_down": 0.3, "p_neutral": 0.2},
        side="BUY", strength=0.5, regime=None, deployment_profile_id=None,
        input_event_ids={"last_price_tick_id": 1},
    )
    eid = await event_writer.write_signal_scored(
        db, producer="model_v4_5", ts_ms=1700000000000, payload=payload,
    )
    cur = await db.execute("SELECT event_type FROM events WHERE id = ?", (eid,))
    row = await cur.fetchone()
    assert row[0] == "signal_scored"


@pytest.mark.asyncio
async def test_event_id_is_monotonic(db):
    p1 = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
    p2 = et.PriceTickPayload(pid="A", price=2, bid=None, ask=None, volume_24h=None, source="ws")
    id1 = await event_writer.write_price_tick(db, producer="ingest", ts_ms=1, payload=p1)
    id2 = await event_writer.write_price_tick(db, producer="ingest", ts_ms=2, payload=p2)
    assert id2 > id1


@pytest.mark.asyncio
async def test_pid_column_populated_when_payload_has_pid(db):
    payload = et.PriceTickPayload(pid="SOL-USD", price=1, bid=None, ask=None, volume_24h=None, source="ws")
    eid = await event_writer.write_price_tick(db, producer="ingest", ts_ms=1, payload=payload)
    cur = await db.execute("SELECT pid FROM events WHERE id = ?", (eid,))
    row = await cur.fetchone()
    assert row[0] == "SOL-USD"
