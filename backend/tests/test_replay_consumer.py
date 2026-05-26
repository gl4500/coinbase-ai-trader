"""Tests for replay_consumer — sandboxed event-stream replay."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema
from tools.replay_consumer import replay_into_sandbox


@pytest.mark.asyncio
async def test_replay_copies_events_into_sandbox(tmp_path):
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        for i in range(5):
            p = et.PriceTickPayload(pid="A", price=float(i), bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p)
    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=0, until_event=None)
    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 5


@pytest.mark.asyncio
async def test_replay_respects_event_id_range(tmp_path):
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        ids = []
        for i in range(10):
            p = et.PriceTickPayload(pid="A", price=float(i), bid=None, ask=None,
                                    volume_24h=None, source="ws")
            ids.append(await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p))
    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=ids[2], until_event=ids[7])
    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 6      # ids 3..7 inclusive
