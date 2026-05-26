"""Tests for EventConsumer — cursor-based polling base class."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.event_consumer import EventConsumer
from services.events_schema import init_events_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
    return path


@pytest.mark.asyncio
async def test_new_consumer_starts_at_max_id(db_path):
    """A first-time consumer skips history by default — starts at current max(id)."""
    async with aiosqlite.connect(db_path) as conn:
        p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
        await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)

    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        events = await consumer.poll()
        assert events == []          # past events skipped
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_consumer_reads_events_after_start(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        events = await consumer.poll()
        assert len(events) == 1
        assert events[0].event_type == "price_tick"
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_commit_advances_cursor(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
            eid = await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        events = await consumer.poll()
        await consumer.commit(events[-1].id)
        events_after = await consumer.poll()
        assert events_after == []
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_cursor_persists_across_restart(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    async with aiosqlite.connect(db_path) as conn:
        p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
        eid = await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
    events = await consumer.poll()
    await consumer.commit(events[-1].id)
    await consumer.stop()

    revived = EventConsumer(db_path, name="model_v3")
    await revived.start()
    try:
        again = await revived.poll()
        assert again == []
    finally:
        await revived.stop()


@pytest.mark.asyncio
async def test_batch_size_limits_returned_events(db_path):
    consumer = EventConsumer(db_path, name="model_v3", batch_size=3)
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            for i in range(10):
                p = et.PriceTickPayload(pid="A", price=i, bid=None, ask=None, volume_24h=None, source="ws")
                await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p)
        events = await consumer.poll()
        assert len(events) == 3
    finally:
        await consumer.stop()
