"""Tests for events_schema — DDL for the event store."""
import aiosqlite
import pytest

from services import events_schema


@pytest.mark.asyncio
async def test_init_creates_events_table(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        row = await cur.fetchone()
    assert row is not None, "events table not created"


@pytest.mark.asyncio
async def test_init_creates_consumer_cursors_table(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consumer_cursors'"
        )
        row = await cur.fetchone()
    assert row is not None, "consumer_cursors table not created"


@pytest.mark.asyncio
async def test_init_is_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_events_schema(db)  # second call must not raise


@pytest.mark.asyncio
async def test_events_indices_exist(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
        )
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert "idx_events_ts" in names
    assert "idx_events_pid_ts" in names
    assert "idx_events_type_ts" in names
    assert "idx_events_producer" in names


@pytest.mark.asyncio
async def test_init_materialized_creates_latest_price(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_materialized_schema(db, model_names=["v3", "v4_5"])
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='materialized_latest_price'"
        )
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_init_materialized_creates_per_model_positions(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_materialized_schema(db, model_names=["v3", "v4_5"])
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'materialized_positions_%'"
        )
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert names == {"materialized_positions_v3", "materialized_positions_v4_5"}


@pytest.mark.asyncio
async def test_init_materialized_rejects_bad_model_name(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        with pytest.raises(ValueError, match="model_name"):
            await events_schema.init_materialized_schema(db, model_names=["v3; DROP TABLE events--"])
