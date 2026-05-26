"""EventConsumer — cursor-based polling primitive for the event store.

Each consumer (a model service, the view materializer, the WS bridge) wraps an
EventConsumer instance.

Discipline:
  - poll() returns ordered events with id > cursor, up to batch_size
  - commit(event_id) advances the cursor to event_id
  - Consumers MUST be idempotent — at-least-once delivery: a consumer that
    crashes after processing but before commit() will re-process the same event
    on restart.
  - First-time consumers start at MAX(id) — they do NOT replay history on first
    boot. Backtest replay tooling uses a separate cursor-reset path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import aiosqlite


@dataclass(frozen=True)
class Event:
    id: int
    ts_ms: int
    event_type: str
    pid: Optional[str]
    payload_json: str
    schema_version: int
    producer: str


class EventConsumer:
    """Cursor-based event reader. Open/close pattern via start()/stop()."""

    def __init__(self, db_path: str, *, name: str, batch_size: int = 1000):
        self._db_path = db_path
        self._name = name
        self._batch_size = batch_size
        self._db: Optional[aiosqlite.Connection] = None
        self._cursor_id: int = 0

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")
        self._cursor_id = await self._load_or_init_cursor()

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _load_or_init_cursor(self) -> int:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT last_processed_id FROM consumer_cursors WHERE consumer_name = ?",
            (self._name,),
        )
        row = await cur.fetchone()
        if row is not None:
            return row[0]

        cur = await self._db.execute("SELECT COALESCE(MAX(id), 0) FROM events")
        max_row = await cur.fetchone()
        max_id = max_row[0] if max_row else 0
        await self._db.execute(
            "INSERT INTO consumer_cursors (consumer_name, last_processed_id) VALUES (?, ?)",
            (self._name, max_id),
        )
        await self._db.commit()
        return max_id

    async def poll(self) -> List[Event]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT id, ts_ms, event_type, pid, payload_json, schema_version, producer "
            "FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (self._cursor_id, self._batch_size),
        )
        rows = await cur.fetchall()
        return [Event(*r) for r in rows]

    async def commit(self, event_id: int) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE consumer_cursors SET last_processed_id = ?, "
            "updated_at = CAST(strftime('%s','now') AS INTEGER) * 1000 "
            "WHERE consumer_name = ?",
            (event_id, self._name),
        )
        await self._db.commit()
        self._cursor_id = event_id
