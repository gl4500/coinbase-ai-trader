"""DDL for the event store.

Event-sourced architecture per docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md.

Append-only events table + per-consumer cursor table + materialized view tables
(materialized_* tables are declared in materialized_schema() so consumers that
don't need them aren't forced to create them).

All DDL is `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` so the
init function is safe to call repeatedly. Tests rely on this idempotency.
"""
from __future__ import annotations

import re
from typing import Iterable

import aiosqlite

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms           INTEGER NOT NULL,
    event_type      TEXT    NOT NULL,
    pid             TEXT,
    payload_json    TEXT    NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    producer        TEXT    NOT NULL,
    write_ts_ms     INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);
"""

_EVENTS_INDICES = (
    "CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(ts_ms);",
    "CREATE INDEX IF NOT EXISTS idx_events_pid_ts   ON events(pid, ts_ms) WHERE pid IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_events_type_ts  ON events(event_type, ts_ms);",
    "CREATE INDEX IF NOT EXISTS idx_events_producer ON events(producer, id);",
)

_CURSORS_DDL = """
CREATE TABLE IF NOT EXISTS consumer_cursors (
    consumer_name      TEXT    PRIMARY KEY,
    last_processed_id  INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);
"""


async def init_events_schema(db: aiosqlite.Connection) -> None:
    """Create events + consumer_cursors tables and indices. Idempotent."""
    await db.execute(_EVENTS_DDL)
    for stmt in _EVENTS_INDICES:
        await db.execute(stmt)
    await db.execute(_CURSORS_DDL)
    await db.commit()


_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")

_MATERIALIZED_LATEST_PRICE_DDL = """
CREATE TABLE IF NOT EXISTS materialized_latest_price (
    pid                 TEXT PRIMARY KEY,
    price               REAL,
    bid                 REAL,
    ask                 REAL,
    pct_change_24h      REAL,
    last_event_id       INTEGER,
    last_updated_ts_ms  INTEGER
);
"""

_MATERIALIZED_POSITIONS_DDL_TMPL = """
CREATE TABLE IF NOT EXISTS materialized_positions_{model} (
    pid               TEXT PRIMARY KEY,
    size              REAL,
    avg_price         REAL,
    position_dollars  REAL,
    entry_time_ms     INTEGER,
    peak_price        REAL,
    peak_pnl_pct      REAL,
    last_event_id     INTEGER
);
"""


async def init_materialized_schema(
    db: aiosqlite.Connection,
    model_names: Iterable[str],
) -> None:
    """Create materialized_latest_price + per-model materialized_positions_* tables.

    model_names is an iterable of safe identifiers (e.g. 'v3', 'v4_5'). Anything
    not matching ^[a-zA-Z0-9_]+$ is rejected to keep the DDL safe — the table
    name is interpolated into a CREATE TABLE so we cannot use parameter binding.
    """
    await db.execute(_MATERIALIZED_LATEST_PRICE_DDL)
    for model in model_names:
        if not _MODEL_NAME_RE.match(model):
            raise ValueError(
                f"model_name {model!r} contains non-identifier characters"
            )
        await db.execute(_MATERIALIZED_POSITIONS_DDL_TMPL.format(model=model))
    await db.commit()
