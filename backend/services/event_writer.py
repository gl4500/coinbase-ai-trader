"""Async typed event INSERT helpers.

One write_<event_type>() coroutine per event type. Each one:
  - Takes the typed payload dataclass
  - Serializes it to JSON
  - Inserts into the events table
  - Returns the new event id

The DB connection is passed in — the writer is agnostic about how the caller
opened it. This lets the same writer be used by the ingest worker (its own
connection) and the model service (its own connection); concurrent writers
are safe because SQLite WAL allows multiple writers serialized by the
journal manager, and writes are tiny (single-row inserts).
"""
from __future__ import annotations

import json
from typing import Optional

import aiosqlite

from services import event_types as et


async def _insert(
    db: aiosqlite.Connection,
    *,
    event_type: str,
    producer: str,
    ts_ms: int,
    pid: Optional[str],
    payload_dict: dict,
) -> int:
    cur = await db.execute(
        "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (ts_ms, event_type, pid, json.dumps(payload_dict, separators=(",", ":")), producer),
    )
    await db.commit()
    return cur.lastrowid


async def write_price_tick(db, *, producer, ts_ms, payload: et.PriceTickPayload) -> int:
    return await _insert(db, event_type="price_tick", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_candle_close(db, *, producer, ts_ms, payload: et.CandleClosePayload) -> int:
    return await _insert(db, event_type="candle_close", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_marketcap_snapshot(db, *, producer, ts_ms, payload: et.MarketcapSnapshotPayload) -> int:
    return await _insert(db, event_type="marketcap_snapshot", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_signal_scored(db, *, producer, ts_ms, payload: et.SignalScoredPayload) -> int:
    return await _insert(db, event_type="signal_scored", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_trade_decided(db, *, producer, ts_ms, payload: et.TradeDecidedPayload) -> int:
    return await _insert(db, event_type="trade_decided", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_trade_closed(db, *, producer, ts_ms, payload: et.TradeClosedPayload) -> int:
    return await _insert(db, event_type="trade_closed", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_exit_triggered(db, *, producer, ts_ms, payload: et.ExitTriggeredPayload) -> int:
    return await _insert(db, event_type="exit_triggered", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())
