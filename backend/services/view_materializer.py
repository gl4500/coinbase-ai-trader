"""View materializer — keeps materialized_* tables in sync with events.

Polls events via EventConsumer (its own cursor name). Routes by event_type:
  price_tick           → UPSERT materialized_latest_price
  trade_decided        → UPSERT materialized_positions_<model>
  trade_closed         → DELETE FROM materialized_positions_<model>
  exit_triggered       → no view update (audit-only)
  signal_scored        → no view update (audit-only)
  candle_close         → no view update (feature_snapshot reads events directly)
  marketcap_snapshot   → no view update (queried from events on demand)

Reuses EventConsumer's cursor semantics so view freshness survives restart.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import aiosqlite

from services.event_consumer import Event, EventConsumer

logger = logging.getLogger(__name__)

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class ViewMaterializer:
    """Subscribes to the event stream and updates materialized_* tables."""

    def __init__(self, db_path: str, *, name: str = "view_materializer", batch_size: int = 1000):
        self._db_path = db_path
        self._consumer = EventConsumer(db_path, name=name, batch_size=batch_size)
        self._write_db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        await self._consumer.start()
        self._write_db = await aiosqlite.connect(self._db_path)
        await self._write_db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._write_db is not None:
            await self._write_db.close()
            self._write_db = None
        await self._consumer.stop()

    async def tick(self) -> int:
        """Process one batch of events. Returns count processed."""
        events = await self._consumer.poll()
        if not events:
            return 0
        for evt in events:
            try:
                await self._apply(evt)
            except Exception:
                logger.exception("view_materializer: failed to apply event id=%s type=%s",
                                 evt.id, evt.event_type)
        await self._consumer.commit(events[-1].id)
        return len(events)

    async def _apply(self, evt: Event) -> None:
        assert self._write_db is not None
        if evt.event_type == "price_tick":
            await self._apply_price_tick(evt)
        elif evt.event_type == "trade_decided":
            await self._apply_trade_decided(evt)
        elif evt.event_type == "trade_closed":
            await self._apply_trade_closed(evt)

    async def _apply_price_tick(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        await self._write_db.execute(
            "INSERT INTO materialized_latest_price "
            "(pid, price, bid, ask, pct_change_24h, last_event_id, last_updated_ts_ms) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?) "
            "ON CONFLICT(pid) DO UPDATE SET "
            "  price=excluded.price, bid=excluded.bid, ask=excluded.ask, "
            "  last_event_id=excluded.last_event_id, "
            "  last_updated_ts_ms=excluded.last_updated_ts_ms",
            (evt.pid, payload.get("price"), payload.get("bid"), payload.get("ask"),
             evt.id, evt.ts_ms),
        )
        await self._write_db.commit()

    async def _apply_trade_decided(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        model = payload["model"]
        if not _MODEL_NAME_RE.match(model):
            logger.error("Skipping trade_decided with unsafe model name: %r", model)
            return
        table = f"materialized_positions_{model}"
        await self._write_db.execute(
            f"INSERT INTO {table} "
            f"(pid, size, avg_price, position_dollars, entry_time_ms, peak_price, peak_pnl_pct, last_event_id) "
            f"VALUES (?, ?, ?, ?, ?, ?, 0.0, ?) "
            f"ON CONFLICT(pid) DO UPDATE SET "
            f"  size=excluded.size, avg_price=excluded.avg_price, "
            f"  position_dollars=excluded.position_dollars, "
            f"  entry_time_ms=excluded.entry_time_ms, peak_price=excluded.peak_price, "
            f"  last_event_id=excluded.last_event_id",
            (payload["pid"], payload["size"], payload["actual_entry_price"],
             payload["size_usd"], evt.ts_ms, payload["actual_entry_price"], evt.id),
        )
        await self._write_db.commit()

    async def _apply_trade_closed(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        cur = await self._write_db.execute(
            "SELECT producer FROM events WHERE id = ?", (payload["decision_event_id"],),
        )
        row = await cur.fetchone()
        if row is None:
            logger.error("trade_closed references unknown decision_event_id=%s", payload["decision_event_id"])
            return
        producer = row[0]
        if not producer.startswith("model_"):
            return
        model = producer[len("model_"):]
        if not _MODEL_NAME_RE.match(model):
            return
        table = f"materialized_positions_{model}"
        await self._write_db.execute(
            f"DELETE FROM {table} WHERE pid = ?", (payload["pid"],),
        )
        await self._write_db.commit()
