"""replay_consumer — copy a range of events from a source DB into a sandbox DB.

Used by replay-determinism tests + operator-driven backtests:
  python -m tools.replay_consumer --src backend/coinbase.db \
                                    --dst /tmp/replay.db \
                                    --from-event 0 --until-event 1000000

The destination DB is created (or reset) with the events schema; then events
in [from_event, until_event] are copied. Consumer cursors in the dst start
clean — a model_service pointed at the sandbox will replay from the beginning.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Optional

import aiosqlite

from services.events_schema import init_events_schema


async def replay_into_sandbox(
    *,
    src_db: str,
    dst_db: str,
    from_event: int = 0,
    until_event: Optional[int] = None,
) -> int:
    """Copy events with id in [from_event, until_event] from src_db to dst_db.
    Returns count copied."""
    if os.path.exists(dst_db):
        os.remove(dst_db)
    async with aiosqlite.connect(dst_db) as dst:
        await init_events_schema(dst)
    async with aiosqlite.connect(src_db) as src, aiosqlite.connect(dst_db) as dst:
        sql = ("SELECT ts_ms, event_type, pid, payload_json, schema_version, producer "
               "FROM events WHERE id >= ?")
        params = [from_event]
        if until_event is not None:
            sql += " AND id <= ?"
            params.append(until_event)
        sql += " ORDER BY id ASC"
        cur = await src.execute(sql, tuple(params))
        rows = await cur.fetchall()
        for r in rows:
            await dst.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, ?, ?, ?, ?, ?)", r,
            )
        await dst.commit()
        return len(rows)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="replay_consumer")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--from-event", type=int, default=0)
    p.add_argument("--until-event", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    n = asyncio.run(replay_into_sandbox(
        src_db=args.src, dst_db=args.dst,
        from_event=args.from_event, until_event=args.until_event,
    ))
    print(f"replayed {n} events into {args.dst}")


if __name__ == "__main__":
    main()
