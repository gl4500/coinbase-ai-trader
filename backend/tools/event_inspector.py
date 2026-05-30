"""event_inspector — CLI to query the event store for debugging.

Examples:
  python -m tools.event_inspector --db backend/coinbase.db --type price_tick --pid BTC-USD --limit 20
  python -m tools.event_inspector --db backend/coinbase.db --from-id 5000 --until-id 5100
"""
from __future__ import annotations

import argparse
import asyncio
import json

import aiosqlite


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="event_inspector",
        description="Query the events table for debugging.")
    p.add_argument("--db", required=True)
    p.add_argument("--type", default=None, help="Filter by event_type")
    p.add_argument("--pid", default=None, help="Filter by pid")
    p.add_argument("--from-id", type=int, default=None)
    p.add_argument("--until-id", type=int, default=None)
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args(argv)


async def query(args):
    where = []
    params = []
    if args.type:
        where.append("event_type = ?"); params.append(args.type)
    if args.pid:
        where.append("pid = ?"); params.append(args.pid)
    if args.from_id is not None:
        where.append("id >= ?"); params.append(args.from_id)
    if args.until_id is not None:
        where.append("id <= ?"); params.append(args.until_id)
    sql = "SELECT id, ts_ms, event_type, pid, payload_json, producer FROM events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(args.limit)

    async with aiosqlite.connect(args.db) as conn:
        cur = await conn.execute(sql, tuple(params))
        for row in await cur.fetchall():
            print(json.dumps({
                "id": row[0], "ts_ms": row[1], "event_type": row[2],
                "pid": row[3], "producer": row[5], "payload": json.loads(row[4]),
            }, default=str))


def main(argv=None):
    asyncio.run(query(_parse_args(argv)))


if __name__ == "__main__":
    main()
