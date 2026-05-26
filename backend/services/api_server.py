"""api_server — FastAPI process serving the frontend from materialized views.

Replaces today's main.py API surface. Drops:
  * Scan loop (moves to model_service)
  * Inference (moves to model_service)
  * Exit checker (moves to model_service)

Reads-only against the event store + materialized_* tables. WebSocket bridge
tails events for frontend push.

Internal state for sandbox/test isolation: build_app(db_path=...) takes the
event-store path so tests can use a tmp DB; CLI main() reads it from --db.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiosqlite
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def build_app(*, db_path: str) -> FastAPI:
    """Construct a FastAPI app reading from the event store at db_path."""

    app = FastAPI(title="Coinbase Trader (event-sourced)", version="3.0.0")
    app.state.db_path = db_path

    @app.get("/api/status")
    async def status():
        return {"ok": True, "ts_ms": int(time.time() * 1000)}

    @app.get("/api/products")
    async def products():
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT pid, price, bid, ask, pct_change_24h, last_updated_ts_ms "
                "FROM materialized_latest_price"
            )
            rows = await cur.fetchall()
        return [
            {"pid": r[0], "price": r[1], "bid": r[2], "ask": r[3],
             "pct_change_24h": r[4], "last_updated_ts_ms": r[5]}
            for r in rows
        ]

    @app.get("/api/trades")
    async def trades(limit: int = 200):
        import json
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT id, ts_ms, event_type, pid, payload_json, producer "
                "FROM events WHERE event_type IN ('trade_decided', 'trade_closed') "
                "ORDER BY id DESC LIMIT ?", (limit,),
            )
            rows = await cur.fetchall()
        return [
            {"id": r[0], "ts_ms": r[1], "event_type": r[2], "pid": r[3],
             "payload": json.loads(r[4]), "producer": r[5]}
            for r in rows
        ]

    @app.get("/api/compare")
    async def compare():
        out = {}
        for model in ("v3", "v4_5"):
            async with aiosqlite.connect(app.state.db_path) as conn:
                try:
                    cur = await conn.execute(
                        f"SELECT pid, size, avg_price, position_dollars, "
                        f"entry_time_ms, peak_price, peak_pnl_pct "
                        f"FROM materialized_positions_{model}"
                    )
                    rows = await cur.fetchall()
                except aiosqlite.OperationalError:
                    rows = []
            out[model] = [
                {"pid": r[0], "size": r[1], "avg_price": r[2],
                 "position_dollars": r[3], "entry_time_ms": r[4],
                 "peak_price": r[5], "peak_pnl_pct": r[6]}
                for r in rows
            ]
        return out

    @app.get("/api/equity_curve")
    async def equity_curve():
        import json
        out = []
        cum_pnl = 0.0
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT id, ts_ms, payload_json FROM events "
                "WHERE event_type='trade_closed' ORDER BY id ASC"
            )
            rows = await cur.fetchall()
        for r in rows:
            payload = json.loads(r[2])
            cum_pnl += float(payload.get("pnl", 0.0))
            out.append({"event_id": r[0], "ts_ms": r[1], "pnl": payload["pnl"],
                        "cumulative_pnl": cum_pnl})
        return out

    @app.post("/api/trading/enable")
    async def trading_enable():
        async with aiosqlite.connect(app.state.db_path) as conn:
            await conn.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, 'system_control', NULL, ?, 1, 'api_server')",
                (int(time.time() * 1000), '{"action": "enable"}'),
            )
            await conn.commit()
        return {"ok": True, "action": "enable"}

    @app.post("/api/trading/disable")
    async def trading_disable():
        async with aiosqlite.connect(app.state.db_path) as conn:
            await conn.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, 'system_control', NULL, ?, 1, 'api_server')",
                (int(time.time() * 1000), '{"action": "disable"}'),
            )
            await conn.commit()
        return {"ok": True, "action": "disable"}

    @app.websocket("/ws")
    async def ws_bridge(ws: WebSocket):
        await ws.accept()
        from services.event_consumer import EventConsumer
        consumer = EventConsumer(app.state.db_path,
                                 name=f"ws_bridge_{id(ws)}", batch_size=200)
        await consumer.start()
        try:
            while True:
                events = await consumer.poll()
                if events:
                    for e in events:
                        if e.event_type in ("price_tick", "trade_decided", "trade_closed",
                                            "exit_triggered", "system_control"):
                            await ws.send_json({
                                "id": e.id, "ts_ms": e.ts_ms,
                                "event_type": e.event_type, "pid": e.pid,
                                "payload": json.loads(e.payload_json),
                                "producer": e.producer,
                            })
                    await consumer.commit(events[-1].id)
                else:
                    await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass
        finally:
            await consumer.stop()

    return app


def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="api_server",
        description="FastAPI process serving frontend from event store.")
    p.add_argument("--db", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args(argv)


async def _run_materializer(db_path: str, stop_evt: asyncio.Event):
    from services.view_materializer import ViewMaterializer
    mat = ViewMaterializer(db_path, name="api_view_materializer")
    await mat.start()
    try:
        while not stop_evt.is_set():
            n = await mat.tick()
            if n == 0:
                await asyncio.sleep(0.5)
    finally:
        await mat.stop()


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    app = build_app(db_path=args.db)
    stop_evt = asyncio.Event()
    mat_task: Optional[asyncio.Task] = None

    @app.on_event("startup")
    async def _startup():
        nonlocal mat_task
        from services.events_schema import init_events_schema, init_materialized_schema
        async with aiosqlite.connect(args.db) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await init_events_schema(conn)
            await init_materialized_schema(conn, model_names=["v3", "v4_5"])
        mat_task = asyncio.create_task(_run_materializer(args.db, stop_evt))

    @app.on_event("shutdown")
    async def _shutdown():
        stop_evt.set()
        if mat_task:
            await mat_task

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
