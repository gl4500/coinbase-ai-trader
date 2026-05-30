"""Tests for api_server — FastAPI surface reading materialized views."""
from contextlib import asynccontextmanager

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from services import event_types as et
from services import event_writer
from services.api_server import build_app
from services.events_schema import init_events_schema, init_materialized_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


def _make_client(db_path):
    app = build_app(db_path=db_path)
    return TestClient(app)


def test_status_returns_ok(db_path):
    client = _make_client(db_path)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "ts_ms" in body


def test_products_returns_materialized_latest_price_rows(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO materialized_latest_price "
                "(pid, price, bid, ask, pct_change_24h, last_event_id, last_updated_ts_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", 95000.0, 94999.0, 95001.0, 1.5, 1, 1_700_000_000_000),
            )
            await conn.commit()
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/products")
    assert r.status_code == 200
    rows = r.json()
    assert any(p["pid"] == "BTC-USD" and p["price"] == 95000.0 for p in rows)


def test_trades_filters_to_trade_events(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=1, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(conn, producer="model_v3", ts_ms=2, payload=sig)
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000, actual_entry_price=95000, fee_paid=0.57,
                trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_2",
            )
            await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=3, payload=td)
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/trades")
    assert r.status_code == 200
    rows = r.json()
    assert all(e["event_type"] in ("trade_decided", "trade_closed") for e in rows)
    assert any(e["pid"] == "BTC-USD" for e in rows)


def test_compare_returns_per_model_positions(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO materialized_positions_v3 (pid, size, avg_price, position_dollars, "
                "entry_time_ms, peak_price, peak_pnl_pct, last_event_id) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", 0.001, 95000, 95.0, 1, 95000, 0.0, 1),
            )
            await conn.commit()
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/compare")
    assert r.status_code == 200
    body = r.json()
    assert "v3" in body and "v4_5" in body
    assert any(p["pid"] == "BTC-USD" for p in body["v3"])


def test_equity_curve_builds_from_trade_closed_events(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(conn, producer="model_v3", ts_ms=1, payload=sig)
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000, actual_entry_price=95000, fee_paid=0.57,
                trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="t1",
            )
            dec_id = await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=2, payload=td)
            tc1 = et.TradeClosedPayload(
                pid="BTC-USD", trade_uid="t1", exit_price=96000, exit_size=0.001,
                pnl=1.0, pct_pnl=1.05, hold_secs=3600, trigger_close="WS_TRAIL_STOP",
                decision_event_id=dec_id, exit_signal_event_id=None,
            )
            await event_writer.write_trade_closed(conn, producer="model_v3", ts_ms=3, payload=tc1)
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/equity_curve")
    assert r.status_code == 200
    points = r.json()
    assert any(p["cumulative_pnl"] == 1.0 for p in points)


def test_trading_disable_writes_system_control_event(db_path):
    client = _make_client(db_path)
    r = client.post("/api/trading/disable")
    assert r.status_code == 200
    import asyncio
    async def check():
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT event_type, payload_json FROM events WHERE event_type='system_control' "
                "ORDER BY id DESC LIMIT 1"
            )
            return await cur.fetchone()
    row = asyncio.get_event_loop().run_until_complete(check())
    assert row is not None
    import json
    assert json.loads(row[1])["action"] == "disable"


def test_websocket_pushes_price_tick(db_path):
    import asyncio, json
    client = _make_client(db_path)
    with client.websocket_connect("/ws") as ws:
        async def emit():
            async with aiosqlite.connect(db_path) as conn:
                p = et.PriceTickPayload(pid="BTC-USD", price=95000, bid=None, ask=None,
                                        volume_24h=None, source="ws")
                await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        asyncio.get_event_loop().run_until_complete(emit())
        msg = ws.receive_json()
        assert msg["event_type"] == "price_tick"
        assert msg["pid"] == "BTC-USD"


from services.api_server import _parse_args


def test_parse_args_db_and_port():
    args = _parse_args(["--db", "/tmp/x.db", "--port", "8001"])
    assert args.db == "/tmp/x.db"
    assert args.port == 8001
