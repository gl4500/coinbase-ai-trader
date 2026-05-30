"""Tests for ingest_worker — WS / REST / marketcap event producers."""
import aiosqlite
import pytest

from services.events_schema import init_events_schema
from services.ingest_worker import _WSIngest


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
    return path


@pytest.mark.asyncio
async def test_ws_tick_handler_emits_price_tick_event(db_path):
    ws = _WSIngest(db_path=db_path, producer="ingest")
    await ws.start()
    try:
        await ws.handle_tick({
            "product_id": "BTC-USD",
            "price": "95000.0",
            "best_bid": "94999.0",
            "best_ask": "95001.0",
            "volume_24_h": "100.0",
        })
    finally:
        await ws.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "price_tick"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["price"] == 95000.0
    assert payload["source"] == "ws"


@pytest.mark.asyncio
async def test_ws_handler_skips_invalid_ticker(db_path):
    ws = _WSIngest(db_path=db_path, producer="ingest")
    await ws.start()
    try:
        await ws.handle_tick({"product_id": "BTC-USD"})    # no price
    finally:
        await ws.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 0


from services.ingest_worker import _CandleIngest


@pytest.mark.asyncio
async def test_candle_ingest_writes_one_event_per_close(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        await ci.emit_close(
            pid="BTC-USD", tier="1h",
            ohlcv={"open": 100, "high": 105, "low": 99, "close": 103, "volume": 50},
            bar_ts_ms=1_700_000_000_000,
        )
    finally:
        await ci.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "candle_close"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["tier"] == "1h"
    assert payload["close"] == 103


@pytest.mark.asyncio
async def test_candle_ingest_is_idempotent_for_same_bar(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        await ci.emit_close(pid="BTC-USD", tier="1h",
                             ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                             bar_ts_ms=1_700_000_000_000)
        await ci.emit_close(pid="BTC-USD", tier="1h",
                             ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                             bar_ts_ms=1_700_000_000_000)
    finally:
        await ci.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='candle_close'")
        (n,) = await cur.fetchone()
    assert n == 1


@pytest.mark.asyncio
async def test_candle_ingest_rejects_unknown_tier(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        with pytest.raises(ValueError, match="tier"):
            await ci.emit_close(pid="X", tier="2h",
                                 ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                                 bar_ts_ms=1)
    finally:
        await ci.stop()


from services.ingest_worker import _MarketcapIngest


@pytest.mark.asyncio
async def test_marketcap_ingest_emits_snapshot_event(db_path):
    mc = _MarketcapIngest(db_path=db_path, producer="ingest")
    await mc.start()
    try:
        await mc.emit_snapshot(pid="BTC-USD", snapshot={
            "market_cap": 1.9e12, "fdv": 2.1e12,
            "circ_supply": 1.97e7, "total_supply": 1.98e7,
            "vol_24h": 4.0e10, "source": "coinpaprika",
        })
    finally:
        await mc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "marketcap_snapshot"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["market_cap"] == 1.9e12


@pytest.mark.asyncio
async def test_marketcap_ingest_accepts_partial_snapshot(db_path):
    mc = _MarketcapIngest(db_path=db_path, producer="ingest")
    await mc.start()
    try:
        await mc.emit_snapshot(pid="X-USD", snapshot={
            "market_cap": None, "fdv": None,
            "circ_supply": None, "total_supply": None,
            "vol_24h": None, "source": "coinpaprika",
        })
    finally:
        await mc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 1


import sys
from unittest.mock import patch

from services.ingest_worker import _parse_args


def test_parse_args_db_and_products():
    args = _parse_args(["--db", "/tmp/x.db", "--products", "BTC-USD,ETH-USD"])
    assert args.db == "/tmp/x.db"
    assert args.products == ["BTC-USD", "ETH-USD"]
    assert args.no_marketcap is False


def test_parse_args_no_marketcap_flag():
    args = _parse_args(["--db", "/tmp/x.db", "--products", "BTC-USD", "--no-marketcap"])
    assert args.no_marketcap is True
