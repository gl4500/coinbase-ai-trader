"""Tests for feature_snapshot — rebuild model features from event history.

Numerical-parity intent: given a set of candle_close + price_tick events for
PID X, the feature vector produced by feature_snapshot.build_for() must
equal (within float tolerance) what the current cnn_agent feature path would
produce given the same input candles/price.
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer, feature_snapshot
from services.events_schema import init_events_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        for i in range(60):
            close = 100.0 + i * 0.5
            payload = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=close - 0.5, high=close + 0.5, low=close - 1.0,
                close=close, volume=10.0 + i,
                bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(
                conn, producer="ingest", ts_ms=payload.bar_ts_ms, payload=payload,
            )
    return path


@pytest.mark.asyncio
async def test_build_for_returns_candles_in_time_order(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=60)
    assert len(snap.candles) == 60
    closes = [c["close"] for c in snap.candles]
    assert closes == sorted(closes)


@pytest.mark.asyncio
async def test_build_for_includes_last_event_ids(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=60)
    assert snap.last_candle_close_id is not None


@pytest.mark.asyncio
async def test_build_for_empty_pid_returns_empty_snapshot(db_path):
    snap = await feature_snapshot.build_for("NOPE-USD", db_path, tier="1h", lookback=60)
    assert snap.candles == []
    assert snap.last_candle_close_id is None


@pytest.mark.asyncio
async def test_build_for_respects_lookback_window(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=10)
    assert len(snap.candles) == 10
    closes = [c["close"] for c in snap.candles]
    assert closes[0] > 100.0   # took the most recent 10
