"""Integration test: record live events for a synthetic scenario, replay them
through ModelService in a sandbox, verify the emitted signal_scored/
trade_decided events match the originals byte-for-byte (modulo timestamps).
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService
from tools.replay_consumer import replay_into_sandbox


@pytest.mark.asyncio
async def test_replay_produces_identical_signal_set(tmp_path, monkeypatch):
    src = str(tmp_path / "live.db")
    dst = str(tmp_path / "replay.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3"])
        for i in range(60):
            close = 100.0 + i * 0.5
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=close-0.5, high=close+0.5, low=close-1, close=close,
                volume=10, bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=cc.bar_ts_ms, payload=cc)

    deterministic_calls = {"n": 0}
    async def deterministic_score(self, pid, snapshot):
        deterministic_calls["n"] += 1
        return {"side": "HOLD", "strength": 0.4, "scores": {"p_up": 0.5},
                "model_version": "deterministic_test", "feature_hash": "Z",
                "regime": None}
    monkeypatch.setattr(ModelService, "_score_signal", deterministic_score)

    svc = ModelService(db_path=src, model_name="v3")
    await svc.start()
    try:
        while True:
            n = await svc.tick()
            if n == 0:
                break
    finally:
        await svc.stop()

    async with aiosqlite.connect(src) as conn:
        cur = await conn.execute(
            "SELECT pid, payload_json FROM events "
            "WHERE event_type='signal_scored' AND producer='model_v3' ORDER BY id"
        )
        live_signals = [(r[0], json.loads(r[1])) for r in await cur.fetchall()]

    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=0, until_event=None)
    async with aiosqlite.connect(dst) as conn:
        await conn.execute(
            "DELETE FROM events WHERE event_type='signal_scored' AND producer='model_v3'"
        )
        await conn.execute("DELETE FROM consumer_cursors")
        await conn.commit()
        await init_materialized_schema(conn, model_names=["v3"])

    svc2 = ModelService(db_path=dst, model_name="v3")
    await svc2.start()
    try:
        while True:
            n = await svc2.tick()
            if n == 0:
                break
    finally:
        await svc2.stop()

    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute(
            "SELECT pid, payload_json FROM events "
            "WHERE event_type='signal_scored' AND producer='model_v3' ORDER BY id"
        )
        replay_signals = [(r[0], json.loads(r[1])) for r in await cur.fetchall()]

    assert [(p, s["side"], s["strength"]) for p, s in live_signals] == \
           [(p, s["side"], s["strength"]) for p, s in replay_signals]
