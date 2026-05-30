"""Tests for model_service — event-driven inference + decisioning."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


@pytest.mark.asyncio
async def test_model_service_starts_with_empty_cursor(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        events = await svc._consumer.poll()
        assert events == []
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_model_service_advances_cursor_after_tick(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=100.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        n = await svc.tick()
        assert n == 1
        n2 = await svc.tick()
        assert n2 == 0
    finally:
        await svc.stop()


import json


@pytest.mark.asyncio
async def test_candle_close_triggers_inference_emits_signal_scored(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.7, "scores": {"p_up": 0.7},
                    "model_version": "test", "feature_hash": "x", "regime": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h", open=1, high=2, low=0.5, close=1.5,
                volume=10, bar_ts_ms=1_700_000_000_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, producer, payload_json FROM events "
            "WHERE event_type='signal_scored' ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "signal_scored"
    assert row[1] == "model_v3"
    payload = json.loads(row[2])
    assert payload["side"] == "BUY"


@pytest.mark.asyncio
async def test_price_tick_alone_does_not_trigger_inference(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    score_calls = []
    async def spy(self, pid, snapshot):
        score_calls.append(pid)
        return {"side": "HOLD", "strength": 0.0, "scores": {},
                "model_version": "t", "feature_hash": "x", "regime": None}
    monkeypatch.setattr(ModelService, "_score_signal", spy)
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="X-USD", price=1.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    assert score_calls == []


@pytest.mark.asyncio
async def test_buy_signal_writes_trade_decided(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.8, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            if side != "BUY":
                return None
            return {"size": 0.001, "size_usd": 95.0,
                    "intended_entry_price": 95000.0,
                    "actual_entry_price": 95000.0, "fee_paid": 0.57,
                    "trigger": "SCAN", "deployment_profile_id": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2, low=0.5,
                                        close=1.5, volume=10, bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, payload_json FROM events WHERE event_type='trade_decided'"
        )
        row = await cur.fetchone()
    assert row is not None
    import json
    payload = json.loads(row[1])
    assert payload["side"] == "BUY"
    assert payload["trade_uid"].startswith("v3_BTC-USD_")


@pytest.mark.asyncio
async def test_buy_signal_populates_positions_by_pid(db_path, monkeypatch):
    """Task #82: after a BUY trade_decided event, ModelService._positions_by_pid
    must hold the new position so a subsequent price_tick can close it via the
    WS_TRAIL_STOP / WS_STOP_LOSS path. Pre-#82, _positions_by_pid was populated
    only by tests; production code emitted trade_decided but never wired the
    in-memory position, leaving _on_price_tick exit logic structurally
    unreachable."""
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.8, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            return {"size": 0.001, "size_usd": 95.0,
                    "intended_entry_price": 95000.0,
                    "actual_entry_price": 95000.0, "fee_paid": 0.57,
                    "trigger": "SCAN", "deployment_profile_id": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2,
                                        low=0.5, close=1.5, volume=10,
                                        bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest",
                                                    ts_ms=1, payload=cc)
        await svc.tick()
        assert "BTC-USD" in svc._positions_by_pid
        pos = svc._positions_by_pid["BTC-USD"]
        assert pos["avg_price"] == 95000.0
        assert pos["peak_price"] == 95000.0
        assert pos["size"] == 0.001
        assert pos["size_usd"] == 95.0
        assert pos["trade_uid"].startswith("v3_BTC-USD_")
        assert pos["entry_ts_ms"] == 1
        # decision_event_id captured from write_trade_decided lastrowid
        assert pos["decision_event_id"] > 0
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_buy_then_stop_loss_drop_closes_position(db_path, monkeypatch):
    """Task #82 full lifecycle: BUY emits trade_decided + populates
    _positions_by_pid → subsequent price tick below stop-loss fires
    exit_triggered + trade_closed → position removed from _positions_by_pid.
    Pre-#82 this whole chain was broken because the population step was missing."""
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.8, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            return {"size": 1.0, "size_usd": 100.0,
                    "intended_entry_price": 100.0, "actual_entry_price": 100.0,
                    "fee_paid": 0.6, "trigger": "SCAN",
                    "deployment_profile_id": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2,
                                        low=0.5, close=1.5, volume=10,
                                        bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest",
                                                    ts_ms=1, payload=cc)
            # 12% drop from entry $100 → below 8% stop-loss
            p = et.PriceTickPayload(pid="BTC-USD", price=88.0, bid=None,
                                     ask=None, volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest",
                                                  ts_ms=2, payload=p)
        await svc.tick()
        assert "BTC-USD" not in svc._positions_by_pid
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type FROM events WHERE event_type IN "
            "('exit_triggered','trade_closed') ORDER BY id"
        )
        types = [r[0] for r in await cur.fetchall()]
    assert types == ["exit_triggered", "trade_closed"]


@pytest.mark.asyncio
async def test_hold_signal_does_not_write_trade_decided(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "HOLD", "strength": 0.4, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            return None
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2, low=0.5,
                                        close=1.5, volume=10, bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='trade_decided'")
        (n,) = await cur.fetchone()
    assert n == 0


@pytest.mark.asyncio
async def test_price_tick_below_stop_loss_writes_exit_triggered_and_trade_closed(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def open_position(pid, entry_price, signal_event_id):
            svc._positions_by_pid[pid] = {
                "trade_uid": f"v3_{pid}_x", "size": 0.001, "avg_price": entry_price,
                "size_usd": entry_price * 0.001, "peak_price": entry_price,
                "entry_ts_ms": 1, "decision_event_id": signal_event_id,
            }
        await open_position("BTC-USD", 100.0, 999)
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=88.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type FROM events WHERE event_type IN "
            "('exit_triggered','trade_closed') ORDER BY id"
        )
        types = [r[0] for r in await cur.fetchall()]
    assert types == ["exit_triggered", "trade_closed"]


@pytest.mark.asyncio
async def test_trade_closed_pnl_uses_exit_notional_for_fee(db_path):
    """Task #87: exit-side fee scales with the exit notional (price * size),
    NOT the entry notional (size_usd). Pre-#87 the calc charged the entry
    notional twice, understating losses (exit price < entry) and overstating
    gains (exit price > entry).

    With entry $100 size 1.0 → size_usd $100, exit at $88:
      fees = (100 + 88*1) * 0.006 = 1.128 (corrected — splits entry+exit)
      pnl  = (88 - 100)*1 - 1.128 = -13.128
    Pre-fix would have charged 100*0.006*2 = 1.2 → pnl -13.2.
    """
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        svc._positions_by_pid["BTC-USD"] = {
            "trade_uid": "v3_BTC-USD_x", "size": 1.0, "avg_price": 100.0,
            "size_usd": 100.0, "peak_price": 100.0, "entry_ts_ms": 1,
            "decision_event_id": 1,
        }
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=88.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2,
                                                payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT payload_json FROM events WHERE event_type='trade_closed'"
        )
        row = await cur.fetchone()
    assert row is not None
    payload = json.loads(row[0])
    assert payload["pnl"] == pytest.approx(-13.128, abs=1e-6)


@pytest.mark.asyncio
async def test_price_tick_above_peak_updates_peak_no_exit(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        svc._positions_by_pid["BTC-USD"] = {
            "trade_uid": "v3_BTC-USD_x", "size": 0.001, "avg_price": 100.0,
            "size_usd": 100.0, "peak_price": 100.0, "entry_ts_ms": 1,
            "decision_event_id": 1,
        }
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=110.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
        assert svc._positions_by_pid["BTC-USD"]["peak_price"] == 110.0
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='trade_closed'")
        (n,) = await cur.fetchone()
    assert n == 0


@pytest.mark.asyncio
async def test_pnl_anchored_trail_fires_after_peak(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        svc._positions_by_pid["BTC-USD"] = {
            "trade_uid": "v3_BTC-USD_x", "size": 1.0, "avg_price": 100.0,
            "size_usd": 100.0, "peak_price": 115.0, "entry_ts_ms": 1,
            "decision_event_id": 1,
        }
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=110.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT event_type FROM events WHERE event_type='exit_triggered'")
        n = len(await cur.fetchall())
    assert n == 1


from services.model_service import _parse_args


def test_parse_args_model_only():
    args = _parse_args(["--model", "v3", "--db", "/tmp/x.db"])
    assert args.model == "v3"
    assert args.deployment is None
    assert args.paper is False


def test_parse_args_with_deployment_and_paper():
    args = _parse_args([
        "--model", "v4_5", "--db", "/tmp/x.db",
        "--deployment", "/tmp/dep.json", "--paper",
    ])
    assert args.model == "v4_5"
    assert args.deployment == "/tmp/dep.json"
    assert args.paper is True
