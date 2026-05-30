"""model_service — event-driven inference + decisioning worker.

One process per active model. Each instance:
  - Owns a consumer cursor (name = 'model_<model_name>')
  - Polls the event stream batch-by-batch
  - For each batch, runs inference/decision/exit hooks
  - Writes signal_scored / trade_decided / trade_closed / exit_triggered events

Concretely the runtime hooks are added in Tasks 14-16; this skeleton just sets
up the poll loop + cursor commit + lifespan.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import aiosqlite

from services import event_writer
from services.event_consumer import Event, EventConsumer

logger = logging.getLogger(__name__)

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_DEFAULT_SCAN_INTERVAL = 15.0


class ModelService:
    """One per active model. Drives one poll → inference → decision loop."""

    _STOP_LOSS_PCT = 0.08
    _GIVEBACK_FRAC = 0.30
    _FEE_RATE     = 0.006

    def __init__(
        self,
        *,
        db_path: str,
        model_name: str,
        deployment_path: Optional[str] = None,
        scan_interval: float = _DEFAULT_SCAN_INTERVAL,
    ):
        if not _MODEL_NAME_RE.match(model_name):
            raise ValueError(f"model_name {model_name!r} not a safe identifier")
        self._db_path = db_path
        self._model = model_name
        self._deployment_path = deployment_path
        self._scan_interval = scan_interval
        self._consumer = EventConsumer(db_path, name=f"model_{model_name}")
        self._write_db: Optional[aiosqlite.Connection] = None
        self._producer = f"model_{model_name}"
        self._last_signal_id_by_pid: dict[str, int] = {}
        self._last_signal_side_by_pid: dict[str, str] = {}
        self._open_trades_by_pid: dict[str, str] = {}
        self._positions_by_pid: dict[str, dict] = {}

    @property
    def model_name(self) -> str:
        return self._model

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
        """Process one batch. Returns number of events processed."""
        events = await self._consumer.poll()
        if not events:
            return 0
        for evt in events:
            try:
                await self._on_event(evt)
            except Exception:
                logger.exception("model_service[%s]: failed on event id=%s type=%s",
                                 self._model, evt.id, evt.event_type)
        await self._consumer.commit(events[-1].id)
        return len(events)

    async def _on_event(self, evt: Event) -> None:
        if evt.event_type == "candle_close" and evt.pid:
            await self._on_candle_close(evt)
        elif evt.event_type == "price_tick" and evt.pid:
            await self._on_price_tick(evt)

    async def _on_candle_close(self, evt: Event) -> None:
        """Score a signal + (Task 15) decide trade."""
        from services.feature_snapshot import build_for
        snapshot = await build_for(evt.pid, self._db_path, tier="1h", lookback=360)
        scored = await self._score_signal(evt.pid, snapshot)
        if scored is None:
            return
        from services import event_types as et
        payload = et.SignalScoredPayload(
            pid=evt.pid,
            model=self._model,
            model_version=scored["model_version"],
            feature_hash=scored["feature_hash"],
            scores=scored["scores"],
            side=scored["side"],
            strength=scored["strength"],
            regime=scored.get("regime"),
            deployment_profile_id=None,
            input_event_ids={
                "last_price_tick_id": snapshot.last_price_tick_id or 0,
                "last_candle_close_id": snapshot.last_candle_close_id or 0,
            },
        )
        assert self._write_db is not None
        sig_id = await event_writer.write_signal_scored(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=payload,
        )
        self._last_signal_id_by_pid[evt.pid] = sig_id
        self._last_signal_side_by_pid[evt.pid] = scored["side"]
        decision = await self._decide_trade(evt.pid, scored["side"], snapshot, sig_id)
        if decision is not None:
            await self._emit_trade_decided(evt, decision, sig_id, scored["side"])

    async def _on_price_tick(self, evt: Event) -> None:
        assert self._write_db is not None
        if evt.pid not in self._positions_by_pid:
            return
        pos = self._positions_by_pid[evt.pid]
        import json
        payload = json.loads(evt.payload_json)
        price = payload["price"]
        avg = pos["avg_price"]
        if avg <= 0 or price <= 0:
            return

        if price > pos["peak_price"]:
            pos["peak_price"] = price

        pct_entry = (price - avg) / avg
        peak_pct = (pos["peak_price"] - avg) / avg
        giveback = max(peak_pct * self._GIVEBACK_FRAC, 2 * self._FEE_RATE)
        trail_floor_pct = peak_pct - giveback
        current_pct = pct_entry

        trigger = None
        if pct_entry <= -self._STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif peak_pct > 0 and current_pct <= trail_floor_pct:
            trigger = "WS_TRAIL_STOP"

        if trigger is None:
            return

        from services import event_types as et
        exit_payload = et.ExitTriggeredPayload(
            pid=evt.pid, trade_uid=pos["trade_uid"], trigger_type=trigger,
            peak_pnl_pct=peak_pct * 100, current_pnl_pct=current_pct * 100,
            exit_threshold=trail_floor_pct * 100, price_at_trigger=price,
            trigger_price_event_id=evt.id,
        )
        await event_writer.write_exit_triggered(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=exit_payload,
        )
        # Task #87: exit-side fee scales with the exit notional, not the entry
        # notional. Pre-#87 the calc charged entry-notional * FEE_RATE * 2,
        # under/over-stating the realized PnL when exit price diverged from entry.
        exit_notional = price * pos["size"]
        fees = (pos["size_usd"] + exit_notional) * self._FEE_RATE
        pnl = (price - avg) * pos["size"] - fees
        close_payload = et.TradeClosedPayload(
            pid=evt.pid, trade_uid=pos["trade_uid"], exit_price=price,
            exit_size=pos["size"], pnl=pnl, pct_pnl=pct_entry * 100,
            hold_secs=int((evt.ts_ms - pos["entry_ts_ms"]) / 1000),
            trigger_close=trigger, decision_event_id=pos["decision_event_id"],
            exit_signal_event_id=None,
        )
        await event_writer.write_trade_closed(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=close_payload,
        )
        del self._positions_by_pid[evt.pid]

    async def _score_signal(self, pid: str, snapshot) -> Optional[dict]:
        """Default: route to agents/xgb_signal for the configured model.

        Overridden in tests via monkeypatch. Returns the inference result dict
        or None to skip emitting a signal for this candle close.
        """
        if not snapshot.candles or len(snapshot.candles) < 60:
            return None
        return {
            "side": "HOLD", "strength": 0.0, "scores": {"p_up": 0.5},
            "model_version": f"{self._model}_runtime",
            "feature_hash": "0",
            "regime": None,
        }

    async def _decide_trade(self, pid, side, snapshot, signal_event_id):
        """Default: HOLD does nothing; BUY/SELL is overridden in tests.

        The real decision logic — Kelly sizing, capital cap, fee math — is wired
        in Task 17 via _decide_trade_v3 / _decide_trade_v4_5 dispatch. Default
        no-op here keeps the unit tests focused on plumbing, not finance.
        """
        return None

    async def _emit_trade_decided(self, evt: Event, decision: dict, signal_event_id: int, side: str) -> None:
        assert self._write_db is not None
        from services import event_types as et
        trade_uid = f"{self._model}_{evt.pid}_{evt.ts_ms}"
        payload = et.TradeDecidedPayload(
            pid=evt.pid, model=self._model, side=side,
            size=decision["size"], size_usd=decision["size_usd"],
            intended_entry_price=decision["intended_entry_price"],
            actual_entry_price=decision["actual_entry_price"],
            fee_paid=decision["fee_paid"], trigger=decision["trigger"],
            signal_event_id=signal_event_id,
            deployment_profile_id=decision.get("deployment_profile_id"),
            trade_uid=trade_uid,
        )
        decision_event_id = await event_writer.write_trade_decided(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=payload,
        )
        self._open_trades_by_pid[evt.pid] = trade_uid
        # Task #82: wire the in-memory position so subsequent price_tick events
        # can fire exit triggers via _on_price_tick. Pre-#82 _positions_by_pid
        # was populated only by tests; production trade_decided emission left
        # the exit logic structurally unreachable.
        entry_price = decision["actual_entry_price"]
        self._positions_by_pid[evt.pid] = {
            "trade_uid":         trade_uid,
            "avg_price":         entry_price,
            "peak_price":        entry_price,
            "size":              decision["size"],
            "size_usd":          decision["size_usd"],
            "entry_ts_ms":       evt.ts_ms,
            "decision_event_id": decision_event_id,
        }

    async def run_forever(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("model_service[%s]: tick failed", self._model)
            await asyncio.sleep(self._scan_interval)


def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="model_service",
        description="Event-driven inference + decisioning worker.")
    p.add_argument("--model", required=True, choices=["v3", "v4_5"],
                   help="Model variant (v3 = binary XGB; v4_5 = 3-class XGB)")
    p.add_argument("--db", required=True,
                   help="SQLite DB path")
    p.add_argument("--deployment", default=None,
                   help="Phase 4 deployment_n{N}.json path (optional)")
    p.add_argument("--paper", action="store_true",
                   help="Dry-run mode; no live orders")
    return p.parse_args(argv)


async def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger.info("model_service starting: model=%s db=%s deployment=%s paper=%s",
                args.model, args.db, args.deployment, args.paper)

    from services.events_schema import init_events_schema, init_materialized_schema
    async with aiosqlite.connect(args.db) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=[args.model])

    svc = ModelService(
        db_path=args.db, model_name=args.model,
        deployment_path=args.deployment,
    )
    await svc.start()
    try:
        await svc.run_forever()
    finally:
        await svc.stop()


if __name__ == "__main__":
    asyncio.run(main())
