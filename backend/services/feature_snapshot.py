"""Reconstruct feature inputs (OHLCV + last price) from the event store.

`build_for(pid, db_path, tier, lookback)` opens a read-only connection,
fetches the last <lookback> candle_close events for the pid+tier, and the
most recent price_tick. Returns a FeatureSnapshot dataclass.

This module deliberately does NOT compute features — it returns raw inputs.
Existing feature compute lives in agents/xgb_signal.py and is reused as-is by
model_service. Keeping the responsibilities separate means model service can
swap feature extractors without touching this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiosqlite


@dataclass(frozen=True)
class FeatureSnapshot:
    pid: str
    tier: str
    candles: List[Dict]                          # oldest → newest
    last_price: Optional[float] = None
    last_price_tick_id: Optional[int] = None
    last_candle_close_id: Optional[int] = None


async def build_for(
    pid: str,
    db_path: str,
    *,
    tier: str = "1h",
    lookback: int = 360,
) -> FeatureSnapshot:
    """Return a FeatureSnapshot with the last <lookback> candles (oldest first)
    + most recent price_tick for <pid>+<tier>."""
    candles: List[Dict] = []
    last_candle_close_id: Optional[int] = None
    last_price: Optional[float] = None
    last_price_tick_id: Optional[int] = None

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA busy_timeout=30000")

        cur = await db.execute(
            "SELECT id, payload_json FROM events "
            "WHERE event_type = 'candle_close' AND pid = ? "
            "ORDER BY id DESC LIMIT ?",
            (pid, lookback),
        )
        rows = await cur.fetchall()
        if rows:
            last_candle_close_id = rows[0][0]
            parsed = [json.loads(r[1]) for r in rows]
            parsed = [p for p in parsed if p.get("tier") == tier]
            parsed.reverse()
            candles = parsed

        cur = await db.execute(
            "SELECT id, payload_json FROM events "
            "WHERE event_type = 'price_tick' AND pid = ? "
            "ORDER BY id DESC LIMIT 1",
            (pid,),
        )
        row = await cur.fetchone()
        if row is not None:
            last_price_tick_id = row[0]
            last_price = json.loads(row[1]).get("price")

    return FeatureSnapshot(
        pid=pid, tier=tier, candles=candles,
        last_price=last_price,
        last_price_tick_id=last_price_tick_id,
        last_candle_close_id=last_candle_close_id,
    )
