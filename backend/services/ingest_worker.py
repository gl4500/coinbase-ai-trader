"""ingest_worker — single source of truth for all market observations.

Process model: one CLI entry point that owns:
  * Coinbase WebSocket subscriber (price_tick events)
  * REST candle backfill + close-detection (candle_close events)
  * Marketcap polling via CoinPaprika (marketcap_snapshot events)

Never runs inference. Never makes trade decisions. Never reads events.

Each concern is a separate inner class so they can be tested + restarted
in isolation. Run together via `python -m services.ingest_worker`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import aiosqlite

from services import event_types as et
from services import event_writer

logger = logging.getLogger(__name__)


class _WSIngest:
    """Subscribes to Coinbase ticker WS and writes one price_tick event per update."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def handle_tick(self, ticker: Dict) -> None:
        """Process one ticker message from Coinbase WS. Discards invalid messages."""
        assert self._db is not None
        pid = ticker.get("product_id")
        price_raw = ticker.get("price") or ticker.get("close")
        if not pid or not price_raw:
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return
        bid = ticker.get("best_bid")
        ask = ticker.get("best_ask")
        vol = ticker.get("volume_24_h")
        payload = et.PriceTickPayload(
            pid=pid,
            price=price,
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            volume_24h=float(vol) if vol is not None else None,
            source="ws",
        )
        await event_writer.write_price_tick(
            self._db, producer=self._producer,
            ts_ms=int(time.time() * 1000), payload=payload,
        )


class _CandleIngest:
    """Emits candle_close events on bar boundaries. Idempotent per (pid, tier, bar_ts_ms)."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None
        self._seen: set = set()  # (pid, tier, bar_ts_ms) — process-local dedupe

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def emit_close(self, *, pid: str, tier: str, ohlcv: Dict, bar_ts_ms: int) -> None:
        assert self._db is not None
        key = (pid, tier, bar_ts_ms)
        if key in self._seen:
            return
        payload = et.CandleClosePayload(
            pid=pid, tier=tier,
            open=float(ohlcv["open"]),
            high=float(ohlcv["high"]),
            low=float(ohlcv["low"]),
            close=float(ohlcv["close"]),
            volume=float(ohlcv["volume"]),
            bar_ts_ms=bar_ts_ms,
        )
        await event_writer.write_candle_close(
            self._db, producer=self._producer, ts_ms=int(time.time() * 1000),
            payload=payload,
        )
        self._seen.add(key)


class _MarketcapIngest:
    """Emits marketcap_snapshot events. Caller drives the cadence (e.g. via
    services.marketcap_history_cache)."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def emit_snapshot(self, *, pid: str, snapshot: Dict) -> None:
        assert self._db is not None
        payload = et.MarketcapSnapshotPayload(
            pid=pid,
            market_cap=snapshot.get("market_cap"),
            fdv=snapshot.get("fdv"),
            circ_supply=snapshot.get("circ_supply"),
            total_supply=snapshot.get("total_supply"),
            vol_24h=snapshot.get("vol_24h"),
            source=snapshot.get("source", "coinpaprika"),
        )
        await event_writer.write_marketcap_snapshot(
            self._db, producer=self._producer, ts_ms=int(time.time() * 1000),
            payload=payload,
        )


def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ingest_worker",
        description="Coinbase market-data ingest worker — single-process WS/REST/marketcap producer.")
    p.add_argument("--db", required=True,
                   help="SQLite DB path (e.g. backend/coinbase.db)")
    p.add_argument("--products", required=True,
                   help="Comma-separated product ids (e.g. BTC-USD,ETH-USD)")
    p.add_argument("--no-marketcap", action="store_true",
                   help="Skip marketcap polling")
    ns = p.parse_args(argv)
    ns.products = [s.strip() for s in ns.products.split(",") if s.strip()]
    return ns


async def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger.info("ingest_worker starting: db=%s products=%s no_marketcap=%s",
                args.db, args.products, args.no_marketcap)

    from services.events_schema import init_events_schema
    async with aiosqlite.connect(args.db) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await init_events_schema(conn)

    ws = _WSIngest(db_path=args.db)
    candles = _CandleIngest(db_path=args.db)
    mc = None if args.no_marketcap else _MarketcapIngest(db_path=args.db)

    await ws.start()
    await candles.start()
    if mc:
        await mc.start()

    try:
        await asyncio.Event().wait()      # block forever; operator stops via SIGTERM
    finally:
        await ws.stop()
        await candles.stop()
        if mc:
            await mc.stop()


if __name__ == "__main__":
    asyncio.run(main())
