"""
Coinbase Advanced Trade WebSocket subscriber.
Subscribes to the public ticker channel for live price + bid/ask updates.
No auth required for public ticker data.
"""
import asyncio
import json
import logging
from typing import Callable, Dict, List, Optional

import websockets

from config import config

logger = logging.getLogger(__name__)


class CoinbaseWSSubscriber:
    def __init__(self, broadcast_fn: Callable):
        self.broadcast_fn  = broadcast_fn
        self.state: Dict[str, Dict] = {}    # product_id → latest ticker
        self._task: Optional[asyncio.Task]  = None
        self._products: List[str]           = []
        self._price_handlers: List[Callable] = []   # async fn(pid, price) callbacks

    def get_price(self, product_id: str) -> Optional[float]:
        return self.state.get(product_id, {}).get("price")

    def get_bid_ask(self, product_id: str) -> Dict:
        t = self.state.get(product_id, {})
        return {"bid": t.get("bid"), "ask": t.get("ask")}

    def register_price_handler(self, fn: Callable) -> None:
        """Register an async callback fn(pid: str, price: float) fired on every tick."""
        self._price_handlers.append(fn)

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def set_products(self, product_ids: List[str]) -> None:
        self._products = list(product_ids)

    async def reconnect(self) -> None:
        """Cancel the current WS connection and immediately reconnect with updated product list."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while True:
            try:
                await self._connect()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"WS disconnected ({e}) — reconnecting in 5s")
                await asyncio.sleep(5)

    async def _connect(self) -> None:
        products = self._products
        if not products:
            await asyncio.sleep(10)
            return

        async with websockets.connect(
            config.coinbase_ws_url,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            await ws.send(json.dumps({
                "type":        "subscribe",
                "product_ids": products,
                "channel":     "ticker",
            }))
            logger.info(f"Coinbase WS subscribed to: {products}")

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle(msg)
                except Exception as e:
                    logger.debug(f"WS parse error: {e}")

    async def _handle(self, msg: Dict) -> None:
        if msg.get("channel") != "ticker":
            return
        for event in msg.get("events", []):
            for ticker in event.get("tickers", []):
                pid   = ticker.get("product_id")
                price = ticker.get("price") or ticker.get("close")
                if not pid or not price:
                    continue
                self.state[pid] = {
                    "product_id": pid,
                    "price":      float(price),
                    "bid":        float(ticker["best_bid"])  if ticker.get("best_bid")  else None,
                    "ask":        float(ticker["best_ask"])  if ticker.get("best_ask")  else None,
                    "volume_24h": float(ticker.get("volume_24_h", 0)),
                    "pct_change": float(ticker.get("price_percent_chg_24_h", 0)),
                }
                await self.broadcast_fn({
                    "type":       "price_update",
                    "product_id": pid,
                    **self.state[pid],
                })
                # Fire real-time trade handlers without blocking the WS receive loop
                for handler in self._price_handlers:
                    asyncio.create_task(handler(pid, float(price)))
