"""Paper maker-entry shadow — did the price path trade through a resting bid?

Measurement only. Places, cancels and simulates no orders; writes only the
``maker_shadow`` table.

ESTIMAND. An instantaneous virtual post-only BUY resting at the WS best bid
captured right after a successful paper BUY, observed for ``window_s``. It is a
PRICE-PATH PROXY, not a fill: the ticker stream carries no queue position,
displayed size or order acknowledgement. It is conditional on the BUYs the paper
book actually took, and it is not the live order path, which quotes and posts
later.

* ``crossed`` — an own-product trade printed strictly BELOW the limit inside the
  window (the level was traded through).
* ``touched`` — an own-product trade printed AT or below the limit.
* ``markout_bps`` — for crossed intents only: the last own-product trade at or
  before ``cross_ts + markout_s``, against the limit. A fixed horizon from the
  crossing; ``mark_age_s`` says how old that trade was at the horizon.
* ``feed_gap`` — the WS feed reconnected while the intent was being observed, so
  a crossing may have been missed. ``None`` when the feed epoch was unreadable.

Every intent is finalised at its own deadline (window end, or the markout
horizon for crossed intents) by ``sweep``, which runs on every tick, on every
register, and from a periodic sweeper — never by waiting for unrelated activity.
Unmeasurable intents are recorded (``no_quote`` / ``duplicate``), not dropped.
Never raises into the WS receive loop (invariant #18) or the scan loop (#14).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Sink = Callable[[dict], Awaitable[None]]


@dataclass
class _Intent:
    pid: str
    limit: float
    ask: float
    created: float
    epoch0: Any
    touched: bool = False
    cross_ts: Optional[float] = None
    close_price: Optional[float] = None
    mark_price: Optional[float] = None
    mark_ts: Optional[float] = None


def _spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4


def _quote_problem(bid: Optional[float], ask: Optional[float]) -> Optional[str]:
    if bid is None or ask is None:
        return "missing quote"
    if bid <= 0:
        return "non-positive bid"
    if ask < bid:
        return "crossed quote"
    return None


class MakerShadow:
    def __init__(
        self,
        sink: Sink,
        clock: Callable[[], float] = time.time,
        window_s: float = 30.0,
        markout_s: float = 60.0,
        feed_epoch: Callable[[], Any] = lambda: 0,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._window = window_s
        self._markout = markout_s
        self._feed_epoch = feed_epoch
        self._open: Dict[str, _Intent] = {}
        self._pending: List[dict] = []
        self._sweeper: Optional[asyncio.Task] = None

    def open_count(self) -> int:
        return len(self._open)

    def _epoch(self) -> Any:
        try:
            return self._feed_epoch()
        except Exception:
            logger.exception("maker_shadow could not read the feed epoch")
            return None

    def _deadline(self, it: _Intent) -> float:
        end = it.created + self._window
        if it.cross_ts is not None:
            end = max(end, it.cross_ts + self._markout)
        return end

    def _row(self, pid: str, status: str, **fields) -> dict:
        base = {
            "product_id": pid,
            "status": status,
            "touched": False,
            "limit_price": None,
            "ask": None,
            "spread_bps": None,
            "created_ts": self._clock(),
            "cross_ts": None,
            "time_to_cross_s": None,
            "window_close_price": None,
            "markout_s": self._markout,
            "markout_bps": None,
            "mark_age_s": None,
            "finalised_late_s": None,
            "feed_gap": None,
            "window_s": self._window,
            "detail": None,
        }
        base.update(fields)
        return base

    def register(self, pid: str, bid: Optional[float], ask: Optional[float]) -> str:
        self._finalise_due(self._clock())
        problem = _quote_problem(bid, ask)
        if problem is not None:
            self._pending.append(self._row(pid, "no_quote", detail=problem))
            return "no_quote"
        if pid in self._open:
            self._pending.append(
                self._row(pid, "duplicate", limit_price=bid, ask=ask, detail="intent already open")
            )
            return "duplicate"
        self._open[pid] = _Intent(
            pid=pid,
            limit=float(bid),
            ask=float(ask),
            created=self._clock(),
            epoch0=self._epoch(),
        )
        return "open"

    def _observe(self, it: _Intent, price: float, now: float) -> None:
        if now >= self._deadline(it):
            return
        if now < it.created + self._window:
            it.close_price = price
            if price <= it.limit:
                it.touched = True
            if price < it.limit and it.cross_ts is None:
                it.cross_ts = now
        if it.cross_ts is not None and now <= it.cross_ts + self._markout:
            it.mark_price = price
            it.mark_ts = now

    def _finalise(self, it: _Intent, now: float) -> dict:
        crossed = it.cross_ts is not None
        deadline = self._deadline(it)
        epoch_now = self._epoch()
        feed_gap = None if it.epoch0 is None or epoch_now is None else epoch_now != it.epoch0
        markout = None
        mark_age = None
        if crossed and it.mark_price is not None:
            markout = (it.mark_price - it.limit) / it.limit * 1e4
            mark_age = (it.cross_ts + self._markout) - it.mark_ts
        return self._row(
            it.pid,
            "crossed" if crossed else "not_crossed",
            touched=it.touched,
            limit_price=it.limit,
            ask=it.ask,
            spread_bps=_spread_bps(it.limit, it.ask),
            created_ts=it.created,
            cross_ts=it.cross_ts,
            time_to_cross_s=(it.cross_ts - it.created) if crossed else None,
            window_close_price=it.close_price,
            markout_bps=markout,
            mark_age_s=mark_age,
            finalised_late_s=now - deadline,
            feed_gap=feed_gap,
        )

    def _finalise_due(self, now: float) -> None:
        for key in [k for k, v in self._open.items() if now >= self._deadline(v)]:
            self._pending.append(self._finalise(self._open.pop(key), now))

    async def on_tick(self, pid: str, price: float) -> None:
        try:
            now = self._clock()
            it = self._open.get(pid)
            if it is not None:
                self._observe(it, price, now)
            self._finalise_due(now)
        except Exception:
            logger.exception("maker_shadow.on_tick failed (pid=%s price=%s)", pid, price)
        await self._flush()

    async def sweep(self) -> None:
        try:
            self._finalise_due(self._clock())
        except Exception:
            logger.exception("maker_shadow.sweep failed")
        await self._flush()

    async def _run_sweeper(self, interval_s: float) -> None:
        while True:
            await asyncio.sleep(interval_s)
            await self.sweep()

    def start_sweeper(self, interval_s: float = 5.0) -> None:
        """Finalise intents at their deadline even when no product ticks."""
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._run_sweeper(interval_s))

    async def _flush(self) -> None:
        rows, self._pending = self._pending, []
        for row in rows:
            try:
                await self._sink(row)
            except Exception:
                logger.exception("maker_shadow sink failed for %s", row.get("product_id"))


def attach(ws_subscriber, agent, sink: Sink, sweep_interval_s: float = 5.0) -> MakerShadow:
    """Build the shadow, feed it the WS reconnect count, register its tick
    handler, start the deadline sweeper and hand it to the agent. Call once per
    backend lifespan, from inside the running event loop."""
    shadow = MakerShadow(sink=sink, feed_epoch=lambda: ws_subscriber.connect_count)
    ws_subscriber.register_price_handler(shadow.on_tick)
    shadow.start_sweeper(sweep_interval_s)
    agent.maker_shadow = shadow
    return shadow
