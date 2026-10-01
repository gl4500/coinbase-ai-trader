"""Paper maker-fill shadow — would a post-only BUY at the best bid have filled?

Measurement only. Places, cancels and simulates no orders; touches no existing
table. A virtual intent rests at the bid captured right after a paper BUY and is
resolved from the live last-trade stream:

* ``filled``  — a trade printed strictly BELOW the limit inside the window, so
  the level was traded through and queue position cannot matter (conservative).
* ``touched`` — a trade printed AT or below the limit (optimistic upper bound).

Rows that could not be measured are recorded as ``no_quote`` / ``duplicate``
rather than dropped, because dropping them would inflate the fill rate.
Never raises into the WS receive loop (invariant #18).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Sink = Callable[[dict], Awaitable[None]]


@dataclass
class _Intent:
    pid: str
    limit: float
    ask: float
    created: float
    touched: bool = False
    fill_ts: Optional[float] = None
    last_price: Optional[float] = None


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
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._window = window_s
        self._open: Dict[str, _Intent] = {}
        self._pending: List[dict] = []

    def open_count(self) -> int:
        return len(self._open)

    def _row(self, pid: str, status: str, **fields) -> dict:
        base = {
            "product_id": pid,
            "status": status,
            "touched": False,
            "limit_price": None,
            "ask": None,
            "spread_bps": None,
            "created_ts": self._clock(),
            "fill_ts": None,
            "time_to_fill_s": None,
            "last_price": None,
            "drift_bps": None,
            "window_s": self._window,
            "detail": None,
        }
        base.update(fields)
        return base

    def register(self, pid: str, bid: Optional[float], ask: Optional[float]) -> str:
        problem = _quote_problem(bid, ask)
        if problem is not None:
            self._pending.append(self._row(pid, "no_quote", detail=problem))
            return "no_quote"
        if pid in self._open:
            self._pending.append(
                self._row(pid, "duplicate", limit_price=bid, ask=ask, detail="intent already open")
            )
            return "duplicate"
        self._open[pid] = _Intent(pid=pid, limit=float(bid), ask=float(ask), created=self._clock())
        return "open"

    def _finalise(self, it: _Intent) -> dict:
        filled = it.fill_ts is not None
        drift = None
        if filled and it.last_price is not None:
            drift = (it.last_price - it.limit) / it.limit * 1e4
        return self._row(
            it.pid,
            "filled" if filled else "unfilled",
            touched=it.touched,
            limit_price=it.limit,
            ask=it.ask,
            spread_bps=_spread_bps(it.limit, it.ask),
            created_ts=it.created,
            fill_ts=it.fill_ts,
            time_to_fill_s=(it.fill_ts - it.created) if filled else None,
            last_price=it.last_price,
            drift_bps=drift,
        )

    async def on_tick(self, pid: str, price: float) -> None:
        try:
            now = self._clock()
            it = self._open.get(pid)
            if it is not None and now < it.created + self._window:
                it.last_price = price
                if price <= it.limit:
                    it.touched = True
                if price < it.limit and it.fill_ts is None:
                    it.fill_ts = now
            elif it is not None:
                it.last_price = price
            for key in [k for k, v in self._open.items() if now >= v.created + self._window]:
                self._pending.append(self._finalise(self._open.pop(key)))
        except Exception:
            logger.exception("maker_shadow.on_tick failed (pid=%s price=%s)", pid, price)
        await self._flush()

    async def _flush(self) -> None:
        rows, self._pending = self._pending, []
        for row in rows:
            try:
                await self._sink(row)
            except Exception:
                logger.exception("maker_shadow sink failed for %s", row.get("product_id"))
