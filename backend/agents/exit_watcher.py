"""WS-driven exit checker.

Registers an async price-tick handler with CoinbaseWSSubscriber and fires
WS_TRAIL_STOP / WS_STOP_LOSS exits on every held position without waiting
for the 60s scan cycle. Max-hold (7-day) exit remains on the scan loop.

Loose coupling per feedback_loose_coupling.md:
  ws_subscriber  ->  doesn't know about exit_watcher
  cnn_agent      ->  doesn't know about exit_watcher
  exit_watcher   ->  reads book.positions; writes via book.sell()

Spec:  docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Constants come from cnn_agent so this module and the scan-loop checker
# stay in lockstep on thresholds. main.py already loads cnn_agent in the
# lifespan, so importing constants here adds no new process cost.
from agents.cnn_agent import _CNN_STOP_LOSS_PCT, _CNN_ATR_TRAIL_MIN

if TYPE_CHECKING:
    from agents.cnn_agent import _CNNBook
    from services.ws_subscriber import CoinbaseWSSubscriber

logger = logging.getLogger(__name__)


async def on_price_tick(pid: str, price: float, book: "_CNNBook") -> None:
    """Per-tick exit checker. Idempotent. Exceptions are caught + logged
    (invariant #18 in CLAUDE.md) so a handler failure cannot crash the
    WS receive loop or poison subsequent ticks.
    """
    try:
        pos = book.positions.get(pid)
        if pos is None:
            return                                              # ~99% of ticks

        avg_price = pos.get("avg_price", 0.0)
        if avg_price <= 0 or price <= 0:
            return

        peak = pos.get("peak_price") or avg_price
        if price > peak:
            pos["peak_price"] = price
            peak = price

        pct_entry     = (price - avg_price) / avg_price
        pct_from_peak = (price - peak) / peak
        trail_pct     = pos.get("trail_pct", _CNN_ATR_TRAIL_MIN)

        trigger = None
        if pct_entry <= -_CNN_STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif pct_from_peak <= -trail_pct:
            trigger = "WS_TRAIL_STOP"

        if trigger:
            await book.sell(pid, price, trigger=trigger)

    except Exception:
        logger.exception(
            "exit_watcher.on_price_tick failed (pid=%s price=%s)", pid, price,
        )


def attach(ws_subscriber: "CoinbaseWSSubscriber", book: "_CNNBook") -> None:
    """Register the per-tick exit handler. Call once per backend lifespan
    (in main.py after ws_subscriber.start() and after cnn_agent is built).
    """
    async def _handler(pid: str, price: float) -> None:
        await on_price_tick(pid, price, book)

    ws_subscriber.register_price_handler(_handler)
    logger.info("exit_watcher attached to ws_subscriber")
