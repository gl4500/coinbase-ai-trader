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
from agents.cnn_agent import (
    _CNN_STOP_LOSS_PCT, _CNN_ATR_TRAIL_MIN, _P_DOWN_EXIT_THRESHOLD,
)
from agents.exit_thresholds import _compute_exit_threshold  # task #46

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

        pct_entry = (price - avg_price) / avg_price
        trail_pct = pos.get("trail_pct", _CNN_ATR_TRAIL_MIN)

        # Task #46: PnL-anchored trail. Ratchet peak_pnl_pct from ratcheted
        # peak_price (consistent with scan-loop initialization in cnn_agent).
        # position_dollars cache is owned by scan-loop (refreshed every ~15min);
        # WS reads but does not write to avoid per-tick mutation contention.
        peak_pnl_from_peak_price = (peak - avg_price) / avg_price if avg_price > 0 else 0.0
        pos["peak_pnl_pct"] = max(pos.get("peak_pnl_pct", 0.0), peak_pnl_from_peak_price)
        position_dollars = pos.get("position_dollars") or (float(pos.get("size", 0.0)) * price)
        total_capital = book.balance + sum(
            p.get("position_dollars", 0.0) for p in book.positions.values()
        )

        exit_threshold = _compute_exit_threshold(
            peak_pnl_pct=pos["peak_pnl_pct"],
            atr_pct=trail_pct,
            position_dollars=position_dollars,
            total_capital=total_capital,
        )

        # Task #46 B2: MODEL_DOWN — cached p_down from generate_signal.
        p_down = pos.get("p_down", 0.0)

        trigger = None
        if pct_entry <= -_CNN_STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif p_down > _P_DOWN_EXIT_THRESHOLD:
            trigger = "WS_MODEL_DOWN"
        elif pct_entry < exit_threshold:
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
