"""Maker/taker routing for live EXIT orders.

Single source of truth for: (1) which exit triggers post as maker vs cross as
taker, (2) the USE_MAKER_EXECUTION + dry-run gate, (3) sourcing bid/ask quotes,
(4) the call into order_executor. Both exit paths — cnn_agent._check_risk_exits
(scan loop) and exit_watcher.on_price_tick (WS) — close the paper book first,
then call execute_live_exit(), which no-ops unless the flag is on and the
executor is live. Mirrors the entry leg (cnn_agent._execute_live_order) but
gates the WHOLE live-order leg behind the flag, because exits place no live
order today (so flag-off must stay byte-for-byte paper-only).

Spec: docs/superpowers/specs/2026-06-21-maker-execution-exit-leg-design.md
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)

# Trail + model-down exits can wait for a post-only fill (30s market fallback
# lives in execute_maker_signal). Hard stops + forced time exits cross now.
_MAKER_EXIT_TRIGGERS: frozenset = frozenset(
    {"TRAIL_STOP", "WS_TRAIL_STOP", "MODEL_DOWN", "WS_MODEL_DOWN"}
)


def is_maker_exit(trigger: str) -> bool:
    """True if `trigger` should post as a maker (post-only) exit."""
    return trigger in _MAKER_EXIT_TRIGGERS


async def execute_live_exit(
    order_executor,
    *,
    pid: str,
    price: float,
    size: float,
    trigger: str,
) -> Optional[Dict]:
    """Place a live SELL liquidating `size` of `pid`, routed by `trigger`.

    No-ops (returns None) unless USE_MAKER_EXECUTION is on and the executor is
    live (not dry-run). Builds a SELL signal with NO `atr` key so order_executor
    sizes from `quote_size`; `quote_size` is set so the executor's
    `base_size = quote_size / fill_price` recovers the held `size`:
      - maker SELL fills at ask   -> quote_size = size * ask
      - taker SELL fills at price  -> quote_size = size * price
    """
    if (order_executor is None
            or getattr(order_executor, "dry_run", True)
            or not config.use_maker_execution):
        return None

    signal: Dict = {
        "product_id":  pid,
        "side":        "SELL",
        "price":       price,
        "signal_type": trigger,
    }

    if is_maker_exit(trigger):
        quotes = await coinbase_client.get_best_bid_ask([pid])
        quote  = quotes.get(pid, {})
        bid    = quote.get("bid") or 0.0
        ask    = quote.get("ask") or 0.0
        if bid <= 0 or ask <= 0:
            logger.warning(
                "maker exit %s for %s missing quotes (bid=%s ask=%s) — "
                "skipping live exit", trigger, pid, bid, ask,
            )
            return None
        signal["bid"]        = bid
        signal["ask"]        = ask
        signal["quote_size"] = round(size * ask, 2)
        return await order_executor.execute_maker_signal(signal)

    signal["quote_size"] = round(size * price, 2)
    return await order_executor.execute_signal(signal)
