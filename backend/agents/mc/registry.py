"""MC filter chain dispatch (#311-mc).

Reads MC_FILTERS at first apply_* call (cache-busted via _reset_chain_cache
for tests). Filters listed but unknown to the registry log a warning and
are skipped. Filter exceptions are caught so one broken filter cannot kill
the rest of the chain or the scan loop.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple, Type

from agents.mc.base import BuyFilter

logger = logging.getLogger(__name__)

# Map of filter name -> class. CIFilter (and future filters) self-register
# here on import. Tests patch this dict to inject spies.
_FILTER_CLASSES: Dict[str, Type[BuyFilter]] = {}

_chain: List[BuyFilter] = []
_chain_built: bool = False


def _build_chain() -> List[BuyFilter]:
    raw = os.getenv("MC_FILTERS", "") or ""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    chain: List[BuyFilter] = []
    for name in names:
        cls = _FILTER_CLASSES.get(name)
        if cls is None:
            logger.warning("MC_FILTERS lists unknown filter %r — skipping", name)
            continue
        try:
            chain.append(cls())
        except Exception:
            logger.exception("MC filter %r failed to instantiate — skipping", name)
    return chain


def _reset_chain_cache() -> None:
    """Test helper: drop the cached chain so the next apply_* rebuilds."""
    global _chain, _chain_built
    _chain = []
    _chain_built = False


def apply_buy_filters(
    side: str,
    model_prob: float,
    pid: str,
    channels: List[List[float]],
    context: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Apply the MC filter chain at the BUY gate.

    Returns (final_side, telemetry_dict). HOLD/SELL side passes through
    untouched. With MC_FILTERS empty, returns (side, {}).
    """
    global _chain, _chain_built
    if side != "BUY":
        return side, {}
    if not _chain_built:
        _chain = _build_chain()
        _chain_built = True
    if not _chain:
        return side, {}
    telemetry: Dict[str, Any] = {}
    cur_side = side
    for f in _chain:
        try:
            cur_side, tele = f.evaluate(cur_side, model_prob, pid, channels, context)
            if tele:
                telemetry.update(tele)
        except Exception as exc:
            logger.warning(
                "MC filter %r raised %s — skipping its decision",
                getattr(f, "name", "unknown"), exc,
            )
    return cur_side, telemetry
