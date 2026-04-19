"""
Deribit IV Service — fetches nearest ATM implied volatility for BTC and ETH.

Computes iv_rv20_spread and iv_rv60_spread (IV minus realized vol) as
CNN feature channels. Positive spread = options overpriced (fear premium).
Results cached for 10 minutes to avoid hammering Deribit public API.
"""
import logging
import math
import time
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_DERIBIT_BASE = "https://www.deribit.com/api/v2/public"
_CACHE_TTL    = 600   # seconds
_cache: Dict[str, tuple] = {}   # currency -> (timestamp, iv_float)

# Map Coinbase product_id → Deribit currency
_PRODUCT_TO_CURRENCY = {
    "BTC-USD": "BTC",
    "ETH-USD": "ETH",
}


async def _fetch_atm_iv(currency: str, spot_price: float) -> Optional[float]:
    """Return nearest-ATM IV (0-1 scale) from Deribit summary endpoint."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{_DERIBIT_BASE}/get_book_summary_by_currency",
                params={"currency": currency, "kind": "option"},
            )
            resp.raise_for_status()
            instruments = resp.json().get("result", [])

        if not instruments:
            return None

        # Keep only calls with IV available and pick closest strike to spot
        candidates = [
            i for i in instruments
            if i.get("mark_iv") and i.get("instrument_name", "").endswith("-C")
        ]
        if not candidates:
            return None

        def _strike(inst: dict) -> float:
            parts = inst["instrument_name"].split("-")
            try:
                return float(parts[2])
            except (IndexError, ValueError):
                return float("inf")

        atm = min(candidates, key=lambda i: abs(_strike(i) - spot_price))
        iv_pct = atm.get("mark_iv")
        if iv_pct is None:
            return None
        return float(iv_pct) / 100.0   # Deribit returns percentage

    except Exception as e:
        logger.debug("Deribit IV fetch failed for %s: %s", currency, e)
        return None


async def get_iv(product_id: str, spot_price: float) -> Optional[float]:
    """
    Return cached ATM IV (0-1 scale) for the given product.
    Returns None for products without Deribit options data.
    """
    currency = _PRODUCT_TO_CURRENCY.get(product_id)
    if not currency:
        return None

    cached = _cache.get(currency)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1]

    iv = await _fetch_atm_iv(currency, spot_price)
    if iv is not None:
        _cache[currency] = (time.time(), iv)
    return iv


def compute_iv_rv_spreads(iv: float, rv20: float, rv60: float) -> Dict[str, float]:
    """
    Returns iv_rv20_spread and iv_rv60_spread, each clipped to [-1, 1].
    Positive = IV > RV (fear / overpriced options).
    Negative = IV < RV (complacency / underpriced options).
    Normalised by dividing by 1.0 (spreads rarely exceed ±100% annualised).
    """
    spread20 = max(-1.0, min(1.0, iv - rv20))
    spread60 = max(-1.0, min(1.0, iv - rv60))
    return {"iv_rv20_spread": round(spread20, 4),
            "iv_rv60_spread": round(spread60, 4)}
