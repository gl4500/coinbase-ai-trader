"""
Binance Futures sentiment — top trader long/short position ratio.

Contrarian signal: heavy long positioning = crowd is long = bearish lean.
sentiment_score = long_ratio - short_ratio, normalised to [-1, 1].
Cached for 5 minutes (Binance updates every 1h anyway).
"""
import logging
import time
from typing import Dict, Optional

import httpx

logger = logging.getLogger(__name__)

_URL      = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
_CACHE_TTL = 300
_cache: Dict[str, tuple] = {}   # binance_symbol -> (timestamp, score)

# Map Coinbase product_id → Binance futures symbol
_PRODUCT_TO_BN = {
    "BTC-USD": "BTCUSDT",
    "ETH-USD": "ETHUSDT",
    "SOL-USD": "SOLUSDT",
    "XRP-USD": "XRPUSDT",
    "BNB-USD": "BNBUSDT",
    "ADA-USD": "ADAUSDT",
    "AVAX-USD": "AVAXUSDT",
    "LINK-USD": "LINKUSDT",
    "DOT-USD": "DOTUSDT",
    "MATIC-USD": "MATICUSDT",
}


async def get_ls_sentiment(product_id: str) -> Optional[float]:
    """
    Return top-trader sentiment score in [-1, 1].
    Positive = more longs than shorts (contrarian bearish lean).
    Negative = more shorts (contrarian bullish lean).
    Returns None if product not on Binance futures or request fails.
    """
    bn_sym = _PRODUCT_TO_BN.get(product_id)
    if not bn_sym:
        return None

    cached = _cache.get(bn_sym)
    if cached and time.time() - cached[0] < _CACHE_TTL:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(_URL, params={"symbol": bn_sym, "period": "1h", "limit": 1})
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None
            row = data[0]
            long_ratio  = float(row["longAccount"])
            short_ratio = float(row["shortAccount"])
            score = max(-1.0, min(1.0, long_ratio - short_ratio))
            _cache[bn_sym] = (time.time(), score)
            return score
    except Exception as e:
        logger.debug("Binance L/S ratio unavailable for %s: %s", product_id, e)
        return None
