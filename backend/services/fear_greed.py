"""
Fear & Greed Index macro gate.

Fetches the CNN Fear & Greed Index (0–100) from alternative.me.
Caches the result for `cache_ttl` seconds (default 10 min) to avoid
hammering the free API.

Usage:
    fg = get_fear_greed()
    if not await fg.is_buy_allowed():
        return  # suppress all BUY entries during extreme fear

Value ranges:
  0–24   Extreme Fear  → suppress BUY (high probability of continued downturn)
  25–44  Fear          → allow with caution
  45–55  Neutral       → allow
  56–74  Greed         → allow
  75–100 Extreme Greed → allow (but consider tighter take-profits)
"""
import logging
import time
from typing import Dict

import httpx

logger = logging.getLogger(__name__)

_API_URL              = "https://api.alternative.me/fng/"
_EXTREME_FEAR_THRESHOLD = 20          # BUY suppressed when value < this
_DEFAULT_CACHE_TTL    = 600           # 10 minutes


class FearGreedIndex:
    def __init__(self, cache_ttl: int = _DEFAULT_CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._cache: Dict | None = None
        self._cache_ts: float = 0.0

    async def fetch(self) -> Dict:
        """
        Return {"value": int, "label": str}.
        Uses cache; on any network failure returns neutral {"value": 50, "label": "Unknown"}.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        try:
            async with httpx.AsyncClient(timeout=8) as client:
                resp = await client.get(_API_URL, params={"limit": 1})
                resp.raise_for_status()
                data = resp.json()["data"][0]
                result = {
                    "value": int(data["value"]),
                    "label": data["value_classification"],
                }
        except Exception as e:
            logger.warning(f"FearGreed fetch failed: {e} — returning neutral 50")
            result = {"value": 50, "label": "Unknown"}

        self._cache    = result
        self._cache_ts = now
        return result

    async def is_buy_allowed(self) -> bool:
        """Returns False only during extreme fear (value < threshold)."""
        data = await self.fetch()
        allowed = data["value"] >= _EXTREME_FEAR_THRESHOLD
        if not allowed:
            logger.info(
                f"FearGreed gate: BUY suppressed | value={data['value']} "
                f"label='{data['label']}' threshold={_EXTREME_FEAR_THRESHOLD}"
            )
        return allowed

    async def is_sell_allowed(self) -> bool:
        """SELL is always allowed (must be able to exit any position)."""
        return True


# Module-level singleton
_instance: FearGreedIndex | None = None


def get_fear_greed() -> FearGreedIndex:
    global _instance
    if _instance is None:
        _instance = FearGreedIndex()
    return _instance
