"""
Market Scanner
──────────────
Discovers and tracks crypto trading pairs from Coinbase.
Fetches live prices, volume, and candle data for all tracked products.
Runs on startup and then every scan_interval seconds.
"""
import asyncio
import logging
from typing import Dict, List, Set

import database
from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)


class MarketScanner:
    def __init__(self):
        self.tracked_ids: Set[str] = set()
        self.is_scanning = False

    async def scan(self) -> List[Dict]:
        """
        Fetch product data for all tracked pairs, update DB.
        Returns the list of tracked products with current prices.
        """
        if self.is_scanning:
            return []
        self.is_scanning = True
        try:
            return await self._do_scan()
        finally:
            self.is_scanning = False

    async def _do_scan(self) -> List[Dict]:
        # Fetch all Coinbase SPOT products and dynamically discover USD pairs
        try:
            all_products = await coinbase_client.get_products()
        except Exception as e:
            logger.warning(f"Coinbase product fetch failed: {e}")
            all_products = []

        # Filter to online, tradeable USD pairs with sufficient volume
        max_tracked   = getattr(config, "max_tracked_products", 100)
        min_volume    = getattr(config, "min_volume_24h", 1_000_000)
        eligible = [
            p for p in all_products
            if p.get("quote_currency_id") == "USD"
            and p.get("status") == "online"
            and not p.get("trading_disabled", False)
            and float(p.get("volume_24h", 0) or 0) >= min_volume
        ]
        # Sort by 24h volume descending and cap
        eligible.sort(key=lambda p: float(p.get("volume_24h", 0) or 0), reverse=True)
        eligible = eligible[:max_tracked]

        self.tracked_ids = {p["product_id"] for p in eligible}
        if eligible:
            logger.info(f"Discovered {len(self.tracked_ids)} USD spot pairs (top by volume)")
        else:
            logger.warning("No eligible USD spot pairs found — check credentials / volume threshold")

        # Build lookup by product_id
        product_map: Dict[str, Dict] = {p.get("product_id", ""): p for p in all_products}

        # Also fetch best bid/ask for tracked pairs
        tracked_list = list(self.tracked_ids)
        try:
            bba = await coinbase_client.get_best_bid_ask(tracked_list)
        except Exception:
            bba = {}

        updated = []
        for pid in tracked_list:
            raw = product_map.get(pid, {})
            if not raw:
                logger.debug(f"Product {pid} not found in Coinbase catalogue")
                continue

            price_data = bba.get(pid, {})
            bid   = price_data.get("bid")
            ask   = price_data.get("ask")
            price = price_data.get("price") or float(raw.get("price", 0) or 0)
            spread = round(ask - bid, 4) if bid and ask else 0

            product = {
                "product_id":           pid,
                "base_currency":        raw.get("base_currency_id",  pid.split("-")[0]),
                "quote_currency":       raw.get("quote_currency_id", "USD"),
                "display_name":         raw.get("display_name",      pid),
                "price":                price,
                "price_pct_change_24h": float(raw.get("price_percentage_change_24h", 0) or 0),
                "volume_24h":           float(raw.get("volume_24h",  0) or 0),
                "high_24h":             0.0,
                "low_24h":              0.0,
                "spread":               spread,
                "is_tracked":           True,
            }
            await database.upsert_product(product)

            # Fetch and cache hourly candles (needed by signal generator + CNN)
            try:
                candles = await coinbase_client.get_candles(pid, "ONE_HOUR", limit=100)
                if candles:
                    await database.save_candles(pid, candles)
                    if candles:
                        closes = [c["close"] for c in candles]
                        product["high_24h"] = max(c["high"]  for c in candles[-24:])
                        product["low_24h"]  = min(c["low"]   for c in candles[-24:])
            except Exception as e:
                logger.debug(f"Candle fetch failed for {pid}: {e}")

            updated.append(product)
            await asyncio.sleep(0.2)   # gentle rate limiting

        logger.info(f"Scanner: updated {len(updated)} products")
        return updated

    async def refresh_prices(self) -> None:
        """Quick price refresh between full scans (no candle fetch)."""
        tracked = list(self.tracked_ids)
        try:
            bba = await coinbase_client.get_best_bid_ask(tracked)
            for pid, data in bba.items():
                if data.get("price"):
                    await database.update_product_price(
                        pid, data["price"], 0.0
                    )
        except Exception as e:
            logger.debug(f"Price refresh failed: {e}")

    async def run_loop(self) -> None:
        """
        Background loop:
         - Full scan (with candles) every config.scan_interval seconds
         - Price refresh every 30 seconds between full scans
        """
        scan_interval    = getattr(config, "scan_interval", 300)
        refresh_interval = 30
        elapsed          = 0

        logger.info(f"Market scanner loop started (full scan every {scan_interval}s)")
        while True:
            try:
                if elapsed == 0 or elapsed >= scan_interval:
                    await self.scan()
                    elapsed = 0
                else:
                    await self.refresh_prices()
            except asyncio.CancelledError:
                logger.info("Market scanner cancelled")
                return
            except Exception as e:
                logger.error(f"Scanner loop error: {e}")
            await asyncio.sleep(refresh_interval)
            elapsed += refresh_interval
