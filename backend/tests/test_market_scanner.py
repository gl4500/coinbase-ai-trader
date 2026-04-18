"""
Tests for MarketScanner minimum-price filter.

Micro-priced tokens (< $0.001) have spreads that wipe out agent profit targets
and produce degenerate CNN features — they must be excluded from tracking.
"""
import os
import sys
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from agents.market_scanner import MarketScanner, MIN_PRICE


def _make_product(pid: str, price: float, volume: float = 5_000_000) -> dict:
    return {
        "product_id":                     pid,
        "quote_currency_id":              "USD",
        "base_currency_id":               pid.split("-")[0],
        "display_name":                   pid,
        "status":                         "online",
        "trading_disabled":               False,
        "volume_24h":                     str(volume),
        "price":                          str(price),
        "price_percentage_change_24h":    "0.5",
    }


class TestMinPriceConstant:
    def test_min_price_exists_and_is_reasonable(self):
        assert MIN_PRICE > 0, "MIN_PRICE must be positive"
        assert MIN_PRICE <= 0.01, "MIN_PRICE should be ≤ $0.01 to allow small-cap alts"
        assert MIN_PRICE >= 0.0001, "MIN_PRICE should be ≥ $0.0001 to exclude micro tokens"


class TestScannerPriceFilter:
    """_do_scan must exclude products whose price is below MIN_PRICE."""

    def _mock_bba(self, products: list) -> dict:
        return {
            p["product_id"]: {"price": float(p["price"]), "bid": float(p["price"]) * 0.999, "ask": float(p["price"]) * 1.001}
            for p in products
        }

    @pytest.mark.asyncio
    async def test_micro_price_product_not_tracked(self):
        """A product priced at $0.000012 (e.g. SHIB) must not be tracked."""
        products = [
            _make_product("XRP-USD",  price=1.33),
            _make_product("SHIB-USD", price=0.000012),   # below MIN_PRICE
        ]
        scanner = MarketScanner()

        with (
            patch("agents.market_scanner.coinbase_client.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.market_scanner.coinbase_client.get_best_bid_ask",
                  new=AsyncMock(return_value=self._mock_bba(products))),
            patch("agents.market_scanner.coinbase_client.get_candles",
                  new=AsyncMock(return_value=[])),
            patch("agents.market_scanner.database.upsert_product", new=AsyncMock()),
            patch("agents.market_scanner.database.save_candles",   new=AsyncMock()),
        ):
            result = await scanner._do_scan()

        pids = {p["product_id"] for p in result}
        assert "XRP-USD"  in pids,  "XRP-USD should be tracked"
        assert "SHIB-USD" not in pids, "SHIB-USD price < MIN_PRICE — must be excluded"
        assert "SHIB-USD" not in scanner.tracked_ids

    @pytest.mark.asyncio
    async def test_product_exactly_at_min_price_is_tracked(self):
        """A product at exactly MIN_PRICE with sufficient volume should be included (boundary)."""
        # volume * price must exceed min_volume_24h ($1M); at $0.01 price need 100M base units
        products = [_make_product("EDGE-USD", price=MIN_PRICE, volume=100_000_000)]
        scanner = MarketScanner()

        with (
            patch("agents.market_scanner.coinbase_client.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.market_scanner.coinbase_client.get_best_bid_ask",
                  new=AsyncMock(return_value=self._mock_bba(products))),
            patch("agents.market_scanner.coinbase_client.get_candles",
                  new=AsyncMock(return_value=[])),
            patch("agents.market_scanner.database.upsert_product", new=AsyncMock()),
            patch("agents.market_scanner.database.save_candles",   new=AsyncMock()),
        ):
            result = await scanner._do_scan()

        assert any(p["product_id"] == "EDGE-USD" for p in result)

    @pytest.mark.asyncio
    async def test_all_micro_price_returns_empty(self):
        """If all products are micro-priced, tracked_ids should be empty."""
        products = [
            _make_product("BONK-USD", price=0.0000139),
            _make_product("PEPE-USD", price=0.0000104),
        ]
        scanner = MarketScanner()

        with (
            patch("agents.market_scanner.coinbase_client.get_products",
                  new=AsyncMock(return_value=products)),
            patch("agents.market_scanner.coinbase_client.get_best_bid_ask",
                  new=AsyncMock(return_value=self._mock_bba(products))),
            patch("agents.market_scanner.coinbase_client.get_candles",
                  new=AsyncMock(return_value=[])),
            patch("agents.market_scanner.database.upsert_product", new=AsyncMock()),
            patch("agents.market_scanner.database.save_candles",   new=AsyncMock()),
        ):
            result = await scanner._do_scan()

        assert result == []
        assert len(scanner.tracked_ids) == 0


class TestAgentPriceGuard:
    """Each agent's entry logic must skip products with price < MIN_PRICE."""

    @pytest.mark.asyncio
    async def test_cnn_skips_micro_price(self):
        """CNN generate_signal returns None for price < MIN_PRICE."""
        from agents.cnn_agent import CoinbaseCNNAgent, MIN_PRICE as CNN_MIN_PRICE

        agent = CoinbaseCNNAgent()
        product = {
            "product_id": "SHIB-USD",
            "price": 0.000012,   # below MIN_PRICE
        }
        result = await agent.generate_signal(product)
        assert result is None, "CNN must not signal on micro-priced assets"

    @pytest.mark.asyncio
    async def test_scalp_skips_micro_price_in_scan(self):
        """ScalpAgent._get_scalp_products must exclude products with price < MIN_PRICE."""
        from agents.scalp_agent import ScalpAgent

        ag = ScalpAgent()
        tracked = [
            {"product_id": "XRP-USD",  "price": 1.33},
            {"product_id": "SHIB-USD", "price": 0.000012},
        ]

        async def fake_get_candles(pid, limit):
            return [{}] * 120

        with (
            patch("agents.scalp_agent.database.get_products",
                  new=AsyncMock(return_value=tracked)),
            patch("agents.scalp_agent.database.get_candles",
                  side_effect=fake_get_candles),
        ):
            result = await ag._get_scalp_products()

        assert "XRP-USD"  in result
        assert "SHIB-USD" not in result
