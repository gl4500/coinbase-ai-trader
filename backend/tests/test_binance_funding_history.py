"""
Tests for services/binance_funding_history.py — historical funding rate fetcher (#57 stage b).

No live API calls — httpx.AsyncClient.get is mocked in every async test.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from services import binance_funding_history as bfh  # noqa: E402


def _make_response(data, status=200):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


class TestProductSymbolMapping:

    def test_known_products_map_to_binance_symbols(self):
        assert bfh._coinbase_to_binance("BTC-USD") == "BTCUSDT"
        assert bfh._coinbase_to_binance("ETH-USD") == "ETHUSDT"
        assert bfh._coinbase_to_binance("SOL-USD") == "SOLUSDT"

    def test_unknown_product_returns_none(self):
        assert bfh._coinbase_to_binance("UNSUPPORTED-USD") is None


class TestFetchFundingHistory:

    @pytest.mark.asyncio
    async def test_returns_sorted_tuples_of_time_and_rate(self):
        payload = [
            {"symbol": "BTCUSDT", "fundingTime": 1_700_000_000_000, "fundingRate": "0.00010"},
            {"symbol": "BTCUSDT", "fundingTime": 1_700_028_800_000, "fundingRate": "-0.00005"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_make_response(payload))

        with patch("services.binance_funding_history.httpx.AsyncClient", return_value=mock_client):
            result = await bfh.fetch_funding_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        assert result == [
            (1_700_000_000_000, 0.00010),
            (1_700_028_800_000, -0.00005),
        ]

    @pytest.mark.asyncio
    async def test_unsupported_product_returns_empty(self):
        result = await bfh.fetch_funding_history(
            "UNSUPPORTED-USD", start_ms=0, end_ms=1
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_symbol_and_window_to_binance(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_make_response([]))

        with patch("services.binance_funding_history.httpx.AsyncClient", return_value=mock_client):
            await bfh.fetch_funding_history(
                "ETH-USD", start_ms=1_700_000_000_000, end_ms=1_700_100_000_000
            )

        args, kwargs = mock_client.get.call_args
        assert "fundingRate" in args[0], "Should hit /fapi/v1/fundingRate"
        params = kwargs.get("params") or (args[1] if len(args) > 1 else {})
        assert params["symbol"] == "ETHUSDT"
        assert params["startTime"] == 1_700_000_000_000
        assert params["endTime"] == 1_700_100_000_000

    @pytest.mark.asyncio
    async def test_returns_empty_on_http_error(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=Exception("network down"))

        with patch("services.binance_funding_history.httpx.AsyncClient", return_value=mock_client):
            result = await bfh.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_200(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_make_response({"err": "rate limit"}, status=418))

        with patch("services.binance_funding_history.httpx.AsyncClient", return_value=mock_client):
            result = await bfh.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_sorts_unsorted_payload(self):
        payload = [
            {"symbol": "BTCUSDT", "fundingTime": 200, "fundingRate": "0.0002"},
            {"symbol": "BTCUSDT", "fundingTime": 100, "fundingRate": "0.0001"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_make_response(payload))

        with patch("services.binance_funding_history.httpx.AsyncClient", return_value=mock_client):
            result = await bfh.fetch_funding_history("BTC-USD", start_ms=0, end_ms=1000)

        assert [t for t, _ in result] == [100, 200]
