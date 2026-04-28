"""
Tests for services/okx_funding_history.py — OKX-sourced funding rate fetcher.

Replaces Binance funding history (geo-blocked from the US, see #80/#81).
OKX exposes funding-rate-history at:
    GET https://www.okx.com/api/v5/public/funding-rate-history?instId=BTC-USDT-SWAP

Differences vs Binance fetcher this test file covers:
  - Response is wrapped as {"code": "0", "data": [...], "msg": ""} — `code != "0"`
    means OKX rejected the request (treat like non-200).
  - Each row has `fundingTime` and `fundingRate` as STRINGS, not numbers.
  - Symbol convention: BTC-USDT-SWAP (USDT-margined perpetuals).
  - `limit` caps at 100 per call (vs Binance 1000), so pagination via `after=`
    cursor is mandatory for windows longer than ~33 days.

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

from services import okx_funding_history as okx  # noqa: E402


def _ok(data, code="0", msg=""):
    """Build an OKX-shaped 200 response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"code": code, "msg": msg, "data": data}
    resp.raise_for_status = MagicMock()
    return resp


def _http(status, body=None):
    """Build a non-200 response."""
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.raise_for_status = MagicMock()
    return resp


# ── Symbol mapping ───────────────────────────────────────────────────────────

class TestProductSymbolMapping:

    def test_known_products_map_to_okx_swap_ids(self):
        assert okx._coinbase_to_okx("BTC-USD")  == "BTC-USDT-SWAP"
        assert okx._coinbase_to_okx("ETH-USD")  == "ETH-USDT-SWAP"
        assert okx._coinbase_to_okx("SOL-USD")  == "SOL-USDT-SWAP"
        assert okx._coinbase_to_okx("DOGE-USD") == "DOGE-USDT-SWAP"

    def test_unsupported_product_returns_none(self):
        # Products without an OKX listing return None — caller treats as no data.
        assert okx._coinbase_to_okx("VVV-USD") is None
        assert okx._coinbase_to_okx("UNKNOWN-USD") is None


# ── Happy path: single-page fetch ────────────────────────────────────────────

class TestFetchFundingHistorySinglePage:

    @pytest.mark.asyncio
    async def test_returns_sorted_tuples_of_ms_time_and_float_rate(self):
        # OKX sends NEWEST first; both fields are STRINGS.
        payload = [
            {"instId": "BTC-USDT-SWAP",
             "fundingTime": "1700028800000", "fundingRate": "-0.00005"},
            {"instId": "BTC-USDT-SWAP",
             "fundingTime": "1700000000000", "fundingRate": "0.00010"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        # Sorted ascending (oldest first) and types are (int_ms, float).
        assert result == [
            (1_700_000_000_000, 0.00010),
            (1_700_028_800_000, -0.00005),
        ]

    @pytest.mark.asyncio
    async def test_passes_instid_and_window_to_okx(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            await okx.fetch_funding_history(
                "ETH-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        args, kwargs = mock_client.get.call_args
        assert "funding-rate-history" in args[0]
        params = kwargs.get("params") or {}
        assert params["instId"] == "ETH-USDT-SWAP"
        # First call should anchor at end_ms via `after=` (newest-first cursor).
        assert int(params["after"]) == 1_700_100_000_000
        assert int(params.get("limit", 100)) == 100


# ── Failure modes ────────────────────────────────────────────────────────────

class TestFetchFundingHistoryFailureModes:

    @pytest.mark.asyncio
    async def test_unsupported_product_returns_empty_no_http(self):
        with patch("services.okx_funding_history.httpx.AsyncClient") as MockClient:
            result = await okx.fetch_funding_history(
                "UNSUPPORTED-USD", start_ms=0, end_ms=1_000
            )
        assert result == []
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_on_non_200(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_http(429, {"err": "rate limit"}))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_okx_code_nonzero(self):
        # OKX returned 200 but `code != "0"` means logical error (e.g. bad instId).
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([], code="51001",
                                                    msg="Instrument id does not exist"))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_network_exception(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=Exception("dns failure"))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_malformed_rows(self):
        payload = [
            {"fundingTime": "1700000000000", "fundingRate": "0.0001"},
            {"fundingTime": "abc", "fundingRate": "0.0002"},   # bad ts
            {"fundingTime": "1700028800000"},                   # missing rate
            {"fundingTime": "1700057600000", "fundingRate": "0.0003"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        # Only the two well-formed rows survive, sorted ascending.
        assert result == [
            (1_700_000_000_000, 0.0001),
            (1_700_057_600_000, 0.0003),
        ]


# ── Kill switch ──────────────────────────────────────────────────────────────

class TestKillSwitch:

    @pytest.mark.asyncio
    async def test_disabled_env_var_short_circuits_without_http(self, monkeypatch):
        monkeypatch.setenv("OKX_FUNDING_DISABLED", "1")
        with patch("services.okx_funding_history.httpx.AsyncClient") as MockClient:
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_env_var_off_does_make_http_call(self, monkeypatch):
        monkeypatch.delenv("OKX_FUNDING_DISABLED", raising=False)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))
        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            await okx.fetch_funding_history("BTC-USD", start_ms=0, end_ms=1_000)
        mock_client.get.assert_called()


# ── Pagination via after= cursor ─────────────────────────────────────────────

class TestPagination:
    """OKX caps `limit` at 100; long windows require multiple calls."""

    @pytest.mark.asyncio
    async def test_paginates_backward_using_after_cursor(self):
        # Three pages of 100 rows each cover ~99 days. We simulate with
        # tiny pages of 2 rows each, walking back in 1ms steps so behaviour
        # is observable but fast.
        page1 = [
            {"fundingTime": "1000", "fundingRate": "0.0010"},
            {"fundingTime": "999",  "fundingRate": "0.0009"},
        ]
        page2 = [
            {"fundingTime": "998",  "fundingRate": "0.0008"},
            {"fundingTime": "997",  "fundingRate": "0.0007"},
        ]
        page3 = [
            {"fundingTime": "996",  "fundingRate": "0.0006"},
            {"fundingTime": "995",  "fundingRate": "0.0005"},
        ]
        page_empty: list = []

        responses = [_ok(page1), _ok(page2), _ok(page3), _ok(page_empty)]

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=responses)

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=995, end_ms=1000
            )

        # All three populated pages should have been merged + sorted ascending.
        times = [t for t, _ in result]
        assert times == [995, 996, 997, 998, 999, 1000]

        # Each call past the first should have used `after=oldest_seen` to walk
        # back in time.
        calls = mock_client.get.call_args_list
        assert len(calls) >= 3
        # First call anchors at end_ms.
        assert int(calls[0].kwargs["params"]["after"]) == 1000
        # Second call's `after` equals the oldest fundingTime from page1 (999).
        assert int(calls[1].kwargs["params"]["after"]) == 999
        # Third call's `after` equals the oldest from page2 (997).
        assert int(calls[2].kwargs["params"]["after"]) == 997

    @pytest.mark.asyncio
    async def test_stops_paginating_when_oldest_ts_below_start_ms(self):
        # Page 1 has rows newer than start_ms — fetcher should NOT call page 2.
        page1 = [
            {"fundingTime": "5000", "fundingRate": "0.0001"},
            {"fundingTime": "4000", "fundingRate": "0.0002"},
            {"fundingTime": "3000", "fundingRate": "0.0003"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(page1))

        with patch("services.okx_funding_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_funding_history(
                "BTC-USD", start_ms=4000, end_ms=5000
            )

        # Only one HTTP call — oldest in page1 (3000) < start_ms (4000),
        # so no further pagination.
        mock_client.get.assert_called_once()
        # And the row below start_ms is filtered out of the result.
        assert [t for t, _ in result] == [4000, 5000]
