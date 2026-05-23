"""
Tests for services/okx_oi_history.py — OKX-sourced open-interest history fetcher.

This is the OI sibling of services/okx_funding_history (#86 / #88). OKX exposes
per-instrument historical open interest at:
    GET https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-history?instId=BTC-USDT-SWAP&period=1H

The fetcher mirrors the funding-history contract:
  - Public coroutine: fetch_oi_history(product_id, start_ms, end_ms, bar="1H")
      -> list[(ts_ms, oi_contracts)] sorted ascending
  - OKX response wrapper: {"code": "0", "msg": "", "data": [...]}
  - Each row's `ts` and `oi` (contracts) are STRINGS.
  - `limit` caps at 100 per call → pagination via `after=<ts_ms>` returns rows
    OLDER than the cursor.
  - Symbol convention: USDT-margined SWAP, e.g. BTC-USDT-SWAP.
  - Kill switch: env OKX_OI_DISABLED=1 short-circuits without an HTTP call.

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

from services import okx_oi_history as okx  # noqa: E402


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


# ── Symbol mapping (mirrors okx_funding_history) ────────────────────────────

class TestProductSymbolMapping:

    def test_known_products_map_to_okx_swap_ids(self):
        assert okx._coinbase_to_okx("BTC-USD")  == "BTC-USDT-SWAP"
        assert okx._coinbase_to_okx("ETH-USD")  == "ETH-USDT-SWAP"
        assert okx._coinbase_to_okx("SOL-USD")  == "SOL-USDT-SWAP"
        assert okx._coinbase_to_okx("DOGE-USD") == "DOGE-USDT-SWAP"

    def test_unsupported_product_returns_none(self):
        assert okx._coinbase_to_okx("VVV-USD") is None
        assert okx._coinbase_to_okx("UNKNOWN-USD") is None

    def test_alt_meme_pids_added_per_211(self):
        """#211 fix: top-N survivorship-aware alts/memes that #210 found
        all-zero in the cache must now resolve to their OKX SWAP instIds.
        Verified live on OKX SWAP listing 2026-05-08."""
        expected = {
            "PENGU-USD":   "PENGU-USDT-SWAP",
            "JTO-USD":     "JTO-USDT-SWAP",
            "POPCAT-USD":  "POPCAT-USDT-SWAP",
            "BONK-USD":    "BONK-USDT-SWAP",
            "ZK-USD":      "ZK-USDT-SWAP",
            "PEPE-USD":    "PEPE-USDT-SWAP",
            "MOODENG-USD": "MOODENG-USDT-SWAP",
            "ONDO-USD":    "ONDO-USDT-SWAP",
            "ALGO-USD":    "ALGO-USDT-SWAP",
            "ZORA-USD":    "ZORA-USDT-SWAP",
            "WIF-USD":     "WIF-USDT-SWAP",
            "RENDER-USD":  "RENDER-USDT-SWAP",
            "FLOKI-USD":   "FLOKI-USDT-SWAP",
            "WLD-USD":     "WLD-USDT-SWAP",
            "BERA-USD":    "BERA-USDT-SWAP",
            "ENA-USD":     "ENA-USDT-SWAP",
            "STRK-USD":    "STRK-USDT-SWAP",
            "TON-USD":     "TON-USDT-SWAP",
            "JUP-USD":     "JUP-USDT-SWAP",
        }
        for pid, inst_id in expected.items():
            assert okx._coinbase_to_okx(pid) == inst_id, (pid, inst_id)

    def test_alts_with_no_okx_swap_still_return_none(self):
        """#211: pids that #210 flagged all-zero AND probe confirmed are
        NOT listed on OKX SWAP must continue returning None (rather than
        being added speculatively, which would just trigger wasted HTTP
        calls)."""
        for pid in (
            "NKN-USD", "AIOZ-USD", "JASMY-USD", "TRU-USD",
            "SKL-USD", "FET-USD", "XCN-USD", "LRDS-USD",
        ):
            assert okx._coinbase_to_okx(pid) is None, pid


# ── Happy path: single-page fetch ────────────────────────────────────────────

class TestFetchOIHistorySinglePage:

    @pytest.mark.asyncio
    async def test_returns_sorted_tuples_of_ms_time_and_float_oi(self):
        # OKX sends NEWEST first; both fields are STRINGS.
        payload = [
            {"instId": "BTC-USDT-SWAP",
             "ts": "1700003600000", "oi": "123456.78", "oiCcy": "12.34"},
            {"instId": "BTC-USDT-SWAP",
             "ts": "1700000000000", "oi": "120000.00", "oiCcy": "12.00"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        # Sorted ascending (oldest first); types are (int_ms, float).
        assert result == [
            (1_700_000_000_000, 120000.00),
            (1_700_003_600_000, 123456.78),
        ]

    @pytest.mark.asyncio
    async def test_passes_instid_period_and_window_to_okx(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            await okx.fetch_oi_history(
                "ETH-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        args, kwargs = mock_client.get.call_args
        assert "open-interest-history" in args[0]
        params = kwargs.get("params") or {}
        assert params["instId"] == "ETH-USDT-SWAP"
        assert params["period"] == "1H"
        # First call should anchor at end_ms via `after=` (newest-first cursor).
        assert int(params["after"]) == 1_700_100_000_000
        assert int(params.get("limit", 100)) == 100

    @pytest.mark.asyncio
    async def test_bar_kwarg_overrides_default_period(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            await okx.fetch_oi_history(
                "BTC-USD",
                start_ms=0, end_ms=1_000,
                bar="4H",
            )

        params = mock_client.get.call_args.kwargs["params"]
        assert params["period"] == "4H"


# ── Failure modes ────────────────────────────────────────────────────────────

class TestFetchOIHistoryFailureModes:

    @pytest.mark.asyncio
    async def test_unsupported_product_returns_empty_no_http(self):
        with patch("services.okx_oi_history.httpx.AsyncClient") as MockClient:
            result = await okx.fetch_oi_history(
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

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
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

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_network_exception(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=Exception("dns failure"))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_malformed_rows(self):
        payload = [
            {"ts": "1700000000000", "oi": "100.0"},
            {"ts": "abc", "oi": "200.0"},                  # bad ts
            {"ts": "1700003600000"},                        # missing oi
            {"ts": "1700007200000", "oi": "300.0"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        # Only the two well-formed rows survive, sorted ascending.
        assert result == [
            (1_700_000_000_000, 100.0),
            (1_700_007_200_000, 300.0),
        ]


# ── Live response shape: rows are arrays, not dicts ─────────────────────────

class TestParsesArrayRowShape:
    """OKX returns rows as positional arrays [ts, oi, oiCcy, oiUsd] — not dicts.
    The original tests mocked dicts; live calls returned [] silently because
    `_parse_rows` did `row["ts"]` against a list. Parser must handle arrays."""

    @pytest.mark.asyncio
    async def test_parses_array_rows_from_live_okx_shape(self):
        # Real OKX shape: ["ts_ms", "oi_contracts", "oi_ccy", "oi_usd"] (all strings).
        payload = [
            ["1700003600000", "123456.78", "12345.6", "1500000.0"],
            ["1700000000000", "120000.00", "12000.0", "1450000.0"],
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        assert result == [
            (1_700_000_000_000, 120000.00),
            (1_700_003_600_000, 123456.78),
        ]

    @pytest.mark.asyncio
    async def test_skips_malformed_array_rows(self):
        payload = [
            ["1700000000000", "100.0", "x", "y"],
            ["abc", "200.0", "x", "y"],          # bad ts
            ["1700003600000"],                    # too short
            ["1700007200000", "300.0", "x", "y"],
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        assert result == [
            (1_700_000_000_000, 100.0),
            (1_700_007_200_000, 300.0),
        ]


# ── Kill switch ──────────────────────────────────────────────────────────────

class TestKillSwitch:

    @pytest.mark.asyncio
    async def test_disabled_env_var_short_circuits_without_http(self, monkeypatch):
        monkeypatch.setenv("OKX_OI_DISABLED", "1")
        with patch("services.okx_oi_history.httpx.AsyncClient") as MockClient:
            result = await okx.fetch_oi_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_env_var_off_does_make_http_call(self, monkeypatch):
        monkeypatch.delenv("OKX_OI_DISABLED", raising=False)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))
        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            await okx.fetch_oi_history("BTC-USD", start_ms=0, end_ms=1_000)
        mock_client.get.assert_called()


# ── Pagination via after= cursor ─────────────────────────────────────────────

class TestPagination:
    """OKX caps `limit` at 100; long windows require multiple calls."""

    @pytest.mark.asyncio
    async def test_paginates_backward_using_after_cursor(self):
        page1 = [
            {"ts": "1000", "oi": "10.0"},
            {"ts": "999",  "oi": "9.0"},
        ]
        page2 = [
            {"ts": "998",  "oi": "8.0"},
            {"ts": "997",  "oi": "7.0"},
        ]
        page3 = [
            {"ts": "996",  "oi": "6.0"},
            {"ts": "995",  "oi": "5.0"},
        ]
        page_empty: list = []

        responses = [_ok(page1), _ok(page2), _ok(page3), _ok(page_empty)]

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=responses)

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD", start_ms=995, end_ms=1000
            )

        times = [t for t, _ in result]
        assert times == [995, 996, 997, 998, 999, 1000]

        calls = mock_client.get.call_args_list
        assert len(calls) >= 3
        # First call anchors at end_ms.
        assert int(calls[0].kwargs["params"]["after"]) == 1000
        # Second call's `after` equals the oldest ts from page1 (999).
        assert int(calls[1].kwargs["params"]["after"]) == 999
        # Third call's `after` equals the oldest from page2 (997).
        assert int(calls[2].kwargs["params"]["after"]) == 997

    @pytest.mark.asyncio
    async def test_stops_paginating_when_oldest_ts_below_start_ms(self):
        page1 = [
            {"ts": "5000", "oi": "1.0"},
            {"ts": "4000", "oi": "2.0"},
            {"ts": "3000", "oi": "3.0"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(page1))

        with patch("services.okx_oi_history.httpx.AsyncClient",
                   return_value=mock_client):
            result = await okx.fetch_oi_history(
                "BTC-USD", start_ms=4000, end_ms=5000
            )

        # Only one HTTP call — oldest in page1 (3000) < start_ms (4000).
        mock_client.get.assert_called_once()
        # Row below start_ms is filtered out.
        assert [t for t, _ in result] == [4000, 5000]
