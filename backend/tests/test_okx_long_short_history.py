"""
Tests for services/okx_long_short_history.py — OKX long/short account-ratio
fetcher.

Sibling of services/okx_oi_history (#141 / #142). OKX exposes per-instrument
historical long/short *account* ratio at:
    GET https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio-contract
        ?instId=BTC-USDT-SWAP&period=1H

(NOT the currency-level `/long-short-account-ratio` endpoint — that one
takes `ccy=BTC`, returns coarser precision, and is keyed off coin codes
rather than per-instrument SWAP ids; see #235g for the discovery.)

The fetcher mirrors the OI-history contract:
  - Public coroutine: fetch_long_short_ratio_history(product_id, start_ms,
        end_ms, bar="1H") -> list[(ts_ms, ratio_float)] sorted ascending
  - OKX response wrapper: {"code": "0", "msg": "", "data": [...]}
  - Each row is a positional array ["ts_ms", "ratio"] (both strings on the
    live API). Dict shape {"ts": "...", "ratio": "..."} also accepted so test
    fixtures stay readable.
  - `limit` caps at 100 per call → pagination via `after=<ts_ms>` returns rows
    OLDER than the cursor.
  - Symbol convention: USDT-margined SWAP, e.g. BTC-USDT-SWAP — same map as
    okx_oi_history.
  - Kill switch: env OKX_LS_DISABLED=1 short-circuits without an HTTP call.

No live API calls — httpx.AsyncClient.get is mocked in every async test.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from services import okx_long_short_history as okx_ls  # noqa: E402


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


# ── Symbol mapping (must match okx_oi_history exactly) ──────────────────────


class TestProductSymbolMapping:
    def test_known_products_map_to_okx_swap_ids(self):
        assert okx_ls._coinbase_to_okx("BTC-USD") == "BTC-USDT-SWAP"
        assert okx_ls._coinbase_to_okx("ETH-USD") == "ETH-USDT-SWAP"
        assert okx_ls._coinbase_to_okx("SOL-USD") == "SOL-USDT-SWAP"
        assert okx_ls._coinbase_to_okx("DOGE-USD") == "DOGE-USDT-SWAP"

    def test_unsupported_product_returns_none(self):
        assert okx_ls._coinbase_to_okx("VVV-USD") is None
        assert okx_ls._coinbase_to_okx("UNKNOWN-USD") is None

    def test_supported_set_matches_oi_history(self):
        """L/S map must mirror OI map: same symbol set, same OKX SWAP ids.
        If the two diverge, single-add probes that pair OI + L/S as Ch 27/28
        will silently coverage-mask different per-pid subsets."""
        from services import okx_oi_history as okx_oi

        assert okx_ls._PRODUCT_TO_OKX == okx_oi._PRODUCT_TO_OKX


# ── Happy path: single-page fetch ────────────────────────────────────────────


class TestFetchLSRatioHistorySinglePage:
    @pytest.mark.asyncio
    async def test_returns_sorted_tuples_of_ms_time_and_float_ratio(self):
        # OKX sends NEWEST first; both fields are STRINGS.
        payload = [
            ["1700003600000", "1.85"],
            ["1700000000000", "1.42"],
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        # Sorted ascending (oldest first); types are (int_ms, float).
        assert result == [
            (1_700_000_000_000, 1.42),
            (1_700_003_600_000, 1.85),
        ]

    @pytest.mark.asyncio
    async def test_passes_instid_period_and_window_to_okx(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            await okx_ls.fetch_long_short_ratio_history(
                "ETH-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        args, kwargs = mock_client.get.call_args
        assert args[0].endswith("/long-short-account-ratio-contract"), (
            f"Expected per-instrument endpoint, got {args[0]!r}. "
            "OKX has TWO L/S endpoints: `/long-short-account-ratio` (takes "
            "ccy=BTC) and `/long-short-account-ratio-contract` (takes "
            "instId=BTC-USDT-SWAP). We need the per-instrument one — see #235g."
        )
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

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD",
                start_ms=0,
                end_ms=1_000,
                bar="4H",
            )

        params = mock_client.get.call_args.kwargs["params"]
        assert params["period"] == "4H"

    @pytest.mark.asyncio
    async def test_accepts_dict_row_shape(self):
        """Test fixtures often supply dicts; the parser should handle both
        `[ts, ratio]` arrays and `{"ts": ..., "ratio": ...}` dicts."""
        payload = [
            {"ts": "1700003600000", "ratio": "1.85"},
            {"ts": "1700000000000", "ratio": "1.42"},
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        assert result == [
            (1_700_000_000_000, 1.42),
            (1_700_003_600_000, 1.85),
        ]


# ── Failure modes ────────────────────────────────────────────────────────────


class TestFetchLSRatioHistoryFailureModes:
    @pytest.mark.asyncio
    async def test_unsupported_product_returns_empty_no_http(self):
        with patch("services.okx_long_short_history.httpx.AsyncClient") as MockClient:
            result = await okx_ls.fetch_long_short_ratio_history(
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

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_okx_code_nonzero(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(
            return_value=_ok([], code="51001", msg="Instrument id does not exist")
        )

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_network_exception(self):
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=Exception("dns failure"))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []

    @pytest.mark.asyncio
    async def test_skips_malformed_array_rows(self):
        payload = [
            ["1700000000000", "1.42"],
            ["abc", "1.50"],  # bad ts
            ["1700003600000"],  # too short
            ["1700007200000", "1.80"],
        ]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(payload))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD",
                start_ms=1_700_000_000_000,
                end_ms=1_700_100_000_000,
            )

        assert result == [
            (1_700_000_000_000, 1.42),
            (1_700_007_200_000, 1.80),
        ]


# ── Kill switch ──────────────────────────────────────────────────────────────


class TestKillSwitch:
    @pytest.mark.asyncio
    async def test_disabled_env_var_short_circuits_without_http(self, monkeypatch):
        monkeypatch.setenv("OKX_LS_DISABLED", "1")
        with patch("services.okx_long_short_history.httpx.AsyncClient") as MockClient:
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=0, end_ms=1_000
            )
        assert result == []
        MockClient.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_env_var_off_does_make_http_call(self, monkeypatch):
        monkeypatch.delenv("OKX_LS_DISABLED", raising=False)
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok([]))
        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            await okx_ls.fetch_long_short_ratio_history("BTC-USD", start_ms=0, end_ms=1_000)
        mock_client.get.assert_called()


# ── Pagination via after= cursor ─────────────────────────────────────────────


class TestPagination:
    """OKX caps `limit` at 100; long windows require multiple calls."""

    @pytest.mark.asyncio
    async def test_paginates_backward_using_after_cursor(self):
        page1 = [["1000", "1.0"], ["999", "0.9"]]
        page2 = [["998", "0.8"], ["997", "0.7"]]
        page3 = [["996", "0.6"], ["995", "0.5"]]
        page_empty: list = []

        responses = [_ok(page1), _ok(page2), _ok(page3), _ok(page_empty)]

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(side_effect=responses)

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=995, end_ms=1000
            )

        times = [t for t, _ in result]
        assert times == [995, 996, 997, 998, 999, 1000]

        calls = mock_client.get.call_args_list
        assert len(calls) >= 3
        assert int(calls[0].kwargs["params"]["after"]) == 1000
        assert int(calls[1].kwargs["params"]["after"]) == 999
        assert int(calls[2].kwargs["params"]["after"]) == 997

    @pytest.mark.asyncio
    async def test_stops_paginating_when_oldest_ts_below_start_ms(self):
        page1 = [["5000", "1.0"], ["4000", "1.1"], ["3000", "1.2"]]
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.get = AsyncMock(return_value=_ok(page1))

        with patch("services.okx_long_short_history.httpx.AsyncClient", return_value=mock_client):
            result = await okx_ls.fetch_long_short_ratio_history(
                "BTC-USD", start_ms=4000, end_ms=5000
            )

        # Only one HTTP call — oldest in page1 (3000) < start_ms (4000).
        mock_client.get.assert_called_once()
        # Row below start_ms is filtered out.
        assert [t for t, _ in result] == [4000, 5000]
