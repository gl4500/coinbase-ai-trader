"""Tests for services/coinpaprika_marketcap.py — free-tier (no key) marketcap
history fetcher (#293).

Mirrors the public surface of services/coingecko_marketcap.py so the probe
harness and parquet builder can swap providers via a --source flag.

Endpoints (CoinPaprika, free tier 25k req/day, no key required):
  GET https://api.coinpaprika.com/v1/coins/{cp_id}/ohlcv/historical?start=YYYY-MM-DD&end=YYYY-MM-DD
      one row per day, includes market_cap field

Public API:
  await fetch_marketcap_history(pid: str, start_ms: int, end_ms: int)
      -> list[(ts_ms, market_cap)] sorted ascending
  _coinbase_to_cp_id(pid) -> Optional[str]   ("BTC-USD" -> "btc-bitcoin")

Kill switch: env COINPAPRIKA_DISABLED=1 short-circuits without an HTTP call.
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

from services import coinpaprika_marketcap as cp  # noqa: E402


def _ok(body):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = body
    resp.raise_for_status = MagicMock()
    return resp


def _http(status, body=None):
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = body or {}
    resp.raise_for_status = MagicMock()
    return resp


# ── Coinbase pid → CoinPaprika id mapping ───────────────────────────────────


class TestIdMapping:
    def test_known_pids_map_to_coinpaprika_ids(self):
        assert cp._coinbase_to_cp_id("BTC-USD") == "btc-bitcoin"
        assert cp._coinbase_to_cp_id("ETH-USD") == "eth-ethereum"
        assert cp._coinbase_to_cp_id("SOL-USD") == "sol-solana"
        assert cp._coinbase_to_cp_id("AVAX-USD") == "avax-avalanche"
        assert cp._coinbase_to_cp_id("LINK-USD") == "link-chainlink"

    def test_unknown_pid_returns_none(self):
        assert cp._coinbase_to_cp_id("DOES-NOT-EXIST-USD") is None
        assert cp._coinbase_to_cp_id("") is None
        assert cp._coinbase_to_cp_id(None) is None


# ── Historical timeseries ───────────────────────────────────────────────────


class TestFetchMarketcapHistory:
    @pytest.mark.asyncio
    async def test_returns_sorted_ts_market_cap_pairs(self):
        body = [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "market_cap": 1_000_000_000_000,
                "open": 90000,
                "high": 91000,
                "low": 89000,
                "close": 90500,
                "volume": 1.0,
            },
            {
                "timestamp": "2025-01-02T00:00:00Z",
                "market_cap": 1_010_000_000_000,
                "open": 90500,
                "high": 91500,
                "low": 90000,
                "close": 91000,
                "volume": 1.0,
            },
            {
                "timestamp": "2025-01-03T00:00:00Z",
                "market_cap": 1_020_000_000_000,
                "open": 91000,
                "high": 92000,
                "low": 90500,
                "close": 91500,
                "volume": 1.0,
            },
        ]
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=_ok(body))
            rows = await cp.fetch_marketcap_history(
                "BTC-USD", start_ms=1735689600000, end_ms=1735862400000
            )
        assert len(rows) == 3
        assert rows == sorted(rows, key=lambda r: r[0])
        assert rows[0][0] == 1735689600 * 1000
        assert rows[0][1] == 1_000_000_000_000
        assert rows[2][1] == 1_020_000_000_000

    @pytest.mark.asyncio
    async def test_unmapped_pid_returns_empty(self):
        rows = await cp.fetch_marketcap_history("UNMAPPED-USD", start_ms=0, end_ms=1)
        assert rows == []

    @pytest.mark.asyncio
    async def test_disabled_env_short_circuits_without_http(self):
        with patch.dict(os.environ, {"COINPAPRIKA_DISABLED": "1"}):
            with patch.object(cp.httpx, "AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=_ok([])
                )
                rows = await cp.fetch_marketcap_history("BTC-USD", start_ms=0, end_ms=1)
                assert rows == []
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_http_429_returns_empty_no_raise(self):
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_http(429)
            )
            rows = await cp.fetch_marketcap_history("BTC-USD", start_ms=0, end_ms=1)
            assert rows == []

    @pytest.mark.asyncio
    async def test_transport_error_returns_empty_no_raise(self):
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("network down")
            )
            rows = await cp.fetch_marketcap_history("BTC-USD", start_ms=0, end_ms=1)
            assert rows == []

    @pytest.mark.asyncio
    async def test_skips_rows_with_missing_or_non_positive_market_cap(self):
        body = [
            {"timestamp": "2025-01-01T00:00:00Z", "market_cap": 1_000_000_000_000},
            {"timestamp": "2025-01-02T00:00:00Z", "market_cap": None},
            {"timestamp": "2025-01-03T00:00:00Z", "market_cap": 0},
            {"timestamp": "2025-01-04T00:00:00Z", "market_cap": 1_020_000_000_000},
        ]
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=_ok(body))
            rows = await cp.fetch_marketcap_history("BTC-USD", start_ms=0, end_ms=10**12)
        assert len(rows) == 2
        assert rows[0][1] == 1_000_000_000_000
        assert rows[1][1] == 1_020_000_000_000

    @pytest.mark.asyncio
    async def test_passes_iso_date_and_interval_params(self):
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=_ok([]))
            mock_client.return_value.__aenter__.return_value.get = mock_get
            await cp.fetch_marketcap_history(
                "BTC-USD",
                start_ms=1735689600 * 1000,  # 2025-01-01
                end_ms=1735862400 * 1000,  # 2025-01-03
            )
            assert mock_get.called
            params = mock_get.call_args.kwargs["params"]
            assert params["start"] == "2025-01-01"
            assert params["interval"] == "1d"

    @pytest.mark.asyncio
    async def test_uses_tickers_historical_endpoint(self):
        """Free tier — must hit /v1/tickers/{id}/historical, not the paid
        /v1/coins/{id}/ohlcv/historical endpoint."""
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_get = AsyncMock(return_value=_ok([]))
            mock_client.return_value.__aenter__.return_value.get = mock_get
            await cp.fetch_marketcap_history(
                "BTC-USD",
                start_ms=1735689600 * 1000,
                end_ms=1735862400 * 1000,
            )
            url = mock_get.call_args.args[0]
            assert "/tickers/btc-bitcoin/historical" in url
            assert "/ohlcv/" not in url


# ── volume_24h extraction (Step A: marketcap bronze v2) ──────────────────────


class TestVolume24hExtraction:
    @pytest.mark.asyncio
    async def test_coinpaprika_parses_volume_24h(self):
        """Fetcher must map the `volume_24h` field from each historical row
        to a volume_24h value in the returned tuple. CoinPaprika
        /v1/tickers/{id}/historical returns rows with keys:
        timestamp, price, volume_24h, market_cap."""
        body = [
            {
                "timestamp": "2025-01-01T00:00:00Z",
                "market_cap": 1.0e9,
                "volume_24h": 5.0e7,
                "price": 100.0,
            },
            {
                "timestamp": "2025-01-02T00:00:00Z",
                "market_cap": 1.01e9,
                "volume_24h": 6.0e7,
                "price": 101.0,
            },
        ]
        with patch.object(cp.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=_ok(body))
            rows = await cp.fetch_marketcap_history(
                "BTC-USD", start_ms=1735689600000, end_ms=1735862400000
            )
        assert len(rows) == 2
        # Returned shape: (ts_ms, market_cap, volume_24h)
        assert all(len(r) == 3 for r in rows)
        assert rows[0][1] == 1.0e9
        assert rows[0][2] == 5.0e7
        assert rows[1][2] == 6.0e7


def test_product_to_cp_id_has_at_least_100_entries():
    """The mapping was extended in 2026-05-23 to support the strategy-discovery
    rebuild's 50-pid universe curation (needs a ~100+ candidate pool)."""
    from services.coinpaprika_marketcap import _PRODUCT_TO_CP_ID

    assert len(_PRODUCT_TO_CP_ID) >= 100, (
        f"_PRODUCT_TO_CP_ID has only {len(_PRODUCT_TO_CP_ID)} entries; "
        f"Phase 1 Task 4 requires >= 100 for universe curation candidate pool"
    )
