"""
Tests for services/coinmarketcap_marketcap.py — CoinMarketCap snapshot
fetcher (#280a).

Sibling of services/coingecko_marketcap (#260). Mirrors the same shape so
both providers can be A/B-compared via the marketcap probe harness, with
provider selection driven by env var.

Free-tier limitation:
    CoinMarketCap historical endpoints (/v1/cryptocurrency/quotes/historical,
    /v2/cryptocurrency/ohlcv/historical) require Hobbyist plan or higher
    ($29/mo+). On the Basic free tier only snapshot endpoints are available:
      GET /v1/cryptocurrency/quotes/latest?id=1,1027,...&convert=USD

    fetch_marketcap_history is therefore a free-tier no-op that returns []
    and emits a one-time warning. Probes that need history must continue
    using services.coingecko_marketcap.

Auth:
    X-CMC_PRO_API_KEY header sourced from env COINMARKETCAP_API_KEY.

Kill switch:
    env COINMARKETCAP_DISABLED=1 short-circuits without an HTTP call.
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
# Test default: a stub key so the module can construct headers; tests that
# care about absence patch this env var explicitly.
os.environ.setdefault("COINMARKETCAP_API_KEY",    "test-key")

from services import coinmarketcap_marketcap as cmc  # noqa: E402


# ── HTTP helpers ────────────────────────────────────────────────────────────

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


# ── Coinbase pid → CMC numeric id mapping ──────────────────────────────────

class TestIdMapping:

    def test_known_pids_map_to_cmc_numeric_ids(self):
        """CMC IDs are numeric and stable (unlike CoinGecko slugs which can
        rename). Hardcoded mapping verified live on CMC 2026-05-09."""
        assert cmc._coinbase_to_cmc_id("BTC-USD")  == 1
        assert cmc._coinbase_to_cmc_id("ETH-USD")  == 1027
        assert cmc._coinbase_to_cmc_id("SOL-USD")  == 5426
        assert cmc._coinbase_to_cmc_id("XRP-USD")  == 52
        assert cmc._coinbase_to_cmc_id("ADA-USD")  == 2010
        assert cmc._coinbase_to_cmc_id("LINK-USD") == 1975
        assert cmc._coinbase_to_cmc_id("DOGE-USD") == 74

    def test_unknown_pid_returns_none(self):
        assert cmc._coinbase_to_cmc_id("DOES-NOT-EXIST-USD") is None
        assert cmc._coinbase_to_cmc_id("") is None
        assert cmc._coinbase_to_cmc_id(None) is None

    def test_mapping_overlaps_coingecko_basket(self):
        """Every Coinbase pid in the CoinGecko mapping that matters for the
        survivorship-aware top-N basket should also resolve in CMC. A
        partial overlap is acceptable for the free-tier snapshot path; the
        probe harness will silently drop unmapped pids per the existing
        contract (test_unmapped_pid_omitted_from_response below)."""
        # Smoke-test only: mapping is non-empty and includes BTC + ETH.
        assert cmc._coinbase_to_cmc_id("BTC-USD") is not None
        assert cmc._coinbase_to_cmc_id("ETH-USD") is not None


# ── Current snapshot ────────────────────────────────────────────────────────

class TestFetchMarketcapSnapshot:

    @pytest.mark.asyncio
    async def test_returns_marketcap_row_per_pid(self):
        """CMC `/v1/cryptocurrency/quotes/latest` returns:
            { data: { "<id>": { id, name, symbol, slug,
                                circulating_supply, total_supply, max_supply,
                                quote: { USD: { price, market_cap,
                                                fully_diluted_market_cap, ... }}}}}
        """
        body = {
            "status": {"error_code": 0},
            "data": {
                "1": {
                    "id": 1, "symbol": "BTC", "slug": "bitcoin",
                    "circulating_supply": 19_500_000,
                    "total_supply":       19_500_000,
                    "max_supply":         21_000_000,
                    "quote": {"USD": {
                        "market_cap": 1_300_000_000_000,
                        "fully_diluted_market_cap": 1_400_000_000_000,
                    }},
                },
                "1027": {
                    "id": 1027, "symbol": "ETH", "slug": "ethereum",
                    "circulating_supply": 120_000_000,
                    "total_supply":       120_000_000,
                    "max_supply":         None,
                    "quote": {"USD": {
                        "market_cap": 350_000_000_000,
                        "fully_diluted_market_cap": 350_000_000_000,
                    }},
                },
            },
        }
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cmc.fetch_marketcap_snapshot(["BTC-USD", "ETH-USD"])
        assert set(out.keys()) == {"BTC-USD", "ETH-USD"}
        assert out["BTC-USD"].market_cap == 1_300_000_000_000
        assert out["BTC-USD"].fdv == 1_400_000_000_000
        assert out["BTC-USD"].circ_supply == 19_500_000
        assert out["BTC-USD"].total_supply == 19_500_000

    @pytest.mark.asyncio
    async def test_missing_fdv_falls_back_to_market_cap(self):
        """CMC returns null fully_diluted_market_cap for some assets where
        max_supply is null. We fall back to market_cap so downstream log-FDV
        math doesn't explode on None — same contract as the CoinGecko service."""
        body = {
            "status": {"error_code": 0},
            "data": {"1": {
                "id": 1, "symbol": "BTC",
                "circulating_supply": 19_500_000,
                "total_supply":       19_500_000,
                "max_supply":         None,
                "quote": {"USD": {
                    "market_cap": 1_300_000_000_000,
                    "fully_diluted_market_cap": None,
                }},
            }},
        }
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cmc.fetch_marketcap_snapshot(["BTC-USD"])
        assert out["BTC-USD"].fdv == 1_300_000_000_000

    @pytest.mark.asyncio
    async def test_unmapped_pid_omitted_from_response(self):
        body = {
            "status": {"error_code": 0},
            "data": {"1": {
                "id": 1, "symbol": "BTC",
                "circulating_supply": 1, "total_supply": 1, "max_supply": 1,
                "quote": {"USD": {"market_cap": 1, "fully_diluted_market_cap": 1}},
            }},
        }
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cmc.fetch_marketcap_snapshot(["BTC-USD", "DOES-NOT-EXIST-USD"])
        assert "BTC-USD" in out
        assert "DOES-NOT-EXIST-USD" not in out

    @pytest.mark.asyncio
    async def test_kill_switch_short_circuits_without_http_call(self):
        with patch.dict(os.environ, {"COINMARKETCAP_DISABLED": "1"}):
            with patch.object(cmc.httpx, "AsyncClient") as mock_client:
                out = await cmc.fetch_marketcap_snapshot(["BTC-USD"])
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert out == {}

    @pytest.mark.asyncio
    async def test_missing_api_key_returns_empty_does_not_raise(self):
        """Callers may not have COINMARKETCAP_API_KEY set yet. Mirror the
        graceful-degradation contract used by other free-tier optional
        feeds (e.g., okx_long_short_history geo-block) — no exception, no
        HTTP call, just an empty dict and a warning."""
        with patch.dict(os.environ, {"COINMARKETCAP_API_KEY": ""}):
            with patch.object(cmc.httpx, "AsyncClient") as mock_client:
                out = await cmc.fetch_marketcap_snapshot(["BTC-USD"])
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert out == {}

    @pytest.mark.asyncio
    async def test_401_returns_empty_does_not_raise(self):
        """Bad API key → 401. Match CoinGecko's graceful-degradation contract."""
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_http(401)
            )
            out = await cmc.fetch_marketcap_snapshot(["BTC-USD"])
        assert out == {}

    @pytest.mark.asyncio
    async def test_429_returns_empty_does_not_raise(self):
        """Rate-limit hit (Basic tier = 30 req/min). Match CoinGecko."""
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_http(429)
            )
            out = await cmc.fetch_marketcap_snapshot(["BTC-USD"])
        assert out == {}

    @pytest.mark.asyncio
    async def test_sends_api_key_in_x_cmc_pro_header(self):
        """CMC requires X-CMC_PRO_API_KEY header; verify we set it."""
        body = {"status": {"error_code": 0}, "data": {}}
        captured = {}

        async def fake_get(url, params=None, headers=None, **kw):
            captured["headers"] = headers or {}
            captured["params"] = params or {}
            return _ok(body)

        with patch.dict(os.environ, {"COINMARKETCAP_API_KEY": "live-key-XYZ"}):
            with patch.object(cmc.httpx, "AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    side_effect=fake_get
                )
                await cmc.fetch_marketcap_snapshot(["BTC-USD"])
        assert captured["headers"].get("X-CMC_PRO_API_KEY") == "live-key-XYZ"
        # Must request by numeric id (not symbol — symbol is ambiguous for forks).
        assert "id" in captured["params"]
        assert captured["params"].get("convert", "USD") == "USD"

    @pytest.mark.asyncio
    async def test_empty_pid_list_short_circuits(self):
        """No pids → no HTTP call (saves a credit)."""
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            out = await cmc.fetch_marketcap_snapshot([])
            mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert out == {}


# ── Historical timeseries (free-tier no-op) ─────────────────────────────────

class TestFetchMarketcapHistoryFreeTier:

    @pytest.mark.asyncio
    async def test_history_is_free_tier_no_op_returns_empty(self):
        """CMC historical endpoints are paid (Hobbyist+). On free tier we
        deliberately return [] without making the HTTP call so the probe
        harness can fall back to CoinGecko cleanly."""
        with patch.object(cmc.httpx, "AsyncClient") as mock_client:
            rows = await cmc.fetch_marketcap_history("BTC-USD", 1, 2)
            mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert rows == []

    @pytest.mark.asyncio
    async def test_history_kill_switch_short_circuits(self):
        with patch.dict(os.environ, {"COINMARKETCAP_DISABLED": "1"}):
            with patch.object(cmc.httpx, "AsyncClient") as mock_client:
                rows = await cmc.fetch_marketcap_history("BTC-USD", 1, 2)
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert rows == []


# ── MarketcapRow dataclass shape ────────────────────────────────────────────

class TestMarketcapRowShape:

    def test_marketcap_row_has_expected_fields(self):
        """Same fields as CoinGecko's MarketcapRow so probe harness can
        treat both sources interchangeably."""
        row = cmc.MarketcapRow(
            market_cap=1.0,
            fdv=1.0,
            circ_supply=1.0,
            total_supply=1.0,
            ingest_ts=1700000000,
            schema_version=1,
        )
        assert row.market_cap == 1.0
        assert row.fdv == 1.0
        assert row.circ_supply == 1.0
        assert row.total_supply == 1.0
        assert row.ingest_ts == 1700000000
        assert row.schema_version == 1
