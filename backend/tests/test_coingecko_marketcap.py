"""
Tests for services/coingecko_marketcap.py — CoinGecko historical marketcap +
FDV fetcher (#260a).

Sibling of services/okx_oi_history (#86 / #88) and services/okx_long_short_history
(#235). Adds marketcap/FDV as a candidate exogenous input after seven sequential
BTC-flavored probes failed the +0.01 mean-AUC gate (see project memory
xgb_feature_optimization_findings.md).

Public API (mirrors funding/OI fetchers):
  await fetch_marketcap_snapshot(pids: list[str])
      -> dict[pid -> MarketcapRow]
  await fetch_marketcap_history(pid: str, start_ms: int, end_ms: int)
      -> list[(ts_ms, market_cap, fdv, circ_supply, total_supply)] sorted asc
  align_to_hourly(rows, hour_grid_ms, lag_secs=86400)
      -> dict[ts_ms -> MarketcapRow]   (forward-filled with strict-causal lag)

CoinGecko endpoints used (free tier, no key):
  GET /api/v3/coins/markets?vs_currency=usd&ids=bitcoin,ethereum,...
  GET /api/v3/coins/{id}/market_chart/range?vs_currency=usd&from=&to=

Kill switch: env COINGECKO_DISABLED=1 short-circuits without an HTTP call.
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

from services import coingecko_marketcap as cg  # noqa: E402


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


# ── Demo API key header (#294) ──────────────────────────────────────────────

class TestDemoApiKey:

    @pytest.mark.asyncio
    async def test_history_sends_demo_key_header_when_env_set(self):
        with patch.dict(os.environ, {"COINGECKO_API_KEY": "demo-abc123"}):
            with patch.object(cg.httpx, "AsyncClient") as mock_client:
                mock_get = AsyncMock(return_value=_ok({"market_caps": []}))
                mock_client.return_value.__aenter__.return_value.get = mock_get
                await cg.fetch_marketcap_history("BTC-USD", 0, 10**12)
                hdrs = mock_get.call_args.kwargs.get("headers") or {}
                assert hdrs.get("x-cg-demo-api-key") == "demo-abc123"

    @pytest.mark.asyncio
    async def test_history_omits_header_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COINGECKO_API_KEY", None)
            with patch.object(cg.httpx, "AsyncClient") as mock_client:
                mock_get = AsyncMock(return_value=_ok({"market_caps": []}))
                mock_client.return_value.__aenter__.return_value.get = mock_get
                await cg.fetch_marketcap_history("BTC-USD", 0, 10**12)
                hdrs = mock_get.call_args.kwargs.get("headers") or {}
                assert "x-cg-demo-api-key" not in hdrs

    @pytest.mark.asyncio
    async def test_snapshot_sends_demo_key_header_when_env_set(self):
        with patch.dict(os.environ, {"COINGECKO_API_KEY": "demo-xyz"}):
            with patch.object(cg.httpx, "AsyncClient") as mock_client:
                mock_get = AsyncMock(return_value=_ok([]))
                mock_client.return_value.__aenter__.return_value.get = mock_get
                await cg.fetch_marketcap_snapshot(["BTC-USD"])
                hdrs = mock_get.call_args.kwargs.get("headers") or {}
                assert hdrs.get("x-cg-demo-api-key") == "demo-xyz"


# ── Coinbase pid → CoinGecko id mapping ─────────────────────────────────────

class TestIdMapping:

    def test_known_pids_map_to_coingecko_ids(self):
        assert cg._coinbase_to_cg_id("BTC-USD")  == "bitcoin"
        assert cg._coinbase_to_cg_id("ETH-USD")  == "ethereum"
        assert cg._coinbase_to_cg_id("SOL-USD")  == "solana"
        assert cg._coinbase_to_cg_id("AVAX-USD") == "avalanche-2"
        assert cg._coinbase_to_cg_id("LINK-USD") == "chainlink"
        assert cg._coinbase_to_cg_id("ARB-USD")  == "arbitrum"

    def test_top20_basket_pids_all_resolve(self):
        """Every pid in the survivorship-aware top-20 basket (#163) used by
        btc_residual_ch9_probe must map to a CoinGecko id. Verified live on
        CoinGecko 2026-05-09 — slugs change occasionally so this test pins
        the snapshot we built the probe against."""
        expected = {
            "PENGU-USD":   "pudgy-penguins",
            "JTO-USD":     "jito-governance-token",
            "POPCAT-USD":  "popcat",
            "BONK-USD":    "bonk",
            "NKN-USD":     "nkn",
            "AIOZ-USD":    "aioz-network",
            "ZK-USD":      "zksync",
            "TRU-USD":     "truefi",
            "ARB-USD":     "arbitrum",
            "SKL-USD":     "skale",
            "AVAX-USD":    "avalanche-2",
            "PEPE-USD":    "pepe",
            "JASMY-USD":   "jasmycoin",
            "LINK-USD":    "chainlink",
            "MOODENG-USD": "moo-deng",
            "FET-USD":     "fetch-ai",
            "ONDO-USD":    "ondo-finance",
            "ALGO-USD":    "algorand",
        }
        for pid, slug in expected.items():
            assert cg._coinbase_to_cg_id(pid) == slug, (pid, slug)

    def test_unknown_pid_returns_none(self):
        assert cg._coinbase_to_cg_id("DOES-NOT-EXIST-USD") is None
        assert cg._coinbase_to_cg_id("") is None


# ── Current snapshot ────────────────────────────────────────────────────────

class TestFetchMarketcapSnapshot:

    @pytest.mark.asyncio
    async def test_returns_marketcap_row_per_pid(self):
        body = [
            {
                "id": "bitcoin",
                "symbol": "btc",
                "market_cap": 1_300_000_000_000,
                "fully_diluted_valuation": 1_400_000_000_000,
                "circulating_supply": 19_500_000,
                "total_supply": 21_000_000,
            },
            {
                "id": "ethereum",
                "symbol": "eth",
                "market_cap": 350_000_000_000,
                "fully_diluted_valuation": 350_000_000_000,
                "circulating_supply": 120_000_000,
                "total_supply": 120_000_000,
            },
        ]
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cg.fetch_marketcap_snapshot(["BTC-USD", "ETH-USD"])
        assert set(out.keys()) == {"BTC-USD", "ETH-USD"}
        assert out["BTC-USD"].market_cap == 1_300_000_000_000
        assert out["BTC-USD"].fdv == 1_400_000_000_000
        assert out["BTC-USD"].circ_supply == 19_500_000
        assert out["BTC-USD"].total_supply == 21_000_000

    @pytest.mark.asyncio
    async def test_missing_fdv_falls_back_to_market_cap(self):
        """CoinGecko returns null `fully_diluted_valuation` for fixed-supply
        coins where FDV == marketcap. We fall back to marketcap so downstream
        log-FDV math doesn't explode on None."""
        body = [{
            "id": "bitcoin", "symbol": "btc",
            "market_cap": 1_300_000_000_000,
            "fully_diluted_valuation": None,
            "circulating_supply": 19_500_000,
            "total_supply": 21_000_000,
        }]
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cg.fetch_marketcap_snapshot(["BTC-USD"])
        assert out["BTC-USD"].fdv == 1_300_000_000_000

    @pytest.mark.asyncio
    async def test_unmapped_pid_omitted_from_response(self):
        body = [{"id": "bitcoin", "symbol": "btc", "market_cap": 1, "fully_diluted_valuation": 1,
                 "circulating_supply": 1, "total_supply": 1}]
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            out = await cg.fetch_marketcap_snapshot(["BTC-USD", "DOES-NOT-EXIST-USD"])
        assert "BTC-USD" in out
        assert "DOES-NOT-EXIST-USD" not in out

    @pytest.mark.asyncio
    async def test_kill_switch_short_circuits_without_http_call(self):
        with patch.dict(os.environ, {"COINGECKO_DISABLED": "1"}):
            with patch.object(cg.httpx, "AsyncClient") as mock_client:
                out = await cg.fetch_marketcap_snapshot(["BTC-USD"])
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert out == {}

    @pytest.mark.asyncio
    async def test_429_returns_empty_does_not_raise(self):
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_http(429)
            )
            out = await cg.fetch_marketcap_snapshot(["BTC-USD"])
        assert out == {}


# ── Historical timeseries ───────────────────────────────────────────────────

class TestFetchMarketcapHistory:

    @pytest.mark.asyncio
    async def test_parses_market_chart_range_response(self):
        """CoinGecko `/market_chart/range` returns three parallel series:
            market_caps:  [[ts_ms, value], ...]
            prices:       [[ts_ms, value], ...]
            total_volumes:[[ts_ms, value], ...]
        We only consume `market_caps`; supply data must be fetched separately."""
        body = {
            "market_caps":   [[1700000000000, 1.3e12], [1700086400000, 1.31e12]],
            "prices":        [[1700000000000, 67000.0], [1700086400000, 67500.0]],
            "total_volumes": [[1700000000000, 5e10],   [1700086400000, 5.1e10]],
        }
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            rows = await cg.fetch_marketcap_history(
                "BTC-USD", start_ms=1700000000000, end_ms=1700086400000
            )
        assert len(rows) == 2
        # Step A: shape extended to (ts_ms, market_cap, volume_24h).
        assert rows[0] == (1700000000000, 1.3e12, 5e10)
        assert rows[1] == (1700086400000, 1.31e12, 5.1e10)

    @pytest.mark.asyncio
    async def test_rows_sorted_ascending_even_if_response_is_not(self):
        body = {"market_caps": [[1700086400000, 2.0], [1700000000000, 1.0]]}
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            rows = await cg.fetch_marketcap_history("BTC-USD", 1, 2)
        assert [r[0] for r in rows] == [1700000000000, 1700086400000]

    @pytest.mark.asyncio
    async def test_unmapped_pid_returns_empty(self):
        rows = await cg.fetch_marketcap_history("DOES-NOT-EXIST-USD", 1, 2)
        assert rows == []

    @pytest.mark.asyncio
    async def test_kill_switch_short_circuits_history(self):
        with patch.dict(os.environ, {"COINGECKO_DISABLED": "1"}):
            with patch.object(cg.httpx, "AsyncClient") as mock_client:
                rows = await cg.fetch_marketcap_history("BTC-USD", 1, 2)
                mock_client.return_value.__aenter__.return_value.get.assert_not_called()
        assert rows == []

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self):
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_http(503)
            )
            rows = await cg.fetch_marketcap_history("BTC-USD", 1, 2)
        assert rows == []


# ── Forward-fill alignment with strict causality ────────────────────────────

class TestAlignToHourly:

    def test_broadcasts_daily_value_to_each_hour_in_day(self):
        """A daily marketcap value at 00:00 UTC fills every hour from
        00:00 + lag (default 1 day) through 00:00 + lag + 23h."""
        BAR = 3600 * 1000  # 1h in ms
        DAY = 86400 * 1000
        rows = [(1700000000000, 1.0e12)]   # 2023-11-14 22:13:20 UTC
        # Build a 1h grid covering the next 2 days
        grid = [1700000000000 + i * BAR for i in range(48)]
        out = cg.align_to_hourly(rows, grid, lag_secs=86400)
        # No fill within the lag window — first 24 entries empty
        assert all(out.get(ts) is None for ts in grid[:24])
        # After lag, all 24 hours of day 2 carry the day-1 value
        assert all(out[ts] == 1.0e12 for ts in grid[24:48])

    def test_uses_most_recent_completed_value_strictly_before_t_minus_lag(self):
        """At t, output uses the most recent rows[i] with rows[i].ts <= t - lag.
        Same-day data must NOT leak across the lag boundary."""
        BAR = 3600 * 1000
        rows = [
            (1700000000000, 1.0),
            (1700086400000, 2.0),   # +1 day later
            (1700172800000, 3.0),   # +2 days later
        ]
        # Grid: 3 days starting at rows[0].ts. With lag=1 day:
        #   day 0 → no value (no rows before t - 1d)
        #   day 1 → 1.0 (rows[0])
        #   day 2 → 2.0 (rows[1])
        grid_day0 = 1700000000000
        grid_day1 = grid_day0 + 86400 * 1000
        grid_day2 = grid_day1 + 86400 * 1000
        out = cg.align_to_hourly(rows, [grid_day0, grid_day1, grid_day2], lag_secs=86400)
        assert out.get(grid_day0) is None
        assert out[grid_day1] == 1.0
        assert out[grid_day2] == 2.0

    def test_empty_rows_returns_empty_dict(self):
        assert cg.align_to_hourly([], [1, 2, 3], lag_secs=86400) == {}

    def test_zero_lag_allows_same_ts_match(self):
        rows = [(1000, 5.0)]
        out = cg.align_to_hourly(rows, [1000, 2000], lag_secs=0)
        assert out[1000] == 5.0
        assert out[2000] == 5.0


# ── MarketcapRow dataclass shape ────────────────────────────────────────────

class TestMarketcapRowShape:

    def test_marketcap_row_has_expected_fields(self):
        row = cg.MarketcapRow(
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


# ── volume_24h extraction (Step A: marketcap bronze v2) ──────────────────────


class TestVolume24hExtraction:

    @pytest.mark.asyncio
    async def test_coingecko_parses_volume_24h(self):
        """Fetcher must extract total_volumes alongside market_caps.

        CoinGecko /coins/{id}/market_chart/range returns prices, market_caps,
        and total_volumes as parallel arrays. The old parser ignored
        total_volumes; the new contract returns (ts_ms, market_cap, volume_24h)
        tuples.
        """
        body = {
            "prices":        [[1746921600000, 100.0],  [1747008000000, 101.0]],
            "market_caps":   [[1746921600000, 1.0e9],  [1747008000000, 1.01e9]],
            "total_volumes": [[1746921600000, 5.0e7],  [1747008000000, 6.0e7]],
        }
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            rows = await cg.fetch_marketcap_history(
                "BTC-USD", 1746921600000, 1747008000000
            )
        assert len(rows) == 2
        ts0, mc0, vol0 = rows[0]
        assert mc0 == 1.0e9
        assert vol0 == 5.0e7
        ts1, mc1, vol1 = rows[1]
        assert vol1 == 6.0e7

    @pytest.mark.asyncio
    async def test_coingecko_handles_missing_total_volumes(self, caplog):
        """Missing/empty total_volumes → all rows get volume_24h=0.0 + warning."""
        import logging
        body = {
            "market_caps":   [[1746921600000, 1.0e9]],
            "total_volumes": [],
        }
        with patch.object(cg.httpx, "AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=_ok(body)
            )
            with caplog.at_level(logging.WARNING):
                rows = await cg.fetch_marketcap_history(
                    "BTC-USD", 0, 9999999999999
                )
        assert rows == [(1746921600000, 1.0e9, 0.0)]
        assert any(
            "total_volumes" in r.message.lower() or "volume_24h" in r.message.lower()
            for r in caplog.records
        )
