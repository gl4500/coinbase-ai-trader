"""
Tests for the Fear & Greed Index macro gate (services/fear_greed.py).

The gate fetches the CNN Fear & Greed Index (0-100) and suppresses BUY entries
during extreme fear (< 20) to avoid catching falling knives.

External HTTP call is always mocked.
"""
import os
import sys
import time
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from services.fear_greed import FearGreedIndex, _EXTREME_FEAR_THRESHOLD


# ── helpers ────────────────────────────────────────────────────────────────────

def _mock_response(value: int, label: str = "Fear") -> dict:
    """Simulates the alternative.me API response format."""
    return {
        "data": [{"value": str(value), "value_classification": label}]
    }


# ── FearGreedIndex ─────────────────────────────────────────────────────────────

class TestFearGreedFetch:

    @pytest.mark.asyncio
    async def test_returns_value_and_label(self):
        fg = FearGreedIndex()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_response(35, "Fear")
        mock_resp.raise_for_status = MagicMock()

        with patch("services.fear_greed.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            result = await fg.fetch()

        assert result["value"] == 35
        assert result["label"] == "Fear"

    @pytest.mark.asyncio
    async def test_extreme_fear_returns_low_value(self):
        fg = FearGreedIndex()
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_response(10, "Extreme Fear")
        mock_resp.raise_for_status = MagicMock()

        with patch("services.fear_greed.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_resp
            )
            result = await fg.fetch()

        assert result["value"] == 10
        assert "Fear" in result["label"]

    @pytest.mark.asyncio
    async def test_network_error_returns_neutral(self):
        """On network failure return neutral value (50) — don't crash the agent."""
        fg = FearGreedIndex()

        with patch("services.fear_greed.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("network error")
            )
            result = await fg.fetch()

        assert result["value"] == 50
        assert "unknown" in result["label"].lower() or result["label"] == "Unknown"


class TestFearGreedCache:

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(self):
        """Repeated calls within TTL must not make a second HTTP request."""
        fg = FearGreedIndex(cache_ttl=600)
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_response(55, "Greed")
        mock_resp.raise_for_status = MagicMock()

        with patch("services.fear_greed.httpx.AsyncClient") as mock_client:
            get_mock = AsyncMock(return_value=mock_resp)
            mock_client.return_value.__aenter__.return_value.get = get_mock

            r1 = await fg.fetch()
            r2 = await fg.fetch()  # should hit cache

        assert get_mock.call_count == 1, "HTTP GET should be called only once within TTL"
        assert r1["value"] == r2["value"] == 55

    @pytest.mark.asyncio
    async def test_cache_expires(self):
        """After TTL expires, a fresh HTTP request should be made."""
        fg = FearGreedIndex(cache_ttl=0)   # TTL=0 → always expired
        mock_resp = MagicMock()
        mock_resp.json.return_value = _mock_response(40, "Fear")
        mock_resp.raise_for_status = MagicMock()

        with patch("services.fear_greed.httpx.AsyncClient") as mock_client:
            get_mock = AsyncMock(return_value=mock_resp)
            mock_client.return_value.__aenter__.return_value.get = get_mock

            await fg.fetch()
            await fg.fetch()  # second fetch — cache expired

        assert get_mock.call_count == 2


class TestFearGreedTradingGate:

    @pytest.mark.asyncio
    async def test_extreme_fear_blocks_buy(self):
        """is_buy_allowed() returns False when F&G < EXTREME_FEAR_THRESHOLD."""
        fg = FearGreedIndex()
        fg._cache = {"value": _EXTREME_FEAR_THRESHOLD - 1, "label": "Extreme Fear"}
        fg._cache_ts = time.time()  # fresh cache

        allowed = await fg.is_buy_allowed()
        assert allowed is False

    @pytest.mark.asyncio
    async def test_normal_fear_allows_buy(self):
        """is_buy_allowed() returns True when F&G >= EXTREME_FEAR_THRESHOLD."""
        fg = FearGreedIndex()
        fg._cache = {"value": _EXTREME_FEAR_THRESHOLD, "label": "Fear"}
        fg._cache_ts = time.time()

        allowed = await fg.is_buy_allowed()
        assert allowed is True

    @pytest.mark.asyncio
    async def test_greed_zone_allows_buy(self):
        """High greed (80+) must not block BUY entries."""
        fg = FearGreedIndex()
        fg._cache = {"value": 82, "label": "Extreme Greed"}
        fg._cache_ts = time.time()

        allowed = await fg.is_buy_allowed()
        assert allowed is True

    @pytest.mark.asyncio
    async def test_sell_always_allowed(self):
        """SELL is always allowed regardless of F&G (need to be able to exit)."""
        fg = FearGreedIndex()
        fg._cache = {"value": 5, "label": "Extreme Fear"}
        fg._cache_ts = time.time()

        # SELL should never be gated by F&G
        allowed = await fg.is_sell_allowed()
        assert allowed is True


class TestFearGreedThreshold:

    def test_extreme_fear_threshold_is_reasonable(self):
        """Threshold should be between 15 and 25 (historical extreme fear zone)."""
        assert 15 <= _EXTREME_FEAR_THRESHOLD <= 25, (
            f"_EXTREME_FEAR_THRESHOLD={_EXTREME_FEAR_THRESHOLD} outside [15, 25]"
        )
