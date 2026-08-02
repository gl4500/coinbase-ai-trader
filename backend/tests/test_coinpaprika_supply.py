"""Tests for the CoinPaprika current-ticker supply-snapshot fetcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.coinpaprika_marketcap import fetch_supply_snapshot


class _MockResp:
    def __init__(self, *, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def _mock_client(resp: _MockResp):
    cm = AsyncMock()
    cm.__aenter__.return_value = AsyncMock(get=AsyncMock(return_value=resp))
    cm.__aexit__.return_value = None
    return cm


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_tuple_for_mapped_pid():
    """fetch_supply_snapshot('BTC-USD') returns (circulating, total, max_or_None)."""
    payload = {
        "id": "btc-bitcoin",
        "symbol": "BTC",
        "circulating_supply": 19_700_000.0,
        "total_supply": 19_700_000.0,
        "max_supply": 21_000_000.0,
        "quotes": {"USD": {"price": 70000.0, "market_cap": 1.4e12}},
    }
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=200, payload=payload)),
    ):
        result = await fetch_supply_snapshot("BTC-USD")

    assert result is not None
    circ, total, max_supply = result
    assert circ == 19_700_000.0
    assert total == 19_700_000.0
    assert max_supply == 21_000_000.0


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_none_for_unmapped_pid():
    """Unmapped pid returns None without hitting the network."""
    result = await fetch_supply_snapshot("ZZZ-USD")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_none_on_non_200():
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=404, payload={})),
    ):
        result = await fetch_supply_snapshot("BTC-USD")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_handles_missing_max_supply():
    """Some tokens (e.g. ETH) have null max_supply — return None in that slot, not 0."""
    payload = {
        "id": "eth-ethereum",
        "circulating_supply": 120_000_000.0,
        "total_supply": 120_000_000.0,
        "max_supply": None,
    }
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=200, payload=payload)),
    ):
        result = await fetch_supply_snapshot("ETH-USD")

    assert result is not None
    circ, total, max_supply = result
    assert circ == 120_000_000.0
    assert total == 120_000_000.0
    assert max_supply is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_respects_disabled_env(monkeypatch):
    """COINPAPRIKA_DISABLED=1 short-circuits without HTTP call."""
    monkeypatch.setenv("COINPAPRIKA_DISABLED", "1")
    result = await fetch_supply_snapshot("BTC-USD")
    assert result is None
