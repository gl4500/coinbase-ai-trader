"""CoinMarketCap /cryptocurrency/quotes/latest service — mirror of CG service shape."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "leader_filter"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def test_coinbase_to_cmc_id_returns_int_for_known_pid():
    from services.coinmarketcap_marketcap import _coinbase_to_cmc_id
    assert _coinbase_to_cmc_id("BTC-USD") == 1
    assert _coinbase_to_cmc_id("ETH-USD") == 1027


def test_coinbase_to_cmc_id_returns_none_for_unmapped():
    from services.coinmarketcap_marketcap import _coinbase_to_cmc_id
    assert _coinbase_to_cmc_id("NOSUCH-USD") is None
    assert _coinbase_to_cmc_id("") is None


def test_kill_switch_short_circuits(monkeypatch):
    from services.coinmarketcap_marketcap import _is_disabled
    monkeypatch.setenv("COINMARKETCAP_DISABLED", "1")
    assert _is_disabled() is True
    monkeypatch.setenv("COINMARKETCAP_DISABLED", "0")
    assert _is_disabled() is False
    monkeypatch.delenv("COINMARKETCAP_DISABLED", raising=False)
    assert _is_disabled() is False


def test_parse_quotes_response_returns_marketcap_rows():
    from services.coinmarketcap_marketcap import parse_quotes_response
    payload = _load("cmc_quotes_sample.json")
    rows = parse_quotes_response(payload, cmc_to_pid={1: "BTC-USD", 1027: "ETH-USD"})
    assert "BTC-USD" in rows
    assert "ETH-USD" in rows
    btc = rows["BTC-USD"]
    assert btc.price == 67100.0
    assert btc.volume_24h == 25500000000
    assert btc.market_cap == 1322000000000
    assert btc.fdv == 1409000000000
    assert btc.price_chg_24h_pct == 2.6
    assert btc.circ_supply == 19700000.0


def test_parse_quotes_response_uses_total_supply_or_max_supply():
    from services.coinmarketcap_marketcap import parse_quotes_response
    rows = parse_quotes_response(_load("cmc_quotes_sample.json"),
                                 cmc_to_pid={1027: "ETH-USD"})
    eth = rows["ETH-USD"]
    assert eth.total_supply == 120280000.0


def test_parse_quotes_response_skips_pid_not_in_map():
    from services.coinmarketcap_marketcap import parse_quotes_response
    rows = parse_quotes_response(_load("cmc_quotes_sample.json"),
                                 cmc_to_pid={1: "BTC-USD"})
    assert "ETH-USD" not in rows
    assert "BTC-USD" in rows


def test_parse_quotes_response_handles_error_envelope():
    from services.coinmarketcap_marketcap import parse_quotes_response
    rows = parse_quotes_response({"status": {"error_code": 1001}, "data": {}},
                                 cmc_to_pid={1: "BTC-USD"})
    assert rows == {}


def test_parse_quotes_response_falls_back_when_fdv_null():
    from services.coinmarketcap_marketcap import parse_quotes_response
    payload = {"status": {"error_code": 0}, "data": {
        "999": {
            "id": 999, "symbol": "X",
            "circulating_supply": 1e9, "total_supply": 2e9, "max_supply": 2e9,
            "quote": {"USD": {
                "price": 5.0, "volume_24h": 1e7, "market_cap": 5e9,
                "fully_diluted_valuation": None, "percent_change_24h": 0.0,
            }},
        }
    }}
    rows = parse_quotes_response(payload, cmc_to_pid={999: "X-USD"})
    assert rows["X-USD"].fdv == pytest.approx(5.0 * 2e9)


@pytest.mark.asyncio
async def test_fetch_marketcap_snapshot_kill_switch_returns_empty(monkeypatch):
    from services.coinmarketcap_marketcap import fetch_marketcap_snapshot
    monkeypatch.setenv("COINMARKETCAP_DISABLED", "1")
    assert await fetch_marketcap_snapshot(["BTC-USD"]) == {}


@pytest.mark.asyncio
async def test_fetch_marketcap_snapshot_no_api_key_returns_empty(monkeypatch):
    from services.coinmarketcap_marketcap import fetch_marketcap_snapshot
    monkeypatch.delenv("COINMARKETCAP_API_KEY", raising=False)
    monkeypatch.delenv("COINMARKETCAP_DISABLED", raising=False)
    assert await fetch_marketcap_snapshot(["BTC-USD"]) == {}


@pytest.mark.asyncio
async def test_fetch_marketcap_snapshot_uses_api_key_header(monkeypatch):
    from services.coinmarketcap_marketcap import fetch_marketcap_snapshot

    captured: dict[str, object] = {}

    class FakeResp:
        status_code = 200
        def json(self):
            return _load("cmc_quotes_sample.json")

    class FakeClient:
        def __init__(self, *a, **kw):
            captured["timeout"] = kw.get("timeout")
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None, headers=None):
            captured["url"] = url
            captured["params"] = params
            captured["headers"] = headers
            return FakeResp()

    monkeypatch.setattr("services.coinmarketcap_marketcap.httpx.AsyncClient", FakeClient)
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "test-key")
    monkeypatch.delenv("COINMARKETCAP_DISABLED", raising=False)

    rows = await fetch_marketcap_snapshot(["BTC-USD", "ETH-USD"])

    assert captured["headers"]["X-CMC_PRO_API_KEY"] == "test-key"
    assert captured["params"]["id"] == "1,1027"
    assert captured["params"]["convert"] == "USD"
    assert "BTC-USD" in rows


@pytest.mark.asyncio
async def test_fetch_marketcap_snapshot_handles_non_200(monkeypatch):
    from services.coinmarketcap_marketcap import fetch_marketcap_snapshot

    class FakeResp:
        status_code = 401
        def json(self): return {"status": {"error_code": 1001}}

    class FakeClient:
        def __init__(self, *a, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, *a, **kw): return FakeResp()

    monkeypatch.setattr("services.coinmarketcap_marketcap.httpx.AsyncClient", FakeClient)
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "k")
    monkeypatch.delenv("COINMARKETCAP_DISABLED", raising=False)

    rows = await fetch_marketcap_snapshot(["BTC-USD"])
    assert rows == {}
