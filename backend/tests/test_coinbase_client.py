from unittest.mock import AsyncMock

import pytest

from clients import coinbase_client as cb


async def test_base_sized_market_sell_sends_base_not_quote(monkeypatch):
    post = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(cb, "_post", post)
    await cb.place_market_order("BTC-USD", "SELL", base_size=0.125)
    payload = post.await_args.args[1]
    assert payload["order_configuration"] == {"market_market_ioc": {"base_size": "0.125"}}
    assert payload["side"] == "SELL"


async def test_quote_sized_market_buy_remains_compatible(monkeypatch):
    post = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(cb, "_post", post)
    await cb.place_market_order("BTC-USD", "BUY", 25.125)
    assert post.await_args.args[1]["order_configuration"] == {
        "market_market_ioc": {"quote_size": "25.12"}
    }


async def test_market_order_rejects_two_size_units(monkeypatch):
    post = AsyncMock()
    monkeypatch.setattr(cb, "_post", post)
    with pytest.raises(ValueError):
        await cb.place_market_order("BTC-USD", "SELL", 25, base_size=0.1)
    post.assert_not_awaited()


async def test_exact_order_lookup_propagates_lookup_failure(monkeypatch):
    get = AsyncMock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr(cb, "_get", get)
    with pytest.raises(RuntimeError):
        await cb.get_order("id-1")
    get.assert_awaited_once_with("/orders/historical/id-1")


async def test_exact_order_lookup_unwraps_order(monkeypatch):
    get = AsyncMock(return_value={"order": {"order_id": "id-1", "status": "CANCELLED"}})
    monkeypatch.setattr(cb, "_get", get)
    assert await cb.get_order("id-1") == {"order_id": "id-1", "status": "CANCELLED"}
