import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import ws_subscriber as wsmod


class _FakeWS:
    def __init__(self):
        self.send = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_connect_count_starts_at_zero():
    sub = wsmod.CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
    assert sub.connect_count == 0


def test_each_subscribed_connection_increments_connect_count():
    sub = wsmod.CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
    sub.set_products(["ABC-USD"])
    with patch.object(
        wsmod.websockets, "connect", MagicMock(side_effect=lambda *a, **k: _FakeWS())
    ):
        _run(sub._connect())
        _run(sub._connect())
    assert sub.connect_count == 2


def test_no_products_does_not_count_as_a_connection():
    sub = wsmod.CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
    with patch.object(wsmod.asyncio, "sleep", AsyncMock()):
        _run(sub._connect())
    assert sub.connect_count == 0
