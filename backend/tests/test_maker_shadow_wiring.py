import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import cnn_agent as ca


def _agent(state):
    ws = MagicMock()
    ws.state = state
    agent = ca.CoinbaseCNNAgent(ws_subscriber=ws)
    agent.maker_shadow = MagicMock()
    return agent


def test_flag_off_never_touches_shadow(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", False)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_not_called()


def test_flag_on_registers_ws_quote(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_called_once_with("ABC-USD", bid=10.0, ask=10.1)


def test_flag_on_missing_ws_state_registers_none(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_called_once_with("ABC-USD", bid=None, ask=None)


def test_shadow_exception_never_escapes(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent.maker_shadow.register.side_effect = RuntimeError("boom")
    agent._shadow_register("ABC-USD")  # must not raise


def test_no_shadow_instance_is_noop(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    ws = MagicMock()
    ws.state = {"ABC-USD": {"bid": 10.0, "ask": 10.1}}
    agent = ca.CoinbaseCNNAgent(ws_subscriber=ws)
    assert agent.maker_shadow is None
    agent._shadow_register("ABC-USD")  # must not raise


def test_attach_wires_epoch_handler_and_sweeper():
    import asyncio

    from services import maker_shadow as ms

    ws = MagicMock()
    ws.connect_count = 3
    agent = MagicMock()

    async def sink(row):
        pass

    async def go():
        shadow = ms.attach(ws, agent, sink=sink, sweep_interval_s=3600)
        assert agent.maker_shadow is shadow
        ws.register_price_handler.assert_called_once_with(shadow.on_tick)
        assert shadow._feed_epoch() == 3
        ws.connect_count = 4
        assert shadow._feed_epoch() == 4
        assert shadow._sweeper is not None and not shadow._sweeper.done()
        shadow._sweeper.cancel()

    asyncio.new_event_loop().run_until_complete(go())
