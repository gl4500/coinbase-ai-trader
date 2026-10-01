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
