"""TDD tests for MODEL_BACKEND env var + _cnn_prob branching (#135 Phase 5).

Two pieces:
    1. config.model_backend reads MODEL_BACKEND env var, default "cnn"
    2. CoinbaseCNNAgent._cnn_prob delegates to xgb_signal.xgb_prob iff
       config.model_backend == "xgb"

Default behavior MUST remain unchanged (Phase 5 ships behind a default
"cnn" so no live trading change). Phase 6 (#136) flips MODEL_BACKEND=xgb
in shadow mode.
"""
import importlib
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Config field ──────────────────────────────────────────────────────────────

class TestConfigField:

    def test_model_backend_defaults_to_cnn(self, monkeypatch):
        """MODEL_BACKEND unset → config.model_backend == 'cnn'."""
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        import config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.config.model_backend == "cnn"

    def test_model_backend_reads_env(self, monkeypatch):
        """MODEL_BACKEND=xgb propagates into config.model_backend."""
        monkeypatch.setenv("MODEL_BACKEND", "xgb")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.config.model_backend == "xgb"

    def test_model_backend_lowercased(self, monkeypatch):
        """MODEL_BACKEND=XGB normalized to 'xgb' so comparisons are stable."""
        monkeypatch.setenv("MODEL_BACKEND", "XGB")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.config.model_backend == "xgb"


# ── _cnn_prob branching ───────────────────────────────────────────────────────

class TestCnnProbBranching:

    def _make_agent(self):
        """Build a minimal CoinbaseCNNAgent without going through __init__'s
        DB / scheduler side effects. We only need the _cnn_prob method.
        """
        from agents.cnn_agent import CoinbaseCNNAgent, FeatureBuilder
        agent = CoinbaseCNNAgent.__new__(CoinbaseCNNAgent)
        agent.fb = FeatureBuilder()
        agent.model = None  # forces _linear path when backend == "cnn"
        return agent

    def test_default_backend_uses_cnn_path(self, monkeypatch):
        """config.model_backend == 'cnn' → _cnn_prob does NOT call xgb_prob."""
        monkeypatch.setenv("MODEL_BACKEND", "cnn")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        # cnn_agent imports `config` at module top, reload it so the agent
        # picks up our reloaded config singleton.
        if "agents.cnn_agent" in sys.modules:
            importlib.reload(sys.modules["agents.cnn_agent"])

        called = {"xgb": False}
        if "agents.xgb_signal" in sys.modules:
            del sys.modules["agents.xgb_signal"]
        import agents.xgb_signal as xs

        def _spy(_channels):
            called["xgb"] = True
            return 0.7
        monkeypatch.setattr(xs, "xgb_prob", _spy)

        agent = self._make_agent()
        channels = [[0.5] * 60 for _ in range(27)]
        agent._cnn_prob(channels)
        assert called["xgb"] is False, "CNN backend must not invoke xgb_prob"

    def test_xgb_backend_calls_xgb_prob(self, monkeypatch):
        """config.model_backend == 'xgb' → _cnn_prob returns xgb_prob's value."""
        monkeypatch.setenv("MODEL_BACKEND", "xgb")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        if "agents.cnn_agent" in sys.modules:
            importlib.reload(sys.modules["agents.cnn_agent"])

        if "agents.xgb_signal" in sys.modules:
            del sys.modules["agents.xgb_signal"]
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "xgb_prob", lambda _channels: 0.73)

        # Re-import cnn_agent AFTER patching xgb_signal so the agent's
        # `from agents.xgb_signal import xgb_prob` (if it does that) sees
        # our spy. Safer: patch the symbol the agent actually uses.
        import agents.cnn_agent as ca
        # The agent should call xgb_signal.xgb_prob via attribute lookup,
        # so monkeypatching the module attr above is the right hook.

        agent = self._make_agent()
        channels = [[0.5] * 60 for _ in range(27)]
        prob = agent._cnn_prob(channels)
        assert prob == 0.73, f"expected xgb_prob's 0.73, got {prob}"
