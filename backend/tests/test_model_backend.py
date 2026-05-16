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
        """MODEL_BACKEND unset → config.model_backend == 'cnn'.

        importlib.reload(config) re-runs load_dotenv() which re-injects any
        MODEL_BACKEND value present in .env. Clear it again post-reload and
        instantiate Config() directly so we test the dataclass default rather
        than the singleton, which load_dotenv has already polluted.
        """
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        import config as cfg_mod
        importlib.reload(cfg_mod)
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        fresh = cfg_mod.Config()
        assert fresh.model_backend == "cnn"

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


# ── Scan-loop interval (#227) ─────────────────────────────────────────────────

class TestScanIntervalConfig:
    """SCAN_INTERVAL_SECS env var drives cnn_agent.run_loop cadence so the
    XGB-eval loop can be expedited (3-5x scan rate) without code changes."""

    def test_scan_interval_secs_defaults_to_900(self, monkeypatch):
        """SCAN_INTERVAL_SECS unset → config.scan_interval_secs == 900."""
        monkeypatch.delenv("SCAN_INTERVAL_SECS", raising=False)
        import config as cfg_mod
        importlib.reload(cfg_mod)
        monkeypatch.delenv("SCAN_INTERVAL_SECS", raising=False)
        fresh = cfg_mod.Config()
        assert fresh.scan_interval_secs == 900

    def test_scan_interval_secs_reads_env(self, monkeypatch):
        """SCAN_INTERVAL_SECS=300 propagates into config.scan_interval_secs."""
        monkeypatch.setenv("SCAN_INTERVAL_SECS", "300")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        assert cfg_mod.config.scan_interval_secs == 300

    def test_scan_interval_secs_is_int(self, monkeypatch):
        """Stored as int — used directly as asyncio.sleep argument."""
        monkeypatch.setenv("SCAN_INTERVAL_SECS", "180")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        assert isinstance(cfg_mod.config.scan_interval_secs, int)
        assert cfg_mod.config.scan_interval_secs == 180


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
        monkeypatch.setattr(xs, "xgb_prob", lambda _channels, pid=None: 0.73)

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


# ── Auto-train gate (#300) ────────────────────────────────────────────────────

class TestAutoTrainGate:
    """When MODEL_BACKEND=xgb the run_loop auto-train tick must NOT spawn
    train_worker.py — XGB mode has no online retrain path; the CNN trained
    on tick is never used for inference and wastes GPU.

    Companion to #232 (Hurst/regime gates) and #250 (Ollama gates): every
    CNN-specific code path in the scan loop must be gated on
    `config.model_backend == 'cnn'`. See feedback_cnn_safeguards_backend_gating.md.
    """

    def _make_agent(self, scan_count=4):
        from agents.cnn_agent import CoinbaseCNNAgent
        agent = CoinbaseCNNAgent.__new__(CoinbaseCNNAgent)
        agent.scan_count = scan_count
        agent.last_trained_at = 0.0
        agent.train_count = 0
        return agent

    @pytest.mark.asyncio
    async def test_xgb_backend_skips_auto_train(self, monkeypatch):
        """MODEL_BACKEND=xgb + scan_count aligned → auto_train_fn NOT invoked."""
        monkeypatch.setenv("MODEL_BACKEND", "xgb")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        if "agents.cnn_agent" in sys.modules:
            importlib.reload(sys.modules["agents.cnn_agent"])

        called = {"fn": False}

        async def _spy():
            called["fn"] = True

        agent = self._make_agent(scan_count=4)
        triggered = await agent._maybe_auto_train(
            train_every_n_scans=4, auto_train_fn=_spy
        )
        assert triggered is False
        assert called["fn"] is False, \
            "auto_train_fn must not run when MODEL_BACKEND=xgb"

    @pytest.mark.asyncio
    async def test_cnn_backend_triggers_auto_train(self, monkeypatch):
        """MODEL_BACKEND=cnn + scan_count aligned → auto_train_fn invoked."""
        monkeypatch.setenv("MODEL_BACKEND", "cnn")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        if "agents.cnn_agent" in sys.modules:
            importlib.reload(sys.modules["agents.cnn_agent"])

        called = {"fn": False}

        async def _spy():
            called["fn"] = True

        agent = self._make_agent(scan_count=4)
        triggered = await agent._maybe_auto_train(
            train_every_n_scans=4, auto_train_fn=_spy
        )
        assert triggered is True
        assert called["fn"] is True

    @pytest.mark.asyncio
    async def test_scan_count_not_aligned_skips_regardless_of_backend(
        self, monkeypatch
    ):
        """scan_count % N != 0 → never triggers, in either backend."""
        monkeypatch.setenv("MODEL_BACKEND", "cnn")
        import config as cfg_mod
        importlib.reload(cfg_mod)
        if "agents.cnn_agent" in sys.modules:
            importlib.reload(sys.modules["agents.cnn_agent"])

        called = {"fn": False}

        async def _spy():
            called["fn"] = True

        agent = self._make_agent(scan_count=3)
        triggered = await agent._maybe_auto_train(
            train_every_n_scans=4, auto_train_fn=_spy
        )
        assert triggered is False
        assert called["fn"] is False


# ── pid plumbing for XGB v3 (added 2026-05-16, #311d) ─────────────────────


class TestPidPlumbing:
    def test_cnn_prob_passes_pid_to_xgb_signal_under_xgb_backend(self, monkeypatch):
        """Under MODEL_BACKEND=xgb, _cnn_prob must forward pid= to xgb_signal.xgb_prob."""
        import config as cfg
        monkeypatch.setattr(cfg.config, "model_backend", "xgb")

        from agents import cnn_agent, xgb_signal
        called = {}

        def fake_prob(channels, pid=None):
            called["pid"] = pid
            return 0.7

        monkeypatch.setattr(xgb_signal, "xgb_prob", fake_prob)

        agent = cnn_agent.CoinbaseCNNAgent.__new__(cnn_agent.CoinbaseCNNAgent)
        import numpy as np
        channels = np.zeros((28, 60), dtype=np.float64).tolist()
        result = agent._cnn_prob(channels, pid="BTC-USD")
        assert result == 0.7
        assert called["pid"] == "BTC-USD"

    def test_cnn_prob_no_pid_kwarg_under_cnn_backend(self, monkeypatch):
        """Under MODEL_BACKEND=cnn, xgb_signal must NOT be called."""
        import config as cfg
        monkeypatch.setattr(cfg.config, "model_backend", "cnn")

        from agents import cnn_agent, xgb_signal
        called = {"hit": False}

        def fake_prob(channels, pid=None):
            called["hit"] = True
            return 0.5

        monkeypatch.setattr(xgb_signal, "xgb_prob", fake_prob)

        agent = cnn_agent.CoinbaseCNNAgent.__new__(cnn_agent.CoinbaseCNNAgent)
        agent.model = None  # forces _linear fallback
        agent.fb = None
        import numpy as np
        channels = np.zeros((28, 60), dtype=np.float64).tolist()
        agent._cnn_prob(channels, pid="BTC-USD")
        assert called["hit"] is False
