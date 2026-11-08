"""Test environment field factories and model-backend routing.

Construct fresh Config values without replacing the shared config singleton.
The supported drivers are xgb and xgb_v45; the legacy CNN driver is rejected.
"""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Config field ──────────────────────────────────────────────────────────────


class TestConfigField:
    def test_model_backend_defaults_to_xgb(self, monkeypatch):
        """MODEL_BACKEND unset → config.model_backend == 'xgb' (CNN deprecated 2026-05-23).

        Instantiate Config directly: field factories read the patched environment.
        Reloading config would replace its singleton while cnn_agent retains the old one.
        """
        monkeypatch.delenv("MODEL_BACKEND", raising=False)
        import config as cfg_mod

        fresh = cfg_mod.Config()
        assert fresh.model_backend == "xgb"

    def test_model_backend_reads_env(self, monkeypatch):
        """MODEL_BACKEND=xgb propagates into config.model_backend."""
        monkeypatch.setenv("MODEL_BACKEND", "xgb")
        import config as cfg_mod

        assert cfg_mod.Config().model_backend == "xgb"

    def test_model_backend_lowercased(self, monkeypatch):
        """MODEL_BACKEND=XGB normalized to 'xgb' so comparisons are stable."""
        monkeypatch.setenv("MODEL_BACKEND", "XGB")
        import config as cfg_mod

        assert cfg_mod.Config().model_backend == "xgb"


# ── Scan-loop interval (#227) ─────────────────────────────────────────────────


class TestScanIntervalConfig:
    """SCAN_INTERVAL_SECS env var drives cnn_agent.run_loop cadence so the
    XGB-eval loop can be expedited (3-5x scan rate) without code changes."""

    def test_scan_interval_secs_defaults_to_900(self, monkeypatch):
        """SCAN_INTERVAL_SECS unset → config.scan_interval_secs == 900."""
        monkeypatch.delenv("SCAN_INTERVAL_SECS", raising=False)
        import config as cfg_mod

        fresh = cfg_mod.Config()
        assert fresh.scan_interval_secs == 900

    def test_scan_interval_secs_reads_env(self, monkeypatch):
        """SCAN_INTERVAL_SECS=300 propagates into config.scan_interval_secs."""
        monkeypatch.setenv("SCAN_INTERVAL_SECS", "300")
        import config as cfg_mod

        assert cfg_mod.Config().scan_interval_secs == 300

    def test_scan_interval_secs_is_int(self, monkeypatch):
        """Stored as int — used directly as asyncio.sleep argument."""
        monkeypatch.setenv("SCAN_INTERVAL_SECS", "180")
        import config as cfg_mod

        assert isinstance(cfg_mod.Config().scan_interval_secs, int)
        assert cfg_mod.Config().scan_interval_secs == 180


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

    def test_cnn_backend_is_deprecated(self, monkeypatch):
        """MODEL_BACKEND=cnn raises ValueError as of 2026-05-23 (CNN deprecated).

        Tests _validate_backend directly to avoid module-reload side-effects that
        can corrupt the config singleton for downstream tests.
        """
        import config as cfg_mod

        with pytest.raises(ValueError, match="MODEL_BACKEND=cnn is deprecated"):
            cfg_mod._validate_backend("cnn")

    def test_xgb_backend_calls_xgb_prob(self, monkeypatch):
        """config.model_backend == 'xgb' → _cnn_prob returns xgb_prob's value."""
        from agents import cnn_agent, xgb_signal

        monkeypatch.setattr(cnn_agent.config, "model_backend", "xgb")
        monkeypatch.setattr(xgb_signal, "xgb_prob", lambda _channels, pid=None: 0.73)

        agent = self._make_agent()
        channels = [[0.5] * 60 for _ in range(27)]
        prob = agent._cnn_prob(channels)
        assert prob == 0.73, f"expected xgb_prob's 0.73, got {prob}"


# ── Auto-train gate tests REMOVED #311-refactor-f ───────────────────────────
# `_maybe_auto_train` was deleted alongside the rest of the dead CNN-backend-
# only branches; the auto-train scheduler hook in `run_loop` was already
# gated off under MODEL_BACKEND=xgb (#300) and a no-op under xgb. Policy
# lock in test_config.py::TestNoCnnBackendOnlyBranches::test_no_maybe_auto_train.
# Auto-train infrastructure cleanup (subprocess wiring, train_worker.py) is
# scoped to module 4c.


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

    def test_cnn_prob_raises_on_cnn_backend(self, monkeypatch):
        """_cnn_prob raises RuntimeError when model_backend is forced to 'cnn' (deprecated)."""
        import config as cfg

        monkeypatch.setattr(cfg.config, "model_backend", "cnn")

        from agents import cnn_agent

        agent = cnn_agent.CoinbaseCNNAgent.__new__(cnn_agent.CoinbaseCNNAgent)
        agent.model = None
        agent.fb = None
        import numpy as np

        channels = np.zeros((28, 60), dtype=np.float64).tolist()
        with pytest.raises(RuntimeError, match="unsupported model_backend"):
            agent._cnn_prob(channels, pid="BTC-USD")
