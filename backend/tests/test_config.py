"""Policy test for refactor sweep (#311-refactor-a).

Locks in: any env var defined in config.py MUST trace to a live consumer.
The four CNN_*_CNN_W / CNN_*_LLM_W env vars were dead-on-arrival (never
read anywhere in backend/) and were deleted. If anyone re-adds them
without a live consumer, this test fails.
"""
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestNoCnnArchEnvVar:
    def test_cnn_arch_env_var_lookup_removed_from_cnn_agent(self):
        """Locks in #311-refactor-e: CNN_ARCH env var lookup was deleted
        when the multi-arch registry was removed. Only glu1 survives.
        Re-introducing the lookup requires reverting the dead-variant
        cleanup, not just adding a config field."""
        src = open(
            os.path.join(BACKEND, "agents", "cnn_agent.py"),
            encoding="utf-8",
        ).read()
        for needle in (
            'os.environ.get("CNN_ARCH"',
            "os.environ.get('CNN_ARCH'",
            'os.getenv("CNN_ARCH"',
            "os.getenv('CNN_ARCH'",
        ):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — CNN_ARCH env-var "
                f"lookup was deleted #311-refactor-e. Multi-arch registry "
                f"committed to single-arch (glu1)."
            )


class TestDeadBlendFieldsStayDeleted:
    def test_no_dead_llm_blend_fields(self):
        from config import config
        dead = (
            "cnn_trending_cnn_w",
            "cnn_trending_llm_w",
            "cnn_ranging_cnn_w",
            "cnn_ranging_llm_w",
        )
        for name in dead:
            assert not hasattr(config, name), (
                f"config.{name} was deleted #311-refactor-a — re-adding it requires "
                f"a live consumer in backend/ first."
            )


class TestNoCnnBackendOnlyBranches:
    """Locks in #311-refactor-f: dead CNN-backend-only inference branches were
    deleted from cnn_agent.py. Under MODEL_BACKEND=xgb (live) these branches
    early-returned or skipped — they only ever fired when MODEL_BACKEND=cnn,
    a configuration that has not run in production for many sessions.

    The deleted dead branches were:
      - Hurst random-walk suppression gate (`if _cnn_only and not hurst_ok`)
      - HMM regime CHAOTIC-only gate (`if _cnn_only and _regime_gate_enabled()`)
      - LGBMFilter entry filter (`_lgbm.allow_buy` + `_lgbm.predict`)
      - Ollama LLM blend (the entire `skip_llm` tree + `_ollama_prob` call)
      - `_maybe_auto_train` (XGB has no online retrain path)

    Re-introducing any of these markers requires reverting the dead-branch
    cleanup, not just patching code back in."""

    @staticmethod
    def _src():
        return open(
            os.path.join(BACKEND, "agents", "cnn_agent.py"),
            encoding="utf-8",
        ).read()

    def test_no_cnn_only_gate(self):
        src = self._src()
        for needle in ("_cnn_only", "if _cnn_only"):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — CNN-backend-only "
                f"suppression gate was deleted #311-refactor-f."
            )

    def test_no_lgbm_filter_machinery(self):
        src = self._src()
        for needle in (
            "_lgbm.allow_buy",
            "_lgbm.predict",
            "from data.lgbm_filter import LGBMFilter",
            "LGBMFilter()",
            "_lgbm_retrain_if_needed",
        ):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — LGBMFilter machinery "
                f"was deleted #311-refactor-f (CNN-tuned, dead under xgb)."
            )

    def test_no_ollama_blend(self):
        src = self._src()
        for needle in (
            "_ollama_prob",
            "skip_llm",
            "from services.fear_greed import get_fear_greed",
        ):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — Ollama LLM blend was "
                f"deleted #311-refactor-f (CNN-backend-only)."
            )

    def test_no_hurst_di_entropy_gates(self):
        src = self._src()
        for needle in (
            "_HURST_MR_THRESH",
            "_HURST_TREND_THRESH",
            "_DI_SUPPRESS_THRESH",
            "_ENTROPY_SKIP_THRESH",
            "_hurst_exponent",
            "_dissimilarity_index",
            "_shannon_entropy",
            "_regime_gate_enabled",
            "CNN_REGIME_GATE",
        ):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — Hurst/DI/entropy/regime "
                f"gates were deleted #311-refactor-f (CNN-backend-only)."
            )

    def test_no_maybe_auto_train(self):
        src = self._src()
        for needle in ("_maybe_auto_train",):
            assert needle not in src, (
                f"cnn_agent.py contains '{needle}' — auto-train scheduler "
                f"was deleted #311-refactor-f (no-op under xgb backend; "
                f"auto-train infrastructure cleanup is module 4c)."
            )


class TestModelBackendValidation:
    """Validates the MODEL_BACKEND env-var contract after CNN deprecation
    (2026-05-23). Valid values: 'xgb' (v3 driver, default) | 'xgb_v45'
    (v4.5 driver). Legacy 'cnn' raises ValueError at startup."""

    def test_model_backend_cnn_raises_value_error(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "cnn")
        from config import Config
        with pytest.raises(ValueError, match="deprecated"):
            Config()

    def test_model_backend_xgb_v45_accepted(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "xgb_v45")
        from config import Config
        cfg = Config()
        assert cfg.model_backend == "xgb_v45"

    def test_model_backend_unknown_raises(self, monkeypatch):
        monkeypatch.setenv("MODEL_BACKEND", "lstm")
        from config import Config
        with pytest.raises(ValueError, match="invalid"):
            Config()

    def test_xgb_v45_threshold_defaults(self, monkeypatch):
        monkeypatch.delenv("XGB_V45_THRESH_UP",   raising=False)
        monkeypatch.delenv("XGB_V45_THRESH_DOWN", raising=False)
        monkeypatch.setenv("MODEL_BACKEND", "xgb")
        from config import Config
        cfg = Config()
        assert cfg.xgb_v45_thresh_up   == 0.50
        assert cfg.xgb_v45_thresh_down == 0.50
