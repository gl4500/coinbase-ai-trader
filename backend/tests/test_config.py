"""Policy test for refactor sweep (#311-refactor-a).

Locks in: any env var defined in config.py MUST trace to a live consumer.
The four CNN_*_CNN_W / CNN_*_LLM_W env vars were dead-on-arrival (never
read anywhere in backend/) and were deleted. If anyone re-adds them
without a live consumer, this test fails.
"""
import os
import sys

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
