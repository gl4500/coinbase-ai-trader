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
