"""Model persistence must remain inside per-test temporary storage."""

from pathlib import Path

import agents.cnn_agent as ca


def test_active_model_paths_are_outside_backend():
    backend = Path(ca.__file__).resolve().parent.parent
    for path in (ca._model_path_for(), ca._best_loss_path_for(), ca._CKPT_PATH):
        assert not Path(path).resolve().is_relative_to(backend), path


def test_config_singleton_matches_the_agent_consumer():
    import config

    assert ca.config is config.config
