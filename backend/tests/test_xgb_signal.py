"""TDD tests for agents/xgb_signal.py — Phase 5 of CNN→XGBoost transition (#135).

Contract:
    xgb_prob(channels) -> float
        - Input: 27x60 nested list (or np.ndarray of shape [27, 60]).
        - Output: float in [0.01, 0.99] when xgb_model.json + xgb_features.json
          are present at the configured paths.
        - Graceful fallback: 0.5 (neutral) when artifacts are missing or fail
          to load. Phase 5 ships behind MODEL_BACKEND=cnn default — XGB path
          MUST not crash production if the model file isn't on disk yet.
        - Deterministic: same input -> same output across calls within a
          process (Booster is loaded once and cached).
        - Mask honored implicitly via tools.xgb_features.extract_features —
          channels 17/18/19 are zeroed before the Booster sees them.
"""
import json
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_channels(seed: int = 0) -> np.ndarray:
    """27-channel x 60-step deterministic synthetic input."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((27, 60)).astype(np.float64)


def _train_tiny_xgb(out_dir, n_samples: int = 64, feature_set: str = "v1"):
    """Train a minimal XGBoost model so xgb_signal.xgb_prob has artifacts to load."""
    import xgboost as xgb
    from tools.xgb_features import extract_features

    rng = np.random.default_rng(0)
    samples = rng.standard_normal((n_samples, 27, 60)).astype(np.float64)
    labels = (rng.standard_normal(n_samples) > 0).astype(np.int64)
    features, names = extract_features(samples, feature_set=feature_set)

    dtrain = xgb.DMatrix(features, label=labels, feature_names=names)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "verbosity": 0},
        dtrain,
        num_boost_round=5,
    )
    model_path = os.path.join(out_dir, "xgb_model.json")
    features_path = os.path.join(out_dir, "xgb_features.json")
    booster.save_model(model_path)
    with open(features_path, "w") as f:
        json.dump({"feature_names": names, "best_params": {}}, f)
    return model_path, features_path


@pytest.fixture
def fresh_xgb_module(monkeypatch):
    """Yield a freshly-imported agents.xgb_signal so each test starts from a
    clean lazy-load state (no cached Booster from a prior test)."""
    if "agents.xgb_signal" in sys.modules:
        del sys.modules["agents.xgb_signal"]
    import agents.xgb_signal as xs
    yield xs
    if "agents.xgb_signal" in sys.modules:
        del sys.modules["agents.xgb_signal"]


# ── Contract: graceful fallback when artifacts missing ────────────────────────

class TestFallback:

    def test_returns_neutral_when_model_file_missing(self, tmp_path, fresh_xgb_module, monkeypatch):
        """No xgb_model.json on disk → return 0.5, never raise."""
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "missing.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "also_missing.json"))
        prob = fresh_xgb_module.xgb_prob(_synthetic_channels())
        assert prob == 0.5

    def test_returns_neutral_when_features_json_missing(self, tmp_path, fresh_xgb_module, monkeypatch):
        """xgb_model.json present but xgb_features.json missing → 0.5."""
        # Touch only the model file, leave features file absent
        (tmp_path / "xgb_model.json").write_text("{}")
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        prob = fresh_xgb_module.xgb_prob(_synthetic_channels())
        assert prob == 0.5


# ── Contract: live model returns calibrated prob ──────────────────────────────

class TestLiveModel:

    def test_returns_float_in_range(self, tmp_path, fresh_xgb_module, monkeypatch):
        """With a real (tiny) booster on disk, xgb_prob returns float ∈ [0.01, 0.99]."""
        model_path, features_path = _train_tiny_xgb(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", model_path)
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", features_path)
        prob = fresh_xgb_module.xgb_prob(_synthetic_channels(seed=1))
        assert isinstance(prob, float)
        assert 0.01 <= prob <= 0.99, f"out of range: {prob}"

    def test_deterministic_same_input(self, tmp_path, fresh_xgb_module, monkeypatch):
        """Repeated calls on identical input → identical output (cached booster)."""
        model_path, features_path = _train_tiny_xgb(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", model_path)
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", features_path)
        x = _synthetic_channels(seed=42)
        a = fresh_xgb_module.xgb_prob(x)
        b = fresh_xgb_module.xgb_prob(x)
        assert a == b

    def test_accepts_nested_list_input(self, tmp_path, fresh_xgb_module, monkeypatch):
        """Mirrors _cnn_prob's contract: 27x60 nested list must work, not just np.ndarray."""
        model_path, features_path = _train_tiny_xgb(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", model_path)
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", features_path)
        ch = _synthetic_channels(seed=2).tolist()
        prob = fresh_xgb_module.xgb_prob(ch)
        assert isinstance(prob, float)
        assert 0.01 <= prob <= 0.99

    def test_mask_channels_are_zeroed(self, tmp_path, fresh_xgb_module, monkeypatch):
        """Ch 17/18/19 are MASKED — values there must NOT change the prediction.

        Rationale: training uses the same mask, so live inference seeing
        anything but zeros there would create train/serve skew.
        """
        model_path, features_path = _train_tiny_xgb(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", model_path)
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", features_path)
        x_a = _synthetic_channels(seed=3)
        x_b = x_a.copy()
        x_b[17, :] = 99.0
        x_b[18, :] = -42.0
        x_b[19, :] = 7.0
        a = fresh_xgb_module.xgb_prob(x_a)
        b = fresh_xgb_module.xgb_prob(x_b)
        assert a == b, (
            "MASKED_CHANNELS values leaked into prediction — "
            "extract_features should be zeroing them"
        )


# ── Module attributes ─────────────────────────────────────────────────────────

class TestModuleAttributes:

    def test_has_default_paths(self, fresh_xgb_module):
        """Module exposes _MODEL_PATH and _FEATURES_PATH as monkeypatch targets."""
        assert hasattr(fresh_xgb_module, "_MODEL_PATH")
        assert hasattr(fresh_xgb_module, "_FEATURES_PATH")
        # Default points to backend/xgb_model.json (sibling of cnn_model_*.pt)
        assert fresh_xgb_module._MODEL_PATH.endswith("xgb_model.json")
        assert fresh_xgb_module._FEATURES_PATH.endswith("xgb_features.json")

    def test_xgb_prob_is_callable(self, fresh_xgb_module):
        assert callable(fresh_xgb_module.xgb_prob)
