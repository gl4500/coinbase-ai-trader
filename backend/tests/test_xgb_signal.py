"""TDD tests for agents/xgb_signal.py — Phase 5 of CNN→XGBoost transition (#135).

Contract:
    xgb_prob(channels) -> float
        - Input: 28x60 nested list (or np.ndarray of shape [28, 60]).
        - Output: float in [0.01, 0.99] when xgb_model.json + xgb_features.json
          are present at the configured paths.
        - Graceful fallback: 0.5 (neutral) when artifacts are missing or fail
          to load. Phase 5 ships behind MODEL_BACKEND=cnn default — XGB path
          MUST not crash production if the model file isn't on disk yet.
        - Deterministic: same input -> same output across calls within a
          process (Booster is loaded once and cached).
        - Mask honored implicitly via tools.xgb_features.extract_features —
          channels 17/18/19 are zeroed before the Booster sees them.
        - **XGB-Step2** (#180): If xgb_calibration.pkl exists, an
          IsotonicRegression remaps the raw booster output to fix the live
          U-shape calibration. Calibration is optional — pkl missing means
          raw passthrough (Phase 5 backwards-compat).
"""
import json
import os
import pickle
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_channels(seed: int = 0) -> np.ndarray:
    """28-channel x 60-step deterministic synthetic input."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((28, 60)).astype(np.float64)


def _train_tiny_xgb(out_dir, n_samples: int = 64, feature_set: str = "v1"):
    """Train a minimal XGBoost model so xgb_signal.xgb_prob has artifacts to load."""
    import xgboost as xgb
    from tools.xgb_features import extract_features

    rng = np.random.default_rng(0)
    samples = rng.standard_normal((n_samples, 28, 60)).astype(np.float64)
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
        """Mirrors _cnn_prob's contract: 28x60 nested list must work, not just np.ndarray."""
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

    def test_has_calibration_path(self, fresh_xgb_module):
        """XGB-Step2 (#180): module exposes _CALIBRATION_PATH for monkeypatch.

        Default points beside the booster artifacts at backend/xgb_calibration.pkl
        so a single Operator runbook can copy all three files together.
        """
        assert hasattr(fresh_xgb_module, "_CALIBRATION_PATH")
        assert fresh_xgb_module._CALIBRATION_PATH.endswith("xgb_calibration.pkl")


# ── XGB-Step2: post-hoc isotonic calibration (#180) ───────────────────────────

class TestCalibration:

    def test_no_calibration_pkl_falls_back_to_raw(
        self, tmp_path, fresh_xgb_module, monkeypatch
    ):
        """Missing xgb_calibration.pkl → raw booster output (no exception)."""
        model_path, features_path = _train_tiny_xgb(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", model_path)
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", features_path)
        monkeypatch.setattr(
            fresh_xgb_module, "_CALIBRATION_PATH",
            str(tmp_path / "does_not_exist.pkl"),
        )
        x = _synthetic_channels(seed=8)
        prob = fresh_xgb_module.xgb_prob(x)
        assert isinstance(prob, float)
        assert 0.01 <= prob <= 0.99

    def test_bare_isotonic_pkl_skipped_with_warning(
        self, tmp_path, monkeypatch, fresh_xgb_module, caplog
    ):
        """Locks in #311-refactor-b: bare-isotonic pickle format is no longer
        supported. A bare pickle on disk is treated as 'unknown shape' —
        skipped with a warning. Raw passthrough remains the failure mode."""
        import logging
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        _train_tiny_xgb(str(tmp_path), feature_set="v1")
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
        )
        with open(tmp_path / "xgb_calibration.pkl", "wb") as f:
            pickle.dump(iso, f)  # bare isotonic — no longer supported

        monkeypatch.setattr(
            fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json")
        )
        monkeypatch.setattr(
            fresh_xgb_module, "_FEATURES_PATH",
            str(tmp_path / "xgb_features.json"),
        )
        monkeypatch.setattr(
            fresh_xgb_module, "_CALIBRATION_PATH",
            str(tmp_path / "xgb_calibration.pkl"),
        )

        with caplog.at_level(logging.WARNING):
            fresh_xgb_module._try_load()
        assert fresh_xgb_module._calibration is None
        assert any(
            "bare-isotonic" in r.message.lower()
            or "not the canonical" in r.message.lower()
            for r in caplog.records
        )


# ── #192: hot-reload after on-disk artifacts change ───────────────────────────

class TestForceReload:
    """Targets the new force_reload() that #192 adds.

    Without it, agents.xgb_signal caches _booster + _calibration on first
    call and never re-reads the pickle. After fitting a fresh calibrator
    on disk (#187) the running backend keeps the old in-memory plateau
    until process restart. force_reload() lets us flip the cached state
    and re-read both artifacts without bouncing the FastAPI process.
    """

    def test_force_reload_function_exists(self, fresh_xgb_module):
        """The reload entrypoint must exist and be callable."""
        assert hasattr(fresh_xgb_module, "force_reload"), (
            "agents.xgb_signal.force_reload missing — required for hot-swap "
            "after on-disk calibrator refit (#187)"
        )
        assert callable(fresh_xgb_module.force_reload)

    def test_force_reload_returns_false_when_artifacts_missing(
        self, tmp_path, fresh_xgb_module, monkeypatch
    ):
        """If model files vanish between loads, force_reload returns False
        and subsequent xgb_prob falls back to the neutral 0.5."""
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "nope.json"))
        monkeypatch.setattr(
            fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "also_nope.json")
        )
        ok = fresh_xgb_module.force_reload()
        assert ok is False
        assert fresh_xgb_module.xgb_prob(_synthetic_channels()) == 0.5


# ── v3 routing tests (added 2026-05-16, #311c) ─────────────────────────────


def _train_tiny_v3(out_dir):
    """Train a tiny v3 booster on synthetic tier inputs."""
    import xgboost as xgb
    from tools.xgb_features import _v3_feature_names, _extract_v3

    rng = np.random.default_rng(0)
    n = 64
    feats = np.zeros((n, 350), dtype=np.float64)
    for i in range(n):
        t = {
            "micro": [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": float(rng.standard_normal()),
                       "volume": 1.0} for j in range(60)],
            "meso":  [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": float(rng.standard_normal()),
                       "volume": 1.0} for j in range(168)],
            "macro": [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                       "close": float(rng.standard_normal()),
                       "volume": 1.0} for j in range(336)],
        }
        f, _ = _extract_v3(t)
        feats[i] = f[0]
    labels = (rng.standard_normal(n) > 0).astype(np.int64)
    names = _v3_feature_names()
    dtrain = xgb.DMatrix(feats, label=labels, feature_names=names)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "verbosity": 0},
        dtrain, num_boost_round=5,
    )
    model_path = os.path.join(out_dir, "xgb_model.json")
    features_path = os.path.join(out_dir, "xgb_features.json")
    booster.save_model(model_path)
    with open(features_path, "w") as f:
        json.dump({"feature_names": names, "feature_set": "v3", "best_params": {}}, f)
    return model_path, features_path


def _fake_v3_tiers():
    return {
        "micro": [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                   "close": 1.0 + j * 0.01, "volume": 1.0} for j in range(60)],
        "meso":  [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                   "close": 1.0 + j * 0.01, "volume": 1.0} for j in range(168)],
        "macro": [{"start": j, "open": 1.0, "high": 1.0, "low": 1.0,
                   "close": 1.0 + j * 0.01, "volume": 1.0} for j in range(336)],
    }


class TestV3Routing:
    def test_v3_booster_auto_detected_from_feature_names(self, tmp_path, monkeypatch, fresh_xgb_module):
        _train_tiny_v3(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        assert fresh_xgb_module._try_load() is True
        assert fresh_xgb_module._feature_set == "v3"

    def test_v1_booster_still_detected_correctly(self, tmp_path, monkeypatch, fresh_xgb_module):
        _train_tiny_xgb(str(tmp_path), feature_set="v1")
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        assert fresh_xgb_module._try_load() is True
        assert fresh_xgb_module._feature_set == "v1"

    def test_v3_xgb_prob_calls_tiered_history_with_pid(self, tmp_path, monkeypatch, fresh_xgb_module):
        _train_tiny_v3(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))

        called = {}
        def fake_fetch(pid, **kwargs):
            called["pid"] = pid
            called["kwargs"] = kwargs
            return _fake_v3_tiers()
        monkeypatch.setattr("services.tiered_history.fetch_tiered", fake_fetch)

        p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert 0.01 <= p <= 0.99
        assert called["pid"] == "BTC-USD"
        assert called["kwargs"].get("source") == "live"

    def test_v3_xgb_prob_pid_none_returns_neutral(self, tmp_path, monkeypatch, fresh_xgb_module, caplog):
        import logging
        _train_tiny_v3(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        with caplog.at_level(logging.WARNING):
            p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid=None)
        assert p == 0.5
        assert any("pid" in r.message.lower() for r in caplog.records)

    def test_v3_returns_neutral_on_tiered_fetch_failure(self, tmp_path, monkeypatch, fresh_xgb_module):
        _train_tiny_v3(str(tmp_path))
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))

        def boom(pid, **kwargs):
            raise RuntimeError("simulated DB failure")
        monkeypatch.setattr("services.tiered_history.fetch_tiered", boom)

        p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert p == 0.5

    def test_v3_skips_v1_calibrator_on_metadata_mismatch(self, tmp_path, monkeypatch, fresh_xgb_module, caplog):
        import logging
        from sklearn.isotonic import IsotonicRegression
        _train_tiny_v3(str(tmp_path))
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
        )
        with open(tmp_path / "xgb_calibration.pkl", "wb") as f:
            pickle.dump({"calibrator": iso, "feature_set": "v1"}, f)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "xgb_calibration.pkl"))

        def fake_fetch(pid, **kwargs):
            return _fake_v3_tiers()
        monkeypatch.setattr("services.tiered_history.fetch_tiered", fake_fetch)

        with caplog.at_level(logging.WARNING):
            fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert fresh_xgb_module._calibration is None
        assert any(
            "feature_set" in r.message.lower() or "calibrator" in r.message.lower()
            for r in caplog.records
        )


# ── v4 shadow path (#xgb-v4 / Step B.1) ───────────────────────────────────


class TestV4ShadowLoad:
    def test_try_load_v4_returns_false_when_artifacts_missing(self, monkeypatch):
        import agents.xgb_signal as xs
        # Force v4 artifacts to point at non-existent files
        monkeypatch.setattr(xs, "_MODEL_PATH_V4", "/nonexistent/v4_model.json")
        monkeypatch.setattr(xs, "_FEATURES_PATH_V4", "/nonexistent/v4_feat.json")
        monkeypatch.setattr(xs, "_load_attempted_v4", False)
        monkeypatch.setattr(xs, "_load_succeeded_v4", False)
        assert xs._try_load_v4() is False


class TestXgbProbV4:
    def test_returns_neutral_when_artifacts_missing(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_MODEL_PATH_V4", "/nonexistent/v4.json")
        monkeypatch.setattr(xs, "_FEATURES_PATH_V4", "/nonexistent/v4f.json")
        monkeypatch.setattr(xs, "_load_attempted_v4", False)
        monkeypatch.setattr(xs, "_load_succeeded_v4", False)
        out = xs.xgb_prob_v4(channels=None, pid="BTC-USD")
        assert out == xs._NEUTRAL

    def test_returns_neutral_when_pid_none(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_load_attempted_v4", True)
        monkeypatch.setattr(xs, "_load_succeeded_v4", True)
        out = xs.xgb_prob_v4(channels=None, pid=None)
        assert out == xs._NEUTRAL


class TestXgbProbShadow:
    def test_shadow_returns_tuple(self, monkeypatch):
        import agents.xgb_signal as xs
        # Stub both v3 and v4 to known values
        monkeypatch.setattr(xs, "xgb_prob",
                            lambda channels, pid=None: 0.7)
        monkeypatch.setattr(xs, "xgb_prob_v4",
                            lambda channels, pid=None: 0.4)
        v3, v4 = xs.xgb_prob_shadow(channels=None, pid="BTC-USD")
        assert v3 == 0.7
        assert v4 == 0.4

    def test_shadow_v4_failure_isolated_from_v3(self, monkeypatch, caplog):
        """v4 raising MUST NOT affect v3 driver path."""
        import logging
        import agents.xgb_signal as xs

        def boom(*a, **kw):
            raise RuntimeError("v4 boom")

        monkeypatch.setattr(xs, "xgb_prob",
                            lambda channels, pid=None: 0.6)
        monkeypatch.setattr(xs, "xgb_prob_v4", boom)
        with caplog.at_level(logging.ERROR):
            v3, v4 = xs.xgb_prob_shadow(channels=None, pid="BTC-USD")
        assert v3 == 0.6     # driver still works
        assert v4 is None    # shadow captured as None
        assert any("v4" in r.message.lower() for r in caplog.records)

    def test_shadow_v3_failure_propagates(self, monkeypatch):
        """v3 path is NOT wrapped — exceptions propagate as before."""
        import agents.xgb_signal as xs

        def boom_v3(*a, **kw):
            raise RuntimeError("v3 boom")

        monkeypatch.setattr(xs, "xgb_prob", boom_v3)
        monkeypatch.setattr(xs, "xgb_prob_v4",
                            lambda channels, pid=None: 0.3)
        with pytest.raises(RuntimeError, match="v3 boom"):
            xs.xgb_prob_shadow(channels=None, pid="BTC-USD")
