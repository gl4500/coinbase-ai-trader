"""TDD tests for agents/mc/ci_filter.py — entry confidence-interval filter.

Contract:
    CIFilter.evaluate(side, model_prob, pid, channels, context)
        - Loads the v3 booster via agents.xgb_signal (already cached).
        - Returns ("BUY", {"ci": {...}}) if lower_bound > cnn_buy_threshold.
        - Returns ("HOLD", {"ci": {...}}) otherwise.
        - K is read from MC_CI_K env, default 1.0.
        - Skips (passes through unchanged) for v1/v2 booster, missing pid,
          missing booster, or any predict failure — telemetry records the
          skip reason.
"""

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
def fresh_xs(monkeypatch):
    for mod in list(sys.modules):
        if mod.startswith("agents.xgb_signal") or mod.startswith("agents.mc"):
            del sys.modules[mod]
    yield


class _FakeBooster:
    """Stand-in for xgboost.Booster — predictions follow a known trajectory."""

    def __init__(self, n_rounds: int = 5):
        self._n = n_rounds

    def num_boosted_rounds(self):
        return self._n

    def predict(self, dmat, iteration_range=None):
        k = iteration_range[1] if iteration_range else self._n
        # Trajectory: 0.50, 0.55, 0.60, 0.65, 0.70 → stdev ~= 0.0707
        val = 0.5 + (k - 1) * 0.05
        return np.array([val], dtype=np.float64)


def _patch_v3_inference(monkeypatch, booster=None):
    """Wire xgb_signal + extract_features + fetch_tiered to safe stubs so
    CIFilter can run end-to-end without hitting the real model."""
    import agents.xgb_signal as xs

    monkeypatch.setattr(xs, "_booster", booster if booster is not None else _FakeBooster(5))
    monkeypatch.setattr(xs, "_feature_set", "v3")
    monkeypatch.setattr(xs, "_feature_names", ["f"] * 350)
    monkeypatch.setattr(xs, "_load_succeeded", True)
    import xgboost as xgb_mod

    class _FakeDM:
        def __init__(self, *a, **kw):
            pass

    monkeypatch.setattr(xgb_mod, "DMatrix", _FakeDM)
    monkeypatch.setattr(
        "services.tiered_history.fetch_tiered",
        lambda pid, **kw: {"micro": [], "meso": [], "macro": []},
    )
    import tools.xgb_features as xf

    monkeypatch.setattr(
        xf,
        "extract_features",
        lambda tiers, feature_set="v3": (np.zeros((1, 350)), ["f"] * 350),
    )


class TestCIFilterCore:
    def test_evaluate_keeps_buy_when_lower_bound_exceeds_threshold(self, fresh_xs, monkeypatch):
        monkeypatch.setenv("MC_CI_K", "1.0")
        _patch_v3_inference(monkeypatch)
        import config as cfg

        monkeypatch.setattr(cfg.config, "cnn_buy_threshold", 0.55)

        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        # Trajectory stdev ~= 0.0707, point=0.70 (final predict)
        # lower = 0.70 - 1.0 * 0.0707 = 0.6293 > 0.55 -> KEEP
        assert side == "BUY"
        assert tele["ci"]["decision"] == "keep"
        assert tele["ci"]["stdev"] == pytest.approx(0.0707, abs=0.005)
        assert tele["ci"]["lower"] == pytest.approx(0.6293, abs=0.005)

    def test_evaluate_blocks_buy_when_lower_bound_below_threshold(self, fresh_xs, monkeypatch):
        monkeypatch.setenv("MC_CI_K", "1.0")
        _patch_v3_inference(monkeypatch)
        import config as cfg

        # Threshold high enough that the lower bound fails (0.6293 < 0.65)
        monkeypatch.setattr(cfg.config, "cnn_buy_threshold", 0.65)

        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "HOLD"
        assert tele["ci"]["decision"] == "block"

    def test_evaluate_skips_under_non_v3_booster(self, fresh_xs, monkeypatch):
        import agents.xgb_signal as xs

        monkeypatch.setattr(xs, "_feature_set", "v1")
        monkeypatch.setattr(xs, "_load_succeeded", True)
        monkeypatch.setattr(xs, "_booster", _FakeBooster(5))
        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "non-v3-booster"

    def test_evaluate_skips_when_pid_none(self, fresh_xs, monkeypatch):
        import agents.xgb_signal as xs

        monkeypatch.setattr(xs, "_feature_set", "v3")
        monkeypatch.setattr(xs, "_booster", _FakeBooster(5))
        monkeypatch.setattr(xs, "_load_succeeded", True)
        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid=None,
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "pid-none"

    def test_evaluate_skips_when_booster_unavailable(self, fresh_xs, monkeypatch):
        import agents.xgb_signal as xs

        monkeypatch.setattr(xs, "_booster", None)
        monkeypatch.setattr(xs, "_load_succeeded", False)
        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "booster-unavailable"

    def test_evaluate_skips_on_predict_error(self, fresh_xs, monkeypatch):
        class _BrokenBooster:
            def num_boosted_rounds(self):
                return 5

            def predict(self, *a, **kw):
                raise RuntimeError("simulated predict failure")

        _patch_v3_inference(monkeypatch, booster=_BrokenBooster())
        from agents.mc.ci_filter import CIFilter

        side, tele = CIFilter().evaluate(
            side="BUY",
            model_prob=0.70,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "predict-error"
