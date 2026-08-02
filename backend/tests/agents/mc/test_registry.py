"""TDD tests for agents/mc/registry.py — MC filter dispatch.

Contract:
    apply_buy_filters(side, model_prob, pid, channels, context) -> (side, telemetry_dict)
        - When MC_FILTERS env is empty (or unset), returns (side, {}) unchanged.
        - When MC_FILTERS="ci", calls CIFilter.evaluate and merges its telemetry.
        - When MC_FILTERS contains an unknown name, logs a warning and skips it.
        - Filter chain order matches the comma-separated MC_FILTERS order.
"""

import logging
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
def fresh_registry(monkeypatch):
    """Yield a freshly-imported agents.mc.registry so each test reads
    MC_FILTERS at import time."""
    for mod in list(sys.modules):
        if mod.startswith("agents.mc"):
            del sys.modules[mod]
    yield
    for mod in list(sys.modules):
        if mod.startswith("agents.mc"):
            del sys.modules[mod]


class TestRegistryDispatch:
    def test_empty_mc_filters_returns_unchanged(self, fresh_registry, monkeypatch):
        monkeypatch.setenv("MC_FILTERS", "")
        from agents.mc import registry

        side, tele = registry.apply_buy_filters(
            side="BUY",
            model_prob=0.7,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele == {}

    def test_unset_mc_filters_returns_unchanged(self, fresh_registry, monkeypatch):
        monkeypatch.delenv("MC_FILTERS", raising=False)
        from agents.mc import registry

        side, tele = registry.apply_buy_filters(
            side="BUY",
            model_prob=0.7,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "BUY"
        assert tele == {}

    def test_ci_filter_invoked_when_listed(self, fresh_registry, monkeypatch):
        monkeypatch.setenv("MC_FILTERS", "ci")
        from agents.mc import registry

        called = {}

        class SpyCI:
            name = "ci"

            def evaluate(self, side, model_prob, pid, channels, context):
                called["hit"] = True
                return side, {"ci": {"stdev": 0.01, "lower": model_prob - 0.01}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES", {"ci": SpyCI})
        registry._reset_chain_cache()
        side, tele = registry.apply_buy_filters(
            side="BUY",
            model_prob=0.7,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert called.get("hit") is True
        assert "ci" in tele

    def test_unknown_filter_warns_and_skips(self, fresh_registry, monkeypatch, caplog):
        monkeypatch.setenv("MC_FILTERS", "bogus")
        from agents.mc import registry

        with caplog.at_level(logging.WARNING):
            side, tele = registry.apply_buy_filters(
                side="BUY",
                model_prob=0.7,
                pid="BTC-USD",
                channels=[[0.0] * 60] * 28,
                context={},
            )
        assert side == "BUY"
        assert tele == {}
        assert any("bogus" in r.message.lower() for r in caplog.records)

    def test_chain_order_matches_env(self, fresh_registry, monkeypatch):
        monkeypatch.setenv("MC_FILTERS", "second,first")
        from agents.mc import registry

        order = []

        class FilterA:
            name = "first"

            def evaluate(self, side, model_prob, pid, channels, context):
                order.append("first")
                return side, {"first": {}}

        class FilterB:
            name = "second"

            def evaluate(self, side, model_prob, pid, channels, context):
                order.append("second")
                return side, {"second": {}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES", {"first": FilterA, "second": FilterB})
        registry._reset_chain_cache()
        registry.apply_buy_filters(
            side="BUY",
            model_prob=0.7,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert order == ["second", "first"]

    def test_filter_can_change_side(self, fresh_registry, monkeypatch):
        monkeypatch.setenv("MC_FILTERS", "blocker")
        from agents.mc import registry

        class Blocker:
            name = "blocker"

            def evaluate(self, side, model_prob, pid, channels, context):
                return "HOLD", {"blocker": {"reason": "test-block"}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES", {"blocker": Blocker})
        registry._reset_chain_cache()
        side, tele = registry.apply_buy_filters(
            side="BUY",
            model_prob=0.7,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "HOLD"
        assert tele["blocker"]["reason"] == "test-block"

    def test_filter_exception_does_not_kill_chain(self, fresh_registry, monkeypatch, caplog):
        monkeypatch.setenv("MC_FILTERS", "broken,working")
        from agents.mc import registry

        class Broken:
            name = "broken"

            def evaluate(self, side, model_prob, pid, channels, context):
                raise RuntimeError("simulated crash")

        class Working:
            name = "working"

            def evaluate(self, side, model_prob, pid, channels, context):
                return side, {"working": {"ok": True}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES", {"broken": Broken, "working": Working})
        registry._reset_chain_cache()
        with caplog.at_level(logging.WARNING):
            side, tele = registry.apply_buy_filters(
                side="BUY",
                model_prob=0.7,
                pid="BTC-USD",
                channels=[[0.0] * 60] * 28,
                context={},
            )
        assert side == "BUY"
        assert "working" in tele
        assert "broken" not in tele
        assert any("broken" in r.message.lower() for r in caplog.records)

    def test_apply_buy_filters_only_runs_for_buy_side(self, fresh_registry, monkeypatch):
        """SELL/HOLD pass through untouched (filters are entry-only for MVP)."""
        monkeypatch.setenv("MC_FILTERS", "ci")
        from agents.mc import registry

        called = {"hit": False}

        class SpyCI:
            name = "ci"

            def evaluate(self, side, model_prob, pid, channels, context):
                called["hit"] = True
                return side, {"ci": {}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES", {"ci": SpyCI})
        registry._reset_chain_cache()
        side, tele = registry.apply_buy_filters(
            side="HOLD",
            model_prob=0.5,
            pid="BTC-USD",
            channels=[[0.0] * 60] * 28,
            context={},
        )
        assert side == "HOLD"
        assert tele == {}
        assert called["hit"] is False
