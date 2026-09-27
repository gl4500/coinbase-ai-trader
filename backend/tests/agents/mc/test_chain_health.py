"""A configured filter that never registers must not be able to hide.

Established by execution on 2026-09-27: commit 589b571 (2026-08-01, "style: ruff lint +
format cleanup") deleted this line from cnn_agent.generate_signal --

    from agents.mc import ci_filter as _ci_filter  # registers ci into _FILTER_CLASSES

-- an import whose ONLY purpose was its registration side effect, which is exactly what a
lint autofix flags as unused. With MC_FILTERS=ci still set, _build_chain finds nothing for
'ci', logs one warning, and apply_buy_filters permits every BUY.

The telemetry confirms the date: mc_telemetry populated on every gate-crossing through
2026-08-06 and on none after. 2026-08-11 alone shows 216 crossings, 216 BUYs, 0 telemetry.

These tests cover the half that is safe to fix now -- OBSERVABILITY. Restoring the
registration changes live trading behaviour (it would block 78-88% of candidates on
June/July rates) and is the operator's decision, so that half is recorded as xfail rather
than silently repaired.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from agents.mc import registry
from agents.mc.base import BuyFilter


class _Spy(BuyFilter):
    name = "spy"

    def evaluate(self, side, model_prob, pid, channels, context):
        return "HOLD", {"spy": {"decision": "block"}}


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setattr(registry, "_FILTER_CLASSES", {}, raising=False)
    registry._reset_chain_cache()
    yield
    registry._reset_chain_cache()


def _apply(prob=0.61):
    return registry.apply_buy_filters("BUY", prob, "AAA-USD", [[1.0]], {})


def test_a_configured_filter_that_is_not_registered_currently_permits_the_buy(monkeypatch):
    """Characterisation of the live defect, so the behaviour is pinned rather than assumed."""
    monkeypatch.setenv("MC_FILTERS", "ci")
    registry._reset_chain_cache()
    assert _apply() == ("BUY", {})


def test_chain_health_reports_configured_filters_that_are_not_active(monkeypatch):
    """The missing capability. Nothing today can answer "is my configured chain running?",
    which is why this survived from 2026-08-01 to 2026-09-27 unnoticed."""
    monkeypatch.setenv("MC_FILTERS", "ci,spy")
    monkeypatch.setitem(registry._FILTER_CLASSES, "spy", _Spy)
    registry._reset_chain_cache()

    health = registry.chain_health()

    assert health["configured"] == ["ci", "spy"]
    assert health["active"] == ["spy"]
    assert health["missing"] == ["ci"]
    assert health["healthy"] is False
    assert "ci" in health["summary"]


def test_chain_health_is_healthy_when_every_configured_filter_resolves(monkeypatch):
    monkeypatch.setenv("MC_FILTERS", "spy")
    monkeypatch.setitem(registry._FILTER_CLASSES, "spy", _Spy)
    registry._reset_chain_cache()

    health = registry.chain_health()
    assert health["configured"] == ["spy"]
    assert health["active"] == ["spy"]
    assert health["missing"] == []
    assert health["healthy"] is True


def test_an_empty_configuration_is_healthy_not_broken(monkeypatch):
    """MC_FILTERS="" is the documented default and a noop. Absence of filters is not a
    fault; only a filter that was ASKED FOR and is missing is."""
    monkeypatch.setenv("MC_FILTERS", "")
    registry._reset_chain_cache()

    health = registry.chain_health()
    assert health["configured"] == []
    assert health["missing"] == []
    assert health["healthy"] is True


def test_chain_health_does_not_disturb_the_cached_chain(monkeypatch):
    """A health query must be safe to call from anywhere, including mid-scan. If it rebuilt
    or cleared the chain it would be a side effect masquerading as an observation."""
    monkeypatch.setenv("MC_FILTERS", "spy")
    monkeypatch.setitem(registry._FILTER_CLASSES, "spy", _Spy)
    registry._reset_chain_cache()

    assert _apply() == ("HOLD", {"spy": {"decision": "block"}})
    built = list(registry._chain)
    registry.chain_health()
    assert registry._chain is built or [type(f) for f in registry._chain] == [
        type(f) for f in built
    ]
    assert _apply() == ("HOLD", {"spy": {"decision": "block"}})


@pytest.mark.xfail(
    reason=(
        "REGRESSION 589b571 (2026-08-01): the registration import was removed from "
        "cnn_agent, so MC_FILTERS=ci resolves to nothing. Restoring it changes LIVE "
        "trading behaviour -- on June/July rates it blocks 78-88% of candidates -- so the "
        "fix is an operator decision, not a lint repair. Recorded as xfail so the defect "
        "is executable and dated without either breaking CI or making that change."
    ),
    strict=True,
)
def test_the_production_entry_point_registers_its_configured_filters(monkeypatch):
    """Importing what production imports must make a configured filter active."""
    monkeypatch.setattr(registry, "_FILTER_CLASSES", {}, raising=False)
    monkeypatch.setenv("MC_FILTERS", "ci")
    registry._reset_chain_cache()

    import agents.cnn_agent  # noqa: F401  -- the production import path

    assert registry.chain_health()["missing"] == [], (
        "MC_FILTERS=ci is configured but importing cnn_agent does not register it"
    )
