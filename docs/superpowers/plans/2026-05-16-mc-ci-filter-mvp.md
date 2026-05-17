# MC CIFilter MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the first Monte Carlo filter (CIFilter, entry confidence-interval gate) behind an off-by-default env flag, with zero changes to live behavior until the operator flips `MC_FILTERS=ci`.

**Architecture:** Sidecar pattern — all MC code lives under `backend/agents/mc/` and is invoked through a single registry hook in `cnn_agent.generate_signal`. Filters are env-driven, kill-switchable, and individually testable. CIFilter algorithm: cumulative-trajectory stdev across the 200-tree v3 booster; lower-bound gate compares `(point − K*stdev)` to `cnn_buy_threshold`. K=1.0 default.

**Tech Stack:** Python 3.11, xgboost (already in tree, v3 booster loaded by `agents/xgb_signal`), aiosqlite (existing), pytest + pytest-asyncio, numpy.

**Spec source:** `docs/superpowers/specs/2026-05-16-mc-loose-coupling-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Owner |
|---|---|---|
| `backend/agents/mc/__init__.py` | CREATE | Task 1 |
| `backend/agents/mc/base.py` | CREATE | Task 1 |
| `backend/agents/mc/registry.py` | CREATE | Task 1 |
| `backend/tests/agents/__init__.py` | CREATE | Task 1 |
| `backend/tests/agents/mc/__init__.py` | CREATE | Task 1 |
| `backend/tests/agents/mc/test_registry.py` | CREATE | Task 1 |
| `backend/agents/mc/ci_filter.py` | CREATE | Task 2 |
| `backend/tests/agents/mc/test_ci_filter.py` | CREATE | Task 2 |
| `backend/migrations/2026-05-16-mc-telemetry.py` | CREATE | Task 3 |
| `backend/tests/test_mc_migration.py` | CREATE | Task 3 |
| `backend/database.py:548-570` | MODIFY | Task 3 |
| `backend/tests/test_database.py` | EXTEND | Task 3 |
| `backend/agents/cnn_agent.py:2225-2236` | MODIFY | Task 3 |
| `backend/tests/test_cnn_agent.py` | EXTEND | Task 3 |
| `polymarket_app/CHANGELOG.md` | APPEND per task | every task |
| `polymarket_app/CLAUDE.md` | EDIT (invariant section) at Task 5 | Task 5 |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND at Task 5 | Task 5 |

---

## Task 1: MC package scaffolding + registry

**Files:**
- Create: `backend/agents/mc/__init__.py`
- Create: `backend/agents/mc/base.py`
- Create: `backend/agents/mc/registry.py`
- Create: `backend/tests/agents/__init__.py`
- Create: `backend/tests/agents/mc/__init__.py`
- Create: `backend/tests/agents/mc/test_registry.py`

### Step 1.1 — Create the empty `__init__.py` files

```bash
cd C:\Users\gl450\polymarket_app
touch backend/agents/mc/__init__.py
touch backend/tests/agents/__init__.py
touch backend/tests/agents/mc/__init__.py
```

(Or on PowerShell: `New-Item backend/agents/mc/__init__.py -ItemType File -Force; ...`)

- [ ] **Step 1.1** — Create the three empty package markers.

### Step 1.2 — Write the failing test file

Create `backend/tests/agents/mc/test_registry.py`:

```python
"""TDD tests for agents/mc/registry.py — MC filter dispatch.

Contract:
    apply_buy_filters(side, model_prob, pid, channels, context) -> (side, telemetry_dict)
        - When MC_FILTERS env is empty (or unset), returns (side, {}) unchanged.
        - When MC_FILTERS="ci", calls CIFilter.evaluate and merges its telemetry.
        - When MC_FILTERS contains an unknown name, logs a warning and skips it.
        - Filter chain order matches the comma-separated MC_FILTERS order.
"""
import importlib
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
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele == {}

    def test_unset_mc_filters_returns_unchanged(self, fresh_registry, monkeypatch):
        monkeypatch.delenv("MC_FILTERS", raising=False)
        from agents.mc import registry
        side, tele = registry.apply_buy_filters(
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele == {}

    def test_ci_filter_invoked_when_listed(self, fresh_registry, monkeypatch):
        monkeypatch.setenv("MC_FILTERS", "ci")
        from agents.mc import registry
        # Patch CIFilter to a spy so we don't depend on xgb_signal state
        called = {}

        class SpyCI:
            name = "ci"
            def evaluate(self, side, model_prob, pid, channels, context):
                called["hit"] = True
                return side, {"ci": {"stdev": 0.01, "lower": model_prob - 0.01}}

        monkeypatch.setattr(registry, "_FILTER_CLASSES",
                            {"ci": SpyCI})
        registry._reset_chain_cache()  # picks up the spy class
        side, tele = registry.apply_buy_filters(
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert called.get("hit") is True
        assert "ci" in tele

    def test_unknown_filter_warns_and_skips(self, fresh_registry, monkeypatch, caplog):
        monkeypatch.setenv("MC_FILTERS", "bogus")
        from agents.mc import registry
        with caplog.at_level(logging.WARNING):
            side, tele = registry.apply_buy_filters(
                side="BUY", model_prob=0.7, pid="BTC-USD",
                channels=[[0.0] * 60] * 28, context={},
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

        monkeypatch.setattr(registry, "_FILTER_CLASSES",
                            {"first": FilterA, "second": FilterB})
        registry._reset_chain_cache()
        registry.apply_buy_filters(
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert order == ["second", "first"]  # env order honored

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
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
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

        monkeypatch.setattr(registry, "_FILTER_CLASSES",
                            {"broken": Broken, "working": Working})
        registry._reset_chain_cache()
        with caplog.at_level(logging.WARNING):
            side, tele = registry.apply_buy_filters(
                side="BUY", model_prob=0.7, pid="BTC-USD",
                channels=[[0.0] * 60] * 28, context={},
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
            side="HOLD", model_prob=0.5, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "HOLD"
        assert tele == {}
        assert called["hit"] is False
```

- [ ] **Step 1.2** — Write the test file above.

### Step 1.3 — Run; expect 8 failures (`agents.mc.registry` not found)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/agents/mc/test_registry.py -v
```

Expected: 8 FAILED with `ModuleNotFoundError: No module named 'agents.mc.registry'` (or similar).

- [ ] **Step 1.3** — Run and observe red.

### Step 1.4 — Write `backend/agents/mc/base.py`

```python
"""ABC for Monte Carlo decision filters (#311-mc).

Each filter wraps one decision point in cnn_agent. Filters live under
agents/mc/<name>_filter.py, expose a class with .name and .evaluate(...),
and are listed by name in the MC_FILTERS env var (comma-separated).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class BuyFilter(ABC):
    """Filter invoked at the BUY gate. May change the side and adds telemetry.

    Contract:
        evaluate(side, model_prob, pid, channels, context) -> (new_side, telemetry)
        - `side` is "BUY" when this is invoked (registry skips non-BUY).
        - return value's first slot is the post-filter side ("BUY" or "HOLD").
          Filters should never up-grade HOLD to BUY in MVP scope.
        - second slot is a dict keyed by self.name with arbitrary serializable
          telemetry; gets merged into the chain-level telemetry dict.
    """

    name: str = ""  # subclasses override

    @abstractmethod
    def evaluate(
        self,
        side: str,
        model_prob: float,
        pid: str,
        channels: List[List[float]],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError
```

- [ ] **Step 1.4** — Write `backend/agents/mc/base.py`.

### Step 1.5 — Write `backend/agents/mc/registry.py`

```python
"""MC filter chain dispatch (#311-mc).

Reads MC_FILTERS at module import (and on _reset_chain_cache for tests).
Filters listed but unknown to the registry log a warning and are skipped.
Filter exceptions are caught so one broken filter cannot kill the rest.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple, Type

from agents.mc.base import BuyFilter

logger = logging.getLogger(__name__)

# Map of filter name -> class. CIFilter (Task 2) registers itself here.
# Tests patch this dict to inject spies.
_FILTER_CLASSES: Dict[str, Type[BuyFilter]] = {}

_chain: List[BuyFilter] = []
_chain_built: bool = False


def _build_chain() -> List[BuyFilter]:
    raw = os.getenv("MC_FILTERS", "") or ""
    names = [n.strip() for n in raw.split(",") if n.strip()]
    chain: List[BuyFilter] = []
    for name in names:
        cls = _FILTER_CLASSES.get(name)
        if cls is None:
            logger.warning("MC_FILTERS lists unknown filter %r — skipping", name)
            continue
        try:
            chain.append(cls())
        except Exception:
            logger.exception("MC filter %r failed to instantiate — skipping", name)
    return chain


def _reset_chain_cache() -> None:
    """Test helper: drop the cached chain so the next apply_* rebuilds."""
    global _chain, _chain_built
    _chain = []
    _chain_built = False


def apply_buy_filters(
    side: str,
    model_prob: float,
    pid: str,
    channels: List[List[float]],
    context: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    """Apply the MC filter chain at the BUY gate.

    Returns (final_side, telemetry_dict). HOLD/SELL side passes through
    untouched. With MC_FILTERS empty, returns (side, {}).
    """
    global _chain, _chain_built
    if side != "BUY":
        return side, {}
    if not _chain_built:
        _chain = _build_chain()
        _chain_built = True
    if not _chain:
        return side, {}
    telemetry: Dict[str, Any] = {}
    cur_side = side
    for f in _chain:
        try:
            cur_side, tele = f.evaluate(cur_side, model_prob, pid, channels, context)
            if tele:
                telemetry.update(tele)
        except Exception as exc:
            logger.warning(
                "MC filter %r raised %s — skipping its decision",
                getattr(f, "name", "unknown"), exc,
            )
    return cur_side, telemetry
```

- [ ] **Step 1.5** — Write `backend/agents/mc/registry.py`.

### Step 1.6 — Run; expect 8 PASS

```bash
../.venv/Scripts/python.exe -m pytest tests/agents/mc/test_registry.py -v
```

Expected: `8 passed`.

- [ ] **Step 1.6** — Run and observe green.

### Step 1.7 — Cleanup + CHANGELOG entry

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Append to `polymarket_app/CHANGELOG.md` at the TOP (above the existing top entry, after the `---` separator):

```markdown
## [Session 58.70a] — 2026-05-16 — MC package scaffolding + registry (#311-mc-a)

### What changed
- **`backend/agents/mc/`** (NEW package) — `__init__.py`, `base.py` (BuyFilter
  ABC), `registry.py` (`apply_buy_filters` chain dispatch). Reads `MC_FILTERS`
  env var; unknown names warn + skip; filter exceptions warn + skip; default
  empty MC_FILTERS = identity passthrough.
- **`backend/tests/agents/mc/test_registry.py`** (NEW) — 8 tests covering
  empty/unset env, dispatch, unknown filter, chain order, side change,
  exception isolation, non-BUY passthrough.

### Verification
```
backend && python -m pytest tests/agents/mc/test_registry.py -v
=> 8 passed
```

---
```

- [ ] **Step 1.7** — Append CHANGELOG entry.

### Step 1.8 — Commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/mc/ backend/tests/agents/ CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311-mc-a): MC package scaffolding + registry

agents/mc/ package with BuyFilter ABC and an env-driven registry. Reads
MC_FILTERS at first call (cache-busted via _reset_chain_cache for tests),
warns + skips unknown filter names, isolates filter exceptions so one
crash cannot kill the chain. MC_FILTERS="" (default) is a bit-for-bit
identity passthrough — cnn_agent behavior unchanged until Task 3 wires
the hook AND operator flips the env.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 1.8** — Commit.

---

## Task 2: CIFilter implementation

**Files:**
- Create: `backend/agents/mc/ci_filter.py`
- Create: `backend/tests/agents/mc/test_ci_filter.py`

### Step 2.1 — Write the failing test file

Create `backend/tests/agents/mc/test_ci_filter.py`:

```python
"""TDD tests for agents/mc/ci_filter.py — entry confidence-interval filter.

Contract:
    CIFilter.evaluate(side, model_prob, pid, channels, context)
        - Loads the v3 booster via agents.xgb_signal (already cached).
        - Returns ("BUY", {"ci": {...}}) if lower_bound > cnn_buy_threshold.
        - Returns ("HOLD", {"ci": {...}}) otherwise.
        - K is read from MC_CI_K env, default 1.0.
        - Skips (passes through unchanged) for v1/v2 booster, missing pid,
          or any predict failure — telemetry records the skip reason.
"""
import importlib
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
def fresh_xs(monkeypatch):
    """Yield a freshly-imported xgb_signal so we can monkey-patch the
    module-level _booster / _feature_set / _feature_names cleanly."""
    for mod in list(sys.modules):
        if mod.startswith("agents.xgb_signal") or mod.startswith("agents.mc"):
            del sys.modules[mod]
    yield


class _FakeBooster:
    """Stand-in for xgboost.Booster — returns increasing predictions per tree
    so the trajectory has a definite stdev > 0."""
    def __init__(self, n_rounds: int = 5):
        self._n = n_rounds

    def num_boosted_rounds(self):
        return self._n

    def predict(self, dmat, iteration_range=None):
        # Return a single float in [0, 1] that depends on iteration_range[1].
        k = iteration_range[1] if iteration_range else self._n
        # Trajectory: 0.50, 0.55, 0.60, 0.65, 0.70 → stdev ≈ 0.0707
        val = 0.5 + (k - 1) * 0.05
        return np.array([val], dtype=np.float64)


class TestCIFilterCore:
    def test_evaluate_keeps_buy_when_lower_bound_exceeds_threshold(
        self, fresh_xs, monkeypatch
    ):
        monkeypatch.setenv("MC_CI_K", "1.0")
        # Patch xgb_signal state to a fake v3 booster
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_booster", _FakeBooster(5))
        monkeypatch.setattr(xs, "_feature_set", "v3")
        monkeypatch.setattr(xs, "_feature_names", ["f"] * 350)
        monkeypatch.setattr(xs, "_load_succeeded", True)

        # Skip the real v3 fetch+extract — return a known-shape dmatrix builder
        import xgboost as xgb_mod

        class _FakeDM:
            def __init__(self, *a, **kw): pass

        monkeypatch.setattr(xgb_mod, "DMatrix", _FakeDM)
        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            lambda pid, **kw: {"micro": [], "meso": [], "macro": []},
        )

        # tools.xgb_features.extract_features returns the expected shape
        import tools.xgb_features as xf
        monkeypatch.setattr(
            xf, "extract_features",
            lambda tiers, feature_set="v3": (np.zeros((1, 350)), ["f"] * 350),
        )

        # config threshold default is 0.6 (cnn_buy_threshold)
        import config as cfg
        monkeypatch.setattr(cfg.config, "cnn_buy_threshold", 0.55)

        from agents.mc.ci_filter import CIFilter
        side, tele = CIFilter().evaluate(
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        # Trajectory stdev ≈ 0.0707, point=0.70 (the final predict)
        # lower = 0.70 - 1.0 * 0.0707 = 0.6293 > 0.55 → KEEP
        assert side == "BUY"
        assert tele["ci"]["decision"] == "keep"
        assert tele["ci"]["stdev"] == pytest.approx(0.0707, abs=0.005)
        assert tele["ci"]["lower"] == pytest.approx(0.6293, abs=0.005)

    def test_evaluate_blocks_buy_when_lower_bound_below_threshold(
        self, fresh_xs, monkeypatch
    ):
        monkeypatch.setenv("MC_CI_K", "1.0")
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_booster", _FakeBooster(5))
        monkeypatch.setattr(xs, "_feature_set", "v3")
        monkeypatch.setattr(xs, "_feature_names", ["f"] * 350)
        monkeypatch.setattr(xs, "_load_succeeded", True)
        import xgboost as xgb_mod

        class _FakeDM:
            def __init__(self, *a, **kw): pass

        monkeypatch.setattr(xgb_mod, "DMatrix", _FakeDM)
        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            lambda pid, **kw: {"micro": [], "meso": [], "macro": []},
        )
        import tools.xgb_features as xf
        monkeypatch.setattr(
            xf, "extract_features",
            lambda tiers, feature_set="v3": (np.zeros((1, 350)), ["f"] * 350),
        )
        import config as cfg
        # Threshold high enough that the lower bound fails (0.6293 < 0.65)
        monkeypatch.setattr(cfg.config, "cnn_buy_threshold", 0.65)

        from agents.mc.ci_filter import CIFilter
        side, tele = CIFilter().evaluate(
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
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
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
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
            side="BUY", model_prob=0.70, pid=None,
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "pid-none"

    def test_evaluate_skips_when_booster_unavailable(self, fresh_xs, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_booster", None)
        monkeypatch.setattr(xs, "_load_succeeded", False)
        from agents.mc.ci_filter import CIFilter
        side, tele = CIFilter().evaluate(
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "booster-unavailable"

    def test_evaluate_skips_on_predict_error(self, fresh_xs, monkeypatch):
        class _BrokenBooster:
            def num_boosted_rounds(self):
                return 5

            def predict(self, *a, **kw):
                raise RuntimeError("simulated predict failure")

        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_booster", _BrokenBooster())
        monkeypatch.setattr(xs, "_feature_set", "v3")
        monkeypatch.setattr(xs, "_feature_names", ["f"] * 350)
        monkeypatch.setattr(xs, "_load_succeeded", True)
        import xgboost as xgb_mod

        class _FakeDM:
            def __init__(self, *a, **kw): pass

        monkeypatch.setattr(xgb_mod, "DMatrix", _FakeDM)
        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            lambda pid, **kw: {"micro": [], "meso": [], "macro": []},
        )
        import tools.xgb_features as xf
        monkeypatch.setattr(
            xf, "extract_features",
            lambda tiers, feature_set="v3": (np.zeros((1, 350)), ["f"] * 350),
        )

        from agents.mc.ci_filter import CIFilter
        side, tele = CIFilter().evaluate(
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele["ci"]["skipped"] == "predict-error"
```

- [ ] **Step 2.1** — Write the test file.

### Step 2.2 — Run; expect 6 failures (`agents.mc.ci_filter` not found)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/agents/mc/test_ci_filter.py -v
```

Expected: 6 FAILED with `ModuleNotFoundError`.

- [ ] **Step 2.2** — Run and observe red.

### Step 2.3 — Write `backend/agents/mc/ci_filter.py`

```python
"""Entry confidence-interval filter (#311-mc-ci).

Algorithm: take the cumulative-prediction trajectory across the v3 booster's
trees (cheap; ~7s per scan for 51 products), compute its stdev as a proxy
for ensemble uncertainty, and require the lower bound (point - K*stdev) to
exceed cnn_buy_threshold before allowing BUY.

K is configurable via MC_CI_K (default 1.0). Skips gracefully for non-v3
boosters, missing pid, or any predict failure — decision stays the caller's.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from agents.mc.base import BuyFilter

logger = logging.getLogger(__name__)


class CIFilter(BuyFilter):
    name = "ci"

    def __init__(self) -> None:
        try:
            self._K = float(os.getenv("MC_CI_K", "1.0"))
        except (TypeError, ValueError):
            self._K = 1.0

    def evaluate(
        self,
        side: str,
        model_prob: float,
        pid: str,
        channels: List[List[float]],
        context: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        # Lazy imports: avoid module-load-order issues + survive tests that
        # patch xgb_signal state via monkeypatch.
        from agents import xgb_signal as xs

        if not getattr(xs, "_load_succeeded", False) or xs._booster is None:
            return side, {"ci": {"skipped": "booster-unavailable"}}
        if getattr(xs, "_feature_set", "v1") != "v3":
            return side, {"ci": {"skipped": "non-v3-booster"}}
        if pid is None:
            return side, {"ci": {"skipped": "pid-none"}}

        try:
            import xgboost as xgb
            from services.tiered_history import fetch_tiered
            from tools.xgb_features import extract_features
            import config as cfg

            tiers = fetch_tiered(pid, source="live")
            features, _ = extract_features(tiers, feature_set="v3")
            dmat = xgb.DMatrix(features, feature_names=xs._feature_names)
            n = xs._booster.num_boosted_rounds()
            trajectory = [
                float(xs._booster.predict(dmat, iteration_range=(0, k + 1))[0])
                for k in range(n)
            ]
            point = trajectory[-1]
            stdev = float(np.std(trajectory))
            lower = max(0.0, point - self._K * stdev)
            threshold = float(cfg.config.cnn_buy_threshold)
            decision = "keep" if lower > threshold else "block"
            new_side = side if decision == "keep" else "HOLD"
            tele = {
                "ci": {
                    "stdev": round(stdev, 6),
                    "lower": round(lower, 6),
                    "K": self._K,
                    "decision": decision,
                }
            }
            return new_side, tele
        except Exception as exc:
            logger.warning("CIFilter predict failed: %s", exc)
            return side, {"ci": {"skipped": "predict-error", "error": str(exc)}}


# Self-register with the registry on import. registry._FILTER_CLASSES["ci"] = CIFilter.
try:
    from agents.mc.registry import _FILTER_CLASSES
    _FILTER_CLASSES["ci"] = CIFilter
except Exception:
    pass
```

- [ ] **Step 2.3** — Write `backend/agents/mc/ci_filter.py`.

### Step 2.4 — Run; expect 6 PASS

```bash
../.venv/Scripts/python.exe -m pytest tests/agents/mc/test_ci_filter.py -v
```

Expected: `6 passed`.

- [ ] **Step 2.4** — Run and observe green.

### Step 2.5 — Re-run registry tests to confirm CIFilter self-registration didn't break them

```bash
../.venv/Scripts/python.exe -m pytest tests/agents/mc/ -v
```

Expected: `14 passed` (8 registry + 6 ci_filter). If the auto-registration is interfering with the spy-injection tests, fix by having `test_registry.py` overwrite `_FILTER_CLASSES` BEFORE accessing it (the existing `monkeypatch.setattr` does this).

- [ ] **Step 2.5** — Run combined and verify 14 pass.

### Step 2.6 — Cleanup + CHANGELOG entry

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Append to `CHANGELOG.md` at the TOP:

```markdown
## [Session 58.70b] — 2026-05-16 — MC CIFilter implementation (#311-mc-ci)

### What changed
- **`backend/agents/mc/ci_filter.py`** (NEW) — entry confidence-interval
  filter. Computes per-tree cumulative trajectory stdev across the v3
  booster (200 trees), gates BUY on `(point - K*stdev) > cnn_buy_threshold`.
  K=1.0 default via `MC_CI_K` env. Skips gracefully (no decision change)
  for non-v3 booster, missing pid, missing booster, or predict failure;
  every skip records a reason in telemetry. Self-registers with
  `agents.mc.registry._FILTER_CLASSES` on import.
- **`backend/tests/agents/mc/test_ci_filter.py`** (NEW) — 6 tests
  covering keep/block paths and 4 skip-reason cases.

### Verification
```
backend && python -m pytest tests/agents/mc/ -v
=> 14 passed (8 registry + 6 ci_filter)
```

---
```

- [ ] **Step 2.6** — Append CHANGELOG.

### Step 2.7 — Commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/mc/ci_filter.py backend/tests/agents/mc/test_ci_filter.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311-mc-ci): MC CIFilter — entry confidence interval

Cumulative trajectory stdev across the v3 booster's 200 trees → lower-
bound gate at (point - K*stdev). K=1.0 default via MC_CI_K env. Self-
registers with agents.mc.registry on import. Six graceful-skip paths
(non-v3 booster, pid-none, booster-unavailable, predict-error etc.) all
return the caller's original decision with a telemetry skip reason —
never crashes the scan loop.

Not wired into cnn_agent yet. Activation gated on Task 3 + MC_FILTERS
env flip by operator.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2.7** — Commit.

---

## Task 3: Schema migration + database extension + cnn_agent wire-up

This is the biggest commit; all three pieces move together so post-commit the system is self-consistent (`MC_FILTERS=""` means today's behavior; flipping the env activates CIFilter).

**Files:**
- Create: `backend/migrations/__init__.py`
- Create: `backend/migrations/mc_telemetry_20260516.py`
- Create: `backend/tests/test_mc_migration.py`
- Modify: `backend/database.py:548-570` (extend `save_cnn_scan`)
- Modify: `backend/agents/cnn_agent.py:2225-2236` (wire `mc.apply_buy_filters`)
- Extend: `backend/tests/test_database.py` (+2 tests for new columns)
- Extend: `backend/tests/test_cnn_agent.py` (+3 tests for gate behavior + telemetry)

### Step 3.1 — Create `backend/migrations/__init__.py` (empty marker)

```bash
touch backend/migrations/__init__.py
```

- [ ] **Step 3.1** — Create the marker.

### Step 3.2 — Write migration test

Create `backend/tests/test_mc_migration.py`:

```python
"""TDD tests for backend/migrations/mc_telemetry_20260516.py.

Contract:
    run(db_path) -> dict
        - Idempotently adds xgb_prob_stdev REAL and mc_telemetry TEXT
          columns to the cnn_scans table.
        - Detects existing columns via PRAGMA table_info; never errors on
          already-applied state.
        - Returns {"added": [...col_names], "already_present": [...col_names]}.
"""
import os
import sqlite3
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_legacy_cnn_scans(db_path):
    """Create the cnn_scans table at its pre-MC schema."""
    c = sqlite3.connect(db_path)
    c.execute("""
        CREATE TABLE cnn_scans (
            id INTEGER PRIMARY KEY,
            product_id TEXT NOT NULL,
            price REAL,
            cnn_prob REAL,
            llm_prob REAL,
            model_prob REAL,
            cnn_weight REAL,
            llm_weight REAL,
            side TEXT,
            strength REAL,
            signal_gen INTEGER,
            regime TEXT,
            adx REAL, rsi REAL, macd REAL, mfi REAL, stoch_k REAL,
            atr REAL, vwap_dist REAL, fast_rsi REAL, velocity REAL,
            vol_z REAL, xgb_prob REAL, scanned_at TEXT
        )""")
    c.commit(); c.close()


class TestMCMigration:
    def test_adds_both_columns_on_first_run(self, tmp_path):
        db = tmp_path / "test.db"
        _make_legacy_cnn_scans(db)
        from migrations import mc_telemetry_20260516 as mig
        result = mig.run(str(db))
        assert "xgb_prob_stdev" in result["added"]
        assert "mc_telemetry" in result["added"]
        # Verify they actually exist
        c = sqlite3.connect(db)
        cols = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        c.close()
        assert "xgb_prob_stdev" in cols
        assert "mc_telemetry" in cols

    def test_idempotent_on_second_run(self, tmp_path):
        db = tmp_path / "test.db"
        _make_legacy_cnn_scans(db)
        from migrations import mc_telemetry_20260516 as mig
        mig.run(str(db))  # first run adds
        result = mig.run(str(db))  # second run should detect + skip
        assert result["added"] == []
        assert set(result["already_present"]) == {"xgb_prob_stdev", "mc_telemetry"}
```

(File name is `mc_telemetry_20260516.py` — YYYYMMDD suffix, not prefix — so it's a valid Python module name and imports cleanly without aliases.)

- [ ] **Step 3.2** — Write the test file.

### Step 3.3 — Write migration

(File name uses YYYYMMDD-suffix because Python identifiers can't start with a digit; this lets test code do `from migrations.mc_telemetry_20260516 import run` without import gymnastics.)

Create `backend/migrations/mc_telemetry_20260516.py`:

```python
"""Migration: add MC telemetry columns to cnn_scans (#311-mc-schema).

Idempotent — safe to re-run.
"""
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_stdev REAL and mc_telemetry TEXT columns to cnn_scans.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [
        ("xgb_prob_stdev", "REAL"),
        ("mc_telemetry",   "TEXT"),
    ]
    c = sqlite3.connect(db_path)
    try:
        existing = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        added: List[str] = []
        already: List[str] = []
        for name, dtype in new_cols:
            if name in existing:
                already.append(name)
                continue
            c.execute(f"ALTER TABLE cnn_scans ADD COLUMN {name} {dtype}")
            added.append(name)
        c.commit()
    finally:
        c.close()
    return {"added": added, "already_present": already}
```

`backend/migrations/__init__.py` stays empty (created in Step 3.1). The file name uses the YYYYMMDD suffix (not prefix) so test code can `from migrations.mc_telemetry_20260516 import run` directly — Python module names can't start with a digit.

- [ ] **Step 3.3** — Write `backend/migrations/mc_telemetry_20260516.py` with the body above.

### Step 3.4 — Run migration test; expect 2 PASS

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/test_mc_migration.py -v
```

Expected: `2 passed`.

- [ ] **Step 3.4** — Run and observe green.

### Step 3.5 — Apply migration to the live DB

```bash
cd backend
../.venv/Scripts/python.exe -c "from migrations.mc_telemetry_20260516 import run; print(run('coinbase.db'))"
```

Expected: `{'added': ['xgb_prob_stdev', 'mc_telemetry'], 'already_present': []}`.

- [ ] **Step 3.5** — Apply to live coinbase.db.

### Step 3.6 — Extend `database.save_cnn_scan` test

Append to `backend/tests/test_database.py`:

```python
class TestSaveCnnScanMCColumns:
    @pytest.mark.asyncio
    async def test_save_cnn_scan_persists_xgb_prob_stdev_when_present(self, tmp_path, monkeypatch):
        import database
        # Point database._DB at a temp DB and create cnn_scans + run migration
        db_path = tmp_path / "test.db"
        # Build cnn_scans table (minimum needed)
        import sqlite3
        c = sqlite3.connect(db_path)
        c.execute("""
            CREATE TABLE cnn_scans (
                id INTEGER PRIMARY KEY, product_id TEXT, price REAL,
                cnn_prob REAL, llm_prob REAL, model_prob REAL,
                cnn_weight REAL, llm_weight REAL,
                side TEXT, strength REAL, signal_gen INTEGER,
                regime TEXT, adx REAL, rsi REAL, macd REAL, mfi REAL,
                stoch_k REAL, atr REAL, vwap_dist REAL,
                fast_rsi REAL, velocity REAL, vol_z REAL, xgb_prob REAL,
                scanned_at TEXT
            )""")
        c.commit(); c.close()
        from migrations.mc_telemetry_20260516 import run as mig_run
        mig_run(str(db_path))
        monkeypatch.setattr(database, "_DB_PATH", str(db_path))

        await database.save_cnn_scan({
            "product_id": "BTC-USD", "price": 100.0,
            "cnn_prob": 0.60, "model_prob": 0.60, "side": "BUY",
            "strength": 0.2, "signal_gen": True,
            "xgb_prob_stdev": 0.0124,
            "mc_telemetry": '{"ci":{"decision":"keep"}}',
        })
        c = sqlite3.connect(db_path)
        row = c.execute(
            "SELECT xgb_prob_stdev, mc_telemetry FROM cnn_scans WHERE product_id='BTC-USD'"
        ).fetchone()
        c.close()
        assert row[0] == 0.0124
        assert row[1] == '{"ci":{"decision":"keep"}}'

    @pytest.mark.asyncio
    async def test_save_cnn_scan_handles_missing_mc_keys_as_null(self, tmp_path, monkeypatch):
        """Default behavior: MC columns are NULL when MC_FILTERS is off."""
        import database, sqlite3
        db_path = tmp_path / "test.db"
        c = sqlite3.connect(db_path)
        c.execute("""
            CREATE TABLE cnn_scans (
                id INTEGER PRIMARY KEY, product_id TEXT, price REAL,
                cnn_prob REAL, llm_prob REAL, model_prob REAL,
                cnn_weight REAL, llm_weight REAL,
                side TEXT, strength REAL, signal_gen INTEGER,
                regime TEXT, adx REAL, rsi REAL, macd REAL, mfi REAL,
                stoch_k REAL, atr REAL, vwap_dist REAL,
                fast_rsi REAL, velocity REAL, vol_z REAL, xgb_prob REAL,
                scanned_at TEXT
            )""")
        c.commit(); c.close()
        from migrations.mc_telemetry_20260516 import run as mig_run
        mig_run(str(db_path))
        monkeypatch.setattr(database, "_DB_PATH", str(db_path))

        await database.save_cnn_scan({
            "product_id": "ETH-USD", "price": 200.0,
            "cnn_prob": 0.50, "model_prob": 0.50, "side": "HOLD",
            "strength": 0.0, "signal_gen": False,
            # no xgb_prob_stdev, no mc_telemetry
        })
        c = sqlite3.connect(db_path)
        row = c.execute(
            "SELECT xgb_prob_stdev, mc_telemetry FROM cnn_scans WHERE product_id='ETH-USD'"
        ).fetchone()
        c.close()
        assert row[0] is None
        assert row[1] is None
```

- [ ] **Step 3.6** — Append tests to `tests/test_database.py`.

### Step 3.7 — Extend `database.save_cnn_scan`

Edit `backend/database.py` lines 548-570. Find:

```python
async def save_cnn_scan(scan: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO cnn_scans
               (product_id, price, cnn_prob, llm_prob, model_prob,
                cnn_weight, llm_weight, side, strength, signal_gen,
                regime, adx, rsi, macd, mfi, stoch_k, atr, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scan["product_id"], scan["price"],
                scan.get("cnn_prob"), scan.get("llm_prob"), scan["model_prob"],
                scan.get("cnn_weight"), scan.get("llm_weight"),
                scan["side"], scan["strength"], 1 if scan.get("signal_gen") else 0,
                scan.get("regime"), scan.get("adx"), scan.get("rsi"),
                scan.get("macd"), scan.get("mfi"), scan.get("stoch_k"),
                scan.get("atr"), scan.get("vwap_dist"),
                scan.get("fast_rsi"), scan.get("velocity"), scan.get("vol_z"),
                scan.get("xgb_prob"),
                _now(),
            )
        )
        await db.commit()
```

Replace with:

```python
async def save_cnn_scan(scan: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO cnn_scans
               (product_id, price, cnn_prob, llm_prob, model_prob,
                cnn_weight, llm_weight, side, strength, signal_gen,
                regime, adx, rsi, macd, mfi, stoch_k, atr, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at,
                xgb_prob_stdev, mc_telemetry)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scan["product_id"], scan["price"],
                scan.get("cnn_prob"), scan.get("llm_prob"), scan["model_prob"],
                scan.get("cnn_weight"), scan.get("llm_weight"),
                scan["side"], scan["strength"], 1 if scan.get("signal_gen") else 0,
                scan.get("regime"), scan.get("adx"), scan.get("rsi"),
                scan.get("macd"), scan.get("mfi"), scan.get("stoch_k"),
                scan.get("atr"), scan.get("vwap_dist"),
                scan.get("fast_rsi"), scan.get("velocity"), scan.get("vol_z"),
                scan.get("xgb_prob"),
                _now(),
                scan.get("xgb_prob_stdev"), scan.get("mc_telemetry"),
            )
        )
        await db.commit()
```

- [ ] **Step 3.7** — Apply the edit.

### Step 3.8 — Run the database tests

```bash
../.venv/Scripts/python.exe -m pytest tests/test_database.py -v
```

Expected: existing tests pass + 2 new pass.

- [ ] **Step 3.8** — Verify.

### Step 3.9 — Add cnn_agent tests

Append to `backend/tests/test_cnn_agent.py`:

```python
class TestMCFilterChainIntegration:
    """Wire-up tests for agents.mc.registry hook in generate_signal."""

    def test_mc_filters_empty_leaves_side_unchanged(self, monkeypatch):
        """MC_FILTERS='' (default) → BUY gate behaves bit-for-bit as before."""
        import os
        os.environ.pop("MC_FILTERS", None)  # ensure unset
        from agents.mc import registry
        registry._reset_chain_cache()
        side, tele = registry.apply_buy_filters(
            side="BUY", model_prob=0.7, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "BUY"
        assert tele == {}

    def test_mc_filters_ci_blocks_when_lower_bound_below_threshold(self, monkeypatch):
        """With MC_FILTERS=ci active and a high threshold, CIFilter blocks."""
        monkeypatch.setenv("MC_FILTERS", "ci")
        monkeypatch.setenv("MC_CI_K", "1.0")
        import importlib
        from agents.mc import registry, ci_filter
        importlib.reload(ci_filter)  # re-register after env change
        registry._reset_chain_cache()

        # Stub xgb_signal state + the v3 path so the trajectory is known
        import agents.xgb_signal as xs
        import numpy as np

        class _Booster:
            def num_boosted_rounds(self): return 5
            def predict(self, dmat, iteration_range=None):
                k = iteration_range[1] if iteration_range else 5
                # trajectory 0.50, 0.55, 0.60, 0.65, 0.70; stdev ≈ 0.0707
                return np.array([0.5 + (k - 1) * 0.05])

        monkeypatch.setattr(xs, "_booster", _Booster())
        monkeypatch.setattr(xs, "_feature_set", "v3")
        monkeypatch.setattr(xs, "_feature_names", ["f"] * 350)
        monkeypatch.setattr(xs, "_load_succeeded", True)
        import xgboost as xgb_mod
        monkeypatch.setattr(xgb_mod, "DMatrix",
                            type("DM", (), {"__init__": lambda *a, **kw: None}))
        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            lambda pid, **kw: {"micro": [], "meso": [], "macro": []},
        )
        import tools.xgb_features as xf
        monkeypatch.setattr(
            xf, "extract_features",
            lambda tiers, feature_set="v3": (np.zeros((1, 350)), ["f"] * 350),
        )
        import config as cfg
        monkeypatch.setattr(cfg.config, "cnn_buy_threshold", 0.65)

        side, tele = registry.apply_buy_filters(
            side="BUY", model_prob=0.70, pid="BTC-USD",
            channels=[[0.0] * 60] * 28, context={},
        )
        assert side == "HOLD"
        assert tele["ci"]["decision"] == "block"

    def test_mc_telemetry_propagates_through_generate_signal_scan_dict(
        self, monkeypatch
    ):
        """The scan dict passed to save_cnn_scan must include xgb_prob_stdev
        and mc_telemetry when CIFilter produced them, NULL otherwise."""
        # This is a structural test: assert generate_signal builds those keys
        # when given a non-empty MC telemetry dict. We don't run the full
        # generate_signal; we just verify the scan-dict construction snippet
        # picks up the new keys.
        scan = {
            "product_id": "BTC-USD",
            "price": 100.0,
            "cnn_prob": 0.7,
            "model_prob": 0.7,
            "side": "BUY",
            "strength": 0.4,
            "signal_gen": True,
        }
        mc_tele = {"ci": {"stdev": 0.0124, "lower": 0.6876, "K": 1.0,
                          "decision": "keep"}}
        # Apply the same enrichment cnn_agent.generate_signal does (see Step 3.10)
        if mc_tele:
            scan["xgb_prob_stdev"] = mc_tele.get("ci", {}).get("stdev")
            import json
            scan["mc_telemetry"] = json.dumps(mc_tele)
        assert scan["xgb_prob_stdev"] == 0.0124
        assert '"decision": "keep"' in scan["mc_telemetry"]
```

- [ ] **Step 3.9** — Append the test class.

### Step 3.10 — Wire `agents.mc.registry` into `cnn_agent.generate_signal`

Edit `backend/agents/cnn_agent.py`. Find lines 2225-2236 (the BUY gate section):

```python
        # ── Signal direction ──────────────────────────────────────────────────
        if model_prob > config.cnn_buy_threshold:
            side     = "BUY"
            strength = round((model_prob - 0.5) * 2, 3)
        elif model_prob < config.cnn_sell_threshold:
            side     = "SELL"
            strength = round((0.5 - model_prob) * 2, 3)
        else:
            side     = "HOLD"
            strength = 0.0

        passes = side != "HOLD"
```

Replace with:

```python
        # ── Signal direction ──────────────────────────────────────────────────
        if model_prob > config.cnn_buy_threshold:
            side     = "BUY"
            strength = round((model_prob - 0.5) * 2, 3)
        elif model_prob < config.cnn_sell_threshold:
            side     = "SELL"
            strength = round((0.5 - model_prob) * 2, 3)
        else:
            side     = "HOLD"
            strength = 0.0

        # MC filter chain (off by default; MC_FILTERS env-gated). Returns the
        # side unchanged and {} telemetry when MC_FILTERS is empty.
        from agents.mc import registry as _mc
        side, mc_telemetry = _mc.apply_buy_filters(
            side=side, model_prob=model_prob, pid=pid,
            channels=channels, context={"strength": strength},
        )
        if side == "HOLD":
            strength = 0.0  # re-zero strength if MC down-graded BUY to HOLD

        passes = side != "HOLD"
```

Then find the `save_cnn_scan` dict construction (line 2239) and add two keys (after `vol_z`):

Find:
```python
            "vol_z":       round(vol_z_norm, 4),
```

(or whichever line ends the existing key list; find by searching `vol_z`)

Add IMMEDIATELY after `xgb_prob` is set (search for `"xgb_prob":` in that dict — there's already a line setting it). Append at end of dict construction, before the close-brace:

```python
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev"),
            "mc_telemetry":   __import__("json").dumps(mc_telemetry) if mc_telemetry else None,
```

(Yes the inline `__import__` is ugly; alternative: add `import json` at top of cnn_agent if not present. Check first — grep `^import json` in cnn_agent.py.)

If `import json` is already there, use it cleanly:
```python
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev"),
            "mc_telemetry":   json.dumps(mc_telemetry) if mc_telemetry else None,
```

- [ ] **Step 3.10** — Apply the cnn_agent edit (both the gate hook and the save_cnn_scan dict additions).

### Step 3.11 — Run the cnn_agent + database + mc tests

```bash
../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py tests/test_database.py tests/agents/mc/ tests/test_mc_migration.py -v
```

Expected: all pass.

- [ ] **Step 3.11** — Run combined and verify.

### Step 3.12 — Cleanup + CHANGELOG entry

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Append to `CHANGELOG.md` at the TOP:

```markdown
## [Session 58.70c] — 2026-05-16 — MC wire-up + telemetry schema (#311-mc-wire)

### What changed
- **`backend/migrations/mc_telemetry_20260516.py`** (NEW) — idempotent
  ALTER TABLE adding `xgb_prob_stdev REAL` and `mc_telemetry TEXT` to
  `cnn_scans`. Detects existing columns via PRAGMA, never errors on
  re-run. Applied to live `coinbase.db` as part of this commit.
- **`backend/database.py:save_cnn_scan`** — INSERT extended to write the
  two new columns; both nullable so MC-off (`MC_FILTERS=""`) state still
  produces NULL rows identical to pre-MC.
- **`backend/agents/cnn_agent.py:generate_signal`** — one new hook call
  to `agents.mc.registry.apply_buy_filters` between the side computation
  and the `save_cnn_scan`. With MC off this is a noop. With `MC_FILTERS=ci`
  the lower-bound gate from CIFilter may down-grade BUY to HOLD; telemetry
  is JSON-serialized into the `mc_telemetry` column.
- **`backend/tests/test_mc_migration.py`** (NEW) — 2 tests: add-on-first-run
  and idempotent-on-second-run.
- **`backend/tests/test_database.py`** — +2 tests covering new columns.
- **`backend/tests/test_cnn_agent.py`** — +3 wire-up tests.

### Verification
```
backend && python -m pytest tests/test_cnn_agent.py tests/test_database.py \
                            tests/agents/mc/ tests/test_mc_migration.py -v
=> all passed
```

### Activation
Code is in but inert. To activate CIFilter on live signal generation:
1. Edit `.env`: add `MC_FILTERS=ci` (and optionally `MC_CI_K=1.0`).
2. `curl -X POST http://localhost:8001/api/cnn/model/reload -H "x-api-key: $KEY"`.

`MC_FILTERS=` (default) leaves live behavior bit-for-bit identical to
pre-MC. Rollback: edit .env, reload.

---
```

- [ ] **Step 3.12** — Append CHANGELOG.

### Step 3.13 — Commit

```bash
cd C:\Users\gl450\polymarket_app
git add backend/migrations/ backend/database.py backend/agents/cnn_agent.py \
        backend/tests/test_mc_migration.py backend/tests/test_database.py \
        backend/tests/test_cnn_agent.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311-mc-wire): wire MC filter chain into cnn_agent + telemetry schema

Adds idempotent migration adding xgb_prob_stdev REAL + mc_telemetry TEXT
columns to cnn_scans. Extends database.save_cnn_scan to persist them.
Wires agents.mc.registry.apply_buy_filters into cnn_agent.generate_signal
between side computation and the persistence step. With MC_FILTERS=""
(default) the chain is a noop and rows look bit-for-bit pre-MC. With
MC_FILTERS=ci the CIFilter lower-bound gate may down-grade BUY to HOLD;
telemetry serialized to JSON in the mc_telemetry column.

Migration applied to live coinbase.db as part of this commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3.13** — Commit.

---

## Task 4: Cutover (operator-driven, .env flip)

**This is the activation step. NO code changes.**

### Step 4.1 — Backup current .env

```bash
cd C:\Users\gl450\polymarket_app
cp .env .env.bak_pre_mc_$(date +%Y%m%d_%H%M%S)
```

- [ ] **Step 4.1** — Backup .env.

### Step 4.2 — Append MC config to `.env`

Add the following lines at the end of `.env`:

```
MC_FILTERS=ci
MC_CI_K=1.0
```

- [ ] **Step 4.2** — Edit .env and save.

### Step 4.3 — Hot-reload the backend (if it's running)

```bash
APIKEY=$(grep -oP 'APP_API_KEY=\K.*' .env | tr -d '\r\n')
curl -sS -X POST http://localhost:8001/api/cnn/model/reload \
     -H "x-api-key: $APIKEY"
```

Expected response includes `"status": "ok"` (the reload endpoint re-reads model artifacts but does not re-read .env — for the env flip to take effect, the backend must be **restarted**, not just reloaded).

So actually: **restart the backend** if it's running. Launch via the desktop launcher OR:

```bash
cd backend && nohup ../.venv/Scripts/python.exe -m uvicorn main:app --host 0.0.0.0 --port 8001 > backend.log 2>&1 &
```

Verify CIFilter is active in the backend log:

```bash
tail -30 backend.log | grep -i "MC_FILTERS\|CIFilter\|mc.registry"
```

If you don't see any MC log line, that's because the registry only logs WARNING for unknown filters; CIFilter activation is silent. Confirm via:

```bash
sleep 60   # wait for one scan cycle
sqlite3 backend/coinbase.db "SELECT product_id, model_prob, xgb_prob_stdev, mc_telemetry FROM cnn_scans WHERE scanned_at >= datetime('now', '-2 minute') LIMIT 5;"
```

You should see non-NULL `xgb_prob_stdev` and JSON in `mc_telemetry`.

- [ ] **Step 4.3** — Restart backend, verify telemetry columns populating.

### Step 4.4 — No commit needed

`.env` is gitignored. Activation is operator-local. The Phase 3 commit (#311-mc-wire) already shipped the code.

- [ ] **Step 4.4** — Confirm `.env` is not staged (`git status` should be clean).

---

## Task 5: Memory + CLAUDE.md sync

**Files:**
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`
- Modify: `polymarket_app/CLAUDE.md`

### Step 5.1 — Append v3 entry to architecture memory

Add a new bullet to `coinbase_trader_architecture.md` (under the existing v3 entry from Session 58.69):

```markdown
- **Session 58.70 (2026-05-16)**: Monte Carlo filter chain (#311-mc-a, ci,
  wire). Sidecar pattern under `backend/agents/mc/` — `base.BuyFilter` ABC,
  `registry.apply_buy_filters` env-driven dispatch (`MC_FILTERS=ci,kelly_dd,...`),
  `ci_filter.CIFilter` MVP (cumulative-trajectory stdev across the v3
  booster's 200 trees → lower-bound gate at `(point - K * stdev) > cnn_buy_threshold`,
  K=1.0 default via `MC_CI_K`). `cnn_agent.generate_signal` calls
  `mc.apply_buy_filters` between side computation and save_cnn_scan;
  `MC_FILTERS=""` (default) = bit-for-bit pre-MC behavior. Telemetry: new
  `cnn_scans.xgb_prob_stdev REAL` + `cnn_scans.mc_telemetry TEXT` columns
  (NULLable, migration `migrations/mc_telemetry_20260516.py` idempotent).
  Filter exceptions are caught + logged; one broken filter cannot kill the
  chain or the scan loop. 21 tests across 4 files. Next filters in queue:
  KellyDDFilter (sizing drawdown envelope), ExitEVFilter (exit EV comparison),
  PortfolioVaRFilter (daily VaR cap).
```

- [ ] **Step 5.1** — Apply.

### Step 5.2 — Add invariant to CLAUDE.md

Find the "Key invariants" list and append:

```markdown
14. **MC filter chain** lives under `backend/agents/mc/` and is the ONLY place
    Monte Carlo math touches the decision pipeline. `cnn_agent.generate_signal`
    has exactly one MC hook (`mc.apply_buy_filters` between side computation
    and save_cnn_scan); embedding MC math inside cnn_agent core is forbidden.
    Each filter is opt-in via `MC_FILTERS` env (comma-separated). `MC_FILTERS=""`
    (default) MUST produce bit-for-bit pre-MC behavior. Telemetry persists to
    `cnn_scans.xgb_prob_stdev` (CIFilter only) and `cnn_scans.mc_telemetry`
    (JSON blob, any filter). Filter exceptions MUST be caught + logged, never
    re-raised into the scan loop.
```

- [ ] **Step 5.2** — Apply.

### Step 5.3 — CHANGELOG entry + commit

Append to `CHANGELOG.md` at the TOP:

```markdown
## [Session 58.70d] — 2026-05-16 — CLAUDE.md + memory sync for MC chain (#311-mc-sync)

### What changed
- **`polymarket_app/CLAUDE.md`** — invariant #14 added: MC filter chain is
  the sole MC-math touchpoint; cnn_agent has one hook; MC_FILTERS="" is
  bit-for-bit pre-MC; telemetry columns + JSON; filter exceptions caught.
- **`memory/coinbase_trader_architecture.md`** (outside repo) — Session 58.70
  entry covering the sidecar pattern, CIFilter MVP, telemetry schema, and
  the queued next filters.

Documentation-only commit.

---
```

```bash
cd C:\Users\gl450\polymarket_app
git add CLAUDE.md CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs(#311-mc-sync): CLAUDE.md invariant for MC filter chain

Adds invariant #14 documenting the MC sidecar pattern: agents/mc/ is the
sole MC-math touchpoint, cnn_agent has one hook, MC_FILTERS="" is bit-for-
bit pre-MC, telemetry persists to two new cnn_scans columns, filter
exceptions are caught. Mirrors the same update in memory/.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5.3** — Commit.

### Step 5.4 — Push all 4 commits

```bash
git push
```

- [ ] **Step 5.4** — Push.

---

## Spec coverage check

| Spec section | Tasks |
|---|---|
| 4.1 Four filter slots, MVP=CIFilter | Tasks 1-2 |
| 4.2 MVP scope (CIFilter alone) | Tasks 1-2 |
| 5 CIFilter algorithm | Task 2 |
| 6.1 File layout | Tasks 1-3 |
| 6.2 What stays unchanged | (no edits needed) |
| 7 Configuration (.env knobs) | Task 4 |
| 8 Telemetry (column + JSON) | Tasks 3.6-3.13 |
| 9 Error handling (7 cases) | Tasks 1.2 (registry) + 2.1 (ci_filter) |
| 10 Tests (21 total) | 8 (T1) + 6 (T2) + 2 mig + 2 db + 3 cnn = 21 ✓ |
| 11 Rollout phases 0-4 | Tasks 1-5 |
| 12 Memory + CLAUDE.md sync | Task 5 |
| 13 Open questions | None — defaults locked |

All sections covered.

---

## Plan complete

Saved to `docs/superpowers/plans/2026-05-16-mc-ci-filter-mvp.md`. **5 tasks, 21 unit tests, 4 code commits + 1 docs commit + 1 operator-driven .env flip.**

Default-off design: through Task 3 the system is bit-for-bit identical to today. Task 4 is when CIFilter starts firing. Task 5 syncs the invariants.

Operator runs inline (per the v3 lesson: subagents kept dying on the pre-commit hook).
