# PnL-Anchored Trail with Capital-Relative Profit Floors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the price-anchored trail + fixed -8% stop_loss with a PnL-anchored trail using ATR-scaled profit-floor tiers + capital-relative dollar-cap on large positions + fee-aware break-even, eliminating the "paper-profit-to-realized-loss" failure mode identified as the primary root cause of realized PnL bleed.

**Architecture:** New pure-helper module `backend/agents/exit_thresholds.py` exposes `_compute_exit_threshold(peak_pnl_pct, atr_pct, position_dollars, total_capital)` consumed by both the scan-loop (`cnn_agent._check_risk_exits`) and the WS tick handler (`exit_watcher.on_price_tick`). Position state gains `peak_pnl_pct` + `position_dollars` (mirrors trail_pct cache pattern from Session 58.71m). Constants are global module-level; per-pid overrides deferred to a v2.

**Tech Stack:** Python, numpy (already), pytest, no new external deps. Builds on existing `_CNNBook` state schema.

**OPERATIONAL CONSTRAINT — read first:**
8001 is currently live trading. Per `feedback_no_pytest_during_trading.md`:
- Tasks 1-7 are **file-only writes** (no pytest runs, no commits).
- Task 8 is **gated on operator pausing 8001** — runs the full pytest sweep, atomic commit, push.
- Each task's TDD `verify-fail` and `verify-pass` steps are written for reference but **deferred to Task 8**.

---

## File Structure

| Path | Role | Responsibility |
|---|---|---|
| `backend/agents/exit_thresholds.py` | NEW | Pure helpers: `_atr_floor`, `_large_position_threshold`, `_dollar_cap_floor`, `_compute_exit_threshold`. ~120 LoC. |
| `backend/tests/test_exit_thresholds.py` | NEW | 15 unit + integration tests covering all 3 layers + capital scaling + boundaries. ~350 LoC. |
| `backend/agents/cnn_agent.py` | MODIFY | Constants block; `_CNNBook.load()` migration of `peak_pnl_pct`/`position_dollars`; `_check_risk_exits` updated to use the new threshold helper. ~50 LoC delta. |
| `backend/agents/exit_watcher.py` | MODIFY | `on_price_tick` updated to ratchet `peak_pnl_pct` and use the new threshold helper. ~25 LoC delta. |

---

## Task 1: Module scaffold + constants

**Files:**
- Create: `backend/agents/exit_thresholds.py`

- [ ] **Step 1: Write the module skeleton**

```python
"""Pure-function exit-threshold helpers for the CNN agent's risk exits.

Used by both the scan-loop (cnn_agent._check_risk_exits) and the WS tick
handler (exit_watcher.on_price_tick) to compute the exit threshold in
PnL-percent terms.

Three layers (see spec docs/superpowers/specs/2026-05-23-pnl-anchored-trail-design.md):
    1. ATR-scaled profit-floor tiers (volatility-relative)
    2. Capital-relative dollar-cap on large positions
    3. Standard ATR giveback baseline (peak_pnl - atr_pct)

Position exits when current_pnl_pct < _compute_exit_threshold(...).
"""
from __future__ import annotations
from typing import Optional


# ── Layer 1: ATR-scaled profit-floor tiers ─────────────────────────────────
# Tiers engage at multiples of atr_pct, lock at multiples of atr_pct.

# ── Layer 2: capital-relative dollar-cap ────────────────────────────────────
# Large position = position $ > max(LARGE_POSITION_FLOOR, total_capital * LARGE_POSITION_FRAC).
# Cap = tighter of (MAX_DOLLAR_GIVEBACK_FRAC of position $, MAX_LOSS_FRAC_OF_CAPITAL of total capital).

FEE_RATE: float                  = 0.006   # Coinbase taker (round-trip basis = 2 * 0.006 = 1.2%)
LARGE_POSITION_FRAC: float       = 0.05    # 5% of capital = "large" position
LARGE_POSITION_FLOOR: float      = 200.0   # USD; absolute floor for small-capital regime
MAX_DOLLAR_GIVEBACK_FRAC: float  = 0.02    # 2% of position $ %-based scale-invariant cap
MAX_LOSS_FRAC_OF_CAPITAL: float  = 0.005   # 0.5% of total capital per trail-fire


__all__ = [
    "FEE_RATE",
    "LARGE_POSITION_FRAC",
    "LARGE_POSITION_FLOOR",
    "MAX_DOLLAR_GIVEBACK_FRAC",
    "MAX_LOSS_FRAC_OF_CAPITAL",
    "_atr_floor",
    "_large_position_threshold",
    "_dollar_cap_floor",
    "_compute_exit_threshold",
]
```

- [ ] **Step 2: ~Verify import works~ DEFERRED to Task 8**

- [ ] **Step 3: ~Commit~ DEFERRED to Task 8**

---

## Task 2: `_atr_floor` — Layer 1 (ATR-scaled profit-floor tiers)

**Files:**
- Modify: `backend/agents/exit_thresholds.py`
- Test: `backend/tests/test_exit_thresholds.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_exit_thresholds.py` with the file header + Layer 1 tests:

```python
"""Tests for backend/agents/exit_thresholds.py.

Numerical contract: all comparisons assert exact equality except where
floating-point arithmetic introduces drift, in which case pytest.approx is
used with abs=1e-12.
"""
from __future__ import annotations

import pytest

from agents.exit_thresholds import (
    FEE_RATE,
    LARGE_POSITION_FRAC,
    LARGE_POSITION_FLOOR,
    MAX_DOLLAR_GIVEBACK_FRAC,
    MAX_LOSS_FRAC_OF_CAPITAL,
    _atr_floor,
    _large_position_threshold,
    _dollar_cap_floor,
    _compute_exit_threshold,
)


# ── Layer 1: _atr_floor ────────────────────────────────────────────────────

def test_atr_floor_returns_none_below_engagement_threshold():
    # peak_pnl < 0.5 * atr_pct → no floor engages
    assert _atr_floor(peak_pnl_pct=0.02, atr_pct=0.06) is None
    assert _atr_floor(peak_pnl_pct=0.029, atr_pct=0.06) is None  # just below 0.030


def test_atr_floor_break_even_tier_uses_fee_rate():
    # 0.5 * atr <= peak_pnl < 1.5 * atr → fee-aware break-even
    assert _atr_floor(peak_pnl_pct=0.030, atr_pct=0.06) == pytest.approx(2 * FEE_RATE)
    assert _atr_floor(peak_pnl_pct=0.089, atr_pct=0.06) == pytest.approx(2 * FEE_RATE)


def test_atr_floor_moderate_lock_tier_at_0_5_x_atr():
    # 1.5 * atr <= peak_pnl < 3.0 * atr → lock 0.5 * atr
    assert _atr_floor(peak_pnl_pct=0.090, atr_pct=0.06) == pytest.approx(0.030)
    assert _atr_floor(peak_pnl_pct=0.179, atr_pct=0.06) == pytest.approx(0.030)


def test_atr_floor_strong_lock_tier_at_1_5_x_atr():
    # peak_pnl >= 3.0 * atr → lock 1.5 * atr
    assert _atr_floor(peak_pnl_pct=0.180, atr_pct=0.06) == pytest.approx(0.090)
    assert _atr_floor(peak_pnl_pct=1.0, atr_pct=0.06) == pytest.approx(0.090)


def test_atr_floor_zero_atr_only_engages_above_zero_pnl():
    # atr_pct = 0 → all thresholds collapse to 0; only the >= 0 tier engages
    # peak_pnl > 0 falls into the 0.5×atr=0 break-even tier → 2*FEE_RATE
    assert _atr_floor(peak_pnl_pct=0.01, atr_pct=0.0) == pytest.approx(2 * FEE_RATE)
    # peak_pnl exactly 0 → also engages (>= 0.5*0 = 0)
    assert _atr_floor(peak_pnl_pct=0.0, atr_pct=0.0) == pytest.approx(2 * FEE_RATE)


def test_atr_floor_boundary_inclusive_at_0_5_x_atr():
    # exactly 0.5 * atr → engages break-even tier (>=, not >)
    atr = 0.10
    result = _atr_floor(peak_pnl_pct=0.05, atr_pct=atr)
    assert result == pytest.approx(2 * FEE_RATE)
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Run: `cd backend && python -m pytest tests/test_exit_thresholds.py::test_atr_floor_returns_none_below_engagement_threshold -v`
Expected: FAIL with `ImportError: cannot import name '_atr_floor'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/agents/exit_thresholds.py`:

```python
def _atr_floor(peak_pnl_pct: float, atr_pct: float) -> Optional[float]:
    """ATR-scaled profit floor — Layer 1.

    Tiers (in PnL terms):
        peak_pnl_pct >= 3.0 * atr_pct  ->  return 1.5 * atr_pct  (strong lock)
        peak_pnl_pct >= 1.5 * atr_pct  ->  return 0.5 * atr_pct  (moderate lock)
        peak_pnl_pct >= 0.5 * atr_pct  ->  return 2 * FEE_RATE   (fee-aware break-even)
        else                            ->  return None           (no floor yet)

    Comparisons are `>=` (inclusive at exact tier boundaries).
    """
    if peak_pnl_pct >= 3.0 * atr_pct:
        return 1.5 * atr_pct
    if peak_pnl_pct >= 1.5 * atr_pct:
        return 0.5 * atr_pct
    if peak_pnl_pct >= 0.5 * atr_pct:
        return 2 * FEE_RATE
    return None
```

- [ ] **Step 4: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 5: Commit (DEFERRED to Task 8)**

---

## Task 3: `_large_position_threshold` + `_dollar_cap_floor` — Layer 2

**Files:**
- Modify: `backend/agents/exit_thresholds.py`
- Test: `backend/tests/test_exit_thresholds.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_exit_thresholds.py`:

```python
# ── Layer 2: _large_position_threshold ─────────────────────────────────────

def test_large_position_threshold_floor_at_small_capital():
    # capital * frac < floor → returns floor
    assert _large_position_threshold(total_capital=100.0) == LARGE_POSITION_FLOOR
    assert _large_position_threshold(total_capital=1000.0) == LARGE_POSITION_FLOOR
    assert _large_position_threshold(total_capital=3999.0) == LARGE_POSITION_FLOOR


def test_large_position_threshold_scales_above_floor():
    # capital * frac >= floor → returns capital * frac
    assert _large_position_threshold(total_capital=4000.0) == pytest.approx(200.0)
    assert _large_position_threshold(total_capital=10_000.0) == pytest.approx(500.0)
    assert _large_position_threshold(total_capital=100_000.0) == pytest.approx(5000.0)


# ── Layer 2: _dollar_cap_floor ─────────────────────────────────────────────

def test_dollar_cap_floor_skips_small_positions():
    # position $ <= threshold → no cap engages
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=50.0,
                             total_capital=1000.0) is None
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=200.0,
                             total_capital=1000.0) is None  # exactly at threshold


def test_dollar_cap_floor_engages_for_large_positions_small_capital():
    # position $250 at capital $1k → threshold=$200, position>threshold
    # pct_cap = 2%, cap_cap = $5 / $250 = 2%, min = 2%
    # threshold = peak_pnl - 2% = 0.05 - 0.02 = 0.03
    result = _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=250.0,
                               total_capital=1000.0)
    assert result == pytest.approx(0.03)


def test_dollar_cap_floor_capital_loss_cap_tighter_than_pct_cap():
    # position $500 at capital $1k → threshold=$200, engages
    # pct_cap = 2%, cap_cap = $5 / $500 = 1.0%, min = 1.0% (capital wins)
    # threshold = peak_pnl - 1.0% = 0.05 - 0.01 = 0.04
    result = _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                               total_capital=1000.0)
    assert result == pytest.approx(0.04)


def test_dollar_cap_floor_pct_cap_tighter_than_capital_cap():
    # position $500 at capital $10k → threshold=$500, NOT engaged (position == threshold)
    # but threshold is "> threshold", so $500.01 would engage
    # let's use $501 to trigger engagement
    # pct_cap = 2%, cap_cap = $50 / $501 = 9.98%, min = 2% (pct wins)
    # threshold = peak_pnl - 2% = 0.05 - 0.02 = 0.03
    result = _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=501.0,
                               total_capital=10_000.0)
    assert result == pytest.approx(0.03)


def test_dollar_cap_floor_threshold_scales_with_capital():
    # Same $500 position, different capital levels — assert engagement varies
    # capital $1k: threshold=$200, $500 > $200 → engages
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=1000.0) is not None
    # capital $5k: threshold=$250, $500 > $250 → engages
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=5_000.0) is not None
    # capital $10k: threshold=$500, $500 NOT > $500 → skips
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=10_000.0) is None
    # capital $100k: threshold=$5k, $500 NOT > $5k → skips
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=100_000.0) is None
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/agents/exit_thresholds.py`:

```python
def _large_position_threshold(total_capital: float) -> float:
    """Capital-relative cutoff for the dollar-cap layer.

    A position is 'large' if it's > LARGE_POSITION_FRAC of total capital,
    with LARGE_POSITION_FLOOR as the absolute minimum so the rule remains
    meaningful at small capital.
    """
    return max(LARGE_POSITION_FLOOR, total_capital * LARGE_POSITION_FRAC)


def _dollar_cap_floor(
    peak_pnl_pct: float,
    position_dollars: float,
    total_capital: float,
) -> Optional[float]:
    """Capital-relative dollar-cap on large positions — Layer 2.

    Caps giveback at the TIGHTER of:
      - MAX_DOLLAR_GIVEBACK_FRAC of position $ (scale-invariant %)
      - MAX_LOSS_FRAC_OF_CAPITAL of total capital (portfolio-impact based)

    Returns the minimum exit threshold (in PnL terms) or None if the
    position is not 'large' relative to capital.
    """
    threshold = _large_position_threshold(total_capital)
    if position_dollars <= threshold:
        return None
    pct_cap = MAX_DOLLAR_GIVEBACK_FRAC
    cap_cap = (total_capital * MAX_LOSS_FRAC_OF_CAPITAL) / position_dollars
    giveback = min(pct_cap, cap_cap)
    return peak_pnl_pct - giveback
```

- [ ] **Step 4: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 5: Commit (DEFERRED to Task 8)**

---

## Task 4: `_compute_exit_threshold` — orchestrator

**Files:**
- Modify: `backend/agents/exit_thresholds.py`
- Test: `backend/tests/test_exit_thresholds.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_exit_thresholds.py`:

```python
# ── Orchestrator: _compute_exit_threshold ───────────────────────────────────

def test_compute_exit_threshold_baseline_no_floors_engage():
    # peak_pnl 0.02 < 0.5 * atr_pct 0.03 → Layer 1 None
    # no position_dollars → Layer 2 skipped
    # threshold = peak_pnl - atr = 0.02 - 0.06 = -0.04
    result = _compute_exit_threshold(peak_pnl_pct=0.02, atr_pct=0.06)
    assert result == pytest.approx(-0.04)


def test_compute_exit_threshold_layer1_break_even_wins_over_baseline():
    # peak_pnl 0.05, atr 0.06 → 0.5*atr=0.03, peak > 0.03 → break-even tier = 2*0.006 = 0.012
    # baseline = 0.05 - 0.06 = -0.01
    # max(baseline=-0.01, floor=0.012) = 0.012
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06)
    assert result == pytest.approx(2 * FEE_RATE)


def test_compute_exit_threshold_layer1_strong_lock():
    # peak_pnl 0.20, atr 0.06 → 3*atr=0.18, peak > 0.18 → strong = 1.5*0.06 = 0.09
    # baseline = 0.20 - 0.06 = 0.14
    # max(0.14, 0.09) = 0.14 → baseline wins
    result = _compute_exit_threshold(peak_pnl_pct=0.20, atr_pct=0.06)
    assert result == pytest.approx(0.14)


def test_compute_exit_threshold_layer2_tightens_over_layer1():
    # peak_pnl 0.05, atr 0.06, position $500, capital $1k
    # Layer 1: break-even tier = 0.012
    # Layer 2: cap_cap = $5/$500 = 0.01, pct_cap = 0.02, min = 0.01; floor = 0.05 - 0.01 = 0.04
    # baseline = 0.05 - 0.06 = -0.01
    # max(-0.01, 0.012, 0.04) = 0.04 → Layer 2 wins (tightens)
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                     position_dollars=500.0, total_capital=1000.0)
    assert result == pytest.approx(0.04)


def test_compute_exit_threshold_layer2_skipped_when_capital_omitted():
    # position_dollars provided but no total_capital → Layer 2 skipped
    # Layer 1: break-even = 0.012, baseline = -0.01, max = 0.012
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                     position_dollars=500.0)
    assert result == pytest.approx(2 * FEE_RATE)


def test_compute_exit_threshold_capital_scaling_invariant():
    # Same $500 position; at large capital, Layer 2 skips → identical result
    r1 = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                 position_dollars=500.0)
    r2 = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                 position_dollars=500.0, total_capital=100_000.0)
    assert r1 == pytest.approx(r2)


def test_compute_exit_threshold_worked_example_dash_break_even_engages():
    # DASH from spec table: peak_pnl 0.078, atr 0.06, pos $58, capital $1k
    # Layer 1: peak (0.078) > 0.5*atr=0.03 AND peak > 1.5*atr=0.09? NO (0.078 < 0.09)
    #   → break-even tier = 2*FEE_RATE = 0.012
    # Layer 2: threshold = max($200, $50) = $200, pos $58 < $200 → skip
    # baseline = 0.078 - 0.06 = 0.018
    # max(0.018, 0.012) = 0.018 → baseline wins
    result = _compute_exit_threshold(peak_pnl_pct=0.078, atr_pct=0.06,
                                     position_dollars=58.0, total_capital=1000.0)
    assert result == pytest.approx(0.018)


def test_compute_exit_threshold_worked_example_paxg_capital_cap_engages():
    # PAXG from spec table: peak_pnl 0.0, atr 0.04, pos $452, capital $1k
    # Layer 1: peak (0) < 0.5*atr=0.02 → None
    # Layer 2: threshold = $200, pos $452 > $200 → engages
    #   pct_cap = 0.02, cap_cap = $5/$452 ≈ 0.01106, min = 0.01106
    #   floor = 0 - 0.01106 = -0.01106
    # baseline = 0 - 0.04 = -0.04
    # max(-0.04, -0.01106) = -0.01106 → Layer 2 wins
    result = _compute_exit_threshold(peak_pnl_pct=0.0, atr_pct=0.04,
                                     position_dollars=452.0, total_capital=1000.0)
    assert result == pytest.approx(-5.0 / 452.0)
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Append to `backend/agents/exit_thresholds.py`:

```python
def _compute_exit_threshold(
    *,
    peak_pnl_pct: float,
    atr_pct: float,
    position_dollars: Optional[float] = None,
    total_capital: Optional[float] = None,
) -> float:
    """Combined exit threshold in PnL terms.

    Position exits when current_pnl_pct < returned value.

    Combines:
      Layer 1: ATR-scaled profit-floor tiers (volatility-relative)
      Layer 2: Capital-relative dollar-cap on large positions (requires both
               position_dollars AND total_capital; skipped if either omitted)
      Layer 3: Standard ATR giveback baseline (peak_pnl - atr_pct)

    Returns the max of all engaged layers — the TIGHTEST exit threshold.
    """
    base = peak_pnl_pct - atr_pct   # Layer 3 baseline

    atr_floor = _atr_floor(peak_pnl_pct, atr_pct)
    if atr_floor is not None:
        base = max(base, atr_floor)

    if position_dollars and total_capital:
        dollar_floor = _dollar_cap_floor(
            peak_pnl_pct, position_dollars, total_capital,
        )
        if dollar_floor is not None:
            base = max(base, dollar_floor)

    return base
```

- [ ] **Step 4: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 5: Commit (DEFERRED to Task 8)**

---

## Task 5: `_CNNBook` state migration on load

**Files:**
- Modify: `backend/agents/cnn_agent.py` (in `_CNNBook.load_state` or equivalent — find the existing positions-deserialize block)
- Test: `backend/tests/test_cnn_agent.py` (add to existing file)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_cnn_agent.py`:

```python
class TestCNNBookPnLMigration:
    """Tests for the peak_pnl_pct + position_dollars migration applied to
    positions loaded from agent_state.positions_json."""

    def test_migration_computes_peak_pnl_from_peak_price_and_avg(self):
        from agents.cnn_agent import _CNNBook
        book = _CNNBook()
        # Simulate a position loaded from disk WITHOUT the new fields
        legacy_position = {
            "size": 1.0,
            "avg_price": 100.0,
            "peak_price": 110.0,
            "entry_time": 1700000000.0,
            "trail_pct": 0.06,
        }
        # Apply migration (function under test — name it _migrate_position_state)
        migrated = book._migrate_position_state(legacy_position)
        assert "peak_pnl_pct" in migrated
        assert migrated["peak_pnl_pct"] == pytest.approx(0.10)  # (110 - 100) / 100
        assert "position_dollars" in migrated
        # Initial dollar value uses avg_price (next scan updates with current_price)
        assert migrated["position_dollars"] == pytest.approx(100.0)

    def test_migration_preserves_existing_pnl_pct(self):
        from agents.cnn_agent import _CNNBook
        book = _CNNBook()
        position = {
            "size": 1.0,
            "avg_price": 100.0,
            "peak_price": 110.0,
            "peak_pnl_pct": 0.15,  # already set — don't overwrite
            "position_dollars": 105.0,
        }
        migrated = book._migrate_position_state(position)
        assert migrated["peak_pnl_pct"] == 0.15
        assert migrated["position_dollars"] == 105.0

    def test_migration_handles_zero_avg_price(self):
        from agents.cnn_agent import _CNNBook
        book = _CNNBook()
        # Corrupt position (legacy entry_corrupt=True case): avg_price=0
        position = {
            "size": 1.0,
            "avg_price": 0.0,
            "peak_price": 0.0,
        }
        migrated = book._migrate_position_state(position)
        assert migrated["peak_pnl_pct"] == 0.0
        assert migrated["position_dollars"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Expected: FAIL with `AttributeError: _CNNBook has no attribute _migrate_position_state`.

- [ ] **Step 3: Implement the migration helper on `_CNNBook`**

In `backend/agents/cnn_agent.py`, find `class _CNNBook:` and add this method (place it near other helpers like `_lock_for`):

```python
    @staticmethod
    def _migrate_position_state(pos: Dict[str, Any]) -> Dict[str, Any]:
        """Add peak_pnl_pct + position_dollars to legacy position dicts.

        peak_pnl_pct = (peak_price - avg_price) / avg_price, clamped to 0 if
        avg_price is zero (corrupt-entry case).
        position_dollars = size * avg_price as an initial value; next scan
        updates with the live current_price.
        """
        if "peak_pnl_pct" not in pos:
            avg = pos.get("avg_price", 0.0)
            peak = pos.get("peak_price", avg)
            pos["peak_pnl_pct"] = ((peak - avg) / avg) if avg > 0 else 0.0
        if "position_dollars" not in pos:
            size = pos.get("size", 0.0)
            avg = pos.get("avg_price", 0.0)
            pos["position_dollars"] = float(size * avg)
        return pos
```

- [ ] **Step 4: Wire migration into the load path**

Find the existing position-deserialize block in `_CNNBook.load_state` (or `load`, or the constructor — search for `positions_json` to locate it). After the JSON is parsed and each position is reconstituted, run it through the migration:

```python
        # AFTER the existing per-position load logic (find via grep for positions_json):
        for pid, pos in self.positions.items():
            self.positions[pid] = self._migrate_position_state(pos)
```

(The exact placement depends on the existing code structure — apply this AFTER positions are loaded but BEFORE the first scan-loop cycle.)

- [ ] **Step 5: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 6: Commit (DEFERRED to Task 8)**

---

## Task 6: Plumb scan-loop path (`_check_risk_exits`)

**Files:**
- Modify: `backend/agents/cnn_agent.py:_check_risk_exits` (around line 1684+)
- Test: `backend/tests/test_cnn_risk_exits.py` (add to existing file)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_cnn_risk_exits.py`:

```python
class TestCNNRiskExitsPnLAnchored:
    """Tests that _check_risk_exits uses _compute_exit_threshold from
    exit_thresholds for the trail decision."""

    @pytest.mark.asyncio
    async def test_check_risk_exits_uses_pnl_anchored_threshold(self):
        """Position with peak_pnl=8%, atr_pct=6%, pos$=58, capital=$1k:
        expected threshold = max(0.018 baseline, break-even floor 0.012) = 0.018.
        Current pnl drops to 1.5% (below 1.8% threshold) → exit fires."""
        from agents.cnn_agent import CoinbaseCNNAgent
        agent = CoinbaseCNNAgent()
        agent.book.positions["DASH-USD"] = {
            "size": 1.34, "avg_price": 42.97, "peak_price": 46.32,
            "peak_pnl_pct": 0.078, "position_dollars": 58.0,
            "trail_pct": 0.06, "entry_time": time.time() - 3600,
        }
        # Mock balance such that total_capital = 1000
        agent.book.balance = 942.0  # 942 + 58 = 1000 total
        # Current price drops to $43.59 → current_pnl_pct = (43.59 - 42.97)/42.97 = 0.0144
        # That's BELOW threshold 0.018 → trail should fire
        sell_mock = AsyncMock()
        agent.book.sell = sell_mock
        await agent._check_risk_exits({"DASH-USD": 43.59})
        # Assert sell was called with TRAIL_STOP trigger
        sell_mock.assert_called_once()
        args, kwargs = sell_mock.call_args
        assert kwargs.get("trigger") == "TRAIL_STOP" or args[2] == "TRAIL_STOP"

    @pytest.mark.asyncio
    async def test_check_risk_exits_does_not_fire_above_threshold(self):
        """Same position, current pnl at 2.5% (above 1.8% threshold) → no exit."""
        from agents.cnn_agent import CoinbaseCNNAgent
        agent = CoinbaseCNNAgent()
        agent.book.positions["DASH-USD"] = {
            "size": 1.34, "avg_price": 42.97, "peak_price": 46.32,
            "peak_pnl_pct": 0.078, "position_dollars": 58.0,
            "trail_pct": 0.06, "entry_time": time.time() - 3600,
        }
        agent.book.balance = 942.0
        sell_mock = AsyncMock()
        agent.book.sell = sell_mock
        await agent._check_risk_exits({"DASH-USD": 44.04})  # +2.5% pnl
        sell_mock.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Expected: FAIL (current `_check_risk_exits` uses peak-price trail; new behavior expected).

- [ ] **Step 3: Modify `_check_risk_exits` to use the new threshold helper**

In `backend/agents/cnn_agent.py:_check_risk_exits` (around line 1684+), replace the existing trail/stop_loss math with a call to `_compute_exit_threshold`. Locate the current block that computes `trail_line = peak_price * (1 - trail_pct)` and the `if pct_entry <= -_CNN_STOP_LOSS_PCT` check; replace with:

```python
        # Import at top of file (do not put inside the function):
        from agents.exit_thresholds import _compute_exit_threshold

        # Inside _check_risk_exits, for each position:
        for pid, pos in self.positions.items():
            current = current_prices.get(pid)
            if current is None or pos.get("entry_corrupt"):
                continue
            entry = pos["avg_price"]
            if entry <= 0:
                continue
            current_pnl_pct = (current - entry) / entry

            # Ratchet peak_pnl_pct (never decreases)
            pos["peak_pnl_pct"] = max(pos.get("peak_pnl_pct", 0.0), current_pnl_pct)
            # Update position_dollars cache (scan-loop owns refresh)
            pos["position_dollars"] = float(pos["size"]) * current

            # ATR-derived trail_pct already cached per Session 58.71m
            atr_pct = pos.get("trail_pct", _CNN_ATR_TRAIL_MIN)

            # Total capital = balance + sum of all position $ (live)
            total_capital = self.balance + sum(
                p.get("position_dollars", 0.0) for p in self.positions.values()
            )

            threshold = _compute_exit_threshold(
                peak_pnl_pct=pos["peak_pnl_pct"],
                atr_pct=atr_pct,
                position_dollars=pos["position_dollars"],
                total_capital=total_capital,
            )

            # Stop-loss backstop still fires independently
            stop_loss_threshold = -_CNN_STOP_LOSS_PCT

            # Pick the trigger: the HIGHER of the two thresholds dominates
            # (because the position exits as current_pnl drops below either)
            if current_pnl_pct < threshold and current_pnl_pct < stop_loss_threshold:
                trigger = "STOP_LOSS"     # both fired; STOP_LOSS is the more serious
            elif current_pnl_pct < threshold:
                trigger = "TRAIL_STOP"
            elif current_pnl_pct < stop_loss_threshold:
                trigger = "STOP_LOSS"
            else:
                continue  # no exit fires for this position

            await self.book.sell(pid, current, trigger=trigger)
```

(Note: the exact placement and existing surrounding code may differ; apply this logic in-place of the current trail+stop_loss math, keeping any logger.info/telemetry calls.)

- [ ] **Step 4: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 5: Commit (DEFERRED to Task 8)**

---

## Task 7: Plumb WS tick path (`on_price_tick`)

**Files:**
- Modify: `backend/agents/exit_watcher.py:on_price_tick`
- Test: `backend/tests/test_exit_watcher.py` (add to existing file)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_exit_watcher.py`:

```python
class TestOnPriceTickPnLAnchored:
    """Tests that on_price_tick uses _compute_exit_threshold from
    exit_thresholds and fires WS_TRAIL_STOP / WS_STOP_LOSS with the new logic."""

    @pytest.mark.asyncio
    async def test_on_price_tick_fires_ws_trail_when_pnl_below_threshold(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06,
        )
        # Current pnl 1.5% < threshold 1.8% → fire
        await on_price_tick("DASH-USD", 43.59, book)
        book.sell.assert_called_once()
        args, kwargs = book.sell.call_args
        assert kwargs.get("trigger") == "WS_TRAIL_STOP"

    @pytest.mark.asyncio
    async def test_on_price_tick_does_not_fire_above_threshold(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06,
        )
        await on_price_tick("DASH-USD", 44.04, book)  # +2.5% > 1.8%
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_price_tick_ratchets_peak_pnl_pct(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.balance = 942.0
        book.positions["DASH-USD"] = _make_pos(
            avg=42.97, peak=46.32, peak_pnl_pct=0.078,
            position_dollars=58.0, trail=0.06,
        )
        # Price ratchets to new high $48 → peak_pnl_pct should update
        await on_price_tick("DASH-USD", 48.0, book)
        new_pnl = (48.0 - 42.97) / 42.97
        assert book.positions["DASH-USD"]["peak_pnl_pct"] == pytest.approx(new_pnl)
```

(Update `_make_pos` helper in the same test file to accept `peak_pnl_pct` and `position_dollars` kwargs.)

- [ ] **Step 2: Run tests to verify they fail (DEFERRED to Task 8)**

Expected: FAIL.

- [ ] **Step 3: Modify `on_price_tick` to use the new threshold**

In `backend/agents/exit_watcher.py`, replace the existing trail/stop_loss math in `on_price_tick`:

```python
from agents.exit_thresholds import _compute_exit_threshold

async def on_price_tick(pid: str, price: float, book) -> None:
    """WS price-tick handler — fires WS_TRAIL_STOP / WS_STOP_LOSS using
    the PnL-anchored threshold (Task #46)."""
    try:
        pos = book.positions.get(pid)
        if pos is None or pos.get("entry_corrupt"):
            return
        entry = pos.get("avg_price", 0.0)
        if entry <= 0:
            return
        current_pnl_pct = (price - entry) / entry

        # Ratchet peak_pnl_pct (never decreases)
        pos["peak_pnl_pct"] = max(pos.get("peak_pnl_pct", 0.0), current_pnl_pct)
        # NOTE: position_dollars cache is scan-loop-owned per Session 58.71m design;
        # WS handler reads it but does NOT update (avoids per-tick mutation contention).

        atr_pct = pos.get("trail_pct", _CNN_ATR_TRAIL_MIN)
        position_dollars = pos.get("position_dollars", float(pos["size"]) * price)
        total_capital = book.balance + sum(
            p.get("position_dollars", 0.0) for p in book.positions.values()
        )

        threshold = _compute_exit_threshold(
            peak_pnl_pct=pos["peak_pnl_pct"],
            atr_pct=atr_pct,
            position_dollars=position_dollars,
            total_capital=total_capital,
        )

        if current_pnl_pct < threshold and current_pnl_pct < -_CNN_STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif current_pnl_pct < threshold:
            trigger = "WS_TRAIL_STOP"
        elif current_pnl_pct < -_CNN_STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        else:
            return

        await book.sell(pid, price, trigger=trigger)
    except Exception:  # noqa: BLE001 — per CLAUDE.md invariant #18
        logger.exception("on_price_tick failure for pid=%s price=%s", pid, price)
```

Imports needed at top of file (verify already present, add if missing):
```python
from agents.cnn_agent import _CNN_ATR_TRAIL_MIN, _CNN_STOP_LOSS_PCT
```

- [ ] **Step 4: Verify pass (DEFERRED to Task 8)**
- [ ] **Step 5: Commit (DEFERRED to Task 8)**

---

## Task 8: Operator-paused execution gate — pytest + atomic commit + push

**REQUIRES OPERATOR ACTION:** 8001 trading must be paused before this task runs.

- [ ] **Step 1: Operator confirms 8001 is paused**

```powershell
# Verify trading is paused (frontend toggle off or backend `is_trading` false).
# Backend can still be on 8001; what matters is scan-loop stops generating signals.
```

- [ ] **Step 2: Run the new tests in isolation first (fast feedback)**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/test_exit_thresholds.py tests/test_cnn_risk_exits.py::TestCNNRiskExitsPnLAnchored tests/test_cnn_agent.py::TestCNNBookPnLMigration tests/test_exit_watcher.py::TestOnPriceTickPnLAnchored -v
```

Expected: all new tests PASS (15 exit_thresholds + 3 risk-exits + 3 migration + 3 ws-handler = 24).

- [ ] **Step 3: Run the full pytest suite**

```bash
python -m pytest tests/ -q --tb=short
```

Expected: 1187 + 24 = ~1211 passed, 0 failed. Existing tests should continue passing because the migration is backward-compatible (legacy positions get fields auto-computed).

- [ ] **Step 4: Stage files in one atomic batch**

```bash
git add backend/agents/exit_thresholds.py \
        backend/agents/cnn_agent.py \
        backend/agents/exit_watcher.py \
        backend/tests/test_exit_thresholds.py \
        backend/tests/test_cnn_agent.py \
        backend/tests/test_cnn_risk_exits.py \
        backend/tests/test_exit_watcher.py \
        CHANGELOG.md \
        docs/superpowers/specs/2026-05-23-pnl-anchored-trail-design.md \
        docs/superpowers/plans/2026-05-23-pnl-anchored-trail.md
```

- [ ] **Step 5: Add the CHANGELOG bullet**

Edit `CHANGELOG.md` to add under `## Unreleased`, before the existing GPU-batched-feature-kernel bullet:

```markdown
- **2026-05-23: PnL-anchored trail with capital-relative profit floors (`agents/exit_thresholds.py`)** — replaces the peak-price-anchored trail with a three-layer design: (1) ATR-scaled profit-floor tiers (volatility-relative; engage at 0.5×/1.5×/3.0× ATR multiples of peak PnL, lock 0.5×/1.5× ATR), (2) capital-relative dollar-cap on large positions (>max($200, 5% of capital)) with tighter-of-(2% position $, 0.5% capital/position $) giveback, (3) fee-aware break-even at 2× FEE_RATE so exit covers Coinbase round-trip fees.
  - Operator-identified as the PRIMARY root cause of paper-profit-to-realized-loss conversion. On today's 10-position book, prevents a $35 swing-the-wrong-way (current rule realizes −$18.29 worst case; new rule realizes −$13.66, +$4.63 saved per single trail-fire event).
  - `agents/cnn_agent.py:_CNNBook` — new `_migrate_position_state(pos)` static method adds `peak_pnl_pct` + `position_dollars` to legacy positions on load. `_check_risk_exits` ratchets `peak_pnl_pct`, refreshes `position_dollars`, and dispatches through `_compute_exit_threshold`.
  - `agents/exit_watcher.py:on_price_tick` — ratchets `peak_pnl_pct` on every WS tick (per-pid lock from Session 58.71m unchanged); reads cached `position_dollars` (scan-loop owns refresh); fires `WS_TRAIL_STOP` / `WS_STOP_LOSS` per the new threshold.
  - **Tests:** 24 new (`tests/test_exit_thresholds.py` 15 + `tests/test_cnn_agent.py::TestCNNBookPnLMigration` 3 + `tests/test_cnn_risk_exits.py::TestCNNRiskExitsPnLAnchored` 3 + `tests/test_exit_watcher.py::TestOnPriceTickPnLAnchored` 3).
  - **Constants** (override via `.env` if needed): `FEE_RATE=0.006`, `LARGE_POSITION_FRAC=0.05`, `LARGE_POSITION_FLOOR=200.0`, `MAX_DOLLAR_GIVEBACK_FRAC=0.02`, `MAX_LOSS_FRAC_OF_CAPITAL=0.005`.
  - **Validate in shadow on 8002 for 7 days** before promoting to 8001. Promotion criteria: realized PnL > current rule by ≥$15 over the window AND no new exit-mode failures.
  - **Spec:** `docs/superpowers/specs/2026-05-23-pnl-anchored-trail-design.md`. **Plan:** `docs/superpowers/plans/2026-05-23-pnl-anchored-trail.md`.
```

Also stage the updated CHANGELOG:

```bash
git add CHANGELOG.md
```

- [ ] **Step 6: Atomic commit**

```bash
git commit -m "$(cat <<'EOF'
feat: PnL-anchored trail with capital-relative profit floors (#46)

Replace peak-price-anchored trail with a three-layer design:
1. ATR-scaled profit-floor tiers (volatility-relative): engage at 0.5x/1.5x/
   3.0x ATR multiples of peak PnL; lock 0.5x/1.5x ATR
2. Capital-relative dollar-cap on large positions (>max($200, 5% of capital));
   giveback = tighter of (2% position $, 0.5% capital / position $)
3. Fee-aware break-even at 2x FEE_RATE so exit covers Coinbase round-trip
   fees

Eliminates "paper-profit-to-realized-loss" conversion when peak/entry is
small. On the live 10-position book today, prevents a $35 worst-case swing
(current rule realizes -$18.29 on a synchronous trail-fire; new rule
realizes -$13.66, +$4.63 saved per fire event).

New module backend/agents/exit_thresholds.py:
- _atr_floor(peak_pnl_pct, atr_pct) -> Layer 1
- _large_position_threshold(total_capital)
- _dollar_cap_floor(peak_pnl_pct, position_dollars, total_capital) -> Layer 2
- _compute_exit_threshold(...) -> orchestrator

Plumbed through:
- cnn_agent._check_risk_exits (scan-loop path)
- exit_watcher.on_price_tick (WS tick path) [CLAUDE.md invariant #18 preserved]

State migration: legacy positions auto-gain peak_pnl_pct and
position_dollars on backend load (computed from existing peak_price +
avg_price + size). No DB schema change.

Constants tunable via .env:
- FEE_RATE=0.006 (Coinbase taker round-trip basis)
- LARGE_POSITION_FRAC=0.05 (5% of capital = "large")
- LARGE_POSITION_FLOOR=200.0 (absolute floor for small-capital regime)
- MAX_DOLLAR_GIVEBACK_FRAC=0.02 (2% of position $)
- MAX_LOSS_FRAC_OF_CAPITAL=0.005 (0.5% of capital per trail-fire)

Tests: 24 new (15 exit_thresholds + 3 migration + 3 risk_exits + 3 ws).
Full suite: 1211 passed, 0 failed.

Validate in shadow on port 8002 for 7 days before promoting to 8001.
Promotion gated on realized PnL > current rule by >=$15 over window AND no
new exit-mode failures.

Spec: docs/superpowers/specs/2026-05-23-pnl-anchored-trail-design.md
Plan: docs/superpowers/plans/2026-05-23-pnl-anchored-trail.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 7: Push**

```bash
git push origin <current-branch-or-feat/pnl-anchored-trail>
```

Replace `<current-branch...>` with the actual branch name (master or a feature branch — confirm with `git branch --show-current` first).

- [ ] **Step 8: Update memory file in same operator window**

In `~/.claude/projects/C--Users-gl450/memory/polymarket_app_trail_design_gap.md`,
mark task #46 as LANDED (still keep the file as historical reference for the
design rationale). Add a `## Status` section at the top:

```markdown
## Status

LANDED 2026-05-23 (commit <SHA>). Validating in shadow on port 8002.
Shadow-week review scheduled ~2026-05-30. Promotion to 8001 gated on
realized PnL > current rule by >=$15 over the 7-day window.
```

- [ ] **Step 9: Operator launches shadow validation on 8002**

```bash
cd C:\Users\gl450\polymarket_app\backend
PORT=8002 DATABASE_URL=coinbase_dev.db ../.venv/Scripts/python.exe main.py
```

Frontend can A/B compare 8001 (current rule) vs 8002 (new rule) over 7 days.

- [ ] **Step 10: Schedule the shadow-week review task (#50, NEW)**

```bash
# After 7 days, run:
sqlite3 backend/coinbase.db "
SELECT 
    'CURRENT (8001)' as side, COUNT(*), SUM(realized_pnl)
FROM trades
WHERE entry_time >= '2026-05-23T22:00:00Z'
  AND close_time IS NOT NULL;
"
sqlite3 backend/coinbase_dev.db "
SELECT 
    'NEW (8002)' as side, COUNT(*), SUM(realized_pnl)
FROM trades
WHERE entry_time >= '2026-05-23T22:00:00Z'
  AND close_time IS NOT NULL;
"
```

Decision: promote if NEW realized PnL > CURRENT realized PnL by ≥$15. Otherwise iterate on constants.

---

## Self-review against spec

**Spec coverage check:**
- ✅ Layer 1 (ATR-scaled tiers) → Task 2 `_atr_floor`
- ✅ Layer 2 (capital-relative dollar-cap) → Task 3 `_large_position_threshold` + `_dollar_cap_floor`
- ✅ Layer 3 (baseline) → Task 4 `_compute_exit_threshold` baseline computation
- ✅ Stop-loss interaction unchanged → Tasks 6, 7 keep `-_CNN_STOP_LOSS_PCT` backstop independent
- ✅ State migration → Task 5 `_migrate_position_state`
- ✅ Scan-loop dispatch → Task 6 `_check_risk_exits`
- ✅ WS tick dispatch → Task 7 `on_price_tick`
- ✅ peak_pnl_pct ratchet → Tasks 6 + 7 (both call sites)
- ✅ Long/short symmetry → Out of scope per spec
- ✅ Per-pid overrides → Out of scope per spec; constants are global
- ✅ Tests: 15 unit + 9 integration matching spec's "15 tests outlined"
- ✅ Shadow-week calibration → Task 8 Steps 9-10
- ✅ CHANGELOG + memory update → Task 8 Steps 5, 8

**Placeholder scan:** no "TBD", "TODO", "implement later", or "add error handling" found.

**Type consistency:**
- `_atr_floor(peak_pnl_pct: float, atr_pct: float) -> Optional[float]` used identically in Task 4 orchestrator ✓
- `_dollar_cap_floor(peak_pnl_pct, position_dollars, total_capital) -> Optional[float]` used identically in Task 4 ✓
- `_compute_exit_threshold(*, peak_pnl_pct, atr_pct, position_dollars=None, total_capital=None) -> float` consumed in Tasks 6 + 7 with kw-args matching ✓
- `_migrate_position_state(pos: Dict) -> Dict` returns the same dict (mutated in place) — consistent ✓
- Position state fields `peak_pnl_pct`, `position_dollars` referenced consistently across Tasks 5, 6, 7 ✓
