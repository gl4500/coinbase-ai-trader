# Macro-Regime Layer — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline macro-regime evaluator (MVRV cycle prior × correlation-gated macro risk → `exposure_scalar`), backfill a daily historical regime series, and a backtest harness that measures whether regime-scaling exposure improves risk-adjusted trade outcomes — producing the go/no-go verdict for Phase 2.

**Architecture:** A pure `RegimeEvaluator` (formula only, no I/O) consumes three scalar inputs (MVRV, BTC-SPX 90d correlation, macro-risk score) and emits a `RegimeState`. A feature module derives the correlation + macro-risk score from daily series; a sources module fetches+caches those series (FRED + CoinMetrics). An offline builder writes the daily regime series to a `regime_state` DB table; an offline backtest harness overlays `exposure_scalar` on the historical `trades` record and reports risk-adjusted deltas. **No live/scan-loop wiring in Phase 1** — that is Phase 2, gated on this phase's verdict.

**Tech Stack:** Python 3.11, numpy, pandas, aiosqlite, urllib (stdlib), pytest + pytest-asyncio, unittest.mock.

## Global Constraints

- **Phase 1 is offline — zero live risk.** No changes to `cnn_agent`, `main.py`, the scan loop, or sizing. No new env flag (that is Phase 2). Nothing in this plan may alter live trading behavior.
- **Branch:** `feat/macro-regime-layer` (off `main`). Surgical pathspec on every commit; never `git commit -a`. Confirm branch (`git rev-parse --abbrev-ref HEAD`) before each `git add`.
- **TDD:** failing test → run red → implement → run green → commit, per task.
- **Formula constants (verbatim, all config-tunable — Phase-1 backtest is the judge):** MVRV anchors `[(0.8,1.25),(1.5,1.05),(3.0,0.95),(3.5,0.85)]` (linear interp between; flat outside); `MACRO_K = 0.3`; `EXPOSURE_CLAMP = (0.4, 1.25)`; `REGIME_STALE_DAYS = 3`.
- **Fail-safe:** any missing/stale input → its factor = 1.0; all missing → `exposure_scalar = 1.0`, `confidence = 0.0`. The evaluator must never raise.
- **Test interpreter:** `C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe` (has numpy/pandas/torch). The worktree `.venv` is a junction to it.
- **Test env stubs (top of every new test file):**
  ```python
  import os
  os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
  os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
  os.environ.setdefault("DRY_RUN",                  "true")
  os.environ.setdefault("LOG_LEVEL",                "WARNING")
  os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")
  ```

## File Structure

- `backend/services/regime/__init__.py` — package marker.
- `backend/services/regime/state.py` — `RegimeState` dataclass + formula constants. (Task 1)
- `backend/services/regime/macro_regime.py` — `RegimeEvaluator`: pure formula (`mvrv_prior`, `macro_mult`, `evaluate`). (Task 2)
- `backend/services/regime/features.py` — derive `corr_spx_90d`, `macro_risk_raw` from series. (Task 3)
- `backend/services/regime/sources.py` — FRED + CoinMetrics fetch + parquet cache + aligned loader. (Task 4)
- `backend/database.py` — add `regime_state` table + `upsert_regime_state` / `get_latest_regime_state` / `get_regime_series`. (Task 5)
- `backend/tools/regime/build_regime_series.py` — offline daily-series builder. (Task 6)
- `backend/tools/regime/backtest_regime.py` — offline backtest harness + verdict. (Task 7)
- Tests: `backend/tests/regime/test_state.py`, `test_macro_regime.py`, `test_features.py`, `test_sources.py`, `test_regime_store.py`, `test_build_regime_series.py`, `test_backtest_regime.py`.

---

### Task 1: `RegimeState` + formula constants

**Files:**
- Create: `backend/services/regime/__init__.py` (empty)
- Create: `backend/services/regime/state.py`
- Test: `backend/tests/regime/__init__.py` (empty), `backend/tests/regime/test_state.py`

**Interfaces:**
- Produces: `RegimeState` dataclass with fields `date:str, mvrv:Optional[float], mvrv_prior:float, corr_spx_90d:Optional[float], macro_risk_raw:Optional[float], macro_mult:float, exposure_scalar:float, confidence:float, components:Dict[str,float]`; module constants `MVRV_ANCHORS`, `MACRO_K`, `EXPOSURE_CLAMP`, `REGIME_STALE_DAYS`; method `RegimeState.to_row() -> dict` and `RegimeState.from_row(dict) -> RegimeState`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/__init__.py` (empty) and `backend/tests/regime/test_state.py`:

```python
import os, sys
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import json
from services.regime.state import (
    RegimeState, MVRV_ANCHORS, MACRO_K, EXPOSURE_CLAMP, REGIME_STALE_DAYS,
)


def test_constants():
    assert MVRV_ANCHORS == [(0.8, 1.25), (1.5, 1.05), (3.0, 0.95), (3.5, 0.85)]
    assert MACRO_K == 0.3
    assert EXPOSURE_CLAMP == (0.4, 1.25)
    assert REGIME_STALE_DAYS == 3


def test_roundtrip_row():
    rs = RegimeState(date="2025-11-30", mvrv=1.0, mvrv_prior=1.1,
                     corr_spx_90d=0.02, macro_risk_raw=0.3, macro_mult=1.0,
                     exposure_scalar=1.1, confidence=1.0,
                     components={"equity_trend": 0.5})
    row = rs.to_row()
    assert row["date"] == "2025-11-30"
    assert row["exposure_scalar"] == 1.1
    assert json.loads(row["components"])["equity_trend"] == 0.5
    back = RegimeState.from_row(row)
    assert back.exposure_scalar == 1.1
    assert back.components["equity_trend"] == 0.5
    assert back.mvrv == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.regime'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/services/regime/__init__.py` (empty file). Create `backend/services/regime/state.py`:

```python
"""RegimeState value object + macro-regime formula constants.

All constants are config-tunable; the Phase-1 backtest is the judge of their
values (see docs/superpowers/specs/2026-07-05-macro-regime-layer-design.md).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# (mvrv, prior) anchors; linear interpolation between, flat outside the ends.
MVRV_ANCHORS: List[Tuple[float, float]] = [(0.8, 1.25), (1.5, 1.05), (3.0, 0.95), (3.5, 0.85)]
MACRO_K: float = 0.3
EXPOSURE_CLAMP: Tuple[float, float] = (0.4, 1.25)
REGIME_STALE_DAYS: int = 3


@dataclass
class RegimeState:
    date: str
    mvrv: Optional[float]
    mvrv_prior: float
    corr_spx_90d: Optional[float]
    macro_risk_raw: Optional[float]
    macro_mult: float
    exposure_scalar: float
    confidence: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict:
        return {
            "date": self.date, "mvrv": self.mvrv, "mvrv_prior": self.mvrv_prior,
            "corr_spx_90d": self.corr_spx_90d, "macro_risk_raw": self.macro_risk_raw,
            "macro_mult": self.macro_mult, "exposure_scalar": self.exposure_scalar,
            "confidence": self.confidence, "components": json.dumps(self.components),
        }

    @classmethod
    def from_row(cls, row: dict) -> "RegimeState":
        comp = row.get("components")
        return cls(
            date=row["date"], mvrv=row.get("mvrv"), mvrv_prior=row["mvrv_prior"],
            corr_spx_90d=row.get("corr_spx_90d"), macro_risk_raw=row.get("macro_risk_raw"),
            macro_mult=row["macro_mult"], exposure_scalar=row["exposure_scalar"],
            confidence=row.get("confidence", 0.0),
            components=json.loads(comp) if isinstance(comp, str) else (comp or {}),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_state.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # expect feat/macro-regime-layer
git add backend/services/regime/__init__.py backend/services/regime/state.py \
        backend/tests/regime/__init__.py backend/tests/regime/test_state.py
git commit -m "feat: RegimeState value object + macro-regime formula constants

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 2: `RegimeEvaluator` pure formula

**Files:**
- Create: `backend/services/regime/macro_regime.py`
- Test: `backend/tests/regime/test_macro_regime.py`

**Interfaces:**
- Consumes: `RegimeState`, constants from Task 1.
- Produces:
  - `mvrv_prior(mvrv: Optional[float]) -> float` — piecewise-linear on `MVRV_ANCHORS`; `None` → 1.0.
  - `macro_mult(corr_spx_90d: Optional[float], macro_risk_raw: Optional[float], k: float = MACRO_K) -> float` — `1 + k*max(0,corr)*risk`; any `None` → 1.0.
  - `evaluate(*, date: str, mvrv: Optional[float], corr_spx_90d: Optional[float], macro_risk_raw: Optional[float], mvrv_age_days: int = 0, macro_age_days: int = 0) -> RegimeState` — applies staleness (age > `REGIME_STALE_DAYS` ⇒ treat input as `None`), combines + clamps, sets `confidence` = fraction of fresh factors, never raises.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_macro_regime.py`:

```python
import os, sys
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import pytest
from services.regime.macro_regime import mvrv_prior, macro_mult, evaluate


class TestMvrvPrior:
    def test_cheap_zone_max_leanin(self):
        assert mvrv_prior(0.5) == 1.25
        assert mvrv_prior(0.8) == 1.25

    def test_extended_zone_trim(self):
        assert mvrv_prior(4.0) == 0.85
        assert mvrv_prior(3.5) == 0.85

    def test_interpolation_midpoints(self):
        assert mvrv_prior(1.15) == pytest.approx(1.15)   # halfway 0.8->1.5 : 1.25->1.05
        assert mvrv_prior(2.25) == pytest.approx(1.00)   # halfway 1.5->3.0 : 1.05->0.95

    def test_none_is_neutral(self):
        assert mvrv_prior(None) == 1.0


class TestMacroMult:
    def test_decoupled_ignores_macro(self):
        # corr 0 -> w=0 -> macro ignored regardless of risk
        assert macro_mult(0.0, -1.0) == 1.0
        assert macro_mult(-0.3, 1.0) == 1.0   # negative corr also gates to 0

    def test_coupled_riskoff_reduces(self):
        assert macro_mult(0.6, -0.8) == pytest.approx(1 - 0.3 * 0.6 * 0.8)

    def test_coupled_riskon_boosts(self):
        assert macro_mult(0.5, 1.0) == pytest.approx(1 + 0.3 * 0.5 * 1.0)

    def test_none_is_neutral(self):
        assert macro_mult(None, 0.5) == 1.0
        assert macro_mult(0.5, None) == 1.0


class TestEvaluate:
    def test_worked_example_2022_coupled_riskoff(self):
        rs = evaluate(date="2022-06-30", mvrv=2.5, corr_spx_90d=0.6, macro_risk_raw=-0.8)
        assert rs.macro_mult == pytest.approx(0.856, abs=1e-3)
        assert rs.exposure_scalar < 1.0
        assert rs.confidence == 1.0

    def test_worked_example_nov2025_decoupled(self):
        rs = evaluate(date="2025-11-30", mvrv=1.0, corr_spx_90d=0.0, macro_risk_raw=-1.0)
        assert rs.macro_mult == 1.0            # macro ignored
        assert rs.exposure_scalar > 1.0        # MVRV prior leads
        assert rs.exposure_scalar == pytest.approx(min(mvrv_prior(1.0), 1.25))

    def test_cycle_bottom_leans_in(self):
        rs = evaluate(date="2018-12-15", mvrv=0.7, corr_spx_90d=0.3, macro_risk_raw=-0.5)
        assert rs.mvrv_prior == 1.25

    def test_clamp_upper(self):
        rs = evaluate(date="2020-01-01", mvrv=0.5, corr_spx_90d=0.6, macro_risk_raw=1.0)
        assert rs.exposure_scalar == 1.25      # 1.25 * 1.18 clamped to 1.25

    def test_clamp_lower(self):
        # force below floor via extreme (defensive) inputs
        rs = evaluate(date="2020-01-01", mvrv=3.5, corr_spx_90d=1.0, macro_risk_raw=-1.0)
        assert rs.exposure_scalar >= 0.4

    def test_all_missing_is_neutral(self):
        rs = evaluate(date="2020-01-01", mvrv=None, corr_spx_90d=None, macro_risk_raw=None)
        assert rs.exposure_scalar == 1.0
        assert rs.confidence == 0.0

    def test_staleness_treated_as_missing(self):
        rs = evaluate(date="2020-01-01", mvrv=0.5, corr_spx_90d=0.6, macro_risk_raw=1.0,
                      mvrv_age_days=10)
        assert rs.mvrv_prior == 1.0            # stale MVRV -> neutral factor
        assert rs.confidence == pytest.approx(0.5)

    def test_never_raises_on_garbage(self):
        rs = evaluate(date="x", mvrv=float("nan"), corr_spx_90d=float("nan"),
                      macro_risk_raw=float("nan"))
        assert 0.4 <= rs.exposure_scalar <= 1.25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_macro_regime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.regime.macro_regime'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/services/regime/macro_regime.py`:

```python
"""RegimeEvaluator — the macro-regime exposure formula (pure, no I/O).

exposure_scalar = clip(mvrv_prior * macro_mult, *EXPOSURE_CLAMP), applied to
ENTRY sizing only. Any missing/stale input degrades its factor to 1.0; all
missing -> 1.0 (neutral). Never raises. See the design spec for rationale.
"""
from __future__ import annotations

import math
from typing import Optional

from services.regime.state import (
    RegimeState, MVRV_ANCHORS, MACRO_K, EXPOSURE_CLAMP, REGIME_STALE_DAYS,
)


def _finite(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return x if math.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def mvrv_prior(mvrv: Optional[float]) -> float:
    mvrv = _finite(mvrv)
    if mvrv is None:
        return 1.0
    lo_m, lo_p = MVRV_ANCHORS[0]
    if mvrv <= lo_m:
        return lo_p
    hi_m, hi_p = MVRV_ANCHORS[-1]
    if mvrv >= hi_m:
        return hi_p
    for (m0, p0), (m1, p1) in zip(MVRV_ANCHORS, MVRV_ANCHORS[1:]):
        if m0 <= mvrv <= m1:
            frac = (mvrv - m0) / (m1 - m0)
            return p0 + frac * (p1 - p0)
    return 1.0


def macro_mult(corr_spx_90d: Optional[float], macro_risk_raw: Optional[float],
               k: float = MACRO_K) -> float:
    corr = _finite(corr_spx_90d)
    risk = _finite(macro_risk_raw)
    if corr is None or risk is None:
        return 1.0
    w = max(0.0, corr)
    return 1.0 + k * w * risk


def evaluate(*, date: str, mvrv: Optional[float], corr_spx_90d: Optional[float],
             macro_risk_raw: Optional[float], mvrv_age_days: int = 0,
             macro_age_days: int = 0) -> RegimeState:
    # Staleness: too-old inputs are treated as missing.
    mvrv_in = _finite(mvrv) if mvrv_age_days <= REGIME_STALE_DAYS else None
    if macro_age_days <= REGIME_STALE_DAYS:
        corr_in, risk_in = _finite(corr_spx_90d), _finite(macro_risk_raw)
    else:
        corr_in, risk_in = None, None

    prior = mvrv_prior(mvrv_in)
    mult = macro_mult(corr_in, risk_in)
    lo, hi = EXPOSURE_CLAMP
    scalar = max(lo, min(hi, prior * mult))

    mvrv_fresh = 1.0 if mvrv_in is not None else 0.0
    macro_fresh = 1.0 if (corr_in is not None and risk_in is not None) else 0.0
    confidence = (mvrv_fresh + macro_fresh) / 2.0

    return RegimeState(
        date=date, mvrv=mvrv_in, mvrv_prior=prior, corr_spx_90d=corr_in,
        macro_risk_raw=risk_in, macro_mult=mult, exposure_scalar=scalar,
        confidence=confidence,
        components={"mvrv_prior": prior, "macro_mult": mult},
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_macro_regime.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/services/regime/macro_regime.py backend/tests/regime/test_macro_regime.py
git commit -m "feat: RegimeEvaluator pure exposure formula (MVRV x corr-gated macro)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 3: Feature derivation (`corr_spx_90d`, `macro_risk_raw`)

**Files:**
- Create: `backend/services/regime/features.py`
- Test: `backend/tests/regime/test_features.py`

**Interfaces:**
- Produces:
  - `corr_spx_90d(btc_closes: list[float], spx_closes: list[float], window: int = 90) -> Optional[float]` — Pearson corr of daily log returns over the last `window` overlapping pairs; `None` if < 30 usable pairs.
  - `macro_risk_raw(spx_closes: list[float], dxy_closes: list[float], real_yield: list[float]) -> Optional[float]` — mean of available standardized sub-signals (equity trend, dollar direction, real-yield direction), clipped to [-1, 1]; `None` if none computable.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_features.py`:

```python
import os, sys, math
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import pytest
from services.regime.features import corr_spx_90d, macro_risk_raw


def test_corr_perfectly_coupled():
    # identical return streams -> corr ~ +1
    base = [100.0]
    for i in range(120):
        base.append(base[-1] * (1.0 + 0.01 * math.sin(i)))
    assert corr_spx_90d(base, base) == pytest.approx(1.0, abs=1e-6)


def test_corr_insufficient_data_none():
    assert corr_spx_90d([100, 101, 102], [100, 101, 102]) is None


def test_macro_risk_riskon_when_equities_up_dollar_down():
    # SPX rising above its 50d MA, DXY falling, real yield falling -> risk-on (+)
    spx = [100 + i for i in range(60)]
    dxy = [100 - 0.05 * i for i in range(60)]
    ry = [2.0 - 0.01 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert v is not None and v > 0.5


def test_macro_risk_riskoff_when_equities_down_dollar_up():
    spx = [160 - i for i in range(60)]
    dxy = [100 + 0.05 * i for i in range(60)]
    ry = [1.0 + 0.01 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert v is not None and v < -0.5


def test_macro_risk_clipped():
    spx = [100 + 5 * i for i in range(60)]
    dxy = [100 - 1.0 * i for i in range(60)]
    ry = [3.0 - 0.1 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert -1.0 <= v <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.regime.features'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/services/regime/features.py`:

```python
"""Derive the macro-regime scalar features from daily series.

corr_spx_90d: BTC-SPX daily-log-return Pearson over the last 90 pairs.
macro_risk_raw: equity trend + dollar direction + real-yield direction,
each standardized to ~[-1,1], averaged over the available sub-signals.
Callers pass already-date-aligned recent series (oldest..newest).
"""
from __future__ import annotations

import math
from typing import List, Optional


def _log_returns(closes: List[float]) -> List[float]:
    out = []
    for a, b in zip(closes, closes[1:]):
        if a and b and a > 0 and b > 0:
            out.append(math.log(b / a))
    return out


def corr_spx_90d(btc_closes: List[float], spx_closes: List[float],
                 window: int = 90) -> Optional[float]:
    n = min(len(btc_closes), len(spx_closes))
    if n < 31:
        return None
    rb = _log_returns(btc_closes[-(window + 1):])
    rs = _log_returns(spx_closes[-(window + 1):])
    m = min(len(rb), len(rs))
    if m < 30:
        return None
    rb, rs = rb[-m:], rs[-m:]
    mb, ms = sum(rb) / m, sum(rs) / m
    num = sum((a - mb) * (b - ms) for a, b in zip(rb, rs))
    db = math.sqrt(sum((a - mb) ** 2 for a in rb))
    ds = math.sqrt(sum((b - ms) ** 2 for b in rs))
    if db == 0 or ds == 0:
        return None
    return max(-1.0, min(1.0, num / (db * ds)))


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


def macro_risk_raw(spx_closes: List[float], dxy_closes: List[float],
                   real_yield: List[float]) -> Optional[float]:
    subs: List[float] = []
    # Equity trend: % of last close above/below its 50d MA, +-5% -> +-1.
    if len(spx_closes) >= 50 and spx_closes[-1] > 0:
        sma = sum(spx_closes[-50:]) / 50.0
        if sma > 0:
            subs.append(_clip((spx_closes[-1] / sma - 1.0) / 0.05))
    # Dollar direction: DXY 20d change; FALLING dollar = risk-on (+). +-2% -> +-1.
    if len(dxy_closes) >= 21 and dxy_closes[-21] > 0:
        subs.append(_clip(-(dxy_closes[-1] / dxy_closes[-21] - 1.0) / 0.02))
    # Real-yield direction: 20d change in pct-points; RISING yield = risk-off (-).
    if len(real_yield) >= 21:
        subs.append(_clip(-(real_yield[-1] - real_yield[-21]) / 0.25))
    if not subs:
        return None
    return _clip(sum(subs) / len(subs))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_features.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/services/regime/features.py backend/tests/regime/test_features.py
git commit -m "feat: regime feature derivation (corr_spx_90d, macro_risk_raw)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 4: Data sources (FRED + CoinMetrics) with parquet cache

**Files:**
- Create: `backend/services/regime/sources.py`
- Test: `backend/tests/regime/test_sources.py`

**Interfaces:**
- Produces:
  - `fetch_fred(series_id: str, start: str, session_get=_urlopen) -> "pd.Series"` — daily series indexed by date; empty Series on failure.
  - `fetch_mvrv(start: str, session_get=_urlopen) -> "pd.Series"` — CoinMetrics `CapMVRVCur`; empty on failure.
  - `load_aligned(start: str, cache_dir: str, session_get=_urlopen) -> "pd.DataFrame"` — columns `btc, spx, dxy, real_yield, mvrv`, daily, ffilled ≤4d; reads a `<cache_dir>/regime_sources.parquet` cache first and refetches only if the cache is stale/absent, then rewrites it. Network failure falls back to whatever the cache holds.
- Notes: `session_get(url, headers)->bytes` is injected so tests never hit the network. CoinMetrics REQUIRES a browser User-Agent (else 403); FRED needs none. Series ids: BTC=`CBBTCUSD`, SPX=`SP500`, DXY=`DTWEXBGS`, real_yield=`DFII10`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_sources.py`:

```python
import os, sys, json
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import pandas as pd
from services.regime import sources


def _fake_get(fred_csv: bytes, cm_json: bytes):
    def _get(url, headers=None):
        if "stlouisfed" in url:
            return fred_csv
        return cm_json
    return _get


def test_fetch_fred_parses_csv():
    csv = b"observation_date,DFII10\n2024-01-02,2.10\n2024-01-03,.\n2024-01-04,2.20\n"
    s = sources.fetch_fred("DFII10", "2024-01-01", session_get=lambda u, headers=None: csv)
    assert len(s) == 2                       # "." dropped
    assert s.iloc[-1] == 2.20


def test_fetch_fred_network_error_returns_empty():
    def boom(u, headers=None):
        raise OSError("net down")
    s = sources.fetch_fred("DFII10", "2024-01-01", session_get=boom)
    assert s.empty


def test_fetch_mvrv_parses_json():
    body = json.dumps({"data": [
        {"time": "2024-01-02T00:00:00.000000000Z", "CapMVRVCur": "1.5"},
        {"time": "2024-01-03T00:00:00.000000000Z", "CapMVRVCur": "1.6"},
    ]}).encode()
    s = sources.fetch_mvrv("2024-01-01", session_get=lambda u, headers=None: body)
    assert s.iloc[-1] == 1.6


def test_load_aligned_uses_cache_on_network_failure(tmp_path):
    # Seed a cache, then force network failure -> load returns cached frame.
    cache = tmp_path / "regime_sources.parquet"
    df = pd.DataFrame(
        {"btc": [1.0, 2.0], "spx": [1.0, 1.0], "dxy": [1.0, 1.0],
         "real_yield": [2.0, 2.0], "mvrv": [1.0, 1.1]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    df.to_parquet(cache)
    def boom(u, headers=None):
        raise OSError("net down")
    out = sources.load_aligned("2024-01-01", str(tmp_path), session_get=boom)
    assert list(out.columns) == ["btc", "spx", "dxy", "real_yield", "mvrv"]
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_sources.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'services.regime.sources'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/services/regime/sources.py`:

```python
"""Fetch + cache the daily series the regime layer needs.

FRED (no key): CBBTCUSD, SP500, DTWEXBGS, DFII10.
CoinMetrics community (needs browser UA): CapMVRVCur.
All fetches are injectable (session_get) so tests never hit the network, and
degrade to the local parquet cache on failure. See memory
btc_macro_drivers_findings for source quirks.
"""
from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_CM = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
       "?assets=btc&metrics=CapMVRVCur&frequency=1d&page_size=10000&start_time={}")
_FRED_IDS = {"btc": "CBBTCUSD", "spx": "SP500", "dxy": "DTWEXBGS", "real_yield": "DFII10"}
_CACHE_NAME = "regime_sources.parquet"


def _urlopen(url: str, headers=None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=40).read()


def fetch_fred(series_id: str, start: str, session_get: Callable = _urlopen) -> pd.Series:
    try:
        raw = session_get(_FRED.format(series_id)).decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "val"]
        df["date"] = pd.to_datetime(df["date"])
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna()
        s = df.set_index("date")["val"].sort_index()
        return s[s.index >= pd.Timestamp(start)]
    except Exception as e:
        logger.warning("fetch_fred(%s) failed: %s", series_id, e)
        return pd.Series(dtype=float)


def fetch_mvrv(start: str, session_get: Callable = _urlopen) -> pd.Series:
    try:
        raw = session_get(_CM.format(start), headers={"User-Agent": _UA, "Accept": "application/json"})
        data = json.loads(raw.decode()).get("data", [])
        idx = pd.to_datetime([d["time"] for d in data]).tz_localize(None).normalize()
        vals = pd.to_numeric([d.get("CapMVRVCur") for d in data], errors="coerce")
        return pd.Series(vals, index=idx).dropna().sort_index()
    except Exception as e:
        logger.warning("fetch_mvrv failed: %s", e)
        return pd.Series(dtype=float)


def load_aligned(start: str, cache_dir: str, session_get: Callable = _urlopen) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, _CACHE_NAME)
    cols = ["btc", "spx", "dxy", "real_yield", "mvrv"]
    series = {}
    for name, sid in _FRED_IDS.items():
        s = fetch_fred(sid, start, session_get)
        s.index = pd.to_datetime(s.index).normalize()
        series[name] = s
    series["mvrv"] = fetch_mvrv(start, session_get)

    if all(s.empty for s in series.values()):
        if os.path.exists(cache):
            return pd.read_parquet(cache)[cols]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(series).sort_index().asfreq("D").ffill(limit=4)
    df = df[df["btc"].notna()][cols]
    try:
        df.to_parquet(cache)
    except Exception as e:
        logger.warning("regime cache write failed: %s", e)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_sources.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/services/regime/sources.py backend/tests/regime/test_sources.py
git commit -m "feat: regime data sources (FRED + CoinMetrics) with parquet cache

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 5: `regime_state` table + store helpers in `database.py`

**Files:**
- Modify: `backend/database.py` (add table to the init block; add three async helpers near the other `upsert_*`/`get_*` functions)
- Test: `backend/tests/regime/test_regime_store.py`

**Interfaces:**
- Consumes: `RegimeState.to_row()` / `from_row()` (Task 1).
- Produces:
  - `async upsert_regime_state(row: dict) -> None` — upsert by `date` PK.
  - `async get_latest_regime_state() -> Optional[dict]` — newest row as dict (or None).
  - `async get_regime_series(start: str, end: str) -> List[dict]` — rows in `[start, end]` ascending by date.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_regime_store.py`:

```python
import os, sys
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import pytest
import database
from services.regime.state import RegimeState


@pytest.mark.asyncio
async def test_upsert_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    rs = RegimeState(date="2025-11-30", mvrv=1.0, mvrv_prior=1.1, corr_spx_90d=0.0,
                     macro_risk_raw=-1.0, macro_mult=1.0, exposure_scalar=1.1,
                     confidence=1.0, components={"a": 1.0})
    await database.upsert_regime_state(rs.to_row())
    # upsert again (idempotent on date)
    rs2 = RegimeState(**{**rs.__dict__, "exposure_scalar": 0.9})
    await database.upsert_regime_state(rs2.to_row())
    latest = await database.get_latest_regime_state()
    assert latest["date"] == "2025-11-30"
    assert latest["exposure_scalar"] == 0.9      # overwritten


@pytest.mark.asyncio
async def test_series_range(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    for d, sc in [("2025-01-01", 1.0), ("2025-06-01", 1.1), ("2025-12-01", 0.8)]:
        await database.upsert_regime_state(RegimeState(
            date=d, mvrv=1, mvrv_prior=1, corr_spx_90d=0, macro_risk_raw=0,
            macro_mult=1, exposure_scalar=sc, confidence=1, components={}).to_row())
    rows = await database.get_regime_series("2025-03-01", "2025-12-31")
    assert [r["date"] for r in rows] == ["2025-06-01", "2025-12-01"]
```

Note: if the DB init entry point is not named `init_db`, match the existing name used elsewhere in the test suite (grep `def init` in `database.py`); use that name in both the test and Step 3.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_regime_store.py -v`
Expected: FAIL — `AttributeError: module 'database' has no attribute 'upsert_regime_state'`.

- [ ] **Step 3: Write minimal implementation**

In `backend/database.py`, add this table inside the existing `CREATE TABLE IF NOT EXISTS ...` init block (alongside the other tables):

```python
            CREATE TABLE IF NOT EXISTS regime_state (
                date            TEXT PRIMARY KEY,
                mvrv            REAL,
                mvrv_prior      REAL NOT NULL,
                corr_spx_90d    REAL,
                macro_risk_raw  REAL,
                macro_mult      REAL NOT NULL,
                exposure_scalar REAL NOT NULL,
                confidence      REAL,
                components      TEXT,
                computed_at     TEXT DEFAULT (datetime('now'))
            );
```

Add these helpers near the other `upsert_*` / `get_*` functions (use the module's existing `DB_PATH`, `_DB_TIMEOUT`, and `aiosqlite.Row` pattern):

```python
async def upsert_regime_state(row: dict) -> None:
    async with aiosqlite.connect(DB_PATH, timeout=_DB_TIMEOUT) as db:
        await db.execute(
            """INSERT INTO regime_state
                 (date, mvrv, mvrv_prior, corr_spx_90d, macro_risk_raw,
                  macro_mult, exposure_scalar, confidence, components)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(date) DO UPDATE SET
                 mvrv=excluded.mvrv, mvrv_prior=excluded.mvrv_prior,
                 corr_spx_90d=excluded.corr_spx_90d,
                 macro_risk_raw=excluded.macro_risk_raw,
                 macro_mult=excluded.macro_mult,
                 exposure_scalar=excluded.exposure_scalar,
                 confidence=excluded.confidence, components=excluded.components,
                 computed_at=datetime('now')""",
            (row["date"], row.get("mvrv"), row["mvrv_prior"], row.get("corr_spx_90d"),
             row.get("macro_risk_raw"), row["macro_mult"], row["exposure_scalar"],
             row.get("confidence"), row.get("components")),
        )
        await db.commit()


async def get_latest_regime_state() -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH, timeout=_DB_TIMEOUT) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM regime_state ORDER BY date DESC LIMIT 1")
        row = await cur.fetchone()
        return dict(row) if row else None


async def get_regime_series(start: str, end: str) -> List[dict]:
    async with aiosqlite.connect(DB_PATH, timeout=_DB_TIMEOUT) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM regime_state WHERE date >= ? AND date <= ? ORDER BY date ASC",
            (start, end),
        )
        return [dict(r) for r in await cur.fetchall()]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_regime_store.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/database.py backend/tests/regime/test_regime_store.py
git commit -m "feat: regime_state table + upsert/get store helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 6: Offline daily-series builder

**Files:**
- Create: `backend/tools/regime/__init__.py` (empty), `backend/tools/regime/build_regime_series.py`
- Test: `backend/tests/regime/test_build_regime_series.py`

**Interfaces:**
- Consumes: `sources.load_aligned` (Task 4), `features.corr_spx_90d`/`macro_risk_raw` (Task 3), `macro_regime.evaluate` (Task 2), `database.upsert_regime_state` (Task 5).
- Produces: `build_series(df: "pd.DataFrame") -> list[RegimeState]` — for each date from index position 90 onward, compute features from the trailing window + evaluate; and `async persist(states: list[RegimeState]) -> int` — upsert each, returns count. A `__main__` CLI wires `load_aligned -> build_series -> persist`.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_build_regime_series.py`:

```python
import os, sys
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import numpy as np
import pandas as pd
import pytest
from tools.regime.build_regime_series import build_series


def _frame(n=200):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    btc = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    spx = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "btc": btc, "spx": spx,
        "dxy": np.linspace(103, 100, n), "real_yield": np.linspace(2.0, 1.5, n),
        "mvrv": np.linspace(0.9, 2.2, n),
    }, index=idx)


def test_build_series_produces_states_after_warmup():
    states = build_series(_frame(200))
    assert len(states) == 200 - 90            # first 90 days are warmup
    s = states[-1]
    assert 0.4 <= s.exposure_scalar <= 1.25
    assert s.date == "2023-07-19"             # 2023-01-01 + 199 days
    assert s.mvrv is not None


def test_build_series_empty_frame():
    assert build_series(pd.DataFrame(columns=["btc", "spx", "dxy", "real_yield", "mvrv"])) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_build_regime_series.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.regime.build_regime_series'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/tools/regime/__init__.py` (empty). Create `backend/tools/regime/build_regime_series.py`:

```python
"""Offline builder: daily series -> per-day RegimeState -> regime_state table.

Run:  python -m tools.regime.build_regime_series [--start 2016-01-01]
Phase-1 offline utility; no live/scan-loop involvement.
"""
from __future__ import annotations

import argparse
import asyncio
import os
from typing import List

import pandas as pd

from services.regime.state import RegimeState, REGIME_STALE_DAYS
from services.regime.features import corr_spx_90d, macro_risk_raw
from services.regime.macro_regime import evaluate

_WARMUP = 90


def build_series(df: "pd.DataFrame") -> List[RegimeState]:
    if df is None or df.empty or len(df) <= _WARMUP:
        return []
    out: List[RegimeState] = []
    btc = df["btc"].tolist(); spx = df["spx"].tolist()
    dxy = df["dxy"].tolist(); ry = df["real_yield"].tolist()
    mvrv = df["mvrv"].tolist(); dates = [d.strftime("%Y-%m-%d") for d in df.index]
    for i in range(_WARMUP, len(df)):
        w = slice(0, i + 1)
        corr = corr_spx_90d(btc[w], spx[w])
        risk = macro_risk_raw(spx[w], dxy[w], ry[w])
        m = mvrv[i]
        out.append(evaluate(
            date=dates[i], mvrv=(None if pd.isna(m) else float(m)),
            corr_spx_90d=corr, macro_risk_raw=risk,
        ))
    return out


async def persist(states: List[RegimeState]) -> int:
    import database
    for st in states:
        await database.upsert_regime_state(st.to_row())
    return len(states)


async def _main(start: str, cache_dir: str) -> None:
    from services.regime import sources
    df = sources.load_aligned(start, cache_dir)
    states = build_series(df)
    n = await persist(states)
    print(f"regime_state upserted: {n} days "
          f"({states[0].date if states else '-'} -> {states[-1].date if states else '-'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__),
                                                        "..", "..", "data", "regime"))
    a = ap.parse_args()
    asyncio.run(_main(a.start, a.cache_dir))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_build_regime_series.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/tools/regime/__init__.py backend/tools/regime/build_regime_series.py \
        backend/tests/regime/test_build_regime_series.py
git commit -m "feat: offline regime-series builder (daily series -> regime_state)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 7: Backtest harness + verdict

**Files:**
- Create: `backend/tools/regime/backtest_regime.py`
- Test: `backend/tests/regime/test_backtest_regime.py`

**Interfaces:**
- Consumes: closed `trades` rows (`pnl`, `usd_open`, `opened_at`), a date→`exposure_scalar` map from `regime_state`.
- Produces:
  - `apply_scaling(trades: list[dict], scalar_by_date: dict[str, float]) -> list[dict]` — adds `scaled_pnl = pnl * scalar` (scalar defaults 1.0 when the trade's `opened_at` date has no regime row); linear-in-size paper model.
  - `metrics(pnls: list[float]) -> dict` — `{total, sharpe, max_drawdown}` over the per-trade PnL series (sharpe = mean/std, 0 if std==0; max_drawdown over the cumulative sum).
  - `compare(trades, scalar_by_date) -> dict` — `{baseline: metrics, scaled: metrics, delta: {...}, by_year: {...}, verdict: str}`. `verdict` = "HELPS" if scaled sharpe ≥ baseline AND scaled max_drawdown ≤ baseline (less negative) AND scaled total ≥ 0.9×baseline; else "NO".
  - A `__main__` CLI loads real trades + regime series from the DB and prints the comparison.

- [ ] **Step 1: Write the failing test**

Create `backend/tests/regime/test_backtest_regime.py`:

```python
import os, sys
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

from tools.regime.backtest_regime import apply_scaling, metrics, compare


def test_apply_scaling_defaults_to_one():
    trades = [{"pnl": 10.0, "usd_open": 100.0, "opened_at": "2025-01-01T00:00:00"}]
    out = apply_scaling(trades, {})            # no regime row -> scalar 1.0
    assert out[0]["scaled_pnl"] == 10.0


def test_apply_scaling_uses_date_scalar():
    trades = [{"pnl": -20.0, "usd_open": 100.0, "opened_at": "2025-11-30T12:00:00"}]
    out = apply_scaling(trades, {"2025-11-30": 0.5})
    assert out[0]["scaled_pnl"] == -10.0       # loss halved by risk-off scaling


def test_metrics_shapes():
    m = metrics([1.0, -2.0, 3.0])
    assert m["total"] == 2.0
    assert "sharpe" in m and "max_drawdown" in m


def test_compare_flags_improvement():
    # Regime halves the big losers, keeps winners -> should HELP.
    trades = [
        {"pnl": 10.0, "usd_open": 100.0, "opened_at": "2025-01-01T00:00:00"},
        {"pnl": -40.0, "usd_open": 100.0, "opened_at": "2025-11-30T00:00:00"},
        {"pnl": 12.0, "usd_open": 100.0, "opened_at": "2025-02-01T00:00:00"},
    ]
    scal = {"2025-11-30": 0.4}                  # risk-off on the loser's day
    res = compare(trades, scal)
    assert res["scaled"]["total"] > res["baseline"]["total"]
    assert res["verdict"] in ("HELPS", "NO")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_backtest_regime.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.regime.backtest_regime'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/tools/regime/backtest_regime.py`:

```python
"""Offline backtest: overlay regime exposure_scalar on the historical trades
record and report whether it improves risk-adjusted outcomes.

Linear paper model: scaled_pnl = pnl * exposure_scalar(opened_at date). This is
the Phase-1 gate — Phase 2 (live wiring) proceeds only if the verdict HELPS and
is robust across sub-periods.

Run:  python -m tools.regime.backtest_regime
"""
from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from typing import Dict, List


def apply_scaling(trades: List[dict], scalar_by_date: Dict[str, float]) -> List[dict]:
    out = []
    for t in trades:
        day = str(t.get("opened_at", ""))[:10]
        scalar = scalar_by_date.get(day, 1.0)
        tt = dict(t)
        tt["scaled_pnl"] = float(t.get("pnl") or 0.0) * scalar
        out.append(tt)
    return out


def metrics(pnls: List[float]) -> dict:
    if not pnls:
        return {"total": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    total = sum(pnls)
    mean = total / len(pnls)
    var = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = math.sqrt(var)
    sharpe = (mean / std) if std > 0 else 0.0
    cum, peak, mdd = 0.0, 0.0, 0.0
    for p in pnls:
        cum += p
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
    return {"total": total, "sharpe": sharpe, "max_drawdown": mdd}


def compare(trades: List[dict], scalar_by_date: Dict[str, float]) -> dict:
    scaled = apply_scaling(trades, scalar_by_date)
    base_m = metrics([float(t.get("pnl") or 0.0) for t in trades])
    scal_m = metrics([t["scaled_pnl"] for t in scaled])

    by_year: Dict[str, dict] = {}
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in scaled:
        buckets[str(t.get("opened_at", ""))[:4]].append(t)
    for yr, ts in sorted(buckets.items()):
        by_year[yr] = {
            "baseline": metrics([float(t.get("pnl") or 0.0) for t in ts]),
            "scaled": metrics([t["scaled_pnl"] for t in ts]),
        }

    helps = (scal_m["sharpe"] >= base_m["sharpe"]
             and scal_m["max_drawdown"] >= base_m["max_drawdown"]   # less negative
             and scal_m["total"] >= 0.9 * base_m["total"])
    return {
        "baseline": base_m, "scaled": scal_m,
        "delta": {k: scal_m[k] - base_m[k] for k in base_m},
        "by_year": by_year, "verdict": "HELPS" if helps else "NO",
    }


async def _main() -> None:
    import database
    trades = await database.get_trades_closed() if hasattr(database, "get_trades_closed") else []
    if not trades:
        import aiosqlite
        async with aiosqlite.connect(database.DB_PATH, timeout=database._DB_TIMEOUT) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT pnl, usd_open, opened_at FROM trades WHERE closed_at IS NOT NULL")
            trades = [dict(r) for r in await cur.fetchall()]
    series = await database.get_regime_series("2000-01-01", "2100-01-01")
    scal = {r["date"]: r["exposure_scalar"] for r in series}
    res = compare(trades, scal)
    print("BASELINE:", res["baseline"])
    print("SCALED:  ", res["scaled"])
    print("DELTA:   ", res["delta"])
    print("BY YEAR: ")
    for yr, m in res["by_year"].items():
        print(f"  {yr}: base_total={m['baseline']['total']:.1f} "
              f"scaled_total={m['scaled']['total']:.1f}")
    print("VERDICT: ", res["verdict"])


if __name__ == "__main__":
    asyncio.run(_main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/test_backtest_regime.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full regime suite + commit**

```bash
cd backend && "C:/Users/gl450/polymarket_app/.venv/Scripts/python.exe" -m pytest tests/regime/ -q
# expect all green
git rev-parse --abbrev-ref HEAD
git add backend/tools/regime/backtest_regime.py backend/tests/regime/test_backtest_regime.py
git commit -m "feat: offline regime backtest harness + go/no-go verdict

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 8: Docs + memory sync

**Files:**
- Modify: `CHANGELOG.md` (new session entry), `CLAUDE.md` (Architecture Quick Reference — note the offline regime layer), memory `coinbase_trader_session_log.md` + `btc_macro_drivers_findings.md` (after commit).

- [ ] **Step 1: CHANGELOG entry** — prepend a Session entry describing the Phase-1 offline macro-regime layer (new `services/regime/*`, `regime_state` table, offline builder + backtest harness, no live wiring, Phase-2 gated on the backtest verdict).

- [ ] **Step 2: CLAUDE.md** — under Architecture Quick Reference, add one line noting `services/regime/` is an **offline** macro-regime evaluator (daily cadence, not yet wired to the scan loop; Phase 2 pending backtest verdict). Do NOT add a "never break" invariant yet — there is no live contract until Phase 2.

- [ ] **Step 3: Run the operator step (offline, manual)** — `python -m tools.regime.build_regime_series --start 2016-01-01` then `python -m tools.regime.backtest_regime`; record the verdict + per-year deltas in the CHANGELOG entry. (This is the Phase-1 gate; if VERDICT=NO, stop and do not plan Phase 2.)

- [ ] **Step 4: Commit docs**

```bash
git rev-parse --abbrev-ref HEAD
git add CHANGELOG.md CLAUDE.md
git commit -m "docs: macro-regime layer Phase 1 (offline) — CHANGELOG + arch note

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

- [ ] **Step 5: Update memory** (after commit): append a Session entry to `coinbase_trader_session_log.md`; update `btc_macro_drivers_findings.md` status with the Phase-1 build + backtest verdict; cross-link `win_factors_improvement_loop.md`.

---

## Self-Review

**Spec coverage:**
- Cascade architecture (slow producer / fast consumer) → the offline half is Tasks 1–7; the fast consumer is Phase 2 (out of scope here, gated). ✓
- Two-phase scope, hard gate → Task 7 verdict + Task 8 Step 3 operator gate. ✓
- Components: `macro_regime.py` T2, `state.py`/`RegimeState` T1, `sources.py` T4, `regime_state` table T5, daily builder T6, backtest harness T7. ✓ (Phase-2 live consumer intentionally deferred.)
- Formula: MVRV prior (asymmetric anchors) T2; correlation-gated macro T2; combine+clamp T2; entry-only + fail-safe + staleness T2 (entry-only application is Phase 2). ✓
- Data sources + cadence + graceful failure → T4. ✓
- Testing (pure evaluator bulk, adapters mocked, store, backtest fixtures) → each task's tests. ✓
- Worked examples (2022 / Nov-2025 / cycle-bottom) reproduced as regression tests → T2. ✓
- Out-of-scope (ETF v1.5, regime-conditional, meta-label feature, no hmm/macro_signals change) → honored; nothing in this plan touches them. ✓

**Placeholder scan:** none — every code/test step shows full content. Task 5 notes the one lookup an implementer must confirm (the DB-init function name) with an explicit grep instruction rather than a guess.

**Type consistency:** `RegimeState` fields identical across T1/T2/T5/T6. `evaluate(*, date, mvrv, corr_spx_90d, macro_risk_raw, mvrv_age_days, macro_age_days)` used identically in T2 and T6. `to_row()`/`from_row()` keys match the `regime_state` columns (T1 ↔ T5). `exposure_scalar` is the single consumed field name throughout. `load_aligned(start, cache_dir, session_get)` signature consistent T4 ↔ T6. `compare/apply_scaling/metrics` signatures consistent T7.
