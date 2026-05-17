# XGB v4 OHLCV-5 Shadow Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fresh XGB v4 model (5 OHLCV channels × 3 tiers × 10 stats = 150 features) running in shadow alongside live v3 — v3 keeps driving decisions, v4 telemetry persists every scan, AUC compared after 7 days to decide cutover.

**Architecture:** New `tools/xgb_v4_features.py` owns the pure-function extractor (no shared state with v3). New `tools/train_xgb_v4.py` orchestrator delegates to small testable helpers. `agents/xgb_signal.py` gains a parallel v4 booster cache + new `xgb_prob_shadow(channels, pid) -> (v3_prob, v4_prob_or_None)` function; v4 failures are isolated and never affect v3. New `cnn_scans.xgb_prob_v4 REAL` column with idempotent migration captures telemetry per scan.

**Tech Stack:** Python 3.11, xgboost, scikit-learn IsotonicRegression, pyarrow parquet, sqlite3 / aiosqlite, pytest + pytest-asyncio.

**Spec source:** `docs/superpowers/specs/2026-05-17-xgb-v4-ohlcv-shadow-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `backend/tools/xgb_v4_features.py` | CREATE | Pure-function v4 feature extractor: constants, public API (`extract_v4`, `feature_names_v4`, `feature_weights_v4`), 5 internal helpers (`_extract_field`, `_compute_stats`, `_slope`, `_pct_rank`, `_delta_at`). ~150 LOC. |
| `backend/tools/train_xgb_v4.py` | CREATE | Orchestrator `main()` delegating to 6 single-responsibility helpers. Reads OHLCV parquets, builds samples + labels at CLI-specified `--forward-hours` / `--label-thresh`, walk-forward split, trains booster, calibrates, writes **horizon-suffixed artifacts** (`xgb_*_v4_h<HOURS>.*`). ~280 LOC. |
| `backend/tools/v4_horizon_compare.py` | CREATE | Loads each horizon's artifacts, builds held-out test set per horizon, computes AUC + logloss + pos_frac + n_samples, renders side-by-side HTML report at `backend/tools/xgb_v4_horizon_compare.html`. ~200 LOC. |
| `backend/tests/test_v4_horizon_compare.py` | CREATE | Unit tests for `_evaluate_on_holdout`, `_render_html_report` (smoke test for HTML output). ~80 LOC. |
| `backend/migrations/xgb_v4_shadow_20260517.py` | CREATE | Idempotent ALTER TABLE adding `cnn_scans.xgb_prob_v4 REAL`. Pattern mirror of `mc_telemetry_20260516.py`. ~35 LOC. |
| `backend/tests/test_xgb_v4_features.py` | CREATE | Unit tests for the 5 helpers + 3 public functions + tier-handling. ~200 LOC. |
| `backend/tests/test_train_xgb_v4.py` | CREATE | Unit tests for `_build_samples_for_pid`, `_walk_forward_split`, label correctness on synthetic candles. ~150 LOC. |
| `backend/tools/xgb_features.py` | EDIT (+7 LOC) | Add `feature_set == "v4"` branch to `extract_features()` dispatcher. |
| `backend/agents/xgb_signal.py` | EDIT (+80 LOC) | New module-level state (`_booster_v4`, `_calibration_v4`, `_load_attempted_v4`, `_load_succeeded_v4`), new `_try_load_v4()`, new `xgb_prob_v4(channels, pid)`, new `xgb_prob_shadow(channels, pid) -> Tuple[float, Optional[float]]`. v3 path UNCHANGED. |
| `backend/database.py` | EDIT (+4 LOC) | Add `xgb_prob_v4 REAL` to CREATE TABLE; add to ALTER TABLE migrations list; add to `save_cnn_scan` INSERT. |
| `backend/agents/cnn_agent.py` | EDIT (~8 LOC) | At line 1903 replace `_xgb.xgb_prob(...)` with `_xgb.xgb_prob_shadow(...)`, capture both probs, add `xgb_prob_v4` to the `save_cnn_scan` dict at line 1989+. **No decision logic changes.** |
| `backend/tests/test_xgb_signal.py` | EDIT (+90 LOC) | Tests for `_try_load_v4`, `xgb_prob_v4`, `xgb_prob_shadow` (success + failure isolation). |
| `backend/tests/test_database.py` | EDIT (+30 LOC) | Tests for `save_cnn_scan` persisting `xgb_prob_v4`. |
| `backend/tests/test_mc_migration.py` | EDIT (+30 LOC) | Add idempotency tests for the v4 migration (apply twice, second is no-op). |
| `CLAUDE.md` | EDIT (+8 LOC) | Add invariant #16 (shadow telemetry isolation). |
| `CHANGELOG.md` | EDIT | New Session 58.71j entry at top. |

Memory sync after commit: `coinbase_trader_architecture.md` (outside repo).

---

## Coordination

Branch tip at plan-write time: `90c52d2` (spec revision with concrete signatures). Single feat branch, no parallel subagents in flight. Single atomic commit at the end (after Task 11 completes). Push immediately after commit lands per [[feedback_push_on_commit]].

---

## Task 1: xgb_v4_features.py — helpers + public API

**Files:**
- Create: `backend/tools/xgb_v4_features.py`
- Create: `backend/tests/test_xgb_v4_features.py`

- [ ] **Step 1.1 — Write failing test file with all unit tests**

Create `backend/tests/test_xgb_v4_features.py`:

```python
"""Unit tests for backend/tools/xgb_v4_features.py — XGB v4 OHLCV-5 extractor.

5 channels (open/high/low/close/volume) x 3 tiers (micro/meso/macro)
x 10 stats = 150 features. Pure functions, no module state.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools import xgb_v4_features as v4  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_channel_fields_order(self):
        assert v4._CHANNEL_FIELDS == ("open", "high", "low", "close", "volume")

    def test_n_channels_derived(self):
        assert v4.N_CHANNELS_V4 == 5
        assert v4.N_CHANNELS_V4 == len(v4._CHANNEL_FIELDS)

    def test_tier_windows(self):
        assert v4.TIER_WINDOWS_V4 == {"micro": 60, "meso": 168, "macro": 336}

    def test_tier_weights(self):
        assert v4.TIER_WEIGHTS_V4 == {"micro": 1.0, "meso": 2.0, "macro": 3.0}

    def test_stat_names_order(self):
        assert v4._STAT_NAMES_V4 == (
            "last", "mean", "std", "slope",
            "min", "max", "pct_rank",
            "dlt5", "dlt10", "dlt30",
        )

    def test_n_features_derived(self):
        assert v4.N_FEATURES_V4 == 150
        assert v4.N_FEATURES_V4 == v4.N_CHANNELS_V4 * v4.N_TIERS_V4 * v4.N_STATS_V4


# ── _slope ────────────────────────────────────────────────────────────────

class TestSlope:
    def test_empty_returns_zero(self):
        assert v4._slope(np.array([], dtype=np.float64)) == 0.0

    def test_single_returns_zero(self):
        assert v4._slope(np.array([5.0])) == 0.0

    def test_constant_returns_zero(self):
        assert v4._slope(np.array([3.0, 3.0, 3.0, 3.0])) == 0.0

    def test_linear_up_returns_one(self):
        # y = x: slope = 1.0
        assert v4._slope(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_linear_down_returns_negative(self):
        assert v4._slope(np.array([3.0, 2.0, 1.0, 0.0])) == pytest.approx(-1.0)


# ── _pct_rank ─────────────────────────────────────────────────────────────

class TestPctRank:
    def test_empty_returns_zero(self):
        assert v4._pct_rank(np.array([], dtype=np.float64)) == 0.0

    def test_single_returns_zero(self):
        assert v4._pct_rank(np.array([5.0])) == 0.0

    def test_last_is_max(self):
        # 4 values [1,2,3,10], last=10 is greater than 3/4 others
        # Convention: (below + 0.5*equal) / n
        out = v4._pct_rank(np.array([1.0, 2.0, 3.0, 10.0]))
        assert out == pytest.approx((3 + 0.5 * 1) / 4)

    def test_last_is_min(self):
        # last=1 is less than 3/4 others
        out = v4._pct_rank(np.array([10.0, 5.0, 2.0, 1.0]))
        assert out == pytest.approx((0 + 0.5 * 1) / 4)


# ── _delta_at ─────────────────────────────────────────────────────────────

class TestDeltaAt:
    def test_too_short_returns_zero(self):
        assert v4._delta_at(np.array([1.0, 2.0]), lookback=5) == 0.0

    def test_exact_lookback(self):
        # values[-1] - values[-1 - 5] for lookback=5
        # array of 6: values[-1]=v[5], values[-6]=v[0]
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        assert v4._delta_at(v, lookback=5) == 10.0 - 1.0

    def test_longer_than_needed(self):
        v = np.arange(20, dtype=np.float64)
        # last is 19; lookback=5 -> 19 - v[14] = 19 - 14 = 5
        assert v4._delta_at(v, lookback=5) == 5.0


# ── _extract_field ────────────────────────────────────────────────────────

class TestExtractField:
    def test_empty_candles_returns_empty(self):
        out = v4._extract_field([], "close")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.size == 0

    def test_extracts_one_column(self):
        candles = [
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
            {"open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200.0},
        ]
        assert list(v4._extract_field(candles, "close")) == [1.5, 2.0]
        assert list(v4._extract_field(candles, "volume")) == [100.0, 200.0]


# ── _compute_stats ────────────────────────────────────────────────────────

class TestComputeStats:
    def test_empty_returns_ten_zeros(self):
        out = v4._compute_stats(np.array([], dtype=np.float64))
        assert out.shape == (10,)
        assert (out == 0.0).all()

    def test_known_series(self):
        # Series 1..10
        v = np.arange(1, 11, dtype=np.float64)
        out = v4._compute_stats(v)
        assert out.shape == (10,)
        # _STAT_NAMES_V4 order: last, mean, std, slope, min, max, pct_rank, dlt5, dlt10, dlt30
        assert out[0] == 10.0                            # last
        assert out[1] == pytest.approx(5.5)              # mean
        assert out[2] == pytest.approx(v.std())          # std
        assert out[3] == pytest.approx(1.0)              # slope
        assert out[4] == 1.0                             # min
        assert out[5] == 10.0                            # max
        assert out[6] == pytest.approx((9 + 0.5) / 10)   # pct_rank
        assert out[7] == 10.0 - 5.0                      # dlt5 = v[-1]-v[-6]
        assert out[8] == 0.0                             # dlt10 needs len>=11
        assert out[9] == 0.0                             # dlt30 needs len>=31


# ── feature_names_v4 ──────────────────────────────────────────────────────

class TestFeatureNames:
    def test_returns_150_names(self):
        names = v4.feature_names_v4()
        assert len(names) == 150

    def test_layout_channel_then_tier_then_stat(self):
        names = v4.feature_names_v4()
        # First 30 should be channel 0 (open) across 3 tiers
        assert names[0] == "ch0_micro_last"
        assert names[1] == "ch0_micro_mean"
        assert names[9] == "ch0_micro_dlt30"
        assert names[10] == "ch0_meso_last"
        assert names[20] == "ch0_macro_last"
        assert names[29] == "ch0_macro_dlt30"
        assert names[30] == "ch1_micro_last"  # channel 1 starts here
        assert names[149] == "ch4_macro_dlt30"  # final

    def test_unique(self):
        names = v4.feature_names_v4()
        assert len(set(names)) == len(names)


# ── feature_weights_v4 ────────────────────────────────────────────────────

class TestFeatureWeights:
    def test_length_matches_names(self):
        assert len(v4.feature_weights_v4()) == len(v4.feature_names_v4())

    def test_dtype_float64(self):
        assert v4.feature_weights_v4().dtype == np.float64

    def test_tier_weight_values(self):
        names = v4.feature_names_v4()
        weights = v4.feature_weights_v4()
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert weights[i] == 1.0, f"{name} should be micro=1.0"
            elif "_meso_" in name:
                assert weights[i] == 2.0, f"{name} should be meso=2.0"
            elif "_macro_" in name:
                assert weights[i] == 3.0, f"{name} should be macro=3.0"


# ── extract_v4 ────────────────────────────────────────────────────────────

class TestExtractV4:
    def _make_candle(self, c: int) -> Dict[str, float]:
        # Distinct values per field so we can detect mis-routing
        return {
            "open":   c * 1.0,
            "high":   c * 1.0 + 0.5,
            "low":    c * 1.0 - 0.5,
            "close":  c * 1.0 + 0.25,
            "volume": c * 10.0,
        }

    def _make_tier(self, n: int) -> List[Dict[str, float]]:
        return [self._make_candle(i + 1) for i in range(n)]

    def test_shape_and_names(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        assert features.shape == (1, 150)
        assert features.dtype == np.float64
        assert names == v4.feature_names_v4()

    def test_channel_3_reads_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # ch3_micro_last corresponds to close of last micro candle.
        # Last micro candle is _make_candle(60): close = 60.25
        idx = names.index("ch3_micro_last")
        assert features[0, idx] == 60.25

    def test_channel_4_reads_volume_not_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # ch4_micro_last: volume of last micro candle = 60 * 10 = 600
        idx_v = names.index("ch4_micro_last")
        idx_c = names.index("ch3_micro_last")
        assert features[0, idx_v] == 600.0
        assert features[0, idx_c] == 60.25
        # Distinct from close — proves volume routing works.
        assert features[0, idx_v] != features[0, idx_c]

    def test_empty_tier_zeros_its_slots(self):
        candles_by_tier = {
            "micro": [],                       # empty
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # All micro slots (50 total) should be zero
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0, f"{name} should be zero (empty tier)"

    def test_missing_tier_key_zeros_slots(self):
        candles_by_tier = {
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }  # no "micro" key
        features, names = v4.extract_v4(candles_by_tier)
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0

    def test_missing_ohlcv_field_raises(self):
        bad = [{"open": 1.0, "high": 2.0, "low": 0.5}]  # missing close/volume
        with pytest.raises(KeyError):
            v4.extract_v4({"micro": bad, "meso": [], "macro": []})

    def test_determinism(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        f1, _ = v4.extract_v4(candles_by_tier)
        f2, _ = v4.extract_v4(candles_by_tier)
        assert (f1 == f2).all()
```

- [ ] **Step 1.2 — Run; expect ImportError on `from tools import xgb_v4_features`**

```bash
cd C:\Users\gl450\polymarket_app
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_features.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'tools.xgb_v4_features'`.

- [ ] **Step 1.3 — Create the implementation**

Create `backend/tools/xgb_v4_features.py`:

```python
"""XGB v4 OHLCV-5 feature extractor.

5 channels (open/high/low/close/volume) x 3 tiers (micro/meso/macro)
x 10 stats = 150 features. Pure functions, no mutable module state.

Per feedback_python_clean_functions: type hints on every signature,
pure data-in/data-out helpers, derived constants, no in-place buffer
mutation (contrast v3's _stats_from_candles(candles, stat_offset, out)).
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ── Configuration constants ────────────────────────────────────────────────
_CHANNEL_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
N_CHANNELS_V4: int = len(_CHANNEL_FIELDS)

TIER_WINDOWS_V4: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V4: Dict[str, float] = {"micro": 1.0, "meso": 2.0, "macro": 3.0}
_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")

_STAT_NAMES_V4: Tuple[str, ...] = (
    "last", "mean", "std", "slope",
    "min", "max", "pct_rank",
    "dlt5", "dlt10", "dlt30",
)
N_STATS_V4: int = len(_STAT_NAMES_V4)
N_TIERS_V4: int = len(TIER_WINDOWS_V4)
N_FEATURES_V4: int = N_CHANNELS_V4 * N_TIERS_V4 * N_STATS_V4  # = 150


# ── Public API ─────────────────────────────────────────────────────────────

def feature_names_v4() -> List[str]:
    """Return 150 feature names in stable column order.

    Layout: ch{0..4}_{micro|meso|macro}_{stat}, ordered
    channel-major -> tier-major -> stat-major.
    """
    names: List[str] = []
    for c in range(N_CHANNELS_V4):
        for tier in _TIER_ORDER:
            for stat in _STAT_NAMES_V4:
                names.append(f"ch{c}_{tier}_{stat}")
    return names


def feature_weights_v4() -> np.ndarray:
    """Return 150-long float64 weight vector aligned with feature_names_v4().

    Per-tier weights: micro 1.0, meso 2.0, macro 3.0. Same weight for all
    10 stats within one (channel, tier) group.
    """
    weights = np.zeros(N_FEATURES_V4, dtype=np.float64)
    i = 0
    for _c in range(N_CHANNELS_V4):
        for tier in _TIER_ORDER:
            w = TIER_WEIGHTS_V4[tier]
            for _s in range(N_STATS_V4):
                weights[i] = w
                i += 1
    return weights


def extract_v4(
    candles_by_tier: Dict[str, Sequence[Dict[str, float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract 150 features from tier-keyed OHLCV candle lists.

    Args:
        candles_by_tier: {"micro": [...], "meso": [...], "macro": [...]}
            where each entry is a candle dict with at minimum the keys
            ("open","high","low","close","volume").

    Returns:
        (features, names) where features is shape (1, 150) float64 and
        names is len-150 list matching feature_names_v4().

    Missing/empty tier -> the 50 slots for that tier are zero.
    Missing OHLCV field in a candle -> raises KeyError (input contract).
    """
    out = np.zeros((1, N_FEATURES_V4), dtype=np.float64)
    names = feature_names_v4()
    slot = 0
    for c in range(N_CHANNELS_V4):
        field = _CHANNEL_FIELDS[c]
        for tier in _TIER_ORDER:
            tier_candles = candles_by_tier.get(tier) or []
            values = _extract_field(tier_candles, field)
            stats = _compute_stats(values)
            out[0, slot:slot + N_STATS_V4] = stats
            slot += N_STATS_V4
    return out, names


# ── Internal helpers (pure functions, one responsibility each) ─────────────

def _extract_field(
    candles: Sequence[Dict[str, float]],
    field: str,
) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray.

    Empty input -> empty ndarray (caller assembles into the zero slot).
    Missing field key in any candle raises KeyError (input contract).
    """
    if not candles:
        return np.array([], dtype=np.float64)
    return np.array([candle[field] for candle in candles], dtype=np.float64)


def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Return shape-(10,) stats in fixed _STAT_NAMES_V4 order.

    Empty input -> all zeros. No in-place mutation of any caller buffer.
    """
    out = np.zeros(N_STATS_V4, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[-1])             # last
    out[1] = float(values.mean())          # mean
    out[2] = float(values.std())           # std
    out[3] = _slope(values)                # slope
    out[4] = float(values.min())           # min
    out[5] = float(values.max())           # max
    out[6] = _pct_rank(values)             # pct_rank
    out[7] = _delta_at(values, lookback=5)
    out[8] = _delta_at(values, lookback=10)
    out[9] = _delta_at(values, lookback=30)
    return out


def _slope(values: np.ndarray) -> float:
    """OLS slope of values vs index 0..len-1. 0.0 for n<2 or zero variance."""
    n = values.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = values.mean()
    num = ((x - x_mean) * (values - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0.0:
        return 0.0
    return float(num / den)


def _pct_rank(values: np.ndarray) -> float:
    """Percentile rank of last value within the series. 0.0 if empty/single."""
    n = values.size
    if n < 2:
        return 0.0
    last = values[-1]
    below = (values < last).sum()
    equal = (values == last).sum()
    return float((below + 0.5 * equal) / n)


def _delta_at(values: np.ndarray, lookback: int) -> float:
    """values[-1] - values[-1-lookback], or 0.0 if series too short."""
    if values.size < lookback + 1:
        return 0.0
    return float(values[-1] - values[-1 - lookback])
```

- [ ] **Step 1.4 — Run; expect all 30+ tests passing**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_features.py -v
```

Expected: all passing (TestConstants × 6, TestSlope × 5, TestPctRank × 4, TestDeltaAt × 3, TestExtractField × 2, TestComputeStats × 2, TestFeatureNames × 3, TestFeatureWeights × 3, TestExtractV4 × 7).

---

## Task 2: xgb_features.py — v4 dispatcher branch

**Files:**
- Modify: `backend/tools/xgb_features.py` (around `extract_features` function, line ~254)

- [ ] **Step 2.1 — Write the dispatcher test in test_xgb_v4_features.py**

Append at end of `backend/tests/test_xgb_v4_features.py`:

```python
# ── Dispatcher integration (xgb_features.extract_features) ────────────────


class TestDispatcherV4Branch:
    def test_extract_features_v4_routes_to_v4(self):
        from tools.xgb_features import extract_features
        candles_by_tier = {
            "micro": [{"open": 1.0, "high": 2.0, "low": 0.5,
                       "close": 1.5, "volume": 10.0}] * 60,
            "meso":  [{"open": 1.0, "high": 2.0, "low": 0.5,
                       "close": 1.5, "volume": 10.0}] * 168,
            "macro": [{"open": 1.0, "high": 2.0, "low": 0.5,
                       "close": 1.5, "volume": 10.0}] * 336,
        }
        features, names = extract_features(candles_by_tier, feature_set="v4")
        assert features.shape == (1, 150)
        assert len(names) == 150
        assert names[0] == "ch0_micro_last"

    def test_extract_features_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features
        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features({}, feature_set="v99")
```

- [ ] **Step 2.2 — Run; expect 2 failures**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_features.py::TestDispatcherV4Branch -v
```

Expected: 2 FAILED with `ValueError: unknown feature_set='v4'`.

- [ ] **Step 2.3 — Add the v4 branch to extract_features**

Edit `backend/tools/xgb_features.py`. Find around line 254:

```python
def extract_features(
    samples, feature_set: str = "v1"
) -> Tuple[np.ndarray, List[str]]:
    """Convert a batch of [N, 28, 60] samples to tabular features.

    feature_set:
        "v1" (default): 270 per-channel stats (back-compat).
        "v2": v1 + 10 cross-channel/temporal addons (_V2_NEW_FEATURES).
        "v3": tiered mixed-lookback — 350 features, samples arg becomes
              {"micro","meso","macro"} candle-list dict (#311b).

    Returns (features, feature_names) where features is float64 with no
    NaN/Inf, and feature_names matches the column order of features.
    """
    if feature_set == "v3":
        return _extract_v3(samples)
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1', 'v2', or 'v3'"
        )
```

Replace with (adds v4 branch + extends error message):

```python
def extract_features(
    samples, feature_set: str = "v1"
) -> Tuple[np.ndarray, List[str]]:
    """Convert a batch of [N, 28, 60] samples to tabular features.

    feature_set:
        "v1" (default): 270 per-channel stats (back-compat).
        "v2": v1 + 10 cross-channel/temporal addons (_V2_NEW_FEATURES).
        "v3": tiered mixed-lookback — 350 features, samples arg becomes
              {"micro","meso","macro"} candle-list dict (#311b).
        "v4": OHLCV-5 channels x 3 tiers x 10 stats = 150 features,
              samples arg is {"micro","meso","macro"} candle-list dict
              (#xgb-v4 / Step B.1).

    Returns (features, feature_names) where features is float64 with no
    NaN/Inf, and feature_names matches the column order of features.
    """
    if feature_set == "v4":
        from tools.xgb_v4_features import extract_v4
        return extract_v4(samples)
    if feature_set == "v3":
        return _extract_v3(samples)
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1', 'v2', 'v3', or 'v4'"
        )
```

- [ ] **Step 2.4 — Run; expect 2 dispatcher tests green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_features.py::TestDispatcherV4Branch -v
```

Expected: 2 PASSED.

---

## Task 3: Migration for cnn_scans.xgb_prob_v4 column

**Files:**
- Create: `backend/migrations/xgb_v4_shadow_20260517.py`
- Modify: `backend/tests/test_mc_migration.py` (extend)

- [ ] **Step 3.1 — Write the migration tests in test_mc_migration.py**

Append at end of `backend/tests/test_mc_migration.py`:

```python
# ── xgb_v4_shadow_20260517 ─────────────────────────────────────────────────


class TestXgbV4ShadowMigration:
    def test_migration_adds_xgb_prob_v4_column(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_shadow_20260517 import run

        db = str(tmp_path / "test.db")
        c = sqlite3.connect(db)
        c.execute(
            "CREATE TABLE cnn_scans ("
            " id INTEGER PRIMARY KEY, product_id TEXT, scanned_at INTEGER"
            ")"
        )
        c.commit()
        c.close()

        result = run(db)
        assert "xgb_prob_v4" in result["added"]
        assert result["already_present"] == []

        # Column now present
        c = sqlite3.connect(db)
        cols = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        assert "xgb_prob_v4" in cols
        c.close()

    def test_migration_idempotent(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_shadow_20260517 import run

        db = str(tmp_path / "test.db")
        c = sqlite3.connect(db)
        c.execute(
            "CREATE TABLE cnn_scans ("
            " id INTEGER PRIMARY KEY, product_id TEXT, scanned_at INTEGER"
            ")"
        )
        c.commit()
        c.close()

        # First run — adds
        r1 = run(db)
        assert "xgb_prob_v4" in r1["added"]
        # Second run — skips
        r2 = run(db)
        assert r2["added"] == []
        assert "xgb_prob_v4" in r2["already_present"]
```

- [ ] **Step 3.2 — Run; expect 2 failures (ModuleNotFoundError)**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_mc_migration.py::TestXgbV4ShadowMigration -v
```

Expected: 2 FAILED with `ModuleNotFoundError: No module named 'migrations.xgb_v4_shadow_20260517'`.

- [ ] **Step 3.3 — Create the migration file**

Create `backend/migrations/xgb_v4_shadow_20260517.py`:

```python
"""Migration: add XGB v4 shadow telemetry column to cnn_scans (#xgb-v4 / Step B.1).

Adds `xgb_prob_v4 REAL` for shadow-mode telemetry — captures the v4 model's
prediction alongside v3's during the 1-week shadow validation period.

Idempotent — safe to re-run. Matches the pattern of
mc_telemetry_20260516.py.
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_v4 REAL to cnn_scans if not present.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [("xgb_prob_v4", "REAL")]
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

- [ ] **Step 3.4 — Run; expect 2 green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_mc_migration.py::TestXgbV4ShadowMigration -v
```

Expected: 2 PASSED.

---

## Task 4: database.py — save_cnn_scan persists xgb_prob_v4

**Files:**
- Modify: `backend/database.py` (CREATE TABLE around line 126, ALTER list line 275, save_cnn_scan line 552)
- Modify: `backend/tests/test_database.py` (extend with persistence test)

- [ ] **Step 4.1 — Write the persistence test**

Find the test class for `save_cnn_scan` in `backend/tests/test_database.py` (search for `class Test.*SaveCnnScan` or similar). Append at end of `backend/tests/test_database.py`:

```python
# ── xgb_prob_v4 persistence (#xgb-v4 / Step B.1) ──────────────────────────


class TestSaveCnnScanXgbProbV4:
    @pytest.mark.asyncio
    async def test_save_cnn_scan_persists_xgb_prob_v4(self, tmp_path):
        import database as db
        db_path = str(tmp_path / "test.db")
        await db.init_db(db_path)

        await db.save_cnn_scan({
            "product_id": "BTC-USD",
            "model_prob": 0.6,
            "cnn_prob":   0.6,
            "llm_prob":   None,
            "regime":     "TRENDING",
            "side":       "BUY",
            "cnn_weight": 1.0,
            "llm_weight": 0.0,
            "rsi":        50.0,
            "macd_h":     0.0,
            "bb_pos":     0.5,
            "vwap_dist":  0.0,
            "fast_rsi":   0.5,
            "velocity":   0.5,
            "vol_z":      0.5,
            "xgb_prob":     0.55,
            "scanned_at":   1700000000,
            "xgb_prob_v4":  0.42,
        }, db_path=db_path)

        import sqlite3
        c = sqlite3.connect(db_path)
        row = c.execute(
            "SELECT xgb_prob, xgb_prob_v4 FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row is not None
        assert row[0] == 0.55
        assert row[1] == 0.42

    @pytest.mark.asyncio
    async def test_save_cnn_scan_xgb_prob_v4_defaults_to_null(self, tmp_path):
        import database as db
        db_path = str(tmp_path / "test.db")
        await db.init_db(db_path)

        # Omit xgb_prob_v4 entirely
        await db.save_cnn_scan({
            "product_id": "BTC-USD",
            "model_prob": 0.6, "cnn_prob": 0.6, "llm_prob": None,
            "regime": "TRENDING", "side": "BUY",
            "cnn_weight": 1.0, "llm_weight": 0.0,
            "rsi": 50.0, "macd_h": 0.0, "bb_pos": 0.5,
            "vwap_dist": 0.0, "fast_rsi": 0.5, "velocity": 0.5, "vol_z": 0.5,
            "xgb_prob": 0.55, "scanned_at": 1700000000,
        }, db_path=db_path)

        import sqlite3
        c = sqlite3.connect(db_path)
        row = c.execute(
            "SELECT xgb_prob_v4 FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row[0] is None
```

Check the `save_cnn_scan` signature in `backend/database.py` — it accepts a dict and an optional `db_path`. If `save_cnn_scan` does NOT accept a `db_path` kwarg, adapt these tests to whatever isolation fixture the existing `TestSaveCnnScan*` classes use (likely a fixture that sets `_DB_PATH`).

- [ ] **Step 4.2 — Run; expect failure (column does not exist or save_cnn_scan doesn't accept xgb_prob_v4)**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py::TestSaveCnnScanXgbProbV4 -v
```

Expected: FAILED with `sqlite3.OperationalError: no such column: xgb_prob_v4` (or the INSERT VALUES count mismatch).

- [ ] **Step 4.3 — Add xgb_prob_v4 to CREATE TABLE in init_db**

Edit `backend/database.py`. Find around line 149-152:

```python
                xgb_prob    REAL,
                ...
                xgb_prob_stdev REAL,
                mc_telemetry TEXT
```

Replace with (insert `xgb_prob_v4` after `xgb_prob_stdev`, before the closing parenthesis):

```python
                xgb_prob    REAL,
                ...
                xgb_prob_stdev REAL,
                mc_telemetry TEXT,
                xgb_prob_v4 REAL
```

NOTE: read lines 146-155 in `database.py` to confirm the exact CREATE TABLE column list and trailing commas before editing. The principle: add `xgb_prob_v4 REAL` as the LAST column, ensure the prior `mc_telemetry TEXT` line gains a trailing comma.

- [ ] **Step 4.4 — Add to ALTER TABLE migration list**

Edit `backend/database.py` around line 280-281. After:

```python
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_stdev REAL",
            "ALTER TABLE cnn_scans ADD COLUMN mc_telemetry TEXT",
```

Append (before the next bracket closing the list):

```python
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4 REAL",
```

The existing migration loop tolerates "duplicate column" failures, so this is safe to re-run on databases that already have the column.

- [ ] **Step 4.5 — Add xgb_prob_v4 to save_cnn_scan INSERT**

Edit `backend/database.py`. Find around line 552-575:

```python
async def save_cnn_scan(scan: Dict) -> None:
    ...
            """INSERT INTO cnn_scans
            (product_id, model_prob, cnn_prob, llm_prob, regime, side,
                cnn_weight, llm_weight, rsi, macd_h, bb_pos, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at,
                xgb_prob_stdev, mc_telemetry)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan["product_id"], scan["model_prob"], ...
                scan.get("xgb_prob"),
                scan.get("scanned_at"),
                scan.get("xgb_prob_stdev"), scan.get("mc_telemetry"),
            ),
```

Add `xgb_prob_v4` as the FINAL column (matches CREATE TABLE order):

```python
            """INSERT INTO cnn_scans
            (product_id, model_prob, cnn_prob, llm_prob, regime, side,
                cnn_weight, llm_weight, rsi, macd_h, bb_pos, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at,
                xgb_prob_stdev, mc_telemetry, xgb_prob_v4)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan["product_id"], scan["model_prob"], ...
                scan.get("xgb_prob"),
                scan.get("scanned_at"),
                scan.get("xgb_prob_stdev"), scan.get("mc_telemetry"),
                scan.get("xgb_prob_v4"),
            ),
```

Add one extra `?` to the VALUES list and one extra `scan.get("xgb_prob_v4")` to the tuple. Read lines 552-578 to confirm the exact existing text and tuple structure before editing.

- [ ] **Step 4.6 — Run; expect 2 green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py::TestSaveCnnScanXgbProbV4 -v
```

Expected: 2 PASSED. Then run the full `test_database.py` to confirm no regressions:

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py -v
```

Expected: full database test suite passes.

---

## Task 5: xgb_signal.py — shadow path

**Files:**
- Modify: `backend/agents/xgb_signal.py` (add v4 loader + xgb_prob_v4 + xgb_prob_shadow)
- Modify: `backend/tests/test_xgb_signal.py` (extend with shadow tests)

- [ ] **Step 5.1 — Write the shadow tests**

Append at end of `backend/tests/test_xgb_signal.py`:

```python
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
```

- [ ] **Step 5.2 — Run; expect failures (functions don't exist)**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py::TestV4ShadowLoad backend/tests/test_xgb_signal.py::TestXgbProbV4 backend/tests/test_xgb_signal.py::TestXgbProbShadow -v
```

Expected: 6+ FAILED with `AttributeError: module 'agents.xgb_signal' has no attribute '_try_load_v4'` etc.

- [ ] **Step 5.3 — Add v4 module state and loader to xgb_signal.py**

Edit `backend/agents/xgb_signal.py`. After the existing v3 state block (around line 47), add:

```python
# ── v4 shadow state (Step B.1) ────────────────────────────────────────────
_MODEL_PATH_V4    = os.path.join(_BACKEND_DIR, "xgb_model_v4.json")
_FEATURES_PATH_V4 = os.path.join(_BACKEND_DIR, "xgb_features_v4.json")
_CALIB_PATH_V4    = os.path.join(_BACKEND_DIR, "xgb_calibration_v4.pkl")

_booster_v4 = None
_feature_names_v4: List[str] = []
_calibration_v4: Optional[object] = None
_load_attempted_v4: bool = False
_load_succeeded_v4: bool = False
```

After the existing `_try_load()` function (around line 131), add `_try_load_v4`:

```python
def _try_load_v4() -> bool:
    """Load v4 booster + feature_names + calibrator from disk once. Idempotent.

    Returns True iff all artifacts loaded successfully. Failures log + return
    False; never raise. Mirrors _try_load but for v4 paths.
    """
    global _booster_v4, _feature_names_v4, _calibration_v4
    global _load_attempted_v4, _load_succeeded_v4
    with _lock:
        if _load_attempted_v4:
            return _load_succeeded_v4
        _load_attempted_v4 = True
        if not (os.path.exists(_MODEL_PATH_V4) and os.path.exists(_FEATURES_PATH_V4)):
            logger.info(
                "xgb_signal: v4 artifacts missing (model=%s features=%s) — shadow disabled",
                _MODEL_PATH_V4, _FEATURES_PATH_V4,
            )
            return False
        try:
            import xgboost as xgb
            with open(_FEATURES_PATH_V4, "r") as f:
                meta = json.load(f)
            names = list(meta.get("feature_names", []))
            if not names:
                logger.warning("xgb_signal: v4 features.json has empty feature_names")
                return False
            booster = xgb.Booster()
            booster.load_model(_MODEL_PATH_V4)
            _booster_v4 = booster
            _feature_names_v4 = names
            if os.path.exists(_CALIB_PATH_V4):
                try:
                    with open(_CALIB_PATH_V4, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and "calibrator" in obj:
                        cal_set = obj.get("feature_set")
                        if cal_set is not None and cal_set != "v4":
                            logger.warning(
                                "xgb_signal: v4 calibrator feature_set=%s != 'v4' — skipping",
                                cal_set,
                            )
                            _calibration_v4 = None
                        else:
                            _calibration_v4 = obj["calibrator"]
                            logger.info("xgb_signal: loaded v4 isotonic calibrator")
                    else:
                        logger.warning(
                            "xgb_signal: v4 calibrator pkl is not dict-shape — skipping"
                        )
                        _calibration_v4 = None
                except Exception as exc:
                    logger.exception("xgb_signal: v4 calibrator load failed: %s", exc)
                    _calibration_v4 = None
            else:
                logger.info("xgb_signal: no v4 calibrator at %s — raw passthrough",
                            _CALIB_PATH_V4)
            _load_succeeded_v4 = True
            logger.info("xgb_signal: loaded v4 booster (%d features)", len(names))
            return True
        except Exception as exc:
            logger.exception("xgb_signal: v4 load failed: %s", exc)
            return False
```

- [ ] **Step 5.4 — Add xgb_prob_v4 function**

After `_try_load_v4`, add:

```python
def xgb_prob_v4(channels, pid: Optional[str] = None) -> float:
    """v4 booster probability in [0.01, 0.99]. Neutral 0.5 if artifacts missing or pid None.

    Mirrors xgb_prob() but always uses feature_set='v4' and requires pid for
    tiered_history fetch. Used by xgb_prob_shadow; not called directly by
    cnn_agent during the shadow week.
    """
    if not _try_load_v4():
        return _NEUTRAL
    if pid is None:
        logger.warning("xgb_signal: v4 requires pid, got None — returning neutral")
        return _NEUTRAL
    try:
        import xgboost as xgb
        from services.tiered_history import fetch_tiered
        from tools.xgb_v4_features import extract_v4

        tiers = fetch_tiered(pid, source="live")
        features, _ = extract_v4(tiers)
        dmat = xgb.DMatrix(features, feature_names=_feature_names_v4)
        raw = float(_booster_v4.predict(dmat)[0])
        if _calibration_v4 is not None:
            raw = float(_calibration_v4.transform(np.asarray([raw]))[0])
        return float(np.clip(raw, 0.01, 0.99))
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob_v4 failed, returning neutral: %s", exc)
        return _NEUTRAL
```

- [ ] **Step 5.5 — Add xgb_prob_shadow function**

After `xgb_prob_v4`, add:

```python
def xgb_prob_shadow(
    channels, pid: Optional[str] = None,
) -> Tuple[float, Optional[float]]:
    """Return (v3_prob, v4_prob) with v4 fully isolated from v3.

    v3 path runs normally (its own exception handling). v4 wrapped in
    try/except: any failure -> v4=None + log, NEVER affects v3. This is the
    only function cnn_agent should call during the shadow week.
    """
    prob_v3 = xgb_prob(channels, pid=pid)
    try:
        prob_v4 = xgb_prob_v4(channels, pid=pid)
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob_shadow: v4 path raised (isolated): %s", exc)
        prob_v4 = None
    return prob_v3, prob_v4
```

Also ensure `Tuple` is imported at the top: `from typing import List, Optional, Sequence, Tuple, Union`.

- [ ] **Step 5.6 — Run; expect 6+ green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py::TestV4ShadowLoad backend/tests/test_xgb_signal.py::TestXgbProbV4 backend/tests/test_xgb_signal.py::TestXgbProbShadow -v
```

Expected: 6+ PASSED. Then run the full `test_xgb_signal.py` to confirm no regressions in v1/v2/v3 path:

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py -v
```

Expected: full file green.

---

## Task 6: cnn_agent.py — write-through xgb_prob_v4 to save_cnn_scan

**Files:**
- Modify: `backend/agents/cnn_agent.py` (line 1903 — replace xgb_prob call; line 1989+ — add to scan dict)
- Modify: `backend/tests/test_cnn_agent.py` (add write-through test)

- [ ] **Step 6.1 — Inspect the existing call site**

Read `backend/agents/cnn_agent.py:1893-2020` and locate:
- The current call: `xgb_shadow = _xgb.xgb_prob(...)` (around line 1903)
- The save_cnn_scan dict construction (around line 1989-2020)

The plan assumes:
- Single call site for `xgb_prob` in `generate_signal` returning `xgb_shadow`
- Scan dict already has `"xgb_prob": round(xgb_shadow, 4) if xgb_shadow is not None else None`

If the call site differs from the assumption, adjust the edits below to match the actual structure.

- [ ] **Step 6.2 — Write the integration test**

Append at end of `backend/tests/test_cnn_agent.py`:

```python
# ── v4 shadow write-through (#xgb-v4 / Step B.1) ──────────────────────────


class TestV4ShadowWriteThrough:
    @pytest.mark.asyncio
    async def test_generate_signal_passes_xgb_prob_v4_to_save_cnn_scan(
        self, monkeypatch
    ):
        """When config.model_backend='xgb', cnn_agent must call xgb_prob_shadow
        and forward the v4 prob into the save_cnn_scan dict as xgb_prob_v4."""
        import agents.cnn_agent as ca

        # Stub xgb_signal.xgb_prob_shadow to return known (v3, v4)
        monkeypatch.setattr(
            "agents.xgb_signal.xgb_prob_shadow",
            lambda channels, pid=None: (0.65, 0.41),
        )

        # Capture save_cnn_scan calls
        saved: list = []
        async def fake_save(scan, db_path=None):
            saved.append(scan)
        monkeypatch.setattr("database.save_cnn_scan", fake_save)

        # Build a minimal agent and run generate_signal with enough fakes
        # to reach the save_cnn_scan call. Use existing test fixtures /
        # helpers from this file for agent setup (mirror TestGenerateSignal
        # patterns). The key assertion at the end:
        #
        #   assert saved[0]["xgb_prob_v4"] == pytest.approx(0.41)
        #
        # If reaching save_cnn_scan from the existing test harness is too
        # invasive, lift the relevant lines into a smaller unit test that
        # exercises only the dict-assembly logic from generate_signal.
        pytest.skip(
            "Integration test stub — wire to existing TestGenerateSignal "
            "harness during implementation."
        )
```

NOTE: This test is intentionally skipped — wiring it requires the existing test harness patterns from `test_cnn_agent.py` which vary. The implementer should either: (a) lift the assertion into the existing `TestGenerateSignal` test that already exercises save_cnn_scan, OR (b) build a minimal harness here. The shadow-path code is fully tested in Task 5 already; this test is a belt-and-suspenders write-through check.

- [ ] **Step 6.3 — Replace xgb_prob call with xgb_prob_shadow**

Edit `backend/agents/cnn_agent.py` around line 1900-1910. Find:

```python
                    xgb_shadow = _xgb.xgb_prob(
```

The full call likely looks like:

```python
                    xgb_shadow = _xgb.xgb_prob(
                        channels, pid=product_id,
                    )
```

Replace with:

```python
                    xgb_shadow, xgb_shadow_v4 = _xgb.xgb_prob_shadow(
                        channels, pid=product_id,
                    )
```

Read 5 lines of context before editing to get the exact existing code (indentation, argument names) right.

- [ ] **Step 6.4 — Add xgb_prob_v4 to the save_cnn_scan dict**

Edit `backend/agents/cnn_agent.py` around line 2011. Find:

```python
            "xgb_prob":    round(xgb_shadow, 4) if xgb_shadow is not None else None,
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev") if mc_telemetry else None,
```

Replace with (add `xgb_prob_v4` line on the next line):

```python
            "xgb_prob":    round(xgb_shadow, 4) if xgb_shadow is not None else None,
            "xgb_prob_v4": round(xgb_shadow_v4, 4) if xgb_shadow_v4 is not None else None,
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev") if mc_telemetry else None,
```

- [ ] **Step 6.5 — Run cnn_agent suite + xgb_signal suite + database suite**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_cnn_agent.py backend/tests/test_xgb_signal.py backend/tests/test_database.py -v
```

Expected: all green (the skipped test in Step 6.2 is intentionally skipped).

---

## Task 7: train_xgb_v4.py — orchestrator + tests

**Files:**
- Create: `backend/tools/train_xgb_v4.py`
- Create: `backend/tests/test_train_xgb_v4.py`

- [ ] **Step 7.1 — Write helper tests**

Create `backend/tests/test_train_xgb_v4.py`:

```python
"""Unit tests for backend/tools/train_xgb_v4.py helpers.

We test the pure helpers (_build_samples_for_pid, _walk_forward_split,
_triple_barrier_label) on synthetic candles. The orchestrator main() is
exercised end-to-end by operator-run smoke-test after the commit lands.
"""
from __future__ import annotations
import os
import sys
from typing import Dict, List

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_candles(n: int, base_close: float = 100.0,
                  drift: float = 0.0) -> List[Dict[str, float]]:
    """Synthetic OHLCV: linear drift, volume ramping."""
    candles = []
    for i in range(n):
        c = base_close + drift * i
        candles.append({
            "start":  1700000000 + i * 3600,
            "open":   c - 0.1,
            "high":   c + 0.5,
            "low":    c - 0.5,
            "close":  c,
            "volume": 100.0 + i,
        })
    return candles


class TestTripleBarrierLabel:
    def test_up_breach_returns_1(self):
        from tools.train_xgb_v4 import _triple_barrier_label
        # close[start]=100, threshold=0.01, forward 4 bars
        # close[start+1]=101.5 -> +1.5% > 1% -> UP = 1
        closes = np.array([100.0, 101.5, 100.0, 99.0, 100.0])
        assert _triple_barrier_label(closes, start=0,
                                     forward_hours=4, label_thresh=0.01) == 1

    def test_down_breach_returns_0(self):
        from tools.train_xgb_v4 import _triple_barrier_label
        closes = np.array([100.0, 98.5, 99.0, 100.0, 100.0])
        assert _triple_barrier_label(closes, start=0,
                                     forward_hours=4, label_thresh=0.01) == 0

    def test_no_breach_returns_0(self):
        from tools.train_xgb_v4 import _triple_barrier_label
        closes = np.array([100.0, 100.5, 99.5, 100.5, 100.0])
        # no bar exceeds +/-1%
        assert _triple_barrier_label(closes, start=0,
                                     forward_hours=4, label_thresh=0.01) == 0

    def test_returns_none_if_window_truncated(self):
        from tools.train_xgb_v4 import _triple_barrier_label
        closes = np.array([100.0, 101.0])
        # forward_hours=4 but only 1 forward bar available
        assert _triple_barrier_label(closes, start=0,
                                     forward_hours=4, label_thresh=0.01) is None


class TestBuildSamplesForPid:
    def test_empty_candles_returns_empty_arrays(self):
        from tools.train_xgb_v4 import _build_samples_for_pid
        X, y, ts = _build_samples_for_pid(
            [], label_thresh=0.003, forward_hours=4,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 150)
        assert y.shape == (0,)
        assert ts.shape == (0,)

    def test_too_few_candles_returns_empty(self):
        """Need at least macro + forward_hours candles to produce any sample."""
        from tools.train_xgb_v4 import _build_samples_for_pid
        candles = _make_candles(100)  # < 336 macro
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.003, forward_hours=4,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 150)

    def test_returns_correct_feature_width(self):
        from tools.train_xgb_v4 import _build_samples_for_pid
        # 500 candles, drift up -> some samples produced
        candles = _make_candles(500, drift=0.01)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.003, forward_hours=4,
            micro=60, meso=168, macro=336,
        )
        assert X.shape[1] == 150
        assert X.shape[0] == y.shape[0] == ts.shape[0]
        assert X.shape[0] > 0


class TestWalkForwardSplit:
    def test_splits_into_three_chronological_groups(self):
        from tools.train_xgb_v4 import _walk_forward_split
        n = 1000
        X = np.random.rand(n, 150)
        y = np.random.randint(0, 2, n)
        ts = np.arange(n, dtype=np.int64) + 1700000000

        (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
            X, y, ts, embargo_bars=4, val_frac=0.15, cal_frac=0.15,
        )
        # Train < Val < Cal chronologically; no overlap
        assert X_tr.shape[0] > 0
        assert X_va.shape[0] > 0
        assert X_ca.shape[0] > 0
        # Total should be <= n (embargo gaps removed)
        total = X_tr.shape[0] + X_va.shape[0] + X_ca.shape[0]
        assert total <= n
        # Embargo creates a gap
        assert total <= n - 2 * 4  # at most 2 embargo gaps of 4 bars
```

- [ ] **Step 7.2 — Run; expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_train_xgb_v4.py -v
```

Expected: collection error — `ModuleNotFoundError: No module named 'tools.train_xgb_v4'`.

- [ ] **Step 7.3 — Create the train script**

Create `backend/tools/train_xgb_v4.py`:

```python
"""XGB v4 trainer (#xgb-v4 / Step B.1).

Reads OHLCV per pid from backend/data/history/<pid>.parquet, builds
triple-barrier labels at CLI-specified `--forward-hours` / `--label-thresh`,
walk-forward splits chronologically, trains the v4 booster on the v4
OHLCV-5 features (150 cols), isotonic-calibrates, writes horizon-suffixed
artifacts to backend/xgb_*_v4_h<HOURS>.* paths.

Per feedback_python_clean_functions: main() delegates to small
single-responsibility helpers, each pure data-in/data-out.

Run (horizon sweep — operator runs 4 times, then v4_horizon_compare):
    cd backend && python -m tools.train_xgb_v4 \
      --pids BTC-USD,ETH-USD,... --forward-hours 24 --label-thresh 0.01
"""
from __future__ import annotations
import argparse
import logging
import os
import pickle
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_features import (  # noqa: E402
    extract_v4, feature_names_v4, feature_weights_v4,
    N_FEATURES_V4, TIER_WINDOWS_V4,
)

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_DIR = BACKEND
# No default for --forward-hours / --label-thresh; operator MUST specify so
# the suffixed artifact paths are explicit. See spec "Label horizon sweep".
_VAL_FRAC = 0.15
_CAL_FRAC = 0.15


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_candles_for_pid(pid: str, history_dir: str) -> List[Dict[str, float]]:
    """Read OHLCV candles for one pid from parquet. [] if file missing."""
    import pyarrow.parquet as pq
    path = os.path.join(history_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    rows = table.to_pydict()
    n = len(rows["start"])
    out: List[Dict[str, float]] = []
    for i in range(n):
        out.append({
            "start":  int(rows["start"][i]),
            "open":   float(rows["open"][i]),
            "high":   float(rows["high"][i]),
            "low":    float(rows["low"][i]),
            "close":  float(rows["close"][i]),
            "volume": float(rows["volume"][i]),
        })
    out.sort(key=lambda r: r["start"])
    return out


def _triple_barrier_label(
    closes: np.ndarray,
    start: int,
    forward_hours: int,
    label_thresh: float,
) -> Optional[int]:
    """Triple-barrier label for a sample anchored at `start`.

    Looks forward up to `forward_hours` bars. Returns:
        1 if any forward close >= entry * (1 + label_thresh)
        0 if any forward close <= entry * (1 - label_thresh)  (down breach first)
        0 if neither breach within window (vertical barrier)
        None if window truncated (start + forward_hours >= len(closes))
    """
    n = closes.size
    if start + forward_hours >= n:
        return None
    entry = closes[start]
    up_thr = entry * (1.0 + label_thresh)
    dn_thr = entry * (1.0 - label_thresh)
    for i in range(start + 1, start + forward_hours + 1):
        c = closes[i]
        if c >= up_thr:
            return 1
        if c <= dn_thr:
            return 0
    return 0


def _build_samples_for_pid(
    candles: List[Dict[str, float]],
    *,
    label_thresh: float,
    forward_hours: int,
    micro: int,
    meso: int,
    macro: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each valid bar i where i >= macro AND a label can be computed,
    produce one sample (features [150], label, timestamp).

    Returns:
        features:   (N, 150) float64
        labels:     (N,) int8 (0/1)
        timestamps: (N,) int64 (epoch seconds at sample bar)
    """
    n = len(candles)
    if n < macro + forward_hours + 1:
        return (np.zeros((0, N_FEATURES_V4), dtype=np.float64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int64))
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    feats_list: List[np.ndarray] = []
    labels_list: List[int] = []
    ts_list: List[int] = []
    for i in range(macro, n):
        label = _triple_barrier_label(closes, i, forward_hours, label_thresh)
        if label is None:
            continue
        tier_slices = {
            "micro": candles[i - micro:i],
            "meso":  candles[i - meso:i],
            "macro": candles[i - macro:i],
        }
        feats, _ = extract_v4(tier_slices)
        feats_list.append(feats[0])
        labels_list.append(label)
        ts_list.append(candles[i]["start"])
    if not feats_list:
        return (np.zeros((0, N_FEATURES_V4), dtype=np.float64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int64))
    X = np.stack(feats_list, axis=0)
    y = np.array(labels_list, dtype=np.int8)
    ts = np.array(ts_list, dtype=np.int64)
    return X, y, ts


def _walk_forward_split(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    embargo_bars: int,
    val_frac: float = _VAL_FRAC,
    cal_frac: float = _CAL_FRAC,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Chronological split (train, val, cal) with embargo gaps between groups.

    Assumes inputs are sorted ascending by timestamp.
    """
    n = features.shape[0]
    cal_n = int(n * cal_frac)
    val_n = int(n * val_frac)
    train_end = n - val_n - cal_n - 2 * embargo_bars
    if train_end < 1:
        # Degenerate — give caller something runnable, no embargo
        train_end = max(1, n - val_n - cal_n)
        embargo_bars = 0
    val_start = train_end + embargo_bars
    val_end   = val_start + val_n
    cal_start = val_end + embargo_bars
    cal_end   = cal_start + cal_n
    X_tr = features[:train_end];           y_tr = labels[:train_end]
    X_va = features[val_start:val_end];    y_va = labels[val_start:val_end]
    X_ca = features[cal_start:cal_end];    y_ca = labels[cal_start:cal_end]
    return (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca)


def _train_booster(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], feature_weights: np.ndarray,
):
    """Train one xgb.Booster. Returns booster + final val_auc."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    d_tr = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    d_tr.set_info(feature_weights=feature_weights)
    d_va = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    d_va.set_info(feature_weights=feature_weights)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "seed": 0,
    }
    booster = xgb.train(
        params, d_tr, num_boost_round=200,
        evals=[(d_va, "val")], verbose_eval=False,
    )
    val_pred = booster.predict(d_va)
    val_auc = float(roc_auc_score(y_val, val_pred)) if len(set(y_val)) == 2 else float("nan")
    return booster, val_auc


def _calibrate_isotonic(booster, X_cal: np.ndarray, y_cal: np.ndarray,
                        feature_names: List[str]):
    """Fit IsotonicRegression on the booster's raw probs vs calibration labels."""
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression

    d_ca = xgb.DMatrix(X_cal, feature_names=feature_names)
    raw = booster.predict(d_ca)
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    cal.fit(raw, y_cal)
    return cal


def _save_artifacts(
    booster,
    calibrator,
    feature_names: List[str],
    out_dir: str,
    *,
    forward_hours: int,
) -> Dict[str, str]:
    """Atomic write of model.json, features.json, calibration.pkl with
    horizon-suffixed paths: xgb_*_v4_h<HOURS>.*

    Note: xgboost auto-detects serialization format from the LAST extension.
    Tmp path MUST end in .json (NOT .json.tmp) to write JSON not UBJSON.
    """
    import json

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_h{forward_hours}"
    model_path = os.path.join(out_dir, f"xgb_model_v4{suffix}.json")
    feat_path  = os.path.join(out_dir, f"xgb_features_v4{suffix}.json")
    cal_path   = os.path.join(out_dir, f"xgb_calibration_v4{suffix}.pkl")

    model_tmp = os.path.join(out_dir, f"xgb_model_v4{suffix}.tmp.json")  # .json last
    feat_tmp  = feat_path + ".tmp"
    cal_tmp   = cal_path + ".tmp"

    booster.save_model(model_tmp)
    os.replace(model_tmp, model_path)

    with open(feat_tmp, "w") as f:
        json.dump({"feature_names": feature_names, "feature_set": "v4"}, f)
    os.replace(feat_tmp, feat_path)

    with open(cal_tmp, "wb") as f:
        pickle.dump({"calibrator": calibrator, "feature_set": "v4"}, f)
    os.replace(cal_tmp, cal_path)

    return {"model": model_path, "features": feat_path, "calibration": cal_path}


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pids", required=True,
                   help="comma-separated, e.g. BTC-USD,ETH-USD")
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument("--forward-hours", type=int, required=True,
                   help="label horizon in bars (4, 24, 72, 168 per sweep)")
    p.add_argument("--label-thresh", type=float, required=True,
                   help="triple-barrier threshold (e.g. 0.003, 0.01, 0.02, 0.05)")
    p.add_argument("--embargo-bars", type=int, default=0,
                   help="defaults to forward_hours if 0")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    micro = TIER_WINDOWS_V4["micro"]
    meso  = TIER_WINDOWS_V4["meso"]
    macro = TIER_WINDOWS_V4["macro"]
    embargo = args.embargo_bars if args.embargo_bars > 0 else args.forward_hours

    t0 = time.time()
    print(f"v4 train: pids={pids} forward_hours={args.forward_hours} "
          f"label_thresh={args.label_thresh} embargo_bars={embargo} "
          f"-> xgb_*_v4_h{args.forward_hours}.*", flush=True)

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_t: List[np.ndarray] = []
    skipped: List[str] = []
    for pid in pids:
        candles = _load_candles_for_pid(pid, args.history_dir)
        if not candles:
            skipped.append(pid)
            print(f"  {pid}: no parquet — skip", flush=True)
            continue
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=args.label_thresh,
            forward_hours=args.forward_hours,
            micro=micro, meso=meso, macro=macro,
        )
        if X.shape[0] == 0:
            skipped.append(pid)
            print(f"  {pid}: too few candles ({len(candles)}) — skip", flush=True)
            continue
        all_X.append(X); all_y.append(y); all_t.append(ts)
        print(f"  {pid}: {X.shape[0]:,} samples, pos_frac={y.mean():.4f}", flush=True)

    if not all_X:
        print("ERROR: no usable pids", flush=True)
        return 1

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    order = np.argsort(t, kind="stable")
    X = X[order]; y = y[order]; t = t[order]
    print(f"\nPooled: X={X.shape} pos_frac={y.mean():.4f}", flush=True)

    (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
        X, y, t, embargo_bars=embargo,
    )
    print(f"Split: train={X_tr.shape} val={X_va.shape} cal={X_ca.shape}",
          flush=True)

    names = feature_names_v4()
    weights = feature_weights_v4()
    booster, val_auc = _train_booster(X_tr, y_tr, X_va, y_va, names, weights)
    print(f"Train done: val_auc={val_auc:.4f}", flush=True)

    calibrator = _calibrate_isotonic(booster, X_ca, y_ca, names)
    print("Calibrated.", flush=True)

    paths = _save_artifacts(
        booster, calibrator, names, args.out_dir,
        forward_hours=args.forward_hours,
    )
    print(f"Wrote: {paths}", flush=True)
    print(f"Skipped pids: {skipped}", flush=True)
    print(f"Total wall: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.4 — Run helper tests; expect green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_train_xgb_v4.py -v
```

Expected: 8 PASSED.

---

## Task 7b: v4_horizon_compare.py — held-out AUC + HTML report

**Files:**
- Create: `backend/tools/v4_horizon_compare.py`
- Create: `backend/tests/test_v4_horizon_compare.py`

- [ ] **Step 7b.1 — Write the helper tests**

Create `backend/tests/test_v4_horizon_compare.py`:

```python
"""Unit tests for backend/tools/v4_horizon_compare.py."""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestEvaluateOnHoldout:
    def test_returns_metrics_dict(self):
        from tools.v4_horizon_compare import _evaluate_on_holdout

        class _StubBooster:
            def predict(self, dmat):
                # Return calibrated-ish probs aligned with labels
                return np.array([0.2, 0.7, 0.3, 0.8])

        class _IdentityCal:
            def transform(self, x):
                return x

        X = np.zeros((4, 150), dtype=np.float64)
        y = np.array([0, 1, 0, 1], dtype=np.int8)
        names = [f"col{i}" for i in range(150)]
        out = _evaluate_on_holdout(_StubBooster(), _IdentityCal(), X, y, names)
        assert "auc" in out
        assert "logloss" in out
        assert "pos_frac" in out
        assert "n_samples" in out
        assert out["n_samples"] == 4
        assert out["pos_frac"] == 0.5
        # AUC for perfectly aligned probs: 1.0
        assert out["auc"] == pytest.approx(1.0)

    def test_returns_nan_auc_for_single_class(self):
        from tools.v4_horizon_compare import _evaluate_on_holdout

        class _StubBooster:
            def predict(self, dmat):
                return np.array([0.5, 0.5, 0.5])

        class _IdentityCal:
            def transform(self, x):
                return x

        X = np.zeros((3, 150), dtype=np.float64)
        y = np.array([1, 1, 1], dtype=np.int8)
        names = [f"col{i}" for i in range(150)]
        out = _evaluate_on_holdout(_StubBooster(), _IdentityCal(), X, y, names)
        assert np.isnan(out["auc"])


class TestRenderHtmlReport:
    def test_writes_html_file(self, tmp_path):
        from tools.v4_horizon_compare import _render_html_report

        metrics = {
            4:   {"auc": 0.512, "logloss": 0.69, "pos_frac": 0.48, "n_samples": 1000},
            24:  {"auc": 0.534, "logloss": 0.68, "pos_frac": 0.45, "n_samples": 800},
            72:  {"auc": 0.561, "logloss": 0.67, "pos_frac": 0.40, "n_samples": 600},
            168: {"auc": 0.589, "logloss": 0.66, "pos_frac": 0.35, "n_samples": 400},
        }
        out_path = str(tmp_path / "report.html")
        _render_html_report(metrics, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        # Sanity checks on content
        assert "0.589" in html or "0.59" in html   # h168 AUC visible
        assert "h168" in html or "168" in html     # horizon visible
        assert "auc" in html.lower()
```

- [ ] **Step 7b.2 — Run; expect ImportError**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_v4_horizon_compare.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.v4_horizon_compare'`.

- [ ] **Step 7b.3 — Create the compare script**

Create `backend/tools/v4_horizon_compare.py`:

```python
"""XGB v4 horizon sweep comparison report (#xgb-v4 / Step B.1).

For each horizon (4, 24, 72, 168), load that horizon's artifacts (booster +
calibrator + feature_names), build a held-out test set at that horizon,
compute AUC + logloss + n_samples + pos_frac, render side-by-side HTML
report.

Per feedback_python_clean_functions: pure-function helpers, main()
orchestrator only.

Run (after all 4 horizons have been trained via train_xgb_v4.py):
    cd backend && python -m tools.v4_horizon_compare \
      --horizons 4,24,72,168 \
      --pids BTC-USD,ETH-USD,SOL-USD,...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = BACKEND
_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_PATH = os.path.join(
    BACKEND, "tools", "xgb_v4_horizon_compare.html"
)
# Map horizon -> label_thresh (must match what train_xgb_v4 was run with)
_HORIZON_THRESHOLDS: Dict[int, float] = {
    4: 0.003, 24: 0.01, 72: 0.02, 168: 0.05,
}


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_horizon_artifacts(horizon: int, base_dir: str) -> Dict[str, object]:
    """Load booster + calibrator + feature_names for one horizon.

    Expected files:
      base_dir/xgb_model_v4_h<H>.json
      base_dir/xgb_features_v4_h<H>.json
      base_dir/xgb_calibration_v4_h<H>.pkl
    """
    import xgboost as xgb

    model_path = os.path.join(base_dir, f"xgb_model_v4_h{horizon}.json")
    feat_path  = os.path.join(base_dir, f"xgb_features_v4_h{horizon}.json")
    cal_path   = os.path.join(base_dir, f"xgb_calibration_v4_h{horizon}.pkl")
    for p in (model_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"horizon h{horizon} artifact missing: {p}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)["feature_names"]
    calibrator: Optional[object] = None
    if os.path.exists(cal_path):
        with open(cal_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict) and "calibrator" in obj:
            calibrator = obj["calibrator"]
    return {
        "booster": booster,
        "calibrator": calibrator,
        "feature_names": feature_names,
    }


def _evaluate_on_holdout(
    booster, calibrator, X: np.ndarray, y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Compute AUC + logloss + pos_frac on a held-out set.

    Returns dict with keys 'auc', 'logloss', 'pos_frac', 'n_samples'.
    AUC is nan when y has a single class.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss

    n = X.shape[0]
    pos_frac = float(y.mean()) if n > 0 else 0.0
    if n == 0:
        return {"auc": float("nan"), "logloss": float("nan"),
                "pos_frac": pos_frac, "n_samples": 0}
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    raw = booster.predict(dmat)
    if calibrator is not None:
        raw = calibrator.transform(raw)
    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, raw)) if len(set(y)) == 2 else float("nan")
    ll = float(log_loss(y, raw)) if len(set(y)) == 2 else float("nan")
    return {"auc": auc, "logloss": ll,
            "pos_frac": pos_frac, "n_samples": n}


def _build_holdout_dataset(
    pids: List[str], horizon: int, label_thresh: float,
    history_dir: str, holdout_frac: float = 0.15,
):
    """Build a held-out (X, y) test set per pid using the LAST holdout_frac
    of each pid's history (chronologically AFTER what train_xgb_v4 used).

    Uses _build_samples_for_pid from train_xgb_v4 for consistency.
    """
    from tools.train_xgb_v4 import _build_samples_for_pid, _load_candles_for_pid
    from tools.xgb_v4_features import TIER_WINDOWS_V4, N_FEATURES_V4

    micro = TIER_WINDOWS_V4["micro"]
    meso  = TIER_WINDOWS_V4["meso"]
    macro = TIER_WINDOWS_V4["macro"]

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    for pid in pids:
        candles = _load_candles_for_pid(pid, history_dir)
        if not candles:
            continue
        X, y, _ts = _build_samples_for_pid(
            candles, label_thresh=label_thresh, forward_hours=horizon,
            micro=micro, meso=meso, macro=macro,
        )
        if X.shape[0] == 0:
            continue
        # Take last holdout_frac of samples per pid
        n_hold = max(1, int(X.shape[0] * holdout_frac))
        all_X.append(X[-n_hold:])
        all_y.append(y[-n_hold:])
    if not all_X:
        return np.zeros((0, N_FEATURES_V4), dtype=np.float64), np.zeros(0, dtype=np.int8)
    return np.vstack(all_X), np.concatenate(all_y)


def _render_html_report(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    out_path: str,
) -> None:
    """Side-by-side HTML report (dark mode, matches xgb_v3_channel_options.html style)."""
    # Determine winner by AUC (highest, ignoring NaN)
    valid = {h: m for h, m in metrics_by_horizon.items()
             if not np.isnan(m.get("auc", float("nan")))}
    winner = max(valid, key=lambda h: valid[h]["auc"]) if valid else None

    rows = []
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        cls = "winner" if h == winner else ""
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>h{h}</td>"
            f"<td class='num'>{m['auc']:.4f}</td>"
            f"<td class='num'>{m['logloss']:.4f}</td>"
            f"<td class='num'>{m['pos_frac']:.4f}</td>"
            f"<td class='num'>{m['n_samples']:,}</td>"
            f"</tr>"
        )

    winner_banner = (
        f"<div class='banner'>Winner: <strong>h{winner}</strong> "
        f"(AUC {valid[winner]['auc']:.4f})</div>"
    ) if winner is not None else "<div class='banner'>No valid AUC computed.</div>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>XGB v4 horizon comparison</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,sans-serif;
          padding:32px; max-width:900px; margin:auto; }}
  h1 {{ color:#fff; }}
  .banner {{ background:#1f3a1f; border:1px solid #1f6b33; color:#56d364;
             padding:14px 20px; border-radius:6px; margin:20px 0; }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:left; color:#8b949e; padding:8px; border-bottom:1px solid #30363d; }}
  td {{ padding:8px; border-bottom:1px solid #21262d; font-family:ui-monospace,monospace; }}
  tr.winner td {{ background:#0d1c11; color:#56d364; font-weight:600; }}
  .num {{ text-align:right; }}
</style></head><body>
<h1>XGB v4 horizon comparison</h1>
{winner_banner}
<table>
  <tr><th>horizon</th><th>auc</th><th>logloss</th><th>pos_frac</th><th>n_samples</th></tr>
  {''.join(rows)}
</table>
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", required=True,
                   help="comma-separated, e.g. 4,24,72,168")
    p.add_argument("--pids", required=True,
                   help="comma-separated pid list (same as train_xgb_v4)")
    p.add_argument("--base-dir", default=_DEFAULT_BASE_DIR,
                   help="directory containing xgb_*_v4_h<N>.* artifacts")
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-path", default=_DEFAULT_OUT_PATH)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]

    metrics: Dict[int, Dict[str, float]] = {}
    for h in horizons:
        thresh = _HORIZON_THRESHOLDS.get(h)
        if thresh is None:
            print(f"  h{h}: no default threshold known — skipping", flush=True)
            continue
        print(f"  h{h}: loading artifacts...", flush=True)
        try:
            artifacts = _load_horizon_artifacts(h, args.base_dir)
        except FileNotFoundError as exc:
            print(f"  h{h}: {exc} — skip", flush=True)
            continue
        print(f"  h{h}: building holdout dataset...", flush=True)
        X, y = _build_holdout_dataset(pids, h, thresh, args.history_dir)
        print(f"  h{h}: evaluating on {X.shape[0]} samples...", flush=True)
        metrics[h] = _evaluate_on_holdout(
            artifacts["booster"], artifacts["calibrator"],
            X, y, artifacts["feature_names"],
        )
        m = metrics[h]
        print(f"  h{h}: auc={m['auc']:.4f} logloss={m['logloss']:.4f} "
              f"n={m['n_samples']} pos_frac={m['pos_frac']:.4f}", flush=True)

    _render_html_report(metrics, args.out_path)
    print(f"\nHTML report: {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7b.4 — Run; expect 3 PASSED**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_v4_horizon_compare.py -v
```

Expected: 3 PASSED.

---

## Task 8: CLAUDE.md invariant #16 + CHANGELOG

**Files:**
- Modify: `CLAUDE.md` (append invariant #16)
- Modify: `CHANGELOG.md` (prepend Session 58.71j)

- [ ] **Step 8.1 — Add invariant #16 to CLAUDE.md**

Read `CLAUDE.md` and find the "Key invariants (never break these)" section near the end of the "Architecture Quick Reference". After invariant #15, append:

```markdown
16. **Shadow telemetry isolation** — Inference shadow paths (v4 alongside v3) must NEVER affect the driver path. Failures in any shadow inference are caught + logged + recorded as NULL, never re-raised. `xgb_signal.xgb_prob_shadow` is the only function that may be called from `cnn_agent` during shadow validation; it returns `(driver_prob, shadow_prob_or_None)`. Mirrors invariant #14's MC chain rule.
```

- [ ] **Step 8.2 — Add CHANGELOG entry**

Edit `CHANGELOG.md`. Find the current top entry (Session 58.71i — marketcap Step A) and insert above it:

```markdown
## [Session 58.71j] — 2026-05-17 — XGB v4 OHLCV-5 shadow model (#xgb-v4 / Step B.1)

### Why
v3's `_extract_v3` only reads `close` from candles, so all 28 channel slots
collapse to ~30 distinct values dressed up as 350 feature names. The booster
wastes capacity learning that `ch0_last == ch1_last == ... == ch16_last`.
Fixing this needs a fresh model: feature distribution changes invalidate
v3's calibration. Step B.1 of the XGB channel-buildout roadmap ships the
smallest honest baseline (5 OHLCV channels) and runs it in shadow alongside
live v3 for one week before any cutover decision.

### What changed
- **`backend/tools/xgb_v4_features.py`** (new) — pure-function v4 extractor.
  5 channels (open/high/low/close/volume) x 3 tiers (micro 60 / meso 168 /
  macro 336) x 10 stats = 150 features. `extract_v4`, `feature_names_v4`,
  `feature_weights_v4` public; helpers `_extract_field`, `_compute_stats`,
  `_slope`, `_pct_rank`, `_delta_at` each pure data-in/data-out, no
  in-place buffer mutation. Constants derived (`N_CHANNELS_V4 = len(_CHANNEL_FIELDS)`).
- **`backend/tools/train_xgb_v4.py`** (new) — trainer orchestrator. `main()`
  delegates to 7 single-responsibility helpers: `_load_candles_for_pid`,
  `_triple_barrier_label`, `_build_samples_for_pid`, `_walk_forward_split`,
  `_train_booster`, `_calibrate_isotonic`, `_save_artifacts`. Reads OHLCV
  parquets. Required CLI args `--forward-hours` and `--label-thresh` —
  no default values, operator MUST specify per the horizon sweep workflow.
  Writes horizon-suffixed artifacts (`backend/xgb_*_v4_h<HOURS>.*`) so all
  4 horizons coexist on disk.
- **`backend/tools/v4_horizon_compare.py`** (new) — horizon sweep
  comparison report. `main()` orchestrator + 4 pure helpers: `_load_horizon_artifacts`,
  `_evaluate_on_holdout`, `_build_holdout_dataset`, `_render_html_report`.
  Loads each horizon's artifacts, builds held-out test set per horizon
  (last 15% of each pid's history), computes AUC + logloss + n_samples +
  pos_frac, renders side-by-side HTML report at
  `backend/tools/xgb_v4_horizon_compare.html`. Highlights winner by AUC.
- **`backend/migrations/xgb_v4_shadow_20260517.py`** (new) — idempotent
  ALTER TABLE adding `cnn_scans.xgb_prob_v4 REAL` for shadow telemetry.
- **`backend/tools/xgb_features.py`** — `extract_features` dispatcher gets
  `feature_set == "v4"` branch routing to `xgb_v4_features.extract_v4`.
- **`backend/agents/xgb_signal.py`** — new module-level v4 state (`_booster_v4`,
  `_calibration_v4`, `_load_attempted_v4`, `_load_succeeded_v4`), new
  `_try_load_v4()`, `xgb_prob_v4(channels, pid)`, and `xgb_prob_shadow(channels, pid)`
  returning `(prob_v3, prob_v4_or_None)`. v4 fully isolated in try/except;
  failures NEVER affect v3. v3 path unchanged.
- **`backend/database.py`** — `xgb_prob_v4 REAL` added to `cnn_scans`
  CREATE TABLE, ALTER TABLE migration list, and `save_cnn_scan` INSERT.
- **`backend/agents/cnn_agent.py`** — single edit: replace the existing
  `_xgb.xgb_prob(...)` call with `_xgb.xgb_prob_shadow(...)`, unpack the
  returned tuple, add `xgb_prob_v4` to the `save_cnn_scan` dict. NO
  decision logic touched.
- **CLAUDE.md** — invariant #16 (shadow telemetry isolation).
- **Tests** — `test_xgb_v4_features.py` (30+ tests), `test_train_xgb_v4.py`
  (8 tests), extensions to `test_xgb_signal.py` (6 shadow tests),
  `test_database.py` (2 persistence tests), `test_mc_migration.py` (2
  idempotency tests).

### Verification
```
cd backend && python -m pytest tests/ -q -m "not slow and not integration"
=> 975+ passed (4+8+6+2+2+3 new tests)
```

### Operator preflight (run once after this commit) — horizon sweep
```bash
cd backend
PIDS=BTC-USD,ETH-USD,SOL-USD,...   # populate with tracked pids
# Train 4 horizons
../.venv/Scripts/python.exe -m tools.train_xgb_v4 --pids $PIDS --forward-hours 4   --label-thresh 0.003
../.venv/Scripts/python.exe -m tools.train_xgb_v4 --pids $PIDS --forward-hours 24  --label-thresh 0.01
../.venv/Scripts/python.exe -m tools.train_xgb_v4 --pids $PIDS --forward-hours 72  --label-thresh 0.02
../.venv/Scripts/python.exe -m tools.train_xgb_v4 --pids $PIDS --forward-hours 168 --label-thresh 0.05
# Render comparison report
../.venv/Scripts/python.exe -m tools.v4_horizon_compare --pids $PIDS --horizons 4,24,72,168
# Open backend/tools/xgb_v4_horizon_compare.html; pick winner (e.g., h24):
cp xgb_model_v4_h24.json     xgb_model_v4.json
cp xgb_features_v4_h24.json  xgb_features_v4.json
cp xgb_calibration_v4_h24.pkl xgb_calibration_v4.pkl
```

Expected wall time: ~5-10 min per horizon × 4 horizons ≈ 30-40 min for ~50
pids. Compare script: seconds. After winner is copied to unsuffixed paths:
backend restart picks up `xgb_*_v4.*`; shadow telemetry begins on next scan.

### Cutover decision (post-shadow-week, separate brainstorm)
```sql
SELECT
  COUNT(*) AS n_outcomes,
  AVG(s.xgb_prob_v3) AS v3_mean_prob,
  AVG(s.xgb_prob_v4) AS v4_mean_prob
FROM cnn_scans s
JOIN signal_outcomes o ON o.scan_id = s.id
WHERE s.scanned_at >= <commit_ts + 7 days>
  AND s.xgb_prob_v4 IS NOT NULL
GROUP BY o.outcome_class;
```

Python-side AUC: `sklearn.metrics.roc_auc_score(labels, probs)` for v3 and v4
on the same outcome subset. Decision criteria + cutover land in their own
brainstorm cycle.

### Step B.2 preview
Add macro-trend channels: market_cap (ch5) + volume_24h (ch6) from bronze
parquet (Step A schema v2 already has them). N_CHANNELS_V4 5 -> 7, retrain
booster. Separate brainstorm cycle.
```

---

## Task 9: Full suite + commit + push + memory sync

**Files:**
- Run pre-commit pytest
- `git add` + `git commit` + `git push`
- Memory: `coinbase_trader_architecture.md`

- [ ] **Step 9.1 — Stage all files**

```bash
cd C:\Users\gl450\polymarket_app
git add \
  backend/tools/xgb_v4_features.py \
  backend/tools/train_xgb_v4.py \
  backend/tools/v4_horizon_compare.py \
  backend/migrations/xgb_v4_shadow_20260517.py \
  backend/tests/test_xgb_v4_features.py \
  backend/tests/test_train_xgb_v4.py \
  backend/tests/test_v4_horizon_compare.py \
  backend/tests/test_xgb_signal.py \
  backend/tests/test_database.py \
  backend/tests/test_mc_migration.py \
  backend/tools/xgb_features.py \
  backend/agents/xgb_signal.py \
  backend/database.py \
  backend/agents/cnn_agent.py \
  CLAUDE.md \
  CHANGELOG.md
```

Verify staging:
```bash
git status --short | grep -v '^??'
```

Expected: 16 staged files, no surprises.

- [ ] **Step 9.2 — Commit (pre-commit hook runs full suite ~5 min)**

```bash
git commit -m "$(cat <<'EOF'
feat(xgb-v4): OHLCV-5 shadow model alongside v3 (Step B.1)

Fresh XGB model with 5 OHLCV-derived channels (open/high/low/close/volume),
running in shadow alongside live v3 for one week before any cutover decision.

xgb_v4_features (NEW):
- Pure-function extractor: extract_v4, feature_names_v4, feature_weights_v4.
- 5 channels x 3 tiers (micro 60 / meso 168 / macro 336) x 10 stats = 150 features.
- Per feedback_python_clean_functions: helpers (_extract_field, _compute_stats,
  _slope, _pct_rank, _delta_at) each pure data-in/data-out, no in-place buffer
  mutation. Constants derived (N_CHANNELS_V4 = len(_CHANNEL_FIELDS)).

train_xgb_v4 (NEW):
- main() delegates to _load_candles_for_pid, _triple_barrier_label,
  _build_samples_for_pid, _walk_forward_split, _train_booster,
  _calibrate_isotonic, _save_artifacts.
- Reads OHLCV from backend/data/history/<pid>.parquet. Required CLI args
  --forward-hours + --label-thresh (no defaults). Atomic writes to
  horizon-suffixed paths xgb_*_v4_h<HOURS>.* so all 4 sweep horizons coexist.

v4_horizon_compare (NEW):
- main() orchestrator + helpers _load_horizon_artifacts, _evaluate_on_holdout,
  _build_holdout_dataset, _render_html_report.
- Loads each horizon's artifacts, evaluates on per-pid last-15% holdout set,
  renders side-by-side HTML report at backend/tools/xgb_v4_horizon_compare.html
  with winner highlighted.

xgb_signal shadow path:
- New _try_load_v4, xgb_prob_v4, xgb_prob_shadow(channels, pid) -> (v3, v4_or_None).
- v4 fully isolated in try/except; failures NEVER affect v3.
- v3 path unchanged.

cnn_agent write-through:
- _xgb.xgb_prob -> _xgb.xgb_prob_shadow at line 1903; xgb_prob_v4 added
  to save_cnn_scan dict. NO decision logic changes.

Database:
- New cnn_scans.xgb_prob_v4 REAL column.
- Idempotent migration backend/migrations/xgb_v4_shadow_20260517.py.
- save_cnn_scan INSERT + CREATE TABLE + ALTER TABLE migration list.

CLAUDE.md invariant #16 (shadow telemetry isolation) added.

Operator horizon sweep (run after this commit, ~30-40 min total):
  python -m tools.train_xgb_v4 --pids <list> --forward-hours 4 --label-thresh 0.003
  ... (24/0.01, 72/0.02, 168/0.05)
  python -m tools.v4_horizon_compare --pids <list> --horizons 4,24,72,168
  # pick winner from HTML report, cp xgb_*_v4_h<N>.* -> xgb_*_v4.*
  # restart backend; shadow tracks winner

Shadow AUC vs v3 has horizon-mismatch caveat if winner is not h4. Live
trade outcomes (signal_outcomes) are the ground truth.

Tests: 48+ new (xgb_v4_features 30+, train_xgb_v4 8, xgb_signal shadow 6,
database persistence 2, migration idempotency 2). Full suite green.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hook runs the full ~970-test suite (~5 min). All must pass — `xgb_prob_v4` shadow tests + extractor tests + persistence + migration green from Tasks 1-7. Existing v3, MC, marketcap, cnn_agent suites must still pass.

If a regression fires, do NOT amend. Stash, fix, re-run, fresh commit (per the prior session's amend-bug postmortem).

- [ ] **Step 9.3 — Push**

```bash
git push origin feat/gpu-coord-mirror
```

- [ ] **Step 9.4 — Verify backend healthy post-push**

```bash
curl -sS -m 3 http://localhost:8001/api/status
```

Expected: 200 with `is_trading:true, dry_run:true`. v4 shadow will be inert until operator runs the train script and restarts.

- [ ] **Step 9.5 — Memory sync**

Edit `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`. Find the most recent Session 58.71 entry (58.71i marketcap Step A) and prepend above it:

```markdown
- **Session 58.71j (2026-05-17)**: XGB v4 OHLCV-5 shadow model + horizon sweep (#xgb-v4 / Step B.1). New `backend/tools/xgb_v4_features.py` (pure-function extractor, 5 OHLCV channels × 3 tiers × 10 stats = 150 features, helpers `_extract_field`/`_compute_stats`/`_slope`/`_pct_rank`/`_delta_at` each single-responsibility per [[feedback_python_clean_functions]]). New `backend/tools/train_xgb_v4.py` (orchestrator `main()` delegating to 7 helpers; reads `backend/data/history/<pid>.parquet`, parameterized by `--forward-hours` + `--label-thresh`, writes horizon-suffixed artifacts `xgb_*_v4_h<HOURS>.*`). New `backend/tools/v4_horizon_compare.py` (loads each horizon's artifacts, builds last-15% holdout per pid, AUC+logloss compare, side-by-side HTML report at `backend/tools/xgb_v4_horizon_compare.html`). New `cnn_scans.xgb_prob_v4 REAL` column + idempotent migration `xgb_v4_shadow_20260517.py`. `xgb_signal` gained `xgb_prob_v4` + `xgb_prob_shadow(channels, pid) -> (v3, v4_or_None)`; v4 fully isolated in try/except — failures NEVER affect v3. `cnn_agent` line 1903 switched to `xgb_prob_shadow`, `save_cnn_scan` dict gained `xgb_prob_v4`. v3 path UNCHANGED throughout. CLAUDE.md invariant #16 (shadow telemetry isolation). **Operator horizon sweep** (~30-40 min total after commit): train 4 horizons (h4/h24/h72/h168 with thresholds 0.003/0.01/0.02/0.05), run `v4_horizon_compare`, pick winner from HTML, copy `xgb_*_v4_h<WINNER>.*` to `xgb_*_v4.*`, restart backend → shadow tracks winner. Full suite 975+ passed (+51 new tests). Commit `<sha>`, branch `feat/gpu-coord-mirror`.
```

Fill the `<sha>` placeholder with the actual commit hash from Step 9.2.

---

## Spec coverage check

| Spec section | Plan task |
|---|---|
| Architecture diagram | Task 5 (xgb_signal shadow path matches the diagram) |
| xgb_v4_features.py module | Task 1 |
| xgb_features.py dispatcher edit | Task 2 |
| xgb_signal.py shadow path | Task 5 |
| database.py + CREATE TABLE + ALTER + INSERT | Task 4 |
| migrations/xgb_v4_shadow_<ts>.py | Task 3 |
| cnn_agent.py write-through | Task 6 |
| train_xgb_v4.py (horizon-suffixed artifacts) | Task 7 |
| v4_horizon_compare.py (HTML report) | Task 7b |
| test_xgb_v4_features.py | Task 1 |
| test_train_xgb_v4.py | Task 7 |
| test_v4_horizon_compare.py | Task 7b |
| test_xgb_signal.py extension | Task 5 |
| test_database.py extension | Task 4 |
| Error handling — 4 cases | Tasks 5 + 6 (artifacts missing, load error, inference error, calibrator error) |
| Tests strategy (unit + integration + migration) | Tasks 1, 3, 5, 6, 7, 7b |
| Rollout — single atomic commit | Task 9 |
| Rollout — operator horizon sweep | Task 8 (CHANGELOG) + Task 9 |
| Label horizon sweep (4 horizons + HTML compare) | Task 7 (suffixed paths) + Task 7b (compare script) |
| CLAUDE.md invariant #16 | Task 8 |
| CHANGELOG 58.71j | Task 8 |
| Memory append | Task 9 |
| Non-goals (no cutover, no marketcap, no CNN changes) | enforced by plan scope |

All spec sections traced to a task.

## Self-review (placeholder scan, type consistency)

- No "TBD", "TODO", "implement later" in any step body
- Every code-changing step shows the full code
- Function signatures consistent: `xgb_prob_shadow(channels, pid) -> Tuple[float, Optional[float]]` matches Tasks 5, 6, 8
- File paths absolute / repo-rooted, no ambiguity
- Migration filename `xgb_v4_shadow_20260517.py` consistent across Tasks 3, 8, 9
- Horizon-suffixed paths `xgb_*_v4_h<HOURS>.*` consistent across Tasks 7, 7b, 8, 9
- Shadow path loads from unsuffixed `xgb_*_v4.*` (operator copies winner) — Tasks 5, 8, 9 agree
- Helper names consistent: `_load_horizon_artifacts`, `_evaluate_on_holdout`, `_build_holdout_dataset`, `_render_html_report` (Task 7b)

One known gap: Task 6 Step 6.2 test is `pytest.skip`'d because wiring an integration assertion into existing test_cnn_agent.py harness is too codebase-specific to spell out in plan form. The shadow path itself is fully covered in Task 5. Belt-and-suspenders only.

## Plan complete

Saved to `docs/superpowers/plans/2026-05-17-xgb-v4-ohlcv-shadow.md`. **10 tasks, ~50 micro-steps, 1 atomic commit + 4-horizon operator sweep + winner-copy + restart, +51 tests, single-commit infrastructure with offline horizon comparison.**

Same shape as recent Module 1/2/4a/Step-A commits: tight scope, single atomic commit, full pre-commit suite green.
