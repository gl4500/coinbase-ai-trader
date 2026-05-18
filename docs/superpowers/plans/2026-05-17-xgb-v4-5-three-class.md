# XGB v4.5 — 3-Class Trend Model + BB Channels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Switch from binary v4 to 3-class triple-barrier labels (DOWN/NEUTRAL/UP), add 2 Bollinger Band channels (bb_position + bb_width), sweep 3 horizons (h24/h72/h168), evaluate 3 decision rules in comparison report — all in a single atomic commit that runs in shadow on PORT=8002 until operator picks (horizon, rule) at promote time.

**Architecture:** Pure-function `tools/xgb_v4_5_features.py` (7 channels × 3 tiers × 10 stats = 210 features) sibling to v4. New `tools/train_xgb_v4_5.py` orchestrator delegates to small helpers (3-class labeler, BB-aware sample builder needing macro+20 bars per sample, 3-class booster trainer using multi:softprob). `agents/xgb_signal.py` gains `xgb_prob_v4_5` returning `(p_down, p_neutral, p_up)` and `xgb_prob_shadow_v4_5` returning `Tuple[float, Optional[Tuple[float,float,float]]]` with v4.5 isolated in try/except per invariants #16/17. `cnn_scans` gets 3 new nullable REAL columns for shadow telemetry.

**Tech Stack:** Python 3.11, xgboost (multi:softprob), pyarrow parquet, sqlite3/aiosqlite, pytest + pytest-asyncio.

**Spec source:** `docs/superpowers/specs/2026-05-17-xgb-v4-5-three-class-design.md`
**Branch:** `feat/gpu-coord-mirror` (continue)

---

## File map

| Path | Action | Responsibility |
|---|---|---|
| `backend/tools/xgb_v4_5_features.py` | CREATE | Pure-function v4.5 extractor. 7 channels (OHLCV + bb_position + bb_width). Constants derived. ~180 LOC. |
| `backend/tools/train_xgb_v4_5.py` | CREATE | 3-class trainer. CLI args `--forward-hours` + `--label-thresh` REQUIRED. Horizon-suffixed artifacts `xgb_*_v4_5_h<H>.*`. ~320 LOC. |
| `backend/tools/v4_5_horizon_compare.py` | CREATE | Per-class AUC + macro-AUC + 3-rule decision-rule sweep + HTML report at `backend/tools/xgb_v4_5_horizon_compare.html`. ~280 LOC. |
| `backend/migrations/xgb_v4_5_shadow_20260517.py` | CREATE | Idempotent ALTER TABLE adds 3 REAL columns to `cnn_scans`. ~40 LOC. |
| `backend/tests/test_xgb_v4_5_features.py` | CREATE | Constants, BB helpers, extractor shape (1, 210), determinism. ~250 LOC. |
| `backend/tests/test_train_xgb_v4_5.py` | CREATE | 3-class label correctness, sample builder, walk-forward split. ~180 LOC. |
| `backend/tests/test_v4_5_horizon_compare.py` | CREATE | Per-class AUC, decision-rule eval, HTML render. ~120 LOC. |
| `backend/tools/xgb_features.py` | EDIT | +5 LOC v4_5 dispatcher branch. |
| `backend/agents/xgb_signal.py` | EDIT | +110 LOC: v4.5 state, `_try_load_v4_5`, `xgb_prob_v4_5`, `xgb_prob_shadow_v4_5`. v4/v3 paths unchanged. |
| `backend/database.py` | EDIT | +12 LOC: 3 columns in CREATE TABLE + ALTER list + `save_cnn_scan` INSERT. |
| `backend/agents/cnn_agent.py` | EDIT | ~10 LOC: replace `_xgb.xgb_prob_shadow` with `_xgb.xgb_prob_shadow_v4_5`, unpack 3-tuple, add 3 dict entries. NO decision logic changes. |
| `backend/tests/test_xgb_signal.py` | EDIT | +90 LOC v4.5 shadow tests. |
| `backend/tests/test_database.py` | EDIT | +35 LOC persistence tests. |
| `backend/tests/test_mc_migration.py` | EDIT | +30 LOC idempotency tests. |
| `CLAUDE.md` | EDIT | +8 LOC invariant #17 (3-class telemetry contract). |
| `CHANGELOG.md` | EDIT | New Session entry at top. |

Memory sync after commit: `coinbase_trader_session_log.md`.

---

## Coordination

Branch tip at plan-write time: `717720f` (v4.5 spec commit). Single feat branch. Single atomic commit at end of Task 9.

**v4 promote in flight separately** (.venv backend on 8001 with v4 binary shadow + v3 driver) — independent of this plan's work. The two paths coexist: production runs v3+v4 shadow on 8001 throughout this plan's implementation, dev backend runs v4.5 shadow on 8002 after operator preflight.

---

## Task 1: xgb_v4_5_features.py — 7-channel extractor + helpers

**Files:**
- Create: `backend/tools/xgb_v4_5_features.py`
- Create: `backend/tests/test_xgb_v4_5_features.py`

- [ ] **Step 1.1 — Write failing test file**

Create `backend/tests/test_xgb_v4_5_features.py`:

```python
"""Unit tests for backend/tools/xgb_v4_5_features.py — v4.5 7-channel extractor.

5 OHLCV channels + 2 BB channels (bb_position, bb_width) × 3 tiers
× 10 stats = 210 features. Pure functions, no module state.
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

from tools import xgb_v4_5_features as v45  # noqa: E402


# ── Constants ─────────────────────────────────────────────────────────────

class TestConstants:
    def test_channel_names_order(self):
        assert v45._CHANNEL_NAMES == (
            "open", "high", "low", "close", "volume",
            "bb_position", "bb_width",
        )

    def test_n_channels_derived(self):
        assert v45.N_CHANNELS_V45 == 7
        assert v45.N_CHANNELS_V45 == len(v45._CHANNEL_NAMES)

    def test_bb_params(self):
        assert v45.BB_PERIOD == 20
        assert v45.BB_MULT == 2.0

    def test_tier_windows(self):
        assert v45.TIER_WINDOWS_V45 == {"micro": 60, "meso": 168, "macro": 336}

    def test_n_features_derived(self):
        assert v45.N_FEATURES_V45 == 210
        assert v45.N_FEATURES_V45 == v45.N_CHANNELS_V45 * v45.N_TIERS_V45 * v45.N_STATS_V45


# ── BB helpers ────────────────────────────────────────────────────────────

class TestBollingerHelpers:
    def test_bb_position_empty(self):
        out = v45._compute_bb_position(np.array([], dtype=np.float64))
        assert out.shape == (0,)

    def test_bb_position_pre_period_fallback(self):
        # Fewer than 20 bars: each bar gets 0.5 (mid) fallback
        out = v45._compute_bb_position(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert (out == 0.5).all()

    def test_bb_position_clamped_0_1(self):
        # 30 bars of constant 100 → mean=100, std=0, bw=0 → pos fallback 0.5
        closes = np.full(30, 100.0)
        out = v45._compute_bb_position(closes)
        # Pre-period (first 19) get 0.5
        assert (out[:19] == 0.5).all()
        # Post-period get 0.5 because zero std → no spread
        assert (out[19:] == 0.5).all()

    def test_bb_position_known_values(self):
        # 25 bars rising linearly: 80..104
        closes = np.arange(80, 105, dtype=np.float64)
        out = v45._compute_bb_position(closes)
        # Last bar: pos should be > 0.5 (close above mean)
        assert out[-1] > 0.5
        assert 0.0 <= out[-1] <= 1.0

    def test_bb_width_empty(self):
        out = v45._compute_bb_width(np.array([], dtype=np.float64))
        assert out.shape == (0,)

    def test_bb_width_pre_period_fallback(self):
        out = v45._compute_bb_width(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert (out == 0.0).all()

    def test_bb_width_formula(self):
        # Rising series so std > 0
        closes = np.arange(80, 110, dtype=np.float64)
        out = v45._compute_bb_width(closes)
        # Width = (upper - lower) / mean = (4 * std) / mean. Sanity check > 0
        assert out[-1] > 0.0


# ── _compute_stats / _slope / _pct_rank / _delta_at ───────────────────────

class TestStatHelpers:
    def test_compute_stats_known_series(self):
        v = np.arange(1, 11, dtype=np.float64)
        out = v45._compute_stats(v)
        assert out.shape == (10,)
        # last, mean, std, slope, min, max, pct_rank, dlt5, dlt10, dlt30
        assert out[0] == 10.0
        assert out[1] == pytest.approx(5.5)
        assert out[3] == pytest.approx(1.0)
        assert out[4] == 1.0
        assert out[5] == 10.0
        assert out[7] == 10.0 - 5.0

    def test_slope_linear(self):
        assert v45._slope(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_pct_rank_empty_zero(self):
        assert v45._pct_rank(np.array([], dtype=np.float64)) == 0.0

    def test_delta_at_too_short(self):
        assert v45._delta_at(np.array([1.0, 2.0]), lookback=5) == 0.0


# ── feature_names_v4_5 + feature_weights_v4_5 ─────────────────────────────

class TestFeatureNames:
    def test_returns_210_names(self):
        names = v45.feature_names_v4_5()
        assert len(names) == 210

    def test_layout_channel_then_tier_then_stat(self):
        names = v45.feature_names_v4_5()
        assert names[0] == "ch0_micro_last"
        assert names[10] == "ch0_meso_last"
        assert names[20] == "ch0_macro_last"
        assert names[30] == "ch1_micro_last"  # next channel
        # Channel 5 = bb_position, channel 6 = bb_width
        assert names[150] == "ch5_micro_last"
        assert names[180] == "ch6_micro_last"
        assert names[209] == "ch6_macro_dlt30"

    def test_unique(self):
        names = v45.feature_names_v4_5()
        assert len(set(names)) == len(names)


class TestFeatureWeights:
    def test_length_210(self):
        assert len(v45.feature_weights_v4_5()) == 210

    def test_tier_weights(self):
        names = v45.feature_names_v4_5()
        weights = v45.feature_weights_v4_5()
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert weights[i] == 1.0
            elif "_meso_" in name:
                assert weights[i] == 2.0
            elif "_macro_" in name:
                assert weights[i] == 3.0


# ── extract_v4_5 (the main public extractor) ──────────────────────────────

class TestExtractV4_5:
    def _make_candle(self, c: int) -> Dict[str, float]:
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
        features, names = v45.extract_v4_5(candles_by_tier)
        assert features.shape == (1, 210)
        assert features.dtype == np.float64
        assert names == v45.feature_names_v4_5()

    def test_channel_3_reads_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch3_micro_last")
        # Last micro candle = _make_candle(60): close = 60.25
        assert features[0, idx] == 60.25

    def test_channel_5_is_bb_position(self):
        # Rising linear closes → bb_position monotonic toward upper band
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch5_micro_last")
        # bb_position is in [0, 1], not raw OHLCV value
        assert 0.0 <= features[0, idx] <= 1.0

    def test_channel_6_is_bb_width(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch6_micro_last")
        # bb_width >= 0 (zero-spread only when std=0)
        assert features[0, idx] >= 0.0

    def test_empty_tier_zeros_its_slots(self):
        candles_by_tier = {
            "micro": [],
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        # 7 channels × 10 stats = 70 micro slots, all zero
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0, f"{name} should be zero"

    def test_missing_ohlcv_field_raises(self):
        bad = [{"open": 1.0, "high": 2.0, "low": 0.5}]  # missing close/volume
        with pytest.raises(KeyError):
            v45.extract_v4_5({"micro": bad, "meso": [], "macro": []})

    def test_determinism(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso":  self._make_tier(168),
            "macro": self._make_tier(336),
        }
        f1, _ = v45.extract_v4_5(candles_by_tier)
        f2, _ = v45.extract_v4_5(candles_by_tier)
        assert (f1 == f2).all()
```

- [ ] **Step 1.2 — Run; expect ImportError**

```bash
cd C:\Users\gl450\polymarket_app
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_5_features.py -v
```

Expected: collection error `ModuleNotFoundError: No module named 'tools.xgb_v4_5_features'`.

- [ ] **Step 1.3 — Create the implementation**

Create `backend/tools/xgb_v4_5_features.py`:

```python
"""XGB v4.5 7-channel feature extractor.

5 OHLCV channels (open/high/low/close/volume) + 2 Bollinger channels
(bb_position, bb_width) x 3 tiers (micro/meso/macro) x 10 stats = 210 features.

Per feedback_python_clean_functions: type hints, pure data-in/data-out helpers,
derived constants, no in-place buffer mutation.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ── Configuration constants ────────────────────────────────────────────────
_OHLCV_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
_BB_CHANNELS:  Tuple[str, ...] = ("bb_position", "bb_width")
_CHANNEL_NAMES: Tuple[str, ...] = _OHLCV_FIELDS + _BB_CHANNELS
N_CHANNELS_V45: int = len(_CHANNEL_NAMES)  # = 7

TIER_WINDOWS_V45: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V45: Dict[str, float] = {"micro": 1.0, "meso": 2.0, "macro": 3.0}
_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")

BB_PERIOD: int = 20
BB_MULT: float = 2.0

_STAT_NAMES_V45: Tuple[str, ...] = (
    "last", "mean", "std", "slope",
    "min", "max", "pct_rank",
    "dlt5", "dlt10", "dlt30",
)
N_STATS_V45: int = len(_STAT_NAMES_V45)        # = 10
N_TIERS_V45: int = len(TIER_WINDOWS_V45)       # = 3
N_FEATURES_V45: int = N_CHANNELS_V45 * N_TIERS_V45 * N_STATS_V45  # = 210


# ── Public API ─────────────────────────────────────────────────────────────

def feature_names_v4_5() -> List[str]:
    """Return 210 feature names in stable column order.

    Layout: ch{0..6}_{micro|meso|macro}_{stat}, channel-major -> tier-major
    -> stat-major. Channels 0..4 = OHLCV, channel 5 = bb_position, channel 6
    = bb_width.
    """
    names: List[str] = []
    for c in range(N_CHANNELS_V45):
        for tier in _TIER_ORDER:
            for stat in _STAT_NAMES_V45:
                names.append(f"ch{c}_{tier}_{stat}")
    return names


def feature_weights_v4_5() -> np.ndarray:
    """Return 210-long float64 weight vector aligned with feature_names_v4_5().

    Per-tier weights: micro=1.0, meso=2.0, macro=3.0. Same weight for all
    10 stats within one (channel, tier) group.
    """
    weights = np.zeros(N_FEATURES_V45, dtype=np.float64)
    i = 0
    for _c in range(N_CHANNELS_V45):
        for tier in _TIER_ORDER:
            w = TIER_WEIGHTS_V45[tier]
            for _s in range(N_STATS_V45):
                weights[i] = w
                i += 1
    return weights


def extract_v4_5(
    candles_by_tier: Dict[str, Sequence[Dict[str, float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract 210 features from tier-keyed OHLCV candle lists.

    For BB channels (ch5, ch6): each bar in the tier slice gets its
    bb_position and bb_width computed from a trailing BB_PERIOD-bar window
    ending at that bar. Bars with fewer than BB_PERIOD preceding bars get
    (0.5, 0.0) fallback.

    Args:
        candles_by_tier: {"micro": [...], "meso": [...], "macro": [...]} where
            each entry is a candle dict with at minimum the OHLCV keys.

    Returns:
        (features, names) where features is shape (1, 210) float64 and
        names is len-210 list matching feature_names_v4_5().

    Missing/empty tier -> the 70 slots for that tier are zero.
    Missing OHLCV field in a candle -> raises KeyError (input contract).
    """
    out = np.zeros((1, N_FEATURES_V45), dtype=np.float64)
    names = feature_names_v4_5()
    slot = 0
    for c in range(N_CHANNELS_V45):
        field = _CHANNEL_NAMES[c]
        for tier in _TIER_ORDER:
            tier_candles = candles_by_tier.get(tier) or []
            if field in _OHLCV_FIELDS:
                values = _extract_ohlcv_field(tier_candles, field)
            elif field == "bb_position":
                closes = _extract_ohlcv_field(tier_candles, "close")
                values = _compute_bb_position(closes)
            elif field == "bb_width":
                closes = _extract_ohlcv_field(tier_candles, "close")
                values = _compute_bb_width(closes)
            else:  # unreachable; defensive
                values = np.array([], dtype=np.float64)
            stats = _compute_stats(values)
            out[0, slot:slot + N_STATS_V45] = stats
            slot += N_STATS_V45
    return out, names


# ── Internal helpers (pure functions, one responsibility each) ─────────────

def _extract_ohlcv_field(
    candles: Sequence[Dict[str, float]],
    field: str,
) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray.

    Empty input -> empty ndarray. Missing field key -> KeyError (input contract).
    """
    if not candles:
        return np.array([], dtype=np.float64)
    return np.array([candle[field] for candle in candles], dtype=np.float64)


def _compute_bb_position(
    closes: np.ndarray,
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> np.ndarray:
    """Bollinger position in [0, 1] at each bar.

    For bar i: pos = (close[i] - lower) / (upper - lower) where upper/lower
    are computed from the trailing `period` bars ending at bar i. Bars with
    fewer than `period` prior bars get 0.5 (mid) fallback. Bars where the
    band has zero spread (std == 0) also get 0.5.
    """
    n = closes.size
    out = np.full(n, 0.5, dtype=np.float64)
    if n < period:
        return out
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        mean = window.mean()
        std = window.std()
        if std == 0.0:
            out[i] = 0.5
            continue
        upper = mean + mult * std
        lower = mean - mult * std
        bw = upper - lower
        if bw <= 0.0:
            out[i] = 0.5
            continue
        pos = (closes[i] - lower) / bw
        out[i] = max(0.0, min(1.0, pos))
    return out


def _compute_bb_width(
    closes: np.ndarray,
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> np.ndarray:
    """(upper - lower) / mean at each bar.

    Pre-period bars (fewer than `period` prior bars) get 0.0 fallback. Bars
    where mean == 0 also get 0.0.
    """
    n = closes.size
    out = np.zeros(n, dtype=np.float64)
    if n < period:
        return out
    for i in range(period - 1, n):
        window = closes[i - period + 1: i + 1]
        mean = window.mean()
        std = window.std()
        if mean == 0.0:
            out[i] = 0.0
            continue
        out[i] = (2.0 * mult * std) / mean
    return out


def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Return shape-(10,) stats in fixed _STAT_NAMES_V45 order.

    Empty input -> all zeros. No in-place mutation of any caller buffer.
    """
    out = np.zeros(N_STATS_V45, dtype=np.float64)
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

- [ ] **Step 1.4 — Run; expect green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_5_features.py -v
```

Expected: ~30+ tests PASSED.

---

## Task 2: xgb_features.py — v4.5 dispatcher branch

**Files:**
- Modify: `backend/tools/xgb_features.py` (around `extract_features`, line ~254)

- [ ] **Step 2.1 — Write dispatcher tests**

Append at END of `backend/tests/test_xgb_v4_5_features.py`:

```python
# ── Dispatcher integration (xgb_features.extract_features) ────────────────


class TestDispatcherV4_5Branch:
    def test_extract_features_v4_5_routes(self):
        from tools.xgb_features import extract_features
        candles = [{"open": 1.0, "high": 2.0, "low": 0.5,
                    "close": 1.5, "volume": 10.0}]
        cbt = {
            "micro": candles * 60,
            "meso":  candles * 168,
            "macro": candles * 336,
        }
        features, names = extract_features(cbt, feature_set="v4_5")
        assert features.shape == (1, 210)
        assert len(names) == 210
        assert names[0] == "ch0_micro_last"

    def test_extract_features_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features
        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features({}, feature_set="v99")
```

- [ ] **Step 2.2 — Run; expect 2 FAILED**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_5_features.py::TestDispatcherV4_5Branch -v
```

Expected: 2 FAILED — `ValueError: unknown feature_set='v4_5'`.

- [ ] **Step 2.3 — Add the v4_5 branch**

Edit `backend/tools/xgb_features.py`. Find the `extract_features` function (around line 254). The current dispatcher already has v1/v2/v3/v4 branches. Add `v4_5` as the FIRST branch (most specific), updating the error message.

Locate the current body:
```python
def extract_features(
    samples, feature_set: str = "v1"
) -> Tuple[np.ndarray, List[str]]:
    """..."""
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

Insert the v4_5 branch ABOVE the v4 branch and update the docstring + error message:

```python
def extract_features(
    samples, feature_set: str = "v1"
) -> Tuple[np.ndarray, List[str]]:
    """Convert a batch of samples to tabular features.

    feature_set:
        "v1": 270 per-channel stats (back-compat).
        "v2": v1 + 10 cross-channel/temporal addons.
        "v3": tiered mixed-lookback — 350 features, dict input.
        "v4": OHLCV-5 channels × 3 tiers × 10 stats = 150 features, dict input.
        "v4_5": v4 + 2 BB channels = 7 channels × 3 tiers × 10 = 210 features
                (#xgb-v4.5 / Step B.1.5).

    Returns (features, feature_names) where features is float64.
    """
    if feature_set == "v4_5":
        from tools.xgb_v4_5_features import extract_v4_5
        return extract_v4_5(samples)
    if feature_set == "v4":
        from tools.xgb_v4_features import extract_v4
        return extract_v4(samples)
    if feature_set == "v3":
        return _extract_v3(samples)
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1', 'v2', 'v3', 'v4', or 'v4_5'"
        )
```

- [ ] **Step 2.4 — Run; expect 2 GREEN**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_5_features.py::TestDispatcherV4_5Branch -v
```

Expected: 2 PASSED. Then run full v4_5 test file to confirm no regressions:
```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_v4_5_features.py -v
```

---

## Task 3: Migration — 3 columns on cnn_scans

**Files:**
- Create: `backend/migrations/xgb_v4_5_shadow_20260517.py`
- Modify: `backend/tests/test_mc_migration.py` (extend)

- [ ] **Step 3.1 — Write migration tests**

Append at END of `backend/tests/test_mc_migration.py`:

```python
# ── xgb_v4_5_shadow_20260517 ──────────────────────────────────────────────


class TestXgbV4_5ShadowMigration:
    def test_migration_adds_three_columns(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_5_shadow_20260517 import run

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
        for col in ("xgb_prob_v4_5_down", "xgb_prob_v4_5_neutral", "xgb_prob_v4_5_up"):
            assert col in result["added"]
        assert result["already_present"] == []

        c = sqlite3.connect(db)
        cols = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        for col in ("xgb_prob_v4_5_down", "xgb_prob_v4_5_neutral", "xgb_prob_v4_5_up"):
            assert col in cols
        c.close()

    def test_migration_idempotent(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_5_shadow_20260517 import run

        db = str(tmp_path / "test.db")
        c = sqlite3.connect(db)
        c.execute(
            "CREATE TABLE cnn_scans ("
            " id INTEGER PRIMARY KEY, product_id TEXT, scanned_at INTEGER"
            ")"
        )
        c.commit()
        c.close()

        r1 = run(db)
        assert len(r1["added"]) == 3
        r2 = run(db)
        assert r2["added"] == []
        for col in ("xgb_prob_v4_5_down", "xgb_prob_v4_5_neutral", "xgb_prob_v4_5_up"):
            assert col in r2["already_present"]
```

- [ ] **Step 3.2 — Run; expect ModuleNotFoundError**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_mc_migration.py::TestXgbV4_5ShadowMigration -v
```

Expected: 2 FAILED with `ModuleNotFoundError: No module named 'migrations.xgb_v4_5_shadow_20260517'`.

- [ ] **Step 3.3 — Create migration**

Create `backend/migrations/xgb_v4_5_shadow_20260517.py`:

```python
"""Migration: add XGB v4.5 3-class shadow telemetry columns (#xgb-v4.5 / Step B.1.5).

Adds three nullable REAL columns to cnn_scans:
  xgb_prob_v4_5_down REAL
  xgb_prob_v4_5_neutral REAL
  xgb_prob_v4_5_up REAL

All three must be written together or all NULL (per CLAUDE.md invariant #17).

Idempotent — safe to re-run. Matches the pattern of mc_telemetry_20260516.py
and xgb_v4_shadow_20260517.py.
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_v4_5_{down,neutral,up} REAL columns to cnn_scans if absent.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [
        ("xgb_prob_v4_5_down",    "REAL"),
        ("xgb_prob_v4_5_neutral", "REAL"),
        ("xgb_prob_v4_5_up",      "REAL"),
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

- [ ] **Step 3.4 — Run; expect 2 GREEN**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_mc_migration.py::TestXgbV4_5ShadowMigration -v
```

Expected: 2 PASSED.

---

## Task 4: database.py — save_cnn_scan persists 3 v4.5 probs

**Files:**
- Modify: `backend/database.py` (CREATE TABLE around line 152, ALTER list line 283, save_cnn_scan INSERT line 552+)
- Modify: `backend/tests/test_database.py` (extend)

- [ ] **Step 4.1 — Write persistence tests**

Append at END of `backend/tests/test_database.py`:

```python
# ── xgb_prob_v4_5_{down,neutral,up} persistence (#xgb-v4.5 / Step B.1.5) ──


class TestSaveCnnScanV4_5Cols:
    @pytest.mark.asyncio
    async def test_save_cnn_scan_persists_three_v45_probs(self, db_module, tmp_path):
        # Reuses the existing db_module fixture (sets _DB_PATH to temp file)
        from database import save_cnn_scan, init_db
        import sqlite3

        await init_db()
        await save_cnn_scan({
            "product_id": "BTC-USD",
            "model_prob": 0.6, "cnn_prob": 0.6, "llm_prob": None,
            "regime": "TRENDING", "side": "BUY",
            "cnn_weight": 1.0, "llm_weight": 0.0,
            "rsi": 50.0, "macd_h": 0.0, "bb_pos": 0.5,
            "vwap_dist": 0.0, "fast_rsi": 0.5, "velocity": 0.5, "vol_z": 0.5,
            "xgb_prob": 0.55, "scanned_at": 1700000000,
            "xgb_prob_v4_5_down":    0.21,
            "xgb_prob_v4_5_neutral": 0.34,
            "xgb_prob_v4_5_up":      0.45,
        })
        # Note: connect via the same path the fixture set
        from database import _DB_PATH
        c = sqlite3.connect(_DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up "
            "FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row is not None
        assert row[0] == pytest.approx(0.21)
        assert row[1] == pytest.approx(0.34)
        assert row[2] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_save_cnn_scan_v45_default_null(self, db_module, tmp_path):
        from database import save_cnn_scan, init_db, _DB_PATH
        import sqlite3

        await init_db()
        await save_cnn_scan({
            "product_id": "BTC-USD",
            "model_prob": 0.6, "cnn_prob": 0.6, "llm_prob": None,
            "regime": "TRENDING", "side": "BUY",
            "cnn_weight": 1.0, "llm_weight": 0.0,
            "rsi": 50.0, "macd_h": 0.0, "bb_pos": 0.5,
            "vwap_dist": 0.0, "fast_rsi": 0.5, "velocity": 0.5, "vol_z": 0.5,
            "xgb_prob": 0.55, "scanned_at": 1700000000,
        })
        c = sqlite3.connect(_DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up "
            "FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None
```

Implementer note: the existing `test_database.py` already has a `db_module` (or similar) fixture pattern from prior v4 tests; reuse that. If the pattern is slightly different, adapt to whatever isolation pattern the existing `TestSaveCnnScanXgbProbV4` class uses.

- [ ] **Step 4.2 — Run; expect 2 FAILED (no such column)**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py::TestSaveCnnScanV4_5Cols -v
```

Expected: 2 FAILED with `sqlite3.OperationalError: no such column: xgb_prob_v4_5_down`.

- [ ] **Step 4.3 — Add 3 columns to CREATE TABLE**

Edit `backend/database.py`. Find around line 152:
```python
                xgb_prob_stdev REAL,
                mc_telemetry TEXT,
                xgb_prob_v4 REAL
```

Replace with (insert 3 new columns after `xgb_prob_v4`, keep `xgb_prob_v4` line trailing comma):
```python
                xgb_prob_stdev REAL,
                mc_telemetry TEXT,
                xgb_prob_v4 REAL,
                xgb_prob_v4_5_down REAL,
                xgb_prob_v4_5_neutral REAL,
                xgb_prob_v4_5_up REAL
```

- [ ] **Step 4.4 — Add 3 ALTER TABLE statements to migration list**

Edit `backend/database.py` around line 283. After:
```python
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4 REAL",
```

Insert:
```python
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4_5_down REAL",
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4_5_neutral REAL",
            "ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4_5_up REAL",
```

- [ ] **Step 4.5 — Add 3 columns to save_cnn_scan INSERT**

Edit `backend/database.py`. Find the `save_cnn_scan` INSERT around line 552-580. The current INSERT looks like:
```python
            """INSERT INTO cnn_scans
            (product_id, model_prob, cnn_prob, llm_prob, regime, side,
                cnn_weight, llm_weight, rsi, macd_h, bb_pos, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at,
                xgb_prob_stdev, mc_telemetry, xgb_prob_v4)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan["product_id"], scan["model_prob"], scan["cnn_prob"],
                scan["llm_prob"], scan["regime"], scan["side"],
                scan["cnn_weight"], scan["llm_weight"],
                scan["rsi"], scan["macd_h"], scan["bb_pos"], scan["vwap_dist"],
                scan["fast_rsi"], scan["velocity"], scan["vol_z"],
                scan.get("xgb_prob"),
                scan.get("scanned_at"),
                scan.get("xgb_prob_stdev"), scan.get("mc_telemetry"),
                scan.get("xgb_prob_v4"),
            ),
```

Replace the column list, VALUES placeholders, and tuple with:
```python
            """INSERT INTO cnn_scans
            (product_id, model_prob, cnn_prob, llm_prob, regime, side,
                cnn_weight, llm_weight, rsi, macd_h, bb_pos, vwap_dist,
                fast_rsi, velocity, vol_z, xgb_prob, scanned_at,
                xgb_prob_stdev, mc_telemetry, xgb_prob_v4,
                xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scan["product_id"], scan["model_prob"], scan["cnn_prob"],
                scan["llm_prob"], scan["regime"], scan["side"],
                scan["cnn_weight"], scan["llm_weight"],
                scan["rsi"], scan["macd_h"], scan["bb_pos"], scan["vwap_dist"],
                scan["fast_rsi"], scan["velocity"], scan["vol_z"],
                scan.get("xgb_prob"),
                scan.get("scanned_at"),
                scan.get("xgb_prob_stdev"), scan.get("mc_telemetry"),
                scan.get("xgb_prob_v4"),
                scan.get("xgb_prob_v4_5_down"),
                scan.get("xgb_prob_v4_5_neutral"),
                scan.get("xgb_prob_v4_5_up"),
            ),
```

3 new `?` placeholders, 3 new tuple elements via `scan.get(...)`. Read lines 552-580 first to confirm exact existing structure.

- [ ] **Step 4.6 — Run; expect 2 GREEN + no regressions**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py::TestSaveCnnScanV4_5Cols -v
.venv/Scripts/python.exe -m pytest backend/tests/test_database.py -v
```

Expected: 2 + all existing PASSED.

---

## Task 5: xgb_signal.py — 3-class shadow path

**Files:**
- Modify: `backend/agents/xgb_signal.py` (add v4.5 loader + xgb_prob_v4_5 + xgb_prob_shadow_v4_5)
- Modify: `backend/tests/test_xgb_signal.py` (extend)

- [ ] **Step 5.1 — Write shadow tests**

Append at END of `backend/tests/test_xgb_signal.py`:

```python
# ── v4.5 shadow path (#xgb-v4.5 / Step B.1.5) ─────────────────────────────


class TestV4_5ShadowLoad:
    def test_try_load_v4_5_false_when_artifacts_missing(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_MODEL_PATH_V45", "/nonexistent/v45.json")
        monkeypatch.setattr(xs, "_FEATURES_PATH_V45", "/nonexistent/v45f.json")
        monkeypatch.setattr(xs, "_load_attempted_v45", False)
        monkeypatch.setattr(xs, "_load_succeeded_v45", False)
        assert xs._try_load_v4_5() is False


class TestXgbProbV4_5:
    def test_neutral_fallback_when_missing(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_MODEL_PATH_V45", "/nope/v45.json")
        monkeypatch.setattr(xs, "_FEATURES_PATH_V45", "/nope/v45f.json")
        monkeypatch.setattr(xs, "_load_attempted_v45", False)
        monkeypatch.setattr(xs, "_load_succeeded_v45", False)
        out = xs.xgb_prob_v4_5(channels=None, pid="BTC-USD")
        assert isinstance(out, tuple)
        assert len(out) == 3
        # Neutral fallback: (0.33, 0.34, 0.33) — sums to 1.0
        assert out == pytest.approx((0.33, 0.34, 0.33))

    def test_returns_tuple_when_pid_none(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "_load_attempted_v45", True)
        monkeypatch.setattr(xs, "_load_succeeded_v45", True)
        out = xs.xgb_prob_v4_5(channels=None, pid=None)
        # pid=None for v4.5 (tiered fetch needed) -> neutral fallback
        assert out == pytest.approx((0.33, 0.34, 0.33))


class TestXgbProbShadowV4_5:
    def test_returns_tuple_shape(self, monkeypatch):
        import agents.xgb_signal as xs
        monkeypatch.setattr(xs, "xgb_prob",
                            lambda channels, pid=None: 0.7)
        monkeypatch.setattr(xs, "xgb_prob_v4_5",
                            lambda channels, pid=None: (0.2, 0.3, 0.5))
        v3, v45 = xs.xgb_prob_shadow_v4_5(channels=None, pid="BTC-USD")
        assert v3 == 0.7
        assert v45 == (0.2, 0.3, 0.5)

    def test_v45_failure_isolated_from_v3(self, monkeypatch, caplog):
        import logging
        import agents.xgb_signal as xs

        def boom(*a, **kw):
            raise RuntimeError("v4.5 boom")

        monkeypatch.setattr(xs, "xgb_prob",
                            lambda channels, pid=None: 0.6)
        monkeypatch.setattr(xs, "xgb_prob_v4_5", boom)
        with caplog.at_level(logging.ERROR):
            v3, v45 = xs.xgb_prob_shadow_v4_5(channels=None, pid="BTC-USD")
        assert v3 == 0.6
        assert v45 is None
        assert any("v4.5" in r.message.lower() or "v4_5" in r.message.lower()
                   for r in caplog.records)

    def test_v3_failure_propagates(self, monkeypatch):
        import agents.xgb_signal as xs

        def boom_v3(*a, **kw):
            raise RuntimeError("v3 boom")

        monkeypatch.setattr(xs, "xgb_prob", boom_v3)
        monkeypatch.setattr(xs, "xgb_prob_v4_5",
                            lambda channels, pid=None: (0.3, 0.3, 0.4))
        with pytest.raises(RuntimeError, match="v3 boom"):
            xs.xgb_prob_shadow_v4_5(channels=None, pid="BTC-USD")
```

- [ ] **Step 5.2 — Run; expect FAILED (AttributeError)**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py::TestV4_5ShadowLoad backend/tests/test_xgb_signal.py::TestXgbProbV4_5 backend/tests/test_xgb_signal.py::TestXgbProbShadowV4_5 -v
```

Expected: 6+ FAILED with `AttributeError: module 'agents.xgb_signal' has no attribute '_try_load_v4_5'`.

- [ ] **Step 5.3 — Add v4.5 module state**

Edit `backend/agents/xgb_signal.py`. After the existing v4 state block (look for `_load_succeeded_v4: bool = False`), add the v4.5 state:

```python
# ── v4.5 shadow state (Step B.1.5) ────────────────────────────────────────
_MODEL_PATH_V45    = os.path.join(_BACKEND_DIR, "xgb_model_v4_5.json")
_FEATURES_PATH_V45 = os.path.join(_BACKEND_DIR, "xgb_features_v4_5.json")
# No calibration path — v4.5 uses raw softmax (see spec)

_booster_v45 = None
_feature_names_v45: List[str] = []
_load_attempted_v45: bool = False
_load_succeeded_v45: bool = False
```

- [ ] **Step 5.4 — Add _try_load_v4_5**

After the existing `_try_load_v4()` function, add:

```python
def _try_load_v4_5() -> bool:
    """Load v4.5 booster + feature_names from disk once. Idempotent.

    Returns True iff load succeeded. Failures log + return False; never raise.
    No calibrator in v4.5 (raw softmax used directly).
    """
    global _booster_v45, _feature_names_v45
    global _load_attempted_v45, _load_succeeded_v45
    with _lock:
        if _load_attempted_v45:
            return _load_succeeded_v45
        _load_attempted_v45 = True
        if not (os.path.exists(_MODEL_PATH_V45) and os.path.exists(_FEATURES_PATH_V45)):
            logger.info(
                "xgb_signal: v4.5 artifacts missing (model=%s features=%s) — shadow disabled",
                _MODEL_PATH_V45, _FEATURES_PATH_V45,
            )
            return False
        try:
            import xgboost as xgb
            with open(_FEATURES_PATH_V45, "r") as f:
                meta = json.load(f)
            names = list(meta.get("feature_names", []))
            if not names:
                logger.warning("xgb_signal: v4.5 features.json has empty feature_names")
                return False
            booster = xgb.Booster()
            booster.load_model(_MODEL_PATH_V45)
            _booster_v45 = booster
            _feature_names_v45 = names
            _load_succeeded_v45 = True
            logger.info("xgb_signal: loaded v4.5 booster (%d features)", len(names))
            return True
        except Exception as exc:
            logger.exception("xgb_signal: v4.5 load failed: %s", exc)
            return False
```

- [ ] **Step 5.5 — Add xgb_prob_v4_5**

After `_try_load_v4_5`, add:

```python
def xgb_prob_v4_5(
    channels, pid: Optional[str] = None,
) -> Tuple[float, float, float]:
    """v4.5 3-class probabilities (p_down, p_neutral, p_up).

    Each clipped to [0.01, 0.99] then renormalized to sum to 1.0. Returns
    neutral fallback (0.33, 0.34, 0.33) if artifacts missing, pid is None,
    or any error during inference.
    """
    _NEUTRAL_3 = (0.33, 0.34, 0.33)
    if not _try_load_v4_5():
        return _NEUTRAL_3
    if pid is None:
        logger.warning(
            "xgb_signal: v4.5 requires pid, got None — returning neutral 3-tuple",
        )
        return _NEUTRAL_3
    try:
        import xgboost as xgb
        from services.tiered_history import fetch_tiered
        from tools.xgb_v4_5_features import extract_v4_5

        tiers = fetch_tiered(pid, source="live")
        features, _ = extract_v4_5(tiers)
        dmat = xgb.DMatrix(features, feature_names=_feature_names_v45)
        raw = _booster_v45.predict(dmat)
        # multi:softprob returns shape (1, 3)
        if raw.ndim != 2 or raw.shape != (1, 3):
            logger.warning(
                "xgb_signal: v4.5 booster output shape %s, expected (1, 3) — neutral",
                raw.shape,
            )
            return _NEUTRAL_3
        p_down    = float(np.clip(raw[0, 0], 0.01, 0.99))
        p_neutral = float(np.clip(raw[0, 1], 0.01, 0.99))
        p_up      = float(np.clip(raw[0, 2], 0.01, 0.99))
        total = p_down + p_neutral + p_up
        if total <= 0.0:
            return _NEUTRAL_3
        # Renormalize after clip so probs still sum to 1.0
        return (p_down / total, p_neutral / total, p_up / total)
    except Exception as exc:
        logger.exception("xgb_signal.xgb_prob_v4_5 failed, returning neutral: %s", exc)
        return _NEUTRAL_3
```

- [ ] **Step 5.6 — Add xgb_prob_shadow_v4_5**

After `xgb_prob_v4_5`, add:

```python
def xgb_prob_shadow_v4_5(
    channels, pid: Optional[str] = None,
) -> Tuple[float, Optional[Tuple[float, float, float]]]:
    """Return (v3_prob, v4_5_3prob_tuple_or_None).

    v3 path runs normally (its own exception handling returns neutral 0.5 on
    failure — no try/except wrapper here). v4.5 wrapped in try/except: any
    failure -> v4_5=None + log, NEVER affects v3. This is the function
    cnn_agent should call during the v4.5 shadow week.
    """
    prob_v3 = xgb_prob(channels, pid=pid)
    try:
        prob_v45 = xgb_prob_v4_5(channels, pid=pid)
    except Exception as exc:
        logger.exception(
            "xgb_signal.xgb_prob_shadow_v4_5: v4.5 path raised (isolated): %s", exc,
        )
        prob_v45 = None
    return prob_v3, prob_v45
```

- [ ] **Step 5.7 — Run; expect 6+ GREEN + no regressions**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py::TestV4_5ShadowLoad backend/tests/test_xgb_signal.py::TestXgbProbV4_5 backend/tests/test_xgb_signal.py::TestXgbProbShadowV4_5 -v
.venv/Scripts/python.exe -m pytest backend/tests/test_xgb_signal.py -v
```

Expected: 6+ PASSED in the v4.5 tests, full file green.

---

## Task 6: cnn_agent.py — write-through 3 v4.5 probs to save_cnn_scan

**Files:**
- Modify: `backend/agents/cnn_agent.py` (line ~1903 — replace shadow call; line ~2014 — add 3 dict entries)

- [ ] **Step 6.1 — Inspect current call site**

Read `backend/agents/cnn_agent.py:1893-2020` to confirm:
- Around line 1903: existing call `xgb_shadow, xgb_shadow_v4 = _xgb.xgb_prob_shadow(...)` from B.1
- Around line ~2014: existing dict entries `"xgb_prob": ...`, `"xgb_prob_v4": ...`

- [ ] **Step 6.2 — Replace xgb_prob_shadow with xgb_prob_shadow_v4_5**

Edit `backend/agents/cnn_agent.py`. Find the existing B.1 line:
```python
                    xgb_shadow, xgb_shadow_v4 = _xgb.xgb_prob_shadow(
                        _mask_training_constant_channels(channels),
                        pid=pid,
                    )
```

Replace with:
```python
                    xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(
                        _mask_training_constant_channels(channels),
                        pid=pid,
                    )
```

Variable name changes: `xgb_shadow_v4` → `xgb_shadow_v45` (a 3-tuple OR None instead of single float OR None).

- [ ] **Step 6.3 — Update the except branch initialization**

In the surrounding try/except (around line 1907-1912), find:
```python
                except Exception:
                    xgb_shadow = None
                    xgb_shadow_v4 = None
```

Replace with:
```python
                except Exception:
                    xgb_shadow = None
                    xgb_shadow_v45 = None
```

And find any earlier init (around line ~1777):
```python
                xgb_shadow_v4: Optional[float] = None
```

Replace with:
```python
                xgb_shadow_v45: Optional[Tuple[float, float, float]] = None
```

(Note: `Tuple` should already be imported at top of cnn_agent.py from prior changes.)

- [ ] **Step 6.4 — Update save_cnn_scan dict to write 3 v4.5 probs (and DROP xgb_prob_v4 since v4 path is no longer called)**

Find around line 2010-2014:
```python
            "xgb_prob":    round(xgb_shadow, 4) if xgb_shadow is not None else None,
            "xgb_prob_v4": round(xgb_shadow_v4, 4) if xgb_shadow_v4 is not None else None,
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev") if mc_telemetry else None,
```

Replace with:
```python
            "xgb_prob":    round(xgb_shadow, 4) if xgb_shadow is not None else None,
            "xgb_prob_v4_5_down":    round(xgb_shadow_v45[0], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_v4_5_neutral": round(xgb_shadow_v45[1], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_v4_5_up":      round(xgb_shadow_v45[2], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev") if mc_telemetry else None,
```

Notes:
- We DROP `xgb_prob_v4` from this dict because v4 path is no longer called from cnn_agent (v4.5 supersedes per spec).
- Per CLAUDE.md invariant #17: all 3 v4.5 probs written together (all from same tuple) or all NULL. The `if xgb_shadow_v45 is not None else None` triple satisfies this.
- The DB column `xgb_prob_v4` stays in the schema; `save_cnn_scan` will write `None` for it via `scan.get("xgb_prob_v4")` returning None. That's correct.

- [ ] **Step 6.5 — Run cnn_agent + xgb_signal + database test suites**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_cnn_agent.py backend/tests/test_xgb_signal.py backend/tests/test_database.py -v
```

Expected: all green. Note: any test that asserted on `xgb_prob_v4` being populated by cnn_agent will need updating — those should now expect `None` (since v4 shadow path is no longer called). Update minimally only if a test breaks; otherwise leave alone.

---

## Task 7: train_xgb_v4_5.py — 3-class trainer

**Files:**
- Create: `backend/tools/train_xgb_v4_5.py`
- Create: `backend/tests/test_train_xgb_v4_5.py`

- [ ] **Step 7.1 — Write helper tests**

Create `backend/tests/test_train_xgb_v4_5.py`:

```python
"""Unit tests for backend/tools/train_xgb_v4_5.py helpers.

Tests pure helpers (_triple_barrier_label_3class, _build_samples_for_pid,
_walk_forward_split) on synthetic candles. Orchestrator main() is
exercised by operator-run smoke test post-commit.
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
    """Synthetic OHLCV with linear drift."""
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


class TestTripleBarrierLabel3Class:
    def test_up_breach_returns_2(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # close[start]=100, threshold=0.01, forward 4 bars
        # close[start+1]=101.5 -> +1.5% > 1% -> UP breach (returns 2)
        closes = np.array([100.0, 101.5, 100.0, 99.0, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 2

    def test_down_breach_returns_0(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        closes = np.array([100.0, 98.5, 99.0, 100.0, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 0

    def test_no_breach_returns_1_neutral(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # No bar exceeds +/-1%
        closes = np.array([100.0, 100.5, 99.5, 100.5, 100.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) == 1

    def test_tie_up_wins(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        # First bar after start hits +exact threshold; subsequent bar would hit -
        # UP barrier checked before DOWN inside the loop -> UP wins
        closes = np.array([100.0, 101.0, 99.0])
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=2, label_thresh=0.01,
        ) == 2

    def test_truncated_returns_none(self):
        from tools.train_xgb_v4_5 import _triple_barrier_label_3class
        closes = np.array([100.0, 101.0])  # only 1 forward bar, need 4
        assert _triple_barrier_label_3class(
            closes, start=0, forward_hours=4, label_thresh=0.01,
        ) is None


class TestBuildSamplesForPid:
    def test_empty_candles_returns_empty_arrays(self):
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        X, y, ts = _build_samples_for_pid(
            [], label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 210)
        assert y.shape == (0,)
        assert ts.shape == (0,)

    def test_too_few_candles_returns_empty(self):
        """Need at least macro + BB_PREFIX + forward_hours candles."""
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(100)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape == (0, 210)

    def test_returns_correct_feature_width(self):
        """500+ candles with drift -> some samples produced."""
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(500, drift=0.05)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert X.shape[1] == 210
        assert X.shape[0] == y.shape[0] == ts.shape[0]
        assert X.shape[0] > 0

    def test_labels_in_valid_set(self):
        from tools.train_xgb_v4_5 import _build_samples_for_pid
        candles = _make_candles(500, drift=0.05)
        X, y, ts = _build_samples_for_pid(
            candles, label_thresh=0.015, forward_hours=24,
            micro=60, meso=168, macro=336,
        )
        assert set(np.unique(y).tolist()).issubset({0, 1, 2})


class TestWalkForwardSplit:
    def test_splits_into_three_chronological_groups(self):
        from tools.train_xgb_v4_5 import _walk_forward_split
        n = 1000
        X = np.random.rand(n, 210)
        y = np.random.randint(0, 3, n)
        ts = np.arange(n, dtype=np.int64) + 1700000000

        (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
            X, y, ts, embargo_bars=24, val_frac=0.15, cal_frac=0.15,
        )
        assert X_tr.shape[0] > 0
        assert X_va.shape[0] > 0
        assert X_ca.shape[0] > 0
        total = X_tr.shape[0] + X_va.shape[0] + X_ca.shape[0]
        assert total <= n
```

- [ ] **Step 7.2 — Run; expect ModuleNotFoundError**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_train_xgb_v4_5.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.train_xgb_v4_5'`.

- [ ] **Step 7.3 — Create the trainer**

Create `backend/tools/train_xgb_v4_5.py`:

```python
"""XGB v4.5 3-class trainer (#xgb-v4.5 / Step B.1.5).

Reads OHLCV per pid from backend/data/history/<pid>.parquet. Builds 3-class
triple-barrier labels (DOWN=0, NEUTRAL=1, UP=2) at CLI-specified
--forward-hours / --label-thresh. Walk-forward splits chronologically.
Trains v4.5 booster (multi:softprob, num_class=3) on 7-channel features
(OHLCV + bb_pos + bb_width = 210 cols). Writes horizon-suffixed artifacts
at backend/xgb_*_v4_5_h<HOURS>.* paths. No calibrator in v4.5 (raw softmax).

Per feedback_python_clean_functions: main() delegates to small
single-responsibility helpers, each pure data-in/data-out.

Run (horizon sweep — operator runs 3 times, then v4_5_horizon_compare):
    cd backend && python -m tools.train_xgb_v4_5 \
      --pids BTC-USD,ETH-USD,... --forward-hours 24 --label-thresh 0.015
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.xgb_v4_5_features import (  # noqa: E402
    extract_v4_5, feature_names_v4_5, feature_weights_v4_5,
    N_FEATURES_V45, TIER_WINDOWS_V45, BB_PERIOD,
)

logger = logging.getLogger(__name__)

_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_DIR = BACKEND
# No defaults for --forward-hours / --label-thresh; operator MUST specify per
# the horizon sweep workflow. See spec "Architecture decisions".
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


def _triple_barrier_label_3class(
    closes: np.ndarray,
    start: int,
    forward_hours: int,
    label_thresh: float,
) -> Optional[int]:
    """3-class triple-barrier label.

    Returns:
        2 (UP)      if any forward close >= entry * (1 + label_thresh) hit first
        0 (DOWN)    if any forward close <= entry * (1 - label_thresh) hit first
        1 (NEUTRAL) if neither barrier hit within window (vertical timeout)
        None        if window truncated (start + forward_hours >= len(closes))

    Tie-break: UP barrier is checked before DOWN within each bar, so a bar
    that simultaneously crosses both gets UP (favors the actionable signal).
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
            return 2
        if c <= dn_thr:
            return 0
    return 1


def _build_samples_for_pid(
    candles: List[Dict[str, float]],
    *,
    label_thresh: float,
    forward_hours: int,
    micro: int,
    meso: int,
    macro: int,
    bb_prefix: int = BB_PERIOD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each valid bar i where i >= macro+bb_prefix AND a label can be
    computed, produce one (features [210], int8 label, int64 timestamp).

    Returns:
        features:   (N, 210) float64
        labels:     (N,) int8 (0=DOWN, 1=NEUTRAL, 2=UP)
        timestamps: (N,) int64 (epoch seconds at sample bar)
    """
    n = len(candles)
    min_needed = macro + bb_prefix + forward_hours + 1
    if n < min_needed:
        return (np.zeros((0, N_FEATURES_V45), dtype=np.float64),
                np.zeros(0, dtype=np.int8),
                np.zeros(0, dtype=np.int64))
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    feats_list: List[np.ndarray] = []
    labels_list: List[int] = []
    ts_list: List[int] = []
    for i in range(macro + bb_prefix, n):
        label = _triple_barrier_label_3class(
            closes, i, forward_hours, label_thresh,
        )
        if label is None:
            continue
        tier_slices = {
            # Include bb_prefix bars BEFORE each tier slice so bb_position
            # can be computed at every bar in the slice (the prefix bars
            # are used only for BB calculation, not for stats — _compute_stats
            # ignores prefix because it sees only the trailing tier_window).
            "micro": candles[i - micro - bb_prefix:i],
            "meso":  candles[i - meso - bb_prefix:i],
            "macro": candles[i - macro - bb_prefix:i],
        }
        feats, _ = extract_v4_5(tier_slices)
        feats_list.append(feats[0])
        labels_list.append(label)
        ts_list.append(candles[i]["start"])
    if not feats_list:
        return (np.zeros((0, N_FEATURES_V45), dtype=np.float64),
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
    """Chronological split (train, val, cal) with embargo gaps."""
    n = features.shape[0]
    cal_n = int(n * cal_frac)
    val_n = int(n * val_frac)
    train_end = n - val_n - cal_n - 2 * embargo_bars
    if train_end < 1:
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


def _train_booster_3class(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], feature_weights: np.ndarray,
):
    """Train one 3-class xgb.Booster (multi:softprob). Returns booster +
    val mlogloss."""
    import xgboost as xgb

    d_tr = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    d_tr.set_info(feature_weights=feature_weights)
    d_va = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    d_va.set_info(feature_weights=feature_weights)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
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
    val_pred = booster.predict(d_va)  # shape (N, 3)
    # mlogloss as quick sanity metric — full AUC per class in compare
    from sklearn.metrics import log_loss
    if len(set(y_val.tolist())) >= 2:
        val_mlogloss = float(log_loss(y_val, val_pred, labels=[0, 1, 2]))
    else:
        val_mlogloss = float("nan")
    return booster, val_mlogloss


def _save_artifacts(
    booster,
    feature_names: List[str],
    out_dir: str,
    *,
    forward_hours: int,
) -> Dict[str, str]:
    """Atomic write of model.json + features.json with horizon suffix.
    No calibrator file in v4.5 (raw softmax used)."""
    import json

    os.makedirs(out_dir, exist_ok=True)
    suffix = f"_h{forward_hours}"
    model_path = os.path.join(out_dir, f"xgb_model_v4_5{suffix}.json")
    feat_path  = os.path.join(out_dir, f"xgb_features_v4_5{suffix}.json")

    # tmp paths: model tmp MUST end in .json (xgboost format auto-detection)
    model_tmp = os.path.join(out_dir, f"xgb_model_v4_5{suffix}.tmp.json")
    feat_tmp  = feat_path + ".tmp"

    booster.save_model(model_tmp)
    os.replace(model_tmp, model_path)

    with open(feat_tmp, "w") as f:
        json.dump({"feature_names": feature_names, "feature_set": "v4_5"}, f)
    os.replace(feat_tmp, feat_path)

    return {"model": model_path, "features": feat_path}


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pids", required=True,
                   help="comma-separated, e.g. BTC-USD,ETH-USD")
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    p.add_argument("--forward-hours", type=int, required=True,
                   help="label horizon in bars (24, 72, 168 per sweep)")
    p.add_argument("--label-thresh", type=float, required=True,
                   help="triple-barrier threshold (e.g. 0.015, 0.03, 0.06)")
    p.add_argument("--embargo-bars", type=int, default=0,
                   help="defaults to forward_hours if 0")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    micro = TIER_WINDOWS_V45["micro"]
    meso  = TIER_WINDOWS_V45["meso"]
    macro = TIER_WINDOWS_V45["macro"]
    embargo = args.embargo_bars if args.embargo_bars > 0 else args.forward_hours

    t0 = time.time()
    print(f"v4.5 train: pids={pids} forward_hours={args.forward_hours} "
          f"label_thresh={args.label_thresh} embargo_bars={embargo} "
          f"-> xgb_*_v4_5_h{args.forward_hours}.*", flush=True)

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
        # Distribution check — for 3-class want all 3 classes represented
        cls_counts = {c: int((y == c).sum()) for c in (0, 1, 2)}
        print(f"  {pid}: {X.shape[0]:,} samples, class counts={cls_counts}",
              flush=True)

    if not all_X:
        print("ERROR: no usable pids", flush=True)
        return 1

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    t = np.concatenate(all_t)
    order = np.argsort(t, kind="stable")
    X = X[order]; y = y[order]; t = t[order]
    cls_counts = {c: int((y == c).sum()) for c in (0, 1, 2)}
    print(f"\nPooled: X={X.shape} class counts={cls_counts}", flush=True)

    (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(
        X, y, t, embargo_bars=embargo,
    )
    print(f"Split: train={X_tr.shape} val={X_va.shape} cal={X_ca.shape}",
          flush=True)

    names = feature_names_v4_5()
    weights = feature_weights_v4_5()
    booster, val_mlogloss = _train_booster_3class(
        X_tr, y_tr, X_va, y_va, names, weights,
    )
    print(f"Train done: val_mlogloss={val_mlogloss:.4f}", flush=True)

    paths = _save_artifacts(
        booster, names, args.out_dir, forward_hours=args.forward_hours,
    )
    print(f"Wrote: {paths}", flush=True)
    print(f"Skipped pids: {skipped}", flush=True)
    print(f"Total wall: {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.4 — Run; expect helper tests green**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_train_xgb_v4_5.py -v
```

Expected: 9+ PASSED.

---

## Task 7b: v4_5_horizon_compare.py — per-class AUC + decision-rule sweep

**Files:**
- Create: `backend/tools/v4_5_horizon_compare.py`
- Create: `backend/tests/test_v4_5_horizon_compare.py`

- [ ] **Step 7b.1 — Write tests**

Create `backend/tests/test_v4_5_horizon_compare.py`:

```python
"""Unit tests for backend/tools/v4_5_horizon_compare.py."""
from __future__ import annotations
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestEvaluateOnHoldout3Class:
    def test_returns_metrics_dict(self):
        from tools.v4_5_horizon_compare import _evaluate_on_holdout_3class

        class _StubBooster:
            def predict(self, dmat):
                # 4 samples, 3 classes — peaked at correct labels
                return np.array([
                    [0.7, 0.2, 0.1],  # DOWN
                    [0.1, 0.2, 0.7],  # UP
                    [0.2, 0.7, 0.1],  # NEUTRAL
                    [0.1, 0.1, 0.8],  # UP
                ])

        X = np.zeros((4, 210), dtype=np.float64)
        y = np.array([0, 2, 1, 2], dtype=np.int8)
        names = [f"col{i}" for i in range(210)]
        out = _evaluate_on_holdout_3class(_StubBooster(), X, y, names)
        for k in ("auc_down", "auc_neutral", "auc_up", "auc_macro",
                  "logloss", "n_samples",
                  "pos_frac_down", "pos_frac_neutral", "pos_frac_up"):
            assert k in out
        assert out["n_samples"] == 4
        assert out["pos_frac_down"] == 0.25
        assert out["pos_frac_up"] == 0.5

    def test_single_class_returns_nan_macro(self):
        from tools.v4_5_horizon_compare import _evaluate_on_holdout_3class

        class _StubBooster:
            def predict(self, dmat):
                return np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])

        X = np.zeros((2, 210), dtype=np.float64)
        y = np.array([0, 0], dtype=np.int8)  # only DOWN
        names = [f"col{i}" for i in range(210)]
        out = _evaluate_on_holdout_3class(_StubBooster(), X, y, names)
        # No UP or NEUTRAL samples -> their AUCs nan
        assert np.isnan(out["auc_up"])
        assert np.isnan(out["auc_neutral"])


class TestDecisionRules:
    def test_argmax_margin_buy(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules
        # Sample 0: p_up dominant with > 10pt margin over p_down -> BUY
        # Sample 1: p_down dominant with > 10pt margin -> SELL
        # Sample 2: p_neutral high, no margin -> HOLD
        probs = np.array([
            [0.1, 0.2, 0.7],  # UP (margin 0.6)
            [0.7, 0.2, 0.1],  # DOWN (margin 0.6)
            [0.3, 0.4, 0.3],  # NEUTRAL
        ])
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "argmax_margin" in out
        # Should fire 1 BUY (correct) and 1 SELL (correct)
        rule = out["argmax_margin"]
        for k in ("buy_precision", "buy_recall", "buy_f1",
                  "sell_precision", "sell_recall", "sell_f1",
                  "trade_rate", "hold_rate"):
            assert k in rule

    def test_indep_thresholds(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules
        probs = np.array([
            [0.10, 0.30, 0.60],
            [0.60, 0.30, 0.10],
            [0.30, 0.40, 0.30],
        ])
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "indep_thresholds" in out

    def test_net_direction(self):
        from tools.v4_5_horizon_compare import _evaluate_decision_rules
        probs = np.array([
            [0.10, 0.20, 0.70],  # net +0.6 -> BUY
            [0.70, 0.20, 0.10],  # net -0.6 -> SELL
            [0.40, 0.30, 0.30],  # net -0.1 -> HOLD (below 0.20 threshold)
        ])
        labels = np.array([2, 0, 1], dtype=np.int8)
        out = _evaluate_decision_rules(probs, labels)
        assert "net_direction" in out


class TestRenderHtmlReport:
    def test_writes_html_with_horizons_and_rules(self, tmp_path):
        from tools.v4_5_horizon_compare import _render_html_report
        metrics = {
            24:  {"auc_macro": 0.55, "auc_down": 0.54, "auc_neutral": 0.50,
                  "auc_up": 0.61, "logloss": 1.0, "n_samples": 1000,
                  "pos_frac_down": 0.3, "pos_frac_neutral": 0.4, "pos_frac_up": 0.3},
            72:  {"auc_macro": 0.57, "auc_down": 0.56, "auc_neutral": 0.51,
                  "auc_up": 0.64, "logloss": 0.98, "n_samples": 800,
                  "pos_frac_down": 0.32, "pos_frac_neutral": 0.38, "pos_frac_up": 0.30},
            168: {"auc_macro": 0.53, "auc_down": 0.52, "auc_neutral": 0.50,
                  "auc_up": 0.57, "logloss": 1.05, "n_samples": 500,
                  "pos_frac_down": 0.35, "pos_frac_neutral": 0.30, "pos_frac_up": 0.35},
        }
        rules = {
            24:  {"argmax_margin": {"buy_f1": 0.30, "sell_f1": 0.25,
                                    "buy_precision": 0.5, "buy_recall": 0.2,
                                    "sell_precision": 0.4, "sell_recall": 0.2,
                                    "trade_rate": 0.4, "hold_rate": 0.6},
                  "indep_thresholds": {"buy_f1": 0.28, "sell_f1": 0.22,
                                       "buy_precision": 0.45, "buy_recall": 0.20,
                                       "sell_precision": 0.40, "sell_recall": 0.15,
                                       "trade_rate": 0.45, "hold_rate": 0.55},
                  "net_direction": {"buy_f1": 0.32, "sell_f1": 0.28,
                                    "buy_precision": 0.50, "buy_recall": 0.24,
                                    "sell_precision": 0.42, "sell_recall": 0.21,
                                    "trade_rate": 0.42, "hold_rate": 0.58}},
            72:  {"argmax_margin": {"buy_f1": 0.35, "sell_f1": 0.30,
                                    "buy_precision": 0.6, "buy_recall": 0.25,
                                    "sell_precision": 0.5, "sell_recall": 0.21,
                                    "trade_rate": 0.4, "hold_rate": 0.6},
                  "indep_thresholds": {"buy_f1": 0.33, "sell_f1": 0.28,
                                       "buy_precision": 0.55, "buy_recall": 0.23,
                                       "sell_precision": 0.45, "sell_recall": 0.20,
                                       "trade_rate": 0.43, "hold_rate": 0.57},
                  "net_direction": {"buy_f1": 0.36, "sell_f1": 0.31,
                                    "buy_precision": 0.58, "buy_recall": 0.26,
                                    "sell_precision": 0.48, "sell_recall": 0.23,
                                    "trade_rate": 0.41, "hold_rate": 0.59}},
            168: {"argmax_margin": {"buy_f1": 0.20, "sell_f1": 0.18,
                                    "buy_precision": 0.4, "buy_recall": 0.13,
                                    "sell_precision": 0.35, "sell_recall": 0.12,
                                    "trade_rate": 0.3, "hold_rate": 0.7},
                  "indep_thresholds": {"buy_f1": 0.19, "sell_f1": 0.17,
                                       "buy_precision": 0.38, "buy_recall": 0.13,
                                       "sell_precision": 0.33, "sell_recall": 0.11,
                                       "trade_rate": 0.32, "hold_rate": 0.68},
                  "net_direction": {"buy_f1": 0.21, "sell_f1": 0.19,
                                    "buy_precision": 0.41, "buy_recall": 0.14,
                                    "sell_precision": 0.36, "sell_recall": 0.13,
                                    "trade_rate": 0.30, "hold_rate": 0.70}},
        }
        out_path = str(tmp_path / "report.html")
        _render_html_report(metrics, rules, out_path)
        assert os.path.exists(out_path)
        html = open(out_path).read()
        # Sanity: each horizon + each rule appears somewhere
        assert "h24" in html or "24" in html
        assert "h72" in html or "72" in html
        assert "h168" in html or "168" in html
        assert "argmax_margin" in html
        assert "indep_thresholds" in html
        assert "net_direction" in html
```

- [ ] **Step 7b.2 — Run; expect ModuleNotFoundError**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_v4_5_horizon_compare.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.v4_5_horizon_compare'`.

- [ ] **Step 7b.3 — Create v4_5_horizon_compare**

Create `backend/tools/v4_5_horizon_compare.py`:

```python
"""XGB v4.5 horizon + decision-rule comparison report (#xgb-v4.5 / Step B.1.5).

For each horizon (24/72/168): load 3-class artifacts, build per-pid last-15%
holdout, predict (N, 3) softmax probs, compute per-class AUC + macro-AUC +
logloss + class distribution. Then evaluate 3 decision rules
(argmax_margin / indep_thresholds / net_direction) on the same holdout —
precision/recall/F1 of BUY signal (labels==UP) and SELL signal (labels==DOWN).

Render side-by-side HTML report at backend/tools/xgb_v4_5_horizon_compare.html
with the (horizon, rule) combo highlighted by best buy_f1 + sell_f1 composite.

Per feedback_python_clean_functions: pure-function helpers, main()
orchestrator only.

Run (after all 3 horizons trained via train_xgb_v4_5.py):
    cd backend && python -m tools.v4_5_horizon_compare \
      --horizons 24,72,168 --pids BTC-USD,ETH-USD,...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = BACKEND
_DEFAULT_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_DEFAULT_OUT_PATH = os.path.join(BACKEND, "tools", "xgb_v4_5_horizon_compare.html")
_HORIZON_THRESHOLDS: Dict[int, float] = {24: 0.015, 72: 0.03, 168: 0.06}


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_horizon_artifacts(horizon: int, base_dir: str) -> Dict[str, object]:
    """Load v4.5 booster + feature_names for one horizon.

    Expected files:
      base_dir/xgb_model_v4_5_h<H>.json
      base_dir/xgb_features_v4_5_h<H>.json
    """
    import xgboost as xgb

    model_path = os.path.join(base_dir, f"xgb_model_v4_5_h{horizon}.json")
    feat_path  = os.path.join(base_dir, f"xgb_features_v4_5_h{horizon}.json")
    for p in (model_path, feat_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"v4.5 horizon h{horizon} artifact missing: {p}")
    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(feat_path, "r") as f:
        feature_names = json.load(f)["feature_names"]
    return {"booster": booster, "feature_names": feature_names}


def _evaluate_on_holdout_3class(
    booster, X: np.ndarray, y: np.ndarray, feature_names: List[str],
) -> Dict[str, float]:
    """Compute per-class AUC + macro-AUC + logloss + class distribution.

    Returns dict with: auc_down, auc_neutral, auc_up, auc_macro, logloss,
    n_samples, pos_frac_down, pos_frac_neutral, pos_frac_up. AUC for a
    class is NaN if no positive examples of that class in holdout.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, log_loss

    n = X.shape[0]
    out: Dict[str, float] = {
        "n_samples": n,
        "pos_frac_down":    float((y == 0).mean()) if n > 0 else 0.0,
        "pos_frac_neutral": float((y == 1).mean()) if n > 0 else 0.0,
        "pos_frac_up":      float((y == 2).mean()) if n > 0 else 0.0,
    }
    if n == 0:
        return {**out, "auc_down": float("nan"), "auc_neutral": float("nan"),
                "auc_up": float("nan"), "auc_macro": float("nan"),
                "logloss": float("nan")}
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    probs = booster.predict(dmat)  # (N, 3)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)

    aucs: List[float] = []
    for cls in (0, 1, 2):
        if (y == cls).sum() == 0 or (y != cls).sum() == 0:
            aucs.append(float("nan"))
            continue
        try:
            aucs.append(float(roc_auc_score((y == cls).astype(np.int8),
                                              probs[:, cls])))
        except ValueError:
            aucs.append(float("nan"))
    valid_aucs = [a for a in aucs if not np.isnan(a)]
    out["auc_down"]    = aucs[0]
    out["auc_neutral"] = aucs[1]
    out["auc_up"]      = aucs[2]
    out["auc_macro"]   = float(np.mean(valid_aucs)) if valid_aucs else float("nan")

    if len(set(y.tolist())) >= 2:
        out["logloss"] = float(log_loss(y, probs, labels=[0, 1, 2]))
    else:
        out["logloss"] = float("nan")
    return out


def _evaluate_decision_rules(
    probs: np.ndarray,    # shape (N, 3)
    labels: np.ndarray,   # shape (N,) — 0/1/2
) -> Dict[str, Dict[str, float]]:
    """Per-rule scorecard.

    Each rule produces BUY/SELL/HOLD decisions. We score BUY signals against
    labels==2 (UP) and SELL signals against labels==0 (DOWN). Precision/recall
    of each (signal_class).

    Returns dict keyed by rule name, each value containing:
      buy_precision, buy_recall, buy_f1,
      sell_precision, sell_recall, sell_f1,
      trade_rate (BUY + SELL fraction), hold_rate.
    """
    n = probs.shape[0]
    p_down, p_neutral, p_up = probs[:, 0], probs[:, 1], probs[:, 2]
    argmax = probs.argmax(axis=1)

    rules_buy_sell: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        "argmax_margin": (
            (argmax == 2) & ((p_up - p_down) > 0.10),
            (argmax == 0) & ((p_down - p_up) > 0.10),
        ),
        "indep_thresholds": (
            (p_up > 0.50) & (p_up >= p_down),
            (p_down > 0.50) & (p_down > p_up),
        ),
        "net_direction": (
            (p_up - p_down) > 0.20,
            (p_down - p_up) > 0.20,
        ),
    }

    out: Dict[str, Dict[str, float]] = {}
    label_up = (labels == 2)
    label_dn = (labels == 0)

    def _prf(signal: np.ndarray, truth: np.ndarray) -> Tuple[float, float, float]:
        if signal.sum() == 0:
            precision = float("nan")
        else:
            precision = float((signal & truth).sum()) / float(signal.sum())
        if truth.sum() == 0:
            recall = float("nan")
        else:
            recall = float((signal & truth).sum()) / float(truth.sum())
        if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    for name, (buy, sell) in rules_buy_sell.items():
        bp, br, bf1 = _prf(buy, label_up)
        sp, sr, sf1 = _prf(sell, label_dn)
        trade_rate = float((buy | sell).mean()) if n > 0 else 0.0
        out[name] = {
            "buy_precision": bp, "buy_recall": br, "buy_f1": bf1,
            "sell_precision": sp, "sell_recall": sr, "sell_f1": sf1,
            "trade_rate": trade_rate, "hold_rate": 1.0 - trade_rate,
        }
    return out


def _build_holdout_dataset(
    pids: List[str], horizon: int, label_thresh: float,
    history_dir: str, holdout_frac: float = 0.15,
):
    """Build held-out (X, y) test set per pid using the LAST holdout_frac
    of each pid's history. Uses _build_samples_for_pid from train_xgb_v4_5."""
    from tools.train_xgb_v4_5 import (
        _build_samples_for_pid, _load_candles_for_pid,
    )
    from tools.xgb_v4_5_features import TIER_WINDOWS_V45, N_FEATURES_V45

    micro = TIER_WINDOWS_V45["micro"]
    meso  = TIER_WINDOWS_V45["meso"]
    macro = TIER_WINDOWS_V45["macro"]

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
        n_hold = max(1, int(X.shape[0] * holdout_frac))
        all_X.append(X[-n_hold:])
        all_y.append(y[-n_hold:])
    if not all_X:
        return (np.zeros((0, N_FEATURES_V45), dtype=np.float64),
                np.zeros(0, dtype=np.int8))
    return np.vstack(all_X), np.concatenate(all_y)


def _render_html_report(
    metrics_by_horizon: Dict[int, Dict[str, float]],
    rules_by_horizon: Dict[int, Dict[str, Dict[str, float]]],
    out_path: str,
) -> None:
    """Side-by-side HTML report with per-horizon AUC + per-rule scorecard.
    Highlights winning horizon (best auc_macro) and winning rule per horizon
    (best composite buy_f1 + sell_f1)."""
    # Winning horizon by macro AUC (ignore NaN)
    valid_h = {h: m for h, m in metrics_by_horizon.items()
               if not np.isnan(m.get("auc_macro", float("nan")))}
    winner_h = max(valid_h, key=lambda h: valid_h[h]["auc_macro"]) if valid_h else None

    # Winning rule per horizon by buy_f1+sell_f1
    winner_rule_by_h: Dict[int, str] = {}
    for h, rules in rules_by_horizon.items():
        scored = {r: rules[r]["buy_f1"] + rules[r]["sell_f1"] for r in rules}
        winner_rule_by_h[h] = max(scored, key=lambda r: scored[r]) if scored else ""

    horizons_rows: List[str] = []
    for h in sorted(metrics_by_horizon.keys()):
        m = metrics_by_horizon[h]
        cls = "winner" if h == winner_h else ""
        horizons_rows.append(
            f"<tr class='{cls}'>"
            f"<td>h{h}</td>"
            f"<td class='num'>{m['auc_macro']:.4f}</td>"
            f"<td class='num'>{m['auc_down']:.4f}</td>"
            f"<td class='num'>{m['auc_neutral']:.4f}</td>"
            f"<td class='num'>{m['auc_up']:.4f}</td>"
            f"<td class='num'>{m['logloss']:.4f}</td>"
            f"<td class='num'>{m['n_samples']:,}</td>"
            f"<td class='num'>{m['pos_frac_down']:.2f}/{m['pos_frac_neutral']:.2f}/{m['pos_frac_up']:.2f}</td>"
            f"</tr>"
        )

    rule_blocks: List[str] = []
    for h in sorted(rules_by_horizon.keys()):
        rules = rules_by_horizon[h]
        w = winner_rule_by_h.get(h, "")
        rule_rows = []
        for r_name, r in rules.items():
            cls = "winner" if r_name == w else ""
            rule_rows.append(
                f"<tr class='{cls}'><td>{r_name}</td>"
                f"<td class='num'>{r['buy_precision']:.3f}</td>"
                f"<td class='num'>{r['buy_recall']:.3f}</td>"
                f"<td class='num'>{r['buy_f1']:.3f}</td>"
                f"<td class='num'>{r['sell_precision']:.3f}</td>"
                f"<td class='num'>{r['sell_recall']:.3f}</td>"
                f"<td class='num'>{r['sell_f1']:.3f}</td>"
                f"<td class='num'>{r['trade_rate']:.3f}</td>"
                f"</tr>"
            )
        rule_blocks.append(
            f"<h3>h{h} decision rules</h3><table>"
            "<tr><th>rule</th><th>buy_p</th><th>buy_r</th><th>buy_f1</th>"
            "<th>sell_p</th><th>sell_r</th><th>sell_f1</th><th>trade_rate</th></tr>"
            + "".join(rule_rows) + "</table>"
        )

    winner_banner = (
        f"<div class='banner'>Winning horizon: <strong>h{winner_h}</strong> "
        f"(auc_macro={valid_h[winner_h]['auc_macro']:.4f}) · "
        f"Winning rule: <strong>{winner_rule_by_h.get(winner_h, 'n/a')}</strong></div>"
    ) if winner_h is not None else "<div class='banner'>No valid metrics.</div>"

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>XGB v4.5 horizon + rule comparison</title>
<style>
  body {{ background:#0d1117; color:#c9d1d9; font-family:-apple-system,sans-serif;
          padding:32px; max-width:1100px; margin:auto; }}
  h1 {{ color:#fff; }}
  h3 {{ color:#79c0ff; margin-top:24px; }}
  .banner {{ background:#1f3a1f; border:1px solid #1f6b33; color:#56d364;
             padding:14px 20px; border-radius:6px; margin:20px 0; }}
  table {{ width:100%; border-collapse:collapse; margin-bottom:16px; }}
  th {{ text-align:left; color:#8b949e; padding:8px; border-bottom:1px solid #30363d; }}
  td {{ padding:8px; border-bottom:1px solid #21262d; font-family:ui-monospace,monospace; }}
  tr.winner td {{ background:#0d1c11; color:#56d364; font-weight:600; }}
  .num {{ text-align:right; }}
</style></head><body>
<h1>XGB v4.5 horizon + rule comparison</h1>
{winner_banner}
<h3>Per-horizon metrics</h3>
<table>
  <tr><th>horizon</th><th>auc_macro</th><th>auc_down</th><th>auc_neutral</th>
      <th>auc_up</th><th>logloss</th><th>n</th><th>class_fracs (D/N/U)</th></tr>
  {''.join(horizons_rows)}
</table>
{''.join(rule_blocks)}
</body></html>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── Orchestrator ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--horizons", required=True,
                   help="comma-separated, e.g. 24,72,168")
    p.add_argument("--pids", required=True,
                   help="comma-separated pid list")
    p.add_argument("--base-dir", default=_DEFAULT_BASE_DIR)
    p.add_argument("--history-dir", default=_DEFAULT_HISTORY_DIR)
    p.add_argument("--out-path", default=_DEFAULT_OUT_PATH)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    pids = [p.strip() for p in args.pids.split(",") if p.strip()]

    metrics: Dict[int, Dict[str, float]] = {}
    rules: Dict[int, Dict[str, Dict[str, float]]] = {}
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
        metrics[h] = _evaluate_on_holdout_3class(
            artifacts["booster"], X, y, artifacts["feature_names"],
        )
        # Re-predict for decision-rule eval (same probs)
        import xgboost as xgb
        if X.shape[0] > 0:
            dmat = xgb.DMatrix(X, feature_names=artifacts["feature_names"])
            probs = artifacts["booster"].predict(dmat)
            rules[h] = _evaluate_decision_rules(probs, y)
        else:
            rules[h] = {}
        m = metrics[h]
        print(f"  h{h}: auc_macro={m['auc_macro']:.4f} logloss={m['logloss']:.4f} "
              f"n={m['n_samples']} class_fracs="
              f"{m['pos_frac_down']:.2f}/{m['pos_frac_neutral']:.2f}/{m['pos_frac_up']:.2f}",
              flush=True)

    _render_html_report(metrics, rules, args.out_path)
    print(f"\nHTML report: {args.out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7b.4 — Run; expect 5+ GREEN**

```bash
.venv/Scripts/python.exe -m pytest backend/tests/test_v4_5_horizon_compare.py -v
```

Expected: 5+ PASSED.

---

## Task 8: CLAUDE.md invariant #17 + CHANGELOG

**Files:**
- Modify: `CLAUDE.md` (append invariant #17 after #16)
- Modify: `CHANGELOG.md` (prepend new Session entry above current top)

- [ ] **Step 8.1 — Add invariant #17 to CLAUDE.md**

Edit `CLAUDE.md`. Find the "Key invariants (never break these)" section. After invariant #16, append:

```markdown
17. **3-class telemetry contract** — When persisting v4.5+ multi-class probabilities to `cnn_scans`, ALL probabilities for a given model version (e.g., all 3 of `xgb_prob_v4_5_down/neutral/up`) must be written together or all NULL — never partial. Probabilities should sum to ~1.0 (after clip + renormalize). Downstream consumers (decision rules, calibration analysis) rely on this invariant. Mirrors invariant #14's MC chain rule for telemetry consistency.
```

- [ ] **Step 8.2 — Add CHANGELOG entry**

Edit `CHANGELOG.md`. Prepend ABOVE the current top entry:

```markdown
## [Session 58.71k] — 2026-05-17 — XGB v4.5 3-class trend model + BB channels (#xgb-v4.5 / Step B.1.5)

### Why
v4 binary collapsed DOWN and NEUTRAL into one class — no SHORT/SELL signal,
and the operator concern that 4hr windows are too short in sustained
downtrends meant v4 binary couldn't help the agent AVOID buying mid-decline.
v4.5 pivots to 3-class triple-barrier labels (DOWN/NEUTRAL/UP) at longer
horizons (h24/h72/h168) with 2 added Bollinger Band channels for
volatility-regime signal.

### What changed
- **`backend/tools/xgb_v4_5_features.py`** (new) — pure-function 7-channel
  extractor (5 OHLCV + bb_position + bb_width). 210 features (7 × 3 tiers
  × 10 stats). Constants derived (`N_CHANNELS_V45 = len(_CHANNEL_NAMES)`).
- **`backend/tools/train_xgb_v4_5.py`** (new) — 3-class trainer. `main()`
  delegates to `_load_candles_for_pid`, `_triple_barrier_label_3class`
  (UP wins tie), `_build_samples_for_pid` (needs macro+BB_PREFIX bars per
  sample), `_walk_forward_split`, `_train_booster_3class`
  (`multi:softprob`, `num_class=3`), `_save_artifacts`. Horizon-suffixed
  artifacts; no calibrator (raw softmax).
- **`backend/tools/v4_5_horizon_compare.py`** (new) — per-class AUC +
  macro-AUC + 3-rule decision-rule sweep (argmax_margin / indep_thresholds /
  net_direction) + side-by-side HTML report with winning (horizon, rule)
  highlighted.
- **`backend/migrations/xgb_v4_5_shadow_20260517.py`** (new) — idempotent
  ALTER TABLE adds `xgb_prob_v4_5_down/neutral/up REAL` to `cnn_scans`.
- **`backend/agents/xgb_signal.py`** — new v4.5 state (`_booster_v45`,
  `_load_*_v45`), `_try_load_v4_5`, `xgb_prob_v4_5(channels, pid) ->
  Tuple[float, float, float]` with neutral fallback `(0.33, 0.34, 0.33)`,
  `xgb_prob_shadow_v4_5(channels, pid) -> Tuple[float, Optional[Tuple]]`
  with v4.5 isolated in try/except per invariant #16/17.
- **`backend/agents/cnn_agent.py`** — single edit: replace
  `_xgb.xgb_prob_shadow` (v4 binary) with `_xgb.xgb_prob_shadow_v4_5`,
  unpack 3-tuple, write 3 v4.5 prob dict entries to `save_cnn_scan`.
  Drops `xgb_prob_v4` from dict (v4 path no longer called from cnn_agent
  per spec; column stays in schema with NULL default).
- **`backend/tools/xgb_features.py`** — `extract_features` dispatcher gets
  `feature_set == "v4_5"` branch routing to `xgb_v4_5_features.extract_v4_5`.
- **`backend/database.py`** — 3 new REAL columns in `cnn_scans` CREATE
  TABLE + ALTER list + `save_cnn_scan` INSERT.
- **CLAUDE.md** invariant #17 (3-class telemetry contract).
- **Tests** — `test_xgb_v4_5_features.py` (30+ tests),
  `test_train_xgb_v4_5.py` (9+ tests), `test_v4_5_horizon_compare.py`
  (5+ tests), extensions to `test_xgb_signal.py` (6+ shadow tests),
  `test_database.py` (2 persistence tests), `test_mc_migration.py`
  (2 idempotency tests).

### Verification
```
cd backend && python -m pytest tests/ -q -m "not slow and not integration"
=> 1010+ passed (52+ new tests)
```

### Operator preflight (run after this commit, ~30-40 min)
```bash
cd backend
PIDS=BTC-USD,ETH-USD,SOL-USD,...
python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 24  --label-thresh 0.015
python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 72  --label-thresh 0.03
python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 168 --label-thresh 0.06
python -m tools.v4_5_horizon_compare --pids $PIDS --horizons 24,72,168
# Open backend/tools/xgb_v4_5_horizon_compare.html — review per-horizon + per-rule
# Launch dev backend on 8002:
PORT=8002 ../.venv/Scripts/python.exe main.py
```

### Shadow week + promote (operator-driven)
After 7 days of v4.5 shadow telemetry accumulates in cnn_scans, query +
join with signal_outcomes to score each decision rule against live trade
outcomes. Pick (horizon, rule) combo. Copy winning artifacts
(`xgb_*_v4_5_h<WINNER>.* -> xgb_*_v4_5.*`), wire chosen decision rule
into `cnn_agent.generate_signal` (one-line edit), restart 8001 backend.

### Non-goals (deferred)
- v4 binary cutover — superseded
- Marketcap channels — Step B.2 (deferred)
- Multi-class calibration — v4.5.1 if needed
- Auto-promotion — operator-gated only
- Modifying CNN — out of scope per [[feedback_xgb_focus_not_cnn]]
```

---

## Task 9: Atomic commit + push + memory sync

**Files:**
- Stage all 14 v4.5 files
- Run pre-commit hook (full ~1010-test suite, ~5 min)
- Commit + push
- Memory: `coinbase_trader_session_log.md`

- [ ] **Step 9.1 — Stage all files**

```bash
cd C:\Users\gl450\polymarket_app
git add \
  backend/tools/xgb_v4_5_features.py \
  backend/tools/train_xgb_v4_5.py \
  backend/tools/v4_5_horizon_compare.py \
  backend/migrations/xgb_v4_5_shadow_20260517.py \
  backend/tests/test_xgb_v4_5_features.py \
  backend/tests/test_train_xgb_v4_5.py \
  backend/tests/test_v4_5_horizon_compare.py \
  backend/tests/test_xgb_signal.py \
  backend/tests/test_database.py \
  backend/tests/test_mc_migration.py \
  backend/tools/xgb_features.py \
  backend/agents/xgb_signal.py \
  backend/database.py \
  backend/agents/cnn_agent.py \
  CLAUDE.md \
  CHANGELOG.md
git status --short | grep -v '^??'
```

Expected: 16 staged files, no surprises.

- [ ] **Step 9.2 — Commit (pre-commit hook runs full suite, ~5 min)**

```bash
git commit -m "$(cat <<'EOF'
feat(xgb-v4.5): 3-class trend model + BB channels (Step B.1.5)

Skips v4 binary cutover. 3-class triple-barrier labels (DOWN/NEUTRAL/UP)
at longer horizons (h24/h72/h168) with 2 Bollinger Band channels added.
210 features total (7 channels x 3 tiers x 10 stats).

xgb_v4_5_features (NEW):
- Pure-function extractor (extract_v4_5, feature_names_v4_5,
  feature_weights_v4_5). 5 OHLCV + bb_position + bb_width channels.
- Per feedback_python_clean_functions: helpers each pure data-in/data-out
  (_extract_ohlcv_field, _compute_bb_position, _compute_bb_width,
  _compute_stats, _slope, _pct_rank, _delta_at). No in-place mutation.
- Constants derived (N_CHANNELS_V45 = len(_CHANNEL_NAMES), etc.).

train_xgb_v4_5 (NEW):
- main() delegates to _load_candles_for_pid, _triple_barrier_label_3class
  (UP wins tie), _build_samples_for_pid (needs macro+BB_PREFIX bars per
  sample for BB calculation), _walk_forward_split, _train_booster_3class
  (multi:softprob, num_class=3), _save_artifacts.
- CLI args --forward-hours + --label-thresh required.
- Horizon-suffixed artifacts (xgb_*_v4_5_h<HOURS>.*).
- No calibrator file (raw softmax used).

v4_5_horizon_compare (NEW):
- Per-class AUC + macro-AUC + 3-rule decision-rule sweep.
- 3 rules: argmax_margin, indep_thresholds, net_direction.
- HTML report with winning (horizon, rule) combo highlighted.

xgb_signal v4.5 shadow path:
- _try_load_v4_5, xgb_prob_v4_5 (returns Tuple[float, float, float] with
  neutral fallback (0.33, 0.34, 0.33)), xgb_prob_shadow_v4_5 (returns
  (v3_prob, v4_5_tuple_or_None) with v4.5 isolated try/except).
- v3 path unchanged.

cnn_agent write-through:
- xgb_prob_shadow -> xgb_prob_shadow_v4_5; unpack 3-tuple; write 3 v4.5
  dict entries to save_cnn_scan. NO decision logic changes (v3 still
  drives until operator promotes after shadow week).
- xgb_prob_v4 dropped from dict (v4 path no longer called from cnn_agent
  per spec; column stays in schema with NULL).

Database:
- 3 new cnn_scans columns: xgb_prob_v4_5_down/neutral/up REAL.
- Idempotent migration xgb_v4_5_shadow_20260517.py.
- save_cnn_scan INSERT + CREATE TABLE + ALTER TABLE migration list.

CLAUDE.md invariant #17 (3-class telemetry contract — all 3 written
together or all NULL, sum ~1.0 after clip+renormalize).

Operator preflight (run after this commit, ~30-40 min):
  python -m tools.train_xgb_v4_5 --pids <list> --forward-hours 24 \
    --label-thresh 0.015
  ... (72/0.03, 168/0.06)
  python -m tools.v4_5_horizon_compare --pids <list> --horizons 24,72,168
  PORT=8002 python main.py    # dev backend on 8002, v3 still on 8001

Shadow week then operator picks (horizon, rule) from compare HTML +
shadow telemetry vs signal_outcomes. Promote = copy h<WINNER> artifacts
to unsuffixed + wire rule into cnn_agent + restart 8001.

Tests: ~52 new (xgb_v4_5_features 30+, train_xgb_v4_5 9+,
v4_5_horizon_compare 5+, xgb_signal shadow 6+, database persistence 2,
migration idempotency 2). Full suite green.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hook runs the full ~1010-test suite (~5 min). All must pass.

- [ ] **Step 9.3 — Push**

```bash
git push origin feat/gpu-coord-mirror
```

- [ ] **Step 9.4 — Verify backend (8001) still healthy**

```bash
curl -sS -m 3 http://localhost:8001/api/status
```

Expected: 200. The v4.5 path is shadow only (no decision changes); production v3 driver on 8001 keeps trading normally.

- [ ] **Step 9.5 — Memory sync**

Edit `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_session_log.md`. Prepend ABOVE the current top entry:

```markdown
- **Session 58.71k (2026-05-17)**: XGB v4.5 3-class trend model + BB channels (#xgb-v4.5 / Step B.1.5). Skips v4 binary cutover (v4 path stays callable but no longer called from cnn_agent). New `backend/tools/xgb_v4_5_features.py` (pure-function 7-channel extractor: 5 OHLCV + bb_position + bb_width; 210 features = 7 × 3 tiers × 10 stats; helpers `_extract_ohlcv_field`/`_compute_bb_position`/`_compute_bb_width`/`_compute_stats`/`_slope`/`_pct_rank`/`_delta_at` per [[feedback_python_clean_functions]]). New `backend/tools/train_xgb_v4_5.py` (3-class trainer, `multi:softprob` `num_class=3`, horizon-suffixed `xgb_*_v4_5_h<H>.*` artifacts, no calibrator). New `backend/tools/v4_5_horizon_compare.py` (per-class AUC + 3-rule decision sweep: argmax_margin/indep_thresholds/net_direction; HTML report). New 3 columns on `cnn_scans` (`xgb_prob_v4_5_down/neutral/up`) + idempotent migration. `xgb_signal` gained `xgb_prob_v4_5` (returns 3-tuple, neutral fallback (0.33, 0.34, 0.33)) + `xgb_prob_shadow_v4_5` (returns `(v3, tuple_or_None)`); v4.5 fully isolated in try/except, never affects v3. `cnn_agent` write-through: replaced `xgb_prob_shadow` with `xgb_prob_shadow_v4_5`, dropped `xgb_prob_v4` dict entry. NO decision logic changes — v3 still drives trading until operator promotes after shadow week. CLAUDE.md invariant #17 (3-class telemetry contract). Operator preflight: train 3 horizons (h24/0.015, h72/0.03, h168/0.06), `v4_5_horizon_compare`, launch dev backend `PORT=8002`. Shadow week then promote with chosen (horizon, rule). Full suite 1010+ passed (+52 new tests). Commit `<sha>`, branch `feat/gpu-coord-mirror`.
```

Fill `<sha>` with the actual commit hash from Step 9.2.

---

## Spec coverage check

| Spec section | Plan task |
|---|---|
| Architecture decisions table | Tasks 1-9 all reflect locked decisions |
| `xgb_v4_5_features.py` module | Task 1 |
| `xgb_features.py` dispatcher | Task 2 |
| `xgb_signal.py` shadow path | Task 5 |
| `database.py` columns + CREATE + ALTER + INSERT | Task 4 |
| `migrations/xgb_v4_5_shadow_*.py` | Task 3 |
| `cnn_agent.py` write-through | Task 6 |
| `train_xgb_v4_5.py` | Task 7 |
| `v4_5_horizon_compare.py` | Task 7b |
| Test classes (12 + extensions) | Tasks 1, 3, 4, 5, 6, 7, 7b |
| BB time-series semantics + pre-period fallback | Task 1 step 1.3 (`_compute_bb_position/_compute_bb_width`) |
| Triple-barrier 3-class with UP-wins tie | Task 7 step 7.3 (`_triple_barrier_label_3class`) |
| BB prefix in sample build (macro+20 = 356) | Task 7 step 7.3 (`_build_samples_for_pid`) |
| 3 decision rules in compare report | Task 7b step 7b.3 (`_evaluate_decision_rules`) |
| invariant #17 | Task 8 |
| CHANGELOG 58.71k | Task 8 |
| Operator preflight (3-horizon sweep + compare + 8002 dev) | Task 8 (CHANGELOG embed) + Task 9 |
| Shadow week + promote workflow | Task 8 (CHANGELOG embed) — separate brainstorm at promote time |
| Memory append | Task 9 step 9.5 |

All spec requirements traced to a task.

## Self-review (placeholder + type consistency)

- No "TBD", "TODO", "implement later" in any step body
- Every code-changing step shows full code blocks
- Function signatures consistent across tasks:
  - `xgb_prob_v4_5(channels, pid) -> Tuple[float, float, float]` matches Tasks 5, 6, 8
  - `xgb_prob_shadow_v4_5(channels, pid) -> Tuple[float, Optional[Tuple[float, float, float]]]` matches Tasks 5, 6, 8
  - `_triple_barrier_label_3class(closes, start, forward_hours, label_thresh) -> Optional[int]` matches Task 7 impl + Task 7b reuse
- Column names consistent: `xgb_prob_v4_5_down/neutral/up` across Tasks 3, 4, 5, 6, 8
- Migration filename `xgb_v4_5_shadow_20260517.py` consistent across Tasks 3, 8, 9
- Horizon-suffixed paths `xgb_*_v4_5_h<HOURS>.*` consistent across Tasks 7, 7b, 8
- Channel layout `ch0..ch4` = OHLCV, `ch5` = bb_position, `ch6` = bb_width consistent in Task 1 tests + impl

One known caveat: Task 6 drops the `xgb_prob_v4` dict entry (v4 path no longer called from cnn_agent per spec). The `xgb_prob_v4` DB column stays in schema — it'll just receive `None` from `scan.get("xgb_prob_v4")`. Any test that asserted `xgb_prob_v4` is populated by cnn_agent will need updating to expect `None` instead. Per Task 6 step 6.5 note.

## Plan complete

Saved to `docs/superpowers/plans/2026-05-17-xgb-v4-5-three-class.md`. **10 tasks, ~50 micro-steps, 1 atomic commit + 3-horizon operator sweep + shadow week + data-driven promote. +52 new tests on top of B.1's 48.**

Same execution shape as B.1: subagent-driven recommended (fresh subagent per task, single atomic commit at end of Task 9).

---

**Plan complete and saved.** Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, single atomic commit at end of Task 9. Same pattern that worked for B.1.
2. **Inline Execution** — execute tasks in this session via `executing-plans`, batched with checkpoints.

Which approach?
