# XGB v4 — OHLCV-5 Shadow Model (Step B.1) — Design

**Date:** 2026-05-17
**Step:** B.1 (of the marketcap-channel-buildout roadmap)
**Branch:** `feat/gpu-coord-mirror`
**Scope:** Build a fresh XGB model with 5 OHLCV-derived channels, run it in shadow alongside live v3 for one week, then compare AUC to decide cutover.

## Goal

Replace v3's 28-channel-but-only-uses-close design with a small, honest 5-channel OHLCV baseline. Validate against live v3 via shadow telemetry before any cutover. This is the foundation for iterative channel additions in subsequent steps (B.2 marketcap, B.3 FDV/supply, B.4+ TBD).

## Why

v3's `_extract_v3` only reads `candles[i]["close"]` for every channel slot. The 350 named features collapse to ~30 distinct values (10 stats × 3 tiers) duplicated 28 times. The booster wastes capacity learning that `ch0_last == ch1_last == … == ch16_last`. Fixing this needs a fresh model — feature distribution changes invalidate v3's calibration.

User direction: start small and clean, add channels iteratively as each demonstrates value, prioritize macro-trend signals first. Per [[feedback_xgb_focus_not_cnn]], the XGB side is built independently of `cnn_agent.py` — no shared modules, no extraction from CNN.

## Architecture

```
                  ┌──────────────────────────────────────┐
                  │  xgb_signal._cnn_prob (per scan)     │
                  │                                      │
   tiered_history │  ┌─────────┐         ┌─────────┐    │
   .fetch_tiered  ├─►│  v3     │ driver  │  v4     │    │
   (OHLCV/tier)   │  │  infer  │ ─────►  │  infer  │    │ shadow
                  │  │ (300)   │ decision│ (150)   │ ──► xgb_prob_v4
                  │  └────┬────┘         └────┬────┘    │
                  │       │                   │         │
                  │       └──► xgb_prob_v3 ───┴──► save_cnn_scan
                  └──────────────────────────────────────┘
```

v3 keeps driving decisions for the full shadow week. v4 telemetry runs every scan, failures isolated, both probs persist per scan to `cnn_scans`.

## Components

### `backend/tools/xgb_v4_features.py` (new, ~120 LOC)

Pure functions. No state. No `xgb_features.py` imports. Per [[feedback_python_clean_functions]] — type hints on every signature, single responsibility per helper, derived constants over hardcoded, no in-place buffer mutation, explicit input-contract docstrings.

```python
"""XGB v4 OHLCV-5 feature extractor.

5 channels (open/high/low/close/volume) x 3 tiers (micro/meso/macro)
x 10 stats = 150 features. Pure functions, no mutable module state.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import numpy as np

# ── Configuration constants ────────────────────────────────────────────────
_CHANNEL_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
N_CHANNELS_V4: int = len(_CHANNEL_FIELDS)                # = 5 (derived)

TIER_WINDOWS_V4: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V4: Dict[str, float] = {"micro": 1.0, "meso": 2.0, "macro": 3.0}

_STAT_NAMES_V4: Tuple[str, ...] = (
    "last", "mean", "std", "slope",
    "min", "max", "pct_rank",
    "dlt5", "dlt10", "dlt30",
)
N_STATS_V4: int = len(_STAT_NAMES_V4)                    # = 10 (derived)
N_TIERS_V4: int = len(TIER_WINDOWS_V4)                   # = 3 (derived)
N_FEATURES_V4: int = N_CHANNELS_V4 * N_TIERS_V4 * N_STATS_V4  # = 150 (derived)


# ── Public API ─────────────────────────────────────────────────────────────

def feature_names_v4() -> List[str]:
    """Return 150 feature names in stable column order.

    Layout: ch{0..4}_{micro|meso|macro}_{stat}, ordered
    channel-major -> tier-major -> stat-major.
    """

def feature_weights_v4() -> np.ndarray:
    """Return 150-long float64 weight vector aligned with feature_names_v4().

    Per-tier weights: micro 1.0, meso 2.0, macro 3.0. Same weight for all
    10 stats within one (channel, tier) group.
    """

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


# ── Internal helpers (pure functions, one responsibility each) ─────────────

def _extract_field(
    candles: Sequence[Dict[str, float]],
    field: str,
) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray.

    Empty input -> empty ndarray (caller assembles into the zero slot).
    """

def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Return shape-(10,) stats in fixed _STAT_NAMES_V4 order.

    Empty input -> all zeros. No in-place mutation of any caller buffer.
    """

def _slope(values: np.ndarray) -> float:
    """OLS slope of values vs index 0..len-1. 0.0 for n<2 or zero variance."""

def _pct_rank(values: np.ndarray) -> float:
    """Percentile rank of last value within the series. 0.0 if empty/single."""

def _delta_at(values: np.ndarray, lookback: int) -> float:
    """values[-1] - values[-1-lookback], or 0.0 if series too short."""
```

**Decomposition rationale** — each helper is independently testable in isolation. `_compute_stats` is pure data-in / data-out (contrast v3's `_stats_from_candles(candles, stat_offset, out)` which mutates a caller-owned buffer at a magic offset). `_slope`, `_pct_rank`, `_delta_at` are each ~5 lines, ~3 test cases each. Constants are derived: adding a 6th channel by appending to `_CHANNEL_FIELDS` automatically updates `N_CHANNELS_V4`, `N_FEATURES_V4`, the weight vector, and the names — no drift.

### `backend/tools/xgb_features.py` (edit, ~5 LOC dispatcher addition)

```python
def extract_features(samples, feature_set="v1"):
    if feature_set == "v4":
        from tools.xgb_v4_features import _extract_v4
        return _extract_v4(samples)
    if feature_set == "v3":
        return _extract_v3(samples)
    # … v1/v2 unchanged
```

### `backend/agents/xgb_signal.py` (edit, ~30 LOC shadow path)

`xgb_prob` (or wherever the v3 loader lives) gets a parallel v4 loader:
- Loads `xgb_model_v4.json` + `xgb_features_v4.json` + `xgb_calibration_v4.pkl` on first call (cached)
- Returns both `(prob_v3, prob_v4)` from a new `xgb_prob_shadow(channels, pid)` function
- v4 failure (load error, inference error, calibration error) caught, logged, `prob_v4 = None`
- v3 path is unchanged — same code as today

Call site in `cnn_agent.generate_signal` (or wherever `_cnn_prob` returns to `save_cnn_scan`):
- Capture `prob_v4` alongside `prob_v3`
- Pass `xgb_prob_v4=prob_v4` to `save_cnn_scan(...)`

### `backend/database.py` (edit, ~3 LOC)

`save_cnn_scan(...)` adds `xgb_prob_v4: Optional[float] = None` kwarg. INSERT statement adds `xgb_prob_v4` column with `?` binding.

### `backend/migrations/xgb_v4_shadow_<ts>.py` (new, idempotent)

```python
ALTER TABLE cnn_scans ADD COLUMN xgb_prob_v4 REAL;
```

Same idempotent pattern as `mc_telemetry_20260516.py`: try-except on duplicate column.

`database.init_db` CREATE TABLE statement also gets the column inline (so fresh DBs include it).

### `backend/tools/train_xgb_v4.py` (new, ~250 LOC)

Per [[feedback_python_clean_functions]] — no god-`main()`. Orchestrator delegates to small, single-responsibility helpers, each pure data-in / data-out:

```python
"""XGB v4 trainer. Reads OHLCV from data/history parquets, builds
triple-barrier labels, walk-forward splits, trains the v4 booster,
calibrates isotonic, writes artifacts."""
from __future__ import annotations
from typing import Dict, List, Tuple
import argparse
import numpy as np
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression


# ── Pure helpers ──────────────────────────────────────────────────────────

def _load_candles_for_pid(pid: str, parquet_dir: str) -> List[Dict]:
    """Read OHLCV candles for one pid; ascending by timestamp. [] if missing."""

def _build_samples_for_pid(
    candles: List[Dict],
    *,
    label_thresh: float,
    forward_hours: int,
    micro: int,
    meso: int,
    macro: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each valid bar i (i >= macro AND i + forward_hours < n), produce
    one sample's features via extract_v4 + one triple-barrier label.

    Returns:
        features: (N, 150) float64
        labels:   (N,) int8  (0/1, triple-barrier UP)
        timestamps: (N,) int64 (epoch seconds at sample bar)
    """

def _walk_forward_split(
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    *,
    embargo_bars: int,
    val_frac: float = 0.15,
    cal_frac: float = 0.15,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Chronological split into (train, val, cal). Embargo gap between
    train end and val start; same between val and cal."""

def _train_booster(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    feature_names: List[str], feature_weights: np.ndarray,
) -> xgb.Booster:
    """Single xgb.train call. Hyperparams hard-coded inside (200 trees,
    depth 4, lr 0.1, colsample_bytree 0.8, binary:logistic). No side
    effects beyond returning the booster."""

def _calibrate_isotonic(
    booster: xgb.Booster, X_cal: np.ndarray, y_cal: np.ndarray,
) -> IsotonicRegression:
    """Fit IsotonicRegression on the booster's raw probs vs calibration
    labels. Returns the fitted calibrator (no side effects)."""

def _save_artifacts(
    booster: xgb.Booster,
    calibrator: IsotonicRegression,
    feature_names: List[str],
    out_dir: str,
) -> None:
    """Atomic write: tmp file -> rename for each of model.json, features.json,
    calibration.pkl. Calibrator pickled as dict {'calibrator', 'feature_set'}.
    Note `.json` must be the LAST extension on booster tmp name (xgboost
    serialization format auto-detection — see prior session bug fix)."""


# ── Orchestrator ──────────────────────────────────────────────────────────

def main(argv: List[str] | None = None) -> int:
    """Parse args, run pipeline, print AUC summary. Returns exit code."""
    args = _parse_args(argv)
    all_features: List[np.ndarray] = []
    all_labels:   List[np.ndarray] = []
    all_ts:       List[np.ndarray] = []
    for pid in args.pids:
        candles = _load_candles_for_pid(pid, args.parquet_dir)
        if not candles:
            continue
        feats, lbls, ts = _build_samples_for_pid(candles, ...)
        all_features.append(feats); all_labels.append(lbls); all_ts.append(ts)
    X = np.vstack(all_features); y = np.concatenate(all_labels); t = np.concatenate(all_ts)
    (X_tr, y_tr), (X_va, y_va), (X_ca, y_ca) = _walk_forward_split(X, y, t, ...)
    booster = _train_booster(X_tr, y_tr, X_va, y_va, feature_names_v4(), feature_weights_v4())
    calibrator = _calibrate_isotonic(booster, X_ca, y_ca)
    _save_artifacts(booster, calibrator, feature_names_v4(), args.out_dir)
    return 0

def _parse_args(argv: List[str] | None) -> argparse.Namespace: ...
```

Each helper is independently testable. The training-data construction (`_build_samples_for_pid`) doesn't mix with the split (`_walk_forward_split`) or train (`_train_booster`) — each can be exercised with synthetic inputs in `test_train_xgb_v4.py`. Labels use the same triple-barrier params as v3 (read from `agents/cnn_agent.py` constants `_FORWARD_HOURS`, `_LABEL_THRESH`) for parity — labels are deterministic given same params + same candles.

Operator runs once after the implementation commit:
```bash
cd backend && python -m tools.train_xgb_v4 --pids <all-tracked>
```

Expected runtime: ~5-10 min for ~50 pids on local CPU.

### `backend/tests/test_xgb_v4_features.py` (new, ~150 LOC)

- Shape: `_extract_v4({"micro":[...60 candles...], "meso":[...168...], "macro":[...336...]})` returns `(np.ndarray shape (1, 150), list len 150)`
- Names: `_v4_feature_names()` returns 150 names matching pattern `ch{0..4}_{micro|meso|macro}_{stat}`
- Weights: `feature_weights_v4()` length 150, micro=1.0 / meso=2.0 / macro=3.0
- Channel 4 reads volume, not close: build candles with distinct close vs volume, verify `ch4_micro_last == volume`, `ch3_micro_last == close`
- Empty tier handling: passing empty list for a tier zeros that tier's 50 slots
- Determinism: same input → same output

### `backend/tests/test_xgb_signal.py` (extend, ~80 LOC)

- Shadow path returns both v3 and v4 probs when v4 artifacts exist
- Shadow returns `(prob_v3, None)` when v4 artifacts missing
- v4 inference error caught, logged, returns `None`
- v3 prob NEVER affected by v4 path

### `backend/tests/test_database.py` (extend, ~30 LOC)

- `save_cnn_scan(xgb_prob_v4=0.42, ...)` persists value to row
- `save_cnn_scan()` default `xgb_prob_v4=None` produces NULL in column

## Data flow

```
Scan loop tick (every 15 min per pid)
  → fetch tiered candles (micro/meso/macro)
  → xgb_prob_shadow(channels, pid)
      → v3 inference (existing path, unchanged) → prob_v3
      → v4 inference (NEW path, isolated try/except) → prob_v4 (or None)
  → side = compute_side(prob_v3, ...)   # v3 still drives
  → mc.apply_buy_filters(...)
  → save_cnn_scan(pid, prob_v3, prob_v4, side, ...)
```

## Error handling

- v4 artifacts missing on startup → log warning once, `prob_v4` is `None` per scan
- v4 booster load error → log warning once, mark v4 as unavailable, do not retry until backend restart
- v4 inference error mid-scan → caught, logged with pid+tick context, `prob_v4 = None`, scan continues normally
- v4 calibrator error → same as inference error
- Schema migration failure → backend startup hard-fails (consistent with existing `init_db` behavior)

## Tests strategy

- **Unit:** `xgb_v4_features` pure-function tests cover shape, weights, stat correctness on hand-built candles
- **Integration:** `xgb_signal` shadow path with mocked booster files
- **Migration:** existing `test_mc_migration.py` pattern — apply twice, second is no-op
- **No live API calls.** Synthetic candles built in tests via dict literals.
- Full suite must stay green at commit time (~5 min pre-commit hook).

## Rollout

1. Land all code + migration in one atomic commit
2. Backend restart picks up new migration (`xgb_prob_v4` column appears)
3. Backend logs "v4 artifacts missing" warning — expected (no trained model yet)
4. Operator runs `python -m tools.train_xgb_v4` (offline, ~5-10 min)
5. Operator restarts backend — v4 loads, shadow telemetry begins
6. After 7 days: query AUC for both models
7. **Decision (separate brainstorm cycle):** if v4 AUC > v3 AUC, cutover. Else, postmortem + iterate.

## AUC comparison query (post-shadow week)

```sql
SELECT
  COUNT(*)             AS n_outcomes,
  AVG(s.xgb_prob_v3)   AS v3_mean_prob,
  AVG(s.xgb_prob_v4)   AS v4_mean_prob
FROM cnn_scans s
JOIN signal_outcomes o ON o.scan_id = s.id
WHERE s.created_at >= '2026-05-24'   -- 1 week after v4 commit
  AND s.xgb_prob_v4 IS NOT NULL
GROUP BY o.outcome_class;
```

AUC itself computed in Python via `sklearn.metrics.roc_auc_score(labels, probs)` for v3 and v4 separately on the same outcome subset. Script lives in `backend/tools/v3_v4_auc_compare.py` (~50 LOC) — created at end of shadow week, not in the B.1 commit.

## Non-goals

- Cutover v3 → v4. That's a separate decision after the shadow week.
- Marketcap channels. That's Step B.2.
- FDV/supply channels. That's Step B.3.
- Refactor v3 itself. v3 stays untouched.
- Modify `cnn_agent.py` decision logic. The only edit to cnn_agent is the `save_cnn_scan` write-through.
- Touch the frontend. No new UI; v3 keeps driving, no user-visible change.
- Retrain CNN. CNN is out of scope per [[feedback_xgb_focus_not_cnn]].

## Open questions / future work (out of B.1 scope)

- **Cutover criteria** — what AUC delta justifies switching? (Decided in next brainstorm.)
- **Shadow telemetry retention** — `cnn_scans.xgb_prob_v4` column grows forever. Trim policy is its own decision.
- **v4 retraining cadence** — currently operator-triggered. Could automate after cutover.
- **Step B.2 channel slot allocation** — market_cap → ch5, volume_24h → ch6. Locked in once B.2 brainstorm runs.

## Files summary

| Action | Path | LOC est |
|---|---|---|
| Create | `backend/tools/xgb_v4_features.py` | ~120 |
| Create | `backend/tools/train_xgb_v4.py` | ~250 |
| Create | `backend/tests/test_xgb_v4_features.py` | ~150 |
| Create | `backend/migrations/xgb_v4_shadow_<ts>.py` | ~30 |
| Edit | `backend/tools/xgb_features.py` | +5 LOC dispatcher |
| Edit | `backend/agents/xgb_signal.py` | +30 LOC shadow path |
| Edit | `backend/database.py` | +3 LOC kwarg + INSERT, +1 LOC CREATE TABLE |
| Edit | `backend/agents/cnn_agent.py` | +5 LOC pass-through to save_cnn_scan |
| Edit | `backend/tests/test_xgb_signal.py` | +80 LOC shadow tests |
| Edit | `backend/tests/test_database.py` | +30 LOC xgb_prob_v4 tests |
| Edit | `CHANGELOG.md` | new Session 58.71j entry |
| Memory | `coinbase_trader_architecture.md` | append entry |

**Net:** 4 new files (~550 LOC), 6 edits (~150 LOC), 1 atomic commit. Plus operator-triggered train run.

## CLAUDE.md / memory invariants to add

After B.1 lands, append invariant #16:

> **Shadow telemetry isolation** — Inference shadow paths (v4 alongside v3) must NEVER affect the driver path. Failures in any shadow inference are caught + logged + recorded as NULL, never re-raised. Mirrors invariant #14's MC chain rule.
