# XGB v4.5 — 3-Class Trend Model + BB Channels (Step B.1.5) — Design

**Date:** 2026-05-17
**Step:** B.1.5 (skips v4 binary cutover; supersedes it)
**Branch:** `feat/gpu-coord-mirror`
**Scope:** Switch from binary `(UP, not-UP)` to 3-class triple-barrier labels `(DOWN, NEUTRAL, UP)`, retrain at longer horizons to capture sustained-trend context, add 2 Bollinger Band channels (bb_position + bb_width) for volatility-regime signal. Train 3 horizons (h24/h72/h168), evaluate 3 decision rules in the comparison report, operator picks (horizon, rule) at promote time.

## Why

**Binary v4's blind spot:** the label collapses `DOWN` and `NEUTRAL` to `0` — the booster cannot distinguish "going to go down" from "going to chop sideways". For trading this means:
- No SHORT/SELL signal — `xgb_prob < threshold` lumps "model thinks down" with "model thinks chop"
- Wastes information — the booster has the data to distinguish, we just throw away the second axis
- In sustained downtrends, a 4-hour binary window catches "down breach" easily but doesn't help the agent AVOID buying during a multi-day decline

**Operator direction (2026-05-17):** *"4hrs windows are too short when crypto is in a downward trend"* + *"do Bollinger Bands also help with this analysis?"* + *"do 4.5, I need better trend analysis"* → switch to 3-class + longer horizon + add BB channels in a single design iteration, skip v4 binary cutover.

Builds on B.1's v4 OHLCV-5 shadow infrastructure (commit `ae9666d`, 17 files, h4 booster val_auc 0.5367 / holdout 0.5514). The shadow plumbing (PORT-env dev backend on 8002, shared SQLite, save_cnn_scan column persistence) is proven; v4.5 reuses it.

Per [[feedback_xgb_focus_not_cnn]] XGB-side only — `cnn_agent` gets one write-through edit, no decision logic touched until promote. Per [[feedback_python_clean_functions]] pure-function helpers + type hints. Per [[feedback_backend_port_isolation]] dev backend on 8002+, promote to 8001 only after shadow week validation.

## Architecture decisions (locked)

| Decision | Choice |
|---|---|
| Label type | 3-class triple-barrier: `0=DOWN, 1=NEUTRAL, 2=UP` |
| Horizons | Sweep 3: h24 (1.5%), h72 (3%), h168 (6%) |
| Channels | 7 = 5 OHLCV (open/high/low/close/volume) + bb_position + bb_width |
| BB params | period=20, mult=2.0 (mirrors existing `_bollinger`) |
| Feature count | 7 channels × 3 tiers × 10 stats = **210 features** (vs v4's 150) |
| Decision rules | All 3 evaluated on holdout in comparison report: argmax+margin, indep thresholds, net-direction |
| XGB objective | `multi:softprob, num_class=3` |
| Calibration | Skip in v4.5 (raw softmax tree calibration is reasonable); add Platt scaling in v4.5.1 if shadow shows miscalibration |
| Promotion | Operator picks (horizon, rule) from comparison + post-shadow-week telemetry; copy artifacts to unsuffixed paths + restart 8001 |
| Shadow path | Persist `(p_down, p_neutral, p_up)` per scan to `cnn_scans`; no decision-logic change in production until promote |
| Backend port | Dev on 8002 per [[feedback_backend_port_isolation]]; 8001 untouched |
| v4 binary fate | Stays callable (`xgb_prob_v4` + `xgb_prob_shadow`) but no longer called from `cnn_agent` after this lands. Eligible for cleanup in v4.6+. |

## Components

### `backend/tools/xgb_v4_5_features.py` (NEW, ~180 LOC)

Pure-function v4.5 extractor. Same purity rules as v4 per [[feedback_python_clean_functions]] — no in-place buffer mutation, derived constants, type hints, explicit docstrings.

```python
"""XGB v4.5 7-channel feature extractor.

5 OHLCV channels (open/high/low/close/volume) + 2 Bollinger channels
(bb_position, bb_width) × 3 tiers (micro/meso/macro) × 10 stats = 210 features.
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
    """Return 210 names in stable column order.
    Layout: ch{0..6}_{micro|meso|macro}_{stat}, channel-major -> tier-major -> stat-major."""

def feature_weights_v4_5() -> np.ndarray:
    """Return 210-long float64 weight vector aligned with feature_names_v4_5()."""

def extract_v4_5(
    candles_by_tier: Dict[str, Sequence[Dict[str, float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract 210 features from tier-keyed OHLCV candle lists.

    For BB channels: each tier's candles include a 20-bar PREFIX so bb_position
    and bb_width can be computed at every bar in the tier slice. The prefix
    bars are used only for BB calculation — they don't feed into stats.

    Returns (features, names) shape (1, 210).
    Missing/empty tier -> 70 zero slots for that tier (7 chans × 10 stats).
    Missing OHLCV field -> KeyError.
    """


# ── Internal helpers (pure functions, single responsibility) ──────────────

def _extract_ohlcv_field(candles: Sequence[Dict[str, float]], field: str) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray."""

def _compute_bb_position(closes: np.ndarray, period: int = BB_PERIOD,
                        mult: float = BB_MULT) -> np.ndarray:
    """Bollinger position [0..1] at each bar. Bars with fewer than `period`
    prior bars get 0.5 (mid) fallback. Returns same length as `closes`."""

def _compute_bb_width(closes: np.ndarray, period: int = BB_PERIOD,
                     mult: float = BB_MULT) -> np.ndarray:
    """(upper - lower) / mean at each bar. Pre-period bars get 0.0 fallback.
    Returns same length as `closes`."""

def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Shape (10,) stats in fixed _STAT_NAMES_V45 order. Empty -> zeros."""

def _slope(values: np.ndarray) -> float: ...
def _pct_rank(values: np.ndarray) -> float: ...
def _delta_at(values: np.ndarray, lookback: int) -> float: ...
```

### `backend/tools/train_xgb_v4_5.py` (NEW, ~320 LOC)

3-class trainer. `main()` delegates to small helpers, each pure data-in/data-out. CLI args `--forward-hours` and `--label-thresh` REQUIRED per the horizon sweep workflow. Writes horizon-suffixed `xgb_*_v4_5_h<H>.*` artifacts.

```python
def _load_candles_for_pid(pid: str, history_dir: str) -> List[Dict]: ...

def _triple_barrier_label_3class(
    closes: np.ndarray, start: int,
    forward_hours: int, label_thresh: float,
) -> Optional[int]:
    """Returns 0 (DOWN — down barrier hit first), 1 (NEUTRAL — neither hit
    within window), 2 (UP — up barrier hit first), or None (truncated).

    Tie-break: if both barriers would hit at the same bar (rare with
    typical close granularity), UP wins (favors the actionable signal).
    """

def _build_samples_for_pid(
    candles: List[Dict],
    *,
    label_thresh: float, forward_hours: int,
    micro: int, meso: int, macro: int,
    bb_prefix: int = BB_PERIOD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each valid bar i where i >= macro+bb_prefix AND a label can be
    computed, produce one (features [210], int8 label, int64 timestamp).
    """

def _walk_forward_split(...) -> Tuple[...]:
    """Same chronological split as v4 trainer."""

def _train_booster_3class(
    X_train, y_train, X_val, y_val,
    feature_names, feature_weights,
) -> xgb.Booster:
    """xgb.train with objective='multi:softprob', num_class=3, 200 trees,
    depth 4, lr 0.05, subsample 0.7, colsample_bytree 0.8."""

def _save_artifacts(
    booster, feature_names: List[str], out_dir: str,
    *, forward_hours: int,
) -> Dict[str, str]:
    """Horizon-suffixed atomic writes: xgb_model_v4_5_h<H>.json + features. No
    calibrator file in v4.5 (skipped — see Architecture decisions)."""

def main(argv: Optional[List[str]] = None) -> int: ...
```

### `backend/tools/v4_5_horizon_compare.py` (NEW, ~280 LOC)

Loads each horizon's 3-class artifacts. Per horizon:
1. Build holdout (last 15% of each pid's history, sorted chronologically)
2. Predict — gives (N, 3) softmax tuples
3. Compute **per-class AUC** (1-vs-rest), **macro-AUC** (mean across 3 classes), logloss, n_samples, pos_frac per class
4. **Decision-rule sweep**: simulate each of 3 rules on holdout outcomes, score precision/recall/F1 of BUY signal vs labels==2, same for SELL vs labels==0

```python
_HORIZON_THRESHOLDS: Dict[int, float] = {24: 0.015, 72: 0.03, 168: 0.06}

def _load_horizon_artifacts(horizon: int, base_dir: str) -> Dict: ...

def _evaluate_on_holdout_3class(
    booster, X: np.ndarray, y: np.ndarray, feature_names: List[str],
) -> Dict[str, float]:
    """Returns dict with keys: auc_down, auc_neutral, auc_up, auc_macro,
    logloss, n_samples, pos_frac_down, pos_frac_neutral, pos_frac_up."""

def _evaluate_decision_rules(
    probs: np.ndarray,    # shape (N, 3)
    labels: np.ndarray,   # shape (N,) — 0/1/2
) -> Dict[str, Dict[str, float]]:
    """Per-rule scorecard: buy_precision, buy_recall, buy_f1, sell_precision,
    sell_recall, sell_f1, trade_rate, hold_rate. 3 rules: argmax_margin,
    indep_thresholds, net_direction."""

def _build_holdout_dataset(
    pids: List[str], horizon: int, label_thresh: float,
    history_dir: str, holdout_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reuses train_xgb_v4_5._build_samples_for_pid (same labeling logic)."""

def _render_html_report(
    metrics_by_horizon: Dict[int, Dict],
    rules_by_horizon: Dict[int, Dict],
    out_path: str,
) -> None:
    """Side-by-side table per horizon + per-rule scorecard. Highlights:
       - winning horizon by auc_macro
       - winning rule per horizon by buy_f1 + sell_f1
       - operator picks the (horizon, rule) combo to promote."""

def main(argv: Optional[List[str]] = None) -> int: ...
```

### `backend/migrations/xgb_v4_5_shadow_20260517.py` (NEW, ~40 LOC)

Idempotent ALTER TABLE adds 3 nullable REAL columns to `cnn_scans`:
- `xgb_prob_v4_5_down`
- `xgb_prob_v4_5_neutral`
- `xgb_prob_v4_5_up`

Pattern mirrors `mc_telemetry_20260516.py`. CREATE TABLE in `init_db` also gets the 3 columns inline.

### `backend/tools/xgb_features.py` (EDIT, +5 LOC)

```python
def extract_features(samples, feature_set="v1"):
    if feature_set == "v4_5":
        from tools.xgb_v4_5_features import extract_v4_5
        return extract_v4_5(samples)
    if feature_set == "v4":
        from tools.xgb_v4_features import extract_v4
        return extract_v4(samples)
    # ... v3/v2/v1 unchanged
```

### `backend/agents/xgb_signal.py` (EDIT, +110 LOC)

New module-level state for v4.5:
- `_MODEL_PATH_V45`, `_FEATURES_PATH_V45` (no calibration path — skipped)
- `_booster_v45`, `_feature_names_v45`
- `_load_attempted_v45`, `_load_succeeded_v45`

New functions:
- `_try_load_v4_5() -> bool` — load booster + features; idempotent; mirrors `_try_load_v4`
- `xgb_prob_v4_5(channels, pid: str) -> Tuple[float, float, float]` — predict, clip each to [0.01, 0.99], renormalize so probs sum to 1.0, return (p_down, p_neutral, p_up). Returns neutral fallback `(0.33, 0.34, 0.33)` on failure
- `xgb_prob_shadow_v4_5(channels, pid: str) -> Tuple[float, Optional[Tuple[float, float, float]]]` — v3 driver prob + v4.5 3-tuple OR None. v4.5 wrapped in try/except per invariant #16/#17. v3 path UNCHANGED.

### `backend/database.py` (EDIT, +12 LOC)

3 new REAL columns in `cnn_scans` CREATE TABLE. 3 new ALTER statements in migration list (idempotent). `save_cnn_scan` INSERT adds 3 placeholders and `scan.get("xgb_prob_v4_5_down/neutral/up")` to tuple.

### `backend/agents/cnn_agent.py` (EDIT, ~10 LOC)

Replace existing `_xgb.xgb_prob_shadow(...)` call (the v4-binary one) with `_xgb.xgb_prob_shadow_v4_5(...)`. Unpack tuple: `xgb_shadow, v45_probs = _xgb.xgb_prob_shadow_v4_5(...)`. Add 3 dict entries to save_cnn_scan: `xgb_prob_v4_5_down/neutral/up` (None if v45_probs is None). **No decision logic changes.**

## Data flow

```
Per scan (every SCAN_INTERVAL_SECS):
  fetch tiered candles (micro/meso/macro)
  xgb_prob_shadow_v4_5(channels, pid)
    → v3 path: xgb_prob_v3(channels, pid) → prob_v3  [drives BUY/SELL decision]
    → v4.5 path: xgb_prob_v4_5(channels, pid) → (p_down, p_neutral, p_up)
                  (wrapped in try/except → None on failure)
  side = compute_side(prob_v3, ...)              # v3 still drives until promote
  side, mc_tele = mc.apply_buy_filters(...)
  save_cnn_scan(pid, prob_v3,
                xgb_prob_v4_5_down=p_down,
                xgb_prob_v4_5_neutral=p_neutral,
                xgb_prob_v4_5_up=p_up,
                ...)
```

## Tests strategy

| Test class | Coverage | Location |
|---|---|---|
| `TestTripleBarrierLabel3Class` | 4 base cases + 1 tie (UP wins) | `test_train_xgb_v4_5.py` |
| `TestBollingerChannels` | bb_pos clamped [0,1], bb_width = (upper-lower)/mean, pre-period fallback (0.5, 0.0) | `test_xgb_v4_5_features.py` |
| `TestExtractV4_5` | Shape (1, 210), 7 chans × 3 tiers × 10 stats, channel 5/6 read BB-derived values, determinism | `test_xgb_v4_5_features.py` |
| `Test3ClassBoosterShape` | Stub booster on synthetic 3-class data, predict returns shape (N, 3) softmax | `test_train_xgb_v4_5.py` |
| `TestBuildSamplesV4_5` | Returns (N, 210) features + (N,) labels with values in {0,1,2} | `test_train_xgb_v4_5.py` |
| `TestXgbProbV4_5` | Returns 3-tuple of floats clipped [0.01, 0.99], renormalized. Neutral fallback on missing artifacts. | `test_xgb_signal.py` |
| `TestXgbProbShadowV4_5` | Returns (v3, tuple_or_None). v4.5 failure → (v3, None), v3 unaffected. | `test_xgb_signal.py` |
| `TestDecisionRules` | 3 rules on hand-built probs/labels yield expected BUY/SELL/HOLD counts | `test_v4_5_horizon_compare.py` |
| `TestEvaluateOnHoldout3Class` | Per-class AUC, macro-AUC. Single-class label input → NaN. | `test_v4_5_horizon_compare.py` |
| `TestRenderHtmlReport` | HTML with all horizons + all rules in scorecard | `test_v4_5_horizon_compare.py` |
| `TestSaveCnnScanV4_5Cols` | Persists 3 new probs, NULL default when omitted | `test_database.py` |
| `TestV4_5Migration` | Idempotency (apply twice → no-op) | `test_mc_migration.py` |

**~35 new tests** added on top of B.1's 48.

## Error handling

- v4.5 artifacts missing on startup → log warning once, `xgb_prob_shadow_v4_5` returns `(v3, None)` per scan
- v4.5 booster load error → log warning once, mark v4.5 as unavailable, do not retry until restart
- v4.5 inference error mid-scan → caught, logged with pid + tick context, `(v3, None)` returned
- Schema migration failure → backend startup hard-fails
- Renormalization of probs that all clip to 0.01 → fall back to (0.33, 0.34, 0.33)
- Filter exceptions in MC chain — already isolated per invariant #14, unchanged

## Rollout (single atomic commit + operator sweep + shadow week)

1. Land all code + migration in one atomic commit (pre-commit full suite ~5 min, ~1010 tests including ~35 new)
2. Operator sweep (~30-40 min):
   ```bash
   cd backend
   PIDS=BTC-USD,ETH-USD,...
   python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 24  --label-thresh 0.015
   python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 72  --label-thresh 0.03
   python -m tools.train_xgb_v4_5 --pids $PIDS --forward-hours 168 --label-thresh 0.06
   python -m tools.v4_5_horizon_compare --pids $PIDS --horizons 24,72,168
   # Open backend/tools/xgb_v4_5_horizon_compare.html — review per-horizon + per-rule
   ```
3. Operator launches dev backend: `PORT=8002 python main.py` from `backend/` cwd
4. Backend restart picks up new code + migration (3 new columns added to cnn_scans)
5. v4.5 shadow path begins persisting `(p_down, p_neutral, p_up)` per scan
6. **Shadow week** (7 days): telemetry accumulates while v3 still drives trading on 8001
7. **Promotion gate** (after shadow week):
   - Query `cnn_scans` for v4.5 probs joined to `signal_outcomes` on (pid, ts)
   - Score each decision rule against live trade outcomes (precision/recall of BUY/SELL signals vs WIN/LOSS classification)
   - Operator picks (horizon, rule) combo
   - Copy winning artifacts: `cp xgb_model_v4_5_h<W>.json xgb_model_v4_5.json` (and features)
   - One-line wire-up edit in `cnn_agent.generate_signal` to use chosen decision rule on v4.5 probs (replaces v3 driver)
   - Commit + restart 8001 backend → v4.5 now drives trading

## AUC-vs-outcomes query (post-shadow-week)

```sql
SELECT
  COUNT(*) AS n_outcomes,
  AVG(s.xgb_prob_v4_5_up)      AS mean_p_up,
  AVG(s.xgb_prob_v4_5_neutral) AS mean_p_neutral,
  AVG(s.xgb_prob_v4_5_down)    AS mean_p_down,
  o.outcome_class
FROM cnn_scans s
JOIN signal_outcomes o ON o.scan_id = s.id
WHERE s.scanned_at >= '<promote_ts - 7d>'
  AND s.xgb_prob_v4_5_up IS NOT NULL
GROUP BY o.outcome_class;
```

Python-side: for each decision rule, compute precision/recall against WIN/LOSS labels. Spec for that script (~50 LOC) is its own deliverable post-shadow; not in this commit.

## Non-goals (out of v4.5 scope)

- **v4 binary cutover** — superseded; v4 path stays callable but isn't called from cnn_agent after this lands
- **Adding marketcap channels** (deferred to v4.6 / Step B.2)
- **Multi-class calibration** — start without; add Platt scaling in v4.5.1 if shadow shows miscalibration
- **Auto-promotion** — operator-gated only per [[feedback_backend_port_isolation]]
- **Modifying CNN** per [[feedback_xgb_focus_not_cnn]]
- **Frontend changes** — frontend keeps hitting 8001 with v3-driven decisions until promote
- **3-class label tuning beyond the sweep horizons** — h24/h72/h168 with chosen thresholds; finer tuning is its own iteration

## Files summary

| Action | Path | LOC est |
|---|---|---|
| Create | `backend/tools/xgb_v4_5_features.py` | ~180 |
| Create | `backend/tools/train_xgb_v4_5.py` | ~320 |
| Create | `backend/tools/v4_5_horizon_compare.py` | ~280 |
| Create | `backend/migrations/xgb_v4_5_shadow_20260517.py` | ~40 |
| Create | `backend/tests/test_xgb_v4_5_features.py` | ~250 |
| Create | `backend/tests/test_train_xgb_v4_5.py` | ~180 |
| Create | `backend/tests/test_v4_5_horizon_compare.py` | ~120 |
| Edit | `backend/tools/xgb_features.py` | +5 LOC v4.5 dispatcher branch |
| Edit | `backend/agents/xgb_signal.py` | +110 LOC v4.5 state + 3 fns |
| Edit | `backend/database.py` | +12 LOC (3 columns + INSERT + ALTER) |
| Edit | `backend/agents/cnn_agent.py` | +10 LOC write-through to save_cnn_scan |
| Edit | `backend/tests/test_xgb_signal.py` | +90 LOC v4.5 shadow tests |
| Edit | `backend/tests/test_database.py` | +35 LOC persistence tests |
| Edit | `backend/tests/test_mc_migration.py` | +30 LOC idempotency tests |
| Edit | `CLAUDE.md` | invariant #17 (3-class telemetry contract) |
| Edit | `CHANGELOG.md` | Session entry |
| Memory | `coinbase_trader_session_log.md` | append entry |

**Net:** 7 new files (~1370 LOC), 7 edits (~290 LOC), single atomic commit. Plus operator-triggered 3-horizon sweep + shadow week + promote.

## CLAUDE.md invariant to add (#17)

> **3-class telemetry contract** — When persisting v4.5+ multi-class probabilities to `cnn_scans`, ALL probabilities for a given model version (e.g., all 3 of `xgb_prob_v4_5_down/neutral/up`) must be written together or all NULL — never partial. Probabilities should sum to ~1.0 (after clip + renormalize). Downstream consumers (decision rules, calibration analysis) rely on this invariant. Mirrors invariant #14's MC chain rule for telemetry consistency.
