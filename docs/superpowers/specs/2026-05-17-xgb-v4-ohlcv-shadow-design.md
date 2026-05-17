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

Pure functions. No state. No `xgb_features.py` imports. Owns its own constants:

```python
N_CHANNELS_V4 = 5
TIER_WINDOWS_V4 = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V4 = {"micro": 1.0, "meso": 2.0, "macro": 3.0}
_STAT_NAMES_V4 = ("last", "mean", "std", "slope", "min", "max",
                  "pct_rank", "dlt5", "dlt10", "dlt30")
_CHANNEL_FIELDS = ("open", "high", "low", "close", "volume")  # idx 0..4
```

Public:
- `_v4_feature_names() -> List[str]`  → 150 names, layout `ch{c}_{tier}_{stat}`
- `feature_weights_v4() -> np.ndarray` → 150-long weight vector
- `_extract_v4(candles_by_tier: dict) -> Tuple[np.ndarray, List[str]]` → `(features[1, 150], names[150])`

Behavior: for each channel c in 0..4, for each tier in (micro, meso, macro), pull the corresponding OHLCV field from the tier's candle list, compute 10 stats, write into the output slot. Channel 4 (volume) just reads `candle["volume"]` instead of `candle["close"]`. Missing/empty tier → 10 zero slots (same convention as v3's `_stats_from_candles`).

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

Mirrors structure of `train_xgb_v3.py`:
- For each tracked pid, read OHLCV from `backend/data/history/<pid>.parquet`
- Build training samples: at every bar `i` (starting from `i >= 336` for macro coverage), produce the 3-tier candle slices ending at `i`, extract 150 features via `_extract_v4`
- Build labels using the **same triple-barrier params as v3** (read from `agents/cnn_agent.py` constants `_FORWARD_HOURS`, `_LABEL_THRESH` etc. for parity — labels are deterministic given the same params + candles)
- Walk-forward split (matches v3's split logic), embargo = `forward_hours`
- xgb.train with `binary:logistic`, 200 trees, depth 4, lr 0.1, colsample_bytree 0.8, `feature_weights = feature_weights_v4()`, `colsample_bytree=0.8`
- Calibrate isotonic on a held-out fold
- Write artifacts:
  - `backend/xgb_model_v4.json` (booster, tmp file then atomic rename — note `.json` last per prior fix)
  - `backend/xgb_features_v4.json` (the 150 feature names)
  - `backend/xgb_calibration_v4.pkl` (dict `{"calibrator": IsotonicRegression, "feature_set": "v4"}`)
- Print AUC on calibration fold

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
