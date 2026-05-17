# XGB Mixed-Lookback Feature Set (v3) — Design

**Date:** 2026-05-16
**Author:** Claude Code (brainstormed with operator)
**Status:** Draft awaiting operator review
**Scope:** `backend/` only (per CLAUDE.md scope rule)
**Skill chain:** `superpowers:brainstorming` → this spec → `superpowers:writing-plans`

---

## 1. Problem

The current XGB booster (`backend/xgb_model.json`, 200 trees, 280 features, feature_set v1) is dominated by short-window intra-bar features. Per-channel gain rollup:

| Top channels (gain) | Bottom channels (gain) |
|---|---|
| ch10 intra-bar pos (325), ch2 HL range (246), ch1 log volume (241), ch14 StochRSI (232) | ch20 funding (20), ch27 OI (17), ch17/18/19 masked (0) |

Live calibration over the 5-day window 2026-05-10 → 2026-05-15 is **inverted** in the [0.55, 0.70) `xgb_prob` band — higher confidence predicts *lower* win-rate:

| xgb_prob bucket | n | WR | sum PnL |
|---|---:|---:|---:|
| [0.55, 0.60) | 340 | 44% | -$30.89 |
| [0.60, 0.65) | 62 | 37% | -$16.81 |
| [0.65, 0.70) | 38 | **29%** | -$9.63 |
| [0.70, 0.75) | 9 | 44% | -$3.71 |
| [0.75, 1.00) | 4 | 25% | -$1.35 |

Hypothesis: the booster has learned intra-bar shape patterns that do not predict 4-hour forward returns under the current market regime. Longer-window macro information (trend regime, vol regime, cross-asset sentiment) is computed and fed into the model but underweighted — top gain features are all single-window short-horizon stats.

## 2. Goal

Re-shape the XGB feature space so longer-window macro signals carry meaningful weight in BUY/SELL decisions, while preserving the v1 fallback for rollback.

## 3. Non-goals

- No change to the CNN architecture or its dataset cache.
- No change to TECH agent, exit logic, order executor, WebSocket plumbing.
- No change to `cnn_buy_threshold` / `cnn_sell_threshold` (0.55 / 0.40 stay; only the underlying probability distribution shifts).
- No live-retraining pipeline. Trainer remains offline (Colab + artifact swap).
- No auto-rollback on inference errors (operator-driven only).

## 4. Approach

Extend the XGB feature extractor to a tiered, mixed-lookback layout (feature_set v3). Each channel is assigned a primary tier; meso/macro-tier channels produce stats over *both* their tier window AND the 60-bar micro baseline ("stacked windows"). Apply XGB `feature_weights` at train time so meso/macro features have higher column-sampling probability.

### 4.1 Tier assignment

| Tier | Lookback | Channels | Stats per channel | Channel count | Live features | Zero-slot features |
|---|---:|---|---:|---:|---:|---:|
| Micro (non-masked) | 60 bars (2.5 d) | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,22,23 | 10 (single window) | 18 | 180 | 0 |
| Micro (masked)     | 60 bars         | 17, 18, 19                                  | 10 (zero stats)    | 3  | 0   | 30 |
| Meso               | 60 + 168 bars (1 wk) | 15, 24, 25, 26                         | 20 (two windows × 10 stats) | 4 | 80 | 0 |
| Macro              | 60 + 336 bars (2 wk) | 20, 21, 27                             | 20 (two windows × 10 stats) | 3 | 60 | 0 |
| **Total**          |                 |                                             |                    | **28** | **320 live** | **30 zero** |

Total feature_names: **350** (320 live + 30 zero-slot for masked channels, kept for parity/forward-compat).

Notes:
- Channels 17/18/19 remain MASKED (constant zeros at inference, per `_TRAINING_CONSTANT_CHANNELS`); they keep 10 zero-feature slots for name-table consistency with v1.
- Channels 21 (BTC return correlation) and 24 (IV/RV20 spread) — currently DROPPED at the trainer (`XGB_DROP_CHANNELS={21,24}`) per #146 — are reactivated under v3 with longer windows (ch21 → macro, ch24 → meso). The +0.002 AUC penalty from #146 was observed with their short 20-bar windows; the longer windows may resolve the noise issue.
- Stats are the existing 10: `last, mean, std, slope, min, max, pct_rank, delta_5, delta_10, delta_30`.

### 4.2 Feature naming scheme

- Micro: `chN_<stat>` (unchanged from v1 — 180 names)
- Meso:  `chN_m060_<stat>` + `chN_m168_<stat>` (80 names)
- Macro: `chN_m060_<stat>` + `chN_m336_<stat>` (60 names)

The `_mWWW_` infix is the load-time tag `xgb_signal._try_load` greps for to detect v3 vs v1.

### 4.3 Feature weights

`xgb.train(params, dtrain, feature_weights=[...], ...)` with weights for all **350** feature_names entries (live + zero-slot):
- 1.0 for all live micro features (180 entries)
- 2.0 for all meso features (80 entries)
- 3.0 for all macro features (60 entries)
- 0.0 for masked-channel zero-slot features (30 entries) — never sampled

This is XGBoost's column-sampling bias (`colsample_bytree`-aware). Soft bias — the booster still chooses the best split; macro features are oversampled into each tree's candidate pool 3× more often than micro. Masked channels' zero-slot features get weight 0 so they're never even considered.

## 5. Architecture

```
                          ┌─────────────────────────────┐
                          │   training pipeline          │
                          │   tools/train_xgb.py (v3)    │
                          └──────────────┬──────────────┘
                                         │
                       ┌─────────────────┴─────────────────┐
              ┌────────▼─────────┐               ┌─────────▼──────────┐
              │ tiered_history    │               │ xgb_features        │
              │ (NEW service)     │◄──────────────┤ feature_set="v3"    │
              │ parquet→{m,e,a}   │   slices       │ (extended module)   │
              └────────▲─────────┘               └─────────┬──────────┘
                       │                                   │
                       │ inference                         ▼
            ┌──────────┴──────────┐         backend/xgb_model.json (v3 booster)
            │ agents/xgb_signal    │         backend/xgb_features.json (feature_set=v3)
            │ xgb_prob(channels,   │         backend/xgb_calibration.pkl ({feature_set,calibrator})
            │          pid=None)   │
            └──────────┬──────────┘
                       │
                       ▼
           agents/cnn_agent._cnn_prob()
           (passes pid=product["product_id"] via new kwarg)
```

### 5.1 Files touched

| Path | Action | Purpose |
|---|---|---|
| `backend/services/tiered_history.py` | NEW | `fetch_tiered(pid, source, now_ts) → {micro,meso,macro}` |
| `backend/tools/xgb_features.py` | EXTEND | Add `_extract_v3()`, `_v3_feature_names()`, `feature_weights_v3()` |
| `backend/agents/xgb_signal.py` | EXTEND | Auto-detect v3 via feature_names; route v3 via tiered_history; accept `pid` kwarg |
| `backend/agents/cnn_agent.py` | EDIT | Pass `pid=product["product_id"]` to `xgb_signal.xgb_prob` under `MODEL_BACKEND=xgb` |
| `backend/tools/train_xgb.py` | EDIT (file already exists) | `--feature-set v3` flag; tiered fetch; feature_weights vector; v3 metadata in features.json. `tools/train_xgb_prod.py` (sibling) gets the same treatment so production retrains pick up v3. |
| `backend/tools/fit_xgb_calibration.py` | EDIT | Refit isotonic on v3 raw outputs; pickle dict `{calibrator, feature_set}` |
| `backend/xgb_model.json`, `xgb_features.json`, `xgb_calibration.pkl` | REPLACED at cutover | v1 backed up to `*.bak_v1_<date>` first |

### 5.2 Unchanged

- `FeatureBuilder` in `cnn_agent.py` (still produces [28×60] tensor for the indicator-snapshot columns persisted to `cnn_scans`)
- `_DATASET_CACHE_VERSION` (v3 reads parquet directly via `tiered_history`, bypassing the CNN cache)
- CNN architecture, auto-train (already gated off under `MODEL_BACKEND=xgb` per #300)
- LGBM, Hurst, regime gates (already skipped under `MODEL_BACKEND!=cnn` per #232)
- Side gate thresholds, exit logic, WebSocket, order executor, TECH agent

## 6. Components

### 6.1 `services/tiered_history.py` (NEW)

```python
def fetch_tiered(
    pid: str,
    source: Literal["parquet","live"] = "live",
    now_ts: float | None = None,
) -> dict:
    """Returns {"micro": List[Candle], "meso": List[Candle], "macro": List[Candle]}.

    micro = last 60 hourly bars
    meso  = last 168 hourly bars (1 week)
    macro = last 336 hourly bars (2 weeks)

    source="parquet": read backend/data/history/{pid}.parquet — used by trainer
    source="live":    sync SQLite read via sqlite3.connect (NOT aiosqlite — see
                      Sync-vs-async note below); if SQLite < 336, fall back to
                      parquet for the prefix.

    Short-history return: any tier whose underlying series is shorter than
    its required length is returned as an empty list. Caller (_extract_v3)
    interprets [] as "fill that tier's slots with 0.0".
    """
```

**Sync-vs-async note:** `fetch_tiered` is **synchronous** even though the rest of the backend uses `aiosqlite`. Reason: the inference call site is `xgb_signal.xgb_prob` (sync) → called from `cnn_agent._cnn_prob` (sync) → called from `generate_signal` (async). Bubbling `await` through `xgb_prob` and `_cnn_prob` would touch many callers + tests. The per-scan inference read is ~400 rows (sub-millisecond) and runs once per product per 300s — briefly blocking the event loop is acceptable. The trainer is sync code anyway.

Pure data layer. ~80 LOC. No model code. Single source of truth for "what does each tier see."

### 6.2 `tools/xgb_features.py` (extended)

```python
MASKED_CHANNELS = frozenset({17, 18, 19})           # existing
MESO_CHANNELS   = frozenset({15, 24, 25, 26})       # NEW
MACRO_CHANNELS  = frozenset({20, 21, 27})           # NEW
# MICRO_CHANNELS = remaining 18 channels (computed, not stored)

TIER_WINDOWS = {"micro": 60, "meso": 168, "macro": 336}

def extract_features(arr_or_dict, feature_set="v1"):
    if feature_set == "v3":
        return _extract_v3(arr_or_dict)   # arr_or_dict = {micro, meso, macro}
    # existing v1/v2 paths unchanged

def _extract_v3(candles_by_tier: dict) -> tuple[np.ndarray, list[str]]: ...
def _v3_feature_names() -> list[str]: ...
def feature_weights_v3() -> np.ndarray: ...
```

### 6.3 `agents/xgb_signal.py` (extended)

Two changes:
1. `_try_load` scans `feature_names` for `_m060_` / `_m168_` / `_m336_` infix; sets module-level `_feature_set = "v3"` if found.
2. `xgb_prob(channels, pid: str | None = None)` (stays **sync**) — when `_feature_set == "v3"`:
   - Ignores `channels` arg (legacy [28×60] tensor — still passed by the caller for cache symmetry but unused).
   - Calls `tiered_history.fetch_tiered(pid, source="live")` (sync — see 6.1).
   - Runs `_extract_v3` on the result.
   - Predicts; applies calibrator if `xgb_calibration.pkl` is dict-shaped with `feature_set == "v3"` (else logs + skips calibration to avoid v1-fit-on-v3 mapping).
   - Clips to [0.01, 0.99].
   - Returns 0.5 if `pid is None` (with warning log).

### 6.4 `agents/cnn_agent.py` (one-line edit)

In `_cnn_prob` (cnn_agent.py:1804–1820 region):
```python
if config.model_backend == "xgb":
    return xgb_signal.xgb_prob(channels, pid=pid)   # add pid kwarg
else:
    return self._cnn_torch_prob(channels)            # existing torch path
```
`pid` is already in scope in `generate_signal` (the caller).

### 6.5 `tools/train_xgb.py` (edit)

Add `--feature-set v3` flag. When set:
- Per training pid: `tiered_history.fetch_tiered(pid, source="parquet", now_ts=sample_ts)` for each rolling sample.
- `extract_features(tiers, feature_set="v3")` builds the 350-element vector (320 live + 30 zero-slot for masked channels).
- Label: `1 if close[t+H] > close[t] else 0`, H=4 (unchanged from v1).
- `xgb.train(..., feature_weights=feature_weights_v3(), ...)`.
- Writes `xgb_model.json` + `xgb_features.json` with `{"feature_set": "v3", "feature_names": [...], "best_params": {...}, "feature_weights": [...]}`.
- Atomic write (tmp + rename) so a mid-run failure leaves prior artifacts intact.

### 6.6 `tools/fit_xgb_calibration.py` (edit)

- Re-fit isotonic against v3 raw outputs vs realized 4h outcomes over the same held-out window protocol as today.
- Pickle as `{"calibrator": isotonic, "feature_set": "v3"}` (dict shape, not bare object — versioning the artifact). `xgb_signal._try_load` accepts both shapes (bare = assume v1, dict = use the stored tag) for backward compatibility through the cutover.

## 7. Data flow

### 7.1 Training (offline, one-shot per retrain)

```
tools/train_xgb.py --feature-set v3
  │
  ├─ for each pid in training set:
  │     candles_by_tier = tiered_history.fetch_tiered(
  │         pid, source="parquet", now_ts=sample_ts)
  │     for each rolling sample inside the parquet window:
  │         feats, names = extract_features(candles_by_tier, "v3")
  │         label = 1 if close[t+H] > close[t] else 0
  │         X.append(feats); y.append(label)
  │
  ├─ dtrain = xgb.DMatrix(X, label=y, feature_names=names)
  ├─ feature_weights = feature_weights_v3()
  ├─ booster = xgb.train(params=best_params, dtrain=dtrain,
  │                      num_boost_round=200,
  │                      feature_weights=feature_weights)
  ├─ booster.save_model("backend/xgb_model.json")
  └─ json.dump({...metadata...}, "backend/xgb_features.json")

Then:
tools/fit_xgb_calibration.py
  └─ refit isotonic on v3 raw outputs
  └─ pickle {"calibrator": isotonic, "feature_set": "v3"} to xgb_calibration.pkl
```

### 7.2 Inference (every cnn_agent scan, every 300 s)

```
cnn_agent.generate_signal(product)
  │
  ├─ pid = product["product_id"]
  ├─ channels = FeatureBuilder.build(...)    # [28 x 60] — unchanged
  │                                          # still used for cnn_scans indicator columns
  ├─ if config.model_backend == "xgb":
  │     cnn_prob = xgb_signal.xgb_prob(channels, pid=pid)
  │     │
  │     └─ xgb_signal.xgb_prob (v3 path, still sync):
  │            tiers = tiered_history.fetch_tiered(pid, source="live")
  │            if tiers["macro"] == []:           # short-history product
  │                # _extract_v3 zero-fills macro slots automatically
  │            feats, _ = extract_features(tiers, "v3")
  │            raw   = booster.predict(DMatrix(feats, feature_names=...))
  │            calib = isotonic.transform(raw) if calibrator else raw
  │            return clip(calib, 0.01, 0.99)
  │
  ├─ side gate (unchanged): >0.55 → BUY, <0.40 → SELL, else HOLD
  └─ persists to cnn_scans (cnn_prob and xgb_prob columns from v3 booster)
```

### 7.3 Inference cost

- Per-scan DB read: `database.get_candles(pid, limit=400)` (vs current ~140). One read per product per scan. 54 products × every 300s ≈ 0.2 reads/sec average, ~11 reads/sec peak burst. SQLite handles trivially.
- Per-scan compute: one extra extractor invocation (vectorized numpy over ~400 candles) + one booster predict on 350 features (320 live + 30 zero). Microseconds; insignificant against the existing scan overhead.

## 8. Error handling

| Condition | Behavior | Test |
|---|---|---|
| Macro tier empty (short history) | `_extract_v3` zero-fills macro slots; micro/meso unaffected | `test_extract_v3_zero_fills_missing_tier` |
| Both meso AND macro empty | Only micro slots populated; booster output dominated by base_score (~0.51 calibrated) — sub-threshold, won't fire | implied by zero-fill semantics |
| `pid=None` while v3 loaded | Log warning, return 0.5 | `test_v3_missing_pid_neutral` |
| `tiered_history.fetch_tiered` raises | Existing `try/except` in `xgb_prob` catches; returns 0.5 | `test_v3_returns_neutral_on_tiered_fetch_failure` |
| Calibrator metadata `feature_set != booster_feature_set` | Log warning, skip calibration (raw passthrough) | `test_v3_skips_v1_calibrator` |
| Booster ↔ features.json column-count mismatch | Existing `DMatrix` `ValueError` caught by `xgb_prob` except block; returns 0.5 | `test_v3_neutral_on_dmatrix_mismatch` |
| Trainer fails mid-run | Atomic write (tmp + rename) leaves prior artifacts intact | `test_train_xgb_v3_atomic_write` |
| Invalid `feature_weights` (length / sign) | `feature_weights_v3()` raises `ValueError` immediately | `test_feature_weights_v3_rejects_bad_weights` |
| OKX funding/OI outage at training time | Channels populated with zeros (existing `_aligned_funding_rates` behavior); zero-fill propagates through tiered_history | not a v3 issue; documented |

### What we don't handle
- **Auto-rollback to v1 on inference errors** — operator-driven via file rename + reload endpoint. Auto-rollback risks oscillation on transient failures.
- **Per-product calibrator** — one global isotonic, same as today.
- **Live retraining in the scan loop** — trainer remains offline.

## 9. Testing

Per CLAUDE.md TDD: failing test first, then implementation, then commit. All tests live under `backend/tests/`.

| File | Status | Tests | Coverage |
|---|---|---:|---|
| `test_tiered_history.py` | NEW | 12 | Slice contracts, short-history empties, source dispatch, leak prevention, chronological order |
| `test_xgb_features_v3.py` | NEW | 15 | 350-element vector (320 live + 30 masked zero) count, name scheme, per-tier stat counts, zero-fill, feature_weights validation, name disjointness from v1 |
| `test_xgb_signal.py` | EXTEND | +6 | v3 auto-detection, pid plumbing, calibrator metadata mismatch, neutral fallbacks |
| `test_cnn_agent.py` | EXTEND | +2 | `pid` kwarg passes under xgb backend; not passed under cnn backend |
| `test_train_xgb.py` | NEW/EXTEND | 5 | v3 flag dispatch, feature_weights wiring, metadata write, atomic write, short-history product skip |
| **Total** | | **40** | |

What we don't unit-test:
- Trained booster's accuracy / AUC — operator decides at training time whether to ship the artifact.
- Isotonic calibrator's exact mapping — covered by existing `test_fit_xgb_calibration.py` shape tests, which re-run against v3 outputs unchanged.
- End-to-end cnn_agent inference — manual smoke before the cutover commit.

Shell cleanup per CLAUDE.md: `Get-Process python | Stop-Process -Force` after each test run.

## 10. Rollout & cutover

Per operator direction: **no shadow window, no AUC gate.** Cutover commit IS the live event. DRY_RUN stays `true` from .env; operator flips to `false` later at their discretion.

### Phase 0 — Build (~1 week)
TDD all 40 tests RED → GREEN. Implement 6 modules. Per-module commits with tests. Sync `coinbase_trader_architecture.md` + `CHANGELOG.md` on each commit. No live behavior change yet — v3 booster doesn't exist on disk.

### Phase 1 — Train v3 (offline)
1. Run `tools/train_xgb.py --feature-set v3` (Colab or local rig).
2. Run `tools/fit_xgb_calibration.py` against the v3 booster.
3. Stage artifacts locally as `xgb_model.json` / `xgb_features.json` / `xgb_calibration.pkl` — production filenames.

### Phase 2 — Cutover commit ("live upon commit")
Single atomic commit does all of:
1. **Backup on host** (gitignored): `mv backend/xgb_model.json backend/xgb_model.json.bak_v1_20260516` (same for features.json + calibration.pkl).
2. **Drop in** the v3 artifacts to production filenames.
3. **Commit** Phase 0 code + CHANGELOG entry noting cutover happened in this commit.
4. **Hot-reload**: `POST /api/cnn/model/reload` (existing endpoint #69) picks up v3 without backend restart.
5. **DRY_RUN=true** stays as-is. v3 immediately drives signal generation; trades are paper only until operator flips DRY_RUN.

### Phase 3 — Observation
- DRY_RUN stays true. Operator watches paper PnL via `trades` / `signal_outcomes`.
- Calibration-by-bucket query (same as the v1 audit run today) is the primary signal:
  ```sql
  SELECT ROUND(cnn_prob,2) bucket, COUNT(*) n,
    SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) wr,
    ROUND(SUM(pnl),2) total_pnl
  FROM trades t
  JOIN cnn_scans s ON s.product_id=t.product_id
   AND s.scanned_at=(SELECT MAX(scanned_at) FROM cnn_scans
                     WHERE product_id=t.product_id AND scanned_at<=t.opened_at)
  WHERE t.agent='CNN' AND t.closed_at >= 'CUTOVER_DATE'
  GROUP BY bucket ORDER BY bucket;
  ```
- When satisfied with realized paper PnL: flip `DRY_RUN=false` in .env and reload — no code change.
- If bad: run rollback.

### Rollback procedure
30-second file rename, no code revert:
```
cd backend
mv xgb_model.json xgb_model.json.bak_v3_<date>
mv xgb_features.json xgb_features.json.bak_v3_<date>
mv xgb_calibration.pkl xgb_calibration.pkl.bak_v3_<date>
mv xgb_model.json.bak_v1_<date> xgb_model.json
mv xgb_features.json.bak_v1_<date> xgb_features.json
mv xgb_calibration.pkl.bak_v1_<date> xgb_calibration.pkl
curl -X POST http://localhost:8000/api/cnn/model/reload -H "x-api-key: <key>"
```

### Accepted risks (operator-acknowledged)
- **No shadow window.** v3 produces signals from commit-second-zero; only realized paper PnL flags problems.
- **No AUC gate.** Trainer's output ships regardless of validation AUC vs v1's 0.5284 baseline.
- DRY_RUN=true at cutover bounds the blast radius to paper losses until operator flips it.

## 11. Memory + CLAUDE.md sync (required per CLAUDE.md)

At cutover commit:
- `memory/coinbase_trader_architecture.md` — feature_set v3, tier layout, feature_weights values, tiered_history service
- `memory/coinbase_trader_schema.md` — no new columns
- `memory/trading_app_bugs_fixed.md` — log the inverted-calibration finding (2026-05-16) and v3 fix attempt
- `memory/trading_app_thresholds.md` — note `cnn_buy_threshold=0.55` unchanged (semantics same, distribution shifted)
- `polymarket_app/CHANGELOG.md` — entry for cutover commit referencing this spec
- `polymarket_app/CLAUDE.md` invariants — add v3 invariant: "Feature set v3 uses 3 tiers (micro 60 / meso 168 / macro 336), 350 feature_names (320 live + 30 zero-slot for masked channels), feature_weights (micro 1.0 / meso 2.0 / macro 3.0 / masked 0.0). Tier assignment in `tools/xgb_features.py:MESO_CHANNELS|MACRO_CHANNELS`."

## 12. Open questions

None blocking — operator has answered all clarifying questions during brainstorming. Remaining choices (calibration window length for refit, exact `num_boost_round`, whether to add `monotone_constraints`) are tuning decisions for Phase 1 and can be made by the operator at training time.

## 13. References

- Brainstorming session: 2026-05-16 (this spec)
- v1 booster build: Session 58.4 (#135), 2026-05-03
- v1 calibrator: Session 58.x (#180/#187)
- Channel ablation finding (ch21/ch24 dropped): #146
- CNN auto-train gate under xgb: #300, Session 58.67
- Inverted calibration finding (live evidence): `backend/tools/xgb_model_breakdown.html`, generated 2026-05-16
- Existing trainer protocol: `tools/train_cloud.py` (CNN side; v3 trainer mirrors structure)
- Hot-reload endpoint: Session 41 #69
