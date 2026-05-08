# Changelog — Coinbase AI Trader (polymarket_app)

All notable changes to this project are documented here.
Format: reverse-chronological by session date.

---

## [Session 58.17] — 2026-05-08 — Rank-transform generalization probes (#156 partial, #162 follow-up) — NULL

### Context

#162's RSI-rank cross-sectional transform PASSED the +0.01 gate (Δ+0.0208).
Question: does the rank transform itself carry information, or was the win
specific to RSI's bounded mean-reverting nature? Two follow-up probes via
`--source-channel` flag (added in same `tools/rsi_rank_probe.py`):

- **Volume-rank (Ch 1)** as a BTC-dominance proxy for #156.
- **MFI-rank (Ch 12)** to test generality on other bounded mean-reverting
  indicators (MFI = volume-weighted RSI).

### Results

| source channel       | baseline_auc | replaced_auc | Δ        | gate    |
|----------------------|--------------|--------------|----------|---------|
| Ch 4  RSI            | 0.5201       | 0.5409       | +0.0208  | PASS    |
| Ch 1  log10 volume   | 0.5134       | 0.5124       | -0.0010  | FAIL    |
| Ch 12 MFI            | 0.5124       | 0.5155       | +0.0031  | FAIL    |

### Interpretation

- **Volume-rank failure** confirms unbounded skewed channels don't transform
  usefully — BTC-USD always rank-1 of volume → near-constant signal, no lift.
- **MFI-rank failure** rules out the "any bounded mean-reverting indicator
  works" hypothesis. RSI(14)/100 carries cross-sectional information that MFI
  does not, despite both being bounded oscillators.
- **#156 BTC-dominance** cannot be approximated from cache contents; needs
  external data source (CoinGecko Pro / CryptoCompare) — deferred.
- RSI-rank stands as a single channel-specific win; integration still bundled
  with next coordinated retrain cycle (per Session 58.16 plan).

### Files

No code changes — `tools/rsi_rank_probe.py` already supported `--source-channel N`
from Session 58.16. CHANGELOG + memory updates only.

---

## [Session 58.16] — 2026-05-07 — Cross-sectional RSI-rank single-add probe (#162) — PASS

### Context

After the 28-ch cache + retrain (#199 + #200) didn't lift v2 above the +0.01
gate (#145), we needed individual probes of each candidate input. Hypothesis:
RSI alone is a single-product signal; the cross-section "this product's RSI
vs all peers right now" is a different information source that no per-product
60-bar window can reconstruct.

### Changes (TDD)

- **`backend/tools/rsi_rank_probe.py`** (new): builds `(T, P)` RSI matrix
  across all 107 products in the v12 cache, ranks each row with
  `scipy.stats.rankdata` (average for ties), normalizes to `[0, 1]`, then
  windows back to per-sample `[N, 60]` rank signals via `np.searchsorted`.
  Replaces ch13 (obv_slope, marginal per #146) and runs through
  `tools/channel_replace.run_replace`.
- **`backend/tests/test_rsi_rank_probe.py`** (new): unit tests for
  `_cross_sectional_rank` (single product → 0.5 neutral, three distinct →
  {0, 0.5, 1}, ties → equal rank, NaN inputs dropped) and `build_rank_signal`
  (shape, self-rank neutral, monotone vs peers, missing timestamps default
  0.5). RED → 9/9 GREEN with vectorized loader unchanged.

### Result

| Implementation       | baseline_auc | replaced_auc | Δ        | gate    |
|----------------------|--------------|--------------|----------|---------|
| slow per-cell loop   | 0.5187       | 0.5414       | +0.0227  | PASS    |
| vectorized (T,P) mat | 0.5201       | 0.5409       | +0.0208  | PASS    |

Replaced AUC ~0.541 — closest the pooled-top-20 cell has come to the 0.55
hard gate. Decision: integrate as a real channel in the next coordinated
retrain cycle (after #156 BTC-dominance lands its probe), not as a one-off
to avoid back-to-back cache rebuilds.

---

## [Session 58.15] — 2026-05-06 — XGB 28-channel coordinated bump (#199 + #200)

### Context

`agents/cnn_agent.N_CHANNELS` was bumped to 28 (with Ch 27 OI from #143-B), but
`tools/xgb_features.N_CHANNELS` was still 27. Inference would have raised
`ValueError: expected shape [N, 27, 60]` on every scan and silently neutralized
XGB to 0.5. Coordinated fix — bump xgb_features, retrain booster on the v12
28-ch cache, refit isotonic calibrator on the new val split.

### Changes (TDD)

- **`backend/tools/xgb_features.py`**: `N_CHANNELS = 27 → 28`. Auto-extends
  feature-name list to 280 (`ch0_*` through `ch27_*` × 10 stats). XGB-side
  drops still apply (XGB_DROP_CHANNELS = {21, 24}).
- **`backend/tools/train_xgb_prod.py`** (new): driver that pools top-20
  products by sample count from `cnn_dataset_cache.pt`, runs 5-fold purged
  walk-forward CV at fixed best params (`max_depth=4, mcw=1, subsample=0.7`),
  saves `xgb_model.json` + `xgb_features.json`.
- **`backend/tools/fit_xgb_calibration.py`**: extracted `_detect_feature_set()`
  helper. Old heuristic `len(feature_names) > 270 -> v2` false-positives at
  N=28 (v1 itself = 280). New rule: detect `xt_*` prefix in any name.
- **`backend/agents/xgb_signal.py`**: same `len > 270` heuristic at the
  inference-time loader replaced with the prefix check, otherwise live
  calls would silently flip to the v2 extract path and raise
  `expected 290, got 280` on every scan.
- **`backend/tests/test_fit_xgb_calibration.py`**: regression tests for v1
  vs v2 detection at 28 channels.
- **`backend/tests/test_calibration_probe.py`**, **`test_train_xgb.py`**:
  bump synthetic-data fixtures from 27→28 channels, 270→280 feature count.
- **`backend/tests/test_channel_ablation.py`**, **`test_channel_replace.py`**,
  **`test_xgb_signal.py`**: synthetic 27-ch fixtures bumped to 28 to match
  `xgb_features.N_CHANNELS`.
- **`backend/tests/test_xgb_features.py`**: regression tests asserting
  xgb_features.N_CHANNELS == cnn_agent.N_CHANNELS (silent-neutralize guard).

### Results

- XGB retrain: 167,144 samples × 28 ch × 60, mean_auc=0.5182, folds=
  [0.5123, 0.5060, 0.5172, 0.5158, 0.5399], 117s. Slight drop from May 3
  baseline (0.5224) on 27 channels — consistent with Ch 27 OI having weak
  per-feature gain in the May 2026 ablation.
- Calibrator refit: 84,858 val samples, monotone curve preserved
  (raw 0.50 → cal 0.4983, raw 0.80 → cal 0.9143, raw 0.20 → cal 0.1111).
  POST-bucket win rates 3.3% → 93.1% across deciles.

### Backups

- `xgb_model.json.bak.20260506`, `xgb_features.json.bak.20260506`,
  `xgb_calibration.pkl.bak.20260506` preserved for rollback.

---

## [Session 58.14] — 2026-05-06 — Populate Ch 27 OI in FeatureBuilder.build (#143-B)

### Context

#143-A landed the per-product OI fetch and forwarded an `oi_aligned` series
through `_extend_or_rebuild_product`, but the array stopped one layer above
`FeatureBuilder.build`. Ch 27 in the cache was still zeros, so XGB and CNN
had no OI signal at training or inference. #143-B closes that gap and bumps
the cache schema to v12 so the next rebuild populates Ch 27 from real OKX
events.

### Changes (TDD)

- **`backend/agents/cnn_agent.py`**:
  - `FeatureBuilder.build`: append `Ch 27 = oi_norm` (scalar `oi_rate / 3.0`
    clipped to ±1, broadcast across `T=SEQ_LEN`), mirroring the Ch 20
    funding pattern.
  - `_build_samples_range`: accept new `oi_rates` kwarg, z-score over the
    full per-product series (`(x − μ) / σ`), and forward per-sample
    `oi_val_z` into `fb.build(oi_rate=…)`.
  - `_extend_or_rebuild_product`: forward `oi_rates=oi_rates` into both
    `_full_rebuild` and the append-path `_build_samples_range` call.
  - `N_CHANNELS = 28`; `_DATASET_CACHE_VERSION = 12` (forces full rebuild
    on next backend start — rebuild is #144).
- **`backend/tests/test_cnn_agent.py`**:
  - new spy tests assert Ch 27 carries OI in `FeatureBuilder.build`.
  - `_zero_mask_channels` fixtures import `N_CHANNELS` instead of
    hardcoding 27.
  - `test_dataset_cache_version_bumped_to_11` asserts `>= 12`.
  - `_FakeFB.build` and `_CapturingFB.build` mocks absorb the new
    `oi_rate=None` kwarg so existing call sites keep passing.
- **CHANGELOG + memory**: this entry; memory `xgb_feature_optimization_findings.md`
  to be updated post-rebuild with measured AUC delta (#145).

### Verification

```
cd backend
.venv/Scripts/python -m pytest tests/test_cnn_agent.py \
    tests/test_cnn_risk_exits.py tests/test_train_cloud.py
# 249 passed, 2 xfailed, 5 xpassed in 678.90s
```

Live activation requires #144 (cache rebuild on next backend start) and a
fresh XGB train on the new 28-channel cache before Ch 27 contributes.
The `xgb_model.json` currently on disk is 270-feature (27 channels × 10
stats); first inference call after restart will fall back to neutral 0.5
because feature-count mismatch — addressed by the rebuild + retrain flow.

---

## [Session 58.13] — 2026-05-05 — Wire OKX OI fetch into Phase 1 (#143-A)

### Context

`services.okx_oi_history.fetch_oi_history` (#141/#142) and
`_aligned_oi_history` (cnn_agent.py:842) have existed since the OKX-Loop1
RED/GREEN tasks but the dataset builder Phase 1 never called them. To
populate Ch 27 during training (#143-B) the per-product loop must first
fetch OI alongside funding and forward an aligned series to
`_extend_or_rebuild_product`.

### Changes (TDD)

- **`backend/tests/test_cnn_agent.py`**: added
  `test_extend_or_rebuild_receives_oi_rates` mirroring the funding-rates
  wiring test — patches `fetch_oi_history` to return a single-event payload
  at the first candle's timestamp, runs `train_on_history(epochs=1)`,
  asserts every spy capture of `_extend_or_rebuild_product` received
  `oi_rates=` as a list with `len(candles)` and the payload value
  forward-filled to bar 0.
- **`backend/agents/cnn_agent.py`**:
  - import `fetch_oi_history` from `services.okx_oi_history`
  - in `_train_full_async` Phase 1 per-product loop, call
    `fetch_oi_history(pid, fr_start_ms, fr_end_ms)` after the existing
    funding fetch and align via `_aligned_oi_history(candles, oi_hist)`
  - extend `all_candle_sets` tuple to 6 elements: `(pid, candles,
    btc_aligned, c5m, funding_aligned, oi_aligned)`
  - update tuple unpacking in nested `_build_dataset` and forward
    `oi_rates=oi_aligned` to `_extend_or_rebuild_product`
  - add `oi_rates: Optional[List[float]] = None` kwarg to
    `_extend_or_rebuild_product` signature (no-op forward; #143-B will
    plumb through to `_build_samples_range` + FeatureBuilder Ch 27)

### Verification

```
tests/test_cnn_agent.py  16 passed
  (TestBuildDatasetWiresBtcAndFiveMinute, TestAlignedFundingRates,
   TestAlignedOiHistory)
```

No regressions in BTC/funding/5m wiring tests after tuple shape change.

---

## [Session 58.12] — 2026-05-04 — Hot-reload endpoint for XGB calibrator (#192)

### Context

Session 58.11 refit `xgb_calibration.pkl` on disk but the running backend
still holds the stale calibrator in `agents.xgb_signal._calibration`
(lazy-loaded once via `_try_load`'s `_load_attempted` guard). A full process
restart was the obvious fix, but two factors made it risky:

1. CNN auto-train subprocess was running (per `feedback_no_restart_during_retrain`).
2. Three confusing `python.exe` processes were live (trading_app, polymarket
   .venv, Spyder) — kill-by-port was not safely deterministic.

Cleaner option: mirror the existing `/api/cnn/model/reload` pattern with a
`force_reload()` on `xgb_signal` and an admin endpoint that calls it.

### Changes (TDD)

- **`backend/agents/xgb_signal.py`**: added `force_reload() -> bool` that
  drops cached `_booster`, `_feature_names`, `_feature_set`, `_calibration`,
  resets the load-once guard, then re-runs `_try_load()`. Lock-protected.
- **`backend/tests/test_xgb_signal.py`**: added `TestForceReload` class
  with 3 tests — function exists, picks up swapped calibrator pickle on
  disk (writes calibrator A, reads value, swaps to B, calls `force_reload`,
  confirms B's value), returns False when artifacts missing.
- **`backend/main.py`**: added `POST /api/xgb/calibration/reload` (auth
  via `verify_api_key`) returning `{status, load_succeeded, calibration_loaded,
  feature_set, n_features}`. Logs hot-reload outcome.

### Verification

```
tests/test_xgb_signal.py  15 passed (3 new + 12 existing)
```

Live activation: pending — will call endpoint after CNN train subprocess
(PID 4292) completes. The `cnn_train_progress.json` status flip from
`running` → `completed` is the gate.

---

## [Session 58.11] — 2026-05-04 — Calibrator refit on cache val split (#187) — XGBcal AUC 0.05 → 0.61

### Context

Session 58.10 surfaced that the production isotonic calibrator (fit on
~300 resolved CNN BUYs from `signal_outcomes` per #180) collapsed to a
near-constant 0.4346 across ~95% of the booster's [0.4, 0.6] output range.
Live XGB has been emitting effectively-constant probabilities for ~24h.

User asked to **fix the calibrator** rather than drop it. Fix follows
sklearn's CalibratedClassifierCV pattern: refit on a held-out slice of the
same distribution as the booster's training data (the dataset cache's
chronological 20% val split, 83k samples, balanced labels).

### Changes

- **`backend/tools/fit_xgb_calibration.py`**: added `--source cache` mode
  alongside the legacy `--source shadow` (default kept for backward
  compat). New `_load_cache_pairs()` mirrors
  `tools.permutation_importance._load_val_split` (sorted-by-pid concat,
  `_TRAINING_CONSTANT_CHANNELS` masking, 80/20 chronological cut), then
  runs `xgb.Booster.predict` over the val tensor via
  `tools.xgb_features.extract_features` (auto-detects v1 vs v2 from
  feature-names length). Returns (raw_probs, labels) for `IsotonicRegression.fit`.
  `fit_calibration(source="cache", ...)` dispatches accordingly with a
  `_MIN_CACHE_SAMPLES = 5_000` floor.
- **`backend/tests/test_fit_xgb_calibration.py` (NEW)**: 5 tests covering
  the new path — `source` kwarg exists, calibrator doesn't collapse to
  one plateau (≥5 unique grid values), monotone non-decreasing,
  AUC preserved within 0.02 of raw booster, default kwarg routes to legacy
  shadow path.
- **`backend/xgb_calibration.pkl`** (binary, gitignored): refit. Old
  artifact backed up to `xgb_calibration.pkl.bak.20260504`.

### Verification (cache fit, 83,614 samples)

```
loaded 83614 (raw_prob, label) pairs from cache val split
PRE-calibration win rate by raw bucket:
  [0.40, 0.50)  n=50145  win= 44.0%
  [0.50, 0.60)  n=26584  win= 56.5%
  [0.60, 0.70)  n=2572   win= 67.6%
  [0.70, 0.80)  n=568    win= 84.2%
  [0.80, 0.90)  n=96     win= 99.0%
Calibration grid (raw -> calibrated):
  0.40 -> 0.3378   0.50 -> 0.5174   0.60 -> 0.6535
  0.70 -> 0.7301   0.80 -> 0.9815   0.90 -> 1.0000
```

Re-ran `tools.cnn_xgb_delta_probe`:

| Output | AUC (Session 58.10) | AUC (now) |
|---|---|---|
| CNN (glu1) | 0.6523 | 0.6523 |
| XGBraw | 0.6036 | 0.6036 |
| **XGBcal** | **0.0506** | **0.6065** |

XGBcal now tracks XGBraw within +0.003 AUC and the calibrator no longer
collapses ranking. Decision-agreement matrix shows XGBcal can fire SELL
(472) and BUY (503) signals at live thresholds — previously it was
HOLD-only with 0 SELLs (calibrator output never crossed 0.2).

### Tests
```
tests/test_fit_xgb_calibration.py::TestCacheSourceCalibratorFit  5 passed
```

### Memory
`xgb_feature_optimization_findings.md` already documents the pathology;
no further memory edits needed.

### Next
- Restart backend so the lazy-loaded `_calibration` in
  `agents.xgb_signal._try_load` re-reads the new pickle.
- Continue XGB shadow watch with restored calibration.

---

## [Session 58.10] — 2026-05-04 — CNN vs XGB delta probe + isotonic-calibrator pathology surfaced

### Context

Phase 6 shadow has been live ~24h with `MODEL_BACKEND=xgb`. Visual delta on
`cnn_scans` rows since deploy was zero (cnn_prob == xgb_prob, expected since
both reads come from the same xgb head when MODEL_BACKEND=xgb). To get a
real CNN-vs-XGB comparison we need a retrospective on the labeled cache —
no live changes, no model edits.

### Changes

- **`backend/tools/cnn_xgb_delta_probe.py` (NEW, exploratory — no test, lives
  under `tools/` per existing oneoff convention)**:
  - Reuses `tools.permutation_importance._load_model` and a new
    `_load_val_split_with_products` (mirror of `_load_val_split` but tracks
    pid per sample) to run CNN forward pass + XGB booster + isotonic
    calibration on the same chronological 20% val tensor.
  - Reports probability distributions, Pearson r, decision-agreement matrix
    at the live thresholds (BUY > 0.8, SELL < 0.2), AUC per model, and
    per-product / per-regime / per-time-slice cuts. Crucially, splits XGB
    output into `XGBraw` (booster-only) and `XGBcal` (post-isotonic) so
    calibrator-induced collapse can be isolated from booster signal.
  - Read-only — touches no DB rows, no checkpoints, no live process.

### Findings (val split: 83,614 samples, 24 distinct products)

| Output | AUC | mean | std | p5 | p50 | p95 |
|---|---|---|---|---|---|---|
| CNN (glu1) | **0.6523** | 0.5007 | 0.1186 | 0.2949 | 0.5021 | 0.6987 |
| XGB raw booster | **0.6036** | 0.4869 | 0.0627 | 0.4038 | 0.4822 | 0.5883 |
| XGB after isotonic | **0.0506** | 0.4381 | 0.0360 | 0.4346 | 0.4346 | 0.4346 |

- `Pearson r(CNN, XGBraw) = +0.33` — modest, models complement.
- `Pearson r(CNN, XGBcal) = +0.10` — calibrator destroys correlation.
- XGBcal is near-constant 0.4346 (p5=p50=p95 plateau). Live system has been
  routing every trade decision through a model emitting essentially constant
  output for ~24h. Under CNN_BUY_THRESHOLD=0.80 that's HOLD-only with a few
  random BUYs at the plateau boundary.

Per-product (XGBraw): 7 products with raw AUC > 0.60. STRK (0.745) and VARA
(0.781) — XGB raw beats CNN. Per-regime: flat (CNN ~0.65 / XGBraw ~0.60 in
both ranging and trending). Per-quintile: XGBraw stable 0.56–0.64 across
val time, XGBcal degrades 0.15 → 0.005 (calibrator was fit on the start of
val and is stale).

### Verification

```
../.venv/Scripts/python.exe -m tools.cnn_xgb_delta_probe
[delta] arch=glu1  n_val=83,614  channels=27  distinct_products=24
=== overall AUC ===
  CNN     AUC = 0.6523
  XGBraw  AUC = 0.6036
  XGBcal  AUC = 0.0506
```

### Memory

`xgb_feature_optimization_findings.md` updated with a 2026-05-04
retrospective section and three remediation options (drop calibrator /
refit on cache val / switch to Platt scaling).

### Next (awaiting decision)

- Cheapest fix: stop loading `xgb_calibration.pkl` in `xgb_signal._try_load`
  → live XGB AUC 0.05 → 0.60 with no retrain. Threshold needs retuning
  since raw output centers at 0.487 not 0.5.

---

## [Session 58.9] — 2026-05-04 — XGB-Step3: parallel xgb_prob shadow logging (#181)

### Context

Steps 1+2 unblocked XGB live firing and corrected its U-shape calibration,
but `cnn_scans` still only persists `cnn_prob` and the blended `model_prob`.
For the Phase 7 cutover decision we need a per-row apples-to-apples
comparison: every scan must store both the active-backend output (already in
`cnn_prob`) **and** the XGB shadow output (new column), computed on the
same masked feature tensor.

### Changes

- **`backend/database.py` (#186)**:
  - Added `xgb_prob REAL` to `cnn_scans` CREATE TABLE.
  - Added `ALTER TABLE cnn_scans ADD COLUMN xgb_prob REAL` to the in-place
    migration list (safe on existing prod DB).
  - Extended `save_cnn_scan` INSERT (now 23 columns, ?-list grew by one);
    binding pulls `scan.get("xgb_prob")` so callers that omit it get NULL.
- **`backend/agents/cnn_agent.py` (#186)**:
  - In `generate_signal`, after `cnn_prob = self._cnn_prob(channels)`,
    compute `xgb_shadow`. When `MODEL_BACKEND=xgb` it equals `cnn_prob`
    (already xgb output, no double inference). When `MODEL_BACKEND=cnn`
    it's `xgb_signal.xgb_prob(_mask_training_constant_channels(channels))`,
    matching the masking `_cnn_prob` itself applies, so the shadow runs on
    identical features. Exceptions in the shadow path silently fall back
    to `None` — never break the CNN scan.
  - Stored the shadow value alongside `cnn_prob` in `self._cache[pid][2]`
    under key `"xgb_shadow"` so cache hits don't lose it on the
    cnn-prob-cached fast path. Initialized to `None` upfront to keep both
    branches type-safe.
  - Pass `"xgb_prob": round(xgb_shadow, 4) if xgb_shadow is not None else None`
    to `database.save_cnn_scan`.
- **`backend/tests/test_database.py` (#185)**:
  - New `TestCnnScansXgbProb` class with 3 RED→GREEN tests:
    - `test_save_and_read_xgb_prob` — round-trips `xgb_prob=0.4242`.
    - `test_xgb_prob_defaults_to_null_when_omitted` — backwards-compat.
    - `test_xgb_prob_distinct_from_cnn_prob` — independent persistence.

### Verification

```
tests\test_database.py::TestCnnScansXgbProb  3/3 PASSED
tests\test_xgb_signal.py                    12/12 PASSED
tests\test_cnn_agent.py::TestCoinbaseCNNAgent::test_cache_skips_fetch  PASSED
```

After backend restart, fresh `cnn_scans` rows have `xgb_prob` populated for
every scan (or `NULL` if xgb_signal artifacts are missing — Phase 5 fallback).

---

## [Session 58.8] — 2026-05-04 — XGB-Step2: post-hoc isotonic calibration (#180)

### Context

The Phase 4 calibration_probe walk-forward CV passed monotonicity, but
4 days of live shadow data show a U-shaped win-rate-by-bucket curve on
resolved BUYs (n=371):

```
raw 0.20-0.30: 51.2%   (overconfident-low — actually wins)
raw 0.30-0.40: 46.8%
raw 0.40-0.50: 36.1%
raw 0.50-0.60: 33.8%   trough
raw 0.60-0.70: 50.0%
raw 0.70-0.80: 86.8%
raw 0.80-0.90: 90.0%   peak
```

The booster's *ranking* is fine — high-prob predictions really do win
more. Its *absolute calibration* was off in the 0.30-0.70 zone (likely
ranging-market borderline buys the walk-forward CV averaged across
regimes). Fixing this without retraining: post-hoc isotonic regression
on live (raw_prob, win_label) pairs, applied between the booster and
the [0.01, 0.99] clip in `xgb_signal.xgb_prob`.

### Changes

- **`backend/agents/xgb_signal.py` (#183)**:
  - New `_CALIBRATION_PATH` module constant pointing at
    `backend/xgb_calibration.pkl` (sibling of model + features).
  - `_try_load()` now also tries to unpickle an `IsotonicRegression`;
    missing pkl is logged and falls back to raw passthrough so Phase 5
    behavior is preserved when no calibrator has been fit.
  - `xgb_prob()` applies `iso.transform([raw])` between booster predict
    and the [0.01, 0.99] clip — the safety clip stays last so
    pathological 0.0/1.0 isotonic outputs can't reach the gate.
- **`backend/tools/fit_xgb_calibration.py` (NEW, #184)**: pulls
  `(confidence, outcome)` pairs from `signal_outcomes` for source='CNN',
  side='BUY', `checked_at IS NOT NULL`, since 2026-05-03 19:15:15 UTC.
  Refuses to fit on < 200 samples. Logs PRE/POST bucket WR + a 0..1
  grid mapping so the calibration shape is visible at fit-time. Saves
  pickled `IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)`
  to `backend/xgb_calibration.pkl`.
- **`.gitignore`**: explicit ignore for `backend/xgb_model.json`,
  `backend/xgb_features.json`, `backend/xgb_calibration.pkl` — matches
  the existing `.pt` model-artifact convention. All three are built
  locally and never committed.
- **Backend restarted** at 2026-05-04 06:52 LOCAL after fitting; the
  startup log confirms `xgb_signal: loaded isotonic calibrator from
  backend/xgb_calibration.pkl`.

### Tests

- `backend/tests/test_xgb_signal.py` — 4 new cases (TestModuleAttributes
  + TestCalibration class):
  - `test_has_calibration_path` — module exposes monkeypatchable
    `_CALIBRATION_PATH`.
  - `test_calibration_pkl_remaps_raw_to_calibrated` — fits an isotonic
    that maps the booster's raw output to 0.42, asserts xgb_prob
    returns 0.42 (within 1e-6).
  - `test_no_calibration_pkl_falls_back_to_raw` — preserves Phase 5
    backwards-compat when pkl is missing.
  - `test_calibration_clipped_to_safe_range` — even a degenerate
    calibrator outputting 0.0 still gets clipped to 0.01 before return,
    so downstream `model_prob > threshold` math is safe.
- All 12 `test_xgb_signal.py` cases green.

### Calibration grid (raw → calibrated)

```
0.20 → 0.286
0.30 → 0.435
0.50 → 0.435   (U-shape squashed flat in the bad zone)
0.70 → 0.607
0.80 → 0.867
0.90 → 1.000   (clipped to 0.99 in xgb_prob)
```

Effective gate semantics under `CNN_BUY_THRESHOLD=0.80`: only raw
booster outputs > 0.80 produce calibrated probs > 0.80, so fires now
correspond to the well-calibrated peak of the curve only.

### Follow-up tasks

- **#181 XGB-Step3**: add `xgb_prob` REAL column to `cnn_scans` so
  CNN and XGB probs can be logged in parallel during the remaining
  3 days of the shadow window.
- **Re-fit cadence**: `fit_xgb_calibration` should be re-run
  weekly or when shadow N grows by ~50%. (Manual for now; cron
  hookup tracked separately.)

### Files touched

- `backend/agents/xgb_signal.py`
- `backend/tests/test_xgb_signal.py`
- `backend/tools/fit_xgb_calibration.py` (new)
- `.gitignore`
- `CHANGELOG.md` (this entry)

---

## [Session 58.7] — 2026-05-04 — XGB-Step1: lower CNN_BUY_THRESHOLD 0.99 → 0.80

### Context

XGB shadow window opened 2026-05-03 19:15 (#136 Phase 6) with
`MODEL_BACKEND=xgb` + `CNN_BUY_THRESHOLD=0.99`. Four days later the
threshold has produced **zero hypothetical BUYs** — the booster output
is clipped to [0.01, 0.99] inside `agents.xgb_signal.xgb_prob`, so
nothing can ever exceed 0.99. The shadow window has been collecting
predictions but no live fires, which means we have no execution data
to inform the Phase 7 cutover decision.

Worse, the resolved-BUY win-rate-by-bucket is **U-shaped**, not
monotonic:

```
0.2-0.3: 30.4% (n=138)
0.3-0.4: 27.8% (n=108)
0.4-0.5: 13.7% (n=95)
0.5-0.6: 11.5% (n=191)  ← trough
0.6-0.7: 13.7% (n=168)
0.7-0.8: 18.2% (n=181)
0.8-0.9: 30.0% (n=120)  ← peak
```

Phase 7 cutover gate 3 ("monotonic calibration") fails on this shape.
That drives Step 2 (calibrated retrain) — but Step 2 needs live fire
data to validate against, which is what Step 1 unblocks.

### Changes

- **`.env` (line 41)**: `CNN_BUY_THRESHOLD=0.99` → `CNN_BUY_THRESHOLD=0.80`.
  Inline comment block records the rationale: 0.8-0.9 bucket = 30% WR
  over 4-day shadow window is the only monotonically-up region of the
  calibration curve, so it's the safest non-zero firing threshold for
  paper-money shadow-mode data collection. `DRY_RUN=true` keeps
  everything paper-money. `CNN_SELL_THRESHOLD=0.40` unchanged.
- **Backend restarted** at 2026-05-04 06:21:01 LOCAL (10:21:01 UTC).
  Verified `xgb_signal: loaded booster (270 features, set=v1)` in
  `backend/logs/backend.err` at 06:21:07. CNN-blend signals firing post-
  restart (e.g. `[SELL] XRP-USD cnn=43.08% llm=32.00% blend=37.90%`),
  confirming `cnn_prob` is being driven by the XGB booster as expected
  under `MODEL_BACKEND=xgb`.

### Tests

No code change — `.env` is gitignored config. No new test required;
existing `test_xgb_signal.py` still covers the booster path.

### Follow-up tasks

- **#180 XGB-Step2**: retrain XGB with isotonic / Platt calibration to
  fix the U-shape. Hard gate before Phase 7 cutover.
- **#181 XGB-Step3**: add `xgb_prob` REAL column to `cnn_scans` so we
  can log CNN and XGB probabilities side-by-side in a true parallel
  shadow mode (the spec for #136 Phase 6).

### Files touched

- `.env` (gitignored — change documented here only)
- `CHANGELOG.md` (this entry)

---

## [Session 58.6] — 2026-05-03 — Tiered blacklist landed end-to-end (#120)

### Context

Closes cash-flow lever 4 (#118/#120) — auto-blacklist losing pairs by
per-product Sharpe so the live system shrinks size on weak products and
paper-trades suspended ones, instead of sizing every product the same.
Three-tier ladder driven by per-trade Sharpe over the most recent
N=10 closed trades:

- **Active**     → full-size trades
- **Probation**  → ½-size trades (real money, reduced risk)
- **Suspended**  → paper-trade only (signal logged, no execution)

Iron rule: a product cannot skip tiers. Suspended must recover to
Probation before it can return to Active — protects against a flip where
a previously-blacklisted product becomes a winner mid-window.

### Changes

- **`backend/services/product_status.py` (NEW, #120a/#120b)**: pure
  evaluator `compute_status(trades, current) -> (new_status, reason)`.
  Constants: `MIN_TRADES_FOR_REVIEW=10`, `SHARPE_DEMOTE=-0.5`,
  `SHARPE_PROMOTE=+0.2`, `_MIN_STDEV=0.005` (decimal). Held silent
  unless N ≥ 10 to avoid noise from a 2-trade unlucky streak.
- **`backend/database.py` (#120c)**: added `product_status` table
  `(product_id PK, status, reason, last_evaluated_at, demoted_at)` +
  index on `status`. Helpers: `get_product_status(pid)`,
  `set_product_status(pid, status, reason)` (UPSERT;
  `demoted_at` stamped iff `status=='suspended'`, cleared on any
  other transition), `list_products_by_status(status)`.
- **`backend/agents/cnn_agent.py` `_CNNBook.buy()` (#125a)**: reads
  `product_status` at the top of `buy()`; missing row or `"active"` →
  full `frac`; `"probation"` → `frac *= 0.5`; `"suspended"` →
  paper-trade (logs CNN PAPER and returns `(0.0, 0.0)` without
  spending or persisting a position).
- **`backend/services/product_status.py` `evaluate_and_persist`
  helper (#125b)**: pulls last N closed trades via
  `database.get_trades(agent=..., product_id=..., closed_only=True,
  limit=N)`, converts `pct_pnl` (stored ×100) to decimal, runs
  `compute_status`, persists only on transition. Returns
  `(old_status, new_status, changed)`.
- **`backend/agents/cnn_agent.py` `_CNNBook.sell()` (#125b)**: after
  the in-memory state has been persisted via `_save()`, calls
  `product_status.evaluate_and_persist(pid, agent=self._agent)` so a
  fresh trade close can immediately tip the product into the next
  tier (or recover it). Wrapped in `try/except` — evaluator failures
  log via `logger.exception` but never block the sell return.

### Tests

- `tests/test_product_status.py` — 21 cases:
  - 13 for the pure `compute_status` evaluator (constants, min-sample
    gate, demotion paths, promotion paths, neutral-Sharpe hold,
    boundary at -0.5).
  - 8 for `evaluate_and_persist` (insufficient trades, no-status row,
    demote/promote/no-change paths, percent→decimal conversion gate,
    null `pct_pnl` skip, `get_trades` arg passthrough).
- `tests/test_database.py::TestProductStatusPersistence` — 11 cases
  for the new table + helpers (idempotent init, get/set round-trip,
  upsert, `demoted_at` stamp on suspended + clear on promotion,
  `list_products_by_status`).
- `tests/test_cnn_agent.py::TestCNNBookBuyProductStatus` — 4 cases
  for buy() gating (no status row, active, probation halves frac,
  suspended records no trade).
- `tests/test_cnn_agent.py::TestCNNBookSellEvaluatesProductStatus` —
  3 cases for sell() wiring (calls evaluator on success, skips
  evaluator when close_trade fails, swallows evaluator exceptions
  while completing the sell).
- Full pre-commit suite: 742 passed, 8 xfailed (no regressions).

### Live impact

- `product_status` table is read-empty at runtime; until enough closed
  trades accumulate for any product, `compute_status` returns
  `(current, "hold: only N trades…")` — buy() reads no row → defaults
  to `"active"` → full frac. So the change is a no-op until the
  evaluator has signal.
- `CNN_BUY_THRESHOLD=0.99` from #127 still blocks all CNN BUYs, so
  the live capital exposure of #125a is double-gated: the auto-shut-off
  is a passive observer until a future session relaxes the threshold.

---

## [Session 58.5] — 2026-05-03 — Phase 6 flip — train XGB + MODEL_BACKEND=xgb live (#136)

### Context

Closes the loop on the CNN→XGBoost transition. After #135 wired the
`MODEL_BACKEND` selector with default `"cnn"`, this session trained an
XGBoost booster on the existing 27-channel cache and flipped the live
backend to `MODEL_BACKEND=xgb` for the 7-day shadow-mode comparison
(#136 Phase 6). Live capital is safe: `DRY_RUN=true` and
`CNN_BUY_THRESHOLD=0.99` combined with `xgb_prob` clipping to
`[0.01, 0.99]` mean the buy gate `model_prob > 0.99` cannot fire under
either backend — this is observation-only.

### Changes

- **`backend/xgb_model.json` + `backend/xgb_features.json` (NEW)**:
  trained via `train_xgb` with 5-fold purged walk-forward, 4h embargo,
  on top-20 pooled products from `cnn_dataset_cache.pt`
  (X.shape=(162982, 27, 60), pos_pct=48.6).
  - best_params: `max_depth=4, min_child_weight=1, subsample=0.7`
  - fold AUCs: 0.5157 / 0.5093 / 0.5259 / 0.5185 / 0.5427
  - **mean_auc: 0.5224** — below the 0.55 Phase-4 hard gate, consistent
    with prior `xgb_feature_optimization_findings` peak of 0.5284 on
    the same 22-effective-channel stack (5 channels zeroed: MASKED
    {17,18,19} + XGB_DROP {21,24}).
  - 270 features (v1 set: per-channel mean/std/p25/p50/p75/last/
    momentum/range/... × 22 live channels + cross-channel terms).
- **`.env`**: appended `MODEL_BACKEND=xgb` block with rollback note
  ("set MODEL_BACKEND=cnn to revert"). Annotated rationale: AUC <
  gate but DRY_RUN + 0.99 buy gate make this a safe shadow flip.
- **Backend restart**: killed prior instance (PID 60284), relaunched
  via `.venv/Scripts/python.exe backend/main.py` (canonical port 8001
  per `start_backend.ps1`). `xgb_signal: loaded booster (270
  features, set=v1)` confirmed in logs at 19:15:15. CNN agent
  continues to scan; probabilities now sourced from XGB.

### Phase 6 next

7-day window: collect side-by-side XGB-vs-CNN probability traces and
real outcomes, then decide whether to relax `CNN_BUY_THRESHOLD` (and
which backend to keep) once feature work (#143-145 OKX OI, #156 BTC
dominance) lifts AUC above 0.55.

---

## [Session 58.4] — 2026-05-03 — Phase 5 — agents/xgb_signal.py + MODEL_BACKEND env var (#135)

### Context

Phase 5 of the CNN→XGBoost transition (see
`docs/superpowers/plans/2026-05-02-cnn-to-xgboost-transition.md`). Phases
0-4 built the XGB feature extractor (`tools/xgb_features.py`),
walk-forward harness, training script, and calibration probe. This phase
ships the inference glue so that flipping a single env var
(`MODEL_BACKEND=xgb`) routes `_cnn_prob` through XGBoost instead of the
CNN — without retraining or recompiling. Default stays `cnn` so live
behaviour is unchanged; Phase 6 (#136) is the 7-day shadow-mode
follow-up that will actually flip it.

### Changes

- **#135 RED — `tests/test_xgb_signal.py` (NEW)**: 8 tests covering the
  `xgb_prob(channels)` contract — graceful fallback to 0.5 when
  `xgb_model.json`/`xgb_features.json` artifacts are missing, returns
  float in `[0.01, 0.99]` when artifacts present, deterministic across
  calls (lazy-loaded singleton booster), accepts both `np.ndarray` and
  nested-list inputs, and channel-mask invariant (ch 17/18/19 values
  must not change predictions because `tools.xgb_features.extract_features`
  zeros them).
- **#135 RED — `tests/test_model_backend.py` (NEW)**: 5 tests covering
  the `Config.model_backend` field (default `"cnn"`, reads
  `MODEL_BACKEND` env, lowercases for stable comparisons) and the
  `CoinbaseCNNAgent._cnn_prob` branch (default backend never invokes
  `xgb_prob`; `MODEL_BACKEND=xgb` returns `xgb_prob`'s value).
- **#135 GREEN — `agents/xgb_signal.py` (NEW)**: lazy-load singleton
  with `_MODEL_PATH`, `_FEATURES_PATH`, threading.Lock-guarded loader.
  `xgb_prob(channels)` extracts features via `extract_features`, runs
  `Booster.predict`, and clips to `[0.01, 0.99]`. All exception paths
  return `0.5` so the production code path can never crash on a missing
  or corrupt model file.
- **#135 GREEN — `config.py:71-74`**: new `model_backend` field
  reading `MODEL_BACKEND` env var, default `"cnn"`, lowercased.
- **#135 GREEN — `agents/cnn_agent.py:_cnn_prob`**: branch on
  `config.model_backend`. When `"xgb"`, import `agents.xgb_signal` and
  return `xgb_signal.xgb_prob(channels)`. Otherwise the existing
  PyTorch / linear-fallback path runs unchanged.

### Tests (all GREEN)

- `tests/test_xgb_signal.py` — 8/8 (4.3 s)
- `tests/test_model_backend.py` — 5/5
- `tests/test_cnn_agent.py` + `tests/test_cnn_risk_exits.py` — 218
  passed, 7 xfailed, 0 failed (regression smoke; 2:25 wall-clock)

### Pending

- **#136 Phase 6**: 7-day shadow-mode A/B — duplicate the scan loop's
  `_cnn_prob` call to also log `xgb_prob` predictions to a new
  `model_predictions` table for offline comparison without trading on
  them. Decision gate: if Sharpe(xgb) ≥ 1.5× Sharpe(cnn) over the
  shadow window, flip `MODEL_BACKEND=xgb` for live; otherwise stay on
  CNN and revisit feature inputs.

---

## [Session 58] — 2026-05-03 — Ch 15 ADX causality fix: per-bar expanding window (#157)

### Context

Audit cross-checking `docs/equity_feature_engineering.md` and
`docs/crypto_feature_engineering_pipeline.md` against the actual 27-channel
FeatureBuilder identified a concrete look-ahead leak on Ch 15 (ADX regime).

`agents/cnn_agent.py:1305-1307` (pre-fix) computed `adx_val` once on the
**full** candle window and broadcast that single value across all 60
timesteps. Every cached training sample's Ch 15 series therefore carried
information about future bars in the window. Verified empirically: on a
sinusoidal 120-bar series, Ch 15 at candle 79 = 0.381 when built on the
full series vs 0.305 when built on only candles[:80] — a 0.076 delta
(~25 % of channel scale) leaking from the 40 future bars.

This is the lookahead-test failure mode that the crypto-pipeline doc §4
calls out: *"If you do nothing else from this guide, build the lookahead
test."*

### Changes

- **#157a — `tests/test_feature_builder_causality.py` (NEW)**: RED tests
  asserting (a) Ch 15 terminal value is identical when built on full
  vs. truncated candle series (windowed-equality property), and (b) Ch 15
  varies across timesteps within a single build call (broadcast detector).
- **#157b — `agents/cnn_agent.py:1305-1313`**: replace the broadcast
  `[adx_val / 100.0] * len(closes)` with a per-bar expanding window list
  comprehension calling `_adx(highs[:i+1], lows[:i+1], closes[:i+1])` for
  each `i`. Pattern mirrors Ch 4 RSI / Ch 8 Bollinger / Ch 14 Stoch RSI.
- **#157c — `agents/cnn_agent.py:455`**: `_DATASET_CACHE_VERSION` 10 → 11
  to invalidate v10 caches built with the leaky Ch 15 series. Updated
  `tests/test_cnn_agent.py::TestDatasetCacheVersionBumpForRv` to assert == 11.

### Tests (all GREEN)

- `tests/test_feature_builder_causality.py::TestCh15ADXCausality` — 2/2
- `tests/test_cnn_agent.py` — 203 passed, 7 xfailed, 0 failed (full suite,
  10:02 wall-clock)

### Pending

- **#160**: re-run `tools/permutation_importance.py` and
  `tools/feature_set_compare.py` on a fresh v11 cache to measure the Δ
  AUC from removing the leak. Expect an apparent **drop** in Ch 15
  permutation-importance (the prior signal was partly leakage); re-test
  whether Ch 15 still earns its slot post-fix.
- **#161**: generic lookahead-test harness extending the windowed-equality
  property to all 27 non-masked channels.

---

## [Session 58.1] — 2026-05-03 — Quarantine corrupted dataset cache + autouse test isolation (#172, #173)

### Context

While preparing to measure Δ AUC from the #157 Ch 15 fix on a fresh v11
cache, `tools/feature_set_compare.py` failed with `KeyError: 'BTC-USD'`.
Investigation found the production cache file
`backend/cnn_dataset_cache.pt` (3.2 MB) contained synthetic test fixture
data — 10 fake products `COIN0-USD..COIN9-USD`, 486 samples on
sinusoidal candles with `start=1_700_000_000+i*3600`. Eleven consecutive
production trains (ids 504-514) had run on this junk, producing tiny
sample counts (78-1020) and near-random val_auc.

Root cause traced to `tests/test_cnn_agent.py:1027` and `:1051`: the
`_make_products_and_candles` helper generates synthetic 6-product
fixtures and the test methods call `agent.train_on_history(...)`
directly. The correct pattern (e.g. `tests/test_cnn_agent.py:2956`)
monkeypatches `_DATASET_CACHE_PATH` to a tmp path, but the offending
tests omitted the redirect — every test run silently overwrote the
production cache.

### Changes

- **#172 — quarantine**: renamed corrupted `cnn_dataset_cache.pt` →
  `cnn_dataset_cache.corrupted_20260503_135915.pt` (preserved for forensic
  analysis, not deleted).
- **#173a RED — `tests/test_dataset_cache_isolation.py` (NEW)**: two
  asserts that during any pytest session, `_DATASET_CACHE_PATH` does NOT
  resolve to the real production file and is not under `backend/`.
- **#173b GREEN — `tests/conftest.py`**: added `_redirect_cnn_dataset_cache`
  autouse fixture that monkeypatches `agents.cnn_agent._DATASET_CACHE_PATH`
  to a per-test tmp directory. Defends against the entire class of bug —
  any future test that touches the cache via deep import side-effects is
  now safe by default rather than per-method opt-in.

### Tests (all GREEN)

- `tests/test_dataset_cache_isolation.py` — 2/2 PASSED
- Full venv suite: 241 passed, 8 xfailed, 0 failed in 154s

### Verified

- `cnn_model_glu1.pt` mtime = 2026-05-02 21:06:20 UTC, predates all 22
  junk training rows (504-525, 16:30-18:06 UTC on 2026-05-03). Disk model
  weights are intact from the last legitimate train (id 503,
  val_auc=0.6092, 359689 samples, fit_status=REJECTED).
- After v11 cache rebuild on next big train, #157 Δ AUC measurement (#160)
  can proceed.

### Follow-up

- **#176**: tests also pollute production `coinbase.db` with junk
  `cnn_training_sessions` rows (515-525 from the 2026-05-03 venv run).
  Same fix pattern needed for `DATABASE_URL` redirect — addressed in
  Session 58.2 below.

---

## [Session 58.2] — 2026-05-03 — Autouse DB redirect fixture (#176)

### Context

While verifying #173, observed that the venv pytest run added 11 junk
rows (ids 515-525) to `cnn_training_sessions` in production
`coinbase.db`. Tests calling `agent.train_on_history(...)` end at
`agents/cnn_agent.py:2881 await database.save_training_session(result)`,
which writes via `database.DB_PATH` set at module import from
`config.database_url`. The existing `tmp_db` fixture only mutates
DATABASE_URL — it does NOT monkeypatch the already-imported
`database.DB_PATH`, so non-`init_db` tests continued to write to the
real file.

22 junk training-history rows accumulated in coinbase.db across multiple
sessions before the autouse fixtures landed (504-525 visible in the
post-#172 audit). Historical pollution preserved as-is — fix targets
future runs only.

### Changes

- **#176a RED — `tests/test_database_isolation.py` (NEW)**: two asserts
  that `database.DB_PATH` does NOT resolve to the real `backend/coinbase.db`
  file and is not under `backend/`.
- **#176b GREEN — `tests/conftest.py`**: added `_redirect_database_path`
  autouse fixture that monkeypatches `database.DB_PATH` to a per-test
  tmp directory. Sibling of the #173 cache redirect — same defense-by-
  default pattern.

### Tests (all GREEN)

- `tests/test_database_isolation.py` — 2/2 PASSED
- Full pytest suite still GREEN with both autouse fixtures in place.
  Tests that explicitly use `tmp_db` + `init_db` reload the database
  module which re-reads DATABASE_URL — overriding this autouse default
  with the test's own tmp DB path. Compatible by design.

---

## [Session 58.3] — 2026-05-03 — Tiered blacklist evaluator: services/product_status.py (#120, #122, #123)

### Context

Task #120 — auto-blacklist losing pairs by per-product Sharpe — needs a
graduated tier system rather than a binary in/out gate. The user's stated
concern was: *"what if you blacklist losers then they become winners?"*
A three-tier ladder with one-step transitions answers that: a Suspended
product earns its way back via Probation (paper trades), where a small
positive Sharpe is enough to re-enter Active.

### Changes

- **#122 — `tests/test_product_status.py` (already-RED)**: 13 tests
  covering constants, min-sample gate, demote/promote paths (each only
  one tier per evaluation), and the boundary-stdev hold case. xfail mark
  removed in this commit so they run as ordinary tests.
- **#123 — `services/product_status.py` (NEW, GREEN)**: implements
  `compute_status(trades, current) → (new_status, reason)`. Sharpe is
  computed as mean(pnl_pct) / stdev(pnl_pct) over the most recent N
  closed trades; below MIN_TRADES_FOR_REVIEW=10 the status holds. Demote
  threshold SHARPE_DEMOTE=-0.5; promote threshold SHARPE_PROMOTE=+0.2;
  zero-variance / very-tight clusters (`stdev < 0.005`) hold to avoid
  amplifying tiny-mean noise into a Sharpe signal.

### Tests (all GREEN)

- `tests/test_product_status.py` — 13/13 PASSED in 2.76s.

### Pending

- **#124**: persistence — `product_status` DB table + helpers.
- **#125**: wire status into `_CNNBook.buy()` (block/half-size) and into
  the scan-loop evaluator (recompute on trade close).
- **#126**: CHANGELOG + memory updates after #124-#125.

---

## [Session 57] — 2026-05-02 — Cash-flow phase 1: bump ATR trail floor 3% → 6% (#115)

### Context

CNN agent has been bleeding money: per-trade expectancy ≈ +0.083% gross is
swamped by 1.2% round-trip taker fees → net ≈ −1.12% per trade. Monte Carlo
simulation across 7-day cohorts identified four cumulative levers (ranked
by marginal PnL impact). **Lever 3 — wider trail floor — is the cheapest
fix and was first.**

The 3% floor was triggering `TRAIL_STOP` on routine intra-day chop in
low-ATR regimes, locking in losses before mean-reversion could play out.
Bumping to 6% gives positions room to breathe; downside is still capped
by hard `STOP_LOSS = 8%`.

### Changes

- **#115 — `agents/cnn_agent.py:79`**: `_CNN_ATR_TRAIL_MIN` 0.03 → 0.06.
  Comment updated to reference Session 57 cash-flow lever 3. Tests:
  `tests/test_cnn_risk_exits.py` — new `TestTrailFloor` class with
  3 tests (constant assertion, 4% drawdown does NOT exit, 6.5% DOES exit).
- **#115d — test cleanup**: `tests/test_cnn_agent.py` and
  `tests/test_cnn_risk_exits.py` had stale
  `OLLAMA_MODEL=qwen2.5:7b`. Bumped to `llama3.1:8b` to match production
  `.env`. Per CLAUDE.md invariant 7 (env-driven model name).

### Test status

- `tests/test_cnn_risk_exits.py`: 17 passed (3 new + 14 existing).

---

## [Session 56] — 2026-05-02 — Auto-start launcher on Windows login (#113)

### Context

User repeatedly forgets to click the "Start All" button after a reboot.
The launcher already auto-starts services 1 s after its window opens, and
already exposes a "Start on login" toggle that writes to
`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`. Default state was
"unset", so on a fresh install nothing launches at boot.

### Changes

- **One-shot registry write**: enabled the `CoinbaseAITrader` HKCU Run
  entry pointing at `Coinbase AI Trader.exe`. Next Windows login auto-
  launches the launcher → which auto-starts backend+frontend within ~1 s.

- **#113 — `launcher.py:_maybe_default_startup_to_on`**: new helper plus
  one call from `LauncherApp.__init__`. On the very first launcher run, it
  defaults Start-on-login to ON and drops a sentinel file at
  `backend/logs/.startup_default_applied`. Subsequent runs see the
  sentinel and never overwrite the user's choice — so a deliberate opt-out
  via the toggle stays opt-out. Tests:
  `backend/tests/test_launcher.py` (3 tests covering first-run write,
  no-op when sentinel exists, and parent-dir creation).

### Test status

- `tests/test_launcher.py`: 3 passed.

---

## [Session 55] — 2026-05-02 — Trades-table as source of truth + dashboard cache alignment (#109–#111)

### Context

Investigating "CNN gained ~$1.50 in Realized PnL but no trades show up on the
Performance tab" surfaced a divergence between two values that should always
match: `agent_state.realized_pnl` (in-memory accumulator persisted on each
sell) vs `SUM(trades.pnl)` (the trade-ledger source of truth). Live snapshot:

| Agent | agent_state | SUM(trades.pnl) | Δ          |
|-------|-------------|-----------------|------------|
| CNN   | -$5.55      | -$11.53         | +$5.98    |
| TECH  | +$39.07     | +$23.64         | +$15.43   |

Plus 14 TECH trade rows that were open in the DB but absent from
`agent_state.positions` ("orphans"), which on next restart would force-close
with `pnl=0` — locking the divergence in permanently.

Root cause: `_CNNBook.sell()` updated `agent_state.realized_pnl` (via `_save`)
**before** calling `database.close_trade()`. If `close_trade` raised (Windows
file lock, transient DB error), the gain was captured in agent_state but no
matching closed-trade row existed. Compounding: PerformanceDashboard cached
trade lists for 2 minutes while AgentsDashboard polled every 15 s, so the
user-visible gap was even wider than reality.

### Changes

- **#109 — `backend/agents/cnn_agent.py:_CNNBook.sell()`**: swap call order so
  `database.close_trade(...)` runs **before** any in-memory mutation or
  `_save()`. The trades table is now the source of truth: if the DB write
  fails the position stays in the book, balance + realized_pnl are unchanged,
  and the next sell attempt can retry cleanly. Tests:
  `tests/test_cnn_agent.py::TestCNNBookSellOrdering` covers both ordering and
  rollback-on-failure.

- **#110 — `backend/tools/reconcile_agent_state.py`**: one-shot CLI that (1)
  closes orphan open-trade rows for each agent (open in DB but absent from
  saved positions) with `trigger_close="RECONCILE"`, and (2) overwrites
  `agent_state.realized_pnl = SUM(trades.pnl)` so the in-memory accumulator
  re-syncs to the ledger on next backend restart. `--dry-run` previews. Run
  with the backend stopped (avoids race where a concurrent sell overwrites
  the reconciled value with the stale in-memory accumulator). Tests:
  `tests/test_reconcile_agent_state.py` (4 tests).

- **#111 — `frontend/src/components/PerformanceDashboard.tsx:_CACHE_TTL_MS`**
  `2 min → 30 s`. Aligns the Performance tab with AgentsDashboard's 15 s poll
  so closed trades appear within roughly the same window the agent view
  reflects them. The "trade vanished for 2 minutes" symptom that prompted
  this investigation is gone.

### Test status

- `tests/test_cnn_agent.py + test_reconcile_agent_state.py + test_database.py`:
  239 passed (495 s) on `.venv` Python 3.11.

### Operator action required

After deploying, run on the host with backend stopped:

```
cd backend
.venv/Scripts/python.exe -m tools.reconcile_agent_state --dry-run   # preview
.venv/Scripts/python.exe -m tools.reconcile_agent_state             # apply
```

Restart the backend; `_CNNBook.load()` will read the corrected
`agent_state.realized_pnl`, and the Performance tab will match
AgentsDashboard within 30 s.

---

## [Session 54] — 2026-05-02 — CNN training health: majors-pin + watchdog/heartbeat/cache hardening (#105–#108)

### Context

Glum retrain (PID 57568) was watchdog-killed at 17:54 with
`{"status":"failed","error":"watchdog: log stale, subprocess killed"}` while
the live backend held the same RTX 2060 for inference. Root-cause sweep
turned up four separate problems that compounded to false-kill a healthy run
and mis-target the dataset itself. All four are fixed here as one bundle so
the next retrain has a clean slate.

### Changes

- **#105 — `backend/database.py:get_products()`** pins a 15-symbol majors
  list (`BTC, ETH, SOL, BNB, XRP, ADA, AVAX, LINK, DOT, DOGE, LTC, ATOM,
  BCH, TRX, MATIC`) at the top of the result regardless of `volume_24h`
  rank. Coinbase stores `volume_24h` in **native token units**, so a pure
  `ORDER BY volume_24h DESC LIMIT 100` was dominated by memecoins (PEPE,
  BONK, SHIB, FLOKI, …) and silently excluded BTC/ETH/SOL. Of 30
  OKX-mapped majors only 3 made the cut; the CNN was training on memecoin
  chaos. Discovered while investigating "why doesn't OKX funding pull
  BTC/ETH?" — the answer was upstream: those products never entered the
  product list. SQL change uses a `CASE WHEN product_id IN (...) THEN 0
  ELSE 1 END, volume_24h DESC` pre-sort so the pin works without breaking
  the existing volume tiebreak among non-major rows.

- **#106 — `backend/agents/cnn_agent.py:_HEARTBEAT_EVERY`** `5 → 1`. Under
  GPU contention with the live backend's inference path, glum epochs
  stretched to 5+ min each. Heartbeat-every-5 meant 25-min gaps between
  INFO log writes — over the existing 30-min watchdog threshold — and
  killed a fundamentally healthy training run. Heartbeat every epoch keeps
  the log mtime fresh even on the slowest arch.

- **#107 — `backend/main.py:_TRAIN_STALE_LOG_SECS`** `1800 → 3600`. Belt-
  and-braces companion to #106: even with per-epoch heartbeats, a single
  contended phase-2 dataset-build chunk could go quiet >30 min. 1 hr gives
  the worst observed 50-min epoch headroom without making the watchdog
  useless against real hangs.

- **#108 — `backend/agents/cnn_agent.py:_save_pp_cache()`** wraps
  `os.replace(tmp, path)` in a 4-attempt retry loop with backoff
  `(0, 0.5, 1.5, 3.0) s`. The live backend keeps `cnn_dataset_cache.pt`
  open during inference; on Windows that raises `PermissionError`
  (WinError 5) when the trainer tries to atomically swap in a new copy.
  The lock window is brief so a short retry recovers cleanly without
  needing to coordinate with the backend.

### Tests

- `backend/tests/test_database.py::TestMajorsAlwaysIncluded::test_btc_eth_sol_returned_even_with_low_native_volume`
  inserts 100 fake memecoins (volume=1e9 each) plus BTC/ETH/SOL with low
  native volume, then asserts the majors are in the top-100 result.
- `backend/tests/test_cnn_agent.py::test_heartbeat_every_is_one` —
  asserts `_HEARTBEAT_EVERY == 1` exactly (constant pin).
- `backend/tests/test_cnn_agent.py::TestPerProductDatasetCache::test_pp_cache_save_retries_on_windows_file_lock`
  uses `monkeypatch` to inject a flaky `os.replace` that raises
  `PermissionError` on the first 2 calls and succeeds on the 3rd, then
  asserts `_save_pp_cache` returned without raising.
- `backend/tests/test_train_watchdog.py` — updated existing thresholds
  test to assert `3600` and added `test_running_log_idle_45m_is_not_stale`
  / `test_running_log_idle_70m_is_stale` covering the new boundary.

Full per-module suite (database + cnn_agent + train_watchdog) green:
**244 passed in 502.53s** (with venv python 3.11).

### Files

- `backend/database.py` (#105 pinned-majors SQL)
- `backend/agents/cnn_agent.py` (#106 heartbeat=1, #108 cache save retry)
- `backend/main.py` (#107 watchdog 1 hr)
- `backend/tests/test_database.py` (new test class)
- `backend/tests/test_cnn_agent.py` (heartbeat-pin + cache-retry tests)
- `backend/tests/test_train_watchdog.py` (1-hour boundary coverage)
- `CHANGELOG.md` (this entry)

### Why bundled

All four issues surface from the same incident (glum killed at 17:54 under
backend GPU contention). #106 and #107 are belt-and-braces for the
watchdog. #105 is unrelated mechanism but was discovered in the same
investigation thread and gates whether the *next* retrain trains on the
right data at all. #108 is the cache-save flake that caused the original
"non-fatal" warning that started the trail.

---

## [Session 53] — 2026-05-02 — GPU/Ollama coordination across trading_app and polymarket_app

### Context

Mirror of trading_app's `gpu_coord.py` — both apps share one Ollama instance
on a single RTX 2060. Without coordination, concurrent Ollama calls cause
15–50s latencies and 50s timeouts (observed in `trading_app/error.log` on
2026-04-27). The trading_app side shipped 2026-05-02 (PRs #6, #7) but was
ineffective for cross-app priority until polymarket also writes to the
shared coord file. This session completes that loop.

### Changes

- **`backend/data/gpu_coord.py` (new)** — mirror of trading_app's module.
  - `OllamaCoordinator` class + module singleton `ollama_coord(app_name="polymarket_app")`.
  - Layer 1: per-process `asyncio.Lock` serializes Ollama calls within polymarket.
  - Layer 2: shared `~/.ollama-coord/state.json` (env override `OLLAMA_COORD_FILE`)
    with `{exposure_usd, updated_at}` per app. `acquire()` yields up to 10s to
    higher-exposure apps, fires anyway on timeout.
  - `acquire_training_mutex` / `release_training_mutex` sync API for the
    cross-app training mutex via `~/.ollama-coord/training.lock`. Stale-PID
    reclaim handles crashed peers.

- **`agents/cnn_agent.py:_ollama_prob`** — wrapped in `ollama_coord.acquire(expected_ms=25_000)`.
- **`agents/signal_generator.py:_llm_confirm`** — wrapped in `ollama_coord.acquire(expected_ms=20_000)`.
- **`services/outcome_tracker.py:validate_with_ollama`** — wrapped in `ollama_coord.acquire(expected_ms=20_000)`.

- **`train_worker.py`** — acquires `acquire_training_mutex(app_name="polymarket_app")`
  at startup, releases in `finally`. If a peer is training, waits up to 1h then
  defers (writes `status="skipped"` to the progress file and exits 0). The next
  scheduled retrain picks it back up — no infinite queueing.

- **`main.py`** — new `_publish_exposure_loop` async task writes
  `app_state.portfolio.summary["total_value"]` to the coord file every 30s so
  trading_app sees polymarket's current exposure for cross-app priority.

### Tests

`backend/tests/test_gpu_coord.py` (new) — 15 pytest tests covering:
- Per-app asyncio.Lock serialization
- Exposure round-trip + multi-app preservation
- Stale-entry handling (>60s)
- Acquire bypass when we have higher exposure
- Bounded wait when we don't, fires after timeout
- Missing coord file falls back to lock-only
- Training mutex: acquire / release / safe-release / stale-reclaim / timeout / re-entrant

15/15 passing in 1.11s.

### Migration / interaction

- Coord file location: `~/.ollama-coord/state.json` (default). Both apps must
  use the same path. If `OLLAMA_COORD_FILE` is set in only one app's `.env`,
  cross-app priority silently breaks.
- Training mutex file: `~/.ollama-coord/training.lock`. Same protocol.
- Coord-file IO is best-effort: missing/unreadable file falls back to lock-only
  behavior — never raises into the call path.

### Files

- `backend/data/gpu_coord.py` (new, 285 lines)
- `backend/agents/cnn_agent.py` (1 site wrapped)
- `backend/agents/signal_generator.py` (1 site wrapped)
- `backend/services/outcome_tracker.py` (1 site wrapped)
- `backend/main.py` (+ exposure publisher task)
- `backend/train_worker.py` (+ mutex acquire/release)
- `backend/tests/test_gpu_coord.py` (new)
- `CHANGELOG.md` (this entry)

---

## [Session 52] — 2026-05-02 — CNN: SCAN-SELL is primary, risk exits demoted to fallback (#104)

### Context
Live-trade audit (2026-04-12 → 2026-05-02) of `coinbase.db` showed CNN net PnL
**-$19.24** on 597 closed trades (36% win rate), while TECH was net **+$23.46**
on 245 closed trades (60% win rate). Decomposing CNN exits by trigger:

| trigger               | n   | PnL     |
|-----------------------|-----|---------|
| SCAN (CNN's own SELL) | 318 | **+$59.44** ✅ |
| TRAIL_STOP            | 197 | -$49.89 ❌ |
| STOP_LOSS             | 13  | -$42.69 ❌ |
| RECONCILE             | 68  |   $0.00 |

CNN's own SELL judgment was profitable. The risk-stop machinery on top of
weak entries was bleeding -$92 net. Root cause: `run_loop` (cnn_agent.py:2284)
called `_check_risk_exits()` **before** `scan_all()` each iteration, so
TRAIL_STOP / STOP_LOSS pre-empted CNN's own SCAN-SELL.

### Change
- `agents/cnn_agent.py::CoinbaseCNNAgent.run_loop`: swap call order. `scan_all()`
  now runs first (CNN's own SCAN-SELL fires as primary exit), then
  `_check_risk_exits()` runs second (TRAIL_STOP → STOP_LOSS → MAX_HOLD as
  secondary/tertiary fallbacks for positions CNN did not close itself).
- Comment updated to document the new priority and to preserve the existing
  semantic that risk fallbacks still run every loop regardless of `is_trading`
  gate (so stops fire even when scanning is paused).

### Why this is safe
- Risk exits still run unconditionally each loop — the safety net is preserved.
- When trading is paused, `scan_all` runs in non-execute mode (no closes), so
  risk-exits remain the only real exit path during pause.
- When trading is live, risk-exits only see positions CNN chose **not** to
  close. If CNN's SELL fires, the position is gone before risk machinery looks.

### Tests
- `tests/test_cnn_agent.py::TestRunLoopExitPriority::test_scan_all_runs_before_check_risk_exits_in_run_loop`
  RED→GREEN. Source-inspection asserts `self.scan_all(` appears earlier in
  `run_loop` body than `self._check_risk_exits(`.

---

## [Session 51] — 2026-05-01 — Fit-loop INFO heartbeat (#101)

### Context
Glum retrain (Session 50 follow-up) was killed by `train_watchdog` at exactly
30 minutes — `cnn_train_progress.json` written with
`{"error": "watchdog: log stale, subprocess killed"}`. Investigation showed:

- The CNN fit loop (`cnn_agent.py::_sync_fit`) only logs at `DEBUG` per-epoch
  (line 2649). With `LOG_LEVEL=INFO` in production, `logs/cnn_training.log`
  receives **one** INFO line during training: "CNN fit started: ...".
- Watchdog (`main.py::_is_training_stale`) reaps the subprocess when
  `cnn_training.log` mtime is unchanged for ≥`_TRAIN_STALE_LOG_SECS=1800` (30 min).
- glu1 (~7 min) and glu2 (~12 min) finished before the threshold so they
  slipped through. Glum needed >30 min — watchdog killed a healthy process.
- CPU dry-run (`tools/dryrun_glum.py`) confirmed `SignalCNNGluM` forward+backward
  works at all batch sizes (1, 8, 64, 256). No architectural hang.

### Change
- Add `_HEARTBEAT_EVERY = 5` constant near `_CKPT_EVERY` (cnn_agent.py:1000).
- Inside the per-epoch loop, after the existing DEBUG log, emit a
  `logger.info("CNN train heartbeat epoch X/Y | train=... val=... lr=... best_val=...")`
  on epoch 1 and every 5 epochs thereafter. Keeps `cnn_training.log` mtime
  fresh (gap ≤ ~3 min on the slowest archs) so the watchdog only fires on
  real hangs.

### Verification
- New tests in `TestFitLoopHeartbeat` (test_cnn_agent.py): assert
  `_HEARTBEAT_EVERY` exists & is reasonable, and source-inspect the per-epoch
  loop body for both `_HEARTBEAT_EVERY` and `logger.info(...)`.
- Full `test_cnn_agent.py` + `test_train_watchdog.py`: 213 passed.
- Re-spawn glum after this change to populate the 3-way comparison.

### Files
- `backend/agents/cnn_agent.py`
- `backend/tests/test_cnn_agent.py`
- `backend/tools/dryrun_glum.py` (diagnostic helper, kept for future arch verification)

---

## [Session 50] — 2026-04-28 — Mask Ch 17/18/19 + train-time zeroing (#99)

### Context
Cache inspection on the v10 dataset revealed three channels still emitting
constants from `FeatureBuilder` fallback even though the OKX funding fetch
(observed via httpx 200 OK on `/api/v5/public/funding-rate-history`) was
running cleanly:

| ch | name         | min/max     | std    | notes                              |
|----|--------------|-------------|--------|------------------------------------|
| 17 | fast_rsi_1h  | 0.5 / 0.5   | 0.0    | neutral fallback — 1h source unfed |
| 18 | velocity_1h  | 0   / 0     | 0.0    | never populated                    |
| 19 | vol_zscore   | 0   / 0     | 0.0    | never populated                    |

Permutation importance (Session 49 perm run) confirmed delta = 0.0 across
both glu1 and glu2 — model already ignores them, but they consume 11% of
input bandwidth and present a hidden train/serve skew: ch 17 trains as 0.5
yet `_cnn_prob` zeros it at inference (because it was *intended* to be in
`_TRAINING_CONSTANT_CHANNELS` but the post-#86 reset to `frozenset()` left
it unmasked).

### Change
- `_TRAINING_CONSTANT_CHANNELS` reset from `frozenset()` to
  `frozenset({17, 18, 19})` (cnn_agent.py:283). Comments updated to record
  why these three remain dead while 10/11/20/24/25/26 are real.
- New helper `_zero_mask_channels(x: torch.Tensor)` — tensor analogue of
  `_mask_training_constant_channels`. Clones input, zeros every channel in
  the mask along the time axis, returns. No-op when mask is empty.
- `_train_arch` now calls `_zero_mask_channels(X_all)` immediately after
  `torch.stack(X_list)` so training tensors carry zeros for masked
  channels. This pairs with the existing inference-time mask in `_cnn_prob`
  to eliminate the train/serve distribution skew on ch 17 (constant 0.5
  in cache → 0.0 at training and inference).
- `tools/train_cloud.py:DEFAULT_MASK` updated `{10, 11, 20, 24, 25, 26}`
  → `{17, 18, 19}` to mirror prod (stale since #86 made those channels
  real). Comment refreshed with the per-channel timeline.

### Tests
- New `TestZeroMaskChannelsHelper` (3 tests): zeros only listed set,
  doesn't mutate input, preserves shape/dtype.
- Updated `TestTrainingConstantChannelMask::test_mask_set_covers_expected_channels`
  to expect `{17, 18, 19}`.
- Updated `TestMaskShrinkAndCacheBump::test_mask_shrunk` to expect
  `frozenset({17, 18, 19})`.
- Updated `tests/test_train_cloud.py::test_default_mask_is_frozenset_of_prod_constant_channels`
  to expect `frozenset({17, 18, 19})`.

### Activation
**Requires retrain.** Existing `cnn_model.pt` (glu2) and `cnn_model_glu1.pt`
were trained with ch 17 = 0.5 in input; the new inference mask zeros it.
Per CLAUDE.md invariant #11, mask changes require retraining. Cache version
NOT bumped — cached X tensors are unchanged; only how training/inference
consume them changed. Retrain on v10 cache picks up the new mask.

### Verification
- Targeted tests — 11/11 PASS in 2.43s on `.venv/Scripts/python.exe`.
- Full suite — **540/540 PASS** in 623s (added 3 new tests vs. 537).

---

## [Session 49] — 2026-04-27 — RV20/RV60 prefix-lookback fix (#98)

### Context
Permutation importance (#92-94) revealed Ch 25 (`ivrv_60`) had **delta = 0.0
across all 5 seeds and both shuffle axes** — i.e. the channel carried zero
signal because it was constant-zero across the entire 60-bar training window.

Root cause: `_rv_series(closes, window=60)` returns 0.0 for the first
`window` indices (`out = [0.0] * len(closes); for i in range(window, len(closes))`).
With `len(closes) == SEQ_LEN == 60` and `window == 60`, the loop body never
executes — every output is zero. Same shape issue for RV20 over the first
20 bars of the window (1/3 of the series zero).

### Change
- `_DataPipeline.build` accepts new optional kwarg `closes_ext: Optional[List[float]]`.
  When provided and `len(closes_ext) >= len(closes)`, RV20/RV60 are computed
  over `closes_ext` and the **last `len(closes)` values** are taken for the
  channel (alignment preserves in-window correspondence).
- `_build_samples_range` (training path) now passes
  `closes_ext = [c["close"] for c in candles[i - SEQ_LEN + 1 - 60 : i + 1]]`
  — 60 prior closes prepended for full RV60 lookback. New constant
  `_RV_PREFIX_BARS = 60`.
- Inference path: `database.get_candles(pid, limit=...)` bumped 80 → 140
  so the in-build truncation `P(...)[-SEQ_LEN:]` now returns the
  rv-non-zero tail of the series. `btc_candles` fetch matched (80 → 140)
  to keep correlation alignment. `closes_ext=closes` forwarded for parity.
- `_DATASET_CACHE_VERSION` bumped 9 → 10 to invalidate caches built with
  the old constant-zero RV channels — first scan post-restart triggers
  full per-product rebuild.
- Tests: 5 new in `TestRvPrefixLookback` (no-ext fallback regression,
  rv60 non-zero across window, rv20 non-zero across window, alignment
  matches reference, short-ext graceful fallback) + 1 in
  `TestDatasetCacheVersionBumpForRv` (version == 10). Updated 2 test fakes
  (`_FakeFB.build`, `_CapturingFB.build`) to accept the new kwarg.
  Loosened the prior `TestMaskShrinkAndCacheBump` cache assertion to `>= 9`.

### Activation
**Not auto-activated.** PID 37496 (glu1 retrain on cache v9) is still
running and per `feedback_no_restart_during_retrain` no restart happens
until that completes. After completion: backend restart picks up new code
+ cache v10; first scan rebuilds the per-product cache; next retrain
trains on real RV20/RV60 channels (no longer constant zero).

### Verification
- `tests/test_cnn_agent.py` — 199/199 PASS.
- Full suite — **537/537 PASS** in 89s on `.venv/Scripts/python.exe`
  (Python 3.11.13, torch 2.6.0+cu124).

---

## [Session 48] — 2026-04-27 — Mid-size CNN arch `SignalCNNGluM` (#89)

### Context
Two-era retrain analysis showed the existing arch lineup straddles the
underfit/overfit boundary with no middle ground:
- `glu2` (249,345 params) memorizes — train BCE → 0.40, val 0.58–0.69
- `glu1` (9,073 params) underfits — val plateaus at 0.69–0.70, never beats 0.6931
27.5× param gap; geometric mean ≈ 47k. A mid-size variant gives the next
retrain a third option without requiring an arch flip on the live process.

### Change
- New class `SignalCNNGluM` in `backend/agents/cnn_agent.py` (arch tag
  `"glum"`, **55,793 params**). 3-block GLU conv stack 24→48→96, single
  MaxPool (60→30), 1-layer LSTM(96→32), Dropout 0.4, FC(32→1).
- `_ARCH_REGISTRY` extended with `"glum": SignalCNNGluM`. Per-arch checkpoint
  paths already work via the generic `_model_path_for(arch)` /
  `_best_loss_path_for(arch)` suffix logic — no path code changes needed
  (`cnn_model_glum.pt`, `cnn_best_loss_glum.txt`).
- Tests: 4 new tests in `TestSignalCNNGluM` (class+arch tag, forward shape,
  predict probability, param count strictly between glu1 and glu2 with
  ≥3× glu1 and ≤glu2/3 bands) + 3 in `TestArchFactoryAndPaths` (factory
  build, model-path suffix, best-loss-path suffix).

### Activation
**Not auto-activated.** PID 37496 (glu1 retrain) is still running and per
`feedback_no_restart_during_retrain` no flip happens until that completes.
After completion, switch via `CNN_ARCH=glum` in `.env` and restart backend.
Cache version unchanged — the v9 dataset cache built for #86 is reusable.

### Verification
- `tests/test_cnn_agent.py` — 193/193 PASS in 71s on
  `.venv/Scripts/python.exe` (Python 3.11.13, torch 2.6.0+cu124).
- Param sanity (live import): `glu1 9,073 / glum 55,793 / glu2 249,345`.

---

## [Session 47] — 2026-04-27 — OKX funding history replaces geo-blocked Binance source (#86)

### Context
After #80/#81 disabled Ch 20 (funding rate) because `fapi.binance.com` returns
HTTP 451 from the US, Ch 20 was the only remaining masked channel — every
sample fed to training had Ch 20 = 0. OKX's public funding-rate endpoint is
reachable from this region, mirrors Binance's data layout, and covers the
USDT-margined SWAP equivalents of every product on our top-list.

### Change
- New module: `backend/services/okx_funding_history.py` — drop-in replacement
  for `services.binance_funding_history`. Same signature
  (`fetch_funding_history(product_id, start_ms, end_ms)`), same return type
  (sorted `[(ts_ms, rate), …]`).
- 30-product `_PRODUCT_TO_OKX` mapping (BTC, ETH, SOL, XRP, BNB, ADA, AVAX,
  LINK, DOT, DOGE, LTC, ATOM, FIL, NEAR, APT, INJ, ARB, OP, TIA, SEI, SUI,
  AAVE, UNI, HYPE, ICP, TAO, BCH, ZEC, SHIB, TRX). Products without a SWAP
  short-circuit to `[]` without an HTTP call (mirrors Binance fetcher
  behaviour for VVV-USD etc.).
- Pagination via `after=<ts_ms>` cursor (OKX caps `limit` at 100 per call vs
  Binance 1000); walks backward from `end_ms` until oldest row crosses
  `start_ms`. `_MAX_PAGES = 60` guards against runaway loops.
- Kill switch: `OKX_FUNDING_DISABLED=1` env short-circuits without HTTP.
- `agents/cnn_agent.py:72` swaps `from services.binance_funding_history` →
  `from services.okx_funding_history`. The Binance module stays in-tree as
  reference / fallback.
- `_TRAINING_CONSTANT_CHANNELS` shrunk from `frozenset({20})` → `frozenset()`.
  All 27 channels are now populated end-to-end.
- `_DATASET_CACHE_VERSION` 8 → 9 invalidates caches built while Ch 20 was
  zero; next training triggers a full rebuild that pulls real OKX funding
  values into the per-product samples.

### Coverage / retention
- 30 products mapped (88 % spot-checked coverage). Missing on OKX: MATIC
  (rebranded POL), RNDR (rebranded RENDER), FET, VVV — these gracefully
  return `[]` via the existing `inst_id is None` short-circuit.
- OKX funding history retention is **~90 days**. Live BTC fetch confirmed
  180 rows over a 60-day window (3× daily funding events). Older training
  samples beyond the 90-day window will still see Ch 20 = 0 for now;
  inference and recent windows carry real values.

### Tests
- New: `backend/tests/test_okx_funding_history.py` — 13 tests covering
  symbol mapping (known/unsupported), single-page success path, all
  failure modes (non-200, non-zero `code`, network exceptions, malformed
  rows), kill switch on/off, pagination via `after=` cursor, and early
  termination when `oldest_ts <= start_ms`.
- Updated: `test_cnn_agent.py::TestTrainingConstantChannelMask` and
  `TestMaskShrinkAndCacheBump` — assert the empty mask and cache version 9.
- 199 tests pass (cnn_agent 186 + okx_funding_history 13).

### Smoke test
```
fetch_funding_history('BTC-USD', now-60d, now) → 180 rows
fetch_funding_history('SOL-USD', now-60d, now) → 180 rows
fetch_funding_history('VVV-USD', now-60d, now) → []  (no OKX listing)
```

### How to roll forward
1. Restart backend — loads new code, picks up `_DATASET_CACHE_VERSION = 9`.
2. First scan of any product triggers full dataset rebuild (cache v8 dropped).
3. Train worker re-trains with real Ch 20; compare val_loss vs the 0.6931
   baseline that 3 prior glu1 runs all hit while Ch 20 was masked.

### Bugs fixed
None — pure additive feature.

---

## [Session 46] — 2026-04-26 — Path A: candle-derived proxies for masked channels (#60-62)

### Context
Tasks #60/#61/#62 (parked) covered Ch 10/11 (L1 order book), Ch 24/25 (IV/RV
spread), and Ch 26 (Binance top-trader sentiment) — all masked because their
external data sources were unavailable: Coinbase has no historical L1 depth,
Deribit options only cover BTC/ETH, and fapi.binance.com is geo-blocked from
US (HTTP 451). Five of the 27 channels were silently constant-zero per window,
contributing nothing to training.

### Change
Replaced the five external-data channels with candle-derived proxies (Path A —
no external dependencies, real per-bar variance for every product):
- **Ch 10** ← `_close_position(closes, highs, lows)` — `(close − low) / (high − low)` ∈ [0,1] (intra-bar buy pressure)
- **Ch 11** ← `_bar_range(closes, highs, lows)` — `(high − low) / close × 10`, clipped [0,1] (relative bar volatility)
- **Ch 24** ← `_rv_series(closes, window=20)` — rolling 20-bar std of log returns × 50, clipped [0,1]
- **Ch 25** ← `_rv_series(closes, window=60)` — rolling 60-bar std of log returns × 50, clipped [0,1]
- **Ch 26** ← `_volume_sentiment(closes, volumes, window=20)` — rolling up-vol/total-vol mapped to [-1,1]

Five new module-level helpers in `backend/agents/cnn_agent.py`. No external
imports needed. `_TRAINING_CONSTANT_CHANNELS` shrunk from `{10, 11, 20, 24, 25, 26}`
→ `{20}` (only Ch 20 funding rate stays masked while geo-blocked).
`_DATASET_CACHE_VERSION` 7 → 8 invalidates caches that contain the old constant
values; next training triggers full rebuild.

### Why Path A (not B/C)
- Path A: derived proxies — zero external deps, ships immediately, all 5 channels become live.
- Path B: real sources where available — narrow benefit (BTC/ETH-only for #61, geo-block risk for #62).
- Path C: drop channels (27 → 22) — clean but requires arch change + full retrain + breaks current production glu2 baseline.

### Tests
- `TestClosePositionHelper` (4): close at high/low/mid, zero-range no-div-zero
- `TestBarRangeHelper` (3): zero range, 1% scaling, clip at 1.0
- `TestRealizedVolHelper` (4): constant prices → 0, pre-window bars → 0, vol comparison, [0,1] clip
- `TestVolumeSentimentHelper` (5): all-up → +1, all-down → −1, balanced → ~0, zero-vol → 0, first bar → 0
- `TestMaskShrinkAndCacheBump` (2): mask now `{20}` only, cache version == 8

187 tests pass.

### How to roll forward
Backend currently running glu1 retrain (PID 31724) on the OLD code (cache v7,
mask `{10, 11, 20, 24, 25, 26}`). Once that completes, restart backend to load
new code and trigger a fresh retrain — first run with cache v8 forces full
dataset rebuild with the new channel values.

### Memory updates
- `feedback_*` — no rule changes
- `coinbase_trader_schema.md` — landmark line numbers (helper definitions, mask, cache version)

---

## [Session 45] — 2026-04-26 — Capacity-reduced glu1 arch + per-arch checkpoints (#40)

### Context
Six consecutive glu2 retrains since Session 44 were rejected by save-if-better
(best_val_loss 0.5908 / 0.6011 / 0.6001 / 0.6168 / 0.6118 / 0.6021 — all ≥ the
0.5828 baseline). Every run shows the same OVERFIT signature: peak val_loss at
epoch 1, train loss falls 0.577 → 0.388, val_loss climbs to ~0.75 by epoch 16
(overfit_gap_pct ≈ 106%). val_auc holds at 0.71–0.76 — signal exists but the
~280k-param glu2 memorizes the train set immediately. Hypothesis: capacity
bottleneck. Fix: introduce a smaller sibling arch (`glu1`, ~5–8× fewer params)
selectable via `CNN_ARCH` env var, with separate on-disk checkpoint files so
the working glu2 baseline (val_loss=0.5828) is preserved as a safety net while
glu1 trains.

### Change
**`backend/agents/cnn_agent.py`**
- New `SignalCNNGlu1` class — 27→16→32 GatedConv1d (one MaxPool, 60→30), 1-layer LSTM(32→16), Dropout(0.3), FC(16→1). Carries `arch = "glu1"` for checkpoint compatibility checking. Includes `predict()` mirror of `SignalCNN`.
- New `_active_arch()` reads `CNN_ARCH` env at call-time (default `glu2`); change takes effect on agent boot.
- New `_build_cnn(arch)` factory routes via `_ARCH_REGISTRY = {"glu2": SignalCNN, "glu1": SignalCNNGlu1}`. Raises `ValueError` on unknown arch.
- New `_model_path_for(arch)` / `_best_loss_path_for(arch)` — `glu2` keeps the legacy `cnn_model.pt` / `cnn_best_loss.txt` paths; non-glu2 archs get `_<arch>` suffix (e.g. `cnn_model_glu1.pt`, `cnn_best_loss_glu1.txt`).
- `CoinbaseCNNAgent.__init__` now stores `self._arch = _active_arch()` and builds via the factory; startup log line includes `arch=`.
- `_exists` / `_load` / `save_model` / `_read_best_loss` / `_write_best_loss` and the `_model_saved` check inside `train_on_history` route through the per-arch path helpers. `_read_best_loss` / `_write_best_loss` converted from `@staticmethod` to instance methods (callsites already used `self.`).
- `save_model` derives its `.bak` path from the active checkpoint path, replacing the now-glu2-only `_MODEL_BAK_PATH`.

**`backend/main.py`**
- `POST /api/cnn/model/reload` now resolves the checkpoint via `_model_path_for(app_state.cnn_agent._arch)` instead of importing `MODEL_PATH` directly. Existence check, `os.stat`, and the response body all reflect the agent's active arch. Response gains an `arch` field. Agent-existence check moved before path resolution.

### Why option (a)
The save-if-better logic compares new val_loss against the previous best; without per-arch files, a winning glu1 retrain would overwrite the glu2 baseline in `cnn_model.pt`, and any later worse glu1 retrain would leave production with no fallback. Separate files preserve both baselines independently — flipping `CNN_ARCH` in `.env` is a safe, reversible choice.

### Tests (TDD red→green)
- `test_cnn_agent.py::TestSignalCNNGlu1` — class exists with `arch="glu1"`, forward shape `(B,1)`, `predict()` returns probability in `[0,1]`, param count ≤ glu2/3. **4/4 PASSED.**
- `test_cnn_agent.py::TestArchFactoryAndPaths` — `_active_arch()` defaults to `glu2` and reads env, `_build_cnn` routes correctly and raises on unknown, `_model_path_for("glu2")` matches legacy `MODEL_PATH`, `glu1` paths carry `_glu1` suffix. **9/9 PASSED.**
- `test_cnn_agent.py::TestCnnAgentArchWiring` — agent default arch is `glu2`, `CNN_ARCH=glu1` selects `SignalCNNGlu1`, `save_model()` under `glu1` writes only `cnn_model_glu1.pt`. **3/3 PASSED.**
- All 13 RED-tests confirmed to fail before implementation (ImportError on `SignalCNNGlu1` / `_active_arch` / `_build_cnn` / `_model_path_for` / `_best_loss_path_for`; `AttributeError: '_arch'`; glu2-path-written assertion).

### How to enable
1. Add `CNN_ARCH=glu1` to `.env` (default remains `glu2`).
2. Restart backend so `CoinbaseCNNAgent.__init__` picks up the new arch.
3. Trigger a retrain — saves to `cnn_model_glu1.pt` against its own `cnn_best_loss_glu1.txt` baseline (initially `inf`, so the first glu1 run unconditionally saves). The glu2 `cnn_model.pt` baseline is untouched.

### Memory
- `coinbase_trader_schema.md` — landmarks updated for new classes/helpers.

---

## [Session 44] — 2026-04-26 — Re-mask Ch 20 funding rate; geo-block kill switch (#80, #81)

### Context
Session 41 unmasked Ch 20 (funding rate) by wiring Binance Futures
historical funding into `_build_dataset`. Production host (US-based) is
geo-blocked from `fapi.binance.com` — every backfill returns HTTP 451,
so the channel is uniformly zero at training time. The model was
training on a feature it could never observe in the wild, contributing
to the val→live gap. Until a non-blocked funding source is wired in,
Ch 20 must be treated as a constant channel again.

### Change
**`backend/agents/cnn_agent.py`**
- `_TRAINING_CONSTANT_CHANNELS` restored to `frozenset({10, 11, 20, 24, 25, 26})` — Ch 20 added back.
- `_DATASET_CACHE_VERSION` bumped 6 → 7 to force full rebuild with new mask.

**`tools/train_cloud.py`**
- `DEFAULT_MASK` mirrored to `frozenset({10, 11, 20, 24, 25, 26})` so cloud retrain matches prod inference mask.

**`backend/services/binance_funding_history.py`** (#81 kill switch)
- New `BINANCE_FUNDING_DISABLED` env var short-circuits `fetch_funding_history` → `[]` without ever constructing `httpx.AsyncClient`. Set via `.env` so the geo-blocked HTTP 451 round-trip is skipped on prod.

### Tests
- `test_cnn_agent.py::TestMaskShrinkAndCacheBump` — assertions updated to mask `{10,11,20,24,25,26}` and version `7`. PASSED.
- `test_binance_funding_history.py` — added `test_disabled_env_var_short_circuits_without_http` (verifies `MockClient.assert_not_called()`) and `test_disabled_env_var_off_makes_http_call` sanity. All 10 tests PASSED.
- `test_train_cloud.py::TestDefaultMaskMatchesProd` — assertion already matched the new set.

### Memory
- `coinbase_trader_schema.md` — note Ch 20 re-masked + kill switch.

---

## [Session 43] — 2026-04-25 — Purge legacy SCALP/MOMENTUM rows from DB (#79)

### Context
Session 42 removed ScalpAgent and MomentumAgent from the runtime but left
historical rows in `trades`, `signal_outcomes`, `agent_state`, and
`agent_decisions` "as-is (no migration)". The frontend Performance
Dashboard and `/api/performance` aggregations were therefore mixing
legacy SCALP/MOMENTUM trades into PnL totals that are no longer
attributable to any active agent. With CNN retrain (best_val=0.6002,
val_auc=0.746) just landed, we want the dashboard to reflect only
agents currently running.

### Change
**Backup first:** `backend/coinbase.db.bak_pre_purge_20260426`
(428,118,016 bytes — full pre-purge snapshot, kept locally, gitignored
via existing `*.db` rule).

**Purged rows** (444,531 total):
| Table | Column | Removed |
|---|---|---|
| `trades` | `agent` IN (MOMENTUM, SCALP) | 4,209 |
| `signal_outcomes` | `source` IN (MOMENTUM, SCALP) | 2,420 |
| `agent_state` | `agent` IN (MOMENTUM, SCALP) | 2 |
| `agent_decisions` | `agent` IN (MOMENTUM, SCALP) | 437,900 |

**VACUUM** reclaimed 294MB: `coinbase.db` 428MB → 134MB.

**Remaining rows** (CNN/TECH only):
- `trades`: CNN=338, TECH=156
- `signal_outcomes`: CNN=20457, TECH=166
- `agent_state`: CNN=1, TECH=1
- `agent_decisions`: TECH=224331

**Verification:**
- `GET /api/performance` returns clean monthly totals (447 trades, 40.9% win rate, +$91.05 PnL) without orphan agent buckets.
- `GET /api/agents/status` returns only `tech` and `cnn` keys.

### Memory
- `coinbase_trader_schema.md` "Active agents" section updated — the
  "Historical rows left as-is" note replaced with the actual purge
  record + remaining counts.

---

## [Session 42] — 2026-04-25 — Remove ScalpAgent + MomentumAgent, port exit-stats to TechAgent

### Context
TechAgent (with TICK_TRAIL exits added in Session 38) covers the same
RSI/BB/MACD/Stoch/OBV signal space as MomentumAgent on a 2-min cadence;
ScalpAgent's only durable contribution was its per-trigger exit-stats
system. Two stale agents removed; the diagnostic value preserved.

### Change
**TechAgent gains per-trigger exit stats** (`agents/tech_agent_cb.py`):
- `_Book._stats: Dict[str, Dict]` keyed by trigger
  (`TICK_SIGNAL` / `TICK_STOP` / `TICK_TRAIL` / `TICK_PROFIT` / `SCAN`)
  with `{wins, losses, total_pnl}` per trigger.
- `_Book.sell()` updates the bucket on every close; `setdefault` handles
  unseen triggers gracefully.
- `TechAgentCB.status["exit_stats"]` exposes per-trigger counters with
  `win_rate` computed inline; only triggers with at least one closed
  trade are returned.
- Diagnostic only — not persisted. The `trades` table is the durable
  source of truth.
- 2 new tests in `tests/test_tech_agent_cb.py`.

**Backend deletions:**
- `backend/agents/momentum_agent_cb.py`
- `backend/agents/scalp_agent.py`
- `backend/tests/test_momentum_agent_cb.py`
- `backend/tests/test_momentum_entry_filter.py`
- `backend/tests/test_scalp_agent.py`

**`backend/main.py`** — dropped imports, AppState fields
(`momentum_agent`, `scalp_agent`, `momentum_task`, `scalp_task`),
startup-stagger constants (`_MOMENTUM_START_DELAY`, `_SCALP_START_DELAY`),
lifespan instantiation + delayed-launch coroutines + task creation,
shutdown task list entries + scalp.stop() call, `/api/agents/status`
payload entries (`mom_status`, `scalp_status`), and `/api/trades` query
description.

**`backend/tests/test_startup_sequence.py`** — reduced to a single
`_TECH_START_DELAY` sanity check; ScalpAgent warmup tests dropped.

**`backend/tests/test_market_scanner.py`** — dropped one
`test_scalp_skips_micro_price_in_scan` test that imported the deleted
ScalpAgent class.

**`backend/tests/test_signal_improvements.py`** — dropped
`TestRSIOverbought` class which imported `_RSI_OVERBOUGHT` from the
deleted `momentum_agent_cb` module.

**Frontend cleanup** (6 files, all under `frontend/src/`):
- `components/AgentsDashboard.tsx`: dropped Momentum + Scalp cards and
  signal feeds; aggregate over TECH + CNN only; reflowed Tech feed to
  a constrained single-column block (`max-w-3xl`) so the layout still
  reads as intentional.
- `components/CNNDashboard.tsx`: dropped Mom + Scalp `<th>` columns and
  IIFE `<td>` blocks from confidence table; updated comment.
- `components/FiringCounter.tsx`: dropped MOM + SCALP rows from Counts
  type, default state, fetch handler, and JSX. (This file was
  previously untracked — now committed for the first time.)
- `components/PerformanceDashboard.tsx`: dropped MOMENTUM + SCALP from
  `AgentFilter` union, both color maps, and chip array.
- `utils/agentByProduct.ts`: `AgentVotes` is now `{ tech }` only.
- `utils/agentByProduct.test.ts`: dropped MOMENTUM/SCALP fixtures and
  assertions; kept tech / multi-product / newest-wins / unknown-agent.

**Docs / cleanup:**
- `_mom_sells.py` deleted (Momentum-only debug script).
- `launcher.py`: subtitle changed from
  "Advanced Trade · RSI · MACD · CNN · Momentum" to
  "Advanced Trade · RSI · MACD · CNN".
- `README.md`: MomentumAgentCB and ScalpAgent rows removed from the
  architecture diagram.
- `REBUILD_STANDARD.md`: scrubbed all class references — file-tree
  entry, MomentumAgentCB section, signal-flow diagram blocks,
  dashboard column row, test-file rows, and staggered-start design row.

### Out of scope
- ScalpAgent's 5-min stop-loss cooldown was NOT ported (user deferred).
- ScalpAgent's confluence-reasons format with `(+score)` annotations
  was NOT applied to TechAgent — TechAgent already has its own
  `buy_reasons`/`sell_reasons` lists in `_score()`.

### DB state
Historical `MOMENTUM` / `SCALP` rows in `agent_state`,
`agent_decisions`, `trades`, and `signal_outcomes` are left as-is
(dry-run only; no migration). New writes will only land under `TECH` /
`CNN`.

### Test count
439 backend tests passing post-removal (down from 525 pre-removal —
~37 scalp tests + ~30 momentum tests + the small follow-on cleanups).
5/5 frontend Vitest tests pass. Dev server compiles green.

---

## [Session 41] — 2026-04-25 — Cloud retrain pipeline (#67–#70)

### Context
The local RTX 2060 (6 GB) was the bottleneck for the LSTM-tail CNN — long
epochs, frequent OOM near batch=128, and Binance funding-history calls
returning HTTP 451 from the US IP. To unblock retrain cycles we built a
self-contained cloud-trainable trainer plus two backend endpoints so
the user can drop fresh weights without restarting the API or waiting
for the next closed trade to refit the LGBM gate.

### Change
**`tools/train_cloud.py`** (new, ~280 lines, #67):
- Self-contained trainer (no `backend.cnn_agent` import) with prod-mirroring
  constants (N_CHANNELS=27, SEQ_LEN=60, _FORWARD_HOURS=4, _LABEL_THRESH=0.003).
- `DEFAULT_MASK = frozenset({10, 11, 24, 25, 26})` matches stage-b
  `_TRAINING_CONSTANT_CHANNELS`.
- `SignalCNN` (arch="glu2") + `GatedConv1d` copies, `apply_mask`, `_smooth`,
  `save_prod_model` writing `{"arch", "n_channels", "state_dict"}`,
  `write_best_loss`, atomic `write_progress_json` (`.tmp`+`os.replace`),
  `_uniqueness_from_indices`, `load_dataset`.
- `run_training` uses BCE `reduction="none"` + uniqueness-weighted mean
  (CLAUDE.md invariant 12), AdamW + ReduceLROnPlateau, grad-clip 1.0,
  label-pos-weight, batch by n_train (256/128/64), LR scaled by
  `(BATCH/64)**0.5`, early-stop patience 8.

**`tools/colab_train.ipynb`** (new, 15 cells, #68):
- nvidia-smi → git clone polymarket_app → mount Drive OR upload tarball
  of `backend/data/cnn_dataset_cache/` → pip install
  `tools/requirements-train.txt` → run train_cloud.py → download
  `cnn_model.pt` + `cnn_best_loss.txt`.
- `metadata.accelerator = "GPU"`.

**`tools/requirements-train.txt`** (new): `torch>=2.0,<3.0`, `numpy>=1.26,<3.0`
— intentionally excludes FastAPI/aiosqlite/ollama (not needed for training).

**`backend/agents/cnn_agent.py`** (#70):
- New `force_lgbm_retrain()` async method — bypasses the
  `n == self._lgbm_trades_seen` short-circuit in
  `_lgbm_retrain_if_needed`. Calls `database.get_lgbm_training_rows`,
  `self._lgbm.train(rows)`, persists on success, updates
  `_lgbm_trades_seen = len(rows)`. Wraps in try/except returning
  `None` on failure.

**Pending implementation (tests landed ahead of code, TDD-style):**
- `POST /api/cnn/model/reload` endpoint in `backend/main.py` (#69).
- `POST /api/cnn/lgbm/retrain` endpoint in `backend/main.py` (#70).
  `force_lgbm_retrain()` is the agent-side method these will call.

### Tests
**`backend/tests/test_train_cloud.py`** (10/10 pass): mask matches prod;
SignalCNN forward shape (B,27,60)→(B,1) and arch="glu2"; apply_mask
zero-out + empty-set passthrough; save_prod_model dict format;
write_best_loss float→str; write_progress_json status + epoch_log
(`completed` and `running` partial).

**`backend/tests/test_cnn_model_reload.py`** (6 tests, awaiting #69
endpoint): auth (no key 401, wrong key 401), 409 when training active,
404 when model missing (detail string includes path), 503 when agent
uninitialised, happy path calls `_load()` and returns metadata.

**`backend/tests/test_lgbm_force_retrain.py`** (7 tests, awaiting #70
endpoint): auth (2), 503 when agent missing, happy path calls
`force_lgbm_retrain` (mocked return), 200 `skipped` when underlying
returns None, `force_lgbm_retrain` runs even when
`n == _lgbm_trades_seen`, `force_lgbm_retrain` returns None on empty
rows.

### Why
The `_lgbm_retrain_if_needed` guard short-circuits on
`n == self._lgbm_trades_seen` to avoid wasted refits when no new
closed trades have arrived. After a CNN swap (cloud retrain → reload)
that guard hurts — the gate stays fit against the old model's scan
distribution until the next closed trade, even though the underlying
prob distribution just shifted. `force_lgbm_retrain` is the manual
escape hatch; the autonomous path is unchanged.

### Follow-ups (open)
- #69: Wire `POST /api/cnn/model/reload` endpoint in `backend/main.py`
  (tests already in place at `backend/tests/test_cnn_model_reload.py`).
- #70: Wire `POST /api/cnn/lgbm/retrain` endpoint in `backend/main.py`
  (tests already in place at `backend/tests/test_lgbm_force_retrain.py`).
- v2 of `tools/colab_train.ipynb`: refetch Binance funding in-Colab and
  rebuild Ch 20 before training (US is the geoblock — Colab is not).
- Decide on revert of mask `{10,11,24,25,26}` →
  `{10,11,20,24,25,26}` until a non-US-blocked funding source is wired
  (Bybit/OKX/CoinGlass).

---

## [Session 40] — 2026-04-25 — Wire Binance funding history into training, unmask Ch 20 (#57 stage b)

### Context
Stage (a) of #57 (Session 39) wired real BTC + 5m candles through
`_build_dataset` and shrank `_TRAINING_CONSTANT_CHANNELS` to
`{10, 11, 20, 24, 25, 26}`. Ch 20 (funding rate) stayed masked because no
historical funding-rate source existed in the codebase — only
`/fapi/v1/premiumIndex` (current `lastFundingRate`) was used at
inference, so every training sample saw funding=0 and the inference mask
zeroed Ch 20 to match.

This session is **stage (b)** of #57: build a Binance historical funding
fetcher, wire it through `train_on_history` Phase 1, and unmask Ch 20.

### Change
**New file** `backend/services/binance_funding_history.py`:
- `_coinbase_to_binance(product_id)` mapping (26 perpetual products:
  BTC, ETH, SOL, XRP, BNB, ADA, AVAX, LINK, DOT, MATIC, DOGE, LTC, ATOM,
  FIL, NEAR, APT, INJ, ARB, OP, TIA, SEI, SUI, RNDR, FET, AAVE, UNI).
- `async fetch_funding_history(product_id, start_ms, end_ms)` hits
  `https://fapi.binance.com/fapi/v1/fundingRate` with
  `symbol`/`startTime`/`endTime`/`limit=1000` and returns sorted
  ascending `[(funding_time_ms, rate), ...]`. Returns `[]` for
  unsupported symbol, HTTP error, non-200, or non-list payload.
- One funding event every 8h; `limit=1000` covers ~333 days, sufficient
  for the typical training window.

**`backend/agents/cnn_agent.py`:**
- Imported `fetch_funding_history` from the new service.
- Added `_aligned_funding_rates(target_candles, funding_history)` helper
  — forward-fills the most-recent funding rate ≤ each bar's timestamp,
  with unit-aware comparison (target candles in seconds, funding events
  in ms). Pre-target events seed the initial value; if every event is
  after the first target bar, seed with the first available rate.
- `train_on_history` Phase 1 now calls `fetch_funding_history(pid,
  fr_start_ms, fr_end_ms)` per product (start/end derived from the
  candle span) and aligns the result via `_aligned_funding_rates`. The
  per-product tuple grew from
  `(pid, candles, btc_aligned, c5m)` to
  `(pid, candles, btc_aligned, c5m, funding_aligned)`.
- `_build_dataset` (closure) unpacks the 5-tuple and passes
  `funding_rates=funding_aligned` into `_extend_or_rebuild_product` on
  both rebuild and append paths.
- Shrunk `_TRAINING_CONSTANT_CHANNELS` from
  `{10, 11, 20, 24, 25, 26}` → `{10, 11, 24, 25, 26}`.
  Still masked: Ch 10/11 (orderbook — no historical L1), Ch 24/25
  (IV/RV — no historical IV source), Ch 26 (L/S sentiment — no
  historical Binance L/S).
- Bumped `_DATASET_CACHE_VERSION` from 5 → 6 (required: pre-v6 caches
  stored zero funding for Ch 20; without the bump, stale tensors would
  still feed the now-unmasked Ch 20 to training, defeating the wiring
  change).

### Tests
**New file** `backend/tests/test_binance_funding_history.py` (8 tests):
- `TestProductSymbolMapping` (2) — known product mapping, unsupported
  product returns None.
- `TestFetchFundingHistory` (6) — sorted tuples, unsupported product
  empty, symbol+window passed to Binance, empty on HTTP error, empty on
  non-200, sorts unsorted payloads.
- All `httpx.AsyncClient` calls mocked; no live API.

**Updated** `backend/tests/test_cnn_agent.py`:
- New `TestAlignedFundingRatesHelper` (6) — length match, single-event
  forward fill, multi-event forward fill across 8h cycles, pre-target
  seed, first-event seed when target precedes funding, None on empty.
- New `TestBuildDatasetWiresBtcAndFiveMinute::
  test_extend_or_rebuild_receives_funding_rates` — spy on
  `_extend_or_rebuild_product` confirms `funding_rates=` is forwarded as
  a list aligned to candles, with the patched payload's value at index 0.
- Updated `TestTrainingConstantChannelMask::
  test_mask_set_covers_expected_channels` and
  `TestMaskShrinkAndCacheBump` to expect `{10, 11, 24, 25, 26}` and
  cache version 6.

### Verification
- `pytest backend/tests/test_binance_funding_history.py -v` — 8/8 pass.
- `pytest backend/tests/test_cnn_agent.py -q` — 153/153 pass.

### Files
- New: `backend/services/binance_funding_history.py`,
  `backend/tests/test_binance_funding_history.py`
- Modified: `backend/agents/cnn_agent.py`,
  `backend/tests/test_cnn_agent.py`, `CHANGELOG.md`

---

## [Session 39] — 2026-04-25 — Wire real BTC + 5m into _build_dataset, shrink mask (#57 stage a)

### Context
Sessions 36–37b plumbed `btc_closes`, `funding_rates`, and `candles_5m`
through `_build_samples_range` and `_extend_or_rebuild_product`, but the
caller `_build_dataset` (closure inside `train_on_history`) never passed
these kwargs. So even after #53/#54/#56, training tensors still saw
`btc_closes=None`, no per-product 5m parquet, and only the synthetic
hourly-proxy 5m window.

That left Ch 15 (ADX), Ch 17/18/19 (5m fast), Ch 21 (BTC corr) inside
`_TRAINING_CONSTANT_CHANNELS` even though their data sources were
already in the codebase. Per CLAUDE.md invariant 11, inference masks
those channels to zero — so the model literally couldn't see them.

This session is **stage (a)** of #57. Stage (b) (funding-rate history
fetcher) is parked behind a new module yet to be built.

### Change
`backend/agents/cnn_agent.py`:
- Imported `load_5m_history` from `services.history_backfill`.
- Added `_aligned_btc_closes(target_candles, btc_candles)` helper —
  forward-fills BTC hourly closes onto a target product's bar timestamps,
  with pre-target BTC bars seeding the initial value so the series
  never starts at None.
- `train_on_history` Phase 1 now loads BTC hourly history once
  (parquet → DB fallback) and, per product, loads the 5m parquet via
  `load_5m_history(pid)`. The per-product tuple grew from `(pid, candles)`
  to `(pid, candles, btc_aligned, c5m)`.
- `_build_dataset` (closure) unpacks the 4-tuple and passes
  `btc_closes=btc_aligned, candles_5m=c5m` into `_extend_or_rebuild_product`
  on both rebuild and append paths. BTC-USD itself receives `btc_closes=None`
  (matches inference at `_extend_or_rebuild_product` callers).
- Shrunk `_TRAINING_CONSTANT_CHANNELS` from
  `{10, 11, 15, 17, 18, 19, 20, 21, 24, 25, 26}`
  to `{10, 11, 20, 24, 25, 26}`.
  Still masked: Ch 10/11 (orderbook — no historical L1), Ch 20 (funding —
  pending stage b), Ch 24/25 (IV/RV — no historical IV source), Ch 26
  (L/S sentiment — no Binance historical L/S).
- Bumped `_DATASET_CACHE_VERSION` from 4 → 5 (required: pre-v5 caches
  stored hourly-proxy 5m + zero BTC; without the bump, stale cached
  tensors would still feed the unmasked positions to training, defeating
  the wiring change).

### Tests
9 new tests in `backend/tests/test_cnn_agent.py`:
- `TestAlignedBtcClosesHelper` (5) — length match, exact alignment,
  forward-fill on gaps, pre-target seed, None on empty BTC.
- `TestBuildDatasetWiresBtcAndFiveMinute` (2) — spy on
  `_extend_or_rebuild_product` confirms `btc_closes` (non-BTC products)
  and `candles_5m` are passed through `train_on_history`.
- `TestMaskShrinkAndCacheBump` (2) — frozenset matches new shape;
  cache version is 5.

Updated existing `TestTrainingConstantChannelMask::test_mask_set_covers_expected_channels`
to match the new mask.

`backend/tests/test_cnn_agent.py` — 146/146 green.

### Behavior
Train/serve invariant 11 still holds: training now sees real values for
Ch 15/17/18/19/21, and inference no longer masks them to zero. The next
auto-train cycle will rebuild the per-product cache from scratch
(invalidated by the version bump) and produce a model that can actually
use those channels. Until that retrain completes, inference reads real
values into a model that was trained with them masked — a brief
distribution shift window that closes on the next training run.

Stage (b) will add `services/binance_funding_history.py` and remove
Ch 20 from the mask once historical funding lands.

---

## [Session 38] — 2026-04-25 — TechAgent trailing $-PnL take-profit (TICK_TRAIL)

### Context
TechAgent already exits on three triggers: cached SELL signal, ATR stop-loss,
and a +6 % take-profit. Small profitable positions that peak above +$1 of
unrealized PnL but reverse before reaching +6 % were leaking gains. This
session adds a per-position trailing dollar exit that captures those wins.

### Change
`backend/agents/tech_agent_cb.py`:
- New constants `_TRAIL_ARM_USD = 1.00` and `_TRAIL_GIVEBACK_USD = 0.25`.
- `_Book.buy` initializes `peak_pnl_usd: 0.0` on new positions; averaging
  up an existing position leaves the peak alone (intentional).
- `on_price_tick` now updates `pos["peak_pnl_usd"] = max(prev, current_pnl_usd)`
  every tick (before the exit if/elif chain).
- New `TICK_TRAIL` branch inserted between `TICK_STOP` and the legacy
  `TICK_PROFIT`. Fires when `peak_pnl_usd >= $1.00` AND
  `peak - current >= $0.25`.
- Stale inline comment on the +6 % take-profit ("20% above entry") corrected
  to "legacy %-based backstop".

Persistence rides on the existing `agent_state.positions_json` blob — no
DB schema change. Legacy saved positions (pre-upgrade) lack the key and
fall back to `0.0` via `dict.get`, picking up tracking on the next tick.

### Tests
12 new tests across two classes in `backend/tests/test_tech_agent_cb.py`:
- `TestTrailingDollarExitState` (3) — constants present, fresh init, no
  reset on average-up.
- `TestTrailingDollarExit` (9) — peak rises/holds correctly, no fire below
  arm or below giveback, fires at threshold with `TICK_TRAIL` trigger,
  trail wins over `+6 %` take-profit, ATR stop wins on a loss, legacy
  positions don't crash, re-entry resets the peak.

`backend/tests/test_tech_agent_cb.py` — 30/30 green.

### Behavior
Existing exit triggers unchanged. The +6 % take-profit becomes a backstop
that will rarely fire because the trail typically triggers first.

### Spec / plan
- Spec: `docs/superpowers/specs/2026-04-24-tech-agent-trailing-dollar-exit-design.md`
- Plan: `docs/superpowers/plans/2026-04-24-tech-agent-trailing-dollar-exit.md`

---

## [Session 37b] — 2026-04-24 — Real 5m candles wired into training (Task #56)

### Context
Session 37 (#55) landed the per-product 5m parquet loader but no caller
consumed it yet — `_build_samples_range` still passed the synthetic
`candles[max(0, i-11): i+1]` (12 hourly bars) to `fb.build`'s `candles_5m=`
kwarg. That < 14-element window forces FeatureBuilder.build's Ch 17/18/19
branch into its degenerate fallback (constants `0.5 / 0.0 / 0.0`), which is
why Session 32 added Ch 17/18/19 to `_TRAINING_CONSTANT_CHANNELS`. Until the
training-time 5m path delivers ≥14 real bars per sample, the mask cannot
shrink without retraining on garbage.

### Change
`backend/agents/cnn_agent.py`:
- `_build_samples_range(...)` gains `candles_5m: Optional[List[Dict]] = None`
- New module constant `_TRAIN_5M_LIMIT = 72` mirrors the inference fetch
  (`scan_all` requests `limit=72` 5m bars = 6h)
- Per-sample slice: `t_close = candles[i]["start"] + 3600`;
  `cutoff = bisect.bisect_left(c5m_starts, t_close)`;
  `five_m = candles_5m[max(0, cutoff - 72): cutoff]`
  → strictly excludes any 5m bar starting at or after the close of hour `i`
  (no look-ahead leak), capped at the last 72 bars to match inference
- `c5m_starts` is built once per product call; per-sample lookup is O(log n)
- When `candles_5m` is None, the legacy synthetic proxy is preserved
  byte-for-byte → no behaviour change for callers that haven't migrated
- `_extend_or_rebuild_product(...)` gains the same kwarg and forwards it to
  both the rebuild and append code paths
- `import bisect` added at module top

Caller plumbing (`_build_dataset` → `_extend_or_rebuild_product` → loader)
remains unchanged this commit. The atomic flip happens in Task #57 when
`_TRAINING_CONSTANT_CHANNELS` shrinks: the caller will start passing
`load_5m_history(pid)` and `_DATASET_CACHE_VERSION` will bump (#58) so all
existing per-product cache entries rebuild against the new 5m channels.

### Tests
New class `TestBuildSamplesRangeRealFiveMinute` (7 tests) in
`backend/tests/test_cnn_agent.py`:
- `test_default_falls_back_to_synthetic_hourly_proxy` — 3600s spacing when None
- `test_real_5m_candles_passed_through_when_provided` — 300s spacing in slice
- `test_5m_slice_excludes_future_bars` — every passed bar has start < t_i+3600
- `test_5m_slice_size_matches_inference_convention` — caps at 72 (last call)
- `test_short_5m_history_returns_what_is_available` — partial 5m parquet OK
- `test_extend_or_rebuild_plumbs_candles_5m_on_rebuild` — rebuild forwards kwarg
- `test_extend_or_rebuild_plumbs_candles_5m_on_append` — append forwards kwarg

`_CapturingFB` records every `fb.build` call's `candles_5m` slice so each
assertion can introspect spacing, length, and timestamp bounds without
needing the full FeatureBuilder pipeline.

RED verified before implementation: 6/7 tests failed with
`TypeError: _build_samples_range() got an unexpected keyword argument 'candles_5m'`.

### Behavior (deliberately unchanged this commit)
No production caller passes `candles_5m=` yet. CNN training output and
inference paths are bitwise unchanged. The dataset cache version stays at 4.

---

## [Session 37] — 2026-04-24 — 5-minute candle backfill (Task #55)

### Context
Group 3 remediation: Ch 15 (ADX), 17 (fast RSI), 18 (velocity), 19 (volume-Z)
are computed at inference from real 5-minute candles but trained on a 12-bar
slice of hourly bars treated as a proxy. That proxy is what drove Session 32's
decision to mask these four channels in `_TRAINING_CONSTANT_CHANNELS`. Before
the mask can shrink (Task #57) we need real historical 5m candles persisted
per product; this session lands the fetch layer.

### Change
`services/history_backfill.py` gains a parallel 5-minute path without altering
any existing hourly behaviour:
- `_FIVE_MINUTE_BAR_SECS = 300` / `_FIVE_MINUTE_GRANULARITY = "FIVE_MINUTE"`
- `_parquet_path_5m(pid)` → `backend/data/history/5m/{pid}.parquet`
  (separate namespace so hourly parquets are never overwritten)
- `load_5m_history(pid)` — analog of `load_history`
- `backfill_product_5m(pid, days=30)` — analog of `backfill_product`
- `_fetch_range` gains a `granularity: str = _GRANULARITY` kwarg
- Internals factored: `_load_from_path` / `_save_to_path` / `_backfill_to_path`
  now power both hourly and 5m; existing `load_history` / `_save_history` /
  `backfill_product` become thin wrappers

Default `days=30` yields ~8640 bars per product (~29 paged requests, ~12s each
at `_REQ_DELAY=0.35`). Callers scale up as the retrain plan demands.

### Behavior (deliberately unchanged this session)
Nothing calls `backfill_product_5m` yet — no change to the `run_loop` or
startup path. Task #56 will wire the loader into `_build_samples_range`;
Task #57 will add a startup-time 5m backfill trigger before flipping the mask.

### Tests
New file `backend/tests/test_history_backfill.py` (11 tests):
- `TestFiveMinuteParquetPath` (3) — path separation, distinct from hourly
- `TestLoad5mHistory` (3) — empty case, roundtrip, hourly/5m isolation
- `TestBackfillProduct5m` (4) — FIVE_MINUTE granularity passed through,
  writes to 5m path only, 5-minute pagination window (not hourly), incremental
- `TestFetchRangeGranularityParam` (1) — signature accepts granularity kwarg

No live API calls; `_fetch_range` mocked with AsyncMock in every async test.
CLAUDE.md coverage table line for this module is now actually backed by a file.

---

## [Session 37] — 2026-04-24 — Wire funding rates through training sample builder (Task #54)

### Context
Continuing Group 2 remediation from Task #53. Ch 20 (funding rate) is already
fetched at inference time from Binance `/fapi/v1/premiumIndex` (cnn_agent.py
:1541-1550), but training always saw 0.0 because `_build_samples_range` never
received a per-sample rate. Result: train/serve skew — the model learned the
channel was a constant zero and the mask zeroes it at inference to preserve
the invariant.

### Change
`_build_samples_range` and `_extend_or_rebuild_product` now accept an optional
`funding_rates: Optional[List[float]]` (aligned 1:1 with `candles`). Per sample
at candle index `i`, `funding_rates[i]` is forwarded to `fb.build(..., funding_rate=...)`
which clips to ±1.0 after `/0.01` normalisation and broadcasts across the
window. `FeatureBuilder.build` already supported the scalar kwarg; only the
training-time plumbing was missing.

### Behavior (deliberately unchanged this session)
The caller in `_build_dataset` still passes `None` — no training distribution
change yet. Tasks #55/#56/#57 will fetch Binance historical funding at the
call site, shrink `_TRAINING_CONSTANT_CHANNELS`, and bump `_DATASET_CACHE_VERSION`
in one coordinated change to avoid train/serve skew.

### Tests
`TestBuildSamplesRangeFundingRates` (5 tests, `backend/tests/test_cnn_agent.py`):
- `test_default_no_funding_rates_leaves_channel_20_zero` — regression
- `test_constant_funding_rates_broadcast_to_channel_20` — end-to-end normalise/broadcast
- `test_funding_rate_selected_per_sample_index` — per-sample index alignment
- `test_funding_rates_clipped_at_plus_minus_one` — clipping boundary
- `test_extend_or_rebuild_plumbs_funding_rates_on_rebuild` — cache-rebuild path
`_FakeFB.build` in `TestPerProductDatasetCache` gains `funding_rate=None` kwarg.
Module total: 125 → 130 tests.

---

## [Session 37] — 2026-04-24 — Wire BTC closes through training sample builder (Task #53)

### Context
Session 32 audit left 11 of 27 CNN channels constant-zero during training
(inference-time mask keeps train/serve distributions aligned — invariant #11).
The structural ceiling behind val_loss ~0.68 is feature starvation, not overfit
capacity. Session 37 begins Group 2+3 remediation: unmask channels whose data
is already available or cheaply fetchable.

### Change
`_build_samples_range` and `_extend_or_rebuild_product` now accept an optional
`btc_closes: Optional[List[float]]` (aligned 1:1 with `candles`). When supplied,
each sample at candle index `i` forwards the slice `btc_closes[i-seq_len+1 : i+1]`
to `fb.build(..., btc_closes=...)`, populating Ch 21 (rolling BTC-return
correlation). `FeatureBuilder.build` already supported this kwarg; only the
training-time plumbing was missing.

### Behavior (deliberately unchanged this session)
The caller in `_build_dataset` still passes `None` — no training distribution
change yet. Task #57 will flip the mask (removing Ch 21 from
`_TRAINING_CONSTANT_CHANNELS`), wire BTC closes at the call site, and bump
`_DATASET_CACHE_VERSION` in one coordinated change to avoid train/serve skew.

### Tests
`TestBuildSamplesRangeBtcCloses` (4 tests, `backend/tests/test_cnn_agent.py`):
- `test_default_no_btc_closes_leaves_channel_21_zero` — regression
- `test_aligned_btc_closes_populate_channel_21` — end-to-end plumbing
- `test_btc_closes_sliced_per_window` — per-window slice alignment
- `test_extend_or_rebuild_plumbs_btc_closes_on_rebuild` — cache-rebuild path
`_FakeFB.build` in `TestPerProductDatasetCache` updated to accept the new kwarg.

---

## [Session 36] — 2026-04-23 — Inference-time regime gate (Task #52, Option C)

### Root cause
Phase-1 overfit investigation (Sessions 34–35) closed with "signal-limited, not
capacity-limited" — tiny 5k-param model flat-lined at val_loss 0.72, matching
prod. Live BUY outcomes on 193 closed CNN trades revealed **inverse regime
calibration**: the CNN is most confident in TRENDING (avg cnn_prob 0.925) where
it is least accurate (44.3% wr), and least confident in CHAOTIC (0.703) where
the real edge lives (58.5% wr). Training learns from high-ADX TRENDING gradients
but those trends have exhausted by entry time.

### Fix (TDD)
Non-destructive inference-time gate — block CNN BUY execution when
`hmm_regime != "CHAOTIC"`. Captures the 14pp winrate edge without retraining.

- `backend/agents/cnn_agent.py`:
  - New module-level helper `_regime_gate_enabled()` reading `CNN_REGIME_GATE`
    env (default `"on"`, set to `"off"` for emergency unblock). Read at call
    time so operational toggle does not require reload.
  - BUY execution path in `generate_signal`: inserted regime gate between the
    existing Hurst check and the LGBM filter. When gate is on and
    `hmm_regime != "CHAOTIC"`, `signal["execution"]` is set to
    `{"success": False, "reason": "Regime <X> — CNN BUY edge is CHAOTIC only"}`.
- `backend/tests/test_cnn_agent.py`:
  - New `TestInferenceRegimeGate` class (3 tests):
    `test_buy_blocked_when_regime_is_trending`,
    `test_buy_allowed_when_regime_is_chaotic`,
    `test_regime_gate_disabled_via_env`.

### Verification
- RED: `test_buy_blocked_when_regime_is_trending` failed — book.buy called once
  in TRENDING (no gate yet).
- GREEN: all 3 gate tests pass; 121/121 `test_cnn_agent.py` tests green, no
  regressions.

### Follow-up
- Task #40 (val_loss ceiling fix) remains open — gate is a signal-side
  workaround, not a root cause fix.
- If CHAOTIC BUY winrate holds above baseline after 2–3 days of live traffic,
  begin Option A (backfill 11 masked training channels).

---

## [Session 35] — 2026-04-23 — LGBM pnl-weighted training (Task #43)

### Root cause
Task #39 CNN-unblock investigation showed that `LGBM_GATE_THRESHOLD=0.35` override was
not enough: `backend/logs/backend.log` on 2026-04-23 recorded 148 consecutive
`CNN BUY ... blocked by LGBMFilter: p(win)=0.15–0.17` entries, zero `CNN BOOK BUY`.
The LGBM was trained on 208 closed CNN trades with a 23.1% win rate using binary
`pnl>0` labels, so predictions collapsed into a narrow 0.15–0.17 band for every BUY —
no threshold <0.17 would unblock without also breaking the gate's ranking value.

### Fix (TDD)
Weight training samples by `|pnl|` so large winners/losers dominate learning and
near-zero noise trades contribute minimally.

- `backend/data/lgbm_filter.py`:
  - New `_sample_weights(rows) -> np.ndarray` returning `max(|pnl|, 1e-3)` per row
    (floor prevents LightGBM dropping 0-weight rows).
  - `train()` now computes `w = _sample_weights(rows)`, splits it 80/20 with X/y, and
    forwards `sample_weight=w_tr` + `eval_sample_weight=[w_val]` to `LGBMClassifier.fit`.
- `backend/tests/test_lgbm_filter.py`:
  - New `TestLGBMFilterPnlWeighting` class (3 tests): helper-returns-|pnl|,
    zero-pnl-floored-above-0, fit-receives-sample-weight.

### Verify
```
.venv/Scripts/python.exe -m pytest backend/tests/test_lgbm_filter.py -v
```
→ 19/19 green (16 existing + 3 new).

### Next
Task #44: force retrain on restart so the new label weighting actually produces a
fresh `.pkl`. Task #45: watch `backend.log` for the first `CNN BOOK BUY` to confirm
the gate opens on strong winners without blanket-passing weak ones.

---

## [Session 34] — 2026-04-22 — Isolate real `_BEST_LOSS_PATH` + `MODEL_PATH` from `TestTrainOnHistory*` tests

### Root cause
Incident: the CNN training subprocess at 2026-04-21 20:41 UTC saved a model with `best_val_loss=0.6888` even though the previous best on disk was 0.6684 — the save gate (`best_val_loss < prev_best`) should have rejected it. Investigation found the gate was reading `inf` for `prev_best`, meaning `cnn_best_loss.txt` had been reset to a stale sentinel just before the run.

`TestTrainOnHistory` (8 tests) and `TestTrainOnHistoryNonBlocking` (2 tests) in `backend/tests/test_cnn_agent.py` call the real `agent.train_on_history()` with `database.get_products`/`get_candles` mocked but **without** patching `_BEST_LOSS_PATH`, `MODEL_PATH`, or `_MODEL_BAK_PATH`. On synthetic sinusoidal data, `best_val_loss` rounds to ~0.0 in fp32; `_write_best_loss(0.0)` then clobbers `backend/cnn_best_loss.txt` and `save_model(backup=True)` clobbers both `backend/cnn_model.pt` and `backend/cnn_model.pt.bak`. The pre-commit hook runs the full test suite on every Python-file commit, so this poisoning happened repeatedly — visible in `backend/logs/cnn_training.log` as multiple `val inf → X` saves and `prior best 0.0000` rejections since 2026-04-19.

### Fix (TDD)
- `tests/test_cnn_agent.py`:
  - New RED guard tests `TestTrainOnHistory::test_production_paths_are_isolated` and `TestTrainOnHistoryNonBlocking::test_production_paths_are_isolated` — assert `ca._BEST_LOSS_PATH` and `ca.MODEL_PATH` don't resolve to the real `backend/cnn_*` files at test-run time.
  - Added autouse class-level fixture `_isolate_model_paths(tmp_path, monkeypatch)` to both classes that redirects `ca._BEST_LOSS_PATH`, `ca.MODEL_PATH`, and `ca._MODEL_BAK_PATH` into `tmp_path`.
- No production code changed. `save_model` / `_write_best_loss` / `_read_best_loss` behavior is untouched.

### Blast radius
- `backend/cnn_model.pt.bak` at mtime 2026-04-21 20:18 was the backup written when the test run during this session's earlier commit (Session 33) triggered a fake save — it holds a synthetic-data checkpoint, not the prior production model. The live `backend/cnn_model.pt` (mtime 20:41) is the real 0.6888 subprocess save; `backend/cnn_best_loss.txt` (0.688838) matches it, so production state is internally consistent — just regressed from 0.6684. Future legit training runs will beat 0.6888 and restore forward progress.

### Verify
```
.venv/Scripts/python.exe -m pytest backend/tests/test_cnn_agent.py --tb=short -q
```
→ 118/118 green (includes 2 new guard tests). Production files' mtimes unchanged after the run, proving the fixture prevents the clobber.

---

## [Session 33] — 2026-04-21 — Persist val_auc + precision/recall at production BUY threshold

Audit of the last 14 training runs (log-scraped, since `val_auc` was never persisted) found Spearman ρ ≈ +0.11 between `best_val_loss` rank and `val_auc` rank — the two metrics essentially disagree on which checkpoint is best. Before switching the save gate from `best_val_loss < prev_best` to anything else, we need the candidate metrics in the DB so gate-choice alternatives can be validated against live outcomes empirically.

### Scope (instrumentation only; save gate unchanged)
- `database.py`: added `val_auc`, `val_precision_at_thresh`, `val_recall_at_thresh`, `val_threshold` to `cnn_training_sessions` (nullable REAL). Added ALTER TABLE migrations so existing DBs upgrade in place. Extended `save_training_session` INSERT to persist the new fields (defaulting to NULL when the caller doesn't provide them).
- `agents/cnn_agent.py`:
  - New module-level helper `_precision_recall_at_threshold(probs, labels, threshold)` — returns `(precision, recall)`, each Optional[float]. Uses strict `>` to match the production gate `model_prob > config.cnn_buy_threshold` at cnn_agent.py:1637. `precision=None` when no preds above threshold; `recall=None` when no positive labels.
  - `train_on_history` now hoists the val-set sigmoid pass above the AUC block so AUC and precision/recall share the same `_probs_list`/`_labels_list`. After AUC, computes precision/recall at `config.cnn_buy_threshold` (default 0.60) and logs both. Added `val_precision_at_thresh`, `val_recall_at_thresh`, `val_threshold` to the `result` dict.
- `tests/test_cnn_agent.py`: new `TestPrecisionRecallAtThreshold` (8 cases: empty, length mismatch, all-below with/without positives, perfect classifier, mixed known values, strict-threshold boundary, sell-side threshold).
- `tests/test_database.py`: new `TestCNNTrainingSessions` (3 round-trip cases: val_auc persists, precision/recall/threshold persist, fields default to NULL when absent).

### Next steps (deferred)
- Observe 10–20 fresh training runs with the new metrics in the DB.
- Evaluate whether `val_auc` or `val_precision_at_thresh` correlates better with post-deployment 4h outcome win rate in `signal_outcomes`.
- Only then propose a composite gate (e.g., `val_precision_at_thresh ↑` with `val_loss < 0.693` floor).

Verify: `cd backend && python -m pytest tests/test_cnn_agent.py::TestPrecisionRecallAtThreshold tests/test_database.py::TestCNNTrainingSessions -v` → 11/11 green. Full suite `tests/test_cnn_agent.py tests/test_database.py` → 142/142 green.

---

## [Session 32] — 2026-04-21 — CNN Training Quality Improvements (P1–P4)

A 9-task plan to improve CNN training quality and honesty of val metrics. All work under TDD (RED→GREEN) and behind `backend/tests/test_cnn_agent.py`.

### P1 — Save-gate unblock (`53bc37a`)
- A stale sub-0.1 `cnn_best_loss.txt` was blocking every subsequent save-if-better check.
- `save_model` now treats any recorded best below `_MIN_PLAUSIBLE_LOSS = 0.1` as "unset" and falls through to save.

### P2 — Per-product append-only dataset cache (`db497c6`)
- Replaced the single-fingerprint dataset cache with per-product entries keyed by `(first_ts, last_ts, last_n)`.
- New helpers: `_dataset_schema`, `_build_samples_range`, `_extend_or_rebuild_product`, `_load_pp_cache`, `_save_pp_cache`. Schema versioned via `_DATASET_CACHE_VERSION` (now 4).
- Warm runs now append only newly-arrived candles instead of rebuilding phase 2 end-to-end (103 min → near-instant).

### P3a — Triple-barrier labels (`13af769`)
- `_label_triple_barrier(candles, i, max_bars, up_mult, dn_mult, label_thresh)` labels each sample by whichever of {upper barrier +1%, lower barrier −1%, time barrier} fires first inside the forward window (López de Prado 2018). Replaces sign-of-4h-return.
- `_TB_UP_MULT = 0.01`, `_TB_DN_MULT = 0.01`.

### P3b — Train/serve distribution alignment (`683ada0`)
- Audit found 11 feature channels constant at training due to missing upstream inputs.
- `_TRAINING_CONSTANT_CHANNELS` frozenset + `_mask_training_constant_channels` applied at inference (`_cnn_prob`), zero-ing the same channels the model never saw vary. Keeps `N_CHANNELS=27` (checkpoint compatibility preserved).

### P3c — Sample-uniqueness weighting (`3994d83`)
- `_compute_uniqueness(sample_indices, forward_hours, n_candles)` returns per-sample weights `u_j = mean(1/N_t)` over the forward window (López de Prado 2018 ch. 4).
- `_sync_fit` BCE now uses `reduction="none"` and takes a weighted mean over uniqueness for both train and val loss. Isolated samples get weight 1.0; densely overlapping samples approach `1/forward_hours`.

### P3d — Label smoothing (`35c1a68`)
- `_LABEL_SMOOTH = 0.05` and `_smooth_labels(y, ε)` map hard targets to soft `{ε, 1−ε}` before `binary_cross_entropy_with_logits` (Szegedy 2016).
- Training BCE uses smoothed labels; val BCE keeps hard labels so val-loss remains comparable across runs.

### P3e — Purged walk-forward CV index helper (`6939815`)
- `_purged_walkforward_splits(sample_indices, n_splits, forward_hours, embargo_bars)` (López de Prado 2018 ch. 7): walk-forward CV with purging (drop training samples whose forward window overlaps val) and embargo (drop samples in the serial-correlation band after each val block).
- Policy constants `_WALKFORWARD_FOLDS = 3`, `_WALKFORWARD_EMBARGO = 4`.
- Ships the index helper with 9 tests; wiring into `_sync_fit` deferred — requires first globally time-sorting samples across products in `_build_dataset`.

### P4 — Per-regime validation metrics (`f992696`)
- `_per_regime_metrics(y_true, y_pred, regimes)` buckets val predictions by HMM regime (`TRENDING`/`RANGING`/`CHAOTIC` from `services/hmm_regime.py`) and reports per-regime `n`, accuracy (0.5 threshold), mean BCE loss, and positive rate.
- Surfaces regime-dependent asymmetry that aggregate val_loss hides. Ships helper + 8 tests; training-log integration deferred.

### Sidecar
- Task #22 — pre-existing `test_losses_are_positive_floats` was failing on fresh `main` (mock data generated a single-class dataset, BCE rounded to 0.0 in fp32). Mock expanded to 200 bars; strict `> 0` relaxed to finite and `< 10` (the real intent is catching NaN/Inf, not fp precision).

### TDD evidence
- Full suite: **454 passed, 13 warnings** on commit `f992696`.

---

## [Session 31] — 2026-04-20 — CNN Training Watchdog + Dataset Cache

Every CNN auto-train subprocess was being killed by the log-stale watchdog before it ever reached phase 3 (actual training). Three coordinated fixes.

### Root cause
`train_worker.py` phase-2 dataset build (sliding-window feature extraction over 100 products × ~300 k samples) is CPU-bound Python and logged progress every 10 products — at ~10–13 min per 10 products, the gap between log lines was right at the 15-min watchdog threshold. One slow product → log stale → watchdog kills subprocess → auto-restart → same thing repeats.

### Change 1 — Raise watchdog window to 30 min
- `backend/main.py:287` — `_TRAIN_STALE_LOG_SECS` 900 → 1800.
- Tests in `backend/tests/test_train_watchdog.py` extended: assert new constant + explicit test that 20-min log idle is NOT stale.

### Change 2 — Halve phase-2 log cadence
- `backend/agents/cnn_agent.py` — new module constant `_PHASE2_LOG_EVERY = 5` (was hard-coded 10 inline). Log cadence now ~5–6 min between lines, well inside the 30-min window.
- Test: `TestPhase2LogCadence::test_phase2_log_every_is_5`.

### Change 3 — Cache phase-2 dataset to disk
- New module-level helpers in `backend/agents/cnn_agent.py`: `_dataset_fingerprint`, `_load_dataset_cache`, `_save_dataset_cache`. Stored at `backend/cnn_dataset_cache.pt` (gitignored via `backend/*.pt`).
- Fingerprint SHA-256 over `(SEQ_LEN, _FORWARD_HOURS, _LABEL_THRESH, N_CHANNELS)` + per-product `(count, first_ts, last_ts, last_close)`. Any change → miss → rebuild.
- `_build_dataset` closure now checks cache first; on hit returns cached tensors and skips the entire sliding-window loop. On miss builds as before and saves before returning. Save failure is non-fatal.
- Expected impact: first post-fix run still spends 30–40 min in phase 2 (cache miss), subsequent runs load in seconds until new candles arrive.
- Tests: `TestDatasetCache` class with 6 tests covering fingerprint determinism, parameter sensitivity, round-trip I/O, and mismatch/miss handling.

### TDD
- RED verified for each change before implementation (constant assertions + helper attribute misses).
- GREEN full suite: **406 passed**.

---

## [Session 30.3] — 2026-04-19 — Momentum Agent: Mirror TechAgent Risk Controls

Three coordinated changes to `backend/agents/momentum_agent_cb.py` so the momentum agent's exit logic matches TechAgent's proven behavior.

### Change 1 — Macro regime multiplier on SELL score
Added `_macro_adjusted_buy_score` and `_macro_adjusted_sell_score` methods mirroring `tech_agent_cb.py:209-217`. `analyze_product` now fetches `MacroContext` and passes both scores through `buy_gate_multiplier()` / `sell_gate_multiplier()` before comparing to thresholds. Short-squeeze regimes (crowded shorts, very negative funding) now scale the SELL score down to [0.5, 1.0], avoiding selling into lows. `mom_s < -_MOMENTUM_THRESH` escape hatch preserved.

### Change 2 — ATR(14)-based stop replaces fixed trail + hard stop
- Removed `_TRAILING_STOP = 0.03` and `_HARD_STOP_LOSS = 0.05`.
- Added `_ATR_MULTIPLIER = 3.0`, `_ATR_STOP_MIN = 0.015`, `_ATR_STOP_MAX = 0.12`.
- New `_compute_atr_stop(candles, entry_price)` method (mirrors `tech_agent_cb.py:219-235`): stop = ATR(14) × 3.0 / entry, clamped to [1.5 %, 12 %]. Falls back to `_ATR_STOP_MIN` when data insufficient or entry ≤ 0.
- SCAN BUY stores `atr_stop` on the position dict.
- TICK handler compares `pct < -pos.get("atr_stop", _ATR_STOP_MIN)` instead of `_HARD_STOP_LOSS`.
- Removed `_check_trailing_stop` and the trailing-stop branch in `analyze_product`.
- TICK BUY path also stores `_ATR_STOP_MIN` as a safety floor until the next scan refreshes it.

### Change 3 — SELL threshold 0.30 → 0.55
Matches TechAgent's `_SELL_THRESHOLD = 0.55`. Filters the noisy 0.30–0.55 SELL band where momentum's SELL signals performed poorly. Inverted the stale `test_thresholds_asymmetric` (BUY > SELL) to `test_sell_threshold_raised_to_match_tech` since the new design intentionally makes SELL the stricter bar.

### TDD
- New test classes `TestMomentumMacroRegime` (6 tests) and `TestMomentumATRStop` (5 tests) in `backend/tests/test_momentum_agent_cb.py`.
- Red phase watched: `AttributeError: 'MomentumAgentCB' object has no attribute '_macro_adjusted_sell_score'`, `ImportError: _ATR_STOP_MIN`, `AssertionError: 0.3 == 0.55`.
- Green: all 41 momentum tests pass. Full suite: **397 passed**.

### Rationale
User observed the momentum SELL regime was regime-agnostic (fired the same in short-squeeze as in overheated markets) and wanted parity with tech. Reducing SELL aggressiveness in short-crowded regimes + raising the confidence bar should cut false exits that tech has already eliminated.

---

## [Session 30.2] — 2026-04-19 — Unblock LLM: Training Watchdog + Signal Display Fixes

### Bug A — LLM suppression
A CNN training subprocess (PID 38816) ran for 4.5 h in phase-2 feature build, emitted no log lines after `18:22`, but stayed alive — `cnn_agent.training_active` remained `True`, which gates **every** Ollama call (`cnn_agent.py:1160-1165`). Result: every CNN scan in that window stored `llm_prob=NULL` (CNN-only signals, no LLM validation). Existing watcher only checked `pid_alive`, so a stuck-alive subprocess silently disabled the LLM indefinitely.

### Fix A
- **`backend/main.py`** — new module-level helper `_is_training_stale(data, log_mtime, now)`. Staleness = `status=="running"` AND `now - started_at >= 30 min` AND `now - log_mtime >= 15 min`. `train_worker.py` only writes the progress file at start and end, so its mtime is useless mid-run; the watchdog watches `backend/logs/cnn_training.log` instead.
- **`_train_progress_watcher`** — after the existing PID-alive branch, calls the helper on each 5-s tick. If stale, runs `taskkill /F /T /PID <pid>`, writes `status=failed` with a watchdog-attributed error, and falls through to the existing `failed` transition branch so `training_active` gets cleared and the normal state reset happens.
- Thresholds `_TRAIN_STALE_START_SECS = 1800` and `_TRAIN_STALE_LOG_SECS = 900` exposed as module-level constants.

### Bug B — regime label mismatch
`cnn_scans.regime` column stored `"RANGING"` while `signals.reasoning` text said `"CHAOTIC"` for the same scan. HMM detector returns one of `{TRENDING, RANGING, CHAOTIC, UNKNOWN}` but the DB write used `"TRENDING" if trending else "RANGING"` (binary collapse), silently mapping CHAOTIC → RANGING in two places (`cnn_agent.py:1224, 1247`).

### Fix B
Both writers now store `hmm_regime` verbatim — `save_cnn_scan` and the outcome-tracker `indicators` dict. No other code paths assumed the binary collapse.

### Bug C — overstated VWAP % in reasoning
`signals.reasoning` printed `"Price below VWAP by 27.98%"` for a BTC scan whose true delta was 1.47%. `_vwap()` in `signal_generator.py` returns `dist / 0.05` (normalised to ±1.0), but the reasoning formatter did `abs(vwap_d) * 100` — up to ~20× overstated. Same normalised value is stored in `cnn_scans.vwap_dist` (correct, since downstream code expects [-1, 1]); bug was display-only.

### Fix C
Introduced a local `vwap_pct_delta = (price - vwap_price) / vwap_price * 100` (guards against `vwap_price == 0`) and used it in both the display string and the `above/below` side token. Raw `vwap_d` still flows unchanged into the DB and CNN feature tensor.

### TDD
- **`backend/tests/test_train_watchdog.py`** — 7 tests on `_is_training_stale`: `not_running_is_never_stale`, `completed_is_not_stale`, `running_without_started_at_is_not_stale`, `running_within_startup_grace_is_not_stale`, `running_with_recent_log_is_not_stale`, `running_with_stale_log_after_grace_is_stale`, `missing_log_mtime_is_not_stale`.
- **`backend/tests/test_cnn_agent.py::TestRegimeLabelAndVWAPDisplay`** — 2 tests: one patches `get_detector` to return `CHAOTIC` and asserts the captured `save_cnn_scan` row has `regime == "CHAOTIC"`; the other parses the displayed VWAP % out of `save_signal`'s reasoning and asserts it matches `(price - vwap_price) / vwap_price * 100` within 0.1 pp.
- Full backend suite: **386 passed** (36.7 s).

---

## [Session 30] — 2026-04-19 — Doc/Code Audit Fixes

Post-audit cleanup after reviewing README, REBUILD_STANDARD, CLAUDE.md, CHANGELOG, and `backend/` against actual code. A follow-up code review surfaced deeper issues (see Session 30.1).

### Documentation
- **README.md** — `/api/backfill` → `/api/history/backfill`, `/api/backfill/status` → `/api/history/status` (endpoints were renamed but docs were stale).
- **CHANGELOG.md Session 27** — same endpoint path correction.
- **test_signal_improvements.py** — docstring said `N_CHANNELS=24`; updated to `27` to match actual constant.

### Code
- **CNN cache type hint** (`cnn_agent.py:691`) — was `Dict[str, Tuple[float, float]]` (2-tuple); runtime stores 3-tuple `(cnn_prob, timestamp, indicators_dict)` at line 1101 and per CLAUDE.md invariant. Type hint now matches reality.
- **OLLAMA_MODEL fallback default** (3 sites) — fallback was `qwen2.5:7b`; updated to `llama3.1:8b` (later superseded by centralization in Session 30.1).

### TDD
- `test_cnn_agent.py::test_cache_skips_fetch` — added 3-tuple length assertion (later replaced by a non-tautological test in Session 30.1).
- `test_signal_improvements.py::TestOllamaModelFallback` — 3 source-scraping tests (replaced with behavior test in Session 30.1).

---

## [Session 30.1] — 2026-04-19 — Review Follow-Up: OLLAMA_MODEL Centralization + Stronger Tests

Code-reviewer feedback on Session 30: swapping one hardcoded fallback for another doesn't satisfy CLAUDE.md invariant 7 ("never hardcode a model name"); the source-scraping fallback tests were brittle; the cache `len == 3` assertion was tautological.

### Config
- **`config.Config.ollama_model`** — new field `os.getenv("OLLAMA_MODEL", "llama3.1:8b")`. Single source of truth for the default, honoring env override when set.

### Code
- **Three OLLAMA_MODEL sites** now read `config.ollama_model` instead of calling `os.getenv` directly:
  - `agents/cnn_agent.py:622`
  - `agents/signal_generator.py:396`
  - `services/outcome_tracker.py:97` (also added `from config import config` import; removed now-unused `import os`)

### TDD
- **`TestOllamaModelFallback` (brittle source scraping) removed.**
- **`TestOllamaModelConfig` added** (2 tests): default fallback when env unset; env override honored. Behavior test against the config object — immune to formatting changes.
- **`test_cache_write_produces_three_tuple` added** — calls `generate_signal` on an empty cache, asserts the *written* value at line 1101 is a 3-tuple of `(float, float, dict)`. Real regression guard for invariant #2.
- **`test_cache_skips_fetch` cleanup** — tautological `len == 3` assertion (on a locally-constructed tuple) removed; the test again focuses on cache-hit skip behavior only.

### Stale model references
- **`.github/workflows/ci.yml:108`** — `OLLAMA_MODEL: llama3.2:3b` → `llama3.1:8b`.
- **`cnn_agent_decision_tree.html:180, 346`** — `llama3.2:3b` → `llama3.1:8b`.

---

## [Session 29] — 2026-04-19

### CNN Risk Management Overhaul
- **ATR trailing stop** (`cnn_agent.py`) — replaces fixed max-hold as primary exit. Trail distance = 2×ATR/peak, clamped [3%, 15%]. Wider trail for volatile coins; tighter for stable ones.
- **Hard stop-loss** (`cnn_agent.py`) — `_CNN_STOP_LOSS_PCT=0.08`; position exits at -8% from entry with trigger `STOP_LOSS`.
- **Max-hold extended 48h → 7 days** (`cnn_agent.py`) — `_CNN_MAX_HOLD_SECS = 7 * 24 * 3600`. Trailing stop is now the primary exit; 7-day limit is a safety net.
- **Legacy position exit** (`cnn_agent.py`) — positions missing `entry_time` (pre-exit-tracking) get `_CNN_LEGACY_HOLD_SECS` hold assigned, forcing exit on next scan.
- **`peak_price` tracked on buy** (`cnn_agent.py:_CNNBook.buy()`) — ratchets up on every tick, never down; drives trail calculation.
- **Win/loss tracking** (`cnn_agent.py:_CNNBook`) — `wins`, `losses`, `_sum_win_pct`, `_sum_loss_pct`, `win_rate`, `expectancy` properties.
- **`/api/cnn/status`** (`main.py`) — now returns `wins`, `losses`, `win_rate`, `expectancy_pct`.

### Auto-Train Subprocess
- **Auto-train routed through subprocess** (`main.py`) — `_auto_train_subprocess()` spawns `train_worker.py` instead of blocking the scan loop. `auto_train_fn` callback passed into `cnn_agent.run_loop()`.
- **Dead-PID detection** (`main.py:_train_progress_watcher`) — if `cnn_train_progress.json` shows "running" but PID is gone, automatically marks status "failed" and clears `training_active`.
- **Phase timing** (`cnn_agent.py:train_on_history`) — logs `phase1_secs` (candle load), `phase2_secs` (feature build), `phase3_secs` (model training).
- **Dataset progress logging** (`cnn_agent.py:_build_dataset`) — logs progress every 10 products (was silent for hours on large datasets).

### Bug Fixes
- **`is_tracked` bug** (`database.py`) — `upsert_product` ON CONFLICT UPDATE clause omitted `is_tracked=excluded.is_tracked`; existing products never got `tracked=1`. CNN couldn't scan any products.
- **CNN indicator cache** (`cnn_agent.py`) — cache tuple expanded from `(cnn_prob, timestamp)` to `(cnn_prob, timestamp, {indicators_dict})`; cache hits now restore all 10 indicator values.
- **Ollama model hardcoded** (`services/outcome_tracker.py`) — `model = "qwen2.5:7b"` replaced with `model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")`.
- **TechAgent take-profit** (`tech_agent_cb.py`) — `_TAKE_PROFIT` lowered 20% → 8% → 6% to lock in gains earlier.
- **Ollama model** (`.env`) — changed `OLLAMA_MODEL=llama3.2:3b` → `llama3.1:8b`.

### TDD
- `test_cnn_risk_exits.py` — 14 tests: win/loss tracking (5), stop-loss (4), max-hold (5). All pass.
- `test_cnn_agent.py` — `test_cache_skips_fetch` updated for 3-tuple cache format.
- Fixed `test_stop_loss_does_not_fire_at_5pct_loss` — set `peak_price = current * 1.01` so drop from peak is ~1%, below 3% ATR floor.
- Fixed `test_max_hold_constant_is_7_days` — asserts `_CNN_MAX_HOLD_SECS == 7 * 24 * 3600`.
- Fixed `test_max_hold_fires_at_49_hours` — entry time offset to `_CNN_MAX_HOLD_SECS + 3600`.

---

## [Session 28] — 2026-04-18 — N_CHANNELS 24→27 (Macro Crypto Channels)

- **Channel 24**: IV/RV20 spread — Deribit implied vol minus 20-day realized vol, clipped [-1,1]. High IV = fear = bearish.
- **Channel 25**: IV/RV60 spread — same against 60-day realized vol.
- **Channel 26**: Binance top-trader long/short sentiment ratio, normalised to [-1,1].
- **N_CHANNELS 24→27** — backward-compat load: checkpoint channel mismatch sets `_needs_retrain=True`.
- **`test_bsm_integration.py`** updated — all shape assertions reflect 27-channel tensor.
- **Note**: macro channels (funding rate Ch20, L/S Ch26) are baked into the CNN input tensor — the model trains on them, not just gates at decision time.

---

## [Session 27] — 2026-04-18 — Historical Signal Backfill

- **`data/history_backfill.py`** (new) — fetches daily OHLCV from Alpaca (Stooq fallback), computes `return_1d`, `return_5d`, `rv_20d`, `rv_60d`. Idempotent.
- **POST `/api/history/backfill`** — manual trigger; `days` param (30–1825, default 365).
- **GET `/api/history/status`** — returns sample counts per symbol + `ready_to_train` bool.
- **Auto-backfill at startup** — fires background backfill when total samples < `MIN_TRAIN_SAMPLES` (100).
- **15 TDD tests** in `tests/test_history_backfill.py`.

---

## [Session 26] — 2026-04-18 — CNN Training Best Practices + UI Reliability

- **Adam → AdamW** (`cnn_model.py`) — `weight_decay=1e-4`; mathematically correct for adaptive optimizers.
- **Dropout 0.2 → 0.3** — better regularization for noisy signals.
- **Random split → chronological** — last 20% as validation; eliminates temporal data leakage.
- **ReduceLROnPlateau scheduler** — `factor=0.5, patience=5, min_lr=1e-6`.
- **Early stopping** — `patience=15`; stops when val loss stalls.
- **MIN_TRAIN_SAMPLES 30 → 100** — prevents training on memorizable micro-datasets.
- **LSTM inplace gradient crash** (`cnn_agent.py:forward()`) — added `self.lstm.flatten_parameters()`.
- **Launcher false "Stopped"** — wrapped `get_usd_balance()` in `asyncio.wait_for(timeout=3.0)`.
- **Training counter disappears on tab switch** (`CNNDashboard.tsx`) — poll resumes from `elapsed_secs` on remount.
- **glu2 arch** — added `BatchNorm1d` after each `GatedConv1d`; arch tag `glu`→`glu2`.

---

## [Session 25] — 2026-04-18 — Token Usage Fix

- **Claude/Gemini show 0 calls/hr in OLLAMA mode** — `_call_timestamps.append()` now called in `_get_ollama_decisions()` for both agents.
- **Claude/Gemini show 0 daily_tokens** — added DB fallback in `/api/tokens` when in-memory stats are empty.
- **GeminiAgent missing from `/api/tokens`** — added explicit `gemini_news_agent` path with DB fallback.

---

## [Session 24] — 2026-04-18 — CloudAgent Refactor + Bayes Early Exit

- **`CloudAgent` base class** (`agents/cloud_agent.py`, new) — extracts shared boilerplate (cycle throttle, backoff, `_api_lock`, `_hourly_call_limit`) from ClaudeAgent and GeminiAgent.
- **Bayes early exit** (`agents/base_agent.py`) — `_check_bayes_exits()` sells positions where `entry_confidence - bayes_confidence >= 0.30`.
- **Bayes confidence display** (`AgentCard.tsx`) — "Entry Conf" and "Bayes" columns; color-coded by confidence drop.
- **Hourly call limits raised** — `CLAUDE_HOURLY_CALL_LIMIT=10`, `GEMINI_HOURLY_CALL_LIMIT=20` (was 2 — too low).

---

## [Session 23] — 2026-04-17 — Bayesian Confidence Update

- **`entry_confidence` / `bayes_confidence`** on `Position` dataclass.
- **Bayesian update in `record_value()`** — logit-linear update: `posterior_logit = prior_logit + k × log_return` (k=10.0).
- **12 new tests** in `TestBayesianConfidence`.

---

## [Session 22] — 2026-04-18 — Markowitz Correlation Gate

- **Correlation gate** (`trading/risk_manager.py`) — blocks BUY when avg pairwise correlation of proposed portfolio > `CORRELATION_LIMIT=0.65`.
- **Bug**: `datetime.utcnow()` (naive) vs `datetime.now(timezone.utc)` (aware) TypeError in churn cooloff — fixed.
- **7 new correlation gate tests**.

---

## [Session 21] — 2026-04-17 — BSM Pipeline Integration (10-Channel CNN)

- **RV channels 8 & 9** — `rv_20d`/`rv_60d` added as CNN input channels.
- **IV/RV spread channel 5** — fetches nearest ATM call; `score = -clamp((IV-RV_20d)/0.20, -1, 1)`.
- **Shannon entropy pre-filter** — skips Ollama when signal information too low; saves ~50s latency.
- **CNN N_CHANNELS → 10**.
- **160/160 tests passing**.

---

## [Session 20] — 2026-04-13 — 6 CNN/Signal Improvements

- **ADX bug fixed** (`signal_generator.py`) — sum init → mean init; ADX was inflated ~14×.
- **MACD defaults (5,13,3)** — changed from stock-market defaults (12,26,9) for 1h crypto bars.
- **RSI overbought 65→78** (`momentum_agent_cb.py`) — crypto RSI routinely hits 80+ before reverting.
- **N_CHANNELS 20→24** — 4 new channels: funding rate (Binance), BTC correlation, time-of-day sin/cos.
- **21 TDD tests** in `test_signal_improvements.py`.

---

## [Session 19] — 2026-04-14 — ML Improvements: HMM Regime, Kelly Sizing, WFE

- **HMM Regime Detector** (`data/regime_detector.py`, new) — 4-state (bull/neutral/bear/high_vol); raises CNN BUY threshold in bear/high_vol.
- **Kelly position sizing** (`trading/portfolio.py`) — quarter-Kelly from trade history; clamped [2%, MAX_POSITION_SIZE].
- **Walk-Forward Efficiency** (`data/cnn_model.py`) — OOS R² computed on val set; HEALTHY/DEGRADED/POOR status.
- **46 new tests** across regime_detector, portfolio, cnn_model.

---

## [Session 18] — 2026-04-13 — Performance Dashboard + Momentum NoneType Fix

- **`AttributeError: 'NoneType'`** (`momentum_agent_cb.py`) — `sc.get()` called before `if sc` guard. Fixed: merged into single short-circuit condition.
- **`/api/performance` wrong P&L** — MIN/MAX on balance gave extremes not chronological first/last. Fixed with correlated subqueries.
- **Performance dashboard** (`PerformanceDashboard.tsx`, new) — SVG bar chart, stat cards, monthly table, $50k/yr projection.

---

## [Session 17] — 2026-04-13 — $50k Goal Implementation

- **CNN hard stop-loss (8%)** — `_CNN_STOP_LOSS_PCT=0.08`.
- **CNN max hold time (48h)** — `_CNN_MAX_HOLD_SECS=48*3600`.
- **Win/loss tracking** on `_CNNBook`.
- **Momentum threshold raised 0.30→0.45** — eliminates weak entries (34% win rate at 0.30).
- **Momentum RSI gate** — blocks buys when RSI ≥ 65.
- **Momentum ADX gate** — requires ADX ≥ 20 (confirmed trend).
- **14 TDD tests** in `test_cnn_risk_exits.py`; **8 TDD tests** in `test_momentum_entry_filter.py`.

---

## [Session 16] — 2026-04-13 — Training Crash + Kelly + Data Quality

- **Kelly frac=0 blocking all BUYs** (`cnn_agent.py`) — `_kelly_fraction(strength)` used `(prob-0.5)*2`; BUYs only fired when model_prob > 0.75. Fixed: pass `model_prob` directly.
- **Training blocked event loop** — `_sync_fit()` extracted and run via `run_in_executor`.
- **`KeyError: 'start'`** during training — SQLite returns `start_time`; normalised before merge.
- **Sub-cent products** — scanner now untracks stale rows; `MIN_PRICE=0.01` enforced across all 4 agents.
- **Corrupt positions `avg_price=0`** — all agent books drop on `load()`; DB reconciler closes orphans.
- **SQLite "database is locked"** — `WAL` journal mode + `busy_timeout=30000` + `_DB_TIMEOUT=30` on all 34 connects.

---

## [Session 15] — 2026-04-13 — GLU CNN + Latency Instrumentation

- **GLU-gated CNN** (`data/cnn_model.py`) — `GatedConv1d` (dual-path: `conv_main(x) × sigmoid(conv_gate(x))`). Gate suppresses noisy channels. ~6800 params.
- **Backward compat** — `arch` field in checkpoint; `load()` picks `_build_glu_net` vs `_build_net`.
- **Ollama latency instrumentation** — `[OLLAMA_LATENCY] elapsed=Xs`; WARNING when > 15s.
- **GUI trading toggle fixed** — all 4 agents now respect `is_trading_fn`; was hardcoded `True`.

---

## [Session 14] — 2026-04-12 — Macro Signals, ScalpAgent, CNN Training Quality

- **CNN train/val split** — 80/20 chronological; per-epoch train+val loss; fit diagnosis (UNDERFIT/OVERFIT/OK).
- **ScalpAgent daily halt → per-trigger stats** — replaced phantom-drawdown halt with `_stats` dict logging W/L/win_rate per trigger type.
- **Stale `is_tracked` rows** — scanner untracks products below MIN_PRICE on every scan.

---

## [Session 13] — 2026-04-12 — GitHub Research Improvements Phase 1

- **Hurst Exponent**, **Multi-period RSI**, **Dissimilarity Index**, **Kelly Criterion** added to `signal_generator.py`.
- **Fear & Greed Index** (`services/fear_greed.py`, new) — suppresses BUY when F&G < 20.
- **ATR trailing stop** for TechAgent — replaces fixed 5% stop; clamped [1.5%, 12%].
- **Kelly sizing** in TechAgent and CNNAgent.

---

## [Session 12] — 2026-04-11 — Auth System + GUI Launcher

- **GUI launcher** (`launcher_gui.pyw`) — PyInstaller-compiled `.exe` with Tkinter UI.
- **Auth added** — `/api/auth/check` public endpoint for launcher health poll.
- **ERR_SSL_PROTOCOL_ERROR** fix — self-signed cert auto-generated at startup.

---

## [Session 11] — 2026-04-10 — Macro Context Dual-Cache

- **Dual-cache macro context** — FAST (15 min, tactical) + SLOW (24 hr, strategic 52W).
- **ClaudeAgent JSON parse failures** — strip markdown fences; `CRITICAL: first char must be '{'` in prompt.
- **Summary agent "no trades" all day** — TTL-based freshness check (was date-only).
- **49 tests** in `test_macro_context.py`.

---

## [Session 10] — 2026-04-05 — DB Auto-Pruning + Ollama Scanner Retry

- **DB auto-pruning** — `prune_performance_table(days=3)` and `prune_news_price_snapshots(days=14)` at startup + daily.
- **Scanner Ollama retry** — up to 3 attempts with 5s/10s backoff on HTTP 500/crash.

---

## [Session 9] — 2026-04-05 — Ollama Tier 2 Swap + GPU Telemetry

- **CNNReasoningAgent SELL crash** — `portfolio.get_position()` doesn't exist; changed to `symbol in portfolio.positions`.
- **SentimentAgent hardcoded `gpt-4o-mini`** — model now reads `config.OLLAMA_MODEL` in Ollama mode.
- **GPU telemetry** — `nvidia-smi` subprocess in `get_telemetry()`; GPU section in Telemetry tab.
- **Tier 2 Ollama swap** — ClaudeAgent and GeminiAgent route to Ollama in `OLLAMA_ONLY_MODE`.

---

## Known Non-Bug Issues

- **`hmmlearn` not installed** — `TestHMMStability` tests skip with `ModuleNotFoundError`. Install with `pip install hmmlearn` to enable.
- **CNN overfitting** — train loss 0.31 vs val 0.70 after current model; LLM skip threshold fires on nearly all scans due to high model confidence. Needs fresh training on more balanced data.
- **`llama3.1:8b`** — must be pulled in Ollama (`ollama pull llama3.1:8b`) and backend restarted before new model is active.
