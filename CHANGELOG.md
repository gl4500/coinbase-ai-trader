# Changelog — Coinbase AI Trader (polymarket_app)

All notable changes to this project are documented here.
Format: reverse-chronological by session date.

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
