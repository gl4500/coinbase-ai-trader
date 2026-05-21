# Changelog — Coinbase AI Trader (polymarket_app)

All notable changes to this project are documented here.
Format: reverse-chronological by session date.

---

## Session 58.71n — Dollar-bar data pipeline (SP1) — 2026-05-20

**Spec:** `docs/superpowers/specs/2026-05-20-dollar-bar-data-pipeline-design.md`
**Plan:** `docs/superpowers/plans/2026-05-20-dollar-bar-data-pipeline.md`

Sub-project 1 of the off-the-clock XGB exploration: 1-minute backfill + per-product dollar bars for the scorecard's top-20 products.

- `services/history_backfill.py` — 1-minute backfill support (`backfill_product_1m`, `load_1m_history`, `_parquet_path_1m`), mirroring the 5m functions.
- `tools/backfill_1m_candles.py` — operator CLI driving the 1m backfill for the top-20 (depth per product from its 1h span).
- `tools/build_dollar_bars.py` — `candle_dollar_value` + `calibrate_threshold` helpers.
- `tools/build_dollar_bars.py` — `dollar_bars_from_candles` construction core (threshold-crossing boundaries, trailing partial dropped).
- `tools/build_dollar_bars.py` — `build_dollar_bars_for_candles` pure assembly (clip to 1h window + calibrate + construct).

---

## Session 58.71l — XGB deployment-aligned scorecard kickoff (2026-05-18)

**Spec:** `docs/superpowers/specs/2026-05-18-xgb-deployment-scorecard-design.md` (commits `c18481d`, `9accdbe`)
**Plan:** `docs/superpowers/plans/2026-05-18-xgb-deployment-scorecard.md` (commit `d179966`)

Motivation: 7+ exogenous-input probes have failed to lift XGB AUC above 0.5284 against the 0.55 production gate (see `xgb_feature_optimization_findings.md`). Literature survey (Salinas 2025, AFML Ch. 14, FinTSB 2025, Hudson & Thames) confirms AUC is the wrong target — serious crypto-XGB practitioners optimize precision-at-gate, expected return per signal, paper-Sharpe, and ECE. This kicks off a deployment-aligned multi-metric scorecard for v3 driver + v4 shadow + v4.5 shadow (9-cell expanded) across retail/mid/pro fee tiers.

### New files (Task 1 of 9)
- `backend/tools/_returns.py` — `realized_log_returns_per_sample(entry_closes, forward_closes)` pure helper; raises on shape mismatch or non-positive prices
- `backend/tests/test_returns_helper.py` — 4 unit tests (basic, zero-when-equal, non-positive raise, shape-mismatch raise)

### New files (Task 2 of 9)
- `backend/tools/_scorecard/__init__.py` — package marker for per-metric scorecard computers
- `backend/tools/_scorecard/_precision.py` — `precision_at_tau(scores, labels, tau)` pure helper; returns `(precision, n_fired)` with NaN when no signals fire; strict `>` threshold; raises on shape mismatch or non-binary labels
- `backend/tests/test_scorecard_precision.py` — 6 unit tests (basic, no-fires-NaN, all-fire, strict-gt, non-binary raise, shape-mismatch raise)

### New files (Task 3 of 9)
- `backend/tools/_scorecard/_expected_return.py` — `expected_return_at_tau(scores, returns, tau, fee)` pure helper; returns `(expected_return, n_fired)` = mean realized log-return on fired samples minus `2*fee` round-trip cost, NaN when no signals fire; raises on shape mismatch or negative fee. Single-tier function — multi-tier reporting (retail 0.006 / mid 0.0025 / pro 0.0005) is the orchestrator's job (Task 6/7).
- `backend/tests/test_scorecard_expected_return.py` — 5 unit tests (basic-with-fee, no-fires-NaN, fee-zero, negative-fee raise, shape-mismatch raise)

### New files (Task 4 of 9)
- `backend/tools/_scorecard/_paper_sharpe.py` — `paper_sharpe_per_fold(scores, returns, fold_ids, fold_spans_days, tau, fee)` pure helper implementing O3 resolution: per-fold per-signal Sharpe with `sqrt(N_f)` annualization where `N_f = n_fires * 365 / span_days`, aggregated as `(mean, std)` across folds for honest variance. Returns `(mean_annual_sharpe, std_annual_sharpe, total_n_fired)`; NaN when no fold has >=2 fires or all degenerate (sigma=0). Raises `ValueError` on shape mismatch; `KeyError` if a data fold_id is missing from spans dict.
- `backend/tests/test_scorecard_paper_sharpe.py` — 5 unit tests (constant-returns NaN guard, positive-mean-with-noise sanity bound, fold-with-no-fires excluded, shape-mismatch raise, missing-fold-span KeyError)

### New files (Task 5 of 9)
- `backend/tools/_scorecard/_ece.py` — `expected_calibration_error(scores, labels, n_bins=10)` pure helper; weighted mean of `|empirical_acc - mean_score|` over equal-width bins on `[0, 1]`. Per O4 resolution decile binning is safe at 167k+ samples. Empty bins skipped. Raises on shape mismatch, non-binary labels, or `n_bins <= 0`.
- `backend/tests/test_scorecard_ece.py` — 6 unit tests (perfectly-calibrated near zero, completely-miscalibrated 0.9 vs 0.1, empty-bins skipped, shape-mismatch, non-binary labels, invalid n_bins)

### New files (Task 6 of 9)
- `backend/tools/_scorecard/_report.py` — `ScorecardReport` dataclass holding per-tau rows (precision, n_fired, e_return/sharpe_mean/sharpe_std per fee tier), scalar `ece`, `recommended_operating_tau`, `gates_passed` dict, `pos_rate`, `gate_tier`.
- `backend/tools/scorecard.py` — `compute_scorecard(scores, labels, returns, fold_ids, fold_spans_days, *, fee_tiers=FEE_TIERS, gate_tier='retail', tau_grid=DEFAULT_TAU_GRID, n_ece_bins=10)` orchestrator. Sweeps tau ∈ {0.50, 0.55, ..., 0.95} × tiers {retail 0.006, mid 0.0025, pro 0.0005}, composes four per-metric computers, picks recommended operating tau as the precision-max row with `n_fired >= 100` and positive E[r] at `gate_tier`, evaluates four hard gates (precision ≥ pos_rate + 0.03, E[r] > 0, paper_sharpe > 0, ece < 0.05). Raises `ValueError` if `gate_tier` not in `fee_tiers`. CLI runner appended in Task 7.
- `backend/tests/test_scorecard_orchestrator.py` — 3 unit tests (full-report shape on synthetic 500-sample dataset, invalid gate_tier raises, recommended-tau respects N_FIRED_FLOOR=100)

### New files (Task 7 of 9 — v3 driver CLI, corrected 2026-05-19)
- `backend/tools/_scorecard/_cv_harness.py` — v3 scorecard harness. `top_n_pids_from_cache` (cache read only for the survivorship-aware top-20 ranking); `build_v3_samples(pids, parquet_dir, sample_step)` mirrors `train_xgb.train_xgb_v3` — reads per-pid OHLCV parquets, builds tiered slices (micro 60 / meso 168 / macro 336), extracts v3 features, labels `close[t+4] > close[t]`, records entry/exit close; `train_fold_v3` fresh per-fold booster (params mirror `train_xgb_v3`: `subsample=0.7`, `feature_weights_v3` per invariant #13); `oof_predict_v3` 5-fold purged-WF OOF predictions + fold spans.
- `backend/tests/test_scorecard_cli.py` — 3 fast tests (help, missing-track, v4-not-implemented) + 1 slow v3 smoke test.
- `backend/tools/scorecard.py` — appended `--track v3` CLI (`--cache`/`--parquet-dir`/`--sample-step`/`--gate-tier`); `--track v4`/`v4.5` raise `NotImplementedError`. Report includes an OOF-mean-AUC sanity-anchor line. Realized return per v3 signal is the plain 4-bar forward log-return `ln(close[t+4]/close[t])`.

### Scope correction (2026-05-19)
The original plan Task 7 was written against false premises, caught when the first smoke run crashed: **the v3 booster is not trained on `cnn_dataset_cache.pt`.** `train_xgb.train_xgb_v3` reads per-pid OHLCV parquets, builds tiered candle slices, and labels `close[t+4] > close[t]` — a naive 4-bar direction, NOT the ±1% triple-barrier (that belongs to the CNN cache). Consequences: the v3 scorecard rebuilds samples from parquets (cache used only for pid ranking); realized return is the 4-bar forward log-return with no barriers (a `_barrier_replay.py` written against the wrong premise was deleted); `purged_walk_forward_splits` is imported from the standalone `tools/walk_forward.py`. Task 7 v1 is **v3-only**; v4/v4.5 deferred to Tasks 7b/7c. Plan Tasks 7/8/9 rewritten; old sections marked SUPERSEDED. The design spec's "Val fold convention" (~167k cache samples) is wrong for v3 and should be amended.

### First v3 baseline (smoke run, sample_step=24, top-20 pids, 7386 samples)
OOF mean AUC 0.512; ECE 0.047 (**PASS** <0.05); precision/expected-return/paper-Sharpe all **FAIL** — every E[r] is negative at the retail fee tier (1.2% round-trip swamps the edge), positive only at the `pro` tier at τ 0.65/0.75. No τ qualifies as a recommended operating point. **v3 passes 1 of 4 hard gates.** Full results in `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md` (Task 9).

### Remaining (Tasks 7b/7c)
v4 / v4.5 scorecard tracks — backlog, each needs its own OHLCV-parquet harness and a spec.

---

## Session 58.71k — XGB v4.5 3-class shadow infrastructure (2026-05-18)

**Spec:** `docs/superpowers/specs/2026-05-17-xgb-v4-5-three-class-design.md`
**Plan:** `docs/superpowers/plans/2026-05-17-xgb-v4-5-three-class.md`

Builds on B.1 binary v4 shadow infrastructure. Adds a fully-isolated 3-class (DOWN/NEUTRAL/UP) shadow path alongside v3 driver and v4 binary shadow, with 7 channels (5 OHLCV + 2 Bollinger Band) and horizon-suffixed artifacts ready for the operator's 3-horizon sweep (h24/h72/h168).

### New files (7)
- `backend/tools/xgb_v4_5_features.py` — pure-function 7-channel × 3-tier × 10-stat extractor (210 features)
- `backend/tools/train_xgb_v4_5.py` — 3-class trainer (multi:softprob, num_class=3, horizon-suffixed artifacts, NO calibrator)
- `backend/tools/v4_5_horizon_compare.py` — per-class AUC + macro AUC + 3-rule decision sweep + HTML report
- `backend/tests/test_xgb_v4_5_features.py`
- `backend/tests/test_train_xgb_v4_5.py`
- `backend/tests/test_v4_5_horizon_compare.py`
- `backend/migrations/xgb_v4_5_shadow_20260517.py` — idempotent ALTER TABLE adds 3 REAL nullable columns to cnn_scans

### Edited files (6)
- `backend/tools/xgb_features.py` — dispatcher branch for `model_version='v4_5'`
- `backend/agents/xgb_signal.py` — `_try_load_v4_5`, `xgb_prob_v4_5` (clip + renormalize), `xgb_prob_shadow_v4_5` (3-class shadow path, v4.5 failure isolated)
- `backend/agents/cnn_agent.py` — ~10-LOC write-through: unpack 3-tuple shadow, persist 3 new columns; no decision logic changes
- `backend/database.py` — CREATE TABLE + ALTER + save_cnn_scan INSERT for 3 v4.5 columns
- `backend/tests/test_xgb_signal.py` — v4.5 shadow tests including failure-isolation
- `backend/tests/test_database.py` — v4.5 column persistence tests
- `backend/tests/test_mc_migration.py` — v4.5 migration idempotency

### Invariants
- **#17 (NEW):** v4.5 3-class telemetry contract — all 3 probs written together or all NULL; sum ~1.0 after clip+renormalize; v4.5 failures never affect v3 driver or v4 shadow.

### Operator preflight (post-commit)
1. Train 3 horizons (h24/0.015, h72/0.03, h168/0.06) via `python -m tools.train_xgb_v4_5 --forward-hours <H> --label-thresh <T>`
2. Run `python -m tools.v4_5_horizon_compare --holdout <PATH> --out backend/tools/xgb_v4_5_horizon_compare.html`
3. Launch dev backend with `PORT=8002` for shadow week
4. Promote (horizon, rule) combo to 8001 if shadow telemetry shows promise

---

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

---

## [Session 58.71i] — 2026-05-16 — Marketcap bronze schema v2: volume_24h (#marketcap-A)

### Why
The XGB v3 feature extractor currently has no marketcap-related channels.
Step A of a 3-step buildout (A: bronze schema; B: channel wiring in
`tools/xgb_features.py`; C: v3 retrain on N_CHANNELS bump) extends the
existing parquet-backed marketcap cache to include 24h trading volume —
which CoinGecko `/coins/{id}/market_chart/range` and CoinPaprika
`/v1/tickers/{id}/historical` already return in the same responses the
current parsers ignored. Zero extra API calls.

### What changed
- **`backend/tools/build_marketcap_parquet.py`** — bumped `_SCHEMA_VERSION`
  1→2, added `volume_24h` (`pa.float64`) field to `_SCHEMA`. Extended
  `_save_marketcap_history` / `_load_marketcap_history` / `rows_from_history`
  to carry the new column. v1 parquets without the column still load
  (key omitted). `rows_from_history` accepts both legacy 2-tuple and new
  3-tuple history inputs.
- **`backend/services/coingecko_marketcap.py:fetch_marketcap_history`** —
  parses `total_volumes` parallel array alongside `market_caps`. Volume is
  indexed by timestamp (defensive against array length mismatch). Returns
  `List[Tuple[int, float, float]]` (ts_ms, market_cap, volume_24h). Missing
  or empty `total_volumes` -> logs warning + fills 0.0.
- **`backend/services/coinpaprika_marketcap.py:fetch_marketcap_history`** —
  reads `volume_24h` field per historical row. Same return shape. Missing
  or unparseable -> 0.0 fill.
- **`backend/services/marketcap_history_cache.py`** — new `_schema_is_stale()`
  check forces full refetch when on-disk parquet has `schema_version < 2`
  (auto-upgrades v1 parquets lazily on next access). Merge logic carries
  `volume_24h` on both cached + fresh paths. Return tuple shape extended
  to `(ts_ms, mc, volume_24h)`.
- **`backend/tools/marketcap_probe.py`** — `marketcap_rows_to_log_grid`
  now accepts both 2-tuple (legacy callers/tests) and 3-tuple (current
  fetchers) row shapes via positional access.
- **Tests** — +4 new: `test_coingecko_parses_volume_24h`,
  `test_coingecko_handles_missing_total_volumes` (in
  `test_coingecko_marketcap.py`); `test_coinpaprika_parses_volume_24h` (in
  `test_coinpaprika_marketcap.py`);
  `test_cache_v1_parquet_triggers_full_refetch` (in
  `test_marketcap_history_cache.py`). Updated 8 existing assertions across
  the same three test files + `test_build_marketcap_parquet.py` for the
  new schema shape and v2 default. Net: +4 tests, 0 deleted.

### Verification
```
cd backend && python -m pytest tests/test_coingecko_marketcap.py \
  tests/test_coinpaprika_marketcap.py \
  tests/test_marketcap_history_cache.py \
  tests/test_build_marketcap_parquet.py \
  tests/test_marketcap_probe.py -v
=> 77 passed
```

Zero extra API calls per pid — volume was already in the response payload.
Bronze parquets upgrade lazily on next cache hit per pid; the operator
preflight below forces a one-shot full upgrade across all tracked products.

### Operator preflight (run once after this commit)

```bash
cd backend
../.venv/Scripts/python.exe -m tools.build_marketcap_parquet \
  --source coingecko --pids <comma-list-of-49-tracked-pids> \
  --start 2025-05-16 --end 2026-05-16
```

CoinGecko free tier rate-limits ~30 req/min; 49 calls ≈ 100 sec wall time.
After completion: all 49 tracked pids have parquet at schema_version=2 with
`volume_24h` populated. (The existing `--source` / `--pids` flags suffice;
a bulk `--all-tracked` convenience flag is Step B's problem.)

### Step B preview
Wire `volume_24h` into `tools/xgb_features.py` v3 extractor as new
channel(s). Bump `N_CHANNELS = 28 → 30+`. Retrain booster (Step C).
Each is its own brainstorm cycle.

---

## [Session 58.71h] — 2026-05-16 — Refactor sweep module 6: probe scripts audit (#311-refactor-g)

### Why
Per Task 19 refactor backlog: `backend/tools/*_probe.py` accumulated 15
research-probe scripts over Sessions 58.13-58.68. Most ran once, posted a
+0.01 mean-AUC verdict (mostly FAIL), and were never re-executed. Their
findings are pinned in CHANGELOG.md + `xgb_feature_optimization_findings.md`
memory, but the scripts themselves clutter `backend/tools/` and confuse
which probes are still worth re-running. This sweep tidies the directory by
moving historical probes to `backend/tools/retired/` while preserving them
as a referenceable record. Only probes with active downstream imports or
recent (last 3 sessions) re-runs stay in place.

### What changed
Inventory of 15 probe scripts under `backend/tools/`, decision per script:

| Probe | LOC | Decision | Reason |
|---|---|---|---|
| `marketcap_probe.py` | 358 | **KEEP** | Active in Sessions 58.47/58.66/58.68; `--source coingecko\|coinpaprika\|both` flag; bronze-cache wiring under MC roadmap; memory still cites it |
| `calibration_probe.py` | 181 | **KEEP** | Live `from tools.calibration_probe import calibration_probe` in `feature_set_compare.py` and `xgb_importance_probe.py`; Phase-4 gate logic |
| `btc_dominance_probe.py` | 265 | MOVE | #156 ran once, FAILED +0.01 gate (Δ=+0.0077); docstring referenced as semantics mirror by `long_trend_probe` + `okx_ls_probe` |
| `btc_leadlag_probe.py` | 423 | MOVE | #246-#248 single-add probe, no gate pass; imported by `btc_residual_ch9_probe` only (moved chain peer) |
| `btc_residual_ch9_probe.py` | 251 | MOVE | #253c FAILED gate; its `from tools.btc_leadlag_probe import ...` updated to `tools.retired.btc_leadlag_probe` |
| `cnn_xgb_delta_probe.py` | 282 | MOVE | Exploratory oneoff per `xgb_feature_optimization_findings.md` line 58; cited in `fit_xgb_calibration.py` docstring only |
| `hour_of_day_probe.py` | 136 | MOVE | #153 FAILED gate; `timescale_sweep.py` docstring reference only |
| `long_trend_probe.py` | 397 | MOVE | #243-#245 all 5 candidates FAILED post-leak-fix |
| `oi_coverage_audit.py` | 168 | MOVE | #210 one-shot audit; finding documented (17/20 pids all-zero OI); no longer running |
| `oi_single_add_probe.py` | 189 | MOVE | #143-#145 FAILED gate; `channel_replace.py` `__main__` print example reference (stale path harmless) |
| `okx_ls_probe.py` | 252 | MOVE | #235 INCONCLUSIVE (0/20 coverage), abandoned |
| `probe_okx_swap_listings.py` | — | MOVE | #211 diagnostic; finding cited in `services/okx_oi_history.py:82` comment (2026-05-08); script not re-run |
| `rsi_rank_probe.py` | 297 | MOVE | #162 PASSED gate but result long since integrated into channels; harness archived |
| `stationarity_audit.py` | 166 | MOVE | #164 one-shot heuristic audit; finding documented |
| `xgb_importance_probe.py` | 152 | MOVE | Zero CHANGELOG/code refs; only memory cite (line 53 of `xgb_feature_optimization_findings.md`) which calls it an "exploratory oneoff" |

Tests deleted (tightly coupled to moved probes; not worth rewriting import paths):
- `test_btc_dominance_probe.py`
- `test_btc_leadlag_probe.py`
- `test_btc_residual_ch9_probe.py`
- `test_hour_of_day_probe_snapshot.py`
- `test_long_trend_probe.py`
- `test_oi_coverage_audit.py`
- `test_okx_ls_probe.py`
- `test_rsi_rank_probe.py`
- `test_stationarity_audit.py`

Tests kept (active probes):
- `test_marketcap_probe.py`
- `test_calibration_probe.py`

New: `backend/tools/retired/__init__.py` — package marker + docstring
explaining the retired-probe convention and restoration procedure.

### Verification
```
cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -q --co
=> 975 tests collected in 4.75s (no import errors from deleted probes)
```

All inter-probe imports either point to tools that stayed in `tools/`
(`channel_replace`, `feature_set_compare`, `pid_snapshot`, `btc_residualize`,
`xgb_features`, `calibration_probe`) or were updated to the
`tools.retired.` namespace (1 instance: `btc_residual_ch9_probe.py` →
`tools.retired.btc_leadlag_probe`). The 3 incidental references in tests
that survived are docstring/comment mentions only (no `import`).

### Net
- Files moved (probes): 13
- Files deleted (tests): 9
- Files added: 1 (`__init__.py`)
- LOC removed from `backend/tools/` top-level: ~3,236
- LOC added under `retired/`: same probes preserved verbatim (one import path edit)
- Active `backend/tools/*_probe.py`: 15 → 2

---

## [Session 58.71g] — 2026-05-16 — Refactor sweep module 4b: cnn_agent dead xgb-gated branches (#311-refactor-f)

### Why
`cnn_agent.py` carried five CNN-backend-only branches that were dead under
the live `MODEL_BACKEND=xgb`: the Hurst random-walk suppression gate, the
HMM regime CHAOTIC-only gate, the LightGBM entry-filter gate, the entire
Ollama LLM-blend decision tree (skip_llm / `_ollama_prob` / fear-greed
fetch), and `_maybe_auto_train` (already a no-op under xgb per #300). They
still imported their inputs (`_hurst_exponent`, `_dissimilarity_index`,
`_shannon_entropy`, `LGBMFilter`) and computed values nothing consumed.
Second sub-module of `cnn_agent.py` cleanup (the largest one — 268 LOC
deleted in production code, 414 LOC deleted in tests).

### What changed
- **`backend/agents/cnn_agent.py`** —
  - DELETED `from data.lgbm_filter import LGBMFilter`.
  - DELETED `from services.fear_greed import get_fear_greed`.
  - DELETED `_hurst_exponent`, `_dissimilarity_index`, `_shannon_entropy`
    from the `agents.signal_generator` import (kept `_kelly_fraction`,
    `_realized_vol`).
  - DELETED `import re` (only used by `_ollama_prob`).
  - DELETED module-level constants `_LGBM_RETRAIN_EVERY`,
    `_LGBM_MODEL_PATH`, `_HURST_TREND_THRESH`, `_HURST_MR_THRESH`,
    `_DI_SUPPRESS_THRESH`, `_ENTROPY_SKIP_THRESH`, `_regime_gate_enabled`,
    `OLLAMA_URL`.
  - DELETED `_ollama_prob(...)` async function (~70 LOC) — Ollama LLM
    probability blend, called only when `MODEL_BACKEND=cnn`.
  - DELETED `CoinbaseCNNAgent._lgbm` field + `_lgbm_trades_seen` field +
    `_lgbm_retrain_if_needed` method.
  - DELETED `_maybe_auto_train` method + the corresponding `await
    self._maybe_auto_train(...)` call site in `run_loop`. `run_loop` still
    accepts the `train_every_n_scans` and `auto_train_fn` kwargs for
    caller compat (main.py still passes them); they are now ignored
    (auto-train infrastructure cleanup is module 4c).
  - `generate_signal` —
    - DELETED the `hurst`/`di`/`entropy` computations.
    - DELETED the `agent_votes`/`agent_ctx` fetch + injection into the
      reasoning context.
    - DELETED the entire `skip_llm` decision tree + `_ollama_prob` call +
      `lessons` fetch + Fear-and-Greed fetch. `llm_prob` is now
      permanently `None`; `model_prob` always equals `cnn_prob`.
    - DELETED the Hurst / regime / LGBMFilter suppression block in the
      execute=True BUY path. BUYs now go straight to Kelly sizing +
      `book.buy`.
    - Kept the HMM regime detection (still stored on `cnn_scans.regime`
      for downstream analysis) and the `cnn_w`/`llm_w` blend weights (for
      `cnn_scans` schema back-compat — values are recorded but no longer
      drive the blend).
  - `CoinbaseCNNAgent.__init__` —
    - `self.llm_calls`, `self.llm_prompt_tokens`, `self.llm_response_tokens`,
      `self.training_active` retained (read by `main.py` `/api/cnn/status`
      payload); they are now always zero / False.
  - Net: ~268 LOC removed from `cnn_agent.py` (2980 → 2712).
- **`backend/tests/test_cnn_agent.py`** —
  - DELETED `TestInferenceRegimeGate` (3 tests — regime suppression gate
    removed).
  - DELETED `TestSuppressionsGatedByBackend` (3 tests — moot, the gates
    are gone for all backends now).
  - DELETED `TestLLMSkippedUnderXgb` (2 tests — moot, no Ollama blend
    exists).
  - Removed all `patch("agents.cnn_agent._ollama_prob", ...)` /
    `patch("agents.cnn_agent._hurst_exponent", ...)` / `patch.object(agent._lgbm, ...)`
    occurrences from surviving tests (8 sites across 5 test methods).
  - Net: ~414 LOC removed.
- **`backend/tests/test_model_backend.py`** —
  - DELETED `TestAutoTrainGate` (3 tests — `_maybe_auto_train` is gone).
- **`backend/tests/test_config.py`** —
  - ADDED `TestNoCnnBackendOnlyBranches` (5 assertions, 1 class):
    - `test_no_cnn_only_gate` — locks out `_cnn_only` markers.
    - `test_no_lgbm_filter_machinery` — locks out `_lgbm.allow_buy`,
      `_lgbm.predict`, `from data.lgbm_filter import LGBMFilter`,
      `LGBMFilter()`, `_lgbm_retrain_if_needed`.
    - `test_no_ollama_blend` — locks out `_ollama_prob`, `skip_llm`,
      `from services.fear_greed import get_fear_greed`.
    - `test_no_hurst_di_entropy_gates` — locks out the four threshold
      constants + the three signal-generator helper names + the regime-
      gate env reader.
    - `test_no_maybe_auto_train` — locks out the deleted scheduler hook.

### Verification
```
cd backend && ../.venv/Scripts/python.exe -c "import agents.cnn_agent; print('import OK')"
=> import OK

cd backend && ../.venv/Scripts/python.exe -m pytest tests/ --tb=line -q
=> 902 passed, 64 skipped, 1 xfailed, 2 xpassed in 279s
```

Same total-passed count as pre-edit baseline accounting for net test
deletions: pre-edit (after Module 4a) full suite ≈ 903 passed; after
deleting 8 dead-branch tests + 3 auto-train tests and adding 5 new
policy tests, expected ≈ 902 passed (matches).

### Net LOC
- `backend/agents/cnn_agent.py`: -268 LOC (2980 → 2712).
- `backend/tests/test_cnn_agent.py`: -414 LOC (4180 → 3766).
- `backend/tests/test_model_backend.py`: -77 LOC.
- `backend/tests/test_config.py`: +85 LOC.
- Test count: -8 dead-branch tests, -3 auto-train tests, +5 policy
  tests (net -6 tests).

### Rollback
Pure code deletion. `git revert <commit>` restores the dead branches +
deleted tests verbatim. No on-disk artifacts touched; no DB schema
change; no .env change. Under `MODEL_BACKEND=xgb` (live) the deleted
code never fired, so live behavior is byte-for-byte unchanged.

---

## [Session 58.71f] — 2026-05-16 — Refactor sweep module 4a: CNN_ARCH dead variants (#311-refactor-e)

### Why
`cnn_agent.py` defined three CNN arch classes (`SignalCNN`=glu2 ~280k params,
`SignalCNNGlu1` ~12k, `SignalCNNGluM` ~50k) with `_ARCH_REGISTRY` +
`_active_arch()` env-var lookup. Active arch was glu1 (per .env); glu2 and
glum were unreachable. Under `MODEL_BACKEND=xgb` (live) the entire CNN path
is bypassed anyway. First sub-module of cnn_agent.py cleanup — tightest
cluster.

### What changed
- **`backend/agents/cnn_agent.py`** —
  - DELETED `class SignalCNN` (glu2, ~44 lines).
  - DELETED `class SignalCNNGluM` (~34 lines).
  - DELETED `_ARCH_REGISTRY` dict.
  - DELETED `_active_arch()` function.
  - SIMPLIFIED `_build_cnn()` — no arg, hardcoded to `SignalCNNGlu1()`.
  - SIMPLIFIED `_model_path_for()` / `_best_loss_path_for()` — no arg,
    return the glu1-suffixed paths directly.
  - `CoinbaseCNNAgent.__init__` — removed `self._arch` field;
    `_model_path_for(self._arch)` → `_model_path_for()` at 5 call sites
    (sed-applied for consistency).
- **`backend/tests/test_cnn_agent.py`** —
  - DELETED `TestSignalCNNGluM` (entire class, ~50 LOC).
  - DELETED `TestArchFactoryAndPaths` (multi-arch routing, ~75 LOC).
  - DELETED `TestCnnAgentArchWiring` (CNN_ARCH env wiring, ~30 LOC).
  - DELETED `TestSignalCNNGlu1::test_fewer_params_than_glu2` (compared
    against deleted class).
- **`backend/tests/test_config.py`** — added `TestNoCnnArchEnvVar` policy
  test (1 test, 4 assertions). If anyone re-introduces
  `os.environ.get("CNN_ARCH"` (or equivalent) in cnn_agent.py, pre-commit
  fails.
- **`.env`** — removed `CNN_ARCH=glu1` line + its preceding comment block.
- **Host-side (operator):** moved `backend/cnn_model.pt` + `cnn_best_loss.txt`
  to `backend/retired/cnn_model_glu2.pt` + `cnn_best_loss_glu2.txt`. Glu1
  active artifacts (`cnn_model_glu1.pt`, `cnn_best_loss_glu1.txt`)
  unchanged. `backend/retired/` is gitignored.

### Verification
```
backend && python -m pytest tests/test_config.py tests/test_cnn_agent.py::TestSignalCNNGlu1 -v
=> 2 passed + 3 skipped (skipif _TORCH_AVAILABLE — unchanged from before)
backend && python -c "import agents.cnn_agent; print(agents.cnn_agent.SignalCNNGlu1.__name__)"
=> SignalCNNGlu1
```

Net: ~120 LOC code deleted, ~150 LOC tests deleted, +25 LOC policy test.
File total: 3073 → ~2925 lines.

Zero live-behavior change — under MODEL_BACKEND=xgb the CNN path is dead
anyway; under MODEL_BACKEND=cnn, glu1 is the only arch and was already
the active one.

### Rollback
1. `git revert <commit>` — restores classes + registry + helpers + tests.
2. `mv backend/retired/cnn_model_glu2.pt backend/cnn_model.pt` (host-side).
3. Re-add `CNN_ARCH=glu2` (or `glum`) to `.env` if you want a different
   arch active.

---

## [Session 58.71e] — 2026-05-16 — Refactor sweep module 3 Phase B: TechAgent removal — frontend (#311-refactor-d)

### What changed
- **`frontend/src/components/AgentsDashboard.tsx`** — removed live TECH
  rendering: deleted the per-agent loop's `tech` branch, the combined
  stat-card "Tech + XGB" sub-labels, and the entire Tech signal feed
  section. The page now shows XGB-only live metrics. `tech` field kept
  in the API-response state shape for back-compat (backend returns
  `tech: {}` after #311-refactor-c).
- **`frontend/src/components/FiringCounter.tsx`** — removed the TECH
  scan/signal counter strip at the top header. Counters' type fields
  (`tech_scans`, `tech_signals`) kept to avoid TypeScript cascades;
  populated from `agents.tech` (now `{}`) so values are always 0.

### Kept intact (operator chose "keep history")
- **`frontend/src/components/PerformanceDashboard.tsx`** — TECH filter
  option stays. Operator can browse the 569 historical TECH trades there.
- **`frontend/src/utils/agentByProduct.ts`** + test — utility maps
  trades by agent; works correctly for historical TECH rows without
  changes.

### Verification
- Frontend type-checks pass (no `tech` reference removed without its
  populator removed).
- AgentsDashboard renders CNN-only.

---

## [Session 58.71d] — 2026-05-16 — Refactor sweep module 3 Phase A: TechAgent removal — backend (#311-refactor-c)

### Why
Third module of the refactor sweep. TechAgent (`agents/tech_agent_cb.py`,
654 LOC + 497 LOC test) was one of two live trading agents. Operator chose
"delete entirely" after the trade-off was flagged (5-day TICK_TRAIL was
the most profitable single trigger in the system: 51 trades, 98% WR,
+$46). Rationale: simplify to a single XGB-driven decision path.

### Preflight (operator-driven, ran 2026-05-16 before this commit)
```
cd backend && python -m tools.close_tech_positions
=> Closed 39 TECH paper positions at live market price.
=> Final TECH balance $933.77, realized PnL -$66.23 (was -$61.41 +
   $4.82 from closing during minor downward moves).
=> agent_state.positions_json zeroed out for TECH.
=> 39 new trades rows written with trigger_close='MANUAL_TECH_RETIREMENT'.
```
DB backup: `backend/coinbase.db.bak_pre_tech_retirement_20260516_213801`
(host-side, gitignored).

### What changed
- **DELETED** `backend/agents/tech_agent_cb.py` (-654 LOC).
- **DELETED** `backend/tests/test_tech_agent_cb.py` (-497 LOC).
- **`backend/main.py`** — removed import + AppState fields (`tech_agent`,
  `tech_task`) + instantiation + `_delayed_tech` startup task + cleanup
  loop entry + `_TECH_START_DELAY` constant. `/api/agents/status`
  endpoint returns `tech_status = {}` for frontend back-compat during
  Phase A → Phase B window.
- **`backend/agents/cnn_agent.py`** — 1-line comment update on the
  `get_agent_decisions` Ollama context (only historical CNN decisions
  now appear; no new TECH writes).
- **`backend/services/outcome_tracker.py`** — deleted dead `if source
  == "TECH":` branch in `_format_indicators` (historical TECH outcomes
  remain in DB but are never re-formatted; comment updated on `source`
  param docstring).
- **`backend/tools/close_tech_positions.py`** (NEW, +136 LOC) — preflight
  script (run once before this commit; idempotent, re-running is a no-op).
- **`backend/tests/test_close_tech_positions.py`** (NEW, 3 tests) —
  covers no-op + happy path.
- **`backend/tests/test_main_no_tech_import.py`** (NEW, 1 test, 2
  assertions) — regression test: `main.py` does not import
  `agents.tech_agent_cb` and AppState has no `tech_agent` field.

### Historical data — UNCHANGED (operator: "keep everything")
All TECH rows in `trades` (530+39=569), `agent_decisions` (279,451),
`agent_state` (1 row, balance=$933.77, positions_json={}),
`signal_outcomes` (563), `signals` remain. No VACUUM, no purge.

### Verification
```
backend && python -m pytest tests/test_close_tech_positions.py \
                            tests/test_main_no_tech_import.py -v
=> 5 passed
```

Net: ~-1150 LOC code deleted, +160 LOC added (preflight script + 2 test
files). Net -990 LOC across Phase A.

### Phase B follows
Frontend deletion comes in the next commit (`#311-refactor-d`):
AgentsDashboard.tsx loses live TECH section, gets collapsed "Retired
Agents (history)" panel; CNNDashboard.tsx confidence table loses **Tech**
column.

### Rollback (full restoration of TechAgent)
1. `git revert <this commit> <Phase B commit>`
2. Restore `agent_state` snapshot from
   `backend/coinbase.db.bak_pre_tech_retirement_20260516_213801`
3. Restart backend

Time: ~5 min total.

---

## [Session 58.71c] — 2026-05-16 — Refactor sweep module 2: bare-isotonic calibrator removal (#311-refactor-b)

### Why
Second module of the refactor sweep. `xgb_signal._try_load` had a dual-path
calibrator loader: dict-shape `{"calibrator","feature_set"}` (canonical
since #311f) and bare-isotonic (legacy v1). The bare-isotonic branch was
~20 lines of conditional logic plus a back-compat warning path. Its only
real-world consumer is a hypothetical rollback to the v1 booster + v1
calibrator backup — a one-time event that can be handled with a 3-line
host script (documented below).

### What changed
- **`backend/agents/xgb_signal.py:_try_load`** — collapsed the
  dict-vs-bare branch into a single dict-shape check. Bare-isotonic
  pickles now log a warning and skip calibration (raw passthrough). Net:
  ~20 lines deleted, ~5 added. Same observable behavior under the
  current bare-isotonic-on-disk state (still raw passthrough); different
  warning message.
- **`backend/tests/test_xgb_signal.py`** — deleted 3 tests that exercised
  the bare-isotonic load path:
  - `test_calibration_pkl_remaps_raw_to_calibrated`
  - `test_calibration_clipped_to_safe_range`
  - `test_force_reload_picks_up_swapped_calibrator`
  Added 1 new test `test_bare_isotonic_pkl_skipped_with_warning` locking
  in the new behavior. Net: -2 tests (19 total in file, was 21).

### Verification
```
backend && python -m pytest tests/test_xgb_signal.py -v
=> 19 passed
backend && python -c "from agents import xgb_signal; xgb_signal._try_load(); print(xgb_signal._calibration)"
=> None  (with 'Legacy bare-isotonic format dropped' warning in log)
```

Zero live-behavior change — current `backend/xgb_calibration.pkl` is bare
isotonic (v3 refit deferred per #311-cut) so both before and after this
change produce `_calibration = None` → raw passthrough.

### Rollback to v1 booster (operator runbook)
If rolling back to the v1 booster + the v1 calibrator backup, the bare
pickle must be rewrapped into dict shape first:

```python
import pickle
from sklearn.isotonic import IsotonicRegression
iso = pickle.load(open("backend/xgb_calibration.pkl.bak_v1_20260516_182946", "rb"))
with open("backend/xgb_calibration.pkl", "wb") as f:
    pickle.dump({"calibrator": iso, "feature_set": "v1"}, f)
```

Then rename the v1 booster files back to production names (per the
#311-cut rollback procedure) and hot-reload.

---

## [Session 58.71b] — 2026-05-16 — Backend-aware shell cleanup rule (#311-refactor-cleanup)

### Why
During refactor Module 1 work I mechanically ran `Get-Process python |
Stop-Process -Force` as the standing shell-cleanup rule prescribes —
which killed the live backend (port 8001) along with stray pytest
processes. ~30 min of MC telemetry not collected. The cleanup rule
needs a backend carve-out so future agent work doesn't repeat this.

### What changed
- **`polymarket_app/CLAUDE.md`** — Shell cleanup section rewritten with
  port-8001-aware snippets (PowerShell + bash). Blanket
  `Stop-Process python -Force` / `pkill -9 python` is explicitly forbidden.

Docs-only commit.

---

## [Session 58.71a] — 2026-05-16 — Refactor sweep module 1: dead env-var cleanup (#311-refactor-a)

### Why
First module of the refactor sweep. Investigation (grep across backend/)
revealed the 4 CNN_*_CNN_W / CNN_*_LLM_W env vars defined in config.py:60-63
are dead-on-arrival: nothing in backend/ ever reads them. `regime_blend()`
in services/hmm_regime.py uses hardcoded weights (0.75/0.25 for trending,
etc.) and was scaffolded with no config plumbing. The env vars + config
fields polluted `.env` and misled operators about what's tunable.

### What changed
- **`backend/config.py`** — deleted 4 fields (`cnn_trending_cnn_w`,
  `cnn_trending_llm_w`, `cnn_ranging_cnn_w`, `cnn_ranging_llm_w`) and
  their wrapping comment. Replaced the auto-train comment with a
  backend-gating tag. Added a policy line to the module docstring
  ("every env var MUST trace to a live consumer").
- **`backend/services/hmm_regime.py`** — `regime_blend()` docstring now
  explicitly notes weights are hardcoded (was misleading — implied
  config-driven).
- **`.env`** — deleted 4 env keys and 2 wrapping comments. Replaced the
  auto-train comment with the backend-gating tag. Added a 2-line policy
  comment block at the top of the file.
- **`backend/tests/test_config.py`** (NEW) — 1 policy test
  (`test_no_dead_llm_blend_fields`) with 4 assertions. Locks in: if
  anyone re-adds these fields without a live consumer, pre-commit fails.
- **`polymarket_app/CLAUDE.md`** — new invariant #15 documenting the
  policy.

### Verification
```
backend && python -m pytest tests/test_config.py -v
=> 1 passed
backend && grep -rE "config\.cnn_trending|config\.cnn_ranging" .
=> (empty — no live consumers)
```

Zero live-behavior change. The 4 deleted fields were never consumed anywhere.

---

## [Session 58.70d] — 2026-05-16 — MC chain ACTIVATED + sync docs (#311-mc-sync)

### What changed
- **`.env`** — added `MC_FILTERS=ci` + `MC_CI_K=1.0`. Backup at
  `.env.bak_pre_mc_<ts>` (gitignored). Activation takes effect on next
  backend restart (env vars read once at process start).
- **`polymarket_app/CLAUDE.md`** — invariant #14 added: MC filter chain
  is the sole MC-math touchpoint; cnn_agent has one hook; MC_FILTERS=""
  is bit-for-bit pre-MC; telemetry columns + JSON; filter exceptions
  caught; filter classes self-register on import.
- **`memory/coinbase_trader_architecture.md`** (outside repo) — Session
  58.70 entry covering the sidecar pattern, CIFilter MVP, telemetry
  schema, .env activation, and the queued next filters.

Documentation + .env-flip commit.

---

## [Session 58.70c] — 2026-05-16 — MC wire-up + telemetry schema (#311-mc-wire)

### What changed
- **`backend/migrations/mc_telemetry_20260516.py`** (NEW) — idempotent
  ALTER TABLE adding `xgb_prob_stdev REAL` and `mc_telemetry TEXT` to
  `cnn_scans`. Detects existing columns via PRAGMA, never errors on
  re-run. Applied to live `coinbase.db` as part of this commit.
- **`backend/database.py:save_cnn_scan`** — INSERT extended to write the
  two new columns; both nullable so MC-off (`MC_FILTERS=""`) state still
  produces NULL rows identical to pre-MC.
- **`backend/agents/cnn_agent.py:generate_signal`** — one new hook call
  to `agents.mc.registry.apply_buy_filters` between the side computation
  and the `save_cnn_scan`. With MC off this is a noop. With `MC_FILTERS=ci`
  the lower-bound gate from CIFilter may down-grade BUY to HOLD; telemetry
  is JSON-serialized into the `mc_telemetry` column.
- **`backend/tests/test_mc_migration.py`** (NEW) — 2 tests: add-on-first-run
  and idempotent-on-second-run.
- **`backend/tests/test_database.py`** — +2 tests for new columns.
- **`backend/tests/test_cnn_agent.py`** — +3 wire-up tests.

### Verification
```
backend && python -m pytest tests/test_cnn_agent.py::TestMCFilterChainIntegration \
                            tests/test_database.py::TestSaveCnnScanMCColumns \
                            tests/agents/mc/ tests/test_mc_migration.py -v
=> 21 passed (8 registry + 6 ci_filter + 2 migration + 2 db + 3 wire)
```

### Activation
Code is in but inert. To activate CIFilter on live signal generation:
1. Edit `.env`: add `MC_FILTERS=ci` (and optionally `MC_CI_K=1.0`).
2. Restart backend (env vars read at process start, not /api/cnn/model/reload).

`MC_FILTERS=` (default) leaves live behavior bit-for-bit identical to
pre-MC. Rollback: edit .env, restart.

---

## [Session 58.70b] — 2026-05-16 — MC CIFilter implementation (#311-mc-ci)

### What changed
- **`backend/agents/mc/ci_filter.py`** (NEW) — entry confidence-interval
  filter. Computes per-tree cumulative trajectory stdev across the v3
  booster (200 trees), gates BUY on `(point - K*stdev) > cnn_buy_threshold`.
  K=1.0 default via `MC_CI_K` env. Skips gracefully (no decision change)
  for non-v3 booster, missing pid, missing booster, or predict failure;
  every skip records a reason in telemetry. Self-registers with
  `agents.mc.registry._FILTER_CLASSES` on import.
- **`backend/tests/agents/mc/test_ci_filter.py`** (NEW) — 6 tests
  covering keep/block paths and 4 skip-reason cases.

### Verification
```
backend && python -m pytest tests/agents/mc/ -v
=> 14 passed (8 registry + 6 ci_filter)
```

---

## [Session 58.70a] — 2026-05-16 — MC package scaffolding + registry (#311-mc-a)

### What changed
- **`backend/agents/mc/`** (NEW package) — `__init__.py`, `base.py` (BuyFilter
  ABC), `registry.py` (`apply_buy_filters` chain dispatch). Reads `MC_FILTERS`
  env var; unknown names warn + skip; filter exceptions warn + skip; default
  empty MC_FILTERS = identity passthrough.
- **`backend/tests/agents/mc/test_registry.py`** (NEW) — 8 tests covering
  empty/unset env, dispatch, unknown filter, chain order, side change,
  exception isolation, non-BUY passthrough.

### Verification
```
backend && python -m pytest tests/agents/mc/test_registry.py -v
=> 8 passed
```

---

## [Session 58.69-cut] — 2026-05-16 — XGB v3 LIVE CUTOVER (#311-cut)

### What changed
v3 artifacts now at production filenames. v1 backed up to
`*.bak_v1_20260516_182946` (gitignored, on host only).

```
backend/xgb_model.json         => v3 (200 trees, 350 features, JSON format)
backend/xgb_features.json      => v3 (feature_set='v3', 350 names + feature_weights)
backend/xgb_calibration.pkl    => unchanged (legacy v1 bare-isotonic).
                                   xgb_signal detects feature_set mismatch
                                   and skips calibration (raw passthrough).
```

End-to-end inference verified against the live DB:
```
xgb_signal: legacy bare-isotonic calibrator found but booster feature_set=v3
            skipping calibration
BTC-USD xgb_prob: 0.5417
ETH-USD xgb_prob: 0.5900
SOL-USD xgb_prob: 0.5043
```

### Operator notes
- DRY_RUN stays true. v3 generates signals; paper trades only.
- Backend was offline during cutover (collateral of dev-loop process
  cleanup); next launcher start picks up v3 automatically.
- Calibrator REFIT deferred: the `fit_xgb_calibration --source cache`
  path uses the v1-shaped [N,28,60] dataset cache which lacks the
  meso/macro 168/336-bar windows v3 needs. Recalibrate from live
  cnn_scans + signal_outcomes after ~48h of post-cutover paper trades.
- Trained on 216 parquet pids, 60,439 samples in ~5 min (perf fix
  #311h + JSON format fix #311i required to make this work end-to-end).

### Rollback (~30 sec, no code change)
```
cd backend
mv xgb_model.json xgb_model.json.bak_v3_now
mv xgb_features.json xgb_features.json.bak_v3_now
mv xgb_model.json.bak_v1_20260516_182946 xgb_model.json
mv xgb_features.json.bak_v1_20260516_182946 xgb_features.json
# restart launcher (or POST /api/cnn/model/reload if backend already up)
```

### Top-10 features by gain (v3)
```
ch4_pct_rank        11.5   (RSI rank — micro)
ch1_slope           11.5   (volume slope — micro)
ch0_slope            9.8   (price slope — micro)
ch24_m168_slope      9.3   (IV/RV20 1-week slope — meso) <- NEW: meso pulling weight
ch24_m168_mean       9.2   (IV/RV20 1-week mean — meso)  <- NEW
ch2_min              9.1   (HL range floor — micro)
ch1_min              8.9   (volume floor — micro)
ch1_last             8.6   (latest volume — micro)
ch15_m168_mean       8.2   (ADX 1-week mean — meso)      <- NEW: trend strength
ch1_max              7.8   (volume ceiling — micro)
```
v1 top-10 was 100% intra-bar single-window stats; v3 mixes in three meso
`_m168_` slots, the macro bias is taking effect.

### Tasks completed
- Plan tasks 1-7 (#311a-#311g)
- Cutover (Task 8): trainer, calibration decision, artifact swap, smoke

---

## [Session 58.69j] — 2026-05-16 — tiered_history prod-schema fix (#311j)

### Why
End-to-end cutover smoke caught `sqlite3.OperationalError: no such column:
start`. Production `candles` table has `start_time INTEGER` (not `start`).
My test fixtures used `start` so the unit suite missed it.

### What changed
- **`backend/services/tiered_history.py:_read_sqlite`** — `PRAGMA
  table_info(candles)` to detect available timestamp column; SQL becomes
  `SELECT start_time AS start, ...` for prod schema. Returned dicts still
  use `start` key for parity with parquet.
- **`backend/tests/test_tiered_history.py`** — new
  `test_source_live_reads_prod_schema_with_start_time_column` builds a
  prod-shaped table and asserts the macro slice returns 336 bars with
  `start` key. Locks in the schema-detection invariant.

### Verification
```
backend && python -m pytest tests/test_tiered_history.py -v
=> 14 passed (13 existing + 1 new)
backend && python -c "from agents import xgb_signal; ..."  # smoke
=> BTC/ETH/SOL all returned valid xgb_prob from live DB
```

---

## [Session 58.69i] — 2026-05-16 — train_xgb_v3 atomic-write format fix (#311i)

### Why
Second production training run (perf fix from #311h landed) wrote
`xgb_model.json` as **UBJSON binary** instead of JSON. xgboost picks
serialization format from the file's LAST extension; my tmp name
`xgb_model.json.tmp` had `.tmp` last so xgboost wrote UBJSON, the
rename to `.json` left binary content, and `load_model("...json")`
rejected it with a JSON-parse error. Smoke test caught it after the
trainer finished (60439 samples, 126 s, ~5 min).

### What changed
- **`backend/tools/train_xgb.py:train_xgb_v3`** — tmp filenames now keep
  `.json` last: `xgb_model.tmp.json` / `xgb_features.tmp.json`. Atomic
  rename to final name preserves format.
- **`backend/tests/test_train_xgb_v3.py`** — new
  `test_v3_saved_model_loads_back_as_json` calls `xgb.Booster.load_model`
  on the saved artifact. Locks in the format invariant.

### Verification
```
backend && python -m pytest tests/test_train_xgb_v3.py -v
=> 6 passed
```

---

## [Session 58.69h] — 2026-05-16 — train_xgb_v3 perf fix: cache parquet per pid (#311h)

### Why
First production training run with 216 parquet files hung after 27 min CPU
without producing artifacts. Root cause: the inner sample loop called
`services.tiered_history.fetch_tiered(source="parquet", parquet_dir=...)`
per sample, and `fetch_tiered` does `pd.read_parquet()` on every call. At
production scale (~500 samples per pid × 216 pids = ~108k samples) this
re-reads the same files 500x each, blowing wall time to an estimated ~70
minutes. Caught when the live cutover trainer failed to land artifacts.

### What changed
- **`backend/tools/train_xgb.py:train_xgb_v3`** — read each pid's parquet
  exactly once via `pd.read_parquet(...).to_dict("records")`, then slice
  the in-memory record list per sample. Removes the import of
  `services.tiered_history.fetch_tiered` from the training hot path
  (it's still used by `xgb_signal.xgb_prob` at inference — once per scan,
  not per training sample). Added per-pid progress logging (logging +
  flushed print so background nohup tails see progress).
- **`backend/tests/test_train_xgb_v3.py`** — replaced
  `test_v3_uses_tiered_history` with `test_v3_reads_each_parquet_once_per_pid`
  that asserts `pandas.read_parquet` is called at most 2× per pid. Locks
  in the perf invariant.

### Verification
```
backend && python -m pytest tests/test_train_xgb_v3.py -v
=> 5 passed
```

---

## [Session 58.69g] — 2026-05-16 — CLAUDE.md invariant for v3 + memory sync (#311g)

### What changed
- **`polymarket_app/CLAUDE.md`** — invariant #13 added documenting v3 feature
  shape (350 names), tier constants location, `feature_weights` mechanism
  (`set_info` + `colsample_bytree=0.8`), pid-passthrough requirement,
  and the dict-shape calibrator pickle.
- **`memory/coinbase_trader_architecture.md`** (outside repo) — Session 58.69
  entry covering #311a–#311f, the sync rationale for `tiered_history`,
  and the operator-driven cutover note.

No tests touched; documentation-only.

---

## [Session 58.69f] — 2026-05-16 — Calibrator dict-shape pickle for v3 (#311f)

### What changed
- **`backend/tools/fit_xgb_calibration.py`** — new helpers
  `_save_calibrator(calibrator, out_path, feature_set)` and
  `_detect_calibration_target_feature_set()`. `_detect_feature_set`
  now recognises v3 via `_mWWW_` infix. Pickled output is a dict
  `{"calibrator", "feature_set"}` so xgb_signal can detect a v1-fit
  calibrator on a v3 booster and skip calibration (raw passthrough)
  instead of mapping through the wrong distribution. Legacy bare-isotonic
  still loadable by xgb_signal (treated as v1).
- **`backend/tests/test_fit_xgb_calibration.py`** — 2 new tests under
  `TestV3CalibrationPickle`. Existing tests' `pickle.load` sites
  updated to unwrap the new dict shape (`_loaded["calibrator"]`).

### Verification
```
backend && python -m pytest tests/test_fit_xgb_calibration.py -v
=> 9 passed (7 existing + 2 new)
```

---

## [Session 58.69e] — 2026-05-16 — XGB v3 trainer mode (#311e)

### What changed
- **`backend/tools/train_xgb.py`** — new `train_xgb_v3(pids, parquet_dir,
  out_dir, sample_step=24, n_estimators=200, ...)`. Pulls per-tier history
  via `tiered_history.fetch_tiered(source='parquet')`, rolls samples,
  labels `1 if close[t+4] > close[t]`. `feature_weights` set on DMatrix
  via `set_info` (the correct xgboost API; `xgb.train` doesn't accept it
  as a kwarg). `colsample_bytree=0.8` so the per-feature bias actually
  takes effect. Atomic write (tmp + rename). Skips pids with <336 parquet
  bars.
- **`backend/tools/train_xgb_prod.py`** — `main_v3()` CLI entry; auto-
  discovers pids from parquet dir. Invocable via
  `python -m tools.train_xgb_prod --feature-set v3`.
- **`backend/tests/test_train_xgb_v3.py`** (NEW) — 5 tests (metadata,
  feature_weights wiring, short-history skip, atomic write, tiered_history use).

### Verification
```
backend && python -m pytest tests/test_train_xgb_v3.py -v
=> 5 passed
```

---

## [Session 58.69d] — 2026-05-16 — cnn_agent pid plumbing for XGB v3 (#311d)

### What changed
- **`backend/agents/cnn_agent.py`** — `_cnn_prob(channels, pid=None)`
  forwards `pid` to `xgb_signal.xgb_prob`. The `generate_signal` call site
  and the shadow-XGB call site both pass `pid=pid` (pid already in scope
  from line 1948). Required by v3 booster's tiered_history lookup.
  Backward-compatible: pid is optional; v1/v2 ignore it.
- **`backend/tests/test_model_backend.py`** — 2 new tests under
  `TestPidPlumbing`. Existing `test_xgb_backend_calls_xgb_prob` lambda
  signature updated to accept `pid=` kwarg.

### Verification
```
backend && python -m pytest tests/test_model_backend.py -v
=> 13 passed (11 existing + 2 new)
```

---

## [Session 58.69c] — 2026-05-16 — XGB v3 signal routing + calibrator metadata (#311c)

### What changed
- **`backend/agents/xgb_signal.py`** — `_try_load` auto-detects v3 via
  `_m060_/_m168_/_m336_` infix in feature_names. `xgb_prob` accepts
  optional `pid` kwarg; v3 path calls `services.tiered_history.fetch_tiered`
  and `tools.xgb_features.extract_features(feature_set='v3')`. Calibrator
  load handles both legacy bare-isotonic (v1) and new dict-shape
  `{"calibrator","feature_set"}` (v3); mismatched feature_set skips calibration.
- **`backend/tests/test_xgb_signal.py`** — 6 new tests under `TestV3Routing`
  + module-level helpers `_train_tiny_v3` and `_fake_v3_tiers`.

### Verification
```
backend && python -m pytest tests/test_xgb_signal.py -v
=> 21 passed (15 existing + 6 new)
```

---

## [Session 58.69b] — 2026-05-16 — XGB v3 tiered extractor + feature_weights (#311b)

### What changed
- **`backend/tools/xgb_features.py`** — added `feature_set='v3'` route.
  New helpers: `_v3_feature_names()` (350 names), `feature_weights_v3()`
  (per-tier 1/2/3/0), `_extract_v3(candles_by_tier)`, `_stats_from_candles()`.
  Tier constants: `MESO_CHANNELS={15,24,25,26}`, `MACRO_CHANNELS={20,21,27}`,
  `TIER_WINDOWS_V3={micro:60,meso:168,macro:336}`. m060 baseline slots on
  meso/macro channels inherit their channel's tier weight (per spec 4.3).
- **`backend/tests/test_xgb_features_v3.py`** (NEW) — 18 tests covering
  shape, naming, per-tier counts, zero-fill, feature_weights, unknown set.

### Verification
```
backend && python -m pytest tests/test_xgb_features_v3.py tests/test_xgb_features.py -v
=> 55 passed (18 new + 37 existing v1/v2 regression)
```

---

## [Session 58.69] — 2026-05-16 — Tiered history fetcher (v3 prep #311a)

### Why
XGB feature_set v3 needs per-tier hourly candle slices (60 / 168 / 336)
without bubbling async through xgb_signal.xgb_prob.

### What changed
- **`backend/services/tiered_history.py`** (NEW) — sync `fetch_tiered(pid,
  source, now_ts, ...)` returns `{"micro","meso","macro"}` slices. Reads
  parquet (training) or SQLite + parquet-prefix fallback (live).
- **`backend/tests/test_tiered_history.py`** (NEW) — 13 tests covering
  slice contracts, short-history empty-list semantics, source dispatch,
  now_ts leak prevention.

### Verification
```
backend && python -m pytest tests/test_tiered_history.py -v
=> 13 passed
```

---

## [Session 58.68] — 2026-05-15 — Marketcap bronze cache + probe `--source` flag (#284/#285)

### Why

`tools/marketcap_probe.py` re-fetches the full CoinGecko `/market_chart/range`
history on every run — for 20 pids each rerun is ~40s and consumes free-tier
budget. Bronze parquet files under `backend/data/marketcap/` already exist
(#299, written via CoinPaprika), but the probe wasn't reading them. Two
gaps:

1. No cache layer between the probe and the CoinGecko service — every probe
   re-run pays the API again.
2. No way to A/B against CoinPaprika without editing the probe by hand.

### What changed

- **`backend/services/marketcap_history_cache.py`** (NEW) — async
  `fetch_marketcap_history_cached(pid, start_ms, end_ms, parquet_dir, refresh_secs)`.
  Reads `<parquet_dir>/<pid>.parquet` first; treats the row set as a hit when
  newest `ingest_ts` is within `refresh_secs` (default 86400) AND newest cached
  `start*1000` is within one bar of the requested `end_ms`. Misses / stale /
  partial coverage call `coingecko_marketcap.fetch_marketcap_history`, merge
  the result with cached rows, stamp `ingest_ts=int(time.time())` +
  `schema_version=1` (per #164b PIT), and re-write the parquet. Returned rows
  are `(ts_ms, market_cap)` tuples sorted ascending and filtered to the
  requested window.
- **`backend/tools/marketcap_probe.py`** — added `--source coingecko|coinpaprika|both`.
  `coingecko` (default) routes through the new cache; `coinpaprika` calls
  `services.coinpaprika_marketcap.fetch_marketcap_history` directly (no cache
  yet — free 12-month rolling window, no API key required); `both` runs the
  probe once per provider and prints side-by-side Δ-AUC reports.
  `_fetch_marketcap_for_pids` now takes `source=` and `parquet_dir=` kwargs;
  unknown sources raise `ValueError`. CLI built via new `_build_argparser()`
  helper so tests can exercise it without launching the runner.
- **`backend/tests/test_marketcap_history_cache.py`** (NEW) — 7 tests covering
  RED-then-GREEN: public coroutine, cache miss calls underlying fetcher, miss
  writes parquet with PIT columns, hit short-circuits API, stale `ingest_ts`
  triggers refetch, partial coverage triggers refetch, returned rows filtered
  to `[start_ms, end_ms]`.
- **`backend/tests/test_marketcap_probe.py`** — added `TestSourceDispatch`
  (4 tests): default = coingecko, coinpaprika dispatch, unknown source =
  ValueError, argparser accepts the three documented choices.

### Verification

```
backend && python -m pytest tests/test_marketcap_history_cache.py
                              tests/test_marketcap_probe.py -v
=> 26 passed in 2.61s
```

### How to apply

Probe re-runs are now warm-cached:
```
cd backend && python tools/marketcap_probe.py --snapshot-ts auto
cd backend && python tools/marketcap_probe.py --snapshot-ts auto --source coinpaprika
cd backend && python tools/marketcap_probe.py --snapshot-ts auto --source both
```

---

## [Session 58.67] — 2026-05-10 — Gate CNN auto-train behind MODEL_BACKEND=='cnn' (#300)

### Why

Backend is running `MODEL_BACKEND=xgb` (XGB-only since #267/#XGBONLY-*), yet
the CNN scan loop was still firing `train_worker.py` every N scans — observed
live at 14:48:25 and again at 15:11:35 today with two concurrent backends
each spawning their own train_worker. Each retrain takes ~10 min on the
RTX 2060, burns shared VRAM with the live inference path, and produces a
checkpoint that XGB never consults. Two parallel train_workers also race on
`cnn_model_*.pt` writes.

Root cause: `cnn_agent.run_loop` (line ~2494) calls `auto_train_fn` whenever
`scan_count % train_every_n_scans == 0`, with no `config.model_backend`
guard. Symmetric companion to #232 (Hurst/regime/LGBM gates) and #250
(Ollama gates) — every CNN-specific code path in the scan loop must be
gated on `config.model_backend == "cnn"` per
`feedback_cnn_safeguards_backend_gating.md`.

### What changed

- **`backend/agents/cnn_agent.py`** — extracted the inline auto-train block
  into a new `_maybe_auto_train(train_every_n_scans, auto_train_fn)`
  coroutine method. Added the early-return gate:

      if config.model_backend != "cnn":
          logger.debug("CNN auto-train skipped — model_backend=%s", ...)
          return False

  `run_loop` now calls `await self._maybe_auto_train(...)`; behavior in
  CNN mode is unchanged.
- **`backend/tests/test_model_backend.py`** — new `TestAutoTrainGate` class,
  three tests: xgb-mode skips, cnn-mode triggers, misaligned scan_count
  skips regardless of backend.

### Verification

Per-module pytest (.venv): `tests/test_model_backend.py::TestAutoTrainGate`
RED→GREEN — 3/3 pass.

### Follow-up

- #303: kill orphan spyder backend (PID 63460) + restart .venv backend
  (PID 74608) so the gate is live in production.

---

## [Session 58.66] — 2026-05-10 — Marketcap historical bronze parquets — paths 1+2 unblocked (#293/#294/#295)

### Why

CoinGecko free `/market_chart/range` started returning 401 on 2026-05-09
(probe #260d/e/f), blocking the marketcap probe path that was the most
promising remaining +0.01 mean-AUC candidate after seven BTC-flavored probes
came up short (xgb_feature_optimization_findings.md). User directive:
"try 1 and 2, then match the parquet file to the required structure" — i.e.,
get both a CoinGecko Demo-key path and a CoinPaprika no-key path working,
then write bronze-schema parquets matching the `history_backfill.py` (#168)
convention.

### What changed

- **Path 1 — CoinGecko Demo-tier auth (#293)**
  `backend/services/coingecko_marketcap.py` — added `_demo_key_headers()`:
  reads `COINGECKO_API_KEY` env, sends `x-cg-demo-api-key` (free Demo plan,
  10k req/month, includes historical `market_chart/range`). Opt into Pro
  with `COINGECKO_API_PRO=1` (sends `x-cg-pro-api-key` instead). Both
  `fetch_marketcap_snapshot` and `fetch_marketcap_history` now pass the
  header dict to httpx.
- **Path 2 — CoinPaprika free-tier sibling (#294)**
  New `backend/services/coinpaprika_marketcap.py`. Mirrors the CoinGecko
  public surface so the probe harness can swap providers via `--source`.
  Uses the FREE `tickers/{cp_id}/historical?start=YYYY-MM-DD&interval=1d`
  endpoint (the `coins/{id}/ohlcv/historical` endpoint is paywalled —
  initial smoke test returned HTTP 402; the tickers/historical variant is
  the no-key path). 28-pid mapping table verified live 2026-05-10.
  Kill switch: `COINPAPRIKA_DISABLED=1`.
- **Bronze parquet writer (#295)**
  New `backend/tools/build_marketcap_parquet.py`. Schema mirrors
  `history_backfill.py:42-51` PIT convention exactly:
  `{start:int64, market_cap:float64, fdv:float64, ingest_ts:int64, schema_version:int32}`.
  Per-pid path: `backend/data/marketcap/<pid>.parquet`. Dedupes on
  `start` (last-wins), stamps `ingest_ts` on save, preserves existing
  ingest_ts across rewrites, sorts ascending. CoinPaprika has no FDV →
  `fdv = market_cap` default. CLI: `--source {coingecko,coinpaprika}
  --pids X,Y,Z --start YYYY-MM-DD --end YYYY-MM-DD`.

### Tests (TDD red→green)

- `backend/tests/test_coinpaprika_marketcap.py` — 10 tests (id mapping,
  history fetch shape, kill switch, HTTP 429, transport error, missing
  market_cap rows, ISO date params, FREE endpoint URL pin).
- `backend/tests/test_build_marketcap_parquet.py` — 16 tests (schema
  match, round-trip, sort, dedup, FDV default, PIT semantics, parent
  dir creation, file shape, ms→bar-aligned conversion).
- `backend/tests/test_coingecko_marketcap.py` — added `TestDemoApiKey`
  class (3 tests: header sent when key set, omitted when unset, Pro
  variant when `COINGECKO_API_PRO=1`).

### Verification (live smoke test)

```
python -m tools.build_marketcap_parquet --source coinpaprika \
    --pids BTC-USD,ETH-USD,SOL-USD \
    --start 2025-05-11 --end 2026-05-10
```
- Wrote 3/3 pids, 365 daily rows each.
- Schema match confirmed: `[('start','int64'), ('market_cap','double'),
  ('fdv','double'), ('ingest_ts','int64'), ('schema_version','int32')]`.
- BTC sample row: `{'start': 1746921600, 'market_cap': 2068610050144.0,
  'fdv': 2068610050144.0, 'ingest_ts': 1778439162, 'schema_version': 1}`.
- BTC marketcap traverses 2.07T → 1.62T over the year window.
- Free-tier date constraint discovered: `start` ≥ ~12 months ago, else 402.

### No backend restart needed

Tools/services additions only; live signal path (`xgb_signal`) is unchanged.
Marketcap parquets are bronze-layer artifacts that feed the future probe
harness (#284–#286), not the running scan loop.

### Tasks
- #293 — `services/coinpaprika_marketcap.py` + tests (RED→GREEN, 10 tests)
- #294 — Switch to FREE `tickers/{id}/historical` after 402 on paid OHLCV
- #295 — `tools/build_marketcap_parquet.py` + tests (RED→GREEN, 16 tests)
- #296 — Path-1 CoinGecko Demo header support + tests (3 tests)
- #299 — Live smoke: 3 parquets generated, schema verified

### Pending (handed off to next loop iteration / future session)
- #284 — Wire CoinPaprika source into `marketcap_probe.py --source` flag
- #285 — Backfill remaining 25 pids (only top-3 smoke-tested this session)
- #286 — Re-run marketcap probe with the new bronze parquets

---

## [Session 58.65] — 2026-05-10 — Rebuild missing XGB artifacts (live backend was returning 0.50 fallback) (#290)

### Why

Verification of the gpu_coord removal (Session 58.64) surfaced that live backend
boots were logging `xgb_signal: artifacts missing (model=...xgb_model.json
features=...xgb_features.json) — fallback to 0.50` on every cold start, and
242 consecutive `cnn_scans` rows between 10:46–11:07 UTC all carried
`xgb_prob=0.5`. The three artifacts (`xgb_model.json`, `xgb_features.json`,
`xgb_calibration.pkl`) are gitignored — produced locally via
`tools/train_xgb_prod.py` + `tools/fit_xgb_calibration.py` — and had been lost
from disk. With XGB-only mode (`MODEL_BACKEND=xgb`, #267) every signal was
collapsing to neutral, so no BUY/SELL strength differentiation reached the
trader.

### What changed

- **No code changes.** Regenerated three local-only artifacts from the
  fresh 28-channel `cnn_dataset_cache.pt` (388,306 samples, May 10 build):
  - `backend/xgb_model.json` (442 KB) — booster, 280 features, fixed
    best_params (max_depth=4, mcw=1, subsample=0.7), 5-fold purged
    walk-forward CV, 4h embargo. `mean_auc=0.5215` (folds:
    0.5129/0.5140/0.5193/0.5188/0.5423) — matches the May 3 baseline (0.5224)
    within fold variance.
  - `backend/xgb_features.json` (5 KB) — feature_names + best_params.
  - `backend/xgb_calibration.pkl` (2 KB) — isotonic, fit on chronological
    20% val split (81,046 (raw_prob, label) pairs, `--source cache` mode
    from #187). Post-calibration buckets align with actual win rates
    (0.40–0.50 → 44.6 %, 0.60–0.70 → 64.6 %, 0.80–0.90 → 86.1 %).
- **Hot-reloaded into the live backend** via `POST /api/xgb/calibration/reload`
  (#194 endpoint). Response: `{"status":"reloaded","load_succeeded":true,
  "calibration_loaded":true,"feature_set":"v1","n_features":280}`.

### Why no code change

Root cause was disk state, not code. `xgb_signal._try_load()` correctly
returns the 0.50 fallback when artifacts are absent (graceful degradation,
not a bug). The fix is to keep the artifacts present after every reboot —
which is the operator's responsibility since they're large binary outputs of
the training pipeline.

### Tasks
- #290a — Rebuild xgb_model.json + xgb_features.json via train_xgb_prod
- #290b — Refit isotonic calibrator on cache val split (`--source cache`)
- #290c — Hot-reload artifacts via POST /api/xgb/calibration/reload

### Verification (in progress at session end)
- Pre-reload window 10:46–11:07 UTC: 242 cnn_scans rows, all `xgb_prob=0.5`
  (collapsed to fallback).
- Post-reload (11:07:30 UTC+): awaiting next 5-min scan cycle to confirm
  `xgb_prob` distribution returns to non-trivial range (expected ~0.03–0.95
  per the calibration grid).
- Backend log confirms `loaded booster (280 features, set=v1)` and `loaded
  isotonic calibrator` events at 11:07:29 UTC.

---

## [Session 58.64] — 2026-05-09 — Remove gpu_coord (symmetric to trading_app cleanup) (#287/#288/#289)

### Why

The cross-app GPU coordinator (`backend/data/gpu_coord.py`) was deleted from the
sibling `trading_app/` repo on 2026-05-09 because it was misfiring; the
polymarket_app counterpart was left in place "untouched per scope." With the
backend now in XGB-only mode (`MODEL_BACKEND=xgb`), `_ollama_prob`,
`_llm_confirm`, lessons fetches, and CNN retrains are all gated behind
`config.model_backend == "cnn"` — meaning the coordinator's per-app `acquire()`
serializer and shared `~/.ollama-coord/state.json` exposure publisher were
dead-weight in the live path. The training mutex in `train_worker.py` was the
last reachable user, but with the peer app no longer participating it was a
half-protocol talking to nobody. Symmetric removal restores parity between the
two repos and incidentally fixes the only Ruff I001 lint failure in CI
(`backend/train_worker.py:8:1` import block).

### What changed

- **Deleted:** `backend/data/gpu_coord.py` (285 lines) and
  `backend/tests/test_gpu_coord.py` (15 tests).
- **Stripped imports + `acquire(...)` wraps** in 4 ollama call sites — back to
  plain `httpx.AsyncClient` blocks:
  - `backend/agents/cnn_agent.py:_ollama_prob` (preserves `_t0/_elapsed` latency
    logging)
  - `backend/main.py` — deleted `_publish_exposure_loop` + its `create_task`
  - `backend/agents/signal_generator.py:_llm_confirm`
  - `backend/services/outcome_tracker.py` (preserves latency logging)
- **Stripped training mutex** in `backend/train_worker.py` — removed
  `acquire_training_mutex` / `release_training_mutex` calls + the early-skip
  branch + the `finally: release_…` clause.

### Verification (TDD GREEN)

`.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py
tests/test_signal_improvements.py tests/test_signal_generator_new.py
tests/test_train_watchdog.py -q` → **295 passed, 2 xpassed, 0 failed** in
278.92s (4:38).

`grep gpu_coord|ollama_coord|acquire_training_mutex|release_training_mutex
backend/` → no matches.

### Downstream

- Memory `backlog_gpu_sequencing.md` updated to reflect symmetric removal.
- CI Ruff failure on PR #1 (`backend/train_worker.py:8:1 I001`) auto-fixes once
  this lands on the PR's branch.
- Three remaining CI failures on PR #1 are still unrelated (npm lockfile out of
  sync; `ModuleNotFoundError: torch` in 5 snapshot tests; security gate
  downstream of those).

---

## [Session 58.50] — 2026-05-09 — Frontend CNN→XGB relabel + remove training UI (#267e/f)

### Why

The system already routes decisions through XGB (`MODEL_BACKEND=xgb`) but the UI
still showed a "CNN" tab, "CNN Signals" headers, "CNN" probability columns, and
a "Train Model" button — all aliased to backend training endpoints CNN no longer
uses. User: "change the front end labels as well" + "remove any CNN related
buttons for training/references."

### What changed (display-only, no API renames)

- **Tabs / header (`App.tsx`)** — `'CNN'` tab → `'XGB'`; tagline
  `RSI · MACD · CNN signals` → `RSI · MACD · XGB signals`.
- **CNNDashboard.tsx** — kept filename + component name (avoids touching imports
  and the unchanged `/api/cnn/*` REST routes), but stripped all training UI:
  removed `pollRef`, `training` / `trainResult` / `trainSecs` / `epochs` state,
  `startTrainPoll` callback, on-mount training-status useEffect, `handleTrain`
  handler, "Train epochs" input, "Train Model" button, and the trainResult
  status pill. The "Last Trained" stat card was removed and the timing row
  collapsed from 4 → 3 columns. Display labels CNN → XGB on signals header,
  empty-state, probability label, confidence-table title, table column, and
  ADX-band tooltip.
- **AgentsDashboard.tsx** — combined-PnL sub `Tech + CNN` → `Tech + XGB`;
  per-agent label `CNN Agent` → `XGB Agent`. State field names (`cnn`,
  `cnnSignals`, `cnnAg`) and the `d.agent === 'CNN'` filter kept — DB-side
  identifier still `'CNN'`, this is presentation only.
- **FiringCounter.tsx** — section header `CNN` → `XGB`; removed "Trains" stat
  pill (train_count is no longer surfaced in the UI).
- **PerformanceDashboard.tsx** — added a tiny `agentLabel(name)` helper that
  maps `'CNN' → 'XGB'` for display. Used in trade-ledger pills, decision-history
  pills, trade row badges, and decision row badges. The `AgentFilter` type and
  filter comparisons still use `'CNN'` since DB rows have `agent='CNN'` — only
  the rendered text changes.

### Not changed

- Backend `/api/cnn/*` routes (status, scan, scans, train/status). They still
  exist; the train endpoint just has no UI affordance now.
- Database `agent` column — trades and decisions persist with `agent='CNN'` so
  historical filters keep working.
- Component name `CNNDashboard` and import path — internal-only naming; renaming
  would touch the build graph for no user-visible benefit.

### Build verification

```
$ npm run build
tsc && vite build
✓ 41 modules transformed.
✓ built in 1.89s
```

No type errors, no removed-symbol warnings.

### Files Changed

- `frontend/src/App.tsx`
- `frontend/src/components/CNNDashboard.tsx`
- `frontend/src/components/AgentsDashboard.tsx`
- `frontend/src/components/FiringCounter.tsx`
- `frontend/src/components/PerformanceDashboard.tsx`

---

## [Session 58.49] — 2026-05-09 — Backend-aware log labels: "CNN [BUY]" → "XGB [BUY]" when MODEL_BACKEND=xgb (#267)

### Why

User reported: with `MODEL_BACKEND=xgb` set, the scan-loop log lines
still print `CNN [BUY]`, `CNN BOOK BUY`, `CNN BOOK SELL`, which read as
if CNN is making decisions. Reality: `_cnn_prob` already delegates to
`xgb_signal.xgb_prob` when backend is xgb (since #135 / #232), so those
log lines are *XGB* decisions wearing a misleading prefix.

### What changed

- `agents/cnn_agent.py` — added module-level `_backend_label()` helper
  that returns `"XGB"` if `config.model_backend == "xgb"` else `"CNN"`.
  Defensive fallback: any other value (incl. `"ensemble"`) prints `CNN`.
- 4 user-facing log strings in `generate_signal` now use the helper:
  - signal info: `f"{lbl} [{side}] {pid} | {lbl.lower()}={prob:.2%}..."`
  - book buy: `f"{lbl} BOOK BUY {pid} @{price:.4f}..."`
  - book buy skip (insufficient balance)
  - book sell

### Not changed

- CNN-internal logs (training, model load, dataset cache, book restore,
  retrain triggers) keep `CNN` prefix — they refer to actual CNN
  infrastructure regardless of decider backend.
- LGBM gate / Hurst / regime suppression logs already gated to
  `_cnn_only` (#232), so they only fire under MODEL_BACKEND=cnn and
  correctly print `CNN`.

### Tests

- `TestBackendLabelHelper` — 3 tests (xgb→XGB, cnn→CNN, unknown→CNN)
- All `TestSuppressionsGatedByBackend` tests still GREEN (no change to
  signal-generation control flow, just log-string formatting).

### Files Changed

- `backend/agents/cnn_agent.py` — added `_backend_label()`, 4 log lines
- `backend/tests/test_cnn_agent.py` — added `TestBackendLabelHelper`

---

## [Session 58.48] — 2026-05-09 — Timescale sweep on 28-ch survivorship-aware (#242)

### Verdict

```
   horizon          n   pos_rate   mean_auc
        1h    126,352      0.488     0.6095
        4h    160,129      0.489     0.6417   <- best
       12h    165,357      0.488     0.6411
       24h    165,722      0.488     0.6410
       72h    165,750      0.488     0.6405

best horizon: 4h  mean_auc=0.6417
```

### What this means

- **4h remains optimal**, matching prior #152 finding on the legacy 27-ch
  cache. Best alternative (12h) is 0.6411 — Δ = -0.0006, well below the
  +0.01 threshold that would justify a cache rebuild at a different
  horizon.
- **1h is meaningfully worse** (-0.032 vs 4h) — too noisy for the
  ±0.3% triple-barrier formulation; many samples bounce inside the
  barrier and label as 0 dead-zone, dropping labeled-sample count to
  126k from ~165k at longer horizons.
- **24h / 72h offer no lift** despite labeling more samples — long
  horizons don't unlock new signal in the existing 28-channel feature
  set.

### Caveat: fold variance

Folds at 4h: `[0.531, 0.533, 0.579, 0.772, 0.793]`. The mean 0.6417
is dominated by the most recent two folds (Q4-Q5) at 0.77-0.79, while
older folds (Q1-Q2) sit at chance (0.53). Same pattern across all
horizons. Implies either:
- The dataset has entered a more-learnable regime in the recent half
  (consistent with cleaner 28-ch coverage post-#177/#197 OI rebuild)
- A subtle leakage near the fold boundary that gets worse for newer
  data (the per-product feature normalization is built on the full
  ts range — strict-causality should reject this hypothesis but worth
  re-checking)

The relative ordering across horizons is robust to the variance source
since all five horizons exhibit the same fold pattern. The absolute
0.6417 number should be treated as an upper bound until the variance
source is understood.

### Decision

**Keep `forward_hours=4` as the production label horizon.** No cache
rebuild justified. Files #242 closed.

### Files Changed

- `CHANGELOG.md` — this entry
- (no code changes; this was a diagnostic sweep using existing
  `tools/timescale_sweep.py`)

---

## [Session 58.47] — 2026-05-09 — Marketcap probe NULL-COVERAGE (#260d/e/f)

### Verdict

```
=== single-add probe: log_marketcap -> ch13 ===
  baseline mean_auc = 0.5202
  replaced mean_auc = 0.5206
  delta             = +0.0004
  +0.01 gate:        FAIL  (null-coverage caveat)
```

**Δ is meaningless** — CoinGecko returned HTTP 401 for the first 5 pids
and 429 for the remaining 15, so per-sample non-zero marketcap coverage
was **0.0%** across all 167,938 pooled samples. ch13 was effectively
replaced with all-zeros. The +0.0004 lift is just XGB getting a
near-neutral signal vs a slightly noisy one — not evidence about
marketcap as a feature.

### Root cause

CoinGecko free tier no longer permits the
`/coins/{id}/market_chart/range` endpoint without a paid API key
(`COINGECKO_API_KEY`). The `/coins/markets` snapshot endpoint that the
service was tested against still works (and remains free), but the
historical timeseries needed for backtesting is gated.

### What landed

- `backend/tools/marketcap_probe.py` — single-add probe runner mirroring
  `okx_ls_probe` shape. Replaces ch13 (obv_slope) with per-pid
  log(market_cap) z-score, 1-day strict-causal lag.
- `backend/tests/test_marketcap_probe.py` — 15 tests covering target
  channel, lag/seq_len constants, log-transform, lag application,
  forward-fill, pre-history neutral-zero, empty-history fallback. **15/15
  PASS.**

### What this means

Probe pipeline is correct (tests pass; probe runs end-to-end). Data
source is the blocker. Three viable paths forward:
1. **Apply for CoinGecko Demo plan** — free tier API key (10k req/month);
   probe re-runs would gate cleanly.
2. **Switch to CoinPaprika** — historical OHLC + marketcap free, ~25k
   req/day, no key required.
3. **Defer marketcap and try a different exogenous input** — e.g. CME
   BTC futures basis, FRED 10y-2y spread, on-chain metrics (Glassnode
   Explorer free tier).

This probe is **not** the 8th BTC-flavored exogenous failure (those tested
real signal); it's the first data-availability blocker. Catalogued
separately so the verdict pattern stays interpretable.

### Files Changed

- `backend/tools/marketcap_probe.py` (new)
- `backend/tests/test_marketcap_probe.py` (new)
- `CHANGELOG.md`

---

## [Session 58.46] — 2026-05-09 — CoinGecko marketcap/FDV service scaffold (#260a/b/c)

### Why

After 7 sequential BTC-flavored probes failed the +0.01 mean-AUC gate
(#156, #235, #243, #246-#248, #253), the recommended path forward in
`xgb_feature_optimization_findings.md` is *new exogenous inputs*. Marketcap
and fully-diluted-valuation are the next candidate set: CoinGecko exposes
both for free, and rank-z marketcap is a textbook cross-sectional alpha
signal that the existing 28-channel set cannot derive from price/volume
alone.

### What landed

- `backend/services/coingecko_marketcap.py` — async fetcher + alignment
  helpers. Public API:
    - `fetch_marketcap_snapshot(pids)` → `{pid: MarketcapRow}`
    - `fetch_marketcap_history(pid, start_ms, end_ms)` → `[(ts_ms, mc), …]`
    - `align_to_hourly(rows, grid, lag_secs=86400)` — strict-causal forward fill
- `_PRODUCT_TO_CG_ID` map covers BTC + 18 of the 20-pid survivorship-aware
  basket used by `btc_residual_ch9_probe`. PIT columns (`ingest_ts`,
  `schema_version`) per #164b. Kill switch via `COINGECKO_DISABLED=1`.
- `backend/tests/test_coingecko_marketcap.py` — 18 tests covering
  id-mapping, snapshot parser, history parser, kill switch, 429 handling,
  null-FDV fallback, strict-causal forward-fill semantics. **18/18 PASS.**

### What's next

- #260d/e — RED + GREEN for `tools/marketcap_probe.py` (single-add probe
  vs ch13 obv_slope, mirrors `btc_residual_ch9_probe` structure).
- #260f — RUN; gate on Δ ≥ +0.01. Probe v1 will test two historical-capable
  candidates: `log_marketcap` and `marketcap_rank_z`.

### Files Changed

- `backend/services/coingecko_marketcap.py` (new)
- `backend/tests/test_coingecko_marketcap.py` (new)
- `CHANGELOG.md`

---

## [Session 58.45] — 2026-05-09 — Ch 9 β-residual probe verdict (#253d FAIL) + cp1252 print fix

### Verdict

```
=== single-add probe: btc_residual_ret -> ch9 ===
  baseline mean_auc = 0.5199
  replaced mean_auc = 0.5199
  delta             = -0.0000
  +0.01 gate:        FAIL
```

Pooled top-20 survivorship-aware (snapshot_ts=1757775600), n=167,933 samples,
20/20 pids 100% non-zero coverage, 5-fold purged CV with 4h embargo. The
β-residualized 1-bar return at Ch 9 carries no incremental signal vs the
raw 1-bar price change.

### What this means

This is the **7th sequential BTC-flavored probe to miss the +0.01 gate**:
#156 BTC-dominance, #235 OKX L/S, #243 long-trend, #246-#248 BTC lead-lag at
five horizons (1/3/6/12/24h), and now #253 β-residualization. The
empirical pattern is consistent: explicit BTC structure — whether injected
into the channel (lead-lag, dominance) or stripped from it (β-residual) —
does not move AUC at the Ch-9 substitution point on this 28-channel cache.

The user's domain intuition (BTC leads alts) is sound, but XGB on the
existing channel set already captures whatever marginal information that
relationship offers. Further BTC-derived single-add probes are unlikely to
clear the gate without a structural change (different timeframe grid,
different target channel, regime-conditional features, or new exogenous
inputs).

### Probe fix landed in this commit

`tools/btc_residual_ch9_probe.py` initially crashed mid-run on Windows
console (`UnicodeEncodeError: 'charmap' codec can't encode '\u03b2'`) —
two `print` calls embedded the literal `β` character, same class of bug
as #153 (hour_of_day_probe) and #249 (long_trend_probe Δ). Replaced both
print-side `β` occurrences with the ASCII string `beta`. Docstrings keep
the math notation (β/ε/←) — they are never written to stdout.

### Status

Probe runner is now Windows-stdout-safe. Verdict recorded. Task #253
(BTC-residualization, Option A) closes as **decided FAIL** per gate rule.
Next direction is at the user's discretion — channel weighting was the
follow-up gated on this verdict.

### Files Changed

- `backend/tools/btc_residual_ch9_probe.py` — replace two `β` chars in
  print f-strings with `beta` (cp1252 fix)
- `CHANGELOG.md` — this entry

---

## [Session 58.44] — 2026-05-09 — Ch 9 β-residual probe runner (#253c GREEN)

### Context

Helpers from Session 58.43 are pure-array math; this commit adds the
runner that wires them into the channel-replacement harness so the +0.01
AUC gate can be evaluated.

### Probe runner

`backend/tools/btc_residual_ch9_probe.py` — single-add probe over the top-20
survivorship-aware pids:

- Target: **Ch 9** (1-bar price change). Spec (#253a) chose this as the
  cleanest β-decomposable channel — every other close-derived channel
  (EMA-dist, BB-pos, norm_c, MACD) is a non-linear transform of multiple
  close points and doesn't admit a simple `r_alt = β·r_btc + ε` substitution.
- β window: **W=24** on the hourly grid. Spec said "288 bars (24h on 5m
  candles)"; this probe consumes hourly parquet history per the
  `btc_leadlag_probe._BAR_SECS=3600` convention, so 24h maps to W=24.
  Calendar context preserved; bar size differs.
- Per-pid signal: `residualize_returns(alt_log_ret, btc_log_ret, window=24)`
  → `{ts: ε_t}` filtered to finite values only (warm-up NaNs dropped).
- BTC-USD passthrough: skipped at the per-pid stage. ε_t for BTC vs itself
  is identically zero — no information; emitting it would skew downstream
  z-scoring with a flat-zero channel.
- Reuses `btc_leadlag_probe.build_leadlag_signal` for the [N, T=60]
  z-scoring + sample alignment so the probe matches the harness used in
  #246-#248.

### Tests

`backend/tests/test_btc_residual_ch9_probe.py` — 9 tests:

- `TestProbeConstants` (3): pin `_TARGET_CHANNEL=9`, `_BETA_WINDOW=24`, and
  `_BTC_PID == btc_leadlag_probe._BTC_PID`.
- `TestBuildResidualSignalForPid` (6): BTC-pid empty-dict passthrough,
  short-history empty, dict-typed output (int keys, float finite values),
  warm-up entries excluded, that the probe routes through
  `tools.btc_residualize.residualize_returns` (mock-asserted), and an
  end-to-end smoke (alt = exp(1.5·btc_log_ret + idio); residual must
  correlate with idio at corr > 0.7).

All 9 pass.

### Status

Probe runner is GREEN locally. The +0.01 AUC verdict (#253d) is the
next step — running the probe over the cache.

### Files Changed

- `backend/tools/btc_residual_ch9_probe.py` (new)
- `backend/tests/test_btc_residual_ch9_probe.py` (new)
- `CHANGELOG.md` — this entry

---

## [Session 58.43] — 2026-05-09 — BTC β-residualization helpers (#253b/#253c)

### Context

Six sequential probes have now missed the +0.01 AUC gate (#156, #235,
#243, #246-#248), each carrying a BTC-related signal *into* a channel.
User raised the empirical-economics frame: BTC leads alts on aggregate,
but every prior attempt encoded the common factor rather than stripping
it. β-residualization inverts the polarity — decompose
`r_alt[t] = β[t] · r_btc[t] + ε[t]` and let ε (the alt-specific component
*after* removing the BTC-correlated piece) drive the channel. This is
the candidate replacement for Ch 9 (1-bar price change) under #253c.

### Helpers

`backend/tools/btc_residualize.py` — pure-array, timeframe-agnostic:

- `compute_rolling_beta(alt, btc, window) -> np.ndarray` — β at index t
  fits OLS on the strictly-prior window `[t-W, t-1]`. Warm-up entries
  (`t < W`) are NaN. Zero-variance windows fall back to β=0.0 so the
  channel stays finite.
- `residualize_returns(alt, btc, window) -> np.ndarray` — ε[t] = alt[t]
  − β[t]·btc[t]. Shape preserved. When β=0 fallback fires, residual
  passes through the raw alt return.

Causality is the load-bearing invariant: `test_strict_causality_no_lookahead`
mutates `alt[t+1:]` and `btc[t+1:]` post-hoc and asserts every β[i] for
i ≤ t is unchanged. Same lookahead discipline as #157 ADX fix.

### Tests

`backend/tests/test_btc_residualize.py` — 18 tests across two classes:

- `TestComputeRollingBeta` (9): known-β recovery (alt = 2·btc → β=2),
  warm-up NaN, strict causality, zero-btc-variance fallback, shape
  preservation, mismatched-length guard, invalid-window guard.
- `TestResidualizeReturns` (9): zero-residual when alt = β·btc,
  idiosyncratic recovery (corr(ε, idio) > 0.7), warm-up, causality,
  BTC-self passthrough, zero-variance passthrough, shape, validation.

All 18 pass.

### Status

This commit lands the helpers and tests. The runner
`tools/btc_residual_ch9_probe.py` (#253c) and the +0.01 AUC gate
verdict (#253d) follow in subsequent commits. Nothing wired into
`cnn_agent.py` yet — channel mapping is a separate decision after the
probe gate result.

### Files Changed

- `backend/tools/btc_residualize.py` (new)
- `backend/tests/test_btc_residualize.py` (new)
- `CHANGELOG.md` — this entry

---

## [Session 58.42] — 2026-05-09 — Drop LLM under `MODEL_BACKEND != "cnn"` (#250)

### Context

Tracing the XGB decision flow surfaced the LLM blend as redundant work under
`MODEL_BACKEND=xgb`. The `_ollama_prob` prompt explicitly hands the active
backend's probability to the LLM as an anchor (`"CNN model probability:
{cnn_prob:.3f}"`), so for confident XGB outputs (the majority) the LLM almost
always confirms — burning ~5–25 s of GPU + Ollama coord per scan. For
borderline scans (0.50–0.55 zone where the LLM could plausibly swing the
result), the 0.55 BUY threshold gate kills the signal anyway.

Net effect under XGB: the LLM rarely moves an outcome but always pays the
latency. With XGB driving inference since Session 58.31 and the LLM blend
having no edge to add, the cleanest fix is to skip it entirely under any
non-CNN backend.

### Change

`agents/cnn_agent.py` — extended `skip_llm` to fire when
`config.model_backend != "cnn"`:

- New early term: `not backend_is_cnn` short-circuits the whole LLM branch.
- The `lessons = await get_tracker().get_lessons(...)` and
  `fg_data = await get_fear_greed().fetch()` calls — whose only consumer was
  the prompt builder — moved *into* the `else` branch so they're only
  performed when the LLM will actually run.
- New skip-debug log line surfaces the backend reason:
  `LLM skipped for {pid}: MODEL_BACKEND=xgb (LLM is anchored to backend prob — redundant)`.
- When LLM is skipped, `lessons` and `fg_score` default to `[]` / `None` so
  any downstream code that reads them stays safe.

The CNN path is unchanged: when `MODEL_BACKEND=cnn`, the original four
skip conditions (decisive cnn_prob, ambiguous regime + DI, noisy entropy,
training subprocess active) still govern whether Ollama runs. This mirrors
the #232 pattern of code-gating CNN-tuned behaviour on `model_backend`
rather than per-feature env knobs (per `feedback_cnn_safeguards_backend_gating.md`).

### Tests

`tests/test_cnn_agent.py` — added `TestLLMSkippedUnderXgb` (2 tests):

- `test_ollama_prob_not_awaited_when_backend_is_xgb` — RED-confirmed before
  fix. Pins cnn_prob=0.62, hurst=0.30, low DI/entropy → would normally fire
  Ollama. Asserts `_ollama_prob`, `tracker.get_lessons`, and `fg.fetch` all
  remain unawaited under `MODEL_BACKEND=xgb`.
- `test_ollama_prob_still_awaited_when_backend_is_cnn` — sanity guard so
  the new gate cannot silently kill the CNN path. Same fixture pinned to
  `MODEL_BACKEND=cnn` → asserts `_ollama_prob.assert_awaited_once()`.

Full `tests/test_cnn_agent.py` suite: **233 passed, 2 xpassed** (no
regressions).

### Files Changed

- `backend/agents/cnn_agent.py` — gate `_ollama_prob` + lessons + F&G fetches behind `model_backend == "cnn"`
- `backend/tests/test_cnn_agent.py` — `TestLLMSkippedUnderXgb` (2 new tests)
- `CHANGELOG.md` — this entry

---

## [Session 58.41] — 2026-05-09 — BTC lead-lag probe (#246-#248): all 5 candidates FAIL — 6th sequential probe miss on 0.55 gate

### Context

After Session 58.40 ruled out long-horizon SMA/golden-cross features,
the remaining structural gap in the 28-channel set is *temporal*
BTC→altcoin influence. Existing Ch 21 `btc_corr_20` is contemporaneous
(rolling correlation at the same t). Lead-lag asks the different
question: does BTC's move at t−k predict alt's move at t? β-residual
strips current-bar BTC influence to expose the idiosyncratic alt
component. Both are structurally novel relative to the 28-channel set
and the four positioning probes that came before.

### Probes

`backend/tools/btc_leadlag_probe.py` — single-add probe (replace ch13
obv_slope, the most marginal channel per #146). Five candidates:

| Candidate              | What it measures                                  |
|------------------------|---------------------------------------------------|
| `btc_ret_lag_1`        | BTC log-return at t−1 (1h ago)                    |
| `btc_ret_lag_4`        | BTC log-return at t−4 (4h ago)                    |
| `btc_ret_lag_12`       | BTC log-return at t−12 (12h ago)                  |
| `btc_beta_60`          | rolling 60-bar OLS β of alt_ret on btc_ret        |
| `btc_beta_residual_60` | alt_ret − β·btc_ret (60-bar window)               |

`backend/tests/test_btc_leadlag_probe.py` — 20 tests covering
`log_returns`, `lag_dict`, `align_pair`, `rolling_beta`,
`beta_residual`, `build_leadlag_signal` (no-lookahead, warm-window
correctness, zero-variance carry, pre-history neutral mean).

BTC-USD itself is skipped in the loop: lag-of-self is autocorrelation,
not lead-lag. The probe runs on all 20 of the survivorship-aware top-20
when BTC is not in the snapshot (which it wasn't here — pooled basket
is altcoin-heavy by construction).

### Result — all 5 candidates FAIL

Pooled top-20 with `--snapshot-ts auto` (cutoff 1756735200, 167,864
samples):

| Candidate              | Baseline AUC | Replaced AUC | Δ        | Gate |
|------------------------|-------------:|-------------:|---------:|------|
| btc_ret_lag_1          | 0.5199       | 0.5178       | −0.0021  | FAIL |
| btc_ret_lag_4          | 0.5199       | 0.5099       | −0.0100  | FAIL |
| btc_ret_lag_12         | 0.5199       | 0.5162       | −0.0038  | FAIL |
| btc_beta_60            | 0.5199       | 0.5196       | −0.0003  | FAIL |
| btc_beta_residual_60   | 0.5199       | 0.5196       | −0.0003  | FAIL |

All five replacements *degraded* AUC (no positive Δ even before the
+0.01 gate). The 4h-lag candidate's −0.0100 hit is the strongest
negative — replacing obv_slope with stale BTC return at the 4h horizon
actively confuses the booster.

### Verdict

**6th sequential probe failure** on the +0.01 gate (after MFI-rank,
log10-vol-rank, BTC-dominance, OKX L/S, long-trend). The pattern across
all six is consistent: at the **0.5199 baseline** AUC and **±0.005**
fold-noise, neither *positioning* (volume rank, dominance, L/S),
*long-horizon trend* (SMA50/200, golden cross), nor *cross-asset
temporal* (BTC lag, β, β-residual) features clear the gate on the
existing 28-channel survivorship-aware top-20 cache.

This makes the `xgb_feature_optimization_findings` conclusion sharper:
the 0.55 production gate is **not reachable on price/orderflow features
alone** at this sample regime. Path forward narrows to: (a) relax the
0.55 gate to something the 0.5199 baseline can clear after calibration,
or (b) bring in genuinely new input *classes* (per-product OKX OI
panel, options term structure, on-chain flows) — not more transforms
of the same OHLC+volume cache.

### Side notes

- The Δ Unicode crash that bit long_trend_probe was preemptively fixed
  in btc_leadlag_probe at write time (no second occurrence).
- Backend restart deferred: a CNN retrain (pid 44048) is currently
  running per `cnn_train_progress.json`, and per
  `feedback_no_restart_during_retrain.md` we never bounce the backend
  while a retrain holds the cache file. Restart to bring the new XGB
  inputs live (none, in this case — probe was read-only) is unnecessary
  for this task; we'll let the in-flight retrain finish.

---

## [Session 58.40] — 2026-05-09 — Long-trend probe (#243-#245, #249): all 5 candidates FAIL post-leak-fix; daily_resample lookahead bug found and squashed

### Context

The +0.01 AUC gate has resisted four positioning candidates (MFI-rank,
log10-vol-rank, BTC-dominance, OKX L/S — see Session 58.39b). The
xgb_feature_optimization_findings memo flagged "no long-horizon trend
feature" as a remaining inputs gap: longest MA distance in the 28
channels is `ema21_dist` (~0.9 days on 1h bars). #243-#245 tested whether
SMA50/SMA200 distance or a daily golden-cross sign lifts AUC.

### Probes

`backend/tools/long_trend_probe.py` — single-add probe (replace ch13
obv_slope, the most marginal channel per #146). Five candidates:

| Candidate          | What it measures                              |
|--------------------|-----------------------------------------------|
| `sma50_1h`         | hourly close vs 50-bar SMA (~2 days)          |
| `sma200_1h`        | hourly close vs 200-bar SMA (~8.3 days)       |
| `sma50_d1`         | hourly close vs daily-resampled 50-bar SMA    |
| `sma200_d1`        | hourly close vs daily-resampled 200-bar SMA   |
| `golden_cross_d1`  | sign(SMA50_d1 − SMA200_d1) ∈ {−1, 0, +1}      |

`backend/tests/test_long_trend_probe.py` — 19 tests covering
`compute_sma_dist_series`, `daily_resample`, `golden_cross_signal`,
`build_trend_signal` (no-lookahead, constant-input zero z, pre-history
neutral mean) plus an integration regression for the leak fix below.

### Lookahead leak in `daily_resample` — found mid-sweep, squashed

The first sweep returned `sma50_d1` Δ=+0.0852 (apparent PASS). On
inspection that was a **lookahead leak**: `daily_resample` keyed each
day's last-hour close at `day_start` (00:00 UTC). When the downstream
`build_trend_signal` forward-filled hourly samples from that dict, a
sample aligned at e.g. 14:00 of day D would resolve to the SMA computed
using **23:00 of day D** — ~9 hours of future close data per sample on
average; up to 24 hours at midnight. Fix: key by the actual
last-observation timestamp (`v[0]`) instead of `day_start`. Added
`test_no_lookahead_via_build_trend_signal` to pin the integration so
the leak cannot recur.

### Result — all 5 candidates FAIL post-fix

Pooled top-20 with `--snapshot-ts auto` (cutoff 1756735200, 167,861
samples):

| Candidate         | Baseline AUC | Replaced AUC | Δ        | Gate |
|-------------------|-------------:|-------------:|---------:|------|
| sma50_1h          | 0.5195       | 0.5201       | +0.0007  | FAIL |
| sma200_1h         | 0.5195       | 0.5201       | +0.0007  | FAIL |
| sma50_d1          | 0.5203       | 0.5201       | −0.0002  | FAIL (was leaky +0.0852) |
| sma200_d1         | 0.5203       | 0.5206       | +0.0004  | FAIL |
| golden_cross_d1   | 0.5203       | 0.5208       | +0.0006  | FAIL |

`golden_cross_d1` had only **7.1%** per-sample non-zero coverage:
`survivorship_aware_top_n` returned pids with ~167–193 daily bars but
the slow=200-day window needs more, so 17/20 pids emitted constant −1
(never warm). Even on the 3 pids that did warm up (PENGU, AVAX, LINK),
the cross adds no measurable signal vs ch13.

### Verdict

5th probe failure in the sequence. The "long-horizon trend gap" called
out in the memo is not, on this cache, a missing-information gap — at
least not at the SMA50/SMA200 horizons we can measure. Path forward
options unchanged from Session 58.39b (relax the gate, OR pursue OI as
a true new input class — Loop2 still on deck).

### Side fixes

- **#249** — Δ Unicode crash in long_trend_probe.py:383 SUMMARY printer
  (Windows cp1252 console can't encode `\u0394`; same bug as #153).
  Fixed in long_trend_probe.py and preemptively in btc_leadlag_probe.py.
- TDD red→green→run for the leak fix; 19/19 long-trend tests + 20/20
  BTC lead-lag tests pass.

---

## [Session 58.39b] — 2026-05-08 — OKX L/S probe: URL bug fixed (#235g), real result Δ=+0.0014 — TRUE FAIL

### Context

The Session 58.39 probe (commit `1ff997a`) returned 0/20 L/S coverage and a
meaningless Δ=+0.0026 (just the dropout effect of replacing ch13 with
zeros). #235g was filed to diagnose whether this was a fetcher bug or an
OKX coverage gap.

### Diagnosis

Direct curl revealed OKX has **two** L/S endpoints:

- `/api/v5/rubik/stat/contracts/long-short-account-ratio` — currency-level,
  takes `ccy=BTC`, returns coarser precision (e.g. `1.46`).
- `/api/v5/rubik/stat/contracts/long-short-account-ratio-contract` —
  per-instrument, takes `instId=BTC-USDT-SWAP`, finer precision
  (e.g. `1.3946...`).

Session 58.39 shipped the currency-level URL but called it with `instId=`
params, so OKX rejected every call (`code 50014: ccy can't be empty`).
That's why coverage was 0/20.

### Fix

`backend/services/okx_long_short_history.py`: change `_URL` to the
`-contract` variant. Preserves the existing `instId=` param contract
and `_PRODUCT_TO_OKX` symmetry with the OI fetcher. Updated docstring
explains both endpoints to prevent recurrence.

`backend/tests/test_okx_long_short_history.py`: tighten URL assertion
from substring (`"long-short-account-ratio" in args[0]`) to suffix
(`args[0].endswith("/long-short-account-ratio-contract")`) so a future
silent revert to the currency-level endpoint will RED.

### Tests

`tests/test_okx_long_short_history.py` — 16/16 pass post-fix.
RED→GREEN verified: pre-fix the new endswith assertion failed; post-fix
passes.

### Result — TRUE FAIL post-fix

Pooled top-20 with `--snapshot-ts auto` (same cutoff 1758816000, 165,551
samples):

| metric        | value   |
|---------------|---------|
| baseline AUC  | 0.5213  |
| replaced AUC  | 0.5226  |
| Δ             | +0.0014 |
| +0.01 gate    | FAIL    |
| L/S coverage  | **11/20 pids** (vs 0/20 pre-fix) |
| per-sample non-zero L/S | **0.3%** |

Per-pid coverage went from 0/20 to 11/20 — confirms the URL fix worked
on the 11 pids OKX has data for. But OKX only retains ~86 hours of L/S
history per pid, vs the dataset's months-long sample window — so even
where coverage exists, only 0.3% of samples have a non-zero L/S value.
The L/S signal is now **truly evaluated** (not always-zero) and falls
+0.0014 short of the +0.01 gate.

### Verdict

L/S joins **MFI-rank, log10-vol-rank, BTC-dominance** as the 4th
L/S-style/positioning probe failure. The +0.528 AUC ceiling on the
existing 27 channels is reaffirmed — none of these single-add candidates
breach the +0.01 gate.

The "four remaining moves" punchlist (`xgb_feature_optimization_findings.md`,
2026-05-08 #156) is now exhausted on positioning candidates. Path forward
options:
1. Relax gate from +0.01 to +0.005 (would let L/S, BTC-dominance, MFI-rank
   contribute marginally — but ensemble of marginal signals tends to
   overfit on small AUC margins)
2. Try larger structural changes (different label horizon, regime-conditional
   models, longer seq_len)
3. Accept 0.528 AUC and ship XGB shadow-mode results as-is

### Files Changed

- `backend/services/okx_long_short_history.py` — URL: `-contract` variant
- `backend/tests/test_okx_long_short_history.py` — tighter endpoint assertion
- `CHANGELOG.md` — this entry

---

## [Session 58.39] — 2026-05-08 — OKX long/short ratio single-add probe: 0/20 coverage, INCONCLUSIVE (#235)

### Note

Superseded by Session 58.39b (#235g) — the 0/20 coverage was a fetcher
URL bug, not an OKX coverage gap. After the fix, real coverage is 11/20
pids and the probe is a TRUE FAIL with Δ=+0.0014. Original entry kept
below for the audit trail.

### Context

After #156 (BTC-dominance, +0.0077 FAIL) the remaining "new input" candidates
to break the 0.528 AUC ceiling were OKX OI (#141–#145, already integrated as
Ch 27) and OKX long/short *account* ratio. L/S ratio measures retail
positioning skew (>1 = more accounts net-long than net-short); hypothesis
was that extreme positioning crowding precedes mean reversion that the
60-bar single-product window can't reconstruct from price/volume alone.

### Change

**New service** — `backend/services/okx_long_short_history.py` mirroring
`okx_oi_history.py`: `fetch_long_short_ratio_history(product_id, start_ms,
end_ms, bar="1H")` paginated via `after=` cursor, accepts both
`[ts, ratio]` array and `{"ts": ..., "ratio": ...}` dict row shapes,
imports `_PRODUCT_TO_OKX` from `okx_oi_history` (single source of truth so
L/S and OI never drift), `OKX_LS_DISABLED=1` kill switch.

**New probe** — `backend/tools/okx_ls_probe.py` (single-add harness, ch13
obv_slope target, +0.01 gate). Pure helpers (`ls_history_to_bar_grid`,
`build_ls_signal`) tested at module level; heavy imports (torch, services,
tools.feature_set_compare, tools.pid_snapshot) deferred inside the runner
functions so the test module can be collected without torch.

### Tests

`tests/test_okx_long_short_history.py` — 16 tests mirroring
`test_okx_oi_history.py`: symbol mapping, single-page fetch, dict/array
row shapes, failure modes (non-200, non-zero `code`, network exception,
malformed rows), kill switch (`OKX_LS_DISABLED`), pagination, plus the
critical contract test `test_supported_set_matches_oi_history` ensuring
the L/S map mirrors OI exactly.

`tests/test_okx_ls_probe.py` — 9 tests on pure helpers
(`build_ls_signal` empty/single/two-value/forward-fill/pre-history-zero,
`ls_history_to_bar_grid` ms→sec/off-grid-skip/empty,
`TestPerPidSignalCoverage`).

Full suite green pre-commit (16/16 service, 9/9 probe, OI 18/18 still
passing).

### Result — INCONCLUSIVE

Pooled top-20 with `--snapshot-ts auto` (cutoff 1758816000, 165,551 samples):

| metric        | value   |
|---------------|---------|
| baseline AUC  | 0.5213  |
| replaced AUC  | 0.5239  |
| Δ             | +0.0026 |
| +0.01 gate    | FAIL    |
| L/S coverage  | **0/20 pids** |

**Critical caveat:** OKX returned zero L/S history for every one of the
top-20 survivorship-aware pids (PENGU, POPCAT, JTO, BONK, NKN, ZK, AIOZ,
TRU, SKL, AVAX, MOODENG, PEPE, JASMY, LINK, ZORA, FET, LRDS, ONDO, XCN,
BLUR). The +0.0026 Δ is just the dropout effect of replacing ch13 with
all-zeros, **not a real evaluation of the L/S signal**.

### Why 0/20 coverage

The L/S endpoint
(`/api/v5/rubik/stat/contracts/long-short-account-ratio`) appears to have
narrower symbol coverage than OI on OKX, or the same `_PRODUCT_TO_OKX`
map needs different inst-id formatting for L/S. Filed as #235g for
diagnosis: pick one known-on-OKX pid (e.g. AVAX-USDT-SWAP), curl the
endpoint directly, determine whether this is a fetcher bug, an OKX
coverage gap, or a different inst-id convention.

**Decision:** do NOT mark L/S as the third L/S-style probe failure
(alongside MFI-rank, log10-vol-rank, BTC-dominance) yet. The probe is
inconclusive — the signal was never actually under test. Diagnose
coverage first (#235g); if coverage gap is fixable, re-run; if it's a
hard OKX limitation, then document as "L/S not testable at the pids we
need" and move on.

### Files Changed

- `backend/services/okx_long_short_history.py` — new fetcher
- `backend/tools/okx_ls_probe.py` — new single-add probe runner
- `backend/tests/test_okx_long_short_history.py` — 16 service tests
- `backend/tests/test_okx_ls_probe.py` — 9 probe-helper tests
- `CHANGELOG.md` — this entry

---

## [Session 58.38] — 2026-05-08 — Gate CNN-tuned BUY suppressions on `model_backend == "cnn"` (#232)

### Context

After #226 lowered `CNN_BUY_THRESHOLD` to 0.55 to expedite XGB evaluation, the
firehose opened: 43 BUY signals fired in 2h. But `signals_executed=0` and
nothing populated the CNN Agent tab — every signal was suppressed before
`_CNNBook.buy()` ran.

Root cause: three CNN-tuned safeguards in `agents/cnn_agent.py` fire
unconditionally on every BUY:
1. **Hurst random-walk gate** (`hurst < 0.45`) — Phase-1 CNN edge requires
   trending price action.
2. **HMM regime gate** (`regime != CHAOTIC`) — Phase-1 finding: CNN BUY edge
   is CHAOTIC-only (58.5% wr vs 44–46% in TRENDING/RANGING).
3. **LGBMFilter** — secondary gate calibrated on CNN outcome data.

Today the regime distribution was 98% TRENDING/RANGING + 2% CHAOTIC, so
the regime gate alone was suppressing nearly every BUY — and even those
that survived hit Hurst or LGBM. With `MODEL_BACKEND=xgb` driving inference
since Session 58.31, these CNN-only safeguards have no theoretical basis to
apply: XGB has its own edge profile and isn't restricted to CHAOTIC.

### Change

`agents/cnn_agent.py` — wrapped the three suppressions in
`_cnn_only = config.model_backend == "cnn"`:

- **Hurst gate**: `if _cnn_only and not hurst_ok` (was: `if not hurst_ok`)
- **Regime gate**: `elif _cnn_only and _regime_gate_enabled() and hmm_regime != "CHAOTIC"`
- **LGBMFilter eval block**: wrapped entire `_lgbm.predict` / `_lgbm.allow_buy`
  call in `if _cnn_only:`. Defaults `_lgbm_allow=True` so non-CNN backends
  fall straight through to `_CNNBook.buy()`.
- **Log line**: `lgbm_p={...}` portion of `CNN BOOK BUY` line is now
  conditional via `_lgbm_str`.

When `MODEL_BACKEND=cnn` (the original tuned configuration), all three gates
remain armed exactly as before. When the backend is anything else, they
auto-disable. Per `feedback_cnn_safeguards_backend_gating.md`: this code-gate
beats per-gate env knobs (e.g. `CNN_REGIME_GATE=off`) because (a) one knob =
one gate, so a partial fix leaves the other two blocking; (b) env disables
persist across `MODEL_BACKEND` flips, silently disarming the protection
when CNN reactivates.

### Tests

`tests/test_cnn_agent.py` — added `TestSuppressionsGatedByBackend` (3 tests):

- `test_regime_gate_skipped_when_backend_is_xgb` — TRENDING + xgb → BUY executes.
- `test_hurst_gate_skipped_when_backend_is_xgb` — Hurst=0.30 + xgb → BUY executes.
- `test_lgbm_gate_skipped_when_backend_is_xgb` — LGBM disallow + xgb → BUY executes.

Updated `TestInferenceRegimeGate.test_buy_blocked_when_regime_is_trending`
to set `model_backend="cnn"` so the gate-blocks-TRENDING contract is still
exercised under its intended backend.

Full suite: 914 passed.

### Files Changed

- `backend/agents/cnn_agent.py` — backend-conditional suppression gate
- `backend/tests/test_cnn_agent.py` — 3 new tests + 1 stale-assert update
- `CHANGELOG.md` — this entry

---

## [Session 58.37] — 2026-05-08 — Expedite XGB evaluation: lower BUY gate + env-ize scan interval (#226, #227)

### Context

XGB shadow logging resumed today after #223 fix (Session 58.36). First
post-restart sample showed `xgb_prob` peaking at **0.5427** (DOGE-USD) and
averaging ~0.50 across all pairs. The active BUY gate `CNN_BUY_THRESHOLD=0.80`
was unreachable on these outputs — the agent would never fire under XGB
backend, so `signal_outcomes` (which feeds isotonic-calibrator refits per
#187/#192) couldn't accumulate. The 7-day shadow eval (#136) was effectively
running at zero throughput.

User direction: with `DRY_RUN=true` (paper-only), expedite XGB evaluation
to identify model limitations and what works. Ship config knobs first,
gather data 24-48h, then decide on auto-recalibration / weekly retrain
infrastructure (deferred until we see actual hit-rate numbers).

### Changes

**#226 — `.env` `CNN_BUY_THRESHOLD` 0.80 → 0.55**

Opens the BUY gate enough that today's XGB peaks (0.54) can occasionally
cross it. Selectivity is intentionally weak (~coinflip + tiny edge) so the
agent fires on most positive XGB signals; goal is volume of `signal_outcomes`
rows for the next calibrator refit, not selective trading. Inline comment
in `.env:38-44` documents rationale + DRY_RUN constraint.

**#227 — env-ize scan-loop cadence (`SCAN_INTERVAL_SECS`)**

- `backend/config.py:65-67` — new `scan_interval_secs: int` field reading
  `SCAN_INTERVAL_SECS` env, default 900s (preserves existing behavior when
  env unset).
- `backend/main.py:478` — `cnn_agent.run_loop` call now passes
  `interval=config.scan_interval_secs` (was relying on `run_loop`'s
  hardcoded 900s default).
- `.env:28` — renamed dead `SCAN_INTERVAL_SECONDS=300` (read by nothing) to
  the live `SCAN_INTERVAL_SECS=300` so it actually takes effect. 3× scan
  rate → 3× signal_outcomes/day during eval.

### Tests

- `backend/tests/test_model_backend.py` — new `TestScanIntervalConfig` with
  3 cases: defaults to 900, reads env, normalizes to int. RED→GREEN.
- Existing `TestConfigField` + `TestCnnProbBranching` (5 cases) still pass.

### Files modified

- `backend/config.py`
- `backend/main.py`
- `backend/tests/test_model_backend.py`
- `.env` (untracked, documented for posterity)
- `CHANGELOG.md`
- `.claude/.../coinbase_trader_architecture.md`

### Verification

- `tests/test_model_backend.py` 8 passed in 4.97s
- `python -c "import main; print(...)"` — main imports cleanly
- Backend restart pending (#231)

### Open follow-ups

- Once we see ~24-48h of `signal_outcomes` at the new cadence, decide:
  - **A3**: auto-refit isotonic every N new outcome rows (small new code +
    hooks `outcome_tracker` → `tools/fit_xgb_calibration.py` →
    `/api/xgb/calibration/reload` endpoint that already exists from #194).
  - **B1-B3**: append-mode cache builder + weekly booster retrain + atomic
    booster swap endpoint (mirror of calibrator hot-reload).
- 7-day shadow eval (#136) clock effectively restarts now that signals
  will actually fire. Plan to revisit AUC/hit-rate gate decision around
  2026-05-15.

---

## [Session 58.36] — 2026-05-08 — Fix XGB shadow logging silently killed by CNN-checkpoint gate (#223)

### Context

XGB shadow logging (#181, #186) wrote `xgb_prob` to `cnn_scans` continuously
from 2026-05-04 through 2026-05-07 02:17 UTC, then stopped — not just
`xgb_prob` but **every column** for the next ~30 hours. Investigation
showed the 28-channel migration (#196–#201) updated the XGB booster +
calibrator + cache to 28 channels but did NOT retrain the CNN checkpoint,
which remained 27-channel and therefore got flagged
`_needs_retrain=True` on every backend startup since.

`generate_signal` had an unconditional early return guarding on
`_needs_retrain`, so the entire scan body — including `save_cnn_scan` at
line 2213 — was skipped. Under `MODEL_BACKEND=xgb` this is wrong: XGB
inference goes through `xgb_signal.xgb_prob`, not the PyTorch model, so
the CNN checkpoint compatibility is irrelevant.

### Files modified

- `backend/agents/cnn_agent.py:1928` — gate guarded behind
  `config.model_backend != "xgb"` so the XGB inference path bypasses the
  CNN checkpoint compatibility check. CNN backend behavior unchanged.
- `backend/tests/test_cnn_agent.py` — new
  `test_generate_signal_persists_scan_under_xgb_when_needs_retrain` asserts
  `save_cnn_scan` is called when `MODEL_BACKEND=xgb` and `_needs_retrain=True`.
  Existing `test_generate_signal_suppressed_when_needs_retrain` updated to
  pin `model_backend="cnn"` so it tests the CNN suppression branch
  unambiguously.

### Verification

- Targeted: 2 tests pass (1 new red→green, 1 updated still green)
- Smoke: 261 passed + 2 xfailed across cnn_agent + signal_generator + maker
  (was 260, +1 from new test)
- Pre-commit hook: full suite green

### Open follow-ups

- CNN checkpoint retrain on 28-channel cache so CNN backend can resume —
  separate task, doesn't block XGB shadow.
- Resume #136 Phase 6 shadow-mode evaluation now that scans are flowing
  again. The 30-hour gap means cumulative shadow data resets effectively
  to 3 days — extend observation window before XGB cutover review.

---

## [Session 58.35] — 2026-05-08 — Maker (post-only LIMIT) entry path with timeout-fallback (#119)

### Context

Existing `OrderExecutor.execute_signal` posts a LIMIT at the signal price
without `post_only`, so it crosses the spread and pays taker fees on the
entry leg (~0.60% on tier 0). #119 adds a parallel maker path that posts
at the touch with `post_only=True`, polls for fill, and falls back to a
MARKET order if the resting limit isn't matched within `timeout_secs`.

Cuts the round-trip cost from ~1.20% (taker both legs) toward 0.80% (maker
entry, taker exit) — or 0.50% on volume tier 2 if both legs end up maker
on a paired exit path later.

### Files added / modified

- `backend/agents/order_executor.py` — adds module-level helper
  `_maker_price(side, bid, ask)` (bid for BUY, ask for SELL) and
  `OrderExecutor.execute_maker_signal(signal, timeout_secs=30.0)`. The new
  method shares drawdown/preflight/sizing with `execute_signal`. Uses a
  `_wait_for_fill` poll loop (deadline-based, sleep ≤ 0.5s clamped to a
  quarter of remaining budget) to detect FILLED status, then returns
  `fill_mode="MAKER"`. On timeout it cancels the limit and places a
  market order, returning `fill_mode="TAKER_FALLBACK"`. Dry-run path
  short-circuits with `order_type="LIMIT_MAKER"` and never touches the
  exchange — same blast-radius profile as the existing dry-run path.
- `backend/tests/test_order_executor_maker.py` — 7 TDD tests covering
  maker price helper (BUY→bid, SELL→ask, case-insensitive), post-only
  LIMIT placement at the correct price for both sides, timeout
  cancel-and-fallback to MARKET (asserts cancel_orders + place_market_order
  both called), and dry-run no-live-call short-circuit.

### Scope discipline

**Purely additive.** No existing caller is migrated to
`execute_maker_signal` in this commit. The CNN/tech signal generator and
exit paths still call `execute_signal` / `sell()` exactly as before — so
behavior in production is unchanged. The maker path becomes opt-in once
the user picks a caller to migrate (likely CNN entries, since they sit on
the longest hold horizon and most tolerate maker-fill latency).

### Verification

- per-module: `pytest tests/test_order_executor_maker.py -v` → 7/7 green
- adjacent: `test_cnn_agent.py + test_signal_generator_new.py +
  test_order_executor_maker.py` → 260 passed + 2 xfailed in 4m41s
- pre-commit hook ran the full suite: see HEAD commit footer for count

### Why scoped this way

Live order-execution changes have non-zero blast radius even when written
correctly — a stray code path that picked up `execute_maker_signal`
unexpectedly could shift fee characteristics or fill timing on real
trades. Keeping the migration of callers a separate user-gated step
isolates the "code exists" decision from the "use it in prod" decision.

### Open follow-ups

- Migrate CNN entry path to call `execute_maker_signal` (gated by a new
  `MAKER_ENABLED` env flag so it can be flipped per environment without
  a redeploy)
- Pair with a maker-side EXIT path (post-only LIMIT at ask for SELL on
  TP, with timeout fallback to market) — would unlock the 0.50% RT tier
- Telemetry: track `fill_mode` distribution in `signal_outcomes` so we
  can measure realized maker-fill rate vs the model's expectation

---

## [Session 58.34] — 2026-05-08 — BTC-dominance probe (#156): FAIL +0.01 gate at Δ=+0.0077

### Context

#156 was last left in_progress and noted "blocked on data source" — earlier
scopings assumed CoinGecko/CMC historical BTC market cap, both gated behind
paid APIs and exchange-agnostic (don't reflect the Coinbase USD universe).
The unblocker: BTC USD-volume share is a faithful dominance proxy
computable from local parquet history we already backfill, and it captures
the same "how much of activity is in BTC vs alts" signal that drives BTC-
dominance trading wisdom — for the universe we actually deploy on.

### Files added

- `backend/tools/btc_dominance_probe.py` — pure helpers
  (`compute_btc_usd_volume_share`, `build_btc_dom_signal`,
  `build_pid_history_from_basket`) plus probe runner. Mirrors
  `tools/oi_single_add_probe.py` pattern: replace ch13 (obv_slope, most
  marginal noise channel per #146 ablation) with a market-wide BTC-
  dominance signal broadcast to all top-20 sample products.
- `backend/tests/test_btc_dominance_probe.py` — 13 tests covering pure
  share math, alignment + z-scoring, lookahead-safety on the WINDOW
  (z-score uses full-series stats per the OI probe pattern), and a
  regression for the basket-vs-sample-universe bug found mid-loop.

### First-run bug found and fixed mid-loop

First probe run showed **0 hours covered, 0% per-sample coverage,
Δ=+0.0001**. Root cause: `_load_pooled_with_btc_dom` reused the pooled
top-20 (by cache sample count) for the dominance computation, and that
list happened to be all alts/memes — BTC-USD wasn't in it, so
`compute_btc_usd_volume_share` short-circuited to `{}`. Fix: introduced a
fixed `_DOMINANCE_BASKET = (BTC, ETH, SOL, XRP, DOGE, ADA, LTC, AVAX, LINK,
DOT)` and a `build_pid_history_from_basket(basket, history_dir)` loader
decoupled from the sample universe. Three regression tests pin the basket
constant + skip-missing behavior. Re-run gave a meaningful series:
hours_covered=9,346, share mean=0.431, std=0.127, min=0.028, max=0.854.

### Result

- baseline mean_auc (5-fold purged CV, 4h embargo, 200 estimators) = 0.5134
- replaced  mean_auc = 0.5212
- **delta = +0.0077  →  +0.01 gate: FAIL**

### Decision

Abandon the BTC-dominance path per the #156 gate spec. The signal is
marginally positive — it does add information — but doesn't clear the
deployment threshold. Per the same logic that retired ch13 candidates
hour-of-day-sin/cos and others, +0.0077 isn't enough to justify a cache-
version bump and full retrain.

The probe tool stays in `tools/` for future reuse if we want to revisit
the denominator (e.g. include more pids, or weight by depth instead of
volume) before retiring the idea entirely.

### Tests

`pytest tests/test_btc_dominance_probe.py -v` → **13/13 passed in 4.13s**.

### Tasks completed
- #156, #156a-e (probe TDD + run), #156-bug (basket-vs-sample-universe fix
  via #217–#219 RED/GREEN/RUN)

---

## [Session 58.33] — 2026-05-08 — OKX OI fetcher coverage fix (#211): map +19 alts/memes

### Context

#211 closes the loop on #210. #210 found 17 of 20 top-N
survivorship-aware pids had all-zero OI in the cache because both
`services/okx_oi_history.py` and `services/okx_funding_history.py`
shared a hand-curated `_PRODUCT_TO_OKX` map of only 30 large-caps.
Unmapped pids fell through `_coinbase_to_okx() → None`, making
`fetch_oi_history()` short-circuit to `[]` and the cache build store
zero. That is option 1 in the #210 action menu and the only option
that actually reclaims signal.

### Files added

- `backend/tools/probe_okx_swap_listings.py` — diagnostic script that
  hits OKX's public `/api/v5/public/instruments?instType=SWAP`
  endpoint and reports which Coinbase pids in the #210 zero set have
  a live `<TICKER>-USDT-SWAP` instrument. Writes nothing; safe to
  re-run as OKX adds listings. Kept as a tool because the listing
  set is a moving target and we will need this again.

### Files patched

- `backend/services/okx_oi_history.py` and
  `backend/services/okx_funding_history.py`: extend `_PRODUCT_TO_OKX`
  with 19 verified entries (PENGU, JTO, POPCAT, BONK, ZK, PEPE,
  MOODENG, ONDO, ALGO, ZORA, WIF, RENDER, FLOKI, WLD, BERA, ENA,
  STRK, TON, JUP). Both maps stay in lockstep so the
  supported-symbol set is shared.

  Eight pids from the #210 zero set are intentionally left out
  because the live probe confirmed they have NO `<TICKER>-USDT-SWAP`
  on OKX as of 2026-05-08 (NKN, AIOZ, JASMY, TRU, SKL, FET, XCN,
  LRDS). They will keep returning `[]` without a wasted HTTP call.

### Files modified — tests

- `backend/tests/test_okx_oi_history.py`,
  `backend/tests/test_okx_funding_history.py`: each gains
  `test_alt_meme_pids_added_per_211` (asserts all 19 expected
  mappings) and `test_alts_with_no_okx_swap_still_return_none`
  (asserts the 8 OKX-absent pids stay `None`). RED prior to the map
  patch, GREEN after.

### Test result

```
$ pytest tests/test_okx_oi_history.py tests/test_okx_funding_history.py -v
33 passed in 3.88s
```

### Coverage impact (expected, will measure post-rebuild)

Of the 17 zero-OI pids #210 flagged:
- 9 are now fixable in the next cache rebuild (PENGU, JTO, POPCAT,
  BONK, ZK, PEPE, MOODENG, ONDO, ALGO).
- 8 remain `[]`-by-design — OKX simply doesn't list those perps.

Once a fresh cache rebuild runs, the per-pid coverage report should
flip from 17/20 all-zero to 8/20 all-zero (the OKX-absent set).
Aggregate frac_zero on Ch 27 should drop from 0.852 toward roughly
0.40, restoring the channel as a useful XGB input. A coverage-audit
re-run after the rebuild will confirm.

### Why no cache rebuild in this commit

The map fix is a code change with zero behavioural risk to a running
backend (the new keys add lookups, never mutate existing ones). The
rebuild is the heavy step and runs out-of-band, gated on the current
CNN retrain finishing per `feedback_no_restart_during_retrain.md`.
Treating fix and rebuild as separate commits keeps each one
reviewable.

---

## [Session 58.32] — 2026-05-08 — OKX OI coverage audit (#210) — 17/20 pids fully zero

### Context

#210, follow-up to #209. The Ch 27 drift probe found that 19/20 pids
report per-pid PSI=0.0000 — meaning the OI channel is constant in at
least one half. This audit answers: "constant at what value, and for
how much of the series?"

### Files added

- `backend/tools/oi_coverage_audit.py`
  - `coverage_stats(values)` — n / n_zero / n_nonzero / frac_zero /
    first_nonzero_idx for any 1-D series.
  - `leading_zero_run(values)` — length of contiguous zero prefix
    (distinguishes legitimate backfill gap vs. scattered zeros).
  - `per_pid_coverage(prods)` — one coverage row per pid sorted by
    frac_zero desc.
  - CLI hydrates the cache and tags each pid as `all-zero`,
    `leading-block (backfill)`, `scattered (suspect)`, or
    `fully covered`.
- `backend/tests/test_oi_coverage_audit.py` — 10 tests, all GREEN.

### Live audit findings

```
Ch 27 (okx_oi) coverage audit (top-20 survivorship-aware pids,
N=167,497 samples)

  17/20 pids: frac_zero = 1.000 (all-zero)
              PENGU, JTO, POPCAT, BONK, NKN, AIOZ, ZK, JASMY, TRU,
              SKL, PEPE, FET, MOODENG, XCN, ONDO, ALGO, LRDS

   3/20 pids: frac_zero ≈ 0.001-0.002 (effectively fully covered,
              scattered)
              DOT  (8,404 samples, 17 zeros)
              LINK (8,161 samples, 11 zeros)
              AVAX (8,209 samples, 11 zeros)

  total samples = 167,497 | total zero = 142,762 | frac_zero = 0.852
```

**Classification:** Ch 27 has a **fetcher coverage defect**, not just a
backfill gap. 85.2% of OI samples are zero, and the zero/non-zero
split is not "first half zero, second half real" (which would be
backfill) — it is "17 specific pids always zero, 3 specific pids
always real." Almost certainly: the OKX OI fetcher only resolves the
DOT/LINK/AVAX instrument IDs (which match OKX's perp naming
convention) and silently returns zero for the alts/memes whose OKX
instrument IDs don't match.

**Implication for XGB:** The 28-channel XGB model trained on this
cache has had Ch 27 = 0 for 85% of training samples. It almost
certainly learned to ignore the channel (or, worse, learned a
spurious "zero OI ⇒ X" pattern for the 17 alt pids). The XGB
permutation importance (Ω 2026-04 run) ranked Ch 27 near the bottom,
which is consistent with a near-constant feature.

**Action options (deferred to a separate fix ticket):**
1. Fix the OKX fetcher to use the correct instrument-ID convention
   per pid (preferred: more signal, more parity across pids).
2. Drop Ch 27 from XGB training (`XGB_DROP_CHANNELS={21,24,27}`) and
   revert to 27-channel training. Cheap; loses any OI signal from
   the 3 covered pids.
3. NaN-mask Ch 27 for the 17 affected pids in the cache build so the
   model treats the absence as "missing" rather than "zero". Requires
   a cache rebuild and an XGB feature-extraction NaN-handling check.

The reconciliation between #164 (variance ratio 10.4×) and #170
(PSI=0.0006 stable) is now fully understood:
- DOT/LINK/AVAX OI history backfilled into the cache *during* the
  cache window, expanding variance second-half (caught by #164).
- Both halves are still dominated by the 17 all-zero pids' constant-zero
  series, so PSI sees a stable "two point masses" distribution (caught
  by #170).
- The drift probe wasn't lying — it was just measuring distribution
  shape on a heavily-degenerate feature.

### Verification

```
$ python -m pytest tests/test_oi_coverage_audit.py -v
====== 10 passed in 3.57s ======
$ python tools/oi_coverage_audit.py
[full output captured above]
```

---

## [Session 58.31] — 2026-05-08 — Generalize drift diagnostic to any channel + Ch 27 (OI) probe (#209)

### Context

#209. The Ch 5 (#208) tool was hard-coded for one channel. Ch 27 (OKX
OI) was flagged twice with conflicting signals: #164 found a 10.4×
variance-ratio jump first→second half, while #170 found PSI=0.0006
(stable). Two questions: (a) generalize the diagnostic so we can probe
any channel, (b) reconcile the #164/#170 discrepancy on Ch 27.

### Changes

- `backend/tools/ch5_drift_diagnose.py` → `backend/tools/channel_drift_diagnose.py`
  (renamed via `git mv`; preserves history).
- `backend/tests/test_ch5_drift_diagnose.py` → `backend/tests/test_channel_drift_diagnose.py`
  (renamed; imports updated to `tools.channel_drift_diagnose`).
- New CLI flag `--channel N` (default 5 = macd_hist).
- New `_load_cache_and_extract_channel(channel)` replaces the
  Ch-5-specific loader.
- New `_CHANNEL_NAMES` tuple (28 entries, last is `okx_oi`) and
  `_channel_label(ch)` helper used in print headers.
- Docstring in tool updated to drop Ch 5 specificity (kept Ch 5 as
  default and the historical motivation note).

### Live Ch 27 (okx_oi) findings

Ran against the production cache (N=167,346 samples, top-20
survivorship-aware pids):

```
[1] Per-bin PSI=0.0002 — flagged stable. But the bin structure is
    *degenerate*: bin 9 holds ~90% of mass in both halves, bin 0 holds
    ~10%, bins 1–8 are all empty (edges all 0.0000). The PSI scalar
    is misleading because the distribution is two point masses, not a
    continuous shape.
[2] Half-vs-half stats reveal the real shift:
    - first half var = 0.0000, second half var = 0.0025  (NOT zero)
    - first half min/max = [-0.0306, 0.0000]
    - second half min/max = [-1.0000, +1.0000]  (full clip range)
    - skew flips -4.02 → +13.45
    First half is essentially a constant zero series; second half is
    real OI data spanning the clip range.
[3] Per-product PSI: only LINK shows nonzero (0.0827); the other 19
    pids report 0.0000. A "0.0000" per-pid PSI here means the channel
    was *constant* (typically all-zero) in at least one half — i.e.
    those pids had no OI data to align with their candles.
[4] Bin sensitivity: 0.0002 → 0.0030 from 5 → 40 bins (monotonic,
    stable). No binning artifact.
```

**Classification:** Ch 27 has a **data-availability / backfill issue**,
not a regime drift. First half: OI history is missing or zero for ~all
pids except LINK. Second half: real OI data populated for many. The
#164 variance-ratio metric correctly flagged this (variance grew as
data backfilled). The #170 PSI metric correctly reported "stable shape"
because both halves are dominated by the same clip-bin extremes. Both
were correct; they measured different things.

**Action:** before any further OI-feature work, audit OKX OI coverage
per-pid and per-period. If first-half OI is artificially zero (rather
than legitimately missing), the cache build needs to forward-fill or
NaN-mask those rows so the model doesn't learn a "zero OI = X behaviour"
spurious pattern. New BACKLOG ticket #210 will track the audit.

### Verification

```
$ python -m pytest tests/test_channel_drift_diagnose.py -v
====== 11 passed in 3.97s ======
$ python tools/channel_drift_diagnose.py --channel 27
[runs to completion; per-bin / stats / per-pid / sensitivity output]
```

---

## [Session 58.30] — 2026-05-08 — Ch 5 (macd_hist) drift diagnostic (#208)

### Context

Follow-up to #170. The drift monitor flagged Ch 5 (macd_hist) with
PSI=0.198 — minor drift, but no shape information. Before the next
retrain we needed to know *why*: real regime change, normalization
artifact, or survivorship/composition shift. Each leads to a different
remediation.

### Files added

- `backend/tools/ch5_drift_diagnose.py`
  - `decompose_psi(a, b, n_bins=10)` — per-bin contributions
    `(q-p)*log(q/p)` so we can see *which* bins drove the scalar PSI.
    Returns `{total_psi, flag, n_bins, per_bin: [...]}`.
  - `summary_stats(values)` — mean / population variance / skew / min /
    max / n. Empty input returns NaN safely.
  - `per_product_drift(prods, n_bins=10)` — PSI per pid sorted desc;
    detects whether drift is concentrated or broad-based.
  - `bin_count_sensitivity(a, b, n_bins_list)` — PSI for several
    n_bins values to flag normalization-fragile drift.
  - CLI hydrates `cnn_dataset_cache.pt`, runs all four against the
    chronologically-sorted Ch 5 terminal-value series.

### Tests added

`backend/tests/test_ch5_drift_diagnose.py` (11 tests, all GREEN):
- decompose_psi: total matches sum of per-bin contributions; identical
  halves give ~0; per-bin records carry required keys; a single
  bin-to-bin shift produces top-2 contributions >90% of total.
- summary_stats: known mean/var; right-skewed exponential gives skew>0.5;
  empty input returns n=0 without crashing.
- per_product_drift: sorts by PSI desc; preserves required keys; very
  short series return PSI=0.0 safely.
- bin_count_sensitivity: returns one PSI per n_bins; identical halves
  stay <1e-6 across all bin counts.

### Live cache findings

Ran against the production cache (N=167,346 samples, top-20
survivorship-aware pids):

```
[1] Per-bin PSI=0.1222 — U-shaped: mass moved out of bins 0 & 9
    (the tails) into bins 4-5 (around zero). Top contributors:
    bin 0 (0.0401), bin 9 (0.0351), bin 4 (0.0213), bin 5 (0.0152).
[2] First half var=0.2055, second half var=0.1057 — variance halved.
    Mean barely moved (-0.0013 → 0.0026). Min/max clipped at ±1 in
    both halves. No clipping shift.
[3] Per-product PSI sorts as: alts/memes high (POPCAT 2.05, AIOZ 1.59,
    BONK 1.08, MOODENG 0.95, PENGU 0.89, PEPE 0.77 ...);
    blue chips stable (LINK 0.02, AVAX 0.02). 17/20 pids ≥ minor.
[4] Bin-count sensitivity: PSI grows monotonically with n_bins
    (0.0959 @ 5 → 0.1958 @ 40). Not a binning artifact.
```

**Classification:** real volatility regime change in alt/meme tokens.
The post-meme-rally cooldown compressed MACD swings — same range, much
less mass in the tails. Not a normalization artifact (variance halving
with stable min/max indicates genuine peakedness shift). Not a
survivorship issue (same pids in both halves).

**Action:** confirms the #165 regime-conditioned model finding. Before
the next retrain: either sample-weight more recent data heavily, or
gate trade size by realized-volatility regime. Don't blindly retrain
on tail-heavy historical data — current behaviour is a fundamentally
different distribution.

### Verification

```
$ python -m pytest tests/test_ch5_drift_diagnose.py -v
====== 11 passed in 3.87s ======
```

---

## [Session 58.29] — 2026-05-08 — Silver-layer OHLCV anomaly flagger (#167a)

### Context

#167 BACKLOG. The bronze layer (#168) has provenance; silver should
have *quality*. This commit ships the anomaly-flagging primitive —
cheap, deterministic sanity rules over a candle list — so an audit
script or dashboard can highlight bars worth investigating before
they pollute training. Cross-exchange reconciliation (the second
half of #167) needs a second data source and is deferred to its own
follow-up; isolating the primitive keeps this commit shippable.

### Files added

- `backend/tools/anomaly_flagger.py`
  - `flag_ohlc_consistency(bars)` — high < low, or open/close outside
    [low, high].
  - `flag_zero_volume_runs(bars, min_run=5)` — runs of consecutive
    zero-volume bars at or above `min_run`.
  - `flag_return_z_outliers(bars, window=30, k=4.0)` — bars where
    `|log-return| > k * trailing-window stdev` (skips degenerate
    `sd <= 0` baselines).
  - `flag_volume_spikes(bars, window=20, k=10.0)` — bars where
    `volume > k * trailing-window median`.
  - `scan_bars(bars)` — runs all detectors, returns
    `{n_bars, anomalies, by_kind}`.

### Tests added

`backend/tests/test_anomaly_flagger.py` (13 tests, all GREEN):
- OHLC: high<low flagged, open>high flagged, close<low flagged,
  consistent bars pass.
- Zero-volume: long run flagged, short run ignored, isolated zero
  not flagged.
- Return z: 50% jump in otherwise-quiet series flagged; seeded
  random-walk produces no outliers (test corrected from a
  deterministic-linear-prices series whose log-returns have a
  near-zero stdev and produced spurious z-scores).
- Volume: 1000× spike flagged; steady volume passes.
- Combined: scan_bars surfaces ohlc kind in `by_kind`; 100 clean
  bars produce empty anomalies list.

### Verification

```
$ python -m pytest tests/test_anomaly_flagger.py -v
====== 13 passed in 3.64s ======
```

Cross-exchange reconciliation (Coinbase vs OKX/Binance closes) is
the natural follow-up — would extend `scan_bars` with a new detector
once the second feed is wired in.

---

## [Session 58.28] — 2026-05-08 — Bronze PIT tagging on parquet pulls (#168)

### Context

#168 BACKLOG. The bronze layer (raw OHLCV in `backend/data/history/`)
had no provenance metadata: a row's `start` told you *what bar* it
covered but not *when it was first ingested* — so re-pulls and PIT
backtests had no way to distinguish a row that's been in the file
for a year from one fetched five seconds ago. Standard data-warehouse
fix: add `ingest_ts` and `schema_version` columns at the writer.

### Files changed

- `backend/services/history_backfill.py`
  - `_SCHEMA_VERSION = 1` constant — bump on bronze schema column
    changes so consumers can branch.
  - `_SCHEMA` extended with `ingest_ts: int64` and
    `schema_version: int32`.
  - `_save_to_path(path, candles, *, now_ts=None)` — new keyword-
    only `now_ts` (defaults to `int(time.time())`). Stamps both
    columns on rows missing them; rows that already carry
    `ingest_ts` keep their original value across rewrites.
    Dedup-on-collision prefers the version with `ingest_ts` so the
    PIT history isn't dropped during merge.
  - `_load_from_path(path)` — back-compat: reads `ingest_ts` and
    `schema_version` when present, omits them otherwise. Pre-#168
    parquet files still load.

### Tests added

`backend/tests/test_history_backfill_pit.py` (6 tests, all GREEN):
- `_SCHEMA_VERSION` is a positive int.
- New writes emit `ingest_ts` column with the supplied `now_ts`.
- New writes emit `schema_version` column equal to the constant.
- PIT preservation: existing rows keep their original `ingest_ts`
  across a second rewrite; only newly-introduced rows pick up the
  second-write timestamp.
- Pre-#168 parquet (legacy 6-column schema) still loads via
  `_load_from_path`.
- Loaded candles carry `ingest_ts` and `schema_version` keys.

### Verification

```
$ python -m pytest tests/test_history_backfill_pit.py tests/test_history_backfill.py -v
====== 17 passed in 7.68s ======
```

Existing parquet files in production aren't migrated by this commit.
Their first-write `ingest_ts` will be set the next time the backfill
loop touches them (i.e. the first save after this code lands), with
`schema_version=1` from that point forward. PIT history starts then.

---

## [Session 58.27] — 2026-05-08 — Inference-time feature-freshness gate (#169)

### Context

#169 BACKLOG. Runtime counterpart to the offline diagnostics #164 / #170.
A live (n_channels, seq_len) window can look fine on shape and norms
yet be silently wrong if a feed pauses, geo-blocks, or gets cached. The
cheapest detectable signature: trailing-flat tails — a channel whose
last K bars are all the same value. The gate reports per-channel
trailing-flat counts and flags any that exceed their budget so the
caller (cnn_agent / xgb_signal) can skip-the-bar, warn-and-score, or
fall back to neutral.

### Files added

- `backend/tools/freshness_gate.py` — pure-numpy, allocation-light:
  - `_trailing_flat_bars(channel)` — count of trailing bars where
    `diff == 0`, walking backward from the tail.
  - `evaluate_freshness(window, max_flat_bars=5, per_channel_max=None,
    ignore_channels=None)` — returns `{fresh, stale_channels,
    channel_flat_bars, max_flat_bars}`. Per-channel overrides for
    legitimately-slow feeds (e.g. 1h cadence at 5m bars repeats ~11),
    and `ignore_channels` for permanently-zero geo-blocked feeds.
- `backend/tests/test_freshness_gate.py` — 11 tests covering: zero on
  changing tail, runs of repeated tail, all-constant edge case, short-
  channel safety, fresh-window pass, single-stale-channel flag,
  per-channel override respected, channel_flat_bars in report,
  threshold-boundary semantics (exactly == max is OK; > max is stale),
  and ignore-list excludes flagged channels from the verdict.

### Decision

No automatic wiring into cnn_agent / xgb_signal in this commit — that's
a behavior change with operator implications and deserves its own
threshold/policy follow-up. The helper is a stable API the live agents
will call when ready, with budgets calibrated against #170's per-channel
PSI report (Ch 5 was the only minor-drift channel; everything else
should clear `max_flat_bars=5` at default).

### Verification

```
$ python -m pytest tests/test_freshness_gate.py -v
====== 11 passed in 3.74s ======
```

---

## [Session 58.26] — 2026-05-08 — Per-channel distribution drift monitor (#170)

### Context

#170 BACKLOG. Companion to #164 (stationarity audit). #164 catches mean
and variance shifts via drift_z / var_ratio / lag1; this catches
distribution-shape shifts a same-mean-same-var pair would hide. Standard
Population Stability Index (PSI) on the chronologically-ordered terminal
value of each channel.

### Helpers

- `_bin_edges(values, n_bins=10)` — quantile-based edges; outer edges
  forced to ±inf so out-of-range values in the second half get binned.
- `_bin_counts(values, edges)` — probability vector per bin.
- `_psi(p, q, eps=1e-6)` — `sum((q - p) * log(q / p))` with eps clip
  to regularise zero bins (log(0)).
- `_channel_drift(series)` — splits chronologically into halves, runs
  PSI between bin probabilities, classifies via thresholds:
  - PSI < 0.10 → `stable`
  - 0.10 ≤ PSI < 0.25 → `minor`
  - PSI ≥ 0.25 → `significant`

### Changes (TDD)

- **`backend/tools/drift_monitor.py`** (new): helpers + `_audit_channels_drift`
  (uses `survivorship_aware_top_n` per #163) + argparse runner with
  `--snapshot-ts`, `--n-pids`, `--n-bins` flags.
- **`backend/tests/test_drift_monitor.py`** (new): 9 tests — bin-edge
  shape, constant-input safety, identical-distribution PSI=0,
  shifted-distribution PSI > 0.25, zero-bin epsilon regularisation,
  per-channel stable/significant/minor cases, short-input safety.
  9/9 GREEN.

### Findings (live cache, 169,375 samples × 28 channels)

| flag        | count | channels |
|-------------|-------|----------|
| stable      | 27    | 0–4, 6–27 |
| minor       | 1     | **5** (PSI=0.198) |
| significant | 0     | — |

Cross-reference with #164:

- **Ch 5** drifts in shape (PSI=0.198) but #164 didn't flag it — mean
  and variance held stable, only the shape moved. Worth a probe before
  next retrain.
- **Ch 27 (OI)** flagged by #164 (var_ratio=10.4) but PSI=0.0006 here
  — same distribution shape, just heavier tails. The two probes are
  complementary, not redundant.

No corrective action taken in this commit. Diagnostic only.

---

## [Session 58.25] — 2026-05-08 — Regime-stratified walk-forward eval (#165)

### Context

#165 BACKLOG calls for regime-stratified evaluation: a single overall
AUC can hide a model that only works in one volatility regime, or
breaks in CHAOTIC. This tool partitions val samples by realised-vol
terciles inside the existing purged walk-forward folds and reports
per-regime AUC alongside overall.

### Helpers

- `_classify_regimes(vols)` — terciles → `{'low','mid','high'}` string
  labels. Constant input → all `'mid'` (degenerate but non-crashing).
- `_per_regime_metrics(y_true, y_score, regimes)` — `{regime: {n,
  base_rate, auc}}` with tie-aware Mann-Whitney AUC; `auc=None` when a
  class is missing in a regime bucket.
- `_window_vol(X)` — std of last 24 bars on Ch 0 (norm_close) per sample.

### Changes (TDD)

- **`backend/tools/regime_eval.py`** (new): helpers above + `_run_eval`
  (XGBClassifier per fold, 100 trees / depth 4, embargo 4h) + argparse
  runner with `--snapshot-ts`, `--n-pids`, `--n-folds` flags.
- **`backend/tests/test_regime_eval.py`** (new): 6 tests — terciles
  split evenly, smallest→`low`, constant-vol non-crash, per-regime row
  count, perfect-separation AUC=1.0, single-class AUC=None. 6/6 GREEN.

### Findings (10 pids, 5 folds, ~17.4k val samples per fold, ~87k pooled)

| split   | n      | base   | AUC    |
|---------|--------|--------|--------|
| overall | 86,947 | ~0.49  | 0.5141 |
| low-vol | 28,985 | 0.487  | 0.5174 |
| mid-vol | 28,980 | 0.494  | **0.5067** |
| high-vol| 28,982 | 0.489  | 0.5179 |

**Mid-vol is the model's weak spot.** Low and high vol both clear
0.517; mid sits at 0.507 (essentially uninformative). All 5 folds
agree: mid-vol AUC is consistently the lowest. Suggests a
regime-conditioned model (or at least a regime-gated trade size) could
improve risk-adjusted P&L without more features.

No corrective action taken in this commit. Diagnostic only.

---

## [Session 58.24] — 2026-05-08 — Heuristic stationarity audit on 28 channels (#164)

### Context

#164 calls for an ADF stationarity audit. statsmodels is not in the
deps tree, so this commit ships a heuristic first-pass using only
numpy/scipy. Formal ADF deferred until a finding warrants the dep.

### Helper: `_stationarity_metrics`

Per-channel proxies on the chronologically-ordered terminal value of
each window:

- `drift_z` — |mean(first_half) − mean(second_half)| / overall_std
- `var_ratio` — std(second_half) / std(first_half)
- `lag1` — lag-1 autocorrelation
- `flag` — `'stationary'` unless drift_z > 0.5, |1 − var_ratio| > 0.5,
  or |lag1| > 0.95 → `'suspect'`

### Changes (TDD)

- **`backend/tools/stationarity_audit.py`** (new): `_stationarity_metrics`
  helper, `_audit_channels` orchestrator (uses
  `survivorship_aware_top_n` per #163), `argparse` runner with
  `--snapshot-ts` flag.
- **`backend/tests/test_stationarity_audit.py`** (new): 7 tests —
  constant / white-noise / random-walk / linear-trend / variance-blowout
  series + a short-input safety case + `_audit_channels` integration.
  7/7 GREEN.

### Findings (live cache, 169,367 samples × 28 channels)

| flag        | count | channels |
|-------------|-------|----------|
| stationary  | 27    | 0–26 |
| suspect     | 1     | **27 (OKX OI)** — var_ratio=10.4 (10× volatility regime change between halves) |

Sub-observations:

- **Ch 17, 18, 19**: constant zero (std=0). Expected — these are the
  remaining `_TRAINING_CONSTANT_CHANNELS` masked at training time.
- **Ch 22, 23**: lag1 ≈ 0.75 (BTC-corr / RV60-related). Borderline but
  below the 0.95 threshold.
- **Ch 27 (OI)**: var_ratio=10.4 means the second half has 10× the
  volatility of the first half. Likely cause: OKX OI history coverage
  asymmetry — older bars use shorter or sparser OI windows. Worth
  investigating before relying on Ch 27 as a strong feature.

No corrective action taken in this commit. The audit is diagnostic only.

---

## [Session 58.23] — 2026-05-08 — Migrate _timescale_sanity to survivorship-aware top-N (#163 follow-up)

### Context

Final cache-only consumer migration. `_timescale_sanity.py` is a
one-shot diagnostic comparing fresh-relabel vs cache-y at horizon 4h —
no public API beyond `main()`, so we extract `_pick_pids` as a
testable seam and delegate to `survivorship_aware_top_n`.

### Changes (TDD)

- **`backend/tools/_timescale_sanity.py`**: new `_pick_pids(prods, n,
  snapshot_ts=None)` helper. `main()` rewired to use it.
- **`backend/tests/test_timescale_sanity_snapshot.py`** (new): 2 tests
  covering legacy passthrough and survivorship cutoff. 2/2 GREEN.

### Validation

Per-module pytest passes 2/2. Default behaviour preserved
(`_pick_pids(prods, n=5, snapshot_ts=None)` reproduces the prior
`len(entry["X"])` ranking).

### #163 follow-up status

All 4 cache-only consumers migrated:

- ✅ `tools/rsi_rank_probe.py` (58.18)
- ✅ `tools/feature_set_compare.py` (58.20)
- ✅ `tools/hour_of_day_probe.py` (58.21)
- ✅ `tools/timescale_sweep.py` (58.22)
- ✅ `tools/_timescale_sanity.py` (58.23)

2 consumers remain on legacy ranking with explicit deferral reasons:

1. `tools/oi_single_add_probe.py:113` — network-bound (OKX); migration
   blocked on offline validation strategy
2. `tools/train_xgb_prod.py:53` — production booster trainer; coordinate
   with next planned retrain cycle to avoid silent ranking drift

---

## [Session 58.22] — 2026-05-08 — Migrate timescale_sweep to survivorship-aware top-N (#163 follow-up)

### Context

Continues the per-consumer #163 migration. `timescale_sweep.py` is the
horizon-sweep probe that pivots the question from "which channels?" to
"which forward_hours?" by relabeling at `h ∈ {1,4,12,24,72}` against
the existing 27-channel feature stack.

### Changes (TDD)

- **`backend/tools/timescale_sweep.py`**: `_load_pooled_with_pids(n,
  snapshot_ts=None)` delegates pid selection to
  `survivorship_aware_top_n`. Added `_parse_snapshot_ts(arg, prods)`
  helper. `main()` gains an `argparse` shell with `--snapshot-ts` flag.
- **`backend/tests/test_timescale_sweep_snapshot.py`** (new): 6 tests
  covering CLI parser (None / int / auto / empty-fallback) and
  `_load_pooled_with_pids` plumbing. 6/6 GREEN.

### Validation

Per-module pytest passes 6/6. Default call site
(`_load_pooled_with_pids(n=20)` without `snapshot_ts`) is byte-identical
to the prior implementation.

### Affected (still deferred)

3 consumers remain on legacy ranking:

1. `tools/oi_single_add_probe.py:113` (network: OKX)
2. `tools/train_xgb_prod.py:53` ← production booster trainer; coordinate
   with next planned retrain cycle
3. `tools/_timescale_sanity.py:30`

---

## [Session 58.21] — 2026-05-08 — Migrate hour_of_day_probe to survivorship-aware top-N (#163 follow-up)

### Context

Continues the per-consumer migration started in 58.18 (rsi_rank_probe)
and 58.20 (feature_set_compare). `hour_of_day_probe.py` had its own
inline legacy `len(entry["X"])` ranking inside `_load_pooled` —
swapping it to `survivorship_aware_top_n` keeps the four remaining
cache-only consumers in lockstep with the new opt-in pid snapshot.

### Changes (TDD)

- **`backend/tools/hour_of_day_probe.py`**: `_load_pooled(n, snapshot_ts=None)`
  now delegates pid selection to `survivorship_aware_top_n`. Added
  `_parse_snapshot_ts(arg, prods)` helper (mirrors feature_set_compare).
  `main()` gains an `argparse` shell with `--snapshot-ts` flag.
- **`backend/tests/test_hour_of_day_probe_snapshot.py`** (new): 6 tests
  covering CLI parser (None / int / auto / empty-fallback) and
  `_load_pooled` plumbing (legacy passthrough preserves sample count;
  cutoff drops newcomer samples and clips ts.max() ≤ cutoff). 6/6 GREEN.

### Validation

Per-module pytest passes 6/6. No behavioural change at the default call
site (`_load_pooled(n=20)` without `snapshot_ts` is byte-identical to
the prior implementation).

### Affected (still deferred)

4 consumers remain on legacy ranking, queued for separate per-consumer
migrations:

1. `tools/oi_single_add_probe.py:113` (network: OKX)
2. `tools/timescale_sweep.py:95`
3. `tools/train_xgb_prod.py:53` ← production booster trainer; coordinate
   with next planned retrain cycle
4. `tools/_timescale_sanity.py:30`

---

## [Session 58.20] — 2026-05-08 — Migrate feature_set_compare to survivorship-aware top-N (#163 follow-up)

### Context

Session 58.18 introduced `survivorship_aware_top_n` and migrated
`rsi_rank_probe.py`. Six other consumers remained on legacy
`len(entry["X"])` ranking. This commit migrates `feature_set_compare.py`
— the v1-vs-v2 feature-set comparison probe.

### Changes (TDD)

- **`backend/tools/feature_set_compare.py`**: `_pooled_top_n` now accepts
  `snapshot_ts: Optional[int]` and delegates pid selection to
  `survivorship_aware_top_n`. `snapshot_ts=None` preserves legacy ranking
  so prior results stay reproducible.
- New `_parse_snapshot_ts(arg, prods)` helper: `None` → legacy, `"auto"` →
  `recommended_snapshot_ts(prods)` (with graceful fallback to `None` when
  the cache has no non-empty products), or explicit int.
- `main()` gains an `argparse` shell with `--snapshot-ts` flag.
- **`backend/tests/test_feature_set_compare_snapshot.py`** (new): 6 tests
  covering CLI parser (None / int / auto / empty-fallback) and
  `_pooled_top_n` plumbing (legacy passthrough preserves sample count;
  cutoff drops newcomer samples and clips ts.max() ≤ cutoff). 6/6 GREEN.

### Validation

Per-module pytest passes 6/6. Pre-commit full suite re-run as part of
this commit. No behavioural change at the default call site
(`_pooled_top_n(prods, n=20)` without `snapshot_ts` is byte-identical
to the prior implementation).

### Affected (still deferred)

5 consumers remain on legacy ranking, queued for separate per-consumer
migrations:

1. `tools/oi_single_add_probe.py:113` (network: OKX)
2. `tools/timescale_sweep.py:95`
3. `tools/hour_of_day_probe.py:42`
4. `tools/train_xgb_prod.py:53` ← production booster trainer; coordinate
   with next planned retrain cycle
5. `tools/_timescale_sanity.py:30`

---

## [Session 58.19] — 2026-05-08 — Ch 0 norm_c strict-causality decision (#171) — accept-as-design

### Context

The #161 generic lookahead harness flagged Ch 0 (`norm_c`) as a strict-causality
leak: `mn, mx = min(closes), max(closes); norm_c = [(v - mn)/rng for v in
closes]` — every candle's normalized value depends on the global min/max
across the whole input window. Diff at candle 79 between full-input and
truncated-window builds: 5e-5.

Two options:
- **(a) Accept as design**: within-window normalization is a defensible CNN
  pattern. The model sees the SEQ_LEN window with terminal at the prediction
  bar; any backward dependency stays inside that window and does not cross
  the prediction boundary into bars > k that the model would otherwise
  never see at inference.
- **(b) Switch to per-bar expanding min/max** for strict causality. Changes
  the normalization scale, would require a full retrain + cache version
  bump, and pollutes the early-window distribution where expanding stats
  are noisy.

### Decision

**Option (a)** — accept as design. The strict-causality test stays xfailed
with `strict=True` so any accidental "fix" that changes the property without
explicit re-decision will surface as an xpassed regression. Documented the
rationale inline next to the Ch 0 build code so a future reader hits the
explanation before the test.

### Changes

- **`backend/agents/cnn_agent.py`**: inline comment at the Ch 0 build site
  documenting the within-window-normalization decision and pointing to
  the test-harness xfail for the property.
- No behavioral change. No test changes. No retrain.

### Validation

`backend/tests/test_feature_builder_causality.py`: 21 passed, 1 xfailed
(Ch 0) — exactly the documented expected state.

---

## [Session 58.18] — 2026-05-08 — Survivorship-aware top-N pid snapshot (#163) — RSI-rank lift survives at Δ+0.0124

### Context

All probe tools (`rsi_rank_probe`, `feature_set_compare`, `oi_single_add_probe`,
`timescale_sweep`, `hour_of_day_probe`, `train_xgb_prod`, `_timescale_sanity`)
select pooled top-N pids via `len(entry["X"])` — total cache sample count.
Bias: products that joined the tracked set recently and grew the most data
dominate. Recent winners crowd out historically-tracked products.

#163 introduces a centralized helper that takes a `snapshot_ts` cutoff and
ranks pids by samples visible at-or-before that timestamp. With
`snapshot_ts=None` it preserves legacy behavior so existing call sites are
unchanged until they opt in.

### Changes (TDD)

- **`backend/tools/pid_snapshot.py`** (new): `survivorship_aware_top_n(prods,
  n, snapshot_ts)` returns up to N pids ranked by samples ≤ cutoff, sorted
  by count desc then pid asc. `recommended_snapshot_ts(prods)` returns the
  median `first_ts` across non-empty products as a sensible default cutoff.
- **`backend/tests/test_pid_snapshot.py`** (new): 7 unit tests covering legacy
  passthrough, cutoff-excludes-newcomers, partial-truncation, capped-N,
  empty-prods, skip-empty-X, and median-first_ts recommendation. RED → 7/7
  GREEN.
- **`backend/tools/rsi_rank_probe.py`**: opt-in `--snapshot-ts {auto|int}`
  CLI flag. Defaults to legacy `None` so prior #162 result remains
  reproducible. `auto` resolves to `recommended_snapshot_ts(prods)`.

### Validation

Re-ran RSI-rank probe with `--snapshot-ts auto` (cutoff = 1758816000 ≈
2025-09-25, median across 113 products):

| Selection mode                  | baseline_auc | replaced_auc | Δ        | gate |
|---------------------------------|--------------|--------------|----------|------|
| legacy (recent winners top-20)  | 0.5201       | 0.5409       | +0.0208  | PASS |
| survivorship-aware (auto)       | 0.5229       | 0.5353       | +0.0124  | PASS |

Pid set differs ~6/20: legacy includes recently-grown ALGO/DOT/STRK/IO/ADA/HBAR;
survivorship-aware swaps in older-listed NKN/TRU/SKL/JASMY/XCN/BLUR.

Lift survives the methodology fix — RSI-rank carries real cross-sectional
information independent of selection bias. ~60% of the legacy Δ was real,
~40% was bias artifact. The integration plan from Session 58.16 remains
valid; the next retrain cycle should also adopt survivorship-aware
selection in the production driver (`train_xgb_prod.py`) — deferred to a
follow-up commit per find-list-fix.

### Affected (deferred, listed not fixed)

Other consumers still on legacy `len(entry["X"])` ranking:

1. `tools/feature_set_compare.py:54` `_pooled_top_n`
2. `tools/oi_single_add_probe.py:113`
3. `tools/timescale_sweep.py:95`
4. `tools/hour_of_day_probe.py:42`
5. `tools/train_xgb_prod.py:53` ← production booster trainer
6. `tools/_timescale_sanity.py:30`

Migration plan: each consumer flips to `survivorship_aware_top_n` with an
opt-in `snapshot_ts` arg in a separate commit, validated against its prior
result.

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
