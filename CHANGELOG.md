# Changelog — Coinbase AI Trader (polymarket_app)

All notable changes to this project are documented here.
Format: reverse-chronological by session date.

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
