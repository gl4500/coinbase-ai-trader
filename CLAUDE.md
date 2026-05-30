# CLAUDE.md — Coinbase AI Trader Coordination Contract

This file is read by all Claude Code agents working on this repository.
It defines responsibilities, workflows, and rules that every agent must follow.

---

## Scope

**Only modify files inside `C:\Users\gl450\polymarket_app\`.**
Never touch `radioconda\`, `.spyder-py3\`, or any other directory in the user's home folder.

---

## Find-List-Fix Workflow — Required whenever issues are identified

When bugs, test failures, stale assertions, or needed refactors are found:

1. **STOP** — don't fix inline. Use `TaskCreate` for every distinct issue; each task names the file, the problem, the intended fix.
2. **Fix in order.** Mark `in_progress` when starting, `completed` immediately when done. No batching.
3. **Run tests after each fix**, not just at the end. If a test fails, resolve before moving on.
4. **No fix without a task.** Discover a new problem mid-fix? Create the task first.

No fix is made silently.

---

## TDD Workflow — Required for every change

```
1. Write a failing test in  backend/tests/test_<module>.py
2. Implement the code change
3. Run tests:  cd backend && python -m pytest tests/test_<module>.py -v
4. Verify GREEN before committing
5. Commit:  git add tests/<file> <module.py> && git commit
```

No code change is committed without a corresponding test. No exceptions for "small" fixes.

---

## Commit Standards

```
feat:     short description of new feature
fix:      short description of bug fixed
test:     add/update tests for X
docs:     update CHANGELOG.md, CLAUDE.md, or README
refactor: internal cleanup, no behavior change
security: security hardening, secret scanning
```

All commits include both the implementation file and its test file.
`Co-Authored-By` line required (added automatically by Claude Code).
Update `CHANGELOG.md` in the same commit as the feature or fix.

### Branch discipline — always use a fresh clean branch

For ANY work that will produce > 1 commit, create a fresh branch off `main` before the first commit. Never commit on top of an existing feature branch that another agent (or session) might have touched. Reusing an existing branch is how parallel-agent collisions happen — `git commit -a`-style races bundle a parallel agent's staged files into your commit (or vice versa), producing commits with misleading messages or files on the wrong branch.

Rule:
- Fresh branch name format: `feat/<scope>-<short-desc>` (e.g. `feat/strategy-discovery-phase2`, `fix/exit-watcher-race`).
- Branch from `main` (or the parent feature branch if explicitly stacking), never from another in-progress agent's branch.
- Every commit MUST use surgical pathspec: `git commit -m "msg" -- <explicit paths>`. NEVER `git commit -a` / `git commit -am`.
- Before every `git add`: `git rev-parse --abbrev-ref HEAD` to confirm you're on YOUR branch.
- After every `git commit`: `git log -1 --stat` to confirm only intended files landed.
- If pre-commit hook fails because of another agent's unrelated WIP in the working tree, use `git stash push -m "<other-agent>-WIP"` to relocate it, retry commit, then `git stash pop`. Never `--no-verify` without explicit operator authorization.

Reinforced 2026-05-24 after a Phase 2 collision where parallel agent's `git commit -a` swept my staged Phase 2 file into their GPU-kernel commit on my branch, AND one of my Phase 2 test commits accidentally landed on their branch and was pushed to origin. See `feedback_parallel_agent_coordination.md` for the full incident log.

---

## Security Gate — Pre-commit Checks

Before every `git push`:

1. **Block staged `.env`** — prevents API keys reaching GitHub (enforced by `.gitignore`)
2. **Verify `.env` is gitignored** — confirm before any commit touching credentials
3. **No plaintext secrets** — never hardcode API keys, private keys, or tokens in `.py`, `.ts`, or `.json` files

Sensitive files that must never be committed:
- `.env` (Coinbase API key + private key)
- `backend/coinbase.db` (live trading database)
- `backend/*.pt` / `backend/*.pth` (model weights — large binary)

---

## Test Conventions

- Framework: `pytest` with `pytest-asyncio`
- Async tests: `@pytest.mark.asyncio`
- **No live API calls** — mock `coinbase_client`, `database`, `httpx`, `ollama`
- **No real DB writes** — patch `database.*` with `AsyncMock`
- **No real file I/O** — mock progress files and model checkpoints
- Shared fixtures live in `tests/conftest.py` — don't duplicate setup across files
- One test file per module: `agents/cnn_agent.py` → `tests/test_cnn_agent.py`

### Running tests
```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/test_<module>.py -v    # per-module (preferred during dev)
python -m pytest backend/tests/ -v            # full suite (run once before commit)
```

### Shell cleanup — required after every test run

Running tests repeatedly causes background python processes to stack up. After any test run:

```powershell
# PowerShell — SKIPS the live backend (port 8001) and the launcher exe
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

```bash
# Bash — same intent: only kill processes that are NOT the backend (port 8001)
BACKEND_PID=$(ss -ltnp 2>/dev/null | awk '/:8001 /{match($0,/pid=[0-9]+/); if (RSTART) print substr($0,RSTART+4,RLENGTH-4)}' | head -1)
ps aux | grep -E 'python|pytest' | grep -v grep | awk -v skip="$BACKEND_PID" '$2 != skip {print $2}' | xargs -r kill -9 2>/dev/null
```

Rules:
- **Prefer per-module tests** over full suite during development — faster, stays foreground, no stacking
- **Only run full suite once** before committing, not repeatedly
- **Always clean up** after a test run completes or is interrupted
- **Never** issue a blanket `Stop-Process python -Force` / `pkill -9 python` — it kills the live backend, breaking the scan loop and stopping MC telemetry accumulation. The port-8001-aware snippets above are the only sanctioned cleanup commands.

---

## Backend port discipline — dev on 8002+, promote to 8001

**Port 8001 is reserved for the live trading backend.** Active paper/live trading runs there. Frontend (port 5174) is hardcoded to hit 8001. Never start a new/experimental/shadow backend on 8001 while the production backend is running — the `_free_port(8001)` helper will kill the production process, stopping live trading.

Rule:
- **Production / live trading** → port 8001 (default, set via `PORT` env or default in `main.py`)
- **Dev / shadow / new-model validation** → port 8002 (or any other unused port). Start with `PORT=8002 python main.py`.
- **Promotion to 8001 is gated on the new model showing promise** — wait for validation evidence (held-out AUC, shadow telemetry, post-week live trade outcomes) before swapping artifacts to the unsuffixed paths (`xgb_*_v4.*` etc.) and restarting the 8001 backend.

Why this matters: SQLite (`coinbase.db`) is a shared file. Multiple backends can run simultaneously, all writing to the same `cnn_scans` / `trades` / `signal_outcomes` tables. A backend on 8002 with new code will populate v4 (or other) telemetry columns in those shared tables — you don't need 8001 to validate the new path's output. Only push to 8001 when you're ready to swap the *driver*.

How to apply when shipping a new XGB model / inference path:
1. Land code + migration (creates new telemetry column nullable).
2. Operator-train the new model → unsuffixed artifacts (or horizon-suffixed during sweep).
3. Operator-start the new backend on **port 8002**: `PORT=8002 python main.py` from `backend/` cwd.
4. Verify new telemetry column populates in `cnn_scans` (rows from PID 8002 backend will have non-NULL).
5. Compare new vs old metrics over a shadow week (or however long the cutover plan specifies).
6. **If new model shows promise** → copy new-model artifacts to the unsuffixed paths the 8001 backend loads, then restart the 8001 backend (which will pick up the new artifacts). The promotion is just a file-copy + restart.
7. If new model doesn't show promise → kill the 8002 backend, leave 8001 untouched, iterate on the model.

Never kill 8001 mid-session unless: (a) the operator explicitly approves, OR (b) you're swapping artifacts after the promotion-gate criteria are met.

---

## Session hygiene — compact periodically

Long Claude Code sessions burn context fast — especially subagent-driven implementation runs, multi-file refactors, and brainstorm → spec → plan → execute cycles. The assistant should **suggest `/compact`** at natural breakpoints rather than let context grow unbounded.

Triggers to suggest `/compact`:
- After 5+ subagent dispatches within one feature
- When estimated conversation exceeds ~100k tokens
- Before starting a NEW feature in a session that already shipped one
- At natural milestones — after a commit lands, after a sweep module finishes, after a brainstorm/plan cycle completes
- When the user mentions cost, slowness, or long-context concerns

How to apply:
- Surface as a single-line suggestion at the breakpoint: *"Context is getting long — want to `/compact` before we continue?"*
- Don't compact silently or force it; the user decides
- `/compact` preserves project state (memory files, recent commits, working tree) — safe at any natural breakpoint
- After `/compact` fires, the same rule applies to the next chunk of work

---

## Memory

- Update relevant memory files immediately after every code change.
- Do not wait until end of session.

---

## CLAUDE.md ↔ Memory Sync Rule

**Both must always be updated together, and memory must be updated after every code change.**

`CLAUDE.md` (this file, in the repo) and the persistent memory files at
`C:\Users\gl450\.claude\projects\C--Users-gl450\memory\` are the two halves of the same contract.

### After every code change — required steps
1. Architecture change (agents, endpoints, file structure) → update `coinbase_trader_architecture.md`.
2. Bug fixed → append the bug + fix to `trading_app_bugs_fixed.md`.
3. Threshold changed → update `trading_app_thresholds.md`.
4. Rule added or modified → update the matching memory file AND this `CLAUDE.md` in the same response.
5. Never commit code without committing any corresponding `CLAUDE.md` update in the same or immediately following commit.

Relevant memory files for this repo (`coinbase-ai-trader` / polymarket_app):
| Memory file | Mirrors |
|---|---|
| `feedback_tdd_workflow.md` | TDD Workflow section |
| `feedback_scope_restriction.md` | Scope section |
| `feedback_shell_cleanup.md` | Shell cleanup section |
| `feedback_sync_rule.md` | Memory sync rule |
| `feedback_python_clean_functions.md` | Code Style + new-code authoring rules |
| `feedback_xgb_focus_not_cnn.md` | XGB-only scope; CNN frozen for new feature work |
| `coinbase_trader_architecture.md` | This file's Architecture Quick Reference + per-session change log |
| `coinbase_trader_schema.md` | DB column lists + code landmarks |

Note: `trading_app_*.md` memory files belong to a different project (not polymarket_app) and should be ignored when working in this repo.

---

## Code Style

- Edit existing files rather than creating new ones.
- Do not add features, refactoring, or abstractions beyond the task scope.
- No comments unless the *why* is non-obvious.

---

## Required Skills — Invoke Before Acting

These trigger clauses are binding. Invoke the matching skill via the `Skill` tool before the action, not after.

| Trigger | Skill |
|---|---|
| Writing or modifying code | `superpowers:test-driven-development` |
| Proposing a fix for a bug or failing test | `superpowers:systematic-debugging` |
| Claiming work complete, or before committing | `superpowers:verification-before-completion` |
| Touching anything under `frontend/` | `frontend-design` |
| Receiving any code-review feedback | `superpowers:receiving-code-review` |
| Before requesting review on a completed change | `superpowers:requesting-code-review` |

A global `SessionStart` hook in `~/.claude/settings.json` also echoes this list at the start of every session so it is always in context.

---

## Architecture Quick Reference

- **Backend:** FastAPI + asyncio, port **8001**
- **Frontend:** React + Vite + Tailwind, port **5174**
- **DB:** SQLite via aiosqlite (`backend/coinbase.db`)
- **Market data:** Coinbase Advanced Trade API (REST + WebSocket)
- **AI agents:** CoinbaseCNNAgent only (XGBoost driver via `MODEL_BACKEND=xgb`; CNN model path still exists but xgb has been the live backend for months — see `agents/xgb_signal.py`). Historical: TechAgent retired #311-refactor-c (2026-05-16); Ollama LLM blend deleted #311-refactor-f; rows remain in DB.
- **Config:** `.env` → environment variables read directly in modules
- **Training:** `train_worker.py` spawned as subprocess to avoid blocking scan loop

### Key invariants (never break these)
1. `_CNNBook.positions[pid]` must always contain `entry_time` and `peak_price` on new entries
2. CNN cache is a 3-tuple: `(cnn_prob, timestamp, indicators_dict)` — never 2-tuple
3. `_CNN_STOP_LOSS_PCT = 0.08` (8%) — required for $50k capital-at-risk math
4. `_CNN_MAX_HOLD_SECS = 7 * 24 * 3600` (7 days) — trailing stop is primary exit; this is safety net
5. `database.upsert_product` ON CONFLICT must include `is_tracked=excluded.is_tracked` — omitting it freezes CNN scanning
6. Auto-train must run in subprocess, never block the async scan loop
7. `OLLAMA_MODEL` must be read from env in every module — never hardcode a model name
8. `__MACRO__.parquet` uses `__`-prefixed filename — `symbols_with_data()` must filter `__`-prefixed entries
9. **Dataset cache is per-product append-only** — entry schema `{first_ts, last_ts, last_n, X, y, indices}`. Schema changes to seq_len/forward_hours/label_thresh/n_channels require bumping `_DATASET_CACHE_VERSION`.
10. **Training BCE uses smoothed labels; validation BCE uses hard labels** — changing either breaks run-to-run val_loss comparability.
11. **Inference must mask `_TRAINING_CONSTANT_CHANNELS` before forward pass** — `_cnn_prob` calls `_mask_training_constant_channels` to prevent train/serve distribution skew. Removing this requires retraining without affected channels.
12. **BCE uses `reduction="none"` + uniqueness-weighted mean** — never `reduction="mean"` on overlapping forward-window samples.
13. **XGB feature_set v3** uses 3 tiers (micro 60 / meso 168 / macro 336), 350 feature_names (320 live + 30 zero-slot for masked ch17/18/19), feature_weights (micro 1.0 / meso 2.0 / macro 3.0 / masked 0.0) set on `DMatrix` via `set_info` with `colsample_bytree=0.8`. Tier assignment lives in `tools/xgb_features.py:MESO_CHANNELS={15,24,25,26}` and `MACRO_CHANNELS={20,21,27}`. Per-tier candle slices come from `services/tiered_history.py:fetch_tiered` (sync). Calibrator pickle is `{"calibrator","feature_set"}` dict; bare isotonic still treated as v1 for back-compat. `xgb_signal` auto-detects via `_m060_/_m168_/_m336_` infix in feature_names. v3 inference REQUIRES `pid` kwarg through `_cnn_prob -> xgb_signal.xgb_prob(channels, pid=pid)`.
14. **MC filter chain** lives under `backend/agents/mc/` and is the ONLY place Monte Carlo math touches the decision pipeline. `cnn_agent.generate_signal` has exactly one MC hook (`mc.apply_buy_filters` between side computation and save_cnn_scan); embedding MC math inside cnn_agent core is forbidden. Each filter is opt-in via `MC_FILTERS` env (comma-separated). `MC_FILTERS=""` (default) MUST produce bit-for-bit pre-MC behavior. Telemetry persists to `cnn_scans.xgb_prob_stdev` (CIFilter only) and `cnn_scans.mc_telemetry` (JSON blob, any filter). Filter exceptions MUST be caught + logged, never re-raised into the scan loop. Filter classes self-register with `agents.mc.registry._FILTER_CLASSES` on import.
15. **Env vars MUST trace to a live consumer.** Every entry in `backend/config.py` and `.env` must be read by production code in `backend/`. Dead entries are deleted on sight per refactor sweep policy (#311-refactor). The `test_no_dead_llm_blend_fields` regression test in `backend/tests/test_config.py` enforces this for known offenders; future deletions extend that test.
16. **Shadow telemetry isolation** — Inference shadow paths (v4 alongside v3) must NEVER affect the driver path. Failures in any shadow inference are caught + logged + recorded as NULL, never re-raised. `xgb_signal.xgb_prob_shadow` is the only function that may be called from `cnn_agent` during shadow validation; it returns `(driver_prob, shadow_prob_or_None)`. Mirrors invariant #14's MC chain rule.
17. **v4.5 3-class telemetry contract.** The three v4.5 probability columns (`xgb_prob_v4_5_down`, `xgb_prob_v4_5_neutral`, `xgb_prob_v4_5_up`) are written atomically: either all three are populated and sum to ~1.0 (after clip [0.01, 0.99] + renormalize), or all three are NULL. A v4.5 failure must NEVER affect the v3 driver path or v4 shadow path. Implemented via isolated try/except in `xgb_signal.xgb_prob_shadow_v4_5`.
18. **WS exit-handler isolation.** `agents/exit_watcher.on_price_tick` MUST catch every exception in its body and log it at ERROR. Exceptions in handlers spawned via `asyncio.create_task` (which is how `CoinbaseWSSubscriber._handle` fires registered price handlers) do not crash the WS receive loop — the task captures them — but unretrieved-Task warnings hide errors from logs. Explicit `try/except + logger.exception` is required so failures stay visible. Mirrors invariant #14's MC chain rule and invariant #16's shadow-telemetry isolation rule. Also: the per-pid `_CNNBook._sell_locks: Dict[str, asyncio.Lock]` (consumed via `_lock_for(pid)`) is the single source of truth for serializing the WS exit handler against scan-loop `_check_risk_exits`; do not duplicate the lock at call sites.
19. **PnL-anchored trail exit.** `_compute_exit_threshold` in `agents/exit_thresholds.py` is the single source of truth for the trail / step-up profit-floor exit. Both `_check_risk_exits` (scan loop) and `exit_watcher.on_price_tick` (WS path) compute `exit_threshold` from `peak_pnl_pct + atr_pct + position_dollars + total_capital`, then fire `TRAIL_STOP` / `WS_TRAIL_STOP` when `pct_entry < exit_threshold`. Do not bypass to the raw `pct_from_peak <= -trail_pct` check that this replaced. `peak_pnl_pct` ratchets upward only and is seeded on `_CNNBook.load()` via `_migrate_position_state` for legacy positions that pre-date the field. `position_dollars` is owned by the scan loop (refreshed each scan); the WS path reads it but does not write, to avoid per-tick mutation contention.
20. **v4.5 MODEL_DOWN exit.** When the v4.5 shadow indicates `p_down > _P_DOWN_EXIT_THRESHOLD` (0.55 in `cnn_agent`) for a held position, fire `MODEL_DOWN` (scan loop) or `WS_MODEL_DOWN` (WS path). `generate_signal` caches `p_down` on `book.positions[pid]` after a successful shadow call, so both exit paths read the same value without re-running inference. The trigger sits between `STOP_LOSS` (capital protection) and `TRAIL_STOP` (profit protection) in the exit ladder. Failures must be isolated per invariant #16 — a missing/None v4.5 result must NEVER re-raise into the scan loop or WS handler; absent `p_down` defaults to 0.0 (no MODEL_DOWN fire).

---

## Current Test Coverage

| Module | Test File | Status |
|---|---|---|
| `agents/cnn_agent.py` | `test_cnn_agent.py` | ✅ covered |
| `agents/cnn_agent.py` (risk exits) | `test_cnn_risk_exits.py` | ✅ covered |
| `agents/xgb_signal.py` | `test_xgb_signal.py` | ✅ covered |
| `agents/mc/*` | `tests/agents/mc/*` | ✅ covered |
| `agents/signal_generator.py` | `test_signal_improvements.py` | ✅ covered |
| `data/cnn_model.py` | `test_cnn_model.py` | ✅ covered |
| `data/history_backfill.py` | `test_history_backfill.py` | ✅ covered |
| `data/macro_history.py` | `test_macro_history.py` | ✅ covered |
| `database.py` | `test_database.py` + integration via agent tests | ✅ covered |
| `services/marketcap_history_cache.py` | `test_marketcap_history_cache.py` | ✅ covered |
| `services/coingecko_marketcap.py` | `test_coingecko_marketcap.py` | ✅ covered |
| `services/coinpaprika_marketcap.py` | `test_coinpaprika_marketcap.py` | ✅ covered |
| `services/tiered_history.py` | `test_tiered_history.py` | ✅ covered |
| `tools/xgb_features.py` (v1/v2/v3) | `test_xgb_features.py`, `test_xgb_features_v3.py` | ✅ covered |

(Test count: ~970 passing as of Session 58.71i.) Modules retired: `tech_agent_cb.py` (#311-refactor-c, 2026-05-16).
