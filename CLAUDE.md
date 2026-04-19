# CLAUDE.md — Coinbase AI Trader Coordination Contract

This file is read by all Claude Code agents working on this repository.
It defines responsibilities, workflows, and rules that every agent must follow.

---

## Scope

**Only modify files inside `C:\Users\gl450\polymarket_app\`.**
Never touch `radioconda\`, `.spyder-py3\`, or any other directory in the user's home folder.

---

## Find-List-Fix Workflow — Required whenever issues are identified

Whenever bugs, test failures, stale assertions, or needed refactors are found during any task:

```
1. STOP — do not fix inline without listing first
2. Write a numbered task list of every issue found (all of them, not just the current one)
3. Fix each item in order, marking it complete as you go
4. Do not move to the next task until the current one is green and committed
```

**Rule:** No fix is made silently. Every issue gets listed before it gets fixed.

Example — found 3 issues while working on a feature:
```
Found issues:
1. [ ] test_cache assertion uses stale tuple size (2-tuple, now 3-tuple)
2. [ ] _CNN_MAX_HOLD_SECS constant not matching test assertion
3. [ ] outcome_tracker.py hardcodes model name instead of reading env var
→ Fix 1, commit. Fix 2, commit. Fix 3, commit.
```

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

## Fix Workflow (REQUIRED)

When you find bugs, inconsistencies, or issues — during an audit, code review,
or in response to a user report — follow this sequence every time:

1. **Create a task list before touching code.**
   Use `TaskCreate` for each distinct fix. Each task must identify:
   - The file
   - The problem
   - The intended fix

2. **Fix in order. Mark tasks as you go.**
   Set `in_progress` when starting a task, `completed` immediately when done.
   Do not batch completions.

3. **Run tests after each fix**, not just at the end.
   If a test fails, resolve it before moving to the next task.

4. **No fix without a task.**
   If you discover a new problem mid-fix, create a task for it first.

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

```bash
# Kill leftover python processes (bash)
ps aux | grep python | grep -v grep | awk '{print $1}' | xargs kill -9 2>/dev/null

# PowerShell equivalent
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Rules:
- **Prefer per-module tests** over full suite during development — faster, stays foreground, no stacking
- **Only run full suite once** before committing, not repeatedly
- **Always clean up** after a test run completes or is interrupted

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

Relevant memory files for this repo:
| Memory file | Mirrors |
|---|---|
| `feedback_tdd_workflow.md` | TDD Workflow section |
| `feedback_scope_restriction.md` | Scope section |
| `feedback_shell_cleanup.md` | Shell cleanup section |
| `feedback_sync_rule.md` | Memory sync rule |
| `coinbase_trader_architecture.md` | Architecture Quick Reference |
| `trading_app_bugs_fixed.md` | Bug history (Sessions 9+) |
| `trading_app_thresholds.md` | Agent decision thresholds |

---

## Code Style

- Edit existing files rather than creating new ones.
- Do not add features, refactoring, or abstractions beyond the task scope.
- No comments unless the *why* is non-obvious.

---

## Architecture Quick Reference

- **Backend:** FastAPI + asyncio, port 8000
- **Frontend:** React + Vite + Tailwind, port 3000
- **DB:** SQLite via aiosqlite (`backend/coinbase.db`)
- **Market data:** Coinbase Advanced Trade API (REST + WebSocket)
- **AI agents:** CNN agent (PyTorch), Tech agent, Momentum agent, Scalp agent
- **Local inference:** Ollama at `http://localhost:11434`; model set via `OLLAMA_MODEL` in `.env`
- **Current Ollama model:** `llama3.1:8b` (~4.7 GB Q4) — fits RTX 2060 with headroom
- **GPU constraint:** RTX 2060 = 6 GB VRAM — only one Q4 model loaded at a time
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

---

## Current Test Coverage

| Module | Test File | Status |
|---|---|---|
| `agents/cnn_agent.py` | `test_cnn_agent.py` | ✅ covered |
| `agents/cnn_agent.py` (risk exits) | `test_cnn_risk_exits.py` | ✅ covered |
| `agents/tech_agent_cb.py` | `test_tech_agent_cb.py` | ✅ covered |
| `agents/momentum_agent_cb.py` | `test_momentum_entry_filter.py` | ✅ covered |
| `agents/signal_generator.py` | `test_signal_improvements.py` | ✅ covered |
| `data/cnn_model.py` | `test_cnn_model.py` | ✅ covered |
| `data/history_backfill.py` | `test_history_backfill.py` | ✅ covered |
| `data/macro_history.py` | `test_macro_history.py` | ✅ covered |
| `database.py` | (integration via agent tests) | partial |
