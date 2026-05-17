# Refactor Sweep — Module 3: TechAgent Removal

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "approved" × 2 across two sections)
**Scope:** `backend/` + `frontend/` (TECH has UI)
**Branch:** continue on `feat/gpu-coord-mirror`
**Sweep position:** Module 3 of N
**Predecessors:** Module 1 (#311-refactor-a, commit `97dc8c9`), Module 2 (#311-refactor-b, commit `cff73a0`)
**Successor:** Module 4 — likely CNN_ARCH dead variants or cnn_agent dead branches under MODEL_BACKEND=xgb (operator picks at next brainstorm)

---

## 1. Problem

`backend/agents/tech_agent_cb.py` (654 LOC) is one of two live trading agents. It runs every 2 minutes with its own indicator stack (RSI/BB/MACD/Stoch/OBV/ADX/MFI), $1000 dry-run paper book, and a 4-trigger exit chain (TICK_SIGNAL/TICK_STOP/TICK_TRAIL/TICK_PROFIT).

5-day trade review (2026-05-10 to 2026-05-15) showed:
- TECH overall: 140 trades, net **-$27**
- TECH **TICK_TRAIL**: 51 trades, **98% WR**, **+$46** — the single best-performing trigger in the entire system
- TECH TICK_STOP: 72 trades, 0% WR by def, **-$85** (dominant cause of TECH's net loss)

Operator chose "delete entirely" after the TICK_TRAIL trade-off was flagged. Rationale: simplify to a single XGB-driven decision path; focus optimization there.

## 2. Goal

Remove TechAgent from the live trading pipeline. Close all 39 open paper positions cleanly at current market price first. Preserve all historical TECH data in the DB (530 trades, 279,451 decisions, 563 outcomes — operator chose "keep everything"). Update frontend to hide TECH from live displays while keeping a collapsed "Retired Agents" history panel.

This module establishes the precedent for retiring a live agent: preflight close → atomic backend commit → frontend commit → history preserved.

## 3. Non-goals

- No retention of TICK_TRAIL logic in the CNN/XGB path (operator explicitly chose "accept the TICK_TRAIL loss" — porting was a separate option that was rejected).
- No DB schema changes. No VACUUM. No purge of historical TECH rows.
- No changes to CNN/XGB inference logic.
- No changes to MC filter chain.
- No changes to launcher/build pipeline beyond removing TechAgent boot wiring.

## 4. Approach

Two-phase commit on `feat/gpu-coord-mirror`:
- **Phase A (backend + preflight script + tests + memory sync)** — one atomic commit
- **Phase B (frontend)** — one atomic commit, immediately after A merges

Preflight script committed in A; operator executes it MANUALLY before the A commit lands. This is the only operator step.

### 4.1 Files touched

#### Phase A — backend
| Path | Action | Diff scope |
|---|---|---|
| `backend/agents/tech_agent_cb.py` | DELETE | -654 LOC |
| `backend/tests/test_tech_agent_cb.py` | DELETE | -497 LOC |
| `backend/main.py:51` | DELETE — `from agents.tech_agent_cb import TechAgentCB` | -1 line |
| `backend/main.py:197` | DELETE — `tech_agent: TechAgentCB = None` field on AppState | -1 line |
| `backend/main.py:411` | DELETE — `app_state.tech_agent = TechAgentCB(ws_subscriber=...)` | -1 line |
| `backend/main.py:474` | DELETE — `await app_state.tech_agent.run_loop(...)` | -1 line |
| `backend/main.py:1138` | EDIT — `tech_status = app_state.tech_agent.status if app_state.tech_agent else {}` → `tech_status = {}` (response still returns the field for frontend back-compat during Phase A→B window) | ~1 line |
| `backend/agents/cnn_agent.py:2133-2141` | EDIT — `get_agent_decisions` call kept (still useful for CNN history under MODEL_BACKEND=cnn); update comment to reflect TECH is gone, the agent_ctx now only contains historical CNN decisions | comment-only |
| `backend/services/outcome_tracker.py:234-241` | DELETE the `if source == "TECH":` branch in `_format_indicators` — dead after no new TECH outcomes are written; historical TECH outcomes remain in DB but are never re-formatted | -8 lines |
| `backend/tools/close_tech_positions.py` | CREATE — preflight one-shot script | +80 LOC |
| `backend/tests/test_main_no_tech_import.py` | CREATE — regression test: `import main` does not pull in `agents.tech_agent_cb` | +20 LOC |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71d entry | new |
| `polymarket_app/CLAUDE.md` | EDIT — Architecture section "Active agents: CNN only" | ~3 lines |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | EDIT — Agents table: drop TECH row; note historical TECH rows remain in DB | outside repo |
| `~/.claude/projects/.../memory/trading_app_architecture.md` | EDIT — same | outside repo |

Net Phase A: ~-1150 LOC code, +100 LOC (script + test), ~+30 LOC docs.

#### Phase B — frontend
| Path | Action | Diff scope |
|---|---|---|
| `frontend/src/components/AgentsDashboard.tsx` | EDIT — remove live TECH rendering; add collapsed "Retired Agents (history)" panel that queries `trades` / `agent_state` with `agent='TECH'` read-only | ~-80 to -120 + ~+30 |
| `frontend/src/components/CNNDashboard.tsx` | EDIT — drop the **Tech** column from the confidence table | ~-5 to -10 |
| `frontend/src/components/SignalDashboard.tsx` | EDIT (if exists) — drop TECH agent filter option | ~-3 |
| `polymarket_app/CHANGELOG.md` | APPEND — Session 58.71e entry | new |

Net Phase B: ~-90 LOC, +30 LOC.

### 4.2 Preflight script (committed in A, run by operator)

`backend/tools/close_tech_positions.py`:

```python
"""One-shot preflight before TechAgent removal (#311-refactor-c).

Closes all open TECH paper positions at current Coinbase price, writes
each close to the trades table with trigger_close='MANUAL_TECH_RETIREMENT',
zeros out the agent_state.positions_json for TECH. Idempotent — re-run
with no open positions is a no-op.

Run from repo root:
    .venv/Scripts/python.exe -m backend.tools.close_tech_positions
"""
import asyncio
import json
import os
import sqlite3
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from clients import coinbase_client


async def main() -> int:
    db_path = os.path.join(BACKEND, "coinbase.db")
    db = sqlite3.connect(db_path)
    row = db.execute(
        "SELECT positions_json, balance, realized_pnl FROM agent_state WHERE agent='TECH'"
    ).fetchone()
    if not row or not row[0]:
        print("No TECH positions to close. Exiting.")
        db.close()
        return 0
    positions = json.loads(row[0])
    balance, realized = float(row[1]), float(row[2])
    if not positions:
        print("TECH agent_state has zero open positions. Exiting.")
        db.close()
        return 0
    print(f"Closing {len(positions)} TECH positions; starting balance ${balance:.2f}, realized ${realized:.2f}")

    closed_at_iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    new_realized = realized
    new_balance = balance
    fees_total = 0.0
    for pid, p in positions.items():
        try:
            data = await coinbase_client.get_product(pid)
            price = float(data.get("price") or p.get("avg_price", 0))
        except Exception:
            price = float(p.get("avg_price", 0))  # fallback to entry price
        size = float(p["size"])
        avg = float(p["avg_price"])
        usd_open = size * avg
        usd_close = size * price
        pnl = usd_close - usd_open
        pct = (price / avg - 1) * 100 if avg else 0.0
        opened_at = p.get("entry_time_iso") or closed_at_iso
        db.execute("""
            INSERT INTO trades (
                agent, product_id, entry_price, exit_price, size,
                usd_open, usd_close, pnl, pct_pnl, hold_secs,
                trigger_open, trigger_close, balance_after, opened_at, closed_at
            ) VALUES ('TECH', ?, ?, ?, ?, ?, ?, ?, ?, 0,
                      'PRE_RETIREMENT', 'MANUAL_TECH_RETIREMENT', ?, ?, ?)
        """, (pid, avg, price, size, usd_open, usd_close, pnl, pct,
              balance + pnl, opened_at, closed_at_iso))
        new_realized += pnl
        new_balance += usd_close
        print(f"  {pid:<14} sz={size:.6f} avg={avg:.4f} now={price:.4f} pnl=${pnl:+.2f}")

    db.execute(
        "UPDATE agent_state SET positions_json=?, balance=?, realized_pnl=? "
        "WHERE agent='TECH'",
        ('{}', new_balance, new_realized),
    )
    db.commit()
    db.close()
    print(f"Done. Final TECH balance ${new_balance:.2f}  realized_pnl ${new_realized:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
```

Operator runs this ONCE before the Phase A code-deletion commit. Script's idempotent — re-running with no positions is a no-op.

### 4.3 Tests

| Test | File | What it covers |
|---|---|---|
| EXISTING `test_tech_agent_cb.py` | DELETE | All 497 LOC of TechAgent tests retire with the module |
| EXISTING `test_cnn_agent.py` | AUDIT — any test that imports/patches `agents.tech_agent_cb`? | Update or delete those specific tests |
| NEW `test_main_no_tech_import.py` | CREATE | `import main` does not pull in `agents.tech_agent_cb`; AppState has no `tech_agent` attribute |
| NEW `test_close_tech_positions.py` | CREATE — 3 tests | (a) no-op when no open positions, (b) writes a trade row per position, (c) zeros agent_state.positions_json |

The 3 new preflight script tests use a tmp DB seeded with a known agent_state row. No live Coinbase calls — `coinbase_client.get_product` mocked.

Total Phase A test delta: -497 (one big file deleted) + ~50 (4 new tests) = net **-447 LOC of tests**.

### 4.4 Architecture (Phase A unchanged paths)

Unchanged:
- CNN agent (`agents/cnn_agent.py` — only the comment around line 2133 changes)
- XGB v3 booster + tier extraction + CIFilter
- Order executor, WebSocket subscriber, market scanner
- Database schema and queries (TECH-row queries still work; they just won't accumulate new rows)
- Frontend `MarketBrowser.tsx`, `OrderBook.tsx`, `PositionTracker.tsx`, `LogViewer.tsx`

### 4.5 Architecture (Phase B frontend)

Two-tier dashboard:
- **Active agents:** CNN-only block (was CNN + TECH)
- **Retired agents (collapsed history):** read-only panel showing TECH's final realized PnL + closed trades count + last-100 trades table; queries `agent_state WHERE agent='TECH'` and `trades WHERE agent='TECH'`

CNN confidence table loses the **Tech** column entirely (and the historical Tech signals that previously appeared there).

## 5. Data flow

**Before (live scan):**
```
2-min interval        → tech_agent_cb.on_scan() → indicators + decision + book.buy/sell
WS tick               → tech_agent_cb.on_price_tick() → exit chain
generate_signal/CNN   → reads recent agent_decisions (CNN + TECH votes) for Ollama context
```

**After Phase A:**
```
2-min interval        → TechAgent gone, no tech scan loop
WS tick               → no TECH tick handler; WS ticks still route to CNN's _check_risk_exits
generate_signal/CNN   → reads recent agent_decisions; only CNN votes appear (no new TECH writes)
```

**After Phase B:**
```
Frontend              → AgentsDashboard renders CNN live + TECH historical panel
CNN confidence table  → no Tech column
```

Zero impact on CNN signal generation, XGB inference, or MC filter chain.

## 6. Error handling

| Condition | Behavior |
|---|---|
| Preflight script run twice | Idempotent — first run zeros positions_json; second run sees empty dict and exits |
| Preflight script can't fetch a live price | Falls back to avg_price (entry price) → pnl=0 for that position; warning printed |
| Phase A deployed but preflight NOT run first | TechAgent still has open positions in agent_state, but the run_loop is gone — positions stay frozen, no new exits fire, no harm. Operator can run the preflight script after the commit too (deletes the now-stale positions_json) |
| Phase A deployed but Phase B not yet | Frontend tries to render TECH; backend returns `tech_status = {}` so frontend shows zeros. Cosmetic glitch, no error |
| Rollback during Phase A→B window | Easy — `git revert` Phase A. agent_state.positions_json is `{}` post-preflight; the reverted code resumes with an empty book. Operator can hand-restore positions_json from a DB backup if they want |

## 7. Tests (summary)

| File | Action | Net |
|---|---|---|
| `tests/test_tech_agent_cb.py` | DELETE entire file | -497 LOC |
| `tests/test_main_no_tech_import.py` | NEW | +1 test (2 assertions) |
| `tests/test_close_tech_positions.py` | NEW | +3 tests |
| `tests/test_cnn_agent.py` | AUDIT (likely no changes) | 0 |
| Existing 1100+ test suite | MUST stay green | enforced by pre-commit hook |

## 8. Rollout

### Phase A (operator-driven preflight + atomic commit)

1. **Operator runs the preflight script:**
   ```
   cd C:\Users\gl450\polymarket_app
   .venv/Scripts/python.exe backend/tools/close_tech_positions.py
   ```
   Output: ~39 lines, one per position closed, final balance + realized_pnl. Idempotent.

2. **Operator backup recommendation (optional):**
   ```
   cp backend/coinbase.db backend/coinbase.db.bak_pre_tech_retirement_<ts>
   ```
   gitignored; just-in-case host-side backup.

3. **Atomic commit:**
   - All Phase A file changes staged together
   - Pre-commit hook runs full suite (~5 min)
   - On green, commit lands as `refactor(#311-refactor-c): delete TechAgent — phase A backend`

4. **Push:**
   ```
   git push
   ```

5. **Restart backend** (live launcher pickup) — TechAgent no longer instantiated.

### Phase B (frontend, immediately after A merges)

1. **Atomic commit:**
   - Frontend file changes + CHANGELOG entry
   - Hook skips (no Python staged)
   - Lands as `refactor(#311-refactor-d): delete TechAgent — phase B frontend`

2. **Push:**
   ```
   git push
   ```

3. **Frontend rebuild + reload** — operator does whatever their normal frontend dev cycle is.

### Rollback (full restoration)

If you want TechAgent back:
1. `git revert <Phase B commit> <Phase A commit>` — restores code + frontend in correct order
2. Restore `agent_state.positions_json` from the DB backup if you want the pre-retirement positions back:
   ```
   sqlite3 backend/coinbase.db \
     "UPDATE agent_state SET positions_json = '<json>', balance = <n>, realized_pnl = <n> WHERE agent='TECH'"
   ```
   (snapshot values printed by the preflight script and recorded in the CHANGELOG entry)
3. Restart backend

Time: ~5 minutes total.

## 9. Memory + CLAUDE.md sync (per CLAUDE.md rule)

Phase A commit bundles:
- `CHANGELOG.md` — Session 58.71d entry
- `CLAUDE.md` — Architecture section updated ("Active agents: CNN only")
- `memory/coinbase_trader_architecture.md` (outside repo) — Agents table updated; note that historical TECH rows remain in DB
- `memory/trading_app_architecture.md` (outside repo) — same

Phase B commit bundles:
- `CHANGELOG.md` — Session 58.71e entry

## 10. Open questions

None — operator approved every clarifying question on 2026-05-16.

## 11. References

- Module 1: `97dc8c9` (refactor #311-refactor-a) — dead env vars
- Module 2: `cff73a0` (refactor #311-refactor-b) — bare-isotonic calibrator
- Session 43 precedent for agent retirement: SCALP/MOMENTUM purge (#79)
- 5-day trade review (TICK_TRAIL +$46 finding): inline data from earlier in this session
- CLAUDE.md sync rule: `polymarket_app/CLAUDE.md` "CLAUDE.md ↔ Memory Sync Rule" section
