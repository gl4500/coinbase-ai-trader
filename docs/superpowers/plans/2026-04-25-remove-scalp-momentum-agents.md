# Remove ScalpAgent & MomentumAgent — Port Exit Stats to TechAgent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove ScalpAgent and MomentumAgent (backend + frontend + tests), and port the per-trigger exit-stats system from ScalpAgent into TechAgent's `_Book` so we keep the diagnostic value (which exit trigger leaks money).

**Architecture:**
- TechAgent's `_Book` gains a `_stats: Dict[str, Dict]` keyed by exit trigger (TICK_SIGNAL, TICK_STOP, TICK_TRAIL, TICK_PROFIT, SCAN). On every `sell()`, the trigger's wins/losses/total_pnl counters update. Exposed via `TechAgentCB.status["exit_stats"]`.
- ScalpAgent and MomentumAgent files, tests, main.py wiring, frontend components, and memory entries are removed wholesale.
- Confluence-reasons list is already present in TechAgent (`buy_reasons`/`sell_reasons` returned from `_score()`). No port needed for that; the plan only enhances the user-facing log/decision strings unchanged.

**Tech Stack:** Python 3.11 (asyncio, FastAPI, aiosqlite), pytest + pytest-asyncio, React + Vite + TypeScript + Tailwind, Vitest.

---

## Note on the "confluence reasons list"

TechAgent's `_score()` (backend/agents/tech_agent_cb.py:371) already returns `buy_reasons` and `sell_reasons` arrays — they are joined into the `reasoning` string saved to `agent_decisions` for every BUY/SELL/TICK exit. **No port needed.** The only stylistic difference vs ScalpAgent is that scalp annotates contributions like `RSI7=28 oversold (+2)`. Tech does not. We are NOT enhancing the format in this plan (out of scope per user instructions). If you want that later, file it separately.

---

## File Map

**Backend modify:**
- `backend/agents/tech_agent_cb.py` — add `_stats` to `_Book`, hook in `buy()`/`sell()`, expose in `status` (~30 LoC)
- `backend/main.py` — strip MomentumAgent + ScalpAgent imports, AppState fields, lifespan startup/shutdown, `/api/agents/status` payload, `/api/trades` query hint
- `backend/tests/test_tech_agent_cb.py` — drop the test that asserts "ADX/MFI moved to MomentumAgent"; add tests for `_stats` accumulation
- `backend/tests/test_startup_sequence.py` — drop ScalpAgent / Momentum delay tests

**Backend delete:**
- `backend/agents/scalp_agent.py`
- `backend/agents/momentum_agent_cb.py`
- `backend/tests/test_scalp_agent.py`
- `backend/tests/test_momentum_agent_cb.py`
- `backend/tests/test_momentum_entry_filter.py`

**Frontend modify:**
- `frontend/src/components/AgentsDashboard.tsx` — drop `momentum` and `scalp` cards/feeds (keep `tech` + `cnn`)
- `frontend/src/components/CNNDashboard.tsx` — drop "Mom" and "Scalp" columns from confidence table
- `frontend/src/components/FiringCounter.tsx` — drop MOM and SCALP rows; remove from counter state
- `frontend/src/components/PerformanceDashboard.tsx` — remove `MOMENTUM` and `SCALP` from `AgentFilter` union and color maps
- `frontend/src/utils/agentByProduct.ts` — drop `mom` and `scalp` fields; keep `tech`
- `frontend/src/utils/agentByProduct.test.ts` — update tests

**Misc cleanup:**
- `polymarket_app/_mom_sells.py` — DELETE
- `polymarket_app/launcher.py` — change "RSI · MACD · CNN · Momentum" subtitle to "RSI · MACD · CNN"
- `polymarket_app/README.md` — remove MomentumAgentCB and ScalpAgent rows from architecture diagram
- `polymarket_app/REBUILD_STANDARD.md` — remove all Mom/Scalp references in agent table, signal pipeline, dashboard columns, test files list, startup section

**Docs / memory:**
- `polymarket_app/CHANGELOG.md` — add Session entry
- `polymarket_app/CLAUDE.md` — drop "Scalp agent" mention in Architecture Quick Reference + remove Mom/Scalp test rows from Test Coverage table
- `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md` — remove MomentumAgentCB and ScalpAgent rows; remove ScalpAgent Exit Stats section; remove momentum signal pipeline step
- `~/.claude/projects/C--Users-gl450/memory/trading_app_thresholds.md` — remove Momentum + Scalp threshold rows
- `~/.claude/projects/C--Users-gl450/memory/MEMORY.md` — no entry change needed (those agents aren't indexed separately)

**DB:** Old `agent_state` rows for `MOMENTUM` and `SCALP` are left in place (dry-run only, won't be written to anymore). Old `agent_decisions` and `trades` rows are also left — historical record. No migration.

**Skip in this plan:** 5-min SL cooldown port (user deferred). Confluence-reasons format enhancement.

---

## Task 1: Add per-trigger exit stats to TechAgent's `_Book`

**Files:**
- Modify: `backend/agents/tech_agent_cb.py:98-177`
- Test: `backend/tests/test_tech_agent_cb.py`

The `_Book` already records `trigger_close` to the trades ledger (line 166). We add an in-memory per-trigger counter that survives the run and is exposed in `status`. We do NOT persist `_stats` to disk — they reset on restart, like ScalpAgent's behavior. Reasoning: `trades` ledger is the durable source of truth; `_stats` is a quick-glance diagnostic.

- [ ] **Step 1: Write failing test for `_stats` accumulation on sell**

Add to `backend/tests/test_tech_agent_cb.py` (append at end of file):

```python
import pytest
from unittest.mock import AsyncMock, patch
from agents.tech_agent_cb import _Book


@pytest.mark.asyncio
async def test_book_records_per_trigger_exit_stats():
    """_Book._stats must increment wins/losses/total_pnl by exit trigger."""
    book = _Book("TECH", balance=1000.0)

    with (
        patch("agents.tech_agent_cb.database.load_agent_state",  new=AsyncMock(return_value=None)),
        patch("agents.tech_agent_cb.database.save_agent_state",  new=AsyncMock()),
        patch("agents.tech_agent_cb.database.open_trade",        new=AsyncMock()),
        patch("agents.tech_agent_cb.database.close_trade",       new=AsyncMock()),
    ):
        await book.buy("BTC-USD", price=100.0, frac=0.10, trigger="SCAN")
        pnl_win = await book.sell("BTC-USD", price=110.0, trigger="TICK_PROFIT")
        assert pnl_win > 0

        await book.buy("ETH-USD", price=100.0, frac=0.10, trigger="SCAN")
        pnl_loss = await book.sell("ETH-USD", price=92.0, trigger="TICK_STOP")
        assert pnl_loss < 0

        stats = book._stats
        assert stats["TICK_PROFIT"]["wins"]   == 1
        assert stats["TICK_PROFIT"]["losses"] == 0
        assert stats["TICK_PROFIT"]["total_pnl"] == pytest.approx(pnl_win, abs=1e-6)

        assert stats["TICK_STOP"]["wins"]   == 0
        assert stats["TICK_STOP"]["losses"] == 1
        assert stats["TICK_STOP"]["total_pnl"] == pytest.approx(pnl_loss, abs=1e-6)


@pytest.mark.asyncio
async def test_status_exposes_exit_stats():
    """TechAgentCB.status must expose exit_stats with win_rate per trigger."""
    from agents.tech_agent_cb import TechAgentCB
    ag = TechAgentCB(ws_subscriber=None)

    with (
        patch("agents.tech_agent_cb.database.load_agent_state", new=AsyncMock(return_value=None)),
        patch("agents.tech_agent_cb.database.save_agent_state", new=AsyncMock()),
        patch("agents.tech_agent_cb.database.open_trade",       new=AsyncMock()),
        patch("agents.tech_agent_cb.database.close_trade",      new=AsyncMock()),
    ):
        await ag.book.buy("BTC-USD", price=100.0, frac=0.10, trigger="SCAN")
        await ag.book.sell("BTC-USD", price=110.0, trigger="TICK_PROFIT")
        await ag.book.buy("ETH-USD", price=100.0, frac=0.10, trigger="SCAN")
        await ag.book.sell("ETH-USD", price=92.0, trigger="TICK_STOP")

        st = ag.status
        assert "exit_stats" in st
        es = st["exit_stats"]
        assert es["TICK_PROFIT"]["wins"]      == 1
        assert es["TICK_PROFIT"]["win_rate"]  == 100.0
        assert es["TICK_STOP"]["losses"]      == 1
        assert es["TICK_STOP"]["win_rate"]    == 0.0
        assert "TICK_TRAIL" not in es  # only triggers with closed trades appear
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/test_tech_agent_cb.py::test_book_records_per_trigger_exit_stats tests/test_tech_agent_cb.py::test_status_exposes_exit_stats -v
```

Expected: FAIL — `_stats` attribute does not exist; `exit_stats` not in status.

- [ ] **Step 3: Add `_stats` to `_Book.__init__`**

In `backend/agents/tech_agent_cb.py`, modify `_Book.__init__` (currently lines 99-103):

```python
class _Book:
    def __init__(self, agent_name: str, balance: float = _DRY_RUN_BALANCE):
        self._agent      = agent_name
        self.balance     = balance
        self.positions: Dict[str, Dict] = {}  # pid → {size, avg_price}
        self.realized_pnl = 0.0
        # Per-trigger exit stats — diagnostic only, not persisted (trades ledger is durable)
        self._stats: Dict[str, Dict] = {
            trigger: {"wins": 0, "losses": 0, "total_pnl": 0.0}
            for trigger in ("TICK_SIGNAL", "TICK_STOP", "TICK_TRAIL", "TICK_PROFIT", "SCAN")
        }
```

- [ ] **Step 4: Hook `_stats` update into `_Book.sell`**

In `backend/agents/tech_agent_cb.py`, modify `_Book.sell` (currently lines 155-170). Add the stats update after PnL is computed, before the return:

```python
    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        """Close position. Returns realized PnL. Writes to trades ledger."""
        if pid not in self.positions:
            return 0.0
        pos = self.positions.pop(pid)
        proceeds = pos["size"] * price
        pnl = proceeds - pos["size"] * pos["avg_price"]
        self.balance += proceeds
        self.realized_pnl += pnl

        # Per-trigger stats (diagnostic — trades table is durable)
        bucket = self._stats.setdefault(
            trigger, {"wins": 0, "losses": 0, "total_pnl": 0.0}
        )
        if pnl > 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
        bucket["total_pnl"] += pnl

        await self._save()
        await database.close_trade(
            agent=self._agent, product_id=pid, exit_price=price,
            size=pos["size"], pnl=pnl, trigger_close=trigger,
            balance_after=self.balance,
        )
        return pnl
```

- [ ] **Step 5: Expose `exit_stats` in `TechAgentCB.status`**

In `backend/agents/tech_agent_cb.py`, modify the `status` property (currently lines 613-628). Append `exit_stats` to the returned dict:

```python
    @property
    def status(self) -> Dict:
        return {
            "agent":         "TECH",
            "balance":       round(self.book.balance, 2),
            "realized_pnl":  round(self.book.realized_pnl, 2),
            "open_positions": len(self.book.positions),
            "positions":     {
                pid: {"size": round(p["size"], 6), "avg_price": round(p["avg_price"], 6)}
                for pid, p in self.book.positions.items()
            },
            "scan_count":    self.scan_count,
            "signals_buy":   self.signals_buy,
            "signals_sell":  self.signals_sell,
            "last_scan_at":  self.last_scan_at,
            "exit_stats":    {
                trigger: {
                    "wins":      s["wins"],
                    "losses":    s["losses"],
                    "win_rate":  round(s["wins"] / max(s["wins"] + s["losses"], 1) * 100, 1),
                    "total_pnl": round(s["total_pnl"], 2),
                }
                for trigger, s in self.book._stats.items()
                if s["wins"] + s["losses"] > 0
            },
        }
```

- [ ] **Step 6: Run tests to verify GREEN**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/test_tech_agent_cb.py -v
```

Expected: PASS for both new tests + all existing tests.

- [ ] **Step 7: Cleanup background python processes**

```bash
ps aux | grep python | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 8: Commit**

```bash
git add backend/agents/tech_agent_cb.py backend/tests/test_tech_agent_cb.py
git commit -m "$(cat <<'EOF'
feat: port per-trigger exit stats from ScalpAgent into TechAgent _Book

Tracks wins/losses/total_pnl per close trigger (TICK_SIGNAL/TICK_STOP/
TICK_TRAIL/TICK_PROFIT/SCAN) and exposes via /api/agents/status.exit_stats.
Diagnostic only, not persisted -- trades ledger is durable source.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Delete MomentumAgent backend + tests

**Files:**
- Delete: `backend/agents/momentum_agent_cb.py`
- Delete: `backend/tests/test_momentum_agent_cb.py`
- Delete: `backend/tests/test_momentum_entry_filter.py`

- [ ] **Step 1: Delete the three files**

```bash
rm /c/Users/gl450/polymarket_app/backend/agents/momentum_agent_cb.py
rm /c/Users/gl450/polymarket_app/backend/tests/test_momentum_agent_cb.py
rm /c/Users/gl450/polymarket_app/backend/tests/test_momentum_entry_filter.py
```

- [ ] **Step 2: Verify no other backend module imports MomentumAgent**

```bash
cd /c/Users/gl450/polymarket_app
grep -rn "MomentumAgentCB\|momentum_agent_cb" backend/ --include='*.py'
```

Expected: matches only in `backend/main.py` (will be cleaned up in Task 4).

- [ ] **Step 3: Commit**

```bash
git add -A backend/agents/momentum_agent_cb.py backend/tests/test_momentum_agent_cb.py backend/tests/test_momentum_entry_filter.py
git commit -m "refactor: delete MomentumAgent backend + tests

Removed in favor of TechAgent which now covers RSI/BB/MACD/Stoch/OBV
on the 2-min cadence with TICK_TRAIL exits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Delete ScalpAgent backend + tests

**Files:**
- Delete: `backend/agents/scalp_agent.py`
- Delete: `backend/tests/test_scalp_agent.py`

- [ ] **Step 1: Delete the two files**

```bash
rm /c/Users/gl450/polymarket_app/backend/agents/scalp_agent.py
rm /c/Users/gl450/polymarket_app/backend/tests/test_scalp_agent.py
```

- [ ] **Step 2: Verify no other backend module imports ScalpAgent**

```bash
cd /c/Users/gl450/polymarket_app
grep -rn "ScalpAgent\|scalp_agent" backend/ --include='*.py'
```

Expected: matches only in `backend/main.py` and `backend/tests/test_startup_sequence.py` (cleaned up in next steps).

- [ ] **Step 3: Commit**

```bash
git add -A backend/agents/scalp_agent.py backend/tests/test_scalp_agent.py
git commit -m "refactor: delete ScalpAgent backend + tests

Removed -- per-trigger exit-stats system was the only piece of value
and has been ported to TechAgent's _Book in the previous commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Strip Momentum + Scalp wiring from `backend/main.py`

**Files:**
- Modify: `backend/main.py`

Make all edits in one pass. The file has been read; here are the exact replacements.

- [ ] **Step 1: Remove imports (lines 52–53)**

Change:
```python
from agents.tech_agent_cb import TechAgentCB
from agents.momentum_agent_cb import MomentumAgentCB
from agents.scalp_agent import ScalpAgent
```
to:
```python
from agents.tech_agent_cb import TechAgentCB
```

- [ ] **Step 2: Remove AppState fields (lines 200–201, 208–209)**

Delete these four lines from `class AppState`:
```python
    momentum_agent:  MomentumAgentCB       = None
    scalp_agent:     ScalpAgent            = None
```
```python
    momentum_task:   asyncio.Task          = None
    scalp_task:      asyncio.Task          = None
```

- [ ] **Step 3: Remove startup-stagger constants (lines 397–398)**

Delete these two lines:
```python
_MOMENTUM_START_DELAY = 10   # Momentum after 10s
_SCALP_START_DELAY    = 15   # Scalp after 15s  (its _entry_loop has no extra warmup)
```

- [ ] **Step 4: Remove agent instantiation (lines 419–420)**

Delete these two lines from the lifespan startup:
```python
    app_state.momentum_agent  = MomentumAgentCB(ws_subscriber=app_state.ws_subscriber)
    app_state.scalp_agent     = ScalpAgent(ws_subscriber=app_state.ws_subscriber, is_trading_fn=_is_trading)
```

- [ ] **Step 5: Remove delayed-launch coroutines (lines 484–494)**

Delete these blocks:
```python
    async def _delayed_momentum():
        await asyncio.sleep(_MOMENTUM_START_DELAY)
        await app_state.momentum_agent.run_loop(is_trading_fn=_is_trading)

    async def _delayed_scalp():
        await asyncio.sleep(_SCALP_START_DELAY)
        await app_state.scalp_agent.start()
```

And change:
```python
    app_state.tech_task     = asyncio.create_task(_delayed_tech())
    app_state.momentum_task = asyncio.create_task(_delayed_momentum())
    app_state.scalp_task          = asyncio.create_task(_delayed_scalp())
    app_state.train_watcher_task  = asyncio.create_task(_train_progress_watcher())
```
to:
```python
    app_state.tech_task          = asyncio.create_task(_delayed_tech())
    app_state.train_watcher_task = asyncio.create_task(_train_progress_watcher())
```

- [ ] **Step 6: Remove shutdown wiring (lines 504–509)**

Change:
```python
    if app_state.scalp_agent:
        await app_state.scalp_agent.stop()
    for task in [
        app_state.scanner_task, app_state.portfolio_task,
        app_state.cnn_task, app_state.tech_task, app_state.momentum_task,
        app_state.scalp_task, app_state.outcome_task, app_state.backfill_task,
        app_state.train_watcher_task,
    ]:
```
to:
```python
    for task in [
        app_state.scanner_task, app_state.portfolio_task,
        app_state.cnn_task, app_state.tech_task,
        app_state.outcome_task, app_state.backfill_task,
        app_state.train_watcher_task,
    ]:
```

- [ ] **Step 7: Update `/api/agents/status` payload (lines 1064–1147)**

a) Delete line 1067:
```python
    mom_status  = app_state.momentum_agent.status if app_state.momentum_agent else {}
```

b) Delete the SCALP block (lines 1089–1092):
```python
    # SCALP book status
    scalp_status: Dict = {}
    if app_state.scalp_agent:
        scalp_status = app_state.scalp_agent.status
```

c) On line 1099 change:
```python
    for st in (tech_status, mom_status, cnn_status, scalp_status):
```
to:
```python
    for st in (tech_status, cnn_status):
```

d) On line 1129 the same change:
```python
    for st in (tech_status, mom_status, cnn_status, scalp_status):
```
→
```python
    for st in (tech_status, cnn_status):
```

e) On line 1147 change:
```python
    return {"tech": tech_status, "momentum": mom_status, "cnn": cnn_status, "scalp": scalp_status}
```
to:
```python
    return {"tech": tech_status, "cnn": cnn_status}
```

- [ ] **Step 8: Update `/api/trades` query hint (line 1311)**

Change:
```python
    agent:        Optional[str] = Query(None, description="TECH or MOMENTUM"),
```
to:
```python
    agent:        Optional[str] = Query(None, description="TECH or CNN"),
```

- [ ] **Step 9: Update `backend/tests/test_startup_sequence.py`**

The file currently tests ScalpAgent warmup and Momentum/Scalp start-delay constants. Replace the file contents with:

```python
"""
Startup-sequence tests for the Coinbase Trader.

Verifies that:
  - Tech start-delay constant is sane (≤ 30s)
"""
import importlib


def test_tech_start_delay_is_sane():
    """_TECH_START_DELAY in main.py must be ≤ 30s so users see signals quickly."""
    m = importlib.import_module("main")
    delay = getattr(m, "_TECH_START_DELAY", None)
    assert delay is not None,         "_TECH_START_DELAY missing from main.py"
    assert isinstance(delay, int),    f"_TECH_START_DELAY must be int, got {type(delay)}"
    assert 0 < delay <= 30,           f"_TECH_START_DELAY {delay}s out of range (0, 30]"
```

- [ ] **Step 10: Update `backend/tests/test_tech_agent_cb.py` docstring + the ADX/MFI test**

Drop the test `test_score_does_not_use_adx_or_mfi` (it asserted "moved to MomentumAgent" — moot now). And in the module docstring at the top of `backend/tests/test_tech_agent_cb.py`, remove the comment about ADX/MFI being in `momentum_agent_cb`.

Find this block (around lines 1–10):
```python
"""
Tests for TechAgentCB — mean-reversion crypto scalper.
...
  ADX and MFI are NOT part of TechAgent (they were removed; see momentum_agent_cb).
"""
```
Replace with:
```python
"""
Tests for TechAgentCB — mean-reversion crypto scalper.
"""
```

Then find and DELETE the test that mentions `MomentumAgent`:
```python
    def test_score_does_not_use_adx_or_mfi(self):
        """ADX and MFI must NOT appear in TechAgent score (moved to MomentumAgent)."""
        ...
```
(Remove the entire test method.)

- [ ] **Step 11: Run full backend tests**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/ -v 2>&1 | tail -60
```

Expected: all tests pass; no `ModuleNotFoundError: No module named 'agents.scalp_agent'` or `... 'agents.momentum_agent_cb'`.

- [ ] **Step 12: Cleanup background python processes**

```bash
ps aux | grep python | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 13: Commit**

```bash
git add backend/main.py backend/tests/test_startup_sequence.py backend/tests/test_tech_agent_cb.py
git commit -m "refactor: remove Momentum + Scalp wiring from main.py and stale tests

main.py: drop imports, AppState fields, lifespan startup/shutdown,
/api/agents/status payload entries, /api/trades query description.
test_startup_sequence: keep only Tech-delay sanity check.
test_tech_agent_cb: drop stale 'moved to MomentumAgent' assertion.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Strip Momentum + Scalp from frontend

**Files:**
- Modify: `frontend/src/utils/agentByProduct.ts`
- Modify: `frontend/src/utils/agentByProduct.test.ts`
- Modify: `frontend/src/components/AgentsDashboard.tsx`
- Modify: `frontend/src/components/CNNDashboard.tsx`
- Modify: `frontend/src/components/FiringCounter.tsx`
- Modify: `frontend/src/components/PerformanceDashboard.tsx`

This task uses the `frontend-design` skill principles for any structural change to keep the layout cohesive after column removal.

- [ ] **Step 1: Update `agentByProduct.ts`**

Replace the entire file with:

```typescript
/**
 * Pure function extracted from CNNDashboard's agentByProduct useMemo.
 * Builds a per-product map of the latest decision from each agent.
 * Exported so it can be unit tested without mounting the component.
 */

export interface AgentDecision {
  id:         number
  agent:      string        // TECH (only -- Momentum and Scalp removed)
  product_id: string
  side:       string        // BUY | SELL | HOLD
  confidence: number
  price:      number
  score:      number | null
  reasoning:  string | null
  balance:    number | null
  pnl:        number | null
  created_at: string
}

export interface AgentVotes {
  tech: AgentDecision | null
}

export function buildAgentByProduct(decisions: AgentDecision[]): Map<string, AgentVotes> {
  const map = new Map<string, AgentVotes>()
  for (const d of decisions) {
    const cur = map.get(d.product_id) ?? { tech: null }
    if (d.agent === 'TECH' && cur.tech === null) cur.tech = d
    map.set(d.product_id, cur)
  }
  return map
}
```

- [ ] **Step 2: Update `agentByProduct.test.ts`**

Read the existing file first, then drop any test that asserts `mom` or `scalp` keys; keep only `tech` assertions.

```bash
cat /c/Users/gl450/polymarket_app/frontend/src/utils/agentByProduct.test.ts
```

Edit the file to:
- remove fixtures with `agent: 'MOMENTUM'` or `agent: 'SCALP'`
- remove any expect(...).mom / .scalp assertions
- keep tests for `.tech` and for "decision with unknown agent is ignored"

- [ ] **Step 3: Update `CNNDashboard.tsx`**

a) Remove the `Mom` and `Scalp` `<th>` columns (lines ~918–919):
```tsx
<th className="px-3 py-2 text-left text-blue-400">Mom</th>
<th className="px-3 py-2 text-left text-amber-400">Scalp</th>
```
Delete both.

b) Remove the corresponding `<td>` cells. Find the "Momentum agent vote" block (around line 1067) and the "Scalp agent vote" block (around line 1096) and delete each block in full (the `<td>` plus the surrounding `{(() => { … })()}` helper).

c) Update the comment around line 312:
```tsx
// Per-product lookup: latest Tech, Momentum & Scalp decision (agentDecisions is newest-first)
```
to:
```tsx
// Per-product lookup: latest Tech decision (agentDecisions is newest-first)
```

- [ ] **Step 4: Update `AgentsDashboard.tsx`**

a) Replace the agentStatus state + fetch (line 104):
```tsx
const [agentStatus, setAgentStatus] = useState<{ tech: SubAgentStatus | null; momentum: SubAgentStatus | null; cnn: SubAgentStatus | null; scalp: SubAgentStatus | null }>({ tech: null, momentum: null, cnn: null, scalp: null })
```
to:
```tsx
const [agentStatus, setAgentStatus] = useState<{ tech: SubAgentStatus | null; cnn: SubAgentStatus | null }>({ tech: null, cnn: null })
```

And the fetch handler (line 112):
```tsx
setAgentStatus({ tech: d.tech ?? null, momentum: d.momentum ?? null, cnn: d.cnn ?? null, scalp: d.scalp ?? null })
```
→
```tsx
setAgentStatus({ tech: d.tech ?? null, cnn: d.cnn ?? null })
```

b) Remove `momSignals`, `momAg`, `scalpAg`:
```tsx
const momSignals  = useMemo(() => signals.filter(d => d.agent === 'MOMENTUM'), [signals])
```
delete entirely; same for `momAg` and `scalpAg`.

c) Update aggregate totals (line 146):
```tsx
const totalPnl  = (techAg?.realized_pnl ?? 0) + (momAg?.realized_pnl ?? 0) + (cnnAg?.realized_pnl ?? 0) + (scalpAg?.realized_pnl ?? 0)
```
→
```tsx
const totalPnl  = (techAg?.realized_pnl ?? 0) + (cnnAg?.realized_pnl ?? 0)
```

And the open-positions sum (line 166):
```tsx
value={(techAg?.open_positions ?? 0) + (momAg?.open_positions ?? 0) + (cnnAg?.open_positions ?? 0) + (scalpAg?.open_positions ?? 0)}
```
→
```tsx
value={(techAg?.open_positions ?? 0) + (cnnAg?.open_positions ?? 0)}
```

d) Update headline strings:
- Line 161: `sub="Tech + Momentum + CNN + Scalp"` → `sub="Tech + CNN"`
- Line 172: `sub={`${techSignals.length} tech · ${momSignals.length} mom`}` → `sub={`${techSignals.length} tech`}`

e) Replace the agent map (line 178):
```tsx
{(['tech', 'momentum', 'cnn', 'scalp'] as const).map(key => {
```
→
```tsx
{(['tech', 'cnn'] as const).map(key => {
```

And inside, simplify the `label` and `color` ternaries to handle only `tech` / `cnn`:
- `const label = key === 'tech' ? 'TechAgent' : 'CNN Agent'`
- `const color = key === 'tech' ? 'text-purple-400' : 'text-yellow-400'`
- Remove all `key === 'momentum'` branches throughout (around lines 184, 227, 247, 284).

f) Delete the entire "Momentum signals" feed block (around lines 392–460):
```tsx
{/* Momentum signals */}
<div ...>
  <h3>... Momentum Signals</h3>
  ...
</div>
```
The Tech feed remains; with Momentum gone, expand the Tech feed to take the full width (remove the parent grid that splits left/right, or set the Tech feed to span full width).

g) Drop unused `high_water` field at line 11:
```tsx
high_water?: number   // Momentum only
```

- [ ] **Step 5: Update `FiringCounter.tsx`**

a) Remove from counts type (lines 17–18):
```tsx
scalp_scans:      number
scalp_signals:    number
```

b) Remove `mom_scans` / `mom_signals` if present in the same type block.

c) Update default state (line 25):
```tsx
scalp_scans: 0, scalp_signals: 0,
```
delete; also delete `mom_scans: 0, mom_signals: 0,`.

d) Remove from fetch (lines 52–53):
```tsx
const mom   = agents.momentum   ?? {}
const scalp = agents.scalp      ?? {}
```
delete both. And lines 67–70:
```tsx
mom_scans:        mom.scan_count                                  ?? 0,
mom_signals:      (mom.signals_buy  ?? 0) + (mom.signals_sell  ?? 0),
scalp_scans:      scalp.scan_count                                ?? 0,
scalp_signals:    (scalp.signals_buy ?? 0) + (scalp.signals_sell ?? 0),
```
delete all four.

e) Remove the Momentum row block (around lines 121–128) and the Scalp row block (around lines 130–135).

- [ ] **Step 6: Update `PerformanceDashboard.tsx`**

a) Line 88:
```tsx
type AgentFilter = 'ALL' | 'TECH' | 'MOMENTUM' | 'CNN' | 'SCALP'
```
→
```tsx
type AgentFilter = 'ALL' | 'TECH' | 'CNN'
```

b) Lines 107, 114: remove the `MOMENTUM:` and `SCALP:` keys from both color maps.

c) Line 248:
```tsx
const AGENTS: AgentFilter[] = ['ALL', 'CNN', 'MOMENTUM', 'SCALP', 'TECH']
```
→
```tsx
const AGENTS: AgentFilter[] = ['ALL', 'CNN', 'TECH']
```

- [ ] **Step 7: Run frontend tests**

```bash
cd /c/Users/gl450/polymarket_app/frontend
npm test -- --run
```

Expected: all `agentByProduct.test.ts` tests pass; no TypeScript errors.

- [ ] **Step 8: Manual visual check (REQUIRED for frontend)**

```bash
cd /c/Users/gl450/polymarket_app/frontend
npm run dev
```

Open `http://localhost:5174`. Verify:
- Agents tab: only TechAgent + CNN Agent cards visible. Tech feed spans full width (or layout looks balanced).
- CNN tab confidence table: Mom + Scalp columns gone. Tech column visible.
- FiringCounter widget: only TECH and CNN rows.
- Performance tab: agent filter chips show only ALL / TECH / CNN.
- No console errors.

If anything looks broken, **stop and fix** before committing.

- [ ] **Step 9: Commit**

```bash
git add frontend/
git commit -m "refactor: remove Momentum + Scalp from frontend

- AgentsDashboard: drop Mom + Scalp cards and feeds; aggregate over TECH+CNN only
- CNNDashboard: drop Mom + Scalp columns from confidence table
- FiringCounter: drop MOM + SCALP rows
- PerformanceDashboard: drop MOMENTUM + SCALP from AgentFilter union and color maps
- agentByProduct: AgentVotes now only has tech

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Cleanup support files

**Files:**
- Delete: `_mom_sells.py`
- Modify: `launcher.py`
- Modify: `README.md`
- Modify: `REBUILD_STANDARD.md`

- [ ] **Step 1: Delete the Momentum debug script**

```bash
rm /c/Users/gl450/polymarket_app/_mom_sells.py
```

- [ ] **Step 2: Update `launcher.py` subtitle (line 319)**

Change:
```python
tk.Label(title_frame, text="Advanced Trade · RSI · MACD · CNN · Momentum",
```
to:
```python
tk.Label(title_frame, text="Advanced Trade · RSI · MACD · CNN",
```

- [ ] **Step 3: Update `README.md` architecture diagram (lines 70–71)**

Remove the lines:
```
                                         │  MomentumAgentCB            │
                                         │  ScalpAgent                 │
```
And renumber/clean any surrounding box-drawing characters so the diagram remains valid.

- [ ] **Step 4: Update `REBUILD_STANDARD.md`**

Search and clean. Remove all Mom/Scalp references at lines 20, 180–186, 210–219, 291, 381–382, 422.

```bash
grep -n -i "scalp\|momentum" /c/Users/gl450/polymarket_app/REBUILD_STANDARD.md
```

For each match, remove or rewrite the surrounding paragraph so only TechAgent + CNN are described. Specifically:
- Remove the `momentum_agent_cb.py` row in the file-tree at line 20.
- Delete the entire `### MomentumAgentCB` section (from line 180 to next `###`).
- Delete the `MomentumAgent (1 min scan)` and `ScalpAgent (5 sec tick loop)` blocks in the signal-flow diagram at lines 210–219.
- Delete the `**Mom**` row in the dashboard column table at line 291.
- Delete the `test_momentum_agent_cb.py` and `test_scalp_agent.py` rows in the test-files list at 381–382.
- Update line 422 startup table to drop `Momentum +60s staggered start`.

- [ ] **Step 5: Commit**

```bash
git add launcher.py README.md REBUILD_STANDARD.md _mom_sells.py
git commit -m "docs: scrub Momentum + Scalp from launcher subtitle, README, REBUILD_STANDARD

Also delete _mom_sells.py debug script -- Momentum agent is gone.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Update CHANGELOG, CLAUDE.md, and memory

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md`
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`
- Modify: `~/.claude/projects/C--Users-gl450/memory/trading_app_thresholds.md`

- [ ] **Step 1: Add CHANGELOG entry**

At the top of `polymarket_app/CHANGELOG.md`, add:

```markdown
## Session 39 — Remove Momentum + Scalp Agents (2026-04-25)

**Why:** TechAgent (with TICK_TRAIL exits added in Session 38) covers the same RSI/BB/MACD/Stoch/OBV signal space as Momentum on a 2-min cadence; Scalp's only durable contribution was its per-trigger exit-stats system.

**Changes:**
- TechAgent `_Book` gains `_stats: Dict[str, Dict]` — per-trigger wins/losses/total_pnl, exposed via `/api/agents/status.exit_stats`. Diagnostic only, not persisted (trades ledger is durable).
- Deleted `agents/momentum_agent_cb.py`, `agents/scalp_agent.py`.
- Deleted tests: `test_momentum_agent_cb.py`, `test_momentum_entry_filter.py`, `test_scalp_agent.py`.
- `main.py`: stripped imports, AppState fields, lifespan startup/shutdown, `/api/agents/status` payload entries, `/api/trades` query description.
- `tests/test_startup_sequence.py`: kept only Tech-delay sanity.
- Frontend: removed Momentum + Scalp from `AgentsDashboard`, `CNNDashboard` (Mom/Scalp columns), `FiringCounter`, `PerformanceDashboard`, `agentByProduct` utility.
- Cleanup: `launcher.py` subtitle, `README.md` and `REBUILD_STANDARD.md` agent diagrams, `_mom_sells.py` deleted.

**Out of scope:** ScalpAgent's 5-min SL cooldown was NOT ported. ScalpAgent's confluence-reasons format with `(+score)` annotations was NOT applied to TechAgent (Tech already has its own reasons list).

**DB state:** historical `MOMENTUM` / `SCALP` rows in `agent_state`, `agent_decisions`, `trades`, `signal_outcomes` are left as-is. No migration.
```

- [ ] **Step 2: Update `CLAUDE.md` Architecture Quick Reference**

Find the "AI agents" line (around line ~150 in the Architecture Quick Reference section):
```
- **AI agents:** CNN agent (PyTorch), Tech agent, Momentum agent, Scalp agent
```
→
```
- **AI agents:** CNN agent (PyTorch), Tech agent
```

And in the Test Coverage table, remove the rows for `momentum_agent_cb.py` and any scalp/momentum-specific test files.

- [ ] **Step 3: Update `coinbase_trader_architecture.md` memory**

```bash
$EDITOR /c/Users/gl450/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md
```

Edits:
- Agents table: delete the `MomentumAgentCB` and `ScalpAgent` rows (and add a note that TechAgent is the only static-indicator agent now).
- Signal Pipeline section: delete step 2 (Momentum) and renumber.
- TechAgent section: append `Exit stats: per-trigger wins/losses/total_pnl tracked in _Book._stats; exposed via status["exit_stats"]. Triggers: TICK_SIGNAL, TICK_STOP, TICK_TRAIL, TICK_PROFIT, SCAN.`
- Delete the entire `## ScalpAgent Exit Stats` section.
- Delete the entire `## MomentumAgent Indicators` section.
- Delete the `_Book("MOMENTUM")` and `_Book("SCALP")` references in any Constants table.

- [ ] **Step 4: Update `trading_app_thresholds.md` memory**

```bash
$EDITOR /c/Users/gl450/.claude/projects/C--Users-gl450/memory/trading_app_thresholds.md
```

Edits:
- Remove all rows for Momentum thresholds (score ≥ 0.45 BUY / ≥ 0.30 SELL, RSI < 65, ADX ≥ 20, 3% trailing stop) and any Scalp thresholds (TP/SL/TIME/MIN_SCORE constants).

- [ ] **Step 5: Commit**

```bash
git add polymarket_app/CHANGELOG.md polymarket_app/CLAUDE.md
git commit -m "docs: CHANGELOG + CLAUDE.md update for Momentum/Scalp removal

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

(Memory files are outside the git repo and not committed; they are saved in step 3–4.)

---

## Task 8: Verification before completion

Use the `superpowers:verification-before-completion` skill — show evidence, do not assert.

- [ ] **Step 1: Confirm no stale imports remain**

```bash
cd /c/Users/gl450/polymarket_app
grep -rn "MomentumAgentCB\|momentum_agent_cb\|ScalpAgent\|scalp_agent\.py" backend/ frontend/src/ --include='*.py' --include='*.ts' --include='*.tsx'
```

Expected: empty output (modulo the historical CHANGELOG.md entry).

- [ ] **Step 2: Run full backend test suite once**

```bash
cd /c/Users/gl450/polymarket_app/backend
python -m pytest tests/ -v 2>&1 | tail -40
```

Expected: all green; no `ModuleNotFoundError`.

- [ ] **Step 3: Run full frontend tests**

```bash
cd /c/Users/gl450/polymarket_app/frontend
npm test -- --run
```

Expected: all green.

- [ ] **Step 4: Boot the backend and confirm `/api/agents/status` shape**

```bash
cd /c/Users/gl450/polymarket_app
.venv/Scripts/python.exe backend/main.py &
sleep 8
curl -s http://localhost:8001/api/agents/status | python -m json.tool | head -20
kill %1
```

Expected: top-level keys are `tech` and `cnn` only. `tech.exit_stats` field present (may be empty dict if no trades closed yet).

- [ ] **Step 5: Cleanup background python processes**

```bash
ps aux | grep python | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
```

- [ ] **Step 6: Push to GitHub**

```bash
cd /c/Users/gl450/polymarket_app
git push
```

---

## Self-Review Checklist (do not skip)

1. **Spec coverage:** every user request mapped to a task?
   - "Remove ScalpAgent backend + frontend": Tasks 3, 4, 5, 6 ✓
   - "Port per-trigger exit stats to TechAgent": Task 1 ✓
   - "Confluence reasons list" — already in TechAgent; called out in plan header ✓
   - "Don't port 5-min SL cooldown": noted as out of scope in plan + CHANGELOG ✓
   - "Remove MomentumAgent": Tasks 2, 4, 5, 6 ✓

2. **Placeholder scan:** no "TBD", "TODO", "implement later" — all code shown.

3. **Type consistency:** `_stats` dict shape `{trigger: {wins, losses, total_pnl}}` consistent across `_Book.__init__`, `_Book.sell`, and `status` property. `exit_stats` shape consistent with test assertions.
