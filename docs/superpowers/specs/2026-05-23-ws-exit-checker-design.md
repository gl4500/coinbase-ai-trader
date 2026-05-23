# WS-Driven Exit Checker — Design Spec

**Date:** 2026-05-23
**Author:** Claude Opus 4.7 (Session 58.71m, post-CNN-driver-removal)
**Status:** Approved (operator 2026-05-23)
**Related:** `2026-05-23-remove-cnn-driver-add-v45-driver-design.md` (predecessor — established `XGB` / `XGB_V45` as the only backends)

---

## Problem

Trail-stop and stop-loss exits fire only at scan-loop cadence. Cadence was 300s on 2026-05-23 morning, dropped to 60s mid-session for faster reactivity, but even 60s is too slow for the operator's primary concern: rapid intra-cycle reversals on a held position can give back significant gains before the next scan check. The exit-decision math is fast; the bottleneck is the polling interval.

The Coinbase Advanced Trade WebSocket ticker channel is already subscribed for every tracked product (frontend price-update broadcasts use it). `services/ws_subscriber.py:33` exposes `register_price_handler(async fn(pid, price))` callbacks fired per-tick via `asyncio.create_task` — but no callback is registered for exit logic today.

## Goal

Fire trail-stop and stop-loss exits with sub-second latency by registering a price-tick handler that reads from `_CNNBook.positions`, evaluates the existing exit conditions, and calls `_CNNBook.sell(pid, price, trigger=...)`. Max-hold (7-day timeout) stays on the scan loop where 60s cadence is appropriate.

## Non-Goals

- Redesigning the max-hold exit (see backlog task #26 — separate spec when prioritized).
- Replacing the scan-loop `_check_risk_exits`. WS handler is additive; scan loop continues to handle exits as a safety net and as the canonical owner of ATR-based trail_pct recomputation.
- New WebSocket connection management. Existing `CoinbaseWSSubscriber` instance is reused.
- Changing peak_price persistence semantics. WS handler ratchets peak in-memory; scan loop continues to persist via `book._save()` on close.

---

## Architecture

### File map

```
NEW   backend/agents/exit_watcher.py        ~80 LoC, async price-tick handler
EDIT  backend/agents/cnn_agent.py           per-pid lock in _CNNBook.sell();
                                            cache trail_pct on pos in _check_risk_exits
EDIT  backend/main.py                       lifespan registers handler after WS start
NEW   backend/tests/test_exit_watcher.py    6 unit + 1 integration test
EDIT  backend/tests/test_cnn_agent.py       1 new race test for sell() lock
EDIT  backend/tests/test_cnn_risk_exits.py  1 new test for trail_pct cache write
```

### Dependency graph (one-way, no back-edges)

```
exit_watcher  →  CoinbaseWSSubscriber.register_price_handler
exit_watcher  →  _CNNBook.sell + _CNNBook.positions (read-only)
main.lifespan →  exit_watcher.attach
```

`ws_subscriber` does not know about `exit_watcher`. `cnn_agent` does not know about `exit_watcher`. Loose coupling per `feedback_loose_coupling.md`.

### Module boundary — `agents/exit_watcher.py`

Single async function plus one attach helper. No state of its own.

```python
async def on_price_tick(pid: str, price: float, book: _CNNBook) -> None:
    """Fire WS_TRAIL_STOP / WS_STOP_LOSS on a held position. No-op otherwise."""

def attach(ws_subscriber: CoinbaseWSSubscriber, book: _CNNBook) -> None:
    """Register the handler. Call once per backend lifespan."""
```

---

## Data Flow

```
Coinbase WS ──► CoinbaseWSSubscriber._handle (existing)
                  • updates self.state[pid]
                  • broadcasts to frontend (existing)
                  • fires registered handlers via asyncio.create_task (existing)
                       │
                       ▼
                exit_watcher.on_price_tick(pid, price, book)         [NEW]
                  │
                  ├─ pos = book.positions.get(pid)
                  ├─ if pos is None: return                          (≈99% of ticks)
                  │
                  ├─ peak = pos.get('peak_price') or pos['avg_price']
                  ├─ if price > peak: pos['peak_price'] = price      (ratchet)
                  │
                  ├─ pct_entry     = (price - pos['avg_price']) / pos['avg_price']
                  ├─ pct_from_peak = (price - peak) / peak
                  ├─ trail_pct     = pos.get('trail_pct', _CNN_ATR_TRAIL_MIN)
                  │
                  ├─ trigger = None
                  ├─ if pct_entry     <= -_CNN_STOP_LOSS_PCT: trigger = 'WS_STOP_LOSS'
                  ├─ elif pct_from_peak <= -trail_pct:        trigger = 'WS_TRAIL_STOP'
                  │
                  └─ if trigger:
                       await book.sell(pid, price, trigger=trigger)
                         │
                         ▼
                       _CNNBook.sell                                 (cnn_agent.py:239)
                         async with self._lock_for(pid):             [NEW]
                            if pid not in self.positions: return 0.0
                            ... existing sell body unchanged ...
```

### Key properties

1. **Hot-path filter.** `book.positions.get(pid)` is a single dict lookup; ≈99% of ticks are not for held products and return immediately. No CPU concern at 580 ticks/sec.
2. **Peak ratcheting on WS.** Currently `_check_risk_exits` ratchets peak once per scan. WS handler ratchets per tick — peak is always fresh, trail-stop never lags reality. Free upgrade.
3. **Trigger names distinguish source.** `WS_TRAIL_STOP` / `WS_STOP_LOSS` vs scan loop's `TRAIL_STOP` / `STOP_LOSS`. Telemetry baked in: per-trigger counts measure latency reduction.
4. **No DB calls in handler.** `trail_pct` comes from `pos['trail_pct']` (cached by scan). ATR not recomputed per tick. Only `book.sell()` writes to DB, inside the lock.
5. **Stop-loss precedes trail-stop.** When both fire, `WS_STOP_LOSS` wins. Matches priority order documented at `cnn_agent.py:1671-1674`.

---

## Race Handling

### Hazard

Both `on_price_tick` and scan-loop `_check_risk_exits` can decide to exit the same position within microseconds of each other. `_CNNBook.sell()` (line 240) has an in-memory guard `if pid not in self.positions: return 0.0`, but `database.close_trade` is called *before* `positions.pop(pid)` — that ordering is intentional crash safety (`cnn_agent.py:247-252`), so the in-memory guard alone allows duplicate DB close rows.

### Lock specifics

```python
# _CNNBook additions
self._sell_locks: Dict[str, asyncio.Lock] = {}

def _lock_for(self, pid: str) -> asyncio.Lock:
    if pid not in self._sell_locks:
        self._sell_locks[pid] = asyncio.Lock()
    return self._sell_locks[pid]

async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
    async with self._lock_for(pid):                          # NEW
        if pid not in self.positions:
            return 0.0
        # ... existing body unchanged (DB-first ordering preserved) ...
```

- **Lock granularity:** per-product. Different products' exits do not serialize against each other.
- **Lock creation:** lazy on first call.
- **Lock cleanup:** none required. ~58 distinct products ever; kilobytes of memory.
- **Lock scope:** entire `sell()` body including `database.close_trade` + in-memory mutation.

### Race timing

```
t=0ms   WS tick arrives, on_price_tick decides trigger=WS_TRAIL_STOP
t=1ms   scan loop arrives at same pid, decides trigger=TRAIL_STOP
t=2ms   WS handler enters book.sell() → acquires lock
t=3ms   scan loop enters book.sell() → BLOCKS on lock
t=50ms  WS handler completes close_trade + pops pid → releases lock
t=51ms  scan loop acquires lock → `pid not in self.positions` → returns 0.0
```

Whichever caller wins, the position closes exactly once at the price observed by the winner. The loser is a no-op.

### Failure modes

| Failure | Effect required | Mechanism |
|---|---|---|
| Exception in `on_price_tick` body | Logged at WARN, swallowed | Top-level `try/except` in handler |
| Exception in `book.sell()` (e.g. DB write fails) | Logged at ERROR with traceback, swallowed at handler boundary | Same handler-level try/except |
| WS disconnect mid-tick | Handler task completes naturally; subscriber reconnects via existing `_run` loop | `register_price_handler` survives reconnects (handlers live on the subscriber instance) |
| `pos['trail_pct']` missing (just-opened position before first scan refresh) | Fall back to `_CNN_ATR_TRAIL_MIN` (6%) | `pos.get('trail_pct', _CNN_ATR_TRAIL_MIN)` |

### New CLAUDE.md invariant

> **#18: WS exit-handler isolation.** `exit_watcher.on_price_tick` MUST catch every exception in its body and log it. Exceptions in `asyncio.create_task`-spawned handlers do not crash the WS receive loop, but unretrieved-Task warnings hide errors from logs — explicit handling is required so failures are visible.

---

## Lifecycle

### Attach point (in `main.py` lifespan)

```python
# After existing line `await app_state.ws_subscriber.start()` (main.py:419)
from agents.exit_watcher import attach as attach_exit_watcher
attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book)
logger.info("WS exit watcher attached")
```

### Product subscription is not the watcher's concern

Scanner already calls `app_state.ws_subscriber.set_products(...)` (`main.py:417`, `:425`, `:571`) when the tracked list changes. Held products are by definition tracked (you can only buy what scans), so every held position receives ticks automatically. The watcher reads from `book.positions` — newly opened positions are picked up on the next tick.

### No detach needed

- `register_price_handler` has exactly one consumer for the lifetime of the backend.
- Shutdown calls `await app_state.ws_subscriber.stop()` (`main.py:487`), cancelling the WS receive task; in-flight handler tasks complete or are cancelled with the event loop.

### Edge cases

| Scenario | Handler behavior |
|---|---|
| Position just opened, scan hasn't run | `pos['trail_pct']` missing → fallback to `_CNN_ATR_TRAIL_MIN` (6%) |
| Position closed by scan microseconds ago | `positions.get(pid)` returns None → immediate no-op |
| WS reconnect | Handlers live on subscriber instance → resumes when subscription re-establishes |
| Product de-tracked while holding | Ticks stop; scan loop continues to manage exit at 60s cadence (graceful degradation) |
| Two ticks same pid rapid succession | Both fire `asyncio.create_task`; first into `sell()` wins lock, second hits `positions.get → None` no-op |
| Backend restart with open positions | `book.load()` restores positions + `peak_price`; handler attaches; `trail_pct` missing until first scan refresh (6% fallback meanwhile) |

---

## Testing

Tests are mock-only (no live API, no real DB writes per CLAUDE.md test conventions). **Safe to write while 8001 is trading; safe to run only during pause windows.**

### `tests/test_exit_watcher.py` (NEW)

| Test | What it pins |
|---|---|
| `test_no_position_returns_immediately` | Tick for unheld product → no `book.sell` call. Hot-path filter. |
| `test_trail_stop_fires_when_price_below_threshold` | `peak=100, price=93, trail_pct=0.06` → fires SELL with trigger=`WS_TRAIL_STOP`. |
| `test_stop_loss_fires_when_price_below_threshold` | `avg=100, price=91` → fires SELL with trigger=`WS_STOP_LOSS`. |
| `test_stop_loss_priority_over_trail` | Both triggered → exactly one `sell` call, trigger=`WS_STOP_LOSS`. |
| `test_peak_ratchets_on_new_high` | `peak=100, price=105` → `pos['peak_price']==105`, no exit fired. |
| `test_handler_swallows_exceptions` | Inject `book.sell` to raise → handler logs error, does NOT re-raise. |
| `test_attach_registers_handler_and_dispatches_ticks` (integration) | Real `CoinbaseWSSubscriber`, call `attach`, manually invoke `ws._handle({fake msg})`, assert `book.sell` called. End-to-end wiring without a real socket. |

### `tests/test_cnn_agent.py` extension

| Test | What it pins |
|---|---|
| `test_sell_lock_serializes_concurrent_callers` | `asyncio.gather(book.sell(pid, 100, "WS_TRAIL_STOP"), book.sell(pid, 100, "TRAIL_STOP"))`. Assert: exactly ONE `database.close_trade` call, exactly ONE position pop, second caller returned 0.0. |

### `tests/test_cnn_risk_exits.py` extension

| Test | What it pins |
|---|---|
| `test_check_risk_exits_writes_trail_pct_to_position` | Run `_check_risk_exits` on a position with mock candles → `pos['trail_pct']` is populated. Contract between scan loop and WS handler. |

**Total new test surface:** 9 tests, ~150 LoC.

---

## Implementation Order

| Atom | Files | TDD red | Implement | TDD green |
|---|---|---|---|---|
| **1. Lock + trail_pct cache** | `agents/cnn_agent.py` | `test_sell_lock_serializes_concurrent_callers`, `test_check_risk_exits_writes_trail_pct_to_position` | `_sell_locks` + `_lock_for`, wrap `sell()` body, add `pos['trail_pct'] = trail_pct` write in `_check_risk_exits` | targeted pytest |
| **2. `exit_watcher` module** | `agents/exit_watcher.py` (new), `tests/test_exit_watcher.py` (new) | 6 unit tests | Write `on_price_tick` + `attach` | targeted pytest |
| **3. Lifespan registration** | `main.py` | integration test | Add import + 2-line attach block after `ws_subscriber.start()` | targeted pytest |
| **4. Memory + CHANGELOG** | `CHANGELOG.md`, `coinbase_trader_architecture.md`, `CLAUDE.md` (invariant #18) | n/a | Docs only | n/a |

### Commit boundary

**Single atomic commit covering atoms 1–4.** Feature only "lands" once atom 3 runs (handler attached); atoms 1–2 alone are dead code. Per `feedback_other_cli_process_improvements.md` atomic-stage rule, ship as one revertible unit.

Commit message:

```
feat: WS-driven exit checker for trail-stop + stop-loss

Subscribes a price-tick handler to the existing Coinbase WS ticker channel
and fires WS_TRAIL_STOP / WS_STOP_LOSS exits on every held position
without waiting for the 60s scan cycle. Adds per-pid asyncio.Lock to
_CNNBook.sell() so the WS handler and scan-loop _check_risk_exits cannot
race a duplicate close_trade DB write. Scan loop now caches the computed
trail_pct onto pos['trail_pct'] so the WS handler reads it without
recomputing ATR per tick. Max-hold exit remains on the scan loop.
```

### Test sequence at commit time (next 8001 pause)

```bash
# Targeted (fast)
cd backend && ../.venv/Scripts/python.exe -m pytest \
  tests/test_exit_watcher.py \
  tests/test_cnn_agent.py::TestCNNBookSellLock \
  tests/test_cnn_risk_exits.py::test_check_risk_exits_writes_trail_pct \
  -v

# Full suite (pre-commit hook runs this regardless)
cd backend && ../.venv/Scripts/python.exe -m pytest tests/ -v
```

### Deployment

1. Commit + push during pause window.
2. Operator restarts 8001 (optionally 8002 first as canary).
3. Verify: log line `WS exit watcher attached` appears on startup.
4. Watch for first `WS_TRAIL_STOP` / `WS_STOP_LOSS` in trades table.
5. After 24h: compare scan-driven vs WS-driven exit counts (trigger strings distinguish source).

### Rollback

Revert the commit, restart 8001. No DB migration to undo. `pos['trail_pct']` is forward-compat (older code ignores extra dict keys).

---

## Open Questions

None. All foundational choices pinned during 2026-05-23 brainstorm:

- Exit scope: trail + stop-loss in WS, max-hold on scan loop (with max-hold redesign tracked separately as backlog #26).
- Lock location: inside `_CNNBook.sell()` (single source of truth).
- Trail-pct caching: scan loop owns refresh, writes to `pos['trail_pct']`; WS handler reads.
