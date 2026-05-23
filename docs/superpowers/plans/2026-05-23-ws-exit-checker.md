# WS-Driven Exit Checker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fire trail-stop and stop-loss exits on Coinbase WS price ticks instead of waiting for the 60s scan cycle.

**Architecture:** New `agents/exit_watcher.py` module registers an async price-tick handler with the existing `CoinbaseWSSubscriber.register_price_handler`. The handler reads from `_CNNBook.positions`, evaluates WS_STOP_LOSS / WS_TRAIL_STOP, and calls `_CNNBook.sell()`. A per-pid `asyncio.Lock` inside `_CNNBook.sell()` serializes the WS handler against the scan-loop `_check_risk_exits` to prevent duplicate `database.close_trade` writes. Scan loop caches the computed `trail_pct` onto `pos['trail_pct']` so the WS handler reads it without recomputing ATR per tick. Max-hold (7-day) exit remains on the scan loop.

**Tech Stack:** Python 3.11, asyncio, websockets, pytest, pytest-asyncio, FastAPI lifespan.

**Spec:** `docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md`

---

## File Structure

```
NEW   backend/agents/exit_watcher.py            ~80 LoC, price-tick handler + attach
EDIT  backend/agents/cnn_agent.py               _sell_locks + lock wrap on sell();
                                                pos['trail_pct'] cache write in _check_risk_exits
EDIT  backend/main.py                           import + attach in lifespan after ws_subscriber.start()
NEW   backend/tests/test_exit_watcher.py        6 unit + 1 integration test
EDIT  backend/tests/test_cnn_agent.py           TestCNNBookSellLock (1 race test)
EDIT  backend/tests/test_cnn_risk_exits.py      test_check_risk_exits_writes_trail_pct (1 cache test)
EDIT  CHANGELOG.md                              bullet under ## Unreleased
EDIT  CLAUDE.md                                 invariant #18 (WS exit-handler isolation)
EDIT  ~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md
                                                note WS exit watcher module + boundary
```

## Operational constraint

**Pre-commit hook runs full ~970-test pytest suite.** Per `feedback_no_pytest_during_trading.md`, this is blocked while 8001 is live or any training subprocess is active. **Tasks 1–6 only write + stage files** — the single atomic commit (Task 7) waits for an 8001 pause window. The implementation engineer must NOT run pytest while 8001 is live; only mock-only test writing is safe during trading.

---

## Task 1: Per-pid Lock around `_CNNBook.sell()`

**Files:**
- Modify: `backend/agents/cnn_agent.py` (class `_CNNBook`, ~line 111-114 init, ~line 239 sell)
- Test: `backend/tests/test_cnn_agent.py` (new class `TestCNNBookSellLock`)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_cnn_agent.py`:

```python
class TestCNNBookSellLock:
    """Per-pid asyncio.Lock in _CNNBook.sell() serializes concurrent callers
    to prevent duplicate database.close_trade writes when the WS exit handler
    races scan-loop _check_risk_exits on the same position.
    """

    @pytest.mark.asyncio
    async def test_sell_lock_serializes_concurrent_callers(self, monkeypatch):
        import asyncio
        from agents.cnn_agent import _CNNBook

        book = _CNNBook()
        book.balance = 1000.0
        book.positions["BTC-USD"] = {
            "size": 1.0, "avg_price": 100.0,
            "entry_time": 0.0, "peak_price": 100.0,
        }

        close_trade_calls = []

        async def _fake_close_trade(**kwargs):
            close_trade_calls.append(kwargs)
            await asyncio.sleep(0.01)  # widen race window

        async def _fake_save(*args, **kwargs):
            pass

        async def _fake_evaluate(*args, **kwargs):
            pass

        monkeypatch.setattr("agents.cnn_agent.database.close_trade", _fake_close_trade)
        monkeypatch.setattr("agents.cnn_agent.database.save_agent_state", _fake_save)
        monkeypatch.setattr(
            "agents.cnn_agent.product_status.evaluate_and_persist", _fake_evaluate,
        )

        results = await asyncio.gather(
            book.sell("BTC-USD", 90.0, trigger="WS_TRAIL_STOP"),
            book.sell("BTC-USD", 90.0, trigger="TRAIL_STOP"),
        )

        assert len(close_trade_calls) == 1, (
            f"expected exactly 1 close_trade call, got {len(close_trade_calls)}: "
            f"{close_trade_calls}"
        )
        assert "BTC-USD" not in book.positions
        assert sorted(results) == [-10.0, 0.0]   # loser returns 0.0; winner returns pnl = (90-100)*1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestCNNBookSellLock::test_sell_lock_serializes_concurrent_callers -v
```
Expected: FAIL — assertion `len(close_trade_calls) == 1` fails with `got 2` because both calls reach `database.close_trade` before either pops `self.positions`.

If 8001 is live, skip this step. Verification deferred to Task 7's full-suite run during the commit pause window.

- [ ] **Step 3: Add `_sell_locks` dict to `_CNNBook.__init__`**

Edit `backend/agents/cnn_agent.py` at the `_CNNBook.__init__` method (around line 111-115). Find:

```python
    def __init__(self):
        self._agent      = "CNN"
        self.balance     = _CNN_DRY_RUN_BALANCE
        self.positions: Dict[str, Dict] = {}   # pid → {size, avg_price}
        self.realized_pnl = 0.0
```

Replace with:

```python
    def __init__(self):
        self._agent      = "CNN"
        self.balance     = _CNN_DRY_RUN_BALANCE
        self.positions: Dict[str, Dict] = {}   # pid → {size, avg_price}
        self.realized_pnl = 0.0
        # Per-pid lock so WS exit handler and scan-loop _check_risk_exits
        # cannot race a duplicate database.close_trade write.
        self._sell_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, pid: str) -> asyncio.Lock:
        if pid not in self._sell_locks:
            self._sell_locks[pid] = asyncio.Lock()
        return self._sell_locks[pid]
```

- [ ] **Step 4: Wrap `_CNNBook.sell()` body in the per-pid lock**

Edit `backend/agents/cnn_agent.py` at the `_CNNBook.sell()` method (around line 239). Find:

```python
    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        if pid not in self.positions:
            return 0.0
        pos = self.positions[pid]
        proceeds = pos["size"] * price
        pnl      = proceeds - pos["size"] * pos["avg_price"]
        pct_pnl  = (price - pos["avg_price"]) / pos["avg_price"] * 100.0
```

Replace with (indent existing body inside the `async with`):

```python
    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        async with self._lock_for(pid):
            if pid not in self.positions:
                return 0.0
            pos = self.positions[pid]
            proceeds = pos["size"] * price
            pnl      = proceeds - pos["size"] * pos["avg_price"]
            pct_pnl  = (price - pos["avg_price"]) / pos["avg_price"] * 100.0
```

Indent every line of the existing sell body (lines 240 through `return pnl`) one extra level so it sits inside the `async with`. The final `return pnl` should still be the function's terminal line, just at the deeper indent.

- [ ] **Step 5: Verify `asyncio` is already imported in `cnn_agent.py`**

Run:
```bash
grep -n "^import asyncio" C:/Users/gl450/polymarket_app/backend/agents/cnn_agent.py
```
Expected: a line like `15:import asyncio` (or any positive match). If missing, add `import asyncio` to the top of the file alongside other stdlib imports.

- [ ] **Step 6: Run test to verify it passes (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_agent.py::TestCNNBookSellLock::test_sell_lock_serializes_concurrent_callers -v
```
Expected: PASS.

Task 1 leaves files staged-locally (no commit yet — single atomic commit in Task 7).

---

## Task 2: Cache `trail_pct` on position in `_check_risk_exits`

**Files:**
- Modify: `backend/agents/cnn_agent.py` (`_check_risk_exits`, ~line 1704-1731)
- Test: `backend/tests/test_cnn_risk_exits.py` (new test function)

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/test_cnn_risk_exits.py`:

```python
@pytest.mark.asyncio
async def test_check_risk_exits_writes_trail_pct_to_position(monkeypatch):
    """Scan loop caches the computed trail_pct on pos['trail_pct'] so the WS
    exit handler can read it without recomputing ATR per tick. Contract
    between scan loop and WS handler (see exit_watcher.on_price_tick).
    """
    import time
    from agents.cnn_agent import CoinbaseCNNAgent, _CNN_ATR_TRAIL_MIN, _CNN_ATR_TRAIL_MAX

    agent = CoinbaseCNNAgent(ws_subscriber=None)
    agent.book.positions["BTC-USD"] = {
        "size": 1.0,
        "avg_price": 100.0,
        "entry_time": time.time(),     # fresh — no max-hold trigger
        "peak_price": 105.0,
    }

    # 20 candles with deterministic non-zero ATR (high-low range = 2)
    fake_candles = [{"high": 100.0, "low": 98.0, "close": 99.0} for _ in range(20)]

    async def _fake_get_candles(pid, limit=20):
        return fake_candles

    monkeypatch.setattr("agents.cnn_agent.database.get_candles", _fake_get_candles)
    # Price between trail and stop-loss → no exit fires; we only check the cache write
    monkeypatch.setattr(agent, "_live_price", lambda pid, fb: 104.0)

    await agent._check_risk_exits()

    pos = agent.book.positions["BTC-USD"]
    assert "trail_pct" in pos, "scan loop must write trail_pct for WS handler"
    assert _CNN_ATR_TRAIL_MIN <= pos["trail_pct"] <= _CNN_ATR_TRAIL_MAX
```

- [ ] **Step 2: Run test to verify it fails (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_risk_exits.py::test_check_risk_exits_writes_trail_pct_to_position -v
```
Expected: FAIL — `assert "trail_pct" in pos` fails because nothing writes it today.

- [ ] **Step 3: Add the cache write in `_check_risk_exits`**

Edit `backend/agents/cnn_agent.py` in `_check_risk_exits` (around line 1704-1717). Find the block:

```python
            # Compute ATR-based trail distance for this product
            trail_pct = _CNN_ATR_TRAIL_MIN  # fallback if candles unavailable
            try:
                candles = await database.get_candles(pid, limit=20)
                if len(candles) >= 15:
                    highs  = [c["high"]  for c in candles]
                    lows   = [c["low"]   for c in candles]
                    closes = [c["close"] for c in candles]
                    atr    = _atr(highs, lows, closes)
                    if atr > 0 and peak_price > 0:
                        raw = atr * _CNN_ATR_TRAIL_MULT / peak_price
                        trail_pct = max(_CNN_ATR_TRAIL_MIN, min(raw, _CNN_ATR_TRAIL_MAX))
            except Exception:
                pass
```

Add one line immediately after the `except Exception: pass` block, before the `# Positions without entry_time` comment:

```python
            except Exception:
                pass

            # Cache for WS exit handler — read by agents/exit_watcher.on_price_tick
            # so it doesn't recompute ATR per tick (~580 ticks/sec aggregate).
            pos["trail_pct"] = trail_pct
```

- [ ] **Step 4: Run test to verify it passes (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_cnn_risk_exits.py::test_check_risk_exits_writes_trail_pct_to_position -v
```
Expected: PASS.

Task 2 leaves files staged-locally.

---

## Task 3: Write `exit_watcher` module + 6 unit tests

**Files:**
- Create: `backend/agents/exit_watcher.py`
- Create: `backend/tests/test_exit_watcher.py`

- [ ] **Step 1: Write the 6 failing unit tests**

Create `backend/tests/test_exit_watcher.py`:

```python
"""Tests for agents/exit_watcher.py — WS-driven trail-stop / stop-loss exits.

Mock-only: no live WS, no real DB. Safe to write while 8001 is trading;
only the pytest invocation itself is gated by feedback_no_pytest_during_trading.
"""
import asyncio
import logging
import os
import sys
from unittest.mock import AsyncMock

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


class _FakeBook:
    """In-memory stand-in for _CNNBook. Tests only need .positions + .sell."""

    def __init__(self):
        self.positions: dict = {}
        self.sell = AsyncMock(return_value=0.0)


def _make_pos(*, avg=100.0, peak=100.0, trail=0.06):
    return {"size": 1.0, "avg_price": avg, "peak_price": peak, "trail_pct": trail}


class TestOnPriceTick:

    @pytest.mark.asyncio
    async def test_no_position_returns_immediately(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        await on_price_tick("BTC-USD", 100.0, book)
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_trail_stop_fires_when_price_below_threshold(self):
        # peak=100, trail=6% → trail threshold = 94. price=93 triggers.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 93.0, book)
        book.sell.assert_called_once_with("BTC-USD", 93.0, trigger="WS_TRAIL_STOP")

    @pytest.mark.asyncio
    async def test_stop_loss_fires_when_price_below_threshold(self):
        # avg=100, _CNN_STOP_LOSS_PCT=8% → stop threshold = 92. price=91 triggers.
        # peak also 100 so trail_threshold=94; both would fire but stop-loss has priority.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 91.0, book)
        book.sell.assert_called_once_with("BTC-USD", 91.0, trigger="WS_STOP_LOSS")

    @pytest.mark.asyncio
    async def test_stop_loss_priority_over_trail(self):
        # Both triggered at price=85: stop-loss must fire, trail must NOT.
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 85.0, book)
        assert book.sell.call_count == 1
        book.sell.assert_called_with("BTC-USD", 85.0, trigger="WS_STOP_LOSS")

    @pytest.mark.asyncio
    async def test_peak_ratchets_on_new_high(self):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        await on_price_tick("BTC-USD", 105.0, book)
        assert book.positions["BTC-USD"]["peak_price"] == 105.0
        book.sell.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_swallows_exceptions(self, caplog):
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)
        book.sell.side_effect = RuntimeError("simulated DB failure")
        with caplog.at_level(logging.ERROR, logger="agents.exit_watcher"):
            # Must NOT re-raise — invariant #18
            await on_price_tick("BTC-USD", 91.0, book)
        assert any(r.levelno == logging.ERROR for r in caplog.records), (
            "exit_watcher must log at ERROR when sell() raises"
        )
```

- [ ] **Step 2: Run the 6 unit tests to verify they fail (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_exit_watcher.py::TestOnPriceTick -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.exit_watcher'`.

- [ ] **Step 3: Create `agents/exit_watcher.py` with `on_price_tick`**

Create `backend/agents/exit_watcher.py`:

```python
"""WS-driven exit checker.

Registers an async price-tick handler with CoinbaseWSSubscriber and fires
WS_TRAIL_STOP / WS_STOP_LOSS exits on every held position without waiting
for the 60s scan cycle. Max-hold (7-day) exit remains on the scan loop.

Loose coupling per feedback_loose_coupling.md:
  ws_subscriber  →  doesn't know about exit_watcher
  cnn_agent      →  doesn't know about exit_watcher
  exit_watcher   →  reads book.positions; writes via book.sell()

Spec:  docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# Constants come from cnn_agent so this module and the scan-loop checker
# stay in lockstep on thresholds. Importing them does load cnn_agent, but
# main.py already loads cnn_agent in the lifespan — no new cost.
from agents.cnn_agent import _CNN_STOP_LOSS_PCT, _CNN_ATR_TRAIL_MIN

if TYPE_CHECKING:
    from agents.cnn_agent import _CNNBook
    from services.ws_subscriber import CoinbaseWSSubscriber

logger = logging.getLogger(__name__)


async def on_price_tick(pid: str, price: float, book: "_CNNBook") -> None:
    """Per-tick exit checker. Idempotent. Exceptions are caught + logged
    (invariant #18 in CLAUDE.md) so a handler failure cannot crash the
    WS receive loop or poison subsequent ticks.
    """
    try:
        pos = book.positions.get(pid)
        if pos is None:
            return                                              # ≈99% of ticks

        avg_price = pos.get("avg_price", 0.0)
        if avg_price <= 0 or price <= 0:
            return

        peak = pos.get("peak_price") or avg_price
        if price > peak:
            pos["peak_price"] = price
            peak = price

        pct_entry     = (price - avg_price) / avg_price
        pct_from_peak = (price - peak) / peak
        trail_pct     = pos.get("trail_pct", _CNN_ATR_TRAIL_MIN)

        trigger = None
        if pct_entry <= -_CNN_STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif pct_from_peak <= -trail_pct:
            trigger = "WS_TRAIL_STOP"

        if trigger:
            await book.sell(pid, price, trigger=trigger)

    except Exception:
        logger.exception(
            "exit_watcher.on_price_tick failed (pid=%s price=%s)", pid, price,
        )
```

`attach()` is intentionally absent — Task 4 adds it.

- [ ] **Step 4: Run the 6 unit tests to verify they pass (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_exit_watcher.py::TestOnPriceTick -v
```
Expected: PASS for all 6.

Task 3 leaves files staged-locally.

---

## Task 4: `attach()` helper + integration test

**Files:**
- Modify: `backend/agents/exit_watcher.py` (add `attach`)
- Modify: `backend/tests/test_exit_watcher.py` (add `TestAttach`)

- [ ] **Step 1: Write the failing integration test**

Append to `backend/tests/test_exit_watcher.py`:

```python
class TestAttach:
    """End-to-end: attach() registers the handler with CoinbaseWSSubscriber,
    and a ticker message routed through ws._handle() reaches book.sell().
    Verifies wiring without opening a real socket.
    """

    @pytest.mark.asyncio
    async def test_attach_registers_handler_and_dispatches_ticks(self):
        from services.ws_subscriber import CoinbaseWSSubscriber
        from agents.exit_watcher import attach

        ws = CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06)

        attach(ws, book)

        fake_msg = {
            "channel": "ticker",
            "events": [{
                "tickers": [{
                    "product_id": "BTC-USD",
                    "price":      "93.0",
                    "best_bid":   "92.99",
                    "best_ask":   "93.01",
                }],
            }],
        }
        await ws._handle(fake_msg)

        # _price_handlers are fired via asyncio.create_task — yield once
        # so the spawned task runs before we assert.
        await asyncio.sleep(0.05)

        book.sell.assert_called_once_with("BTC-USD", 93.0, trigger="WS_TRAIL_STOP")
```

- [ ] **Step 2: Run the integration test to verify it fails (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_exit_watcher.py::TestAttach -v
```
Expected: FAIL — `ImportError: cannot import name 'attach' from 'agents.exit_watcher'`.

- [ ] **Step 3: Add `attach()` to `agents/exit_watcher.py`**

Append to `backend/agents/exit_watcher.py`:

```python
def attach(ws_subscriber: "CoinbaseWSSubscriber", book: "_CNNBook") -> None:
    """Register the per-tick exit handler. Call once per backend lifespan
    (in main.py after ws_subscriber.start() and after cnn_agent is built).
    """
    async def _handler(pid: str, price: float) -> None:
        await on_price_tick(pid, price, book)

    ws_subscriber.register_price_handler(_handler)
    logger.info("exit_watcher attached to ws_subscriber")
```

- [ ] **Step 4: Run the integration test to verify it passes (defer if 8001 live)**

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_exit_watcher.py::TestAttach -v
```
Expected: PASS.

Task 4 leaves files staged-locally.

---

## Task 5: Register handler in `main.py` lifespan

**Files:**
- Modify: `backend/main.py` (top imports + lifespan after line 419)

- [ ] **Step 1: Add the import at the top of `main.py`**

Edit `backend/main.py`. Find the existing import line for `CoinbaseWSSubscriber`:

```python
from services.ws_subscriber import CoinbaseWSSubscriber
```

Add directly below it:

```python
from agents.exit_watcher import attach as attach_exit_watcher
```

- [ ] **Step 2: Add the attach call in the lifespan**

Edit `backend/main.py`. Find the existing line (around line 419):

```python
    await app_state.ws_subscriber.start()
```

Insert immediately after it:

```python

    # WS-driven exit checker: fires WS_TRAIL_STOP / WS_STOP_LOSS on every
    # held position without waiting for the 60s scan cycle.
    # Spec: docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md
    attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book)
    logger.info("WS exit watcher attached")
```

- [ ] **Step 3: Re-run the integration test as a smoke check (defer if 8001 live)**

The integration test from Task 4 already covers `attach()` correctness. Re-running it after the lifespan edit catches any accidental regression to either side.

Run (only when 8001 is paused):
```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/test_exit_watcher.py -v
```
Expected: PASS for all 7 tests.

There is no lifespan-level unit test — `main.py` lifespan is exercised end-to-end via the backend startup itself. Deployment verification (Task 7) confirms the `WS exit watcher attached` log line appears on real startup.

Task 5 leaves files staged-locally.

---

## Task 6: Docs — CHANGELOG, CLAUDE.md invariant #18, memory file

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md`
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`

- [ ] **Step 1: Add the CHANGELOG bullet under `## Unreleased`**

Edit `CHANGELOG.md`. Locate the `## Unreleased` section header and add this bullet beneath it (preserve any existing bullets):

```markdown
- **WS-driven exit checker.** New `backend/agents/exit_watcher.py` registers an async price-tick handler on the existing Coinbase WS ticker channel. Fires `WS_TRAIL_STOP` / `WS_STOP_LOSS` exits on held positions in sub-second latency instead of waiting for the 60s scan cycle. Max-hold (7-day) exit stays on the scan loop. Adds per-pid `asyncio.Lock` to `_CNNBook.sell()` so the WS handler and scan-loop `_check_risk_exits` cannot race a duplicate `database.close_trade` write. Scan loop caches the computed `trail_pct` on `pos['trail_pct']` so the WS handler reads it without recomputing ATR per tick. Spec: `docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md`. Invariant #18 added to `CLAUDE.md`.
```

- [ ] **Step 2: Add invariant #18 to `CLAUDE.md`**

Edit `CLAUDE.md`. Find the existing invariant block (look for "**v4.5 3-class telemetry contract**" — that's invariant #17). Append invariant #18 directly after it, before the `---` separator that ends the Architecture Quick Reference section:

```markdown
18. **WS exit-handler isolation.** `agents/exit_watcher.on_price_tick` MUST catch every exception in its body and log it at ERROR. Exceptions in handlers spawned via `asyncio.create_task` do not crash the WS receive loop (the task captures them), but unretrieved-Task warnings hide errors from logs — explicit `try/except + logger.exception` is required so failures stay visible. Mirrors invariant #14's MC chain rule and invariant #16's shadow-telemetry isolation rule.
```

- [ ] **Step 3: Update the architecture memory file**

Edit `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`. Add a new subsection (placement: under the most recent dated heading, or under a new `### 2026-05-23 — WS exit checker` heading if no current section exists):

```markdown
### WS-driven exit checker (2026-05-23, Session 58.71m)

`backend/agents/exit_watcher.py` — new module, ~80 LoC. Registers async
price-tick handler with `CoinbaseWSSubscriber.register_price_handler`.
Per-tick: reads `book.positions[pid]`, ratchets `peak_price`, fires
`WS_STOP_LOSS` (priority) or `WS_TRAIL_STOP` via `book.sell(pid, price, trigger=...)`.
Max-hold (7d) exit stays on scan loop's `_check_risk_exits`.

**Race protection:** per-pid `asyncio.Lock` inside `_CNNBook.sell()`
(field `_sell_locks: Dict[str, asyncio.Lock]`, helper `_lock_for(pid)`).
Lock wraps the entire sell body so the WS handler and scan-loop
`_check_risk_exits` cannot race a duplicate `database.close_trade` write.

**Trail-pct cache:** `_check_risk_exits` writes `pos['trail_pct'] = trail_pct`
after computing the ATR-based value. WS handler reads `pos.get('trail_pct',
_CNN_ATR_TRAIL_MIN)` — no DB calls per tick. Up to 60s lag on regime
shifts; acceptable since ATR drifts slowly.

**Wired in:** `main.py` lifespan, after `ws_subscriber.start()`:
`attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book)`.

**Telemetry:** trigger strings `WS_TRAIL_STOP` / `WS_STOP_LOSS` distinguish
WS-driven exits from scan-driven `TRAIL_STOP` / `STOP_LOSS`. Compare counts
in `trades` table to measure latency reduction.

## See also
- [[coinbase_trader_session_log]] — Session 58.71m entry
- [[feedback_loose_coupling]] — module boundary rationale
- CLAUDE.md invariant #18 — handler isolation rule
```

Task 6 leaves files staged-locally.

---

## Task 7: Atomic commit + push (during 8001 pause window)

**Operational gate:** This task is the ONLY one that runs the full pytest suite and commits. It is blocked while 8001 is live or any training subprocess is active. Operator must pause 8001 first; coordinate before starting.

**Files staged for commit:**
- `backend/agents/exit_watcher.py` (new)
- `backend/agents/cnn_agent.py`
- `backend/main.py`
- `backend/tests/test_exit_watcher.py` (new)
- `backend/tests/test_cnn_agent.py`
- `backend/tests/test_cnn_risk_exits.py`
- `CHANGELOG.md`
- `CLAUDE.md`

(The memory file at `~/.claude/projects/...` is outside the repo and not part of the commit.)

- [ ] **Step 1: Operator confirms 8001 is paused**

Visual confirmation in the frontend (Start/Stop button) or via:
```bash
curl -s http://127.0.0.1:8001/api/status | grep -o '"is_trading":[^,]*'
```
Expected: `"is_trading":false`.

If `true`, STOP — do not proceed. Coordinate with operator.

- [ ] **Step 2: Run the full pytest suite**

```bash
cd C:/Users/gl450/polymarket_app/backend && ../.venv/Scripts/python.exe -m pytest tests/ -v
```
Expected: ~970 PASS, 0 FAIL (the 9 new tests from Tasks 1–4 are included). If anything FAILs, STOP. Diagnose. Do not commit a red suite.

- [ ] **Step 3: Inspect `git status` for unexpected files**

```bash
git -C C:/Users/gl450/polymarket_app status --short
```
Expected output (in any order):
```
 M backend/agents/cnn_agent.py
?? backend/agents/exit_watcher.py
 M backend/main.py
?? backend/tests/test_exit_watcher.py
 M backend/tests/test_cnn_agent.py
 M backend/tests/test_cnn_risk_exits.py
 M CHANGELOG.md
 M CLAUDE.md
```

If `.env`, `*.db`, `*.pt`, or any other untracked-and-sensitive file appears, STOP and remove it from staging consideration. CLAUDE.md security gate forbids those.

- [ ] **Step 4: Stage exactly the 8 files**

```bash
git -C C:/Users/gl450/polymarket_app add \
  backend/agents/exit_watcher.py \
  backend/agents/cnn_agent.py \
  backend/main.py \
  backend/tests/test_exit_watcher.py \
  backend/tests/test_cnn_agent.py \
  backend/tests/test_cnn_risk_exits.py \
  CHANGELOG.md \
  CLAUDE.md
```

Re-check with `git status --short` — every listed file should show `A` (added) or `M` (modified) in the left column.

- [ ] **Step 5: Commit (pre-commit hook runs full suite again)**

```bash
git -C C:/Users/gl450/polymarket_app commit -m "$(cat <<'EOF'
feat: WS-driven exit checker for trail-stop + stop-loss

Subscribes a price-tick handler to the existing Coinbase WS ticker channel
and fires WS_TRAIL_STOP / WS_STOP_LOSS exits on every held position
without waiting for the 60s scan cycle. Adds per-pid asyncio.Lock to
_CNNBook.sell() so the WS handler and scan-loop _check_risk_exits cannot
race a duplicate close_trade DB write. Scan loop now caches the computed
trail_pct onto pos['trail_pct'] so the WS handler reads it without
recomputing ATR per tick. Max-hold exit remains on the scan loop.

Spec:  docs/superpowers/specs/2026-05-23-ws-exit-checker-design.md
Plan:  docs/superpowers/plans/2026-05-23-ws-exit-checker.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```
Expected: pre-commit hook runs the full suite (or whatever subset is configured), suite passes, commit succeeds with a new SHA printed.

If pre-commit hook FAILS: read the failure output, fix the issue, re-stage, create a NEW commit (do NOT `--amend` — per CLAUDE.md commit standards).

- [ ] **Step 6: Push to origin**

```bash
git -C C:/Users/gl450/polymarket_app push
```
Expected: push succeeds to current branch on origin.

- [ ] **Step 7: Operator restarts 8001**

Operator action — not automated by this plan.

- [ ] **Step 8: Deployment verification**

After 8001 restart, tail the backend log for the attach line:
```bash
tail -n 200 C:/Users/gl450/polymarket_app/backend/backend.log | grep "WS exit watcher attached"
```
Expected: exactly one line `WS exit watcher attached` shortly after the `Coinbase WS subscribed to:` line.

After 24h of live trading, compare trigger counts:
```bash
sqlite3 C:/Users/gl450/polymarket_app/backend/coinbase.db \
  "SELECT trigger_close, COUNT(*) FROM trades WHERE status='closed' AND closed_at > datetime('now','-1 day') GROUP BY trigger_close ORDER BY trigger_close;"
```
Expected: rows for `WS_TRAIL_STOP` and/or `WS_STOP_LOSS` alongside the existing `TRAIL_STOP` / `STOP_LOSS` rows. WS-driven rows confirm end-to-end wiring.

- [ ] **Step 9: Memory + session-log update**

Update memory files per `feedback_memory_after_changes.md`:
- `coinbase_trader_session_log.md` — add Session 58.71m WS-exit-checker entry.
- `coinbase_trader_bugs_fixed.md` — no entry (this is a feature, not a bug fix).
- `MEMORY.md` index — no new file (architecture memory already linked).

Task 7 ends the plan.

---

## Rollback

If the WS exit checker misbehaves in production (false exits, missed exits, log spam):

1. `git -C C:/Users/gl450/polymarket_app revert <COMMIT_SHA>` (the commit from Task 7).
2. `git push`.
3. Operator restarts 8001.
4. No DB migration to undo. `pos['trail_pct']` is forward-compat (older code ignores extra dict keys); `_sell_locks` is in-memory only.

---

## Acceptance Criteria

The feature is "done" when:

- All 9 new tests pass (6 unit + 1 integration in `test_exit_watcher.py`, 1 race test in `test_cnn_agent.py`, 1 cache-write test in `test_cnn_risk_exits.py`).
- Full ~970-test pytest suite passes.
- Pre-commit hook accepts the single atomic commit.
- Backend logs show `WS exit watcher attached` on startup.
- After 24h of live trading, `trades` table contains at least one row with `trigger_close LIKE 'WS_%'` (confirms the path actually fires).
- `_CNNBook.sell()` is race-safe under `asyncio.gather` from two callers (the Task 1 test pins this).
- `pos['trail_pct']` is populated for every held position after the scan loop iterates (the Task 2 test pins this).
- Memory file `coinbase_trader_architecture.md` documents the new module + boundary.
- CLAUDE.md invariant #18 documents the handler-isolation rule.
