# Maker-Execution Exit Leg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route live exit orders maker/taker by trigger (trail + model-down → maker post-only; stop-loss + max-hold → taker), gated behind the existing `USE_MAKER_EXECUTION` flag, across both the scan-loop and WS exit paths.

**Architecture:** A new single-responsibility module `agents/exit_execution.py` owns the trigger→mode classification, the flag/dry-run gate, the bid/ask fetch, and the call into `order_executor`. Both exit paths (`cnn_agent._check_risk_exits`, `exit_watcher.on_price_tick`) close the position on the paper book exactly as today, then make one unconditional call to `exit_execution.execute_live_exit(...)`, which no-ops unless `USE_MAKER_EXECUTION=true` and the executor is live (`not dry_run`).

**Tech Stack:** Python 3, asyncio, pytest + pytest-asyncio, `unittest.mock`.

## Global Constraints

- **Default-off contract:** with `config.use_maker_execution == False` (default), behavior is byte-for-byte unchanged — no live order, no bid/ask fetch. (CLAUDE.md invariant #21 / #14 pattern.)
- **Paper book is the source of truth:** always `book.sell(...)` first, then the live exit (mirrors the entry leg's "paper first, then live" ordering).
- **Isolation:** a live-exit failure must NEVER re-raise into the scan loop or WS handler (CLAUDE.md invariants #16/#18).
- **Branch:** work on `feat/maker-exit-leg` (stacked on `feat/maker-execution-shadow`). Surgical pathspec on every commit; never `git commit -a`. Confirm branch before each `git add`.
- **TDD:** failing test → run red → implement → run green → commit, per task.
- **No new env var** — reuse `config.use_maker_execution`.
- **Trigger routing (verbatim):** maker = `{TRAIL_STOP, WS_TRAIL_STOP, MODEL_DOWN, WS_MODEL_DOWN}`; taker = `{STOP_LOSS, WS_STOP_LOSS, MAX_HOLD, LEGACY_EXIT}`.

---

### Task 1: `exit_execution.py` module — classification, gate, routing

**Files:**
- Create: `backend/agents/exit_execution.py`
- Test: `backend/tests/test_exit_execution.py`

**Interfaces:**
- Consumes: `config.use_maker_execution` (bool); `clients.coinbase_client.get_best_bid_ask(list[str]) -> dict[str, dict]` returning `{pid: {"bid": float, "ask": float, ...}}`; `order_executor.execute_maker_signal(signal) -> dict`; `order_executor.execute_signal(signal) -> dict`; `order_executor.dry_run` (bool).
- Produces:
  - `is_maker_exit(trigger: str) -> bool`
  - `async execute_live_exit(order_executor, *, pid: str, price: float, size: float, trigger: str) -> Optional[dict]` — returns the executor result dict, or `None` when gated off / quotes missing.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_exit_execution.py`:

```python
"""Tests for agents/exit_execution.py — maker/taker routing for live exits.

Mock-only: no live network, no real DB. Safe to write during 8001 trading;
only the pytest invocation itself is gated by feedback_no_pytest_during_trading.
"""
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import agents.exit_execution as exit_mod
from agents.exit_execution import is_maker_exit, execute_live_exit


def _live_executor():
    ex = AsyncMock()
    ex.dry_run = False
    ex.execute_maker_signal.return_value = {"success": True, "fill_mode": "MAKER"}
    ex.execute_signal.return_value = {"success": True, "fill_mode": "TAKER"}
    return ex


class TestClassification:
    def test_maker_triggers(self):
        for t in ("TRAIL_STOP", "WS_TRAIL_STOP", "MODEL_DOWN", "WS_MODEL_DOWN"):
            assert is_maker_exit(t) is True

    def test_taker_triggers(self):
        for t in ("STOP_LOSS", "WS_STOP_LOSS", "MAX_HOLD", "LEGACY_EXIT"):
            assert is_maker_exit(t) is False


class TestGate:
    @pytest.mark.asyncio
    async def test_none_executor_noops(self):
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True):
            result = await execute_live_exit(
                None, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP")
        assert result is None

    @pytest.mark.asyncio
    async def test_flag_off_noops(self):
        ex = _live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", False, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP")
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()
        mock_bba.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_executor_noops(self):
        ex = _live_executor()
        ex.dry_run = True
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="TRAIL_STOP")
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()


class TestMakerRouting:
    @pytest.mark.asyncio
    async def test_maker_trigger_routes_with_quotes_and_sizing(self):
        ex = _live_executor()
        quotes = {"BTC-USD": {"bid": 99.0, "ask": 101.0, "price": 100.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)) as mock_bba:
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=2.0, trigger="TRAIL_STOP")

        mock_bba.assert_awaited_once_with(["BTC-USD"])
        ex.execute_signal.assert_not_called()
        ex.execute_maker_signal.assert_awaited_once()
        sig = ex.execute_maker_signal.call_args.args[0]
        assert sig["side"] == "SELL"
        assert sig["product_id"] == "BTC-USD"
        assert sig["signal_type"] == "TRAIL_STOP"
        assert sig["bid"] == 99.0 and sig["ask"] == 101.0
        assert sig["quote_size"] == round(2.0 * 101.0, 2)   # size * ask
        assert "atr" not in sig
        assert result == {"success": True, "fill_mode": "MAKER"}

    @pytest.mark.asyncio
    async def test_maker_trigger_missing_quotes_noops(self):
        ex = _live_executor()
        quotes = {"BTC-USD": {"bid": 0.0, "ask": 0.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            result = await execute_live_exit(
                ex, pid="BTC-USD", price=100.0, size=1.0, trigger="MODEL_DOWN")
        assert result is None
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()


class TestTakerRouting:
    @pytest.mark.asyncio
    async def test_taker_trigger_routes_without_quote_fetch(self):
        ex = _live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            result = await execute_live_exit(
                ex, pid="ETH-USD", price=50.0, size=3.0, trigger="STOP_LOSS")

        mock_bba.assert_not_called()
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_awaited_once()
        sig = ex.execute_signal.call_args.args[0]
        assert sig["side"] == "SELL"
        assert sig["product_id"] == "ETH-USD"
        assert sig["signal_type"] == "STOP_LOSS"
        assert sig["quote_size"] == round(3.0 * 50.0, 2)    # size * price
        assert "bid" not in sig and "ask" not in sig and "atr" not in sig
        assert result == {"success": True, "fill_mode": "TAKER"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_exit_execution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agents.exit_execution'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/agents/exit_execution.py`:

```python
"""Maker/taker routing for live EXIT orders.

Single source of truth for: (1) which exit triggers post as maker vs cross as
taker, (2) the USE_MAKER_EXECUTION + dry-run gate, (3) sourcing bid/ask quotes,
(4) the call into order_executor. Both exit paths — cnn_agent._check_risk_exits
(scan loop) and exit_watcher.on_price_tick (WS) — close the paper book first,
then call execute_live_exit(), which no-ops unless the flag is on and the
executor is live. Mirrors the entry leg (cnn_agent._execute_live_order) but
gates the WHOLE live-order leg behind the flag, because exits place no live
order today (so flag-off must stay byte-for-byte paper-only).

Spec: docs/superpowers/specs/2026-06-21-maker-execution-exit-leg-design.md
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)

# Trail + model-down exits can wait for a post-only fill (30s market fallback
# lives in execute_maker_signal). Hard stops + forced time exits cross now.
_MAKER_EXIT_TRIGGERS: frozenset = frozenset(
    {"TRAIL_STOP", "WS_TRAIL_STOP", "MODEL_DOWN", "WS_MODEL_DOWN"}
)


def is_maker_exit(trigger: str) -> bool:
    """True if `trigger` should post as a maker (post-only) exit."""
    return trigger in _MAKER_EXIT_TRIGGERS


async def execute_live_exit(
    order_executor,
    *,
    pid: str,
    price: float,
    size: float,
    trigger: str,
) -> Optional[Dict]:
    """Place a live SELL liquidating `size` of `pid`, routed by `trigger`.

    No-ops (returns None) unless USE_MAKER_EXECUTION is on and the executor is
    live (not dry-run). Builds a SELL signal with NO `atr` key so order_executor
    sizes from `quote_size`; `quote_size` is set so the executor's
    `base_size = quote_size / fill_price` recovers the held `size`:
      - maker SELL fills at ask   -> quote_size = size * ask
      - taker SELL fills at price  -> quote_size = size * price
    """
    if (order_executor is None
            or getattr(order_executor, "dry_run", True)
            or not config.use_maker_execution):
        return None

    signal: Dict = {
        "product_id":  pid,
        "side":        "SELL",
        "price":       price,
        "signal_type": trigger,
    }

    if is_maker_exit(trigger):
        quotes = await coinbase_client.get_best_bid_ask([pid])
        quote  = quotes.get(pid, {})
        bid    = quote.get("bid") or 0.0
        ask    = quote.get("ask") or 0.0
        if bid <= 0 or ask <= 0:
            logger.warning(
                "maker exit %s for %s missing quotes (bid=%s ask=%s) — "
                "skipping live exit", trigger, pid, bid, ask,
            )
            return None
        signal["bid"]        = bid
        signal["ask"]        = ask
        signal["quote_size"] = round(size * ask, 2)
        return await order_executor.execute_maker_signal(signal)

    signal["quote_size"] = round(size * price, 2)
    return await order_executor.execute_signal(signal)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_exit_execution.py -v`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD   # expect feat/maker-exit-leg
git add backend/agents/exit_execution.py backend/tests/test_exit_execution.py
git commit -m "feat: exit_execution module — maker/taker routing for live exits

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 2: Wire the scan-loop exit path (`_check_risk_exits`)

**Files:**
- Modify: `backend/agents/cnn_agent.py` (import `exit_execution`; `_check_risk_exits` signature + live-exit call; `run_loop` forwards `order_executor`)
- Test: `backend/tests/test_cnn_risk_exits.py` (append a new test class)

**Interfaces:**
- Consumes: `exit_execution.execute_live_exit(order_executor, *, pid, price, size, trigger)` from Task 1.
- Produces: `_check_risk_exits(self, order_executor=None)` — live exit fired after the paper close when gated on.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_cnn_risk_exits.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# Maker-execution exit leg — _check_risk_exits live routing
# ═══════════════════════════════════════════════════════════════════════════

class TestCheckRiskExitsLiveRouting:
    """_check_risk_exits places a live exit after the paper close, routed
    maker/taker by trigger, only when USE_MAKER_EXECUTION is on and the
    executor is live. Flag-off / dry-run / no-executor stay paper-only."""

    def _live_executor(self):
        ex = AsyncMock()
        ex.dry_run = False
        ex.execute_maker_signal.return_value = {"success": True, "fill_mode": "MAKER"}
        ex.execute_signal.return_value = {"success": True, "fill_mode": "TAKER"}
        return ex

    def _agent_with_trail_setup(self, pid="BTC-USD", price=105.0):
        # peak=120, avg=100 -> peak_pnl 20%; price 105 (+5%) << trail threshold
        # -> TRAIL_STOP. price 105 > stop (92) so no STOP_LOSS.
        agent = CoinbaseCNNAgent()
        agent.book = _book_with_position(pid, avg_price=100.0, size=2.0,
                                         peak_price=120.0)
        ws = MagicMock()
        ws.get_price.return_value = price
        agent.ws = ws
        return agent

    @pytest.mark.asyncio
    async def test_trail_stop_routes_maker_when_flag_on(self):
        import agents.exit_execution as exit_mod
        agent = self._agent_with_trail_setup(price=105.0)
        ex = self._live_executor()
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])), \
             patch.object(agent.book, "sell", new=AsyncMock(return_value=10.0)) as sell_mock, \
             patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            await agent._check_risk_exits(ex)

        assert sell_mock.call_args.kwargs.get("trigger") == "TRAIL_STOP"
        ex.execute_maker_signal.assert_awaited_once()
        ex.execute_signal.assert_not_called()
        sig = ex.execute_maker_signal.call_args.args[0]
        assert sig["side"] == "SELL" and sig["signal_type"] == "TRAIL_STOP"
        assert sig["quote_size"] == round(2.0 * 106.0, 2)

    @pytest.mark.asyncio
    async def test_stop_loss_routes_taker_when_flag_on(self):
        import agents.exit_execution as exit_mod
        agent = CoinbaseCNNAgent()
        agent.book = _book_with_position("ETH-USD", avg_price=100.0, size=2.0,
                                         peak_price=100.0)
        ws = MagicMock(); ws.get_price.return_value = 91.0   # -9% -> STOP_LOSS
        agent.ws = ws
        ex = self._live_executor()
        with patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])), \
             patch.object(agent.book, "sell", new=AsyncMock(return_value=-9.0)) as sell_mock, \
             patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await agent._check_risk_exits(ex)

        assert sell_mock.call_args.kwargs.get("trigger") == "STOP_LOSS"
        mock_bba.assert_not_called()
        ex.execute_signal.assert_awaited_once()
        ex.execute_maker_signal.assert_not_called()
        sig = ex.execute_signal.call_args.args[0]
        assert sig["quote_size"] == round(2.0 * 91.0, 2)

    @pytest.mark.asyncio
    async def test_flag_off_stays_paper_only(self):
        import agents.exit_execution as exit_mod
        agent = self._agent_with_trail_setup(price=105.0)
        ex = self._live_executor()
        with patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])), \
             patch.object(agent.book, "sell", new=AsyncMock(return_value=10.0)) as sell_mock, \
             patch.object(exit_mod.config, "use_maker_execution", False, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await agent._check_risk_exits(ex)

        sell_mock.assert_called_once()           # paper close still happens
        mock_bba.assert_not_called()
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_executor_stays_paper_only(self):
        agent = self._agent_with_trail_setup(price=105.0)
        with patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])), \
             patch.object(agent.book, "sell", new=AsyncMock(return_value=10.0)) as sell_mock:
            await agent._check_risk_exits()       # default order_executor=None
        sell_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_exit_exception_swallowed(self):
        import agents.exit_execution as exit_mod
        agent = self._agent_with_trail_setup(price=105.0)
        ex = self._live_executor()
        ex.execute_maker_signal.side_effect = RuntimeError("boom")
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])), \
             patch.object(agent.book, "sell", new=AsyncMock(return_value=10.0)) as sell_mock, \
             patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            # must NOT raise
            await agent._check_risk_exits(ex)
        sell_mock.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_cnn_risk_exits.py::TestCheckRiskExitsLiveRouting -v`
Expected: FAIL — `_check_risk_exits()` takes no `order_executor` arg / no executor calls made.

- [ ] **Step 3: Write minimal implementation**

In `backend/agents/cnn_agent.py`:

(a) Add the import near the other `agents.*` imports (top of file, alongside the `exit_thresholds` import):

```python
from agents import exit_execution
```

(b) Change the `_check_risk_exits` signature (currently `async def _check_risk_exits(self) -> None:`):

```python
    async def _check_risk_exits(self, order_executor=None) -> None:
```

(c) Replace the existing `if trigger:` block (the `pnl = await self.book.sell(...)` + logger.info) with:

```python
            if trigger:
                size = pos["size"]
                pnl  = await self.book.sell(pid, price, trigger=trigger)
                try:
                    await exit_execution.execute_live_exit(
                        order_executor, pid=pid, price=price,
                        size=size, trigger=trigger,
                    )
                except Exception:
                    logger.exception(
                        "live exit execution failed for %s (%s)", pid, trigger,
                    )
                logger.info(
                    f"CNN RISK EXIT {pid} @{price:.6f} | {trigger} | "
                    f"entry={pct_entry*100:+.2f}% peak={peak_price:.6f} "
                    f"peak_pnl={pos['peak_pnl_pct']*100:+.2f}% "
                    f"exit_thr={exit_threshold*100:+.2f}% (atr_trail={trail_pct*100:.1f}%) "
                    f"hold={hold_secs/3600:.1f}h | "
                    f"pnl=${pnl:+.2f} | balance=${self.book.balance:.2f}"
                )
```

(d) In `run_loop`, forward the executor (currently `await self._check_risk_exits()`):

```python
                await self._check_risk_exits(order_executor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_cnn_risk_exits.py -v`
Expected: PASS (new class + all pre-existing risk-exit tests still green — the default `order_executor=None` keeps them paper-only).

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/agents/cnn_agent.py backend/tests/test_cnn_risk_exits.py
git commit -m "feat: route scan-loop exits through maker/taker live execution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 3: Wire the WS exit path (`exit_watcher`) + `main.py`

**Files:**
- Modify: `backend/agents/exit_watcher.py` (`on_price_tick` + `attach` accept `order_executor`; live-exit call)
- Modify: `backend/main.py:425` (pass `app_state.order_executor` to `attach_exit_watcher`)
- Test: `backend/tests/test_exit_watcher.py` (append a new test class)

**Interfaces:**
- Consumes: `exit_execution.execute_live_exit(...)` from Task 1.
- Produces: `on_price_tick(pid, price, book, order_executor=None)`; `attach(ws_subscriber, book, order_executor=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_exit_watcher.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════
# Maker-execution exit leg — WS live routing
# ═══════════════════════════════════════════════════════════════════════════

class TestOnPriceTickLiveRouting:
    """on_price_tick places a live exit after the paper close, routed
    maker/taker by trigger, only when USE_MAKER_EXECUTION is on + executor live."""

    def _live_executor(self):
        ex = AsyncMock()
        ex.dry_run = False
        ex.execute_maker_signal.return_value = {"success": True, "fill_mode": "MAKER"}
        ex.execute_signal.return_value = {"success": True, "fill_mode": "TAKER"}
        return ex

    @pytest.mark.asyncio
    async def test_ws_trail_stop_routes_maker(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            await on_price_tick("BTC-USD", 105.0, book, ex)   # +5% << trail thr
        book.sell.assert_called_once_with("BTC-USD", 105.0, trigger="WS_TRAIL_STOP")
        ex.execute_maker_signal.assert_awaited_once()
        sig = ex.execute_maker_signal.call_args.args[0]
        assert sig["side"] == "SELL" and sig["signal_type"] == "WS_TRAIL_STOP"
        assert sig["quote_size"] == round(2.0 * 106.0, 2)

    @pytest.mark.asyncio
    async def test_ws_stop_loss_routes_taker(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["ETH-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await on_price_tick("ETH-USD", 91.0, book, ex)    # -9% -> WS_STOP_LOSS
        book.sell.assert_called_once_with("ETH-USD", 91.0, trigger="WS_STOP_LOSS")
        mock_bba.assert_not_called()
        ex.execute_signal.assert_awaited_once()
        ex.execute_maker_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_ws_model_down_routes_maker(self):
        import time as _t
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD
        book = _FakeBook()
        book.positions["ABC-USD"] = _make_pos(avg=100.0, peak=100.0, trail=0.06, size=2.0)
        book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05
        book.positions["ABC-USD"]["p_down_ts_ms"] = int(_t.time() * 1000)
        ex = self._live_executor()
        quotes = {"ABC-USD": {"bid": 100.5, "ask": 101.5}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            await on_price_tick("ABC-USD", 101.0, book, ex)   # +1% -> WS_MODEL_DOWN
        book.sell.assert_called_once_with("ABC-USD", 101.0, trigger="WS_MODEL_DOWN")
        ex.execute_maker_signal.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flag_off_stays_paper_only(self):
        import agents.exit_execution as exit_mod
        from agents.exit_watcher import on_price_tick
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        with patch.object(exit_mod.config, "use_maker_execution", False, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock()) as mock_bba:
            await on_price_tick("BTC-USD", 105.0, book, ex)
        book.sell.assert_called_once()
        mock_bba.assert_not_called()
        ex.execute_maker_signal.assert_not_called()
        ex.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_attach_forwards_order_executor(self):
        import agents.exit_execution as exit_mod
        from services.ws_subscriber import CoinbaseWSSubscriber
        from agents.exit_watcher import attach
        ws = CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
        book = _FakeBook()
        book.positions["BTC-USD"] = _make_pos(avg=100.0, peak=120.0, trail=0.06, size=2.0)
        ex = self._live_executor()
        quotes = {"BTC-USD": {"bid": 104.0, "ask": 106.0}}
        with patch.object(exit_mod.config, "use_maker_execution", True, create=True), \
             patch.object(exit_mod.coinbase_client, "get_best_bid_ask",
                          new=AsyncMock(return_value=quotes)):
            attach(ws, book, ex)
            await ws._handle({
                "channel": "ticker",
                "events": [{"tickers": [{"product_id": "BTC-USD", "price": "105.0"}]}],
            })
            await asyncio.sleep(0.05)
        ex.execute_maker_signal.assert_awaited_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_exit_watcher.py::TestOnPriceTickLiveRouting -v`
Expected: FAIL — `on_price_tick()` / `attach()` take no `order_executor` arg.

- [ ] **Step 3: Write minimal implementation**

(a) In `backend/agents/exit_watcher.py`, add the import (with the other `agents.*` imports):

```python
from agents import exit_execution
```

(b) Change `on_price_tick` signature:

```python
async def on_price_tick(pid: str, price: float, book: "_CNNBook",
                        order_executor=None) -> None:
```

(c) Replace the `if trigger:` block (currently `await book.sell(pid, price, trigger=trigger)`):

```python
        if trigger:
            size = pos.get("size", 0.0)
            await book.sell(pid, price, trigger=trigger)
            await exit_execution.execute_live_exit(
                order_executor, pid=pid, price=price, size=size, trigger=trigger,
            )
```

(The whole body is already inside the `try/except` that logs at ERROR — invariant #18 — so a live-exit failure is caught there.)

(d) Change `attach` to accept + forward the executor:

```python
def attach(ws_subscriber: "CoinbaseWSSubscriber", book: "_CNNBook",
           order_executor=None) -> None:
    """Register the per-tick exit handler. Call once per backend lifespan
    (in main.py after ws_subscriber.start() and after cnn_agent is built).
    """
    async def _handler(pid: str, price: float) -> None:
        await on_price_tick(pid, price, book, order_executor)

    ws_subscriber.register_price_handler(_handler)
    logger.info("exit_watcher attached to ws_subscriber")
```

(e) In `backend/main.py:425`, pass the executor:

```python
    attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book,
                        app_state.order_executor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_exit_watcher.py -v`
Expected: PASS (new class + all pre-existing WS exit tests still green — the default `order_executor=None` keeps them paper-only).

- [ ] **Step 5: Commit**

```bash
git rev-parse --abbrev-ref HEAD
git add backend/agents/exit_watcher.py backend/main.py backend/tests/test_exit_watcher.py
git commit -m "feat: route WS exits through maker/taker live execution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

---

### Task 4: Docs + memory sync, full-suite verification

**Files:**
- Modify: `CLAUDE.md` (amend invariant #21)
- Modify: `CHANGELOG.md` (new session entry)
- Modify: memory `coinbase_trader_session_log.md`, `win_factors_improvement_loop.md` (after commit)

- [ ] **Step 1: Amend CLAUDE.md invariant #21**

Replace the final sentence of invariant #21 ("**Entry leg only** — profit-target maker exits are a separate, not-yet-built piece.") with:

```
**Exit leg (this session):** the entire exit live-order path is gated behind
the same flag (exits place NO live order today, so flag-off MUST stay paper-only
— `book.sell` only). `agents/exit_execution.execute_live_exit` is the single
source of truth for exit routing: maker (post-only) for `TRAIL_STOP`/
`WS_TRAIL_STOP`/`MODEL_DOWN`/`WS_MODEL_DOWN`, taker for `STOP_LOSS`/
`WS_STOP_LOSS`/`MAX_HOLD`/`LEGACY_EXIT`. Both exit paths (`_check_risk_exits`
scan loop, `exit_watcher.on_price_tick` WS) close the paper book first, then
call `execute_live_exit`, which no-ops unless the flag is on and the executor is
live. A live-exit failure is caught + logged, never re-raised (invariants
#16/#18). Has zero effect on tracked paper PnL (paper book models no fees);
purely a live-execution-path change for the 8002 shadow.
```

- [ ] **Step 2: Add CHANGELOG entry**

Prepend a Session entry to `CHANGELOG.md` describing the exit leg (new `exit_execution.py`, both paths wired, trigger routing, default-off contract, TDD count).

- [ ] **Step 3: Run the full suite once**

Run: `cd backend && python -m pytest tests/ -q`
Expected: all pass (baseline 1284 passed / 65 skipped / 1 xfailed / 2 xpassed from 58.72, plus the new exit-leg tests). Note exact counts.

- [ ] **Step 4: Shell cleanup** (port-8001-aware snippet from CLAUDE.md).

- [ ] **Step 5: Commit docs**

```bash
git rev-parse --abbrev-ref HEAD
git add CLAUDE.md CHANGELOG.md
git commit -m "docs: maker-execution exit leg — invariant #21 + CHANGELOG

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git log -1 --stat
```

- [ ] **Step 6: Update memory** (after commit, per sync rule): append a Session entry to `coinbase_trader_session_log.md` and update the status block in `win_factors_improvement_loop.md` (entry leg + exit leg both shipped; next = 8002 shadow fill-rate validation).

---

## Self-Review

**Spec coverage:**
- New `exit_execution.py` module → Task 1. ✓
- Trigger routing table (maker/taker) → Task 1 `_MAKER_EXIT_TRIGGERS` + tests. ✓
- Scan path wiring + `run_loop` forward → Task 2. ✓
- WS path wiring + `attach` + `main.py` → Task 3. ✓
- Default-off byte-for-byte contract → Task 2/3 flag-off + no-executor tests. ✓
- Paper-first ordering → Task 2/3 implementation (sell before live exit). ✓
- Isolation (no re-raise) → Task 2 try/except + test; Task 3 existing handler try/except + test. ✓
- Sizing (`size*ask` maker / `size*price` taker) → Task 1 tests. ✓
- Reuse `config.use_maker_execution`, no new flag → Task 1 gate. ✓
- CLAUDE.md #21 amend + CHANGELOG + memory → Task 4. ✓
- Out-of-scope `_preflight` SELL wart → intentionally untouched (not in any task). ✓

**Placeholder scan:** none — every code/test step shows full content.

**Type consistency:** `execute_live_exit(order_executor, *, pid, price, size, trigger)` and `is_maker_exit(trigger)` used identically in Tasks 1–3. `on_price_tick(..., order_executor=None)` and `attach(..., order_executor=None)` consistent between Task 3 impl, tests, and `main.py` call. Signal keys (`product_id`, `side`, `price`, `signal_type`, `bid`, `ask`, `quote_size`) consistent with `order_executor.execute_maker_signal`/`execute_signal` expectations.
