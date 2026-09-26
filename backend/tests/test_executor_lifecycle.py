"""Executor lifecycle across enable / disable / re-enable (execution finding 1).

Audit: docs/audits/2026-09-26-strategy-audit-report.md section 5
Analysis: docs/handoffs/2026-09-26-execution-findings.md finding 1

`POST /api/trading/enable` rebinds `app_state.order_executor`, but the two
long-running automated paths captured the startup instance by value:

  * main.py passed the instance into `attach_exit_watcher`, which closes over it;
  * main.py bound it as a `run_loop` argument once, at task-creation time.

The tell is that the adjacent `is_trading_fn=lambda: app_state.is_trading` is
passed as a *callable* precisely so it reads fresh. These tests pin that both
consumers now accept a resolver and read the current instance, that they resolve
exactly **once per operation** (so a single tick or cycle can never mix two
objects across an await), and that passing a plain instance still works.

The last test documents what late binding does NOT fix.
"""

import asyncio
import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

from agents import exit_execution, exit_watcher  # noqa: E402
from agents import order_executor as oe  # noqa: E402


class _AppState:
    """Stands in for main.py's app_state: one mutable attribute, rebound on enable."""

    def __init__(self, executor):
        self.order_executor = executor


class _WS:
    def __init__(self):
        self.handler = None

    def register_price_handler(self, fn):
        self.handler = fn


class _Book:
    balance = 1000.0

    def __init__(self, with_position=True):
        self.positions = (
            {
                "BTC-USD": {
                    "avg_price": 100.0,
                    "peak_price": 100.0,
                    "size": 1.0,
                    "position_dollars": 100.0,
                }
            }
            if with_position
            else {}
        )
        self.sold = []

    async def sell(self, pid, price, trigger=None):
        self.sold.append(trigger)


# ── WebSocket exit path ───────────────────────────────────────────────────────


async def test_ws_path_uses_the_current_executor_after_re_enable(monkeypatch):
    seen = []

    async def _spy(pid, price, book, order_executor=None):
        seen.append(order_executor)

    monkeypatch.setattr(exit_watcher, "on_price_tick", _spy)

    startup = oe.OrderExecutor(dry_run=True)
    state = _AppState(startup)
    ws = _WS()
    exit_watcher.attach(ws, _Book(), executor_fn=lambda: state.order_executor)

    await ws.handler("BTC-USD", 100.0)

    replacement = oe.OrderExecutor(dry_run=True)
    state.order_executor = replacement  # what enable_trading does
    await ws.handler("BTC-USD", 100.0)

    assert seen == [startup, replacement]


async def test_ws_path_accepts_a_plain_instance_for_back_compat(monkeypatch):
    seen = []

    async def _spy(pid, price, book, order_executor=None):
        seen.append(order_executor)

    monkeypatch.setattr(exit_watcher, "on_price_tick", _spy)

    executor = oe.OrderExecutor(dry_run=True)
    ws = _WS()
    exit_watcher.attach(ws, _Book(), executor)

    await ws.handler("BTC-USD", 100.0)

    assert seen == [executor]


async def test_ws_path_resolves_exactly_once_per_tick(monkeypatch):
    """Resolving more than once inside one tick could mix two objects across an
    await, which is worse than holding one stale object."""
    calls = {"n": 0}
    executor = oe.OrderExecutor(dry_run=True)

    def _resolver():
        calls["n"] += 1
        return executor

    async def _spy(pid, price, book, order_executor=None):
        return None

    monkeypatch.setattr(exit_watcher, "on_price_tick", _spy)

    ws = _WS()
    exit_watcher.attach(ws, _Book(), executor_fn=_resolver)
    await ws.handler("BTC-USD", 100.0)

    assert calls["n"] == 1


async def test_ws_resolver_failure_does_not_crash_the_tick(monkeypatch):
    """Invariant #18: the tick handler must never raise into the WS loop."""

    def _boom():
        raise RuntimeError("app_state not ready")

    ws = _WS()
    exit_watcher.attach(ws, _Book(), executor_fn=_boom)

    await ws.handler("BTC-USD", 80.0)  # must not raise


async def test_ws_exit_actually_fires_with_a_resolver(monkeypatch):
    """End to end: a 20% drop still trips WS_STOP_LOSS when late-bound."""
    passed = []

    async def _fake_live_exit(executor, **kwargs):
        passed.append(executor)
        return None

    monkeypatch.setattr(exit_execution, "execute_live_exit", _fake_live_exit)

    executor = oe.OrderExecutor(dry_run=True)
    state = _AppState(executor)
    book = _Book()
    ws = _WS()
    exit_watcher.attach(ws, book, executor_fn=lambda: state.order_executor)

    await ws.handler("BTC-USD", 80.0)

    assert book.sold == ["WS_STOP_LOSS"]
    assert passed == [executor]


# ── Scan-loop path ────────────────────────────────────────────────────────────


def _agent():
    from agents.cnn_agent import CoinbaseCNNAgent

    return CoinbaseCNNAgent(ws_subscriber=None)


async def _drive_run_loop(agent, monkeypatch, executor_arg, cycles=2):
    """Run `cycles` iterations of run_loop, then cancel it. Returns the executor
    object handed to each _scan_cycle call."""
    seen = []

    async def _no_start():
        return None

    async def _spy_cycle(execute, order_executor, timeout):
        seen.append(order_executor)
        if len(seen) >= cycles:
            raise asyncio.CancelledError

    monkeypatch.setattr(agent, "start", _no_start)
    monkeypatch.setattr(agent, "_scan_cycle", _spy_cycle)

    kwargs = (
        {"executor_fn": executor_arg}
        if callable(executor_arg)
        else {"order_executor": executor_arg}
    )
    await agent.run_loop(interval=0, is_trading_fn=lambda: True, **kwargs)
    return seen


async def test_scan_loop_uses_the_current_executor_after_re_enable(monkeypatch):
    startup = oe.OrderExecutor(dry_run=True)
    replacement = oe.OrderExecutor(dry_run=True)
    state = _AppState(startup)
    swapped = {"done": False}

    def _resolver():
        # First cycle sees the startup instance; enable_trading then rebinds.
        if swapped["done"]:
            return state.order_executor
        swapped["done"] = True
        return state.order_executor

    agent = _agent()
    seen = []

    async def _no_start():
        return None

    async def _spy_cycle(execute, order_executor, timeout):
        seen.append(order_executor)
        if len(seen) == 1:
            state.order_executor = replacement  # enable_trading fires here
        if len(seen) >= 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(agent, "start", _no_start)
    monkeypatch.setattr(agent, "_scan_cycle", _spy_cycle)

    await agent.run_loop(interval=0, executor_fn=_resolver, is_trading_fn=lambda: True)

    assert seen == [startup, replacement]


async def test_scan_loop_accepts_a_plain_instance_for_back_compat(monkeypatch):
    executor = oe.OrderExecutor(dry_run=True)
    agent = _agent()
    seen = await _drive_run_loop(agent, monkeypatch, executor, cycles=2)
    assert seen == [executor, executor]


async def test_scan_loop_resolves_exactly_once_per_cycle(monkeypatch):
    calls = {"n": 0}
    executor = oe.OrderExecutor(dry_run=True)

    def _resolver():
        calls["n"] += 1
        return executor

    agent = _agent()
    await _drive_run_loop(agent, monkeypatch, _resolver, cycles=3)

    assert calls["n"] == 3  # one per cycle, not per use within a cycle


async def test_a_callable_executor_is_not_mistaken_for_a_resolver(monkeypatch):
    """MagicMock and any __call__-defining object are callable, so a callable()
    heuristic would invoke the executor instead of using it. The explicit
    executor_fn parameter removes the ambiguity."""
    from unittest.mock import MagicMock

    seen = []

    async def _spy(pid, price, book, order_executor=None):
        seen.append(order_executor)

    monkeypatch.setattr(exit_watcher, "on_price_tick", _spy)

    callable_executor = MagicMock()  # callable, but IS the executor
    ws = _WS()
    exit_watcher.attach(ws, _Book(), callable_executor)
    await ws.handler("BTC-USD", 100.0)

    assert seen == [callable_executor]
    callable_executor.assert_not_called()


# ── What late binding does NOT fix ────────────────────────────────────────────


def test_replacing_the_executor_still_discards_risk_state():
    """UNRESOLVED, recorded deliberately.

    `enable_trading` constructs a NEW OrderExecutor, so the drawdown circuit
    breaker and paper balance start from scratch on every enable/disable/
    re-enable cycle. Late binding fixes *which* object the automated paths see;
    it does not preserve state across the swap. Fixing that means mutating a
    single long-lived executor instead of replacing it, which needs a safe
    transition contract for an in-flight executor and is deliberately out of
    scope here.
    """
    first = oe.OrderExecutor(dry_run=True)
    first._dry_run_balance = 123.45
    first._halted = True
    first._halt_reason = "daily drawdown"

    second = oe.OrderExecutor(dry_run=True)  # what enable_trading builds

    assert second._dry_run_balance != 123.45
    assert second._halted is False
    assert second._halt_reason == ""
