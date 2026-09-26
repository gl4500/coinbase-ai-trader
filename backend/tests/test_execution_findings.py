"""Characterisation tests for the four execution findings (Task 5).

Audit: docs/audits/2026-09-26-strategy-audit-report.md section 5
Analysis + proposed fixes: docs/handoffs/strategy-prerequisites.md section 9

READ THIS BEFORE "FIXING" A FAILURE HERE.

These tests pin **current** behaviour, including behaviour that is defective.
They exist so that the defect cannot change silently and so a deliberate fix has
to update an explicit assertion. Task 5 was scoped to investigate without
altering live-execution semantics, and findings 2 and 3 are documented as
intentional in CLAUDE.md invariant #21.

No test here places an order: every exchange call is a stub.
"""

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

import pytest  # noqa: E402

from agents import exit_execution, exit_watcher  # noqa: E402
from agents import order_executor as oe
from config import config  # noqa: E402

# ── Finding 1: enabling trading leaves background handlers on the old executor ─


async def test_f1_attach_captures_the_executor_instance_not_the_live_reference(monkeypatch):
    """CONFIRMED. `exit_watcher.attach` closes over the executor *object*.

    Rebinding `app_state.order_executor` (which POST /api/trading/enable does)
    cannot reach that closure, so the WS exit path keeps using the executor
    built at startup. Contrast `is_trading_fn=lambda: app_state.is_trading`,
    which main.py passes as a callable precisely so it reads fresh.
    """
    startup_executor = oe.OrderExecutor(dry_run=True)
    replacement_executor = oe.OrderExecutor(dry_run=True)
    assert startup_executor is not replacement_executor

    seen = []

    async def _spy(pid, price, book, order_executor=None):
        seen.append(order_executor)

    monkeypatch.setattr(exit_watcher, "on_price_tick", _spy)

    class _WS:
        def __init__(self):
            self.handler = None

        def register_price_handler(self, fn):
            self.handler = fn

    ws = _WS()
    book = type("B", (), {"positions": {}})()
    exit_watcher.attach(ws, book, startup_executor)

    # Simulate what enable_trading does: rebind the shared reference.
    class _AppState:
        order_executor = startup_executor

    _AppState.order_executor = replacement_executor

    await ws.handler("BTC-USD", 100.0)

    assert seen == [startup_executor], "handler should still hold the startup instance"
    assert seen[0] is not _AppState.order_executor, (
        "this inequality IS the defect: the automated path and the request "
        "handlers now use different executor instances, with separate "
        "_dry_run_balance and drawdown state"
    )


def test_f1_each_executor_instance_owns_its_own_paper_balance():
    """Why finding 1 matters for accounting: balance is per-instance state."""
    a = oe.OrderExecutor(dry_run=True)
    b = oe.OrderExecutor(dry_run=True)
    a._dry_run_balance = 123.45
    assert b._dry_run_balance != 123.45


# ── Finding 2: live risk exits are suppressed while the maker flag is off ──────


async def test_f2_live_exit_no_ops_for_every_trigger_when_maker_flag_is_off(monkeypatch):
    """CONFIRMED, and documented as intentional in CLAUDE.md invariant #21.

    The risk is the *asymmetry*: entries route live through the taker path
    regardless of this flag, while exits no-op. On a funded account with the
    flag off, automation could open positions it would never close.
    """
    monkeypatch.setattr(config, "use_maker_execution", False)
    live_executor = oe.OrderExecutor(dry_run=False)

    for trigger in (
        "STOP_LOSS",
        "WS_STOP_LOSS",
        "TRAIL_STOP",
        "WS_TRAIL_STOP",
        "MODEL_DOWN",
        "MAX_HOLD",
    ):
        result = await exit_execution.execute_live_exit(
            live_executor, pid="BTC-USD", price=100.0, size=1.0, trigger=trigger
        )
        assert result is None, f"{trigger} unexpectedly placed a live exit"


async def test_f2_dry_run_executor_also_no_ops_even_with_the_flag_on(monkeypatch):
    monkeypatch.setattr(config, "use_maker_execution", True)
    paper_executor = oe.OrderExecutor(dry_run=True)
    result = await exit_execution.execute_live_exit(
        paper_executor, pid="BTC-USD", price=100.0, size=1.0, trigger="STOP_LOSS"
    )
    assert result is None


# ── Finding 3: the paper book closes before the exchange confirms ──────────────


async def test_f3_paper_book_is_closed_before_the_live_exit_is_attempted(monkeypatch):
    """CONFIRMED. `book.sell` is awaited first, then `execute_live_exit`."""
    calls = []

    class _Book:
        balance = 1000.0

        def __init__(self):
            self.positions = {
                "BTC-USD": {
                    "avg_price": 100.0,
                    "peak_price": 100.0,
                    "size": 1.0,
                    "position_dollars": 100.0,
                }
            }

        async def sell(self, pid, price, trigger=None):
            calls.append(("book.sell", trigger))

    async def _fake_live_exit(executor, **kwargs):
        calls.append(("execute_live_exit", kwargs["trigger"]))
        return None

    monkeypatch.setattr(exit_execution, "execute_live_exit", _fake_live_exit)

    # A 20% drop trips WS_STOP_LOSS.
    await exit_watcher.on_price_tick("BTC-USD", 80.0, _Book(), oe.OrderExecutor(dry_run=True))

    assert [c[0] for c in calls] == ["book.sell", "execute_live_exit"]


async def test_f3_paper_close_stands_even_when_the_live_exit_raises(monkeypatch):
    """The paper book is already closed, and the exception is swallowed by
    invariant #18's blanket handler, so the two records diverge silently."""
    sold = []

    class _Book:
        balance = 1000.0

        def __init__(self):
            self.positions = {
                "BTC-USD": {
                    "avg_price": 100.0,
                    "peak_price": 100.0,
                    "size": 1.0,
                    "position_dollars": 100.0,
                }
            }

        async def sell(self, pid, price, trigger=None):
            sold.append(trigger)

    async def _boom(executor, **kwargs):
        raise RuntimeError("exchange rejected the exit")

    monkeypatch.setattr(exit_execution, "execute_live_exit", _boom)

    await exit_watcher.on_price_tick("BTC-USD", 80.0, _Book(), oe.OrderExecutor(dry_run=True))

    assert sold == ["WS_STOP_LOSS"], "paper book recorded the close"


# ── Finding 4: market fallback runs even when the cancel failed ────────────────


@pytest.fixture
def maker_env(monkeypatch):
    """Minimal stubbing to reach the timeout-fallback block, no real calls."""
    executor = oe.OrderExecutor(dry_run=False)

    async def _no_drawdown(self):
        return None

    async def _no_preflight(self, quote_size):
        return None

    monkeypatch.setattr(oe.OrderExecutor, "_check_drawdown", _no_drawdown)
    monkeypatch.setattr(oe.OrderExecutor, "_preflight", _no_preflight)

    async def _never_fills(self, pid, order_id, timeout_secs):
        return False

    monkeypatch.setattr(oe.OrderExecutor, "_wait_for_fill", _never_fills)

    async def _place_limit(*a, **k):
        return {"success_response": {"order_id": "limit-1"}}

    async def _save_order(*a, **k):
        return None

    async def _update_status(*a, **k):
        return None

    monkeypatch.setattr(oe.coinbase_client, "place_limit_order", _place_limit)
    monkeypatch.setattr(oe.database, "save_order", _save_order)
    monkeypatch.setattr(oe.database, "update_order_status", _update_status)
    return executor


async def test_f4_market_order_is_placed_even_when_cancel_raises(maker_env, monkeypatch):
    """CONFIRMED. The cancel exception is logged and execution falls through.

    Consequence: if the resting limit is still live (or filled in the race
    between the poll timing out and the cancel landing), the account can end up
    with roughly double the intended exposure.
    """
    market_calls = []

    async def _cancel_raises(order_ids):
        raise RuntimeError("cancel failed")

    async def _place_market(pid, side, quote_size):
        market_calls.append((pid, side, quote_size))
        return {"success_response": {"order_id": "market-1"}}

    monkeypatch.setattr(oe.coinbase_client, "cancel_orders", _cancel_raises)
    monkeypatch.setattr(oe.coinbase_client, "place_market_order", _place_market)

    result = await maker_env.execute_maker_signal(
        {"product_id": "BTC-USD", "side": "BUY", "bid": 100.0, "ask": 100.1, "quote_size": 50.0},
        timeout_secs=0.01,
    )

    assert market_calls == [("BTC-USD", "BUY", 50.0)]
    assert result["fill_mode"] == "TAKER_FALLBACK"
    assert result["success"] is True


async def test_f4_cancel_response_body_is_never_inspected(maker_env, monkeypatch):
    """A cancel that reports failure *without raising* also falls through."""
    market_calls = []

    async def _cancel_reports_failure(order_ids):
        return {"results": [{"success": False, "failure_reason": "UNKNOWN_CANCEL_ORDER"}]}

    async def _place_market(pid, side, quote_size):
        market_calls.append(quote_size)
        return {"success_response": {"order_id": "market-2"}}

    monkeypatch.setattr(oe.coinbase_client, "cancel_orders", _cancel_reports_failure)
    monkeypatch.setattr(oe.coinbase_client, "place_market_order", _place_market)

    await maker_env.execute_maker_signal(
        {"product_id": "ETH-USD", "side": "BUY", "bid": 10.0, "ask": 10.01, "quote_size": 25.0},
        timeout_secs=0.01,
    )

    assert market_calls == [25.0], "fallback ran despite a failed cancel"


async def test_f4_partial_fill_is_treated_as_no_fill(maker_env, monkeypatch):
    """`_wait_for_fill` only accepts status == FILLED, so a partially filled
    maker order still triggers a full-size market order on top of it."""
    market_calls = []

    async def _cancel_ok(order_ids):
        return {"results": [{"success": True}]}

    async def _place_market(pid, side, quote_size):
        market_calls.append(quote_size)
        return {"success_response": {"order_id": "market-3"}}

    monkeypatch.setattr(oe.coinbase_client, "cancel_orders", _cancel_ok)
    monkeypatch.setattr(oe.coinbase_client, "place_market_order", _place_market)

    await maker_env.execute_maker_signal(
        {"product_id": "SOL-USD", "side": "BUY", "bid": 20.0, "ask": 20.02, "quote_size": 40.0},
        timeout_secs=0.01,
    )

    # Full quote_size, not the unfilled remainder.
    assert market_calls == [40.0]
