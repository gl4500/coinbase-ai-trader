# Execution findings — confirm/refute + fix proposals (Task 5)

**Scope boundary:** investigation only. **No live-execution semantics were changed.** Each finding
has an isolated regression test in `backend/tests/test_execution_findings.py` that pins *current*
behaviour, including behaviour that is defective, so it cannot change silently. Fixes below are
proposals for review, not applied work.

**All four findings are CONFIRMED.** Two are documented as intentional in CLAUDE.md invariant #21;
that does not make their consequences safe, and both are stated below.

---

## Finding 1 — Enabling trading replaces the executor; background handlers keep the old one

**CONFIRMED.**

### Affected path

| Location | Code | Binding |
|---|---|---|
| `main.py:438` | `app_state.order_executor = OrderExecutor(dry_run=...)` | startup instance created |
| `main.py:459` | `attach_exit_watcher(ws, book, app_state.order_executor)` | **instance passed by value** |
| `exit_watcher.py:120-126` | `attach()` closes over `order_executor`, `_handler` calls `on_price_tick(..., order_executor)` | closure captures that instance forever |
| `main.py:502` | `cnn_agent.run_loop(order_executor=app_state.order_executor, ...)` | task created once; argument bound once |
| `main.py:1141` | `enable_trading`: `app_state.order_executor = OrderExecutor(dry_run=dry_run)` | **rebinds the name only** |

### Existing contract, and why this is clearly a defect

The very same `run_loop` call passes `is_trading_fn=lambda: app_state.is_trading` — a **callable**,
specifically so the flag is read fresh on every cycle. The executor beside it is passed **by value**.
The codebase already knows the late-binding pattern and applied it inconsistently to the two pieces of
state that change at the same moment.

After `POST /api/trading/enable`:

- the WS exit path and the scan loop hold the **startup** executor;
- request handlers (`main.py:816`, `1113`, `1124`, `1132`) read `app_state.order_executor` at call
  time and get the **new** one;
- the dashboard reads `app_state.order_executor._dry_run_balance` (`main.py:568`, `747`) and
  `.drawdown_status` (`751`) from the **new** instance.

`_dry_run_balance`, `_halted`, `_day_start_balance` and `_week_start_balance` are all per-instance
(`order_executor.py:46-56`). So automated trading debits one balance while the dashboard displays
another, and the drawdown circuit breaker accumulates on an instance nobody is reading.

**This is a plausible contributor to the class of state drift behind the $5.98 discrepancy** (see
`2026-09-26-accounting-reconciliation.md` §1), because it splits a single logical account into two
objects.

### Tests

`test_f1_attach_captures_the_executor_instance_not_the_live_reference`,
`test_f1_each_executor_instance_owns_its_own_paper_balance`

### Proposed fix

Pass the executor the way the trading flag is already passed — by accessor, not by value.

```python
# main.py
attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book,
                    lambda: app_state.order_executor)
...
app_state.cnn_agent.run_loop(executor_fn=lambda: app_state.order_executor, ...)
```

`exit_watcher.attach` and `cnn_agent.run_loop` resolve it per tick / per cycle.

*Alternative, preferred if a wider change is acceptable:* stop replacing the object. Give
`OrderExecutor` a `set_dry_run()` and have `enable_trading` mutate the single long-lived instance, so
no reference can ever go stale and balance/drawdown state survives an enable/disable cycle. This also
removes the reset-on-enable behaviour that currently discards accumulated drawdown state.

- **Behaviour change:** automated paths would use the post-enable executor. Intended.
- **Schema change:** none.
- **Risk:** low; signature changes are confined to two call sites plus the two consumers, both of
  which already accept `order_executor=None`.

---

## Finding 2 — Automated live risk exits are suppressed when the maker flag is off

**CONFIRMED.** Documented as intentional in CLAUDE.md invariant #21 ("flag-off MUST stay paper-only").

### Affected path

`exit_execution.execute_live_exit` (`exit_execution.py:54-59`) returns `None` unless
`config.use_maker_execution` is true and the executor is live:

```python
if (order_executor is None
        or getattr(order_executor, "dry_run", True)
        or not config.use_maker_execution):
    return None
```

Both exit paths — `cnn_agent._check_risk_exits` and `exit_watcher.on_price_tick` — route every
trigger (`STOP_LOSS`, `WS_STOP_LOSS`, `TRAIL_STOP`, `WS_TRAIL_STOP`, `MODEL_DOWN`, `WS_MODEL_DOWN`,
`MAX_HOLD`, `LEGACY_EXIT`) through it.

### The real risk is asymmetry, not the gate itself

Invariant #21 also specifies that with the flag **off**, live BUYs still route through the taker path
"exactly as before". So on a funded, non-dry-run account with `USE_MAKER_EXECUTION=false`:

- **entries place real orders**,
- **exits place none** — they only close the paper book.

Automation could therefore open real positions it can never close, with stop-losses that fire only in
the paper ledger. Today this is masked because the account runs `dry_run=true`; the gate is one
environment variable away from being load-bearing in the wrong direction.

### Tests

`test_f2_live_exit_no_ops_for_every_trigger_when_maker_flag_is_off`,
`test_f2_dry_run_executor_also_no_ops_even_with_the_flag_on`

### Proposed fix (needs review — invariant #21 is deliberate)

Gate the *routing style*, not the *existence* of the exit. `USE_MAKER_EXECUTION` should choose
maker-vs-taker for exits; it should not decide whether a live account exits at all:

- flag **on** → current behaviour (maker for trail/model-down, taker for stops/max-hold);
- flag **off** → route **all** exits through the plain taker path, exactly as entries already do.

Safer interim option if that is too large a change: make the asymmetry impossible to hit by refusing
to run live at all in the inconsistent configuration — fail startup when
`dry_run == False and use_maker_execution == False`, with an explicit error naming the reason.

- **Behaviour change:** yes, on live accounts only; none while `dry_run=true`.
- **Schema change:** none.
- **Requires:** an update to invariant #21 and an operator decision. Not to be applied silently.

---

## Finding 3 — The paper book closes before exchange execution is confirmed

**CONFIRMED.** Also documented in invariant #21 ("close the paper book first, then call
`execute_live_exit`").

### Affected path

`exit_watcher.py:100-108`:

```python
size = pos.get("size", 0.0)
await book.sell(pid, price, trigger=trigger)          # paper close committed
await exit_execution.execute_live_exit(...)           # exchange attempt afterwards
```

`cnn_agent._check_risk_exits` follows the same order. `execute_live_exit` catches and logs its own
failures (invariants #16/#18) and never re-raises, and `on_price_tick` wraps everything in a blanket
`except Exception: logger.exception(...)`. So when a live exit fails, the paper book already records
the position as closed and nothing reconciles the two.

### Tests

`test_f3_paper_book_is_closed_before_the_live_exit_is_attempted`,
`test_f3_paper_close_stands_even_when_the_live_exit_raises`

### Proposed fix

Make the book state reflect exchange reality rather than intent:

1. Introduce an intermediate position state — `exiting` / `pending_close` — written *before* the
   exchange call.
2. Commit the close only on confirmed fill; on failure revert to `open` and let the next tick retry.
3. On partial fill, reduce the position by `filled_size` instead of closing it.
4. Emit a reconciliation warning whenever a paper close has no corresponding confirmed fill.

- **Behaviour change:** yes — exits become two-phase. Paper P&L timing shifts slightly.
- **Schema change:** yes — a position status field, and the `order_id`/`fill_id`/`filled_size`
  columns Task 6 §5 already requires on `trades`.
- **Dependency:** pointless until `orders` is actually populated (currently 0 rows), so this should
  land together with the provenance work, not before it.

---

## Finding 4 — Maker timeout fallback can submit a market order after cancellation fails

**CONFIRMED — and broader than the finding as stated.**

### Affected path

`order_executor.py:418-432`:

```python
try:
    await coinbase_client.cancel_orders([order_id])
    await database.update_order_status(order_id, "canceled")
except Exception as e:
    logger.error(f"Cancel during maker timeout failed: {e}")   # swallowed

try:
    mkt_resp = await coinbase_client.place_market_order(pid, side, quote_size)
```

Three distinct problems:

1. **The cancel exception is swallowed and execution falls through** to `place_market_order`
   unconditionally. If the resting post-only limit is still live, the account can end up with roughly
   **double the intended exposure**.
2. **The cancel response body is never inspected.** `cancel_orders` returning
   `{"results": [{"success": false, ...}]}` is treated as success, so a cleanly-reported cancel
   failure also falls through. This needs no exception at all to trigger.
3. **A partial fill is treated as no fill.** `_wait_for_fill` (`order_executor.py:295`) returns True
   only on `status == "FILLED"`, so a partially filled maker order proceeds to a **full-size** market
   order on top of the filled portion.

There is also an inherent race: the limit can fill between the poll timing out and the cancel
landing, which is exactly the case where cancel legitimately fails.

### Tests

`test_f4_market_order_is_placed_even_when_cancel_raises`,
`test_f4_cancel_response_body_is_never_inspected`,
`test_f4_partial_fill_is_treated_as_no_fill`

### Proposed fix

**The market fallback must be conditional on a confirmed, complete cancel.**

```
1. poll -> not filled
2. cancel
3. re-query the order's terminal state (do NOT trust the cancel response alone)
4. branch on what the exchange says:
     CANCELLED, filled_size == 0   -> place market order for the full quote_size
     CANCELLED, 0 < filled_size    -> place market order for the REMAINDER only
     FILLED                        -> place nothing; report fill_mode=MAKER
     anything else / unknown       -> place nothing; return success=False and
                                      surface it for operator attention
```

Never place the fallback while the limit's state is unknown. Sizing must come from
`quote_size - filled_notional`, never the original `quote_size`.

- **Behaviour change:** yes — some timeouts that currently force an entry would decline it. That is
  the point: a missed entry is cheap, double exposure is not.
- **Schema change:** none strictly; recording `filled_size`/`avg_fill_price` on the maker order (Task
  6 §5) makes the remainder computation auditable.
- **Risk:** this is the only one of the four that can lose real money on a funded account, and it is
  reachable as soon as `USE_MAKER_EXECUTION=true` goes live. Recommend fixing it **before** the 8002
  maker shadow is promoted.

---

## Summary

| # | Finding | Verdict | Documented in CLAUDE.md | Fix risk | Recommended order |
|---|---|---|---|---|---|
| 4 | Market fallback after failed cancel | CONFIRMED (+2 extra defects) | No | Low | **1st — real-money exposure** |
| 1 | Stale executor after enable | CONFIRMED | No | Low | 2nd |
| 2 | Exits suppressed by maker flag | CONFIRMED | Yes, invariant #21 | Medium — needs operator decision | 3rd |
| 3 | Paper close before confirmation | CONFIRMED | Yes, invariant #21 | High — schema + two-phase exits | 4th, with provenance work |

Findings 2 and 3 require an invariant #21 amendment and an explicit operator decision; they are
presented for review rather than proposed for immediate implementation.
