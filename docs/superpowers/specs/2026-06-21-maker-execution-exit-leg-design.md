# Maker-Execution Exit Leg — Design

**Date:** 2026-06-21
**Branch:** `feat/maker-exit-leg` (stacked on `feat/maker-execution-shadow`, entry leg `c262efc`)
**Status:** approved design, pre-implementation
**Related:** Session 58.72 (entry leg), `win_factors_improvement_loop`, CLAUDE.md invariant #21

## Background

The win-factors analysis concluded the durable edge is **execution-cost efficiency**
(maker/post-only fills), not direction prediction. Read-only sims showed routing
profit/trail exits through maker flips full-history paper PnL from −$414 (taker) to
+$169. Session 58.72 shipped the **entry leg**: `config.use_maker_execution` (env
`USE_MAKER_EXECUTION`, default false) gates `cnn_agent._execute_live_order`, which on
the flag-on path sources best bid/ask and routes BUYs through
`order_executor.execute_maker_signal`. The entry leg explicitly deferred the **exit
leg** (profit-target maker exits, disaster stops stay taker) as the next piece.

## Critical baseline difference from the entry leg

Entries **already** place live taker orders today, so the entry leg's default-off path
was "unchanged taker." **Exits place no live order at all today** — both exit paths
close positions only via the paper book (`_CNNBook.sell`):

- `cnn_agent._check_risk_exits` (scan loop, fires every cycle)
- `agents/exit_watcher.on_price_tick` (WS path, ~98% of live exits, all `WS_TRAIL_STOP`)

Therefore, to keep `USE_MAKER_EXECUTION=false` **byte-for-byte unchanged**, the *entire*
exit live-order leg must sit behind the flag. When off: exits stay paper-only exactly as
today (no bid/ask fetch, no live order). When on: exits place a live SELL alongside the
paper close, routed maker/taker by trigger.

This is intentionally asymmetric with the entry leg (where taker live orders fire even
when the flag is off) — the asymmetry is forced by the no-live-exits-today baseline plus
the default-off contract.

Note also: the paper book models **no fees** (`proceeds = size * price`); tracked 8001
PnL is gross. So this leg, like the entry leg, has **zero effect on tracked paper PnL** —
it is purely a live-execution-path change for the 8002 shadow / future live trading. The
+$169 figure came from offline sims applying assumed fees, not from the paper book.

## Trigger routing

| Trigger | Mode | Rationale |
|---|---|---|
| `TRAIL_STOP`, `WS_TRAIL_STOP` | **maker** | profit-protecting pullback from peak; position usually green; can wait for a fill (30s market fallback covers non-fill) |
| `MODEL_DOWN`, `WS_MODEL_DOWN` | **maker** | managed exit; sits between STOP_LOSS and TRAIL_STOP in the ladder; position not necessarily underwater; 30s fallback still guarantees exit |
| `STOP_LOSS`, `WS_STOP_LOSS` | **taker** | capital protection — must cross now |
| `MAX_HOLD`, `LEGACY_EXIT` | **taker** | forced time exit |

Matches the win-factors sim (`trail-exit` maker, `stop/maxhold-exit` taker);
`MODEL_DOWN` classification confirmed by operator on 2026-06-21.

## Components

### 1. New module `backend/agents/exit_execution.py`

Single-responsibility policy/adapter — the one place that classifies an exit trigger and
places the live order. Both exit paths call it; neither duplicates the routing
(loose coupling per `feedback_loose_coupling`).

```python
_MAKER_EXIT_TRIGGERS = frozenset(
    {"TRAIL_STOP", "WS_TRAIL_STOP", "MODEL_DOWN", "WS_MODEL_DOWN"}
)

def is_maker_exit(trigger: str) -> bool:
    return trigger in _MAKER_EXIT_TRIGGERS

async def execute_live_exit(
    order_executor,
    *,
    pid: str,
    price: float,
    size: float,
    trigger: str,
    bid: float = 0.0,
    ask: float = 0.0,
) -> Optional[Dict]:
    """Place a live SELL liquidating `size` of `pid`, routed maker/taker by trigger.

    Builds a SELL signal with NO `atr` key so order_executor sizes from `quote_size`.
    Sets quote_size so order_executor's `base_size = quote_size / fill_price` rounds
    back to the exact held `size`:
      - maker SELL fills at ask  -> quote_size = size * ask
      - taker SELL fills at price -> quote_size = size * price
    Returns the executor result dict, or None if quotes are missing for a maker exit.
    """
```

- For a maker trigger: requires `bid > 0 and ask > 0`; attaches them to the signal and
  calls `order_executor.execute_maker_signal(signal)`.
- For a taker trigger: calls `order_executor.execute_signal(signal)`.
- `signal_type` is set to the trigger string for order-row traceability.

### 2. Scan path — `cnn_agent._check_risk_exits`

After computing `trigger` and before/at the existing `book.sell` call:

1. Capture `size = pos["size"]` **before** `book.sell` pops the position.
2. `pnl = await self.book.sell(pid, price, trigger=trigger)` (unchanged — paper book is
   the source of truth, mirrors entry-leg ordering: paper first, then live).
3. If `order_executor and not order_executor.dry_run and config.use_maker_execution`:
   - If `exit_execution.is_maker_exit(trigger)`: fetch `bid/ask` via
     `coinbase_client.get_best_bid_ask([pid])`.
   - `await exit_execution.execute_live_exit(order_executor, pid=pid, price=price,
     size=size, trigger=trigger, bid=bid, ask=ask)`.
   - Wrap in try/except + log (a live-order failure must not crash the scan loop or
     poison subsequent positions).

`run_loop` already holds `order_executor`; forward it: `_check_risk_exits(order_executor)`.
Default `order_executor=None` keeps the method callable without live execution (tests,
paused trading).

### 3. WS path — `exit_watcher.on_price_tick` + `attach`

- `attach(ws_subscriber, book, order_executor=None)` — capture `order_executor` in the
  handler closure.
- `on_price_tick(pid, price, book, order_executor=None)` — after the existing
  `book.sell(pid, price, trigger=trigger)`:
  - capture `size` from `pos` before the sell;
  - same gated `execute_live_exit` call, inside the existing `try/except` that already
    guards the handler (invariant #18).

### 4. `main.py`

Line ~425: `attach_exit_watcher(app_state.ws_subscriber, app_state.cnn_agent.book,
app_state.order_executor)`.

### 5. Config & invariants

- **No new flag** — reuses `config.use_maker_execution`.
- **CLAUDE.md invariant #21** amended: remove "Entry leg only — ... not-yet-built";
  document the exit leg (entire exit live-order path gated behind the flag; maker triggers
  TRAIL/MODEL_DOWN, taker triggers STOP/MAX_HOLD; both scan + WS paths covered).

## Data flow (flag-on, maker trail exit, WS path)

```
WS tick -> on_price_tick: price < exit_threshold -> trigger=WS_TRAIL_STOP
  -> size = pos["size"]
  -> book.sell(pid, price, WS_TRAIL_STOP)        # paper close (source of truth)
  -> flag on & not dry_run:
       is_maker_exit(WS_TRAIL_STOP) -> True
       get_best_bid_ask([pid]) -> bid, ask
       execute_live_exit(... ask ...) -> execute_maker_signal(SELL @ ask, post_only,
                                          30s poll, market fallback)
```

## Error handling

- Scan path: new try/except around the live-exit block; log at ERROR; never re-raise.
- WS path: reuse the existing handler try/except (invariant #18); the live-exit block is
  inside it.
- `execute_live_exit` returns `None` (no-op) when a maker exit lacks quotes, rather than
  raising — the paper close already happened, so a missing quote must not error.

## Testing (TDD)

**New `tests/test_exit_execution.py`:**
- `is_maker_exit` classification: each of the 8 triggers maps to the correct mode.
- maker trigger routes to `execute_maker_signal` with `bid`/`ask` attached and
  `quote_size == size * ask`; signal carries no `atr` key.
- taker trigger routes to `execute_signal` with `quote_size == size * price`.
- maker exit with missing/zero quotes returns `None`, calls neither executor method.
- `signal_type == trigger`.

**Extend `tests/test_cnn_risk_exits.py`:**
- flag-on + not dry_run: a `TRAIL_STOP` places a live maker exit (mock executor);
  a `STOP_LOSS` places a live taker exit.
- flag-off (default): no executor method called, no `get_best_bid_ask` call — paper-only,
  unchanged.
- dry_run executor: no live exit placed.
- live-exit exception is swallowed (scan loop continues).

**Extend `tests/test_exit_watcher.py`:**
- flag-on: `WS_TRAIL_STOP` → maker, `WS_STOP_LOSS` → taker, `WS_MODEL_DOWN` → maker.
- flag-off: paper-only, unchanged (no executor / quote calls).
- `attach` forwards `order_executor` into the handler.

Full suite run once before commit (~1284 baseline from 58.72).

## Out of scope (known limitations)

- **`_preflight` gates SELLs on USD balance** (semantically wrong for a sell, which
  receives USD). Incidentally fine in the non-dry-run mode where the live path actually
  runs (real USD balance > sell proceeds). Left as-is to keep the change symmetric with
  the entry leg and avoid altering shared `execute_signal` behavior. Tracked as a
  follow-up, not fixed here.
- **SCAN sells** already route through the entry leg's `_execute_live_order`
  (generate_signal SELL branch) — unchanged by this leg.
- **Partial fills / size precision**: the `quote_size = size * fill_price` trick relies on
  `base_size = round(quote_size / fill_price, 8)` recovering `size`; rounding is at 1e-8,
  acceptable for the shadow. No partial-fill reconciliation (the paper book already closed
  the full position).

## Deployment

Per port discipline: operator launches the 8002 shadow with
`USE_MAKER_EXECUTION=true PORT=8002 python main.py`. The live exit path only fires when
`not dry_run`; promotion to 8001 is gated on the shadow confirming real maker fill rates
(fills are a live property, not backtestable).
