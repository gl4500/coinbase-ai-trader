# TechAgent Trailing Dollar Exit — Design

**Date:** 2026-04-24
**Owner:** gl4500
**Status:** approved (pending user spec review)
**Scope:** `backend/agents/tech_agent_cb.py`, `backend/tests/test_tech_agent_cb.py`

---

## Goal

Capture small profit gains on TechAgent positions before they reverse.
Add a per-position trailing take-profit measured in dollars: arm at
≥ $1.00 unrealized PnL, fire SELL when the position has given back
≥ $0.25 from its peak unrealized PnL.

This sits alongside the existing exit triggers; it is not a replacement
for the +6% percentage take-profit, the ATR stop-loss, or signal-based
exits.

## Non-goals

- No change to the BUY side. Entry logic untouched.
- No change to MomentumAgent, CNNAgent, or ScalpAgent.
- No DB schema change. Per-position state lives inside the existing
  `agent_state.positions_json` blob.
- No frontend change.

---

## Rule semantics

For each open TECH position, every WS price tick:

1. Compute `current_pnl_usd = (price − avg_price) × size`.
2. Update `pos["peak_pnl_usd"] = max(pos["peak_pnl_usd"], current_pnl_usd)`.
3. If `pos["peak_pnl_usd"] ≥ _TRAIL_ARM_USD` AND
   `pos["peak_pnl_usd"] − current_pnl_usd ≥ _TRAIL_GIVEBACK_USD` →
   fire SELL with trigger `TICK_TRAIL`.

**Constants:**
- `_TRAIL_ARM_USD = 1.00`
- `_TRAIL_GIVEBACK_USD = 0.25`

Once armed (peak ≥ $1), the position remains armed for its lifetime.
Peak is monotonic — falling PnL never lowers it. On any sell (trail,
ATR, signal, +6% take-profit), `_Book.sell` pops the pid; peak state
is gone, and re-entry starts fresh.

The rule is strictly per-position. Total realized PnL is not part of
the trigger.

## Exit-chain priority

Inside `on_price_tick` → "has position" branch
(`tech_agent_cb.py` lines 288–338), the new order:

1. `TICK_SIGNAL` — cached SELL score ≥ `_SELL_THRESHOLD` (unchanged)
2. `TICK_STOP` — ATR stop-loss (unchanged; only fires on loss, no
   conflict with the trailing rule)
3. `TICK_TRAIL` — **new trailing $ take-profit**
4. `TICK_PROFIT` — +6% take-profit (unchanged; remains a backstop and
   will rarely fire because the trail typically triggers first)

`peak_pnl_usd` is updated unconditionally before the exit checks run,
so peak-tracking happens on every tick regardless of which (if any)
exit fires.

## State storage

Add one field to each entry in `_Book.positions[pid]`:

```
{
  "size": float,
  "avg_price": float,
  "atr_stop": float,         # existing
  "peak_pnl_usd": float,     # NEW — initialized to 0.0 on buy
}
```

- `_Book.buy` initializes `peak_pnl_usd = 0.0` only on a fresh entry.
  When averaging up an existing position, `peak_pnl_usd` is **not
  reset** — keep tracking the run-up.
- `_Book._save()` already serializes the full positions dict to
  `agent_state.positions_json`; the new key flows automatically.
- `_Book.load()` deserializes it back. For positions saved before
  this change, the key will be absent — call sites use
  `pos.get("peak_pnl_usd", 0.0)` so legacy positions pick up tracking
  from their next tick without crashing.
- `_Book.sell` already pops the pid → state is naturally cleared on
  exit. No special cleanup needed.

No DB schema migration. No JSON-shape version bump.

## Tests (TDD, RED-first)

New class `TestTrailingDollarExit` in
`backend/tests/test_tech_agent_cb.py`. All async, all using
`AsyncMock` for `database.*` per repo conventions. No live API calls,
no real file I/O.

| # | Test | Asserts |
|---|------|---------|
| 1 | `test_peak_pnl_updates_when_price_rises` | tick at higher price → `peak_pnl_usd` increases |
| 2 | `test_peak_pnl_held_when_price_falls` | tick at lower price → `peak_pnl_usd` unchanged |
| 3 | `test_no_trail_sell_when_peak_below_arm_threshold` | peak = $0.80, full giveback → no sell |
| 4 | `test_no_trail_sell_when_giveback_below_threshold` | peak = $1.50, current = $1.30 (giveback $0.20) → no sell |
| 5 | `test_trail_sell_fires_when_armed_and_giveback_reached` | peak = $1.50, current = $1.25 (giveback $0.25) → SELL with trigger `TICK_TRAIL`, `signals_sell += 1`, `agent_decisions` row written |
| 6 | `test_trail_fires_before_6pct_take_profit` | scenario where both conditions are true → trigger = `TICK_TRAIL`, not `TICK_PROFIT` |
| 7 | `test_atr_stop_still_wins_over_trail` | position down (ATR stop active), trail never reached |
| 8 | `test_position_without_peak_key_loads_safely` | legacy saved position (no `peak_pnl_usd` key) → tick does not crash |
| 9 | `test_reentry_after_trail_sell_starts_with_fresh_peak` | trail fires → buy again same pid → `peak_pnl_usd == 0.0` |

## Files touched

| File | Change |
|---|---|
| `backend/agents/tech_agent_cb.py` | 2 new constants; init `peak_pnl_usd: 0.0` in `_Book.buy` (new entries only); ~10 lines in `on_price_tick` to update peak and check trail |
| `backend/tests/test_tech_agent_cb.py` | 9 new tests in `TestTrailingDollarExit` |
| `CHANGELOG.md` | new Session entry |
| `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` | TechAgent row + exit-trigger list |
| `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\trading_app_thresholds.md` | record `_TRAIL_ARM_USD`, `_TRAIL_GIVEBACK_USD` |

`CLAUDE.md` is not touched — no new architectural invariant is added.

## Commit / branch

Single commit on the existing `master` branch, including both
implementation and tests, per repo convention. Memory files updated
in the same response.

## Risks / edge cases considered

- **Averaging up.** If TechAgent buys more of a pid it already holds,
  `_Book.buy` recomputes `avg_price`. Resetting `peak_pnl_usd` would
  lose the run-up; not resetting may leave a stale peak that the new
  combined position can't reach. **Decision:** keep the existing peak
  unchanged. The combined position has a lower avg_price, so any
  subsequent tick will recompute current_pnl_usd against the new
  avg_price and update peak naturally. The previous peak remains a
  valid floor for the trail.

- **Initial peak value at buy.** Set to `0.0`, not to the current PnL
  (which is also 0 the moment of buy). If the user adds slippage
  modeling later, the buy-side PnL might be slightly negative; the
  `max(...)` update keeps peak at the highest observed value.

- **Legacy saved positions.** Before this change ships, the DB has
  positions without the new key. The `pos.get("peak_pnl_usd", 0.0)`
  fallback ensures they pick up tracking from their next tick — no
  retroactive peak; the rule simply waits until live PnL crosses $1
  again.

- **Tick rate.** WS ticks can fire many times per second. The peak
  update is O(1) and the trail check is two comparisons; cost is
  negligible.

- **Race with the per-product tick lock.** `on_price_tick` already
  short-circuits when the tick lock is held (entry path). Exit checks
  in the existing code path do not acquire that lock; the new trail
  check follows the same pattern as `TICK_STOP` / `TICK_PROFIT`.

## Done criteria

- All 9 new tests pass; existing `test_tech_agent_cb.py` tests still
  pass.
- Full backend suite passes.
- CHANGELOG entry written.
- Memory files updated in the same session.
- Single commit on master.
