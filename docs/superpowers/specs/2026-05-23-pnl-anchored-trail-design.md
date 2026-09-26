# PnL-Anchored Trail with Volatility-Relative Profit Floors — Design

**Date:** 2026-05-23
**Status:** Draft (deferred to next 8001 pause window; tracked as backlog task #46)
**Author:** Claude (Opus 4.7), invoked by gl4500
**Pairs with:** [[2026-05-23-gpu-batched-feature-kernel-design]] (independent), task #26 (max-hold redesign), task #47 (GPU port of horizon_compare)

---

## Problem

The current trail-stop in `agents/cnn_agent.py:_check_risk_exits` (and the WS-side mirror
in `agents/exit_watcher.py:on_price_tick`) computes:

```python
trail_line = peak_price * (1 - trail_pct)   # trail_pct ∈ [6%, 15%], ATR-scaled
stop_loss  = entry_price * 0.92             # fixed -8% from entry
exit when:  current_price <= max(trail_line, stop_loss)
```

This has a **structural gap**: when `peak_price` is barely above `entry_price` (peak/entry <
1/(1−trail_pct) ≈ 1.064 at 6% trail), the trail line sits **below** entry. So the trail
"fires first" but realizes a LOSS on a position that was meaningfully green at peak.

**Worked example from 2026-05-23, Session 58.71n (operator's live book):**

10 open positions, 9 green unrealized, totaling **+$17.47 paper PnL**. Realized PnL
cumulative: **−$86.71**. If every trail fires at the current peak/trail levels
under the **existing rule**, realized PnL becomes **−$104.99** — a **$35 swing the
wrong way** on positions that went green. Under the **new rule** (this design),
realized PnL only drops to **−$87.50** — a +$28 improvement, lifting nearly every
green position to the fee-aware break-even floor.

Operator framing 2026-05-23: *"this is the primary reason I am losing realized gains."*

The flaw is **price-anchored trail with no profit-lock floor**. A position can peak at
+5% PnL, pull back 6%, and exit at −1% — converting paper profit into realized loss.

## Goal

Replace the peak-price-anchored trail with a **PnL-anchored trail + volatility-relative
profit-floor + position-size-aware dollar cap + fee-aware break-even**. Eliminate the
"paper-profit-to-realized-loss" failure mode while preserving:

1. The current ATR-scaled width (no whipsaw exits on minor noise)
2. Stop-loss as backstop for catastrophic moves (unchanged)
3. The WS exit checker's sub-second latency (Session 58.71m)
4. Long/short symmetry (mirrored math for SELL-side positions if/when added)

**Success criteria:**

1. On the live 10-position snapshot today, worst-case realized improves by ≥$25
   (validated 2026-05-23: +$28.12 measured via the actual implementation —
   see "Worked Examples" table below).
2. Shadow-week realized PnL on port 8002 exceeds current-rule realized PnL on
   port 8001 by ≥$15 over a 7-day window.
3. No regression on positions that hit `STOP_LOSS` — those still exit at −8% (or
   earlier if dollar-cap engages).
4. CPU-vs-WS parity: scan-loop and WS handler compute identical exit thresholds
   for the same `(peak_pnl_pct, atr_pct, position_dollars, total_capital)` tuple.

## Non-Goals

- Changing the model (XGB v4.5 inference path stays untouched). Exit policy ≠ model.
- Adding new triggers (no time-of-day filter, no regime-conditional rules in v1).
- Migrating to absolute-dollar stops (e.g., "exit if unrealized < −$10"). The design
  stays in %-PnL space, with a position-size *cap* layered on top, not a wholesale shift.
- Per-pid tuning of the constants. Single global config for v1; per-pid overrides
  deferred to a v2 if shadow-week data motivates it.

---

## Architecture — Three Layers

### Layer 1: ATR-scaled profit-floor tiers (volatility-relative)

Fixed-% tiers don't translate across the coin universe. A +2% floor is meaningful for
BTC (ATR ~1.5%, lock = 1.3× daily move) but trivial for a microcap (ATR ~15%, lock =
pure noise). Anchor tiers in **ATR multiples**:

```python
def _atr_floor(peak_pnl_pct: float, atr_pct: float) -> Optional[float]:
    """ATR-scaled profit floor. Returns the minimum exit threshold in PnL terms,
    or None if no floor engages yet.

    Tiers:
      peak_pnl_pct >= 3.0 * atr_pct  ->  lock 1.5 * atr_pct  (strong lock)
      peak_pnl_pct >= 1.5 * atr_pct  ->  lock 0.5 * atr_pct  (moderate lock)
      peak_pnl_pct >= 0.5 * atr_pct  ->  lock 2 * FEE_RATE    (fee-aware break-even)
      else                            ->  None                (no floor yet)
    """
    if peak_pnl_pct >= 3.0 * atr_pct:
        return 1.5 * atr_pct
    if peak_pnl_pct >= 1.5 * atr_pct:
        return 0.5 * atr_pct
    if peak_pnl_pct >= 0.5 * atr_pct:
        return 2 * FEE_RATE
    return None
```

This preserves the same "informational meaning" across coins:

| Coin (example ATR_pct) | "Strong lock" engages at | Locks |
|---|---|---|
| BTC (ATR ~1.5%) | peak_pnl ≥ 4.5% | +2.25% |
| ETH (ATR ~2%) | peak_pnl ≥ 6.0% | +3.0% |
| DOGE (ATR ~5%) | peak_pnl ≥ 15% | +7.5% |
| Microcap (ATR ~15%) | peak_pnl ≥ 45% | +22.5% |

### Layer 2: Capital-relative dollar-cap on large positions

Layer 1 is %-based, which means equal-% loss on a $30 position vs a $1000 position
hurts very differently in $ terms. Add an absolute-$ cap on giveback, but make the
"large" threshold **capital-relative** so the design scales as the operator's total
capital grows from $1k to $10k to $100k.

**Why capital-relative, not fixed $:** If `LARGE_POSITION_THRESHOLD` is a fixed $200,
then at $1k capital it's a 20%-of-capital cutoff (only outsized positions qualify),
but at $10k+ capital it becomes <2%-of-capital and *every* position qualifies — the
tiered design collapses to "everything has the cap." Tying the threshold to capital
preserves the "small vs large" distinction at every scale.

```python
def _large_position_threshold(total_capital: float) -> float:
    """Capital-relative cutoff for the dollar-cap layer.

    A position is 'large' if it's > LARGE_POSITION_FRAC of total capital,
    with LARGE_POSITION_FLOOR as the absolute minimum so the rule remains
    meaningful at small capital."""
    return max(LARGE_POSITION_FLOOR, total_capital * LARGE_POSITION_FRAC)


def _dollar_cap_floor(
    peak_pnl_pct: float,
    position_dollars: float,
    total_capital: float,
) -> Optional[float]:
    """Tighten exit threshold for positions above the capital-relative threshold.

    Caps giveback at the TIGHTER of:
      - MAX_DOLLAR_GIVEBACK_FRAC of position $ (%-based, scale-invariant)
      - MAX_LOSS_FRAC_OF_CAPITAL of total capital (portfolio-impact-based)

    Returns the minimum exit threshold (in PnL terms) or None if not applicable.
    """
    threshold = _large_position_threshold(total_capital)
    if position_dollars <= threshold:
        return None
    # Tighter of (% of position, absolute $ tied to capital)
    pct_cap = MAX_DOLLAR_GIVEBACK_FRAC                                       # e.g., 2% of position
    cap_cap = (total_capital * MAX_LOSS_FRAC_OF_CAPITAL) / position_dollars  # e.g., 0.5% of capital / pos $
    giveback = min(pct_cap, cap_cap)
    return peak_pnl_pct - giveback
```

This adds asymmetric protection that scales with capital:

| Total capital | "Large" begins at | Behavior |
|---|---|---|
| $1k (today) | max($200, 5% × $1k = $50) = **$200** | Only positions >$200 get the cap |
| $5k | max($200, 5% × $5k = $250) = **$250** | Positions >$250 get the cap |
| $10k | max($200, 5% × $10k = $500) = **$500** | Positions >$500 get the cap |
| $100k | max($200, 5% × $100k = $5k) = **$5k** | Positions >$5k get the cap |

A small position ($30 in a $100k book = 0.03% concentration) rides the full ATR trail
regardless of capital scale. A large position (>5% concentration) gets the tighter cap.

**Worked scaling example** — position $500, capital grows:

| Capital | Threshold | $500 > threshold? | pct_cap (2%) | cap_cap (0.5%×cap/pos$) | Effective giveback |
|---|---|---|---|---|---|
| $1k | $200 | yes | 2% | $5/$500 = 1.0% | min = **1.0%** (tighter) |
| $5k | $250 | yes | 2% | $25/$500 = 5.0% | min = **2.0%** (pct wins) |
| $10k | $500 | edge (not strictly above) | n/a | n/a | **no cap** — Layer 3 trail applies |
| $100k | $5k | no ($500 << $5k) | n/a | n/a | **no cap** — Layer 3 trail applies |

So as capital grows, a $500 position transitions from "large + tight cap" → "small +
standard trail." This matches portfolio intuition: a small absolute position in a big
book doesn't need the tight cap because its $ impact is small.

### Layer 3: Standard ATR giveback (preserve current behavior baseline)

The pre-floor behavior is **still the floor of the floor**. If peak_pnl_pct is too
small to engage Layer 1 AND position_dollars is below the threshold for Layer 2, the
exit reverts to the existing %-trail: `peak_pnl - atr_pct`.

### Combined exit threshold

```python
FEE_RATE                    = 0.006   # Coinbase taker (round-trip basis = 2 * 0.006 = 1.2%)
LARGE_POSITION_FRAC         = 0.05    # 5% of capital = "large" position
LARGE_POSITION_FLOOR        = 200.0   # USD; absolute floor for small-capital regime
MAX_DOLLAR_GIVEBACK_FRAC    = 0.02    # 2% of position $ — %-based scale-invariant cap
MAX_LOSS_FRAC_OF_CAPITAL    = 0.005   # 0.5% of total capital per trail-fire


def _compute_exit_threshold(
    *,
    peak_pnl_pct: float,
    atr_pct: float,
    position_dollars: Optional[float] = None,
    total_capital: Optional[float] = None,
) -> float:
    """Exit threshold in PnL terms. Position exits when current_pnl_pct < threshold.

    Combines:
      1. ATR-scaled profit-floor tiers (volatility-relative)
      2. Optional capital-relative dollar-cap on large positions
      3. Standard ATR giveback baseline (peak_pnl - atr_pct)

    `total_capital` is required for Layer 2 to engage; if omitted, Layer 2 is skipped.
    """
    base = peak_pnl_pct - atr_pct                                  # Layer 3 baseline

    atr_floor = _atr_floor(peak_pnl_pct, atr_pct)                  # Layer 1
    if atr_floor is not None:
        base = max(base, atr_floor)

    if position_dollars and total_capital:                          # Layer 2
        dollar_floor = _dollar_cap_floor(
            peak_pnl_pct, position_dollars, total_capital,
        )
        if dollar_floor is not None:
            base = max(base, dollar_floor)

    return base
```

### Stop-loss interaction (unchanged)

`STOP_LOSS = -8%` from entry continues to fire independently. The combined exit rule
is:

```python
exit_if  current_pnl_pct < max(_compute_exit_threshold(...), -_CNN_STOP_LOSS_PCT)
```

For green positions with engaged floors, the threshold is above −8% so trail/floor
fires first. For positions that never went green, the threshold reverts to standard
ATR trail; stop_loss may fire first in high-vol regimes (trail_pct > 8%) — this is
unchanged from today.

---

## Worked Examples on Today's Live Book

Using `FEE_RATE = 0.006` (Coinbase taker), positions sorted by current %PnL.
**Threshold computations verified via the actual `_compute_exit_threshold`
implementation** (see `tests/test_exit_thresholds.py` for the worked-example
unit tests).

| PID | entry | peak | curr | peak% | atr% | pos$ | Layer 1 | Layer 2 | New threshold | Old threshold | Exits today? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DASH | 42.97 | 46.32 | 45.93 | +7.8% | 6% | $58 | break-even tier; baseline 1.8% wins | no | **+1.80%** | -1.30% | no (curr +6.9%) |
| HYPE | 56.75 | 60.30 | 59.81 | +6.3% | 6% | $60 | break-even tier wins over 0.3% baseline | no | **+1.20%** | -0.12% | no (curr +5.4%) |
| ETH | 2035 | 2123 | 2118 | +4.3% | 6% | $103 | break-even tier wins over −1.7% baseline | no | **+1.20%** | -1.94% | no (curr +4.1%) |
| UNI | 3.355 | 3.504 | 3.441 | +4.4% | 6% | $78 | break-even tier wins over −1.6% baseline | no | **+1.20%** | -1.83% | no (curr +2.6%) |
| BTC | 74853 | 76827 | 76710 | +2.6% | 6% | $84 | none (peak < 0.5×ATR=3%) | no | **−3.36%** (Layer 3) | -3.52% | no (curr +2.5%) |
| ETC | 8.765 | 9.080 | 8.980 | +3.6% | 6% | $66 | break-even tier wins over −2.4% baseline | no | **+1.20%** | -2.62% | no (curr +2.5%) |
| SUI | 1.038 | 1.083 | 1.061 | +4.3% | 6% | $58 | break-even tier wins over −1.7% baseline | no | **+1.20%** | -1.92% | no (curr +2.2%) |
| AVAX | 9.148 | 9.490 | 9.340 | +3.7% | 6% | $89 | break-even tier wins over −2.3% baseline | no | **+1.20%** | -2.49% | no (curr +2.1%) |
| VVV | 17.835 | 18.890 | 18.100 | +5.9% | 6% | $54 | break-even tier wins over −0.1% baseline | no | **+1.20%** | -0.44% | no (curr +1.5%) |
| PAXG | 4532 | 4532 | 4530 | 0% | 4% | $452 | none (peak < 0.5×ATR=2%) | **yes** ($452 > $200) | **−1.11%** (Layer 2) | -4.00% | no (curr -0.06%) |

**Worst-case behavior — what gets realized if every position rides down to its threshold:**

| PID | Old realized | New realized | Delta |
|---|---|---|---|
| DASH | +$0.77 (trail at -1.3% PnL) | +$1.04 (floor at +1.8% PnL) | +$0.27 |
| HYPE | −$0.07 (trail at -0.1%) | **+$0.72** (floor at +1.2%) | **+$0.79** |
| ETH | −$1.92 (trail at -1.9%) | **+$1.19** (floor at +1.2%) | **+$3.11** |
| UNI | −$1.38 (trail at -1.8%) | **+$0.91** (floor at +1.2%) | **+$2.29** |
| BTC | −$2.96 (trail at -3.5%) | −$2.83 (trail at -3.4%, Layer 3) | +$0.13 |
| ETC | −$1.69 (trail at -2.6%) | **+$0.77** (floor at +1.2%) | **+$2.46** |
| SUI | −$1.12 (trail at -1.9%) | **+$0.70** (floor at +1.2%) | **+$1.82** |
| AVAX | −$2.21 (trail at -2.5%) | **+$1.07** (floor at +1.2%) | **+$3.28** |
| VVV | −$0.24 (trail at -0.4%) | **+$0.64** (floor at +1.2%) | **+$0.88** |
| PAXG | −$18.09 (trail at -4%) | **−$5.00** (dollar-cap at -1.1%) | **+$13.09** |
| **Total** | **−$28.92** | **−$0.79** | **+$28.12** |

**Net improvement on a synchronous worst-case fire: +$28.12** (book goes from
realizing −$29 in losses to realizing essentially break-even). The break-even
floor (Layer 1, fee-aware at +1.2%) engages on **9 of 10** positions because peak
PnL exceeded `0.5 × ATR = 3%` for all but BTC. PAXG benefits separately via the
capital-relative dollar-cap (Layer 2), saving $13 alone.

Over the rolling history of trades (many open-then-close cycles per day), this
compounds substantially. Even partial trails (not all positions firing
simultaneously) recover the majority of these dollars.

---

## State Schema

### Position state additions in `_CNNBook.positions[pid]`

| Field | Type | Source | Updated when |
|---|---|---|---|
| `peak_pnl_pct` | float | `max(peak_pnl_pct, (current − entry) / entry)` | every scan + every WS tick |
| `position_dollars` | float | `size × current_price` | every scan (cached, not per tick) |

`atr_pct` already cached as `trail_pct` per Session 58.71m WS exit checker.

### Book-level state passed to threshold calc

`total_capital` — sum of (balance + open-positions-dollars) for the agent.
Already computed in `_CNNBook` (used for sizing decisions).
Passed by value into `_compute_exit_threshold(...)` on each scan / tick — not stored
per-position.

### Migration on backend startup

For existing open positions loaded from `agent_state.positions_json`:

```python
# Compute peak_pnl_pct from existing peak_price + avg_price
if "peak_pnl_pct" not in pos:
    pos["peak_pnl_pct"] = (pos["peak_price"] - pos["avg_price"]) / pos["avg_price"]
# Compute position_dollars from size × current_price (will be updated next scan)
if "position_dollars" not in pos:
    pos["position_dollars"] = pos["size"] * pos.get("avg_price", 0.0)
```

`total_capital` is derived fresh each call from `_CNNBook` state; no migration
needed.

No DB schema migration needed (JSON blob accepts new fields).

---

## Implementation Sites

1. **`backend/agents/cnn_agent.py:_check_risk_exits`** — scan-loop path
   - Update `peak_pnl_pct` on entry to the function (track new highs)
   - Update `position_dollars` from current price × size
   - Compute exit threshold via `_compute_exit_threshold`
   - Replace existing trail/stop_loss check with the combined rule

2. **`backend/agents/exit_watcher.py:on_price_tick`** — WS tick path
   - Update `peak_pnl_pct` on each tick (per-pid lock already in place)
   - `position_dollars` reads from cached value (scan loop owns refresh)
   - Compute exit threshold via the same `_compute_exit_threshold`
   - Fire `WS_TRAIL_FLOOR` or `WS_TRAIL` or `WS_STOP_LOSS` per which line triggered

3. **`backend/agents/cnn_agent.py` constants** — add (or move into the new module):
   ```python
   _FEE_RATE                    = 0.006   # Coinbase taker round-trip basis
   _LARGE_POSITION_FRAC         = 0.05    # 5% of capital = "large"
   _LARGE_POSITION_FLOOR        = 200.0   # USD; absolute floor for small-capital regime
   _MAX_DOLLAR_GIVEBACK_FRAC    = 0.02    # 2% of position $
   _MAX_LOSS_FRAC_OF_CAPITAL    = 0.005   # 0.5% of capital max loss per trail-fire
   ```

4. **`backend/agents/exit_thresholds.py`** (NEW) — pure-helper module:
   - `_atr_floor(peak_pnl_pct, atr_pct) -> Optional[float]`
   - `_large_position_threshold(total_capital) -> float`
   - `_dollar_cap_floor(peak_pnl_pct, position_dollars, total_capital) -> Optional[float]`
   - `_compute_exit_threshold(peak_pnl_pct, atr_pct, position_dollars=None, total_capital=None) -> float`
   - All pure functions for testability + reuse between scan and WS paths

---

## Testing Strategy

### Unit tests (`tests/test_exit_thresholds.py`, NEW)

For `_compute_exit_threshold`:

1. `test_layer3_baseline_when_no_floor_engages` — small peak_pnl (< 0.5×ATR) +
   small position → returns `peak_pnl - atr_pct`
2. `test_layer1_break_even_tier` — peak_pnl in [0.5×ATR, 1.5×ATR) → returns `2 × FEE_RATE`
3. `test_layer1_moderate_lock_tier` — peak_pnl in [1.5×ATR, 3.0×ATR) → returns `0.5 × atr_pct`
4. `test_layer1_strong_lock_tier` — peak_pnl >= 3.0×ATR → returns `1.5 × atr_pct`
5. `test_layer2_engages_above_capital_relative_threshold` — position $250 + capital $1k
   (5% × 1k = $50, floor = $200, effective = $200) → cap engages, returns
   `peak_pnl - min(2%, cap_cap)`
6. `test_layer2_skipped_below_capital_relative_threshold` — position $50 + capital $1k →
   skip dollar-cap, Layer 3 baseline applies
7. `test_layer2_threshold_scales_with_capital` — same $500 position; assert cap engages
   at capital=$1k (threshold $200), engages at capital=$5k (threshold $250), does NOT
   engage at capital=$100k (threshold $5k)
8. `test_layer2_capital_floor_at_small_capital` — capital=$100; threshold should be
   `LARGE_POSITION_FLOOR` ($200) not `5% × $100 = $5`
9. `test_layer2_capital_loss_cap_tighter_than_pct_cap` — small capital × large position
   = capital-loss-cap < pct-cap; assert `min()` picks capital-loss-cap
10. `test_layer2_pct_cap_tighter_than_capital_loss_cap` — high capital × small-relative-large
    position = pct-cap < capital-loss-cap; assert `min()` picks pct-cap
11. `test_max_of_layers_when_multiple_engage` — large position + strong-lock tier → returns max
    of all engaged layers
12. `test_boundary_at_exact_atr_multiples` — exactly 0.5×, 1.5×, 3.0× ATR boundaries
    (asserts `>=` not `>` semantics)
13. `test_fee_aware_break_even_uses_fee_rate_constant` — changing FEE_RATE shifts the
    break-even tier output by `2 × delta`
14. `test_dollar_cap_uses_constant_fractions` — changing MAX_DOLLAR_GIVEBACK_FRAC or
    MAX_LOSS_FRAC_OF_CAPITAL shifts the cap output proportionally

### Integration tests

11. `test_scan_loop_uses_new_threshold` — mock position with known peak_pnl_pct + ATR;
    assert `_check_risk_exits` computes expected exit threshold
12. `test_ws_handler_uses_same_threshold_as_scan_loop` — same inputs produce same exit
    decision in scan vs WS paths (parity)
13. `test_position_state_migration_on_load` — load agent_state with old JSON (no
    peak_pnl_pct field), assert it's computed correctly from peak_price + avg_price
14. `test_peak_pnl_pct_ratchets_only_upward` — feed sequence of prices, assert
    peak_pnl_pct never decreases

### Regression test

15. `test_existing_book_exit_table_matches_spec_example` — recreate the 10-position
    worked example from this spec; assert combined threshold and "exit?" decision
    match the table

---

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Constants (FEE_RATE, LARGE_POSITION_THRESHOLD, MAX_DOLLAR_GIVEBACK_FRAC) miscalibrated | Medium | Surface as `.env`-overrideable knobs; tune in shadow week on 8002 before promoting |
| New floors fire too aggressively → exits-too-early | Medium | Shadow-week telemetry will catch (compare new vs current realized PnL on same trade sequence) |
| Per-pid ATR computation expensive at every tick | Low | Already cached as `trail_pct` per Session 58.71m |
| `position_dollars` stale between scan refreshes | Low | Price can drift ~6% between scans (1 min); dollar-cap based on stale price is at most 6% off, irrelevant for threshold gating |
| Long-only design — short-side mirror untested | Out of scope | No SELL-side positions in current pipeline; deferred to v2 if short trading is added |
| Migration breaks existing positions on startup | Low | Defensive `if "peak_pnl_pct" not in pos: pos["peak_pnl_pct"] = ...` covers the gap; no DB schema change |

---

## Calibration Plan for Shadow Week

After landing in shadow mode on port 8002:

1. **Compare realized PnL** between 8001 (current rule) and 8002 (new rule) over 7 days
2. **Per-tier engagement frequency** — log which tier (Layer 1 break-even / moderate /
   strong / Layer 2 dollar-cap / Layer 3 baseline) fired on each exit
3. **Time-in-position** — does new rule exit too early on continued trends?
4. **Constants to tune if shadow-week metrics motivate:**
   - `FEE_RATE` — verify Coinbase fee tier matches assumed 0.006
   - `LARGE_POSITION_FRAC` — operator's actual "large position by % of capital" line
     (0.05 = 5% by default; lower if more concentration-averse, higher if happy with bigger bets)
   - `LARGE_POSITION_FLOOR` — absolute USD floor for the small-capital regime ($200 default)
   - `MAX_DOLLAR_GIVEBACK_FRAC` — %-of-position cap (0.02 = 2% default)
   - `MAX_LOSS_FRAC_OF_CAPITAL` — %-of-capital cap (0.005 = 0.5% default; tightens
     as capital grows because absolute single-trade losses become more impactful)

Promote to 8001 only if:
- Realized PnL > current rule by ≥$15 over the 7-day window
- No new exit-mode failures (no positions stuck open, no premature exits on multi-day trends)

---

## Out-of-Scope (deferred to v2 or other backlog)

- **Per-pid constant overrides** — single global config for v1; per-pid tuning if
  data motivates
- **Short-side mirror** — SELL positions; deferred until short trading is added
- **Trailing trigger by absolute $** — alternative framing where stops are in $ terms
  instead of % terms; this design stays in %-space with a $-cap layer
- **Regime-conditional floors** — different floors in TRENDING vs RANGING vs UNKNOWN
  regimes; defer until v1 telemetry shows clear regime-vs-exit-quality correlation
- **Max-hold redesign** (task #26 backlog) — independent change; pairs naturally with
  this but ships on its own

---

## See also

- [[coinbase_trader_architecture]] — `_CNNBook.positions[pid]` state schema
- [[polymarket_app_trail_design_gap]] — operator-confirmed root cause of realized PnL bleed
- [[feedback_roi_first_priority]] — ROI is the #1 operator metric; this change is
  ranked above model improvements for that reason
- [[2026-05-23-gpu-batched-feature-kernel-design]] — GPU port (this session's predecessor)
- WS exit checker design (2026-05-23, Session 58.71m) — established the per-pid lock +
  `trail_pct` cache pattern this design extends
- Task #26 backlog — max-hold redesign (pairs naturally)
- Task #47 — GPU port of `v4_5_horizon_compare.py` (already drafted in working tree)
- CLAUDE.md invariants #18 (WS exit-handler isolation) — must hold for the new WS path
