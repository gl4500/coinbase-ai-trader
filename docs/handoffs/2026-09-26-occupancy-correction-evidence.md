# Occupancy correction: what changed, what it was measured on, and what is still unknown

**Date:** 2026-09-26 / 2026-09-27
**Branches:** `feat/label-endpoint-publication` (#76) → `feat/endpoint-consumer-integration` (#77)
→ `feat/endpoint-consumer-wiring` (#78)
**Spec:** `docs/specs/2026-09-26-label-endpoint-contract.md`, §9
**Status:** method corrected; **archived results not re-measured**

---

## 1. Read this section before quoting any number below

Every figure in this note comes from a **synthetic fixture**, not from the real universe. No
Phase 2 / Phase 4 run was regenerated, and no archived artifact was recomputed. That was a
deliberate boundary, not an omission: the ATR blocker is unresolved, so a re-run would produce
new numbers resting on the same unresolved input and would invite exactly the reading this note
exists to prevent.

So this note records **that a measurement method was defective and has been repaired**. It does
**not** establish what the corrected method would say about any past verdict. Those are different
claims, and conflating them is how a correction turns into an unearned improvement story.

**Nothing here is evidence of profitability.** Phase 4 still emits `deployment_eligible=false`
with explicit blockers (invariant 24), and that is unchanged by this work.

---

## 2. The defect

Two components answered the same question differently, and nothing compared them:

| component | how it decided when a position ends |
|---|---|
| the label producer | the exit rule that actually fired — stop, trail, cap, or horizon |
| the portfolio replay | `entry_ts + horizon × 3_600_000`, a wall clock |
| the miner | `build_next_eligible(ts_ms, horizon_bars=horizon)`, row arithmetic |

A single wall-clock `exit_ts` was serving two distinct roles at once:

- an **eligibility boundary**, which is a **POSITION** in the working frame — the row from which
  the capital is free again;
- an **accounting time**, which is an **INSTANT** — when the outcome becomes observable.

That conflation is why the error was hard to see: it points in **opposite directions** depending
on the case, so it never presented as a consistent bias either way.

| case | wall clock says | truth | direction of error |
|---|---|---|---|
| gap in the source bars | exit lands on a bar that does not exist | exit at the last real bar | slot released **early** and resold — occupancy understated, trade count overstated |
| stop or trail fires early | holds the slot to the full horizon | exit when the rule fired | a real later entry **suppressed** — occupancy overstated, trade count understated |

They are now separate functions: `eligibility_boundaries()` (positions) and `accounting_times()`
(instants), in `replay_timeline.py`, which contains no decision logic at all.

---

## 3. What was actually measured — toy fixtures, exact values

### 3.1 The double count, demonstrated end to end

Gap fixture, one product, six candidate entries:

```
clock-derived next-eligible:     [1, 2, 3, 4, 5, 6]
endpoint-derived next-eligible:  [2, 3, 4, 5, 6, 6]
```

Run through the unchanged `walk_and_sum` on the same candidate set:

```
on the clock:      0.6
on the endpoints:  0.3
```

The clock counted one entry **twice**. This is the clearest single artifact in the effort, because
the summing function is identical in both runs — only the eligibility vector differs.

### 3.2 The occupancy denominator, established by falsification rather than by passing

A checkpoint-only instant exists so a position that is due can be examined. It must never become
a sampling point. The regression injects 25 instants that belong to no position, leaving the
candidates, labels and decision instants untouched (via `patch.object` on the checkpoint
provider). With the guard in place nothing moves. With the guard deliberately removed:

```
pct_slots_full        0.75 -> 0.18181818181818182
mean_concurrent       0.75 -> 0.18181818181818182
trade_count           3    -> 3                     (unchanged)
cumulative_profit_raw 0.30000000000000004 -> 0.30000000000000004   (unchanged)
```

Only the denominator moved, under an input that changed nothing about the trading — which is
precisely the claim. A test that merely passes proves nothing; this one was watched to fail.

### 3.3 The equivalence gate is non-vacuous

The endpoint-derived eligibility builder is checked against the original `build_next_eligible` on
a horizon-only fixture where the two must agree: 9 rows, **7 distinct values**, vector
`[3,4,5,6,7,8,9,9,9]`. `build_next_eligible` is deliberately **kept** rather than deleted, as the
baseline that keeps this comparison possible.

### 3.4 A bug the defaults almost hid

The default exit cap is **168 bars**, which **equals the longest default horizon**. Reconstructing
a record's expected cap from its own horizon therefore still validates horizon-168 records, and
only a short horizon exposes the error. Against the real producer:

```
horizon=1   max_hold_bars=168   bars_held=1
```

The cap comes from the declared `exit_config`, never from the record under test. The regression
uses horizon 1 for exactly this reason.

---

## 4. Scope — what is wired and what is not

**Wired:** `simulate_portfolio`, `knapsack_search`, `build_phase4`.

**NOT wired: the miner.** `mine_profiles.py:317` still calls the wall-clock
`build_next_eligible`. `build_next_eligible_from_endpoints` is built and tested but has **no
caller**. Mined eligibility therefore still carries the §2 error, and endpoint-driven mining is
required and pending.

This is recorded plainly because an earlier draft of the CHANGELOG and of CLAUDE.md invariant 26
claimed that "mining and portfolio replay" both read endpoints. They did not. Codex review
`17814674` caught it and commit `27fd5ae` corrected both documents. That was the **fifth** time in
this effort that prose claimed more than the code delivered — the same defect class the endpoint
contract exists to remove: one fact living in two places with only one of them checked.

### 4.1 One direction is contained rather than exercised

PR #72's contiguity guard means the replay refuses non-contiguous hourly input, and the miner
raises on it too (`timestamps must be unique and contiguous hourly`). So the **gap** direction
cannot arise in either path today. It is contained, not fixed-and-proven, and the eligibility-side
proof in §3.1 stands in for it. The **early-exit** direction is live in both paths and is what the
wiring actually repairs. Recorded in spec §9 rather than left to be rediscovered.

---

## 5. The archived Phase 4 verdict — what this does and does not say about it

The archived strategy-discovery run (2026-05-28, 25 profiles) returned **ABORT**: raw profit
positive, deflation flipping it negative.

Two statements must stay separate, and spec §9.6 keeps them separate:

1. **The method that produced that verdict was defective.** Occupancy and eligibility were
   measured on a clock that disagreed with the labels, and — independently — the `purged_wf`
   leakage finding put post-test rows into TRAIN. Both are established.
2. **"This archived verdict is wrong"** — *not established, and not claimed here.*

A defective method makes its output **uninformative**, not inverted. An ABORT produced by a broken
measurement is not thereby a GO; it is a verdict carrying no evidentiary weight in either
direction. Saying anything further would require a re-run, and that has deliberately not been done
(§1).

---

## 6. Verification performed

| gate | result |
|---|---|
| full pre-commit hook on `b8b3694` | 2002 passed, 65 skipped, 1 deselected, 1 xfailed, 2 xpassed (410.18s) |
| `tests/tools/strategy_discovery/` | 618 passed (75.05s) |
| pinned `ruff==0.9.0` `check backend/` | clean |
| pinned `ruff==0.9.0` `format --check backend/` | clean, 267 files |
| independent review runs (Codex) | 101 adapter/helper/timeline · 45 portfolio/scorecard · 11 Phase 4/search |

### 6.1 Two of these tests were worthless when first written

Both Phase-4 integration tests initially **passed against an empty universe** — instrumentation
showed 0 products loaded and 3 excluded, so the assertions were never reached. The fixture now
publishes through the real `_publish_endpoints` (3 products, 228 records), one test drives labels
*and* endpoints through `simulate_labels_with_endpoints`, and non-vacuity is an **assertion**
rather than a one-time manual check.

A related trap, worth recording because it will recur: patching `build_phase4.os.replace` patches
the **shared** `os` module attribute, so the injected failure fired inside an unrelated atomic
write and the test passed against deliberately broken code. The patch now fires only for a
destination ending `.endpoints.json`, and was falsified by reverting the production code to
`write_text`.

---

## 7. Still open

1. **Miner wiring** — helper and tests exist; the call site does not (§4).
2. **The ATR blocker is UNRESOLVED.** It is not addressed, worked around, or diminished by
   anything in this note.
3. **No re-measurement.** No live regeneration was run; §5 stands unresolved by design.
4. **Operator decisions, not taken here:** whether to wire the miner while the ATR blocker is
   open, and merge order beyond #76 → #77 → #78. No merge, deploy, or production change was made.
