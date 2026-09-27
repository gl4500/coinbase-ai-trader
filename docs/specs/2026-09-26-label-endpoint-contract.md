# Shared label endpoint contract

**Status:** design input. Docs only — no code, no mining, no retraining, no archive edits, no
threshold changes, no live changes. No trading policy is proposed or implied.

**Problem it exists to remove.** Two consumers independently derive when a labelled trade ends, from
different clocks, and they disagree:

| Consumer | How it decides the trade is over |
|---|---|
| mining eligibility (`profit_split.build_next_eligible`) | **wall-clock**: first bar with `ts >= entry_ts + h × 1h` |
| portfolio replay (`portfolio_sim`) | **wall-clock**: `exit_ts = entry_ts + h × 3_600_000` |
| the label itself (`labels._simulate_one`) | **row offset**, or an earlier triggered exit |

On contiguous hourly data the first two coincide with the third's *nominal* case. On gapped data they
do not, which produced overlapping positions inside one leaf and premature PnL realisation with early
slot release in the portfolio. Draft PR #72 contains that by rejecting non-contiguous source frames.

Containment is not a fix. The underlying defect is that **three components each re-derive a fact only
the simulation knows.** This contract makes the simulation *publish* that fact and the others *consume*
it. Nothing here claims to improve any result; it removes a class of disagreement.

---

## 1. What the simulation actually does

Stated from the code, because a contract built on a paraphrase would be worse than none.
`_simulate_one(entry_idx, horizon, …)` in `backend/tools/strategy_discovery/labels.py`:

- **Entry** at `closes[entry_idx]` — the close of the signal bar.
- `horizon_cap = min(horizon, max_hold_bars)`; `last_idx = entry_idx + horizon_cap`. **If
  `last_idx >= n` it returns `NaN` before evaluating any bar** — so a row whose full horizon does not
  fit is unlabelled even when a stop would have triggered on its first bar.
- Walking `s = 1 … horizon_cap`, at each bar `i = entry_idx + s`:
  1. **Stop-loss first** (documented as matching `cnn_agent._check_risk_exits`): if
     `lows[i]/entry − 1 ≤ −stop_loss_pct`, exit at the **assumed stop price**
     `entry × (1 − stop_loss_pct)` — not at the observed low.
  2. **Trail**: `peak` is raised to `highs[i]` **including the current bar**, then if
     `lows[i]/peak − 1 ≤ −atr_pct` the exit is at `peak × (1 − atr_pct)`, where
     `atr_pct = max(atr_pcts[i], atr_trail_floor)` and a non-finite ATR falls back to the floor.
  3. Otherwise continue.
- **Horizon exit** at `closes[last_idx]`.
- Every return is **net of `round_trip_fee`**.

So there are exactly three exit kinds — `stop`, `trail`, `horizon` — and only the third lands on a bar
boundary the outside world can name.

## 2. The endpoint record

The simulation is the only component that knows which exit occurred. It must emit, per labelled row:

| Field | Meaning |
|---|---|
| `entry_row_id`, `entry_ts` | original-frame row identity and timestamp of the entry bar |
| `exit_row_id`, `exit_ts` | original-frame row identity and timestamp of the bar **on which** the exit condition was observed |
| `exit_kind` | `stop` \| `trail` \| `horizon` |
| `exit_price_basis` | `assumed_stop_level` \| `assumed_trail_level` \| `bar_close` |
| `bars_held` | `exit_row_id − entry_row_id`, recorded rather than recomputed |
| `label_version`, `cost_version`, `config_id`, `data_id` | provenance: label semantics, cost model, parameter set, and the input frame |
| `intrabar_timing_known` | **always `false`** for `stop` and `trail` (see §4) |

Identities are **original-frame row ids**, never positions in a filtered or reindexed frame. A row id
that cannot be resolved in the declared `data_id` is a malformed endpoint, not a recoverable one.

## 3. Both consumers read it; neither re-derives it

- **Mining eligibility**: occupancy runs from `entry_row_id` to `exit_row_id`. `build_next_eligible`'s
  wall-clock arithmetic is replaced by the published endpoint, so a gap cannot make eligibility shorter
  than the label.
- **Portfolio replay**: the position closes at the published `exit_ts` and the cap slot is released
  then — not at `entry_ts + h × 1h`.
- **Validation at both**: a profile whose rows lack endpoints, or whose endpoint `data_id` does not
  match the frame being replayed, is **excluded with a named reason**. Neither consumer may fall back
  to horizon arithmetic, because a silent fallback is what this contract removes.

**Tie-order.** Occupancy is the **half-open interval `[entry_row_id, exit_row_id)`**: the exit bar
itself is eligible to open the next position. This preserves today's behaviour — `searchsorted(…,
right=False)`, `if i < open_until`, and `exit_ts <= ts` all treat the boundary bar as available — and
the point of writing it down is that three independent implementations currently agree by coincidence.
The convention is only defensible where it is consistent with information availability: entering on
the exit bar's close is fine, since that bar is complete; it would not be fine to enter *within* the
exit bar on the strength of an intrabar exit whose time is unknown (§4).

## 4. A simulated endpoint is not a fill

`exit_ts` names **the bar on which the condition was observed**. It is not the time of a fill, and for
`stop` and `trail` it is not even a known time *within* that bar — an OHLC bar records four prices and
no ordering. Reports may not describe it as an execution time, and `intrabar_timing_known` is `false`
precisely so no downstream consumer can quietly assume otherwise.

The prices are assumptions too: both triggered exits fill **exactly at the threshold** (the stop level,
or `peak × (1 − atr_pct)`), never at the observed low. Real execution gaps through stops. Labels are
net of a constant `round_trip_fee` and model **no slippage and no spread**, so the cost treatment is
partial and the fill assumption is optimistic. `cost_version` exists so this can be stated rather than
inferred, and a report may not call these returns net-of-costs without qualifying which costs.

## 5. Two unresolved concerns this contract records and does not fix

**(a) Intrabar high/low ordering — an in-bar look-ahead.** In step 2 the `peak` is raised using bar
`i`'s **high** and then compared against bar `i`'s **low**. Within one bar the order of the high and the
low is unknown. If the low occurred *first*, the trail exit cannot have been triggered by a peak set
later in the same bar — yet the simulation triggers it. So the trail path assumes, without evidence,
that the high preceded the low, which is the assumption most favourable to the trail. The endpoint
record must therefore carry `exit_kind = trail` explicitly so the affected labels are identifiable, and
an `intrabar_order_assumption` field naming the assumption (`high_before_low`). Resolving it requires
finer-grained data or a deliberately conservative rule; either is a semantic change and out of scope
here.

**(b) Contemporaneous ATR — requires verification, not assertion.** The trail threshold at bar `i` uses
`atr_pcts[i]`, taken from the phase-2 `atr14_pct` column. Whether that value is computed **using bar
`i`'s own high, low and close** has not been verified in this document, and it decides whether the trail
rule consults information that was unavailable while bar `i` was forming. The check is narrow: if
`atr14_pct` at row `i` includes row `i`'s own range — as a conventional ATR does — then using it for an
intrabar decision at row `i` is contemporaneous and the labels inherit a causality defect that no
endpoint record repairs. Until that is settled, `label_version` must not be described as causal, and
the `cost_version`/`config_id` pair must record which ATR column was used.

Both are stated as open because a contract that quietly assumed them away would be the same failure as
the clock mismatch it is written to remove.

## 6. Migration

- **New `label_version`.** Existing labels carry no endpoints and none can be reconstructed:
  re-deriving an endpoint from the horizon is exactly the wrong-clock arithmetic being removed, and an
  early-stopped trade's real exit is unrecoverable from the label value alone.
- **Legacy rows are excluded from endpoint-requiring evidence with a named diagnostic, never
  relabelled** — consistent with `chronological_distinct_folds_v2` and the exact-rule cohorts.
- **The contiguity guard (#72) stays.** It is independent containment, and it remains the only
  protection for any path still reading a legacy label.
- Endpoints do **not** retire the diagnostics requirements: a published endpoint makes occupancy
  agreeable, not the evidence sufficient.

## 7. Regression cases

Every one must be asserted, and each must survive serialisation:

| Case | Required behaviour |
|---|---|
| **gapped source frame** | endpoint row ids still bound occupancy correctly; no consumer falls back to wall-clock |
| **interior missing label** | the row has no endpoint and is excluded; the surrounding rows' endpoints are unaffected |
| **early stop** | `exit_row_id < entry_row_id + horizon_cap`, `exit_kind = stop`, occupancy ends at the real exit — not the nominal horizon |
| **capped horizon** | `horizon > max_hold_bars` uses `horizon_cap`; the recorded `bars_held` reflects the cap, not the request |
| **trail at the same bar as a new high** | `exit_kind = trail` with `intrabar_order_assumption` recorded (§5a) |
| **end of data** | `last_idx >= n` yields no label and no endpoint; it is not a zero-return trade |
| **malformed endpoint** | `exit_row_id` before `entry_row_id`, unresolvable ids, or a `data_id` mismatch → excluded with a distinct reason per failure |
| **both consumers** | identical occupancy for identical endpoints, asserted against the real `build_next_eligible` and `simulate_portfolio`, not a reimplementation |

The last row is the one that matters most: the defect being removed was two implementations of one
rule, so the test must compare the two *actual* consumers.

## 8. What this does not resolve

- Intrabar ordering (§5a) and ATR causality (§5b) — recorded, not fixed.
- Execution realism: assumed threshold fills, no slippage, no spread.
- Everything in the per-fold policy evidence specification: frozen policies, per-fold reporting,
  funded-capital ledgers, matched-cost baselines, holdout discipline.
- Whether any labelled return corresponds to an achievable trade.
