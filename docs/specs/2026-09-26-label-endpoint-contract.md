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

The simulation is the only component that knows which exit occurred, so it must publish one record per
labelled row.

**A bar timestamp is not an event time, and the record must not let one stand in for the other.**
`build_phase2._load_history_parquet` renames `start` to `ts`, so `ts` is the bar's **opening** instant,
while the entry price is that bar's **close**. A single `entry_ts`/`exit_ts` pair would therefore mean
two different things at once — and releasing a portfolio slot at a raw exit-bar `ts` would release it a
full bar early. The record separates them:

| Field | Meaning |
|---|---|
| `entry_row_id`, `exit_row_id` | original-frame row identities (the authoritative occupancy boundary) |
| `entry_bar_start`, `exit_bar_start` | the bars' opening instants, i.e. their `ts` values |
| `bar_duration_ms` | **declared**, not inferred (3_600_000 for hourly). Gap duration cannot be read off the next observed row, because the next row may itself be missing. |
| `entry_available_at`, `exit_observable_at` | `bar_start + bar_duration_ms` — the earliest instant at which that bar's close, high and low are known |
| `exit_kind` | `stop` \| `trail` \| `horizon` |
| `exit_price_basis` | `assumed_stop_level` \| `assumed_trail_level` \| `bar_close` |
| `bars_held` | `exit_row_id − entry_row_id`, recorded rather than recomputed |
| `product_id`, `horizon` | what this endpoint belongs to |
| `label_value` (or its digest) | the PnL this endpoint produced, bound in |
| `label_version`, `cost_version`, `config_id`, `data_id` | label semantics, cost model, parameter set, input frame |
| `intrabar_timing_known` | **always `false`** for `stop` and `trail` (see §4) |

**No PnL realisation and no slot release at a raw bar start.** An OHLC stop or trail condition is
observable only once the bar has completed, so the earliest defensible realisation instant is
`exit_observable_at`. Occupancy itself is expressed in **row ids**; timestamps are derived, never the
primary key.

**Why `product_id`, `horizon` and `label_value` are bound in:** without them a structurally valid
endpoint can be paired with a *different* cached PnL or a different horizon and nothing detects it. The
binding must be independently digested or referenced, so a mismatch fails rather than resolving.

Identities are **original-frame row ids**, never positions in a filtered or reindexed frame.

### Required validations

Reject, rather than repair: `exit_row_id > entry_row_id`; `bars_held` equal to the difference **and**
`≤ min(horizon, max_hold_bars)`; all prices and the return finite; and `exit_kind` consistent with
`exit_price_basis` (`horizon` ⇔ `bar_close`, `stop` ⇔ `assumed_stop_level`, `trail` ⇔
`assumed_trail_level`). A row id unresolvable in the declared `data_id` is malformed, not recoverable.

## 3. Both consumers read it; neither re-derives it

- **Mining eligibility**: occupancy runs from `entry_row_id` to `exit_row_id`. `build_next_eligible`'s
  wall-clock arithmetic is replaced by the published endpoint, so a gap cannot make eligibility shorter
  than the label.
- **Portfolio replay**: the position closes at the published endpoint's `exit_observable_at` and the cap
  slot is released then — not at `entry_bar_start + h × 1h`, and not at the exit bar's *start*. (The
  field names in the table at the top of this document are the CURRENT code's variables, which §2
  replaces.)
- **Validation at both**: an endpoint whose `data_id` does not match the frame being replayed is
  **excluded with a named reason** (granularity defined in §3a). Neither consumer may fall back to
  horizon arithmetic, because a silent fallback is what this contract removes.

**Row ids must be translated, never compared raw.** Endpoint ids index the ORIGINAL frame, while
eligibility and replay operate on a frame that has been sorted, label-filtered and reindexed — so a raw
source index and a working-array index are different coordinate systems, and comparing them directly is
the same error class as comparing a row's horizon against the file it came from. Each consumer must
carry an explicit original-id → working-position map alongside its frame, or resolve by timestamp
search; and the label producer must **preserve each row's original identity across the `dropna`** so the
map can exist at all. A row id that does not resolve in the working frame is a validation failure, not
an occasion to fall back to positional arithmetic.

### 3a. Exclusion granularity

The unit of exclusion is the **row**, and the profile-level consequence is stated separately so the two
cannot drift:

| Situation | Behaviour |
|---|---|
| tail row with no label (`last_idx >= n`) | **normal and expected.** No endpoint, no trade, reported as unlabelled — not malformed, not an error |
| row whose endpoint is missing where one is required | that row is excluded with reason `endpoint_missing`; the profile survives |
| row whose endpoint is **malformed** (fails §2's validations, unresolvable id, `data_id` mismatch) | that row is excluded with a distinct reason, and the profile is **flagged** rather than silently thinned |
| any consumer requiring complete coverage | a profile with any excluded row is **not** complete evidence; it is reported as incomplete, with counts by reason |

So a malformed endpoint never silently removes one observation from an otherwise-complete-looking set:
completeness is reported, not assumed.

**Tie-order.** Occupancy is the **half-open interval `[entry_row_id, exit_row_id)`**: the exit bar
itself is eligible to open the next position. This preserves today's behaviour — `searchsorted(…,
right=False)`, `if i < open_until`, and `exit_ts <= ts` all treat the boundary bar as available — and
the point of writing it down is that three independent implementations currently agree by coincidence.
The convention is only defensible where it is consistent with information availability: entering on
the exit bar's close is fine, since that bar is complete; it would not be fine to enter *within* the
exit bar on the strength of an intrabar exit whose time is unknown (§4).

## 4. A simulated endpoint is not a fill

`exit_row_id` and `exit_bar_start` name **the bar on which the condition was observed**, and
`exit_observable_at` names the earliest instant at which that observation was possible. None of them is
the time of a fill, and for `stop` and `trail` none is even a known time *within* the bar — an OHLC bar
records four prices and no ordering. So `exit_observable_at` is a lower bound on when a fill could have
happened, not an estimate of when it did. Reports may not describe any of these as an execution time,
and `intrabar_timing_known` is `false` precisely so no downstream consumer can quietly assume
otherwise.

The prices are assumptions too: both triggered exits fill **exactly at the threshold** (the stop level,
or `peak × (1 − atr_pct)`), never at the observed low. Real execution gaps through stops. Labels are
net of a constant `round_trip_fee` and model **no slippage and no spread**, so the cost treatment is
partial and the fill assumption is optimistic. `cost_version` exists so this can be stated rather than
inferred, and a report may not call these returns net-of-costs without qualifying which costs.

## 5. Two unresolved concerns this contract records and does not fix

**(a) Intrabar high/low ordering — an unsupported assumption, with bias of unknown sign.** In step 2
the `peak` is raised using bar `i`'s **high** and then compared against bar `i`'s **low**. Within one bar
the order of the high and the low is unknown. If the low occurred *first*, the trail exit cannot have
been triggered by a peak set later in the same bar — yet the simulation triggers it.

So the trail path assumes `high_before_low` **without evidence**. It would be an overclaim to call that
assumption uniformly favourable: it manufactures trail exits that could not have happened, and such an
exit may lock in a gain OR cut short a position that would have recovered, so the resulting bias can run
in either direction and its sign is not established. What can be said is that the affected labels are
not derivable from the data alone. The record therefore carries `exit_kind = trail` and an
`intrabar_order_assumption` field naming the assumption, so those labels are identifiable. Resolving it
requires finer-grained data or a deliberately conservative rule; either is a semantic change and out of
scope here.

**(b) Contemporaneous ATR — CONFIRMED, no longer an open question.** An earlier draft of this document
left this as requiring verification. It has since been verified in the source, and the answer is the
unfavourable one: `features._wilder_atr14` computes the true range from the **current** bar's high−low
and from the current high and low against the *previous* close, and `add_trend_features` then divides by
the **current** close. So `atr14_pct[i]` is a function of bar `i`'s own high, low and close.

`labels._simulate_one` uses exactly that value — `atr_pcts[i]` — to decide an exit **inside** bar `i`.
The threshold therefore depends on the completed bar while purporting to govern a decision taken during
it. **This is a causality defect in the labels themselves, which no endpoint record repairs**, and it
is recorded here rather than fixed: silently altering the ATR input would change every label and every
downstream number, which is a semantic change requiring its own contract and re-mining.

**Reproduced end-to-end, not merely read.** Two synthetic bars, `ts = [0, 3_600_000]`,
`open = [100, 100]`, `high = [100, 200]`, `low = [100, 93]`, entry close `100`. Changing **only the
second bar's close** — every other input identical — changes the label:

| second bar's close | `atr14_pct[1]` | `label_h1` |
|---|---|---|
| `100` | 0.076428571… | **0.835142857…** |
| `190` | 0.040225564… (then the 6 % floor) | **0.868000000…** |

No stop fires in either case (the 8 % stop sits at 92, the low is 93), so the difference is entirely the
trail. **The supposed intrabar trail fill therefore depends on the bar's final closing price** — a
number that does not exist at the moment the fill is claimed to occur. Verified independently by both
sessions; reproduction retained at `.coordination/atr-current-close-reproduction.json`.

This is what makes the defect unarguable: it is not an inference from how ATR is conventionally
computed, it is an observed dependence of a label on information from after its own decision point.

Consequences that must be honoured until it is resolved: `label_version` may **not** be described as
causal; the record must name which ATR column was used; and any trail-exit label must be treated as
conditioned on information not available at decision time. Neither a lagged ATR nor a changed fill
ordering may be introduced as part of an endpoint change — each is a separate versioned semantic
decision requiring its own contract and re-mining.

Both are stated as open because a contract that quietly assumed them away would be the same failure as
the clock mismatch it is written to remove.

## 6. Migration

- **New `label_version`.** Existing labels carry no endpoints, and **the label scalar alone cannot
  reconstruct one**: re-deriving an endpoint from the horizon is exactly the wrong-clock arithmetic being
  removed, and an early-stopped trade's real exit is unrecoverable from a PnL number. Note the precise
  claim — endpoints *can* be regenerated by **re-running the original simulation over versioned
  inputs**, which produces new artifacts under the new `label_version`. What is impossible is inferring
  them from existing label values, which is why legacy rows are excluded rather than back-filled.
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
| **interior NaN from bad input** | the tail check explains tail NaNs but does **not** prove interiors are always labelled — a non-finite price or a malformed input row can produce an interior `NaN`. Such a row must be excluded with its own reason, never treated as a zero-return trade |
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
