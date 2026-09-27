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

**Require** all of the following, and **reject any violation** rather than repairing it:

- `exit_row_id > entry_row_id` — a zero-duration endpoint (`exit == entry`) is rejected, as is any
  reversed pair.
- `bars_held == exit_row_id − entry_row_id`, and `bars_held ≤ min(horizon, max_hold_bars)`.
- `entry_bar_start` and `exit_bar_start` equal to the **source bars'** timestamps at those row ids in
  the declared `data_id` — not merely plausible values.
- `entry_available_at == entry_bar_start + bar_duration_ms`, and likewise for `exit_observable_at`, with
  the **declared** `bar_duration_ms`.
- all prices and the return finite.
- `exit_kind` consistent with `exit_price_basis`: `horizon` ⇔ `bar_close`, `stop` ⇔
  `assumed_stop_level`, `trail` ⇔ `assumed_trail_level`.

A row id unresolvable in the declared `data_id` is malformed, not recoverable. (An earlier draft opened
this list with "Reject, rather than repair: `exit_row_id > entry_row_id`", which read as an instruction
to reject every *valid* exit — the requirement and its violation had been collapsed into one sentence.)

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
map can exist at all.

**An absent exit row is not an invalid endpoint.** A perfectly valid exit can land on a source row that
the working frame dropped, because that row has no label of its own — so requiring the exit row to be
present among the filtered candidates would reject correct endpoints. Resolution is therefore two-step:
resolve `entry_row_id` and `exit_row_id` against the **original** frame identified by `data_id`, then map
the exit to the **first retained eligible row at or after the source exit** — or to a terminal sentinel
when no retained row follows it.

> Original rows at hours `[0, 1, 2, 3, 4]`; retained `[0, 1, 3, 4]`; a valid exit at source row `2`.
> The next eligible candidate is source row `3`. It is **not** a rejection.

Only an id that cannot be resolved **in the original frame** is a validation failure. Absence from the
working frame is a mapping step, not an error.

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
`exit_observable_at` names the earliest instant at which **that evidence** is available under the
declared bar model. None of them is the time of a fill, and for `stop` and `trail` none is even a known
time *within* the bar — an OHLC bar records four prices and no ordering.

**`exit_observable_at` is not a bound on the fill in either direction.** An earlier draft called it a
lower bound on when a fill could have happened, which is backwards: a real stop or trail fill would occur
**during** the bar, therefore *before* `exit_observable_at`. The field says only when the completed-bar
evidence became available. Realising PnL at the bar close is an **accounting convention** adopted because
nothing finer is knowable from OHLC — it is not a claim about execution timing, and not an attestation of
anything. Reports may not describe any of these fields as an execution time, and
`intrabar_timing_known` is `false` precisely so no downstream consumer can quietly assume otherwise.

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

**The contamination is version-wide, not confined to trail exits.** An earlier draft of this document
said only that trail-exit labels are affected. That understates it: the ATR comparison is evaluated at
**every** bar the walk traverses, so deciding *not* to trail at bar `i` also consults bar `i`'s
contemporaneous ATR. A `horizon` label and a later `stop` label therefore inherit the same defect
through the bars they survived. `exit_kind` cannot isolate the affected rows, and filtering on it would
imply a cleanliness that does not exist.

The one narrow exception is not usable as a filter either: because the code is **stop-first**, a label
whose stop fires on its very first bar returns before any ATR is read. But `exit_kind = stop` alone does
not establish that — a stop on bar five consulted the ATR on bars one to four — so distinguishing the
exempt rows requires knowing the whole prior path, which is exactly what the label scalar does not carry.

Consequences that must be honoured until it is resolved: `label_version` may **not** be described as
causal; the record must name which ATR column was used; **the entire label version carries a causality
blocker** pending a corrected replay, rather than a per-row annotation; and neither a lagged ATR nor a
changed fill ordering may be introduced as part of an endpoint change — each is a separate versioned
semantic decision requiring its own contract and re-mining.

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

## 9. Consumer integration: the operational contract

§3 establishes that both consumers read the endpoint and neither re-derives it. This section
adds what turned out to be needed to actually do it, all of it found by reading and running
the current code rather than by reasoning about it. Nothing here weakens §3 or §3a; §9.3
reconciles an apparent conflict with §3a.

### 9.1. Two quantities, two units, never substituted

| quantity | type | unit | decides |
|---|---|---|---|
| eligibility boundary | position | index into the consumer's working frame | when a slot reopens |
| accounting time | instant | milliseconds | when realized PnL lands on the equity curve |

The code this replaces used a single wall-clock `exit_ts` for both roles, and that is why one
defect produced errors in **opposite directions**: an early exit released the slot late and
realized PnL late, while a gap realized PnL early and released the slot early, permitting two
positions in one leaf. A position is not a time. Any function that accepts one where the other
belongs is wrong even when the arithmetic happens to agree.

**Entry decisions are available at the entry bar's CLOSE, not its start.** The record states
this directly — `entry_available_at = entry_bar_start + bar_duration_ms` — and the reason is
information availability: the features a rule reads are close-derived. The replay currently
iterates raw bar *starts* and evaluates rules there, which dates every entry one bar early.
§3 fixed the exit clock and left this one unstated; it is stated here.

A consumer MUST cross-check each record's `entry_available_at` against
`row.ts + bar_duration_ms` for the row it claims, and reject a disagreement. The frame and the
endpoints must describe the same bars, and this is the cheapest place to find out that they do
not.

### 9.2. The anchor chain, and where it terminates

The producer writes, beside each product's labelled frame:

- `endpoints/{pid}/endpoints.jsonl` and `endpoints_manifest.json` — the dataset and its manifest;
- `{pid}.endpoints.json` — a **sidecar** holding `manifest_digest`, `data_id`, `product_id`,
  `horizons`, `bar_duration_ms`, `feature_recipe`, `label_version`, `cost_version`,
  `exit_config` (`stop_loss_pct`, `atr_trail_floor`, `max_hold_bars`, `round_trip_fee`), and
  `record_digests` keyed `"{horizon}:{entry_row_id}"`.

A consumer verifies the manifest against the sidecar's `manifest_digest` — a value it did not
compute — and each record against `record_digests`, which were computed **at publication**. A
digest recomputed from the record under test attests nothing, which is the circularity the
external anchor exists to break.

**The sidecar is an anchor, not a root of trust.** Replacing the frame, the dataset and the
sidecar together produces a coherent triple that nothing in this pipeline can reject. The chain
terminates at whatever the consumer independently retains, and here that is the sidecar. This is
a documented boundary, not a solved problem, and it must not be described as tamper-proof.

**`config_id` equals `data_id` by construction.** The exit config is an input to
`build_data_id`, so the producer sets them to the same value. Two consequences, both worth
stating because neither is obvious:

1. Recomputing `build_data_id` from the frame's own arrays **and** the declared `exit_config`
   verifies frame identity and config identity in one step. Quoting the sidecar's `data_id`
   back into a loader proves only that the manifest agrees with the sidecar; it does **not**
   bind the frame. The recompute is what binds it.
2. On a mismatch, the id alone cannot say whether the frame moved or the config did. That is a
   diagnostic limitation, not a soundness one, and separating them would be a producer change.

### 9.3. Coverage, and reconciling this with §3a

§3a makes a row whose endpoint is missing an **exclusion with a named reason**, leaving the
profile alive but reported incomplete. That remains correct for a consumer reading an artifact
whose coverage it has not verified. It would be wrong to read it as licence to proceed past an
**artifact-level** mismatch, so the two levels are named separately:

| level | situation | behaviour |
|---|---|---|
| artifact | the dataset does not cover the frame's finite-label rows, or the recomputed `data_id` disagrees | **fail loud.** The two artifacts are not a matched pair, and no per-row accounting can repair that |
| row | one row lacks an endpoint inside an otherwise coverage-verified artifact | §3a: exclude with `endpoint_missing`, profile survives, completeness reported |

In this pipeline the producer publishes an endpoint for every finite-label row, and the consumer
verifies complete coverage at load. The row-level path is therefore a **defensive assertion**
rather than the expected path, and a consumer must never reach for horizon arithmetic at either
level — a silent fallback is precisely what this contract removes.

**Expectations must be built independently of the records, and for every declared horizon.**
Enumerating expected keys *from* the records makes a missing record undetectable, because
removing a record removes its own expectation. And a loader that rejects records absent from
the expectations will reject every record of every *other* horizon if it is handed one
horizon's expectations — so expectations are built from the **unfiltered** frame across all
horizons the sidecar declares, and complete coverage is required there.

**"Finite" means `isfinite`, not "not null".** `dropna` retains `+inf` and `-inf`. A consumer
that retains rows with `notna` while building expectations with `isfinite` disagrees with
itself: an infinite-labelled row is retained, has no expectation and no endpoint, and then
fails as a spurious coverage error rather than as the data problem it is. Both sides use
`isfinite`.

**Two id spaces, two validations.** The unfiltered frame's `source_row_id` must be exactly
`0..n-1` — the producer writes `arange(n)`, so anything else means the frame was filtered,
reordered or concatenated after labelling and its positions no longer mean what the endpoints
reference. Uniqueness must be checked *before* these become dictionary keys, because a
duplicate silently overwrites rather than fails. A *filtered* frame's ids must be strictly
increasing and unique, with gaps expected — gaps are the filter's whole purpose.

**Ids are validated, never coerced.** `to_numpy(dtype="int64")` and `int()` truncate silently.
A truncated timestamp produces a bar start that does not describe the frame; a truncated row id
is worse, because it still points at a real row — just the wrong one.

### 9.4. The expected cap is the configured cap, not the horizon

A record publishes `max_hold_bars` from the **exit configuration**, not from its own horizon.
The simulation uses `min(horizon, max_hold_bars)` internally, but the record carries the
configured value: a `horizon=1` record published under the default configuration carries
`max_hold_bars = 168`.

Any consumer that reconstructs the expected cap from the horizon **rejects every valid
short-horizon record.** The cap comes from the declared `exit_config`, strictly validated.

### 9.5. Occupancy metrics keep their sampling basis

`pct_slots_full` and `mean_concurrent` are sampled **once per unique decision instant** across
the participating products — the same union-of-product-times convention the pre-integration loop
used. Two consequences for any implementation:

- Additional instants introduced only so that a due position can be examined (for example an
  exit observable after the last decision instant) MUST NOT become sampling points. Folding them
  into the sampled set changes both metrics with nothing in the code looking wrong.
- Only **participating** products contribute instants. Including every supplied product would let
  an unrelated input change the denominator.

A time-weighted occupancy measure is defensible and is **not** part of this contract. Introducing
one in the same change that corrects the exit clock would make a metric redefinition
indistinguishable from the correction.

### 9.6. What consumer integration does not establish

Verifying integrity, attribution and coverage is not the same as validating semantics, and a
clean load is not evidence of a correct simulation. In particular:

- The `label_atr_contemporaneous_causality` blocker of §5 remains **UNRESOLVED**. Making the
  pipeline self-consistent makes its numbers *coherent*; it does not make them *trustworthy*, and
  a coherent number reads as a credible one, which is a hazard worth naming.
- Every research verdict recorded before this integration was computed on the pre-integration
  occupancy model and on entries dated one bar early. Those numbers are not comparable with
  anything produced afterwards.
- The fold purge remains as it is. Setting it from observed `bars_held` would leak an **outcome**
  into fold construction — the same error class as post-test rows appearing in TRAIN. If it is
  ever changed, the defensible quantity is `max_hold_bars`, known before any outcome exists.
