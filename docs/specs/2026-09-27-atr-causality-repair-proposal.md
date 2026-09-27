# Proposal: repairing the ATR causality defect in dynamic-exit labels

**Date:** 2026-09-27
**Status:** PROPOSAL for the LABEL POLICY. Nothing here is implemented.
**Measured:** §10 carries real per-variant results from the offline probe. They show the
repairable half moves 0.06% of records and the unrepairable half 1.25%, which inverts the
priority this document originally implied. Read §10 before §3.
**Gate:** operator approval is required before any change to **production label semantics**
(a v3 label version, a new blocker set, anything the producer writes). It is NOT required for
the offline diagnostic in §5 and its tests, which preserve v2 and live semantics and are
already within the standing authorisation — an earlier header gated those too, which was
over-broad (Codex `c3c5acc8`).
**Blocker addressed:** `label_atr_contemporaneous_causality` (`endpoint_records.py:41`)
**Related:** `docs/specs/2026-09-26-label-endpoint-contract.md`,
`docs/handoffs/2026-09-26-occupancy-correction-evidence.md`

---

## 1. The defect, located exactly

`labels.py:_simulate_one` walks bars `i = entry_idx + s` for `s = 1..horizon_cap`. At each bar:

```python
if bar_high > peak:
    peak = bar_high                 # highs[i]
atr_now = float(atr_pcts[i])        # <-- the defect
atr_pct = max(atr_now, atr_trail_floor)
if bar_low / peak - 1.0 <= -atr_pct:
    exit_price = peak * (1.0 - atr_pct)
```

`atr_pcts[i]` is **a function of bar `i` itself**, twice over:

- `features.py:_wilder_atr14` computes `TR_t = max(high_t - low_t, |high_t - close_{t-1}|,
  |low_t - close_{t-1}|)`. The first term uses **bar `t`'s own high and low**.
- `features.py:66` then divides by `close`: `out["atr14_pct"] = _wilder_atr14(...) / close`, so the
  published ratio also depends on **bar `t`'s close**.

There is no `shift` anywhere in that path — verified, not assumed.

So the threshold that decides whether an exit fires *during* bar `i` is computed from bar `i`'s
completed high, low and close. A trader standing at bar `i`'s open cannot know it. That is a
genuine lookahead, and it is why every published record carries the blocker.

### 1.1 What is NOT a defect, and must not be "fixed" by accident

`atr14_pct` is also a **feature** (`mine_profiles._FEATURE_COLUMNS` includes it), and there it is
correct. A rule evaluates features at row `r` and the entry becomes available at
`entry_bar_start + bar_duration_ms`, i.e. row `r`'s close — the instant at which `atr14_pct[r]` is
exactly known. Lagging the feature would degrade a sound signal. **The repair must change the
label's threshold only, never the feature column.**

---

## 2. The defect splits in two, and only one half can be repaired

This distinction is the substance of the proposal. Treating the blocker as one problem is what
has made it look intractable.

| # | issue | nature | repairable? |
|---|---|---|---|
| A | the trail **threshold** at bar `i` is computed from bar `i`'s own OHLC | genuine lookahead — information from after the decision point | **YES**, by lagging |
| B | `peak` is raised from `high_i` and then compared against `low_i` | ordering **ambiguity**: an OHLC bar records four prices and no sequence | **NO** — not recoverable from hourly OHLC at all |

Issue B is already declared honestly: `intrabar_order_assumption = "high_before_low"` on trail
records, and `intrabar_timing_known = False` for every triggered exit. It is a property of the
input data, not a coding error. **No amount of lagging removes it**, and a proposal that claimed
to "fix the causality blocker" outright would be overclaiming in exactly the way this effort keeps
catching.

**Therefore the blocker should be SPLIT, not cleared.** Proposed replacement:

- `label_atr_contemporaneous_causality` — repaired by §3, absent from v3 records.
- `label_intrabar_order_assumption` — **permanent, and for EVERY record kind including
  `horizon`**, because survival to the horizon is itself ordering-dependent (see below). §4.1 shows
  it is not a technicality: on a two-bar path the two orderings differ by 3.5x.

**Correction: a `horizon` record needs the ordering blocker too** (Codex `df49b175`). I first
wrote that it carries neither, reasoning that its exit is the close of a known bar so there is no
ordering question. That is wrong, and my own §4.1 counterexample disproves it: under B1 the trade
trails out at bar 1, under B2 it **survives to the horizon**. So whether a record is a horizon
record *at all* depends on the ordering.

The distinction I had collapsed:

- `intrabar_timing_known` is about the exit **instant**. It is legitimately `True` for a horizon
  exit — the close of a known bar.
- Ordering dependence is about **which exit fired**. It reaches horizon records as well, because
  survival to the horizon means no trail or stop triggered first, and whether one triggered is
  exactly the ordering question.

This is the second error I made in this section from the same root: treating the ordering as if it
only affected the exit price or instant, when it also decides which exit occurs. §4.1 was the first.

So under v3 **every** `stop`, `trail` and `horizon` record carries `label_intrabar_order_assumption`,
and ambiguous ones carry a second blocker. The only records genuinely free of the question are
those where both orderings agree *and* no bar in the holding window came within either threshold —
which is **checkable, not assumable**, so it belongs in the §5 probe as a measured
order-insensitive fraction rather than as a blocker exemption.

An earlier draft also said the blocker was "present on both", which read as every record and
contradicted §2's own stop/trail wording and §7's test — three sections disagreeing about one fact
(Codex `c3c5acc8`). The resolution is the strict one: all records, no exemption by exit kind.

---

## 3. Proposed policy: lag-1 ATR

Use the ATR of the **last completed bar** as the threshold for the bar being traded through:

```python
atr_now = float(atr_pcts[i - 1])
```

Why lag exactly 1, and why this is sufficient:

- The trade is decided at row `r`'s close and `entry_price = closes[entry_idx]`, so at the first
  simulated bar `i = r + 1` the lagged value is `atr_pcts[r]` — known at the entry instant. No
  boundary special-case is needed, and no record loses its first bar.
- Lag 1 is the smallest lag that removes the lookahead. A larger lag would be a *modelling*
  choice (a smoother threshold), not a causality requirement, and should not be smuggled in under
  a correctness banner.

### 3.1 Warm-up and non-finite handling

Current behaviour substitutes the floor when `atr_pcts[i]` is not finite:

```python
if not np.isfinite(atr_now):
    atr_now = atr_trail_floor
```

Keep that rule, but note what changes: the NaN region shifts by one row, so the **set of records
that fall back to the floor changes**. That is a real label difference and must show up in the
§5 reproduction rather than being waved through as equivalent.

`atr_trail_floor` (0.06) and `stop_loss_pct` (0.08) are unchanged. **No threshold is being
tuned** — that is explicitly outside this proposal and outside the current authorisation.

---

## 4. Explicit fill and stop ordering

Issue B cannot be resolved, so it must be *declared and bounded*.

### 4.1 Correction: neither ordering dominates

An earlier version of this section claimed the conservative ordering `low_before_high` (B2) was
strictly no better than `high_before_low` (B1), recommended it as the published basis, and
specified a test asserting the inequality. **That was wrong.** Codex review `c3c5acc8` refuted it
by construction and the counterexample reproduces by execution:

```
entry 100, trail floor 6%, stop 8%
bar 1: high 120  low  95  close 115
bar 2: high 150  low 114  close 145

high_before_low   exit 112.80 (trail @ bar 1)   pnl +0.1280
low_before_high   exit 145.00 (horizon)         pnl +0.4500
```

The reasoning error is worth naming because it is easy to repeat: I considered only the exit
**price** — a lower peak gives a lower trail level, hence a lower exit — and ignored that a lower
level also makes the trail **less likely to fire at all**. Under B2 the position survives bar 1
and then rides bar 2's rally. Delaying an exit is not conservative; it trades one risk for
another.

So neither ordering dominates the other, in either direction, and **there is no single
"conservative path" available to publish.**

### 4.2 What follows: bracketing, proposed as the only defensible option

| option | rule | status |
|---|---|---|
| B1 `high_before_low` | raise peak from `high_i`, then compare `low_i` | one arbitrary branch of an ambiguity |
| B2 `low_before_high` | compare `low_i` against the peak as of `i-1`, then raise | the other arbitrary branch; **not** a conservative one |
| **B3 bracket both** | compute B1 and B2; agree → one label; disagree → declare it | **proposed**, pending operator question 1 |

Proposed B3 semantics:

- Compute both orderings for every record.
- **They agree** (same exit row, kind and value): publish that value, unambiguous.
- **They disagree**: publish `min(pnl_B1, pnl_B2)` and carry a new blocker
  `label_intrabar_order_ambiguous`, plus both values in the record so nothing downstream has to
  re-derive them.

`min` is PROPOSED, not approved -- operator question 1 decides it, and Codex `2cc2feec` is right
that it is a diagnostic comparison until then. It is defensible as a candidate precisely because it
does **not** pretend to be a simulated path: it is an explicit lower bound and the record says so.
When it is reported, the smaller-PnL branch must carry **its own** exit, timing and occupancy --
pairing one branch's value with the other's metadata would describe a trade neither ordering
produces. The probe enforces that with `lower_bound_ordering` and `lower_bound_result`. Publishing either branch alone would report a number
the ordering assumption manufactured, which §4.1 shows can differ by 3.5x on a two-bar path.

**But be exact about what it bounds** (Codex `c3c5acc8`): `min(B1, B2)` is a bound over the **two
enumerated orderings only**. It is **not** a bound over all intrabar price paths. A real bar may
visit its low, recover, set its high and fall back, touching a trail level that neither B1 nor B2
triggers; within-bar paths are unbounded in number and hourly OHLC constrains only the extremes.
So the honest claim is:

> `min(B1, B2)` bounds the two orderings this simulation can distinguish. The true worst case over
> all paths consistent with the bar is not computed, and is not claimed to be.

Anyone wanting a genuine worst-case bound needs sub-hourly data, which is a different project. The
record must therefore carry the ordering blocker even when B1 and B2 agree — agreement between two
scenarios is not proof that a third would agree.

The ambiguity rate therefore becomes a **headline figure of the §5 probe, not a footnote**. If it
is small, that is the strongest available statement about the assumption's materiality. If it is
large, the honest conclusion is that hourly OHLC is too coarse to label a trail-stop strategy at
all — a finding worth having explicitly rather than hidden inside a single published branch.

### 4.3 Gap-through fills must be stated, because the current rule is optimistic

Codex `c3c5acc8` asks for this explicitly and it is a real gap in both the code and the earlier
draft. Today:

```python
if bar_low / entry_price - 1.0 <= -stop_loss_pct:
    exit_price = entry_price * (1.0 - stop_loss_pct)     # fills AT the stop level
```

The fill is assumed to occur exactly at the stop level. When a bar **gaps through** the level —
its open is already below it — a real order fills at or near the open, which is worse. The same
applies to the trail branch, which fills at `peak * (1 - atr_pct)`.

The current code therefore reports a better price than the data supports whenever a gap occurs.
Proposed v3 rule, and note it needs the `open` column which the frame already carries:

- `stop`: `exit_price = min(stop_level, open_i)`.
- `trail`: `exit_price = min(trail_level, open_i)`.
- Record a distinct `exit_price_basis` when the gap branch is taken (e.g. `gapped_open`), so a
  consumer can count them rather than having to infer them.

This is a **separate correction from the ATR lag** and should be measured separately in §5, not
folded into the lag's effect. It is included here because a proposal that fixed the threshold
while leaving an optimistic fill would still publish numbers the data does not support.

Stop-versus-trail priority is unchanged: stop-loss is checked first, matching
`cnn_agent._check_risk_exits`. That correspondence with the live exit ladder is deliberate and is
not part of this repair.

## 5. Before/after reproduction — offline, no regeneration

A new script, `backend/tools/strategy_discovery/atr_causality_probe.py`, run on existing Phase 2
frames, writing to a **new output path**. It must not touch `phase2/`, any published endpoint
directory, any model artifact, or the live database. Building it needs no operator gate: it is
offline, additive, and changes no label the producer writes.

**Each change is measured in isolation, then combined** (Codex `c3c5acc8`), because a single
combined diff cannot attribute an effect to a cause:

| variant | lag-1 ATR | ordering | gap fill |
|---|---|---|---|
| v2 baseline | no | B1 | at-level |
| L | **yes** | B1 | at-level |
| O | no | **B3 bracket** | at-level |
| G | no | B1 | **min(level, open)** |
| v3 combined | yes | B3 | min(level, open) |

Reporting L, O and G separately is what makes "the lag changed N records" a statement about the
lag. It also guards against the variants interacting in a way a combined run would hide.

For each variant against v2 it computes, per product and horizon:

1. **Coverage**: records emitted by each version, and the rows where only one version emits.
2. **Exit-kind migration**: a 3x3 matrix of `stop`/`trail`/`horizon` v2 → v3. The interesting cell
   is `trail → horizon` (the threshold moved and the exit no longer fires).
3. **Label deltas**: distribution of `v3.label_value - v2.label_value`, plus the count and
   magnitude of sign flips.
4. **Bars-held deltas**: distribution of `v3.bars_held - v2.bars_held`.
5. **Floor-fallback set change**: rows whose threshold came from the floor in one version only.
6. **B3 ambiguity rate**: fraction of records where `high_before_low` and `low_before_high` give
   different exits, broken down by the v2 exit kind -- horizon records included, since §4.2 shows
   they are not exempt.
7. **Order-insensitive fraction**: records where both orderings agree AND no bar in the holding
   window came within either threshold. These are the only records the ordering provably does not
   touch, and the number is worth knowing precisely because it cannot be assumed.

Reported per product, never pooled into a single headline. **This is a measurement of a label
change, not an evaluation of a strategy**: it says nothing about profitability, and the report
must state so in its own header.

---

## 6. Compatibility and versioning

Non-negotiable: **v2 records are never recomputed, relabelled, or pooled with v3.** The same rule
already governs outcome-label v1 versus v2 (invariant 22), and for the same reason.

| artifact | change |
|---|---|
| `LABEL_VERSION` | new value, e.g. `label_endpoint_v3`; v2 stays valid for existing records |
| `exit_config` | gains `atr_lag_bars: 1` and `intrabar_order: "low_before_high"` |
| `config_id` / `data_id` | change automatically — `build_data_id` covers the config, so a v3 dataset cannot validate against a v2 frame binding, and vice versa. This part genuinely is free. |
| blockers | `label_atr_contemporaneous_causality` absent on v3; `label_intrabar_order_assumption` present on both; `label_intrabar_order_ambiguous` on v3 records where the two orderings disagree (§4.2) |
| loader | must **reject a mixed-version dataset** (already rejects heterogeneous `label_version` via `_HEADER_FIELDS`) |
| **validator — REQUIRED CODE CHANGE** | `_validate_blockers` must gain **version dispatch**. See §6.1: it currently rejects every v3 record. |
| consumers | no signature change; they read whatever version the sidecar declares |
| Phase 4 | `deployment_eligible` stays `false`. v3 removes one blocker; the others (independent holdout, cost/fill, accounting, prospective execution) are untouched. |

### 6.1 Correction: versioning does NOT come free

An earlier draft of this section said the existing `data_id` anchor did all the versioning work
and no new mechanism was needed. That was wrong for the blockers, and I verified it by execution
rather than reading:

```
_validate_blockers(("label_intrabar_order_assumption",))
  -> ValueError: every record of this simulation version must carry the
     label_atr_contemporaneous_causality blocker
inspect.getsource(_validate_blockers): "label_version" in source -> False
```

`_validate_blockers` requires `CAUSALITY_BLOCKER` **unconditionally, with no version dispatch**.
Every v3 record would be rejected by the validator that exists today. The required change:

- `_validate_blockers` takes the record's `label_version` and enforces a **per-version required
  set**: v2 → `{label_atr_contemporaneous_causality}`; v3 → `{label_intrabar_order_assumption}`
  for `stop`/`trail`, `{}` for `horizon`.
- A v2 record that drops its blocker must still be rejected, so the v2 path cannot be weakened by
  the addition. That is a test, not a comment.
- The mandatory-set table lives in one place, keyed by version, so a future version cannot acquire
  a silently empty requirement.

`_EXIT_BASIS` also needs the `gapped_open` basis from §4.3, and `_validate_exit_semantics` pairs
`exit_kind` with basis, so that mapping becomes version-dependent too. Both are why §7 test 5 and
test 6 exist.

Archived v2-based verdicts are **not** invalidated by this change and **not** validated by it
either. They remain uninformative for the reasons already recorded in the occupancy note §5.

---

## 7. Tests required before any of this is believed

1. **The lag is real.** A frame where `atr_pcts[i]` and `atr_pcts[i-1]` straddle the trigger:
   v2 exits at bar `i`, v3 does not. Falsify by removing the lag and watching it fail.
2. **The feature is untouched.** Assert the `atr14_pct` **column** is bit-identical between v2 and
   v3 builds (`float.hex()`, not a tolerance), so the repair provably did not lag the feature.
3. **First-bar boundary.** A record whose exit fires at `s = 1` uses `atr_pcts[entry_idx]`, which
   is known at the entry instant. Assert the exact value used, not merely that it ran.
4. **Warm-up.** Rows inside the first 14 bars fall back to the floor in v3, and the set differs
   from v2 by exactly one row's shift.
5. **Blocker split.** Every v3 record -- `stop`, `trail` AND `horizon` -- carries
   `label_intrabar_order_assumption` and NOT `label_atr_contemporaneous_causality`. A v3 record of
   ANY kind missing the ordering blocker is rejected, horizon included: pin §4.1's fixture, whose
   B2 branch produces a horizon record whose very existence is ordering-dependent. And a **v2**
   record missing the ATR blocker is still rejected, proving the version dispatch did not weaken
   the v2 path -- that last one is what would silently rot.
6. **Version isolation.** A dataset mixing v2 and v3 records fails to load. A v3 dataset fails to
   validate against a v2 frame binding.
7. **Neither ordering dominates, and the bound holds.** Pin §4.1's counterexample exactly as a
   fixture -- entry 100, bars (120, 95, 115) and (150, 114, 145), floor 6% -- and assert
   `high_before_low` yields +0.1280 while `low_before_high` yields +0.4500. This is the
   regression for a claim I actually got wrong, so it is asserted with the very numbers that
   refuted it. Then, as a property test over generated bars: the published value equals
   `min(B1, B2)`, is `<=` both, and equals both exactly whenever the orderings agree. Do NOT
   assert any dominance between B1 and B2 in either direction -- that is the false claim.
8. **No silent regeneration.** The probe writes only under its own output directory; assert the
   Phase 2 directory's contents are unchanged after a run.
9. **Equivalence where it should hold.** On a frame with no trail exits and no gaps, v2 and v3
   labels are bit-identical. The non-vacuity guard: if it fails, the change is broader than
   claimed; without such a case the other tests cannot localise the change.
10. **Gap fills.** A bar whose open is already below the stop level fills at the open, not the
   level, and records the `gapped_open` basis. Falsify by reverting to the at-level fill.
11. **Variant isolation.** The probe's L, O and G variants each differ from v2 in exactly the
   records their own change can touch: L only where the lagged and contemporaneous ATR straddle a
   trigger, G only where a gap occurs, O only where the two orderings disagree.

---

## 8. What this proposal does not do

- It does **not** clear the causality blocker. It repairs one half and makes the other half
  permanent and explicit (§2).
- It does **not** touch thresholds, retrain anything, regenerate any live artifact, or alter v2
  label semantics.
- It does **not** make any profitability claim, and it does not make Phase 4 deployable.
- It does **not** establish that any archived verdict was wrong.

## 10. MEASURED RESULTS, and they invert this document's priority

Run on 2026-09-27 with `atr_causality_report.run_probe`, read-only over the first 8 products of
`backend/data/phase2`, horizon 24, first 1500 entry rows each. Output went to an isolated
scratchpad directory; the frames directory was fingerprinted before and after and was unchanged.

**Scope of these numbers, stated before them:** 8 products, 12,000 comparable records, one
horizon. Not the full universe. **Not a profitability measurement** and not a re-measurement of
any archived verdict. The `legacy` self-check showed 0 changes against itself on every product,
without which none of the rest would be worth reading.

| variant | records changed | share |
|---|---|---|
| `lag_only` — the ATR causality repair | **7** | **0.06%** |
| `ordering_only` — the unrepairable ambiguity | **150** | **1.25%** |
| `gap_only` | 8 | 0.07% |
| `combined` | 299 | 2.49% |

Ordering ambiguity (the two enumerated orderings disagreeing at all): **547 of 12,000 = 4.56%**,
with **51 sign flips** among the records `ordering_only` moved.

### 10.1 The repairable defect is nearly inert, and here is why

`lag_only` changes 0.06% of records. The reason is structural, not a quirk of the sample: the
trail threshold is `max(atr14_pct, atr_trail_floor)` with the floor at **0.06**, and the measured
ATR almost never reaches it.

| product | median `atr14_pct` | bars above the 0.06 floor |
|---|---|---|
| AAVE-USD | 0.0121 | 3 of 8714 (0.03%) |
| ABT-USD | 0.0195 | 285 of 7628 (3.74%) |
| ADA-USD | 0.0110 | 0 of 8714 (0.00%) |

**The floor masks the ATR.** Where `atr <= 0.06` the threshold is the constant floor, so which
bar's ATR is used cannot matter, and lagging it by one bar is a no-op. The lookahead §1 documents
is real in the code and almost absent in effect on this data. Only BOBA-USD showed any
sensitivity at all (7 records).

This does **not** make the repair pointless: a lookahead that is currently masked by a
configuration value is still a lookahead, and it would become live the moment the floor were
lowered or a more volatile universe were mined. But it does mean the repair should not be
described, or prioritised, as materially changing the labels.

### 10.2 The unrepairable half is roughly twenty times larger

`ordering_only` moves 1.25% of records against `lag_only`'s 0.06%, with 51 sign flips and, on
ABT-USD, a per-record PnL delta spanning `-0.0776` to `+0.1924`. Per-product ambiguity varies
enormously — ADA-USD 0 of 1500, AVAX-USD and BNB-USD 1 each, ABT-USD 261, BOBA-USD 206 — so a
single pooled figure would hide the shape.

So the blocker names the half that barely matters, and the half that matters cannot be fixed with
better code. Per §4.1 and §4.2 the honest response is to keep declaring it, not to claim a repair.

### 10.3 The variants interact, which is why isolating them was necessary

`combined` changes 299 records; the isolated variants sum to 7 + 150 + 8 = **165**. The
combination is nearly twice the sum of its parts, so these changes are **not additive** — a lagged
threshold alters which bar triggers, which changes what the ordering and gap rules then see. A
single combined before/after diff would have reported 2.49% with no way to attribute it, and
anyone estimating the parts from the whole would be wrong by ~80%.

### 10.4 What would change my recommendation

If the operator lowers `atr_trail_floor` or extends the universe to products whose ATR routinely
exceeds it, `lag_only` stops being inert and §10.1 no longer applies. The probe should be re-run
in that case rather than these numbers quoted.

---

## 9. Open questions for the operator

1. **Whether `min(B1, B2)` as a declared lower bound is acceptable as the published label**
   (§4.2), given that §4.1 rules out publishing either branch as "the conservative one". The
   alternative is to keep B1 for continuity with v2 exit kinds and rely on the ambiguity blocker
   alone. A modelling decision, not a correctness one -- but note the earlier version of this
   document recommended a conservative single branch that does not exist, and §10.2 shows this is
   the consequential half of the change.

1a. **Given §10.1, is the lag repair still worth a label version at all?** It moves 0.06% of
   records because the 0.06 floor masks the ATR. Three coherent answers, and I do not think this
   one is mine to pick: ship it anyway because a masked lookahead is still a lookahead and the
   mask is a config value; defer it and spend the version on the ordering declaration alone;
   or reconsider `atr_trail_floor` itself, which would make the lag live and is a threshold
   change outside my current authorisation.
2. Whether the probe should run over the full universe or a named subset first.
3. Whether v3 should be built at all before the other Phase 4 blockers are addressed, given that
   removing one of four does not change `deployment_eligible`.
