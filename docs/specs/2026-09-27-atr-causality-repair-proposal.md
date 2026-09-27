# Proposal: repairing the ATR causality defect in dynamic-exit labels

**Date:** 2026-09-27
**Status:** PROPOSAL. Nothing here is implemented. Requires operator approval before any code change.
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
- `label_intrabar_order_assumption` — **permanent** for `stop`/`trail` records under bar data,
  present on v3 records too, and it must remain a blocker on any deployment claim.

A v3 record therefore still carries a blocker. That is the correct outcome, and anyone reading a
"causality fixed" headline should be pointed at this paragraph.

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

Issue B cannot be resolved, so it must be *declared and bounded*. Three options:

| option | rule | effect on labels | recommendation |
|---|---|---|---|
| **B1** keep `high_before_low` | peak raised from `high_i`, then compared to `low_i` | status quo; the most optimistic reading (a higher peak means a higher trail exit price) | **not recommended as the only basis** — it is the favourable branch of an ambiguity |
| **B2** conservative `low_before_high` | compare `low_i` against the peak **as of `i-1`**, then raise the peak | strictly ≤ B1 PnL; exits fire at a lower level | **recommended as the published basis** |
| **B3** bracket both | compute B1 and B2; agree → one label, disagree → mark the record ambiguous | most honest, but yields a third state downstream consumers do not model | **recommended as a diagnostic**, reported alongside B2, not as the label |

Recommending B2 for the label and B3 as a measured diagnostic: B2 never reports a profit the
ordering assumption manufactured, and B3 quantifies how many records the ambiguity actually
touches. If B3 shows the disagreement is rare, that is itself the strongest available statement
about the assumption's materiality; if it is common, the honest conclusion is that hourly OHLC is
too coarse for a trail-stop label, which is a finding worth having explicitly.

Stop-versus-trail priority is unchanged: stop-loss is checked first, matching
`cnn_agent._check_risk_exits`. That ordering is a deliberate correspondence with the live exit
ladder and is not part of this repair.

---

## 5. Before/after reproduction — offline, no regeneration

A new script, `backend/tools/strategy_discovery/atr_causality_probe.py`, run on existing Phase 2
frames, writing to a **new output path**. It must not touch `phase2/`, any published endpoint
directory, any model artifact, or the live database.

It computes, per product and horizon, for v2 (current) against v3 (lagged + B2):

1. **Coverage**: records emitted by each version, and the rows where only one version emits.
2. **Exit-kind migration**: a 3x3 matrix of `stop`/`trail`/`horizon` v2 → v3. The interesting cell
   is `trail → horizon` (the threshold moved and the exit no longer fires).
3. **Label deltas**: distribution of `v3.label_value - v2.label_value`, plus the count and
   magnitude of sign flips.
4. **Bars-held deltas**: distribution of `v3.bars_held - v2.bars_held`.
5. **Floor-fallback set change**: rows whose threshold came from the floor in one version only.
6. **B3 ambiguity rate**: fraction of records where `high_before_low` and `low_before_high` give
   different exits.

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
| `config_id` / `data_id` | change automatically — `build_data_id` covers the config, so a v3 dataset cannot validate against a v2 frame binding, and vice versa. The existing anchor does this work; no new mechanism needed. |
| blockers | `label_atr_contemporaneous_causality` absent on v3; `label_intrabar_order_assumption` present on both |
| loader | must **reject a mixed-version dataset** (already rejects heterogeneous `label_version` via `_HEADER_FIELDS`) |
| consumers | no signature change; they read whatever version the sidecar declares |
| Phase 4 | `deployment_eligible` stays `false`. v3 removes one blocker; the others (independent holdout, cost/fill, accounting, prospective execution) are untouched. |

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
5. **Blocker split.** v3 records carry `label_intrabar_order_assumption` and NOT
   `label_atr_contemporaneous_causality`; a v3 record missing the intrabar blocker is rejected.
6. **Version isolation.** A dataset mixing v2 and v3 records fails to load. A v3 dataset fails to
   validate against a v2 frame binding.
7. **B2 is conservative.** For every record where the two orderings disagree, v3 (B2) PnL is
   strictly less than the B1 value. A property test over generated bars, not one example.
8. **No silent regeneration.** The probe writes only under its own output directory; assert the
   Phase 2 directory's contents are unchanged after a run.
9. **Equivalence where it should hold.** On a frame with no trail exits at all, v2 and v3 labels
   are bit-identical. This is the non-vacuity guard: if it fails, the change is broader than
   claimed; if the suite has no such case, the other tests cannot localise the change.

---

## 8. What this proposal does not do

- It does **not** clear the causality blocker. It repairs one half and makes the other half
  permanent and explicit (§2).
- It does **not** touch thresholds, retrain anything, regenerate any live artifact, or alter v2
  label semantics.
- It does **not** make any profitability claim, and it does not make Phase 4 deployable.
- It does **not** establish that any archived verdict was wrong.

## 9. Open questions for the operator

1. **B2 versus B1** as the published basis (§4). B2 is the conservative choice and my
   recommendation; B1 preserves comparability with v2 exit kinds. This is a modelling decision,
   not a correctness one, so it is yours.
2. Whether the probe should run over the full universe or a named subset first.
3. Whether v3 should be built at all before the other Phase 4 blockers are addressed, given that
   removing one of four does not change `deployment_eligible`.
