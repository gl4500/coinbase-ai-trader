# What the evidence establishes, and what would change the answer

**Date:** 2026-09-27
**Status:** decision input, not a decision. No recommendation to stop or continue is made here.
**Scope:** polymarket_app, the live paper-trading backend on port 8001. Not `trading_app`.

## SUPERSEDING SECTION — added 2026-09-28 after eight measured reversals

Everything below this section was written before the corrections here. Where they conflict,
**this section controls.** Evidence is labelled with the tier Codex proposed, because the
single largest source of error in this session was treating one tier as another:

| tier | meaning |
|---|---|
| **M** mechanism verified | the code does what it is said to do |
| **P** predictive evidence | a statistical association measured out of sample |
| **E** economic evidence | positive net expectancy after realistic costs |
| **X** execution verified | attainable with recorded fills and fees |

**Nothing in this repository reaches tier E. Nothing reaches tier X.** `orders` has 0 rows.

### The conclusion

**On this universe, in this period, at these frictions, the correct action is to not trade.**
Every strategy family testable with existing data converges on that:

| family | result |
|---|---|
| entry picking on the score | no edge — 45.0% up at +4h vs 43.9% for rejected candidates [P] |
| bracket exits, 111 configurations | best +0.210% before spread; negative after [P] |
| horizon variation 24/72/168/336/720h | apparent gains were 47–80% **timer** exits, not price exits [M] |
| liquidity restriction | selected stablecoins; one product carried 49 clusters [M] |
| veto / state filter on holdings | degenerates to cash — median time held **0.0%** [P] |

This is Codex's abstain-first rule reached empirically: when the lower confidence bound on net
expected value is never positive, HOLD *is* the answer. The system's defect is not that it
picks badly; it is that it trades at all.

### Corrections to the numbers below

1. **"−$95.77" is GROSS and PAPER.** `cnn_agent.py:322` computes `pnl = proceeds − size ×
   avg_price` with no fee term, and `orders` is empty, so no cash was lost and net expectancy
   is **unmeasurable**, not merely unmeasured. An earlier draft of this session stated a
   "net −$958.34"; that was a fee *scenario* presented as a measurement and is withdrawn.
2. **Agent tag is not model provenance.** The 1,582-trade figure mixes model eras. Codex's
   v3-provenance cohort (319 closed trades since 2026-05-23 with `xgb_prob == model_prob`) is
   **−$5.54 gross** — effectively flat.
3. **Criterion 1 below is mis-framed.** It centres the mining harness. The binding constraints
   are frictions and instrument scaling, neither of which is model work.

### Findings established 2026-09-28, all [M] unless noted

- **Levels were never scaled to instruments.** ATR as a share of price ranges from 0.0096%
  (USDT-USD) to 11.95% (WAXL-USD) — a **1,244× spread**. The fixed 8% stop used throughout
  this session was a **0.7-ATR** stop on one instrument and an **833-ATR** stop on another.
  Any fixed-percentage rule measures which products happen to match the chosen number.
  *Diagnostic worth keeping: express every parameter in ATR units; if p5→p95 spans more than
  ~3×, it is not a parameter.*
- **The universe contains stablecoins.** USDT-USD, USD1-USD, DAI-USD, USDS-USD are scanned and
  scored above threshold. An 8% move in USDT is 833 ATR, so those positions can only ever exit
  on a timer — and because stablecoins have the tightest spreads, a "restrict to tight spread"
  filter selects *toward* them.
- **The max-hold cap is the primary exit, not a safety net.** Under an 8% stop / 12% trail it
  ended 80.2% of positions at 24h and 49.2% at 168h. Invariant #4 describes it as a safety net;
  that does not match its behaviour.
- **Retrospective feature reconstruction leaks.** `backend/services/tiered_history.py:48`
  filters `df["start"] < now_ts` — on bar **start** only — so a still-forming candle is
  admitted. Harmless live (the partial bar holds only past data); in replay from parquet the
  candle is complete, so up to 59 minutes of future high/low/close enters the features at a
  15-minute scan cadence. Every retrospective re-score built this way is invalid. Filter on
  bar **end**.
- **Scan scores are not independent observations.** `cnn_scans` holds 3,668 distinct
  `model_prob` values across 64,662 BUY rows; one value covers 9.5%. These are cache hits from
  `_cnn_prob`, so any per-scan n is inflated, and percentile slicing on the score selects by
  recency within ties.
- **Scanning is bursty.** Median product: 287 scans over only **6 distinct days** within a
  1,003-hour span, median signal coverage 34.9% at a 4h staleness limit. Scan volume fell from
  388k rows in May to 9k in September. A continuous state-filter policy is implementable on
  only ~48 of 351 products.
- **Frictions, measured as a scenario [P]:** spread/price across the 225 scanned products is
  median 0.1158%, mean 0.5366%, p90 1.1976%. A taker round trip crosses it twice and
  `USE_MAKER_EXECUTION` is default-off. Against the best measured gross edge (~1.4% per round
  trip over 7 days) total frictions are 1.43–2.27%. **This is a sensitivity scenario, not the
  historical sign** — the spread reading is a current snapshot and no fills exist.
- **The one effect that survived every correction — and it is a HYPOTHESIS, not a finding.**
  The score appears to separate *decline*: rejected candidates returned −5.2% to −7.5% at 168h
  against ~0% to +0.8% for gated ones, consistently across configurations, with controls far
  from zero. I had labelled this [P]. Codex downgraded it and I accept the downgrade: the
  comparison is **not** adjusted for product and regime composition, ATR/volatility, repeated
  overlapping windows, scan availability, or the inference-cache ties — and a product×time
  bootstrap **does not cure selection or target leakage**, only dependence. Establishing it
  requires testing within product and contemporaneous week, stratifying on causally-computed
  ATR%, using independent signal episodes with an overlap embargo, and reporting coverage and
  missingness. None of that has been done. Even if it held, it is not monetisable long-only:
  harvesting it needs a short leg the spot architecture does not have.

### Multiplicity, stated so it cannot be laundered

Over 120 configurations were evaluated against one sample. **That sample is discovery data
permanently** — no multiplicity correction converts a selected winner into a confirmation.
Confirmation requires chronologically later, uninspected data with at least a 168h purge at
the boundary and no interim tuning.

### Methodology resolutions agreed with the peer session, 2026-09-28

These answer the questions this document previously left open, and they are binding on any
follow-up work:

- **ATR is admissible as a scale variable only if computed exactly as production computes it,
  from fully closed bars, with formula, version and warm-up frozen.** Wilder ATR is recursive
  smoothing, not a clean 14-bar window; do not imply the latter.
- **ATR does not supply an eligibility floor, and no cutoff should be chosen from this
  dataset.** The stablecoin problem was discovered in these residuals, so any threshold picked
  now and evaluated here is post-hoc. Eligibility must instead come from **independent venue
  and product constraints** — the exchange's own product classification — declared before a new
  forward period is collected. This supersedes the ATR%-floor proposal made earlier in this
  document's drafting.
- **Further search on this sample is not confirmatory.** With >120 configurations tried and
  known target/data defects, the options are: freeze the current universe and stop, or freeze
  one fully specified policy and test it **once** on untouched forward data, always against a
  **no-trade comparator**.
- **Closed-bar filtering is mandatory for historical feature replay.** Restated here because it
  is the defect most likely to be reintroduced silently: the production filter reads correctly
  and only misbehaves in replay.

### The four blockers — no future result is trustworthy until these are fixed

1. No model hash, config, or `scan_id` on trade rows, so no number is attributable to a version.
2. The `tiered_history` bar-start filter above.
3. `purged_wf` places post-test rows in TRAIN, making every mining verdict uninformative.
4. Mining sidecars store rounded rule summaries, so no exact rule is reproducible.

### What remains genuinely untested, none of it model work

- **Maker execution measured rather than assumed** — the only lever with a large enough
  coefficient (breakeven hit rate 65.1% → 51.3%), unmeasurable today because no fills exist.
- Point-in-time quotes, so frictions become facts instead of snapshots.
- A different signal family.
- A short or market-neutral construction, which is architecturally unavailable on spot.

**Recommendation: the next work is measurement infrastructure, not strategy search.** Eight
conclusions inverted in one session -- endpoint target, deduplication, fee framing, a scenario
stated as a measurement, spread, timer artifacts, instrument selection, and the exit-versus-
entry attribution -- each time because a measurement tier was assumed rather than established.

---

## Why this document exists

Six investigations over this session and the ones before it went looking for an edge in a
different layer each time. Each came back empty, unmeasurable, or worth twenty dollars. The
purpose here is to stop a seventh from re-deriving the same six findings, and to write down in
advance what result would justify further investment — because a criterion chosen after seeing
the number is not a criterion.

**What this document is NOT.** It is not a claim that the strategy has no edge. Absence of
demonstrated edge is not proof of absence, and six probes on layers *we chose* do not cover
the space of layers. Every negative below is bounded by what it actually tested.

## The ledger, re-verified 2026-09-27

Agent-scoped and closed-only, from `trades` in `backend/coinbase.db` (read-only):

| agent | n | PnL | win % | first close | last close |
|---|---|---|---|---|---|
| CNN | 1,582 | **−95.77** *(gross, paper; see superseding section)* | 39.0 | 2026-04-12 | 2026-09-27 |
| TECH | 517 | −82.54 | 53.97 | 2026-04-12 | 2026-05-17 |

**Both separations are load-bearing.** TECH was retired 2026-05-17; pooling the two agents
mixes a live system with a dead one. Filtering `closed_at IS NOT NULL` matters too — 52
positions are open as of this writing and carry no realised PnL. An earlier pass in this
session reported exit attribution across both agents and had to be withdrawn in full, and
Codex correctly widened that: **any aggregate over `trades` without agent and era separation
is invalid**, not merely imprecise.

Two figures previously repeated in this session are corrected here. CNN's win rate is **39.0%**
— the 42.73% figure is the *all-agent* snapshot and was wrongly attributed to CNN. And the
line "100% of PnL is made in the exit layer by model-free rules" belongs to `trading_app`'s
HistoricalTrendsAgent, a different project; it does not describe this system.

### CNN by close trigger

| trigger | n | PnL | win % |
|---|---|---|---|
| `SCAN` (model's own SELL) | 750 | **+145.56** | 53.47 |
| `WS_TRAIL_STOP` | 286 | +49.69 | 40.91 |
| `MAX_HOLD` | 8 | +44.98 | 75.0 |
| `LEGACY_EXIT` | 1 | +13.89 | 100.0 |
| `WS_MODEL_DOWN` | 15 | +10.96 | 60.0 |
| `MODEL_DOWN` | 1 | +2.74 | 100.0 |
| `RECONCILE` / `STARTUP_CLEANUP` | 69 | 0.00 | — |
| `WS_STOP_LOSS` | 10 | −103.08 | 0.0 |
| `TRAIL_STOP` | 391 | −122.11 | 20.97 |
| `STOP_LOSS` | 51 | −138.40 | 0.0 |

Columns sum to exactly −95.77, matching the agent total — an internal check that could have
failed and did not.

**The shape worth noticing:** the largest positive column is the model's own discretionary
exit, not a model-free rule. The three loss columns are all risk exits, which is what risk
exits are for — they are where losses are *realised*, not where they are *caused*. Reading
them as the problem is the error that produced the withdrawn thesis below.

### Activity collapsed, and has not recovered

CNN closed trades per month: 526 (Apr), 802 (May), 116 (Jun), 58 (Jul), 65 (Aug), **15 (Sep,
+24.66)**. September is the only positive month, on n=15 — far too few to mean anything. Two
causes were established earlier: score-distribution compression against a fixed 0.60 gate, and
the CI filter over June/July.

## The six lines, and the limit of each

**1. XGB feature sweeps — AUC ceiling 0.5284.** Establishes that the feature families tried do
not separate outcomes materially better than chance. *Cannot rule out:* a feature class not
tried, a different label definition, or a different horizon. This is a ceiling **under the
features tried**, never a proof of no edge. (Provenance: recorded before tonight; not
re-verified here.)

**2. Live CNN agent — −95.77 over 1,582 closed trades.** Establishes that as configured and
actually run, the system has not made money over five and a half months. *Cannot rule out:*
that a configuration inside the same architecture would. Note also the suppression finding in
§"Correctness defects" — for roughly two months the system was not running the filter chain it
was configured to run, so this period is not a clean test of the configured design.

**3. Tier scoring — 23 of 24 products negative per-trade Sharpe, driving suspensions.**
Establishes the suspensions were consistent with the measured per-product record. *Cannot rule
out:* that per-trade Sharpe on these sample sizes is mostly noise, which would make the
suspensions arbitrary rather than protective. (Provenance: earlier session; not re-verified.)

**4. Phase 4 mining — verdict ABORT, and the verdict is UNINFORMATIVE.** This is the one most
often misread. `purged_wf` leaked post-test rows into TRAIN, so the ABORT is **not** evidence
that mining is safe or that the strategies are bad — it is evidence of nothing either way.
Codex escalated a second defect: `mine_universe.py` writes `rule_path_summary` (deliberately
threshold-rounded) into the sidecar, so **no exact rule exists anywhere in the pipeline** and
every Phase 4 trade set is approximate by construction.

**5. Exit counterfactual — inconclusive, every interval spans zero.** Replaying 58 of 61 live
CNN stop exits with the `STOP_LOSS` rung removed gave +3.49 pts mean at 168h, 95% CI
[−1.60, +8.35] under a cluster bootstrap over products. All six horizon/completeness cells
span zero. The raw figure was +5.88 before clustering, so **the effect shrank at every step of
added rigour** — numbers that behave that way were never findings. The real content is the
tail: BILL-USD realised −8.31% and reaches −66.20% without the stop. Direction is consistently
positive, which is mild evidence and nothing more. Every figure is also biased *toward*
holding, because `MODEL_DOWN` is not replayable from `trades`. The thesis that exit policy
caused the per-trade Sharpe that suspended 23 products is **withdrawn**.

**6. Stop overshoot — latency, not gaps, and worth about twenty dollars.** An 8% stop realises
mean −9.494% over 51 `STOP_LOSS` exits. Classifying each exit against its own bar's open
(n=53 scored of 61; 5 exits postdate their product's last stored bar, 3 products have no
history file): **88.7% `INTRABAR_CROSS` vs 11.3% `GAP_AT_OPEN`**. Of the mean overshoot past
the stop, −0.189 pts is unavoidable and −1.215 pts is attributable to the path. By path, scan
loop −1.365 vs tick path −0.229 — and that 1.14-point spread independently reproduces the
1.17-point gap between the two paths' realised means, which is the check that makes the
decomposition credible. **But in dollars the attributable column is an upper bound of −$19.98
against −$191.63 realised, 10%.** The largest percentage overshoots were small positions.
*Cannot rule out:* that a faster path is worth more in a future regime with larger positions.
But on this record, routing work is not justified on PnL grounds. **In points this reads as an
87%-recoverable finding; in dollars it is 10%. The framing decided the conclusion.**

## Correctness defects, which are true regardless of whether the strategy ever earns

**The configured filter chain was silently inert for two months.** Commit `589b571`
(2026-08-01, "style: ruff lint + format cleanup") deleted an import held solely for its
registration side effect. `backend/agents/cnn_agent.py` now reads, literally:

```python
try:
    pass  # registers ci into _FILTER_CLASSES
except Exception:
    pass
```

The comment survived; the import did not. With `MC_FILTERS=ci` still set, `_build_chain` finds
nothing, logs one warning, and `apply_buy_filters` permits every BUY. Telemetry agrees on the
date: populated through 2026-08-06 and on none after; 2026-08-11 alone shows 216 gate
crossings, 216 BUYs, 0 telemetry. Codex reproduced it independently in a fresh process.

**What that regression cannot tell us.** PID 19060, the process that traded in September, no
longer exists (port 8001 is now PID 17184). What that process actually loaded is therefore
**permanently unknown**, not merely unmeasured, and September behaviour must not be attributed
to this defect by inference from current source.

**Restoring the registration changes live trading.** On June/July rates it would block 78–88%
of candidates. It is therefore an operator decision, not a lint repair, and is recorded as a
strict `xfail` in `backend/tests/agents/mc/test_chain_health.py` so the defect is executable
and dated without either breaking CI or changing behaviour.

## What would justify continued investment

Written as falsifiable criteria, deliberately before the next probe runs:

1. **A feature or label class not yet tried** that lifts held-out AUC above ~0.55 on a purged,
   embargoed split — with the leakage defect in §4 fixed first, because the current harness
   cannot support any verdict.
2. **A frozen executable policy** (one exact rule, exit, sizing and cost model, fixed *before*
   the scored period) that is positive across folds. Per PR #68's spec: a direction group can
   have a measurement; only a frozen policy can have evidence. The same rule with different
   sizing does not inherit its evidence.
3. **A dollar-material effect**, stated in dollars. §6 is the cautionary case: a mechanism that
   is 87% of the percentage points and 10% of the money. Any future finding reports the money.
4. **A clean test of the configured design** — i.e. a period where the filter chain the system
   was configured to run was actually running. No such period exists after 2026-08-01.

## What would justify winding down

1. Criterion 1 fails **after** the leakage defect is repaired, so the negative is informative
   rather than uninformative.
2. Activity stays at September levels (n=15/month), at which point no result reaches
   significance in any reasonable horizon regardless of edge.
3. The correctness backlog is closed and the ledger is still negative on a clean period.

None of these is satisfied today. Criterion 1 in particular **cannot** be evaluated yet.

## Open decisions — operator's, not mine

1. **Fail-open, fail-closed, or fail-loud** when a configured filter cannot resolve. Invariant
   #14 forbids filter exceptions re-raising into the scan loop, which argues against fail-loud
   *inside* the loop, but says nothing about startup.
2. **Whether to restore the `ci` registration at all**, given it changes live trading.
3. **Merge order across #76–#79**, all still draft.
4. **The Snyk credential.** `SNYK_TOKEN` (last set 2026-08-09) now returns `SNYK-0005 / 401
   Unauthorized`, reproduced on rerun. Until it is fixed, CI red is not a signal about code.
5. **Whether the security gate should distinguish "scan errored" from "scan found something."**
   It currently reads `needs.snyk.result`, so a 401 is indistinguishable from a high-severity
   finding. Same defect class as everything else in this document.

## Provenance and limits of this document

Lines 2 and 6 and the ledger tables were verified by execution on 2026-09-27 against a
read-only copy of the live database, and line 6's classifier is committed with tests
(`b07a943`). Lines 1, 3, 4 and 5 are carried forward from earlier work in this session and its
predecessors and were **not** re-verified tonight; they are cited with that status rather than
as fresh measurements. The backend is **paper-trading only** — all PnL here is simulated and no
funds were at risk. `docs/` is the home for this kind of record in this repo; the per-trade
artifact behind §6 lives in the session scratchpad as `stop_overshoot_report.json`.

**The recurring defect class across every line above**, worth stating once: a partial or
errored answer presented as a real one. An uninformative ABORT read as safe. A confidence
interval spanning zero read as a direction. A percentage-point decomposition read as money. A
truncated mailbox read as a silent peer. A failed scanner read as a vulnerability. In each case
the fix was not more analysis but making the incompleteness visible.
