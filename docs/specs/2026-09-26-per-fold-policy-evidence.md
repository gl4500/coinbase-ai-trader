# Per-fold policy evidence

**Status:** design input. Docs only — no code, no mining, no retraining, no archive edits, no
threshold changes, no deployment.

**What this document is.** A specification of the evidence a strategy profile must carry before its
reported performance may be read as out-of-sample evidence for a policy someone could run.

**What this document is not.** It makes no claim that any profile is profitable, that any profile
will become profitable, or that meeting these requirements would produce a deployable strategy. It
defines an evidence bar, not a result. Every requirement here is a way for a profile to *fail*
honestly rather than pass ambiguously.

---

## 0. Why this exists

The strategy-discovery pipeline reports numbers that are individually correct and collectively answer
a different question than their readers ask. Three repairs were implemented on 2026-09-26 —
chronological folds, distinct-fold counting, and exact executable rules — and none of them changed
that, because none of them was about it. All three are **unmerged draft PRs** (#62, #67, #69) at the
time of writing, so nothing described here is in `main`.

After those repairs, an emitted profile still says:

| Field | What a reader assumes | What it actually reports |
|---|---|---|
| `n_folds_passed_q0 = 4` | this rule qualified in 4 of 5 held-out periods | **some leaf on this root side** qualified in 4 of 5 periods |
| `cumulative_profit_raw` | what this rule earned | the pooled trades of **every qualifying leaf** in that direction group, across folds |
| `rule_path` / machine rule | the rule those numbers came from | **one representative leaf**, the last one written during mining |
| `max_dd` | worst peak-to-trough on funded capital | worst point of an **additive series of per-trade returns** |

So the profit figure, the rule, and the pass count describe three different objects. That is the
problem this specification addresses, and it is not fixed by making any one of them more precise.

---

## 1. The distinction that everything else rests on

**A direction group** is the set of leaves sharing the root split's **feature and direction**,
accumulated across outer folds. The root *threshold* may differ from fold to fold and the leaves still
group together, so the group is even looser than "same split" suggests. It has no executable form: its members are different rules, fitted on different training windows, and no
single predicate reproduces its trade set. A direction group can have a *measurement*. It cannot have
a *policy*.

**A frozen executable policy** is a fully specified decision process together with everything needed
to act on it — entry condition, exit rule, holding period, sizing, and cost model — fixed and
identified **before** the period over which it is evaluated.

The frozen object does **not** have to be a static predicate. A *predeclared algorithm* — "refit this
tree on a trailing window at this cadence, with these hyperparameters and this leaf-selection rule,
and trade the resulting leaf" — is equally a frozen policy, provided every input, parameter and
decision rule is fixed in advance and nothing is chosen with knowledge of the evaluation period. What
must be frozen is the **decision procedure**, not necessarily a single fixed rule. A static rule is
merely the simplest case, and requiring only static rules would rule out the retraining designs most
likely to be worth evaluating.

> **Only a frozen executable policy can produce out-of-sample evidence.** A direction-group aggregate
> is a summary of a search, and the search saw the data it is being scored on.

Every requirement below follows from that sentence.

---

## 2. A concrete counterexample

One product, horizon 24 h, five chronological outer folds `P1…P5`. The tree is refitted per fold and
the root split is on `price_over_ema20` in the same direction each time. Its threshold **may vary by
fold without preventing grouping**; it is held constant at `1.02` in the table below purely for
legibility. Either way every left-side leaf falls in one direction group. In each
of the first four folds a *different* sub-split qualifies:

| Fold | Qualifying leaf | Its exact rule | Return in that fold |
|---|---|---|---|
| P1 | A | `price_over_ema20 <= 1.02 AND vol_over_mc <= 0.05` | **+6.0 %** |
| P2 | B | `price_over_ema20 <= 1.02 AND vol_over_mc > 0.05` | **+5.0 %** |
| P3 | C | `price_over_ema20 <= 1.02 AND ret_24h <= 0.00` | **+4.0 %** |
| P4 | D | `price_over_ema20 <= 1.02 AND ret_24h > 0.00` | **+1.0 %** |
| P5 | *(none qualified)* | — | — |

What is emitted today, all of it correctly computed:

- `n_folds_passed_q0 = 4` — correct under distinct-fold counting: four separate periods each
  contained a qualifying leaf.
- `cumulative_profit_raw = +16.0 %` — the pooled trades of A, B, C and D.
- the machine rule of **D**, the last leaf written into the fold summary.

Now ask what **D alone** would have done across all five folds. **This is not out-of-sample
evidence for D, and the reason is specific:** D was *selected using* P4, so the selection procedure had
already seen the outcomes it is now being scored against. The disqualifying fact is the **access**, not
the calendar — see R3 on why a sealed historical holdout can be genuine out-of-sample evidence even
though its bars predate the freeze.

The figures below are therefore a **contaminated retrospective evaluation**. They remain decisive for
the point at hand — they show the advertised aggregate is **not attributable to the rule the artifact
names** — but they are evidence neither for nor against D:

| Fold | P1 | P2 | P3 | P4 | P5 | Total |
|---|---|---|---|---|---|---|
| D's return | −2.0 % | −1.0 % | 0.0 % | +1.0 % | −3.0 % | **−5.0 %** |

The artifact advertises **+16.0 %, passed 4 of 5 folds** for a policy whose only executable form
returns **−5.0 %** over the same span. No number in it is wrong. The aggregation simply answers
"did anything on this side of the split ever work?" while the reader hears "does this rule work?"

**Overlap is a separate defect, and it needs a separate illustration.** It cannot be shown with A and
B above: those are `vol_over_mc <= 0.05` and `vol_over_mc > 0.05`, mutually exclusive predicates drawn
from different folds, so they can never occupy the same bar. Overlap requires two rules that can be
*concurrently active*.

Take two hypothetical profiles on one product, from different folds and therefore different trees,
whose predicates are not mutually exclusive:

- **X** — `price_over_ema20 <= 1.02 AND vol_over_mc <= 0.05`
- **Y** — `rsi_14 <= 30 AND ret_24h > 0.00`

Assume **unlevered unit capital**: one account, one unit, no borrowing. On a stretch of bars where
both predicates hold, each profile records its own trade — say **+6 %** and **+5 %**. Summed additively
that reads **+11 %**. But one unit of capital cannot fund both positions: it funds one, or half of
each for **+5.5 %**.

So an additive per-trade sum can report more than any unlevered account could have earned, with no
error in any single trade's arithmetic. The overstatement is a property of the *summation*, not of the
trades — which is why R4 requires a ledger rather than better per-trade accounting.

---

## 3. Requirements

Each requirement is stated so that it can fail, and carries a blocker name for the report.

### R1 — Every fold is reported, including the ones that failed
`blocker: per_fold_reporting_incomplete`

Report one row per outer fold, for **all** folds, each carrying: trade count, gross return, costs, net
return, funded capital, and whether that fold qualified. Today only qualifying leaves contribute
trades, so the aggregate is conditioned on selection — a survivor sum. A fold in which the policy lost
money is evidence, not an omission.

### R2 — The aggregate is over the frozen policy, not the search
`blocker: aggregate_is_group_not_policy`

The headline figure must be the frozen policy's result over all evaluated folds. A direction-group
aggregate may still be reported, but must be labelled as a property of the search and must never
appear as the policy's performance.

### R3 — Selection data and evaluation data are disjoint, and the holdout is used once
`blocker: selection_contaminates_evaluation`

Three levels, explicitly separated: inner CV chooses hyperparameters; outer folds provide selection
evidence; an **untouched holdout** provides the evidence of record.

**Holdout access is accounted at the campaign level, not per policy identity.** An earlier draft of
this requirement said "scored once per policy identity", which is self-defeating: since R7 makes
identity a content digest, any change of sizing or cost model mints a *fresh* identity for free, so an
unlimited number of candidates could each claim a first look at the same holdout. That is the
multiple-testing leak this requirement exists to prevent, reintroduced by the granularity of the rule
itself.

Instead: a holdout window is **reserved** for a research campaign, and every candidate evaluated
against it is logged — including the ones discarded. The report states how many candidates have
touched that window, so a result can be read against the number of attempts that produced it. Any
scoring after the first, by any identity, is a **selection** use and the report must say so. A new
identity does not buy a fresh holdout; a new **data window** does.

**Three distinct clocks, and the rule is about information access — not about the calendar.**

| Clock | What it is |
|---|---|
| **wall-clock freeze time** | when the policy was fixed and its identity recorded |
| **simulated information cutoff** | the latest data the policy may consult when making each decision |
| **holdout access time** | when the researcher first saw the holdout's *outcomes* |

Out-of-sample status is determined by the third, not the first. **A sealed historical holdout may be
evaluated after the policy is frozen and still yield genuine out-of-sample evidence, even though its
bars predate the freeze** — what matters is that nobody consulted its outcomes while selecting. An
earlier draft of this document got this wrong, declaring all retrospective evaluation a counterfactual;
that rule would have forbidden the one honest use of archived data.

So the prohibition is stated as access: a policy's evidence is void over any period whose outcomes
informed its selection. D in §2 fails on exactly that ground — it was selected using P4 — and not
because P1–P3 precede its freeze.

**A deterministic rerun is not a new attempt.** Re-evaluating an unchanged policy on unchanged data to
reproduce a result consumes no additional selection budget. Log reruns separately from candidate
evaluations, so the campaign counter measures attempts to *find* something rather than attempts to
*verify* it. A rerun that produces a different number is a reproducibility failure, which is its own
finding.

### R4 — Capital is a ledger, and concurrent exposure is explicit
`blocker: additive_returns_not_funded`

Returns are computed from a position/capital ledger, never by summing per-trade percentages. For each
product, concurrent open positions must be represented explicitly; overlapping trades are either
merged into one position or constrained by available capital. The report states peak concurrent
exposure per product and in aggregate.

### R5 — Costs are applied at the funded size, against a comparable baseline
`blocker: costs_or_baseline_missing`

Fees, spread or slippage, and any funding cost are applied at the size the ledger actually funds. A
baseline — at minimum buy-and-hold on the same capital, over the same periods, under the same cost
model — is reported alongside. A return without a baseline on identical capital and costs is not a
comparison, and "beat zero" is not a finding.

### R6 — Drawdown comes from the funded equity curve
`blocker: drawdown_not_equity_based`

Maximum drawdown is the worst peak-to-trough of the ledger's equity curve, and that curve must
**mark open positions to market** — a drawdown computed from closed trades alone cannot see an
unrealised loss on a position still held, which is exactly when capital is most at risk.

Drawdown computed on an additive return series describes a curve that was never funded, and it can
distort risk in **either direction**: it understates when overlapping positions demand more capital
than the series assumes, and it can overstate when sequentially-scaled returns are summed as though
each were taken on the full account. Neither error is safe to carry.

### R7 — The policy is identified by content
`blocker: policy_identity_unbound`

A policy identity is a digest over the frozen decision procedure and everything needed to act on it:
for a static policy, its exact machine rule; for a **predeclared adaptive procedure** (§1), the fitting
and selection configuration — trailing-window definition, refit cadence, hyperparameters, and
leaf-selection rule — since for such a policy *that configuration is the rule*. In both cases the
digest also covers the exit and holding rule, the sizing rule, and the cost model. An ordinal, a name, or a position in a sorted list is not an identity: it is not
stable across runs. **Rule alone is not a policy** — the same rule with different sizing is a
different policy and must not inherit the rule's evidence.

**A digest proves content integrity, not execution.** It establishes that the description has not
changed since it was recorded; it says nothing about whether the code that ran implements that
description. Claiming otherwise would repeat the accepted-versus-confirmed error in a new place.

The evidence must therefore also carry a **manifest**, and the report is not complete without it:

| Manifest field | Why |
|---|---|
| executable implementation + config identity | the description is not the executor; a config change is a different policy |
| feature schema, ordered | a rule naming a column means nothing without the schema that defines it |
| evaluation data identity and time boundaries | fixes *what* was evaluated and *over which span*, so a later re-run is comparable |
| freeze time, information cutoff, holdout access log | makes R3's three clocks checkable rather than asserted — in particular that no evaluated period's outcomes informed selection |

The manifest is integrity evidence too, not attestation: it records what was claimed to run, and an
independent execution record remains a separate requirement this specification does not satisfy.

### R8 — Reporting fails closed
`blocker: report_incomplete_fail_closed`

If any required per-fold row, cost input, or identity field is missing, the report emits **no
aggregate at all** — not an aggregate with a caveat. A number with a footnote gets quoted without the
footnote. This is the same rule the execution layer now follows: absence of evidence must not default
to the permissive reading.

### R9 — Migration excludes, and never relabels
`blocker: legacy_evidence_excluded`

Reports gain a version. Artifacts predating it are **excluded** from evidence cohorts with a
diagnostic and are never rewritten, because no post-hoc inspection can tell a genuine per-period pass
from a pooled one. Each requirement above keeps its own named blocker, so partial progress stays
visible and a report satisfying six of nine cannot be summarised as passing.

---

## 4. What a compliant report looks like

```
policy_identity   : <digest>          # rule + exit + sizing + costs
report_version    : per_fold_v1
deployment_eligible: false
blockers          : [ ... any unmet requirement, by name ... ]

per_fold:
  P1  trades=..  gross=..  costs=..  net=..  capital=..  qualified=yes
  P2  ...                                            qualified=no      # reported, not dropped
  ...
frozen_policy_total : net=..  over folds P1..P5
group_aggregate     : net=..  LABEL: property of the search, not of the policy
baseline            : buy_and_hold  net=..  same capital, same costs
equity_drawdown     : ..  # from the ledger curve
peak_concurrent_exposure: per_pid={..}  aggregate=..
holdout             : scored=<once|never>  at=<identity>
```

`deployment_eligible` stays `false` while any blocker is present, consistent with the research-only
gating already in place. Nothing in this specification is a route to setting it `true`; that requires
prospective execution evidence, which by definition cannot come from a backtest.

---

## 5. Relationship to the 2026-09-26 repairs

These repairs are prerequisites, not substitutes — and all are unmerged drafts:

- **Chronological folds** removed leakage from fold construction. Necessary before any per-fold number
  means anything.
- **Distinct-fold counting** made the pass count count *periods*. It fixed the arithmetic; §1 is about
  what is being counted.
- **Exact executable rules** made the rule reproducible. Necessary for R2 and R7 — a frozen policy
  cannot be frozen around a rounded display string.
- **Provenance versioning** established the pattern R9 follows: exclude, diagnose, never relabel.

None of them licenses a profitability claim, and prior outputs — including `ABORT` verdicts — remain
uninformative rather than conservatively safe, because they were produced by builders now known to be
wrong in ways that affect the verdict in both directions.

## 6. Open items this specification does not resolve

Stated so they are not mistaken for covered:

- **Direction-group aggregate metrics vs representative rule** — this spec says the two must be
  separated and labelled; it does not prescribe which representative to choose, or whether a direction
  group should emit a policy at all.
- **Predeclared policy selection** — how a single policy is chosen before evaluation, rather than
  after, is a separate design.
- **Overlapping leaf exposure within a product** — R4 requires it be explicit and capital-constrained;
  the portfolio construction that resolves competing signals is out of scope.
- **Whether any surviving policy has economic value** — unaddressed and unaddressable here.
