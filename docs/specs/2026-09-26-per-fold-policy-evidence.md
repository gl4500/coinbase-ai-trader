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
a different question than their readers ask. Three repairs landed on 2026-09-26 — chronological
folds, distinct-fold counting, and exact executable rules — and none of them changed that, because
none of them was about it.

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

**A direction group** is the set of leaves sharing a root split, accumulated across outer folds. It
has no executable form: its members are different rules, fitted on different training windows, and no
single predicate reproduces its trade set. A direction group can have a *measurement*. It cannot have
a *policy*.

**A frozen executable policy** is one exact machine rule together with everything needed to act on
it — entry condition, exit rule, holding period, sizing, and cost model — fixed and identified
**before** the period over which it is evaluated.

> **Only a frozen executable policy can produce out-of-sample evidence.** A direction-group aggregate
> is a summary of a search, and the search saw the data it is being scored on.

Every requirement below follows from that sentence.

---

## 2. A concrete counterexample

One product, horizon 24 h, five chronological outer folds `P1…P5`. The tree is refitted per fold and
the root split is `price_over_ema20 <= 1.02` in all five, so every left-side leaf shares one direction
group. In each of the first four folds a *different* sub-split qualifies:

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

Now evaluate **D alone**, as a frozen policy, across all five folds — the question a reader believes
has been answered:

| Fold | P1 | P2 | P3 | P4 | P5 | Total |
|---|---|---|---|---|---|---|
| D's return | −2.0 % | −1.0 % | 0.0 % | +1.0 % | −3.0 % | **−5.0 %** |

The artifact advertises **+16.0 %, passed 4 of 5 folds** for a policy whose only executable form
returns **−5.0 %** over the same span. No number in it is wrong. The aggregation simply answers
"did anything on this side of the split ever work?" while the reader hears "does this rule work?"

**Overlap makes it worse, and separately.** If A's and B's trades occupy the *same* bars on the same
product, the additive sum says +11 % while a one-unit capital account could hold only one position, or
half of each for +5.5 %. An additive per-trade sum can therefore exceed what any funded account could
have earned, with no error in any single trade's arithmetic.

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
evidence; an **untouched holdout** provides the evidence of record. The holdout is scored once per
policy identity. A second scoring of the same holdout after any change makes it a selection set, and
the report must say so rather than silently reusing it.

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

Maximum drawdown is the worst peak-to-trough of the ledger's equity curve. Drawdown computed on an
additive return series describes a curve that was never funded and understates the capital at risk
whenever positions overlap.

### R7 — The policy is identified by content
`blocker: policy_identity_unbound`

A policy identity is a digest over its exact machine rule, its exit and holding rule, its sizing rule,
and its cost model. The identity is recorded with the evidence, so the object evaluated is provably
the object later run. An ordinal, a name, or a position in a sorted list is not an identity: it is not
stable across runs. (This mirrors the exact-rule binding already being built; the addition here is
that **rule alone is not a policy** — the same rule with different sizing is a different policy and
must not inherit the rule's evidence.)

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

These repairs are prerequisites, not substitutes:

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
