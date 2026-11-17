# Strategy evaluation and promotion evidence

## Current status

No winning strategy established. This change repairs research validation boundaries; it does not retrain models, regenerate historical performance, implement a complete evaluation runner or authorize trading.

The pre-fix Phase 3 splitter trained on future rows. Every result produced through it needs a new run with preserved input hashes and a corrected validation version, including prior ABORT verdicts. Preserve original artifacts as invalidated historical records. Do not overwrite production artifacts to make an old report appear repaired.

## Chronological contract implemented here

- Sort complete hourly rows by timestamp before tensor creation; reject missing/noninteger timestamps, duplicate timestamps and spacing shorter than an hour. Gaps and omitted unlabeled rows make the row-count embargo conservative.
- Reserve the first block and division remainder for warmup; use five equal outer test blocks and expanding training prefixes. Inner selection uses the same rule with three test blocks.
- Exclude the label horizon immediately before each test. No future training, empty test set or empty training set. The split utility omits unusable folds, but mining requires all five outer folds and three inner folds per outer fold, each with at least max(horizon, twice the smallest min_leaf grid) training rows. Reject incomplete history before fitting and log an explicit insufficient-history diagnostic; never lower the four-of-five pass threshold. Emitted profiles record n_folds_evaluated=5 and validation_version=chronological_v1. New parquet rows use schema 2. Legacy/default profiles remain legacy_unverified; existing rows are not relabeled.
- Features and labels retain the same chronological row identity. The tree fitter currently uses local IDs against full label arrays, so all training slices must remain prefixes.
- These are walk-forward splits, not CPCV. This agrees with the earlier-training/later-testing and gap semantics documented by [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

## Research-only output contract implemented here

Legacy filenames deployment_nN.json remain for compatibility. Their content explicitly records evaluation_scope=research_selection, deployment_eligible=false, deployment_blockers, and gates.scope=research_only. The CLI success code means the research screen passed; it never grants deployment approval. Existing numeric gates are preserved as research heuristics, not validated risk limits.

Both Markdown and JSON report missing evidence. There is deliberately no flag that turns deployment_eligible on without implementing and reviewing the evidence checks below. Old files are not rewritten automatically. Any future execution consumer must reject absent/false eligibility and independently verify a frozen evaluation manifest.

## Required next implementation and experiment

1. Enforce chronological validation metadata and fold counts at profile loading, with counted legacy exclusions. Repair all-fold reporting, horizon-qualified identities and funded-equity accounting. Aggregate losing as well as winning folds. Keep policy/threshold changes explicit instead of describing changing tree leaves as one fixed strategy. Resolve execution findings 1-3 and account drift before trusting paper/live comparisons.
2. Freeze a manifest before running a final holdout: data snapshot/hash and cutoff, completed-bar availability times, label version and matching prediction target, model/code hash, selection grid and all trials, universe, sizing, risk limits, fee scenario, and train/validation/holdout intervals. Choose the holdout without inspecting its results. If it has already been inspected, reserve later unseen data instead.
3. Use chronological nested selection with horizon purging only on development data. Freeze the entire selection policy, including portfolio cap and rules, before the final holdout. Do not reuse holdout results to tune thresholds and call the same period untouched.
4. Compare on identical capital and timing: cash/no trade; BTC buy-and-hold; a predeclared low-turnover trend rule; the frozen current XGB policy. No candidate gets a different cost model or favorable start date. Match calibration metrics to the actual trained target; endpoint-return labels cannot calibrate triple-barrier probabilities.
5. Simulate causal signals, next-available execution prices, spread/slippage, latency, rejected/missed/partial fills, capital constraints and both entry/exit fees. Record account-specific maker/taker fees; the existing 0.6% per side assumption is a scenario, not a verified account rate. [Coinbase describes fee tiers and maker/taker pricing](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/advanced-trade-fees). Stress costs and turnover explicitly.
6. Report net equity/P&L, peak-to-trough funded drawdown, exposure, turnover, trade count, fill/reconciliation rates and results by time period. Use time-block uncertainty estimates that respect overlapping returns; account for strategy selection. A point estimate or IID bootstrap alone is insufficient.
7. Predeclare acceptance thresholds before viewing holdout outcomes: sufficient independent observations, positive after-cost incremental performance versus baselines with uncertainty accounted for, acceptable funded drawdown, and no unexplained accounting/exposure mismatch. Insufficient evidence is inconclusive, not a winning result. Do not weaken gates after failure.
8. Run a prospective shadow with isolated storage and no live credentials/orders. Compare intended orders with observed market availability and reconcile every signal/order/fill/position transition. Only a separately reviewed evidence bundle can support a later operator promotion decision.

## Open execution contract

Maker cancellation is not a fill. Persist a stable intent ID before submission; preserve exchange IDs and unknown outcomes; do not retry ambiguous placement. Track pending, confirmed filled, partial, rejected and reconciliation-required transitions. Position reductions and realized P&L must follow confirmed quantities/fees, with idempotent fill events. A strategy exit signal cannot itself close a live position. Claude is preparing the detailed implementation contract; no live lifecycle migration is part of this validation patch.
