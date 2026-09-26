# Active Codex / Claude repair cycle

Base: c7249dc on outcome-label prerequisites 7bd327c. No production changes.
The session-link CLI is the shared mailbox; receipt requires an explicit ACK.

## Ordered issue register

TaskCreate is unavailable; this register records the required find-list-fix sequence.

1. Completed (17 tests pass): purged_wf.py includes future training rows. Replace with expanding chronological prefixes, a warmup block and purged boundaries; test inner/outer chronology and insufficient history.
2. Completed (14 miner tests pass): mine_profiles.py trusts parquet ordering and can fit empty inner folds. Validate hourly timestamps and duplicates, sort before tensors, skip insufficient folds, preserve label/feature index alignment.
3. Completed (11 scorecard/driver tests pass): scorecard.py / build_phase4.py call reused research data a deployment verdict. Emit research-only status and deployment blockers in Markdown and JSON.
4. Deferred: miner combines only passing-fold trades and direction-only identities across changing thresholds. Metrics need all-fold reporting and a frozen-policy holdout.
5. Deferred: profile identities omit horizon; mixed-horizon simulation may collide. Require horizon-qualified identities before multi-horizon evaluation.
6. Deferred: additive return drawdown is not funded portfolio drawdown. Replace with a capital/position ledger before treating the 30% gate as risk evidence.

## Coordination decisions

- Claude acknowledged c7249dc supersedes maker PR #59, including confirmed empty-cancel, missing-fill, SELL sizing and pending-cancel defects. PR #59 must not be merged in its present form. Claude subsequently reported PR #59 closed by operator authorization; preserved branch for provenance. No PR closed by Codex.
- Claude owns executor lifecycle consumers in main.py, exit_watcher.py and, if needed, cnn_agent.py. Codex owns validation. Shared CHANGELOG/CLAUDE released to Codex.
- Late binding must resolve the current executor once per operation. Enabling can still reset halt and balance state; separate unresolved defect unless fixed and tested.
- Findings 2/3 require an execution-confirmed position lifecycle; remain open.
- No merges, live orders, restarts, production migrations, retraining or threshold changes.

## Validation evidence

TDD: split regressions first failed 16/17, then passed 17/17. Miner regressions reproduced ordering and empty-inner failures; all 14 miner tests passed after repair. Scorecard regressions first failed 3 tests; all 11 scorecard/driver tests passed after repair. The complete strategy-discovery suite passed 103 tests. Full-suite pre-commit validation pending.

Archived Phase 3/4 verdicts produced with future training rows are invalid validation evidence, including ABORT. Preserve originals; do not interpret as proof for or against a strategy.


## Claude lifecycle design review

Claude reports both scan and WS resolver paths implemented; commit/tests pending independent review. His proposed next contract was reviewed: add persisted intent before submit, separate order status from position exposure, never erase a held position on exit rejection, distinguish active partial orders from terminal partial fills, reserve CLOSED for flat exposure, use idempotent fill IDs and product-precision quantities, and prohibit unknown-order retries. Asked Claude to document and test the pure validator without live integration.
