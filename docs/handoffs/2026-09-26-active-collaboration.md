# Active Codex / Claude repair cycle

Base: c7249dc on outcome-label prerequisites 7bd327c. No production changes.
The session-link CLI is the shared mailbox; receipt requires an explicit ACK.

## Ordered issue register

TaskCreate is unavailable; this register records the required find-list-fix sequence.

1. Completed (17 tests pass): purged_wf.py includes future training rows. Replace with expanding chronological prefixes, a warmup block and purged boundaries; test inner/outer chronology and insufficient history.
2. Completed (14 miner tests pass): mine_profiles.py trusts parquet ordering and can fit empty inner folds. Validate hourly timestamps and duplicates, sort before tensors, skip insufficient folds, preserve label/feature index alignment.
3. Completed (11 scorecard/driver tests pass): scorecard.py / build_phase4.py call reused research data a deployment verdict. Emit research-only status and deployment blockers in Markdown and JSON.
4. Deferred: miner combines only passing-fold trades and direction-only identities across changing thresholds. Metrics need all-fold reporting and a frozen-policy holdout.
5. Implemented, full suite pending: loaded profile identities now include horizon; simulator rule lookups use the same complete ID. Regression tests reproduced wrong labels, exit horizons and overwritten entry rules under input-order changes. All 35 targeted loader/simulator/selection/driver tests pass. Existing per-horizon sidecar keys remain compatible.
6. Deferred: additive return drawdown is not funded portfolio drawdown. Replace with a capital/position ledger before treating the 30% gate as risk evidence.

7. Completed, targeted tests pass (Claude review): a variable count of usable folds makes the four-pass gate ambiguous and permits tiny training sets. Require five evaluable outer folds, three inner folds each, and minimum training rows max(horizon, twice the smallest min_leaf). Record evaluated fold count/version on profiles; report insufficient history explicitly.

8. Completed, round-trip tests pass: parquet schema assertion exposed newly added validation metadata. Version new profile rows as schema 2 and round-trip both causal and legacy-unverified defaults. Existing rows are not relabeled.

9. Completed, 18 targeted and 1501 full-suite tests passed: profile_loader accepted missing/legacy validation metadata. Enforce schema 2, chronological_v1 and five evaluated folds with counted exclusions before treating archived profiles as current research inputs. Reject malformed or impossible pass counts without integer truncation. Phase 4 remains deployment-blocked regardless.

## Coordination decisions

- Claude acknowledged c7249dc supersedes maker PR #59, including confirmed empty-cancel, missing-fill, SELL sizing and pending-cancel defects. PR #59 must not be merged in its present form. Claude subsequently reported PR #59 closed by operator authorization; preserved branch for provenance. No PR closed by Codex.
- Claude owns executor lifecycle consumers in main.py, exit_watcher.py and, if needed, cnn_agent.py. Codex owns validation. Shared CHANGELOG/CLAUDE released to Codex.
- Late binding must resolve the current executor once per operation. Enabling can still reset halt and balance state; separate unresolved defect unless fixed and tested.
- Findings 2/3 require an execution-confirmed position lifecycle; remain open.
- No merges, live orders, restarts, production migrations, retraining or threshold changes.

## Validation evidence

TDD: split regressions first failed 16/17, then passed 17/17. Miner regressions reproduced ordering and empty-inner failures; all 14 miner tests passed after repair. Scorecard regressions first failed 3 tests; all 11 scorecard/driver tests passed after repair. The complete strategy-discovery suite passed 103 tests. Initial full-suite pre-commit passed: 1437 passed, 65 skipped, 1 deselected, 1 xfailed, 2 xpassed (380.82s).

Archived Phase 3/4 verdicts produced with future training rows are invalid validation evidence, including ABORT. Preserve originals; do not interpret as proof for or against a strategy.


## Claude lifecycle design review

Claude reports both scan and WS resolver paths implemented; commit/tests pending independent review. His proposed next contract was reviewed: add persisted intent before submit, separate order status from position exposure, never erase a held position on exit rejection, distinguish active partial orders from terminal partial fills, reserve CLOSED for flat exposure, use idempotent fill IDs and product-precision quantities, and prohibit unknown-order retries. Asked Claude to document and test the pure validator without live integration.


## Review evidence and delivery

- Maker/label draft PR: https://github.com/gl4500/coinbase-ai-trader/pull/60 (c7249dc), stacked on prerequisite PR #58. No merge or deployment.
- Claude executor patch: 7d9913f on fix/executor-lifecycle-late-binding. Codex source review found no blocker for the narrow stale-reference repair; independent lifecycle + exit-watcher tests passed 29/29. Tests simulate replacement, not actual enable/disable endpoints. Risk-state reset and exit accounting remain open.
- Claude caught variable-fold comparability and tiny-history issues in the first causal patch. The follow-up requires complete 5x3 evaluation and minimum training rows. Three new regressions failed before the change. The synthetic positive-cohort fixture was lengthened from 1000 to 2000 rows to meet the new warmup precondition; profit criteria were unchanged.
- Additional source defect registered with Claude: manual execute_market_order accepts a missing exchange success/order ID, and cancel_order marks canceled without verifying per-order success. These are separate from the repaired maker path and remain pending a shared execution adapter.


Follow-up validation: all 17 miner tests passed, including the unchanged positive-cohort profitability assertions on adequate history. The parquet metadata test then exposed its obsolete schema expectation; new rows now use schema 2 and round-trip tests verify causal metadata versus legacy-unverified defaults. Final targeted check: 19 passed, one already-passed expensive cohort test deselected. Final full pre-commit suite passed: 1440 passed, 65 skipped, 1 deselected, 1 xfailed, 2 xpassed; 14 existing sklearn warnings; 425.30 seconds. Implementation commits f301298 and 06f376e. Scoped Ruff and whitespace checks pass. No tests were bypassed.


## Review closure

Claude reviewed PR #60 with no blockers: https://github.com/gl4500/coinbase-ai-trader/pull/60#issuecomment-5849379672 . GitHub checks on that commit passed. Legacy rows are excluded in both query and mutation guards, which also prevents old pending rows consuming the resolver queue indefinitely. An explicit legacy-pending diagnostic remains an optional follow-up.

PR #61 contains Claude's independently based executor patch. Its full-suite result is Claude-reported (1315 passed); Codex independently verified the 29 lifecycle/exit tests. No integration merge was performed.

The dollar-bar worktree uses bar-count horizons and requires its own integration review; hourly timestamp spacing validation must not be copied into it blindly. The agreed position/order lifecycle spec and pure validator are Claude's next local deliverables, not completed components of this patch.


## Resumed dialogue after operator status check

Claude delivered validator PR #63 at 3bb8aa1. Independent pure-function probes reproduced six evidence defects: cancellation accepted OPEN/negative fills; numeric order IDs accepted; partial settlement without terminal proof; failed entry erased known exposure; infinite increment made held exposure flat; string false cleared reconciliation. Sent as blockers for Claude to fix before the false-success manual order/cancel repairs. No production calls involved.

Root is integrating upstream 3afccfc into #60/#62 and enforcing validation metadata at profile loading. Claude owns validator and executor fixes; root does not edit those files. Full test runs are coordinated to avoid contention. Promotion remains blocked.

PR #60 integration is now 634bf8f, pushed after 1464 full-suite tests passed
(65 skipped, one deselected, one xfailed, two xpassed). PR #62 cleanly merged
that parent with its loader fix as e472c2e, pushed after 1501 tests passed
(65 skipped, one deselected, one xfailed, two xpassed; 14 existing warnings,
371.54 seconds). Scoped Ruff and whitespace checks passed. Draft PR #62
reflects the implementation; CI on the new head is pending.

## Regular session checks

At the operator's request, both sessions now run 30-second mailbox watchers.
Codex uses the installed native `codex queue` command against this existing
thread, one outstanding challenge maximum, next cycle 120 seconds after the
assistant receipt. A missing receipt after 180 seconds is stale/busy, not proof
of process failure. Runtime status is `.coordination/ping-status.json`; separate
receipt records prevent process liveness from masquerading as agent response.
The initial queued follow-up actually resumed Codex; Claude independently
replied to PING-ROUNDTRIP-1 after its background watcher notification. Claude's
reported acknowledgement took about 116 seconds, response creation about 129
seconds. No strict response latency is guaranteed during long tool calls.
Both clients must remain available and the computer awake. Stop instructions
are in `tools/session_bridge/README.md` in the main workspace.

Second validator review found four remaining evidence contradictions and sent
them to Claude: unidentified/nonzero-fill rejection from UNKNOWN, zero-fill
cancellation after a known partial, zero-fill partial settlement, and FILLED
despite explicit live remainder. These are unresolved review blockers until
Claude's follow-up is independently checked. No live execution was exercised.

Latest validator review: independently ran 101 targeted tests successfully,
then reproduced two remaining branch-specific bypasses (non-boolean remainder
flags and overfill in partial cancellation) plus inconsistent cancelled/full-fill
classification. Sent these to Claude with a request for one shared classifier
and cross-event regression coverage. PR #63 remains blocked on that review.
