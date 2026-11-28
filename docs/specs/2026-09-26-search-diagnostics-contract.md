# Complete search diagnostics contract

Status: proposed implementation contract, not implemented evidence collection.
Baseline: main `cfc7d3b10f7bf3a461d8f2000f8fe004d057ddf6` (2026-11-23 source review),
including the endpoint integration in PRs #78/#79. Exact-rule reports from
[PR #70](https://github.com/gl4500/coinbase-ai-trader/pull/70) are merged (`a4084ef`).
This complements the proposed per-fold policy specification in open
[PR #68](https://github.com/gl4500/coinbase-ai-trader/pull/68); it does not satisfy its
policy-performance requirements. PR #71 targets main, not the former #70 branch.

## Purpose and limits

Preserve what the search evaluated before selection removes observations. The
artifact is `search_diagnostics_v1`, with `scope: search_diagnostics` and
`deployment_eligible: false`. It is not a frozen-policy backtest, a funded ledger,
or an independent holdout. No field may assert otherwise.

The current miner loses evidence at input/task creation, early input/history
returns, fold sufficiency checks, empty leaf skips, leaf qualification, and final
root-direction qualification. Recording only successful profiles cannot reveal
what was attempted or why nothing survived.

This contract adds an independent diagnostic output. Existing archived files
remain untouched. It does not change tree fitting, selection thresholds, model
training, live trading, or the meaning of the existing profile artifacts.

## Run and coverage records

Before input-file checks, declare the complete deduplicated requested product by
horizon cross-product. Materialize horizon iterables once. Reject invalid horizon
values rather than coercing booleans or fractions into identities.

The run manifest contains:

- Version, scope, unique run ID, research campaign ID, and producer commit/config.
- Ordered feature schema, label version if available, and input content identities.
  Unknown label provenance is explicitly unknown, never inferred from a filename.
- Requested pairs, each with a unique pair ID and its input location.
- Expected outer fold count (five), inner count (three), minimum training rows,
  embargo/holding assumptions, and qualification thresholds/config identity.
- Start/finish UTC times and lifecycle status: `running`, `complete`, `incomplete`.
- Access-accounting requirements and explicit unresolved evidence blockers.

Create one pair record for every requested pair, initially `pending`. Pair states
are `pending`, `running`, `completed`, `excluded`, or `error`. `excluded` and
`error` require structured reason codes and observed facts. At minimum cover:
missing input, missing label column, invalid schema/timestamps, insufficient labeled
rows, insufficient fold structure, and evaluation exception. Name at least
`missing_input`, `missing_label_column`, `invalid_timestamps`,
`missing_endpoint_publication`, `invalid_endpoint_publication`,
`insufficient_labeled_rows`, `insufficient_fold_structure`, and `evaluation_exception`.
`invalid_timestamps` covers missing/noninteger/null timestamps and non-contiguous,
duplicate or out-of-order source rows; do not sort or filter them to repair input.
`missing_endpoint_publication` identifies the absent endpoint sidecar separately from
a missing product parquet. `invalid_endpoint_publication` includes malformed sidecars,
missing/corrupted declared dataset content, and failed frame/config/record validation.
Unreadable paths and unexpected I/O remain errors with their exception evidence.
Assign reason codes from the originating check stage, never from the exception class
alone: preserve a stage marker/check identity at the boundary inside the pair call
alongside the exception type/message, since unrelated checks can raise the same type.

Today the miner raises `ValueError` for invalid timestamps and `FileNotFoundError`
for a missing publication sidecar; endpoint-loader validation exceptions propagate.
These abort the pair call rather than returning `[]`. By contrast, a missing label
column, fewer than 200 finite-label rows, or insufficient nested folds returns `[]`
when that check is reached. Preserve check order: publication validation precedes
the finite-label count, so an invalid publication cannot be relabeled as short history.
The producer capture boundary must record a terminal excluded/error disposition and
the observed exception type/message before preserving the existing exception or
run-failure behavior. Missing required publication and invalid publication are errors
for current endpoint mining, not successful empty evaluations. Do not swallow invalid
publication errors to continue a reduced universe with an apparently complete verdict.
Instrumentation may record evidence; changing exception propagation or legacy profile
output is outside this collection change. Record raw/labeled
row counts, actual outer count and per-outer inner counts when available. A failed
run must preserve pending/running pairs as incomplete; it cannot fabricate empty
successful results for them.

Coverage has two independent meanings: all requested pairs have recorded dispositions,
and all required folds have complete evaluation evidence. A fully accounted run
with excluded pairs is not fully evaluated. Consumers must report both explicitly.

Write immutable run-specific outputs. Persist the declared manifest before work;
atomically replace individual checkpoints and the final manifest. A crash may leave
an incomplete run, never an apparently complete file assembled from mixed runs.
Content digests detect alteration; they do not attest that code executed correctly.

## Fold and leaf records

For each evaluated outer fold record fold ID, sorted-frame index boundaries,
training/test timestamp boundaries with interval convention, actual inner boundaries,
chosen fitting parameters, tree digest, and declared fitted leaf IDs/count. Also
preserve that fold's `inner_cv_se`, `raw_max`, chosen depth and minimum leaf size,
not just the values surviving in an emitted profile. Also preserve the actual
candidate parameter grid searched, its measured count, every
candidate's score on each inner fold, and the aggregation/dispersion definition.
Record the candidate count actually passed to deflation separately; disagreement
with the searched grid is a named `deflation_search_count_mismatch` blocker.
The current literal nine matches today's grid but is not evidence of what ran.
Retaining the score matrix makes dispersion auditable; it does not establish that
the current cross-candidate standard deviation is a valid standard error.

Preserve seed, RNG algorithm/library version, bootstrap iteration count and ordered
group emission identities. Current bootstrap RNG state is shared sequentially
within one product/horizon call: inserting/removing a qualifying group can change
later groups' confidence intervals without changing their trade lists. Each call
creates its own RNG, so this is not cross-product coupling in the current driver.
Reproducibility identity includes that order and configuration, not just the seed.
Changing to identity-derived independent generators is a separate algorithm change.

Preserve actual row membership or a membership digest so dropped-label gaps are not hidden by
min/max timestamps. The contract does not prove label maturity from those boundaries;
that requires independent label provenance and causal-cutoff validation.

Record one row for every fitted leaf, before the empty-row skip or qualification
filter. Leaf identity is `(run_id, pair_id, outer_fold_id, source_leaf_id)`, bound
to the exact machine rule, source tree digest and ordered feature schema. Do not
substitute the cross-fold group ordinal for the source leaf ID.

Each leaf row carries routed-row count, replay trade count, descriptive return
metrics, qualification result/reasons, and root-feature/direction group key.
Record failed as well as successful leaves. A fitted zero-trade leaf is an observed
empty sample, not a missing record: count is zero, undefined sample statistics are
null, and qualification is false with `no_trades` reason. In particular, do not emit
zero win rate, average win/loss or Sortino for an empty sample. Conditional averages
with no winning or no losing observations are null even when other trades exist.
Use strict JSON: NaN and infinity must not escape as numeric literals. Record
undefined-metric reasons rather than pretending an undefined value is a zero.

Diagnostic null semantics are independent of the legacy qualification implementation.
Record the actual qualification decision and its configuration; do not silently
change existing gates as part of this collection repair.

Declare replay assumptions in machine-readable fields: the next-eligible-index
exclusion rule, its clock and units, label holding/exit clock and units, and actual
label/cost semantics. Record the eligibility source and endpoint publication identity
per pair: sidecar content digest, validated `data_id`, frame fingerprint, product,
horizon, and retained source-row membership. A filename alone is not that identity.

At main `cfc7d3b`, `mine_profiles_for_pid_horizon` validates unique, ascending,
contiguous hourly timestamps in the frame as read, then requires published endpoints
against the unfiltered frame in published order, before retaining finite labels.
`build_next_eligible_from_endpoints` delegates to `eligibility_boundaries`: its output
is a position in the retained working frame, derived from the published exit.
The miner does not import the legacy wall-clock builder and has no fallback to it.
`simulate_portfolio` requires validated endpoints and accounts for an exit at
`exit_observable_at`, not at entry plus the configured horizon. Causal purge and
embargo still use the configured horizon; diagnostics must not substitute observed
holding times for those validation boundaries.

**Legacy blocker: `replay_label_clock_mismatch`.** Pre-endpoint miners used
`build_next_eligible` with `horizon * 1h` timestamp arithmetic, while labels advanced
by source rows and could exit early. For the legacy example with hours
`[0, 2, 4, 6, 8]` and horizon 2, a row-count exit at row 2 is hour 4, while the
clock builder admits row 1 at hour 2. This is a legacy counterexample, not an input
accepted by the current miner: its contiguous-hourly check rejects that frame.
The old builder remains in `profit_split` for equivalence tests; its presence does
not establish use by the miner or current portfolio replay. Do not attach the old
early-PnL claim to endpoint-based Phase 4.

The validator must flag pre-endpoint artifacts without publication identity as
legacy/unverified with `replay_label_clock_mismatch`, not invent an identity or infer
that all such artifacts empirically overlapped. Missing or invalid publications on
a current run are separate errors, never a silent switch to legacy replay.
Endpoint alignment supports the declared eligibility and accounting semantics; it
does not establish causal labels, cross-leaf/cross-product funded-capital feasibility,
or trading profitability. Preserve those evidence limits. Unknown cost semantics
remain unknown; do not infer gross/net classification from filenames.

## Group dispositions

Record every root-feature/direction group observed in fitted trees, including groups
with no qualifying leaves. Thresholds can differ across folds within one group.
For each group record observed fold IDs, qualifying fold IDs, distinct passing count,
required passing count, emitted-profile identity if any, and rejection reasons.
Absent groups in a particular fold are marked absent, not as a fabricated leaf.
Group statistics remain search summaries. No representative leaf inherits the
aggregate as its own performance. Leaf records cannot be dropped because their
group failed the four-of-five gate.

The current `fold_summaries[direction_key]` is assigned for every qualifying leaf:
later passing leaves overwrite earlier ones, including within the same outer fold.
The emitted group's deflation, `inner_cv_se`, chosen depth/minimum leaf size and
representative rule/binding use the last assigned summary, although its trade list
aggregates qualifying leaves across folds. Preserve all per-fold fitting evidence
and each qualifying leaf's source identity before that overwrite. The group record
must name the exact source outer fold AND source leaf of the summary used by the
emitted profile, with its tree digest and fitting values. For rejected groups,
record the last summary source if one exists, with no emitted-profile identity;
if no leaf passed, that source is null. Do not replace the existing selection of
representative values with an average or a different fold as part of collection.

## Reading, accounting, and fail-closed behavior

A validator checks declared-pair coverage, unique identities, pair/fold/leaf
referential consistency, leaf counts, schema versions, rule bindings and content
digests before producing a validated diagnostic view. It distinguishes incomplete
coverage from a completed evaluation with no survivors. A missing sidecar is not
an empty successful run. Legacy artifacts are excluded with a named diagnostic;
no automatic migration invents missing observations.

Audit inspection and candidate selection are separate access purposes. Record both.
Before a supported selection consumer reveals results, append an access event with
campaign ID, artifact digest, declared candidate universe/identities examined,
selection procedure identity, purpose and time. Scanning all leaves counts all
examined candidates, not just the final winner. Raw diagnostics are selection data;
no later consumer can rename them an untouched holdout. Reproducibility reruns use
the same immutable inputs/procedure and are logged separately from new candidates.

Until this accounting consumer exists, declare selection consumption unsupported
and block supported selection entry points rather than silently omitting the log.
A scope string is not access control: manually reading a local file can bypass an
application log. Reports must disclose that enforcement boundary and cannot claim
complete access history without independent access controls. Audit collection does
not itself establish a numerical research budget or authorize further holdout use.

## Implementation slices and acceptance cases

1. Pure records and validator: incomplete/duplicate pair coverage; missing or duplicate
   fold/leaf IDs; explicit exclusions; zero-trade nulls; unknown provenance; corrupted
   rule/data bindings. No writer or miner changes in this first slice.
2. Producer capture: declare all requested pairs before file checks, preserve every
   early-return and raised-error disposition, record each fitted leaf before filters,
   and record rejected groups. Stage/check instrumentation inside the pair call is
   part of this slice; preserve existing profile-return and exception behavior.
   Tests stub fitting
   and use synthetic data; they must not mine real archives.
3. Run writer: atomic run-isolated persistence, interruption recovery, strict JSON,
   and no mixing of prior-run outputs. Test failures at each persistence boundary.
4. Read-only diagnostic reporting and accountable selection consumption: reject
   incomplete evidence where complete evaluation is required; report exclusions
   without policy-performance claims; require selection access events before use.

Acceptance fixtures include: missing product file, missing horizon label, too few
labeled rows, missing/noninteger/null timestamps, non-contiguous/duplicate/out-of-order
timestamps, a missing endpoint sidecar, and a publication that fails frame validation.
Also cover incomplete nested folds, a losing leaf, a zero-trade
leaf, several passing leaves in one fold, a group below threshold, all groups rejected,
an interrupted run, and a legacy gapped-timestamp counterexample where clock-hour
eligibility precedes the row-count exit. The current-miner fixture must reject that
frame, and a valid endpoint fixture must preserve endpoint-derived eligibility and
accounting provenance. Assert that losses and exclusions survive serialization and
that collection does not change which legacy profiles are emitted. Include passing
folds with distinct fitting values and multiple qualifying leaves in one fold;
assert preservation of every fold's values and the exact last-summary fold/leaf
used by the emitted profile, including its deflation inputs.
Include a timestamp `ValueError` and a sidecar `JSONDecodeError` receiving different
stage-derived codes, and a vanished product parquet receiving `missing_input`, never
`missing_endpoint_publication`. Assert that propagation remains unchanged in each case.

A completed implementation still requires a separately predeclared decision process,
causal feature/label validation, holdout access discipline, funded-equity replay,
matched-cost baselines and prospective execution evidence before trading conclusions.

## Source verification for this revision

Inspected at main `cfc7d3b` on 2026-11-23; these are source/assertion checks, not a new
execution of the referenced tests or evidence that the proposed collector exists.

- [Miner](../../backend/tools/strategy_discovery/mine_profiles.py):
  `mine_profiles_for_pid_horizon` validates timestamp order/spacing, calls
  `_require_published_endpoints` before filtering finite labels, and uses
  `build_next_eligible_from_endpoints`. Its qualifying-leaf loop overwrites
  `fold_summaries`; final profile construction reads that summary's fitting fields.
- [Endpoint adapter](../../backend/tools/strategy_discovery/endpoint_consumers.py):
  `load_validated_endpoints`, `eligibility_boundaries`, and `accounting_times`
  separate publication validation, retained positions, and observable exit instants.
- [Profit replay](../../backend/tools/strategy_discovery/profit_split.py):
  `build_next_eligible_from_endpoints` delegates to the adapter; the legacy
  `build_next_eligible` remains available for baseline comparisons.
- [Portfolio replay](../../backend/tools/strategy_discovery/portfolio_sim.py):
  `simulate_portfolio` requires validated endpoints and uses observable exit times.
- [Existing endpoint-miner regression assertions](../../backend/tests/tools/strategy_discovery/test_mine_profiles_endpoints.py)
  cover early exits differing from the clock, missing/invalid publications, full-frame
  validation, finite-label retention, and an interior filtered source row.
- [Existing adapter regression assertions](../../backend/tests/tools/strategy_discovery/test_endpoint_consumers.py)
  cover publication/frame binding, retained source-row identities and eligibility/
  accounting separation. These support the current semantic contract, not funded
  profitability or causal validity of ATR-derived labels.
