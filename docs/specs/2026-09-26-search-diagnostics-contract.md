# Complete search diagnostics contract

Status: proposed implementation contract, not implemented evidence collection.
Parent: exact-rule reports in draft PR #70. This complements the per-fold policy
specification in draft PR #68; it does not satisfy its policy-performance requirements.

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
rows, insufficient fold structure, and evaluation exception. Record raw/labeled
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
preserve the actual candidate parameter grid searched, its measured count, every
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
label/cost semantics. The current builder advances eligibility by `horizon * 1h`
in timestamp space, whereas dynamic labels advance by row count (capped at 168)
and may exit earlier at stops. These are different clocks, not verified trade exits.

**Confirmed blocker: `replay_label_clock_mismatch`.** On timestamps at hours
`[0, 2, 4, 6, 8]` with horizon 2, a label entered at row 0 can exit at row 2
(hour 4), while replay admits row 1 at hour 2. Two labelled trades then overlap
inside one leaf despite the max-one replay description. The miner accepts such
gaps because it requires spacing of at least one hour, not exactly one hour.
Dropping missing-label rows can introduce additional replay-index differences.

A synthetic reproduction using `_simulate_one`, `build_next_eligible`, and
`_replay_trades` with rising prices `[100, 101, 102, 103, 104]`, a 6% trailing floor,
8% stop and 1.2% round-trip fee admits both rows 0 and 1 with returns about
0.008 and 0.007802. No stops fire; the first label's exit is after the second entry. The Phase 4
portfolio simulator also realizes the first label at hour 2 and releases the slot,
two hours before its labelled exit. Trade counts and return statistics can therefore
include overlapping occupancy, and the portfolio equity path can recognize PnL early.

Diagnostics must record this exclusion assumption without asserting actual
within-leaf non-overlap. Resolving it requires label exit identities/timestamps and
replay eligibility aligned to those exits or a separately specified conservative
holding interval. That semantic repair is outside collection and requires its own
regressions. Until then, funded-capital and non-overlap claims remain blocked even
within a leaf. Cross-leaf or cross-product feasibility is also unverified. Unknown
cost semantics remain unknown; do not infer gross/net classification from filenames.

## Group dispositions

Record every root-feature/direction group observed in fitted trees, including groups
with no qualifying leaves. Thresholds can differ across folds within one group.
For each group record observed fold IDs, qualifying fold IDs, distinct passing count,
required passing count, emitted-profile identity if any, and rejection reasons.
Absent groups in a particular fold are marked absent, not as a fabricated leaf.
Group statistics remain search summaries. No representative leaf inherits the
aggregate as its own performance. Leaf records cannot be dropped because their
group failed the four-of-five gate.

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
   early-return disposition, record each fitted leaf before filters, and record
   rejected groups. Preserve existing profile-return behavior. Tests stub fitting
   and use synthetic data; they must not mine real archives.
3. Run writer: atomic run-isolated persistence, interruption recovery, strict JSON,
   and no mixing of prior-run outputs. Test failures at each persistence boundary.
4. Read-only diagnostic reporting and accountable selection consumption: reject
   incomplete evidence where complete evaluation is required; report exclusions
   without policy-performance claims; require selection access events before use.

Acceptance fixtures include: missing product file, missing horizon label, too few
labeled rows, invalid timestamps, incomplete nested folds, a losing leaf, a zero-trade
leaf, several passing leaves in one fold, a group below threshold, all groups rejected,
an interrupted run, and gapped timestamps where clock-hour eligibility precedes
the row-count label exit. Assert that losses and exclusions survive serialization and
that collection does not change which legacy profiles are emitted.

A completed implementation still requires a separately predeclared decision process,
causal feature/label validation, holdout access discipline, funded-equity replay,
matched-cost baselines and prospective execution evidence before trading conclusions.
