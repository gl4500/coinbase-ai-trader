# CI setup caching and measured shard balancing

These changes follow PR #112 and issue #117. The comparison baseline is main commit
`76a21d5e509518d11f55101b5b4b25754648aca1`, run
[42492861418](https://github.com/gl4500/coinbase-ai-trader/actions/runs/42492861418).
That run took 4m24s overall. Its backend shard jobs took 129, 138, 218 and 146 seconds;
the aggregate coverage job added 13 seconds. These job durations include setup and
cleanup, not just test execution.

## Phase 1: installed environment cache

`.github/actions/setup-backend` is a Linux-only composite action. The backend shards
all use its shared manifests: `backend/requirements-ci.txt` includes the existing
application requirements and pinned test extras; CPU Torch uses a separate manifest
and its own CPU index. No production requirement or application code is changed.
The extra package versions were taken from the successful baseline's install log.

The cache contains `.ci-venv`, so an exact hit skips package resolution, installation
and wheel extraction. Every runner still restores its own environment and collects
the test suite. Cache transfer/decompression is real overhead and must be measured.

The key includes runner OS/architecture/image, the exact resolved Python version,
all backend requirement manifests, the action definition, a manual cache generation,
and the ISO UTC week. There are no partial restore keys. Runtime requirements contain
lower bounds, so weekly refresh bounds the age of their resolved dependency graph.
The action runs `pip check` and prints the actual package versions on both paths.
Only the environment directory is cached; repository files and test outputs are not.

A cold or invalidated cache still installs dependencies on each runner. A cache hit
is a performance optimization, never a prerequisite for correct tests. A single
prepare job or a prebuilt image remains an alternative if transfer overhead wins.

## Phase 2: measured durations

`tools/ci/timings.py` records the actual selected node IDs and setup/call/teardown
durations for each shard. The aggregate job requires one successful report for every
shard, the same source revision, disjoint selections and complete finite measurements.
It uploads a merged timing candidate for review. It never silently changes a shard's
input during an active workflow run.

The recorder is loaded as `timings` through `PYTHONPATH=tools/ci`, rather than as
`tools.ci.timings`: the latter would pre-load a namespace that conflicts with the
application's existing `backend/tools` package. A subprocess regression recreates
that import ordering and verifies the application tools remain importable.

A shared `.test_durations` baseline will be committed from a successful hosted run;
all four shards then use that identical file with pytest-split's `least_duration`.
New tests get the plugin's estimated average until the reviewed baseline is refreshed.
This cannot split one individually long test or guarantee perfect balance when fixture
costs or machine load change. The fast-test marker, test assertions, CPU Torch,
combined coverage, stable required checks and release gate remain in place.

## Verification and benchmarks

- Unit and real-pytest subprocess tests exercise the timing recorder/merger, including
  missing, duplicate, overlapping, failed, mixed-revision and invalid-duration inputs.
- The private CI sandbox runs the same setup action with the representative ML
  dependency manifests, rather than timing only a tiny pytest installation.
- Record cold and warm hits separately. Compare the same source head's selected IDs,
  combined coverage, slowest shard, whole-pipeline elapsed time and summed job time.
- Preserve existing checks on deliberate shard failure, and obtain exact-head peer
  review before merge. No performance result is claimed before hosted measurement.
