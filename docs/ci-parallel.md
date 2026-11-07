# Parallel backend CI

Baseline: [main run 41758551278](https://github.com/gl4500/coinbase-ai-trader/actions/runs/41758551278),
2026-11-04, commit `52292c44ae7ffaa3e7c79737633f644acf4433f1`.
Backend job: 5m31s; dependency install: 41s; test step: 4m21s.
Most other jobs already execute concurrently. This is an execution-time bottleneck in that run,
not evidence of runner queue pressure across all runs.

The fast-test selector remains `not slow and not integration`. Four `backend-shards` jobs use
`pytest-split==0.11.0`, one-based groups and `least_duration`. Each installs dependencies and
collects the suite independently. No test workers share a runner, database or GPU.
The secret scan remains a prerequisite. No additional paid runner infrastructure is introduced.

Without a committed `.test_durations` file, shards initially balance by test count. Each prints
its 20 slowest tests. If imbalanced, generate a complete baseline with the same selector and
`pytest --store-durations`, review the resulting `.test_durations`, and commit it so every shard
uses identical timing inputs. Do not generate separate conflicting timing inputs per shard.
A single long test, repeated collection or expensive fixtures can limit speedup.

Each shard uploads only its explicitly named hidden coverage data file; missing files fail the
job. Coverage uses relative source paths and the same pinned coverage version on all runners.
`test-backend` waits for the complete matrix, requires success, combines data and publishes the
existing `coverage-report` XML artifact. It retains the existing `Backend tests` check name.
The security gate requires every listed prerequisite to succeed; release tagging waits for it.
Snyk's existing no-token success behavior and the advisory dependency audit are unchanged.

Validation: `python -m unittest discover -s tools/ci -p 'test_*.py' -v` tests gate exit behavior
for success, failure, skip, cancellation and malformed/missing inputs. Validate sharding by
comparing the union and intersections of collected node IDs against an unsharded collection
with the same marker selector. Validate coverage by running complementary branches in separate
shards, combining the raw data, and checking the XML includes both branches.

After the first GitHub run, compare total elapsed time, each shard's test/setup/queue time and
aggregate coverage with the baseline. Four shards repeat setup and can increase runner minutes;
no 4x speedup is promised. If total elapsed time does not improve, rebalance or reduce shard count.
Rollback is a revert of this commit. Existing required check names remain available.

## Local validation ? 2026-11-07

- Real suite collection: 1,870 selected tests (one slow/integration test deselected).
- Shards: 468 / 468 / 467 / 467. Their node-ID sets are disjoint and their union equals
  the unsharded selection. Collection is not test execution.
- Coverage integration smoke test: four separate pytest-split/pytest-cov subprocesses,
  complementary source branches, raw `.coverage.shard-N` files, `coverage combine`,
  and XML generation all passed; combined toy-source line coverage is 100%.
- Gate regression tests passed after demonstrating failure before implementation.
- YAML parses and `git diff --check` passes. Actual hosted execution/speedup remains
  to be measured; equal test counts do not establish equal elapsed durations.
