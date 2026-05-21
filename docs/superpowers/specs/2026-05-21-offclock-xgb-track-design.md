# Off-the-Clock XGB Track — Design Spec

**Date:** 2026-05-21
**Status:** Draft — pending user review
**Scope:** polymarket_app — Sub-project 2 of the off-the-clock XGB exploration

## Problem

The deployment scorecard showed the v3 driver fails 3 of 4 hard gates — net-negative after retail fees. The operator's hypothesis: fixed time-interval sampling is the structural ceiling. Sub-project 1 built the dollar-bar data pipeline (activity-based bars persisted to `data/history/dollar/`). SP2 puts the hypothesis to the test: it trains XGB on dollar bars and on matched time bars, and asks the deployment scorecard which wins.

## This is Sub-project 2 of 2

- **SP1 (shipped, Session 58.71n)** — the dollar-bar data pipeline. Output: per-product dollar-bar parquet at `data/history/dollar/<pid>.parquet`.
- **SP2 (this spec)** — the off-the-clock XGB track. Builds and scorecard-evaluates a sweep of XGB configs across both bar substrates and reports a verdict.

**Operator prerequisite:** SP2's sweep requires `data/history/dollar/` to be populated — the operator must have run SP1's `backfill_1m_candles` + `build_dollar_bars` steps first. The time-bar substrate (`data/history/<pid>.parquet`) already exists.

## Decisions locked (from brainstorming, 2026-05-21)

| Decision | Choice | Rationale |
|---|---|---|
| Substrates | **Dollar bars AND 1h time bars** (matched control) | Running the identical pipeline on both makes dollar-vs-time a clean A/B — bar structure is the only thing that differs, so a scorecard delta is attributable to it. |
| Label variants | **Direction AND triple-barrier** | Direction (`close[t+k] > close[t]`) isolates the bar-structure effect against the v3-style label; triple-barrier adds the fully event-relative framing. Two variants separate the bar effect from the labeling effect. |
| Horizons | **k ∈ {4, 24, 72, 168}** bars | v4-style horizon sweep; k counts bars (dollar-bars or time-bars depending on substrate). |
| Features | **`extract_v4`** | 5 OHLCV channels × 3 tiers (60/168/336) × 10 stats = 150 features. Pure, tested; a dollar bar is an OHLCV bar so it applies directly. |
| Total configs | **2 × 2 × 4 = 16** | Each is one trained-and-scored XGB model. |

## Architecture

Two new files. The harness lives in the scorecard package and reuses its primitives (`compute_scorecard`, `purged_walk_forward_splits`, `realized_log_returns_per_sample`). It is a sibling of `_cv_harness.py`, not an extension of it — `_cv_harness.py` is v3-specific (v3 extractor, naive label) and must stay untangled.

| File | Responsibility |
|---|---|
| `backend/tools/_scorecard/_offclock_harness.py` | Sample building for both substrates and both label variants; per-fold OOF prediction. |
| `backend/tools/offclock_sweep.py` | CLI: loop the 16 configs, run `compute_scorecard` on each, write the results doc. |

## Sample building

For one `(substrate, label_variant, horizon=k)` config, for each survivorship-aware top-20 product:

1. Load the product's bars — dollar bars from `data/history/dollar/<pid>.parquet`, or 1h time bars from `data/history/<pid>.parquet`. Both are time-sorted OHLCV bar lists.
2. Roll samples from bar index 336 (the macro tier needs 336 bars of lookback) up to `len(bars) - k` (the label needs k bars ahead).
3. Per sample at bar index `t`: slice tiered lookback `{micro: bars[t-60:t], meso: bars[t-168:t], macro: bars[t-336:t]}`, run `extract_v4` → 150 features.
4. Compute the label and the entry/exit close prices per the label variant (below).
5. Record the sample's entry timestamp as bar `t`'s `start` (present on both substrates).

Pool all 20 products' samples and sort by entry timestamp. `sample_step` (roll one sample every N bars) is a CLI parameter so the 16-config sweep stays tractable; its default is set in the implementation plan.

## The two label variants

Both operate on the entry bar `t` and a horizon of `k` bars.

**Direction.** `label = 1 if close[t+k] > close[t] else 0`. Realized log-return = `ln(close[t+k] / close[t])` — entry close `close[t]`, exit close `close[t+k]`.

**Triple-barrier.** Upper barrier `close[t] × 1.01`, lower barrier `close[t] × 0.99`, vertical timeout at `t+k`. Scan bars `t+1 … t+k`: if a bar's `high ≥ upper` the label is 1 (UP); if `low ≤ lower` the label is 0 (DOWN); first barrier hit wins; if both hit in one bar the bar's close direction breaks the tie. If neither barrier is hit by `t+k`, the label is the sign of `close[t+k] − close[t]`. Realized log-return is barrier-aware: UP hit → `ln(1.01)`, DOWN hit → `ln(0.99)`, timeout → `ln(close[t+k] / close[t])`. A small barrier-resolve helper computes the label and the exit price together.

## Evaluation

Each of the 16 configs is evaluated identically:

1. Build the pooled, time-sorted sample set (features, labels, entry/exit closes, entry timestamps).
2. 5-fold purged walk-forward CV with a 4h embargo (`purged_walk_forward_splits` on the entry timestamps).
3. Per fold: train a fresh XGB booster on the fold's train rows using the v4 production parameters; predict the validation rows. Collect out-of-fold scores, fold ids, and per-fold calendar spans.
4. Compute realized returns via `realized_log_returns_per_sample(entry_closes, exit_closes)`.
5. Run `compute_scorecard(scores, labels, returns, fold_ids, fold_spans_days)` → a `ScorecardReport` (precision-at-gate, expected return, paper-Sharpe, ECE; the 4 hard gates).

## Output

`docs/superpowers/specs/2026-05-21-offclock-sweep-results.md`, written by the sweep CLI:

- A 16-row table: `substrate × label_variant × horizon` → OOF AUC, the 4 gate outcomes, recommended operating τ.
- The **dollar-minus-time delta** per `(label_variant, horizon)` cell — the clean A/B that isolates bar structure.
- A verdict: does any dollar-bar config pass gates the matched time-bar config fails? Does the operator's "time bars are the ceiling" hypothesis hold?

## Error handling

- A product with too few bars to roll a single sample (fewer than `336 + k + 1`) is skipped and logged; it does not abort the config.
- A config that yields no samples (e.g. dollar bars not yet built) fails loud with a clear message naming the missing `data/history/dollar/` prerequisite.
- The sweep runs the 16 configs independently — one failing config is logged and skipped, the rest proceed.

## Testing

- Sample-building and both label-variant functions are pure (bars in, samples/labels/returns out) and unit-tested on synthetic bars: tier slicing, the direction label + return, the triple-barrier label + barrier-aware return (UP / DOWN / timeout / tie), the skip of too-short products.
- The per-config OOF and the 16-config sweep loop are tested with a mocked harness / mocked `compute_scorecard` — no live training in unit tests.
- TDD red-green-refactor per the project workflow.

## Out of scope

- Promoting any config to the live driver or a shadow track — SP2 produces a verdict, not a deployment. Promotion, if warranted, is a separate decision.
- Dollar-bar-native features (`n_candles`, `dollar_value`, activity-intensity) — SP2 uses `extract_v4`'s OHLCV features only; the native features are a possible follow-on variant.
- SELL-side evaluation — the scorecard v1 measures LONG signals only.
- Tick / trade-level bars — SP1 capped the base at 1-minute candles.

## See also

- `2026-05-20-dollar-bar-data-pipeline-design.md` — SP1, which produces the dollar bars SP2 consumes
- `2026-05-18-xgb-deployment-scorecard-design.md` — the scorecard SP2 evaluates every config against
- `2026-05-18-xgb-scorecard-baseline-results.md` — the v3 time-bar baseline (1 of 4 gates) this exploration aims to beat
- `xgb_post_scorecard_roadmap.md` (memory) — the exploration roadmap; SP1+SP2 are the elevated bar-structure path
