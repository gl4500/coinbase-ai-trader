# Dollar-Bar Data Pipeline — Design Spec

**Date:** 2026-05-20
**Status:** Draft — pending user review
**Scope:** polymarket_app — Sub-project 1 of the off-the-clock XGB exploration

## Problem

The deployment scorecard (Session 58.71l–m) showed the v3 driver fails 3 of 4 hard gates — net-negative after retail fees. The operator's hypothesis (2026-05-20): the lackluster performance is structural — fixed time-interval sampling is the ceiling, not the features. Today every XGB feature is built from fixed 1-hour candles, so a dead 3am hour and a news-spike hour are each one bar; information arrives in bursts but time bars sample uniformly, producing fat-tailed, heteroskedastic returns.

The fix being explored is **activity-based bars** — bars that close when the market transacts a fixed quantum of value, not when the clock ticks.

## This is Sub-project 1 of 2

The full "off-the-clock" exploration decomposes into:

- **SP1 (this spec) — Activity-bar data pipeline.** Backfill finer base candles, construct dollar bars, persist them. Output: per-product dollar-bar parquet series.
- **SP2 (separate spec, later) — Off-the-clock XGB track.** Event-relative labels on the dollar bars, feature extraction, a new XGB training path, scorecard evaluation vs. the v3 time-bar baseline.

SP2 cannot start until SP1 produces bars. The cross-asset alignment / pooling problem (dollar bars close at different wall-clock moments per product) is **SP2's** concern — SP1 only persists per-product series with timestamps so SP2 can align them however it chooses.

## Decisions locked (from brainstorming, 2026-05-20)

| Decision | Choice | Rationale |
|---|---|---|
| Base candle granularity | **1-minute** | The base candle is atomic — a dollar bar can only close on a candle boundary, so base granularity is the resolution ceiling. 1m is the finest the Coinbase candle API offers and gives the cleanest hypothesis test: a flat result can't be blamed on coarse bars. |
| Product scope | **Survivorship-aware top-20** | Same 20 products the deployment scorecard evaluates, keeping the experiment comparable to the v3 baseline. ~11× lighter backfill than all ~220 products. |
| Bar type | **Dollar bars** | Bar closes on cumulative price×volume. `$1M of activity` is comparable across BTC and a small-cap (good for a pooled model) and stays meaningful as price drifts. |
| Threshold | **Calibrated per product to match the 1h bar count** | `threshold_p = Σ dollar value ÷ n_1h_bars`. Each product yields ≈ its current 1h bar count — same dataset size as the v3 baseline, only the bar *placement* changes. |
| Pipeline structure | **Two-stage, both materialized** | Persist the 1m layer and the dollar layer as separate parquet sets. Each stage independently testable and re-runnable; the 1m layer is reusable by other tasks (e.g. the CUSUM path). |

## Architecture

Two standalone, operator-run tools (not part of the live scan loop):

| Stage | Tool | Reads | Writes |
|---|---|---|---|
| 1. Backfill | `backend/tools/backfill_1m_candles.py` | Coinbase Advanced Trade API, `granularity=ONE_MINUTE` | `backend/data/history/1m/<pid>.parquet` |
| 2. Construct | `backend/tools/build_dollar_bars.py` | `data/history/1m/<pid>.parquet` + `data/history/<pid>.parquet` (1h, for calibration) | `backend/data/history/dollar/<pid>.parquet` |

The `data/history/<granularity>/` subdirectory convention already exists in `services/history_backfill.py` (it defines a `5m/` path). The 1m and dollar layers follow it.

Stage 2's core is a pure function — `dollar_bars_from_candles(candles, threshold) -> list[bar]` — with no I/O, so bar-boundary logic is unit-tested directly.

## Stage 1 — 1-minute backfill

- Fetches 1-minute OHLCV candles for each of the top-20 products from the Coinbase Advanced Trade API. Reuses `services/history_backfill.py`'s Coinbase candle-fetch path where feasible (parameterized by granularity) rather than duplicating it; per the loose-coupling rule.
- **Depth:** covers exactly the calendar span of each product's existing 1h parquet (`first_ts`→`last_ts`). Matching the span exactly keeps the threshold calibration coherent — `Σ dollar value` and `n_1h_bars` are then measured over the same window — and keeps the dollar-bar dataset comparable to the v3 baseline period.
- **Pagination:** the candles endpoint caps candles per request; the backfill paginates across the full span.
- **Rate limiting:** respects Coinbase rate limits with backoff. This is a long operator-run job (1-minute candles over a multi-year span × 20 products) — it is run once, offline, by the operator; it is not on the scan-loop path.
- **Output schema:** matches the existing 1h parquet — `start, open, high, low, close, volume`.

## Stage 2 — dollar-bar construction

- **Dollar value per 1m candle:** `volume × typical_price`, where `typical_price = (high + low + close) / 3`.
- **Per-product threshold:** `threshold_p = Σ(dollar value over all 1m candles for p) ÷ (row count of p's 1h parquet)`.
- **Boundary rule:** walk the 1m candles in time order, accumulating dollar value. When the accumulator ≥ `threshold_p`, close the bar — including the candle that crossed the threshold — and reset the accumulator. A 1m candle is atomic and is never split across bars (the accepted 1-minute resolution ceiling). Each dollar bar is therefore an integer number of consecutive 1m candles.
- **Trailing partial bar:** if residual accumulated dollar value at the end of the series is below the threshold, that final partial bar is dropped (incomplete).

## Dollar-bar schema

One parquet row per dollar bar:

| Field | Meaning |
|---|---|
| `start` | timestamp of the first underlying 1m candle |
| `end` | timestamp of the last underlying 1m candle |
| `open` | open of the first 1m candle |
| `high` | max high across the bar's 1m candles |
| `low` | min low across the bar's 1m candles |
| `close` | close of the last 1m candle |
| `volume` | Σ of 1m volumes |
| `dollar_value` | Σ of 1m dollar values (the accumulator at close; ≥ threshold) |
| `n_candles` | count of underlying 1m candles in the bar |

`n_candles` against the `end − start` span exposes wall-clock gaps (a bar that spans missing minutes).

## Error handling

- **API gaps:** Coinbase 1m history can have missing minutes. The backfill records gaps; the dollar-bar builder accumulates whatever candles exist — a bar may span a gap, surfaced via `n_candles` vs `end − start`.
- **Insufficient data:** a product with too few 1m candles to form bars is skipped and logged; it does not abort the run.
- **Idempotency:** both stages overwrite their parquet outputs and are safe to re-run.

## Testing

- **Stage 2 core** (`dollar_bars_from_candles`) — pure, unit-tested on synthetic 1m candles: boundary placement at the threshold, OHLC aggregation, `volume`/`dollar_value`/`n_candles` sums, dropped trailing partial, and the threshold-calibration formula.
- **Stage 1** — tested against a **mocked** Coinbase client (no live API calls, per CLAUDE.md): pagination across the span, gap handling, parquet output schema.
- TDD red-green-refactor per the project workflow.

## Out of scope (→ Sub-project 2)

- Cross-asset alignment / pooling of dollar bars across products.
- Event-relative (triple-barrier) labeling.
- Feature extraction on dollar bars.
- XGB training and scorecard evaluation.
- Tick / trade-level data — 1-minute candles are the base; true dollar bars from trades are not in scope.

## See also

- `2026-05-18-xgb-deployment-scorecard-design.md` — the scorecard SP2 will evaluate against
- `2026-05-18-xgb-scorecard-baseline-results.md` — the v3 time-bar baseline (1 of 4 gates) this exploration aims to beat
- `xgb_post_scorecard_roadmap.md` (memory) — the 8-path exploration roadmap; this is the elevated bar-structure path
