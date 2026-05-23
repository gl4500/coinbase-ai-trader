# Strategy Discovery Rebuild — Brainstorming Notes

> **Status:** in-progress brainstorming, started 2026-05-23. Not a finalized spec.
> This doc captures the context and the walking questions so the brainstorm can
> proceed step by step across sessions. Decisions are recorded inline as each
> question is resolved.

## Why we're here

Three findings stack to motivate a step away from v3:

1. **Scorecard verdict (Session 58.71l):** v3 fails 3 of 4 deployment gates and is net-negative at the retail fee tier. See `2026-05-18-xgb-scorecard-baseline-results.md`.
2. **Probe selection-bias verdict (Session 58.71p, this session):** v3's honest deployable OOF AUC (0.5120) is *below* the best-of-N noise floor under the empirical fold-level SE. The "best documented" 0.5284 deflates to marginal (0.735). The only "confirmed PASS" channel-add (RSI-rank +0.0124) is below the best-of-17 noise floor. See `2026-05-22-probe-selection-bias-results.md`.
3. **Operator framing (2026-05-23):** "the current XGB model is trading on noise, fragility and really isn't doing anything novel. It is missing key signal characteristics." SP1/SP2 (bar-structure exploration) probe one structural variable while holding v3's other framing constant — informative as a partial answer, but cannot themselves answer the rebuild-scope question.

## What v3 frames that this rebuild may step away from

| Row | What v3 does today | What this rebuild may do |
|---|---|---|
| Inputs | OHLCV tier-summary stats (`extract_v4`, 150 features) | **Tokenomic + activity state** (market cap, FDV, supply ratios, 24h volume) **plus a price-trend signal** — confirmed direction 2026-05-23 |
| Label | Binary UP/DOWN direction (or 3-class for v4.5) | 3-class direction (v4.5 infra reusable) — *or* PnL regression / per-strategy realized-return — TBD |
| Optimization target | AUC / logloss | **Realized PnL** — confirmed direction 2026-05-23 |
| Model class | XGBoost gradient-boosted trees | TBD — depends on rule-mining vs predictor choice (Q1–Q5 below) |
| Trade horizon | k bars ahead, k ∈ {4..168} (1h candles) | TBD — **Q1** |
| Decision artifact | Single classifier → threshold → BUY/SELL every bar | **Discovery / rule-mining of conditional strategies**, plural — confirmed 2026-05-23 |
| Validation | 5-fold purged walk-forward CV, 4h embargo | TBD — depends on horizon + sample density |
| Universe | Survivorship-aware top-20 Coinbase USD spot | TBD — **Q5** |
| Execution gating | Threshold on a continuous score | TBD — filter / trigger / hybrid — **Q3** |

## Stage

**DISCOVERY** — we are *mining* for which combinations of crypto-native state produce reliable realized-PnL wins, not testing a pre-committed hypothesis. No pre-committed model architecture beyond "fits within rule-mining or small-model regime given the data scale."

## Candidate input variables (crypto-native, confirmed 2026-05-23)

| Variable | What it really means in crypto |
|---|---|
| Circulating supply | Tokens actually in the market right now |
| Total supply | Tokens that will eventually exist — tells you future dilution |
| FDV ÷ Market cap ratio | Forward-dilution overhang signal — how much "unlock supply" still has to absorb |
| Market cap (price × circulating) | Current valuation of the liquid float |
| FDV (price × total supply) | Forward valuation including all unlocks |
| 24h trading volume | Activity / liquidity state right now |
| Volume ÷ Market cap (turnover) | Velocity of trading — how hot is the float |
| Price-trend signal | TBD form — Q3 |

These are **not** equity fundamentals (no earnings, P/E, dividends). They are crypto-native state variables — tokenomics + activity. Distinction matters because it rules out equity-style factor-rotation strategies as a blueprint.

## Walking questions — step by step

Each question gets answered in order. Once answered, the row in the table above gets updated and the decision logged at the bottom.

### Q1 — Trade horizon

What timescale does a "winning trade" live on? Drives label horizon, validation fold structure, feature-window choice, and which input variables matter (tokenomic state evolves slowly; price-trend evolves fast).

Crypto-native horizon buckets to choose between:

| Horizon | Hold | Tokenomic state relevance | Trend-signal relevance |
|---|---|---|---|
| Microstructure | minutes to <1h | barely moves | dominant |
| Swing | hours to ~1 day | slow drift, surfaces between trades | dominant |
| Cohort rotation | days to ~weeks | actively changing (unlocks, supply events) | meaningful but slower |
| Positional | weeks to months | dominant (thesis trades) | secondary |

**Status: OPEN**

### Q2 — Tokenomic state as filter vs continuous input

When the strategy fires, is the tokenomic state used as a **hard filter** ("only consider tokens with MC > X AND volume > Y"), or as a **continuous input gradient** (the model uses the actual numeric values, not bucket inclusion)?

- *Filter*: shrinks the universe to a cohort, then a simpler trigger acts inside the cohort.
- *Continuous input*: tokenomic values feed a model that scores tokens on a gradient.

Affects model class — filters favor rule mining / decision trees; continuous inputs favor gradient boosting / regression.

**Status: OPEN**

### Q3 — Trend signal as trigger vs gate

What does the price-trend signal *do* in the strategy?

- *Trigger*: the entry event — strategy fires the moment a defined price-action pattern hits (breakout from consolidation, EMA cross, retest of support).
- *Gate*: only opens trades in a permissive trend regime — strategy may have other entry rules but won't fire against the trend.
- *Both*: trend gates *whether* to look, then a specific trigger fires *when*.

Affects how often the strategy can fire and how the entry condition is defined.

**Status: OPEN**

### Q4 — Single strategy or family of strategies

Are we mining for *one* winning strategy that works on the whole universe, or a *family* of strategies — different rules for different cohorts (small-cap vs mega-cap, low-volume vs high-volume, etc.)?

- *Single*: one rule set; simpler; risk that no universal pattern exists.
- *Family*: per-cohort strategies; matches the intuition that different parts of the market behave differently; more complex to maintain.

**Status: OPEN**

### Q5 — Universe scope

What's the universe the strategy operates over?

- v3's choice: survivorship-aware top-20 Coinbase USD spot.
- Alternatives: all Coinbase USD pairs; all pairs with sufficient tokenomic data; regime-stratified subsets; explicitly include the long tail (which v3 explicitly excluded).

Affects data availability — CoinPaprika's free tier and the marketcap parquet writer cover a basket but not all products.

**Status: OPEN**

## Decisions locked so far

- 2026-05-23: **Stage = discovery** (rule mining for conditional strategies, not single-model prediction).
- 2026-05-23: **Inputs = crypto-native tokenomic + activity state** + a price-trend signal — *not* equity fundamentals, *not* OHLCV summary stats alone.
- 2026-05-23: **Optimization target = realized PnL**, not AUC. (The scorecard infrastructure already operates on realized PnL.)
- 2026-05-23: **3-class direction labels (up / neutral / down) acceptable** as one labeling option. v4.5 infrastructure reusable.
- 2026-05-23: **SP1/SP2 not load-bearing for this rebuild.** They probe substrate while inheriting v3's other framing — their verdict feeds in as one data point if/when the operator runs the sweep, but the rebuild does not gate on it.

## What this rebuild is NOT

- Not equity-style factor rotation (different paradigm; equity fundamentals don't apply).
- Not stock-screening (no P/E, earnings, dividends).
- Not v3 polish or refinement on the existing 28-channel cache.
- Not another single-add channel probe.
- Not a tick-level / order-book microstructure system (we have no L2 data ingestion).

## See also

- `2026-05-18-xgb-scorecard-baseline-results.md` — v3's deployment failure (1 of 4 gates)
- `2026-05-22-probe-selection-bias-results.md` — diagnostic that v3 trades on noise + selection
- `xgb_post_scorecard_roadmap.md` (memory) — original 8-path roadmap (#16 was the gate; this rebuild supersedes the rest if it proceeds)
- `xgb_probe_results_log.md` (memory) — probe history showing the OHLCV-cache search space is exhausted
- `services/coinpaprika_marketcap.py` + `tools/build_marketcap_parquet.py` — existing data path for market cap and 24h volume; FDV + supply use the same endpoint pattern
