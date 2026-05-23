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

## Methodology (decided 2026-05-23)

**Target-spec-first approach.** Borrowed in spirit from gradient-based learning ("back-propagation"): define what a "winning strategy" *looks like* as a measurable target FIRST; then choose / build the optimizer that hunts for strategies meeting it. Loss-function-first design rather than model-first.

This integrates cleanly with the existing deployment scorecard pattern (Session 58.71l). The scorecard already defines pass/fail gates + a ranking for v3 at the *deployment* stage. The same shape applies one level up — at the *discovery* stage:

- **Gates** = hard constraints a strategy must meet to *qualify* as winning.
- **Ranking** = metric to compare qualifying strategies against each other.
- Mining produces candidates → gates filter → ranking picks among survivors.

The walking questions below are re-ordered: target specification (Q0) leads; the optimizer-shaped questions (Q1–Q5) follow once the target is concrete.

## Walking questions — step by step

Each question gets answered in order. Once answered, the row in the table above gets updated and the decision logged at the bottom.

### Q0 — Winning-strategy target specification

What concretely makes a strategy "winning"? Pass/fail gates + a ranking metric.

| Component | What it asks | Status |
|---|---|---|
| **Optimization criterion** | What does the mining maximize? | **RESOLVED 2026-05-23: maximize CUMULATIVE PROFIT (after retail fees) — primary ranking.** Mining outputs a ranked set of profiles; top profiles by cumulative profit are the "winners." Operator: "produce profiles that maximize profit." |
| Fee convention | Wins/losses measured at which fee tier? | **RESOLVED 2026-05-23: retail (0.6%/side, 1.2% round-trip).** All win/loss magnitudes are after-fee, net. |
| Min avg win magnitude | What's the smallest meaningful winning trade size (after fees)? | **RESOLVED 2026-05-23: ≥ +5% net (the asymmetric-momentum floor — operator selected the band Asymmetric momentum / Cohort rotation / Long-shot).** |
| Max avg loss magnitude | Cap on losing-trade size? | **RESOLVED 2026-05-23: ≤ −10% magnitude net (the long-shot ceiling).** |
| Min win rate | What precision must a strategy achieve to qualify? | **RESOLVED 2026-05-23 (revised): ranking direction ONLY — higher is better, but NOT a hard gate.** Operator pivoted from "70% gate" to "profit-maximize over just 70%-win-rate fixation." A 55%-win-rate strategy with rich per-trade profit beats a 75%-win-rate strategy with thin profit. The 70% number is now informational. |
| Min net expectancy | Avg P&L per trade floor (after fees)? | **RESOLVED 2026-05-23: no explicit floor — subsumed by the profit-maximize ranking. Cumulative profit = expectancy × frequency; ranking on cumulative is sufficient.** |
| Position sizing | Fixed-$, fixed-fraction, concurrency-capped, Kelly, or conviction-weighted? | **RESOLVED 2026-05-23: concurrency-capped fixed-fraction, max 3-5 concurrent positions.** Per-slot size = capital / max_concurrent. Mining simulates at max=5 as the primary case, with sensitivity at max=3 and max=4. When all slots are full the strategy cannot fire; closing a position frees a slot. |
| Min trade frequency | Min trades/year — high enough for statistical validity, low enough to allow meaningful per-trade sizing (depends on sizing approach) | **RESOLVED 2026-05-23: NO HARD GATE.** Trade count is reported on each profile (informational). Qualification is defined by the other gates (win/loss band, max DD, profit ranking). Statistical validity of low-frequency strategies is enforced via the long-shot caveat (extra validation rigor), not via a frequency floor. Operator: "I don't want to have a trade limit, more of what meets the model standards." |
| Max drawdown | Worst cumulative loss tolerated? | **RESOLVED 2026-05-23: ≤ 30% peak-to-trough.** Aggressive / risk-tolerant. Strategies whose historical drawdown exceeded 30% are rejected. |
| Risk-adjusted return | Sharpe / Sortino / Calmar floor for ranking? | **RESOLVED 2026-05-23: Sortino as SECONDARY ranking, no hard gate.** Primary rank = cumulative profit; Sortino surfaces alongside as a risk-quality tiebreaker / context column. Sortino chosen over Sharpe because it doesn't penalize upside volatility — appropriate for the asymmetric (Asymmetric Momentum / Cohort Rotation / Long-Shot) profile band the operator selected. |

These don't all have to be hard gates — some can be ranking metrics, some can be informational-only (reported but not enforced). The point is to pin down which dimensions matter to the operator BEFORE building any mining infrastructure, so the mining knows what it's optimizing toward.

**Status: COMPLETE 2026-05-23.** All 9 components resolved. See "Decisions locked" at the bottom for the consolidated target spec. Q1–Q5 below (optimizer-shape) follow from the resolved target.

### Q1 — Trade horizon

What timescale does a "winning trade" live on? Drives label horizon, validation fold structure, feature-window choice, and which input variables matter (tokenomic state evolves slowly; price-trend evolves fast).

Crypto-native horizon buckets:

| Horizon | Hold | Tokenomic state relevance | Trend-signal relevance |
|---|---|---|---|
| Microstructure | minutes to <1h | barely moves | dominant |
| Swing | hours to ~1 day | slow drift, surfaces between trades | dominant |
| Cohort rotation | days to ~weeks | actively changing (unlocks, supply events) | meaningful but slower |
| Positional | weeks to months | dominant (thesis trades) | secondary |

**Status: RESOLVED 2026-05-23 — horizon is a SEARCH DIMENSION inside the data-feasible band (hours to ~weeks), not a pre-fixed pick.** The operator's framing is: optimize for win rate × meaningful magnitude, and let the winning horizon emerge from the search. Concrete implications:

- **Outer envelope set by data:** microstructure (<1h) is out (no L2, no 1m OHLCV backfilled); positional (>weeks) is out (12-month CoinPaprika free-tier history yields too few non-overlapping trades for honest validation). The feasible band is **roughly hours to ~weeks.**
- **Mining must be multi-horizon:** each candidate (entry-bar, token) gets outcomes labeled at multiple horizons (e.g. {1h, 4h, 24h, 72h, 168h, 336h}); each candidate rule is evaluated against the win-rate × magnitude properties of its outcomes at each horizon; the winning horizon is whatever produces the best properties for the rule.
- **Parallel action:** expand CoinPaprika tokenomic coverage before mining work begins. See "Parallel actions" section below.

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
- 2026-05-23: **Methodology = target-spec-first ("backprop"-style).** Define the winning-strategy target (gates + ranking) BEFORE building the optimizer. Mining hunts for strategies that pass the target.
- 2026-05-23: **Trade horizon = SEARCH DIMENSION** within the data-feasible band (hours to ~weeks), not pre-fixed. Optimization criterion is high win rate × meaningful magnitude; horizon emerges from the search.
- 2026-05-23: **Optimization criterion: high win rate × meaningful win magnitude** (specific thresholds TBD via Q0). Operator stated framing: "high win rates, timescale is based on the win rate and percentage of the win."
- 2026-05-23: **Min win rate gate = 70%.** Higher is better — also enters the ranking metric. Operator: "70% win rate ... but higher is better."
- 2026-05-23: **Fee convention = retail tier (0.6%/side, 1.2% round-trip).** All win/loss magnitudes throughout this discovery are NET of round-trip fees. A trade that nets +0.5% gross is a LOSS under this convention.
- 2026-05-23: **Per-trade profile band** — operator selected Asymmetric Momentum + Cohort Rotation + Long-Shot. Concrete gates: **avg_win ≥ +5% net** AND **avg_loss ≤ −10% net magnitude**. Strategies whose profile lands outside this band are rejected. Strategies on tight-scalp or standard-swing profiles are out of scope.
- 2026-05-23: **Long-shot caveat** — strategies whose profile lands at the long-shot end (avg_win ≥ +15%, avg_loss ≥ −7%) at 70%+ win rate are statistically unusual on this data scale. Validation methodology must require extra rigor for them (higher minimum trade count, out-of-sample re-test, possibly bootstrap CI). To be revisited at the validation-methodology question.
- 2026-05-23: **PRIMARY OPTIMIZATION = MAXIMIZE CUMULATIVE PROFIT.** Mining outputs a ranked set of profiles; top profiles by cumulative profit (after retail fees) are the "winners." Operator: "produce profiles that maximize profit."
- 2026-05-23: **70% win rate DOWNGRADED from hard gate to ranking direction.** Higher win rates still rank better all else equal, but a 55%-win-rate strategy with rich per-trade profit beats a 75%-win-rate strategy with thin profit. Operator: "I would like to base it on profit wins over just 70% winning rates."
- 2026-05-23: **No explicit per-trade expectancy floor** — subsumed by cumulative-profit ranking. Per-trade economics are still gated by the avg_win/avg_loss band so the ranking doesn't reward degenerate per-trade profiles.
- 2026-05-23: **Per-trade sizing matters and is a Q0 component.** Operator: "higher investment amount will equal a greater return..vs. many small trades that dilute the amount of available cash." Mining must surface sizing-aware metrics; min trade frequency must balance statistical validity against capital dilution. Cumulative profit is sensitive to sizing — the ranking metric should be computed under whatever sizing approach is chosen, not under an arbitrary fixed-$.
- 2026-05-23: **Sizing approach = concurrency-capped fixed-fraction, max 3–5 concurrent.** Per-slot size = capital / max_concurrent (so at max=5, each slot = 20% of capital). Mining simulates at max=5 as primary, with sensitivity runs at max=3 and max=4. When all slots are full, the strategy cannot fire; closing a position frees a slot. Higher firing frequencies are effectively capped by slot availability — this is the structural answer to the "dilution" concern.
- 2026-05-23: **Max drawdown gate = ≤ 30% peak-to-trough.** Aggressive / risk-tolerant. Strategies whose historical drawdown exceeded 30% are rejected outright, regardless of cumulative profit.
- 2026-05-23: **No min-trade-frequency gate.** Operator: "I don't want to have a trade limit, more of what meets the model standards." Trade count is reported per profile; statistical validity for low-frequency profiles is enforced via the long-shot caveat (extra OOS / bootstrap-CI rigor at validation time), not via a hard floor.
- 2026-05-23: **Risk-adjusted return = Sortino as SECONDARY ranking, no hard gate.** Primary ranking = cumulative profit. Sortino is surfaced alongside as a risk-quality tiebreaker. Sortino over Sharpe because the asymmetric profile band (Asym Momentum / Cohort Rotation / Long-Shot) shouldn't be penalized for upside volatility. Max DD gate (≤30%) already handles catastrophic-risk protection.
- **2026-05-23: Q0 COMPLETE — target spec finalized.** Hard gates: avg_win ≥ +5%, avg_loss ≤ -10% magnitude, max DD ≤ 30%, profile inside band {Asym Momentum / Cohort Rotation / Long-Shot}, at retail fees. Primary ranking: cumulative profit. Secondary ranking: Sortino. Informational: win rate (higher better), trade count. Sizing: concurrency-capped fixed-fraction, max 3-5 concurrent.

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
