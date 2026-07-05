# Macro-Regime Layer — Design

**Date:** 2026-07-05
**Branch:** `feat/macro-regime-layer` (off `main`)
**Status:** approved design, pre-implementation
**Related:** `btc_macro_drivers_findings` (memory), [[win_factors_improvement_loop]], `hmm_regime.py`, CLAUDE.md invariants #14/#16/#20/#21

## Background

The cross-layer BTC-driver study (2026-07-05, memory `btc_macro_drivers_findings`)
established two things that motivate this work:

1. **Direction prediction is exhausted at every layer** — technical signals sit at the
   no-signal floor (|corr| < 0.06 vs forward return); macro, sentiment, ETF-flow, and
   on-chain signals are all *contemporaneous / regime-setting*, not predictive. The one
   genuine forward signal in the entire study is **MVRV < 1** (deep-undervaluation
   capitulation → +14.8% median forward-90d return, every cycle).
2. **What actually moves BTC is regime + structural demand.** The BTC-equity correlation
   is a *dial* (0 → 0.6, collapsed to ~0 in 2025); ETF flows explained the 2025
   macro-decoupling (monthly BTC~flow +0.77 vs BTC~SPX +0.32).

The live micro model is **blind to all of this** — it decides entries from price/indicator
features that carry no directional edge, with no notion of *which regime it is trading in*.
This design adds a **macro-regime layer** that classifies the regime on a slow (daily)
clock and modulates **how much** the micro loop is exposed — never **which way** it bets
(a slow, non-predictive signal has no business making per-tick directional calls).

## Existing regime mechanisms (and why neither is this)

- **`services/hmm_regime.py`** — a 3-state Gaussian HMM on hourly returns+vol. A
  *volatility/price* regime (ranging / trending / crisis). Micro, hourly. Complementary,
  not a substitute.
- **`services/macro_signals.py`** — despite the name, crypto *derivatives micro-structure*
  (funding, L/S, OI, dominance, premium), **live-fetch only, not persisted / not
  backtestable**. Provides `buy_gate_multiplier()` etc.

There is **no true macro-regime layer** (equity-coupling regime, dollar/yield backdrop,
cycle position via MVRV, structural demand). That is the gap this fills.

## Goal

A daily **Macro Regime Evaluator** that emits a small, stable `RegimeState`
(`exposure_scalar` + components + confidence), persisted as a backtestable daily series,
consumed by the scan loop as an **entry-sizing** modulator — behind a default-off flag,
validated offline first, then shadowed on 8002.

## Architecture — separation of timescales (cascade)

```
DAILY job ─► RegimeEvaluator ─► RegimeState ─► regime_state DB table (daily series + latest)
             (MVRV + corr-gated              {date, mvrv, corr_spx_90d, macro_risk_raw,
              macro-risk)                      exposure_scalar, confidence, components_json}
                                                          │
60s scan loop ─► reads latest RegimeState (cheap DB/cache read) ─► scales ENTRY size
```

The macro layer is a **slow producer**; the scan loop is a **fast consumer** doing a cheap
read. **No macro math or network calls in the hot path.** Three cadence tiers stay
separated: macro (daily) → vol regime (hourly, existing `hmm_regime`) → micro execution
(60s / tick). Loose coupling (per `feedback_loose_coupling`): the scan loop reads one
`RegimeState` field and never knows how it was computed — mirrors how it already reads
cached `p_down` / `position_dollars`.

## Two-phase scope (one spec, phased plan)

**Phase 1 — offline, zero live risk.** Build evaluator + data adapters + `regime_state`
store, backfill the daily historical regime series (2016+), and an offline **backtest
harness** that overlays `exposure_scalar` on the historical paper-trade record and measures
whether regime-scaling exposure improves risk-adjusted outcomes (Sharpe / max-drawdown /
return-per-unit-risk). **Hard gate:** proceed to Phase 2 only if it demonstrably helps.

**Phase 2 — live, opt-in, default-off.** A consumer hook multiplies entry sizing by
`exposure_scalar`, gated behind a new flag `USE_REGIME_EXPOSURE` (default false →
byte-for-byte unchanged), deployed to the **8002 shadow** first. Mirrors the
`USE_MAKER_EXECUTION` / `MC_FILTERS=""` default-off contract (invariants #14/#21).

## Components

1. **`services/regime/macro_regime.py`** — `RegimeEvaluator`: a **pure function**, series
   in → `RegimeState` out, no I/O. The formula (below) lives here and only here.
2. **`RegimeState`** dataclass — `{date, mvrv, mvrv_prior, corr_spx_90d, macro_risk_raw,
   macro_mult, exposure_scalar, confidence, components}`.
3. **`services/regime/sources.py`** — data adapters: FRED (CBBTCUSD / SP500 / DTWEXBGS /
   DFII10) + CoinMetrics community MVRV (`CapMVRVCur`), each with a daily parquet cache.
   Access details (headers, series ids, quirks) documented in memory
   `btc_macro_drivers_findings`. Network failures degrade gracefully (stale cache → lower
   confidence).
4. **`regime_state` DB table** — daily rows (backtestable series) + a latest-row lookup.
   Columns mirror `RegimeState`. New migration, additive/nullable.
5. **Daily trigger** — piggyback the existing daily cadence (`services/history_backfill`)
   or a dedicated scheduled task; runs the evaluator once/day and upserts `regime_state`.
6. **`tools/regime/backtest_regime.py`** — Phase-1 offline validation harness (reads the
   historical trade record + daily regime series; reports risk-adjusted deltas).
7. **Phase-2 live consumer** — a flag-gated hook at the entry-sizing step in `cnn_agent`.

## The exposure-scalar formula (`RegimeEvaluator`)

Two multiplicative factors, **each defaulting to 1.0 (no-op) when its data is missing**,
combined and clamped. All anchors/constants are config-tunable.

### Factor 1 — MVRV cycle prior (`mvrv_prior`)

Piecewise-linear on MVRV, **asymmetric** — the validated edge is the *low* end; high MVRV
was momentum-y in-sample, so the high-side trim is mild tail-insurance, **not** a
data-claimed reversal (documented as a judgment overlay):

| MVRV | prior | rationale |
|---|---|---|
| ≤ 0.8 | 1.25 | capitulation → the one real forward edge (+14.8% median fwd-90d) → lean in |
| 0.8 → 1.5 | 1.25 → 1.05 (interp) | historically favorable |
| 1.5 → 3.0 | 1.05 → 0.95 (interp) | neutral band |
| ≥ 3.5 | 0.85 | late-cycle drawdown insurance (judgment overlay, not a data claim) |

Between 3.0 and 3.5, interpolate 0.95 → 0.85.

### Factor 2 — correlation-gated macro risk (`macro_mult`)

- `macro_risk_raw ∈ [−1, +1]` = blend of: equity trend (SP500 vs its 50-day MA →
  +1 above / −1 below, scaled), dollar direction (DTWEXBGS falling = risk-on = +),
  real-yield direction (DFII10 rising = risk-off = −). Simple average of the standardized
  sub-signals, clipped to [−1, 1].
- `w = max(0, corr_spx_90d)` — the **gate** (BTC-SP500 90-day return correlation).
- `macro_mult = 1 + K · w · macro_risk_raw`, with `K = 0.3`.

When BTC is **decoupled** (`corr_spx_90d ≈ 0`, e.g. 2025): `w ≈ 0` → `macro_mult ≈ 1` →
the macro tape is **ignored** (the empirically-correct 2025 behavior). When **coupled +
risk-off** (2022): `w · macro_risk_raw` is negative → exposure shrinks.

### Combine, clamp, apply

```
exposure_scalar = clip(mvrv_prior * macro_mult, 0.4, 1.25)
```

- **Applied to ENTRY sizing only** — never to exits (regime must never trap a position the
  exit logic would otherwise close).
- **Confidence & fail-safe:** if an input is stale (> `_REGIME_STALE_DAYS`, default 3,
  mirroring the `p_down` staleness gate inv #20) or missing, that factor → 1.0 and
  `confidence` drops; if **all** inputs stale/missing → `exposure_scalar = 1.0` (fully
  neutral). A broken regime layer must **never** halt the book.

### Worked examples

- **2022 (coupled risk-off):** `corr≈0.6`, `macro_risk_raw≈−0.8` → `macro_mult ≈ 1 −
  0.3·0.6·0.8 = 0.856`; MVRV≈2.5 → prior≈0.97 → `exposure_scalar ≈ 0.83` (reduce). ✅
- **Nov 2025 (decoupled):** `corr≈0` → `macro_mult≈1`; MVRV≈1 → prior≈1.1 →
  `exposure_scalar ≈ 1.1`. Macro ignored (correct). **Honest gap:** v1 does *not* catch
  that month's ETF-flow-driven crash — that is precisely what v1.5 (ETF flows) adds.
- **Cycle bottom (MVRV < 0.8):** prior = 1.25 dominates → lean in (accumulate the low). ✅

## Data flow (Phase 2, flag on)

```
daily job → RegimeEvaluator(series) → RegimeState → upsert regime_state
scan cycle → load latest RegimeState → base_size *= exposure_scalar (entry only)
           → persist regime_scalar used, for telemetry
```

## Error handling & isolation

- All network fetches (FRED / CoinMetrics) are wrapped; failure → use cached parquet →
  if no cache, that factor is neutral (1.0). Never raises into the daily job or scan loop.
- Phase-2 consumer: reading/applying the scalar is wrapped in try/except; any failure →
  `exposure_scalar = 1.0` (unmodified sizing) + log. Mirrors invariants #16/#18/#20 —
  a regime failure must never affect the driver path or halt trading.
- `USE_REGIME_EXPOSURE=false` (default) → the consumer hook is a no-op; sizing is
  byte-for-byte unchanged.

## Testing (TDD)

**`RegimeEvaluator` (pure, the bulk of coverage):**
- MVRV prior anchors: ≤0.8 → 1.25; ~2.0 → ~1.0; ≥3.5 → 0.85; interpolation midpoints.
- Correlation gate: `corr=0` → macro ignored (`macro_mult==1`) regardless of
  `macro_risk_raw`; `corr>0` + risk-off → `macro_mult<1`; + risk-on → `>1`.
- Combine + clamp bounds (0.4 / 1.25).
- Fail-safe: missing MVRV → prior=1.0; missing macro → macro_mult=1.0; all missing →
  exposure_scalar=1.0, confidence=0.
- Staleness: input older than `_REGIME_STALE_DAYS` treated as missing.
- Worked examples above reproduced as regression tests (2022 / Nov-2025 / cycle-bottom).

**Data adapters:** mocked HTTP; parse fixtures; network failure → cached/neutral (no raise).

**`regime_state` store:** upsert + latest-row lookup (mocked DB / in-memory).

**Phase-2 consumer:** flag-off → sizing unchanged (byte-for-byte); flag-on + scalar<1 →
entry size scaled; exits never scaled; consumer exception → sizing unmodified.

**Backtest harness:** deterministic on a fixture trade record + fixture regime series.

## Out of scope / future

- **v1.5 — ETF-flow trend** as a third factor (regime *confirmer*). Deferred: Farside
  history is 2024+ only (not backtestable pre-2024) and reflexive (0.05 forward corr).
  Added once the two-axis core is validated offline.
- **Regime-conditional micro models** (separate thresholds per regime) — rejected for v1
  (few regime transitions → overfit risk; a slow signal shouldn't switch fast behavior).
- **Regime as a meta-label feature** for the win-factors meta-labeler — natural once both
  this layer and the meta-labeler exist; `RegimeState` is persisted precisely so it can be
  consumed that way later.
- No change to `hmm_regime` (vol regime) or `macro_signals` (derivatives) — complementary.

## Deployment

Per port discipline: Phase 1 is offline (no backend). Phase 2 operator launches the 8002
shadow with `USE_REGIME_EXPOSURE=true PORT=8002 python main.py`; promotion to 8001 only
after the shadow confirms the regime-scaled sizing behaves as backtested. Zero effect on
tracked paper PnL until the flag is on and validated.
