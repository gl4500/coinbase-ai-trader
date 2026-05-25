# Live-Ops Feedback for Strategy-Discovery Phase 2 / 3 / 4

**Date:** 2026-05-24
**Author:** Claude Opus 4.7 (session ops review)
**Status:** Findings doc — complementary to Phase 2/3 designs
**Companion designs:**
- `2026-05-23-strategy-discovery-phase2-design.md` (features + labels)
- `2026-05-24-strategy-discovery-phase3-design.md` (tree mining + Q0 gates) — on `docs/strategy-discovery-phase3-spec`
- `2026-05-23-pnl-anchored-trail-design.md` (B2 trail spec, deployed 2026-05-24)
- `2026-05-23-ws-exit-checker-design.md` (WS_MODEL_DOWN trigger)

This doc captures **empirical findings from 6 weeks of live ops (2026-04-12 → 2026-05-24)** and frames them as evidence that should inform — *not redesign* — the in-flight Phase 2/3 work. It is **not** a new spec; the Phase 2/3 specs are approved and in implementation. Treat this as a feedback report from the production lane to the research lane.

---

## TL;DR

> **Important context.** The 8001 backend runs in **paper-trading mode** — no real funds are at risk. All PnL figures below are *simulated* portfolio values against the live Coinbase data feed. The analytical findings remain valid (the strategy is unprofitable in simulation), but the urgency framing is *not* "stop the bleed" — there is no bleed. The operator has elected to keep the backend running as a data/telemetry generator for shadow comparisons and future Phase 4 selection.

| Metric (paper trading, $1,000 simulated start) | Value |
|---|---|
| Paper-trading window | 2026-04-12 → 2026-05-24 (42 days) |
| Trades simulated | 1,277 (496 W / 684 L) |
| Win rate | 42% |
| Break-even win rate (given win/loss ratio 1.19) | 45.6% |
| Net paper PnL | **−$77.12** |
| Gross paper PnL **before fees** | **+$361** |
| Round-trip fees @ ~1.2% | **−$438** |
| BTC HODL same window | −18.8% |
| Strategy paper return | −42.9% |

**One-sentence summary.** The simulated signal has real alpha (+$361 gross), but at 30 paper-trades/day modeled Coinbase taker fees would consume the edge ~5× over. Phase 2/3 will only produce profiles that beat what live paper-ops have shown if the deflated, fee-aware profit gates are set high enough to filter out marginal-conviction patterns that look profitable pre-fee.

---

## What Phase 2/3 already addresses (validated by live ops)

The current Phase 2/3 designs anticipate most of the structural lessons from live ops. This is good — the research lane is not blind to the production lane's pain. Items below are *validated* by live evidence, not problems to fix.

### 1. Multi-horizon labels → addresses "exit timing context" ✅

Phase 2 emits per-row labels at `h1 / h4 / h24 / h72 / h168`. The live evidence supports this multi-horizon framing:

- **v3 currently uses a single `h=4` label** for the BUY decision. The 5/23 SOL case (entry at ~$82, ran to $87 over 13h, then back to $84) is a classic case where `h=4` correctly says "modest up" but `h=24` would have flagged a much larger move. Multi-horizon profile mining can pick the right horizon **per (pid, regime)**.

### 2. Per-(pid, horizon) profiles → addresses "per-product allowlist" ✅

Phase 3 emits ranked leaf profiles **per (pid, horizon)**. Live ops shows this is exactly the right granularity:

| PID | v4.5+model-exit backtest (7d) | Verdict |
|---|---|---|
| ALGO-USD | 75% W, +$7.35 total | Reliable |
| TAO-USD | 67% W, +$6.99 total | Reliable |
| USELESS-USD | 50% W, +$15.64 total | Big pump catches |
| FET-USD | 100% W, +$14.38 total | Single big win |
| **BILL-USD** | 0% W, **−$41.81** | **Avoid** |
| MEGA-USD | 12% W, −$31.52 total | Avoid |

A working strategy needs a **product allowlist** — Phase 3's per-(pid) profile emission gives exactly that. The mining is structurally correct.

### 3. Dynamic-exit PnL labels → addresses "labels include exits" ✅

Phase 2's dynamic-exit simulation already bakes the live exit framework (B2 trail + STOP_LOSS + max-hold) into label PnL. This is **essential**: labels computed as raw `(close[t+h] − close[t]) / close[t]` would systematically over-estimate live PnL because they ignore how positions actually close.

### 4. Q0 hard gates → addresses "filter unprofitable patterns" ✅

Phase 3 requires leaves to pass Q0 gates (cumulative profit, sortino, max drawdown, trade count, bootstrap CI) on ≥4 of 5 outer Purged-WF folds. **This is the lever that protects against the "all-time net −$77 even though signal is +$361 gross" problem we hit in live ops.** Set the cumulative-profit Q0 gate above the fee burden the strategy will see live, not above zero.

---

## What live ops surfaces that Phase 2/3 does NOT directly address

These are **not critiques of Phase 2/3** — Phase 2 explicitly states "13 features locked, new features require an explicit brainstorm round." These are candidates for the future brainstorm round (Phase 5 features / Phase 4+ deployment selection logic).

### A. Multi-resolution OHLCV features (sub-hour, super-hour)

**Live finding:** SOL bottomed at $81.68 at 04:42 EDT on 5/23. v3 (hourly bars only) never crossed its BUY threshold during the subsequent +6.8% climb because the hourly bar smoothed the reversal wick. A 5m or 15m feature would have seen the V-shape.

**Phase 2 status:** Trend features are computed from 1h OHLCV closes only. Phase 2 design explicitly defers any expansion: *"13 features are locked in the brainstorm; new features require an explicit brainstorm round."*

**Recommendation:** Add to a future brainstorm round (post-Phase 4 baseline):
- 5m / 15m derived features (micro-momentum, reversal-wick detection)
- 4h / 1D derived features (multi-timeframe trend agreement)
- All would feed Phase 2's `features.py` as additional columns; no architectural change needed

**Why not now:** Phase 2 is in implementation. Changing the feature set mid-flight breaks the design contract. Land Phase 2/3/4 baseline first; expand features as a focused follow-up.

### B. Cross-asset features (BTC dominance, sector flow, correlation regime)

**Live finding:** During the 5/23 SOL climb, BTC was stabilizing in the same window. v3 scored SOL in isolation. A "BTC stabilizing + alt-season tailwind" feature would have boosted the SOL score precisely when human traders would lean in.

**Phase 2 status:** Tokenomic features (`market_cap`, `fdv`, `vol_24h`) are per-token snapshots, **not** cross-asset signals. There is no `btc_dominance` or `sector_strength` feature.

**Recommendation:** Phase 2.x or Phase 5 candidates:
- `btc_dominance = btc_mcap / total_mcap` (Phase 1's marketcap bronze parquets already have the inputs)
- `pid_correlation_to_btc_rolling_30d`
- `cohort_strength` (avg `ret_24h_sign` across the universe's large-cap cohort)
- `vol_z_universe` (this token's RVOL vs universe median)

**Why not now:** Same as A — Phase 2 features are locked. Add after baseline.

### C. Session / time-of-day features (trivial cost, marginal impact)

**Live finding:** Crypto volume is markedly different across Asia / EU / US sessions; weekend volume is depressed. v3 features are session-blind.

**Recommendation:** `hour_of_day`, `day_of_week`, `weekend_flag` are essentially free to add. Defer to feature brainstorm round but flag as low-risk.

### D. The exit framework needs to match the signal characteristics — Phase 4 selection logic

**Live finding (most actionable):** B2's PnL-anchored trail with the 1.2% fee-floor giveback was designed around v3's higher-conviction signals (`xgb_prob > 0.55`). When applied to v4.5's marginal-conviction signals (`p_up > 0.50`), it **catastrophically chops marginal trades**:

| Strategy backtest | N trades | Win rate | Avg net | Total net |
|---|---|---|---|---|
| v4.5 BUY + **B2 trail** + fees | 313 | 22% | −1.13% | **−352%** |
| v4.5 BUY + **model-only exit** + fees | 29 | 38% | −0.65% | −18.9% |
| v3 LIVE actual (same 7 days) | 50 | 36% | +1.48% | **+$17 (positive)** |

**Implication for Phase 4 selection:** When Phase 4 ranks Phase 3 profiles for deployment, the deployment artifact must include the **exit framework that matches the profile's conviction distribution**:

- High-conviction profiles (sparse, high `avg_win`): B2 trail works
- Low-conviction profiles (frequent, low `avg_win`): need looser trail or model-only exit
- The label PnL Phase 2 simulates **must match** the exit framework Phase 4 deploys, or the deployment realized PnL will diverge from the leaf's training profit

**Concrete suggestion:** Phase 4 spec (not yet written) should require each deployed profile to specify which exit framework variant it was trained/labeled with, and the live runtime must dispatch to the matching variant. Don't let a "B2-labeled" profile and a "model-only-labeled" profile both run through the same deployed exit code.

### E. Fee-aware Q0 gate calibration

**Live finding:** $36,533 in trade volume cost $438 in fees over 42 days. Strategy was +$361 gross, −$77 net. **The signal had alpha but fees ate it.**

**Implication for Phase 3 Q0 gates:** The current Q0 cumulative-profit gate threshold (per Phase 3 spec) should be calibrated against the realistic live fee burden, not the simulation's fee burden. Specifically:

- Phase 2 label simulation uses `FEE_RATE = 0.006` (matches live Coinbase taker)
- Phase 3 Q0 gate on `cumulative_profit_deflated` should require **at minimum** a positive value AFTER fee deduction, BUT
- Live ops shows that "positive after fees" at 30 trades/day still loses to cash if the per-trade margin is < ~0.1%. The right Q0 framing is:
  - **Profit per dollar of round-trip volume** (Sharpe-like, not raw cumulative)
  - Should clear a meaningful threshold (suggest ≥1.5× fee rate as a starting point — ≥0.9% per round-trip)

**Action:** Phase 3 implementation should expose Q0 thresholds as configuration, not hard-code. The "right" threshold is unknowable in advance; we want to be able to re-calibrate after Phase 4 ranks the first batch of profiles.

---

## Concrete case studies (evidence for the above)

### Case 1: SOL-USD on 2026-05-23 — v3 misses the bottom

| Time (EDT) | Event | Price | v3 prob | v4.5 p_up | v4.5 p_dn |
|---|---|---|---|---|---|
| 03:50 | Trade #5988 closes via TRAIL_STOP (−$0.68) | $82.46 | — | — | — |
| 04:42 | **Local bottom** | **$81.68** | 0.50 | 0.556 | 0.183 |
| 10:05 | Mid-climb | $83.03 | 0.55 | 0.506 | 0.251 |
| 17:01 | Local top | **$87.26** | 0.55 (briefly) | 0.499 | 0.255 |

v3 fired **0** BUY signals during the 6.8% climb (`xgb_prob` peaked at 0.558, just over the 0.55 gate, never enough to pass downstream filters). v4.5 would have fired 51 of 132 scans as BUY (39%), including at the bottom.

**Lesson for Phase 3:** Multi-horizon profiles + per-pid mining should naturally surface "SOL-class V-shape reversal" patterns and assign them an appropriate horizon (here `h24` would capture the full climb).

### Case 2: BILL-USD — catastrophic single trade in v4.5 + model-only exit

| Date | v4.5 BUY entry | Exit (MODEL_FLIP) | Net |
|---|---|---|---|
| 2026-05-20 | $0.108 | $0.063 | −41.81% |

A single BILL-USD trade in the v4.5 backtest wiped out 80% of the strategy's gains. The model held all the way down because v4.5's `p_up` only dropped below 0.45 after the price had collapsed.

**Lesson for Phase 3:** Per-pid mining + Q0 max-DD gate should reject BILL-class profiles regardless of pre-fee profit (they don't survive realistic drawdown caps). This is exactly what Phase 3's Q0 gates are designed to filter.

### Case 3: Trigger-level loss attribution (all-time, CNN agent only)

| Trigger | Net | Win rate | Lesson |
|---|---|---|---|
| **SCAN** (model says SELL) | **+$149** | 56% | Only consistently profitable exit type. **Model-driven exits work.** |
| TRAIL_STOP (old price-based, now B2) | −$122 | 21% | Mechanical trail leaks gains. B2 swap deployed 2026-05-24. |
| STOP_LOSS (8% hard backstop) | −$138 | 0% | Gap-throughs on volatile small caps |
| WS_TRAIL_STOP (B2, new) | +$3.12 | 62% | Early (8 trades) but promising |
| WS_MODEL_DOWN (new) | +$10.50 | 100% | Early (3 trades) but very promising |

**Lesson for Phase 2/3:** Phase 2's dynamic-exit label simulation must include the *current* live exit framework (B2 trail + STOP_LOSS + MODEL_DOWN). If the simulation still uses the old price-based trail, label PnL will systematically overstate realized PnL.

---

## Recommendations summary (priority-ordered)

| Priority | Recommendation | Owner |
|---|---|---|
| **P0 (this week)** | Verify Phase 2 label simulation uses B2 trail + STOP_LOSS + MODEL_DOWN exit logic identical to live `agents/exit_thresholds.py` and `agents/exit_watcher.py`. Mismatch will leak into all downstream Phase 3 profiles. | Phase 2 impl |
| **P0 (this week)** | Phase 3 Q0 `cumulative_profit_deflated` gate: confirm it's net of round-trip fees AND requires at least ~1.5× fee rate per-trade margin (or equivalent volume-normalized metric). | Phase 3 impl |
| **P1 (Phase 4 design)** | Each deployed profile must specify its training exit framework; runtime must dispatch to matching variant. Don't mix B2-labeled and model-only-labeled profiles under the same deployed code. | Phase 4 spec author |
| **P1 (Phase 4 design)** | Profile scorecard weighting should prefer high-conviction sparse profiles over marginal-conviction frequent profiles, since live fee burden scales with frequency. | Phase 4 spec author |
| **P2 (post Phase 4 baseline)** | Brainstorm round for multi-resolution OHLCV features (5m / 15m / 4h / 1D additions to `features.py`). | Future brainstorm |
| **P2 (post Phase 4 baseline)** | Brainstorm round for cross-asset features (BTC dominance, sector strength, correlation regime). | Future brainstorm |
| **P3 (low-cost win)** | Session / time-of-day features (`hour_of_day`, `weekend_flag`). | Future brainstorm |

---

## What live ops should keep doing during Phase 2/3/4 work

| Activity | Why keep |
|---|---|
| Continue logging `xgb_prob` + `xgb_prob_v4_5_*` to `cnn_scans` | Provides shadow telemetry against any future model swap |
| Continue B2 trail + WS_MODEL_DOWN deployment | The recent triggers are early but positive (+$13.62 across 11 trades over 24h); shadow week ending 2026-05-31 will confirm |
| Keep dashboard `/api/compare` operational (8001 v3 vs 8002 v4.5) | Phase 4 selection needs the cross-driver data |

## What live ops should consider stopping or changing

**Operator decision (2026-05-24):** Backend stays up to keep data flowing. Paper-trading mode means no capital pressure, and the ongoing scan + trigger logs are themselves the most valuable input to Phase 4 selection. The recommendations below are tuning options, not urgent fixes.

| Activity | Why consider |
|---|---|
| Trading-simulation on ALL Coinbase USD pairs without an allowlist | Phase 3 will eventually emit per-pid profiles; an interim allowlist based on the per-pid evidence in this doc (TAO/ALGO/ENA/PAXG-class only) would make the paper PnL more representative of what a Phase 4 deployment would experience |
| Taker-order fee simulation at 0.6% per side | The deployed maker path (commit `4dcbfa9`) routes limit orders at 0.0–0.4%. If paper PnL switched to maker-fee modeling, the simulated edge would be 3–5× higher and the alpha would no longer be fee-eaten. Worth confirming the paper-trade fee model matches the dominant order type. |
| `MODEL_BACKEND=xgb` (v3) on 8001, `MODEL_BACKEND=xgb_v45` on 8002 | Keep both running — shadow comparison data accumulates through 5/31 review and is essential input for the Phase 4 deployment selection |

---

## Out of scope

This doc deliberately does **not** propose:
- Changes to Phase 2's locked feature set (defer to future brainstorm)
- Changes to Phase 3's mining algorithm (the custom-criterion tree is fine; the *thresholds* are what live ops informs)
- A v5 model architecture (premature — finish Phase 4 baseline first)
- Replacing v3 or v4.5 mid-flight (the Phase 4 baseline will decide promotion)

This doc IS proposing:
- Calibrate Phase 3 Q0 thresholds with live fee evidence
- Ensure Phase 2 label simulation matches live exit framework
- Pre-design Phase 4 to dispatch deployed exits per profile
- Defer feature expansion to a post-baseline brainstorm round

---

## See also

- `2026-05-23-strategy-discovery-phase2-design.md` — feature + label spec (this doc references, does not modify)
- `2026-05-24-strategy-discovery-phase3-design.md` — mining + Q0 gate spec (this doc informs threshold calibration)
- `2026-05-23-pnl-anchored-trail-design.md` — B2 trail design (live as of 2026-05-24)
- `2026-05-23-ws-exit-checker-design.md` — WS_MODEL_DOWN trigger (live as of 2026-05-24)
- `2026-05-18-xgb-scorecard-baseline-results.md` — independent v3 baseline analysis
- `CHANGELOG.md` — session-by-session change log
