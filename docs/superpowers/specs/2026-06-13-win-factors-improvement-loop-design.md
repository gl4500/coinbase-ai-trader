# Win-Factors Improvement Loop — Design Spec

**Date:** 2026-06-13
**Status:** Design complete (research phase done; implementation gated on operator approval)
**Tracker:** `docs/win-factors-strategy/progress.md` (factor ledger + decisions log)
**Memory:** `[[win_factors_improvement_loop]]`, governed by `[[feedback_fail_fast_iterate]]`

---

## 1. Problem

The strategy is net-negative on paper (−$132 realized over 1,830 closed trades). Prior effort chased a better *direction signal* — XGB sits at the ~0.51 AUC no-signal floor and strategy-discovery mining hit ABORT twice. The operator reframed the question: not "which factor predicts direction," but **"what analysis tooling selects the right factors and reacts consistently, and how do we keep improving it?"** — anchored on a repeatable *improvement loop* rather than any single technique, run under a **fail-fast** discipline with a **persistent memory** so each iteration makes better decisions.

## 2. What the research phase found (fail-fast probes, all read-only)

Four cheap probes, logged to the factor ledger, overturned the obvious hypothesis and found the real lever:

| Probe | Finding |
|---|---|
| #1 Recovery diagnostic | 73% of stopped-out losers recover to breakeven+fee within 7d → stops *look* too tight. **But optimistic** (ignores the non-recoverer tail; 23.5% draw down ≤ −20%, worst −70%). |
| #2 Stop re-sim (tail charged, 1,932 entries) | **Widening stops does NOT help on net** — win rate rises 35%→43% but net worsens; the tail eats the recovery. Best policy = time-gate (−20% floor first 24h, then −8%): −$414 vs −$599 flat-−8%. A bleed reducer, not a cure. |
| #3 Fee sensitivity | **Gross expectancy = +0.60%/trade, 67% green before fees. The 1.2% taker fee is the binding constraint.** maker-both (0.4%) flips to +$233. |
| #4 Fill-aware maker sim + robustness | Realistic maker (entry 0.2%, trail-exit 0.2%, stop/maxhold-exit 0.6%) → **net +0.12%/trade, +$169** (full history positive). Robust to micro-caps. ⚠️ **Edge decaying: 1st half maker +$184 vs 2nd half −$16.** |

**Core conclusion:** the factor that gives consistent wins is **execution-cost efficiency**, not direction prediction (a coin-flip with a small structural drift) and not stop tuning (a modest bleed reducer). The win exists in gross terms; it is being donated to taker fees.

## 3. The factor hierarchy (priority order)

1. **Maker / post-only execution** — the dominant lever. Code path already exists (`order_executor.execute_maker_signal`). Strictly improves every period; flips full-history backtest from −$414 to +$169.
2. **Time-gated / vol-aware stop** — secondary bleed reducer (~$185 vs flat −8%). −20% catastrophic floor for the first ~24h (median recovery window), then −8%.
3. **Selectivity / abstention (meta-labeling)** — real role is to *cut trade count and raise average gross drift*, not to predict direction. Becomes important precisely because the gross edge is decaying.

## 4. The improvement loop (durable framework)

A standardized gauntlet every candidate passes through identically, with an append-only ledger as its memory:

```
REGISTER → BUILD → FALSIFY → VERDICT → RECORD → SHADOW
   │                                              │
   └────────── persistent factor ledger ──────────┘
        verdict · trial count N · live-vs-backtest delta
```

- **FALSIFY** is the product: financial gates (net-of-fee expectancy, precision, paper-Sharpe, ECE) under purged-WF/CPCV + DSR/PBO deflation **at the running trial count N** (deflation tightens as N grows — the guard strategy-discovery lacked). **Gates are financial, never AUC.**
- **VERDICT** is mechanical: REJECT / SHADOW / DEPLOY — same judgment every run ("consistent reactions").
- **The ledger has a memory:** trial count N and per-candidate live-vs-backtest deltas, so the loop calibrates its own optimism and compounds learning across sessions.
- Reuses existing components: `tools/_scorecard/*`, `tools/probe_selection_bias.py`, purged-WF harness, 8002→8001 shadow infra.
- **Candidate Protocol supports two shapes:** *rule-candidates* (exit/execution changes, evaluated by trade re-simulation on realized-PnL distribution) and *signal-candidates* (classifiers, evaluated by OOF + CPCV). Both required from day one.

The research probes above were the loop run manually; productizing it (`tools/factor_loop/`) is justified now that there is a proven DEPLOY candidate to house.

## 5. First DEPLOY candidate — maker-execution shadow

- **Change:** route entries and profit-target exits through the existing maker (post-only LIMIT) path; keep disaster stops as taker (they must cross).
- **Validation (cannot be backtested — fill rates are a live property):** run on **port 8002** per port discipline, measuring *actual* maker fill rate, realized blended fee, and missed-entry selection effect, against the 8001 taker baseline. Promote to 8001 only if the shadow confirms the blended fee lands near the modeled 0.4–0.5% and net expectancy is positive.
- **Backtest expectation to beat:** +0.12%/trade net, +$169 full-history (but expect the recent-period marginality from the decay finding).

## 6. Risks / outcome-aware caveats

- **Edge decay** — the gross drift halved between the two halves of history; maker execution is *necessary, not sufficient*. The loop must monitor backtest-vs-live delta and be ready to lean on selectivity (#3).
- **Maker fill uncertainty** — entries may miss (selectively dropping fast movers); profit exits as resting limits may not fill in fast moves. The +$169 assumes neutral misses — shadow must measure the real selection effect.
- **Paper data** — all of this is simulated PnL; no real capital at risk, but also no real slippage/fill realism until live.
- **Intrabar pessimism** — sims assume stop-first on both-touched candles (conservative); real fills could differ either way.

## 7. Scope / YAGNI

- No auto-promotion (operator-gated 8002→8001 per port discipline).
- Implement one rule-candidate (maker execution) first; meta-labeling is candidate #2, built only after the loop is productized and maker is validated.
- Do NOT modify live 8001 trading behavior without the shadow gate + TDD.

## 8. Next step

Implementation is a **live-trading change** requiring operator approval, TDD, and the shadow gate — out of scope for the autonomous research loop. The terminal deliverable of this phase is this spec + the populated factor ledger. On approval, the implementation plan covers: (a) maker-execution shadow wiring + telemetry on 8002, (b) the `tools/factor_loop/` gauntlet+ledger productization, (c) the time-gate stop as a second rule-candidate.
