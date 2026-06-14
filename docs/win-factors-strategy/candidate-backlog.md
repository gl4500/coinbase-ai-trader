# Win-Factors — Improvement-Candidate Backlog

**Purpose:** the forward-looking worklist for the continuous-improvement loop. The
[`progress.md`](progress.md) factor ledger records *past* attempts (#1–#4, all read-only sims, 2026-06-13);
this file records *future* candidates queued for the gauntlet. Governed by the operating philosophy
**fail fast → learn → iterate → succeed** with a persistent ledger (see memory `feedback_fail_fast_iterate`).

**Loop discipline (this is documentation only):**
- Every candidate must name a **cheap probe first** (hours, read-only) and a **falsifiable financial gate** (net-of-fee, never AUC).
- Gates tighten with the running trial count N (DSR/PBO deflation) — the guard strategy-discovery lacked.
- A candidate that *matures to "build it"* leaves this loop and enters the full brainstorm → spec → operator-approval → plan gate. Nothing here authorizes a live-trading or code change.
- Verdict vocabulary is mechanical: **REJECT / SHADOW / DEPLOY**.

---

## Priority queue (cheapest + unblocked first)

| # | Candidate | Type | Blocked? | Cheap probe | Falsifiable gate |
|---|---|---|---|---|---|
| C1 | **Maker / post-only execution** | rule | ⛔ operator approval (live-trading) | done — ledger #3/#4 (+$169 fill-aware) | live 8002 shadow: real blended fee ≤ ~0.5% **and** net > taker baseline |
| C2 | **Time-gated / vol-aware stop** | rule | ✅ unblocked | done — ledger #2 (time-gate −$414 best of the stop sweep) | re-sim **with maker fees layered**: net-of-fee expectancy > current flat −8% stop, purged-WF |
| C3 | **Selectivity / meta-labeling** | signal | ✅ unblocked | Phase-0 pulse: triple-barrier labels over **all** primary signals (taken+skipped), quick meta-labeler P(win) on regime/vol/RSI/ADX/model_prob | OOS AUC clears 0.5 by a meaningful margin under purged-WF; **else REJECT** (acceptable, informative) |
| C4 | **CUSUM event filter** (roadmap #20) | sampling | ✅ unblocked | build CUSUM event series on top-20 pids; count events; estimate K-deflation reduction | deflation factor at event-sampled N materially below time-bar N at equal coverage |
| C5 | **Sample-uniqueness weighting** (roadmap #18) | sampling | ✅ unblocked | compute label-overlap weights on existing samples; measure effective-N shrink | effective N (and thus deflation) drops without harming OOS expectancy |
| C6 | **Regime-conditional XGB** (roadmap #23) | signal | ✅ unblocked | segment OOF returns by HMM/vol regime; check per-regime expectancy dispersion | at least one regime shows net-of-fee positive expectancy that the blended model hides |

---

## Why this ordering

- **C1 is highest-value but blocked** on operator approval + a live shadow — it cannot run inside an autonomous read-only loop, so it sits at the head as the *known DEPLOY-grade* candidate awaiting the human gate, not as loop work.
- **C2 and C3 are the live worklist** for the loop's ideation: both have cheap, read-only probes and attack the two quantified levers behind C1 (stop bleed; trade-count selectivity). C2 most directly tightens the +$169 number by pairing the best stop with maker fees; C3 is the decaying-edge insurance (its real job is to *cut trade count*, not predict direction).
- **C4–C6 attack the deflation gate at its source** — the wall that ABORTed strategy-discovery twice. They are research-grade, lower-confidence, and queued behind C2/C3.

## Loop log (iteration → artifact)

- **2026-06-14 · iteration 1:** created this backlog from the design-doc factor hierarchy (§3) + the post-scorecard roadmap. No probes executed (read-only/docs-only). Next iteration: deepen **C2** — draft the maker-fee-layered stop re-sim *design* (method, data, gate, expected result), still documentation-only; execution stays operator-gated.

## See also
- [`progress.md`](progress.md) — the factor ledger (past attempts #1–#4) + decisions log
- [`maker-execution-readout.html`](maker-execution-readout.html) · [`maker-execution-sequence.html`](maker-execution-sequence.html) — visual readouts of C1
- `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md` — the loop framework spec (committed `fa161a4`)
