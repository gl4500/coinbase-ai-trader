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
| C1a | **Maker execution — entry leg** | rule | ✅ **SHIPPED** `c262efc` (default-off flag) | done — ledger #3/#4 (+$169 fill-aware) | live 8002 shadow: real maker **fill rate** + blended fee ≤ ~0.5% on entries |
| C1b | **Maker execution — profit-target exit leg** | rule | ⛔ gated on C1a shadow proving entries fill | n/a — depends on C1a shadow telemetry | live 8002 shadow: blended round-trip fee ≤ ~0.5% **and** net > taker baseline |
| C2 | **Time-gated / vol-aware stop** | rule | ✅ unblocked | done — ledger #2 (time-gate −$414 best of the stop sweep) | re-sim **with maker fees layered**: net-of-fee expectancy > current flat −8% stop, purged-WF |
| C3 | **Selectivity / meta-labeling** | signal | ✅ unblocked | Phase-0 pulse: triple-barrier labels over **all** primary signals (taken+skipped), quick meta-labeler P(win) on regime/vol/RSI/ADX/model_prob | OOS AUC clears 0.5 by a meaningful margin under purged-WF; **else REJECT** (acceptable, informative) |
| C4 | **CUSUM event filter** (roadmap #20) | sampling | ✅ unblocked | build CUSUM event series on top-20 pids; count events; estimate K-deflation reduction | deflation factor at event-sampled N materially below time-bar N at equal coverage |
| C5 | **Sample-uniqueness weighting** (roadmap #18) | sampling | ✅ unblocked | compute label-overlap weights on existing samples; measure effective-N shrink | effective N (and thus deflation) drops without harming OOS expectancy |
| C6 | **Regime-conditional XGB** (roadmap #23) | signal | ✅ unblocked | segment OOF returns by HMM/vol regime; check per-regime expectancy dispersion | at least one regime shows net-of-fee positive expectancy that the blended model hides |

---

## Why this ordering

- **C1a (entry leg) is shipped and now awaits the operator's 8002 shadow** — fills are a live property, not backtestable, so the next gate is measurement, not more code. **C1b (exit leg) is deliberately sequenced *after* the entry-leg shadow proves entries fill** — building the net-new exit-side routing before knowing entry fill rates would be premature (if maker entries miss badly, the whole maker thesis weakens and C1b's design changes). Fail-fast: validate the cheap shipped thing before building the next thing.
- **C2 and C3 are the live worklist** for the loop's ideation: both have cheap, read-only probes and attack the two quantified levers behind C1 (stop bleed; trade-count selectivity). C2 most directly tightens the +$169 number by pairing the best stop with maker fees; C3 is the decaying-edge insurance (its real job is to *cut trade count*, not predict direction).
- **C4–C6 attack the deflation gate at its source** — the wall that ABORTed strategy-discovery twice. They are research-grade, lower-confidence, and queued behind C2/C3.

## C1a entry-leg — 8002 shadow validation checklist (the next gate, operator-run)

Launch: `USE_MAKER_EXECUTION=true PORT=8002 python main.py` from `backend/` (8001 stays the taker baseline). Then, over a shadow window, decide from telemetry — **not** from backtest:

| Metric | Where | PASS signal |
|---|---|---|
| **Maker fill rate** | `orders` rows with `fill_mode="MAKER"` vs `"TAKER_FALLBACK"` | high enough that the blended entry fee lands near 0.2–0.5%, not creeping back to 0.6% taker |
| **Missed-entry selection** | entries that fell to `TAKER_FALLBACK` after the 30s poll | fallbacks are not concentrated on the fast movers (which would selectively drop the winners) |
| **Blended entry fee** | realized fee on filled entries | ≤ ~0.5% (the modeled maker-entry assumption) |
| **No 8001 disruption** | 8001 scan loop + telemetry unaffected | default-off flag holds; 8001 behavior byte-for-byte unchanged |

If entries fill well → green-light **C1b** (profit-target maker exits, keep disaster stops taker). If fills are poor or adversely selected → the maker thesis weakens; pivot to **C2** (stop) / **C3** (selectivity) instead.

## Loop log (iteration → artifact)

- **2026-06-14 · iteration 1:** created this backlog from the design-doc factor hierarchy (§3) + the post-scorecard roadmap. No probes executed (read-only/docs-only). Next iteration: deepen **C2** — draft the maker-fee-layered stop re-sim *design* (method, data, gate, expected result), still documentation-only; execution stays operator-gated.
- **2026-06-15 · iteration 2:** operator approved active building (run pytest+commit alongside live 8001). **C1a entry leg SHIPPED** via TDD on `feat/maker-execution-shadow` (`c262efc`, full suite 1284 passed) — gated `USE_MAKER_EXECUTION` flag, sources bid/ask, routes to `execute_maker_signal`. Split C1 → C1a (shipped) / C1b (exit, gated on the entry-leg shadow). Added the 8002 shadow validation checklist above as the next gate. **Deliberately did NOT pre-build the C1b exit design** — premature before the entry-leg shadow measures real fills. Next: when the operator runs the shadow, read its telemetry → green-light C1b or pivot to C2/C3.
- **2026-06-15 · iteration 3:** drafted [`c2-stop-resim-design.md`](c2-stop-resim-design.md) — the maker-fee-layered stop re-sim that tests whether the time-gate stop (ledger #2's winner at the taker tier) still adds net **on top of** maker execution, or whether maker absorbs the benefit. Design only; the cheap read-only probe is the next executable step. **End of the clearly-safe autonomous doc backlog** — remaining moves need the operator (run the 8002 shadow, or OK executing the C2/C3 probes against the live DB).

## See also
- [`progress.md`](progress.md) — the factor ledger (past attempts #1–#4) + decisions log
- [`maker-execution-readout.html`](maker-execution-readout.html) · [`maker-execution-sequence.html`](maker-execution-sequence.html) — visual readouts of C1
- `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md` — the loop framework spec (committed `fa161a4`)
