# Win-Factors Strategy — Brainstorm & Outcomes Tracker

**Started:** 2026-06-13
**Goal:** Identify the *analysis tooling* that selects the right factors and produces consistent, repeatable trading reactions — and a process to keep improving it. Not "which single factor," but the method that chooses factors and adapts.

---

## Grounding facts (from this session's data review)

- **Entry edge is statistically exhausted.** XGB v3/v4.5 sit at the ~0.51 AUC no-signal floor; feature engineering fully extracted (Session 58.71v). Strategy-discovery profile mining hit **ABORT** twice (time-bars 2026-05-28, dollar-bars 2026-06-08) — selection-bias deflation flips raw profit negative.
- **Exit machinery already wins.** On 1,830 closed paper trades: profit-exits (TICK_PROFIT 100%/+$166, TICK_TRAIL 99.5%/+$188, SCAN 55%/+$185) total **≈ +$580**. Stop-exits (TICK_STOP 0%/−$469, STOP_LOSS 0%/−$138, TRAIL_STOP 21%/−$122) total **≈ −$730**. Net **−$132**.
- **Hold-duration is bimodal.** Wins cluster at the extremes (5–60 min scalps win often but tiny; 3–7 d holds win 48% but +3.19% avg via asymmetric tail). The **1–24 h band is the graveyard** (35–43% win, negative) and holds 65% of trades.
- **Fee reality:** 1.2% taker round-trip is the binding economic constraint; live sizing ≈ 15% of cash balance (Kelly clipped by the 0.15 cap, since buy-gate 0.60 ⇒ full-Kelly 0.20 > cap).

**Implication:** consistent wins are a *management/asymmetry/selection* problem, not a *direction-prediction* problem.

---

## Brainstorm checklist (mirrors harness tasks #1–#5)

| # | Item | Status |
|---|---|---|
| 1 | Explore project context | ✅ done (data + history above) |
| 2 | Clarify "factors / consistent wins" → reframed to **analysis-tool selection** | ✅ done (reframe in decisions log 2026-06-13) |
| 3 | Propose 2–3 approaches w/ tradeoffs | ✅ done (Approaches A/B/C; A chosen) |
| 4 | Present design, get approval | ✅ done (Approach A approved 2026-06-13) |
| 5 | Write spec, self-review, user review, → writing-plans | ✅ done (spec `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md`, self-reviewed) |

## Resume state (2026-06-14 — after power interruption)

**Research / brainstorm phase is COMPLETE.** Terminal deliverable = the spec above + this factor ledger (4 probe entries). Nothing lost to the power cut; branch commit `57d7206` (phase-3 progress logging) is pushed.

**Next step (NOT started — needs operator go-ahead):** maker-execution shadow on **port 8002** — route entries + profit-target exits through the existing `order_executor.execute_maker_signal` (post-only LIMIT); keep disaster stops as taker. This is a **live-trading change** requiring TDD + the 8002→8001 shadow gate per port discipline (CLAUDE.md). Backtest target to beat: +0.12%/trade net, +$169 full-history (expect recent-period marginality from the decay finding).

**Superseded — do NOT resume:** phase-3 strategy-discovery mining re-run. The research concluded the edge is in *execution cost*, not entry-signal selection (mining ABORTed ×2). The `feat/phase3-mining-progress-logging` branch was supporting polish, not the live thread.

---

## Factor ledger (every attempt, even throwaway — the memory spine)

### Entry #1 — Step 1 recovery diagnostic (2026-06-13, read-only, throwaway)
**Question:** of stopped-out losers (TICK_STOP n=211 −$469, STOP_LOSS n=51 −$138), how many would have recovered to a profit target within the 7-day hold window if the stop hadn't fired? Reconstructed from `candles` (1h OHLCV) + `entry_price` + `opened_at`. 260/262 analyzed (2 no-coverage).

**Result:**
- Recovery rate within 7d: **+1.2% (breakeven+fee): 73.1%** · +2%: 62.7% · +5%: 41.2%
- Drawdown to endure: median −12.1%, mean −14.6%, **worst −70%**, **23.5% reach ≤ −20%**
- Time-to-recover to +2%: median 23.9h, mean 44.3h
- Crude swing on +2% recoverers: +$166 gain vs −$414 realized → **+$579 (OPTIMISTIC upper bound — does NOT charge deeper losses on the 37% non-recoverers)**

**Verdict: STOPS ARE TOO TIGHT — the problem is (at least partly) exit-side, not purely entry selection.** Majority mean-revert to breakeven. BUT the −20%/−70% drawdown tail forbids naive "remove/widen the stop"; the catastrophic-continuation 37% is what an 8% hard stop (invariant #3) exists to cap. → Fix direction = **volatility-scaled + time-aware stop**, not a flat wider stop. Validate net effect (charging the tail) in Step 2 re-simulation. Meta-labeling correctly demoted to candidate #2.

**Caveats:** peak-touch ≠ realizable fill (needs a resting take-profit); micro-cap composition (sub-cent tokens reverse AND continue violently); paper data; +$579 is an upper bound only.

### Entry #2 — Step 2 stop re-simulation (2026-06-13, read-only, tail charged)
**Method:** re-simulate ALL 1,932 long entries (same entries, vary only the hard stop) forward over 1h candles; common trailing TP (1.5% giveback after +1%); intrabar both-touched → stop-first (pessimistic); fee 1.2%.
**Result (avg_net / sum_usd):** hard −4% −0.73%/−$534 · −8% −0.80%/−$599 · −12% −0.89%/−$662 · floor −20% −0.95%/−$666 · vol 2.5·ATR −0.78%/−$563 · **time-gate (−20% floor first 24h, then −8%) −0.60%/−$414 ← best**.
**Verdict: widening stops does NOT help on net.** Win rate rises with width (35%→43%, more recovery) but net WORSENS — the non-recoverer tail costs more than recoverers gain. **Step 1's +$579 was the optimistic upper bound; charging the tail makes every policy negative.** time-gate is the best exit (−$414 vs −$599 flat-−8%, ~$185 leak reduction) but is a bleed-reducer, not a cure.

### Entry #3 — Fee sensitivity on time-gate policy (2026-06-13, read-only) — **PIVOTAL**
**GROSS expectancy = +0.60%/trade, 67.2% of trades green before fees.** The 1.2% taker fee is what makes it negative.
| Fee scenario | round-trip | avg_net | win% | sum_usd |
|---|---|---|---|---|
| taker both (current) | 1.2% | −0.60% | 41.6% | −$414 |
| maker entry + taker exit | 0.8% | −0.20% | 49.3% | −$91 |
| **maker both** | **0.4%** | **+0.20%** | **57.0%** | **+$233** |
| zero-fee ceiling | 0.0% | +0.60% | 67.2% | +$556 |

**Verdict: the binding factor is EXECUTION COST, not signal or stop.** The strategy has a small real positive gross drift; fees donate it away. Maker entry alone (1.2%→0.8%) swings −$414 → −$91 (+$323). **Caveat (outcome-aware): maker fills are not guaranteed — disaster stops MUST cross (taker), so "maker both" (0.4%) is an idealized ceiling; realistic = maker entry + maker exit on restable profit-targets + taker on stops, landing between 0.4–0.8% → near break-even.** Matches the scorecard's independent "positive only at pro/maker tier."

**Reframe of the whole effort:** the "factor that gives consistent wins" = (1) **fee efficiency / maker execution** (biggest lever; code path already exists in `order_executor.execute_maker_signal`), (2) time-gated/vol-aware stop (bleed reducer), (3) selectivity/abstention (meta-labeling's real role = cut trade count, raise avg drift). Direction *prediction* is NOT a winnable factor here.

### Entry #4 — Fill-aware maker sim + robustness (2026-06-13, read-only) — **CONFIRMS + TEMPERS**
**Realistic maker model:** entry 0.2% (maker); exit 0.2% if via trailing-TP (restable limit), 0.6% if via stop/max-hold (must cross). Exit mix: trail 1541 / stop 245 / maxhold 146.
- **ALL (n=1932): gross +0.60% → realistic-maker net +0.12%/trade, 56.8% win, +$169.** Maker execution flips the full-history backtest POSITIVE (vs −$414 taker).
- **Robust to micro-caps:** price≥1¢ (n=1897) net +0.10%/+$153; sub-cent (n=35) +0.99%/+$15 — edge is NOT a micro-cap artifact.
- ⚠️ **EDGE IS DECAYING:** 1st half gross +0.92% (maker +$184) vs **2nd half gross +0.27% (maker −$16)**. Recent period is break-even-to-slightly-negative even WITH maker.

**Verdict:** maker execution is the correct #1 lever (strictly improves every period, flips full history to +$169) but is **necessary, not sufficient** — the underlying gross drift is weakening, so durability needs selectivity (#3) + live monitoring of the backtest-vs-live delta. Caveat: maker fill rates are a live-market property; +$169 assumes maker entries fill (neutral on missed) and winners exit as restable limits — must be validated by shadow, not asserted.

## Decisions / outcomes log

- **2026-06-13:** User rejected the win-rate / expectancy / smoothness framing as "short-sighted." Reframed the question to: *what analysis tools choose the right factors and give consistent reactions?* → pivoting brainstorm to the **methodology / factor-selection-engine** layer.
- **2026-06-13:** Tool anchor chosen = **Process / improvement loop** (the repeatable propose→falsify→keep→re-select cycle), over any single technique. Rationale: the loop is the meta-tool that makes meta-labeling / regime selection trustworthy rather than one-off overfits.

---

- **2026-06-13:** Loop structure chosen = **Approach A** (standardized gauntlet + persistent ledger), **with meta-labeling as the first test subject**. B (meta-labeling factory) and C (full automation) deferred until A's gauntlet proves it calibrates well. Rationale: A is the foundation B/C both need; lowest risk; mostly wires together existing components (scorecard, probe_selection_bias, shadow infra).

---

## Operating philosophy (operator directive 2026-06-13)

**Fail fast → learn → iterate → succeed**, with a **persistent memory** so every iteration makes better decisions. See [[feedback_fail_fast_iterate]]. The memory (ledger) is the *spine* of the loop, not a component. Cheap falsification up front; log every attempt's verdict; consult prior outcomes (trial count N, live-vs-backtest deltas) before each decision; never loosen an honest falsifier.

## Design — revised to phased / fail-fast (exit-side first)

Two corrections applied after self-review: (a) Phase gates are **financial (net-of-fee expectancy), never AUC** — AUC only as a cheap *kill* filter; (b) first candidate reordered to the **cheap exit-side experiment** (attacks the quantified −$730 stop bleed), meta-labeling becomes candidate #2. This is strictly more fail-fast than meta-labeling-first.

| Step | What | Gate | Status |
|---|---|---|---|
| **1 — Recovery diagnostic** (hours, throwaway) | Of the stopped-out losers (TICK_STOP −$469, STOP_LOSS −$138), how many would have recovered to a profit target within the hold window? Use price parquets + entry timestamps. No model. Log verdict to ledger. | Most recovered → stop too tight (config fix, huge payoff). Most kept falling → bleed unavoidable, problem is upstream entries → meta-labeling earns its place. **Either answer is decisive.** | 🔄 designing |
| **2 — Exit re-simulation** (only if Step 1 = "too tight") | Re-simulate forward price path with alternative stops (wider / vol-scaled / time-gated to skip 1–24h graveyard). | Net-of-fee expectancy lift vs current rule | ⬜ |
| **3 — Productize loop + meta-labeling #2** | Build `tools/factor_loop/` (gauntlet + ledger); candidate Protocol supports **rule-candidates AND signal-candidates**. Bring meta-labeling through (labels = triple-barrier over ALL signals taken+skipped; bet = structure in exit/regime conditioning). Falsify = purged-WF + CPCV + financial scorecard + DSR/PBO deflation at trial-count N. | Mechanical REJECT/SHADOW/DEPLOY | ⬜ |

**Why exit-first:** each step is cheaper than the next and gates it; attacks a *quantified* loss instead of hunting exhausted edge; and the Step-1 diagnostic answers *"is the problem exits or entries?"* — which tells us whether meta-labeling is even worth building, before spending a day on it.
