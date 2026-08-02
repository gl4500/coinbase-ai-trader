# C2 — Time-Gated / Vol-Aware Stop, Maker-Fee-Layered Re-Sim (design)

**Candidate:** C2 in [`candidate-backlog.md`](candidate-backlog.md). **Type:** rule-candidate (exit-side).
**Status:** design only (read-only). Execution is the next loop tick, pending operator OK to run a
cheap probe alongside live 8001. Governed by **fail fast → log every verdict** (memory `feedback_fail_fast_iterate`).

## The question

Ledger #2 found the **time-gate stop** (−20% floor for the first 24h, then −8%) was the best of the
stop sweep — **−$414 vs −$599** for flat −8% (~$185 leak reduction) — but that re-sim charged the
**taker** fee (1.2% round-trip). Ledger #3/#4 then showed **execution cost is the dominant lever**
(maker flips full history −$414 → +$169).

So the open question is an **interaction effect**: *does the stop improvement still add net once maker
fees are already applied — or does maker execution absorb most of the benefit?* A bleed-reducer that
only helps at the expensive taker tier is far less interesting than one that stacks on top of maker.

## Method

Same shape as ledger #2/#4 — hold the entries fixed, vary only the stop policy and the fee model.

- **Entries:** the 1,932 long paper entries (DB `trades`: `entry_price`, `opened_at`, `product_id`).
- **Forward path:** `candles` (1h OHLCV) over the 7-day hold window per entry.
- **Stop policies (vary only this):**
  1. flat −8% — current production (CLAUDE.md invariant #3)
  2. time-gate −20%/24h → −8% — ledger #2's winner
  3. vol-aware 2.5·ATR
  4. (reference) flat −12%, −20% floor
- **Common exit:** trailing TP, 1.5% giveback after +1% (matches ledger #2).
- **Fee model — the new axis (run every policy under BOTH):**
  - **taker** 1.2% round-trip (reproduces ledger #2, the control)
  - **realistic maker** (ledger #4 model): entry 0.2%, profit-target exit 0.2% (restable limit),
    **stop / max-hold exit 0.6%** (must cross — taker). This is the maker-aware tier.
- **Intrabar pessimism:** both-touched candle → stop-first (conservative).
- **Metrics per (policy × fee model):** avg net %/trade, win%, sum_usd. Plus split-half (the ledger #4
  decay check) so a verdict isn't a single-period artifact.

## Falsifiable gate

Under the **maker** fee model: does any stop policy beat **flat-−8% + maker** on net-of-fee
expectancy by a margin that survives the split-half / purged-WF robustness check?

- **PASS →** C2 is a real *additive* lever on top of C1; promote to an implementation design
  (vol/time-aware stop in `agents/exit_thresholds.py`, which already owns the trail/floor logic).
- **FAIL →** maker execution absorbs the stop benefit; C2 is demoted, effort stays on C1 (maker) + C3
  (selectivity). **This is an acceptable, informative outcome** — log it and move on.

## Honest prior

Expect **modest**. Maker already removes much of the fee drag the wide-stop recovery was fighting; the
time-gate's real edge is *avoiding the 1–24h graveyard*, which is a path effect independent of fees, so
**some** additive net is plausible — but it was always "a bleed reducer, not a cure," and the maker-tier
gap between policies will likely shrink. Plausible verdict: within noise → FAIL/DEMOTE. Either answer is
decisive and cheap.

## Caveats

Paper data; peak-touch ≠ realizable fill (needs a resting TP); reconstructed from 1h candles;
micro-cap reversal/continuation violence; same caveat set as ledger #1–#4. The maker exit assumption
(profit-targets fill as restable limits) is the same live-property uncertainty the C1a 8002 shadow
exists to measure — a maker-tier C2 verdict inherits that dependency.

## See also
- [`candidate-backlog.md`](candidate-backlog.md) — C2 row + priority rationale
- [`progress.md`](progress.md) — factor ledger #1–#4 (the prior stop + fee sims this extends)
- [`maker-execution-readout.html`](maker-execution-readout.html) — the C1 fee-lever case this stacks on
