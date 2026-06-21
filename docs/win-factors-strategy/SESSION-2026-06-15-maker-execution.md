# Session Summary — Maker Execution + Win-Factors Loop (2026-06-14 → 06-15)

**Trigger:** resumed after a power interruption. Reconstructed in-flight state from git + persistent
memory + the in-repo tracker, then shipped the win-factors research deliverables and the first
implementation lever (maker execution, entry leg) under a continuous-improvement `/loop`.

**Branches / PRs:**
- `docs/win-factors-improvement-loop` → **PR #21** — all win-factors documentation
- `feat/maker-execution-shadow` → entry-leg implementation (pushed; PR not yet opened)

---

## What happened (chronological)

1. **Power-loss recovery.** Confirmed nothing was lost — the last commit (`57d7206`, phase-3 progress
   logging) was already pushed; in-flight files were on disk. Reconstructed state from
   [`progress.md`](progress.md), memory `win_factors_improvement_loop`, and the session log.
2. **Committed the win-factors research deliverables** (`fa161a4`): the improvement-loop design spec +
   the factor ledger (4 read-only probe entries, 2026-06-13). Conclusion: **execution-cost efficiency
   (maker / post-only execution) is the #1 lever**, not direction prediction or stop tuning.
3. **Visual readouts** (`2fb2e9b`): two self-contained HTML pages —
   [`maker-execution-readout.html`](maker-execution-readout.html) (the −$414 → +$169 fee case) and
   [`maker-execution-sequence.html`](maker-execution-sequence.html) (a faithful execution trace of
   `execute_maker_signal()` read from `order_executor.py:285–424`).
4. **Continuous-improvement `/loop`** (self-paced, dynamic mode). Scope started read-only, then the
   operator approved active building (running pytest+commit alongside the live 8001 paper backend —
   the `feedback_no_pytest_during_trading` "ask first" gate was satisfied).
5. **Shipped the maker-execution entry leg** (`c262efc`) via TDD — see deliverables below.
6. **Loop documentation iterations:** candidate backlog → 8002 shadow validation checklist → C2 design.
7. **Paused the loop** at the end of the clearly-safe autonomous doc backlog; remaining steps are
   operator-gated (run the shadow, or OK executing the probes).

---

## Deliverables

### Code — `feat/maker-execution-shadow` (`c262efc`, full suite 1284 passed)
Opt-in maker (post-only LIMIT) routing for live BUY entries, gated behind `USE_MAKER_EXECUTION`
(default **false** → byte-for-byte unchanged taker behavior).
- `backend/config.py` — `use_maker_execution` flag.
- `backend/agents/cnn_agent.py` — `_execute_live_order()`: flag on → sources best bid/ask via
  `coinbase_client.get_best_bid_ask`, attaches to the signal (the maker path requires them), routes to
  `execute_maker_signal`; flag off → taker `execute_signal`.
- `backend/tests/test_cnn_agent.py` — `TestMakerExecutionRouting` (2 tests, TDD red→green verified).
- `CHANGELOG.md` (Session 58.72) + `CLAUDE.md` invariant #21.
- **Key build discovery:** the `signal` dict carried no bid/ask, which the maker path requires — so
  sourcing live quotes IS the entry-leg work; a naive routing swap would have silently no-op'd.

### Docs — `docs/win-factors-improvement-loop` / PR #21
| Commit | File(s) |
|---|---|
| `fa161a4` | `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md`, `progress.md` (factor ledger) |
| `2fb2e9b` | `maker-execution-readout.html`, `maker-execution-sequence.html`, `candidate-backlog.md` |
| `9968579` | `candidate-backlog.md` — C1a shipped, C1b gated, **8002 shadow validation checklist** |
| `9633893` | `c2-stop-resim-design.md` — maker-fee-layered stop re-sim design |
| (this doc) | `SESSION-2026-06-15-maker-execution.md` |

### Memory (persistent, outside the repo)
- `coinbase_trader_session_log` — Session 58.72 entry (the maker entry leg).
- `win_factors_improvement_loop` — status block: research phase complete, entry leg shipped, next = shadow.

---

## Current state — where it rests

- **Entry leg:** built, tested, pushed. Default-off → live 8001 untouched. Working tree checked out on
  `feat/maker-execution-shadow`, shadow-ready.
- **The #1 open step is the operator's:** launch the 8002 maker shadow
  (`USE_MAKER_EXECUTION=true PORT=8002 python main.py`) to measure real maker fill rates — not
  backtestable. Decision criteria in `candidate-backlog.md` → "8002 shadow validation checklist."
- **Loop:** paused (no scheduled wakeup), not killed. Tasks: #2 done, #3 paused-pending-operator.

## Open decisions / next steps
1. **Run the 8002 shadow** → telemetry green-lights **C1b** (profit-target maker exits) or pivots to C2/C3.
2. **OK executing the C2 probe** (read-only stop re-sim per `c2-stop-resim-design.md`) — flagged for
   review rather than blind autonomous run, to keep the honest-falsifier ledger clean.
3. **Open the `feat/maker-execution-shadow` PR** when ready.

## Guardrails respected
- Port discipline: maker path is for the **8002 shadow**; 8001 promotion gated on shadow evidence.
- `no-pytest-during-trading`: asked first; operator approved running tests/commit alongside live 8001.
- TDD: red→green verified before every code commit; docs-only commits skip the pre-commit suite.
- Surgical pathspec commits; branch discipline (feature work off `main`).

## See also
- [`candidate-backlog.md`](candidate-backlog.md) · [`c2-stop-resim-design.md`](c2-stop-resim-design.md) · [`progress.md`](progress.md)
- `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md` — the loop framework spec
