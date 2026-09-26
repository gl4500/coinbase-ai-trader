# Strategy prerequisites — Claude handoff

**Status:** Tasks 1–6 COMPLETE. Task 7 (this document) complete. **Nothing pushed, merged, or
deployed. No migration applied to any production database.**
**Owner:** Claude Code session (interactive)
**Coordinating with:** separate Codex session (manual coordination)
**Date:** 2026-09-26

---

## 1. Workspace

| Item | Value |
|---|---|
| Branch | `feat/outcome-label-provenance` |
| Worktree | `C:\Users\gl450\polymarket_app\.claude\worktrees\outcome-labels` |
| Base branch | `main` |
| Starting commit | `26dddc28ffb301fc577e66f29ec1095a5560afc8` ("Merge pull request #51 from gl4500/chore/snyk-blocking-gate") |
| **Final commit** | **see §8 — filled in after the commit landed** |
| Python | `3.11.13` via `.venv` junction → `C:\Users\gl450\polymarket_app\.venv` |

Created with `git worktree add -b feat/outcome-label-provenance .claude\worktrees\outcome-labels 26dddc2`,
matching the `.claude/worktrees/` convention already used by `maker-exit-leg`.

### Task-tracking substitution

`CLAUDE.md` §Find-List-Fix mandates `TaskCreate` for every distinct issue. **That tool is not
available in this session.** Substitute: the numbered register in §4. It preserves the required
semantics — all issues listed before any fix, fixed in order, nothing fixed silently, tests run after
each fix. Flagged because it is a deviation from the written contract.

---

## 2. Pre-existing state at start (preserved, not modified)

### Worktrees

| Path | Branch | Owner |
|---|---|---|
| `polymarket_app` | `main` | shared; Codex's audit ran here |
| `.claude/worktrees/maker-exit-leg` | `feat/macro-regime-layer` | **this Claude session, earlier task** |
| `.wt-dbar` | `feat/dollar-bar-strategy-discovery` | earlier session |
| `.claude/worktrees/outcome-labels` | `feat/outcome-label-provenance` | **this task** |

### Unpushed local work — PRESERVED

- **`feat/macro-regime-layer` is unpushed and must not be deleted.** It carries the macro-regime
  Phase 1 work (offline layer, gate verdict INCONCLUSIVE) plus a merge of `origin/main`, ending at
  `61bb1c8`. Untouched by this task: different branch, different worktree.
- `feat/pnl-anchored-trail` (behind 118) and `fix/price-chart-dropdown` (behind 67): stale, untouched.
- `main`: no unpushed commits.

### Codex artifacts and other untracked work — NOT TOUCHED

`artifacts/strategy_audit_2026_09_26/` (incl. the 110 MB `extract.json` + `extract.sha256`),
`docs/audits/2026-09-26-strategy-audit-{plan,report}.md`, and `.agents/skills/` (CrewAI Agent Skills)
are all **untracked in the main worktree**. They were read, never edited. Because they are untracked
they do not appear in this worktree; the audit documents were read from the main worktree path.

---

## 3. Overlap analysis — before editing anything

| File / area | Codex touched it? | Decision |
|---|---|---|
| `docs/audits/*`, `artifacts/strategy_audit_2026_09_26/*` | Yes — authored, frozen | **Never edit.** Read-only input. |
| `backend/services/outcome_tracker.py` | No (audit was read-only) | Mine (Task 3) |
| `backend/database.py` (`signal_outcomes`) | No | Mine, additive migration only |
| `backend/services/diagnostics.py` | No | Mine (Task 4) |
| `backend/agents/{cnn_agent,order_executor,exit_*}.py` | No | **Investigate only** (Task 5); no semantic change |
| `backend/database.py` (other regions) | n/a | **Known self-overlap:** `feat/macro-regime-layer` appends `regime_state` helpers at end of file; this branch edits the `signal_outcomes` region and the migration list. Both additive; whoever merges second resolves mechanically. |
| `CHANGELOG.md` | n/a | **Session numbering:** `main` ends at 58.81; `feat/macro-regime-layer` claims **58.82**; this branch claims **58.83**. No collision, but the second to merge should re-check. |

---

## 4. Issue register (substitutes for TaskCreate)

| # | Area | Issue | Status |
|---|---|---|---|
| 1 | `outcome_tracker.check_pending` | Resolved delayed outcomes with the *current* price | **FIXED** (Task 3) |
| 2 | `outcome_tracker.check_pending` | `get_candles(limit=1)` can return an **incomplete** bar | **FIXED** (Task 3) |
| 3 | `signal_outcomes` schema | No label version / target time / price timestamp / provenance | **FIXED** (additive migration) |
| 4 | `outcome_tracker.check_pending` | Unresolvable rows retried forever; no terminal state | **FIXED** (attempts + `UNAVAILABLE`) |
| 5 | `outcome_tracker.check_pending` | `lesson_text` asserted "after 4h" regardless of actual delay | **FIXED** (now true by construction) |
| 6 | `diagnostics.py` | Accuracy pooled legacy + new labels | **FIXED** (Task 4) |
| 7 | `diagnostics.py` | Confidence deciles presented as calibration against a mismatched target | **FIXED** (suppressed + reason) |
| 8 | `tests/test_diagnostics.py` | Stale fixture/assertions assumed version-mixed pooling and calibration buckets | **FIXED** (updated; behaviour change was mandated, not a product bug) |
| 9 | `database.insert_signal_outcome` | Referenced an unimported module alias → `NameError` on the default (`signal_time` omitted) path — i.e. exactly how production calls it | **FIXED** + regression test added. Caught by ruff F821, *not* by my tests, because every test passed `signal_time` explicitly. |
| 10 | `main.py` / `exit_watcher` | Executor captured by value; `enable_trading` rebinding cannot reach it | **CONFIRMED, NOT FIXED** — Task 5 is investigate-only; fix proposed |
| 11 | `exit_execution.execute_live_exit` | Exits no-op while the maker flag is off, but entries route live — asymmetric gate | **CONFIRMED, NOT FIXED** — documented as intentional (invariant #21); needs operator decision |
| 12 | exit paths | Paper book closes before exchange confirmation | **CONFIRMED, NOT FIXED** — invariant #21; needs schema work |
| 13 | `order_executor.execute_maker_signal` | Market fallback runs after a failed cancel; cancel response never inspected; partial fill treated as no fill | **FIXED** 2026-09-26 on `fix/maker-fallback-cancel-confirm` — fallback now requires a confirmed cancel and is sized to the unfilled remainder |
| 14 | `tests/test_order_executor_maker.py` | `test_timeout_cancels_and_falls_back_to_market` pinned the old unconditional fallback (mocked `get_orders` as permanently OPEN) | **FIXED** — same intent, updated so the exchange confirms the cancel after `cancel_orders` |
| 15 | `tests/test_execution_findings.py` | The three finding-4 characterisation tests pinned the defect they were written to expose | **FIXED** — deliberately inverted; they now pin the corrected behaviour |

---

## 5. Task status

| Task | Description | Status |
|---|---|---|
| 1 | Isolated workspace + handoff | **COMPLETE** |
| 2 | Outcome-label contract specification | **COMPLETE** — `docs/specs/2026-09-26-outcome-label-contract.md` |
| 3 | Versioned, correctly timed outcomes | **COMPLETE** |
| 4 | Non-misleading diagnostics | **COMPLETE** |
| 5 | Reproduce 4 execution findings | **COMPLETE** — all 4 CONFIRMED; `docs/handoffs/2026-09-26-execution-findings.md` |
| 6 | Accounting reconciliation | **COMPLETE** — `docs/handoffs/2026-09-26-accounting-reconciliation.md` |
| 7 | Reviewable handoff | **COMPLETE** (this document) |

### Deliberately NOT done

- No model retrained, no threshold touched, no strategy promoted.
- No live-execution semantics changed (Task 5 boundary).
- **Migration not applied to any production database.** `coinbase.db` verified still has no
  `label_version` column.
- Legacy labels not recomputed.
- Nothing pushed, merged, or deployed.

---

## 6. Changed files

### New

| File | Purpose |
|---|---|
| `backend/services/outcome_labels.py` | Pure label math, no I/O |
| `backend/tests/test_outcome_labels.py` | 17 tests — timing, boundaries, direction, retries |
| `backend/tests/test_outcome_store_v2.py` | 15 tests — migration, provenance, idempotency |
| `backend/tests/test_outcome_tracker.py` | 9 tests — resolver end-to-end (module had **no** coverage before) |
| `backend/tests/test_diagnostics_label_versions.py` | 9 tests — version separation + denominators |
| `backend/tests/test_execution_findings.py` | 9 characterisation tests for the 4 findings |
| `docs/specs/2026-09-26-outcome-label-contract.md` | Task 2 contract |
| `docs/handoffs/strategy-prerequisites.md` | This handoff |
| `docs/handoffs/2026-09-26-execution-findings.md` | Task 5 |
| `docs/handoffs/2026-09-26-accounting-reconciliation.md` | Task 6 |

### Modified

| File | Change |
|---|---|
| `backend/database.py` | 12 additive `signal_outcomes` columns; `insert_signal_outcome` stamps the v2 schedule; pending queue matures on `target_time`; 5 new accessors |
| `backend/services/outcome_tracker.py` | `check_pending` rewritten onto the contract; live-price paths removed; `record` stamps `signal_time` |
| `backend/services/diagnostics.py` | `signal_edge` version-scoped + counts + legacy block + calibration suppression; `signal_funnel.matured` version-scoped |
| `backend/tests/test_diagnostics.py` | Fixture + assertions updated for the versioned contract (issue #8) |
| `CLAUDE.md` | New invariant **#22** (outcome-label version contract) |
| `CHANGELOG.md` | Session 58.83 entry |

### Migration instructions

The migration is **additive and idempotent** — 12 `ALTER TABLE signal_outcomes ADD COLUMN`
statements inside the existing `init_db` try/except list. It applies automatically the next time
`init_db()` runs against a database.

```powershell
# 1. ALWAYS back up first
Copy-Item backend\coinbase.db backend\coinbase.db.bak-2026-09-26

# 2. Verify against a disposable copy before touching production
.venv\Scripts\python.exe -c "import sqlite3; s=sqlite3.connect('file:backend/coinbase.db?mode=ro',uri=True); d=sqlite3.connect('C:/temp/probe.db'); s.backup(d)"
$env:DATABASE_URL='C:/temp/probe.db'
.venv\Scripts\python.exe -c "import asyncio,sys; sys.path.insert(0,'backend'); import database; asyncio.run(database.init_db())"
```

Production application requires operator authorisation and a backend restart, neither of which was
performed. Rollback: the added columns are nullable and unread by older code, so reverting the code
alone is sufficient — no down-migration needed.

**One behaviour note for whoever deploys this:** legacy *pending* rows (74 unresolved rows exist in
production) have no `target_time`. The resolver derives their schedule from `check_after` and labels
them as version 2. Legacy *resolved* rows are never touched.

---

## 7. Tests run, and results

All commands from the worktree root.

| Command | Result |
|---|---|
| `pytest backend/tests/test_outcome_labels.py -q` | **17 passed** |
| `pytest backend/tests/test_outcome_store_v2.py -q` | **15 passed** |
| `pytest backend/tests/test_outcome_tracker.py -q` | **9 passed** |
| `pytest backend/tests/test_diagnostics.py -q` | **10 passed** |
| `pytest backend/tests/test_diagnostics_label_versions.py -q` | **9 passed** |
| `pytest backend/tests/test_execution_findings.py -q` | **9 passed** |
| `ruff check backend/` (main's config) | **All checks passed** |
| `ruff format --check` (changed files) | **clean** |
| `pytest backend/tests -m "not slow and not integration" -q` | **see §8** |

Baseline for comparison: `main` at `26dddc2` runs **1345 passed / 65 skipped / 1 deselected /
1 xfailed / 2 xpassed**.

Caveats:
- Local ruff is **0.15.9**; CI pins **0.9.0**. Lint verified with main's config extracted into a
  `ruff.toml`.
- No test touches a real database, the exchange, or the network. `test_outcome_tracker.py` actively
  asserts the resolver never reaches `coinbase_client.get_candles` or `database.get_product`.

---

## 8. Final commit and full-suite result

| | |
|---|---|
| Branch | `feat/outcome-label-provenance` |
| Final commit | ``17131f8` (code + tests + contract spec). The docs commit that adds this handoff is the branch HEAD immediately after it.` |
| Full suite | `**1364 passed**, 65 skipped, 1 deselected, 1 xfailed, 2 xpassed in 6m09s (`-m "not slow and not integration"`). The pre-commit hook re-ran the same suite on the committed tree: 1364 passed, commit allowed.` |

---

## 9. Sample corrected outcome records

From `test_outcome_tracker.py::test_delayed_resolution_uses_the_target_bar_not_the_latest_bar`
(synthetic; temp database). Signal at `2026-01-01T00:00:00Z`, so entry bar `01:00`, exit bar `04:00`,
`target_time` `05:00`. Resolver run **200 hours late**:

| field | value | note |
|---|---|---|
| `label_version` | 2 | |
| `entry_candle_start` | 1767229200 (01:00Z) | first bar strictly after the signal |
| `exit_candle_start` | 1767240000 (04:00Z) | |
| `target_time` | 1767243600 (05:00Z) | exit bar's close instant |
| `price_observed_at` | 1767243600 | **equals target_time — not the processing time** |
| `entry_price` | 999.0 | scan quote, **preserved unchanged** |
| `entry_price_v2` | 100.0 | open of the entry bar |
| `target_price` | 102.0 | close of the exit bar (a later 50.0 bar existed and was ignored) |
| `signed_return` | +0.02 | fraction |
| `outcome` | `WIN` | +2% > +0.5% |
| `price_source` | `local_candles` | |
| `processed_at` | run timestamp | carries no pricing meaning |

Under version 1 the same row would have been labelled from the 50.0 bar (a **LOSS**), because the
resolver fetched the latest available price.

Unresolvable example (`test_repeated_failures_end_as_unavailable`): after 5 attempts with no bars,
`outcome='UNAVAILABLE'`, `unresolved_reason='missing_entry_candle'`, `target_price=NULL`. It is
excluded from every accuracy denominator and is never counted as a WIN, LOSS, or NEUTRAL.

---

## 10. Execution findings — verdicts

All four **CONFIRMED**. Detail and fix proposals: `docs/handoffs/2026-09-26-execution-findings.md`.

| # | Finding | Verdict | In CLAUDE.md? | Fix priority |
|---|---|---|---|---|
| 1 | Enabling trading replaces the executor; background handlers keep the old instance | **CONFIRMED** | No | 2nd |
| 2 | Live risk exits suppressed while the maker flag is off | **CONFIRMED** | Yes, #21 | 3rd — needs operator decision |
| 3 | Paper book closes before exchange confirmation | **CONFIRMED** | Yes, #21 | 4th — needs schema |
| 4 | Maker timeout fallback can market-order after a failed cancel | **FIXED 2026-09-26** | Now yes, #21 | done |

Finding 1 mechanism: `main.py:459` and `:502` pass the executor **by value** while the adjacent
`is_trading_fn=lambda: app_state.is_trading` is passed as a **callable**. `enable_trading`
(`main.py:1141`) rebinds `app_state.order_executor`, so the WS exit path and scan loop keep the
startup instance while handlers and the dashboard see the new one — and `_dry_run_balance` /
drawdown state are per-instance.

Finding 4 is broader than the audit stated: (a) the cancel exception is swallowed and execution falls
through unconditionally; (b) `cancel_orders`' response body is never inspected, so a cancel that
*reports* failure without raising also falls through; (c) `_wait_for_fill` accepts only
`status == "FILLED"`, so a partial fill triggers a **full-size** market order on top of it.

---

## 11. Accounting findings

Full detail: `docs/handoffs/2026-09-26-accounting-reconciliation.md`. Read-only
(`mode=ro` + `PRAGMA query_only=ON`).

- **The $5.98 is reproduced to the cent:** CNN ledger −$82.6357 vs `agent_state.realized_pnl`
  −$76.6569 = **−$5.9788**.
- It is **not** a lost-tail update: the last CNN close (`16:43:03.892743Z`) and the state write
  (`16:43:03.899741Z`) are 7 ms apart, and zero CNN rows closed afterwards. No single `trigger_close`
  subset sums to it either. It is accumulated drift between two independently maintained totals with
  no transactional coupling and no event log — bounded and explained as a class, not attributable to
  specific rows.
- **A second, larger defect found:** **1,905** closed rows have `pnl != usd_close − usd_open`
  (Σ **−$13.36**). The worst cases record `exit_price == entry_price` (implying $0) while `pnl` is
  materially negative — concentrated in tick-driven exits. `usd_open == size * entry_price` for every
  row, so the entry side is sound; the close-side price is unreliable.
- **Zero exchange evidence:** `orders` has **0 rows**; no `filled_size`, no `fee_paid`; `trades` has
  no order/fill column. Every figure is paper bookkeeping. No fills were invented or backfilled.
- **Three disagreeing position records:** 52 open `trades` rows vs 3 `positions` rows vs an empty CNN
  `positions_json`. 12 open trades (ATH, CORECHAIN, FAI, KAT, MON, NOICE, ONDO, RED, SUP, USDT, VET,
  XYO) exist in no state at all; 38 belong to TECH's `positions_json`, frozen since 2026-05-17 and
  still claiming $118.28; FLR-USD sits in `positions` with no open trade row.
- **No signal→trade→fill linkage** anywhere: 698,050 scans and 134,818 outcome rows with no foreign
  key to a trade.
- Label-timing defect **independently reproduced** from live data: mean 45.74 h, median 17.35 h,
  p90 140.92 h, max 495.32 h; 57.68% resolved >6 h late, 43.64% >24 h late (audit: 45.75 / 17.45 /
  140.91 / 57.76% / 43.67%).
- Minimum additional provenance required (run ID, model hash, config/strategy version, signal ID,
  execution mode, order/fill IDs, quantities, prices, fees, timestamps) is tabulated in that
  document's §5, with the recommendation to **stop maintaining `agent_state.realized_pnl` as an
  independent scalar** and derive it from the ledger or an append-only event log.

---

## 12. Remaining limitations

- Version-2 labels exist only in tests. Production has **no** v2 rows, so `signal_edge` will report
  `n=0` with real `counts` until the migration is applied and signals mature. That is intended: a
  truthful zero beats a misleading 22%.
- The ±0.5% band was kept from version 1 so a v1↔v2 comparison isolates the timing fix. It is a
  **gross** band; at the audit's 0.60%/side stress assumption a +0.5% WIN is a net loss.
- The v2 label still matches **no** model's training target, so calibration stays suppressed. Fixing
  that needs a triple-barrier label computed to each model's own parameters — not done here.
- Price source is the local `candles` table only. It has real gaps (one 6-hour, two multi-week), so
  some rows will legitimately end `UNAVAILABLE`. No backfill-on-demand was added.
- The $5.98 is characterised, not repaired.
- Execution findings are pinned by characterisation tests but **unfixed**; findings 2 and 3 require an
  invariant #21 amendment and an operator decision.
- `reconcile.py` lives in the scratchpad (absolute production path), not committed.

---

## 13. Next recommended task

~~Fix execution finding 4 (maker timeout fallback), TDD, before the 8002 maker shadow is promoted.~~ **DONE 2026-09-26** on `fix/maker-fallback-cancel-confirm` (stacked on this branch). **The next task is now execution finding 1** — the executor late-binding fix: small, self-contained, and it removes a state-drift source that feeds the accounting problem.

It is the only one of the four that can lose real money, it is reachable the moment
`USE_MAKER_EXECUTION=true` runs against a funded account, and its fix is self-contained: make the
market fallback conditional on a re-queried terminal order state, and size it from the unfilled
remainder. The characterisation tests in `test_execution_findings.py` already encode the current
behaviour, so the fix is a deliberate inversion of three assertions.

Then, in order: finding 1 (executor late-binding — small, and it removes a state-drift source that
feeds the accounting problem), then the provenance columns from §5 of the accounting report (they
gate findings 3 and any real fill reconciliation), and only then apply this migration to production
and let v2 labels accumulate before any model claim is made.

---

## 13b. Codex session-link handshake

Codex installed a CrewAI-based session link (`tools/session_bridge`, MCP server
`polymarket_session_link`, SQLite mailbox under `.coordination/`) and queued handshake
`f363ed1a-e4ab-482e-ba18-b833649d1728`.

- **MCP tools are NOT available in this Claude session** — the server was registered at ~13:27
  while this session started at 12:04, and the MCP server list is fixed at session start.
  `claude mcp get polymarket_session_link` reports **Connected**, scoped to project
  `C:/Users/gl450/polymarket_app`, so a session launched from that directory will load it.
- The documented CLI fallback works without a restart and was used here.
- Handshake **acknowledged** (`acknowledged: 1790447002`), task **claimed** as
  `outcome-label-provenance` with the file scopes in §6, and a status **reply sent**
  (`e6d228d7-4759-42d9-8dd2-09a077dc7703`, pending in Codex's mailbox).
- `tools/session_bridge` and `docs/handoffs/session-link.md` are Codex-owned and were not touched.

## 14. Boundaries honoured

- Production backend **not** restarted or stopped. Operator had disabled trading before this task
  (`is_trading:false`); the process was left alive and untouched.
- **No** migration applied to a production database; verified `coinbase.db` still lacks
  `label_version`.
- No orders placed, no credentials read or changed, no model retrained, no strategy threshold altered.
- Codex's audit artifacts unmodified.
- Nothing merged, deployed, or pushed.
