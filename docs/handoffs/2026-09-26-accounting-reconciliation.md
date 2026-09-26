# Accounting reconciliation — 2026-09-26 (Task 6)

**Method:** production `backend/coinbase.db` opened with `mode=ro` **and** `PRAGMA query_only=ON`.
Nothing was written, migrated, invented, or backfilled. Script:
`scratchpad/reconcile.py` (disposable; reproduced verbatim in §7).

**Relationship to the audit:** this independently reproduces the audit's headline numbers from live
data rather than the frozen extract, then goes further on mechanism. Counts differ slightly from
`docs/audits/2026-09-26-strategy-audit-report.md` because the app kept running after the audit froze
its extract.

---

## 1. Ledger vs persisted agent state

| Quantity | Value |
|---|---:|
| `trades` rows | 2,138 |
| closed | 2,086 |
| still open | 52 |
| `SUM(pnl)` over all closed | **−$165.17** |
| synthetic closes (`RECONCILE`, `STARTUP_CLEANUP`) | 69 rows, $0.00 |

`agent_state` holds **two** agents:

| agent | balance | realized_pnl | updated_at | positions_json |
|---|---:|---:|---|---|
| `CNN` | $923.34 | **−$76.66** | 2026-09-26T16:43:03Z | 0 positions |
| `TECH` | $118.28 | **−$62.29** | 2026-05-17T01:41:13Z | **38 stale positions** |

### The $5.98 discrepancy — reproduced exactly

| | |
|---|---:|
| CNN ledger, all closed rows | −$82.6357 |
| CNN `agent_state.realized_pnl` | −$76.6569 |
| **delta** | **−$5.9788** |

This matches the audit's $5.98 to the cent.

**What it is not.** The state is *current*, not truncated: the last CNN trade closed at
`2026-09-26T16:43:03.892743Z` and CNN state was written at `2026-09-26T16:43:03.899741Z` — 7 ms
later. Zero CNN rows closed after the last state write, so this is not a lost-tail-update artifact.
No single `trigger_close` subset sums to −$5.9788 either, so it is not one bad exit class.

**What it is: accumulated drift between two independently maintained totals.**
`agent_state.realized_pnl` is a running scalar mutated in memory and persisted periodically, while
`trades.pnl` is written per close. Nothing couples them transactionally and there is no event log, so
once they diverge the difference cannot be attributed after the fact. Two concrete arithmetic
inconsistencies in the same data are large enough to produce this class of drift:

1. **1,905 closed rows where `pnl != usd_close − usd_open`** (Σ of differences = **−$13.36**).
2. **69 synthetic closes** that remove a position with `pnl = 0.0` and NULL `usd_open`/`usd_close`
   (55 `RECONCILE` + 1 `STARTUP_CLEANUP` have NULL usd columns) — book-state mutations carrying no
   P&L record at all.

Because both exist, the $5.98 is **bounded and explained as a class**, but not uniquely attributable
to specific rows. That matches the audit's position: quantify, do not repair.

### A second defect found while reconciling: the close-side price is unreliable

Ranking the 1,905 divergent rows by size exposes the mechanism — the worst offenders have
`exit_price` recorded **equal to `entry_price`**, so the price columns imply exactly $0 P&L while the
stored `pnl` is materially negative:

| id | product | entry | exit | usd_open | usd_close | stored pnl | implied | trigger |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 167 | NOM-USD | 0.00381 | 0.00381 | 118.879072 | 118.879072 | **−6.5524** | 0.0000 | TICK_STOP |
| 84 | KAT-USD | 0.00777 | 0.00777 | 73.826513 | 73.826513 | **−3.8956** | 0.0000 | TICK_STOP |
| 146 | L3-USD | 0.01115 | 0.01115 | 52.417174 | 52.417174 | **−2.8677** | 0.0000 | TICK_STOP |
| 6209 | ETH-USD | 1577.5 | 1577.5 | 135.894577 | 135.894577 | −0.0465 | 0.0000 | WS_MODEL_DOWN |

Meanwhile `usd_open == size * entry_price` for **every** row (0 mismatches), so the entry side is
sound. The defect is on the close side, concentrated in tick-driven exits: `pnl` was computed from
the live tick price, but `exit_price`/`usd_close` were persisted echoing the entry price. Smaller
divergences (e.g. id 4978, +$0.00005) are ordinary float rounding.

**Consequence:** any attempt to recompute P&L from the stored prices will disagree with the stored
`pnl`, and `exit_price` must not be treated as the realised exit for tick exits. This is independent
of the $5.98 and is arguably the more serious record-keeping problem.

---

## 2. Paper bookkeeping vs verifiable exchange execution

**There is no verifiable exchange execution in this database at all.**

| Evidence | Count |
|---|---:|
| `orders` rows | **0** |
| rows with `filled_size` | 0 |
| rows with `fee_paid` | 0 |
| `trades` columns referencing an order or fill | **none** |

Every number in §1 is therefore **paper bookkeeping**: an internal simulation of fills at observed
prices, with no fees modelled and no counterparty confirmation. The `orders` table has the right
shape (`order_id`, `filled_size`, `avg_fill_price`, `fee_paid`, `status`) but was never populated,
because the live order path has not run against a funded account in this history.

The retired `signals` table does carry `acted` + `order_id`, so the intent existed on the old TECH
path; the CNN/XGB path that produced this history has no equivalent.

**Nothing here can be reconciled against an exchange, and no fills were invented or backfilled.**

---

## 3. Orphans and open-position disagreement

Three different records of "what is held" disagree:

| Source | Count | Contents |
|---|---:|---|
| `trades` with `closed_at IS NULL` | **52** | opens from 2026-04-12 and 2026-05-16 |
| `positions` table | **3** | FLR-USD, LINK-USD, XRP-USD (updated today 17:56Z) |
| `agent_state` CNN `positions_json` | **0** | empty |
| `agent_state` TECH `positions_json` | **38** | frozen since 2026-05-17 |

- **12 open trades exist in no state at all:** ATH, CORECHAIN, FAI, KAT, MON, NOICE, ONDO, RED, SUP,
  USDT, VET, XYO — all opened 2026-04-12/13 and never closed or adopted. These are orphan ledger
  rows roughly five months stale.
- **38 open trades belong to TECH's frozen `positions_json`**, orphaned by the TechAgent retirement
  (#311-refactor-c, 2026-05-16). TECH state has not been written since 2026-05-17 yet still claims
  $118.28 of balance and 38 positions.
- **FLR-USD sits in the `positions` table with no open `trades` row**, so the positions table and the
  ledger disagree in the opposite direction too.
- CNN's `positions_json` says flat while the `positions` table shows three holdings, because the two
  are written by different code paths at different times.

None of this is repaired here. It is recorded so that no equity curve is reconstructed from these
rows without acknowledging the disagreement.

---

## 4. Missing signal → order → fill linkage

| Link | Present? |
|---|---|
| `cnn_scans` → trade or order | **No** (no such column) |
| `signal_outcomes` → trade or order | **No** |
| `trades` → order | **No** |
| order → fill | n/a (`orders` empty) |

698,050 scans and 134,818 outcome rows exist with **no foreign key to any trade**. The audit's
"preceding same-product scan within one hour" heuristic is the only available association, and it is
a heuristic, not a link. Per-model attribution is impossible: `agent='CNN'` is a book name spanning
multiple CNN and XGB versions with no immutable artifact identifier.

Independently reproduced from live data, the label-timing defect behind Task 2/3:

| Metric | This run (n=134,744) | Audit (frozen extract) |
|---|---:|---:|
| mean resolution delay | 45.74 h | 45.75 h |
| median | 17.35 h | 17.45 h |
| p90 | 140.92 h | 140.91 h |
| max | 495.32 h | 495.32 h |
| resolved >6 h late | 57.68% | 57.76% |
| resolved >24 h late | 43.64% | 43.67% |

Outcome distribution: LOSS 56,651 / WIN 51,440 / NEUTRAL 26,653 / unresolved 74. `label_version` is
**absent** from the production database, confirming the Task 3 migration has not been applied there.

---

## 5. Minimum additional provenance required

Nothing above can be fixed by better queries; the fields do not exist. Minimum set, per the brief:

| Field | Attach to | Why |
|---|---|---|
| `run_id` | every scan, signal, order, trade | Ties a row to one process lifetime + configuration. Without it, "which code produced this" is unanswerable. |
| `model_hash` | every scan/signal | The audit could not attribute any trade to a model version. An immutable artifact digest is the only fix. |
| `config_version` / `strategy_version` | every signal, order | Thresholds and exit policy changed mid-history; attribution is currently impossible. |
| `signal_id` (FK) | `trades`, `orders` | Replaces the one-hour proximity heuristic with a real link. |
| `execution_mode` (`paper` \| `live`) | every order, trade | Makes the paper/live boundary explicit in the data instead of inferred from `dry_run` at runtime. |
| `order_id`, `fill_id` | `trades` | Currently absent; without them no exchange reconciliation is possible ever. |
| `filled_size`, `avg_fill_price` | fills | Distinguishes partial from complete fills — directly relevant to execution finding 4. |
| `fee_paid` (+ `fee_currency`) | fills | The audit had to model fees as scenarios because actual fees are nowhere recorded. |
| `price_source` + `price_observed_at` | trade close | Would have prevented the `exit_price == entry_price` defect in §1. |
| `event_seq` / append-only ledger | state mutations | The $5.98 is unattributable *because* state is a mutable scalar with no event history. |

**Recommendation on realized P&L:** stop maintaining `agent_state.realized_pnl` as an independent
running total. Either derive it from the `trades` ledger on read, or make state mutations append-only
events that reconcile to the ledger by construction. A parallel scalar with no transactional coupling
will drift again, and the drift will again be unattributable.

---

## 6. Limitations

- Paper-only data: no exchange fill exists to reconcile against, so "correct" here means internally
  consistent, never externally verified.
- The $5.98 is bounded and explained as a class, not attributed to specific rows. Attribution would
  require an event log that does not exist.
- The 52 open rows span five months and overlap a retired agent; no equity curve is reconstructable.
- Counts differ slightly from the audit because the app continued running after its extract froze.
- Read-only by construction: no repair, migration, or backfill was performed or is proposed here
  without separate authorisation.

---

## 7. Reproduction

```powershell
# read-only; requires no credentials and places no orders
.venv\Scripts\python.exe <path-to>\reconcile.py
```

The script opens the database `mode=ro` with `PRAGMA query_only=ON` and prints §1–§4 verbatim. It was
kept in the scratchpad rather than committed because it targets an absolute production path; promote
it to `backend/tools/` with a `--db` argument if it should become a standing report.
