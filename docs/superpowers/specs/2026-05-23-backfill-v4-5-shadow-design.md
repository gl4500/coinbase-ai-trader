# Backfill v4.5 Shadow Telemetry — Design Spec

**Date:** 2026-05-23
**Topic:** Retroactively populate `xgb_prob_v4_5_*` columns on historical `cnn_scans` rows
**Status:** Design approved, plan pending

---

## Goal

Populate the three v4.5 shadow columns (`xgb_prob_v4_5_down`, `_neutral`, `_up`) on `cnn_scans` rows that were written before the always-on v4.5 shadow gate landed (commit `6052566`). Scope: last 7 days. Enables direct v3-vs-v4.5 prediction comparison on historical live scans without waiting for the natural 7-day shadow accumulation.

## Why

- Post-deploy of always-on shadow (Session 58.71m), forward-looking scans on 8001 carry v4.5 shadow probs. The 294K historical rows do not.
- `tools/v4_5_horizon_compare.py` already gave us model-quality AUC on holdout data (0.6399 h168). What it did not give: side-by-side v3-vs-v4.5 calls on the same LIVE scans, where regime / market microstructure matches what the agent actually saw.
- Per `feedback_roi_first_priority.md`, ROI is the operator's #1 metric. This backfill is the prediction layer needed before any ROI replay (Option B, out of scope here).

## Scope

**IN:**
- New function param `now_ts` on `agents.xgb_signal.xgb_prob_v4_5(channels, pid, now_ts=None)`. Backward-compat (default `None` preserves current live behavior).
- New CLI tool `backend/tools/backfill_v4_5_shadow.py` with flags `--days`, `--batch-size`, `--db`, `--dry-run`.
- Test file `backend/tests/test_backfill_v4_5_shadow.py` — 5 unit tests (will run during next 8001 pause window).
- Operator-style execution: this session runs the backfill end-to-end on coinbase.db.

**OUT (deferred):**
- Decision replay (apply `_indep_thresholds_decision` to backfilled probs → synthesize hypothetical BUY/SELL signals)
- Paper-PnL simulation (trail-stop, position sizing, exit logic against historical price stream)
- Comparison query helpers / SQL views
- Pytest verification (8001 is live trading; pytest deferred per `feedback_no_pytest_during_trading.md`)
- Commit (deferred to next pause window)

## Architecture

### Function change (`agents/xgb_signal.py`)

```python
def xgb_prob_v4_5(
    channels, pid: Optional[str] = None,
    now_ts: Optional[float] = None,            # NEW
) -> Tuple[float, float, float]:
    """v4.5 3-class probabilities.

    When now_ts is provided, fetch_tiered uses it to look up the tier slices
    as they would have been at that historical timestamp (drops candles with
    start >= now_ts). Default None = live (current behavior).
    """
    ...
    tiers = fetch_tiered(pid, source="live", now_ts=now_ts)   # CHANGED
    ...
```

`channels` arg remains vestigial (already unused by the function — kept for back-compat with `xgb_prob_shadow_v4_5` caller signature).

### Backfill tool

```
backend/tools/backfill_v4_5_shadow.py

USAGE
  python -m tools.backfill_v4_5_shadow [--days N] [--batch-size 100] [--db PATH] [--dry-run]

ARGS
  --days N         Look back N days of cnn_scans (default 7)
  --batch-size N   Rows per UPDATE transaction (default 100)
  --db PATH        SQLite path (default backend/coinbase.db)
  --dry-run        Print scope + sample, no UPDATEs

FLOW
  1. Open DB (read-only initial query)
  2. PRAGMA journal_mode = WAL  (idempotent — only changes if not WAL already)
  3. SELECT id, product_id, scanned_at FROM cnn_scans
       WHERE xgb_prob_v4_5_up IS NULL
         AND scanned_at > <cutoff iso>
       ORDER BY id
  4. If --dry-run: print count + first 5 sample rows. Exit.
  5. Else: for each batch of <batch-size>:
       For each row:
         now_ts = parse_iso_to_unix(row.scanned_at)
         try:
           p_down, p_neutral, p_up = xgb_signal.xgb_prob_v4_5(
               channels=None, pid=row.pid, now_ts=now_ts,
           )
           updates.append((round(p_down,4), round(p_neutral,4), round(p_up,4), row.id))
         except Exception as e:
           logger.warning("row %d (%s @ %s): %s — skipping", row.id, pid, scanned_at, e)
       UPDATE cnn_scans SET xgb_prob_v4_5_down=?, xgb_prob_v4_5_neutral=?, xgb_prob_v4_5_up=?
         WHERE id=?  (executemany)
       COMMIT (per batch)
       Print [HH:MM] progress: {processed}/{total} rows ({pct:.1f}%) rate={k/sec:.1f}/s eta={mm:ss}
  6. Final summary: total processed, total skipped (inference failures), total succeeded
```

### Safety

| Concern | Mitigation |
|---|---|
| Concurrent writes from live backend on cnn_scans | WAL mode + 100-row batches → each transaction holds writer lock <50ms |
| Long-running tool blocks scan loop | Batched commits — scan loop can interleave between batches |
| Inference failure crashes whole batch | Per-row try/except → log + skip + continue |
| Already-backfilled rows | NULL filter excludes them (idempotent re-runs) |
| Wrong scope (too far back) | `--dry-run` flag prints count + sample first |
| v4.5 artifacts not loaded | `_try_load_v4_5()` returns False → `xgb_prob_v4_5` returns neutral `(0.33, 0.34, 0.33)` — tool detects neutral and skips writing those (or writes; operator's call — see Q1) |

### Edge cases

- **Insufficient historical candles at `now_ts`**: `fetch_tiered` returns shorter tiers; `extract_v4_5` may raise or return zero-padded features → caught by per-row try/except, skip + log.
- **`now_ts` before product had data**: same as above.
- **`now_ts` very recent (< macro window of 336h ago)**: macro tier short → extract_v4_5 may pad-zero certain features → still produces a prob but lower quality. Acceptable since we're scoring directionality + the model was trained with similar warm-up edge cases.
- **Crash mid-run**: Idempotent re-run picks up remaining NULL rows. Operator can ctrl-C any time.

## Test Plan

These tests are written but NOT run in this session (8001 live trading per `feedback_no_pytest_during_trading.md`). Operator runs them in the next pause window before committing.

`backend/tests/test_backfill_v4_5_shadow.py`:

1. **`test_xgb_prob_v4_5_passes_now_ts_to_fetch_tiered`** — patch `fetch_tiered`, call `xgb_prob_v4_5(..., now_ts=1700000000.0)`, assert `fetch_tiered.call_args.kwargs["now_ts"] == 1700000000.0`.
2. **`test_xgb_prob_v4_5_now_ts_default_none`** — call without now_ts, assert fetch_tiered called with `now_ts=None`. Back-compat guard.
3. **`test_backfill_selects_only_null_v45_rows`** — seed test DB with 3 rows (1 NULL, 1 populated, 1 NULL outside window) → tool selects only the 1 NULL row in window.
4. **`test_backfill_writes_three_probs_atomically`** — mock `xgb_prob_v4_5` to return `(0.1, 0.2, 0.7)`, run on 1 NULL row, query row after — all three cols populated, rounded to 4 dp.
5. **`test_backfill_handles_inference_failure`** — mock `xgb_prob_v4_5` to raise on row 1, return ok on row 2 → row 1 stays NULL, row 2 populated, no crash.

## Open question

**Q1: When v4.5 returns the neutral fallback (0.33, 0.34, 0.33) — write or skip?**

When `_try_load_v4_5()` fails or features can't be extracted, `xgb_prob_v4_5` returns the neutral fallback. If we write it, NULL columns show as `(0.33, 0.34, 0.33)` — indistinguishable from a real-model neutral prediction. If we skip, the row stays NULL and a future re-run will retry.

**Recommendation**: skip on neutral-fallback. Detect via exact value `(0.33, 0.34, 0.33)` and treat as inference failure. This way the NULL state is a meaningful "couldn't infer" signal, and any tier-fetch issues can be diagnosed by re-running with verbose logging.

## Rollback

If the tool produces bad data:
```sql
-- Wipe all backfilled rows (preserves forward-looking always-on rows
-- since those were written by cnn_agent.generate_signal, not this tool;
-- distinguish by scanned_at < <backfill-start-ts>).
UPDATE cnn_scans
SET xgb_prob_v4_5_down = NULL,
    xgb_prob_v4_5_neutral = NULL,
    xgb_prob_v4_5_up = NULL
WHERE scanned_at < datetime('<backfill_started_at>')
  AND scanned_at > datetime('now', '-7 days');
```

Tool prints `<backfill_started_at>` at start so operator can rollback if needed.

## Invariants preserved

- **#16 Shadow telemetry isolation** — `xgb_prob_v4_5` is wrapped in try/except in `xgb_prob_shadow_v4_5` (live path). The backfill tool doesn't touch this; it calls `xgb_prob_v4_5` directly with its own per-row try/except. Live path unaffected.
- **#17 v4.5 3-class telemetry contract (atomic write or all-NULL)** — preserved: per-row UPDATE writes all three columns or none.
