# v3 Diagnostics Dashboard — Design

**Date:** 2026-08-08
**Status:** Approved (brainstorm), ready for implementation plan
**Branch:** `feat/diagnostics-dashboard`

## Purpose

The live 8001 v3 (XGB) system is net-negative in paper trading (−$115 / −11.5%
over ~4 months, 39% win rate) and the existing **PerformanceDashboard** only
shows PnL accounting (trades, win-rate, return %, per-trigger, equity, balance).
It does **not** answer *why* it loses. This adds a **Diagnostics** tab that turns
the ad-hoc analysis done during the 2026-08 review into a repeatable,
always-current view — decision-support, **not** another model/prediction
experiment (the user's explicit framing: "different objective, not R&D").

A grounding finding this should surface prominently: **calibration reveals edge
hiding in v3.** BUY-signal realized win-rate rises monotonically with confidence
in the upper buckets (0.5→39%, 0.6→46%, 0.9→**53%** with positive avg return),
even though blended precision is only ~22%. "The top-confidence slice actually
works" is exactly the kind of insight this tab makes visible.

## Approach (chosen: A — on-demand endpoint + TTL cache + hand-rolled SVG)

Rejected alternatives: **B** (materialized snapshot table + background job —
overkill for v1; natural later evolution for historical trending), **C**
(browser-side compute from raw endpoints — infeasible; the signal-edge/
calibration views aggregate over 670k `cnn_scans`).

## Architecture & boundaries

### Backend — `backend/services/diagnostics.py` (new, read-only)
Pure, typed, single-responsibility functions (per clean-function rules). Opens
its **own read-only** connection (`file:coinbase.db?mode=ro`, WAL-safe alongside
live writes) — never writes, never imports the trading loop, so it physically
cannot affect 8001. One function per view + a `compute_diagnostics(window)`
orchestrator. A tiny module-level TTL cache (`{window: (ts, payload)}`, 60s) so
670k-row aggregations don't recompute per poll.

- `signal_edge(conn, window) -> dict`
- `exit_attribution(conn, window) -> dict`
- `regime_and_asset(conn, window) -> dict`
- `signal_funnel(conn, window) -> dict`
- `compute_diagnostics(window) -> dict` (opens conn, calls the four, caches)

All queries filter to the live agent (`agent='CNN'` / `source='CNN'`; TECH is
retired) and to the selected window.

### Backend — `main.py`
`@app.get("/api/diagnostics")` with `window` ∈ {`30d`,`90d`,`all`} (default
`30d`). Returns the orchestrator JSON. Wrapped in try/except → 500 with message;
a diagnostics failure never propagates to trading state.

### Frontend — `frontend/src/components/DiagnosticsDashboard.tsx` (new)
Register `'Diagnostics'` in `App.tsx`'s `TABS` array + conditional render. Four
sections; charts hand-rolled in `<svg>` (matching the existing `EquityCurve`
pattern — **no new charting dependency**, CSP-safe). Window selector + manual
refresh button. Fetches on mount / window-change / refresh — **not** on the WS
state poll (too heavy for a per-tick broadcast).

## The four views (data sources grounded in the current schema)

Tables: `cnn_scans` (670k: `side, model_prob, xgb_prob, xgb_prob_stdev, regime,
scanned_at, product_id, mc_telemetry`), `signal_outcomes` (132k: `source, side,
confidence, pct_change, outcome, created_at`), `trades` (`trigger_open,
trigger_close, pnl, pct_pnl, hold_secs, opened_at, closed_at, product_id`),
`products` (`is_tracked`).

1. **Signal edge & calibration** (`signal_outcomes`, BUY, matured):
   precision (WIN/LOSS/NEUTRAL %), E[return] = avg(`pct_change`); **calibration
   chart** = `confidence` decile → realized WIN-rate + count per bucket, with the
   y=x reference diagonal. Centerpiece.
2. **Exit attribution** (`trades.trigger_close`): n / sumPnL / avg `pct_pnl` / WR
   per trigger; hold-time histogram; SCAN-SELL share trend (the WS-pre-emption
   story surfaced in the review).
3. **Regime & per-asset**: per-asset PnL from `trades` (group by `product_id`);
   regime PnL via **nearest-scan join** (each trade's `product_id` + `opened_at`
   → latest `cnn_scans.regime` at/before entry, since `signal_outcomes` has no
   regime); plus regime distribution from `cnn_scans`.
4. **Signal funnel**: scans (`cnn_scans`) → BUY signals → executed (`trades`
   opened) → matured outcomes (`signal_outcomes`), with conversion/drop-off at
   each stage.

## Data flow, caching, refresh

Tab open / window change → `GET /api/diagnostics?window=X` → orchestrator checks
60s TTL cache → miss: open RO conn, run the four query-sets, close, cache,
return → frontend renders four SVG sections. Manual refresh re-fetches (cache
still applies within 60s). No auto-poll.

## Error handling & isolation

Read-only connection (WAL-safe). Each view function is independently try-wrapped
so one failing view degrades to an error card, not a blank tab. Endpoint-level
try/except returns 500 without touching app state. Zero coupling to
`cnn_agent`/trading loop (loose-coupling rule).

## Testing (TDD)

`backend/tests/test_diagnostics.py`: seed a temp SQLite with known
`trades`/`signal_outcomes`/`cnn_scans` rows; assert each pure function's numbers
(precision, calibration buckets, per-trigger sums, nearest-scan regime mapping,
funnel counts) and windowing. Mock-only, no live DB. Optional light frontend
render test.

## Scope — YAGNI exclusions

- No historical snapshot / trend-over-time (that's Approach B, later).
- No new charting library (hand-rolled SVG).
- No auto-refresh polling.
- CNN agent only (TECH retired).
