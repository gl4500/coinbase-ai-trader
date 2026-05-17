# Monte Carlo Loose-Coupling — Approved Design

**Date:** 2026-05-16
**Status:** APPROVED 2026-05-16 (operator: "go with your recommendations")
**Scope:** `backend/` only
**Branch:** continue on `feat/gpu-coord-mirror`
**Predecessor:** XGB v3 cutover (#311-cut)
**Successor:** plan at `docs/superpowers/plans/2026-05-16-mc-ci-filter-mvp.md`

---

## 1. Problem

XGB v3 is live. Every BUY/SELL/HOLD decision still rests on a **point estimate** — `model_prob > config.cnn_buy_threshold`. There is no notion of how confident the booster is, no draw-down envelope on Kelly sizing, no expected-value comparison at exit time, no portfolio-level VaR cap. Per `xgb_model_breakdown.html`, the live calibration in the 0.55–0.70 band was *inverted* under v1 — higher confidence predicted lower WR. Even with v3, we have no defense against the same failure mode beyond the point threshold itself.

Monte Carlo / bootstrap-style filtering can add that defense. The risk: embedding stochastic code inside `cnn_agent` creates a tangled diff and a hard rollback path.

## 2. Goal

Add four MC-style filters to the trading decision pipeline, **each independently kill-switchable**, **none touching `cnn_agent` core logic**, ship the MVP filter first behind an off-by-default flag.

## 3. Non-goals

- No changes to `cnn_agent` decision math beyond a single hook call.
- No new dependencies (uses `numpy` + `xgboost` already in tree).
- No backtesting harness in this scope (separate concern).
- No frontend changes.
- No touching TECH agent (TICK_TRAIL already pulling weight).
- No live retraining of any model.

## 4. Approach — loose-coupling pattern

Each MC intervention is a self-contained **filter** module under `backend/agents/mc/`. A filter takes the current decision + context, returns `(modified_decision, telemetry_dict)`. `cnn_agent.generate_signal` gets ONE new call: `mc.apply_buy_filters(...)`. Filters are dispatched from a registry that reads the `MC_FILTERS` env var.

```
MC_FILTERS=                ← default; registry returns identity; cnn_agent path bit-for-bit unchanged
MC_FILTERS=ci              ← CIFilter active
MC_FILTERS=ci,kelly_dd     ← both active, applied in chain order
```

### 4.1 The four filter slots

| # | Name | Decision point | Status |
|---|---|---|---|
| 1 | **CIFilter** | BUY gate: `lower_ci(model_prob) > threshold` | **MVP, this spec** |
| 2 | KellyDDFilter | position size: cap frac by P(drawdown > 8%) tolerance | next phase |
| 3 | ExitEVFilter | tick exit chain: only fire if EV(exit) > EV(hold) | later |
| 4 | PortfolioVaRFilter | BUY gate: block if VaR_95(portfolio + new) > cap | later |

### 4.2 MVP scope (approved)

**CIFilter alone.** Subsequent filters get their own spec + plan once we have ≥24h of `cnn_scans` telemetry confirming CIFilter behaves as expected.

## 5. CIFilter algorithm

```python
def compute_ci(channels, pid, K=1.0) -> tuple[float, float]:
    """Returns (point_prob, ci_stdev) for the v3 booster."""
    booster = xgb_signal._booster                     # already loaded
    dmat = build_v3_dmatrix(channels, pid)            # reuse v3 path
    n = booster.num_boosted_rounds()                  # 200 today
    trajectory = [
        float(booster.predict(dmat, iteration_range=(0, k + 1))[0])
        for k in range(n)
    ]
    point = trajectory[-1]
    stdev = float(np.std(trajectory))
    return point, stdev
```

Lower-bound BUY gate:

```python
lower_bound = max(0.0, point - K * stdev)
buy_allowed = (lower_bound > config.cnn_buy_threshold)
```

**K=1.0** is the locked default (one-stdev lower bound). Tuneable via `MC_CI_K`.

## 6. Architecture

```
                              ┌─────────────────────────────────┐
                              │ cnn_agent.generate_signal()      │
                              │   ... compute model_prob ...     │
                              └───────────────┬─────────────────┘
                                              │
                                              ▼
                              ┌─────────────────────────────────┐
                              │ agents.mc.registry               │
                              │   apply_buy_filters(side,        │
                              │     model_prob, pid, channels)   │
                              │                                  │
                              │   reads MC_FILTERS env once at   │
                              │   import; dispatch chain         │
                              └───────────────┬─────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
                       │ CIFilter    │ │ KellyDD     │ │ ExitEV      │
                       │ (MVP)       │ │ (next)      │ │ (later)     │
                       └─────────────┘ └─────────────┘ └─────────────┘
                              │
                              ▼
                       (side, telemetry_dict)
                              │
                              ▼
                              ┌─────────────────────────────────┐
                              │ cnn_scans persistence:           │
                              │   xgb_prob_stdev REAL NULL       │
                              │   mc_telemetry   TEXT NULL  (JSON) │
                              └─────────────────────────────────┘
```

### 6.1 File layout

| Path | Action | Purpose |
|---|---|---|
| `backend/agents/mc/__init__.py` | CREATE | empty package marker |
| `backend/agents/mc/base.py` | CREATE | `BuyFilter` ABC: `evaluate(side, model_prob, pid, channels, context) -> (decision, telemetry)` |
| `backend/agents/mc/registry.py` | CREATE | reads `MC_FILTERS`, returns active chain; `apply_buy_filters(...)` entry point |
| `backend/agents/mc/ci_filter.py` | CREATE | CIFilter implementation |
| `backend/tests/agents/__init__.py` | CREATE | new test subpackage marker |
| `backend/tests/agents/mc/__init__.py` | CREATE | new test subpackage marker |
| `backend/tests/agents/mc/test_registry.py` | CREATE | 8 tests: env dispatch, chain order, unknown filter handling |
| `backend/tests/agents/mc/test_ci_filter.py` | CREATE | 6 tests: trajectory math, edge cases, neutral fallbacks |
| `backend/agents/cnn_agent.py` | EDIT | +1 call to `mc.apply_buy_filters` at BUY gate; +propagation of stdev/telemetry |
| `backend/database.py` | EDIT | extend `save_cnn_scan` to accept + persist `xgb_prob_stdev`, `mc_telemetry` |
| `backend/migrations/2026-05-16-mc-telemetry.py` | CREATE | idempotent `ALTER TABLE cnn_scans ADD COLUMN ...` |
| `backend/tests/test_cnn_agent.py` | EXTEND | +3 tests for MC-on / MC-off gate behavior + telemetry |
| `backend/tests/test_database.py` | EXTEND | +2 tests for column persistence |

### 6.2 Unchanged

- `xgb_signal.py` (CIFilter calls its already-loaded `_booster` directly via the module-level reference)
- `tiered_history`, `xgb_features`, `train_xgb`, `fit_xgb_calibration`
- TECH agent, exit chain, order executor, WebSocket
- Side gate threshold semantics (still `> threshold`; just compared against `lower_bound` instead of `point` when CI is active)
- Frontend

## 7. Configuration (.env knobs — all default disabled)

```
MC_FILTERS=                # comma-separated list; empty = MC off entirely
MC_CI_K=1.0                # stdev multiplier for CIFilter lower bound
MC_CI_TRAJ_STEP=1          # 1 = full 200-tree trajectory; bump to 10 for 10x speed at ~5% stdev error
```

## 8. Telemetry (approved: both column + JSON blob)

Two new columns on `cnn_scans`:

```sql
ALTER TABLE cnn_scans ADD COLUMN xgb_prob_stdev REAL;
ALTER TABLE cnn_scans ADD COLUMN mc_telemetry   TEXT;     -- JSON blob: {filter_name: telemetry_dict, ...}
```

CIFilter populates:
- `xgb_prob_stdev`: the trajectory stdev (float)
- `mc_telemetry`: `{"ci": {"stdev": 0.0124, "lower": 0.5776, "K": 1.0, "decision": "keep"}}`

Both columns are nullable. Pre-MC rows stay NULL. Post-MC rows where `MC_FILTERS=""` also stay NULL (registry returns identity, no telemetry to persist).

## 9. Error handling

| Condition | Behavior | Test |
|---|---|---|
| `MC_FILTERS=""` (default) | `apply_buy_filters` returns `(side, {})` unchanged | `test_registry_empty_mc_filters_noop` |
| `MC_FILTERS=unknown` | log warning, skip the unknown name | `test_registry_unknown_filter_warns` |
| CIFilter under v1/v2 booster | returns `(side, {"ci": {"skipped": "non-v3-booster"}})`, decision unchanged | `test_ci_filter_skips_under_v1` |
| CIFilter when `pid=None` | log warning, decision unchanged, telemetry `{"ci": {"skipped": "pid-none"}}` | `test_ci_filter_skips_when_pid_none` |
| CIFilter when booster unavailable | log warning, decision unchanged | `test_ci_filter_skips_when_booster_missing` |
| CIFilter when trajectory predict raises | exception caught; decision unchanged; telemetry records the error | `test_ci_filter_skips_on_predict_error` |
| Migration column already exists | idempotent (PRAGMA detect → skip ALTER) | `test_migration_idempotent` |

## 10. Tests (TDD, all RED → GREEN per CLAUDE.md)

| File | Status | Tests |
|---|---|---:|
| `tests/agents/mc/test_registry.py` | NEW | 8 |
| `tests/agents/mc/test_ci_filter.py` | NEW | 6 |
| `tests/test_cnn_agent.py` | EXTEND | +3 |
| `tests/test_database.py` | EXTEND | +2 |
| `tests/test_mc_migration.py` | NEW | 2 |
| **Total** | | **21** |

## 11. Rollout

### Phase 0 — Scaffolding (commit 1)
Empty `agents/mc/` package + `base.py` ABC + `registry.py` with empty-MC noop path. 8 registry tests. No behavior change. Default backend state unchanged.

### Phase 1 — CIFilter impl (commit 2)
`ci_filter.py` + 6 unit tests. Filter is registered but not invoked yet (no cnn_agent wire-up). Tests pass in isolation.

### Phase 2 — Schema + cnn_agent wire (commit 3)
Migration + `database.save_cnn_scan` extension + `cnn_agent.generate_signal` hook + 5 wire-up tests. **MC_FILTERS still empty in .env**, so live behavior unchanged. CIFilter's existence is observable only when operator flips the env.

### Phase 3 — Cutover (commit 4 / `.env` edit only)
Operator flips `MC_FILTERS=ci` + `POST /api/cnn/model/reload`. CIFilter starts evaluating every scan and persisting telemetry. DRY_RUN stays true.

### Phase 4 — Observation (no code, operator-driven)
After 24-48h: query `cnn_scans` for `xgb_prob_stdev` distribution + per-bucket WR, decide whether to keep K=1.0 / tune / add KellyDDFilter next.

### Rollback
- Per-filter: edit `MC_FILTERS`, reload (~5 sec).
- All MC: `MC_FILTERS=`, reload.
- Code-level: `git revert <commits>` on `feat/gpu-coord-mirror` (~30 sec).
- Schema: not needed — columns are NULLABLE.

## 12. Memory + CLAUDE.md sync (per CLAUDE.md sync rule)

At Phase 3 commit:
- `memory/coinbase_trader_architecture.md` — new "Monte Carlo Filters" section: registry pattern, MC_FILTERS knob, CIFilter MVP, the 3 deferred filter slots.
- `polymarket_app/CLAUDE.md` — new invariant: "MC filters live under `backend/agents/mc/` with one-line registry hook in `cnn_agent.generate_signal`. `MC_FILTERS=` default means today's bit-for-bit behavior. Each filter has its own env-var kill-switch. Never embed MC math inside cnn_agent core."
- `polymarket_app/CHANGELOG.md` — one entry per phase commit.

## 13. Open questions

None — operator approved defaults on 2026-05-16. Tuning of K, traj_step, and observation window happens in Phase 4 based on live telemetry.

## 14. References

- HTML overview: `backend/tools/mc_loose_coupling_plan.html` (2026-05-16)
- v3 cutover: `backend/CHANGELOG.md` Session 58.69-cut (2026-05-16)
- Inverted v1 calibration finding: `backend/tools/xgb_model_breakdown.html` (2026-05-16)
- 502-trade K-sensitivity table: HTML overview §"K sensitivity at threshold 0.55"
