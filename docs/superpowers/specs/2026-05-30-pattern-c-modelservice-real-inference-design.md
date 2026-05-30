# ModelService Real Inference + Decision Logic — Design Spec

**Date:** 2026-05-30
**Author:** Claude Opus 4.7 (Session 58.71u, post-PR-#15 review-fixes commit `615a52c`)
**Status:** Approved (operator 2026-05-30; sketch presented in chat, option A chosen — "write this up as a proper spec, then a separate session implements via TDD")
**Branch:** `feat/event-sourced-architecture`
**Worktree:** `C:/Users/gl450/AppData/Local/Temp/pattern-c`
**Related:**
- `2026-05-25-event-sourced-architecture-design.md` (parent — defines ModelService role)
- `2026-05-25-event-sourced-architecture-decisions.md` (deployment_n{N}.json artifact)
- `polymarket_app/backend/agents/cnn_agent.py:441` (`_indep_thresholds_decision` — source rule)
- `polymarket_app/backend/agents/xgb_signal.py` (`xgb_prob`, `xgb_prob_shadow_v4_5` — source inference fns)
- Task #83 in operator's task tracker

---

## Problem

`backend/services/model_service.py` ships two plumbing stubs (`_score_signal`, `_decide_trade`) that always return HOLD / `None`. The event-sourced pipeline is structurally complete — events flow, materialized views update, exits fire (task #82 + #87) — but the inference and decision hooks are no-ops. Until they're wired, ModelService cannot drive any 8001 cutover; it's a watchful skeleton.

This spec defines the production wiring: how `_score_signal` consumes a `feature_snapshot` and emits a real signal dict, and how `_decide_trade` converts a BUY signal into a sized trade decision. Both must keep parity with the monolith's `cnn_agent` semantics so the cutover preserves trade behavior.

## Goal

Replace the stubs with production inference + decision logic so a fresh ModelService process, fed candle-close events, emits the same signal + trade decisions a monolith `cnn_agent` would emit for the same inputs. Parity, not novelty.

Specifically:
1. `_score_signal` routes by `self._model` to `xgb_signal.xgb_prob` (v3) or `xgb_signal.xgb_prob_shadow_v4_5` (v4_5), returns the signal dict in the shape the existing emit code expects.
2. `_decide_trade` returns a sized trade dict on BUY (with concurrency + balance guards) or `None` on HOLD / SELL.

## Non-Goals

- **Replacing the monolith.** Phase 4 of the parent spec is parallel-stream; this spec lands the inference in ModelService, the monolith keeps running. Promotion is a separate operator gate.
- **Kelly sizing.** Win-rate + payoff stats haven't accumulated yet in the event store. Fixed-fraction is the v1; Kelly is deferred to its own backlog item once `trade_closed` history accrues.
- **`deployment_n{N}.json` consumption.** The deployment artifact tells you *which model variant to run* and which PIDs to gate. ModelService already receives `--deployment`; threading the profile into per-trade sizing is a separate task (depends on Phase 4 selection being finalized).
- **Regime gating.** The v3 path in monolith uses an HMM regime flag; v4.5 does not. Mirror v4.5's simplicity here. If regime gating returns, it returns through `xgb_signal`, not through `_score_signal`.
- **SELL / short positions.** Monolith ignores SELL signals on un-held PIDs (BUY-only paper trading). Mirror that.
- **Per-tick SELL on a held position.** Exit logic in `_on_price_tick` already owns held-position exits (stop / trail). `_score_signal` returning SELL does not trigger a position-side change.
- **Real-money order routing.** Paper trading only; `actual_entry_price == intended_entry_price` always.

---

## Architecture

### File map

```
EDIT  backend/services/model_service.py        ~80 LoC: replace 2 stub methods
                                                +1 helper: _read_balance
                                                +2 module consts: _CAPITAL_PER_TRADE_FRAC, _MIN_TRADE_USD
                                                +1 module const: _MAX_CONCURRENT_PER_MODEL
EDIT  backend/services/events_schema.py        no change (materialized_balance already not needed)
EDIT  backend/tests/test_model_service.py      replace existing _score_signal / _decide_trade monkeypatches
                                                with real-inference tests; add fixtures for snapshot stubs
NONE                                            no new files; no schema changes
```

### Dependency graph (one-way, no back-edges)

```
model_service._score_signal  →  agents.xgb_signal.xgb_prob              (v3)
model_service._score_signal  →  agents.xgb_signal.xgb_prob_shadow_v4_5  (v4_5)
model_service._score_signal  →  cnn_agent._indep_thresholds_decision    (v4_5 rule)
model_service._decide_trade  →  model_service._read_balance             (helper)
model_service._read_balance  →  events table (aggregate over trade_decided / trade_closed)
```

ModelService imports from `agents.xgb_signal` and `agents.cnn_agent` only. It does NOT import `_CNNBook`, `database.py`, or any monolith state — Pattern C is event-sourced; the events table is the single source of truth.

### Module boundary — `model_service.py`

Two stub replacements + one helper + three module constants. No new classes.

```python
# Module-level (constants):
_CAPITAL_PER_TRADE_FRAC  = 0.10   # 10% of model's current balance per BUY
_MIN_TRADE_USD           = 10.0   # below this, the entry fee math degenerates
_MAX_CONCURRENT_PER_MODEL = 5     # max open positions per ModelService instance

# Instance methods (replace stubs):
async def _score_signal(self, pid: str, snapshot) -> Optional[dict]: ...
async def _decide_trade(self, pid, side, snapshot, signal_event_id) -> Optional[dict]: ...

# Instance method (new helper):
async def _read_balance(self) -> float: ...
```

---

## Data Flow

### `_score_signal` — candle-close handler

```
on_candle_close(evt)
  → build_for(evt.pid, db_path, tier='1h', lookback=360)    [existing]
      • returns FeatureSnapshot with .candles (list of 28-channel rows)
  → _score_signal(evt.pid, snapshot)                          [NEW WIRING]
      • if len(snapshot.candles) < 60: return None
      • if self._model == 'v3':
          channels = _build_channels(snapshot)               # 28×T transpose
          prob = xgb_signal.xgb_prob(channels, pid=pid)      # binary
          side = 'BUY' if prob > 0.50 else ('SELL' if prob < 0.50 else 'HOLD')
          return {'side': side, 'strength': prob,
                  'scores': {'p_up': prob},
                  'model_version': _xgb_artifact_hash('xgb_model.json'),
                  'feature_hash': _features_hash('xgb_features.json'),
                  'regime': None}
      • if self._model == 'v4_5':
          channels = _build_channels(snapshot)
          _v3, v45 = xgb_signal.xgb_prob_shadow_v4_5(channels, pid=pid)
          if v45 is None: return None                       # shadow failure — skip
          p_down, p_neu, p_up = v45
          side, strength = _indep_thresholds_decision(
              p_down, p_neu, p_up,
              thresh_up=float(os.getenv('XGB_V45_THRESH_UP', '0.50')),
              thresh_down=float(os.getenv('XGB_V45_THRESH_DOWN', '0.50')),
          )
          return {'side': side, 'strength': strength,
                  'scores': {'p_down': p_down, 'p_neutral': p_neu, 'p_up': p_up},
                  'model_version': _xgb_artifact_hash('xgb_model_v4_5.json'),
                  'feature_hash': _features_hash('xgb_features_v4_5.json'),
                  'regime': None}
  → emit signal_scored event with returned dict             [existing wiring]
  → call _decide_trade(pid, scored['side'], snapshot, sig_id) [existing wiring]
```

Note: `model_version` and `feature_hash` are short-form artifact hashes (first 12 hex chars of SHA-256 over the artifact file bytes). They identify *which artifact produced this signal*, enabling replay-determinism debugging. Helpers `_xgb_artifact_hash` and `_features_hash` are LRU-cached on path; first call hashes, subsequent calls return cached value.

### `_decide_trade` — BUY-decision gate

```
on_candle_close (continued)
  → _decide_trade(pid, side, snapshot, signal_event_id)
      • if side != 'BUY':                  return None    (HOLD or SELL → no trade)
      • if pid in self._positions_by_pid:  return None    (no stacking — one open per pid)
      • if len(self._positions_by_pid) >= _MAX_CONCURRENT_PER_MODEL: return None
      • balance = await self._read_balance()
      • size_usd = max(_MIN_TRADE_USD, balance * _CAPITAL_PER_TRADE_FRAC)
      • if size_usd > balance:             return None    (under-funded)
      • entry_price = snapshot.last_price
      • size        = size_usd / entry_price
      • fee_paid    = size_usd * self._FEE_RATE
      • return {
            'size':                  size,
            'size_usd':              size_usd,
            'intended_entry_price':  entry_price,
            'actual_entry_price':    entry_price,   # paper
            'fee_paid':              fee_paid,
            'trigger':               'SCAN',
            'deployment_profile_id': None,          # threaded later
        }
  → _emit_trade_decided(evt, decision, sig_id, side)       [existing wiring]
```

### `_read_balance` — event-sourced balance derivation

The events table is the single source of truth. ModelService computes its own balance by aggregating every `trade_decided` / `trade_closed` event it has previously produced.

```python
_STARTING_BALANCE_USD = 50_000.0  # mirrors _CNN_DRY_RUN_BALANCE in cnn_agent

async def _read_balance(self) -> float:
    """Reconstruct this model's paper-trading balance from event history.

    starting_balance
      - sum(trade_decided.size_usd + trade_decided.fee_paid)  ← capital tied up + entry fee
      - sum(trade_closed.pnl < 0 ? -pnl : 0) - sum(exit_fee)  ← already counted in entry; pnl resolves
    Re-credit closed-trade proceeds:
      + sum(trade_closed.exit_price * trade_closed.exit_size - exit_fee)

    Equivalently and more cleanly:
      balance = _STARTING_BALANCE_USD
              - sum(trade_decided.size_usd + fee_paid for OPEN positions)   ← capital in flight
              + sum(trade_closed.pnl for CLOSED trades)                     ← realized PnL
    """
    assert self._write_db is not None
    starting = _STARTING_BALANCE_USD

    open_capital_q = """
      SELECT COALESCE(SUM(json_extract(payload_json,'$.size_usd')
                        + json_extract(payload_json,'$.fee_paid')), 0)
      FROM events
      WHERE event_type='trade_decided'
        AND producer = ?
        AND json_extract(payload_json,'$.trade_uid') NOT IN (
          SELECT json_extract(payload_json,'$.trade_uid')
          FROM events
          WHERE event_type='trade_closed'
            AND producer = ?
        )
    """
    realized_pnl_q = """
      SELECT COALESCE(SUM(json_extract(payload_json,'$.pnl')), 0)
      FROM events
      WHERE event_type='trade_closed' AND producer = ?
    """
    cur = await self._write_db.execute(open_capital_q, (self._producer, self._producer))
    (open_capital,) = await cur.fetchone()
    cur = await self._write_db.execute(realized_pnl_q, (self._producer,))
    (realized_pnl,) = await cur.fetchone()
    return float(starting - open_capital + realized_pnl)
```

**Why this and not a materialized_balance table:** the events table already has everything; one SQL aggregate per BUY-candidate is cheap (~10ms over a year of trades with the existing `idx_events_producer` index). Materializing it adds a write path and a divergence risk. If the aggregate gets hot, materialize later.

**Producer scoping** prevents cross-model balance leakage — `model_v3` and `model_v4_5` run as separate ModelService processes with separate `producer` fields, each keeps its own balance. This matches the parent spec's "one ModelService per active model" rule.

---

## Decision: Constants

| Constant | Value | Reason |
|---|---|---|
| `_CAPITAL_PER_TRADE_FRAC` | `0.10` | 10% of balance per BUY, conservative (monolith uses Kelly which floats 0–25%; 10% sits below the cap as a Kelly-less default) |
| `_MIN_TRADE_USD` | `10.0` | Below ~$10 the round-trip fee (0.6% × 2 = 1.2%) eats meaningful edge |
| `_MAX_CONCURRENT_PER_MODEL` | `5` | Caps tail risk; monolith has no explicit cap but scans ~80 PIDs, rarely opens >5 at once |
| `_STARTING_BALANCE_USD` | `50_000.0` | Mirrors `_CNN_DRY_RUN_BALANCE` so 8001 ↔ Pattern C balance trajectories are comparable |
| `XGB_V45_THRESH_UP / DOWN` | `0.50 / 0.50` | Read from env, matches `config.py:103-104`. Env override allowed |

All constants live at module scope, not on `ModelService`, so test fixtures can monkeypatch them per-test.

## Decision: Inference path reuse

ModelService imports `xgb_signal.xgb_prob` (v3) and `xgb_signal.xgb_prob_shadow_v4_5` (v4_5) **directly from polymarket_app's `agents/xgb_signal.py`** — no copy, no fork. Pattern C's worktree shares the polymarket_app source tree at the file level; the import path is `from agents import xgb_signal`. This guarantees parity: when the monolith retrains and swaps artifacts, ModelService picks up the same artifacts on the next scan.

`_indep_thresholds_decision` is also imported (`from agents.cnn_agent import _indep_thresholds_decision`). It's a pure function with no side effects.

**Risk:** if `xgb_signal.xgb_prob` evolves a breaking signature change, ModelService breaks. Mitigation: the existing `xgb_prob` signature has been stable for ~5 months; any future signature change should bump a version (already true for `xgb_prob_v2`, `xgb_prob_v3`).

## Decision: Channels construction

`build_for(pid, db_path, tier='1h', lookback=360).candles` returns a list of dicts (per-candle rows with all 28 channels in named fields). `xgb_signal.xgb_prob` expects a list-of-lists: 28 sub-lists, each `T=360` floats. The transpose is identical to what `cnn_agent.generate_signal` builds today (`_build_channels_from_candles` at `cnn_agent.py:~1840`).

Decision: extract the transpose into a small helper inside `model_service.py` (not importing the monolith's `_build_channels_from_candles`, because that function is method-bound to `_CNNBook` and pulls in dependencies we don't want). 30 LoC, deterministic, easy to unit-test.

```python
def _build_channels_from_snapshot(snapshot) -> List[List[float]]:
    """Transpose snapshot.candles (list of 28-field dicts) to 28×T list-of-lists.
    Matches the layout xgb_signal expects. Last candle is the most recent.
    """
    rows = snapshot.candles
    ch_names = _CHANNEL_NAMES_28  # frozen constant — see cnn_agent._CHANNEL_NAMES
    return [[float(r.get(name, 0.0)) for r in rows] for name in ch_names]
```

`_CHANNEL_NAMES_28` is the canonical 28-channel ordered tuple. Lives in `services/feature_snapshot.py` (a `_CHANNEL_NAMES` constant alongside `build_for`), exported for re-use. If it doesn't already exist, add it as part of this work.

---

## Error Handling

| Failure point | Behavior |
|---|---|
| `snapshot.candles` has < 60 rows | `_score_signal` returns `None` → no signal_scored event emitted, no decision |
| `xgb_signal.xgb_prob` raises | Propagates to `_on_event`'s `try/except` → logged at ERROR, batch continues. Mirrors invariant #16. |
| `xgb_signal.xgb_prob_shadow_v4_5` returns `(_, None)` | `_score_signal` returns `None` (treats v4.5 inference failure as no signal, NOT as HOLD). Prevents spurious HOLD events. |
| `_indep_thresholds_decision` returns HOLD | `_score_signal` returns HOLD dict; signal_scored fires; no trade decided. |
| `_read_balance` returns 0 or negative | `_decide_trade` returns `None` (under-funded — explicit guard). |
| `_read_balance` raises (DB transient) | Propagates to `_on_event` `try/except` → logged, batch continues. Next candle close retries. |
| Position cap reached | `_decide_trade` returns `None`, no event emitted (silent skip). |
| Duplicate-pid BUY | `_decide_trade` returns `None`, no event emitted. Mirrors monolith's "one position per pid" rule. |

No failure path mutates `_positions_by_pid` or `_open_trades_by_pid`. Population happens only inside `_emit_trade_decided` (already wired in task #82).

---

## Testing Strategy

| Test | Purpose |
|---|---|
| `test_score_signal_v3_buy` | Mocks `xgb_signal.xgb_prob` to return 0.7 → asserts side='BUY', strength=0.7 |
| `test_score_signal_v3_hold` | Mocks `xgb_prob` to return 0.5 → asserts side='HOLD' |
| `test_score_signal_v3_sell` | Mocks `xgb_prob` to return 0.3 → asserts side='SELL', strength=0.3 |
| `test_score_signal_v4_5_indep_rule_buy` | Mocks `xgb_prob_shadow_v4_5` → `(p_down=0.1, p_neu=0.3, p_up=0.6)` with thresh 0.5 → BUY |
| `test_score_signal_v4_5_indep_rule_sell` | Mocks shadow → `(0.6, 0.3, 0.1)` → SELL |
| `test_score_signal_v4_5_shadow_returns_none` | Mocks shadow to return `(_, None)` → `_score_signal` returns None |
| `test_score_signal_short_snapshot_returns_none` | snapshot.candles len < 60 → returns None |
| `test_score_signal_inference_raises_propagates` | Mocked `xgb_prob` raises → exception propagates (caught by `_on_event`) |
| `test_decide_trade_buy_sizes_correctly` | balance=50_000, frac=0.10 → size_usd=5_000, size=5_000/price, fee=30 |
| `test_decide_trade_hold_returns_none` | side='HOLD' → None |
| `test_decide_trade_sell_returns_none` | side='SELL' → None |
| `test_decide_trade_existing_position_returns_none` | pid already in `_positions_by_pid` → None (no stacking) |
| `test_decide_trade_at_concurrency_cap_returns_none` | `_positions_by_pid` has 5 entries → None |
| `test_decide_trade_below_min_returns_none` | balance × 0.10 < 10.0 (low balance) → respects `_MIN_TRADE_USD` floor; if floor > balance → None |
| `test_decide_trade_negative_balance_returns_none` | `_read_balance` returns -1 → None |
| `test_read_balance_starting` | Empty events table → 50_000.0 |
| `test_read_balance_open_position` | One trade_decided of $5k+$30 fee, no close → 50_000 - 5030 = 44_970 |
| `test_read_balance_closed_winning_trade` | trade_decided $5k + trade_closed pnl=+$100 → 50_100 |
| `test_read_balance_scopes_to_producer` | Two producers' trade_decided events present → only own producer counted |

Test fixtures use the existing `tmp_path` aiosqlite pattern from `test_model_service.py`. Inference is monkeypatched at the `services.model_service.xgb_signal` module attribute (not at `agents.xgb_signal`), so the production import path is exercised but the heavy artifact load is bypassed. `_xgb_artifact_hash` and `_features_hash` are also monkeypatched to return fixed strings.

Total: 17 new tests (replace the existing 2-3 stub-monkeypatch tests with these; some existing test_model_service.py tests are repurposed as integration tests of the full candle-close → signal_scored → trade_decided → positions_by_pid path).

## Acceptance Criteria

1. Pattern C `test_model_service.py` test count grows by ≥10 (new inference + decision tests).
2. Full Pattern C pytest suite remains green (currently 1232 passing).
3. A new candle-close event for a v3 model with synthetic snapshot + mocked inference → signal_scored event written + trade_decided event written + `_positions_by_pid` populated.
4. A second candle-close event for the same pid does NOT emit a second trade_decided (concurrency guard).
5. `_read_balance` correctly handles three states: empty, one open position, one closed winning trade.
6. No new files outside `services/model_service.py`, `tests/test_model_service.py`, and (if missing) `services/feature_snapshot.py` (`_CHANNEL_NAMES_28` export).
7. No imports from monolith state (`_CNNBook`, `database.py`); only pure-function imports from `agents.xgb_signal` and `agents.cnn_agent._indep_thresholds_decision`.

---

## Open Questions (deferred, not blockers)

1. **`deployment_n{N}.json` consumption.** Phase 4 may select a profile per (pid, horizon). Currently `_deployment_path` is stored on the instance but unused. Threading it into `_decide_trade` (e.g., per-pid horizon-specific thresholds) is a separate task once Phase 4 selection is finalized. Defer.
2. **Kelly sizing.** Once `_read_balance` is in place and `trade_closed` history accumulates, switch `_CAPITAL_PER_TRADE_FRAC` to a Kelly fraction derived from rolling win-rate × payoff. Defer to its own backlog item.
3. **Exit-driven SELL path.** Monolith fires SELL only via exit triggers (stop / trail / model-down). Pattern C's exit path is `_on_price_tick`. If a `_score_signal=SELL` on a held position should also exit, that's a behavior change — separate spec.
4. **Concurrency cap = 5 — empirical?** No data backs this; pick + revisit. Reasonable bet for paper trading; tune once we have a week of head-to-head 8001 vs ModelService telemetry.
5. **Balance drift between ModelService and monolith.** As long as both run in parallel-stream, both maintain independent balances. Operator-visible comparison via the existing `/api/compare` endpoint. Promotion gate: ModelService balance trajectory tracks monolith within ~2% over a head-to-head week.

---

## Out of Scope (explicit non-goals, repeated for clarity)

- Removing the monolith.
- New schema columns.
- New tables.
- Touching `agents/xgb_signal.py` (frozen — it's the parity boundary).
- Touching port 8001 in any way.
- Adding env vars beyond the existing `XGB_V45_THRESH_UP` / `XGB_V45_THRESH_DOWN`.
- Changing `_FEE_RATE`, `_STOP_LOSS_PCT`, or `_GIVEBACK_FRAC` (already in `model_service.py`; tasks #84/#85 will converge those with PR #14 after merge).
