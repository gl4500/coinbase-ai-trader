# Event-Sourced Operational Architecture — Design Spec

**Date:** 2026-05-25
**Author:** Claude Opus 4.7
**Status:** Approved scope from operator (Pattern C chosen 2026-05-24); operator review pending on full spec
**Predecessor decisions:** `2026-05-25-event-sourced-architecture-decisions.md` (open-question answers on 6 architectural decisions)
**Predecessor findings:** `2026-05-24-live-ops-feedback-phase2-phase3.md` (live-ops evidence that surfaced the architectural need)

This spec describes a 3-process split of the current monolithic 8001 backend into **ingest worker**, **model service**, and **api server**, with a shared SQLite-backed event log as the substrate. It is **complementary to** — not a replacement for — the strategy-discovery research lane:

| Research lane (other CLIs) | This spec (operational lane) |
|---|---|
| Phase 1: data foundation | — |
| Phase 2: features + labels (per-token parquet) | — |
| Phase 3: tree mining + Q0 leaf profiles | — |
| Phase 4: portfolio knapsack + `deployment_n{N}.json` selection artifact | Pattern C consumes `deployment_n{N}.json` at runtime |
| Phase 4 explicit non-goal: "runtime inference path into `agents/cnn_agent.py`" | **Pattern C IS the integration commit Phase 4 punts on** |

---

## Goal

Replace the monolithic 8001 backend with:
1. **One ingest worker** that owns all data acquisition (WS, REST, candle backfill, marketcap) and persists every observation as an immutable event
2. **N model services** (one per active model) that consume events, run inference per their loaded `deployment_n{N}.json`, and persist their decisions as events
3. **One API server** that serves the frontend and reads materialized views built from events

The substrate is an append-only `events` table in SQLite-WAL. Every consumer maintains its own cursor. Replay = "advance any consumer's cursor over historical events to reproduce that consumer's decisions."

**Why this matters operationally:** today, restarting 8001 to swap a model also restarts the WS subscriber, loses cached price state, re-warms the scan loop. Multiple model experiments require multiple full backends. Backtest replay uses different code paths than live. After Pattern C: model swap = restart one process, WS state stays, dashboard stays responsive, backtest = same code reading historical events.

**Why this matters research-lane-wise:** Phase 4 emits a deployment artifact that says "run profile X on PID A at horizon h24, profile Y on PID B at horizon h72, sized via Kelly fraction Z." Today, no clean runtime hook accepts that artifact. Pattern C provides that hook (`model_service.py --deployment deployment_n3.json`).

---

## Non-goals (what this spec does NOT cover)

- **Replacing SQLite as a data store.** All event reads + writes go to `backend/coinbase.db` (production) or `backend/coinbase_dev.db` (dev). Future migration to Redis Streams / NATS is explicit Phase-2.x scope, NOT this spec.
- **Cross-machine deployment.** Single-host, single-sqlite-file. If we ever need multi-host, that's a separate redesign.
- **Multi-resolution OHLCV features** (5m/15m/4h/1D for live decisions). The `candle_close` event schema accommodates a `tier` column so future expansion is non-breaking, but adding new tier consumers is out of scope here. Wired in [[live_ops_feedback_phase2_phase3]] as a future brainstorm.
- **Cross-asset features** (BTC dominance, sector flow). Same deferral as above.
- **Re-architecting frontend.** Frontend continues to hit `http://localhost:8001` via REST + WebSocket. Internally `api_server.py` replaces today's `main.py` API surface; the frontend contract is unchanged.
- **Changing the inference algorithms** (v3, v4.5 logic, B2 exit, MODEL_DOWN). All math is preserved; this is purely an architectural reshape.

---

## Architecture overview

```
                          ┌──────────────────────────┐
                          │   Coinbase REST + WS     │
                          │   CoinPaprika, etc.      │
                          └────────┬─────────────────┘
                                   │ external API
                                   ▼
                   ┌──────────────────────────────┐
                   │   ingest_worker.py           │  ← single process
                   │   - WS subscriber             │
                   │   - REST fallback / backfill  │
                   │   - Marketcap fetch           │
                   │   - HMM regime emit (optional)│
                   └────────┬─────────────────────┘
                            │ INSERT events
                            ▼
                ┌────────────────────────────────────────┐
                │   coinbase.db (SQLite WAL)             │
                │   ┌──────────────┐                     │
                │   │ events table │ (append-only)       │
                │   └──────────────┘                     │
                │   ┌─────────────────────────┐          │
                │   │ consumer_cursors table  │          │
                │   └─────────────────────────┘          │
                │   ┌────────────────────────────┐       │
                │   │ materialized_* views/tables│       │
                │   └────────────────────────────┘       │
                └────────────┬───────────────────────────┘
                             │ poll cursor + read events
            ┌────────────────┼───────────────────┬──────────────────┐
            ▼                ▼                   ▼                  ▼
  ┌──────────────────┐ ┌──────────────────┐ ┌────────────────┐  ┌──────────────────┐
  │ model_service    │ │ model_service    │ │ api_server.py  │  │ exit_watcher     │
  │ --model v3       │ │ --model v4.5     │ │ (port 8001)    │  │ (or in-model)    │
  │ --deployment …   │ │ --deployment …   │ │ - REST + WS    │  │ - WS price tick  │
  └──────────────────┘ └──────────────────┘ │ - materialize │  │ - emit exit evt  │
            │                  │            │   views        │  └──────────────────┘
            └──────────────────┴────────────┴────────────────┘
                               │
                               ▼
                      INSERT signal_scored,
                      trade_decided,
                      trade_closed events
```

**Process inventory after cutover:**

| Process | Role | Port | What it writes | What it reads |
|---|---|---|---|---|
| `ingest_worker.py` | Data acquisition | none (worker) | `price_tick`, `candle_close`, `marketcap_snapshot`, `regime_classified` events | Coinbase API, CoinPaprika, env config |
| `model_service.py --model v3 --deployment <path>` | v3 inference + decisioning | none (worker) | `signal_scored`, `trade_decided`, `trade_closed`, `exit_triggered` events | events table (own cursor), Phase 4 deployment JSON |
| `model_service.py --model v4_5 --deployment <path>` | v4.5 inference + decisioning | none (worker) | same as above | same |
| `api_server.py` | Frontend bridge | **8001** | Updates `materialized_*` views | events table + materialized views |

`exit_watcher` becomes either (a) a sub-thread inside each `model_service`, or (b) a small standalone process. Defaulting to (a) for simplicity — each model service owns its own positions and runs its own exit checks. See [Component spec: model_service.py](#component-spec-model_servicepy).

---

## Event store schema

### `events` table — append-only, immutable

```sql
CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,  -- monotonic; gives total ordering
    ts_ms           INTEGER NOT NULL,                    -- event observation time, milliseconds
    event_type      TEXT    NOT NULL,                    -- enum (see catalog below)
    pid             TEXT,                                -- nullable for non-PID events (regime, etc.)
    payload_json    TEXT    NOT NULL,                    -- typed payload, schema per event_type
    schema_version  INTEGER NOT NULL DEFAULT 1,
    producer        TEXT    NOT NULL,                    -- 'ingest' | 'model_v3' | 'model_v4_5' | ...
    write_ts_ms     INTEGER NOT NULL DEFAULT (strftime('%s','now')*1000)  -- internal: when SQLite saw it
);

CREATE INDEX idx_events_ts          ON events(ts_ms);
CREATE INDEX idx_events_pid_ts      ON events(pid, ts_ms)
                                    WHERE pid IS NOT NULL;
CREATE INDEX idx_events_type_ts     ON events(event_type, ts_ms);
CREATE INDEX idx_events_producer    ON events(producer, id);  -- for consumer cursor advances
```

### `consumer_cursors` table

```sql
CREATE TABLE consumer_cursors (
    consumer_name       TEXT    PRIMARY KEY,    -- 'model_v3' | 'model_v4_5' | 'api_view_builder' | etc.
    last_processed_id   INTEGER NOT NULL,
    updated_at          INTEGER NOT NULL DEFAULT (strftime('%s','now')*1000)
);
```

### Materialized view tables (built by consumers)

These are NOT events; they're current-state snapshots maintained by consumers for fast frontend reads. Each consumer rebuilds its own views from events.

```sql
-- maintained by api_server.py
CREATE TABLE materialized_latest_price (
    pid          TEXT PRIMARY KEY,
    price        REAL,
    bid          REAL,
    ask          REAL,
    pct_change_24h REAL,
    last_event_id INTEGER,
    last_updated_ts_ms INTEGER
);

-- maintained by each model_service per its deployment
CREATE TABLE materialized_positions_<model_name> (
    pid          TEXT PRIMARY KEY,
    size         REAL,
    avg_price    REAL,
    position_dollars REAL,
    entry_time_ms INTEGER,
    peak_price   REAL,
    peak_pnl_pct REAL,
    last_event_id INTEGER
);
```

---

## Event type catalog

All payloads serialized as JSON in `payload_json`. The schema below is the canonical contract for `schema_version = 1`.

### 1. `price_tick`
**Producer:** `ingest`
**Payload:**
```jsonc
{
  "pid": "BTC-USD",         // also in events.pid column for indexing
  "price": 95123.45,
  "bid": 95120.00,          // nullable
  "ask": 95125.00,          // nullable
  "volume_24h": 12345.67,   // nullable
  "source": "ws"            // 'ws' | 'rest_fallback'
}
```
**Causality:** None. Pure observation.

### 2. `candle_close`
**Producer:** `ingest`
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "tier": "1h",             // '1m' | '5m' | '15m' | '1h' | '4h' | '1d'
  "open":  95000.0,
  "high":  95400.0,
  "low":   94800.0,
  "close": 95123.45,
  "volume": 1234.56,
  "bar_ts_ms": 1716595200000  // start of the bar; events.ts_ms = close time
}
```
**Causality:** None. Emitted once when a candle is sealed (REST-validated for completeness).

### 3. `marketcap_snapshot`
**Producer:** `ingest`
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "market_cap": 1_900_000_000_000.0,
  "fdv":        2_100_000_000_000.0,
  "circ_supply":  19_700_000,
  "total_supply": 19_800_000,
  "vol_24h":      40_000_000_000.0,
  "source": "coinpaprika"
}
```
**Causality:** None.

### 4. `signal_scored`
**Producer:** `model_v3`, `model_v4_5`, etc.
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "model": "v4_5",
  "model_version": "h168_2026-05-23",   // sha or artifact tag
  "feature_hash": "ab12cd...",            // hash of the input features for replay verification
  "scores": {                              // model-specific
    "p_up": 0.556, "p_down": 0.183, "p_neutral": 0.261
  },
  "side": "BUY",                           // 'BUY' | 'SELL' | 'HOLD'
  "strength": 0.556,                       // normalized 0..1
  "regime": "TRENDING",                    // nullable
  "deployment_profile_id": "profile_42",   // nullable — set if a Phase 4 deployment_n{N}.json
                                            //   profile triggered the score
  "input_event_ids": {                     // explicit causal references
    "last_price_tick_id": 12345,
    "last_candle_close_id": 12342
  }
}
```
**Causality:** References the price/candle events that fed inference. Allows deterministic re-derivation in replay.

### 5. `trade_decided`
**Producer:** `model_v3`, `model_v4_5`, etc.
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "model": "v4_5",
  "side": "BUY",
  "size":  0.001,
  "size_usd": 95.12,
  "intended_entry_price": 95123.45,
  "actual_entry_price":   95123.45,         // = intended in dry-run; differs with slippage in live
  "fee_paid": 0.5707,                       // 0.6% of size_usd
  "trigger": "SCAN",                        // 'SCAN' | other entry trigger
  "signal_event_id": 67890,                 // explicit: which signal_scored caused this
  "deployment_profile_id": "profile_42",    // nullable
  "trade_uid": "v4_5_BTC-USD_1716595800"   // model-unique id; appears in subsequent trade_closed
}
```
**Causality:** `signal_event_id` is mandatory. A `trade_decided` without it is malformed.

### 6. `trade_closed`
**Producer:** `model_v3`, `model_v4_5`, etc.
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "trade_uid": "v4_5_BTC-USD_1716595800",   // matches trade_decided
  "exit_price": 96000.0,
  "exit_size":  0.001,
  "pnl":  +0.871,
  "pct_pnl": +0.91,
  "hold_secs": 14400,
  "trigger_close": "WS_TRAIL_STOP",          // 'WS_TRAIL_STOP' | 'STOP_LOSS' | 'WS_MODEL_DOWN' | ...
  "decision_event_id": 67890,                // links to trade_decided
  "exit_signal_event_id": 71234              // nullable: signal_scored that triggered the close
}
```

### 7. `exit_triggered`
**Producer:** `model_v3.exit_watcher`, `model_v4_5.exit_watcher`, etc.
**Payload:**
```jsonc
{
  "pid": "BTC-USD",
  "trade_uid": "v4_5_BTC-USD_1716595800",
  "trigger_type": "WS_TRAIL_STOP",           // 'WS_TRAIL_STOP' | 'STOP_LOSS' | 'WS_MODEL_DOWN' | 'MAX_HOLD'
  "peak_pnl_pct": 1.15,
  "current_pnl_pct": 0.91,
  "exit_threshold": 0.93,                     // B2 computed threshold at the time
  "price_at_trigger": 96000.0,
  "trigger_price_event_id": 78901             // which price_tick crossed the threshold
}
```
**Causality:** `trigger_price_event_id` is mandatory. Enables replay to re-verify exit firing.

---

## Consumer cursor management

Every consumer (each model service, the API view builder) operates the same way:

```python
# Pseudocode — actual implementation in services/event_consumer.py
class EventConsumer:
    def __init__(self, name: str, db_path: str):
        self.name = name
        self.db = sqlite3.connect(db_path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.cursor_id = self._load_or_init_cursor()

    def _load_or_init_cursor(self) -> int:
        row = self.db.execute(
            "SELECT last_processed_id FROM consumer_cursors WHERE consumer_name = ?",
            (self.name,)
        ).fetchone()
        if row: return row[0]
        # Default: start from current event (don't replay all history on first run)
        last_id = self.db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0]
        self.db.execute(
            "INSERT INTO consumer_cursors (consumer_name, last_processed_id) VALUES (?, ?)",
            (self.name, last_id)
        )
        return last_id

    def poll(self, batch_size: int = 1000) -> List[Event]:
        rows = self.db.execute("""
            SELECT id, ts_ms, event_type, pid, payload_json, schema_version, producer
            FROM events WHERE id > ? ORDER BY id ASC LIMIT ?
        """, (self.cursor_id, batch_size)).fetchall()
        return [Event.from_row(r) for r in rows]

    def commit(self, event_id: int) -> None:
        self.db.execute(
            "UPDATE consumer_cursors SET last_processed_id = ?, updated_at = strftime('%s','now')*1000 "
            "WHERE consumer_name = ?",
            (event_id, self.name)
        )
        self.cursor_id = event_id
```

**Cursor commit discipline:**
- Consumer processes events strictly in order
- Commits AFTER processing (at-least-once delivery; consumers must be idempotent)
- If a consumer crashes mid-batch, on restart it re-processes from `last_processed_id + 1` — which means processing the same event again. Consumers MUST detect this (e.g., via `trade_uid` uniqueness in `materialized_positions_*`).

---

## Component spec: `ingest_worker.py`

**Single source of truth for all observations.** Replaces:
- `backend/services/coinbase_ws_subscriber.py` (WS handling stays internally but write-out goes to events)
- The WS price update path inside `main.py`
- Candle backfill logic
- Marketcap fetches scheduled in `main.py`

**Public CLI:**
```bash
python -m services.ingest_worker [--db backend/coinbase.db] [--products BTC-USD,ETH-USD,...] [--no-marketcap]
```

**Responsibilities:**
1. WS subscriber loop (existing logic adapted) — every price update writes one `price_tick` event
2. Candle close detection — when a candle's bar boundary passes + REST-confirmed, emit `candle_close` for every tracked tier per PID
3. Marketcap polling (existing CoinPaprika integration) — every poll writes a `marketcap_snapshot` event
4. HMM regime classification (if `HMM_ENABLED=true`) — on each candle close, computes regime, emits `regime_classified` event
5. **Never** runs inference. Never makes trade decisions. Never modifies the book.

**Failure modes:**
- WS disconnect → reconnect logic, no events lost (WS will replay from sequence)
- REST API rate limit → backoff, emit `ingest_error` event with details (new event type for failures)
- DB write failure → buffered in-memory, retry, alert via `ingest_error`

**Ops health endpoint (optional, port 8101):**
- `GET /health` → `{"ws_connected": true, "last_event_ms_ago": 142, "events_written_total": 12345}`

---

## Component spec: `model_service.py`

**One process per active model.** Loads a Phase 4 `deployment_n{N}.json` artifact (or runs in "raw mode" without one) and runs inference per profile.

**Public CLI:**
```bash
python -m services.model_service \
    --model v4_5 \
    --deployment backend/data/phase4/deployment_n3.json \
    --db backend/coinbase.db \
    [--paper]    # dry-run mode; no live orders
```

**What it does each tick** (every `SCAN_INTERVAL_SECONDS`, default 15s):

1. Poll events via cursor (`EventConsumer.poll()`)
2. Build per-PID feature snapshots from event history (price ticks + candle closes + marketcap snapshots)
3. For each PID in its deployment's universe:
   - Run model inference (v3, v4_5, etc.) — emit `signal_scored` event
   - Apply deployment profile rules → decide BUY / HOLD / SELL
   - If BUY and no open position: open position, emit `trade_decided`
   - If position open: run exit checks (B2 trail, STOP_LOSS, MODEL_DOWN) on every price_tick event read this batch — emit `exit_triggered` + `trade_closed`
4. Commit cursor

**State management:**
- Positions live in `materialized_positions_<model_name>` table — updated on every `trade_decided`/`trade_closed` event processed
- Cache: `feature_snapshot_cache` is in-memory but reconstructed from events on restart (read backwards N hours from current cursor)

**Phase 4 deployment integration:**
- `deployment_n{N}.json` contains per-PID profile assignments (which Phase 3 leaf profile to apply to which PID at which horizon)
- Model service loads this on startup; each scan iteration applies the deployment rules
- If `--deployment` is omitted, falls back to a "global threshold" mode matching today's behavior (single BUY threshold across all PIDs)

**Concurrency safety:**
- Multiple `model_service` instances can run side-by-side (different `--model` flags)
- Each maintains independent cursor + materialized view tables (no shared state)
- Exit watcher per model uses the existing per-pid asyncio.Lock pattern (preserved from current `exit_watcher.py`)

---

## Component spec: `api_server.py`

**Replaces today's `main.py` API surface.** No scan loop. No model inference. No exit checks. Just reads materialized views + tails events for WebSocket push.

**Public CLI:**
```bash
PORT=8001 python -m services.api_server [--db backend/coinbase.db] [--frontend-origin http://localhost:5174]
```

**Endpoints (preserved from today, internally re-implemented):**
- `GET /api/status` → reads `materialized_*` tables
- `GET /api/agents/status` → reads `materialized_positions_<model_name>` for each active model
- `GET /api/products` → reads `materialized_latest_price` + tracked-products config
- `GET /api/trades` → reads events directly: `SELECT ... FROM events WHERE event_type IN ('trade_decided', 'trade_closed') ...`
- `GET /api/compare` → reads `materialized_positions_v3` + `materialized_positions_v4_5`
- `GET /api/equity_curve` → reads events: cumulative PnL series from `trade_closed` events
- `POST /api/trading/{enable,disable}` → writes a `system_control` event; model services listen for it

**WebSocket bridge (`/ws`):**
- Tails the events table (cursor-based, same pattern as model services)
- Filters for events relevant to the frontend (`price_tick`, `trade_decided`, `trade_closed`, system_control)
- Pushes selected events as state-update messages (preserves today's frontend protocol)

**View materializer (background task inside api_server):**
- Same cursor pattern as a model service consumer
- Updates `materialized_latest_price` on every `price_tick`
- Updates `materialized_*` views on relevant events
- Decouples the heavy view logic from the API request path (frontend requests don't trigger expensive aggregations)

---

## Determinism + replay

**Goal:** given the same `events` table state, replaying any consumer from event id 0 to event id N produces the same set of `signal_scored` + `trade_decided` + `trade_closed` events that the original consumer emitted live.

**Deterministic components:**
- Feature computation (pure function of `candle_close` + `price_tick` events)
- XGB inference (deterministic given model artifact + features)
- B2 exit math (pure function of peak_pnl, position_dollars, fee constants)
- Deployment profile application (pure logic given profile rule + feature inputs)

**Non-deterministic — flagged as caveats:**
1. **External API call results** (marketcap, regime). Live: ingest_worker calls API and emits an event. Replay: events are pre-recorded; no API call. Determinism preserved IF the original event was recorded; lost if a live event has no recorded equivalent (e.g., the regime classifier was added mid-window).
2. **WS jitter and arrival ordering.** Two `price_tick` events that arrive within ms can have observed ordering swapped between live and replay. Mitigation: replay uses `events.id` ordering, not `ts_ms`, so a single canonical ordering is enforced.
3. **Concurrent trade decisions across model services.** Each model service has its own state; running multiple replays in parallel doesn't change a single replay's determinism.

**Replay tooling (delivered in implementation):**
- `python -m tools.replay_consumer --model v4_5 --from-event 0 --until-event 1000000 --output replayed_events.parquet`
- Runs in a sandboxed DB (clone of production DB) so replay doesn't pollute live cursors
- Output is the model's emitted events; diff against original = determinism verification

---

## Migration plan — 4 phases

### Phase 1: Add event store + ingest_worker (1-2 weeks)

**Land in `coinbase.db`:**
- `events` table + indices
- `consumer_cursors` table

**Build:**
- `services/event_writer.py` — typed event INSERT helpers per event_type
- `services/ingest_worker.py` — pulls existing WS + REST + marketcap logic, wraps with event_writer
- `tools/event_inspector.py` — CLI to query events for debugging

**Verify:**
- Run `ingest_worker` alongside the existing 8001 monolith
- Event stream contains every observation; no gaps
- Existing 8001 monolith still works (no change to its behavior; ingest_worker writes events as a side channel)

**Cutover gate:** events table has 24+ hours of data, no gaps, no schema drift.

### Phase 2: First consumer — model_v3 service running parallel (1-2 weeks)

**Build:**
- `services/event_consumer.py` — base class with cursor logic
- `services/model_service.py` — parameterized for `--model v3`
- `services/feature_snapshot.py` — rebuilds feature inputs from event history

**Verify:**
- `model_service --model v3` runs alongside the existing 8001 monolith scan loop
- Both produce `signal_scored`-equivalent records (monolith writes to `cnn_scans`, service writes to events)
- **Cross-check protocol:** every 1h, run a diff: for the same `(pid, scan_ts_ms ± 30s)`, do the two paths produce the same side + strength? Acceptable divergence: <1% of scans differ (due to feature timing). Anything more is a bug.

**Cutover gate:** 48+ hours of cross-check with <1% divergence.

### Phase 3: Cutover — model_service replaces monolith scan loop (1 week)

**Change:**
- Disable scan loop in monolith `main.py` (gate behind `MONOLITH_SCAN_DISABLED=true` env)
- Promote `model_service.py` to primary v3 driver
- Bring up `model_service.py --model v4_5` for shadow
- Monolith `main.py` becomes today's API server until step 4

**Verify:**
- Trade ledger continuity (no missing trades; trade_decided + trade_closed events match what would have been written to `trades` table — and in fact the `trades` table is replaced by a view over events)
- Frontend keeps working (still hits the monolith's port 8001 — see step 4)

**Cutover gate:** 1 week of model_service-only paper trading; no regressions vs Phase 2 baseline.

### Phase 4: Replace monolith with `api_server.py` (1 week)

**Change:**
- Build `services/api_server.py` from existing FastAPI endpoints in `main.py`
- Move all endpoint logic to read from materialized views or events
- Cut over: stop the monolith `main.py`, start `api_server.py` on port 8001
- Frontend sees no behavior change

**Cutover gate:** Frontend smoke tests pass; all dashboards render correctly.

### Phase 5 (optional, out of scope for this spec): Phase 4 deployment integration

- Once Phase 4 (research lane) lands deployment_n{N}.json artifacts, wire model_service `--deployment` flag
- This is a single follow-on commit, not a full phase

---

## Testing strategy

### Per-module unit tests
- `test_event_writer.py` — typed inserts, schema validation, idempotency
- `test_event_consumer.py` — cursor advance, batch reads, restart recovery
- `test_feature_snapshot.py` — rebuild feature inputs from synthetic events; assert numerical equality with current cnn_agent feature path
- `test_model_service.py` — feed synthetic event stream; assert signal_scored + trade_decided outputs

### Integration tests
- `test_ingest_to_consumer_e2e.py` — boot ingest_worker against a mock WS feed; assert events land in DB; assert model_service consumes them; assert decisions match expected
- `test_replay_determinism.py` — record live events for 1h, replay model_service against the recording, assert emitted events match the live emissions byte-for-byte

### Migration safety tests
- `test_phase2_crosscheck.py` — runs both monolith and event-driven path on synthetic input, asserts <1% divergence
- `test_no_event_gaps.py` — guarantees there's no event_type producer-consumer pair where the consumer expects something the producer doesn't emit

---

## Operator integration (post-implementation)

**Starting the new stack:**
```bash
# Production (port 8001) — start in this order
python -m services.ingest_worker --db backend/coinbase.db &
python -m services.model_service --model v3 --db backend/coinbase.db &
PORT=8001 python -m services.api_server --db backend/coinbase.db

# Optional shadow on dev DB
python -m services.ingest_worker --db backend/coinbase_dev.db &
python -m services.model_service --model v4_5 --db backend/coinbase_dev.db &
```

**Stopping a model service** (e.g., to swap to a new version):
```bash
kill $MODEL_SERVICE_PID
# Model state preserved in materialized_positions_v3 + events
# Restart with same args; resumes from cursor
python -m services.model_service --model v3 --db backend/coinbase.db &
```

**Frontend impact:** zero. `http://localhost:8001` is still the API; the WS protocol is preserved.

---

## What this spec is NOT

- A redesign of the dashboard or frontend
- A change to model accuracy or trade decision logic
- A migration to a different DB engine
- A multi-host deployment story
- A real-time / sub-second push system (cursor polling at 5-15s matches today's scan cadence)
- A "rewrite the whole backend" — most existing logic (B2 exits, MC filters, XGB inference, exit_watcher math) is reused; only the dispatch boundary changes

---

## Backlog (deferred)

| ID | Item | Trigger to revisit |
|---|---|---|
| #61 | Redis Streams side-channel for sub-second push to consumers | When poll-latency becomes a measurable bottleneck (current scan loop is 5-15s — far from this limit) |
| #62 | Multi-resolution OHLCV features (5m/15m/4h/1D in `candle_close.tier`) | After Phase 4 baseline lands; needs feature brainstorm round |
| #63 | Cross-asset features (BTC dominance, sector flow) emitted as events | Same as #62 |
| #64 | Event store partitioning by month/quarter (single-file SQLite grows unboundedly) | When `events` table exceeds ~100GB or query times degrade noticeably |
| #65 | Replay-driven backtest harness as separate CLI | After Phase 4 baseline; current backtest tooling is separate from this architecture |

---

## See also

- `2026-05-25-event-sourced-architecture-decisions.md` — open-question decisions (Q1-Q6) this spec builds on
- `2026-05-24-live-ops-feedback-phase2-phase3.md` — empirical evidence motivating this work
- `2026-05-23-strategy-discovery-phase2-design.md` — features + labels (per-token parquet aligns with our `candle_close` events)
- `2026-05-24-strategy-discovery-phase3-design.md` — tree mining + Q0 leaf profiles
- `2026-05-24-strategy-discovery-phase4-design.md` — portfolio knapsack + `deployment_n{N}.json` (the artifact this architecture consumes at runtime)
- `2026-05-23-pnl-anchored-trail-design.md` — B2 trail math (reused inside model_service exit checks)
- `2026-05-23-ws-exit-checker-design.md` — current `exit_watcher.py` (becomes a sub-thread of model_service)
