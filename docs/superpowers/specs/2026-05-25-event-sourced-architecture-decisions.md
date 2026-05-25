# Pattern C — Event-Sourced Architecture: Open-Question Decisions

**Date:** 2026-05-25
**Author:** Claude Opus 4.7
**Status:** Decisions draft — for operator review before spec expansion
**Context:** Operator chose Pattern C over A/B/defer in 2026-05-24 session. Goal is to separate operational data ingest from model actions so:
- Model swaps don't restart WS state
- Multiple model services can subscribe to identical data
- Backtest replay uses same event interface as live
- Failure isolation by concern

**Predecessor:** `2026-05-24-live-ops-feedback-phase2-phase3.md` (live-ops feedback that surfaced the architectural need)

This doc answers 6 design questions with my best judgment. **Operator review gate before any spec writing.** If a decision is wrong, redirect now — cheaper than rewriting a 500-line spec.

---

## Q1. Event store substrate

**Options considered:**

| Option | Pros | Cons |
|---|---|---|
| **Per-PID Parquet partitions** (one parquet per PID per day) | Aligns with Phase 2 / Phase 3 per-token parquet schema; cheap reads via DuckDB / pyarrow; natural backtest replay (just read the parquets) | Eventual-consistency write latency (minute-scale); not great for real-time consumer push |
| **SQLite append-only with WAL mode** | Same SQLite we already use; trivial to integrate; readers don't block writers in WAL mode; transactional | Single-file bottleneck at high volumes; no built-in push semantics — consumers poll |
| **Redis Streams** | Real-time push to consumers; cheap on local box; replay via XRANGE | New infra dep; persistence story weaker (snapshot + AOF, but not great for years of events); RAM footprint grows |
| **NATS JetStream / Kafka** | Production-grade pub/sub; battle-tested replay | Massive ops overhead for retail-scale local single-box; orthogonal to existing SQLite stack |

**Decision: SQLite append-only with WAL mode** — at least to start.

**Why:**
- We're already on SQLite (`coinbase.db`); zero new infrastructure to learn
- WAL mode gives reader/writer concurrency (no read blocking)
- Pure append (no UPDATE/DELETE on event rows) is a clean discipline
- Poll-based consumers are fine at scan-loop cadence (every 5-15 seconds isn't real-time, but it matches today's behavior)
- **Future migration path:** if poll-latency becomes a bottleneck, add Redis Streams as a side channel for low-latency push *while keeping SQLite as the durable log*. This is a one-way door we don't have to walk through now.

**What this looks like:**

```sql
CREATE TABLE events (
    id            INTEGER PRIMARY KEY,    -- monotonic append-only
    ts_ms         INTEGER NOT NULL,        -- event timestamp in ms (not write time)
    event_type    TEXT NOT NULL,            -- 'price_tick' | 'signal_scored' | 'trade_decided' | 'trade_closed' | 'exit_triggered'
    pid           TEXT,                     -- nullable for non-PID events
    payload_json  TEXT NOT NULL,            -- typed payload per event_type
    schema_version INTEGER NOT NULL DEFAULT 1,
    producer      TEXT NOT NULL             -- 'ingest' | 'model_v3' | 'model_v45' | etc.
);
CREATE INDEX idx_events_ts ON events(ts_ms);
CREATE INDEX idx_events_pid_ts ON events(pid, ts_ms);
CREATE INDEX idx_events_type_ts ON events(event_type, ts_ms);
```

---

## Q2. Event schema (what's an "event"?)

**Decision: 5 core event types, all immutable:**

| event_type | Producer | Payload | Why an event |
|---|---|---|---|
| `price_tick` | ingest | `{pid, price, bid, ask, vol, source}` | Every WS update; also REST fallback fetches |
| `candle_close` | ingest | `{pid, ts_ms, open, high, low, close, volume, tier}` | Hourly + 5m + 15m bar closes as separate events |
| `signal_scored` | model_v3 / model_v45 | `{pid, ts_ms, side, strength, prob_up, prob_down, features_hash, model_version}` | One per scan loop iteration per pid |
| `trade_decided` | model service | `{pid, ts_ms, side, size, entry_price, reason, signal_id}` | Decision moment; references the scoring event |
| `trade_closed` | model service / exit watcher | `{trade_id, pid, exit_price, pnl, hold_secs, trigger_close}` | Closes a position |
| `exit_triggered` | exit watcher | `{pid, trigger_type, peak_pnl_pct, current_pnl_pct}` | When trail / stop_loss / model_down fires |

**Key constraints:**
- **Events are immutable** — never UPDATE a row, never DELETE. Any correction is a new event.
- **Events are causal** — `trade_decided` carries `signal_id` referencing the `signal_scored` event that triggered it. This makes replay deterministic.
- **No derived state in events** — only raw observations + decisions. Aggregates (balance, open positions) are materialized views built by consumers.

---

## Q3. Consumer model

**Decision: Cursor-based polling consumers.**

Each model service maintains a `last_processed_event_id` cursor (per-service-instance, persisted in its own table). On each tick:

```python
# Pseudocode in each model service
while True:
    events = db.execute("""
        SELECT * FROM events WHERE id > ? ORDER BY id ASC LIMIT 1000
    """, (last_processed_id,))
    for event in events:
        process(event)
        last_processed_id = event.id
        persist_cursor(last_processed_id)
    sleep(scan_interval)
```

**Why not push:**
- SQLite has no native push; bolting on LISTEN/NOTIFY or a side-channel adds complexity
- Polling at 5-15s intervals is what the scan loop already does; matches current cadence
- Multiple consumers don't fight each other — each has its own cursor

**Why cursor-based not timestamp-based:**
- `id` is monotonic; `ts_ms` can have ties or out-of-order arrivals
- Cursor is recoverable if a consumer crashes mid-batch

---

## Q4. Time-travel / replay guarantees

**Decision: Best-effort deterministic replay, with explicit non-determinism flags.**

Replay = "given events 1..N, can model service v3 produce the same `signal_scored` events it produced live?"

**Will be deterministic:**
- Feature computation from `candle_close` events (same code, same input → same features)
- XGB inference (deterministic given the model artifact)
- B2 exit threshold math (pure function of peak_pnl + position_dollars)

**Will NOT be deterministic — flagged as out-of-band state:**
- Network latency / WS arrival jitter (the raw `price_tick` event timestamps capture observed-time, not actual-market-time — replay can't change this)
- Race conditions in concurrent buy/sell (the per-pid asyncio.Lock makes the live order deterministic given the event stream, but if events arrive out of order in replay vs live, lock ordering may differ)
- External REST calls during scan (marketcap fetch, etc.) — these become events themselves: `marketcap_snapshot` event captures what was returned at that time

**Implication for spec:**
- Any external API call during a scan must be wrapped: result goes into an event, then consumer reads from the event (not the API). Live mode: ingest calls API + emits event. Replay mode: events are pre-existing, no API call.

---

## Q5. Migration path

**Decision: Parallel-stream — both old and new systems run simultaneously during cutover.**

**Phase 0 (this design):** Approved
**Phase 1 (event store):** Add `events` table to existing `coinbase.db`; ingest worker writes price ticks + candle closes alongside existing direct writes. No consumers yet — just verify event stream is correct + complete.
**Phase 2 (first consumer):** New model_v3 service reads `events` AND writes to its own `model_v3_signals` table. Existing 8001 scan loop also continues, writes to existing `cnn_scans`. Cross-check: do the two paths agree?
**Phase 3 (cutover):** Once Phase 2 shows agreement, the existing scan loop in 8001 gets replaced by the event-consumer model service. Trades, exits, telemetry all flow through events.
**Phase 4 (parallel models):** Add model_v45 service, model_v5 service, etc. — each subscribes to same event stream independently.

**Why parallel-stream:**
- Stop-the-world cutover risks losing paper-trading data continuity (operator decision: keep data flowing)
- Cross-check phase (Phase 2) is the only way to validate the event-driven path produces same decisions as monolith
- If event-driven path diverges from monolith, we can debug live by comparing both

---

## Q6. What 8001 looks like after the refactor

**Decision: 8001 splits into THREE processes:**

| Process | Role | Talks to |
|---|---|---|
| `ingest_worker.py` | WS subscriber + REST fallback + candle backfill + marketcap fetch. Writes `price_tick`, `candle_close`, `marketcap_snapshot` events. No model logic. | Coinbase API, `coinbase.db` (writes events) |
| `model_service.py --model v3` | Polls events, runs v3 inference, writes `signal_scored` + `trade_decided` + `trade_closed`. Maintains its own simulated book. | `coinbase.db` (reads events, writes own events) |
| `api_server.py` (replaces 8001 main) | FastAPI + WebSocket frontend bridge. Reads materialized views from events. Doesn't run inference. | `coinbase.db` (reads), frontend (WS) |

`model_service.py` can spawn multiple instances: `--model v45`, `--model v5`, etc. Each tracks own cursor + book. The current 8002 dev backend pattern (`PORT=8002 MODEL_BACKEND=xgb_v45`) becomes `python model_service.py --model v45 --paper`.

**Port allocation after:**
- Port 8001: `api_server.py` (frontend's hardcoded target stays)
- No port for ingest_worker — it's a worker, not a server
- No port for model services — they're workers too
- Optional: a debug HTTP endpoint per worker on 8101+, for ops health checks

---

## What's IN scope for the spec (next iteration)

- The `events` table schema (locked from Q2)
- Cursor management (per-service `consumer_cursors` table)
- The 6 event types with full payload schemas
- `ingest_worker.py` design (state machines for WS, REST fallback, candle backfill)
- `model_service.py` design (cursor loop, B2 exit logic adaptation, book management from events)
- `api_server.py` redesign (drop scan loop, add event-stream WebSocket for frontend)
- Phase 1-4 migration plan with detailed cross-check protocols
- Test strategy (event determinism tests, replay tests, parallel-stream divergence detector)

## What's OUT of scope for now

- Real-time push consumers (Redis Streams etc.) — future if poll-latency becomes a problem
- Cross-machine deployment (everything stays single-box, single sqlite file)
- Replacing the dashboard's WebSocket protocol (frontend stays the same; API server adapts internally)
- Multi-resolution candle features (separate Phase-2.x brainstorm per live-ops feedback doc)
- Cross-asset features (separate brainstorm)

---

## Decisions needing operator confirmation

Tagging each decision for explicit go/no-go:

| # | Decision | Confidence |
|---|---|---|
| Q1 | SQLite append-only WAL as event store | **High** — minimizes new infra, matches existing stack |
| Q2 | 6 event types, immutable, causal references via signal_id | **High** — standard event-sourcing pattern |
| Q3 | Cursor-based polling consumers | **High** — matches existing scan-loop cadence |
| Q4 | Best-effort deterministic replay with flagged non-determinism | **Medium** — non-determinism is real; whether "best-effort" is enough for operator's debug needs is a judgment call |
| Q5 | Parallel-stream migration (3 phases of co-running) | **High** — safest path, aligns with operator's "keep data flowing" preference |
| Q6 | 3-process split (ingest / model_service / api_server) | **Medium** — clean conceptually, but adds 2 processes to manage. Alternative: just split ingest off (one extra process) and let api_server keep scan loop internally |

**Specific operator feedback wanted:**
1. Is "best-effort" determinism in Q4 acceptable, or do we need event-stream additions (e.g., capturing more external state) to make replay bit-exact?
2. For Q6, prefer 3-process split (clean) or 2-process split (ingest separate, scan+api stays)?
3. Any of the IN-scope items in the next-iteration list above that should be OUT-of-scope instead (or vice versa)?

---

## See also

- `2026-05-24-live-ops-feedback-phase2-phase3.md` — surfaced the architectural need
- `2026-05-23-strategy-discovery-phase2-design.md` — Phase 2 features+labels (per-token parquet aligns with Q2 candle_close events)
- `2026-05-23-pnl-anchored-trail-design.md` — B2 trail; the math becomes a consumer that reads `trade_decided` + `price_tick` events
- `2026-05-23-ws-exit-checker-design.md` — exit_watcher; becomes the producer of `exit_triggered` events
