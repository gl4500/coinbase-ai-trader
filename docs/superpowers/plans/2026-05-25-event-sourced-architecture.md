# Event-Sourced Operational Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the monolithic 8001 backend into three cooperating processes (ingest_worker / model_service / api_server) that communicate through an append-only SQLite event log, so model swaps don't restart WS state, multiple model experiments can run in parallel, and backtest replay uses the same code paths as live.

**Architecture:** New `backend/services/` modules: an `event_writer.py` + typed `event_types.py` pair writes immutable rows into a new `events` SQLite table; an `event_consumer.py` base class polls events by monotonic id with a per-consumer cursor; three CLI entry points (`ingest_worker.py`, `model_service.py`, `api_server.py`) each focus on one concern. `api_server.py` becomes the new owner of port 8001 once the monolith is retired. Migration is parallel-stream — the monolith keeps running while every new component is validated alongside.

**Tech Stack:** Python 3.11, `aiosqlite` (existing async DB layer), FastAPI + Uvicorn (existing API surface), `websockets` (existing Coinbase WS client), `pytest` + `pytest-asyncio` (existing test conventions). No new infrastructure.

**Spec:** `docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md`
**Decisions:** `docs/superpowers/specs/2026-05-25-event-sourced-architecture-decisions.md`
**Predecessor evidence:** `docs/superpowers/specs/2026-05-24-live-ops-feedback-phase2-phase3.md`

---

## Operational constraint

**The 8001 backend is running live paper-trading for the duration of this plan.** Per `feedback_no_pytest_during_trading.md`:

- The pre-commit hook runs the full ~1180-test pytest suite (5-9 min). It is blocked while 8001 is live.
- Tasks 1–25 write + stage files only. Within each task, "Run test to verify it fails/passes" steps describe expected outcomes for traceability but the executing agent MUST NOT actually invoke pytest while 8001 is live.
- Task 27 is the single operator-paused commit gate — full pytest sweep + atomic commit + push, executed in one shot during a sanctioned 8001 pause window.
- All code changes additive: nothing in this plan modifies behavior of the running monolith until Task 23, and even Task 23 is a no-op when `MONOLITH_SCAN_DISABLED` is unset (default).

Per `feedback_backend_port_isolation.md`: the new `api_server.py` does NOT bind 8001 in any test, only in operator-controlled rollout (Phase 4 of the spec). Tests use ephemeral ports.

Per `feedback_scope_restriction.md`: every change is inside `C:\Users\gl450\polymarket_app\` (this worktree's repo root).

---

## File Structure

| File | Purpose | Touch |
|---|---|---|
| `backend/services/events_schema.py` | DDL for `events`, `consumer_cursors`, `materialized_*` | NEW |
| `backend/services/event_types.py` | Typed dataclasses for 7 event payloads | NEW |
| `backend/services/event_writer.py` | Async typed insert helpers (one per event_type) | NEW |
| `backend/services/event_consumer.py` | `EventConsumer` base class — cursor poll/commit | NEW |
| `backend/services/feature_snapshot.py` | Rebuild feature inputs from event history | NEW |
| `backend/services/view_materializer.py` | Maintains `materialized_*` tables from events | NEW |
| `backend/services/ingest_worker.py` | Process: WS + REST + marketcap → events | NEW |
| `backend/services/model_service.py` | Process: poll events → inference → decisions | NEW |
| `backend/services/api_server.py` | Process: FastAPI surface reading materialized views | NEW |
| `backend/tools/event_inspector.py` | CLI query tool for events | NEW |
| `backend/tools/replay_consumer.py` | Sandboxed event-stream replay | NEW |
| `backend/tests/test_events_schema.py` | Schema init + indices | NEW |
| `backend/tests/test_event_types.py` | Dataclass validation | NEW |
| `backend/tests/test_event_writer.py` | Per-type insert correctness | NEW |
| `backend/tests/test_event_consumer.py` | Cursor advance, restart recovery, batch reads | NEW |
| `backend/tests/test_feature_snapshot.py` | Synthetic-event → feature parity | NEW |
| `backend/tests/test_view_materializer.py` | Materialized view freshness | NEW |
| `backend/tests/test_ingest_worker.py` | WS/REST/marketcap → event emission | NEW |
| `backend/tests/test_model_service.py` | Synthetic events → signal_scored + trade_decided | NEW |
| `backend/tests/test_api_server.py` | Endpoint correctness against materialized views | NEW |
| `backend/tests/test_replay_determinism.py` | Record → replay → byte-equal output | NEW |
| `backend/tests/test_phase2_crosscheck.py` | Monolith vs event-driven decision parity | NEW |
| `backend/main.py` | Add `MONOLITH_SCAN_DISABLED` env gate (Phase 3 toggle) | EDIT |
| `backend/database.py` | Call `events_schema.init_events_schema()` from `init_db()` | EDIT |
| `backend/config.py` | Add event-store config keys (no behavior change) | EDIT |
| `CHANGELOG.md` | Pattern C session entry | EDIT |
| `CLAUDE.md` | New invariant #19 (events append-only, never UPDATE/DELETE) | EDIT |
| `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md` | Event-store + 3-process layout | EDIT |

**Module boundary discipline** (per `feedback_loose_coupling`):

- `events_schema.py` knows about SQLite DDL; nothing about event semantics.
- `event_types.py` knows about Python dataclasses + JSON serialization; nothing about DB.
- `event_writer.py` imports `event_types` + uses `aiosqlite`; nothing about consumers.
- `event_consumer.py` knows about polling + cursor commits; nothing about event semantics or consumer logic.
- `feature_snapshot.py` knows about event history → feature vectors; nothing about model inference or trade decisions.
- `view_materializer.py` imports `event_consumer`; knows about materialized table schemas. No inference.
- `ingest_worker.py` imports `event_writer` + `ws_subscriber` + existing REST clients; never reads events.
- `model_service.py` imports `event_consumer` + `event_writer` + `feature_snapshot` + existing inference (`agents/xgb_signal.py`). Never opens WS, never makes REST calls.
- `api_server.py` imports `event_consumer` + `view_materializer`. Reads only.

Tests for each module never import siblings unless the test is explicitly integration-flavored.

---

## Task 1: Events schema — `events` + `consumer_cursors` tables

**Files:**
- Create: `backend/services/events_schema.py`
- Create: `backend/tests/test_events_schema.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_events_schema.py`:

```python
"""Tests for events_schema — DDL for the event store."""
import aiosqlite
import pytest

from services import events_schema


@pytest.mark.asyncio
async def test_init_creates_events_table(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        row = await cur.fetchone()
    assert row is not None, "events table not created"


@pytest.mark.asyncio
async def test_init_creates_consumer_cursors_table(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='consumer_cursors'"
        )
        row = await cur.fetchone()
    assert row is not None, "consumer_cursors table not created"


@pytest.mark.asyncio
async def test_init_is_idempotent(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_events_schema(db)  # second call must not raise


@pytest.mark.asyncio
async def test_events_indices_exist(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='events'"
        )
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert "idx_events_ts" in names
    assert "idx_events_pid_ts" in names
    assert "idx_events_type_ts" in names
    assert "idx_events_producer" in names
```

- [ ] **Step 2: Run test to verify it fails** *(skip while 8001 is live; expected: `ModuleNotFoundError: services.events_schema`)*

- [ ] **Step 3: Implement `services/events_schema.py`**

```python
"""DDL for the event store.

Event-sourced architecture per docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md.

Append-only events table + per-consumer cursor table + materialized view tables
(materialized_* tables are declared in materialized_schema() so consumers that
don't need them aren't forced to create them).

All DDL is `CREATE TABLE IF NOT EXISTS` + `CREATE INDEX IF NOT EXISTS` so the
init function is safe to call repeatedly. Tests rely on this idempotency.
"""
from __future__ import annotations

import aiosqlite

_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms           INTEGER NOT NULL,
    event_type      TEXT    NOT NULL,
    pid             TEXT,
    payload_json    TEXT    NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 1,
    producer        TEXT    NOT NULL,
    write_ts_ms     INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);
"""

_EVENTS_INDICES = (
    "CREATE INDEX IF NOT EXISTS idx_events_ts       ON events(ts_ms);",
    "CREATE INDEX IF NOT EXISTS idx_events_pid_ts   ON events(pid, ts_ms) WHERE pid IS NOT NULL;",
    "CREATE INDEX IF NOT EXISTS idx_events_type_ts  ON events(event_type, ts_ms);",
    "CREATE INDEX IF NOT EXISTS idx_events_producer ON events(producer, id);",
)

_CURSORS_DDL = """
CREATE TABLE IF NOT EXISTS consumer_cursors (
    consumer_name      TEXT    PRIMARY KEY,
    last_processed_id  INTEGER NOT NULL,
    updated_at         INTEGER NOT NULL DEFAULT (CAST(strftime('%s','now') AS INTEGER) * 1000)
);
"""


async def init_events_schema(db: aiosqlite.Connection) -> None:
    """Create events + consumer_cursors tables and indices. Idempotent."""
    await db.execute(_EVENTS_DDL)
    for stmt in _EVENTS_INDICES:
        await db.execute(stmt)
    await db.execute(_CURSORS_DDL)
    await db.commit()
```

- [ ] **Step 4: Run test to verify it passes** *(deferred)*

- [ ] **Step 5: Stage (no commit yet)**

```bash
git add backend/services/events_schema.py backend/tests/test_events_schema.py
```

---

## Task 2: Events schema — `materialized_*` tables

**Files:**
- Modify: `backend/services/events_schema.py` (add `init_materialized_schema`)
- Modify: `backend/tests/test_events_schema.py` (3 new tests)

- [ ] **Step 1: Append failing tests to `backend/tests/test_events_schema.py`**

```python
@pytest.mark.asyncio
async def test_init_materialized_creates_latest_price(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_materialized_schema(db, model_names=["v3", "v4_5"])
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='materialized_latest_price'"
        )
        row = await cur.fetchone()
    assert row is not None


@pytest.mark.asyncio
async def test_init_materialized_creates_per_model_positions(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        await events_schema.init_materialized_schema(db, model_names=["v3", "v4_5"])
        cur = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'materialized_positions_%'"
        )
        rows = await cur.fetchall()
    names = {r[0] for r in rows}
    assert names == {"materialized_positions_v3", "materialized_positions_v4_5"}


@pytest.mark.asyncio
async def test_init_materialized_rejects_bad_model_name(tmp_path):
    db_path = str(tmp_path / "test.db")
    async with aiosqlite.connect(db_path) as db:
        await events_schema.init_events_schema(db)
        with pytest.raises(ValueError, match="model_name"):
            await events_schema.init_materialized_schema(db, model_names=["v3; DROP TABLE events--"])
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add `init_materialized_schema` to `backend/services/events_schema.py`**

Append:

```python
import re
from typing import Iterable

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")

_MATERIALIZED_LATEST_PRICE_DDL = """
CREATE TABLE IF NOT EXISTS materialized_latest_price (
    pid                 TEXT PRIMARY KEY,
    price               REAL,
    bid                 REAL,
    ask                 REAL,
    pct_change_24h      REAL,
    last_event_id       INTEGER,
    last_updated_ts_ms  INTEGER
);
"""

_MATERIALIZED_POSITIONS_DDL_TMPL = """
CREATE TABLE IF NOT EXISTS materialized_positions_{model} (
    pid               TEXT PRIMARY KEY,
    size              REAL,
    avg_price         REAL,
    position_dollars  REAL,
    entry_time_ms     INTEGER,
    peak_price        REAL,
    peak_pnl_pct      REAL,
    last_event_id     INTEGER
);
"""


async def init_materialized_schema(
    db: aiosqlite.Connection,
    model_names: Iterable[str],
) -> None:
    """Create materialized_latest_price + per-model materialized_positions_* tables.

    model_names is an iterable of safe identifiers (e.g. 'v3', 'v4_5'). Anything
    not matching ^[a-zA-Z0-9_]+$ is rejected to keep the DDL safe — the table
    name is interpolated into a CREATE TABLE so we cannot use parameter binding.
    """
    await db.execute(_MATERIALIZED_LATEST_PRICE_DDL)
    for model in model_names:
        if not _MODEL_NAME_RE.match(model):
            raise ValueError(
                f"model_name {model!r} contains non-identifier characters"
            )
        await db.execute(_MATERIALIZED_POSITIONS_DDL_TMPL.format(model=model))
    await db.commit()
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/events_schema.py backend/tests/test_events_schema.py
```

---

## Task 3: Event payload dataclasses (`event_types.py`)

**Files:**
- Create: `backend/services/event_types.py`
- Create: `backend/tests/test_event_types.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_event_types.py`:

```python
"""Tests for typed event payload dataclasses."""
import json

import pytest

from services import event_types as et


def test_price_tick_roundtrip():
    tick = et.PriceTickPayload(pid="BTC-USD", price=95123.45, bid=95120.0, ask=95125.0,
                               volume_24h=12345.67, source="ws")
    blob = json.dumps(tick.to_dict())
    restored = et.PriceTickPayload.from_dict(json.loads(blob))
    assert restored == tick


def test_candle_close_requires_tier():
    with pytest.raises(ValueError, match="tier"):
        et.CandleClosePayload(
            pid="BTC-USD", tier="bogus", open=1, high=2, low=0.5,
            close=1.5, volume=10, bar_ts_ms=1716595200000,
        )


def test_signal_scored_carries_input_event_ids():
    s = et.SignalScoredPayload(
        pid="BTC-USD", model="v4_5", model_version="h168_2026-05-23",
        feature_hash="abcdef", scores={"p_up": 0.6, "p_down": 0.2, "p_neutral": 0.2},
        side="BUY", strength=0.6, regime="TRENDING", deployment_profile_id=None,
        input_event_ids={"last_price_tick_id": 100, "last_candle_close_id": 99},
    )
    blob = s.to_dict()
    assert blob["input_event_ids"]["last_price_tick_id"] == 100


def test_trade_decided_requires_signal_event_id():
    with pytest.raises(ValueError, match="signal_event_id"):
        et.TradeDecidedPayload(
            pid="BTC-USD", model="v4_5", side="BUY", size=0.001, size_usd=95.0,
            intended_entry_price=95000.0, actual_entry_price=95000.0,
            fee_paid=0.57, trigger="SCAN", signal_event_id=None,
            deployment_profile_id=None, trade_uid="v4_5_BTC-USD_x",
        )


def test_exit_triggered_requires_trigger_price_event_id():
    with pytest.raises(ValueError, match="trigger_price_event_id"):
        et.ExitTriggeredPayload(
            pid="BTC-USD", trade_uid="x", trigger_type="WS_TRAIL_STOP",
            peak_pnl_pct=1.0, current_pnl_pct=0.5, exit_threshold=0.7,
            price_at_trigger=100.0, trigger_price_event_id=None,
        )


def test_event_type_enum_includes_all_seven():
    assert set(et.EVENT_TYPES) == {
        "price_tick", "candle_close", "marketcap_snapshot",
        "signal_scored", "trade_decided", "trade_closed", "exit_triggered",
    }
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred — `ModuleNotFoundError`)*

- [ ] **Step 3: Implement `backend/services/event_types.py`**

```python
"""Typed payload dataclasses for the 7 event types in the spec.

Each dataclass:
  - Validates its own invariants in __post_init__
  - Round-trips via to_dict() / from_dict() (JSON-safe primitive dicts)
  - Mirrors the payload schema documented in 2026-05-25-event-sourced-architecture-design.md

We use frozen=True so payloads are immutable once constructed — events are
append-only and the dataclass instance carries that intent through the code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Optional


EVENT_TYPES = (
    "price_tick",
    "candle_close",
    "marketcap_snapshot",
    "signal_scored",
    "trade_decided",
    "trade_closed",
    "exit_triggered",
)

_VALID_TIERS = {"1m", "5m", "15m", "1h", "4h", "1d"}
_VALID_SIDES = {"BUY", "SELL", "HOLD"}
_VALID_TRIGGERS = {"WS_TRAIL_STOP", "WS_STOP_LOSS", "STOP_LOSS",
                   "WS_MODEL_DOWN", "MAX_HOLD", "TRAIL_STOP", "RECONCILE"}


@dataclass(frozen=True)
class PriceTickPayload:
    pid: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    volume_24h: Optional[float]
    source: Literal["ws", "rest_fallback"]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PriceTickPayload":
        return cls(**d)


@dataclass(frozen=True)
class CandleClosePayload:
    pid: str
    tier: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_ts_ms: int

    def __post_init__(self):
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"tier {self.tier!r} must be one of {_VALID_TIERS}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CandleClosePayload":
        return cls(**d)


@dataclass(frozen=True)
class MarketcapSnapshotPayload:
    pid: str
    market_cap: Optional[float]
    fdv: Optional[float]
    circ_supply: Optional[float]
    total_supply: Optional[float]
    vol_24h: Optional[float]
    source: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MarketcapSnapshotPayload":
        return cls(**d)


@dataclass(frozen=True)
class SignalScoredPayload:
    pid: str
    model: str
    model_version: str
    feature_hash: str
    scores: Dict[str, float]
    side: str
    strength: float
    regime: Optional[str]
    deployment_profile_id: Optional[str]
    input_event_ids: Dict[str, int]

    def __post_init__(self):
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side {self.side!r} must be one of {_VALID_SIDES}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SignalScoredPayload":
        return cls(**d)


@dataclass(frozen=True)
class TradeDecidedPayload:
    pid: str
    model: str
    side: str
    size: float
    size_usd: float
    intended_entry_price: float
    actual_entry_price: float
    fee_paid: float
    trigger: str
    signal_event_id: Optional[int]
    deployment_profile_id: Optional[str]
    trade_uid: str

    def __post_init__(self):
        if self.signal_event_id is None:
            raise ValueError("signal_event_id is required (no orphan trades)")
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side {self.side!r} must be one of {_VALID_SIDES}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeDecidedPayload":
        return cls(**d)


@dataclass(frozen=True)
class TradeClosedPayload:
    pid: str
    trade_uid: str
    exit_price: float
    exit_size: float
    pnl: float
    pct_pnl: float
    hold_secs: int
    trigger_close: str
    decision_event_id: int
    exit_signal_event_id: Optional[int]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeClosedPayload":
        return cls(**d)


@dataclass(frozen=True)
class ExitTriggeredPayload:
    pid: str
    trade_uid: str
    trigger_type: str
    peak_pnl_pct: float
    current_pnl_pct: float
    exit_threshold: float
    price_at_trigger: float
    trigger_price_event_id: Optional[int]

    def __post_init__(self):
        if self.trigger_price_event_id is None:
            raise ValueError("trigger_price_event_id is required for replay determinism")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExitTriggeredPayload":
        return cls(**d)
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/event_types.py backend/tests/test_event_types.py
```

---

## Task 4: Typed event writers (`event_writer.py`)

**Files:**
- Create: `backend/services/event_writer.py`
- Create: `backend/tests/test_event_writer.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_event_writer.py`:

```python
"""Tests for typed event INSERT helpers."""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema


@pytest.fixture
async def db(tmp_path):
    path = str(tmp_path / "test.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        yield conn


@pytest.mark.asyncio
async def test_write_price_tick_inserts_row(db):
    payload = et.PriceTickPayload(
        pid="BTC-USD", price=100.0, bid=99.9, ask=100.1,
        volume_24h=1.0, source="ws",
    )
    eid = await event_writer.write_price_tick(
        db, producer="ingest", ts_ms=1700000000000, payload=payload,
    )
    cur = await db.execute(
        "SELECT event_type, pid, payload_json, producer FROM events WHERE id = ?",
        (eid,),
    )
    row = await cur.fetchone()
    assert row[0] == "price_tick"
    assert row[1] == "BTC-USD"
    assert json.loads(row[2])["price"] == 100.0
    assert row[3] == "ingest"


@pytest.mark.asyncio
async def test_write_signal_scored_routes_to_correct_event_type(db):
    payload = et.SignalScoredPayload(
        pid="ETH-USD", model="v4_5", model_version="h168_2026-05-23",
        feature_hash="x", scores={"p_up": 0.5, "p_down": 0.3, "p_neutral": 0.2},
        side="BUY", strength=0.5, regime=None, deployment_profile_id=None,
        input_event_ids={"last_price_tick_id": 1},
    )
    eid = await event_writer.write_signal_scored(
        db, producer="model_v4_5", ts_ms=1700000000000, payload=payload,
    )
    cur = await db.execute("SELECT event_type FROM events WHERE id = ?", (eid,))
    row = await cur.fetchone()
    assert row[0] == "signal_scored"


@pytest.mark.asyncio
async def test_event_id_is_monotonic(db):
    p1 = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
    p2 = et.PriceTickPayload(pid="A", price=2, bid=None, ask=None, volume_24h=None, source="ws")
    id1 = await event_writer.write_price_tick(db, producer="ingest", ts_ms=1, payload=p1)
    id2 = await event_writer.write_price_tick(db, producer="ingest", ts_ms=2, payload=p2)
    assert id2 > id1


@pytest.mark.asyncio
async def test_pid_column_populated_when_payload_has_pid(db):
    payload = et.PriceTickPayload(pid="SOL-USD", price=1, bid=None, ask=None, volume_24h=None, source="ws")
    eid = await event_writer.write_price_tick(db, producer="ingest", ts_ms=1, payload=payload)
    cur = await db.execute("SELECT pid FROM events WHERE id = ?", (eid,))
    row = await cur.fetchone()
    assert row[0] == "SOL-USD"
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `backend/services/event_writer.py`**

```python
"""Async typed event INSERT helpers.

One write_<event_type>() coroutine per event type. Each one:
  - Takes the typed payload dataclass
  - Serializes it to JSON
  - Inserts into the events table
  - Returns the new event id

The DB connection is passed in — the writer is agnostic about how the caller
opened it. This lets the same writer be used by the ingest worker (its own
connection) and the model service (its own connection); concurrent writers
are safe because SQLite WAL allows multiple writers serialized by the
journal manager, and writes are tiny (single-row inserts).
"""
from __future__ import annotations

import json
from typing import Optional

import aiosqlite

from services import event_types as et


async def _insert(
    db: aiosqlite.Connection,
    *,
    event_type: str,
    producer: str,
    ts_ms: int,
    pid: Optional[str],
    payload_dict: dict,
) -> int:
    cur = await db.execute(
        "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (ts_ms, event_type, pid, json.dumps(payload_dict, separators=(",", ":")), producer),
    )
    await db.commit()
    return cur.lastrowid


async def write_price_tick(db, *, producer, ts_ms, payload: et.PriceTickPayload) -> int:
    return await _insert(db, event_type="price_tick", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_candle_close(db, *, producer, ts_ms, payload: et.CandleClosePayload) -> int:
    return await _insert(db, event_type="candle_close", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_marketcap_snapshot(db, *, producer, ts_ms, payload: et.MarketcapSnapshotPayload) -> int:
    return await _insert(db, event_type="marketcap_snapshot", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_signal_scored(db, *, producer, ts_ms, payload: et.SignalScoredPayload) -> int:
    return await _insert(db, event_type="signal_scored", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_trade_decided(db, *, producer, ts_ms, payload: et.TradeDecidedPayload) -> int:
    return await _insert(db, event_type="trade_decided", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_trade_closed(db, *, producer, ts_ms, payload: et.TradeClosedPayload) -> int:
    return await _insert(db, event_type="trade_closed", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())


async def write_exit_triggered(db, *, producer, ts_ms, payload: et.ExitTriggeredPayload) -> int:
    return await _insert(db, event_type="exit_triggered", producer=producer,
                          ts_ms=ts_ms, pid=payload.pid, payload_dict=payload.to_dict())
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/event_writer.py backend/tests/test_event_writer.py
```

---

## Task 5: Event consumer base class (`event_consumer.py`)

**Files:**
- Create: `backend/services/event_consumer.py`
- Create: `backend/tests/test_event_consumer.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_event_consumer.py`:

```python
"""Tests for EventConsumer — cursor-based polling base class."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.event_consumer import EventConsumer
from services.events_schema import init_events_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
    return path


@pytest.mark.asyncio
async def test_new_consumer_starts_at_max_id(db_path):
    """A first-time consumer skips history by default — starts at current max(id)."""
    async with aiosqlite.connect(db_path) as conn:
        p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
        await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)

    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        events = await consumer.poll()
        assert events == []          # past events skipped
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_consumer_reads_events_after_start(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        events = await consumer.poll()
        assert len(events) == 1
        assert events[0].event_type == "price_tick"
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_commit_advances_cursor(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
            eid = await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        events = await consumer.poll()
        await consumer.commit(events[-1].id)
        events_after = await consumer.poll()
        assert events_after == []
    finally:
        await consumer.stop()


@pytest.mark.asyncio
async def test_cursor_persists_across_restart(db_path):
    consumer = EventConsumer(db_path, name="model_v3")
    await consumer.start()
    async with aiosqlite.connect(db_path) as conn:
        p = et.PriceTickPayload(pid="A", price=1, bid=None, ask=None, volume_24h=None, source="ws")
        eid = await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
    events = await consumer.poll()
    await consumer.commit(events[-1].id)
    await consumer.stop()

    revived = EventConsumer(db_path, name="model_v3")
    await revived.start()
    try:
        again = await revived.poll()
        assert again == []
    finally:
        await revived.stop()


@pytest.mark.asyncio
async def test_batch_size_limits_returned_events(db_path):
    consumer = EventConsumer(db_path, name="model_v3", batch_size=3)
    await consumer.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            for i in range(10):
                p = et.PriceTickPayload(pid="A", price=i, bid=None, ask=None, volume_24h=None, source="ws")
                await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p)
        events = await consumer.poll()
        assert len(events) == 3
    finally:
        await consumer.stop()
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `backend/services/event_consumer.py`**

```python
"""EventConsumer — cursor-based polling primitive for the event store.

Each consumer (a model service, the view materializer, the WS bridge) wraps an
EventConsumer instance.

Discipline:
  - poll() returns ordered events with id > cursor, up to batch_size
  - commit(event_id) advances the cursor to event_id
  - Consumers MUST be idempotent — at-least-once delivery: a consumer that
    crashes after processing but before commit() will re-process the same event
    on restart.
  - First-time consumers start at MAX(id) — they do NOT replay history on first
    boot. Backtest replay tooling uses a separate cursor-reset path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import aiosqlite


@dataclass(frozen=True)
class Event:
    id: int
    ts_ms: int
    event_type: str
    pid: Optional[str]
    payload_json: str
    schema_version: int
    producer: str


class EventConsumer:
    """Cursor-based event reader. Open/close pattern via start()/stop()."""

    def __init__(self, db_path: str, *, name: str, batch_size: int = 1000):
        self._db_path = db_path
        self._name = name
        self._batch_size = batch_size
        self._db: Optional[aiosqlite.Connection] = None
        self._cursor_id: int = 0

    @property
    def name(self) -> str:
        return self._name

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")
        self._cursor_id = await self._load_or_init_cursor()

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def _load_or_init_cursor(self) -> int:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT last_processed_id FROM consumer_cursors WHERE consumer_name = ?",
            (self._name,),
        )
        row = await cur.fetchone()
        if row is not None:
            return row[0]

        cur = await self._db.execute("SELECT COALESCE(MAX(id), 0) FROM events")
        max_row = await cur.fetchone()
        max_id = max_row[0] if max_row else 0
        await self._db.execute(
            "INSERT INTO consumer_cursors (consumer_name, last_processed_id) VALUES (?, ?)",
            (self._name, max_id),
        )
        await self._db.commit()
        return max_id

    async def poll(self) -> List[Event]:
        assert self._db is not None
        cur = await self._db.execute(
            "SELECT id, ts_ms, event_type, pid, payload_json, schema_version, producer "
            "FROM events WHERE id > ? ORDER BY id ASC LIMIT ?",
            (self._cursor_id, self._batch_size),
        )
        rows = await cur.fetchall()
        return [Event(*r) for r in rows]

    async def commit(self, event_id: int) -> None:
        assert self._db is not None
        await self._db.execute(
            "UPDATE consumer_cursors SET last_processed_id = ?, "
            "updated_at = CAST(strftime('%s','now') AS INTEGER) * 1000 "
            "WHERE consumer_name = ?",
            (event_id, self._name),
        )
        await self._db.commit()
        self._cursor_id = event_id
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/event_consumer.py backend/tests/test_event_consumer.py
```

---

## Task 6: Feature snapshot reconstruction (`feature_snapshot.py`)

**Files:**
- Create: `backend/services/feature_snapshot.py`
- Create: `backend/tests/test_feature_snapshot.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_feature_snapshot.py`:

```python
"""Tests for feature_snapshot — rebuild model features from event history.

Numerical-parity intent: given a set of candle_close + price_tick events for
PID X, the feature vector produced by feature_snapshot.build_for() must
equal (within float tolerance) what the current cnn_agent feature path would
produce given the same input candles/price.
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer, feature_snapshot
from services.events_schema import init_events_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        for i in range(60):
            close = 100.0 + i * 0.5
            payload = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=close - 0.5, high=close + 0.5, low=close - 1.0,
                close=close, volume=10.0 + i,
                bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(
                conn, producer="ingest", ts_ms=payload.bar_ts_ms, payload=payload,
            )
    return path


@pytest.mark.asyncio
async def test_build_for_returns_candles_in_time_order(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=60)
    assert len(snap.candles) == 60
    closes = [c["close"] for c in snap.candles]
    assert closes == sorted(closes)


@pytest.mark.asyncio
async def test_build_for_includes_last_event_ids(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=60)
    assert snap.last_candle_close_id is not None


@pytest.mark.asyncio
async def test_build_for_empty_pid_returns_empty_snapshot(db_path):
    snap = await feature_snapshot.build_for("NOPE-USD", db_path, tier="1h", lookback=60)
    assert snap.candles == []
    assert snap.last_candle_close_id is None


@pytest.mark.asyncio
async def test_build_for_respects_lookback_window(db_path):
    snap = await feature_snapshot.build_for("BTC-USD", db_path, tier="1h", lookback=10)
    assert len(snap.candles) == 10
    closes = [c["close"] for c in snap.candles]
    assert closes[0] > 100.0   # took the most recent 10
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `backend/services/feature_snapshot.py`**

```python
"""Reconstruct feature inputs (OHLCV + last price) from the event store.

`build_for(pid, db_path, tier, lookback)` opens a read-only connection,
fetches the last <lookback> candle_close events for the pid+tier, and the
most recent price_tick. Returns a FeatureSnapshot dataclass.

This module deliberately does NOT compute features — it returns raw inputs.
Existing feature compute lives in agents/xgb_signal.py and is reused as-is by
model_service. Keeping the responsibilities separate means model service can
swap feature extractors without touching this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiosqlite


@dataclass(frozen=True)
class FeatureSnapshot:
    pid: str
    tier: str
    candles: List[Dict]                          # oldest → newest
    last_price: Optional[float] = None
    last_price_tick_id: Optional[int] = None
    last_candle_close_id: Optional[int] = None


async def build_for(
    pid: str,
    db_path: str,
    *,
    tier: str = "1h",
    lookback: int = 360,
) -> FeatureSnapshot:
    """Return a FeatureSnapshot with the last <lookback> candles (oldest first)
    + most recent price_tick for <pid>+<tier>."""
    candles: List[Dict] = []
    last_candle_close_id: Optional[int] = None
    last_price: Optional[float] = None
    last_price_tick_id: Optional[int] = None

    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA busy_timeout=30000")

        cur = await db.execute(
            "SELECT id, payload_json FROM events "
            "WHERE event_type = 'candle_close' AND pid = ? "
            "ORDER BY id DESC LIMIT ?",
            (pid, lookback),
        )
        rows = await cur.fetchall()
        if rows:
            last_candle_close_id = rows[0][0]
            parsed = [json.loads(r[1]) for r in rows]
            parsed = [p for p in parsed if p.get("tier") == tier]
            parsed.reverse()
            candles = parsed

        cur = await db.execute(
            "SELECT id, payload_json FROM events "
            "WHERE event_type = 'price_tick' AND pid = ? "
            "ORDER BY id DESC LIMIT 1",
            (pid,),
        )
        row = await cur.fetchone()
        if row is not None:
            last_price_tick_id = row[0]
            last_price = json.loads(row[1]).get("price")

    return FeatureSnapshot(
        pid=pid, tier=tier, candles=candles,
        last_price=last_price,
        last_price_tick_id=last_price_tick_id,
        last_candle_close_id=last_candle_close_id,
    )
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/feature_snapshot.py backend/tests/test_feature_snapshot.py
```

---

## Task 7: View materializer (`view_materializer.py`)

**Files:**
- Create: `backend/services/view_materializer.py`
- Create: `backend/tests/test_view_materializer.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_view_materializer.py`:

```python
"""Tests for ViewMaterializer — keeps materialized_* tables up to date from events."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.view_materializer import ViewMaterializer


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


@pytest.mark.asyncio
async def test_price_tick_updates_latest_price(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            payload = et.PriceTickPayload(
                pid="BTC-USD", price=95000.0, bid=94999.0, ask=95001.0,
                volume_24h=100.0, source="ws",
            )
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1_700_000_000_000, payload=payload)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT price, bid, ask FROM materialized_latest_price WHERE pid = ?",
                ("BTC-USD",),
            )
            row = await cur.fetchone()
        assert row == (95000.0, 94999.0, 95001.0)
    finally:
        await mat.stop()


@pytest.mark.asyncio
async def test_trade_decided_updates_positions_for_correct_model(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={"p_up": 0.6}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(
                conn, producer="model_v3", ts_ms=1, payload=sig,
            )
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000.0, actual_entry_price=95000.0,
                fee_paid=0.57, trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_1",
            )
            await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=2, payload=td)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT size, avg_price FROM materialized_positions_v3 WHERE pid = ?",
                ("BTC-USD",),
            )
            row = await cur.fetchone()
        assert row == (0.001, 95000.0)


@pytest.mark.asyncio
async def test_trade_closed_clears_position(db_path):
    mat = ViewMaterializer(db_path, name="api_view")
    await mat.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(
                conn, producer="model_v3", ts_ms=1, payload=sig,
            )
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000.0, actual_entry_price=95000.0,
                fee_paid=0.57, trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_1",
            )
            dec_id = await event_writer.write_trade_decided(
                conn, producer="model_v3", ts_ms=2, payload=td,
            )
            tc = et.TradeClosedPayload(
                pid="BTC-USD", trade_uid="v3_BTC-USD_1", exit_price=96000.0,
                exit_size=0.001, pnl=1.0, pct_pnl=1.05, hold_secs=3600,
                trigger_close="WS_TRAIL_STOP", decision_event_id=dec_id,
                exit_signal_event_id=None,
            )
            await event_writer.write_trade_closed(conn, producer="model_v3", ts_ms=3, payload=tc)
        await mat.tick()
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM materialized_positions_v3 WHERE pid = ?",
                ("BTC-USD",),
            )
            (n,) = await cur.fetchone()
        assert n == 0
    finally:
        await mat.stop()
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `backend/services/view_materializer.py`**

```python
"""View materializer — keeps materialized_* tables in sync with events.

Polls events via EventConsumer (its own cursor name). Routes by event_type:
  price_tick           → UPSERT materialized_latest_price
  trade_decided        → UPSERT materialized_positions_<model>
  trade_closed         → DELETE FROM materialized_positions_<model>
  exit_triggered       → no view update (audit-only)
  signal_scored        → no view update (audit-only)
  candle_close         → no view update (feature_snapshot reads events directly)
  marketcap_snapshot   → no view update (queried from events on demand)

Reuses EventConsumer's cursor semantics so view freshness survives restart.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import aiosqlite

from services.event_consumer import Event, EventConsumer

logger = logging.getLogger(__name__)

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")


class ViewMaterializer:
    """Subscribes to the event stream and updates materialized_* tables."""

    def __init__(self, db_path: str, *, name: str = "view_materializer", batch_size: int = 1000):
        self._db_path = db_path
        self._consumer = EventConsumer(db_path, name=name, batch_size=batch_size)
        self._write_db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        await self._consumer.start()
        self._write_db = await aiosqlite.connect(self._db_path)
        await self._write_db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._write_db is not None:
            await self._write_db.close()
            self._write_db = None
        await self._consumer.stop()

    async def tick(self) -> int:
        """Process one batch of events. Returns count processed."""
        events = await self._consumer.poll()
        if not events:
            return 0
        for evt in events:
            try:
                await self._apply(evt)
            except Exception:
                logger.exception("view_materializer: failed to apply event id=%s type=%s",
                                 evt.id, evt.event_type)
        await self._consumer.commit(events[-1].id)
        return len(events)

    async def _apply(self, evt: Event) -> None:
        assert self._write_db is not None
        if evt.event_type == "price_tick":
            await self._apply_price_tick(evt)
        elif evt.event_type == "trade_decided":
            await self._apply_trade_decided(evt)
        elif evt.event_type == "trade_closed":
            await self._apply_trade_closed(evt)

    async def _apply_price_tick(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        await self._write_db.execute(
            "INSERT INTO materialized_latest_price "
            "(pid, price, bid, ask, pct_change_24h, last_event_id, last_updated_ts_ms) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?) "
            "ON CONFLICT(pid) DO UPDATE SET "
            "  price=excluded.price, bid=excluded.bid, ask=excluded.ask, "
            "  last_event_id=excluded.last_event_id, "
            "  last_updated_ts_ms=excluded.last_updated_ts_ms",
            (evt.pid, payload.get("price"), payload.get("bid"), payload.get("ask"),
             evt.id, evt.ts_ms),
        )
        await self._write_db.commit()

    async def _apply_trade_decided(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        model = payload["model"]
        if not _MODEL_NAME_RE.match(model):
            logger.error("Skipping trade_decided with unsafe model name: %r", model)
            return
        table = f"materialized_positions_{model}"
        await self._write_db.execute(
            f"INSERT INTO {table} "
            f"(pid, size, avg_price, position_dollars, entry_time_ms, peak_price, peak_pnl_pct, last_event_id) "
            f"VALUES (?, ?, ?, ?, ?, ?, 0.0, ?) "
            f"ON CONFLICT(pid) DO UPDATE SET "
            f"  size=excluded.size, avg_price=excluded.avg_price, "
            f"  position_dollars=excluded.position_dollars, "
            f"  entry_time_ms=excluded.entry_time_ms, peak_price=excluded.peak_price, "
            f"  last_event_id=excluded.last_event_id",
            (payload["pid"], payload["size"], payload["actual_entry_price"],
             payload["size_usd"], evt.ts_ms, payload["actual_entry_price"], evt.id),
        )
        await self._write_db.commit()

    async def _apply_trade_closed(self, evt: Event) -> None:
        assert self._write_db is not None
        payload = json.loads(evt.payload_json)
        cur = await self._write_db.execute(
            "SELECT producer FROM events WHERE id = ?", (payload["decision_event_id"],),
        )
        row = await cur.fetchone()
        if row is None:
            logger.error("trade_closed references unknown decision_event_id=%s", payload["decision_event_id"])
            return
        producer = row[0]
        if not producer.startswith("model_"):
            return
        model = producer[len("model_"):]
        if not _MODEL_NAME_RE.match(model):
            return
        table = f"materialized_positions_{model}"
        await self._write_db.execute(
            f"DELETE FROM {table} WHERE pid = ?", (payload["pid"],),
        )
        await self._write_db.commit()
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/view_materializer.py backend/tests/test_view_materializer.py
```

---

## Task 8: Hook `init_events_schema` into existing `init_db`

**Files:**
- Modify: `backend/database.py` (call into events_schema from init_db)
- Modify: `backend/tests/test_database.py` (1 new test)

- [ ] **Step 1: Append failing test to `backend/tests/test_database.py`**

```python
@pytest.mark.asyncio
async def test_init_db_creates_events_schema(tmp_db, monkeypatch):
    import importlib
    import database as db_mod
    importlib.reload(db_mod)
    await db_mod.init_db()
    async with aiosqlite.connect(tmp_db) as conn:
        cur = await conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name IN ('events', 'consumer_cursors')"
        )
        names = {r[0] for r in await cur.fetchall()}
    assert names == {"events", "consumer_cursors"}
```

- [ ] **Step 2: Run test to verify it fails** *(deferred)*

- [ ] **Step 3: Wire `events_schema` into `init_db()` in `backend/database.py`**

At the end of `init_db()` after the existing executescript + commit and before the function returns, add:

```python
        # Event-sourced architecture substrate (Pattern C, 2026-05-25 spec)
        from services.events_schema import init_events_schema
        await init_events_schema(db)
```

The call is idempotent (`CREATE TABLE IF NOT EXISTS`), safe to invoke on every `init_db()` call. The legacy `cnn_scans` / `trades` / `positions` tables are unaffected; the new tables coexist.

- [ ] **Step 4: Run test to verify it passes** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/database.py backend/tests/test_database.py
```

---

## Task 9: Ingest worker — WS price-tick emission

**Files:**
- Create: `backend/services/ingest_worker.py` (skeleton + WS class)
- Create: `backend/tests/test_ingest_worker.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_ingest_worker.py`:

```python
"""Tests for ingest_worker — WS / REST / marketcap event producers."""
import aiosqlite
import pytest

from services.events_schema import init_events_schema
from services.ingest_worker import _WSIngest


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
    return path


@pytest.mark.asyncio
async def test_ws_tick_handler_emits_price_tick_event(db_path):
    ws = _WSIngest(db_path=db_path, producer="ingest")
    await ws.start()
    try:
        await ws.handle_tick({
            "product_id": "BTC-USD",
            "price": "95000.0",
            "best_bid": "94999.0",
            "best_ask": "95001.0",
            "volume_24_h": "100.0",
        })
    finally:
        await ws.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "price_tick"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["price"] == 95000.0
    assert payload["source"] == "ws"


@pytest.mark.asyncio
async def test_ws_handler_skips_invalid_ticker(db_path):
    ws = _WSIngest(db_path=db_path, producer="ingest")
    await ws.start()
    try:
        await ws.handle_tick({"product_id": "BTC-USD"})    # no price
    finally:
        await ws.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 0
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement skeleton + `_WSIngest`**

Create `backend/services/ingest_worker.py`:

```python
"""ingest_worker — single source of truth for all market observations.

Process model: one CLI entry point that owns:
  * Coinbase WebSocket subscriber (price_tick events)
  * REST candle backfill + close-detection (candle_close events)
  * Marketcap polling via CoinPaprika (marketcap_snapshot events)

Never runs inference. Never makes trade decisions. Never reads events.

Each concern is a separate inner class so they can be tested + restarted
in isolation. Run together via `python -m services.ingest_worker`.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Optional

import aiosqlite

from services import event_types as et
from services import event_writer

logger = logging.getLogger(__name__)


class _WSIngest:
    """Subscribes to Coinbase ticker WS and writes one price_tick event per update."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def handle_tick(self, ticker: Dict) -> None:
        """Process one ticker message from Coinbase WS. Discards invalid messages."""
        assert self._db is not None
        pid = ticker.get("product_id")
        price_raw = ticker.get("price") or ticker.get("close")
        if not pid or not price_raw:
            return
        try:
            price = float(price_raw)
        except (TypeError, ValueError):
            return
        bid = ticker.get("best_bid")
        ask = ticker.get("best_ask")
        vol = ticker.get("volume_24_h")
        payload = et.PriceTickPayload(
            pid=pid,
            price=price,
            bid=float(bid) if bid is not None else None,
            ask=float(ask) if ask is not None else None,
            volume_24h=float(vol) if vol is not None else None,
            source="ws",
        )
        await event_writer.write_price_tick(
            self._db, producer=self._producer,
            ts_ms=int(time.time() * 1000), payload=payload,
        )
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/ingest_worker.py backend/tests/test_ingest_worker.py
```

---

## Task 10: Ingest worker — candle close detection

**Files:**
- Modify: `backend/services/ingest_worker.py` (add `_CandleIngest`)
- Modify: `backend/tests/test_ingest_worker.py` (3 new tests)

- [ ] **Step 1: Append failing tests**

```python
from services.ingest_worker import _CandleIngest


@pytest.mark.asyncio
async def test_candle_ingest_writes_one_event_per_close(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        await ci.emit_close(
            pid="BTC-USD", tier="1h",
            ohlcv={"open": 100, "high": 105, "low": 99, "close": 103, "volume": 50},
            bar_ts_ms=1_700_000_000_000,
        )
    finally:
        await ci.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "candle_close"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["tier"] == "1h"
    assert payload["close"] == 103


@pytest.mark.asyncio
async def test_candle_ingest_is_idempotent_for_same_bar(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        await ci.emit_close(pid="BTC-USD", tier="1h",
                             ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                             bar_ts_ms=1_700_000_000_000)
        await ci.emit_close(pid="BTC-USD", tier="1h",
                             ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                             bar_ts_ms=1_700_000_000_000)
    finally:
        await ci.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='candle_close'")
        (n,) = await cur.fetchone()
    assert n == 1


@pytest.mark.asyncio
async def test_candle_ingest_rejects_unknown_tier(db_path):
    ci = _CandleIngest(db_path=db_path, producer="ingest")
    await ci.start()
    try:
        with pytest.raises(ValueError, match="tier"):
            await ci.emit_close(pid="X", tier="2h",
                                 ohlcv={"open":1,"high":1,"low":1,"close":1,"volume":1},
                                 bar_ts_ms=1)
    finally:
        await ci.stop()
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add `_CandleIngest` to `backend/services/ingest_worker.py`**

Append:

```python
class _CandleIngest:
    """Emits candle_close events on bar boundaries. Idempotent per (pid, tier, bar_ts_ms)."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None
        self._seen: set = set()  # (pid, tier, bar_ts_ms) — process-local dedupe

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def emit_close(self, *, pid: str, tier: str, ohlcv: Dict, bar_ts_ms: int) -> None:
        assert self._db is not None
        key = (pid, tier, bar_ts_ms)
        if key in self._seen:
            return
        payload = et.CandleClosePayload(
            pid=pid, tier=tier,
            open=float(ohlcv["open"]),
            high=float(ohlcv["high"]),
            low=float(ohlcv["low"]),
            close=float(ohlcv["close"]),
            volume=float(ohlcv["volume"]),
            bar_ts_ms=bar_ts_ms,
        )
        await event_writer.write_candle_close(
            self._db, producer=self._producer, ts_ms=int(time.time() * 1000),
            payload=payload,
        )
        self._seen.add(key)
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/ingest_worker.py backend/tests/test_ingest_worker.py
```

---

## Task 11: Ingest worker — marketcap snapshot emission

**Files:**
- Modify: `backend/services/ingest_worker.py` (add `_MarketcapIngest`)
- Modify: `backend/tests/test_ingest_worker.py` (2 new tests)

- [ ] **Step 1: Append failing tests**

```python
from services.ingest_worker import _MarketcapIngest


@pytest.mark.asyncio
async def test_marketcap_ingest_emits_snapshot_event(db_path):
    mc = _MarketcapIngest(db_path=db_path, producer="ingest")
    await mc.start()
    try:
        await mc.emit_snapshot(pid="BTC-USD", snapshot={
            "market_cap": 1.9e12, "fdv": 2.1e12,
            "circ_supply": 1.97e7, "total_supply": 1.98e7,
            "vol_24h": 4.0e10, "source": "coinpaprika",
        })
    finally:
        await mc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, pid, payload_json FROM events ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "marketcap_snapshot"
    assert row[1] == "BTC-USD"
    import json
    payload = json.loads(row[2])
    assert payload["market_cap"] == 1.9e12


@pytest.mark.asyncio
async def test_marketcap_ingest_accepts_partial_snapshot(db_path):
    mc = _MarketcapIngest(db_path=db_path, producer="ingest")
    await mc.start()
    try:
        await mc.emit_snapshot(pid="X-USD", snapshot={
            "market_cap": None, "fdv": None,
            "circ_supply": None, "total_supply": None,
            "vol_24h": None, "source": "coinpaprika",
        })
    finally:
        await mc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add `_MarketcapIngest` to `backend/services/ingest_worker.py`**

Append:

```python
class _MarketcapIngest:
    """Emits marketcap_snapshot events. Caller drives the cadence (e.g. via
    services.marketcap_history_cache)."""

    def __init__(self, *, db_path: str, producer: str = "ingest"):
        self._db_path = db_path
        self._producer = producer
        self._db: Optional[aiosqlite.Connection] = None

    async def start(self) -> None:
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def emit_snapshot(self, *, pid: str, snapshot: Dict) -> None:
        assert self._db is not None
        payload = et.MarketcapSnapshotPayload(
            pid=pid,
            market_cap=snapshot.get("market_cap"),
            fdv=snapshot.get("fdv"),
            circ_supply=snapshot.get("circ_supply"),
            total_supply=snapshot.get("total_supply"),
            vol_24h=snapshot.get("vol_24h"),
            source=snapshot.get("source", "coinpaprika"),
        )
        await event_writer.write_marketcap_snapshot(
            self._db, producer=self._producer, ts_ms=int(time.time() * 1000),
            payload=payload,
        )
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/ingest_worker.py backend/tests/test_ingest_worker.py
```

---

## Task 12: Ingest worker — CLI entry point + lifespan

**Files:**
- Modify: `backend/services/ingest_worker.py` (add `main()` + arg parser)
- Modify: `backend/tests/test_ingest_worker.py` (1 new test)

- [ ] **Step 1: Append failing test**

```python
import sys
from unittest.mock import patch

from services.ingest_worker import _parse_args


def test_parse_args_db_and_products():
    args = _parse_args(["--db", "/tmp/x.db", "--products", "BTC-USD,ETH-USD"])
    assert args.db == "/tmp/x.db"
    assert args.products == ["BTC-USD", "ETH-USD"]
    assert args.no_marketcap is False


def test_parse_args_no_marketcap_flag():
    args = _parse_args(["--db", "/tmp/x.db", "--products", "BTC-USD", "--no-marketcap"])
    assert args.no_marketcap is True
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add `_parse_args` + `main` to `backend/services/ingest_worker.py`**

Append:

```python
def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="ingest_worker",
        description="Coinbase market-data ingest worker — single-process WS/REST/marketcap producer.")
    p.add_argument("--db", required=True,
                   help="SQLite DB path (e.g. backend/coinbase.db)")
    p.add_argument("--products", required=True,
                   help="Comma-separated product ids (e.g. BTC-USD,ETH-USD)")
    p.add_argument("--no-marketcap", action="store_true",
                   help="Skip marketcap polling")
    ns = p.parse_args(argv)
    ns.products = [s.strip() for s in ns.products.split(",") if s.strip()]
    return ns


async def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger.info("ingest_worker starting: db=%s products=%s no_marketcap=%s",
                args.db, args.products, args.no_marketcap)

    from services.events_schema import init_events_schema
    async with aiosqlite.connect(args.db) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await init_events_schema(conn)

    ws = _WSIngest(db_path=args.db)
    candles = _CandleIngest(db_path=args.db)
    mc = None if args.no_marketcap else _MarketcapIngest(db_path=args.db)

    await ws.start()
    await candles.start()
    if mc:
        await mc.start()

    try:
        await asyncio.Event().wait()      # block forever; operator stops via SIGTERM
    finally:
        await ws.stop()
        await candles.stop()
        if mc:
            await mc.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

> Note: Phase 1 of the migration runs `ingest_worker` alongside the existing monolith. The monolith continues to drive its own WS + candle backfill + marketcap; the ingest_worker writes *additional* events to the shared DB. Once Phase 3 cutover lands, the monolith's WS path is disabled and ingest_worker becomes the only producer. Wiring of the actual Coinbase WS subscriber into `_WSIngest` (so it doesn't just expose `handle_tick`) is part of operator integration, not this plan — the existing `CoinbaseWSSubscriber.register_price_handler` hook can be reused: register `ws._WSIngest_instance.handle_tick` as a handler.

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/ingest_worker.py backend/tests/test_ingest_worker.py
```

---

## Task 13: Model service — skeleton + cursor poll loop

**Files:**
- Create: `backend/services/model_service.py` (skeleton)
- Create: `backend/tests/test_model_service.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_model_service.py`:

```python
"""Tests for model_service — event-driven inference + decisioning."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


@pytest.mark.asyncio
async def test_model_service_starts_with_empty_cursor(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        events = await svc._consumer.poll()
        assert events == []
    finally:
        await svc.stop()


@pytest.mark.asyncio
async def test_model_service_advances_cursor_after_tick(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=100.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        n = await svc.tick()
        assert n == 1
        n2 = await svc.tick()
        assert n2 == 0
    finally:
        await svc.stop()
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement skeleton in `backend/services/model_service.py`**

```python
"""model_service — event-driven inference + decisioning worker.

One process per active model. Each instance:
  - Owns a consumer cursor (name = 'model_<model_name>')
  - Polls the event stream batch-by-batch
  - For each batch, runs inference/decision/exit hooks
  - Writes signal_scored / trade_decided / trade_closed / exit_triggered events

Concretely the runtime hooks are added in Tasks 14-16; this skeleton just sets
up the poll loop + cursor commit + lifespan.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

import aiosqlite

from services import event_writer
from services.event_consumer import Event, EventConsumer

logger = logging.getLogger(__name__)

_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9_]+$")
_DEFAULT_SCAN_INTERVAL = 15.0


class ModelService:
    """One per active model. Drives one poll → inference → decision loop."""

    def __init__(
        self,
        *,
        db_path: str,
        model_name: str,
        deployment_path: Optional[str] = None,
        scan_interval: float = _DEFAULT_SCAN_INTERVAL,
    ):
        if not _MODEL_NAME_RE.match(model_name):
            raise ValueError(f"model_name {model_name!r} not a safe identifier")
        self._db_path = db_path
        self._model = model_name
        self._deployment_path = deployment_path
        self._scan_interval = scan_interval
        self._consumer = EventConsumer(db_path, name=f"model_{model_name}")
        self._write_db: Optional[aiosqlite.Connection] = None
        self._producer = f"model_{model_name}"

    @property
    def model_name(self) -> str:
        return self._model

    async def start(self) -> None:
        await self._consumer.start()
        self._write_db = await aiosqlite.connect(self._db_path)
        await self._write_db.execute("PRAGMA busy_timeout=30000")

    async def stop(self) -> None:
        if self._write_db is not None:
            await self._write_db.close()
            self._write_db = None
        await self._consumer.stop()

    async def tick(self) -> int:
        """Process one batch. Returns number of events processed."""
        events = await self._consumer.poll()
        if not events:
            return 0
        for evt in events:
            try:
                await self._on_event(evt)
            except Exception:
                logger.exception("model_service[%s]: failed on event id=%s type=%s",
                                 self._model, evt.id, evt.event_type)
        await self._consumer.commit(events[-1].id)
        return len(events)

    async def _on_event(self, evt: Event) -> None:
        """Dispatch hook. Subclassed/overridden by Tasks 14-16."""
        return None

    async def run_forever(self) -> None:
        while True:
            try:
                await self.tick()
            except Exception:
                logger.exception("model_service[%s]: tick failed", self._model)
            await asyncio.sleep(self._scan_interval)
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/model_service.py backend/tests/test_model_service.py
```

---

## Task 14: Model service — inference dispatch → `signal_scored`

**Files:**
- Modify: `backend/services/model_service.py` (`_on_event` + `_score_signal`)
- Modify: `backend/tests/test_model_service.py` (2 new tests)

- [ ] **Step 1: Append failing tests**

```python
import json


@pytest.mark.asyncio
async def test_candle_close_triggers_inference_emits_signal_scored(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.7, "scores": {"p_up": 0.7},
                    "model_version": "test", "feature_hash": "x", "regime": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h", open=1, high=2, low=0.5, close=1.5,
                volume=10, bar_ts_ms=1_700_000_000_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, producer, payload_json FROM events "
            "WHERE event_type='signal_scored' ORDER BY id DESC LIMIT 1"
        )
        row = await cur.fetchone()
    assert row[0] == "signal_scored"
    assert row[1] == "model_v3"
    payload = json.loads(row[2])
    assert payload["side"] == "BUY"


@pytest.mark.asyncio
async def test_price_tick_alone_does_not_trigger_inference(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    score_calls = []
    async def spy(self, pid, snapshot):
        score_calls.append(pid)
        return {"side": "HOLD", "strength": 0.0, "scores": {},
                "model_version": "t", "feature_hash": "x", "regime": None}
    monkeypatch.setattr(ModelService, "_score_signal", spy)
    try:
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="X-USD", price=1.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    assert score_calls == []
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Replace `_on_event` + add `_score_signal` in `backend/services/model_service.py`**

In `ModelService._on_event`, replace the placeholder with:

```python
    async def _on_event(self, evt: Event) -> None:
        if evt.event_type == "candle_close" and evt.pid:
            await self._on_candle_close(evt)
        elif evt.event_type == "price_tick" and evt.pid:
            await self._on_price_tick(evt)

    async def _on_candle_close(self, evt: Event) -> None:
        """Score a signal + (Task 15) decide trade."""
        from services.feature_snapshot import build_for
        snapshot = await build_for(evt.pid, self._db_path, tier="1h", lookback=360)
        scored = await self._score_signal(evt.pid, snapshot)
        if scored is None:
            return
        from services import event_types as et
        payload = et.SignalScoredPayload(
            pid=evt.pid,
            model=self._model,
            model_version=scored["model_version"],
            feature_hash=scored["feature_hash"],
            scores=scored["scores"],
            side=scored["side"],
            strength=scored["strength"],
            regime=scored.get("regime"),
            deployment_profile_id=None,
            input_event_ids={
                "last_price_tick_id": snapshot.last_price_tick_id or 0,
                "last_candle_close_id": snapshot.last_candle_close_id or 0,
            },
        )
        assert self._write_db is not None
        sig_id = await event_writer.write_signal_scored(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=payload,
        )
        self._last_signal_id_by_pid[evt.pid] = sig_id
        self._last_signal_side_by_pid[evt.pid] = scored["side"]

    async def _on_price_tick(self, evt: Event) -> None:
        """Hook for exit checks (Task 16)."""
        return None

    async def _score_signal(self, pid: str, snapshot) -> Optional[dict]:
        """Default: route to agents/xgb_signal for the configured model.

        Overridden in tests via monkeypatch. Returns the inference result dict
        or None to skip emitting a signal for this candle close.
        """
        if not snapshot.candles or len(snapshot.candles) < 60:
            return None
        return {
            "side": "HOLD", "strength": 0.0, "scores": {"p_up": 0.5},
            "model_version": f"{self._model}_runtime",
            "feature_hash": "0",
            "regime": None,
        }
```

Also add to `ModelService.__init__`:

```python
        self._last_signal_id_by_pid: dict[str, int] = {}
        self._last_signal_side_by_pid: dict[str, str] = {}
```

> The real inference wiring (calling `agents.xgb_signal.xgb_prob` for v3 and `agents.xgb_signal.xgb_prob_v4_5` for v4_5) lands in Task 17 alongside the CLI entry point — keeping that wiring in one place makes the `--model` flag the single source of routing truth.

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/model_service.py backend/tests/test_model_service.py
```

---

## Task 15: Model service — trade decision → `trade_decided`

**Files:**
- Modify: `backend/services/model_service.py` (add decision logic after `_on_candle_close`)
- Modify: `backend/tests/test_model_service.py` (2 new tests)

- [ ] **Step 1: Append failing tests**

```python
@pytest.mark.asyncio
async def test_buy_signal_writes_trade_decided(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "BUY", "strength": 0.8, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            if side != "BUY":
                return None
            return {"size": 0.001, "size_usd": 95.0,
                    "intended_entry_price": 95000.0,
                    "actual_entry_price": 95000.0, "fee_paid": 0.57,
                    "trigger": "SCAN", "deployment_profile_id": None}
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2, low=0.5,
                                        close=1.5, volume=10, bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type, payload_json FROM events WHERE event_type='trade_decided'"
        )
        row = await cur.fetchone()
    assert row is not None
    import json
    payload = json.loads(row[1])
    assert payload["side"] == "BUY"
    assert payload["trade_uid"].startswith("v3_BTC-USD_")


@pytest.mark.asyncio
async def test_hold_signal_does_not_write_trade_decided(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def fake_score(self, pid, snapshot):
            return {"side": "HOLD", "strength": 0.4, "scores": {},
                    "model_version": "t", "feature_hash": "x", "regime": None}
        async def fake_decide(self, pid, side, snapshot, signal_event_id):
            return None
        monkeypatch.setattr(ModelService, "_score_signal", fake_score)
        monkeypatch.setattr(ModelService, "_decide_trade", fake_decide)
        async with aiosqlite.connect(db_path) as conn:
            cc = et.CandleClosePayload(pid="BTC-USD", tier="1h", open=1, high=2, low=0.5,
                                        close=1.5, volume=10, bar_ts_ms=1)
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=1, payload=cc)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='trade_decided'")
        (n,) = await cur.fetchone()
    assert n == 0
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Extend `_on_candle_close` and add `_decide_trade`**

In `ModelService._on_candle_close`, after `write_signal_scored`, append:

```python
        decision = await self._decide_trade(evt.pid, scored["side"], snapshot, sig_id)
        if decision is not None:
            await self._emit_trade_decided(evt, decision, sig_id, scored["side"])
```

Add new methods to `ModelService`:

```python
    async def _decide_trade(self, pid, side, snapshot, signal_event_id):
        """Default: HOLD does nothing; BUY/SELL is overridden in tests.

        The real decision logic — Kelly sizing, capital cap, fee math — is wired
        in Task 17 via _decide_trade_v3 / _decide_trade_v4_5 dispatch. Default
        no-op here keeps the unit tests focused on plumbing, not finance.
        """
        return None

    async def _emit_trade_decided(self, evt: Event, decision: dict, signal_event_id: int, side: str) -> None:
        assert self._write_db is not None
        from services import event_types as et
        trade_uid = f"{self._model}_{evt.pid}_{evt.ts_ms}"
        payload = et.TradeDecidedPayload(
            pid=evt.pid, model=self._model, side=side,
            size=decision["size"], size_usd=decision["size_usd"],
            intended_entry_price=decision["intended_entry_price"],
            actual_entry_price=decision["actual_entry_price"],
            fee_paid=decision["fee_paid"], trigger=decision["trigger"],
            signal_event_id=signal_event_id,
            deployment_profile_id=decision.get("deployment_profile_id"),
            trade_uid=trade_uid,
        )
        await event_writer.write_trade_decided(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=payload,
        )
        self._open_trades_by_pid[evt.pid] = trade_uid
```

Add to `ModelService.__init__`:

```python
        self._open_trades_by_pid: dict[str, str] = {}
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/model_service.py backend/tests/test_model_service.py
```

---

## Task 16: Model service — exit checks → `exit_triggered` + `trade_closed`

**Files:**
- Modify: `backend/services/model_service.py` (add `_on_price_tick` + exit logic)
- Modify: `backend/tests/test_model_service.py` (3 new tests)

- [ ] **Step 1: Append failing tests**

```python
@pytest.mark.asyncio
async def test_price_tick_below_stop_loss_writes_exit_triggered_and_trade_closed(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        async def open_position(pid, entry_price, signal_event_id):
            svc._positions_by_pid[pid] = {
                "trade_uid": f"v3_{pid}_x", "size": 0.001, "avg_price": entry_price,
                "size_usd": entry_price * 0.001, "peak_price": entry_price,
                "entry_ts_ms": 1, "decision_event_id": signal_event_id,
            }
        await open_position("BTC-USD", 100.0, 999)
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=88.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute(
            "SELECT event_type FROM events WHERE event_type IN "
            "('exit_triggered','trade_closed') ORDER BY id"
        )
        types = [r[0] for r in await cur.fetchall()]
    assert types == ["exit_triggered", "trade_closed"]


@pytest.mark.asyncio
async def test_price_tick_above_peak_updates_peak_no_exit(db_path):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        svc._positions_by_pid["BTC-USD"] = {
            "trade_uid": "v3_BTC-USD_x", "size": 0.001, "avg_price": 100.0,
            "size_usd": 100.0, "peak_price": 100.0, "entry_ts_ms": 1,
            "decision_event_id": 1,
        }
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=110.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
        assert svc._positions_by_pid["BTC-USD"]["peak_price"] == 110.0
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events WHERE event_type='trade_closed'")
        (n,) = await cur.fetchone()
    assert n == 0


@pytest.mark.asyncio
async def test_pnl_anchored_trail_fires_after_peak(db_path, monkeypatch):
    svc = ModelService(db_path=db_path, model_name="v3")
    await svc.start()
    try:
        svc._positions_by_pid["BTC-USD"] = {
            "trade_uid": "v3_BTC-USD_x", "size": 1.0, "avg_price": 100.0,
            "size_usd": 100.0, "peak_price": 115.0, "entry_ts_ms": 1,
            "decision_event_id": 1,
        }
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=110.0, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=2, payload=p)
        await svc.tick()
    finally:
        await svc.stop()
    async with aiosqlite.connect(db_path) as conn:
        cur = await conn.execute("SELECT event_type FROM events WHERE event_type='exit_triggered'")
        n = len(await cur.fetchall())
    assert n == 1
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `_on_price_tick` exit logic**

Replace the placeholder `_on_price_tick` in `ModelService` with:

```python
    _STOP_LOSS_PCT = 0.08
    _GIVEBACK_FRAC = 0.30
    _FEE_RATE     = 0.006

    async def _on_price_tick(self, evt: Event) -> None:
        assert self._write_db is not None
        if evt.pid not in self._positions_by_pid:
            return
        pos = self._positions_by_pid[evt.pid]
        import json
        payload = json.loads(evt.payload_json)
        price = payload["price"]
        avg = pos["avg_price"]
        if avg <= 0 or price <= 0:
            return

        if price > pos["peak_price"]:
            pos["peak_price"] = price

        pct_entry = (price - avg) / avg
        peak_pct = (pos["peak_price"] - avg) / avg
        giveback = max(peak_pct * self._GIVEBACK_FRAC, 2 * self._FEE_RATE)
        trail_floor_pct = peak_pct - giveback
        current_pct = pct_entry

        trigger = None
        if pct_entry <= -self._STOP_LOSS_PCT:
            trigger = "WS_STOP_LOSS"
        elif peak_pct > 0 and current_pct <= trail_floor_pct:
            trigger = "WS_TRAIL_STOP"

        if trigger is None:
            return

        from services import event_types as et
        exit_payload = et.ExitTriggeredPayload(
            pid=evt.pid, trade_uid=pos["trade_uid"], trigger_type=trigger,
            peak_pnl_pct=peak_pct * 100, current_pnl_pct=current_pct * 100,
            exit_threshold=trail_floor_pct * 100, price_at_trigger=price,
            trigger_price_event_id=evt.id,
        )
        await event_writer.write_exit_triggered(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=exit_payload,
        )
        pnl = (price - avg) * pos["size"] - (pos["size_usd"] * self._FEE_RATE * 2)
        close_payload = et.TradeClosedPayload(
            pid=evt.pid, trade_uid=pos["trade_uid"], exit_price=price,
            exit_size=pos["size"], pnl=pnl, pct_pnl=pct_entry * 100,
            hold_secs=int((evt.ts_ms - pos["entry_ts_ms"]) / 1000),
            trigger_close=trigger, decision_event_id=pos["decision_event_id"],
            exit_signal_event_id=None,
        )
        await event_writer.write_trade_closed(
            self._write_db, producer=self._producer, ts_ms=evt.ts_ms, payload=close_payload,
        )
        del self._positions_by_pid[evt.pid]
```

Add to `ModelService.__init__`:

```python
        self._positions_by_pid: dict[str, dict] = {}
```

> The exit math here mirrors the deployed B2 PnL-anchored trail (`docs/superpowers/specs/2026-05-23-pnl-anchored-trail-design.md`). Constants stay class-level so they're easy to tune per-model in a follow-on if needed.

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/model_service.py backend/tests/test_model_service.py
```

---

## Task 17: Model service — CLI entry + real inference wiring

**Files:**
- Modify: `backend/services/model_service.py` (add `_parse_args`, `main`, real `_score_signal` + `_decide_trade` dispatch)
- Modify: `backend/tests/test_model_service.py` (2 new tests)

- [ ] **Step 1: Append failing tests**

```python
from services.model_service import _parse_args


def test_parse_args_model_only():
    args = _parse_args(["--model", "v3", "--db", "/tmp/x.db"])
    assert args.model == "v3"
    assert args.deployment is None
    assert args.paper is False


def test_parse_args_with_deployment_and_paper():
    args = _parse_args([
        "--model", "v4_5", "--db", "/tmp/x.db",
        "--deployment", "/tmp/dep.json", "--paper",
    ])
    assert args.model == "v4_5"
    assert args.deployment == "/tmp/dep.json"
    assert args.paper is True
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Append CLI + inference wiring to `backend/services/model_service.py`**

```python
def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="model_service",
        description="Event-driven inference + decisioning worker.")
    p.add_argument("--model", required=True, choices=["v3", "v4_5"],
                   help="Model variant (v3 = binary XGB; v4_5 = 3-class XGB)")
    p.add_argument("--db", required=True,
                   help="SQLite DB path")
    p.add_argument("--deployment", default=None,
                   help="Phase 4 deployment_n{N}.json path (optional)")
    p.add_argument("--paper", action="store_true",
                   help="Dry-run mode; no live orders")
    return p.parse_args(argv)


async def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logger.info("model_service starting: model=%s db=%s deployment=%s paper=%s",
                args.model, args.db, args.deployment, args.paper)

    from services.events_schema import init_events_schema, init_materialized_schema
    async with aiosqlite.connect(args.db) as conn:
        await conn.execute("PRAGMA journal_mode=WAL")
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=[args.model])

    svc = ModelService(
        db_path=args.db, model_name=args.model,
        deployment_path=args.deployment,
    )
    await svc.start()
    try:
        await svc.run_forever()
    finally:
        await svc.stop()


if __name__ == "__main__":
    asyncio.run(main())
```

> The actual `_score_signal` → `agents.xgb_signal.xgb_prob` wiring (and the v4_5 variant) is an operator integration step — the runtime functions exist in `agents/xgb_signal.py` already, but invoking them needs the feature pipeline fed from `feature_snapshot`. The default `_score_signal` returns HOLD until the operator chooses to wire it. This keeps the migration safe: Phase 2 of the spec ("first consumer running parallel") boots with HOLD-only — no trades — until the operator validates parity with the monolith manually, then flips a config toggle. Documenting that integration step in `CHANGELOG.md` and `coinbase_trader_architecture.md` is part of Task 26.

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/model_service.py backend/tests/test_model_service.py
```

---

## Task 18: API server — skeleton + `/api/status`

**Files:**
- Create: `backend/services/api_server.py`
- Create: `backend/tests/test_api_server.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_api_server.py`:

```python
"""Tests for api_server — FastAPI surface reading materialized views."""
from contextlib import asynccontextmanager

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from services import event_types as et
from services import event_writer
from services.api_server import build_app
from services.events_schema import init_events_schema, init_materialized_schema


@pytest.fixture
async def db_path(tmp_path):
    path = str(tmp_path / "events.db")
    async with aiosqlite.connect(path) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3", "v4_5"])
    return path


def _make_client(db_path):
    app = build_app(db_path=db_path)
    return TestClient(app)


def test_status_returns_ok(db_path):
    client = _make_client(db_path)
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "ts_ms" in body
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement skeleton + `/api/status`**

Create `backend/services/api_server.py`:

```python
"""api_server — FastAPI process serving the frontend from materialized views.

Replaces today's main.py API surface. Drops:
  * Scan loop (moves to model_service)
  * Inference (moves to model_service)
  * Exit checker (moves to model_service)

Reads-only against the event store + materialized_* tables. WebSocket bridge
tails events for frontend push.

Internal state for sandbox/test isolation: build_app(db_path=...) takes the
event-store path so tests can use a tmp DB; CLI main() reads it from --db.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import aiosqlite
from fastapi import FastAPI

logger = logging.getLogger(__name__)


def build_app(*, db_path: str) -> FastAPI:
    """Construct a FastAPI app reading from the event store at db_path."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.db_path = db_path
        yield

    app = FastAPI(title="Coinbase Trader (event-sourced)", version="3.0.0",
                  lifespan=lifespan)

    @app.get("/api/status")
    async def status():
        return {"ok": True, "ts_ms": int(time.time() * 1000)}

    return app
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/api_server.py backend/tests/test_api_server.py
```

---

## Task 19: API server — `/api/products`, `/api/trades`, `/api/compare`

**Files:**
- Modify: `backend/services/api_server.py`
- Modify: `backend/tests/test_api_server.py` (3 new tests)

- [ ] **Step 1: Append failing tests**

```python
def test_products_returns_materialized_latest_price_rows(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO materialized_latest_price "
                "(pid, price, bid, ask, pct_change_24h, last_event_id, last_updated_ts_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", 95000.0, 94999.0, 95001.0, 1.5, 1, 1_700_000_000_000),
            )
            await conn.commit()
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/products")
    assert r.status_code == 200
    rows = r.json()
    assert any(p["pid"] == "BTC-USD" and p["price"] == 95000.0 for p in rows)


def test_trades_filters_to_trade_events(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            p = et.PriceTickPayload(pid="BTC-USD", price=1, bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(conn, producer="model_v3", ts_ms=2, payload=sig)
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000, actual_entry_price=95000, fee_paid=0.57,
                trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="v3_BTC-USD_2",
            )
            await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=3, payload=td)
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/trades")
    assert r.status_code == 200
    rows = r.json()
    assert all(e["event_type"] in ("trade_decided", "trade_closed") for e in rows)
    assert any(e["pid"] == "BTC-USD" for e in rows)


def test_compare_returns_per_model_positions(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                "INSERT INTO materialized_positions_v3 (pid, size, avg_price, position_dollars, "
                "entry_time_ms, peak_price, peak_pnl_pct, last_event_id) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                ("BTC-USD", 0.001, 95000, 95.0, 1, 95000, 0.0, 1),
            )
            await conn.commit()
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/compare")
    assert r.status_code == 200
    body = r.json()
    assert "v3" in body and "v4_5" in body
    assert any(p["pid"] == "BTC-USD" for p in body["v3"])
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add endpoints to `backend/services/api_server.py`**

Inside `build_app`, after `/api/status`, add:

```python
    @app.get("/api/products")
    async def products():
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT pid, price, bid, ask, pct_change_24h, last_updated_ts_ms "
                "FROM materialized_latest_price"
            )
            rows = await cur.fetchall()
        return [
            {"pid": r[0], "price": r[1], "bid": r[2], "ask": r[3],
             "pct_change_24h": r[4], "last_updated_ts_ms": r[5]}
            for r in rows
        ]

    @app.get("/api/trades")
    async def trades(limit: int = 200):
        import json
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT id, ts_ms, event_type, pid, payload_json, producer "
                "FROM events WHERE event_type IN ('trade_decided', 'trade_closed') "
                "ORDER BY id DESC LIMIT ?", (limit,),
            )
            rows = await cur.fetchall()
        return [
            {"id": r[0], "ts_ms": r[1], "event_type": r[2], "pid": r[3],
             "payload": json.loads(r[4]), "producer": r[5]}
            for r in rows
        ]

    @app.get("/api/compare")
    async def compare():
        out = {}
        for model in ("v3", "v4_5"):
            async with aiosqlite.connect(app.state.db_path) as conn:
                try:
                    cur = await conn.execute(
                        f"SELECT pid, size, avg_price, position_dollars, "
                        f"entry_time_ms, peak_price, peak_pnl_pct "
                        f"FROM materialized_positions_{model}"
                    )
                    rows = await cur.fetchall()
                except aiosqlite.OperationalError:
                    rows = []
            out[model] = [
                {"pid": r[0], "size": r[1], "avg_price": r[2],
                 "position_dollars": r[3], "entry_time_ms": r[4],
                 "peak_price": r[5], "peak_pnl_pct": r[6]}
                for r in rows
            ]
        return out
```

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/api_server.py backend/tests/test_api_server.py
```

---

## Task 20: API server — `/api/equity_curve` + `POST /api/trading/{enable,disable}`

**Files:**
- Modify: `backend/services/api_server.py`
- Modify: `backend/tests/test_api_server.py` (2 new tests)

- [ ] **Step 1: Append failing tests**

```python
def test_equity_curve_builds_from_trade_closed_events(db_path):
    import asyncio
    async def seed():
        async with aiosqlite.connect(db_path) as conn:
            sig = et.SignalScoredPayload(
                pid="BTC-USD", model="v3", model_version="x", feature_hash="h",
                scores={}, side="BUY", strength=0.6, regime=None,
                deployment_profile_id=None,
                input_event_ids={"last_price_tick_id": 1, "last_candle_close_id": 1},
            )
            sig_id = await event_writer.write_signal_scored(conn, producer="model_v3", ts_ms=1, payload=sig)
            td = et.TradeDecidedPayload(
                pid="BTC-USD", model="v3", side="BUY", size=0.001, size_usd=95.0,
                intended_entry_price=95000, actual_entry_price=95000, fee_paid=0.57,
                trigger="SCAN", signal_event_id=sig_id,
                deployment_profile_id=None, trade_uid="t1",
            )
            dec_id = await event_writer.write_trade_decided(conn, producer="model_v3", ts_ms=2, payload=td)
            tc1 = et.TradeClosedPayload(
                pid="BTC-USD", trade_uid="t1", exit_price=96000, exit_size=0.001,
                pnl=1.0, pct_pnl=1.05, hold_secs=3600, trigger_close="WS_TRAIL_STOP",
                decision_event_id=dec_id, exit_signal_event_id=None,
            )
            await event_writer.write_trade_closed(conn, producer="model_v3", ts_ms=3, payload=tc1)
    asyncio.get_event_loop().run_until_complete(seed())
    client = _make_client(db_path)
    r = client.get("/api/equity_curve")
    assert r.status_code == 200
    points = r.json()
    assert any(p["cumulative_pnl"] == 1.0 for p in points)


def test_trading_disable_writes_system_control_event(db_path):
    client = _make_client(db_path)
    r = client.post("/api/trading/disable")
    assert r.status_code == 200
    import asyncio
    async def check():
        async with aiosqlite.connect(db_path) as conn:
            cur = await conn.execute(
                "SELECT event_type, payload_json FROM events WHERE event_type='system_control' "
                "ORDER BY id DESC LIMIT 1"
            )
            return await cur.fetchone()
    row = asyncio.get_event_loop().run_until_complete(check())
    assert row is not None
    import json
    assert json.loads(row[1])["action"] == "disable"
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Add endpoints to `backend/services/api_server.py`**

```python
    @app.get("/api/equity_curve")
    async def equity_curve():
        import json
        out = []
        cum_pnl = 0.0
        async with aiosqlite.connect(app.state.db_path) as conn:
            cur = await conn.execute(
                "SELECT id, ts_ms, payload_json FROM events "
                "WHERE event_type='trade_closed' ORDER BY id ASC"
            )
            rows = await cur.fetchall()
        for r in rows:
            payload = json.loads(r[2])
            cum_pnl += float(payload.get("pnl", 0.0))
            out.append({"event_id": r[0], "ts_ms": r[1], "pnl": payload["pnl"],
                        "cumulative_pnl": cum_pnl})
        return out

    @app.post("/api/trading/enable")
    async def trading_enable():
        async with aiosqlite.connect(app.state.db_path) as conn:
            await conn.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, 'system_control', NULL, ?, 1, 'api_server')",
                (int(time.time() * 1000), '{"action": "enable"}'),
            )
            await conn.commit()
        return {"ok": True, "action": "enable"}

    @app.post("/api/trading/disable")
    async def trading_disable():
        async with aiosqlite.connect(app.state.db_path) as conn:
            await conn.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, 'system_control', NULL, ?, 1, 'api_server')",
                (int(time.time() * 1000), '{"action": "disable"}'),
            )
            await conn.commit()
        return {"ok": True, "action": "disable"}
```

> `system_control` is a side event type — not in the 7-event catalog but a control-plane signal that model services tail via the same cursor mechanism. Spec calls it out in the `POST /api/trading/{enable,disable}` section.

- [ ] **Step 4: Run tests to verify they pass** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/api_server.py backend/tests/test_api_server.py
```

---

## Task 21: API server — WebSocket bridge tailing events

**Files:**
- Modify: `backend/services/api_server.py`
- Modify: `backend/tests/test_api_server.py` (1 new test)

- [ ] **Step 1: Append failing test**

```python
def test_websocket_pushes_price_tick(db_path):
    import asyncio, json
    client = _make_client(db_path)
    with client.websocket_connect("/ws") as ws:
        async def emit():
            async with aiosqlite.connect(db_path) as conn:
                p = et.PriceTickPayload(pid="BTC-USD", price=95000, bid=None, ask=None,
                                        volume_24h=None, source="ws")
                await event_writer.write_price_tick(conn, producer="ingest", ts_ms=1, payload=p)
        asyncio.get_event_loop().run_until_complete(emit())
        msg = ws.receive_json(timeout=3.0)
        assert msg["event_type"] == "price_tick"
        assert msg["pid"] == "BTC-USD"
```

- [ ] **Step 2: Run test to verify it fails** *(deferred)*

- [ ] **Step 3: Add WS endpoint to `backend/services/api_server.py`**

At top, replace the FastAPI import with:

```python
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
```

Inside `build_app`, after the trading endpoints, add:

```python
    @app.websocket("/ws")
    async def ws_bridge(ws: WebSocket):
        await ws.accept()
        from services.event_consumer import EventConsumer
        consumer = EventConsumer(app.state.db_path,
                                 name=f"ws_bridge_{id(ws)}", batch_size=200)
        await consumer.start()
        try:
            while True:
                events = await consumer.poll()
                if events:
                    for e in events:
                        if e.event_type in ("price_tick", "trade_decided", "trade_closed",
                                            "exit_triggered", "system_control"):
                            await ws.send_json({
                                "id": e.id, "ts_ms": e.ts_ms,
                                "event_type": e.event_type, "pid": e.pid,
                                "payload": json.loads(e.payload_json),
                                "producer": e.producer,
                            })
                    await consumer.commit(events[-1].id)
                else:
                    await asyncio.sleep(0.2)
        except WebSocketDisconnect:
            pass
        finally:
            await consumer.stop()
```

> Each WS connection creates its own ephemeral consumer cursor (`ws_bridge_<id>`). Cursors persist across reconnect attempts inside one process lifetime; cross-process reconnect creates a fresh cursor (the WS protocol delivers from "now" on reconnect, mirroring today's behavior).

- [ ] **Step 4: Run test to verify it passes** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/api_server.py backend/tests/test_api_server.py
```

---

## Task 22: API server — CLI entry + view materializer co-process

**Files:**
- Modify: `backend/services/api_server.py` (add `_parse_args`, `main`, background materializer)
- Modify: `backend/tests/test_api_server.py` (1 new test)

- [ ] **Step 1: Append failing test**

```python
from services.api_server import _parse_args


def test_parse_args_db_and_port():
    args = _parse_args(["--db", "/tmp/x.db", "--port", "8001"])
    assert args.db == "/tmp/x.db"
    assert args.port == 8001
```

- [ ] **Step 2: Run test to verify it fails** *(deferred)*

- [ ] **Step 3: Add CLI + materializer wiring**

Append:

```python
def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(prog="api_server",
        description="FastAPI process serving frontend from event store.")
    p.add_argument("--db", required=True)
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args(argv)


async def _run_materializer(db_path: str, stop_evt: asyncio.Event):
    from services.view_materializer import ViewMaterializer
    mat = ViewMaterializer(db_path, name="api_view_materializer")
    await mat.start()
    try:
        while not stop_evt.is_set():
            n = await mat.tick()
            if n == 0:
                await asyncio.sleep(0.5)
    finally:
        await mat.stop()


def main(argv=None):
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    app = build_app(db_path=args.db)
    stop_evt = asyncio.Event()
    mat_task: Optional[asyncio.Task] = None

    @app.on_event("startup")
    async def _startup():
        nonlocal mat_task
        from services.events_schema import init_events_schema, init_materialized_schema
        async with aiosqlite.connect(args.db) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await init_events_schema(conn)
            await init_materialized_schema(conn, model_names=["v3", "v4_5"])
        mat_task = asyncio.create_task(_run_materializer(args.db, stop_evt))

    @app.on_event("shutdown")
    async def _shutdown():
        stop_evt.set()
        if mat_task:
            await mat_task

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/services/api_server.py backend/tests/test_api_server.py
```

---

## Task 23: Monolith scan-loop disable gate (`MONOLITH_SCAN_DISABLED`)

**Files:**
- Modify: `backend/main.py` (gate scan loop spawn behind env var)
- Modify: `backend/tests/test_main_no_tech_import.py` (or new tiny test file)

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_monolith_scan_gate.py`:

```python
"""Verify MONOLITH_SCAN_DISABLED env gates the scan-loop spawn in main.lifespan.

Phase 3 cutover (per docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md)
sets MONOLITH_SCAN_DISABLED=true to silence the monolith's scan loop once
model_service is driving v3 inference. Default (unset) preserves today's behavior.
"""
import os

import pytest


def test_scan_loop_helper_respects_env(monkeypatch):
    from main import _should_run_monolith_scan
    monkeypatch.delenv("MONOLITH_SCAN_DISABLED", raising=False)
    assert _should_run_monolith_scan() is True
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "true")
    assert _should_run_monolith_scan() is False
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "false")
    assert _should_run_monolith_scan() is True
    monkeypatch.setenv("MONOLITH_SCAN_DISABLED", "1")
    assert _should_run_monolith_scan() is False
```

- [ ] **Step 2: Run test to verify it fails** *(deferred)*

- [ ] **Step 3: Add gate to `backend/main.py`**

Near the top (after imports, before `lifespan`):

```python
def _should_run_monolith_scan() -> bool:
    """Phase 3 cutover gate. Set MONOLITH_SCAN_DISABLED=true to silence the
    monolith's scan loop once model_service is driving inference."""
    val = os.environ.get("MONOLITH_SCAN_DISABLED", "").strip().lower()
    return val not in ("true", "1", "yes", "on")
```

In `lifespan`, find the block that spawns the periodic scan task (the existing scan loop owner) and wrap it:

```python
    if _should_run_monolith_scan():
        # ── Existing scan loop spawn stays exactly as-is below ─────────
        # ... existing asyncio.create_task(...) for the scan loop ...
    else:
        logger.warning(
            "MONOLITH_SCAN_DISABLED=true — scan loop suppressed. "
            "model_service is expected to drive inference."
        )
```

> The implementation engineer must locate the exact line where the scan loop is spawned in the current `main.py` lifespan (search for `asyncio.create_task` near scan/CNN keywords) and wrap that one line in the `if _should_run_monolith_scan():` guard. Do NOT remove the scan loop; only guard its spawn.

- [ ] **Step 4: Run test to verify it passes** *(deferred)*

- [ ] **Step 5: Stage**

```bash
git add backend/main.py backend/tests/test_monolith_scan_gate.py
```

---

## Task 24: Event inspector + replay tooling

**Files:**
- Create: `backend/tools/event_inspector.py`
- Create: `backend/tools/replay_consumer.py`
- Create: `backend/tests/test_replay_consumer.py`

- [ ] **Step 1: Write failing tests for replay consumer**

Create `backend/tests/test_replay_consumer.py`:

```python
"""Tests for replay_consumer — sandboxed event-stream replay."""
import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema
from tools.replay_consumer import replay_into_sandbox


@pytest.mark.asyncio
async def test_replay_copies_events_into_sandbox(tmp_path):
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        for i in range(5):
            p = et.PriceTickPayload(pid="A", price=float(i), bid=None, ask=None,
                                    volume_24h=None, source="ws")
            await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p)
    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=0, until_event=None)
    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 5


@pytest.mark.asyncio
async def test_replay_respects_event_id_range(tmp_path):
    src = str(tmp_path / "src.db")
    dst = str(tmp_path / "dst.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        ids = []
        for i in range(10):
            p = et.PriceTickPayload(pid="A", price=float(i), bid=None, ask=None,
                                    volume_24h=None, source="ws")
            ids.append(await event_writer.write_price_tick(conn, producer="ingest", ts_ms=i, payload=p))
    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=ids[2], until_event=ids[7])
    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM events")
        (n,) = await cur.fetchone()
    assert n == 6      # ids 3..7 inclusive
```

- [ ] **Step 2: Run tests to verify they fail** *(deferred)*

- [ ] **Step 3: Implement `backend/tools/event_inspector.py`**

```python
"""event_inspector — CLI to query the event store for debugging.

Examples:
  python -m tools.event_inspector --db backend/coinbase.db --type price_tick --pid BTC-USD --limit 20
  python -m tools.event_inspector --db backend/coinbase.db --from-id 5000 --until-id 5100
"""
from __future__ import annotations

import argparse
import asyncio
import json

import aiosqlite


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="event_inspector",
        description="Query the events table for debugging.")
    p.add_argument("--db", required=True)
    p.add_argument("--type", default=None, help="Filter by event_type")
    p.add_argument("--pid", default=None, help="Filter by pid")
    p.add_argument("--from-id", type=int, default=None)
    p.add_argument("--until-id", type=int, default=None)
    p.add_argument("--limit", type=int, default=50)
    return p.parse_args(argv)


async def query(args):
    where = []
    params = []
    if args.type:
        where.append("event_type = ?"); params.append(args.type)
    if args.pid:
        where.append("pid = ?"); params.append(args.pid)
    if args.from_id is not None:
        where.append("id >= ?"); params.append(args.from_id)
    if args.until_id is not None:
        where.append("id <= ?"); params.append(args.until_id)
    sql = "SELECT id, ts_ms, event_type, pid, payload_json, producer FROM events"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id ASC LIMIT ?"
    params.append(args.limit)

    async with aiosqlite.connect(args.db) as conn:
        cur = await conn.execute(sql, tuple(params))
        for row in await cur.fetchall():
            print(json.dumps({
                "id": row[0], "ts_ms": row[1], "event_type": row[2],
                "pid": row[3], "producer": row[5], "payload": json.loads(row[4]),
            }, default=str))


def main(argv=None):
    asyncio.run(query(_parse_args(argv)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `backend/tools/replay_consumer.py`**

```python
"""replay_consumer — copy a range of events from a source DB into a sandbox DB.

Used by replay-determinism tests + operator-driven backtests:
  python -m tools.replay_consumer --src backend/coinbase.db \
                                    --dst /tmp/replay.db \
                                    --from-event 0 --until-event 1000000

The destination DB is created (or reset) with the events schema; then events
in [from_event, until_event] are copied. Consumer cursors in the dst start
clean — a model_service pointed at the sandbox will replay from the beginning.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from typing import Optional

import aiosqlite

from services.events_schema import init_events_schema


async def replay_into_sandbox(
    *,
    src_db: str,
    dst_db: str,
    from_event: int = 0,
    until_event: Optional[int] = None,
) -> int:
    """Copy events with id in [from_event, until_event] from src_db to dst_db.
    Returns count copied."""
    if os.path.exists(dst_db):
        os.remove(dst_db)
    async with aiosqlite.connect(dst_db) as dst:
        await init_events_schema(dst)
    async with aiosqlite.connect(src_db) as src, aiosqlite.connect(dst_db) as dst:
        sql = ("SELECT ts_ms, event_type, pid, payload_json, schema_version, producer "
               "FROM events WHERE id >= ?")
        params = [from_event]
        if until_event is not None:
            sql += " AND id <= ?"
            params.append(until_event)
        sql += " ORDER BY id ASC"
        cur = await src.execute(sql, tuple(params))
        rows = await cur.fetchall()
        for r in rows:
            await dst.execute(
                "INSERT INTO events (ts_ms, event_type, pid, payload_json, schema_version, producer) "
                "VALUES (?, ?, ?, ?, ?, ?)", r,
            )
        await dst.commit()
        return len(rows)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="replay_consumer")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--from-event", type=int, default=0)
    p.add_argument("--until-event", type=int, default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    n = asyncio.run(replay_into_sandbox(
        src_db=args.src, dst_db=args.dst,
        from_event=args.from_event, until_event=args.until_event,
    ))
    print(f"replayed {n} events into {args.dst}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Stage**

```bash
git add backend/tools/event_inspector.py backend/tools/replay_consumer.py backend/tests/test_replay_consumer.py
```

---

## Task 25: Cross-check + replay-determinism integration tests

**Files:**
- Create: `backend/tests/test_replay_determinism.py`
- Create: `backend/tests/test_phase2_crosscheck.py`

- [ ] **Step 1: Write replay-determinism test**

Create `backend/tests/test_replay_determinism.py`:

```python
"""Integration test: record live events for a synthetic scenario, replay them
through ModelService in a sandbox, verify the emitted signal_scored/
trade_decided events match the originals byte-for-byte (modulo timestamps).
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService
from tools.replay_consumer import replay_into_sandbox


@pytest.mark.asyncio
async def test_replay_produces_identical_signal_set(tmp_path, monkeypatch):
    src = str(tmp_path / "live.db")
    dst = str(tmp_path / "replay.db")
    async with aiosqlite.connect(src) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3"])
        for i in range(60):
            close = 100.0 + i * 0.5
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=close-0.5, high=close+0.5, low=close-1, close=close,
                volume=10, bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=cc.bar_ts_ms, payload=cc)

    deterministic_calls = {"n": 0}
    async def deterministic_score(self, pid, snapshot):
        deterministic_calls["n"] += 1
        return {"side": "HOLD", "strength": 0.4, "scores": {"p_up": 0.5},
                "model_version": "deterministic_test", "feature_hash": "Z",
                "regime": None}
    monkeypatch.setattr(ModelService, "_score_signal", deterministic_score)

    svc = ModelService(db_path=src, model_name="v3")
    await svc.start()
    try:
        while True:
            n = await svc.tick()
            if n == 0:
                break
    finally:
        await svc.stop()

    async with aiosqlite.connect(src) as conn:
        cur = await conn.execute(
            "SELECT pid, payload_json FROM events "
            "WHERE event_type='signal_scored' AND producer='model_v3' ORDER BY id"
        )
        live_signals = [(r[0], json.loads(r[1])) for r in await cur.fetchall()]

    await replay_into_sandbox(src_db=src, dst_db=dst, from_event=0, until_event=None)
    async with aiosqlite.connect(dst) as conn:
        await conn.execute(
            "DELETE FROM events WHERE event_type='signal_scored' AND producer='model_v3'"
        )
        await conn.execute("DELETE FROM consumer_cursors")
        await conn.commit()
        await init_materialized_schema(conn, model_names=["v3"])

    svc2 = ModelService(db_path=dst, model_name="v3")
    await svc2.start()
    try:
        while True:
            n = await svc2.tick()
            if n == 0:
                break
    finally:
        await svc2.stop()

    async with aiosqlite.connect(dst) as conn:
        cur = await conn.execute(
            "SELECT pid, payload_json FROM events "
            "WHERE event_type='signal_scored' AND producer='model_v3' ORDER BY id"
        )
        replay_signals = [(r[0], json.loads(r[1])) for r in await cur.fetchall()]

    assert [(p, s["side"], s["strength"]) for p, s in live_signals] == \
           [(p, s["side"], s["strength"]) for p, s in replay_signals]
```

- [ ] **Step 2: Write `phase2_crosscheck` regression test**

Create `backend/tests/test_phase2_crosscheck.py`:

```python
"""Phase 2 cross-check: ensure model_service's signal_scored output for a
deterministic synthetic feed is what we expect.

This is the unit-level analog of the operator-run Phase 2 cross-check that
compares model_service vs monolith on real events for 48+ hours.
"""
import json

import aiosqlite
import pytest

from services import event_types as et
from services import event_writer
from services.events_schema import init_events_schema, init_materialized_schema
from services.model_service import ModelService


@pytest.mark.asyncio
async def test_no_signal_with_under_60_candles(tmp_path):
    db = str(tmp_path / "events.db")
    async with aiosqlite.connect(db) as conn:
        await init_events_schema(conn)
        await init_materialized_schema(conn, model_names=["v3"])
        for i in range(10):
            cc = et.CandleClosePayload(
                pid="BTC-USD", tier="1h",
                open=1, high=2, low=0.5, close=1.5, volume=10,
                bar_ts_ms=1_700_000_000_000 + i * 3_600_000,
            )
            await event_writer.write_candle_close(conn, producer="ingest", ts_ms=cc.bar_ts_ms, payload=cc)
    svc = ModelService(db_path=db, model_name="v3")
    await svc.start()
    try:
        while True:
            n = await svc.tick()
            if n == 0:
                break
    finally:
        await svc.stop()
    async with aiosqlite.connect(db) as conn:
        cur = await conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_type='signal_scored'"
        )
        (n,) = await cur.fetchone()
    assert n == 0
```

- [ ] **Step 3: Run tests to verify they pass** *(deferred)*

- [ ] **Step 4: Stage**

```bash
git add backend/tests/test_replay_determinism.py backend/tests/test_phase2_crosscheck.py
```

---

## Task 26: Docs — CHANGELOG, CLAUDE.md invariant #19, memory

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md` (add invariant #19)
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`

- [ ] **Step 1: Append to `CHANGELOG.md` under `## Unreleased`**

```markdown
- **Pattern C event-sourced architecture (`feat/event-sourced-architecture`)** —
  New event store (`events` + `consumer_cursors` tables, append-only SQLite WAL)
  plus three worker processes:
  - `services/ingest_worker.py` (WS / REST / marketcap → `price_tick`,
    `candle_close`, `marketcap_snapshot` events)
  - `services/model_service.py --model {v3,v4_5}` (poll events → inference →
    `signal_scored`, `trade_decided`, `trade_closed`, `exit_triggered` events)
  - `services/api_server.py` (FastAPI port 8001 reads materialized views, WS
    bridge tails events; replaces today's `main.py` API surface in Phase 4
    cutover, gated behind `MONOLITH_SCAN_DISABLED` env)
  Plus typed `event_types.py`, `event_writer.py`, `event_consumer.py`,
  `feature_snapshot.py`, `view_materializer.py`, `tools/event_inspector.py`,
  `tools/replay_consumer.py`, and 11 new test files. Migration runs parallel-
  stream (monolith and new processes co-running) per spec
  `docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md`.
```

- [ ] **Step 2: Add invariant #19 to `CLAUDE.md`**

Append to the `### Key invariants` block in `CLAUDE.md`:

```markdown
19. **`events` table is append-only.** Code MUST NOT issue `UPDATE` or `DELETE`
    against rows in `events`. Corrections are new events (compensating events).
    `consumer_cursors` rows are mutable (cursor advance); `materialized_*`
    rows are mutable views built from events. The append-only discipline is
    what makes replay deterministic and is the foundation of the Pattern C
    architecture (`docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md`).
```

- [ ] **Step 3: Update memory file**

Append to `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`:

```markdown
## Pattern C — Event-Sourced Architecture (2026-05-25)

Three-process split via shared SQLite event log:
- `services/ingest_worker.py`: WS + REST + marketcap → events (no inference)
- `services/model_service.py --model {v3,v4_5}`: poll events → inference → decisions
- `services/api_server.py`: FastAPI 8001, materialized views from events

Event store: `events` table (append-only) + `consumer_cursors` (per-consumer cursor).
7 event types: `price_tick`, `candle_close`, `marketcap_snapshot`, `signal_scored`,
`trade_decided`, `trade_closed`, `exit_triggered`. Plus `system_control` for trading on/off.

Migration: Phase 1 add events alongside monolith. Phase 2 model_service runs parallel.
Phase 3 cutover (set `MONOLITH_SCAN_DISABLED=true`). Phase 4 retire monolith main.py.

See:
- spec: docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md
- plan: docs/superpowers/plans/2026-05-25-event-sourced-architecture.md
```

- [ ] **Step 4: Stage**

```bash
git add CHANGELOG.md CLAUDE.md
# Memory is outside the repo; mention the update in the commit message instead.
```

---

## Task 27 (OPERATOR-GATED): Full pytest sweep + atomic commit + push

**This task runs only during an operator-sanctioned 8001 pause window.**

- [ ] **Step 1: Operator pauses the 8001 backend**

Operator action: stop the 8001 backend (`Stop-Process` on the python.exe owning the LISTEN on TCP 8001). Confirm:

```powershell
Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue
```

Expected: no rows.

- [ ] **Step 2: Verify staged changes are intact**

```bash
git status
git diff --stat HEAD
```

Expected: all new files from Tasks 1-26 listed under "Changes to be committed".

- [ ] **Step 3: Run the full pytest suite from the worktree**

```bash
cd C:/Users/gl450/AppData/Local/Temp/pattern-c/backend && ../.venv/Scripts/python.exe -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/pattern-c-pytest.log
```

Expected: all existing tests pass + all new Pattern C tests pass. Approx test count: ~1180 baseline + ~80 new = ~1260.

If any test fails, stop. Do not commit. Diagnose, fix, restage, re-run.

- [ ] **Step 4: Atomic commit (HEREDOC for clean message)**

```bash
cd C:/Users/gl450/AppData/Local/Temp/pattern-c && git commit -m "$(cat <<'EOF'
feat: Pattern C event-sourced architecture — 3-process split, append-only event log

Implements the spec at docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md.

New process boundaries:
  * services/ingest_worker.py  — WS + REST + marketcap → price_tick / candle_close
                                 / marketcap_snapshot events
  * services/model_service.py  — event-driven inference, one process per model
                                 (--model v3 or --model v4_5). Emits signal_scored,
                                 trade_decided, trade_closed, exit_triggered events.
  * services/api_server.py     — FastAPI port 8001, reads materialized views from
                                 events; WS bridge tails events for frontend push.

Substrate:
  * events table (append-only, SQLite WAL)
  * consumer_cursors table (per-consumer cursor, at-least-once delivery)
  * materialized_latest_price + materialized_positions_{v3,v4_5} (UPSERT views)

Tooling:
  * tools/event_inspector.py   — CLI query
  * tools/replay_consumer.py   — sandboxed event-stream replay for backtesting

Migration is parallel-stream: monolith main.py keeps running until Phase 3
cutover (set MONOLITH_SCAN_DISABLED=true). Phase 4 retires the monolith API
surface in favor of api_server.py on port 8001. No behavior change to the
deployed 8001 backend until the operator flips the gate.

New invariant: CLAUDE.md #19 — events table is append-only.

11 new test files (~80 tests). Full suite green.

Refs:
  - Spec: docs/superpowers/specs/2026-05-25-event-sourced-architecture-design.md
  - Decisions: docs/superpowers/specs/2026-05-25-event-sourced-architecture-decisions.md
  - Predecessor: docs/superpowers/specs/2026-05-24-live-ops-feedback-phase2-phase3.md

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 5: Push and verify**

```bash
git push origin feat/event-sourced-architecture
git log --oneline -3
```

Expected: new commit at HEAD of `feat/event-sourced-architecture`; remote tracking branch updated.

- [ ] **Step 6: Memory update (post-commit, before operator restart)**

Confirm `coinbase_trader_architecture.md` update from Task 26 was saved to `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\`. Per `feedback_memory_after_changes`, this happens in the same step as commit, not at end-of-session.

- [ ] **Step 7: Operator restart 8001 + report**

Operator action: restart the 8001 backend (`cd backend && python main.py`). No behavior change is expected vs pre-commit because `MONOLITH_SCAN_DISABLED` is not set; the new code paths are dormant until Phase 1 of the migration is executed by the operator. Report in chat:

> Pattern C code shipped on `feat/event-sourced-architecture` (commit `<sha>`).
> Pre-commit hook passed full suite (`<N>` passed, 0 failed).
> 8001 restarted, no behavior change. Phase 1 ready when operator chooses to launch `services/ingest_worker.py` alongside.

---

## Self-Review

**Spec coverage check** — every spec section is implemented:

| Spec section | Plan task(s) |
|---|---|
| Event store schema (`events`, `consumer_cursors`, `materialized_*`) | Tasks 1, 2 |
| Event type catalog (7 types) | Tasks 3, 4 |
| Consumer cursor management | Task 5 |
| `feature_snapshot` rebuild | Task 6 |
| `view_materializer` | Task 7 |
| Schema wiring into existing init_db | Task 8 |
| `ingest_worker.py` (WS / candle / marketcap / CLI) | Tasks 9-12 |
| `model_service.py` (cursor + score + decide + exit + CLI) | Tasks 13-17 |
| `api_server.py` (status / products / trades / compare / equity_curve / trading / WS / CLI) | Tasks 18-22 |
| `MONOLITH_SCAN_DISABLED` gate (Phase 3) | Task 23 |
| `event_inspector` + `replay_consumer` | Task 24 |
| Replay determinism + cross-check integration tests | Task 25 |
| CHANGELOG + invariant #19 + memory | Task 26 |
| Operator-paused commit + push | Task 27 |

**Placeholder scan** — no "TBD", "implement later", or "fill in". All TDD steps include concrete code. The one judgment-call deferral (real inference wiring in Task 17) is explicitly called out as operator-integration scope, with the default returning HOLD so the migration is safe.

**Type consistency check** — `ModelService._on_event`, `_on_candle_close`, `_on_price_tick`, `_decide_trade`, `_score_signal` names are consistent across Tasks 13-17. `EventConsumer.poll()` returns `List[Event]`. `_MODEL_NAME_RE` regex matches in `events_schema`, `model_service`, `view_materializer`.

**Constraint check** — Tasks 1-26 stage files only (no pytest, no commit), conforming to `feedback_no_pytest_during_trading`. Task 27 is the single operator-paused commit window. All code changes are additive until `MONOLITH_SCAN_DISABLED=true` is set; the deployed 8001 monolith continues to run unchanged through Phase 1.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-25-event-sourced-architecture.md`. Two execution options:

**1. Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Critical for a 27-task plan: one fresh subagent per task keeps context tight.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`, with operator checkpoints at task boundaries.

Either way, **Tasks 1-26 are file-write-only**. Task 27 is the single operator-paused commit gate. The 8001 backend must remain undisturbed until Task 27.
