# Maker Fill Shadow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure, without placing any order, how often a post-only BUY resting at the best bid would have filled within the 30 s maker window, and what the price did afterwards — so the maker-execution lever stops being an assumption.

**Architecture:** A pure, clock-injected state machine (`services/maker_shadow.py`) holds one virtual post-only intent per product, created right after every successful paper BUY from the WS bid/ask snapshot, and resolves it from the live ticker last-trade stream (the same `register_price_handler` hook `exit_watcher` uses). Resolved intents go to an additive `maker_shadow` table; a read-only report turns rows into fill rate, time-to-fill, spread saved and post-fill drift. Everything is behind `MAKER_SHADOW` (default off).

**Tech Stack:** Python 3.11 (`.venv`), aiosqlite, pytest (repo fixtures `db_module`, `run`), ruff 0.9.0 pinned.

**Spec:** this plan's Design section, motivated by `docs/superpowers/specs/2026-06-13-win-factors-improvement-loop-design.md` (Step 4: maker flips history −$414 → +$169 *if* fills happen) and `docs/specs/2026-09-27-strategy-evidence-and-decision.md` ("maker execution measured rather than assumed … unmeasurable today because no fills exist").

## Design

- **Intent.** On a successful paper BUY of `pid`, snapshot `bid`, `ask` from `ws_subscriber.state[pid]`. Limit = `bid`. Window = 30 s (matches `order_executor.execute_maker_signal`'s fill poll).
- **Fill rule — two bounds, both recorded.**
  - `filled` (conservative, the headline): some tick inside the window has last-trade `price < limit` (strictly traded THROUGH the level, so queue position cannot matter).
  - `touched` (optimistic upper bound): some tick inside the window has `price <= limit`.
- **Resolution.** An intent is finalised by the first tick (any product) at or after `created + window`. Fields: `status` (`filled` | `unfilled`), `touched`, `fill_ts`, `time_to_fill_s`, `last_price` (latest tick for that pid seen by finalisation), `drift_bps` = `(last_price − limit) / limit × 1e4`, `spread_bps` = `(ask − bid) / mid × 1e4`.
- **No-quote rows are recorded, not dropped** (`status = 'no_quote'`, reason in `detail`): missing bid/ask, `bid <= 0`, `ask < bid`. Dropping them would inflate the fill rate.
- **One open intent per pid.** A second BUY on a pid with an open intent records `status='duplicate'` and does not replace the first.
- **Never raises** into the WS receive loop (invariant #18) or the scan loop (invariant #14). Sink failures are logged and swallowed.
- **Does not** place, cancel or simulate orders in `order_executor`; does not change the paper book, `trades`, or any existing table.

## Global Constraints

- `MAKER_SHADOW` env flag, default `false`; flag off ⇒ no intent created, no handler work, no table rows.
- Additive schema only: `CREATE TABLE IF NOT EXISTS maker_shadow`; no `ALTER` of existing tables.
- No network calls from the new code; quotes come only from `ws_subscriber.state`.
- Run tests with `.venv/Scripts/python.exe -m pytest` from `backend/`.
- Pinned `ruff==0.9.0` `check` and `format --check` must pass on touched files.
- Paper-trading only system: no real funds; nothing here places an order.

## Review Focus

1. **Tick exactly at the limit** (`price == bid`): must set `touched` but NOT `filled`. → Task 1 test `test_touch_is_not_a_fill`.
2. **A product that never ticks again** must still finalise when another product ticks past the deadline. → Task 1 test `test_intent_finalised_by_other_products_tick`.
3. **Crossed / missing / zero quote** must produce a `no_quote` row, never an intent and never an exception. → Task 1 test `test_bad_quotes_recorded_as_no_quote`.
4. **Sink raising** (DB locked) must not propagate out of `on_tick`. → Task 1 test `test_sink_failure_is_swallowed`.
5. **Flag off** must be byte-for-byte unchanged: no `MakerShadow` method is called from the buy path. → Task 3 test `test_flag_off_never_touches_shadow`.

---

### Task 1: Pure maker-shadow state machine

**Files:**
- Create: `backend/services/maker_shadow.py`
- Test: `backend/tests/test_maker_shadow.py`

**Interfaces:**
- Produces:
  - `class MakerShadow(sink: Callable[[dict], Awaitable[None]], clock: Callable[[], float] = time.time, window_s: float = 30.0)`
  - `MakerShadow.register(pid: str, bid: float | None, ask: float | None) -> str` — returns `"open"`, `"no_quote"` or `"duplicate"`; non-open outcomes are queued for the sink immediately.
  - `async MakerShadow.on_tick(pid: str, price: float) -> None` — updates intents, finalises expired ones, flushes queued rows to `sink`; never raises.
  - `MakerShadow.open_count() -> int`
  - Row dict keys: `product_id, status, touched, limit_price, ask, spread_bps, created_ts, fill_ts, time_to_fill_s, last_price, drift_bps, window_s, detail`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for services.maker_shadow — paper post-only fill measurement."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.maker_shadow import MakerShadow


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def _make(window_s=30.0):
    rows = []

    async def sink(row):
        rows.append(row)

    clock = _Clock()
    return MakerShadow(sink=sink, clock=clock, window_s=window_s), clock, rows


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_trade_through_inside_window_is_filled():
    shadow, clock, rows = _make()
    assert shadow.register("ABC-USD", bid=10.0, ask=10.1) == "open"
    clock.t += 5
    _run(shadow.on_tick("ABC-USD", 9.99))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 10.05))
    assert len(rows) == 1
    r = rows[0]
    assert r["status"] == "filled" and r["touched"] is True
    assert r["time_to_fill_s"] == pytest.approx(5.0)
    assert r["last_price"] == pytest.approx(10.05)
    assert r["drift_bps"] == pytest.approx((10.05 - 10.0) / 10.0 * 1e4)
    assert r["spread_bps"] == pytest.approx(0.1 / 10.05 * 1e4)


def test_touch_is_not_a_fill():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 3
    _run(shadow.on_tick("ABC-USD", 10.0))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 10.02))
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["touched"] is True
    assert rows[0]["fill_ts"] is None


def test_trade_through_after_window_does_not_count():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 9.5))
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["touched"] is False


def test_intent_finalised_by_other_products_tick():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 40
    _run(shadow.on_tick("XYZ-USD", 1.0))
    assert len(rows) == 1 and rows[0]["product_id"] == "ABC-USD"
    assert rows[0]["status"] == "unfilled"
    assert rows[0]["last_price"] is None
    assert rows[0]["drift_bps"] is None
    assert shadow.open_count() == 0


@pytest.mark.parametrize(
    "bid,ask,detail",
    [(None, 10.1, "missing quote"), (10.0, None, "missing quote"),
     (0.0, 10.1, "non-positive bid"), (10.2, 10.1, "crossed quote")],
)
def test_bad_quotes_recorded_as_no_quote(bid, ask, detail):
    shadow, clock, rows = _make()
    assert shadow.register("ABC-USD", bid=bid, ask=ask) == "no_quote"
    _run(shadow.on_tick("ABC-USD", 10.0))
    assert rows[0]["status"] == "no_quote"
    assert rows[0]["detail"] == detail
    assert shadow.open_count() == 0


def test_second_buy_while_open_is_duplicate_and_keeps_first():
    shadow, clock, rows = _make()
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    assert shadow.register("ABC-USD", bid=11.0, ask=11.1) == "duplicate"
    clock.t += 1
    _run(shadow.on_tick("ABC-USD", 9.9))
    clock.t += 30
    _run(shadow.on_tick("ABC-USD", 9.9))
    statuses = sorted(r["status"] for r in rows)
    assert statuses == ["duplicate", "filled"]
    filled = next(r for r in rows if r["status"] == "filled")
    assert filled["limit_price"] == 10.0


def test_sink_failure_is_swallowed():
    async def bad_sink(row):
        raise RuntimeError("database is locked")

    clock = _Clock()
    shadow = MakerShadow(sink=bad_sink, clock=clock)
    shadow.register("ABC-USD", bid=10.0, ask=10.1)
    clock.t += 31
    _run(shadow.on_tick("ABC-USD", 10.0))  # must not raise
    assert shadow.open_count() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'services.maker_shadow'`

- [ ] **Step 3: Write the implementation**

```python
"""Paper maker-fill shadow — would a post-only BUY at the best bid have filled?

Measurement only. Places, cancels and simulates no orders; touches no existing
table. A virtual intent rests at the bid captured right after a paper BUY and is
resolved from the live last-trade stream:

* ``filled``  — a trade printed strictly BELOW the limit inside the window, so
  the level was traded through and queue position cannot matter (conservative).
* ``touched`` — a trade printed AT or below the limit (optimistic upper bound).

Rows that could not be measured are recorded as ``no_quote`` / ``duplicate``
rather than dropped, because dropping them would inflate the fill rate.
Never raises into the WS receive loop (invariant #18).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

Sink = Callable[[dict], Awaitable[None]]


@dataclass
class _Intent:
    pid: str
    limit: float
    ask: float
    created: float
    touched: bool = False
    fill_ts: Optional[float] = None
    last_price: Optional[float] = None


def _spread_bps(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid * 1e4


def _quote_problem(bid: Optional[float], ask: Optional[float]) -> Optional[str]:
    if bid is None or ask is None:
        return "missing quote"
    if bid <= 0:
        return "non-positive bid"
    if ask < bid:
        return "crossed quote"
    return None


class MakerShadow:
    def __init__(
        self,
        sink: Sink,
        clock: Callable[[], float] = time.time,
        window_s: float = 30.0,
    ) -> None:
        self._sink = sink
        self._clock = clock
        self._window = window_s
        self._open: Dict[str, _Intent] = {}
        self._pending: List[dict] = []

    def open_count(self) -> int:
        return len(self._open)

    def _row(self, pid: str, status: str, **fields) -> dict:
        base = {
            "product_id": pid,
            "status": status,
            "touched": False,
            "limit_price": None,
            "ask": None,
            "spread_bps": None,
            "created_ts": self._clock(),
            "fill_ts": None,
            "time_to_fill_s": None,
            "last_price": None,
            "drift_bps": None,
            "window_s": self._window,
            "detail": None,
        }
        base.update(fields)
        return base

    def register(self, pid: str, bid: Optional[float], ask: Optional[float]) -> str:
        problem = _quote_problem(bid, ask)
        if problem is not None:
            self._pending.append(self._row(pid, "no_quote", detail=problem))
            return "no_quote"
        if pid in self._open:
            self._pending.append(
                self._row(pid, "duplicate", limit_price=bid, ask=ask, detail="intent already open")
            )
            return "duplicate"
        self._open[pid] = _Intent(pid=pid, limit=float(bid), ask=float(ask), created=self._clock())
        return "open"

    def _finalise(self, it: _Intent) -> dict:
        filled = it.fill_ts is not None
        drift = None
        if filled and it.last_price is not None:
            drift = (it.last_price - it.limit) / it.limit * 1e4
        return self._row(
            it.pid,
            "filled" if filled else "unfilled",
            touched=it.touched,
            limit_price=it.limit,
            ask=it.ask,
            spread_bps=_spread_bps(it.limit, it.ask),
            created_ts=it.created,
            fill_ts=it.fill_ts,
            time_to_fill_s=(it.fill_ts - it.created) if filled else None,
            last_price=it.last_price,
            drift_bps=drift,
        )

    async def on_tick(self, pid: str, price: float) -> None:
        try:
            now = self._clock()
            it = self._open.get(pid)
            if it is not None and now < it.created + self._window:
                it.last_price = price
                if price <= it.limit:
                    it.touched = True
                if price < it.limit and it.fill_ts is None:
                    it.fill_ts = now
            elif it is not None:
                it.last_price = price
            for key in [k for k, v in self._open.items() if now >= v.created + self._window]:
                self._pending.append(self._finalise(self._open.pop(key)))
        except Exception:
            logger.exception("maker_shadow.on_tick failed (pid=%s price=%s)", pid, price)
        await self._flush()

    async def _flush(self) -> None:
        rows, self._pending = self._pending, []
        for row in rows:
            try:
                await self._sink(row)
            except Exception:
                logger.exception("maker_shadow sink failed for %s", row.get("product_id"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow.py -q`
Expected: all pass (10 tests incl. parametrised).

- [ ] **Step 5: Lint + commit**

```bash
.venv/Scripts/python.exe -m ruff check backend/services/maker_shadow.py backend/tests/test_maker_shadow.py
.venv/Scripts/python.exe -m ruff format --check backend/services/maker_shadow.py backend/tests/test_maker_shadow.py
git add backend/services/maker_shadow.py backend/tests/test_maker_shadow.py
git commit -m "feat: pure paper maker-fill shadow state machine"
```

---

### Task 2: Persistence — additive `maker_shadow` table

**Files:**
- Modify: `backend/database.py` (add table inside `init_db`, after `regime_state`; add two functions at end of file)
- Test: `backend/tests/test_database_maker_shadow.py`

**Interfaces:**
- Consumes: Task 1 row dict keys.
- Produces: `async save_maker_shadow(row: dict) -> None`, `async get_maker_shadow_rows(since_ts: float | None = None) -> list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def db(tmp_path):
    import database

    importlib.reload(database)
    database.DB_PATH = str(tmp_path / "t.db")
    asyncio.new_event_loop().run_until_complete(database.init_db())
    return database


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _row(**kw):
    base = {
        "product_id": "ABC-USD", "status": "filled", "touched": True,
        "limit_price": 10.0, "ask": 10.1, "spread_bps": 99.5, "created_ts": 1000.0,
        "fill_ts": 1005.0, "time_to_fill_s": 5.0, "last_price": 10.05,
        "drift_bps": 50.0, "window_s": 30.0, "detail": None,
    }
    base.update(kw)
    return base


def test_round_trip_preserves_every_field(db):
    _run(db.save_maker_shadow(_row()))
    rows = _run(db.get_maker_shadow_rows())
    assert len(rows) == 1
    r = rows[0]
    for k, v in _row().items():
        assert r[k] == v, k


def test_no_quote_row_with_nulls_round_trips(db):
    _run(db.save_maker_shadow(_row(status="no_quote", touched=False, limit_price=None,
                                   ask=None, spread_bps=None, fill_ts=None,
                                   time_to_fill_s=None, last_price=None, drift_bps=None,
                                   detail="missing quote")))
    r = _run(db.get_maker_shadow_rows())[0]
    assert r["status"] == "no_quote" and r["touched"] is False and r["limit_price"] is None


def test_since_filter(db):
    _run(db.save_maker_shadow(_row(created_ts=100.0)))
    _run(db.save_maker_shadow(_row(created_ts=200.0)))
    assert [r["created_ts"] for r in _run(db.get_maker_shadow_rows(since_ts=150.0))] == [200.0]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_database_maker_shadow.py -q`
Expected: FAIL `AttributeError: module 'database' has no attribute 'save_maker_shadow'`

- [ ] **Step 3: Implement**

In `init_db`, after the `regime_state` `executescript`, add:

```python
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS maker_shadow (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id      TEXT NOT NULL,
                status          TEXT NOT NULL,
                touched         INTEGER NOT NULL,
                limit_price     REAL,
                ask             REAL,
                spread_bps      REAL,
                created_ts      REAL NOT NULL,
                fill_ts         REAL,
                time_to_fill_s  REAL,
                last_price      REAL,
                drift_bps       REAL,
                window_s        REAL NOT NULL,
                detail          TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_maker_shadow_created ON maker_shadow(created_ts);
        """)
```

At end of `database.py`:

```python
_MAKER_SHADOW_COLS = (
    "product_id", "status", "touched", "limit_price", "ask", "spread_bps", "created_ts",
    "fill_ts", "time_to_fill_s", "last_price", "drift_bps", "window_s", "detail",
)


async def save_maker_shadow(row: Dict) -> None:
    """Persist one resolved maker-shadow intent (measurement only)."""
    values = [row[c] for c in _MAKER_SHADOW_COLS]
    values[2] = 1 if row["touched"] else 0
    async with _db() as db:
        await db.execute(
            f"INSERT INTO maker_shadow ({','.join(_MAKER_SHADOW_COLS)}) "
            f"VALUES ({','.join('?' * len(_MAKER_SHADOW_COLS))})",
            values,
        )
        await db.commit()


async def get_maker_shadow_rows(since_ts: Optional[float] = None) -> List[Dict]:
    sql = f"SELECT {','.join(_MAKER_SHADOW_COLS)} FROM maker_shadow"
    args: tuple = ()
    if since_ts is not None:
        sql += " WHERE created_ts >= ?"
        args = (since_ts,)
    sql += " ORDER BY created_ts, id"
    async with _db() as db:
        async with db.execute(sql, args) as cur:
            rows = await cur.fetchall()
    out = []
    for r in rows:
        d = dict(zip(_MAKER_SHADOW_COLS, r))
        d["touched"] = bool(d["touched"])
        out.append(d)
    return out
```

(Check the existing imports at the top of `database.py` include `Optional` and `List`; add them to the `typing` import if not.)

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_database_maker_shadow.py tests/test_database.py -q`
Expected: all pass.

- [ ] **Step 5: Lint + commit**

```bash
.venv/Scripts/python.exe -m ruff check backend/database.py backend/tests/test_database_maker_shadow.py
.venv/Scripts/python.exe -m ruff format --check backend/database.py backend/tests/test_database_maker_shadow.py
git add backend/database.py backend/tests/test_database_maker_shadow.py
git commit -m "feat: additive maker_shadow table and accessors"
```

---

### Task 3: Wiring — flag, buy-path hook, WS attach

**Files:**
- Modify: `backend/config.py` (after `use_maker_execution`)
- Modify: `backend/agents/cnn_agent.py` (`__init__` ~1665; buy path after `if spent > 0:` ~2356)
- Modify: `backend/main.py` (after `attach_exit_watcher(...)` ~465)
- Test: `backend/tests/test_maker_shadow_wiring.py`

**Interfaces:**
- Consumes: `MakerShadow.register`, `MakerShadow.on_tick`, `database.save_maker_shadow`.
- Produces: `config.maker_shadow: bool`; `CoinbaseCNNAgent.maker_shadow: Optional[MakerShadow]` (default `None`); `CoinbaseCNNAgent._shadow_register(pid: str) -> None`.

- [ ] **Step 1: Write the failing tests**

```python
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents import cnn_agent as ca


def _agent(state):
    ws = MagicMock()
    ws.state = state
    agent = ca.CoinbaseCNNAgent(ws_subscriber=ws)
    agent.maker_shadow = MagicMock()
    return agent


def test_flag_off_never_touches_shadow(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", False)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_not_called()


def test_flag_on_registers_ws_quote(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_called_once_with("ABC-USD", bid=10.0, ask=10.1)


def test_flag_on_missing_ws_state_registers_none(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({})
    agent._shadow_register("ABC-USD")
    agent.maker_shadow.register.assert_called_once_with("ABC-USD", bid=None, ask=None)


def test_shadow_exception_never_escapes(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    agent = _agent({"ABC-USD": {"bid": 10.0, "ask": 10.1}})
    agent.maker_shadow.register.side_effect = RuntimeError("boom")
    agent._shadow_register("ABC-USD")  # must not raise


def test_no_shadow_instance_is_noop(monkeypatch):
    monkeypatch.setattr(ca.config, "maker_shadow", True)
    ws = MagicMock()
    ws.state = {"ABC-USD": {"bid": 10.0, "ask": 10.1}}
    agent = ca.CoinbaseCNNAgent(ws_subscriber=ws)
    assert agent.maker_shadow is None
    agent._shadow_register("ABC-USD")  # must not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow_wiring.py -q`
Expected: FAIL `AttributeError: ... has no attribute 'maker_shadow'` / `_shadow_register`.

- [ ] **Step 3: Implement**

`config.py`, after the `use_maker_execution` field:

```python
    # Paper maker-fill SHADOW: after each paper BUY, record whether a post-only
    # BUY at the WS best bid would have filled within 30 s (services/maker_shadow).
    # Measurement only — places no orders. Default false → unchanged behaviour.
    maker_shadow: bool = field(
        default_factory=lambda: os.getenv("MAKER_SHADOW", "false").lower() == "true"
    )
```

`cnn_agent.py` `__init__`, after `self.book = _CNNBook()`:

```python
        self.maker_shadow = None  # services.maker_shadow.MakerShadow, set by main.py
```

`cnn_agent.py`, new method on `CoinbaseCNNAgent` (place just before `_execute_live_order`):

```python
    def _shadow_register(self, pid: str) -> None:
        """Record a paper maker-fill intent for pid. Measurement only; never raises."""
        if not config.maker_shadow or self.maker_shadow is None:
            return
        try:
            quote = (self.ws.state.get(pid) if self.ws else None) or {}
            self.maker_shadow.register(pid, bid=quote.get("bid"), ask=quote.get("ask"))
        except Exception:
            logger.exception("maker_shadow register failed for %s", pid)
```

`cnn_agent.py` buy path, first line inside `if spent > 0:`:

```python
                    self._shadow_register(pid)
```

`main.py`, after the `attach_exit_watcher(...)` call and its log line:

```python
    if config.maker_shadow:
        from services.maker_shadow import MakerShadow

        app_state.cnn_agent.maker_shadow = MakerShadow(sink=database.save_maker_shadow)
        app_state.ws_subscriber.register_price_handler(app_state.cnn_agent.maker_shadow.on_tick)
        logger.info("Maker fill shadow attached (measurement only, no orders)")
```

(Confirm `config` and `database` are already imported in `main.py`; they are used elsewhere in the lifespan.)

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow_wiring.py tests/test_maker_shadow.py tests/test_cnn_agent.py -q`
Expected: all pass.

- [ ] **Step 5: Lint + commit**

```bash
.venv/Scripts/python.exe -m ruff check backend/config.py backend/agents/cnn_agent.py backend/main.py backend/tests/test_maker_shadow_wiring.py
.venv/Scripts/python.exe -m ruff format --check backend/config.py backend/agents/cnn_agent.py backend/main.py backend/tests/test_maker_shadow_wiring.py
git add backend/config.py backend/agents/cnn_agent.py backend/main.py backend/tests/test_maker_shadow_wiring.py
git commit -m "feat: wire paper maker-fill shadow behind MAKER_SHADOW (default off)"
```

---

### Task 4: Read-only report + docs

**Files:**
- Create: `backend/tools/maker_shadow_report.py`
- Test: `backend/tests/test_maker_shadow_report.py`
- Modify: `CHANGELOG.md` (new session entry at top), `CLAUDE.md` (invariant: maker shadow is measurement-only, default off)

**Interfaces:**
- Consumes: row dicts from `database.get_maker_shadow_rows`.
- Produces: `summarize(rows: list[dict]) -> dict` with keys `n_total, n_measured, n_no_quote, n_duplicate, n_filled, n_touched, fill_rate, touch_rate, median_time_to_fill_s, median_spread_bps, median_drift_bps_filled`; CLI `python -m tools.maker_shadow_report [--since-hours H]` prints it as JSON.

- [ ] **Step 1: Write the failing test**

```python
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.maker_shadow_report import summarize


def _r(status, touched=False, ttf=None, spread=None, drift=None):
    return {"status": status, "touched": touched, "time_to_fill_s": ttf,
            "spread_bps": spread, "drift_bps": drift}


def test_rates_use_measured_denominator_only():
    rows = [_r("filled", True, 4.0, 20.0, -10.0), _r("unfilled", True, None, 30.0),
            _r("unfilled", False, None, 40.0), _r("no_quote"), _r("duplicate")]
    s = summarize(rows)
    assert s["n_total"] == 5 and s["n_measured"] == 3
    assert s["n_no_quote"] == 1 and s["n_duplicate"] == 1
    assert s["fill_rate"] == pytest.approx(1 / 3)
    assert s["touch_rate"] == pytest.approx(2 / 3)
    assert s["median_time_to_fill_s"] == 4.0
    assert s["median_spread_bps"] == 30.0
    assert s["median_drift_bps_filled"] == -10.0


def test_empty_is_none_not_zero():
    s = summarize([_r("no_quote")])
    assert s["n_measured"] == 0
    assert s["fill_rate"] is None and s["median_spread_bps"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow_report.py -q`
Expected: `ModuleNotFoundError: No module named 'tools.maker_shadow_report'`

- [ ] **Step 3: Implement**

```python
"""Read-only summary of the paper maker-fill shadow (services/maker_shadow).

Rates use MEASURED intents only (filled + unfilled). no_quote / duplicate rows
are counted and reported, never silently folded into either side.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _med(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def summarize(rows: List[Dict]) -> Dict:
    measured = [r for r in rows if r["status"] in ("filled", "unfilled")]
    filled = [r for r in measured if r["status"] == "filled"]
    n = len(measured)
    return {
        "n_total": len(rows),
        "n_measured": n,
        "n_no_quote": sum(r["status"] == "no_quote" for r in rows),
        "n_duplicate": sum(r["status"] == "duplicate" for r in rows),
        "n_filled": len(filled),
        "n_touched": sum(bool(r["touched"]) for r in measured),
        "fill_rate": len(filled) / n if n else None,
        "touch_rate": sum(bool(r["touched"]) for r in measured) / n if n else None,
        "median_time_to_fill_s": _med([r["time_to_fill_s"] for r in filled]),
        "median_spread_bps": _med([r["spread_bps"] for r in measured]),
        "median_drift_bps_filled": _med([r["drift_bps"] for r in filled]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since-hours", type=float, default=None)
    args = ap.parse_args()
    import database

    since = time.time() - args.since_hours * 3600 if args.since_hours else None
    rows = asyncio.run(database.get_maker_shadow_rows(since_ts=since))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_maker_shadow_report.py -q`
Expected: pass.

- [ ] **Step 5: Docs.** Add a CHANGELOG session entry summarising Tasks 1–4 (flag, fill rule with both bounds, no orders, how to run: `MAKER_SHADOW=true` on the 8002 dev backend per port discipline, then `python -m tools.maker_shadow_report --since-hours 24`). Add a CLAUDE.md invariant: "The maker shadow is measurement-only: it must never place, cancel or simulate orders in `order_executor`, never write existing tables, and stays default-off (`MAKER_SHADOW`)."

- [ ] **Step 6: Lint + commit + push**

```bash
.venv/Scripts/python.exe -m ruff check backend/tools/maker_shadow_report.py backend/tests/test_maker_shadow_report.py
.venv/Scripts/python.exe -m ruff format --check backend/tools/maker_shadow_report.py backend/tests/test_maker_shadow_report.py
git add backend/tools/maker_shadow_report.py backend/tests/test_maker_shadow_report.py CHANGELOG.md CLAUDE.md
git commit -m "feat: maker shadow report + docs"
git push -u origin feat/maker-fill-shadow
```

## Known limitations (state them in the CHANGELOG)

- The ticker channel reports last trade per update, not every print; a brief trade-through between updates can be missed → `filled` is a lower bound, `touched` an upper bound.
- `ws_subscriber.state` carries no quote timestamp; a stale bid cannot be detected here.
- Measures the ENTRY leg only. Exit-leg maker fills (trail/model-down exits) are a follow-up.
- Fill is necessary, not sufficient: `drift_bps` on filled intents is the adverse-selection signal and must be read alongside the fill rate.
