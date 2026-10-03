# Market Data Recorder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone, read-only recorder that captures as-of market data the trading app never
stored: Coinbase BTC/ETH order book, trades and heartbeats, plus derivatives funding, open
interest, basis, options and daily catalogue snapshots. Every record says when we received it
and whether it was valid.

**Architecture:** Package `backend/tools/recorder/`, run as its own process
(`python -m tools.recorder.run`). It imports nothing from the app (`agents`, `services`,
`database`, `config`, `clients`, `main`) and never opens `coinbase.db`. Output is append-only
gzip JSONL under `C:\Users\gl450\market_recorder_data\<stream>\<UTC-day>.jsonl.gz`, outside the
repo. Raw payloads are stored verbatim. Parsing and feature building happen later, offline, so
the recorder can never bake a parsing mistake into history.

**Tech Stack:** Python 3.11, asyncio, `websockets` 13.1, `httpx`, stdlib `gzip`/`json`/`hashlib`,
pytest.

**Spec:** the Design section below. It comes from the Claude/Codex indicator comparison
(`C:\Users\gl450\analysis_archive\indicator_research_2026-10-03\comparison.md`, Codex
"Priority 0" + "Proposed start sequence") and from source probes run 2026-10-03.

## Design (spec)

**Sources, fixed after reachability probes from this machine:**

| Stream | Source | Cadence | Probe result |
|---|---|---|---|
| `coinbase_ws/l2_data`, `coinbase_ws/market_trades`, `coinbase_ws/heartbeats` | `wss://advanced-trade-ws.coinbase.com`, no auth, BTC-USD + ETH-USD | streaming | OK; ~0.36 GB/day gz steady-state |
| `poll/okx_funding_*`, `poll/okx_oi_*`, `poll/okx_mark_*`, `poll/okx_index_*` | OKX public v5, BTC/ETH-USDT-SWAP | 5 min | HTTP 200 |
| `poll/intx_quote_*` | Coinbase International `/instruments/{BTC,ETH}-PERP/quote` | 5 min | HTTP 200 |
| `poll/deribit_futures_*` | Deribit `get_book_summary_by_currency kind=future` | 5 min | HTTP 200 |
| `poll/deribit_options_*` | the same endpoint, `kind=option` (implied-volatility surface) | 15 min | HTTP 200 |
| `poll/coinbase_spot_catalogue`, `poll/coinbase_futures_catalogue` | Coinbase `/market/products` | daily at 00:00 UTC, plus once at start | HTTP 200 |

Binance futures (HTTP 451) and Bybit (HTTP 403) are blocked from this location. They are
deliberately not used, and no geo-restriction is routed around.

**Record envelope** (one JSON object per line):

```
{"received_at_ns", "source", "kind": "message|poll|event", "status", "error", "meta", "payload", "schema": 1}
```

- `received_at_ns` is our receive time, from `time.time_ns()`. The exchange's own timestamps stay
  inside the verbatim `payload`.
- A failed poll is recorded with `error` and/or a non-200 `status`. It is NEVER converted to 0,
  1, 50 or any neutral value. This is the exact defect found in `services/macro_signals.py`.

**Continuity events** go to `coinbase_ws/events` and `recorder/events`:
- `connect`;
- `disconnect` (with the error);
- `stale`: no message for 30 s, which forces a reconnect;
- `gap`: Coinbase per-connection `sequence_num` not equal to previous + 1;
- `raw_paused_low_disk` and `raw_resumed`;
- `start` and `stop`.

A reconnect re-subscribes, and Coinbase then sends a fresh L2 snapshot. Books are rebuilt
offline, and any gap invalidates a book until the next snapshot.

**Files:**
- One file per stream per UTC day, append-only. A restart on the same day appends a new gzip
  member.
- Completed days get a `<file>.sha256` sidecar: at rollover, and at startup for any past day
  missing one.

**Disk guard:** below 100 GB free, raw WS capture pauses (with an event). Heartbeats and all
pollers continue, and capture resumes automatically above the threshold.

**Health:** `status.json` is rewritten atomically every 60 s with:
- pid and free disk;
- whether raw capture is enabled;
- per stream: count, last receive time and age in seconds.

**Out of scope (YAGNI):**
- minute summaries and book reconstruction (derived offline from raw later);
- autostart at logon (an operator decision);
- any consumer inside the trading app.

## Global Constraints

- **Standalone:** no imports from `agents`, `services`, `database`, `config`, `clients` or
  `main`; never touch `coinbase.db`, ports 8001/8002 or the bot. Enforced by a test.
- **Public, unauthenticated, read-only endpoints only.** No credentials, no `.env`.
- **Output** goes to `C:\Users\gl450\market_recorder_data` (override with `--out`), never inside
  the repo.
- **Errors are data:** record them; never substitute values.
- **8001 is live paper trading:** the pre-commit hook runs the full suite (~8 min). Commit once
  per green package, as the operator previously agreed. Pinned ruff 0.9.0 for check/format.

## Review Focus

1. **UTC midnight rollover mid-stream:** records after 00:00 go to the next day's file and the
   finished day gets its sha256 → `test_rollover_seals_previous_day` (Task 1).
2. **Coinbase sequence gap:** must produce a `gap` event, never pass silently →
   `test_ws_records_messages_and_gap_events` (Task 2).
3. **Silent socket** (connected, but no data): must record `stale` and reconnect →
   `test_ws_silence_records_stale_and_reconnects` (Task 2).
4. **HTTP error or non-200:** must be recorded with status/error and no fabricated value →
   `test_poll_once_records_errors_not_values` (Task 3).
5. **Low disk:** raw L2/trades pause with an event while heartbeats and polls continue →
   `test_raw_paused_still_records_heartbeats` (Task 2), `test_disk_guard_transitions` (Task 4).

---

### Task 1: Store and envelope

**Files:**
- Create: `backend/tools/recorder/__init__.py` (empty)
- Create: `backend/tools/recorder/store.py`
- Create: `backend/tests/tools/recorder/__init__.py` (empty)
- Test: `backend/tests/tools/recorder/test_store.py`

**Interfaces:**
- Produces:
  - `utc_day(ns: int) -> str`;
  - `envelope(source, kind, received_at_ns, *, payload=None, status=None, error=None, meta=None) -> dict`;
  - `DailyStore(root)`, with `.write(stream, record)`, `.flush_all()`, `.close()`,
    `.raw_enabled: bool` and `.stats: dict[str, {"count", "last_received_ns"}]`;
  - `seal_past_days(root, today: str) -> int`;
  - `read_records(path) -> list[dict]` (test/offline helper).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/recorder/test_store.py
import gzip

from tools.recorder.store import DailyStore, envelope, read_records, seal_past_days, utc_day

DAY_NS = 86_400 * 10**9
T0 = 1_791_000_000 * 10**9  # 2026-10-03 UTC


def test_utc_day():
    assert utc_day(T0) == "2026-10-03"


def test_envelope_shape():
    e = envelope("okx", "poll", T0, payload="{}", status=200, meta={"url": "u"})
    assert e == {"received_at_ns": T0, "source": "okx", "kind": "poll", "status": 200,
                 "error": None, "meta": {"url": "u"}, "payload": "{}", "schema": 1}


def test_write_and_read_back_with_stats(tmp_path):
    s = DailyStore(tmp_path)
    s.write("poll/x", envelope("okx", "poll", T0, payload="a"))
    s.write("poll/x", envelope("okx", "poll", T0 + 1, payload="b"))
    s.close()
    recs = read_records(tmp_path / "poll" / "x" / "2026-10-03.jsonl.gz")
    assert [r["payload"] for r in recs] == ["a", "b"]
    assert s.stats["poll/x"] == {"count": 2, "last_received_ns": T0 + 1}


def test_rollover_seals_previous_day(tmp_path):
    s = DailyStore(tmp_path)
    s.write("ws/l2", envelope("cb", "message", T0, payload="day1"))
    s.write("ws/l2", envelope("cb", "message", T0 + DAY_NS, payload="day2"))
    s.close()
    d1 = tmp_path / "ws" / "l2" / "2026-10-03.jsonl.gz"
    assert (d1.parent / "2026-10-03.jsonl.gz.sha256").exists()
    assert not (d1.parent / "2026-10-04.jsonl.gz.sha256").exists()  # close() never seals today
    assert [r["payload"] for r in read_records(d1.parent / "2026-10-04.jsonl.gz")] == ["day2"]


def test_restart_same_day_appends_new_member(tmp_path):
    for p in ("first", "second"):
        s = DailyStore(tmp_path)
        s.write("poll/x", envelope("okx", "poll", T0, payload=p))
        s.close()
    path = tmp_path / "poll" / "x" / "2026-10-03.jsonl.gz"
    assert [r["payload"] for r in read_records(path)] == ["first", "second"]
    with gzip.open(path, "rt") as f:
        assert len(f.read().splitlines()) == 2


def test_seal_past_days_only_seals_unsealed_past_files(tmp_path):
    s = DailyStore(tmp_path)
    s.write("poll/x", envelope("okx", "poll", T0, payload="old"))
    s.write("poll/y", envelope("okx", "poll", T0 + DAY_NS, payload="today"))
    s.close()
    assert seal_past_days(tmp_path, "2026-10-04") == 1
    assert (tmp_path / "poll" / "x" / "2026-10-03.jsonl.gz.sha256").exists()
    assert seal_past_days(tmp_path, "2026-10-04") == 0
```

- [ ] **Step 2: Run to verify failure**

Run (from `backend/`): `../.venv/Scripts/python.exe -m pytest tests/tools/recorder -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.recorder'`

- [ ] **Step 3: Implement**

```python
# backend/tools/recorder/store.py
"""Append-only daily gzip JSONL store: one file per stream per UTC day.

Completed days get a sha256 sidecar (at rollover, or at startup via seal_past_days). close()
never seals the current day, because a restart on the same day appends another gzip member.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utc_day(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def envelope(source: str, kind: str, received_at_ns: int, *, payload: Optional[str] = None,
             status: Optional[int] = None, error: Optional[str] = None,
             meta: Optional[dict] = None) -> Dict[str, Any]:
    return {"received_at_ns": received_at_ns, "source": source, "kind": kind, "status": status,
            "error": error, "meta": meta or {}, "payload": payload, "schema": 1}


def _seal(path: Path) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(digest + "\n")


def seal_past_days(root: Path, today: str) -> int:
    sealed = 0
    for f in Path(root).rglob("*.jsonl.gz"):
        if f.name[:10] < today and not f.with_name(f.name + ".sha256").exists():
            _seal(f)
            sealed += 1
    return sealed


def read_records(path: Path) -> list:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class DailyStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.raw_enabled = True
        self.stats: Dict[str, Dict[str, int]] = {}
        self._open: Dict[str, tuple] = {}

    def write(self, stream: str, record: dict) -> None:
        day = utc_day(record["received_at_ns"])
        cur = self._open.get(stream)
        if cur is not None and cur[0] != day:
            self._close_one(stream, seal=True)
            cur = None
        if cur is None:
            path = self.root / stream / f"{day}.jsonl.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            cur = (day, path, gzip.open(path, "at", encoding="utf-8"))
            self._open[stream] = cur
        cur[2].write(json.dumps(record, separators=(",", ":")) + "\n")
        st = self.stats.setdefault(stream, {"count": 0, "last_received_ns": 0})
        st["count"] += 1
        st["last_received_ns"] = record["received_at_ns"]

    def flush_all(self) -> None:
        for _, _, handle in self._open.values():
            handle.flush()

    def _close_one(self, stream: str, seal: bool) -> None:
        _, path, handle = self._open.pop(stream)
        handle.close()
        if seal:
            _seal(path)

    def close(self) -> None:
        for stream in list(self._open):
            self._close_one(stream, seal=False)
```

- [ ] **Step 4: Run to verify pass.** Expected: 6 passed.

### Task 2: Coinbase WebSocket capture

**Files:**
- Create: `backend/tools/recorder/coinbase_ws.py`
- Test: `backend/tests/tools/recorder/test_coinbase_ws.py`

**Interfaces:**
- Consumes: `DailyStore`, `envelope`, `read_records`.
- Produces:
  - `SeqTracker().observe(seq) -> tuple[int, int] | None`;
  - `backoff_s(attempt) -> float`;
  - `async run_coinbase_ws(store, products, stop, *, connect=None, clock=time.time_ns, max_silence_s=30.0, url=WS_URL, sleep=asyncio.sleep)`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/recorder/test_coinbase_ws.py
import asyncio
import json

from tools.recorder.coinbase_ws import SeqTracker, backoff_s, run_coinbase_ws
from tools.recorder.store import DailyStore, read_records

T0 = 1_791_000_000 * 10**9


def _msg(channel, seq):
    return json.dumps({"channel": channel, "sequence_num": seq, "events": []})


class FakeWS:
    def __init__(self, msgs):
        self.msgs, self.sent = list(msgs), []

    async def send(self, m):
        self.sent.append(json.loads(m))

    async def recv(self):
        if self.msgs:
            return self.msgs.pop(0)
        await asyncio.sleep(3600)  # silence


class FakeConnect:
    def __init__(self, sockets):
        self.sockets, self.calls = list(sockets), 0

    def __call__(self, url):
        self.calls += 1
        ws = self.sockets.pop(0) if self.sockets else FakeWS([])

        class CM:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *a):
                return False

        return CM()


def _clock():
    t = [T0]

    def tick():
        t[0] += 1000
        return t[0]

    return tick


def _run(store, connect, stop_after_sleeps=1):
    stop = asyncio.Event()
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
        if len(sleeps) >= stop_after_sleeps:
            stop.set()

    asyncio.run(run_coinbase_ws(store, ["BTC-USD"], stop, connect=connect, clock=_clock(),
                                max_silence_s=0.05, sleep=sleep))
    return sleeps


def _stream(tmp_path, name):
    return read_records(tmp_path / "coinbase_ws" / name / "2026-10-03.jsonl.gz")


def test_seq_tracker_reports_gaps_only():
    t = SeqTracker()
    assert [t.observe(s) for s in (0, 1, 2, 5, 6)] == [None, None, None, (3, 5), None]


def test_backoff_caps_at_60():
    assert [backoff_s(a) for a in (0, 1, 3, 10)] == [1.0, 2.0, 8.0, 60.0]


def test_ws_records_messages_and_gap_events(tmp_path):
    store = DailyStore(tmp_path)
    ws = FakeWS([_msg("l2_data", 0), _msg("market_trades", 1), _msg("l2_data", 3)])
    _run(store, FakeConnect([ws]))
    store.close()
    assert [json.loads(r["payload"])["sequence_num"] for r in _stream(tmp_path, "l2_data")] == [0, 3]
    events = [r["meta"] for r in _stream(tmp_path, "events")]
    assert {"event": "gap", "expected": 2, "got": 3} in events
    assert {s["channel"] for s in ws.sent} == {"level2", "market_trades", "heartbeats"}


def test_ws_silence_records_stale_and_reconnects(tmp_path):
    store = DailyStore(tmp_path)
    connect = FakeConnect([FakeWS([]), FakeWS([])])
    sleeps = _run(store, connect, stop_after_sleeps=2)
    store.close()
    names = [r["meta"].get("event") for r in _stream(tmp_path, "events")]
    assert names.count("stale") == 2 and names.count("connect") == 2
    assert connect.calls == 2 and sleeps == [1.0, 1.0]  # stale resets backoff: clean connects


def test_ws_disconnect_error_is_recorded_with_backoff(tmp_path):
    store = DailyStore(tmp_path)

    def boom(url):
        raise OSError("network down")

    sleeps = _run(store, boom, stop_after_sleeps=2)
    store.close()
    recs = _stream(tmp_path, "events")
    assert recs[0]["meta"]["event"] == "disconnect" and "network down" in recs[0]["error"]
    assert sleeps == [1.0, 2.0]


def test_raw_paused_still_records_heartbeats(tmp_path):
    store = DailyStore(tmp_path)
    store.raw_enabled = False
    _run(store, FakeConnect([FakeWS([_msg("l2_data", 0), _msg("heartbeats", 1)])]))
    store.close()
    assert len(_stream(tmp_path, "heartbeats")) == 1
    assert not (tmp_path / "coinbase_ws" / "l2_data").exists()
```

- [ ] **Step 2: Run to verify failure.** `pytest tests/tools/recorder/test_coinbase_ws.py -q` →
  `ModuleNotFoundError ... coinbase_ws`

- [ ] **Step 3: Implement**

```python
# backend/tools/recorder/coinbase_ws.py
"""Coinbase Advanced Trade public WebSocket capture (no auth). Payloads are stored verbatim;
continuity (connect / disconnect / stale / sequence gap) is recorded as events, never inferred."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, Tuple

from tools.recorder.store import envelope

WS_URL = "wss://advanced-trade-ws.coinbase.com"
CHANNELS = ("level2", "market_trades", "heartbeats")
SOURCE = "coinbase_ws"
EVENTS = f"{SOURCE}/events"


class SeqTracker:
    """Coinbase sequence_num is per connection; create a new tracker for each connection."""

    def __init__(self) -> None:
        self.last: Optional[int] = None

    def observe(self, seq: int) -> Optional[Tuple[int, int]]:
        gap = None
        if self.last is not None and seq != self.last + 1:
            gap = (self.last + 1, seq)
        self.last = seq
        return gap


def backoff_s(attempt: int) -> float:
    return float(min(60, 2**attempt))


def _default_connect(url: str):
    import websockets

    return websockets.connect(url, max_size=None, ping_interval=20)


async def run_coinbase_ws(store, products, stop: asyncio.Event, *, connect=None,
                          clock=time.time_ns, max_silence_s: float = 30.0, url: str = WS_URL,
                          sleep=asyncio.sleep) -> None:
    connect = connect or _default_connect
    attempt = 0
    while not stop.is_set():
        tracker = SeqTracker()
        try:
            async with connect(url) as ws:
                store.write(EVENTS, envelope(SOURCE, "event", clock(),
                                             meta={"event": "connect", "products": list(products)}))
                for ch in CHANNELS:
                    await ws.send(json.dumps({"type": "subscribe", "product_ids": list(products),
                                              "channel": ch}))
                attempt = 0
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=max_silence_s)
                    except asyncio.TimeoutError:
                        store.write(EVENTS, envelope(SOURCE, "event", clock(), meta={
                            "event": "stale", "silence_s": max_silence_s}))
                        break
                    received = clock()
                    try:
                        head = json.loads(msg)
                        channel, seq = str(head.get("channel", "unknown")), head.get("sequence_num")
                    except ValueError:
                        channel, seq = "unparsed", None
                    if isinstance(seq, int):
                        gap = tracker.observe(seq)
                        if gap:
                            store.write(EVENTS, envelope(SOURCE, "event", received, meta={
                                "event": "gap", "expected": gap[0], "got": gap[1]}))
                    if store.raw_enabled or channel == "heartbeats":
                        store.write(f"{SOURCE}/{channel}",
                                    envelope(SOURCE, "message", received, payload=msg))
        except Exception as exc:
            store.write(EVENTS, envelope(SOURCE, "event", clock(), error=repr(exc),
                                         meta={"event": "disconnect"}))
        if stop.is_set():
            break
        await sleep(backoff_s(attempt))
        attempt += 1
```

Note on `test_ws_silence_records_stale_and_reconnects`: each stale socket did connect, so
`attempt` resets to 0 before the sleep; both sleeps are therefore `backoff_s(0) = 1.0`. With a
failing `connect`, `attempt` keeps growing: sleeps of 1.0, then 2.0.

- [ ] **Step 4: Run to verify pass.** Expected: 6 passed.

### Task 3: REST pollers

**Files:**
- Create: `backend/tools/recorder/pollers.py`
- Test: `backend/tests/tools/recorder/test_pollers.py`

**Interfaces:**
- Consumes: `DailyStore`, `envelope`.
- Produces:
  - `Poll(name, url, interval_s)` and `default_polls() -> tuple[Poll, ...]`;
  - `next_due(now_s, interval_s) -> float`;
  - `async poll_once(poll, store, http_get, clock)`;
  - `async run_poller(poll, store, http_get, stop, *, clock=time.time_ns, wall=time.time)`;
  - `make_http_get(client)`, where `http_get(url)` returns `(status, text)`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/recorder/test_pollers.py
import asyncio

from tools.recorder.pollers import Poll, default_polls, next_due, poll_once, run_poller
from tools.recorder.store import DailyStore, read_records

T0 = 1_791_000_000 * 10**9
P = Poll("okx_funding_BTC", "https://example/f", 300)


def _recs(tmp_path, name="okx_funding_BTC"):
    return read_records(tmp_path / "poll" / name / "2026-10-03.jsonl.gz")


def test_default_polls_cover_every_source_and_both_coins():
    names = {p.name for p in default_polls()}
    for coin in ("BTC", "ETH"):
        for kind in ("okx_funding", "okx_oi", "okx_mark", "okx_index", "intx_quote",
                     "deribit_futures", "deribit_options"):
            assert f"{kind}_{coin}" in names
    assert {"coinbase_spot_catalogue", "coinbase_futures_catalogue"} <= names
    intervals = {p.name: p.interval_s for p in default_polls()}
    assert intervals["deribit_options_BTC"] == 900 and intervals["coinbase_spot_catalogue"] == 86400
    assert all(p.url.startswith("https://") for p in default_polls())


def test_next_due_aligns_to_wall_clock_boundaries():
    assert next_due(1000.0, 300) == 1200.0
    assert next_due(1200.0, 300) == 1500.0  # exactly on a boundary -> the next one


def test_poll_once_records_body_and_status(tmp_path):
    store = DailyStore(tmp_path)

    async def get(url):
        return 200, '{"data":[1]}'

    asyncio.run(poll_once(P, store, get, lambda: T0))
    store.close()
    (r,) = _recs(tmp_path)
    assert (r["status"], r["payload"], r["error"], r["meta"]["url"]) == (
        200, '{"data":[1]}', None, "https://example/f")


def test_poll_once_records_errors_not_values(tmp_path):
    store = DailyStore(tmp_path)

    async def get_451(url):
        return 451, "restricted location"

    async def get_raise(url):
        raise TimeoutError("read timeout")

    asyncio.run(poll_once(P, store, get_451, lambda: T0))
    asyncio.run(poll_once(P, store, get_raise, lambda: T0 + 1))
    store.close()
    a, b = _recs(tmp_path)
    assert (a["status"], a["payload"]) == (451, "restricted location")
    assert b["status"] is None and b["payload"] is None and "read timeout" in b["error"]


def test_run_poller_polls_immediately_and_stops(tmp_path):
    store = DailyStore(tmp_path)
    calls = []

    async def get(url):
        calls.append(url)
        stop.set()
        return 200, "{}"

    stop = asyncio.Event()

    async def go():
        await asyncio.wait_for(run_poller(P, store, get, stop, clock=lambda: T0), timeout=2)

    asyncio.run(go())
    store.close()
    assert calls == ["https://example/f"] and len(_recs(tmp_path)) == 1
```

- [ ] **Step 2: Run to verify failure.** `pytest tests/tools/recorder/test_pollers.py -q` →
  `ModuleNotFoundError ... pollers`

- [ ] **Step 3: Implement**

```python
# backend/tools/recorder/pollers.py
"""Public REST pollers. A failed or non-200 poll is recorded as such; nothing is substituted."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass

from tools.recorder.store import envelope

OKX = "https://www.okx.com"
DERIBIT = "https://www.deribit.com/api/v2/public"
INTX = "https://api.international.coinbase.com/api/v1"
COINBASE = "https://api.coinbase.com/api/v3/brokerage/market"


@dataclass(frozen=True)
class Poll:
    name: str
    url: str
    interval_s: int


def default_polls() -> tuple:
    polls = []
    for coin in ("BTC", "ETH"):
        swap = f"{coin}-USDT-SWAP"
        polls += [
            Poll(f"okx_funding_{coin}", f"{OKX}/api/v5/public/funding-rate?instId={swap}", 300),
            Poll(f"okx_oi_{coin}",
                 f"{OKX}/api/v5/public/open-interest?instType=SWAP&instId={swap}", 300),
            Poll(f"okx_mark_{coin}",
                 f"{OKX}/api/v5/public/mark-price?instType=SWAP&instId={swap}", 300),
            Poll(f"okx_index_{coin}", f"{OKX}/api/v5/market/index-tickers?instId={coin}-USDT", 300),
            Poll(f"intx_quote_{coin}", f"{INTX}/instruments/{coin}-PERP/quote", 300),
            Poll(f"deribit_futures_{coin}",
                 f"{DERIBIT}/get_book_summary_by_currency?currency={coin}&kind=future", 300),
            Poll(f"deribit_options_{coin}",
                 f"{DERIBIT}/get_book_summary_by_currency?currency={coin}&kind=option", 900),
        ]
    polls += [
        Poll("coinbase_spot_catalogue", f"{COINBASE}/products?limit=5000", 86400),
        Poll("coinbase_futures_catalogue", f"{COINBASE}/products?limit=5000&product_type=FUTURE",
             86400),
    ]
    return tuple(polls)


def next_due(now_s: float, interval_s: int) -> float:
    return (math.floor(now_s / interval_s) + 1) * interval_s


async def poll_once(poll: Poll, store, http_get, clock) -> None:
    started = clock()
    try:
        status, text = await http_get(poll.url)
        rec = envelope("poll", poll.name, started, payload=text, status=status,
                       meta={"url": poll.url})
    except Exception as exc:
        rec = envelope("poll", poll.name, started, error=repr(exc), meta={"url": poll.url})
    store.write(f"poll/{poll.name}", rec)


async def run_poller(poll: Poll, store, http_get, stop: asyncio.Event, *, clock=time.time_ns,
                     wall=time.time) -> None:
    await poll_once(poll, store, http_get, clock)
    while not stop.is_set():
        delay = max(0.0, next_due(wall(), poll.interval_s) - wall())
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            await poll_once(poll, store, http_get, clock)


def make_http_get(client):
    async def get(url: str):
        resp = await client.get(url, timeout=20)
        return resp.status_code, resp.text

    return get
```

- [ ] **Step 4: Run to verify pass.** Expected: 5 passed.

### Task 4: Health, disk guard, entry point, standalone check

**Files:**
- Create: `backend/tools/recorder/health.py`
- Create: `backend/tools/recorder/run.py`
- Test: `backend/tests/tools/recorder/test_health.py`
- Test: `backend/tests/tools/recorder/test_standalone.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `MIN_FREE_BYTES`;
  - `disk_ok(free, min_free) -> bool`;
  - `apply_disk_guard(store, free, clock, min_free)`;
  - `status_snapshot(store, now_ns, free, pid) -> dict`;
  - `async run_health(store, out, stop, *, clock, free, interval_s, pid)`;
  - `run.main(out, products)` and the `run.cli()` command line.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/recorder/test_health.py
import asyncio
import json

from tools.recorder.health import (apply_disk_guard, disk_ok, run_health, status_snapshot)
from tools.recorder.store import DailyStore, envelope, read_records

T0 = 1_791_000_000 * 10**9
GB = 1024**3


def test_disk_ok_threshold():
    assert disk_ok(100 * GB, 100 * GB) and not disk_ok(100 * GB - 1, 100 * GB)


def test_disk_guard_transitions(tmp_path):
    store = DailyStore(tmp_path)
    apply_disk_guard(store, 50 * GB, lambda: T0, 100 * GB)
    apply_disk_guard(store, 50 * GB, lambda: T0 + 1, 100 * GB)  # no duplicate event
    assert store.raw_enabled is False
    apply_disk_guard(store, 200 * GB, lambda: T0 + 2, 100 * GB)
    assert store.raw_enabled is True
    store.close()
    ev = [r["meta"]["event"] for r in read_records(tmp_path / "recorder" / "events" / "2026-10-03.jsonl.gz")]
    assert ev == ["raw_paused_low_disk", "raw_resumed"]


def test_status_snapshot_reports_age_per_stream(tmp_path):
    store = DailyStore(tmp_path)
    store.write("poll/x", envelope("okx", "poll", T0))
    snap = status_snapshot(store, T0 + 5 * 10**9, 123, 42)
    store.close()
    assert snap["pid"] == 42 and snap["free_bytes"] == 123 and snap["raw_enabled"] is True
    assert snap["streams"]["poll/x"] == {"count": 1, "last_received_ns": T0, "age_s": 5.0}


def test_run_health_writes_status_atomically(tmp_path):
    store = DailyStore(tmp_path)
    stop = asyncio.Event()

    def free(path):
        stop.set()
        return 500 * GB

    asyncio.run(run_health(store, tmp_path, stop, clock=lambda: T0, free=free, interval_s=0.01,
                           pid=7))
    store.close()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["pid"] == 7 and not (tmp_path / "status.json.tmp").exists()
```

```python
# backend/tests/tools/recorder/test_standalone.py
"""The recorder must never import the trading app: it runs beside it, not inside it."""
import ast
from pathlib import Path

PKG = Path(__file__).resolve().parents[3] / "tools" / "recorder"
FORBIDDEN = {"agents", "services", "database", "config", "clients", "main"}


def test_recorder_imports_nothing_from_the_app():
    files = sorted(PKG.glob("*.py"))
    assert len(files) >= 5
    for py in files:
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN, f"{py.name} imports {n}"
```

- [ ] **Step 2: Run to verify failure.** `pytest tests/tools/recorder -q` → `ModuleNotFoundError ... health`

- [ ] **Step 3: Implement**

```python
# backend/tools/recorder/health.py
"""Health file + disk guard. Low disk pauses raw WS capture (heartbeats and polls continue)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from tools.recorder.store import envelope

MIN_FREE_BYTES = 100 * 1024**3


def disk_ok(free_bytes: int, min_free: int = MIN_FREE_BYTES) -> bool:
    return free_bytes >= min_free


def apply_disk_guard(store, free_bytes: int, clock, min_free: int = MIN_FREE_BYTES) -> None:
    ok = disk_ok(free_bytes, min_free)
    if ok != store.raw_enabled:
        store.raw_enabled = ok
        store.write("recorder/events", envelope("recorder", "event", clock(), meta={
            "event": "raw_resumed" if ok else "raw_paused_low_disk", "free_bytes": free_bytes}))


def status_snapshot(store, now_ns: int, free_bytes: int, pid: int) -> dict:
    return {
        "now_ns": now_ns, "pid": pid, "free_bytes": free_bytes, "raw_enabled": store.raw_enabled,
        "streams": {name: {**st, "age_s": round((now_ns - st["last_received_ns"]) / 1e9, 3)}
                    for name, st in sorted(store.stats.items())},
    }


async def run_health(store, out: Path, stop: asyncio.Event, *, clock=time.time_ns,
                     free=lambda p: shutil.disk_usage(p).free, interval_s: float = 60.0,
                     pid: int = os.getpid()) -> None:
    out = Path(out)
    while True:
        now, free_bytes = clock(), free(out)
        apply_disk_guard(store, free_bytes, clock)
        store.flush_all()
        tmp = out / "status.json.tmp"
        tmp.write_text(json.dumps(status_snapshot(store, now, free_bytes, pid), indent=1))
        tmp.replace(out / "status.json")
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
```

```python
# backend/tools/recorder/run.py
"""Entry point:  python -m tools.recorder.run [--out DIR] [--products BTC-USD,ETH-USD]"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from tools.recorder.coinbase_ws import run_coinbase_ws
from tools.recorder.health import run_health
from tools.recorder.pollers import default_polls, make_http_get, run_poller
from tools.recorder.store import DailyStore, envelope, seal_past_days, utc_day

DEFAULT_OUT = Path(r"C:\Users\gl450\market_recorder_data")


async def main(out: Path, products: list) -> None:
    import httpx

    out.mkdir(parents=True, exist_ok=True)
    seal_past_days(out, utc_day(time.time_ns()))
    store, stop = DailyStore(out), asyncio.Event()
    store.write("recorder/events", envelope("recorder", "event", time.time_ns(), meta={
        "event": "start", "pid": os.getpid(), "products": products}))
    try:
        async with httpx.AsyncClient(headers={"User-Agent": "market-recorder/1"}) as client:
            get = make_http_get(client)
            tasks = [run_coinbase_ws(store, products, stop), run_health(store, out, stop)]
            tasks += [run_poller(p, store, get, stop) for p in default_polls()]
            await asyncio.gather(*tasks)
    finally:
        stop.set()
        store.write("recorder/events", envelope("recorder", "event", time.time_ns(),
                                                meta={"event": "stop"}))
        store.close()


def cli(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Standalone read-only market data recorder")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--products", default="BTC-USD,ETH-USD")
    args = ap.parse_args(argv)
    try:
        asyncio.run(main(args.out, [p.strip() for p in args.products.split(",") if p.strip()]))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Run to verify pass.** Expected: 5 passed (4 health + 1 standalone). Then run the
  whole package with `pytest tests/tools/recorder -q`; expected 22 passed.

### Task 5: Commit, live pilot, launch

- [ ] **Step 1:** Run pinned ruff 0.9.0 `format` and `check` on `backend/tools/recorder` and
  `backend/tests/tools/recorder`. Add a CHANGELOG entry. Then make one commit (the hook runs the
  full suite) and push.
- [ ] **Step 2:** Pilot for 3 minutes into a temporary `--out`, then stop it:

  `cd backend && timeout 180 ../.venv/Scripts/python.exe -m tools.recorder.run --out <tmp>`

  Verify all of the following:
  - every expected stream has at least 1 record;
  - every poll status is 200;
  - `status.json` exists;
  - events show `connect` and no `gap`;
  - files read back with `read_records`.
- [ ] **Step 3:** Launch detached into `C:\Users\gl450\market_recorder_data` with
  `Start-Process pythonw -ArgumentList '-m','tools.recorder.run' -WorkingDirectory <worktree>\backend -WindowStyle Hidden`.
  Record the PID from `status.json`. Autostart at logon is left as an operator decision.
</content>
</invoke>
<parameter name="file_path">C:\Users\gl450\polymarket_app\.claude\worktrees\market-recorder\docs\superpowers\plans\2026-10-03-market-recorder.md