"""Tests for db_open: bounded open/configure retry and post-I/O integrity check.

Temp databases and fakes only; never touches coinbase.db.
"""

import sqlite3

import pytest

import db_open


class FakeConn:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    async def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def _script(*outcomes):
    """connect() stand-in: each call yields the next outcome (an exception or a FakeConn)."""
    calls = []

    async def connect(path, busy_ms):
        calls.append((path, busy_ms))
        o = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(o, BaseException):
            raise o
        return o

    return connect, calls


def _op(msg):
    return sqlite3.OperationalError(msg)


async def _open(connect, clock, deadline=30.0):
    return await db_open.open_with_retry(
        "x.db", deadline_secs=deadline, connect=connect, sleep=clock.sleep, clock=clock.now
    )


@pytest.mark.asyncio
async def test_first_attempt_success():
    c = FakeConn()
    connect, calls = _script(c)
    conn, had_io = await _open(connect, Clock())
    assert conn is c and had_io is False and len(calls) == 1
    assert calls[0][1] == db_open.INIT_BUSY_MS


@pytest.mark.asyncio
async def test_locked_then_success_uses_fresh_connection_and_backoff():
    c = FakeConn()
    clock = Clock()
    connect, calls = _script(_op("database is locked"), c)
    conn, had_io = await _open(connect, clock)
    assert conn is c and len(calls) == 2 and clock.sleeps == [0.5] and had_io is False


@pytest.mark.asyncio
async def test_disk_io_error_then_success_flags_io():
    c = FakeConn()
    connect, _ = _script(_op("disk I/O error"), c)
    conn, had_io = await _open(connect, Clock())
    assert had_io is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exc",
    [
        sqlite3.DatabaseError("database disk image is malformed"),
        sqlite3.DatabaseError("file is not a database"),
        _op("no such table: foo"),
        _op("database disk image is malformed"),
    ],
)
async def test_non_retryable_errors_propagate_immediately(exc):
    clock = Clock()
    connect, calls = _script(exc, FakeConn())
    with pytest.raises(type(exc)):
        await _open(connect, clock)
    assert len(calls) == 1 and clock.sleeps == []


@pytest.mark.asyncio
async def test_exhaustion_propagates_last_error_within_deadline():
    clock = Clock()
    connect, calls = _script(_op("database is locked"))
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        await _open(connect, clock, deadline=10.0)
    assert clock.t <= 10.0
    assert clock.sleeps[:3] == [0.5, 1, 2]


@pytest.mark.asyncio
async def test_connection_closed_when_configuration_fails():
    holder = {}

    async def connect(path, busy_ms):
        conn = FakeConn()
        holder.setdefault("conns", []).append(conn)
        if len(holder["conns"]) == 1:
            raise db_open.ConfigureError(conn, _op("database is locked"))
        return conn

    conn, _ = await _open(connect, Clock())
    assert holder["conns"][0].closed is True and conn is holder["conns"][1]


def _db_with_rows(path, n=20000):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    con.executemany("INSERT INTO t VALUES (?, ?)", ((i, "x" * 40) for i in range(n)))
    con.execute("CREATE INDEX ix ON t(a)")
    con.commit()
    con.close()


def test_quick_check_ok_on_healthy_db(tmp_path):
    p = tmp_path / "ok.db"
    _db_with_rows(p, 100)
    assert db_open.quick_check_sync(str(p), cap_secs=60) == "ok"


def test_quick_check_interrupted_by_cap(tmp_path):
    p = tmp_path / "big.db"
    _db_with_rows(p)
    assert db_open.quick_check_sync(str(p), cap_secs=-1) == "interrupted"


@pytest.mark.asyncio
@pytest.mark.parametrize("result", ["interrupted", "*** in database main ***\nPage 5: never used"])
async def test_integrity_gate_fails_startup_on_non_ok(result):
    with pytest.raises(db_open.DatabaseIntegrityError):
        await db_open.integrity_gate("x.db", check=lambda path, cap_secs: result)


@pytest.mark.asyncio
async def test_integrity_gate_passes_on_ok():
    await db_open.integrity_gate("x.db", check=lambda path, cap_secs: "ok")


@pytest.mark.asyncio
async def test_real_open_sets_wal_and_busy_timeout(tmp_path):
    p = tmp_path / "real.db"
    conn, had_io = await db_open.open_with_retry(str(p))
    try:
        async with conn.execute("PRAGMA journal_mode") as cur:
            assert (await cur.fetchone())[0].lower() == "wal"
        async with conn.execute("PRAGMA busy_timeout") as cur:
            assert (await cur.fetchone())[0] == db_open.INIT_BUSY_MS
    finally:
        await conn.close()
    assert had_io is False


@pytest.mark.asyncio
async def test_init_db_runs_integrity_gate_after_io_recovery(tmp_path, monkeypatch):
    import database

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "init.db"))
    real_open = db_open.open_with_retry

    async def open_flagging_io(path, **kw):
        conn, _ = await real_open(path, **kw)
        return conn, True

    async def failing_gate(path, **kw):
        raise db_open.DatabaseIntegrityError("quick_check failed: forced")

    monkeypatch.setattr(db_open, "open_with_retry", open_flagging_io)
    monkeypatch.setattr(db_open, "integrity_gate", failing_gate)
    with pytest.raises(db_open.DatabaseIntegrityError):
        await database.init_db()


@pytest.mark.asyncio
async def test_init_db_creates_schema_through_bounded_open(tmp_path, monkeypatch):
    import database

    p = tmp_path / "fresh.db"
    monkeypatch.setattr(database, "DB_PATH", str(p))
    calls = []
    real_open = db_open.open_with_retry

    async def spy(path, **kw):
        calls.append(path)
        return await real_open(path, **kw)

    monkeypatch.setattr(db_open, "open_with_retry", spy)
    await database.init_db()
    assert calls == [str(p)]
    con = sqlite3.connect(p)
    assert con.execute("SELECT name FROM sqlite_master WHERE name='products'").fetchone()
    con.close()
