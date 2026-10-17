"""Bounded database open for startup.

Retries only the open/configure step, only for transient OperationalErrors, with a
fresh connection each attempt and a total deadline. Corruption is never retried.
After a recovery from `disk I/O error`, a time-capped quick_check must return "ok"
or startup fails (an interrupted check also fails: fail closed). A successful
retry does not establish what caused the error.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

import aiosqlite

logger = logging.getLogger(__name__)

INIT_BUSY_MS = 5000  # per-attempt busy timeout while opening; normal operation keeps 30 s
DEADLINE_SECS = 30.0
BACKOFFS = (0.5, 1, 2, 4)
QUICK_CHECK_CAP_SECS = 60.0
_RETRYABLE = ("disk i/o error", "database is locked", "unable to open database file")


class DatabaseIntegrityError(RuntimeError):
    pass


class ConfigureError(Exception):
    """Raised by connect() when the file opened but configuration failed; carries the conn to close."""

    def __init__(self, conn, cause: BaseException):
        super().__init__(str(cause))
        self.conn = conn
        self.cause = cause


async def _connect_configured(path: str, busy_ms: int):
    conn = await aiosqlite.connect(path, timeout=busy_ms / 1000)
    try:
        await conn.execute(f"PRAGMA busy_timeout={int(busy_ms)}")
        await conn.execute("PRAGMA journal_mode=WAL")
    except BaseException as e:
        raise ConfigureError(conn, e) from e
    return conn


def _retryable(exc: BaseException) -> bool:
    return type(exc) is sqlite3.OperationalError and any(m in str(exc).lower() for m in _RETRYABLE)


async def open_with_retry(
    path: str,
    deadline_secs: float = DEADLINE_SECS,
    connect=_connect_configured,
    sleep=asyncio.sleep,
    clock=time.monotonic,
):
    """Return (connection, had_io_error). Raises the last error once the deadline would be passed."""
    start = clock()
    had_io = False
    attempt = 0
    while True:
        attempt += 1
        try:
            return await connect(path, INIT_BUSY_MS), had_io
        except ConfigureError as ce:
            await ce.conn.close()
            exc = ce.cause
        except sqlite3.Error as e:
            exc = e
        if not _retryable(exc):
            raise exc
        had_io = had_io or "disk i/o error" in str(exc).lower()
        backoff = BACKOFFS[min(attempt - 1, len(BACKOFFS) - 1)]
        elapsed = clock() - start
        logger.warning(
            "DB open attempt %d failed after %.1fs: %s %s (%s)",
            attempt,
            elapsed,
            type(exc).__name__,
            exc,
            getattr(exc, "sqlite_errorname", "?"),
        )
        if elapsed + backoff > deadline_secs:
            raise exc
        await sleep(backoff)


def quick_check_sync(path: str, cap_secs: float = QUICK_CHECK_CAP_SECS) -> str:
    """'ok', 'interrupted' (cap reached), or the non-ok quick_check output. Separate read-only connection."""
    deadline = time.monotonic() + cap_secs
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 100)
        try:
            rows = con.execute("PRAGMA quick_check").fetchall()
        except sqlite3.OperationalError as e:
            if "interrupt" in str(e).lower():
                return "interrupted"
            raise
        return "\n".join(str(r[0]) for r in rows)
    finally:
        con.close()


async def integrity_gate(
    path: str, check=quick_check_sync, cap_secs: float = QUICK_CHECK_CAP_SECS
) -> None:
    result = await asyncio.to_thread(check, path, cap_secs)
    if result == "ok":
        logger.warning("quick_check ok after disk I/O error recovery")
        return
    if result == "interrupted":
        logger.critical(
            "quick_check interrupted at %.0fs cap after disk I/O recovery; failing startup",
            cap_secs,
        )
        raise DatabaseIntegrityError(
            f"quick_check inconclusive (interrupted at {cap_secs:.0f}s cap)"
        )
    logger.critical("quick_check failed after disk I/O recovery: %s", result[:500])
    raise DatabaseIntegrityError(f"quick_check failed: {result[:200]}")
