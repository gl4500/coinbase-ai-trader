"""Single policy point for starting the backend on a port.

Default is to REFUSE an occupied port or a held instance lock; nothing is killed.
Replacement happens only on explicit request and only for an instance this guard
itself registered (lock record PID + process creation time + backend dir), with
fresh revalidation right before signaling. See
docs/superpowers/specs/2026-10-17-backend-startup-guard-design.md.

Limitation: the PID/taskkill design leaves a small check-to-signal window after
the final revalidation; it is narrowed, not eliminated.

Stdlib only: imported by main.py before the app's own modules matter.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import List, Optional

MODES = ("refuse", "replace", "replace-unhealthy")

EXIT_PORT_IN_USE = 2
EXIT_LOCK_HELD = 3
EXIT_OWNER_UNVERIFIED = 4
EXIT_NOT_UNHEALTHY = 5
EXIT_REVALIDATION_FAILED = 6
EXIT_EXIT_TIMEOUT = 7
EXIT_INCONSISTENT = 8

HEALTH_INTERVAL_SECS = 10.0
HEALTH_TIMEOUT_SECS = 5.0
HEALTH_MIN_FAILURES = 3
HEALTH_MIN_SPAN_SECS = 30.0
HEALTH_BUDGET_SECS = 75.0
EXIT_WAIT_SECS = 30.0
POLL_SECS = 0.5
LOCK_OFFSET = 1 << 20  # lock a byte far past the record so readers never hit the locked region


class StartupRefused(Exception):
    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code = code
        self.reason = reason


# ── Parsing / classification (pure) ───────────────────────────────────────────

_ROW = re.compile(r"^\s*TCP\s+(\S+)\s+\S+\s+LISTENING\s+(\d+)\s*$", re.IGNORECASE)


def parse_netstat(text: str, port: int) -> List[int]:
    """PIDs LISTENING on exactly `port` (IPv4 and IPv6), unique and sorted."""
    pids = set()
    for line in text.splitlines():
        m = _ROW.match(line)
        if m and m.group(1).rsplit(":", 1)[-1] == str(port):
            pids.add(int(m.group(2)))
    return sorted(pids)


def classify_health(
    status: Optional[int], body: Optional[bytes], exc: Optional[BaseException]
) -> str:
    """'ok' | 'unhealthy' | 'ambiguous'. Only positively identified failure counts as unhealthy."""
    if exc is not None:
        if isinstance(exc, (ConnectionRefusedError, TimeoutError)):
            return "unhealthy"
        return "ambiguous"
    try:
        payload = json.loads(body or b"")
    except ValueError:
        return "ambiguous"
    state = payload.get("status") if isinstance(payload, dict) else None
    if status == 200 and state == "ok":
        return "ok"
    if status == 503 and state == "unhealthy":
        return "unhealthy"
    return "ambiguous"


def mode_from(argv, env) -> str:
    mode = env.get("BACKEND_START_MODE", "refuse")
    if "--replace" in argv:
        mode = "replace"
    if "--mode" in argv:
        i = list(argv).index("--mode")
        mode = argv[i + 1] if i + 1 < len(argv) else ""
    if mode not in MODES:
        raise ValueError(f"unknown start mode {mode!r}; expected one of {MODES}")
    return mode


# ── OS access (real implementation; tests inject a fake) ──────────────────────


class System:
    def listeners(self, port: int) -> List[int]:
        out = ""
        for proto in ("TCP", "TCPv6"):
            out += subprocess.run(
                ["netstat", "-ano", "-p", proto],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            ).stdout
        return parse_netstat(out, port)

    @staticmethod
    def _open(pid: int):
        import ctypes

        return ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION

    def creation_time(self, pid: int) -> Optional[str]:
        import ctypes
        from ctypes import wintypes

        h = self._open(pid)
        if not h:
            return None
        try:
            c, e, k, u = (wintypes.FILETIME() for _ in range(4))
            if not ctypes.windll.kernel32.GetProcessTimes(
                h, *(ctypes.byref(x) for x in (c, e, k, u))
            ):
                return None
            return str((c.dwHighDateTime << 32) | c.dwLowDateTime)
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    def pid_alive(self, pid: int) -> bool:
        import ctypes
        from ctypes import wintypes

        h = self._open(pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        finally:
            ctypes.windll.kernel32.CloseHandle(h)

    def kill_tree(self, pid: int) -> int:
        return subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).returncode

    def probe_health(self, port: int, timeout: float) -> str:
        import socket
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/api/health", timeout=timeout
            ) as r:
                return classify_health(r.status, r.read(), None)
        except urllib.error.HTTPError as e:
            return classify_health(e.code, e.read(), None)
        except urllib.error.URLError as e:
            reason = e.reason if isinstance(e.reason, BaseException) else OSError(str(e.reason))
            if isinstance(reason, socket.timeout):
                reason = TimeoutError()
            return classify_health(None, None, reason)
        except (socket.timeout, TimeoutError):
            return classify_health(None, None, TimeoutError())
        except OSError as e:
            return classify_health(None, None, e)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, secs: float) -> None:
        time.sleep(secs)


class PortLock:
    """Exclusive lock held on one stable handle for the process lifetime.

    Opening never truncates; only the holder rewrites the record. The OS drops the
    lock when the holder exits, including after a forced kill.
    """

    def __init__(self, path: str):
        self.path = path
        self.fd: Optional[int] = None

    def try_acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
        try:
            os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the deployment target
                import fcntl

                fcntl.lockf(fd, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, LOCK_OFFSET)
        except OSError:
            os.close(fd)
            return False
        self.fd = fd
        return True

    def read_record(self) -> Optional[dict]:
        try:
            with open(self.path, "rb") as f:
                rec = json.loads(f.read().decode("utf-8"))
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) else None

    def write_record(self, rec: dict) -> None:
        assert self.fd is not None, "write_record requires the lock"
        data = json.dumps(rec, sort_keys=True).encode("utf-8")
        os.lseek(self.fd, 0, os.SEEK_SET)
        os.ftruncate(self.fd, 0)
        os.write(self.fd, data)
        os.fsync(self.fd)

    def release(self) -> None:
        if self.fd is None:
            return
        try:
            os.lseek(self.fd, LOCK_OFFSET, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(self.fd)
        self.fd = None


# ── Policy ────────────────────────────────────────────────────────────────────


def _valid_record(rec, port: int) -> bool:
    return (
        isinstance(rec, dict)
        and isinstance(rec.get("pid"), int)
        and isinstance(rec.get("creation_time"), str)
        and rec.get("port") == port
        and isinstance(rec.get("backend_dir"), str)
    )


def _same_dir(a: str, b: str) -> bool:
    return os.path.normcase(os.path.normpath(a)) == os.path.normcase(os.path.normpath(b))


def _proven_unhealthy(system, port: int) -> None:
    """Return only when unhealthiness is positively identified over >= HEALTH_MIN_SPAN_SECS."""
    start = system.monotonic()
    failures: List[float] = []
    while True:
        result = system.probe_health(port, HEALTH_TIMEOUT_SECS)
        now = system.monotonic()
        if result == "ok":
            raise StartupRefused(EXIT_NOT_UNHEALTHY, "backend healthy; not replacing")
        if result != "unhealthy":
            raise StartupRefused(EXIT_NOT_UNHEALTHY, "health reply ambiguous; not replacing")
        failures.append(now)
        if (
            len(failures) >= HEALTH_MIN_FAILURES
            and failures[-1] - failures[0] >= HEALTH_MIN_SPAN_SECS
        ):
            return
        if now - start + HEALTH_INTERVAL_SECS > HEALTH_BUDGET_SECS:
            raise StartupRefused(EXIT_NOT_UNHEALTHY, "unhealthiness not established within budget")
        system.sleep(HEALTH_INTERVAL_SECS)


def _replace_holder(port, mode, backend_dir, system, lock) -> None:
    listeners = system.listeners(port)
    if not listeners:
        raise StartupRefused(EXIT_LOCK_HELD, "another backend instance holds the lock (starting)")
    rec = lock.read_record()
    if not _valid_record(rec, port):
        raise StartupRefused(
            EXIT_OWNER_UNVERIFIED, "lock record missing or unreadable; owner not verifiable"
        )
    if listeners != [rec["pid"]]:
        raise StartupRefused(
            EXIT_INCONSISTENT, f"listeners {listeners} disagree with lock owner {rec['pid']}"
        )
    if not _same_dir(rec["backend_dir"], backend_dir):
        raise StartupRefused(EXIT_OWNER_UNVERIFIED, "lock owner runs a different backend directory")
    pid = rec["pid"]
    if system.creation_time(pid) != rec["creation_time"]:
        raise StartupRefused(
            EXIT_OWNER_UNVERIFIED, f"PID {pid} creation time does not match the lock record"
        )
    if mode == "replace-unhealthy":
        _proven_unhealthy(system, port)
    # Revalidate immediately before signaling.
    if (
        system.listeners(port) != [pid]
        or system.creation_time(pid) != rec["creation_time"]
        or lock.read_record() != rec
    ):
        raise StartupRefused(
            EXIT_REVALIDATION_FAILED, "owner changed before signaling; not killing"
        )
    rc = system.kill_tree(pid)
    if rc != 0:
        raise StartupRefused(EXIT_REVALIDATION_FAILED, f"taskkill returned {rc}")
    deadline = system.monotonic() + EXIT_WAIT_SECS
    while system.pid_alive(pid) or system.listeners(port):
        if system.monotonic() >= deadline:
            raise StartupRefused(
                EXIT_EXIT_TIMEOUT, f"PID {pid} or port {port} not released within {EXIT_WAIT_SECS}s"
            )
        system.sleep(POLL_SECS)
    while not lock.try_acquire():
        if system.monotonic() >= deadline:
            raise StartupRefused(EXIT_LOCK_HELD, "lock not released after replacement")
        system.sleep(POLL_SECS)


def guard_startup(
    port: int,
    mode: str,
    backend_dir: str,
    system=None,
    lock=None,
    self_pid: Optional[int] = None,
    self_created: Optional[str] = None,
):
    """Return the held lock (keep it referenced until exit) or raise StartupRefused."""
    if mode not in MODES:
        raise ValueError(f"unknown start mode {mode!r}")
    system = system or System()
    lock = lock or PortLock(os.path.join(backend_dir, ".locks", f"backend-{port}.lock"))
    if not lock.try_acquire():
        if mode == "refuse":
            rec = lock.read_record()
            owner = rec.get("pid") if isinstance(rec, dict) else "unknown"
            raise StartupRefused(
                EXIT_LOCK_HELD, f"another backend instance holds the lock (PID {owner})"
            )
        _replace_holder(port, mode, backend_dir, system, lock)
        replaced = True
    else:
        replaced = False
    listeners = system.listeners(port)
    if listeners:
        lock.release()
        if replaced:
            raise StartupRefused(
                EXIT_INCONSISTENT, f"port {port} taken by {listeners} after replacement"
            )
        if mode == "refuse":
            raise StartupRefused(EXIT_PORT_IN_USE, f"port {port} already in use by PID {listeners}")
        raise StartupRefused(
            EXIT_OWNER_UNVERIFIED, f"port {port} owner {listeners} holds no lock; stop it manually"
        )
    pid = self_pid if self_pid is not None else os.getpid()
    created = self_created if self_created is not None else system.creation_time(pid)
    lock.write_record(
        {"pid": pid, "creation_time": created or "", "port": port, "backend_dir": backend_dir}
    )
    return lock


def run_or_exit(port: int, backend_dir: str, logger, argv=None, env=None):
    """main.py entry: hold the lock or exit with a distinct code and a logged reason."""
    try:
        mode = mode_from(
            argv if argv is not None else sys.argv, env if env is not None else os.environ
        )
        return guard_startup(port, mode, backend_dir)
    except StartupRefused as e:
        logger.error("Startup refused (exit %d): %s", e.code, e.reason)
        print(f"Startup refused (exit {e.code}): {e.reason}", file=sys.stderr)
        sys.exit(e.code)
    except ValueError as e:
        logger.error("Startup refused: %s", e)
        print(f"Startup refused: {e}", file=sys.stderr)
        sys.exit(EXIT_INCONSISTENT)
