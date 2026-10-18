"""Tests for startup_guard: refuse-by-default, single-instance lock, verified replacement.

Every OS interaction goes through a fake System; nothing touches the live 8001,
its processes or coinbase.db. The real-lock tests use temp files and a helper
subprocess that holds the lock.
"""

import json
import os
import subprocess
import sys
import textwrap
import time

import pytest

import startup_guard as sg

PORT = 18001
BACKEND = r"C:\repo\backend"


class FakeSystem:
    """Scriptable stand-in for the OS. Tracks every kill request."""

    def __init__(
        self, listeners=(), created=None, alive=None, health=(), kill_rc=0, exits_after_kill=True
    ):
        self._listeners = list(listeners)
        self.created = dict(created or {})
        self.alive = set(alive if alive is not None else self._listeners)
        self.health = list(health)
        self.kill_rc = kill_rc
        self.exits_after_kill = exits_after_kill
        self.killed = []
        self.clock = 0.0
        self.probe_times = []
        self.listener_script = None  # optional callable(call_index) -> list

        self._calls = 0

    def listeners(self, port):
        self._calls += 1
        if self.listener_script:
            return self.listener_script(self._calls)
        return list(self._listeners)

    def creation_time(self, pid):
        return self.created.get(pid)

    def pid_alive(self, pid):
        return pid in self.alive

    def kill_tree(self, pid):
        self.killed.append(pid)
        if self.kill_rc == 0 and self.exits_after_kill:
            self.alive.discard(pid)
            self._listeners = [p for p in self._listeners if p != pid]
        return self.kill_rc

    def probe_health(self, port, timeout):
        self.probe_times.append(self.clock)
        self.clock += 0.5
        return self.health.pop(0) if self.health else "ok"

    def monotonic(self):
        return self.clock

    def sleep(self, secs):
        self.clock += secs


class FakeLock:
    """In-memory lock: `held_by_other` simulates another live holder."""

    def __init__(self, held_by_other=False, record=None, release_on_kill_of=None, sys_=None):
        self.held_by_other = held_by_other
        self.record = record
        self.acquired = False
        self.written = []
        self.release_on_kill_of = release_on_kill_of
        self.sys_ = sys_

    def try_acquire(self):
        if self.held_by_other and self.release_on_kill_of is not None and self.sys_ is not None:
            if self.release_on_kill_of not in self.sys_.alive:
                self.held_by_other = False
        if self.held_by_other:
            return False
        self.acquired = True
        return True

    def read_record(self):
        return self.record

    def write_record(self, rec):
        assert self.acquired
        self.record = rec
        self.written.append(rec)

    def release(self):
        self.acquired = False


def _record(pid, created="133000000000000000", backend=BACKEND):
    return {"pid": pid, "creation_time": created, "port": PORT, "backend_dir": backend}


def _guard(sys_, lock, mode="refuse"):
    return sg.guard_startup(
        PORT, mode, BACKEND, system=sys_, lock=lock, self_pid=999, self_created="1"
    )


# ---------- netstat parsing ----------
NETSTAT = """
  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:18001          0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:180010       0.0.0.0:0              LISTENING       5555
  TCP    127.0.0.1:18001        127.0.0.1:50000        ESTABLISHED     4242
  TCP    [::]:18001             [::]:0                 LISTENING       4242
  TCP    [::1]:18002            [::]:0                 LISTENING       7777
  TCP    0.0.0.0:28001          0.0.0.0:0              LISTENING       8888
"""


def test_parse_netstat_exact_port_listening_dedup():
    assert sg.parse_netstat(NETSTAT, PORT) == [4242]


def test_parse_netstat_multiple_owners_sorted_unique():
    text = NETSTAT + "  TCP    0.0.0.0:18001   0.0.0.0:0   LISTENING   1111\n"
    assert sg.parse_netstat(text, PORT) == [1111, 4242]


def test_parse_netstat_empty():
    assert sg.parse_netstat("", PORT) == []


# ---------- refuse (default) ----------
def test_free_port_starts_and_writes_own_record():
    s, lk = FakeSystem(), FakeLock()
    held = _guard(s, lk)
    assert held is lk and lk.acquired
    assert lk.record == {"pid": 999, "creation_time": "1", "port": PORT, "backend_dir": BACKEND}
    assert s.killed == []


def test_refuse_when_port_in_use_and_never_kills():
    s, lk = FakeSystem(listeners=[4242], created={4242: "7"}), FakeLock()
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk)
    assert ei.value.code == sg.EXIT_PORT_IN_USE and s.killed == [] and not lk.acquired


def test_refuse_when_lock_held_by_other_instance():
    s = FakeSystem(listeners=[4242], created={4242: "7"})
    lk = FakeLock(held_by_other=True, record=_record(4242, "7"))
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk)
    assert ei.value.code == sg.EXIT_LOCK_HELD and s.killed == []


# ---------- ownership ----------
@pytest.mark.parametrize("mode", ["replace", "replace-unhealthy"])
def test_legacy_listener_without_lock_holder_is_unverifiable(mode):
    s, lk = FakeSystem(listeners=[4242], created={4242: "7"}), FakeLock()
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, mode)
    assert ei.value.code == sg.EXIT_OWNER_UNVERIFIED and s.killed == [] and not lk.acquired


@pytest.mark.parametrize(
    "record",
    [
        None,
        {"garbage": True},
        {"pid": "4242", "creation_time": "7", "port": PORT, "backend_dir": BACKEND},
        _record(4242, "8"),
        _record(4242, "7", backend=r"C:\other\backend"),
        _record(5151, "7"),
    ],
)
def test_unverifiable_or_mismatched_lock_record_refuses(record):
    s = FakeSystem(listeners=[4242], created={4242: "7", 5151: "7"})
    lk = FakeLock(held_by_other=True, record=record)
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code in (sg.EXIT_OWNER_UNVERIFIED, sg.EXIT_INCONSISTENT) and s.killed == []


def test_wrapper_child_pid_ownership_matched_by_record_not_cmdline():
    # Listener is the base interpreter (child of the venv stub); the record names it exactly.
    s = FakeSystem(listeners=[27796], created={27796: "133"})
    lk = FakeLock(
        held_by_other=True, record=_record(27796, "133"), release_on_kill_of=27796, sys_=s
    )
    held = _guard(s, lk, "replace")
    assert s.killed == [27796] and held is lk and lk.record["pid"] == 999


def test_multiple_listeners_with_lock_held_refuses_inconsistent():
    s = FakeSystem(listeners=[4242, 1111], created={4242: "7", 1111: "9"})
    lk = FakeLock(held_by_other=True, record=_record(4242, "7"))
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code == sg.EXIT_INCONSISTENT and s.killed == []


def test_lock_held_but_no_listener_means_other_instance_starting():
    s = FakeSystem(listeners=[], created={4242: "7"}, alive=[4242])
    lk = FakeLock(held_by_other=True, record=_record(4242, "7"))
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code == sg.EXIT_LOCK_HELD and s.killed == []


# ---------- replace-unhealthy health rules ----------
def _unhealthy_case(health):
    s = FakeSystem(listeners=[4242], created={4242: "7"}, health=health)
    lk = FakeLock(held_by_other=True, record=_record(4242, "7"), release_on_kill_of=4242, sys_=s)
    return s, lk


def test_replace_unhealthy_requires_failures_spanning_30s():
    s, lk = _unhealthy_case(["unhealthy"] * 4)
    _guard(s, lk, "replace-unhealthy")
    assert s.killed == [4242]
    assert s.probe_times[-1] - s.probe_times[0] >= 30


def test_replace_unhealthy_refuses_when_backend_recovers():
    s, lk = _unhealthy_case(["unhealthy", "ok", "unhealthy", "unhealthy", "unhealthy"])
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace-unhealthy")
    assert ei.value.code == sg.EXIT_NOT_UNHEALTHY and s.killed == []


def test_replace_unhealthy_refuses_ambiguous_reply():
    s, lk = _unhealthy_case(["unhealthy", "ambiguous", "unhealthy", "unhealthy"])
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace-unhealthy")
    assert ei.value.code == sg.EXIT_NOT_UNHEALTHY and s.killed == []


def test_three_quick_failures_alone_are_not_enough():
    # Probe cadence is fixed by the guard; three failures can never span 30 s, so a 4th is needed.
    s, lk = _unhealthy_case(["unhealthy"] * 3 + ["ok"])
    with pytest.raises(sg.StartupRefused):
        _guard(s, lk, "replace-unhealthy")
    assert s.killed == []


def test_plain_replace_does_not_probe_health():
    s, lk = _unhealthy_case([])
    _guard(s, lk, "replace")
    assert s.probe_times == [] and s.killed == [4242]


# ---------- revalidation, kill result, exit wait ----------
def test_revalidation_detects_pid_reuse_before_kill():
    s, lk = _unhealthy_case([])
    orig = s.creation_time
    calls = {"n": 0}

    def created(pid):
        calls["n"] += 1
        return orig(pid) if calls["n"] == 1 else "999999"  # creation time changed at revalidation

    s.creation_time = created
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code == sg.EXIT_REVALIDATION_FAILED and s.killed == []


def test_kill_failure_aborts():
    s, lk = _unhealthy_case([])
    s.kill_rc = 128
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code == sg.EXIT_REVALIDATION_FAILED and not lk.acquired


def test_exit_wait_timeout_aborts():
    s, lk = _unhealthy_case([])
    s.exits_after_kill = False
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code == sg.EXIT_EXIT_TIMEOUT
    assert s.clock <= sg.EXIT_WAIT_SECS + 5


def test_listener_rechecked_after_reacquire_refuses_race_winner():
    s, lk = _unhealthy_case([])
    real = s.kill_tree

    def kill_and_race(pid):
        rc = real(pid)
        s._listeners = [6060]  # a foreign process grabs the port right after the kill
        s.alive.add(6060)
        return rc

    s.kill_tree = kill_and_race
    s.exits_after_kill = True
    with pytest.raises(sg.StartupRefused) as ei:
        _guard(s, lk, "replace")
    assert ei.value.code in (sg.EXIT_EXIT_TIMEOUT, sg.EXIT_INCONSISTENT)
    assert s.killed == [4242] and not lk.acquired


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        _guard(FakeSystem(), FakeLock(), "kill-everything")


def test_mode_from_argv_and_env():
    assert sg.mode_from(["main.py"], {}) == "refuse"
    assert sg.mode_from(["main.py", "--replace"], {}) == "replace"
    assert sg.mode_from(["main.py", "--mode", "replace-unhealthy"], {}) == "replace-unhealthy"
    assert sg.mode_from(["main.py"], {"BACKEND_START_MODE": "replace"}) == "replace"


# ---------- real file lock ----------
HOLDER = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[2])
    import startup_guard as sg
    lk = sg.PortLock(sys.argv[1])
    assert lk.try_acquire()
    lk.write_record({"pid": 1, "creation_time": "x", "port": 1, "backend_dir": "d"})
    print("held", flush=True)
    time.sleep(60)
""")


@pytest.mark.skipif(os.name != "nt", reason="msvcrt lock semantics")
def test_real_lock_is_exclusive_readable_and_released_on_holder_death(tmp_path):
    path = tmp_path / "backend-1.lock"
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    holder = subprocess.Popen(
        [sys.executable, "-c", HOLDER, str(path), backend_dir], stdout=subprocess.PIPE, text=True
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        mine = sg.PortLock(str(path))
        assert mine.try_acquire() is False
        assert mine.read_record() == {"pid": 1, "creation_time": "x", "port": 1, "backend_dir": "d"}
    finally:
        holder.kill()
        holder.wait(10)
    deadline = time.monotonic() + 5
    while not mine.try_acquire():
        assert time.monotonic() < deadline
        time.sleep(0.1)
    mine.write_record({"pid": 2, "creation_time": "y", "port": 1, "backend_dir": "d"})
    assert json.loads(path.read_text())["pid"] == 2
    mine.release()


@pytest.mark.skipif(os.name != "nt", reason="msvcrt lock semantics")
def test_real_lock_open_does_not_truncate_existing_record(tmp_path):
    path = tmp_path / "backend-2.lock"
    path.write_text(json.dumps({"pid": 5, "creation_time": "z", "port": 2, "backend_dir": "d"}))
    other = sg.PortLock(str(path))
    assert other.read_record()["pid"] == 5  # opening/reading never truncates
    assert json.loads(path.read_text())["pid"] == 5


def test_read_record_corrupt_returns_none(tmp_path):
    path = tmp_path / "backend-3.lock"
    path.write_text("{not json")
    assert sg.PortLock(str(path)).read_record() is None


# ---------- health reply classification ----------
@pytest.mark.parametrize(
    "status,body,exc,want",
    [
        (200, b'{"status": "ok", "last_scan_at": 1}', None, "ok"),
        (503, b'{"status": "unhealthy"}', None, "unhealthy"),
        (503, b"service unavailable", None, "ambiguous"),
        (200, b'{"status": "unhealthy"}', None, "ambiguous"),
        (401, b'{"detail": "Invalid"}', None, "ambiguous"),
        (500, b'{"status": "unhealthy"}', None, "ambiguous"),
        (200, b"not json", None, "ambiguous"),
        (None, None, ConnectionRefusedError(), "unhealthy"),
        (None, None, TimeoutError(), "unhealthy"),
        (None, None, OSError("weird"), "ambiguous"),
    ],
)
def test_classify_health(status, body, exc, want):
    assert sg.classify_health(status, body, exc) == want
