"""Launcher watchdog state machine and status-loop dispatch (Codex PR #105 P105-01/02)."""

import importlib
import os
import sys
import types

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


@pytest.fixture
def launcher():
    try:
        import launcher as mod
    except Exception as e:  # tkinter missing on some runners
        pytest.skip(f"launcher import unavailable: {e}")
    return importlib.reload(mod)


def _run(wd, steps):
    """steps: (health, own_alive, now) tuples -> list of actions."""
    return [wd.step(h, own, now) for h, own, now in steps]


def test_external_unhealthy_backend_triggers_one_replacement(launcher):
    wd = launcher._Watchdog()
    acts = _run(wd, [("unhealthy", False, t) for t in (0, 1, 2)])
    assert acts == ["none", "none", "replace_external"]


def test_external_down_or_ambiguous_never_acts(launcher):
    wd = launcher._Watchdog()
    acts = _run(
        wd,
        [("down", False, t) for t in range(10)] + [("ambiguous", False, t) for t in range(10, 20)],
    )
    assert set(acts) == {"none"}


def test_inflight_helper_is_not_killed_by_fast_503s(launcher):
    wd = launcher._Watchdog()
    _run(wd, [("unhealthy", False, t) for t in (0, 1, 2)])
    wd.begin_inflight(2.5)  # launcher spawned main.py --mode replace-unhealthy (now its own child)
    acts = _run(wd, [("unhealthy", True, 3 + t) for t in range(60)])  # helper verifying for 60 s
    assert set(acts) == {"wait"}


def test_inflight_ends_when_backend_healthy_then_normal_monitoring(launcher):
    wd = launcher._Watchdog()
    wd.begin_inflight(0)
    assert wd.step("unhealthy", True, 1) == "wait"
    assert wd.step("ok", True, 40) == "none"
    acts = _run(wd, [("unhealthy", True, 41 + t) for t in range(3)])
    assert acts[-1] == "restart_own"


def test_refused_replacement_enters_cooldown_no_respawn_loop(launcher):
    wd = launcher._Watchdog()
    _run(wd, [("unhealthy", False, t) for t in (0, 1, 2)])
    wd.begin_inflight(2.5)
    assert wd.step("unhealthy", False, 3) == "none"  # helper exited (guard refused), inflight ends
    acts = _run(wd, [("unhealthy", False, 4 + t) for t in range(int(wd.cooldown_secs) - 10)])
    assert "replace_external" not in acts
    later = _run(wd, [("unhealthy", False, 3 + wd.cooldown_secs + t) for t in range(3)])
    assert later[-1] == "replace_external"


def test_healthy_reply_clears_cooldown(launcher):
    wd = launcher._Watchdog()
    _run(wd, [("unhealthy", False, t) for t in (0, 1, 2)])
    wd.begin_inflight(2.5)
    wd.step("unhealthy", False, 3)
    wd.step("ok", False, 4)
    acts = _run(wd, [("unhealthy", False, 5 + t) for t in range(3)])
    assert acts[-1] == "replace_external"


def test_own_child_failures_restart_own(launcher):
    wd = launcher._Watchdog()
    acts = _run(wd, [("down", True, t) for t in range(3)])
    assert acts == ["none", "none", "restart_own"]


def test_inflight_deadline_resumes_monitoring(launcher):
    wd = launcher._Watchdog()
    wd.begin_inflight(0)
    assert wd.step("unhealthy", True, wd.inflight_secs - 1) == "wait"
    acts = _run(wd, [("unhealthy", True, wd.inflight_secs + t) for t in range(3)])
    assert acts[-1] == "restart_own"


@pytest.mark.parametrize(
    "outcome,want",
    [
        ((200, None), "ok"),
        ((503, None), "unhealthy"),
        ((401, None), "ambiguous"),
        ((None, ConnectionRefusedError()), "down"),
        ((None, TimeoutError()), "unhealthy"),
        ((None, OSError("x")), "ambiguous"),
    ],
)
def test_health_state_classification(launcher, outcome, want):
    assert launcher._health_state(*outcome) == want


def test_status_tick_dispatches_replacement_for_unhealthy_external_backend(launcher):
    """Full tick path with no owned child: 4 unhealthy probes -> exactly one replace-unhealthy start."""
    calls = []
    app = types.SimpleNamespace(
        _backend_proc=None,
        _watchdog=launcher._Watchdog(),
        after=lambda ms, fn, *a: fn(*a),
        _update_status=lambda *a: None,
        _queue_log=lambda *a: None,
        _start_backend=lambda mode=None: calls.append(mode),
        _restart_backend=lambda: calls.append("restart_own"),
    )
    for t in range(4):
        launcher.LauncherApp._status_tick(app, "unhealthy", True, float(t))
    assert calls == ["replace-unhealthy"]
