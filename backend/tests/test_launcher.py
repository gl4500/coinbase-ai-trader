"""
Tests for launcher.py first-run defaults (#113).

The launcher already auto-starts backend+frontend 1 s after its window
opens, and exposes a 'Start on login' checkbox that writes to
HKCU Run. Issue: the checkbox defaults to whatever's in the registry —
on a fresh install nothing is there, so the user has to toggle it once
manually. They keep forgetting to do so (or to click Start). The fix:
on the very first launcher run, default the toggle ON and remember
the decision via a sentinel file so subsequent runs respect any
user override.
"""

import importlib
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _import_launcher():
    """Import launcher.py from the repo root; tolerate missing tkinter."""
    try:
        import launcher
    except Exception as e:
        pytest.skip(f"launcher import unavailable: {e}")
    importlib.reload(launcher)
    return launcher


def test_first_run_calls_set_startup_true_and_writes_sentinel(tmp_path):
    launcher = _import_launcher()

    sentinel = tmp_path / "first_run.flag"
    captured = {"calls": []}

    def fake_set_startup(enabled: bool) -> bool:
        captured["calls"].append(enabled)
        return True

    changed = launcher._maybe_default_startup_to_on(sentinel, set_fn=fake_set_startup)

    assert changed is True
    assert captured["calls"] == [True], (
        f"Expected one set_startup(True) call on first run, got {captured['calls']}"
    )
    assert sentinel.exists(), "first-run sentinel must be created"


def test_subsequent_runs_are_no_op(tmp_path):
    launcher = _import_launcher()

    sentinel = tmp_path / "first_run.flag"
    sentinel.touch()  # simulates a prior run

    captured = {"calls": []}

    def fake_set_startup(enabled: bool) -> bool:
        captured["calls"].append(enabled)
        return True

    changed = launcher._maybe_default_startup_to_on(sentinel, set_fn=fake_set_startup)

    assert changed is False
    assert captured["calls"] == [], (
        "set_startup must not be called when sentinel exists (would clobber a user opt-out)"
    )


def test_sentinel_parent_is_created_if_missing(tmp_path):
    launcher = _import_launcher()

    sentinel = tmp_path / "deep" / "subdir" / "first_run.flag"
    assert not sentinel.parent.exists()

    def noop_set(enabled: bool) -> bool:
        return True

    launcher._maybe_default_startup_to_on(sentinel, set_fn=noop_set)

    assert sentinel.exists()
