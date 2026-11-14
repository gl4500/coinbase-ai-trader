"""Tests for live_guard.py. Standard library only.

Run from the repository root:
    <python> -m unittest tools/agent_guard/test_live_guard.py -v
"""
import io
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import live_guard as g  # noqa: E402

NOW = datetime(2026, 11, 14, 18, 0, tzinfo=timezone.utc)
CFG = g.Config(port=8001, training_markers=("train_worker.py",), commit_needs_staged_py=True)


def _never(*_a, **_k):
    raise AssertionError("probe must not be called for a harmless command")


def decide(command, *, port_open=False, training=False, staged_py=False, window=False, cfg=CFG):
    return g.decide(
        command, cfg,
        port_open=lambda port: port_open,
        training_running=lambda markers: training,
        staged_py=lambda: staged_py,
        window_open=lambda: window,
    )


class Classify(unittest.TestCase):
    def test_pytest_in_any_form_is_a_test_run(self):
        for cmd in ("python -m pytest tests/ -v", "pytest backend/tests", ".venv/Scripts/python.exe -m pytest -q"):
            self.assertIn("test", g.classify(cmd), cmd)

    def test_trading_app_runner_and_unittest_are_test_runs(self):
        for cmd in (r"C:\Users\gl450\trading_app\runtime\python\python.exe run_tests.py -v",
                    r"..\runtime\python\python.exe -m unittest tests.test_portfolio"):
            self.assertIn("test", g.classify(cmd), cmd)

    def test_git_commit_is_a_commit(self):
        self.assertIn("commit", g.classify('git commit -m "x" -- a.py'))
        self.assertIn("commit", g.classify("git -C repo commit -q -F msg.txt"))

    def test_process_kills_are_kills(self):
        for cmd in ("Stop-Process -Id 5 -Force", "taskkill /F /PID 3", "kill -9 123", "pkill -f python",
                    "Get-Process python | Stop-Process"):
            self.assertIn("kill", g.classify(cmd), cmd)

    def test_backend_starts_are_restarts(self):
        for cmd in ("python main.py", "PORT=8002 python main.py", "uvicorn main:app --port 8001",
                    "python -m backend.startup_guard --mode replace", "start_backend.ps1 -Replace"):
            self.assertIn("restart", g.classify(cmd), cmd)

    def test_harmless_commands_have_no_risk(self):
        for cmd in ("git status", "git log -1 --stat", "ls backend/tests", "gh pr view 122", "cat main.py"):
            self.assertEqual(g.classify(cmd), frozenset(), cmd)


class Decide(unittest.TestCase):
    def test_test_run_while_live_is_blocked_and_names_the_port(self):
        d = decide("python -m pytest tests/", port_open=True)
        self.assertTrue(d.block)
        self.assertIn("8001", d.reason)

    def test_test_run_while_training_is_blocked(self):
        d = decide("python -m pytest tests/", training=True)
        self.assertTrue(d.block)
        self.assertIn("training", d.reason)

    def test_test_run_with_nothing_live_is_allowed(self):
        self.assertFalse(decide("python -m pytest tests/").block)

    def test_open_test_window_allows_a_test_run_while_live(self):
        self.assertFalse(decide("python -m pytest tests/", port_open=True, window=True).block)

    def test_harmless_command_while_live_never_probes(self):
        d = g.decide("git status", CFG, port_open=_never, training_running=_never,
                     staged_py=_never, window_open=_never)
        self.assertFalse(d.block)

    def test_commit_without_staged_python_is_allowed_while_live(self):
        self.assertFalse(decide('git commit -m "docs" -- AGENTS.md', port_open=True, staged_py=False).block)

    def test_commit_with_staged_python_is_blocked_while_live(self):
        d = decide('git commit -m "fix" -- a.py', port_open=True, staged_py=True)
        self.assertTrue(d.block)
        self.assertIn("commit", d.reason)

    def test_commit_is_harmless_when_repo_hook_runs_no_tests(self):
        cfg = g.Config(port=8000, training_markers=(), commit_needs_staged_py=False)
        self.assertFalse(decide('git commit -m "fix" -- a.py', port_open=True, staged_py=True, cfg=cfg).block)

    def test_kill_while_live_is_blocked_even_with_test_window(self):
        d = decide("Stop-Process -Id 5", port_open=True, window=True)
        self.assertTrue(d.block)

    def test_restart_while_training_is_blocked_even_with_test_window(self):
        d = decide("python main.py", training=True, window=True)
        self.assertTrue(d.block)

    def test_backend_start_while_live_but_not_training_is_left_to_startup_guard(self):
        # The sanctioned 8002 shadow starts while 8001 is live; startup_guard refuses an occupied port.
        self.assertFalse(decide("PORT=8002 python main.py", port_open=True).block)

    def test_probe_failure_on_a_risky_command_blocks(self):
        def broken(*_a):
            raise OSError("cannot list processes")
        d = g.decide("python -m pytest", CFG, port_open=lambda p: False, training_running=broken,
                     staged_py=lambda: False, window_open=lambda: False)
        self.assertTrue(d.block)
        self.assertIn("could not check", d.reason)


class TrainingDetection(unittest.TestCase):
    M = ("train_worker.py",)

    def test_real_training_worker_is_detected(self):
        self.assertTrue(g.training_in([("python.exe", "python backend/train_worker.py --epochs 3")], self.M))

    def test_guards_own_command_line_is_not_training(self):
        # The guard is launched with --training-marker train_worker.py; it must not match itself.
        own = r"C:\py\python.exe tools\agent_guard\live_guard.py --port 8001 --training-marker train_worker.py"
        self.assertFalse(g.training_in([("python.exe", own)], self.M))

    def test_shell_that_merely_mentions_the_marker_is_not_training(self):
        self.assertFalse(g.training_in([("bash.exe", "bash -c 'grep -n train_worker.py cnn_agent.py'")], self.M))

    def test_multiline_command_line_is_one_process(self):
        # A shell running a heredoc has newlines in its command line; it must be judged as one record.
        script = "bash -c \"python - <<EOF\nargs=['--training-marker','train_worker.py']\nrun('live_guard.py')\nEOF\""
        self.assertFalse(g.training_in([("bash.exe", script)], self.M))

    def test_process_records_parse_from_tab_separated_output(self):
        out = "python.exe\tpython train_worker.py\nbash.exe\tbash -c x\n"
        self.assertEqual(g.parse_process_records(out),
                         [("python.exe", "python train_worker.py"), ("bash.exe", "bash -c x")])


class Window(unittest.TestCase):
    def _file(self, text):
        f = tempfile.NamedTemporaryFile("w", delete=False, suffix=".window", encoding="utf-8")
        f.write(text)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_future_expiry_is_open(self):
        self.assertTrue(g.window_is_open(self._file((NOW + timedelta(hours=1)).isoformat()), NOW))

    def test_past_expiry_is_closed(self):
        self.assertFalse(g.window_is_open(self._file((NOW - timedelta(minutes=1)).isoformat()), NOW))

    def test_missing_or_garbled_file_is_closed(self):
        self.assertFalse(g.window_is_open(os.path.join(tempfile.gettempdir(), "no-such.window"), NOW))
        self.assertFalse(g.window_is_open(self._file("tomorrow-ish"), NOW))

    def test_expiry_without_timezone_is_closed(self):
        self.assertFalse(g.window_is_open(self._file("2099-01-01T00:00:00"), NOW))


class Main(unittest.TestCase):
    def run_main(self, payload, port_open=True):
        stdin = io.StringIO(payload if isinstance(payload, str) else json.dumps(payload))
        stderr = io.StringIO()
        code = g.main(["--port", "8001", "--training-marker", "train_worker.py", "--commit-needs-staged-py"],
                      stdin=stdin, stderr=stderr,
                      probes=dict(port_open=lambda p: port_open, training_running=lambda m: False,
                                  staged_py=lambda: True, window_open=lambda: False))
        return code, stderr.getvalue()

    def test_blocked_shell_command_exits_2_with_reason_on_stderr(self):
        code, err = self.run_main({"tool_name": "Bash", "tool_input": {"command": "pytest -q"}, "cwd": "."})
        self.assertEqual(code, 2)
        self.assertIn("8001", err)

    def test_powershell_tool_is_guarded_too(self):
        code, _ = self.run_main({"tool_name": "PowerShell", "tool_input": {"command": "Stop-Process -Id 1"}})
        self.assertEqual(code, 2)

    def test_allowed_command_exits_0(self):
        code, _ = self.run_main({"tool_name": "Bash", "tool_input": {"command": "git status"}})
        self.assertEqual(code, 0)

    def test_other_tools_are_ignored(self):
        code, _ = self.run_main({"tool_name": "Edit", "tool_input": {"file_path": "main.py"}})
        self.assertEqual(code, 0)

    def test_unparseable_payload_fails_open_with_a_warning(self):
        code, err = self.run_main("not json")
        self.assertEqual(code, 0)
        self.assertIn("live_guard", err)


if __name__ == "__main__":
    unittest.main()
