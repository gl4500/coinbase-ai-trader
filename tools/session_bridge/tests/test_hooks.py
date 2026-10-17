import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from session_bridge.hooks import handle_hook, in_scope
from session_bridge.store import Store

ROOT = Path(r"C:\Users\gl450\polymarket_app")
HOME = Path(r"C:\Users\gl450")


class InScopeTests(unittest.TestCase):
    def test_root_worktree_and_home_are_in_scope(self):
        self.assertTrue(in_scope(str(ROOT), ROOT, HOME))
        self.assertTrue(in_scope(str(ROOT / ".claude/worktrees/x/backend"), ROOT, HOME))
        self.assertTrue(in_scope(str(HOME), ROOT, HOME))

    def test_other_projects_and_home_subdirs_are_out_of_scope(self):
        self.assertFalse(in_scope(str(HOME / "trading_app"), ROOT, HOME))
        self.assertFalse(in_scope(str(HOME / "grain-marketing"), ROOT, HOME))
        self.assertFalse(in_scope(str(HOME / "polymarket_app_old"), ROOT, HOME))

    def test_in_scope_normalizes_case_and_separators(self):
        self.assertTrue(in_scope("c:/users/gl450/polymarket_app/backend/", ROOT, HOME))
        self.assertTrue(in_scope("C:\\USERS\\GL450\\", ROOT, HOME))

    def test_relative_or_empty_cwd_is_out_of_scope(self):
        for cwd in ("temporary", "", "."):
            self.assertFalse(in_scope(cwd, ROOT, HOME))


class HandleHookScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "mail.sqlite3"
        Store(self.db, "codex").send("claude", "hello", "k1")

    def tearDown(self):
        self.temp.cleanup()

    def _run(self, cwd):
        event = {"hook_event_name": "PostToolUse", "session_id": "s1", "cwd": cwd}
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO(json.dumps(event))), redirect_stdout(out):
            handle_hook(self.db, "claude", ROOT / "tools/session_bridge/bridge.py", root=ROOT, home=HOME)
        return out.getvalue()

    def test_out_of_scope_session_gets_nothing_and_is_not_registered(self):
        self.assertEqual(self._run(str(HOME / "trading_app")), "")
        with Store(self.db, "claude").connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0], 0)

    def test_in_scope_session_gets_notice(self):
        out = self._run(str(ROOT))
        self.assertIn("hello", json.loads(out)["hookSpecificOutput"]["additionalContext"])
