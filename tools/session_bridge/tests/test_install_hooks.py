import json
import tempfile
import unittest
from pathlib import Path

import install_hooks as ih


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def bridge_cmds(data, event):
    return [h["command"] for e in data.get("hooks", {}).get(event, []) for h in e["hooks"]
            if ih.MARKER in h["command"]]


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        base = Path(self.temp.name)
        self.root, self.home = base / "repo", base / "home"
        self.python = self.root / ".coordination-runtime/Scripts/python.exe"
        (self.root / ".claude/worktrees/wt/.claude").mkdir(parents=True)
        (self.home / ".claude").mkdir(parents=True)
        (self.home / ".codex").mkdir(parents=True)
        self.user = self.home / ".claude/settings.json"
        self.codex = self.home / ".codex/hooks.json"
        self.local = self.root / ".claude/settings.local.json"
        self.wt_local = self.root / ".claude/worktrees/wt/.claude/settings.local.json"
        echo = {"hooks": [{"type": "command", "command": "echo AUTO-INVOKE"}]}
        herdr = {"hooks": [{"type": "command", "command": "powershell herdr-agent-state.ps1 session", "timeout": 10}]}
        self.user.write_text("﻿" + json.dumps({"model": "x", "hooks": {"SessionStart": [echo]}}), encoding="utf-8")
        self.codex.write_text(json.dumps({"hooks": {"SessionStart": [herdr]}}), encoding="utf-8")
        old = ih.hook_command(self.python, self.root / "tools/session_bridge/bridge.py", "claude")
        legacy = {"hooks": {e: [{"hooks": [{"type": "command", "command": old, "timeout": 8}]}] for e in ih.EVENTS}}
        self.local.write_text(json.dumps({**legacy, "permissions": {"allow": ["Bash(ls)"]}}), encoding="utf-8")
        self.wt_local.write_text(json.dumps(legacy), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_install(self, **kw):
        return ih.install(self.root, self.home, self.python, **kw)

    def test_installs_user_level_for_both_roles_and_preserves_others(self):
        self.run_install()
        user, codex = read(self.user), read(self.codex)
        for event in ih.EVENTS:
            self.assertEqual(len(bridge_cmds(user, event)), 1)
            self.assertIn("--role claude hook", bridge_cmds(user, event)[0])
            self.assertEqual(len(bridge_cmds(codex, event)), 1)
            self.assertIn("--role codex hook", bridge_cmds(codex, event)[0])
        self.assertEqual(user["model"], "x")
        self.assertIn("echo AUTO-INVOKE", json.dumps(user))
        self.assertIn("herdr-agent-state.ps1", json.dumps(codex))

    def test_strips_project_local_duplicates_and_leaves_valid_json(self):
        self.run_install()
        local, wt = read(self.local), read(self.wt_local)
        self.assertEqual(local, {"permissions": {"allow": ["Bash(ls)"]}})
        self.assertEqual(wt, {})

    def test_second_run_is_idempotent(self):
        self.run_install()
        before = [p.read_text(encoding="utf-8") for p in (self.user, self.codex)]
        lines = self.run_install()
        self.assertEqual([p.read_text(encoding="utf-8") for p in (self.user, self.codex)], before)
        self.assertTrue(any(line.startswith("Already installed:") for line in lines))

    def test_backup_written_before_change(self):
        self.run_install()
        backups = list((self.root / ".coordination/config-backups").glob("*.json"))
        self.assertGreaterEqual(len(backups), 4)

    def test_uninstall_removes_only_bridge_entries(self):
        self.run_install()
        self.run_install(uninstall=True)
        user, codex = read(self.user), read(self.codex)
        for event in ih.EVENTS:
            self.assertEqual(bridge_cmds(user, event), [])
            self.assertEqual(bridge_cmds(codex, event), [])
        self.assertIn("echo AUTO-INVOKE", json.dumps(user))
        self.assertIn("herdr-agent-state.ps1", json.dumps(codex))

    def test_malformed_json_is_reported_and_never_overwritten(self):
        self.codex.write_text("{not json", encoding="utf-8")
        lines = self.run_install()
        self.assertEqual(self.codex.read_text(encoding="utf-8"), "{not json")
        self.assertTrue(any(line.startswith("ERROR:") and "hooks.json" in line for line in lines))

    def test_missing_codex_dir_is_skipped_not_created(self):
        for p in (self.home / ".codex").iterdir():
            p.unlink()
        (self.home / ".codex").rmdir()
        lines = self.run_install()
        self.assertFalse((self.home / ".codex").exists())
        self.assertTrue(any("Codex not installed" in line for line in lines))
