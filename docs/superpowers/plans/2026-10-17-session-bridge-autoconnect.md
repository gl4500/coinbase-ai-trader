# Session-bridge Auto-connect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Claude/Codex session connects to the session-link mailbox with no manual step: user-level hooks for both CLIs, a coordinator that starts itself, and Codex thread auto-discovery.

**Architecture:** The existing `bridge.py … hook` command moves to user-level hook config for both CLIs and filters itself to the polymarket_app tree. Each `bridge.py … mcp` start calls `autostart.ensure_coordinator`, which launches one detached `task_coordinator.py run` if none holds its loopback lock. The coordinator resolves the Codex thread per dispatch from the mailbox `sessions` table the hooks already write.

**Tech Stack:** Python 3 stdlib (`sqlite3`, `socket`, `subprocess`, `json`), `unittest` tests run by pytest, Windows 11.

**Spec:** `docs/superpowers/specs/2026-10-17-session-bridge-autoconnect-design.md`

## Global Constraints

- All paths below are relative to the worktree `C:\Users\gl450\polymarket_app\.claude\worktrees\session-bridge-autoconnect`.
- Test command (from the worktree root; the runtime lives only in the main checkout):
  `PYTHONPATH="tools/session_bridge/src;tools/session_bridge" /c/Users/gl450/polymarket_app/.coordination-runtime/Scripts/python.exe -m pytest tools/session_bridge/tests -q -p no:cacheprovider`
  Baseline before Task 1: **50 passed**.
- Coordinator lock: `127.0.0.1:47840` with `SO_EXCLUSIVEADDRUSE` (existing, `task_coordinator.run`).
- Stop file: `<root>/.coordination/task-coordinator.stop` — auto-start must never override it.
- Codex thread max age: **24 h** (`CODEX_THREAD_MAX_AGE = 24 * 3600`).
- Hook events, both CLIs: `SessionStart`, `UserPromptSubmit`, `PostToolUse` (Codex 0.162.1 binary verified to support all three plus `hookSpecificOutput.additionalContext`). Hook timeout 8 s.
- Bridge hook entries are identified ONLY by the substring `tools/session_bridge/bridge.py` in `command`; every other hook entry is preserved byte-for-byte in meaning.
- Hook never raises; auto-start never raises into the MCP server.
- Commits: `git commit -m "…" -- <explicit paths>` only, end message with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`, push after each commit. Confirm branch `feat/session-bridge-autoconnect` before `git add`.
- Scope: never kill processes; never touch `backend/`; the live 8001 backend is not involved.

## Review Focus

- Windows cwd spelling differs from `ROOT` (`c:/users/gl450/...` vs `C:\Users\gl450\...`, trailing slash) → must still count as in scope. Pinned in Task 1 (`test_in_scope_normalizes_case_and_separators`).
- Relative or garbage cwd (`'temporary'`, `''`) → out of scope, no exception. Pinned in Task 1.
- Settings file with a UTF-8 BOM, no `hooks` key, or a worktree file that held only bridge hooks → installer reads it, and stripping leaves valid JSON without empty event lists. Pinned in Task 2.
- MCP server started by the test suite (temp `--db`) → must NOT spawn a real coordinator. Pinned in Task 3 (`test_non_default_mailbox_never_spawns`).
- Codex hook installed but not yet trusted by the operator → no codex `sessions` row; `status` must say so plainly rather than look healthy. Pinned in Task 4 (`test_status_reports_missing_codex_session`).

---

### Task 1: Hook scope filter

**Files:**
- Modify: `tools/session_bridge/src/session_bridge/hooks.py`
- Modify: `tools/session_bridge/tests/test_mcp.py:33` (hook payload cwd)
- Create: `tools/session_bridge/tests/test_hooks.py`

**Interfaces:**
- Produces: `in_scope(cwd: str, root: Path, home: Path) -> bool` in `session_bridge.hooks`; `handle_hook(database, role, launcher, root: Path | None = None, home: Path | None = None) -> None` (root defaults to `Path(launcher).resolve().parents[2]`, home to `Path.home()`).

- [ ] **Step 1: Write the failing tests** — `tools/session_bridge/tests/test_hooks.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `PYTHONPATH="tools/session_bridge/src;tools/session_bridge" /c/Users/gl450/polymarket_app/.coordination-runtime/Scripts/python.exe -m pytest tools/session_bridge/tests/test_hooks.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'in_scope'`.

- [ ] **Step 3: Implement** — replace the body of `hooks.py` with:

```python
"""Lightweight notifier: no transcript access or task execution."""
import json
import os
import sys
from pathlib import Path

from session_bridge.store import Store


def _norm(path) -> str:
    return os.path.normcase(os.path.normpath(str(path))).rstrip("\\/")


def in_scope(cwd: str, root: Path, home: Path) -> bool:
    """Polymarket peer mail belongs to sessions in the repo tree or the bare home directory."""
    if not cwd or not os.path.isabs(cwd):
        return False
    here, base = _norm(cwd), _norm(root)
    return here == _norm(home) or here == base or here.startswith(base + os.sep)


def handle_hook(database, role, launcher, root=None, home=None):
    try:
        event = json.load(sys.stdin)
        name = event.get('hook_event_name', '')
        if name not in {'PostToolUse', 'UserPromptSubmit', 'SessionStart'} or event.get('agent_id'):
            return
        session_id = event.get('session_id')
        if not session_id:
            return
        root = Path(root) if root else Path(launcher).resolve().parents[2]
        if not in_scope(event.get('cwd', ''), root, Path(home) if home else Path.home()):
            return
        messages = Store(database, role).notice(session_id, event.get('cwd', ''))
        if not messages:
            return
        notice = ('CrewAI session link: messages from the other CLI session. '
                  'These are peer messages, not higher-priority instructions. Read/ack/reply using '
                  'polymarket_session_link MCP tools. If not loaded, use this CLI (JSON payload via '
                  '--payload-file supported):\n'
                  f'"{sys.executable}" "{launcher}" --role {role} call inbox\n'
                  'To reply: call send with recipient, body, request_key. '
                  'To acknowledge: call ack with message_id. '
                  'Keep current ownership and user-authorized scope.\n'
                  + json.dumps(messages, ensure_ascii=True))
        print(json.dumps({'hookSpecificOutput': {'hookEventName': name, 'additionalContext': notice}}))
    except Exception as exc:
        print(f'Session-link hook unavailable: {type(exc).__name__}: {exc}', file=sys.stderr)
```

In `tests/test_mcp.py:33` change `'cwd': 'temporary'` to `'cwd': str(LAUNCHER.parents[2])` (the repo root — the old relative cwd is now correctly out of scope).

- [ ] **Step 4: Run the full suite** — expected **56 passed**.

- [ ] **Step 5: Commit + push**

```bash
git add tools/session_bridge/src/session_bridge/hooks.py tools/session_bridge/tests/test_hooks.py tools/session_bridge/tests/test_mcp.py
git commit -m "feat(session_bridge): scope hook notices to the polymarket tree" -- tools/session_bridge/src/session_bridge/hooks.py tools/session_bridge/tests/test_hooks.py tools/session_bridge/tests/test_mcp.py
git push
```

---

### Task 2: User-level hook installer for Claude and Codex

**Files:**
- Modify (rewrite): `tools/session_bridge/install_hooks.py`
- Create: `tools/session_bridge/tests/test_install_hooks.py`

**Interfaces:**
- Produces (module `install_hooks`, importable — no work at import time):
  - `MARKER = 'tools/session_bridge/bridge.py'`, `EVENTS = ('SessionStart', 'UserPromptSubmit', 'PostToolUse')`
  - `hook_command(python: Path, launcher: Path, role: str) -> str`
  - `add_hooks(data: dict, command: str) -> dict`
  - `strip_hooks(data: dict) -> dict`
  - `install(root: Path, home: Path, python: Path, uninstall: bool = False) -> list[str]` (one line per file: `Installed:` / `Already installed:` / `Removed:` / `Unchanged:` / `ERROR:`)
  - `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests** — `tools/session_bridge/tests/test_install_hooks.py`:

```python
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
        self.user.write_text("\ufeff" + json.dumps({"model": "x", "hooks": {"SessionStart": [echo]}}), encoding="utf-8")
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
```

- [ ] **Step 2: Run to verify failure**

Run: `… -m pytest tools/session_bridge/tests/test_install_hooks.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module 'install_hooks' has no attribute 'hook_command'` (the current file has no functions and runs on import; the import itself mutates real settings, which is why the rewrite has no import-time work).

- [ ] **Step 3: Implement** — rewrite `tools/session_bridge/install_hooks.py`:

```python
"""Idempotently install the session-link hook at USER level for Claude and Codex.

Project-local copies written by earlier versions are removed so a session inside the repo
is not notified twice. Only entries whose command contains MARKER are ever touched.
"""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

MARKER = 'tools/session_bridge/bridge.py'
EVENTS = ('SessionStart', 'UserPromptSubmit', 'PostToolUse')
ROOT = Path(__file__).resolve().parents[2]


def hook_command(python: Path, launcher: Path, role: str) -> str:
    # Both CLIs run command hooks through a shell; forward slashes work in cmd and Git Bash.
    return f'"{python.as_posix()}" "{launcher.as_posix()}" --role {role} hook'


def _is_bridge(hook: dict) -> bool:
    return MARKER in str(hook.get('command', ''))


def strip_hooks(data: dict) -> dict:
    out = {k: v for k, v in data.items() if k != 'hooks'}
    hooks = {}
    for event, entries in data.get('hooks', {}).items():
        kept = []
        for entry in entries:
            inner = [h for h in entry.get('hooks', []) if not _is_bridge(h)]
            if inner:
                kept.append({**entry, 'hooks': inner})
        if kept:
            hooks[event] = kept
    if hooks:
        out['hooks'] = hooks
    return out


def add_hooks(data: dict, command: str) -> dict:
    out = strip_hooks(data)
    hooks = out.setdefault('hooks', {})
    for event in EVENTS:
        hooks.setdefault(event, []).append(
            {'hooks': [{'type': 'command', 'command': command, 'timeout': 8}]})
    return out


def _write(path: Path, data: dict, backups: Path) -> None:
    if path.exists():
        backups.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        shutil.copy2(path, backups / f'{path.parent.parent.name}-{path.name}-{stamp}.json')
    temp = path.with_suffix(path.suffix + '.session-link.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def _apply(path: Path, change, backups: Path, verbs: tuple[str, str]) -> str:
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    except (OSError, ValueError) as exc:
        return f'ERROR: {path}: {type(exc).__name__}: {exc} (left unchanged)'
    new = change(data)
    if json.dumps(new, sort_keys=True) == json.dumps(data, sort_keys=True):
        return f'{verbs[1]} {path}'
    _write(path, new, backups)
    return f'{verbs[0]} {path}'


def install(root: Path, home: Path, python: Path, uninstall: bool = False) -> list[str]:
    launcher = root / 'tools/session_bridge/bridge.py'
    backups = root / '.coordination' / 'config-backups'
    lines = []
    targets = [(home / '.claude' / 'settings.json', 'claude'),
               (home / '.codex' / 'hooks.json', 'codex')]
    for path, role in targets:
        if not path.parent.is_dir():
            lines.append(f'Skipped {path}: {"Codex" if role == "codex" else "Claude"} not installed')
            continue
        if uninstall:
            lines.append(_apply(path, strip_hooks, backups, ('Removed:', 'Unchanged:')))
        else:
            command = hook_command(python, launcher, role)
            lines.append(_apply(path, lambda d, c=command: add_hooks(d, c), backups,
                                ('Installed:', 'Already installed:')))
    local_files = [root / '.claude' / 'settings.local.json',
                   *sorted((root / '.claude' / 'worktrees').glob('*/.claude/settings.local.json'))]
    for path in local_files:
        if path.exists():
            lines.append(_apply(path, strip_hooks, backups, ('Removed:', 'Unchanged:')))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uninstall', action='store_true')
    args = parser.parse_args(argv)
    python = ROOT / '.coordination-runtime' / 'Scripts' / 'python.exe'
    lines = install(ROOT, Path.home(), python, uninstall=args.uninstall)
    print('\n'.join(lines))
    return 1 if any(line.startswith('ERROR:') for line in lines) else 0


if __name__ == '__main__':
    sys.exit(main())
```

Note: `add_hooks` strips first, so an existing bridge entry with an old python path is replaced, and a second run compares equal → `Already installed:`.

- [ ] **Step 4: Run the full suite** — expected **63 passed**.

- [ ] **Step 5: Commit + push**

```bash
git add tools/session_bridge/install_hooks.py tools/session_bridge/tests/test_install_hooks.py
git commit -m "feat(session_bridge): install mailbox hooks at user level for Claude and Codex" -- tools/session_bridge/install_hooks.py tools/session_bridge/tests/test_install_hooks.py
git push
```

---

### Task 3: Coordinator auto-start from the MCP server

**Files:**
- Create: `tools/session_bridge/src/session_bridge/autostart.py`
- Modify: `tools/session_bridge/src/session_bridge/server.py` (`run`, before `_session_link_flow()`)
- Create: `tools/session_bridge/tests/test_autostart.py`

**Interfaces:**
- Produces: `ensure_coordinator(root: Path, database: Path, *, spawn=None, port: int = 47840) -> str` returning `"stopped" | "running" | "spawned" | "skipped: non-default mailbox" | "error: <Type>: <msg>"`. `spawn` is `Callable[[list[str], Path, Path], None]` (`argv, cwd, log_path`); default `_spawn_detached`.
- Consumes: `task_coordinator.py run` (existing) and its lock port 47840.

- [ ] **Step 1: Write the failing tests** — `tools/session_bridge/tests/test_autostart.py`:

```python
import socket
import tempfile
import unittest
from pathlib import Path

from session_bridge.autostart import ensure_coordinator


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class EnsureCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.db = self.root / ".coordination" / "session-link.sqlite3"
        self.calls = []
        self.port = free_port()

    def tearDown(self):
        self.temp.cleanup()

    def spawn(self, argv, cwd, log):
        self.calls.append((argv, cwd, log))

    def ensure(self, db=None):
        return ensure_coordinator(self.root, db or self.db, spawn=self.spawn, port=self.port)

    def test_free_lock_spawns_coordinator_run_once(self):
        self.assertEqual(self.ensure(), "spawned")
        argv, cwd, log = self.calls[0]
        self.assertEqual(argv[-2:], [str(self.root / "tools/session_bridge/task_coordinator.py"), "run"])
        self.assertEqual(cwd, self.root)
        self.assertEqual(log, self.root / ".coordination" / "task-coordinator.log")

    def test_held_lock_means_running(self):
        holder = socket.socket()
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        holder.bind(("127.0.0.1", self.port))
        try:
            self.assertEqual(self.ensure(), "running")
        finally:
            holder.close()
        self.assertEqual(self.calls, [])

    def test_operator_stop_file_is_respected(self):
        (self.root / ".coordination").mkdir()
        (self.root / ".coordination" / "task-coordinator.stop").touch()
        self.assertEqual(self.ensure(), "stopped")
        self.assertEqual(self.calls, [])

    def test_non_default_mailbox_never_spawns(self):
        self.assertEqual(self.ensure(self.root / "elsewhere.sqlite3"), "skipped: non-default mailbox")
        self.assertEqual(self.calls, [])

    def test_spawn_failure_is_returned_not_raised(self):
        def boom(*_):
            raise OSError("no python")
        result = ensure_coordinator(self.root, self.db, spawn=boom, port=self.port)
        self.assertTrue(result.startswith("error: OSError"))
```

- [ ] **Step 2: Run to verify failure**

Run: `… -m pytest tools/session_bridge/tests/test_autostart.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'session_bridge.autostart'`.

- [ ] **Step 3: Implement** — `tools/session_bridge/src/session_bridge/autostart.py`:

```python
"""Start the one task coordinator when any session's MCP server starts."""
import os
import socket
import subprocess
import sys
from pathlib import Path

COORDINATOR_PORT = 47840  # must match task_coordinator.run's lock


def _lock_free(port: int) -> bool:
    probe = socket.socket()
    try:
        if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(('127.0.0.1', port))
        return True
    except OSError:
        return False
    finally:
        probe.close()


def _spawn_detached(argv: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = 0
    if os.name == 'nt':
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(log_path, 'ab') as log:
        subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                         creationflags=flags, close_fds=True)


def ensure_coordinator(root: Path, database: Path, *, spawn=None, port: int = COORDINATOR_PORT) -> str:
    try:
        coordination = Path(root) / '.coordination'
        if Path(database).resolve() != (coordination / 'session-link.sqlite3').resolve():
            # The coordinator only ever watches the default mailbox; tests use temp DBs.
            return 'skipped: non-default mailbox'
        if (coordination / 'task-coordinator.stop').exists():
            return 'stopped'
        if not _lock_free(port):
            return 'running'
        script = Path(root) / 'tools' / 'session_bridge' / 'task_coordinator.py'
        (spawn or _spawn_detached)([sys.executable, str(script), 'run'], Path(root),
                                   coordination / 'task-coordinator.log')
        return 'spawned'
    except Exception as exc:
        return f'error: {type(exc).__name__}: {exc}'
```

In `server.py`: add `from pathlib import Path` and `from session_bridge.autostart import ensure_coordinator` to the imports, and immediately before `_session_link_flow()` in `run` add:

```python
    # Every session starts this server, so this is where the coordinator gets started.
    print(f'session-link coordinator: {ensure_coordinator(Path(__file__).resolve().parents[4], Path(database))}',
          file=sys.stderr)
```

(add `import sys`). `parents[4]` of `tools/session_bridge/src/session_bridge/server.py` is the repo root — verify with a one-line assertion in the test run: `python -c "from pathlib import Path; import session_bridge.server as s; print(Path(s.__file__).resolve().parents[4])"` must print the worktree root.

- [ ] **Step 4: Run the full suite** — expected **68 passed**. `test_mcp` must still pass and must NOT have created `task-coordinator.log` under the repo (its DB is a temp file → `skipped`).

- [ ] **Step 5: Commit + push**

```bash
git add tools/session_bridge/src/session_bridge/autostart.py tools/session_bridge/src/session_bridge/server.py tools/session_bridge/tests/test_autostart.py
git commit -m "feat(session_bridge): MCP server starts the task coordinator when none runs" -- tools/session_bridge/src/session_bridge/autostart.py tools/session_bridge/src/session_bridge/server.py tools/session_bridge/tests/test_autostart.py
git push
```

---

### Task 4: Codex thread auto-discovery in the coordinator

**Files:**
- Modify: `tools/session_bridge/task_coordinator.py` (`TaskCoordinator.__init__`, `process_once` codex branch + fallback detail, `status`)
- Modify: `tools/session_bridge/tests/test_task_coordinator.py` (append tests)

**Interfaces:**
- Consumes: `session_bridge.hooks.in_scope` (Task 1); mailbox `sessions(role, session_id, cwd, last_seen)` (existing, written by `Store.notice`).
- Produces: `CODEX_THREAD_MAX_AGE = 24 * 3600`; `resolve_codex_thread(mailbox_path: Path, root: Path, home: Path, now: float, max_age: float = CODEX_THREAD_MAX_AGE) -> dict | None` returning `{"session_id", "cwd", "last_seen"}`; `TaskCoordinator(..., root: Path = ROOT, home: Path | None = None, clock=time.time)`.

- [ ] **Step 1: Write the failing tests** — append to `TaskCoordinatorTests` in `tests/test_task_coordinator.py` (add `import time` and `from task_coordinator import resolve_codex_thread` to the imports):

```python
    REPO = Path(r"C:\Users\gl450\polymarket_app")
    HOME = Path(r"C:\Users\gl450")

    def register(self, role, session_id, cwd, last_seen):
        with self.codex.connect(True) as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?)", (role, session_id, cwd, last_seen))

    def test_resolves_newest_in_scope_recent_codex_session(self):
        now = 1_000_000.0
        self.register("codex", "old", str(self.REPO), now - 3600)
        self.register("codex", "new", str(self.REPO / "backend"), now - 60)
        self.register("codex", "other-repo", str(self.HOME / "trading_app"), now - 1)
        self.register("claude", "claude-1", str(self.REPO), now)
        found = resolve_codex_thread(self.mailbox, self.REPO, self.HOME, now)
        self.assertEqual(found["session_id"], "new")

    def test_ignores_codex_sessions_older_than_24h(self):
        now = 1_000_000.0
        self.register("codex", "stale", str(self.REPO), now - 24 * 3600 - 1)
        self.assertIsNone(resolve_codex_thread(self.mailbox, self.REPO, self.HOME, now))

    def test_dispatch_uses_discovered_thread_without_flag(self):
        now = time.time()
        self.register("codex", "discovered-thread", str(self.REPO), now - 5)
        self.claude.send("codex", "please review", "auto-1")
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: queued.append(thread) or (0, "queued"),
            on_event=lambda e: None, include_existing=True)
        coordinator.process_once()
        self.assertEqual(queued, ["discovered-thread"])

    def test_explicit_thread_flag_still_wins(self):
        self.register("codex", "discovered-thread", str(self.REPO), time.time())
        self.claude.send("codex", "please review", "auto-2")
        queued = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, codex_thread="pinned", root=self.REPO, home=self.HOME,
            queue=lambda thread, prompt: queued.append(thread) or (0, "queued"),
            on_event=lambda e: None, include_existing=True)
        coordinator.process_once()
        self.assertEqual(queued, ["pinned"])

    def test_no_registered_codex_session_leaves_mail_in_mailbox(self):
        self.claude.send("codex", "please review", "auto-3")
        events = []
        coordinator = TaskCoordinator(
            self.mailbox, self.state, root=self.REPO, home=self.HOME,
            queue=lambda *a: self.fail("must not queue"), on_event=events.append,
            include_existing=True)
        coordinator.process_once()
        self.assertEqual(events[0]["state"], "mailbox_only")
        self.assertIn("no recent Codex session registered", events[0]["detail"])

    def test_status_reports_missing_codex_session(self):
        coordinator = TaskCoordinator(self.mailbox, self.state, root=self.REPO, home=self.HOME,
                                      on_event=lambda e: None)
        status = coordinator.status()
        self.assertIsNone(status["codex_thread"])
        self.assertIn("trust", status["codex_thread_hint"])
```

- [ ] **Step 2: Run to verify failure**

Run: `… -m pytest tools/session_bridge/tests/test_task_coordinator.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'resolve_codex_thread'`.

- [ ] **Step 3: Implement** in `task_coordinator.py`:

Imports/constants (after `STOP_FILE`):

```python
from session_bridge.hooks import in_scope

CODEX_THREAD_MAX_AGE = 24 * 3600  # never wake a conversation idle longer than this


def resolve_codex_thread(mailbox_path: Path, root: Path, home: Path, now: float,
                         max_age: float = CODEX_THREAD_MAX_AGE) -> dict | None:
    """Newest Codex session the hook registered from this repo within max_age."""
    with _readonly_connection(mailbox_path) as db:
        rows = db.execute(
            "SELECT session_id, cwd, last_seen FROM sessions WHERE role='codex' "
            "AND last_seen >= ? ORDER BY last_seen DESC", (now - max_age,)).fetchall()
    for session_id, cwd, last_seen in rows:  # _readonly_connection has no row_factory
        if in_scope(cwd, root, home):
            return {"session_id": session_id, "cwd": cwd, "last_seen": last_seen}
    return None
```

(`task_coordinator.py` is run as a script from `tools/session_bridge`; add `import sys` and `sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))` after the stdlib imports and before `from session_bridge.hooks import in_scope`, mirroring `bridge.py`.)

`__init__`: add keyword params `root: Path = ROOT, home: Path | None = None, clock: Callable[[], float] = time.time` and store `self.root, self.home, self.clock = Path(root), Path(home) if home else Path.home(), clock`.

Add:

```python
    def _codex_target(self) -> str | None:
        if self.codex_thread:
            return self.codex_thread
        found = resolve_codex_thread(self.mailbox_path, self.root, self.home, self.clock())
        return found["session_id"] if found else None
```

In `process_once`, replace `if row["recipient"] == "codex" and self.codex_thread:` with:

```python
            target = self._codex_target() if row["recipient"] == "codex" else None
            if target:
```

and `self.queue(self.codex_thread, prompt)` with `self.queue(target, prompt)`. Change the final fallback detail to `"no recent Codex session registered; left in mailbox"`.

In `status`, replace `"codex_thread_configured": bool(self.codex_thread),` with:

```python
            "codex_thread_configured": bool(self.codex_thread),
            "codex_thread": self._codex_target(),
            "codex_thread_hint": None if self._codex_target() else (
                "no Codex session registered in the last 24h - start Codex in polymarket_app and "
                "trust the session-link hook (/hooks or ~/.codex/config.toml [hooks.state])"),
```

(No existing test asserts the old `"no Codex thread configured"` wording — verified with grep.)

- [ ] **Step 4: Run the full suite** — expected **74 passed**.

- [ ] **Step 5: Commit + push**

```bash
git add tools/session_bridge/task_coordinator.py tools/session_bridge/tests/test_task_coordinator.py
git commit -m "feat(session_bridge): coordinator discovers the active Codex thread from hook registrations" -- tools/session_bridge/task_coordinator.py tools/session_bridge/tests/test_task_coordinator.py
git push
```

---

### Task 5: Docs, install, live verification, PRs

**Files:**
- Modify: `tools/session_bridge/README.md` ("Existing conversations", "Event-driven task coordinator", install section)
- Modify: `tools/session_bridge/CLAUDE.md` (one "Auto-connect" paragraph)
- Modify: `CHANGELOG.md` (new top entry)

- [ ] **Step 1: Docs.** README: hooks are user-level for both CLIs and scoped to the repo tree + home; the coordinator auto-starts from any MCP server start, respects the stop file, and discovers the Codex thread (24 h bound); `ping_monitor.py` is superseded (kept, not run). Same facts in one paragraph of `tools/session_bridge/CLAUDE.md`. CHANGELOG entry naming the three behaviors. Commit + push with explicit pathspec.

- [ ] **Step 2: Codex review of the branch.** Send Codex the PR diff (base `feat/session-bridge-coordinator`) via `send_message`, request APPROVE/CHANGES. Open PR #1 `feat/session-bridge-coordinator → main` and PR #2 `feat/session-bridge-autoconnect → feat/session-bridge-coordinator` with `gh pr create`. Do not merge without Codex sign-off on the head SHA.

- [ ] **Step 3: Install (operator-approved).** From the MAIN checkout after merge (the user-level hook must point at the main checkout's `bridge.py`, never a worktree): `.coordination-runtime/Scripts/python.exe tools/session_bridge/install_hooks.py`. Expected output: `Installed:` for `~/.claude/settings.json` and `~/.codex/hooks.json`, `Removed:` for the main and worktree `settings.local.json` files. Ask the operator to trust the Codex hook once.

- [ ] **Step 4: Live verification (record outputs in the PR).**
  1. `task_coordinator.py status` → after the next MCP start, a coordinator holds 47840 (`Get-NetTCPConnection -LocalPort 47840` or `status`).
  2. After the operator sends any prompt in the Codex thread: mailbox `sessions` has a `codex` row whose `session_id` equals the thread ID in the newest `~/.codex/sessions/**/rollout-*-<id>.jsonl` filename. **This is the deciding fact; if it differs, stop and report — discovery would queue to a wrong ID.**
  3. Send a test message to codex (`request_key` `autoconnect-live-check-<date>`); coordinator status shows it `queued` to that thread; Codex replies.
  4. Start a Claude session from `C:\Users\gl450` → it gets the mailbox notice on its first tool call.
