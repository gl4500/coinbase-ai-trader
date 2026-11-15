"""PreToolUse guard for Claude Code and Codex: protect a live backend from agent shell commands.

Standard library only, so any Python 3.8+ interpreter can run it. Both agents send the same
payload on stdin ({"tool_name": "Bash" | "PowerShell", "tool_input": {"command": ...}, "cwd": ...}),
and both treat exit code 2 with a reason on stderr as "block this tool call".

While the live backend port is listening or a training process runs, it blocks:
  * test runs (pytest, run_tests.py, python -m unittest), unless the operator opened a test window;
  * commits with staged .py files, where the repo's pre-commit hook then runs tests (same window rule);
  * process kills (Stop-Process, taskkill, kill, pkill, killall), window or not;
  * backend starts and restarts while training runs (startup_guard already refuses an occupied port).

A test window is a file `.agent-test-window` at the repository root holding an ISO-8601 expiry with a
timezone, e.g. `2026-11-14T22:00:00-05:00`. It is gitignored and written by the operator.

Failure policy: if a check needed for a risky command fails, the command is blocked. A payload that
cannot be parsed is let through with a warning, so a malformed event never locks an agent out.
This is a guardrail against mistakes, not a security boundary: commands can be phrased to evade it.
"""
from __future__ import annotations

import argparse
import errno
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, FrozenSet, Iterable, Optional, Sequence, TextIO, Tuple

WINDOW_FILENAME = ".agent-test-window"
SHELL_TOOLS = ("Bash", "PowerShell")

_PATTERNS = {
    "test": re.compile(r"\bpytest\b|\brun_tests\.py\b|-m\s+unittest\b", re.IGNORECASE),
    "commit": re.compile(r"\bgit\b[^|;&]*?\bcommit\b"),
    "kill": re.compile(r"\bStop-Process\b|\btaskkill\b|\bpkill\b|\bkillall\b|(?:^|[\s;&|(])kill\s+-?\w",
                       re.IGNORECASE),
    # Direct starts plus the repositories' own launchers (start_backend.ps1/.bat, start_app.py,
    # start.bat, launcher.py). start_frontend.* is deliberately not a backend start.
    "restart": re.compile(r"python\S*\s+(?:-\S+\s+)*\S*main\.py\b|\buvicorn\b|\bstartup_guard\b|\s-Replace\b"
                          r"|\bstart_backend\.(?:ps1|bat|cmd)\b|\bstart_app\.py\b|(?:^|[\\/\s])start\.bat\b"
                          r"|\blauncher\.py\b", re.IGNORECASE),
}

# Splits a shell line into commands; any of these separators may hide a second git operation.
_SEGMENT_SPLIT = re.compile(r"&&|\|\||[;|\n]")
_DIR_CHANGE = re.compile(r"(?:^|\s)(?:cd|pushd|chdir|Set-Location|sl)(?:\s|$)", re.IGNORECASE)
_INDEX_CHANGE = re.compile(r"\bgit\b[^|;&]*?\b(?:add|stage|rm|mv|restore|apply|checkout|stash|reset)\b")
_COMMIT_OPTS_WITH_VALUE = frozenset({
    "-m", "--message", "-F", "--file", "-C", "--reuse-message", "-c", "--reedit-message", "--author",
    "--date", "-t", "--template", "--fixup", "--squash", "--cleanup", "--trailer", "--pathspec-from-file"})
_COMMIT_STAGING_LONG = frozenset({"--all", "--include", "--interactive", "--patch", "--pathspec-from-file"})
_COMMIT_STAGING_SHORT = frozenset("aip")


def _pathspec_may_be_python(path: str) -> bool:
    if any(ch in path for ch in "*?["):
        return True
    ext = os.path.splitext(os.path.basename(path.rstrip("/")))[1].lower()
    return ext in ("", ".py", ".pyw")  # no extension: could be a directory


def _commit_segment_may_include_py(segment: str, staged_py: Callable[[str], bool]) -> bool:
    try:
        tokens = shlex.split(segment.replace("\\", "/"))
    except ValueError:
        return True  # unbalanced quotes: cannot tell
    git_at = next((i for i, t in enumerate(tokens) if os.path.basename(t).lower() in ("git", "git.exe")), None)
    if git_at is None or "commit" not in tokens[git_at:]:
        return True
    target, i = ".", git_at + 1
    while tokens[i] != "commit":  # global options between `git` and `commit`
        if tokens[i] == "-C" and i + 1 < len(tokens):
            target = tokens[i + 1]
            i += 1
        elif tokens[i] == "-c":
            i += 1
        i += 1
    paths, i, after_dd = [], i + 1, False
    while i < len(tokens):
        t = tokens[i]
        if after_dd or not t.startswith("-") or t == "-":
            paths.append(t)
        elif t == "--":
            after_dd = True
        elif t.split("=", 1)[0] in _COMMIT_STAGING_LONG:
            return True
        elif t in _COMMIT_OPTS_WITH_VALUE:
            i += 1
        elif not t.startswith("--"):
            flags = t[1:]
            value_at = min((flags.find(c) for c in "mFCct" if c in flags), default=-1)
            leading = flags if value_at < 0 else flags[:value_at]
            if set(leading) & _COMMIT_STAGING_SHORT:
                return True
            if value_at >= 0 and value_at == len(flags) - 1:
                i += 1  # e.g. -qm "msg": the message is the next token
        i += 1
    return staged_py(target) or any(_pathspec_may_be_python(p) for p in paths)


def commit_may_include_py(command: str, staged_py: Callable[[str], bool]) -> bool:
    """Conservatively decide whether a command's commit can contain .py files (which makes the
    pre-commit hook run tests). Only a plain commit of explicit non-Python files with nothing
    Python staged, in a command that neither changes directory nor touches the index, is harmless."""
    segments = [s.strip() for s in _SEGMENT_SPLIT.split(command) if s.strip()]
    commits = [s for s in segments if _PATTERNS["commit"].search(s)]
    if not commits:
        return False
    if any(_DIR_CHANGE.search(s) or _INDEX_CHANGE.search(s) for s in segments):
        return True
    return any(_commit_segment_may_include_py(s, staged_py) for s in commits)


@dataclass(frozen=True)
class Config:
    port: int
    training_markers: Tuple[str, ...]
    commit_needs_staged_py: bool


@dataclass(frozen=True)
class Decision:
    block: bool
    reason: str = ""


def classify(command: str) -> FrozenSet[str]:
    """Return the risk categories a shell command falls into (empty if harmless)."""
    return frozenset(name for name, pattern in _PATTERNS.items() if pattern.search(command))


def decide(
    command: str,
    cfg: Config,
    *,
    port_open: Callable[[int], bool],
    training_running: Callable[[Tuple[str, ...]], bool],
    staged_py: Callable[[str], bool],
    window_open: Callable[[], bool],
) -> Decision:
    """Decide whether to block a command. Probes are only called for risky commands.
    `staged_py(target)` reports staged .py files in the repo at `target` (relative to the event cwd)."""
    risks = classify(command)
    if not risks:
        return Decision(False)
    try:
        if "commit" in risks and not (cfg.commit_needs_staged_py and commit_may_include_py(command, staged_py)):
            risks = risks - {"commit"}
        if not risks:
            return Decision(False)
        live = port_open(cfg.port)
        training = bool(cfg.training_markers) and training_running(cfg.training_markers)
        window = window_open() if (risks & {"test", "commit"}) and (live or training) else False
    except Exception as exc:  # fail closed: a risky command we cannot check is blocked
        return Decision(True, f"live_guard: blocked; could not check the live backend ({exc}). Ask the operator.")

    state = " and ".join(part for part in (
        f"port {cfg.port} is live" if live else "",
        "a training process is running" if training else "",
    ) if part)
    if "kill" in risks and (live or training):
        return Decision(True, f"live_guard: blocked a process kill because {state}. Ask the operator first.")
    if "restart" in risks and training:
        return Decision(True, "live_guard: blocked a backend start/restart because a training process is "
                              "running. Wait for training to finish.")
    if (risks & {"test", "commit"}) and (live or training) and not window:
        what = "a test run" if "test" in risks else "a commit with staged .py files (the hook runs tests)"
        return Decision(True, f"live_guard: blocked {what} because {state}. Ask the operator for a test "
                              f"window: an ISO-8601 expiry with timezone in <repo>/{WINDOW_FILENAME}.")
    return Decision(False)


def window_is_open(path: str, now: datetime) -> bool:
    """True if `path` holds a timezone-aware ISO-8601 expiry later than `now`."""
    try:
        with open(path, encoding="utf-8") as fh:
            expiry = datetime.fromisoformat(fh.read().strip())
    except (OSError, ValueError):
        return False
    return expiry.tzinfo is not None and expiry > now


# Only an actively refused connection proves nothing is listening (POSIX ECONNREFUSED, WSAECONNREFUSED).
REFUSED_ERRNOS = frozenset({errno.ECONNREFUSED, 10061})


# Windows retries a refused localhost connect for about 2 s before reporting WSAECONNREFUSED; a shorter
# timeout returns WSAEWOULDBLOCK (10035), which is "unknown" and would block every risky command.
PROBE_TIMEOUT_SECS = 5.0


def _connect_ex(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(PROBE_TIMEOUT_SECS)
        return sock.connect_ex(("127.0.0.1", port))


def port_listening(port: int, connect: Callable[[int], int] = _connect_ex) -> bool:
    """True if something listens on 127.0.0.1:port, False if the connection is refused.
    Any other outcome (timeout, permission denied, ...) is unknown and raises, so risky commands block."""
    code = connect(port)
    if code == 0:
        return True
    if code in REFUSED_ERRNOS:
        return False
    raise OSError(code, f"port {port} probe returned error {code}")


ProcessRecord = Tuple[str, str]  # (executable name, full command line on one line)

# One record per line: "<name>\t<command line>", with newlines inside a command line flattened.
_PS_LIST = ("Get-CimInstance Win32_Process | ForEach-Object { "
            "$_.Name + \"`t\" + (($_.CommandLine -as [string]) -replace '[\\r\\n]+', ' ') }")


def parse_process_records(out: str) -> list:
    records = []
    for line in out.splitlines():
        name, _, cmd = line.partition("\t")
        if name:
            records.append((name.strip(), cmd.strip()))
    return records


def _process_records() -> list:
    if os.name == "nt":
        out = subprocess.run(["powershell", "-NoProfile", "-Command", _PS_LIST],
                             capture_output=True, text=True, timeout=15, check=True).stdout
    else:
        out = subprocess.run(["ps", "-eo", "comm=,args="], capture_output=True, text=True,
                             timeout=15, check=True).stdout
        out = "\n".join("\t".join(part.split(None, 1)) for part in out.splitlines() if part.strip())
    return parse_process_records(out)


def training_in(records: Iterable[ProcessRecord], markers: Tuple[str, ...]) -> bool:
    """True if a Python process names a training marker. Shells that merely mention the marker and
    this guard's own invocations (which carry `--training-marker <marker>`) do not count."""
    for name, cmd in records:
        if not name.lower().startswith("python") or "live_guard" in cmd:
            continue
        if any(m in cmd for m in markers):
            return True
    return False


def _training_running(markers: Tuple[str, ...]) -> bool:
    return training_in(_process_records(), markers)


def _repo_root(cwd: str) -> str:
    out = subprocess.run(["git", "-C", cwd, "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    return out or cwd


def _staged_py(cwd: str) -> bool:
    out = subprocess.run(["git", "-C", cwd, "diff", "--cached", "--name-only"],
                         capture_output=True, text=True, timeout=10, check=True).stdout
    return any(name.strip().endswith(".py") for name in out.splitlines())


def _default_probes(cwd: str) -> dict:
    return dict(
        port_open=port_listening,
        training_running=_training_running,
        staged_py=lambda target: _staged_py(os.path.join(cwd, target)),
        window_open=lambda: window_is_open(os.path.join(_repo_root(cwd), WINDOW_FILENAME),
                                           datetime.now(timezone.utc)),
    )


def _parse_args(argv: Sequence[str]) -> Config:
    p = argparse.ArgumentParser(description="PreToolUse live-backend guard")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--training-marker", action="append", default=[])
    p.add_argument("--commit-needs-staged-py", action="store_true")
    a = p.parse_args(list(argv))
    return Config(a.port, tuple(a.training_marker), a.commit_needs_staged_py)


def main(argv: Sequence[str], stdin: TextIO = sys.stdin, stderr: TextIO = sys.stderr,
         probes: Optional[dict] = None, stdout: TextIO = sys.stdout) -> int:
    cfg = _parse_args(argv)
    try:
        event = json.loads(stdin.read().lstrip("﻿") or "{}")  # Windows PowerShell pipes add a BOM
        tool = event.get("tool_name", "")
        command = (event.get("tool_input") or {}).get("command", "")
        cwd = event.get("cwd") or os.getcwd()
    except (ValueError, AttributeError) as exc:
        print(f"live_guard: could not read the hook payload ({exc}); allowing.", file=stderr)
        return 0
    if tool not in SHELL_TOOLS or not isinstance(command, str) or not command:
        return 0
    decision = decide(command, cfg, **(probes or _default_probes(cwd)))
    if decision.block:
        # Both signals: exit 2 + reason on stderr, and the JSON deny decision both agents accept on
        # stdout, so the block survives a shell wrapper that loses the exit code.
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                                 "permissionDecisionReason": decision.reason}}), file=stdout)
        print(decision.reason, file=stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
