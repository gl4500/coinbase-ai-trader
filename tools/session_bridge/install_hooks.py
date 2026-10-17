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


def _shell_arg(path: Path) -> str:
    # A leading quoted path is a parse error in PowerShell, so quote only when required.
    text = path.as_posix()
    return f'"{text}"' if any(c.isspace() for c in text) else text


def hook_command(python: Path, launcher: Path, role: str) -> str:
    # Forward slashes work in cmd, Git Bash and PowerShell.
    return f'{_shell_arg(python)} {_shell_arg(launcher)} --role {role} hook'


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
    # Strip first so a stale bridge entry (old interpreter path) is replaced, not duplicated.
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
    parser.add_argument('--force', action='store_true', help='allow running from a worktree')
    args = parser.parse_args(argv)
    if '/.claude/worktrees/' in ROOT.as_posix() + '/' and not args.force:
        # User-level hooks would point every session on the machine at this worktree.
        print(f'ERROR: {ROOT} is a worktree; run install_hooks.py from the main checkout')
        return 2
    python = ROOT / '.coordination-runtime' / 'Scripts' / 'python.exe'
    lines = install(ROOT, Path.home(), python, uninstall=args.uninstall)
    print('\n'.join(lines))
    return 1 if any(line.startswith('ERROR:') for line in lines) else 0


if __name__ == '__main__':
    sys.exit(main())
