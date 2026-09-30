"""Idempotently add project-local Claude notifications, preserving existing settings."""
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
launcher = ROOT / 'tools/session_bridge/bridge.py'
python = ROOT / '.coordination-runtime/Scripts/python.exe'
# Claude uses a shell for command hooks; forward slashes work in Windows and Git Bash.
command = f'"{python.as_posix()}" "{launcher.as_posix()}" --role claude hook'
backups = ROOT / '.coordination' / 'config-backups'
for project in [ROOT, ROOT / '.claude/worktrees/outcome-labels']:
    if not project.is_dir():
        continue
    path = project / '.claude/settings.local.json'
    data = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    original = json.dumps(data, sort_keys=True)
    for event in ['PostToolUse', 'UserPromptSubmit', 'SessionStart']:
        entries = data.setdefault('hooks', {}).setdefault(event, [])
        if not any(h.get('command') == command for e in entries for h in e.get('hooks', [])):
            entries.append({'hooks': [{'type': 'command', 'command': command, 'timeout': 8}]})
    if json.dumps(data, sort_keys=True) == original:
        print(f'Already installed: {path}')
        continue
    if path.exists():
        backups.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        shutil.copy2(path, backups / f'{project.name}-{stamp}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.session-link.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    print(f'Installed: {path}')
