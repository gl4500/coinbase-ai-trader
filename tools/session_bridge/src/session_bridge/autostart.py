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
