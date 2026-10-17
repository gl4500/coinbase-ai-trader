"""Give a console-less process (pythonw) real stdout/stderr before anything logs.

The logon autostart task (`CoinbaseTraderBackend`) runs `pythonw.exe main.py`, which starts with
`sys.stdout` and `sys.stderr` set to None. uvicorn's default log formatter calls
`sys.stdout.isatty()`, so the backend exited 1 before writing a single log line.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO

MAX_BYTES = 10_000_000  # rotate at startup above this; one previous file is kept


def _rotate_if_large(path: str, max_bytes: int) -> None:
    """Startup-only retention: the open handle lives for the whole process, so the file can
    only be rotated before it is opened. If another process holds it (Windows refuses the
    rename), keep appending rather than fail startup."""
    try:
        if os.path.getsize(path) > max_bytes:
            os.replace(path, path + ".1")
    except OSError:
        pass


def ensure_std_streams(
    log_path: "os.PathLike[str] | str", max_bytes: int = MAX_BYTES
) -> Optional[TextIO]:
    """Point any missing std stream at an append-only, line-buffered file. Returns that file,
    or None when both streams already exist (python.exe, the launcher) and nothing changed."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    os.makedirs(os.path.dirname(os.fspath(log_path)) or ".", exist_ok=True)
    _rotate_if_large(os.fspath(log_path), max_bytes)
    stream = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return stream
