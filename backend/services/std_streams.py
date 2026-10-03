"""Give a console-less process (pythonw) real stdout/stderr before anything logs.

The logon autostart task (`CoinbaseTraderBackend`) runs `pythonw.exe main.py`, which starts with
`sys.stdout` and `sys.stderr` set to None. uvicorn's default log formatter calls
`sys.stdout.isatty()`, so the backend exited 1 before writing a single log line.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, TextIO


def ensure_std_streams(log_path: "os.PathLike[str] | str") -> Optional[TextIO]:
    """Point any missing std stream at an append-only, line-buffered file. Returns that file,
    or None when both streams already exist (python.exe, the launcher) and nothing changed."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    os.makedirs(os.path.dirname(os.fspath(log_path)) or ".", exist_ok=True)
    stream = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream
    return stream
