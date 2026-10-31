"""Exclusive writer lock on the output root: two recorders must never write the same archive.
The OS releases the lock if the process dies, so a crash never leaves a stale lock."""

from __future__ import annotations

import os
from pathlib import Path


class WriterLock:
    def __init__(self, root: Path):
        self.path = Path(root) / ".writer.lock"
        self._fh = None

    def acquire(self) -> "WriterLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            raise RuntimeError(f"another recorder holds {self.path}: {exc!r}") from exc
        self._fh = fh
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self._fh.close()
            self._fh = None
