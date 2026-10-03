"""Append-only, crash-safe segment store.

Each run writes its own immutable hourly segments:
`<root>/<stream>/<UTC-day>/<HH>00_<run_id>.jsonl.gz`. A segment is written as `.part` and
renamed with a sha256 sidecar only when cleanly finalised. A run never appends to another run's
file, so an unclean exit can damage only its own `.part` files. `recover_incomplete` keeps those
bytes, salvages the readable prefix and marks them `.incomplete` — never complete.

Durability: data is flushed (gzip Z_SYNC_FLUSH) at most every `flush_s` seconds, so a process
crash loses at most that window. Flush is not a power-loss (fsync) guarantee.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class StoreError(RuntimeError):
    """Storage failure. Fatal by design: never to be mistaken for a network event."""


def utc_day(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%Y-%m-%d")


def _utc_hour(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).strftime("%H")


def envelope(
    source: str,
    kind: str,
    received_at_ns: int,
    *,
    payload: Optional[str] = None,
    status: Optional[int] = None,
    error: Optional[str] = None,
    meta: Optional[dict] = None,
) -> Dict[str, Any]:
    return {
        "received_at_ns": received_at_ns,
        "source": source,
        "kind": kind,
        "status": status,
        "error": error,
        "meta": meta or {},
        "payload": payload,
        "schema": 1,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_records(path: Path, tolerant: bool = False):
    """Strict: return records or raise. Tolerant: return (records_before_failure, error_or_None)."""
    records, error = [], None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as exc:
        if not tolerant:
            raise
        error = repr(exc)
    return (records, error) if tolerant else records


def recover_incomplete(root: Path) -> list:
    report = []
    for part in sorted(Path(root).rglob("*.jsonl.gz.part")):
        records, error = read_records(part, tolerant=True)
        target = part.with_name(part.name.removesuffix(".part") + ".incomplete")
        part.rename(target)
        info = {
            "path": str(target),
            "records_recovered": len(records),
            "error": error,
            "bytes": target.stat().st_size,
        }
        target.with_name(target.name + ".salvage.json").write_text(json.dumps(info, indent=1))
        report.append(info)
    return report


class SegmentStore:
    def __init__(
        self,
        root: Path,
        run_id: str,
        *,
        flush_s: float = 5.0,
        mono: Callable[[], int] = time.monotonic_ns,
    ):
        self.root = Path(root)
        self.run_id = run_id
        self.raw_enabled = True
        self.stats: Dict[str, Dict[str, Any]] = {}
        self._flush_ns = int(flush_s * 1e9)
        self._mono = mono
        self._open: Dict[str, tuple] = {}  # stream -> (key, part_path, handle, last_flush_mono)

    def write(self, stream: str, record: dict) -> None:
        ns = record["received_at_ns"]
        key = (utc_day(ns), _utc_hour(ns))
        try:
            cur = self._open.get(stream)
            if cur is not None and cur[0] != key:
                self._finalise(stream)
                cur = None
            if cur is None:
                part = self.root / stream / key[0] / f"{key[1]}00_{self.run_id}.jsonl.gz.part"
                part.parent.mkdir(parents=True, exist_ok=True)
                cur = (key, part, gzip.open(part, "wt", encoding="utf-8"), self._mono())
                self._open[stream] = cur
            stamped = {**record, "run_id": self.run_id, "written_mono_ns": self._mono()}
            cur[2].write(json.dumps(stamped, separators=(",", ":")) + "\n")
            now = self._mono()
            if now - cur[3] >= self._flush_ns:
                cur[2].flush()
                self._open[stream] = (cur[0], cur[1], cur[2], now)
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f"{stream}: {exc!r}") from exc
        st = self.stats.setdefault(stream, {"count": 0, "last_received_ns": 0})
        st["count"] += 1
        st["last_received_ns"] = ns

    def flush_all(self) -> None:
        try:
            for stream, (key, part, handle, _) in list(self._open.items()):
                handle.flush()
                self._open[stream] = (key, part, handle, self._mono())
        except Exception as exc:
            raise StoreError(f"flush: {exc!r}") from exc

    def _finalise(self, stream: str) -> None:
        _, part, handle, _ = self._open.pop(stream)
        handle.close()
        final = part.with_name(part.name.removesuffix(".part"))
        part.rename(final)
        final.with_name(final.name + ".sha256").write_text(_sha256(final) + "\n")

    def close(self) -> None:
        try:
            for stream in list(self._open):
                self._finalise(stream)
        except Exception as exc:
            raise StoreError(f"close: {exc!r}") from exc
