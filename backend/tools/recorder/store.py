"""Append-only, crash-safe segment store.

Each run writes its own immutable hourly segments:
`<root>/<stream>/<UTC-day>/<HH>00_<run_id>.jsonl.gz`. A segment is written as `.part` and
renamed with a sha256 sidecar only when cleanly finalised. A run never appends to another run's
file, so an unclean exit can damage only its own `.part` files. `recover_incomplete` keeps those
bytes, salvages the readable prefix and marks them `.incomplete` — never complete.

Durability: a stream with unflushed data is flushed (gzip Z_SYNC_FLUSH) once `flush_s` seconds
have passed since its last flush, both on write and by the independent `run_flusher` timer
(1 s tick). A process crash therefore loses at most about flush_s + 1 s (plus event-loop delay),
even for a stream that is never written again. Flush is not a power-loss (fsync) guarantee.
Segment names carry a per-run ordinal: a wall clock stepping back into an earlier hour opens a
NEW segment and can never overwrite a finalised one.
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
        self._open: Dict[str, list] = {}  # stream -> [key, part, handle, last_flush, dirty]
        self._seq = 0

    def write(self, stream: str, record: dict) -> None:
        ns = record["received_at_ns"]
        key = (utc_day(ns), _utc_hour(ns))
        try:
            cur = self._open.get(stream)
            if cur is not None and cur[0] != key:
                self._finalise(stream)
                cur = None
            if cur is None:
                self._seq += 1
                name = f"{key[1]}00_{self.run_id}_s{self._seq:04d}.jsonl.gz.part"
                part = self.root / stream / key[0] / name
                part.parent.mkdir(parents=True, exist_ok=True)
                cur = [key, part, gzip.open(part, "xt", encoding="utf-8"), self._mono(), False]
                self._open[stream] = cur
            stamped = {**record, "run_id": self.run_id, "written_mono_ns": self._mono()}
            cur[2].write(json.dumps(stamped, separators=(",", ":")) + "\n")
            cur[4] = True
            self._flush_if_due(cur, self._mono())
        except StoreError:
            raise
        except Exception as exc:
            raise StoreError(f"{stream}: {exc!r}") from exc
        st = self.stats.setdefault(stream, {"count": 0, "last_received_ns": 0})
        st["count"] += 1
        st["last_received_ns"] = ns

    def _flush_if_due(self, cur: list, now: int, force: bool = False) -> bool:
        if cur[4] and (force or now - cur[3] >= self._flush_ns):
            cur[2].flush()
            cur[3], cur[4] = now, False
            return True
        return False

    def flush_due(self) -> int:
        """Flush every stream whose unflushed data has waited >= flush_s. Returns the count."""
        try:
            now = self._mono()
            return sum(self._flush_if_due(cur, now) for cur in self._open.values())
        except Exception as exc:
            raise StoreError(f"flush: {exc!r}") from exc

    def flush_all(self) -> None:
        try:
            now = self._mono()
            for cur in self._open.values():
                self._flush_if_due(cur, now, force=True)
        except Exception as exc:
            raise StoreError(f"flush: {exc!r}") from exc

    def _finalise(self, stream: str) -> None:
        _, part, handle, _, _ = self._open.pop(stream)
        handle.close()
        final = part.with_name(part.name.removesuffix(".part"))
        if final.exists():
            raise FileExistsError(f"refusing to overwrite finalised segment {final}")
        part.rename(final)
        final.with_name(final.name + ".sha256").write_text(_sha256(final) + "\n")

    def close(self) -> None:
        """Best effort: try to finalise EVERY open segment, then raise if any failed. A failed
        segment stays `.part` (salvaged as incomplete at the next start), never claimed complete."""
        errors = []
        for stream in list(self._open):
            try:
                self._finalise(stream)
            except Exception as exc:
                errors.append(f"{stream}: {exc!r}")
        if errors:
            raise StoreError("close: " + "; ".join(errors))
