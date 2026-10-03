"""Append-only, crash-safe segment store.

Each run writes its own immutable hourly segments:
`<root>/<stream>/<UTC-day>/<HH>00_<run_id>_s<NNNN>.jsonl.gz`. A segment is written as `.part`.
Finalisation publishes the `.sha256` seal first (temp file + rename) and only then renames the
data, so final-named data never exists without its complete seal: a COMPLETE segment is a data
file plus a well-formed seal, nothing less. A run never appends to another run's file.
`recover_incomplete` (startup, under the writer lock) keeps every byte and marks anything short
of that pair `.incomplete` (or sets a stray seal aside as `.orphan`) — never complete.

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
import re
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


_SEAL = re.compile(r"[0-9a-f]{64}\r?\n?")  # text-mode seals on Windows end in CRLF


def _seal_ok(seal: Path) -> bool:
    """Structural check only (cheap at startup): a complete 64-hex digest. Content is verified
    against the data by consumers, not re-hashed here on every start."""
    try:
        return _SEAL.fullmatch(seal.read_bytes().decode("ascii")) is not None
    except (OSError, UnicodeDecodeError):
        return False


def _mark_incomplete(data: Path, target: Path, reason: str) -> dict:
    if target.exists():
        raise StoreError(f"recovery: refusing to overwrite {target}")
    records, error = read_records(data, tolerant=True)
    data.rename(target)
    info = {
        "path": str(target),
        "reason": reason,
        "records_recovered": len(records),
        "error": error,
        "bytes": target.stat().st_size,
    }
    target.with_name(target.name + ".salvage.json").write_text(json.dumps(info, indent=1))
    return info


def _set_aside(path: Path, suffix: str) -> None:
    target = path.with_name(path.name + suffix)
    if target.exists():
        raise StoreError(f"recovery: refusing to overwrite {target}")
    path.rename(target)


def recover_incomplete(root: Path) -> list:
    """Run at startup under the writer lock. A segment is complete only as a data file plus a
    well-formed `.sha256` seal. Anything else is kept byte-for-byte and reported, never deleted:
    - `.part` data (unclean exit)            -> `.incomplete` + salvage report
    - final data with a missing/partial seal -> `.incomplete` (+ the bad seal set aside)
    - a seal whose data never got its name   -> `.sha256.orphan`
    - a seal temp file                       -> `.sha256.tmp.orphan`"""
    root, report = Path(root), []
    for part in sorted(root.rglob("*.jsonl.gz.part")):
        target = part.with_name(part.name.removesuffix(".part") + ".incomplete")
        report.append(_mark_incomplete(part, target, "unfinalised .part"))
    for data in sorted(root.rglob("*.jsonl.gz")):
        seal = data.with_name(data.name + ".sha256")
        if seal.exists() and _seal_ok(seal):
            continue
        reason = "seal malformed" if seal.exists() else "seal missing"
        if seal.exists():
            _set_aside(seal, ".orphan")
        report.append(_mark_incomplete(data, data.with_name(data.name + ".incomplete"), reason))
    for seal in sorted(root.rglob("*.jsonl.gz.sha256")):
        if not seal.with_name(seal.name.removesuffix(".sha256")).exists():
            _set_aside(seal, ".orphan")
    for tmp in sorted(root.rglob("*.jsonl.gz.sha256.tmp")):
        _set_aside(tmp, ".orphan")
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
        """Publish the seal FIRST (temp file + rename), then the data. An interruption at any
        point leaves either `.part` data (recovered as incomplete) or no final-named data at all:
        final-named data never exists without its complete seal."""
        _, part, handle, _, _ = self._open.pop(stream)
        handle.close()
        final = part.with_name(part.name.removesuffix(".part"))
        seal = final.with_name(final.name + ".sha256")
        for existing in (final, seal):
            if existing.exists():
                raise FileExistsError(f"refusing to overwrite finalised file {existing}")
        tmp = seal.with_name(seal.name + ".tmp")
        with open(tmp, "x", encoding="utf-8") as f:
            f.write(_sha256(part) + "\n")
        tmp.rename(seal)
        part.rename(final)

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
