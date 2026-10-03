"""Verified, read-only snapshot of backend/data plus the SQLite database.

  python -m tools.data_snapshot.snapshot [--data-dir DIR] [--db PATH] [--out ROOT]

Guarantees, stated exactly:
- Sources are only ever READ. Each snapshot goes to a new `<out>/<UTC stamp>/` directory and
  never overwrites an existing one.
- A file is `stable` only if its (size, mtime_ns) are unchanged across the copy AND a second
  read of the source hashes identically to the copy. This detects concurrent rewrites,
  including same-size/same-mtime ones that differ by the second read. It cannot detect a writer
  that changes the file and restores it between our two reads. Otherwise the file is marked
  `unstable` after `retries` attempts and is never claimed clean.
- The sha256 recorded is that of the retained COPY.
- Parquet copies are fully read (every row group), not just their metadata. Other files are
  hash-only.
- The database is copied with SQLite's online backup API, which is transactionally consistent
  for the DB itself, then checked with `PRAGMA integrity_check`.
- The snapshot as a whole is NOT a globally consistent as-of view: files are captured one by one
  over the recorded interval.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

CONSISTENCY_NOTE = (
    "Files were captured one at a time between captured_from_ns and captured_to_ns; this is "
    "not a globally consistent as-of snapshot. Each file's own capture time is recorded."
)
EXCLUDE_DIRS = {"__pycache__"}


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def copy_stable(
    src: Path,
    dst: Path,
    *,
    retries: int = 3,
    stat: Callable = os.stat,
    read: Callable[[Path], bytes] = lambda p: Path(p).read_bytes(),
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    for attempts in range(1, retries + 1):
        before = stat(src)
        data = read(src)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(dst)
        after = stat(src)
        same_stat = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        if same_stat and _sha(read(src)) == _sha(data):
            return {
                "status": "stable",
                "attempts": attempts,
                "bytes": len(data),
                "sha256": _sha(dst.read_bytes()),
                "captured_at_ns": time.time_ns(),
            }
        sleep(0.5)
    return {
        "status": "unstable",
        "attempts": attempts,
        "bytes": dst.stat().st_size,
        "sha256": _sha(dst.read_bytes()),
        "captured_at_ns": time.time_ns(),
    }


def validate_copy(path: Path) -> dict:
    if path.suffix != ".parquet":
        return {"kind": "hash_only", "ok": True, "rows": None, "error": None}
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)  # full read: every row group and page
        return {"kind": "parquet_full_read", "ok": True, "rows": table.num_rows, "error": None}
    except Exception as exc:
        return {"kind": "parquet_full_read", "ok": False, "rows": None, "error": repr(exc)}


def backup_sqlite(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(f"file:{Path(src).as_posix()}?mode=ro", uri=True)
    target = sqlite3.connect(dst)
    try:
        started = time.time_ns()
        source.backup(target)
        integrity = target.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        target.close()
        source.close()
    return {
        "status": "ok" if integrity == "ok" else "integrity_failed",
        "integrity": integrity,
        "bytes": dst.stat().st_size,
        "sha256": _sha(dst.read_bytes()),
        "started_ns": started,
        "finished_ns": time.time_ns(),
    }


def _files(data_dir: Path) -> list:
    return sorted(
        p
        for p in data_dir.rglob("*")
        if p.is_file() and not EXCLUDE_DIRS.intersection(p.relative_to(data_dir).parts)
    )


def take_snapshot(
    data_dir: Path, db_path: Optional[Path], out_root: Path, *, stamp: Optional[str] = None
) -> Path:
    data_dir, out_root = Path(data_dir), Path(out_root)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=False)  # FileExistsError: never overwrite a snapshot
    t0 = time.time_ns()
    files = []
    for src in _files(data_dir):
        rel = src.relative_to(data_dir).as_posix()
        dst = out / "data" / rel
        rec = copy_stable(src, dst)
        rec.update(rel=rel, validation=validate_copy(dst))
        files.append(rec)
    database = backup_sqlite(Path(db_path), out / "coinbase.db") if db_path else None
    t1 = time.time_ns()
    manifest = {
        "schema": 1,
        "stamp": stamp,
        "source_data_dir": str(data_dir),
        "source_db": str(db_path) if db_path else None,
        "captured_from_ns": t0,
        "captured_to_ns": t1,
        "consistency_note": CONSISTENCY_NOTE,
        "tool_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "summary": {
            "files": len(files),
            "stable": sum(f["status"] == "stable" for f in files),
            "unstable": sum(f["status"] == "unstable" for f in files),
            "invalid": sum(not f["validation"]["ok"] for f in files),
        },
        "files": files,
        "database": database,
    }
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1))
    tmp.replace(out / "manifest.json")
    return out


def cli(argv=None) -> int:
    backend = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=backend / "data")
    ap.add_argument("--db", type=Path, default=backend / "coinbase.db")
    ap.add_argument("--out", type=Path, default=Path(r"C:\Users\gl450\polymarket_data_snapshots"))
    args = ap.parse_args(argv)
    out = take_snapshot(args.data_dir, args.db if args.db.exists() else None, args.out)
    m = json.loads((out / "manifest.json").read_text())
    print(
        json.dumps(
            {
                "snapshot": str(out),
                "summary": m["summary"],
                "database": m["database"] and m["database"]["integrity"],
            },
            indent=1,
        )
    )
    bad = (
        m["summary"]["unstable"]
        or m["summary"]["invalid"]
        or (m["database"] and m["database"]["integrity"] != "ok")
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(cli())
