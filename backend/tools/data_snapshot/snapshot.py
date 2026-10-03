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
    attempts, copied, error = 0, False, None
    for attempts in range(1, retries + 1):
        try:  # source-side failures are retried, then recorded; never fatal
            before = stat(src)
            data = read(src)
        except OSError as exc:
            error = repr(exc)
            sleep(0.5)
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.write_bytes(data)  # destination failures stay fatal
        tmp.replace(dst)
        copied = True
        try:
            after = stat(src)
            reread = read(src)
        except OSError as exc:
            error = repr(exc)
            sleep(0.5)
            continue
        same_stat = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        if same_stat and _sha(reread) == _sha(data):
            return {
                "status": "stable",
                "attempts": attempts,
                "bytes": len(data),
                "sha256": _sha(dst.read_bytes()),
                "captured_at_ns": time.time_ns(),
                "error": None,
            }
        sleep(0.5)
    if not copied:
        return {
            "status": "failed",
            "attempts": attempts,
            "bytes": None,
            "sha256": None,
            "captured_at_ns": time.time_ns(),
            "error": error,
        }
    return {
        "status": "unstable",
        "attempts": attempts,
        "bytes": dst.stat().st_size,
        "sha256": _sha(dst.read_bytes()),
        "captured_at_ns": time.time_ns(),
        "error": error,
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


COVERAGE_NOTE = (
    "Only the roots in included_roots are captured. NOT included: research outputs kept in "
    "worktrees (e.g. slow_trend locked raw/report/ledger), model weights/caches/checkpoints/logs "
    "outside the data dir, and the market recorder archive. See "
    r"C:\Users\gl450\analysis_archive\do_now_2026-10-03\writer_inventory.md."
)


def _check_inputs(data_dir: Path, db_path: Optional[Path], out: Path, data_only: bool) -> None:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"source data dir not found: {data_dir}")
    if db_path is None and not data_only:
        raise ValueError("db_path is None: pass data_only=True to omit the database explicitly")
    if db_path is not None and not Path(db_path).is_file():
        raise FileNotFoundError(f"source database not found: {db_path}")
    src, dst = data_dir.resolve(), out.resolve()
    if dst == src or src in dst.parents:
        raise ValueError(f"snapshot directory {dst} is inside the source data tree {src}")


def take_snapshot(
    data_dir: Path,
    db_path: Optional[Path],
    out_root: Path,
    *,
    stamp: Optional[str] = None,
    data_only: bool = False,
) -> Path:
    data_dir, out_root = Path(data_dir), Path(out_root)
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_root / stamp
    _check_inputs(data_dir, db_path, out, data_only)  # before ANY write
    out.mkdir(parents=True, exist_ok=False)  # FileExistsError: never overwrite a snapshot
    t0 = time.time_ns()
    files = []
    for src in _files(data_dir):
        rel = src.relative_to(data_dir).as_posix()
        dst = out / "data" / rel
        rec = copy_stable(src, dst)
        if rec["status"] == "failed":
            rec.update(
                rel=rel,
                validation={
                    "kind": "none",
                    "ok": False,
                    "rows": None,
                    "error": "source could not be read",
                },
            )
        else:
            rec.update(rel=rel, validation=validate_copy(dst))
        files.append(rec)
    if db_path is not None:
        database = backup_sqlite(Path(db_path), out / "coinbase.db")
    else:
        database = {"status": "omitted_explicitly"}
    t1 = time.time_ns()
    roots = [str(data_dir.resolve())] + ([str(Path(db_path).resolve())] if db_path else [])
    manifest = {
        "schema": 2,
        "stamp": stamp,
        "included_roots": roots,
        "coverage_note": COVERAGE_NOTE,
        "captured_from_ns": t0,
        "captured_to_ns": t1,
        "consistency_note": CONSISTENCY_NOTE,
        "tool_sha256": _sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
        "summary": {
            "files": len(files),
            "stable": sum(f["status"] == "stable" for f in files),
            "unstable": sum(f["status"] == "unstable" for f in files),
            "failed": sum(f["status"] == "failed" for f in files),
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
    ap.add_argument("--no-db", action="store_true", help="omit the database explicitly")
    ap.add_argument("--out", type=Path, default=Path(r"C:\Users\gl450\polymarket_data_snapshots"))
    args = ap.parse_args(argv)
    try:
        out = take_snapshot(
            args.data_dir, None if args.no_db else args.db, args.out, data_only=args.no_db
        )
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        print(f"data_snapshot: refused: {exc}", file=sys.stderr)
        return 2
    m = json.loads((out / "manifest.json").read_text())
    db = m["database"]
    print(
        json.dumps(
            {
                "snapshot": str(out),
                "summary": m["summary"],
                "database": db.get("integrity", db["status"]),
            },
            indent=1,
        )
    )
    s = m["summary"]
    bad = s["unstable"] or s["failed"] or s["invalid"] or db.get("status") == "integrity_failed"
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(cli())
