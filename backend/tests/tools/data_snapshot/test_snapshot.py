import hashlib
import json
import sqlite3

import pandas as pd
import pytest

from tools.data_snapshot.snapshot import (
    backup_sqlite,
    copy_stable,
    take_snapshot,
    validate_copy,
)


def _sha(p):
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _parquet(path, n=3):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"start": range(n), "close": [1.0] * n}).to_parquet(path)
    return path


def _db(path, rows=5):
    con = sqlite3.connect(path)
    con.execute("create table candles(start int, close real)")
    con.executemany("insert into candles values(?,?)", [(i, 1.0) for i in range(rows)])
    con.commit()
    con.close()
    return path


def test_copy_stable_copies_bytes_and_hashes_the_copy(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"hello")
    dst = tmp_path / "out" / "a.bin"
    r = copy_stable(src, dst)
    assert dst.read_bytes() == b"hello"
    assert r["status"] == "stable" and r["attempts"] == 1
    assert r["sha256"] == _sha(dst) and r["bytes"] == 5


def test_copy_stable_marks_a_changing_file_unstable_instead_of_claiming_it(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"x")
    ticks = iter(range(1000))

    class FakeStat:
        def __init__(self):
            self.st_size, self.st_mtime_ns = 1, next(ticks)

    r = copy_stable(
        src, tmp_path / "out" / "a.bin", retries=3, stat=lambda p: FakeStat(), sleep=lambda s: None
    )
    assert r["status"] == "unstable" and r["attempts"] == 3


def test_validate_copy_reads_parquet_metadata_and_rejects_truncation(tmp_path):
    good = _parquet(tmp_path / "g.parquet", n=4)
    assert validate_copy(good) == {
        "kind": "parquet_full_read",
        "ok": True,
        "rows": 4,
        "error": None,
    }
    bad = tmp_path / "b.parquet"
    bad.write_bytes(good.read_bytes()[:-20])
    v = validate_copy(bad)
    assert v["kind"] == "parquet_full_read" and v["ok"] is False and v["error"]
    other = tmp_path / "x.pkl"
    other.write_bytes(b"\x80")
    assert validate_copy(other) == {"kind": "hash_only", "ok": True, "rows": None, "error": None}


def test_backup_sqlite_is_consistent_and_leaves_source_untouched(tmp_path):
    src = _db(tmp_path / "src.db", rows=7)
    before = _sha(src)
    r = backup_sqlite(src, tmp_path / "out" / "copy.db")
    assert r["integrity"] == "ok" and r["status"] == "ok"
    con = sqlite3.connect(tmp_path / "out" / "copy.db")
    assert con.execute("select count(*) from candles").fetchone() == (7,)
    con.close()
    assert _sha(src) == before and r["sha256"] == _sha(tmp_path / "out" / "copy.db")


def test_take_snapshot_end_to_end(tmp_path):
    data = tmp_path / "data"
    _parquet(data / "history" / "BTC-USD.parquet")
    _parquet(data / "history" / "5m" / "BTC-USD.parquet")
    (data / "cg_snapshot.json").write_text('{"a": 1}')
    (data / "__pycache__").mkdir()
    (data / "__pycache__" / "x.pyc").write_bytes(b"0")
    db = _db(tmp_path / "coinbase.db")
    src_hashes = {p: _sha(p) for p in data.rglob("*") if p.is_file()}

    out = take_snapshot(data, db, tmp_path / "snaps", stamp="20261003T200000Z")
    m = json.loads((out / "manifest.json").read_text())
    rels = sorted(f["rel"] for f in m["files"])
    assert rels == ["cg_snapshot.json", "history/5m/BTC-USD.parquet", "history/BTC-USD.parquet"]
    for f in m["files"]:
        assert f["status"] == "stable" and f["validation"]["ok"] is True
        assert f["sha256"] == _sha(out / "data" / f["rel"])
    assert m["database"]["integrity"] == "ok"
    assert m["captured_from_ns"] <= m["captured_to_ns"]
    assert "not a globally consistent" in m["consistency_note"]
    assert m["summary"] == {"files": 3, "stable": 3, "unstable": 0, "invalid": 0}
    assert not list(out.rglob("*.tmp"))
    assert {p: _sha(p) for p in data.rglob("*") if p.is_file()} == src_hashes  # sources untouched
    with pytest.raises(FileExistsError):
        take_snapshot(data, db, tmp_path / "snaps", stamp="20261003T200000Z")


def test_copy_stable_detects_same_stat_rewrite_by_rereading_the_source(tmp_path):
    src = tmp_path / "a.bin"
    src.write_bytes(b"v1")
    reads = iter([b"v1", b"v2"] * 5)  # every re-read differs from what was copied
    r = copy_stable(
        src, tmp_path / "out" / "a.bin", retries=2, read=lambda p: next(reads), sleep=lambda s: None
    )
    assert r["status"] == "unstable" and r["attempts"] == 2
