import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.recorder.store import (
    SegmentStore,
    StoreError,
    envelope,
    read_records,
    recover_incomplete,
    utc_day,
)

HOUR_NS = 3600 * 10**9
T0 = 1_791_000_000 * 10**9  # 2026-10-03 04:00 UTC


def _segments(root, stream):
    return sorted((root / stream).rglob("*.jsonl.gz"))


def test_utc_day():
    assert utc_day(T0) == "2026-10-03"


def test_envelope_shape():
    e = envelope("okx", "poll", T0, payload="{}", status=200, meta={"url": "u"})
    assert e == {
        "received_at_ns": T0,
        "source": "okx",
        "kind": "poll",
        "status": 200,
        "error": None,
        "meta": {"url": "u"},
        "payload": "{}",
        "schema": 1,
    }


def test_write_stamps_run_and_monotonic_and_counts_after_success(tmp_path):
    s = SegmentStore(tmp_path, "run1", mono=lambda: 42)
    s.write("poll/x", envelope("okx", "poll", T0, payload="a"))
    s.close()
    (seg,) = _segments(tmp_path, "poll/x")
    (rec,) = read_records(seg)
    assert rec["run_id"] == "run1" and rec["written_mono_ns"] == 42 and rec["payload"] == "a"
    assert s.stats["poll/x"]["count"] == 1 and s.stats["poll/x"]["last_received_ns"] == T0


def test_hourly_segments_are_finalised_with_sha256(tmp_path):
    s = SegmentStore(tmp_path, "run1")
    s.write("ws/l2", envelope("cb", "message", T0, payload="h04"))
    s.write("ws/l2", envelope("cb", "message", T0 + HOUR_NS, payload="h05"))
    s.close()
    segs = _segments(tmp_path, "ws/l2")
    assert [p.name for p in segs] == ["0400_run1.jsonl.gz", "0500_run1.jsonl.gz"]
    assert all(p.parent.name == "2026-10-03" for p in segs)
    for p in segs:
        assert p.with_name(p.name + ".sha256").exists()
    assert not list(tmp_path.rglob("*.part"))


def test_midnight_record_goes_to_next_day_directory(tmp_path):
    s = SegmentStore(tmp_path, "r")
    s.write("ws/l2", envelope("cb", "message", T0 + 20 * HOUR_NS, payload="next"))
    s.close()
    (seg,) = _segments(tmp_path, "ws/l2")
    assert seg.parent.name == "2026-10-04" and seg.name.startswith("0000_")


def test_interrupted_segment_is_salvaged_not_marked_complete(tmp_path):
    code = (
        "import os, sys; sys.path.insert(0, '.');"
        "from tools.recorder.store import SegmentStore, envelope;"
        f"s = SegmentStore(r'{tmp_path}', 'crashed', flush_s=0.0);"
        f"[s.write('ws/l2', envelope('cb', 'message', {T0} + i, payload=f'm{{i}}')) for i in range(3)];"
        "os._exit(1)"
    )  # abrupt exit: no close, no gzip trailer
    backend = Path(__file__).resolve().parents[3]
    r = subprocess.run([sys.executable, "-c", code], cwd=backend, capture_output=True, text=True)
    assert r.returncode == 1, r.stderr  # the crash itself, not an import failure
    (part,) = list(tmp_path.rglob("*.part"))
    report = recover_incomplete(tmp_path)
    assert len(report) == 1 and report[0]["records_recovered"] == 3
    incomplete = part.with_name(part.name.removesuffix(".part") + ".incomplete")
    assert incomplete.exists() and not part.exists()
    assert not incomplete.with_name(incomplete.name + ".sha256").exists()
    sal = json.loads(incomplete.with_name(incomplete.name + ".salvage.json").read_text())
    assert sal["records_recovered"] == 3 and sal["error"]
    # a restarted run writes its own segment and history stays readable
    s2 = SegmentStore(tmp_path, "restart")
    s2.write("ws/l2", envelope("cb", "message", T0 + 10, payload="after"))
    s2.close()
    (seg,) = _segments(tmp_path, "ws/l2")
    assert [r["payload"] for r in read_records(seg)] == ["after"]


def test_tolerant_read_of_truncated_gzip(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    p.write_bytes(gzip.compress(b'{"a":1}\n{"a":2}\n')[:-8])
    with pytest.raises((EOFError, gzip.BadGzipFile, OSError)):
        read_records(p)
    recs, err = read_records(p, tolerant=True)
    assert recs == [{"a": 1}, {"a": 2}] and err


def test_write_failure_raises_store_error_and_does_not_count(tmp_path):
    s = SegmentStore(tmp_path, "r")
    blocker = tmp_path / "poll"
    blocker.write_text("a file where a directory must go")
    with pytest.raises(StoreError, match="poll/x"):
        s.write("poll/x", envelope("okx", "poll", T0))
    assert "poll/x" not in s.stats
