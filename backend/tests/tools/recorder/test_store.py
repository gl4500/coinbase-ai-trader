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
    assert [p.name for p in segs] == ["0400_run1_s0001.jsonl.gz", "0500_run1_s0002.jsonl.gz"]
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


def test_clock_returning_to_an_earlier_hour_never_reuses_a_segment(tmp_path):
    s = SegmentStore(tmp_path, "r")
    for t, tag in ((T0, "A1"), (T0 + HOUR_NS, "B"), (T0 + 60 * 10**9, "A2")):  # A -> B -> A
        s.write("ws/l2", envelope("cb", "message", t, payload=tag))
    s.close()
    segs = _segments(tmp_path, "ws/l2")
    assert [p.name for p in segs] == [
        "0400_r_s0001.jsonl.gz",
        "0400_r_s0003.jsonl.gz",
        "0500_r_s0002.jsonl.gz",
    ]
    assert sorted(r["payload"] for p in segs for r in read_records(p)) == ["A1", "A2", "B"]


def test_timer_flush_makes_a_quiet_stream_durable_within_the_bound(tmp_path):
    now = [0]
    s = SegmentStore(tmp_path, "r", flush_s=5.0, mono=lambda: now[0])
    s.write("poll/x", envelope("okx", "poll", T0, payload="only"))  # no later write ever comes
    (part,) = list(tmp_path.rglob("*.part"))
    now[0] = 4 * 10**9
    assert s.flush_due() == 0
    now[0] = 5 * 10**9
    assert s.flush_due() == 1
    recs, _ = read_records(part, tolerant=True)
    assert [r["payload"] for r in recs] == ["only"]
    s.close()


def test_close_tries_every_segment_then_raises(tmp_path):
    s = SegmentStore(tmp_path, "r")
    s.write("poll/a", envelope("okx", "poll", T0, payload="a"))
    s.write("poll/b", envelope("okx", "poll", T0, payload="b"))
    real = s._finalise

    def flaky(stream):
        if stream == "poll/a":
            raise OSError("rename failed")
        return real(stream)

    s._finalise = flaky
    with pytest.raises(StoreError, match="rename failed"):
        s.close()
    assert len(_segments(tmp_path, "poll/b")) == 1  # b still finalised
    assert list((tmp_path / "poll" / "a").rglob("*.part"))  # a left incomplete, not claimed done


def _fail_after_sidecar(monkeypatch):
    """Inject a failure at the real stage between checksum and data publication."""
    import tools.recorder.store as st

    real_rename = Path.rename

    def rename(self, target):
        if str(self).endswith(".jsonl.gz.part"):
            raise OSError("interrupted before data rename")
        return real_rename(self, target)

    monkeypatch.setattr(st.Path, "rename", rename)


def test_failed_finalisation_never_leaves_final_data_without_its_seal(tmp_path, monkeypatch):
    """Codex B2: completion means a verified data/checksum pair, never data alone."""
    s = SegmentStore(tmp_path, "r")
    s.write("poll/a", envelope("okx", "poll", T0, payload="a"))
    _fail_after_sidecar(monkeypatch)
    with pytest.raises(StoreError, match="interrupted"):
        s.close()
    monkeypatch.undo()
    for seg in tmp_path.rglob("*.jsonl.gz"):  # any final-named data must carry a valid seal
        assert seg.with_name(seg.name + ".sha256").exists()
    report = recover_incomplete(tmp_path)
    assert [r["records_recovered"] for r in report] == [1]
    assert not list(tmp_path.rglob("*.jsonl.gz"))
    assert not list(tmp_path.rglob("*.sha256"))  # stale seal moved aside, not left claiming data
    assert list(tmp_path.rglob("*.sha256.orphan"))


def test_recovery_marks_final_data_with_missing_or_bad_seal_incomplete(tmp_path):
    """Segments finalised by older code may lack a seal or carry a partial one."""
    s = SegmentStore(tmp_path, "r")
    s.write("poll/a", envelope("okx", "poll", T0, payload="a"))
    s.write("poll/b", envelope("okx", "poll", T0, payload="b"))
    s.write("poll/c", envelope("okx", "poll", T0, payload="c"))
    s.close()
    a, b, c = (_segments(tmp_path, f"poll/{x}")[0] for x in "abc")
    a.with_name(a.name + ".sha256").unlink()  # missing seal
    b.with_name(b.name + ".sha256").write_text("deadbe")  # partial seal
    report = recover_incomplete(tmp_path)
    assert sorted(Path(r["path"]).parent.parent.name for r in report) == ["a", "b"]
    assert all(r["reason"] for r in report)
    assert _segments(tmp_path, "poll/c") == [c]  # a sealed segment is untouched
    assert len(list(tmp_path.rglob("*.incomplete"))) == 2


def test_failed_seal_write_leaves_no_final_looking_segment(tmp_path, monkeypatch):
    """Codex B2, the exact stage: the checksum write itself fails."""
    import tools.recorder.store as st

    def failing_open(path, *a, **k):
        if ".sha256" in str(path):
            raise OSError("sidecar write failed")
        return open(path, *a, **k)

    s = SegmentStore(tmp_path, "r")
    s.write("poll/a", envelope("okx", "poll", T0, payload="a"))
    monkeypatch.setattr(st, "open", failing_open, raising=False)
    with pytest.raises(StoreError, match="sidecar"):
        s.close()
    monkeypatch.undo()
    assert not list(tmp_path.rglob("*.jsonl.gz"))  # nothing claims to be complete
    assert [r["records_recovered"] for r in recover_incomplete(tmp_path)] == [1]
