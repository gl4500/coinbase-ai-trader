import asyncio
import json

import pytest

from tools.recorder.health import apply_disk_guard, disk_ok, run_health, status_snapshot
from tools.recorder.lock import WriterLock
from tools.recorder.run import expected_streams, package_digest
from tools.recorder.store import SegmentStore, envelope, read_records

T0 = 1_791_000_000 * 10**9
GB = 1024**3


def test_disk_ok_threshold():
    assert disk_ok(100 * GB, 100 * GB) and not disk_ok(100 * GB - 1, 100 * GB)


def test_disk_guard_transitions_once_each_way(tmp_path):
    store = SegmentStore(tmp_path, "r")
    apply_disk_guard(store, 50 * GB, lambda: T0, 100 * GB)
    apply_disk_guard(store, 50 * GB, lambda: T0 + 1, 100 * GB)
    assert store.raw_enabled is False
    apply_disk_guard(store, 200 * GB, lambda: T0 + 2, 100 * GB)
    assert store.raw_enabled is True
    store.close()
    seg = next((tmp_path / "recorder" / "events").rglob("*.jsonl.gz"))
    assert [r["meta"]["event"] for r in read_records(seg)] == ["raw_paused_low_disk", "raw_resumed"]


def test_status_reports_age_app_status_and_never_seen_streams(tmp_path):
    store = SegmentStore(tmp_path, "r")
    store.write("poll/x", envelope("okx", "poll", T0))
    store.stats["poll/x"]["last_app_status"] = "app_error"
    snap = status_snapshot(store, T0 + 5 * 10**9, 123, 42, ["poll/x", "coinbase_ws/l2_data"])
    store.close()
    assert snap["pid"] == 42 and snap["free_bytes"] == 123 and snap["raw_enabled"] is True
    assert snap["streams"]["poll/x"]["age_s"] == 5.0
    assert snap["streams"]["poll/x"]["last_app_status"] == "app_error"
    assert snap["never_seen"] == ["coinbase_ws/l2_data"]


def test_run_health_writes_status_atomically(tmp_path):
    store = SegmentStore(tmp_path, "r")
    stop = asyncio.Event()

    def free(path):
        stop.set()
        return 500 * GB

    asyncio.run(
        run_health(
            store, tmp_path, stop, clock=lambda: T0, free=free, interval_s=0.01, pid=7, expected=[]
        )
    )
    store.close()
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["pid"] == 7 and not (tmp_path / "status.json.tmp").exists()


def test_second_writer_is_refused_and_lock_is_released(tmp_path):
    first = WriterLock(tmp_path).acquire()
    with pytest.raises(RuntimeError, match="another recorder"):
        WriterLock(tmp_path).acquire()
    first.release()
    WriterLock(tmp_path).acquire().release()


def test_expected_streams_cover_ws_channels_and_every_poll():
    exp = expected_streams()
    assert {"coinbase_ws/l2_data", "coinbase_ws/market_trades", "coinbase_ws/heartbeats"} <= set(
        exp
    )
    assert "poll/okx_funding_BTC" in exp and "poll/coinbase_spot_catalogue" in exp


def test_package_digest_is_stable_and_hex():
    d = package_digest()
    assert d == package_digest() and d.startswith("sha256:") and len(d) == 71


def test_stop_file_requests_graceful_shutdown(tmp_path):
    store = SegmentStore(tmp_path, "r")
    stop = asyncio.Event()
    (tmp_path / "STOP").write_text("")

    async def go():
        await asyncio.wait_for(
            run_health(
                store,
                tmp_path,
                stop,
                clock=lambda: T0,
                free=lambda p: 500 * GB,
                interval_s=60,
                pid=1,
                expected=[],
            ),
            timeout=2,
        )

    asyncio.run(go())
    store.close()
    assert stop.is_set() and not (tmp_path / "STOP").exists()
