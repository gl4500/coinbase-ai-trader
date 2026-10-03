import asyncio
import json
from pathlib import Path

import pytest

from tools.recorder.coinbase_ws import SeqTracker, backoff_s, run_coinbase_ws
from tools.recorder.store import SegmentStore, StoreError, read_records

T0 = 1_791_000_000 * 10**9
FIXTURE = Path(__file__).parent / "fixtures" / "coinbase_ws_sequence_sample.jsonl"


def _msg(channel, seq):
    return json.dumps({"channel": channel, "sequence_num": seq, "events": []})


class FakeWS:
    """Items are raw messages, or callables run between messages (to change state mid-stream)."""

    def __init__(self, items):
        self.items, self.sent = list(items), []

    async def send(self, m):
        self.sent.append(json.loads(m))

    async def recv(self):
        while self.items:
            item = self.items.pop(0)
            if callable(item):
                item()
                continue
            return item
        await asyncio.sleep(3600)  # silence


class FakeConnect:
    def __init__(self, sockets):
        self.sockets, self.calls = list(sockets), 0

    def __call__(self, url):
        self.calls += 1
        ws = self.sockets.pop(0) if self.sockets else FakeWS([])

        class CM:
            async def __aenter__(self):
                return ws

            async def __aexit__(self, *a):
                return False

        return CM()


def _clock():
    t = [T0]

    def tick():
        t[0] += 1000
        return t[0]

    return tick


def _run(store, connect, stop_after_sleeps=1):
    stop = asyncio.Event()
    sleeps = []

    async def sleep(s):
        sleeps.append(s)
        if len(sleeps) >= stop_after_sleeps:
            stop.set()

    asyncio.run(
        run_coinbase_ws(
            store,
            ["BTC-USD"],
            stop,
            connect=connect,
            clock=_clock(),
            max_silence_s=0.05,
            sleep=sleep,
        )
    )
    return sleeps


def _stream(root, name):
    recs = []
    for seg in sorted((root / "coinbase_ws" / name).rglob("*.jsonl.gz")):
        recs += read_records(seg)
    return recs


def _events(root):
    return [r["meta"] for r in _stream(root, "events")]


def test_seq_tracker_reports_gaps_only():
    t = SeqTracker()
    assert [t.observe(s) for s in (0, 1, 2, 5, 6)] == [None, None, None, (3, 5), None]


def test_real_capture_has_one_sequence_per_connection_across_channels_and_products():
    rows = [json.loads(x) for x in FIXTURE.read_text().splitlines() if x.strip()]
    assert {r["channel"] for r in rows} >= {"l2_data", "market_trades", "heartbeats"}
    assert {p for r in rows for p in r["product_ids"]} >= {"BTC-USD", "ETH-USD"}
    t = SeqTracker()
    assert [t.observe(r["sequence_num"]) for r in rows].count(None) == len(rows)


def test_backoff_caps_at_60():
    assert [backoff_s(a) for a in (0, 1, 3, 10)] == [1.0, 2.0, 8.0, 60.0]


def test_gap_is_recorded_with_connection_and_forces_resnapshot(tmp_path):
    store = SegmentStore(tmp_path, "r")
    ws = FakeWS(
        [_msg("l2_data", 0), _msg("market_trades", 1), _msg("l2_data", 3), _msg("l2_data", 4)]
    )
    connect = FakeConnect([ws])
    _run(store, connect)
    store.close()
    seqs = [json.loads(r["payload"])["sequence_num"] for r in _stream(tmp_path, "l2_data")]
    assert seqs == [0, 3]  # message 4 never read: reconnected for a fresh snapshot
    assert {"event": "gap", "expected": 2, "got": 3, "channel": "l2_data", "conn_id": 1} in _events(
        tmp_path
    )
    assert {s["channel"] for s in ws.sent} == {"level2", "market_trades", "heartbeats"}


def test_silence_records_stale_and_reconnects(tmp_path):
    store = SegmentStore(tmp_path, "r")
    connect = FakeConnect([FakeWS([]), FakeWS([])])
    sleeps = _run(store, connect, stop_after_sleeps=2)
    store.close()
    names = [e.get("event") for e in _events(tmp_path)]
    assert names.count("stale") == 2 and names.count("connect") == 2
    assert connect.calls == 2 and sleeps == [1.0, 1.0]


def test_network_error_is_a_disconnect_with_backoff(tmp_path):
    store = SegmentStore(tmp_path, "r")

    def boom(url):
        raise OSError("network down")

    sleeps = _run(store, boom, stop_after_sleeps=2)
    store.close()
    recs = _stream(tmp_path, "events")
    assert recs[0]["meta"]["event"] == "disconnect" and "network down" in recs[0]["error"]
    assert sleeps == [1.0, 2.0]


def test_raw_pause_keeps_heartbeats_and_resume_forces_resubscribe(tmp_path):
    store = SegmentStore(tmp_path, "r")
    store.raw_enabled = False

    def resume():
        store.raw_enabled = True

    ws = FakeWS([_msg("l2_data", 0), _msg("heartbeats", 1), resume, _msg("l2_data", 2)])
    connect = FakeConnect([ws])
    _run(store, connect)
    store.close()
    assert len(_stream(tmp_path, "heartbeats")) == 1
    assert not (tmp_path / "coinbase_ws" / "l2_data").exists()  # no delta without its snapshot
    names = [e.get("event") for e in _events(tmp_path)]
    assert "raw_discard_start" in names and "resubscribe_after_pause" in names


class FailingDataStore(SegmentStore):
    def write(self, stream, record):
        if stream.endswith("l2_data"):
            raise StoreError(f"{stream}: disk full")
        return super().write(stream, record)


def test_storage_failure_is_fatal_not_a_disconnect(tmp_path):
    store = FailingDataStore(tmp_path, "r")
    with pytest.raises(StoreError, match="disk full"):
        _run(store, FakeConnect([FakeWS([_msg("l2_data", 0)])]))
    store.close()
    assert "disconnect" not in [e.get("event") for e in _events(tmp_path)]


def test_non_object_json_is_kept_as_unparsed_not_a_disconnect(tmp_path):
    store = SegmentStore(tmp_path, "r")
    _run(store, FakeConnect([FakeWS(["[]", "null", _msg("l2_data", 0)])]))
    store.close()
    assert [r["payload"] for r in _stream(tmp_path, "unparsed")] == ["[]", "null"]
    assert "disconnect" not in [e.get("event") for e in _events(tmp_path)]
