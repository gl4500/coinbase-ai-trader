"""Tests for supply-snapshot parquet writer."""

from __future__ import annotations

import pytest

from tools.strategy_discovery import build_supply_snapshot as bss


def test_save_and_load_supply_snapshot_roundtrip(tmp_path):
    """save -> load returns the same rows."""
    rows = [
        {
            "pid": "BTC-USD",
            "circulating": 19_700_000.0,
            "total": 19_700_000.0,
            "max_supply": 21_000_000.0,
            "ingest_ts": 1_700_000_000,
        },
        {
            "pid": "ETH-USD",
            "circulating": 120_000_000.0,
            "total": 120_000_000.0,
            "max_supply": None,
            "ingest_ts": 1_700_000_000,
        },
    ]
    path = tmp_path / "snapshot.parquet"
    bss.save_snapshot(str(path), rows)
    loaded = bss.load_snapshot(str(path))

    assert len(loaded) == 2
    by_pid = {r["pid"]: r for r in loaded}
    assert by_pid["BTC-USD"]["circulating"] == 19_700_000.0
    assert by_pid["BTC-USD"]["max_supply"] == 21_000_000.0
    assert by_pid["ETH-USD"]["max_supply"] is None


def test_load_snapshot_missing_returns_empty_list(tmp_path):
    assert bss.load_snapshot(str(tmp_path / "no_such.parquet")) == []


def test_save_snapshot_dedups_by_pid_last_wins(tmp_path):
    """If the same pid appears twice in input, the LAST row wins."""
    rows = [
        {
            "pid": "BTC-USD",
            "circulating": 1.0,
            "total": 1.0,
            "max_supply": 21_000_000.0,
            "ingest_ts": 100,
        },
        {
            "pid": "BTC-USD",
            "circulating": 2.0,
            "total": 2.0,
            "max_supply": 21_000_000.0,
            "ingest_ts": 200,
        },
    ]
    path = tmp_path / "snapshot.parquet"
    bss.save_snapshot(str(path), rows)
    loaded = bss.load_snapshot(str(path))
    assert len(loaded) == 1
    assert loaded[0]["circulating"] == 2.0
    assert loaded[0]["ingest_ts"] == 200


@pytest.mark.asyncio
async def test_fetch_and_persist_one_pid_merges_with_existing(tmp_path, monkeypatch):
    """fetch_and_persist appends a new pid's snapshot to the existing parquet."""
    path = tmp_path / "snapshot.parquet"
    # Seed with ETH already present.
    bss.save_snapshot(
        str(path),
        [
            {
                "pid": "ETH-USD",
                "circulating": 120e6,
                "total": 120e6,
                "max_supply": None,
                "ingest_ts": 100,
            },
        ],
    )

    async def _fake_fetch(pid):
        return (19_700_000.0, 19_700_000.0, 21_000_000.0)

    monkeypatch.setattr(bss, "fetch_supply_snapshot", _fake_fetch)
    monkeypatch.setattr(bss, "_now_ts", lambda: 200)

    await bss.fetch_and_persist(["BTC-USD"], parquet_path=str(path), sleep_secs=0.0)

    loaded = bss.load_snapshot(str(path))
    by_pid = {r["pid"]: r for r in loaded}
    assert set(by_pid.keys()) == {"BTC-USD", "ETH-USD"}
    assert by_pid["BTC-USD"]["circulating"] == 19_700_000.0
    assert by_pid["BTC-USD"]["ingest_ts"] == 200
    # ETH untouched
    assert by_pid["ETH-USD"]["ingest_ts"] == 100
