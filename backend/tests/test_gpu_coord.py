"""Tests for data/gpu_coord.py — polymarket_app side of cross-app GPU coordination.

Mirrors trading_app's test_gpu_coord.py. Same coord file format, same
training-mutex protocol — they coordinate via the shared file.
"""
import asyncio
import json
import os
import tempfile
import time

import pytest

from data.gpu_coord import OllamaCoordinator, STALE_AFTER_SECS


# ── OllamaCoordinator: per-app asyncio.Lock ───────────────────────────────

@pytest.mark.asyncio
async def test_concurrent_acquire_serializes(tmp_path):
    """Two coroutines both call acquire() — must run one at a time."""
    coord_path = str(tmp_path / "state.json")
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    order = []

    async def worker(label: str, work_secs: float):
        async with coord.acquire(expected_ms=int(work_secs * 1000)):
            order.append(f"start:{label}")
            await asyncio.sleep(work_secs)
            order.append(f"end:{label}")

    await asyncio.gather(worker("A", 0.05), worker("B", 0.05))
    assert len(order) == 4
    first = order[0].split(":", 1)[1]
    second = order[2].split(":", 1)[1]
    assert order[0] == f"start:{first}"
    assert order[1] == f"end:{first}"
    assert order[2] == f"start:{second}"
    assert order[3] == f"end:{second}"


# ── OllamaCoordinator: exposure round-trip ────────────────────────────────

def test_update_writes_app_record(tmp_path):
    coord_path = str(tmp_path / "state.json")
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    coord.update_exposure(8_500.50)

    with open(coord_path) as f:
        state = json.load(f)
    assert "polymarket_app" in state
    assert state["polymarket_app"]["exposure_usd"] == pytest.approx(8_500.50)
    assert state["polymarket_app"]["updated_at"] > time.time() - 5


def test_update_preserves_other_apps(tmp_path):
    """Updating polymarket_app must not erase trading_app's record."""
    coord_path = str(tmp_path / "state.json")
    os.makedirs(os.path.dirname(coord_path), exist_ok=True)
    with open(coord_path, "w") as f:
        json.dump({
            "trading_app": {"exposure_usd": 12_500.0, "updated_at": time.time()},
        }, f)
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    coord.update_exposure(2_000.0)
    with open(coord_path) as f:
        state = json.load(f)
    assert "polymarket_app" in state
    assert "trading_app" in state
    assert state["trading_app"]["exposure_usd"] == pytest.approx(12_500.0)


def test_update_failure_does_not_raise(tmp_path):
    """A path the coordinator can't write to must be a soft failure."""
    blocking_file = tmp_path / "blocker.file"
    blocking_file.write_text("not a directory")
    bad_path = str(blocking_file / "state.json")
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=bad_path)
    coord.update_exposure(100.0)  # must not raise


# ── Stale-entry handling ──────────────────────────────────────────────────

def test_stale_other_app_treated_as_zero(tmp_path):
    coord_path = str(tmp_path / "state.json")
    with open(coord_path, "w") as f:
        json.dump({
            "trading_app": {
                "exposure_usd": 50_000.0,
                "updated_at":   time.time() - (STALE_AFTER_SECS + 60),
            },
        }, f)
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    assert coord._other_app_priority_exposure() == 0.0


def test_fresh_other_app_counts(tmp_path):
    coord_path = str(tmp_path / "state.json")
    with open(coord_path, "w") as f:
        json.dump({
            "trading_app": {
                "exposure_usd": 50_000.0,
                "updated_at":   time.time() - 5,
            },
        }, f)
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    assert coord._other_app_priority_exposure() == pytest.approx(50_000.0)


# ── acquire bypass + bounded wait ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_higher_exposure_fires_immediately(tmp_path):
    coord_path = str(tmp_path / "state.json")
    with open(coord_path, "w") as f:
        json.dump({
            "trading_app": {"exposure_usd": 1_000.0, "updated_at": time.time()},
        }, f)
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    coord.update_exposure(50_000.0)

    t0 = time.monotonic()
    async with coord.acquire(expected_ms=1_000):
        pass
    assert time.monotonic() - t0 < 0.2


@pytest.mark.asyncio
async def test_lower_exposure_waits_then_fires(tmp_path, monkeypatch):
    coord_path = str(tmp_path / "state.json")
    with open(coord_path, "w") as f:
        json.dump({
            "trading_app": {"exposure_usd": 50_000.0, "updated_at": time.time()},
        }, f)
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    coord.update_exposure(1_000.0)

    from data import gpu_coord as _gc
    monkeypatch.setattr(_gc, "MAX_WAIT_SECS", 0.5)
    monkeypatch.setattr(_gc, "POLL_WAIT_SECS", 0.1)

    t0 = time.monotonic()
    async with coord.acquire(expected_ms=1_000):
        pass
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.4
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_missing_coord_file_does_not_block(tmp_path):
    coord_path = str(tmp_path / "missing" / "state.json")
    coord = OllamaCoordinator(app_name="polymarket_app", coord_path=coord_path)
    t0 = time.monotonic()
    async with coord.acquire(expected_ms=100):
        pass
    assert time.monotonic() - t0 < 0.2


# ── Training mutex ────────────────────────────────────────────────────────

def test_acquire_when_lock_missing(tmp_path):
    from data.gpu_coord import acquire_training_mutex, release_training_mutex
    lock_path = str(tmp_path / "training.lock")
    try:
        ok = acquire_training_mutex(app_name="polymarket_app", lock_path=lock_path)
        assert ok
        assert os.path.exists(lock_path)
    finally:
        release_training_mutex(app_name="polymarket_app", lock_path=lock_path)


def test_release_removes_lock_file(tmp_path):
    from data.gpu_coord import acquire_training_mutex, release_training_mutex
    lock_path = str(tmp_path / "training.lock")
    acquire_training_mutex(app_name="polymarket_app", lock_path=lock_path)
    release_training_mutex(app_name="polymarket_app", lock_path=lock_path)
    assert not os.path.exists(lock_path)


def test_release_when_not_holder_does_not_raise(tmp_path):
    from data.gpu_coord import release_training_mutex
    lock_path = str(tmp_path / "training.lock")
    with open(lock_path, "w") as f:
        json.dump({"app": "trading_app", "pid": 99999, "started_at": time.time()}, f)
    release_training_mutex(app_name="polymarket_app", lock_path=lock_path)
    assert os.path.exists(lock_path)


def test_reclaim_when_holder_stale(tmp_path):
    """Stale lock (held >2h) → reclaim immediately."""
    from data.gpu_coord import acquire_training_mutex, release_training_mutex
    lock_path = str(tmp_path / "training.lock")
    with open(lock_path, "w") as f:
        json.dump({
            "app": "trading_app",
            "pid": os.getpid(),
            "started_at": time.time() - (3 * 60 * 60),  # 3h ago
        }, f)
    try:
        ok = acquire_training_mutex(
            app_name="polymarket_app", lock_path=lock_path, max_wait_secs=1
        )
        assert ok
    finally:
        release_training_mutex(app_name="polymarket_app", lock_path=lock_path)


def test_timeout_when_held_by_live_peer(tmp_path, monkeypatch):
    """Live peer + fresh lock → wait up to max_wait_secs, return False."""
    from data.gpu_coord import acquire_training_mutex
    lock_path = str(tmp_path / "training.lock")
    with open(lock_path, "w") as f:
        json.dump({
            "app": "trading_app",
            "pid": os.getpid(),
            "started_at": time.time(),
        }, f)
    monkeypatch.setattr("data.gpu_coord.TRAINING_LOCK_POLL_SECS", 0.05)
    t0 = time.monotonic()
    ok = acquire_training_mutex(
        app_name="polymarket_app", lock_path=lock_path, max_wait_secs=0.3
    )
    elapsed = time.monotonic() - t0
    assert not ok
    assert elapsed >= 0.25
    assert elapsed < 1.0


def test_reentrant_same_app_same_pid(tmp_path):
    from data.gpu_coord import acquire_training_mutex, release_training_mutex
    lock_path = str(tmp_path / "training.lock")
    try:
        first  = acquire_training_mutex(app_name="polymarket_app", lock_path=lock_path)
        second = acquire_training_mutex(app_name="polymarket_app", lock_path=lock_path)
        assert first
        assert second
    finally:
        release_training_mutex(app_name="polymarket_app", lock_path=lock_path)
