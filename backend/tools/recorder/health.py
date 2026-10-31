"""Health file + disk guard. Low disk pauses raw WS capture (heartbeats and polls continue)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from tools.recorder.store import envelope

MIN_FREE_BYTES = 100 * 1024**3


def disk_ok(free_bytes: int, min_free: int = MIN_FREE_BYTES) -> bool:
    return free_bytes >= min_free


def apply_disk_guard(store, free_bytes: int, clock, min_free: int = MIN_FREE_BYTES) -> None:
    ok = disk_ok(free_bytes, min_free)
    if ok != store.raw_enabled:
        store.raw_enabled = ok
        store.write(
            "recorder/events",
            envelope(
                "recorder",
                "event",
                clock(),
                meta={
                    "event": "raw_resumed" if ok else "raw_paused_low_disk",
                    "free_bytes": free_bytes,
                },
            ),
        )


def status_snapshot(store, now_ns: int, free_bytes: int, pid: int, expected) -> dict:
    streams = {
        name: {**st, "age_s": round((now_ns - st["last_received_ns"]) / 1e9, 3)}
        for name, st in sorted(store.stats.items())
    }
    return {
        "now_ns": now_ns,
        "mono_ns": time.monotonic_ns(),
        "pid": pid,
        "run_id": store.run_id,
        "free_bytes": free_bytes,
        "raw_enabled": store.raw_enabled,
        "streams": streams,
        "never_seen": sorted(s for s in expected if s not in store.stats),
    }


async def run_health(
    store,
    out: Path,
    stop: asyncio.Event,
    *,
    clock=time.time_ns,
    free=lambda p: shutil.disk_usage(p).free,
    interval_s: float = 60.0,
    pid: int = os.getpid(),
    expected=(),
) -> None:
    out = Path(out)
    stop_file = out / "STOP"  # Windows cannot deliver Ctrl+C to a detached process
    while True:
        if stop_file.exists():
            stop_file.unlink()
            store.write(
                "recorder/events",
                envelope("recorder", "event", clock(), meta={"event": "stop_requested"}),
            )
            stop.set()
        now, free_bytes = clock(), free(out)
        apply_disk_guard(store, free_bytes, clock)
        store.flush_all()
        tmp = out / "status.json.tmp"
        tmp.write_text(json.dumps(status_snapshot(store, now, free_bytes, pid, expected), indent=1))
        tmp.replace(out / "status.json")
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass
