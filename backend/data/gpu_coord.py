"""
GPU/Ollama coordination across apps.

Mirror of trading_app's `data/gpu_coord.py` — same coord file, same
training-mutex file, same protocol. The two apps coordinate by both
writing to `~/.ollama-coord/state.json` and `~/.ollama-coord/training.lock`.

Two coordination primitives:

  • OllamaCoordinator — for inference (seconds-long calls). Layered as
    per-app asyncio.Lock + cross-app dollar-at-risk priority via shared
    coord file. Best-effort: missing file is fine, stale entries are
    fine, never raises.

  • acquire_training_mutex / release_training_mutex — for retraining
    (minutes-long blocking GPU events). Sync API (training runs in a
    subprocess). One holder at a time across apps via a separate lock
    file. Stale-PID reclaim handles crashed peers.

trading_app side: docs/superpowers/plans/2026-04-28-gpu-sequencing-design.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

# How long an entry in the coord file is considered fresh. Beyond this,
# the entry is treated as if exposure_usd=0 (the other app is down or stuck).
STALE_AFTER_SECS = 60.0

# Maximum total wait in acquire() before firing anyway.
MAX_WAIT_SECS = 10.0

# Per-poll wait when yielding to a higher-priority app.
POLL_WAIT_SECS = 1.0


def _default_coord_path() -> str:
    """Default coord file location: ~/.ollama-coord/state.json.
    Override via the OLLAMA_COORD_FILE env var.

    NB: trading_app uses the same default — both apps coordinate via the
    same file. Setting OLLAMA_COORD_FILE in only one app's .env breaks
    coordination silently."""
    override = os.getenv("OLLAMA_COORD_FILE")
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".ollama-coord", "state.json")


def _atomic_write_json(path: str, payload: dict) -> None:
    """Best-effort atomic JSON write: write to .tmp, then rename. On Windows,
    rename across the same directory is atomic."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, path)


def _read_json(path: str) -> dict:
    """Best-effort JSON read. Returns {} on any IO/parse error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


class OllamaCoordinator:
    """Coordinates Ollama access for one app.

    Usage:
        coord = OllamaCoordinator(app_name="polymarket_app")
        coord.update_exposure(8_500.0)
        async with coord.acquire(expected_ms=25_000):
            response = await httpx_client.post(...)
    """

    def __init__(
        self,
        app_name: str,
        coord_path: Optional[str] = None,
    ):
        self._app_name = app_name
        self._coord_path = coord_path or _default_coord_path()
        # Per-process serializer: at most one outstanding Ollama call from this app.
        self._lock = asyncio.Lock()
        # Cached last known exposure for telemetry.
        self._last_exposure: float = 0.0

    @property
    def coord_path(self) -> str:
        return self._coord_path

    @property
    def last_exposure(self) -> float:
        return self._last_exposure

    def update_exposure(self, exposure_usd: float) -> None:
        """Record this app's current capital at risk to the coord file.
        Call periodically (every cycle or every 30s) so cross-app priority
        reflects current state. Failures here are logged at debug and
        don't raise — coordination is best-effort."""
        try:
            now = time.time()
            state = _read_json(self._coord_path)
            state[self._app_name] = {
                "exposure_usd": float(exposure_usd),
                "updated_at":   now,
            }
            _atomic_write_json(self._coord_path, state)
            self._last_exposure = float(exposure_usd)
        except Exception as exc:
            logger.debug(f"gpu_coord: failed to update exposure for {self._app_name}: {exc}")

    def _other_app_priority_exposure(self) -> float:
        """Sum of OTHER apps' fresh exposures. Stale entries (>STALE_AFTER_SECS)
        are excluded — they likely belong to a crashed/stopped app."""
        try:
            state = _read_json(self._coord_path)
            now = time.time()
            total = 0.0
            for name, entry in state.items():
                if name == self._app_name:
                    continue
                if not isinstance(entry, dict):
                    continue
                ts = float(entry.get("updated_at", 0) or 0)
                if now - ts > STALE_AFTER_SECS:
                    continue
                total += float(entry.get("exposure_usd", 0) or 0)
            return total
        except Exception:
            return 0.0

    @asynccontextmanager
    async def acquire(self, expected_ms: int = 30_000):
        """Async context manager: serialize Ollama calls per-app, yield to
        any other app whose fresh exposure exceeds ours, with a bounded wait.

        Always yields control eventually — if the bounded wait expires while
        another app is winning priority, fires anyway rather than skipping
        the cycle. Better to miss priority than starve the agent.
        """
        async with self._lock:  # Layer 1: per-app serializer (always applies)
            # Layer 2: cross-app priority wait (best-effort)
            t_start = time.monotonic()
            while True:
                other_exposure = self._other_app_priority_exposure()
                if self._last_exposure >= other_exposure:
                    break  # we win or tie — fire now
                elapsed = time.monotonic() - t_start
                if elapsed >= MAX_WAIT_SECS:
                    logger.debug(
                        f"gpu_coord: {self._app_name} exceeded "
                        f"{MAX_WAIT_SECS:.0f}s wait — firing anyway "
                        f"(our exposure ${self._last_exposure:,.0f} < "
                        f"other ${other_exposure:,.0f})"
                    )
                    break
                # Wait briefly for the other app to release, then re-check
                await asyncio.sleep(POLL_WAIT_SECS)
            yield  # caller now makes the Ollama call


# ── Module-level singleton for polymarket_app ─────────────────────────────
# All Ollama callers in polymarket import this instance directly so they
# share the same per-process lock.
ollama_coord = OllamaCoordinator(app_name="polymarket_app")


# ── Training mutex (Option F) ─────────────────────────────────────────────
# Separate from inference coordination because retrains take MINUTES (not
# seconds) and must be exclusive. Implemented as a sync API since
# polymarket runs training in a separate subprocess (train_worker.py).

# A held training lock older than this is assumed crashed and is reclaimed.
TRAINING_LOCK_STALE_SECS = 2 * 60 * 60  # 2 hours
# Maximum total wait when contending for the training lock.
TRAINING_LOCK_MAX_WAIT_SECS = 60 * 60   # 1 hour
# Poll interval while waiting.
TRAINING_LOCK_POLL_SECS = 30.0


def _training_lock_path() -> str:
    return os.path.join(os.path.dirname(_default_coord_path()), "training.lock")


def _pid_alive(pid: int) -> bool:
    """True when a process with this PID is alive on the local machine.
    Returns True on any error so we don't spuriously steal a live lock."""
    if pid <= 0:
        return False
    try:
        import psutil  # type: ignore
        return psutil.pid_exists(pid)
    except Exception:
        # If we can't check, conservatively assume alive — never steal a
        # lock just because the check itself failed.
        return True


def acquire_training_mutex(
    app_name: str = "polymarket_app",
    lock_path: Optional[str] = None,
    max_wait_secs: float = TRAINING_LOCK_MAX_WAIT_SECS,
) -> bool:
    """Sync-API training mutex. Returns True when acquired, False on timeout.

    Acquisition rules:
      • If lock file is missing → claim it.
      • If lock file holds another app's PID and that PID is alive AND
        the lock is fresh (< TRAINING_LOCK_STALE_SECS) → wait + retry.
      • If the holder's PID is dead OR the lock is older than the stale
        threshold → reclaim (the holder crashed mid-train).
      • Re-entrant: if same app + same PID already holds, return True
        immediately without rewriting.

    Caller is responsible for calling release_training_mutex() in a
    finally block.
    """
    path = lock_path or _training_lock_path()
    pid = os.getpid()
    t_start = time.monotonic()
    while True:
        existing = _read_json(path)
        if not existing:
            try:
                _atomic_write_json(path, {
                    "app": app_name, "pid": pid, "started_at": time.time(),
                })
                logger.info(f"gpu_coord: training mutex acquired by {app_name} pid={pid}")
                return True
            except Exception as exc:
                logger.warning(f"gpu_coord: could not write training mutex: {exc}")
                return False

        held_app = existing.get("app", "")
        held_pid = int(existing.get("pid", 0) or 0)
        held_started = float(existing.get("started_at", 0) or 0)
        age = time.time() - held_started

        # Re-entrant case: we already hold it
        if held_app == app_name and held_pid == pid:
            return True

        # Reclaim on stale or dead-holder
        if age > TRAINING_LOCK_STALE_SECS or not _pid_alive(held_pid):
            logger.warning(
                f"gpu_coord: reclaiming stale training mutex held by "
                f"{held_app} pid={held_pid} age={age:.0f}s "
                f"(alive={_pid_alive(held_pid)})"
            )
            try:
                _atomic_write_json(path, {
                    "app": app_name, "pid": pid, "started_at": time.time(),
                })
                return True
            except Exception as exc:
                logger.warning(f"gpu_coord: reclaim write failed: {exc}")
                return False

        # Held by a live peer — wait
        if time.monotonic() - t_start >= max_wait_secs:
            logger.warning(
                f"gpu_coord: training mutex contention timeout — "
                f"{held_app} pid={held_pid} still holds (age={age:.0f}s)"
            )
            return False
        time.sleep(TRAINING_LOCK_POLL_SECS)


def release_training_mutex(
    app_name: str = "polymarket_app",
    lock_path: Optional[str] = None,
) -> None:
    """Release the training mutex if we hold it. Safe to call when we don't —
    silently no-ops in that case so finally-blocks are simple."""
    path = lock_path or _training_lock_path()
    try:
        existing = _read_json(path)
        if (existing.get("app") == app_name
                and int(existing.get("pid", 0) or 0) == os.getpid()):
            os.remove(path)
            logger.info(f"gpu_coord: training mutex released by {app_name}")
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"gpu_coord: failed to release training mutex: {exc}")
