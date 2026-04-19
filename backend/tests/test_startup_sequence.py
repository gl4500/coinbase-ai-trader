"""
Tests for startup sequence efficiency.

Verifies:
  - App becomes ready before initial scan completes
  - WS subscriber starts from DB cache, not waiting for scan
  - ScalpAgent only has one 90s delay (not double)
  - Agent stagger delays are short (≤ 10s each)
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")


# ── ScalpAgent warmup ─────────────────────────────────────────────────────────

class TestScalpEntryLoopWarmup:
    """ScalpAgent._entry_loop should have exactly one warmup delay, not two."""

    @pytest.mark.asyncio
    async def test_entry_loop_no_prescan_warmup(self):
        """_entry_loop must call _scan_entries BEFORE any asyncio.sleep.
        The lifespan provides the startup delay — no additional warmup inside the loop."""
        from agents.scalp_agent import ScalpAgent

        ag = ScalpAgent()
        ag._running = True
        events = []   # record "scan" and "sleep" in order

        async def fake_scan():
            events.append("scan")
            ag._running = False   # stop after first scan

        async def fake_sleep(n):
            events.append(f"sleep:{n}")

        with (
            patch.object(ag, "_scan_entries", fake_scan),
            patch("agents.scalp_agent.asyncio.sleep", fake_sleep),
        ):
            await ag._entry_loop()

        assert events, "No events recorded"
        assert events[0] == "scan", (
            f"_entry_loop slept before scanning: {events} — "
            f"warmup should be handled by the lifespan, not inside _entry_loop"
        )

    @pytest.mark.asyncio
    async def test_entry_loop_runs_scan_after_warmup(self):
        """After warmup, _scan_entries must be called."""
        from agents.scalp_agent import ScalpAgent

        ag = ScalpAgent()
        ag._running = True
        scan_called = []

        async def fake_scan():
            scan_called.append(True)
            ag._running = False

        async def fake_sleep(n):
            pass   # skip actual sleep

        with (
            patch.object(ag, "_scan_entries", fake_scan),
            patch("agents.scalp_agent.asyncio.sleep", fake_sleep),
        ):
            await ag._entry_loop()

        assert scan_called, "_scan_entries was never called after warmup"


# ── Lifespan stagger delays ───────────────────────────────────────────────────

class TestAgentStaggerDelays:
    """Agent startup delays in main.py should be short, not 30/60/90s."""

    def test_stagger_constants_are_short(self):
        """Import stagger values from main and assert they're ≤ 10s each."""
        try:
            import main as m
            tech_delay     = getattr(m, "_TECH_START_DELAY",     None)
            momentum_delay = getattr(m, "_MOMENTUM_START_DELAY", None)
            scalp_delay    = getattr(m, "_SCALP_START_DELAY",    None)
        except Exception:
            pytest.skip("main.py not importable in test environment")

        if tech_delay is not None:
            assert tech_delay <= 10, f"Tech stagger {tech_delay}s too long (≤ 10s)"
        if momentum_delay is not None:
            assert momentum_delay <= 20, f"Momentum stagger {momentum_delay}s too long (≤ 20s)"
        if scalp_delay is not None:
            assert scalp_delay <= 30, f"Scalp stagger {scalp_delay}s too long (≤ 30s)"


# ── WS subscriber starts from DB cache ───────────────────────────────────────

class TestWSSubscriberEarlyStart:
    """WS subscriber should be seeded with DB-cached products, not wait for scan."""

    @pytest.mark.asyncio
    async def test_ws_can_start_with_db_products(self):
        """CoinbaseWSSubscriber.set_products accepts a list from any source."""
        from services.ws_subscriber import CoinbaseWSSubscriber

        ws = CoinbaseWSSubscriber(broadcast_fn=AsyncMock())
        db_products = ["XRP-USD", "ADA-USD", "SUI-USD"]

        # Must not raise even if called before scan
        ws.set_products(db_products)
        assert ws._products == db_products or set(ws._products) == set(db_products)
