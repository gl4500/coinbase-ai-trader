"""Resilience for the Aug-1 network-stall incident:

1. scan_loop_stale() — pure liveness predicate powering GET /api/health, so the
   launcher watchdog can restart a backend whose scan loop is hung while HTTP is up.
2. CoinbaseCNNAgent._scan_cycle() — wraps scan_all + _check_risk_exits in
   asyncio.wait_for so a hung network call can't freeze the loop for hours.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from main import scan_loop_stale  # noqa: E402


class TestScanLoopStale:
    def test_fresh_scan_not_stale(self):
        now = 1_000_000.0
        assert scan_loop_stale(now - 60, now, interval_secs=900, k=3) is False

    def test_old_scan_is_stale(self):
        now = 1_000_000.0
        assert scan_loop_stale(now - 3000, now, interval_secs=900, k=3) is True

    def test_none_is_grace_not_stale(self):
        # Loop hasn't scanned yet (just started) — don't trigger a restart loop.
        assert scan_loop_stale(None, 1_000_000.0, interval_secs=900, k=3) is False

    def test_boundary_just_under_threshold(self):
        now = 1_000_000.0
        assert scan_loop_stale(now - 2699, now, 900, 3) is False

    def test_boundary_just_over_threshold(self):
        now = 1_000_000.0
        assert scan_loop_stale(now - 2701, now, 900, 3) is True


@pytest.mark.asyncio
class TestScanCycleTimeout:
    async def test_hanging_scan_raises_timeout(self):
        from agents.cnn_agent import CoinbaseCNNAgent

        agent = CoinbaseCNNAgent()

        async def _hang(*a, **k):
            await asyncio.sleep(30)

        async def _noop(*a, **k):
            return None

        agent.scan_all = _hang
        agent._check_risk_exits = _noop
        with pytest.raises(asyncio.TimeoutError):
            await agent._scan_cycle(execute=False, order_executor=None, timeout=0.1)

    async def test_normal_cycle_runs_scan_then_risk(self):
        from agents.cnn_agent import CoinbaseCNNAgent

        agent = CoinbaseCNNAgent()
        calls: list[str] = []

        async def _scan(*a, **k):
            calls.append("scan")
            return []

        async def _risk(*a, **k):
            calls.append("risk")
            return None

        agent.scan_all = _scan
        agent._check_risk_exits = _risk
        await agent._scan_cycle(execute=True, order_executor=None, timeout=5)
        assert calls == ["scan", "risk"]
