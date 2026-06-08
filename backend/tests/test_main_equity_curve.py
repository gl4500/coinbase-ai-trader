"""Tests for _equity_curve_series in main.py (task #81, PR #14 review).

Must be an async function backed by aiosqlite — not sync sqlite3 — so the
FastAPI /api/equity_curve endpoint doesn't block the event loop while
reading two DBs sequentially.
"""
from __future__ import annotations

import inspect
import os
import sqlite3
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from main import _equity_curve_series


def _seed_trades_db(db_path: str, rows: list[tuple[str, float]]) -> None:
    """Create a trades table with (closed_at, pnl) rows. Uses sync sqlite3 —
    fixture-only; runtime code uses aiosqlite."""
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE trades (id INTEGER PRIMARY KEY, closed_at TEXT, pnl REAL)"
    )
    con.executemany(
        "INSERT INTO trades (closed_at, pnl) VALUES (?, ?)", rows,
    )
    con.commit()
    con.close()


class TestEquityCurveSeries:
    def test_helper_is_async_function(self):
        """Locked-in contract: helper MUST be async so FastAPI await-points it."""
        assert inspect.iscoroutinefunction(_equity_curve_series), (
            "_equity_curve_series must be async — sync sqlite3 blocks the event "
            "loop while /api/equity_curve reads two DBs sequentially."
        )

    @pytest.mark.asyncio
    async def test_returns_cumulative_series_in_order(self, tmp_path):
        from datetime import datetime, timezone, timedelta
        db = str(tmp_path / "trades.db")
        now = datetime.now(timezone.utc)
        rows = [
            ((now - timedelta(days=3)).isoformat(), 5.0),
            ((now - timedelta(days=2)).isoformat(), -2.0),
            ((now - timedelta(days=1)).isoformat(), 3.0),
        ]
        _seed_trades_db(db, rows)

        series = await _equity_curve_series(db, days=7)

        assert len(series) == 3
        # cumulative: 5, 3, 6
        assert [p[1] for p in series] == [5.0, 3.0, 6.0]
        # chronological order preserved
        assert series[0][0] < series[1][0] < series[2][0]

    @pytest.mark.asyncio
    async def test_returns_empty_when_db_missing(self, tmp_path):
        """Missing DB file → empty list, NOT exception (silent failure preserved
        from sync version; matches the endpoint's caller contract)."""
        missing = str(tmp_path / "nope.db")
        series = await _equity_curve_series(missing, days=7)
        assert series == []

    @pytest.mark.asyncio
    async def test_excludes_rows_outside_days_window(self, tmp_path):
        from datetime import datetime, timezone, timedelta
        db = str(tmp_path / "trades.db")
        now = datetime.now(timezone.utc)
        rows = [
            ((now - timedelta(days=10)).isoformat(), 100.0),  # outside 7d
            ((now - timedelta(days=2)).isoformat(),   5.0),
        ]
        _seed_trades_db(db, rows)

        series = await _equity_curve_series(db, days=7)

        assert len(series) == 1
        assert series[0][1] == 5.0
