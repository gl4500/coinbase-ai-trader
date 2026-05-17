"""Tests for backend/tools/close_tech_positions.py (#311-refactor-c).

Three cases:
1. No-op when no TECH agent_state row exists.
2. No-op when agent_state.positions_json is empty dict.
3. Writes one trade row per open position; zeros positions_json.
"""
import asyncio
import json
import os
import sqlite3
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_trade_schema(db_path):
    """Create minimal trades + agent_state tables matching prod schema."""
    c = sqlite3.connect(db_path)
    c.execute("""
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            agent TEXT, product_id TEXT,
            entry_price REAL, exit_price REAL, size REAL,
            usd_open REAL, usd_close REAL, pnl REAL, pct_pnl REAL,
            hold_secs INTEGER,
            trigger_open TEXT, trigger_close TEXT,
            balance_after REAL,
            opened_at TEXT, closed_at TEXT
        )""")
    c.execute("""
        CREATE TABLE agent_state (
            id INTEGER PRIMARY KEY,
            agent TEXT, balance REAL, realized_pnl REAL,
            positions_json TEXT, high_water_json TEXT
        )""")
    c.commit(); c.close()


class TestCloseTechPositions:
    def test_no_op_when_no_tech_agent_state_row(self, tmp_path):
        from tools.close_tech_positions import close_tech_positions
        db = tmp_path / "test.db"
        _make_trade_schema(db)
        result = asyncio.run(close_tech_positions(db_path=str(db)))
        assert result["n_closed"] == 0
        c = sqlite3.connect(db)
        n_trades = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        c.close()
        assert n_trades == 0

    def test_no_op_when_positions_json_empty(self, tmp_path):
        from tools.close_tech_positions import close_tech_positions
        db = tmp_path / "test.db"
        _make_trade_schema(db)
        c = sqlite3.connect(db)
        c.execute(
            "INSERT INTO agent_state (agent, balance, realized_pnl, positions_json) "
            "VALUES ('TECH', 1000.0, 0.0, ?)",
            ("{}",),
        )
        c.commit(); c.close()
        result = asyncio.run(close_tech_positions(db_path=str(db)))
        assert result["n_closed"] == 0
        c = sqlite3.connect(db)
        n_trades = c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        c.close()
        assert n_trades == 0

    def test_writes_trade_row_per_position_and_zeros_state(
        self, tmp_path, monkeypatch
    ):
        from tools.close_tech_positions import close_tech_positions
        db = tmp_path / "test.db"
        _make_trade_schema(db)
        positions = {
            "BTC-USD": {"size": 0.001, "avg_price": 50000.0},
            "ETH-USD": {"size": 0.5,    "avg_price": 3000.0},
            "SOL-USD": {"size": 10.0,   "avg_price": 100.0},
        }
        c = sqlite3.connect(db)
        c.execute(
            "INSERT INTO agent_state (agent, balance, realized_pnl, positions_json) "
            "VALUES ('TECH', 100.0, -50.0, ?)",
            (json.dumps(positions),),
        )
        c.commit(); c.close()

        # Stub the live-price fetch so the test doesn't hit Coinbase
        async def _fake_get_product(pid):
            return {"price": {"BTC-USD": "55000.0",
                              "ETH-USD": "3300.0",
                              "SOL-USD": "110.0"}[pid]}

        import clients.coinbase_client as cb
        monkeypatch.setattr(cb, "get_product", _fake_get_product)

        result = asyncio.run(close_tech_positions(db_path=str(db)))
        assert result["n_closed"] == 3
        assert result["fallback_used"] == 0  # all prices fetched OK
        # 3 positions * ~10% gain each → realized_pnl moves up
        assert result["final_realized_pnl"] > -50.0

        # Verify trades + agent_state mutations
        c = sqlite3.connect(db)
        rows = c.execute(
            "SELECT product_id, trigger_close FROM trades WHERE agent='TECH'"
        ).fetchall()
        ps_after = c.execute(
            "SELECT positions_json FROM agent_state WHERE agent='TECH'"
        ).fetchone()[0]
        c.close()
        assert len(rows) == 3
        assert all(r[1] == "MANUAL_TECH_RETIREMENT" for r in rows)
        assert json.loads(ps_after) == {}
