"""Tests for shadow-week monitoring endpoints — /api/compare and /api/equity_curve.

These power the dashboard's sticky comparison header (#52) and equity curve
chart (#54). Both endpoints read from sqlite databases (coinbase.db for v3,
coinbase_dev.db for v4.5). Tests use a tmp_path with synthetic DBs.
"""
from __future__ import annotations
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

from main import (  # noqa: E402
    _read_agent_state,
    _equity_curve_series,
)


def _make_agent_state_db(path: str, *, balance: float, realized: float,
                        positions: dict) -> None:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE agent_state (
        agent TEXT PRIMARY KEY,
        balance REAL, realized_pnl REAL, positions_json TEXT
    )""")
    con.execute(
        "INSERT INTO agent_state VALUES (?, ?, ?, ?)",
        ("CNN", balance, realized, json.dumps(positions)),
    )
    con.commit()
    con.close()


def _make_trades_db(path: str, trades: list) -> None:
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE trades (
        id INTEGER PRIMARY KEY,
        agent TEXT, product_id TEXT, entry_price REAL, exit_price REAL,
        size REAL, usd_open REAL, usd_close REAL, pnl REAL, pct_pnl REAL,
        hold_secs INTEGER, trigger_open TEXT, trigger_close TEXT,
        balance_after REAL, opened_at TEXT, closed_at TEXT
    )""")
    for t in trades:
        con.execute(
            "INSERT INTO trades (agent, product_id, pnl, closed_at) "
            "VALUES (?, ?, ?, ?)",
            ("CNN", t["pid"], t["pnl"], t["closed_at"]),
        )
    con.commit()
    con.close()


class TestReadAgentState:
    def test_returns_none_when_db_missing(self, tmp_path):
        result = _read_agent_state(str(tmp_path / "does_not_exist.db"))
        assert result is None

    def test_zero_state_when_no_agent_row(self, tmp_path):
        path = str(tmp_path / "empty.db")
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE agent_state (
            agent TEXT PRIMARY KEY, balance REAL, realized_pnl REAL,
            positions_json TEXT)""")
        con.commit(); con.close()
        result = _read_agent_state(path)
        assert result == {
            "balance": 0.0, "realized_pnl": 0.0,
            "open_positions": 0, "unrealized_pnl_est": 0.0,
        }

    def test_reads_state_with_no_positions(self, tmp_path):
        path = str(tmp_path / "v3.db")
        _make_agent_state_db(path, balance=500.0, realized=-25.0, positions={})
        result = _read_agent_state(path)
        assert result["balance"] == 500.0
        assert result["realized_pnl"] == -25.0
        assert result["open_positions"] == 0
        assert result["unrealized_pnl_est"] == 0.0

    def test_reads_state_with_positions_and_unrealized(self, tmp_path):
        path = str(tmp_path / "v3.db")
        positions = {
            "BTC-USD": {"size": 0.001, "avg_price": 75000.0,
                        "position_dollars": 76.5},  # +$1.5 unrealized
            "ETH-USD": {"size": 0.05, "avg_price": 2000.0,
                        "position_dollars": 105.0},  # +$5 unrealized
        }
        _make_agent_state_db(path, balance=300.0, realized=-50.0,
                             positions=positions)
        result = _read_agent_state(path)
        assert result["balance"] == 300.0
        assert result["realized_pnl"] == -50.0
        assert result["open_positions"] == 2
        # Unrealized = $76.5 - 0.001*75000=$75 → +$1.5
        #            + $105 - 0.05*2000=$100  → +$5
        #            = +$6.5
        assert result["unrealized_pnl_est"] == pytest.approx(6.5)

    def test_unrealized_falls_back_to_zero_when_position_dollars_absent(self, tmp_path):
        path = str(tmp_path / "v3.db")
        positions = {
            "BTC-USD": {"size": 0.001, "avg_price": 75000.0},  # no position_dollars
        }
        _make_agent_state_db(path, balance=300.0, realized=0.0, positions=positions)
        result = _read_agent_state(path)
        # position_dollars defaults to size * avg_price = 75 → unrealized 0
        assert result["unrealized_pnl_est"] == pytest.approx(0.0)


class TestEquityCurveSeries:
    def test_empty_series_when_db_missing(self, tmp_path):
        result = _equity_curve_series(str(tmp_path / "missing.db"), days=7)
        assert result == []

    def test_empty_series_when_no_trades(self, tmp_path):
        path = str(tmp_path / "empty.db")
        _make_trades_db(path, trades=[])
        assert _equity_curve_series(path, days=7) == []

    def test_cumulative_within_window(self, tmp_path):
        path = str(tmp_path / "trades.db")
        now = datetime.now(timezone.utc)
        trades = [
            {"pid": "A-USD", "pnl": 2.0,
             "closed_at": (now - timedelta(days=2)).isoformat()},
            {"pid": "B-USD", "pnl": -1.0,
             "closed_at": (now - timedelta(days=1)).isoformat()},
            {"pid": "C-USD", "pnl": 5.0,
             "closed_at": (now - timedelta(hours=1)).isoformat()},
        ]
        _make_trades_db(path, trades)
        series = _equity_curve_series(path, days=7)
        assert len(series) == 3
        # Cumulative: 2.0, 1.0, 6.0
        assert series[0][1] == pytest.approx(2.0)
        assert series[1][1] == pytest.approx(1.0)
        assert series[2][1] == pytest.approx(6.0)

    def test_excludes_trades_outside_window(self, tmp_path):
        path = str(tmp_path / "trades.db")
        now = datetime.now(timezone.utc)
        trades = [
            {"pid": "OLD-USD", "pnl": 100.0,
             "closed_at": (now - timedelta(days=30)).isoformat()},
            {"pid": "NEW-USD", "pnl": 3.0,
             "closed_at": (now - timedelta(hours=1)).isoformat()},
        ]
        _make_trades_db(path, trades)
        series = _equity_curve_series(path, days=7)
        assert len(series) == 1
        assert series[0][1] == pytest.approx(3.0)

    def test_chronological_order(self, tmp_path):
        path = str(tmp_path / "trades.db")
        now = datetime.now(timezone.utc)
        # Insert out of order
        trades = [
            {"pid": "C-USD", "pnl": 1.0, "closed_at": (now - timedelta(hours=1)).isoformat()},
            {"pid": "A-USD", "pnl": 2.0, "closed_at": (now - timedelta(hours=3)).isoformat()},
            {"pid": "B-USD", "pnl": 3.0, "closed_at": (now - timedelta(hours=2)).isoformat()},
        ]
        _make_trades_db(path, trades)
        series = _equity_curve_series(path, days=7)
        # Series must be sorted ascending by time, so cumulative reflects time order:
        # A (-3h) = 2.0, B (-2h) = 5.0, C (-1h) = 6.0
        assert series[0][1] == pytest.approx(2.0)
        assert series[1][1] == pytest.approx(5.0)
        assert series[2][1] == pytest.approx(6.0)

    def test_handles_null_pnl_as_zero(self, tmp_path):
        path = str(tmp_path / "trades.db")
        con = sqlite3.connect(path)
        con.execute("""CREATE TABLE trades (
            id INTEGER PRIMARY KEY, agent TEXT, product_id TEXT,
            entry_price REAL, exit_price REAL, size REAL,
            usd_open REAL, usd_close REAL, pnl REAL, pct_pnl REAL,
            hold_secs INTEGER, trigger_open TEXT, trigger_close TEXT,
            balance_after REAL, opened_at TEXT, closed_at TEXT)""")
        now_iso = datetime.now(timezone.utc).isoformat()
        con.execute(
            "INSERT INTO trades (agent, product_id, pnl, closed_at) "
            "VALUES (?, ?, ?, ?)",
            ("CNN", "X-USD", None, now_iso),
        )
        con.commit(); con.close()
        series = _equity_curve_series(path, days=7)
        assert len(series) == 1
        assert series[0][1] == pytest.approx(0.0)
