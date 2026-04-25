"""
Smoke tests for tools/checkin.py — proves the operator snapshot script
imports cleanly and prints all five sections without crashing on an empty
(but schema-correct) database, with no progress file present.

Catches: import errors, schema drift in the SELECT statements, and crashes
when the four hot tables exist but contain zero rows.
"""

import importlib.util
import io
import os
import sqlite3
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKIN_PATH = REPO_ROOT / "tools" / "checkin.py"


@pytest.fixture
def checkin_module():
    """Load tools/checkin.py as a module without requiring a package layout."""
    spec = importlib.util.spec_from_file_location("checkin", CHECKIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def empty_schema_db(tmp_path):
    """SQLite file with the four hot tables created but no rows inserted."""
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY, agent TEXT, product_id TEXT,
            entry_price REAL, exit_price REAL, size REAL,
            usd_open REAL, usd_close REAL, pnl REAL, pct_pnl REAL,
            hold_secs INTEGER, trigger_open TEXT, trigger_close TEXT,
            balance_after REAL, opened_at REAL, closed_at REAL
        );
        CREATE TABLE cnn_scans (
            id INTEGER PRIMARY KEY, product_id TEXT, price REAL,
            cnn_prob REAL, llm_prob REAL, model_prob REAL,
            cnn_weight REAL, llm_weight REAL,
            side TEXT, strength REAL, signal_gen TEXT, regime TEXT,
            adx REAL, rsi REAL, macd REAL, mfi REAL, stoch_k REAL, atr REAL,
            scanned_at REAL, vwap_dist REAL, fast_rsi REAL,
            velocity REAL, vol_z REAL
        );
        CREATE TABLE signal_outcomes (
            id INTEGER PRIMARY KEY, source TEXT, product_id TEXT,
            side TEXT, confidence REAL,
            entry_price REAL, exit_price REAL, pct_change REAL,
            outcome TEXT, indicators_json TEXT, lesson_text TEXT,
            check_after REAL, checked_at REAL, created_at REAL
        );
        CREATE TABLE positions (
            id INTEGER PRIMARY KEY, product_id TEXT, base_currency TEXT,
            side TEXT, size REAL, avg_price REAL, current_price REAL,
            initial_value REAL, current_value REAL,
            cash_pnl REAL, pct_pnl REAL, updated_at REAL
        );
        """
    )
    conn.commit()
    conn.close()
    return str(db)


def _run(checkin_module, db_path, progress_path):
    """Run checkin.main() with stdout captured, return (rc, captured_text)."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = checkin_module.main(
            ["--db", db_path, "--progress", progress_path]
        )
    return rc, buf.getvalue()


class TestCheckinSmoke:
    def test_main_returns_zero_on_empty_db(self, checkin_module, empty_schema_db, tmp_path):
        rc, _ = _run(checkin_module, empty_schema_db, str(tmp_path / "missing.json"))
        assert rc == 0

    def test_all_section_headers_print(self, checkin_module, empty_schema_db, tmp_path):
        _, out = _run(checkin_module, empty_schema_db, str(tmp_path / "missing.json"))
        for header in (
            "Training",
            "Scans (last 20 min)",
            "Outcomes (last 60 min)",
            "Recent closed trades (24h, newest first)",
            "Open positions",
        ):
            assert header in out, f"missing section: {header!r}\n----\n{out}"

    def test_handles_missing_progress_file(self, checkin_module, empty_schema_db, tmp_path):
        _, out = _run(checkin_module, empty_schema_db, str(tmp_path / "does_not_exist.json"))
        assert "(no progress file)" in out

    def test_renders_progress_file_when_present(self, checkin_module, empty_schema_db, tmp_path):
        prog = tmp_path / "prog.json"
        prog.write_text('{"status": "running", "pid": 9999, "started_at": 1700000000.0}')
        _, out = _run(checkin_module, empty_schema_db, str(prog))
        assert "running" in out
        assert "9999" in out

    def test_returns_one_when_db_missing(self, checkin_module, tmp_path):
        rc, _ = _run(
            checkin_module,
            str(tmp_path / "no_such.db"),
            str(tmp_path / "no_such.json"),
        )
        assert rc == 1
