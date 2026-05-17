"""TDD tests for backend/migrations/mc_telemetry_20260516.py.

Contract:
    run(db_path) -> dict
        - Idempotently adds xgb_prob_stdev REAL and mc_telemetry TEXT
          columns to the cnn_scans table.
        - Detects existing columns via PRAGMA table_info; never errors on
          already-applied state.
        - Returns {"added": [...col_names], "already_present": [...col_names]}.
"""
import os
import sqlite3
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _make_legacy_cnn_scans(db_path):
    """Create the cnn_scans table at its pre-MC schema."""
    c = sqlite3.connect(db_path)
    c.execute("""
        CREATE TABLE cnn_scans (
            id INTEGER PRIMARY KEY,
            product_id TEXT NOT NULL,
            price REAL,
            cnn_prob REAL,
            llm_prob REAL,
            model_prob REAL,
            cnn_weight REAL,
            llm_weight REAL,
            side TEXT,
            strength REAL,
            signal_gen INTEGER,
            regime TEXT,
            adx REAL, rsi REAL, macd REAL, mfi REAL, stoch_k REAL,
            atr REAL, vwap_dist REAL, fast_rsi REAL, velocity REAL,
            vol_z REAL, xgb_prob REAL, scanned_at TEXT
        )""")
    c.commit(); c.close()


class TestMCMigration:
    def test_adds_both_columns_on_first_run(self, tmp_path):
        db = tmp_path / "test.db"
        _make_legacy_cnn_scans(db)
        from migrations import mc_telemetry_20260516 as mig
        result = mig.run(str(db))
        assert "xgb_prob_stdev" in result["added"]
        assert "mc_telemetry" in result["added"]
        c = sqlite3.connect(db)
        cols = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        c.close()
        assert "xgb_prob_stdev" in cols
        assert "mc_telemetry" in cols

    def test_idempotent_on_second_run(self, tmp_path):
        db = tmp_path / "test.db"
        _make_legacy_cnn_scans(db)
        from migrations import mc_telemetry_20260516 as mig
        mig.run(str(db))
        result = mig.run(str(db))
        assert result["added"] == []
        assert set(result["already_present"]) == {"xgb_prob_stdev", "mc_telemetry"}


# ── xgb_v4_shadow_20260517 ─────────────────────────────────────────────────


class TestXgbV4ShadowMigration:
    def test_migration_adds_xgb_prob_v4_column(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_shadow_20260517 import run

        db = str(tmp_path / "test.db")
        c = sqlite3.connect(db)
        c.execute(
            "CREATE TABLE cnn_scans ("
            " id INTEGER PRIMARY KEY, product_id TEXT, scanned_at INTEGER"
            ")"
        )
        c.commit()
        c.close()

        result = run(db)
        assert "xgb_prob_v4" in result["added"]
        assert result["already_present"] == []

        # Column now present
        c = sqlite3.connect(db)
        cols = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        assert "xgb_prob_v4" in cols
        c.close()

    def test_migration_idempotent(self, tmp_path):
        import sqlite3
        from migrations.xgb_v4_shadow_20260517 import run

        db = str(tmp_path / "test.db")
        c = sqlite3.connect(db)
        c.execute(
            "CREATE TABLE cnn_scans ("
            " id INTEGER PRIMARY KEY, product_id TEXT, scanned_at INTEGER"
            ")"
        )
        c.commit()
        c.close()

        # First run — adds
        r1 = run(db)
        assert "xgb_prob_v4" in r1["added"]
        # Second run — skips
        r2 = run(db)
        assert r2["added"] == []
        assert "xgb_prob_v4" in r2["already_present"]
