import os
import sqlite3
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
from migrations import diagnostics_indexes_20260808 as mig  # noqa: E402


def _seed(db):
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cnn_scans (product_id TEXT, scanned_at TEXT)")
    con.execute("CREATE TABLE trades (agent TEXT, closed_at TEXT)")
    con.commit()
    con.close()


def test_creates_indexes_and_is_idempotent(tmp_path):
    db = str(tmp_path / "t.db")
    _seed(db)
    first = mig.run(db)
    assert set(first["created"]) == {"idx_cnn_scans_pid_scanned", "idx_trades_agent_closed"}
    second = mig.run(db)  # idempotent — nothing new created
    assert second["created"] == []
    assert set(second["already_present"]) == {
        "idx_cnn_scans_pid_scanned",
        "idx_trades_agent_closed",
    }
    con = sqlite3.connect(db)
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    con.close()
    assert {"idx_cnn_scans_pid_scanned", "idx_trades_agent_closed"} <= idx
