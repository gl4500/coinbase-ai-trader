import os
import sqlite3
import sys
from pathlib import Path

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
import pytest  # noqa: E402

from services import diagnostics as d  # noqa: E402

_NOW = 1_700_000_000.0  # fixed epoch for determinism


class TestWindowCutoff:
    def test_all_is_none(self):
        assert d.window_cutoff("all", _NOW) is None

    def test_30d_is_iso_30_days_back(self):
        cut = d.window_cutoff("30d", _NOW)
        assert cut is not None and cut.endswith("+00:00") and "T" in cut

    def test_90d_older_than_30d(self):
        assert d.window_cutoff("90d", _NOW) < d.window_cutoff("30d", _NOW)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            d.window_cutoff("7d", _NOW)


def _seed(tmp_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(tmp_path / "d.db")
    con.executescript(
        """
        CREATE TABLE signal_outcomes (source TEXT, side TEXT, confidence REAL,
            pct_change REAL, outcome TEXT, created_at TEXT);
        CREATE TABLE trades (agent TEXT, product_id TEXT, pnl REAL, pct_pnl REAL,
            hold_secs REAL, trigger_close TEXT, opened_at TEXT, closed_at TEXT);
        CREATE TABLE cnn_scans (product_id TEXT, side TEXT, model_prob REAL,
            regime TEXT, scanned_at TEXT);
        """
    )
    return con


class TestSignalEdge:
    def test_precision_and_calibration(self, tmp_path: Path) -> None:
        con = _seed(tmp_path)
        rows = [
            ("CNN", "BUY", 0.90, 0.02, "WIN", "2026-08-08T00:00:00+00:00"),
            ("CNN", "BUY", 0.92, -0.01, "LOSS", "2026-08-08T00:00:00+00:00"),
            ("CNN", "BUY", 0.20, -0.03, "LOSS", "2026-08-08T00:00:00+00:00"),
            ("CNN", "BUY", 0.20, 0.00, "NEUTRAL", "2026-08-08T00:00:00+00:00"),
            ("TECH", "BUY", 0.90, 0.05, "WIN", "2026-08-08T00:00:00+00:00"),  # excluded
        ]
        con.executemany(
            "INSERT INTO signal_outcomes VALUES (?,?,?,?,?,?)", rows
        )
        con.commit()
        out = d.signal_edge(con, cutoff=None)
        assert out["n"] == 4 and out["wins"] == 1 and out["losses"] == 2
        assert out["precision"] == pytest.approx(0.25)
        b9 = next(b for b in out["calibration"] if b["bucket"] == 0.9)
        assert b9["win_rate"] == pytest.approx(0.5)  # 1 win / (1 win + 1 loss)
