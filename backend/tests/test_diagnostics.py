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


class TestExitAttribution:
    def test_by_trigger_and_share(self, tmp_path: Path):
        con = _seed(tmp_path)
        rows = [
            ("CNN", "SOL-USD", 5.0, 0.01, 3600, "SCAN", "2026-08-01T00:00:00+00:00",
             "2026-08-08T00:00:00+00:00"),
            ("CNN", "SOL-USD", -3.0, -0.02, 7200, "STOP_LOSS",
             "2026-08-01T00:00:00+00:00", "2026-08-08T00:00:00+00:00"),
            ("CNN", "ETH-USD", 2.0, 0.005, 100, "SCAN", "2026-08-01T00:00:00+00:00",
             "2026-08-08T00:00:00+00:00"),
        ]
        con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)", rows)
        con.commit()
        out = d.exit_attribution(con, cutoff=None)
        scan = next(t for t in out["by_trigger"] if t["trigger"] == "SCAN")
        assert scan["n"] == 2 and scan["sum_pnl"] == pytest.approx(7.0)
        assert scan["win_rate"] == pytest.approx(1.0)
        assert out["scan_sell_share"] == pytest.approx(2 / 3)


class TestRegimeAndAsset:
    def test_asset_and_nearest_scan_regime(self, tmp_path):
        con = _seed(tmp_path)
        con.executemany(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
            [
                ("CNN", "SOL-USD", 5.0, 0.01, 3600, "SCAN",
                 "2026-08-05T10:00:00+00:00", "2026-08-05T14:00:00+00:00"),
                ("CNN", "SOL-USD", -2.0, -0.01, 3600, "STOP_LOSS",
                 "2026-08-06T10:00:00+00:00", "2026-08-06T14:00:00+00:00"),
            ],
        )
        con.executemany(
            "INSERT INTO cnn_scans (product_id, side, model_prob, regime, scanned_at) "
            "VALUES (?,?,?,?,?)",
            [
                ("SOL-USD", "BUY", 0.6, "TRENDING", "2026-08-05T09:00:00+00:00"),
                ("SOL-USD", "HOLD", 0.5, "RANGING", "2026-08-06T09:00:00+00:00"),
            ],
        )
        con.commit()
        out = d.regime_and_asset(con, cutoff=None)
        sol = next(a for a in out["by_asset"] if a["product_id"] == "SOL-USD")
        assert sol["n"] == 2 and sol["sum_pnl"] == pytest.approx(3.0)
        regimes = {r["regime"]: r for r in out["by_regime"]}
        assert regimes["TRENDING"]["sum_pnl"] == pytest.approx(5.0)
        assert regimes["RANGING"]["sum_pnl"] == pytest.approx(-2.0)


class TestSignalFunnel:
    def test_counts(self, tmp_path):
        con = _seed(tmp_path)
        con.executemany(
            "INSERT INTO cnn_scans (product_id, side, model_prob, regime, scanned_at) "
            "VALUES (?,?,?,?,?)",
            [("SOL-USD", "BUY", 0.6, "TRENDING", "2026-08-08T00:00:00+00:00"),
             ("SOL-USD", "HOLD", 0.5, "RANGING", "2026-08-08T00:00:00+00:00")],
        )
        con.execute(
            "INSERT INTO trades VALUES ('CNN','SOL-USD',1,0.01,10,'SCAN',"
            "'2026-08-08T00:00:00+00:00','2026-08-08T01:00:00+00:00')"
        )
        con.execute(
            "INSERT INTO signal_outcomes VALUES "
            "('CNN','BUY',0.6,0.01,'WIN','2026-08-08T00:00:00+00:00')"
        )
        con.commit()
        out = d.signal_funnel(con, cutoff=None)
        assert out == {"scans": 2, "buy_signals": 1, "executed": 1, "matured": 1}
