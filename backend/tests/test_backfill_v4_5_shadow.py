"""Tests for v4.5 shadow backfill tool.

NOT run during 2026-05-23 session (8001 was live). Run during next pause:
  cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_backfill_v4_5_shadow.py -v
"""
import os
import sqlite3
import sys

import pytest

_BACKEND = os.path.join(os.path.dirname(__file__), "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)


class _FakeBooster:
    def predict(self, dmat):
        import numpy as np
        return np.array([[0.2, 0.3, 0.5]])


class TestXgbProbV45NowTs:
    """Plumbing: now_ts arg flows through to fetch_tiered."""

    def test_passes_now_ts_to_fetch_tiered(self, monkeypatch):
        from agents import xgb_signal as xs

        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        monkeypatch.setattr(xs, "_feature_names_v45", ["f"] * 210, raising=False)

        captured = {}

        def _fake_fetch(pid, source="live", now_ts=None):
            captured["pid"] = pid
            captured["source"] = source
            captured["now_ts"] = now_ts
            return {"micro": [], "meso": [], "macro": []}

        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered", _fake_fetch,
        )

        def _fake_extract(tiers):
            import numpy as np
            return (np.zeros((1, 210)), [])

        monkeypatch.setattr(
            "tools.xgb_v4_5_features.extract_v4_5", _fake_extract,
        )

        xs.xgb_prob_v4_5(channels=None, pid="BTC-USD", now_ts=1700000000.0)

        assert captured["pid"] == "BTC-USD"
        assert captured["source"] == "live"
        assert captured["now_ts"] == pytest.approx(1700000000.0, abs=1e-6)

    def test_default_now_ts_is_none(self, monkeypatch):
        from agents import xgb_signal as xs

        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        monkeypatch.setattr(xs, "_feature_names_v45", ["f"] * 210, raising=False)

        captured = {}

        def _fake_fetch(pid, source="live", now_ts=None):
            captured["now_ts"] = now_ts
            return {"micro": [], "meso": [], "macro": []}

        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered", _fake_fetch,
        )

        def _fake_extract(tiers):
            import numpy as np
            return (np.zeros((1, 210)), [])

        monkeypatch.setattr(
            "tools.xgb_v4_5_features.extract_v4_5", _fake_extract,
        )

        xs.xgb_prob_v4_5(channels=None, pid="BTC-USD")  # no now_ts

        assert captured["now_ts"] is None


class TestBackfillTool:
    """Backfill tool selects + writes correctly."""

    def _make_db(self, tmp_path):
        # Timestamps anchored to datetime.now() so the days=7 window check
        # stays valid as wall-clock time advances. Original hardcoded
        # "2026-05-23T20:00:00" aged out on 2026-06-07 (15 days later) —
        # all in-window rows fell outside the window and the four
        # TestBackfillTool tests started failing.
        from datetime import datetime, timedelta, timezone

        now = datetime.now(timezone.utc)
        in_window_iso  = (now - timedelta(hours=1)).isoformat()
        out_window_iso = (now - timedelta(days=8)).isoformat()

        db = str(tmp_path / "test_coinbase.db")
        conn = sqlite3.connect(db)
        conn.execute(
            """
            CREATE TABLE cnn_scans (
                id INTEGER PRIMARY KEY,
                product_id TEXT, scanned_at TEXT,
                xgb_prob_v4_5_down REAL,
                xgb_prob_v4_5_neutral REAL,
                xgb_prob_v4_5_up REAL
            )
            """
        )
        # Row 1: NULL, in window
        conn.execute(
            "INSERT INTO cnn_scans VALUES (1, 'BTC-USD', ?, NULL, NULL, NULL)",
            (in_window_iso,),
        )
        # Row 2: already populated, in window — should be skipped
        conn.execute(
            "INSERT INTO cnn_scans VALUES (2, 'ETH-USD', ?, 0.1, 0.2, 0.7)",
            (in_window_iso,),
        )
        # Row 3: NULL, OUTSIDE window (8 days ago) — should be skipped
        conn.execute(
            "INSERT INTO cnn_scans VALUES (3, 'SOL-USD', ?, NULL, NULL, NULL)",
            (out_window_iso,),
        )
        conn.commit()
        conn.close()
        return db

    def test_selects_only_null_in_window(self, tmp_path):
        from tools.backfill_v4_5_shadow import _select_null_rows
        db = self._make_db(tmp_path)
        conn = sqlite3.connect(db)
        rows = _select_null_rows(conn, days=7)
        conn.close()
        assert len(rows) == 1
        assert rows[0][0] == 1
        assert rows[0][1] == "BTC-USD"

    def test_writes_three_probs_atomically(self, tmp_path, monkeypatch):
        from tools.backfill_v4_5_shadow import backfill
        db = self._make_db(tmp_path)

        monkeypatch.setattr(
            "tools.backfill_v4_5_shadow.xgb_signal.xgb_prob_v4_5",
            lambda channels, pid, now_ts: (0.10, 0.20, 0.70),
        )

        total, processed, skipped = backfill(
            db_path=db, days=7, batch_size=10, dry_run=False,
        )
        assert total == 1
        assert processed == 1
        assert skipped == 0

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up "
            "FROM cnn_scans WHERE id=1"
        ).fetchone()
        conn.close()
        assert row == (0.1, 0.2, 0.7)

    def test_handles_inference_failure(self, tmp_path, monkeypatch):
        from tools.backfill_v4_5_shadow import backfill
        db = self._make_db(tmp_path)

        def _raise(channels, pid, now_ts):
            raise RuntimeError("simulated inference failure")

        monkeypatch.setattr(
            "tools.backfill_v4_5_shadow.xgb_signal.xgb_prob_v4_5", _raise,
        )

        total, processed, skipped = backfill(
            db_path=db, days=7, batch_size=10, dry_run=False,
        )
        assert total == 1
        assert processed == 1
        assert skipped == 1

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT xgb_prob_v4_5_down FROM cnn_scans WHERE id=1"
        ).fetchone()
        conn.close()
        assert row[0] is None  # stayed NULL

    def test_neutral_fallback_treated_as_failure(self, tmp_path, monkeypatch):
        """When v4.5 returns the (0.33, 0.34, 0.33) fallback, skip writing."""
        from tools.backfill_v4_5_shadow import backfill
        db = self._make_db(tmp_path)

        monkeypatch.setattr(
            "tools.backfill_v4_5_shadow.xgb_signal.xgb_prob_v4_5",
            lambda channels, pid, now_ts: (0.33, 0.34, 0.33),
        )

        total, processed, skipped = backfill(
            db_path=db, days=7, batch_size=10, dry_run=False,
        )
        assert skipped == 1

        conn = sqlite3.connect(db)
        row = conn.execute(
            "SELECT xgb_prob_v4_5_down FROM cnn_scans WHERE id=1"
        ).fetchone()
        conn.close()
        assert row[0] is None
