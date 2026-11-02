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

        def _fake_fetch(pid, source="live", now_ts=None, closed_only=False):
            captured["pid"] = pid
            captured["source"] = source
            captured["now_ts"] = now_ts
            captured["closed_only"] = closed_only
            return {"micro": [], "meso": [], "macro": []}

        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            _fake_fetch,
        )

        def _fake_extract(tiers):
            import numpy as np

            return (np.zeros((1, 210)), [])

        monkeypatch.setattr(
            "tools.xgb_v4_5_features.extract_v4_5",
            _fake_extract,
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

        def _fake_fetch(pid, source="live", now_ts=None, closed_only=False):
            captured["now_ts"] = now_ts
            return {"micro": [], "meso": [], "macro": []}

        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            _fake_fetch,
        )

        def _fake_extract(tiers):
            import numpy as np

            return (np.zeros((1, 210)), [])

        monkeypatch.setattr(
            "tools.xgb_v4_5_features.extract_v4_5",
            _fake_extract,
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
        in_window_iso = (now - timedelta(hours=1)).isoformat()
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
            db_path=db,
            days=7,
            batch_size=10,
            dry_run=False,
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
            "tools.backfill_v4_5_shadow.xgb_signal.xgb_prob_v4_5",
            _raise,
        )

        total, processed, skipped = backfill(
            db_path=db,
            days=7,
            batch_size=10,
            dry_run=False,
        )
        assert total == 1
        assert processed == 1
        assert skipped == 1

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT xgb_prob_v4_5_down FROM cnn_scans WHERE id=1").fetchone()
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
            db_path=db,
            days=7,
            batch_size=10,
            dry_run=False,
        )
        assert skipped == 1

        conn = sqlite3.connect(db)
        row = conn.execute("SELECT xgb_prob_v4_5_down FROM cnn_scans WHERE id=1").fetchone()
        conn.close()
        assert row[0] is None


class TestReplayAsksForClosedBarsOnly:
    """A historical re-score must not consume a candle that had not closed.

    `fetch_tiered`'s default filter keeps bars whose START precedes `now_ts`, which admits the
    bar that had begun but not finished. Replaying from a completed parquet, that bar carries
    its final high/low/close/volume, so up to an hour of future information enters the
    features at a 15-minute scan cadence -- and every metric computed that way is invalid.

    The opt-in is tied to `now_ts` rather than exposed as a separate switch, which makes the
    live path byte-identical BY CONSTRUCTION rather than by discipline: live callers
    (`cnn_agent` via `xgb_prob_shadow_v4_5`) pass no `now_ts`, so they cannot accidentally
    acquire the replay semantics, and the only caller that does pass one is
    `tools/backfill_v4_5_shadow.py`, which is a replay by definition.
    """

    @staticmethod
    def _spy(monkeypatch):
        from agents import xgb_signal as xs

        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        # unique names: duplicates make DMatrix raise, so the call would only be observable
        # through the function's own except branch -- a weaker test than the real path.
        monkeypatch.setattr(xs, "_feature_names_v45", [f"f{i}" for i in range(210)], raising=False)

        seen = {}

        def _fake_fetch(pid, source="live", now_ts=None, closed_only=False):
            seen["now_ts"] = now_ts
            seen["closed_only"] = closed_only
            return {"micro": [], "meso": [], "macro": []}

        monkeypatch.setattr("services.tiered_history.fetch_tiered", _fake_fetch)

        def _fake_extract(tiers):
            import numpy as np

            return (np.zeros((1, 210)), [])

        monkeypatch.setattr("tools.xgb_v4_5_features.extract_v4_5", _fake_extract)
        return xs, seen

    def test_supplying_now_ts_requests_closed_bars_only(self, monkeypatch):
        xs, seen = self._spy(monkeypatch)
        xs.xgb_prob_v4_5(channels=None, pid="BTC-USD", now_ts=1700000000.0)
        assert seen, "fetch_tiered was never called -- assertion would be vacuous"
        assert seen["now_ts"] == 1700000000.0
        assert seen["closed_only"] is True

    def test_the_live_path_is_unchanged(self, monkeypatch):
        """Non-vacuity for the safety claim. Without an as-of instant there is nothing to be
        "as of", so the live filter must stay exactly as it was."""
        xs, seen = self._spy(monkeypatch)
        xs.xgb_prob_v4_5(channels=None, pid="BTC-USD")
        assert seen, "fetch_tiered was never called -- assertion would be vacuous"
        assert seen["now_ts"] is None
        assert seen["closed_only"] is False


class TestIsolatedFailureStaysObservable:
    """Invariants 16/17 require a v4.5 failure to be isolated -- it must never reach the
    driver, and a neutral 3-tuple is the correct return. But isolation must not become
    SILENCE: the same broad `except` that protects the scan loop also catches programming
    errors such as a signature mismatch, and a mismatch that degrades every call to neutral
    while logging nothing would be invisible in production.

    This pins the observability half of the invariant so a later refactor cannot downgrade
    `logger.exception` to a debug line or a bare `pass`. I hit exactly this shape today: an
    unexpected kwarg raised TypeError inside the try, and the neutral fallback made the call
    look successful.
    """

    def test_a_signature_mismatch_returns_neutral_AND_logs_at_error(self, monkeypatch, caplog):
        import logging

        from agents import xgb_signal as xs

        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        monkeypatch.setattr(xs, "_feature_names_v45", [f"f{i}" for i in range(210)], raising=False)

        def _stale_signature(pid, source="live"):
            raise TypeError("fetch_tiered() got an unexpected keyword argument 'closed_only'")

        monkeypatch.setattr("services.tiered_history.fetch_tiered", _stale_signature)

        with caplog.at_level(logging.ERROR):
            out = xs.xgb_prob_v4_5(channels=None, pid="BTC-USD", now_ts=1700000000.0)

        assert out == (0.33, 0.34, 0.33), "isolation broken -- must degrade to neutral"
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "the failure was SILENT -- isolation must not hide a defect"
        assert any("v4_5" in r.getMessage() or "v4.5" in r.getMessage() for r in errors)
        assert any(r.exc_info for r in errors), "no traceback captured; cause is undiagnosable"

    def test_the_success_path_logs_no_error(self, monkeypatch, caplog):
        """Non-vacuity for the test above: the assertion must be able to fail."""
        import logging

        from agents import xgb_signal as xs

        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        monkeypatch.setattr(xs, "_feature_names_v45", [f"f{i}" for i in range(210)], raising=False)
        monkeypatch.setattr(
            "services.tiered_history.fetch_tiered",
            lambda pid, source="live", now_ts=None, closed_only=False: {
                "micro": [],
                "meso": [],
                "macro": [],
            },
        )

        def _fake_extract(tiers):
            import numpy as np

            return (np.zeros((1, 210)), [])

        monkeypatch.setattr("tools.xgb_v4_5_features.extract_v4_5", _fake_extract)

        with caplog.at_level(logging.ERROR):
            out = xs.xgb_prob_v4_5(channels=None, pid="BTC-USD", now_ts=1700000000.0)

        assert out != (0.33, 0.34, 0.33), "stub should have produced a real 3-tuple"
        assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
