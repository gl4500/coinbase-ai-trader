# Backfill v4.5 Shadow Telemetry Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax. This plan is executed by the same session as the spec was written; tests are written but NOT run (8001 live trading per `feedback_no_pytest_during_trading.md`). Operator runs pytest + commits during next pause window.

**Goal:** Backfill `xgb_prob_v4_5_*` columns on last 7 days of `cnn_scans` rows.

**Architecture:** New `now_ts` param on `xgb_signal.xgb_prob_v4_5` plumbed through to `fetch_tiered`. New CLI tool `tools/backfill_v4_5_shadow.py` walks NULL rows in batches, calls inference with historical timestamp, batch-UPDATEs.

**Tech Stack:** Python 3.11, `xgboost`, `aiosqlite` not needed (tool is sync via stdlib `sqlite3`), `argparse`, `datetime`.

**Spec:** `docs/superpowers/specs/2026-05-23-backfill-v4-5-shadow-design.md`

**Pre-conditions:**
- 8001 backend running (live or paused — both fine). WAL mode allows concurrent reads/writes.
- v4.5 model artifacts present at `backend/xgb_model_v4_5.json` + `backend/xgb_features_v4_5.json`. Verify with `ls backend/xgb_model_v4_5.json`.
- Branch state: any branch is fine. No commits during this session (8001 live).

---

## Task 1: Plumb `now_ts` into `xgb_prob_v4_5`

**Files:**
- Modify: `backend/agents/xgb_signal.py:286-329` (the existing `xgb_prob_v4_5` function)

- [ ] **Step 1: Read the existing function**

```bash
grep -n "def xgb_prob_v4_5" backend/agents/xgb_signal.py
```

Confirm signature at line 286 looks like:
```python
def xgb_prob_v4_5(
    channels, pid: Optional[str] = None,
) -> Tuple[float, float, float]:
```

- [ ] **Step 2: Update signature + fetch_tiered call**

Edit `backend/agents/xgb_signal.py` to add `now_ts` param + pass through:

```python
def xgb_prob_v4_5(
    channels, pid: Optional[str] = None,
    now_ts: Optional[float] = None,
) -> Tuple[float, float, float]:
    """v4.5 3-class probabilities (p_down, p_neutral, p_up).

    When now_ts is provided, fetch_tiered uses it to look up the tier slices
    as they would have been at that historical timestamp (drops candles
    with start >= now_ts). Default None = live (current behavior).

    Each clipped to [0.01, 0.99] then renormalized to sum to 1.0. Returns
    neutral fallback (0.33, 0.34, 0.33) if artifacts missing, pid is None,
    or any error during inference.
    """
    _NEUTRAL_3 = (0.33, 0.34, 0.33)
    if not _try_load_v4_5():
        return _NEUTRAL_3
    if pid is None:
        logger.warning(
            "xgb_signal: v4.5 requires pid, got None — returning neutral 3-tuple",
        )
        return _NEUTRAL_3
    try:
        import xgboost as xgb
        from services.tiered_history import fetch_tiered
        from tools.xgb_v4_5_features import extract_v4_5

        tiers = fetch_tiered(pid, source="live", now_ts=now_ts)
        features, _ = extract_v4_5(tiers)
        ...
```

Only two lines change in practice: the signature gains `now_ts`, and the `fetch_tiered` call passes `now_ts=now_ts`. Body otherwise identical.

- [ ] **Step 3: Smoke import check**

```bash
cd backend && ../.venv/Scripts/python.exe -c "from agents.xgb_signal import xgb_prob_v4_5; import inspect; sig = inspect.signature(xgb_prob_v4_5); print('params:', list(sig.parameters.keys()))"
```

Expected: `params: ['channels', 'pid', 'now_ts']`

---

## Task 2: Write the backfill tool

**Files:**
- Create: `backend/tools/backfill_v4_5_shadow.py`

- [ ] **Step 1: Create the tool**

Create `backend/tools/backfill_v4_5_shadow.py` with this exact content:

```python
"""Backfill v4.5 shadow telemetry onto historical cnn_scans rows.

For each NULL row in scope, fetch tiers AS THEY WOULD HAVE BEEN at
scanned_at, run v4.5 inference, UPDATE the row's three v4.5 columns.

Safe to run while live backend (port 8001) is trading: SQLite WAL +
100-row batched transactions hold writer lock <50ms per batch.

USAGE
  cd backend
  ../.venv/Scripts/python.exe -m tools.backfill_v4_5_shadow [--days N]
       [--batch-size 100] [--db backend/coinbase.db] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

# Make backend/ importable when run via `python -m tools.backfill_v4_5_shadow`.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from agents import xgb_signal  # noqa: E402


logger = logging.getLogger(__name__)


_NEUTRAL_FALLBACK = (0.33, 0.34, 0.33)


def _parse_iso_to_unix(s: str) -> float:
    """Parse '2026-05-23T21:18:19.313951+00:00' to unix seconds."""
    # SQLite stored values may end with '+00:00' or 'Z'. Python 3.11+ fromisoformat
    # accepts both.
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _select_null_rows(conn: sqlite3.Connection, days: int) -> List[Tuple[int, str, str]]:
    """Select (id, product_id, scanned_at) for rows missing v4.5 in the window."""
    cutoff_unix = time.time() - days * 86400.0
    cutoff_iso  = datetime.fromtimestamp(cutoff_unix, tz=timezone.utc).isoformat()
    cur = conn.execute(
        """
        SELECT id, product_id, scanned_at
        FROM cnn_scans
        WHERE xgb_prob_v4_5_up IS NULL
          AND scanned_at > ?
        ORDER BY id
        """,
        (cutoff_iso,),
    )
    return cur.fetchall()


def _infer_one(pid: str, scanned_at: str) -> Tuple[float, float, float]:
    """Run v4.5 inference at historical timestamp. Raises on any error."""
    now_ts = _parse_iso_to_unix(scanned_at)
    probs = xgb_signal.xgb_prob_v4_5(channels=None, pid=pid, now_ts=now_ts)
    if probs == _NEUTRAL_FALLBACK:
        # Inference returned the neutral fallback — treat as failure so the
        # row stays NULL and operator can diagnose. See spec Q1.
        raise RuntimeError("xgb_prob_v4_5 returned neutral fallback")
    return probs


def backfill(
    db_path: str, days: int, batch_size: int, dry_run: bool,
) -> Tuple[int, int, int]:
    """Returns (total_scope, processed, skipped)."""
    logger.info("[%s] connecting to %s", _hhmm(), db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    rows = _select_null_rows(conn, days)
    total = len(rows)
    logger.info("[%s] scope: %d NULL rows in last %d days", _hhmm(), total, days)

    if total == 0:
        logger.info("[%s] nothing to do — all rows in scope already populated", _hhmm())
        conn.close()
        return total, 0, 0

    if dry_run:
        logger.info("[%s] DRY RUN — first 5 sample rows:", _hhmm())
        for r in rows[:5]:
            logger.info("  id=%d pid=%s scanned_at=%s", r[0], r[1], r[2])
        conn.close()
        return total, 0, 0

    started_at_iso = datetime.now(timezone.utc).isoformat()
    logger.info("[%s] backfill started_at=%s  (use for rollback scope)",
                _hhmm(), started_at_iso)

    processed = 0
    skipped   = 0
    t0        = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        updates: List[Tuple[float, float, float, int]] = []

        for row_id, pid, scanned_at in batch:
            try:
                p_down, p_neutral, p_up = _infer_one(pid, scanned_at)
                updates.append((
                    round(p_down, 4), round(p_neutral, 4), round(p_up, 4), row_id,
                ))
            except Exception as e:
                logger.debug("skip row %d (%s @ %s): %s", row_id, pid, scanned_at, e)
                skipped += 1

        if updates:
            conn.executemany(
                """
                UPDATE cnn_scans
                SET xgb_prob_v4_5_down    = ?,
                    xgb_prob_v4_5_neutral = ?,
                    xgb_prob_v4_5_up      = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()

        processed += len(batch)
        elapsed = time.time() - t0
        rate    = processed / elapsed if elapsed > 0 else 0.0
        remain  = total - processed
        eta_sec = remain / rate if rate > 0 else 0
        eta_mm  = int(eta_sec // 60)
        eta_ss  = int(eta_sec % 60)
        pct     = 100.0 * processed / total
        logger.info(
            "[%s] progress: %d/%d (%.1f%%) rate=%.1f/s eta=%d:%02d skipped=%d",
            _hhmm(), processed, total, pct, rate, eta_mm, eta_ss, skipped,
        )

    conn.close()
    succeeded = processed - skipped
    logger.info(
        "[%s] done. processed=%d succeeded=%d skipped=%d total_elapsed=%.1fs",
        _hhmm(), processed, succeeded, skipped, time.time() - t0,
    )
    return total, processed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill v4.5 shadow probs.")
    parser.add_argument("--days", type=int, default=7, help="Look back N days (default 7)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Rows per UPDATE transaction (default 100)")
    parser.add_argument(
        "--db",
        default=os.path.join(_BACKEND, "coinbase.db"),
        help="SQLite path (default backend/coinbase.db)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scope + sample, no UPDATEs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backfill(
        db_path=args.db,
        days=args.days,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke check the tool imports**

```bash
cd backend && ../.venv/Scripts/python.exe -c "from tools.backfill_v4_5_shadow import backfill, _parse_iso_to_unix; print('OK')"
```

Expected: `OK`. If ImportError, check the missing dependency.

---

## Task 3: Write tests (DO NOT RUN — 8001 live)

**Files:**
- Create: `backend/tests/test_backfill_v4_5_shadow.py`

These tests document the contract; operator runs them during the next 8001 pause window.

- [ ] **Step 1: Create the test file**

Create `backend/tests/test_backfill_v4_5_shadow.py`:

```python
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


class TestXgbProbV45NowTs:
    """Plumbing: now_ts arg flows through to fetch_tiered."""

    def test_passes_now_ts_to_fetch_tiered(self, monkeypatch):
        from agents import xgb_signal as xs

        # Make _try_load_v4_5 succeed without real model file
        monkeypatch.setattr(xs, "_try_load_v4_5", lambda: True)
        # Stash any pre-existing _booster_v45 reference (set during load)
        monkeypatch.setattr(xs, "_booster_v45", _FakeBooster(), raising=False)
        monkeypatch.setattr(xs, "_feature_names_v45", ["f"] * 210, raising=False)

        captured = {}
        def _fake_fetch(pid, source="live", now_ts=None):
            captured["pid"] = pid
            captured["source"] = source
            captured["now_ts"] = now_ts
            return {"micro": [], "meso": [], "macro": []}
        monkeypatch.setattr("services.tiered_history.fetch_tiered", _fake_fetch)

        def _fake_extract(tiers):
            import numpy as np
            return (np.zeros((1, 210)), [])
        monkeypatch.setattr("tools.xgb_v4_5_features.extract_v4_5", _fake_extract)

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
        monkeypatch.setattr("services.tiered_history.fetch_tiered", _fake_fetch)

        def _fake_extract(tiers):
            import numpy as np
            return (np.zeros((1, 210)), [])
        monkeypatch.setattr("tools.xgb_v4_5_features.extract_v4_5", _fake_extract)

        xs.xgb_prob_v4_5(channels=None, pid="BTC-USD")  # no now_ts

        assert captured["now_ts"] is None


class _FakeBooster:
    def predict(self, dmat):
        import numpy as np
        return np.array([[0.2, 0.3, 0.5]])


class TestBackfillTool:
    """Backfill tool selects + writes correctly."""

    def _make_db(self, tmp_path):
        db = str(tmp_path / "test_coinbase.db")
        conn = sqlite3.connect(db)
        conn.execute("""
            CREATE TABLE cnn_scans (
                id INTEGER PRIMARY KEY,
                product_id TEXT, scanned_at TEXT,
                xgb_prob_v4_5_down REAL,
                xgb_prob_v4_5_neutral REAL,
                xgb_prob_v4_5_up REAL
            )
        """)
        # Row 1: NULL, in window
        conn.execute(
            "INSERT INTO cnn_scans VALUES (1, 'BTC-USD', ?, NULL, NULL, NULL)",
            ("2026-05-23T20:00:00+00:00",),
        )
        # Row 2: already populated, in window — should be skipped
        conn.execute(
            "INSERT INTO cnn_scans VALUES (2, 'ETH-USD', ?, 0.1, 0.2, 0.7)",
            ("2026-05-23T20:00:00+00:00",),
        )
        # Row 3: NULL, OUTSIDE window (8 days ago) — should be skipped
        conn.execute(
            "INSERT INTO cnn_scans VALUES (3, 'SOL-USD', ?, NULL, NULL, NULL)",
            ("2026-05-15T20:00:00+00:00",),
        )
        conn.commit()
        conn.close()
        return db

    def test_selects_only_null_in_window(self, tmp_path, monkeypatch):
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
```

---

## Task 4: Dry-run + execute + verify

- [ ] **Step 1: Dry-run to confirm scope**

```bash
cd backend && ../.venv/Scripts/python.exe -m tools.backfill_v4_5_shadow --days 7 --dry-run
```

Expected: scope count (likely 5K-25K rows) + 5 sample rows printed. No DB mutations.

- [ ] **Step 2: Execute**

```bash
cd backend && ../.venv/Scripts/python.exe -m tools.backfill_v4_5_shadow --days 7
```

Expected: streaming progress lines every batch, ETA, final summary. Estimated runtime: ~15-45 min for ~15K rows depending on inference latency.

- [ ] **Step 3: Verify**

```bash
cd backend && ../.venv/Scripts/python.exe -c "
import sqlite3
c = sqlite3.connect('coinbase.db')
# Fill rate
total, with_v45 = c.execute(
    \"SELECT COUNT(*), COUNT(xgb_prob_v4_5_up) FROM cnn_scans \"
    \"WHERE scanned_at > datetime('now', '-7 days')\"
).fetchone()
print(f'Last 7 days: {total} total, {with_v45} with v4.5 ({100*with_v45/total:.1f}%)')

# Spot-check: rows where v3 and v4.5 disagree on direction
disagree = c.execute(
    'SELECT COUNT(*) FROM cnn_scans '
    'WHERE xgb_prob > 0.55 AND xgb_prob_v4_5_down > 0.40 '
    \"  AND scanned_at > datetime('now', '-7 days')\"
).fetchone()[0]
print(f'v3 bullish ({chr(62)}0.55) but v4.5 bearish ({chr(62)}0.40 p_down): {disagree} rows')
"
```

Expected: fill rate ≥ 95%, some non-zero count for the disagreement query.

---

## Operator handoff after this session

1. Tests not yet run. During next 8001 pause, run:
   ```bash
   cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_backfill_v4_5_shadow.py -v
   ```
2. If GREEN, commit:
   - `backend/agents/xgb_signal.py` (now_ts param added)
   - `backend/tools/backfill_v4_5_shadow.py` (new file)
   - `backend/tests/test_backfill_v4_5_shadow.py` (new file)
   - `docs/superpowers/specs/2026-05-23-backfill-v4-5-shadow-design.md`
   - `docs/superpowers/plans/2026-05-23-backfill-v4-5-shadow.md`

Suggested commit message: `feat(backfill): retroactive v4.5 shadow telemetry on cnn_scans`.
