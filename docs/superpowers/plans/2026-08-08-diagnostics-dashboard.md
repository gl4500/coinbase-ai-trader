# v3 Diagnostics Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only **Diagnostics** tab that explains *why* v3 loses (signal edge + calibration, exit attribution, regime/per-asset, signal funnel), complementing the PnL-only PerformanceDashboard.

**Architecture:** A new read-only `backend/services/diagnostics.py` (pure functions, own `mode=ro` sqlite connection, 60s TTL cache) feeds a thin `GET /api/diagnostics?window=` endpoint in `main.py`; a new `DiagnosticsDashboard.tsx` React tab renders four sections with hand-rolled SVG (no new deps). One additive index migration makes the regime nearest-scan join fast.

**Tech Stack:** Python 3.11, FastAPI, sqlite3 (stdlib, read-only), React + Vite + TypeScript, Tailwind, inline SVG.

## Global Constraints

- Read-only: `diagnostics.py` opens `sqlite3.connect("file:<db>?mode=ro", uri=True)` — never writes, never imports `cnn_agent`/trading loop (loose-coupling rule).
- All queries filter to the live agent: `agent='CNN'` (trades) / `source='CNN'` (signal_outcomes). TECH is retired.
- Ruff 0.9.0 clean (CI-pinned): `ruff check backend/` + `ruff format --check backend/` must pass. Type hints, pure single-responsibility functions.
- Tests mock-only: seed a temp SQLite (`tmp_path`), no live DB / network / real file I/O.
- DB timestamps are ISO-8601 strings (e.g. `2026-08-08T15:30:30.366569+00:00`); string comparison is valid for range filters.
- Window values: `30d`, `90d`, `all` (default `30d`).

---

### Task 1: Index migration for the regime nearest-scan join

**Files:**
- Create: `backend/migrations/diagnostics_indexes_20260808.py`
- Test: `backend/tests/test_diagnostics_migration.py`

**Interfaces:**
- Produces: idempotent `run(conn)` that creates `idx_cnn_scans_pid_scanned` on `cnn_scans(product_id, scanned_at)` and `idx_trades_agent_closed` on `trades(agent, closed_at)`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_diagnostics_migration.py
import os, sqlite3, sys
BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
from migrations import diagnostics_indexes_20260808 as mig  # noqa: E402


def test_creates_indexes_idempotently(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE cnn_scans (product_id TEXT, scanned_at TEXT)")
    con.execute("CREATE TABLE trades (agent TEXT, closed_at TEXT)")
    con.commit()
    mig.run(con)
    mig.run(con)  # idempotent
    idx = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    assert "idx_cnn_scans_pid_scanned" in idx
    assert "idx_trades_agent_closed" in idx
    con.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics_migration.py -v`
Expected: FAIL (module `diagnostics_indexes_20260808` not found)

- [ ] **Step 3: Write minimal implementation**

```python
# backend/migrations/diagnostics_indexes_20260808.py
"""Additive indexes for the diagnostics dashboard. Idempotent."""


def run(conn):
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cnn_scans_pid_scanned "
        "ON cnn_scans(product_id, scanned_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_agent_closed ON trades(agent, closed_at)"
    )
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics_migration.py -v`
Expected: PASS

- [ ] **Step 5: Wire into the migration runner**

Find the migration registry in `backend/database.py` (the list the startup runner iterates — search `migrations` / `run(`). Append an import + call to `diagnostics_indexes_20260808.run(conn)` following the existing pattern for `mc_telemetry_20260516` etc. (match the exact registration style already used).

- [ ] **Step 6: Commit**

```bash
git add backend/migrations/diagnostics_indexes_20260808.py backend/tests/test_diagnostics_migration.py backend/database.py
git commit -m "feat: additive indexes for diagnostics regime/exit queries"
```

---

### Task 2: Window-cutoff helper

**Files:**
- Create: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Produces: `window_cutoff(window: str, now: float) -> Optional[str]` — ISO cutoff for `"30d"`/`"90d"`, `None` for `"all"` (meaning no lower bound). Raises `ValueError` on unknown window.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_diagnostics.py
import os, sys
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestWindowCutoff -v`
Expected: FAIL (module/function missing)

- [ ] **Step 3: Write minimal implementation**

```python
# backend/services/diagnostics.py
"""Read-only diagnostics aggregations for the Diagnostics dashboard tab.

Never writes; opens its own mode=ro connection; no coupling to the trading loop.
"""
from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

_WINDOW_DAYS = {"30d": 30, "90d": 90}


def window_cutoff(window: str, now: float) -> Optional[str]:
    if window == "all":
        return None
    if window not in _WINDOW_DAYS:
        raise ValueError(f"unknown window: {window!r}")
    dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(days=_WINDOW_DAYS[window])
    return dt.isoformat()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestWindowCutoff -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics window_cutoff helper"
```

---

### Task 3: signal_edge (precision, E[return], calibration)

**Files:**
- Modify: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Consumes: `window_cutoff`.
- Produces: `signal_edge(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict` returning
  `{"n": int, "wins": int, "losses": int, "neutrals": int, "precision": float,
   "e_return": float, "calibration": [{"bucket": float, "n": int, "win_rate": float, "avg_ret": float}, ...]}`.
  precision = wins/n; calibration win_rate excludes NEUTRAL (wins/(wins+losses) per decile).

- [ ] **Step 1: Write the failing test**

```python
# add to backend/tests/test_diagnostics.py
import sqlite3


def _seed(tmp_path):
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
    def test_precision_and_calibration(self, tmp_path):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestSignalEdge -v`
Expected: FAIL (`signal_edge` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# add to backend/services/diagnostics.py
def _where_since(col: str, cutoff: Optional[str]) -> tuple[str, list]:
    return (f" AND {col} >= ?", [cutoff]) if cutoff else ("", [])


def signal_edge(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict:
    clause, params = _where_since("created_at", cutoff)
    base = (
        "FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        "AND outcome IN ('WIN','LOSS','NEUTRAL')" + clause
    )
    n, wins, losses, neutrals, e_return = conn.execute(
        "SELECT COUNT(*), "
        "SUM(outcome='WIN'), SUM(outcome='LOSS'), SUM(outcome='NEUTRAL'), "
        "AVG(pct_change) " + base,
        params,
    ).fetchone()
    n = n or 0
    calibration = []
    for r in conn.execute(
        "SELECT CAST(confidence*10 AS INT) AS b, COUNT(*), "
        "SUM(outcome='WIN'), SUM(outcome IN ('WIN','LOSS')), AVG(pct_change) "
        + base + " GROUP BY b ORDER BY b",
        params,
    ):
        bucket, cnt, w, wl, avg_ret = r
        calibration.append({
            "bucket": round(bucket / 10.0, 1),
            "n": cnt,
            "win_rate": (w / wl) if wl else 0.0,
            "avg_ret": avg_ret or 0.0,
        })
    return {
        "n": n, "wins": wins or 0, "losses": losses or 0, "neutrals": neutrals or 0,
        "precision": (wins / n) if n else 0.0,
        "e_return": e_return or 0.0,
        "calibration": calibration,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestSignalEdge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics signal_edge (precision + calibration)"
```

---

### Task 4: exit_attribution (per-trigger, hold-time, SCAN-SELL share)

**Files:**
- Modify: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Produces: `exit_attribution(conn, cutoff) -> Dict` = `{"by_trigger": [{"trigger": str, "n": int,
  "sum_pnl": float, "avg_pct": float, "win_rate": float}, ...], "scan_sell_share": float}`.
  Filters `agent='CNN' AND closed_at IS NOT NULL` (+ cutoff on `closed_at`). scan_sell_share = SCAN closes / all closes.

- [ ] **Step 1: Write the failing test**

```python
class TestExitAttribution:
    def test_by_trigger_and_share(self, tmp_path):
        con = _seed(tmp_path)
        rows = [
            ("CNN", "SOL-USD", 5.0, 0.01, 3600, "SCAN", "2026-08-01T00:00:00+00:00",
             "2026-08-08T00:00:00+00:00"),
            ("CNN", "SOL-USD", -3.0, -0.02, 7200, "STOP_LOSS", "2026-08-01T00:00:00+00:00",
             "2026-08-08T00:00:00+00:00"),
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestExitAttribution -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def exit_attribution(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict:
    clause, params = _where_since("closed_at", cutoff)
    base = "FROM trades WHERE agent='CNN' AND closed_at IS NOT NULL" + clause
    by_trigger = []
    total = 0
    scan = 0
    for r in conn.execute(
        "SELECT trigger_close, COUNT(*), SUM(pnl), AVG(pct_pnl), "
        "SUM(pnl>0)*1.0/COUNT(*) " + base + " GROUP BY trigger_close ORDER BY SUM(pnl)",
        params,
    ):
        trig, cnt, sum_pnl, avg_pct, wr = r
        by_trigger.append({
            "trigger": trig, "n": cnt, "sum_pnl": sum_pnl or 0.0,
            "avg_pct": avg_pct or 0.0, "win_rate": wr or 0.0,
        })
        total += cnt
        if trig == "SCAN":
            scan += cnt
    return {"by_trigger": by_trigger, "scan_sell_share": (scan / total) if total else 0.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestExitAttribution -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics exit_attribution"
```

---

### Task 5: regime_and_asset (per-asset PnL + nearest-scan regime)

**Files:**
- Modify: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Produces: `regime_and_asset(conn, cutoff) -> Dict` = `{"by_asset": [{"product_id": str, "n": int,
  "sum_pnl": float, "win_rate": float}, ...], "by_regime": [{"regime": str, "n": int, "sum_pnl": float}, ...]}`.
  Regime per trade = latest `cnn_scans.regime` for that `product_id` with `scanned_at <= opened_at` (correlated subquery; uses `idx_cnn_scans_pid_scanned`). Trades with no prior scan → regime `"UNKNOWN"`.

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestRegimeAndAsset -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def regime_and_asset(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict:
    clause, params = _where_since("closed_at", cutoff)
    base = "FROM trades t WHERE t.agent='CNN' AND t.closed_at IS NOT NULL" + clause
    by_asset = [
        {"product_id": r[0], "n": r[1], "sum_pnl": r[2] or 0.0, "win_rate": r[3] or 0.0}
        for r in conn.execute(
            "SELECT product_id, COUNT(*), SUM(pnl), SUM(pnl>0)*1.0/COUNT(*) "
            + base + " GROUP BY product_id ORDER BY SUM(pnl)",
            params,
        )
    ]
    regime_agg: Dict[str, list] = {}
    for pnl, regime in conn.execute(
        "SELECT t.pnl, COALESCE((SELECT s.regime FROM cnn_scans s "
        "WHERE s.product_id=t.product_id AND s.scanned_at<=t.opened_at "
        "ORDER BY s.scanned_at DESC LIMIT 1), 'UNKNOWN') " + base,
        params,
    ):
        agg = regime_agg.setdefault(regime, [0, 0.0])
        agg[0] += 1
        agg[1] += pnl or 0.0
    by_regime = [
        {"regime": k, "n": v[0], "sum_pnl": v[1]}
        for k, v in sorted(regime_agg.items(), key=lambda kv: kv[1][1])
    ]
    return {"by_asset": by_asset, "by_regime": by_regime}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestRegimeAndAsset -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics regime_and_asset (nearest-scan regime join)"
```

---

### Task 6: signal_funnel

**Files:**
- Modify: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Produces: `signal_funnel(conn, cutoff) -> Dict` = `{"scans": int, "buy_signals": int,
  "executed": int, "matured": int}`. scans/buy from `cnn_scans` (cutoff on `scanned_at`), executed =
  trades opened (cutoff on `opened_at`), matured = `signal_outcomes` with outcome set (cutoff on `created_at`).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestSignalFunnel -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
def signal_funnel(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict:
    sc_cl, sc_p = _where_since("scanned_at", cutoff)
    op_cl, op_p = _where_since("opened_at", cutoff)
    cr_cl, cr_p = _where_since("created_at", cutoff)
    scans = conn.execute(
        "SELECT COUNT(*) FROM cnn_scans WHERE 1=1" + sc_cl, sc_p).fetchone()[0]
    buys = conn.execute(
        "SELECT COUNT(*) FROM cnn_scans WHERE side='BUY'" + sc_cl, sc_p).fetchone()[0]
    executed = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE agent='CNN'" + op_cl, op_p).fetchone()[0]
    matured = conn.execute(
        "SELECT COUNT(*) FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        "AND outcome IN ('WIN','LOSS','NEUTRAL')" + cr_cl, cr_p).fetchone()[0]
    return {"scans": scans, "buy_signals": buys, "executed": executed, "matured": matured}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestSignalFunnel -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics signal_funnel"
```

---

### Task 7: compute_diagnostics orchestrator + 60s TTL cache

**Files:**
- Modify: `backend/services/diagnostics.py`
- Test: `backend/tests/test_diagnostics.py`

**Interfaces:**
- Produces: `compute_diagnostics(window: str, db_path: str, now: Optional[float]=None) -> Dict` =
  `{"window", "generated_at", "signal_edge", "exit_attribution", "regime_and_asset", "signal_funnel"}`.
  Opens `mode=ro` connection to `db_path`. 60s TTL cache keyed by `window` (module-level dict); a second
  call within TTL returns the cached object (identity-equal).

- [ ] **Step 1: Write the failing test**

```python
class TestComputeDiagnostics:
    def test_shape_and_cache(self, tmp_path):
        con = _seed(tmp_path)
        con.commit()
        con.close()
        db = str(tmp_path / "d.db")
        d._CACHE.clear()
        a = d.compute_diagnostics("all", db_path=db, now=_NOW)
        assert set(a) == {"window", "generated_at", "signal_edge",
                          "exit_attribution", "regime_and_asset", "signal_funnel"}
        b = d.compute_diagnostics("all", db_path=db, now=_NOW + 30)  # within TTL
        assert a is b  # cache hit returns same object
        c = d.compute_diagnostics("all", db_path=db, now=_NOW + 90)  # TTL expired
        assert c is not a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py::TestComputeDiagnostics -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
_CACHE: Dict[str, tuple] = {}  # window -> (expires_at, payload)
_TTL_SECS = 60.0


def compute_diagnostics(window: str, db_path: str, now: Optional[float] = None) -> Dict:
    now = time.time() if now is None else now
    hit = _CACHE.get(window)
    if hit and hit[0] > now:
        return hit[1]
    cutoff = window_cutoff(window, now)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        payload = {
            "window": window,
            "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "signal_edge": signal_edge(conn, cutoff),
            "exit_attribution": exit_attribution(conn, cutoff),
            "regime_and_asset": regime_and_asset(conn, cutoff),
            "signal_funnel": signal_funnel(conn, cutoff),
        }
    finally:
        conn.close()
    _CACHE[window] = (now + _TTL_SECS, payload)
    return payload
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics.py -v`
Expected: PASS (all diagnostics tests)

- [ ] **Step 5: Commit**

```bash
git add backend/services/diagnostics.py backend/tests/test_diagnostics.py
git commit -m "feat: diagnostics compute orchestrator + TTL cache"
```

---

### Task 8: GET /api/diagnostics endpoint

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_diagnostics_api.py`

**Interfaces:**
- Consumes: `services.diagnostics.compute_diagnostics`, the app's DB path (find how other endpoints
  resolve it — search `coinbase.db` / `DATABASE_URL` / `database._db_path` in `main.py` and reuse it).
- Produces: `GET /api/diagnostics?window=30d|90d|all` → JSON payload; unknown window → 400; internal
  error → 500. Never touches `app_state`/trading.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_diagnostics_api.py
import inspect, os, sys
BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
import main  # noqa: E402


def test_endpoint_exists_and_is_async():
    assert hasattr(main, "get_diagnostics")
    assert inspect.iscoroutinefunction(main.get_diagnostics)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics_api.py -v`
Expected: FAIL (`get_diagnostics` missing)

- [ ] **Step 3: Write minimal implementation**

Add near the other `@app.get` routes in `main.py` (import `from services import diagnostics` at top with the other service imports; reuse the existing DB-path resolution used by other read endpoints):

```python
@app.get("/api/diagnostics")
async def get_diagnostics(window: str = "30d"):
    try:
        return diagnostics.compute_diagnostics(window, db_path=_DIAG_DB_PATH)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:  # never propagate into trading state
        logger.exception("diagnostics failed")
        return JSONResponse({"error": str(e)}, status_code=500)
```

Where `_DIAG_DB_PATH` is the same path other endpoints read (e.g. the value passed to `database`/aiosqlite). If a module-level constant does not already exist, define `_DIAG_DB_PATH` next to the other path constants using the same source.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_diagnostics_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_diagnostics_api.py
git commit -m "feat: GET /api/diagnostics endpoint"
```

---

### Task 9: DiagnosticsDashboard.tsx tab (frontend)

**Files:**
- Create: `frontend/src/components/DiagnosticsDashboard.tsx`
- Modify: `frontend/src/App.tsx` (add `'Diagnostics'` to `TABS`, import + conditional render)

**Interfaces:**
- Consumes: `GET /api/diagnostics?window=`.
- Produces: a tab component rendering four sections; hand-rolled `<svg>` calibration line + bar charts,
  matching `EquityCurve.tsx` conventions. No new npm dependency.

- [ ] **Step 1: Read the reference component**

Read `frontend/src/components/EquityCurve.tsx` fully for the SVG idiom (viewBox, scales, theming
classes) and `frontend/src/App.tsx` for the `TABS` array + `activeTab === '...' && <Component/>` pattern.

- [ ] **Step 2: Create the component (fetch + render)**

```tsx
// frontend/src/components/DiagnosticsDashboard.tsx
import { useEffect, useState } from 'react'

type Cal = { bucket: number; n: number; win_rate: number; avg_ret: number }
type Diag = {
  window: string
  signal_edge: { n: number; precision: number; e_return: number; calibration: Cal[] }
  exit_attribution: { by_trigger: { trigger: string; n: number; sum_pnl: number;
    avg_pct: number; win_rate: number }[]; scan_sell_share: number }
  regime_and_asset: { by_asset: { product_id: string; n: number; sum_pnl: number;
    win_rate: number }[]; by_regime: { regime: string; n: number; sum_pnl: number }[] }
  signal_funnel: { scans: number; buy_signals: number; executed: number; matured: number }
}

export default function DiagnosticsDashboard() {
  const [window, setWindow] = useState<'30d' | '90d' | 'all'>('30d')
  const [data, setData] = useState<Diag | null>(null)
  const [err, setErr] = useState<string | null>(null)

  const load = () => {
    setErr(null)
    fetch(`/api/diagnostics?window=${window}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
      .then(setData)
      .catch((e) => setErr(String(e)))
  }
  useEffect(load, [window])

  if (err) return <div className="p-4 text-red-400">Diagnostics error: {err}</div>
  if (!data) return <div className="p-4 text-slate-400">Loading diagnostics…</div>

  return (
    <div className="p-4 space-y-6">
      <div className="flex gap-2 items-center">
        {(['30d', '90d', 'all'] as const).map((w) => (
          <button key={w} onClick={() => setWindow(w)}
            className={`px-3 py-1 rounded ${window === w ? 'bg-indigo-600' : 'bg-slate-700'}`}>
            {w}
          </button>
        ))}
        <button onClick={load} className="px-3 py-1 rounded bg-slate-700">refresh</button>
      </div>
      <CalibrationChart cal={data.signal_edge.calibration}
        precision={data.signal_edge.precision} eReturn={data.signal_edge.e_return} />
      <ExitTable rows={data.exit_attribution.by_trigger}
        share={data.exit_attribution.scan_sell_share} />
      <RegimeAsset ra={data.regime_and_asset} />
      <Funnel f={data.signal_funnel} />
    </div>
  )
}
```

- [ ] **Step 3: Add the sub-components (SVG calibration + tables)**

```tsx
function CalibrationChart({ cal, precision, eReturn }:
  { cal: { bucket: number; win_rate: number; n: number }[]; precision: number; eReturn: number }) {
  const W = 320, H = 200, pad = 30
  const x = (b: number) => pad + b * (W - 2 * pad)
  const y = (v: number) => H - pad - v * (H - 2 * pad)
  return (
    <section>
      <h3 className="font-semibold mb-1">Signal edge & calibration</h3>
      <div className="text-sm text-slate-400 mb-2">
        precision {(precision * 100).toFixed(1)}% · E[return] {(eReturn * 100).toFixed(2)}%
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full max-w-md bg-slate-900 rounded">
        <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} stroke="#475569" strokeDasharray="4" />
        <polyline fill="none" stroke="#818cf8" strokeWidth="2"
          points={cal.map((c) => `${x(c.bucket)},${y(c.win_rate)}`).join(' ')} />
        {cal.map((c) => <circle key={c.bucket} cx={x(c.bucket)} cy={y(c.win_rate)} r={3} fill="#818cf8" />)}
      </svg>
    </section>
  )
}

function ExitTable({ rows, share }:
  { rows: { trigger: string; n: number; sum_pnl: number; avg_pct: number; win_rate: number }[]; share: number }) {
  return (
    <section>
      <h3 className="font-semibold mb-1">Exit attribution
        <span className="text-sm text-slate-400"> · SCAN-SELL share {(share * 100).toFixed(0)}%</span></h3>
      <table className="text-sm w-full max-w-lg">
        <thead><tr className="text-slate-400 text-left">
          <th>trigger</th><th>n</th><th>sum PnL</th><th>avg %</th><th>WR</th></tr></thead>
        <tbody>{rows.map((r) => (
          <tr key={r.trigger}>
            <td>{r.trigger}</td><td>{r.n}</td>
            <td className={r.sum_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>${r.sum_pnl.toFixed(2)}</td>
            <td>{(r.avg_pct * 100).toFixed(2)}%</td><td>{(r.win_rate * 100).toFixed(0)}%</td>
          </tr>))}</tbody>
      </table>
    </section>
  )
}

function RegimeAsset({ ra }: { ra: Diag['regime_and_asset'] }) {
  return (
    <section className="flex gap-8 flex-wrap">
      <div>
        <h3 className="font-semibold mb-1">By regime</h3>
        {ra.by_regime.map((r) => (
          <div key={r.regime} className="text-sm">{r.regime}: {r.n} · ${r.sum_pnl.toFixed(2)}</div>
        ))}
      </div>
      <div>
        <h3 className="font-semibold mb-1">By asset (worst first)</h3>
        {ra.by_asset.slice(0, 15).map((a) => (
          <div key={a.product_id} className="text-sm">{a.product_id}: {a.n} · ${a.sum_pnl.toFixed(2)}</div>
        ))}
      </div>
    </section>
  )
}

function Funnel({ f }: { f: Diag['signal_funnel'] }) {
  const stages: [string, number][] = [
    ['scans', f.scans], ['BUY signals', f.buy_signals],
    ['executed', f.executed], ['matured', f.matured]]
  return (
    <section>
      <h3 className="font-semibold mb-1">Signal funnel</h3>
      {stages.map(([label, v]) => (
        <div key={label} className="text-sm">{label}: {v}</div>
      ))}
    </section>
  )
}
```

- [ ] **Step 4: Register the tab in App.tsx**

Add `'Diagnostics'` to the `TABS` array, `import DiagnosticsDashboard from './components/DiagnosticsDashboard'`, and a render branch: `{activeTab === 'Diagnostics' && <DiagnosticsDashboard />}` (match the existing branch style).

- [ ] **Step 5: Type-check + build**

Run: `cd frontend && npm install --ignore-scripts && npx tsc --noEmit && npm run build`
Expected: no TS errors; build succeeds.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/DiagnosticsDashboard.tsx frontend/src/App.tsx
git commit -m "feat: Diagnostics dashboard tab (signal/exit/regime/funnel)"
```

---

## Self-Review

**Spec coverage:** signal-edge+calibration (Task 3), exit attribution (Task 4), regime+per-asset (Task 5), signal funnel (Task 6), read-only diagnostics.py + TTL cache (Tasks 2/7), /api/diagnostics (Task 8), DiagnosticsDashboard tab + SVG + window selector (Task 9), performance index for the regime join (Task 1). All spec sections mapped.

**Placeholder scan:** no TBD/TODO; two deliberate "reuse the existing pattern" pointers (migration registry in `database.py` Task 1 Step 5; DB-path resolution in `main.py` Task 8 Step 3) — these are lookups in existing code, not code placeholders; exact symbol names can't be hard-coded without reading those files, which the step directs.

**Type consistency:** dict shapes in each backend task match the `Diag` TS type in Task 9 (calibration `{bucket,n,win_rate,avg_ret}`, by_trigger `{trigger,n,sum_pnl,avg_pct,win_rate}`, by_asset `{product_id,n,sum_pnl,win_rate}`, by_regime `{regime,n,sum_pnl}`, funnel `{scans,buy_signals,executed,matured}`). `compute_diagnostics` keys match the endpoint + frontend consumers.

## Deployment note

Backend changes take effect on the next 8001 restart (operator-gated); the index migration applies automatically on that restart. Frontend changes require a Vite rebuild/redeploy of the 5174 app. Default behavior of the trading loop is unchanged (purely additive read-only surface).
