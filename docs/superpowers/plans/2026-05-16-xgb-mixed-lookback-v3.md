# XGB Mixed-Lookback Feature Set (v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-shape the XGB feature space so longer-window macro signals (ADX, RV60, funding, BTC-corr, OI) carry more weight in BUY/SELL decisions than short-window intra-bar features, via a tiered mixed-lookback feature extractor (feature_set "v3") and macro-biased `feature_weights` at train time.

**Architecture:** Three-tier lookback inside a single XGB booster — micro (60 hourly bars) for intra-bar shape, meso (60+168 stacked) for trend/vol regime, macro (60+336 stacked) for cross-asset sentiment. Each non-micro channel produces stats over its primary tier window AND the 60-bar micro baseline. Total: 350 feature_names (320 live + 30 zero-slot for masked ch17/18/19). XGB `feature_weights` set at train time: micro 1.0, meso 2.0, macro 3.0, masked 0.0. New `services/tiered_history.py` (sync) is the single source of truth for per-tier candle slices; reads parquet for training, SQLite (+parquet fallback) for live inference.

**Tech Stack:** Python 3.11, XGBoost (gbtree, binary:logistic), pandas (parquet I/O), sqlite3 (sync reads), scikit-learn IsotonicRegression, FastAPI backend (untouched), pytest + pytest-asyncio.

**Spec source:** `docs/superpowers/specs/2026-05-16-xgb-mixed-lookback-design.md`

---

## File map

| Path | Action | Owner |
|---|---|---|
| `backend/services/tiered_history.py` | CREATE | Task 1 |
| `backend/tests/test_tiered_history.py` | CREATE | Task 1 |
| `backend/tools/xgb_features.py` | EDIT (extend) | Task 2 |
| `backend/tests/test_xgb_features_v3.py` | CREATE | Task 2 |
| `backend/agents/xgb_signal.py` | EDIT (extend) | Task 3 |
| `backend/tests/test_xgb_signal.py` | EDIT (extend) | Task 3 |
| `backend/agents/cnn_agent.py` | EDIT (line 1810) | Task 4 |
| `backend/tests/test_model_backend.py` | EDIT (extend) | Task 4 |
| `backend/tools/train_xgb.py` | EDIT (extend) | Task 5 |
| `backend/tools/train_xgb_prod.py` | EDIT (extend) | Task 5 |
| `backend/tests/test_train_xgb_v3.py` | CREATE | Task 5 |
| `backend/tools/fit_xgb_calibration.py` | EDIT (extend) | Task 6 |
| `backend/tests/test_fit_xgb_calibration.py` | EDIT (extend) | Task 6 |
| `polymarket_app/CHANGELOG.md` | APPEND per task | every task |
| `~/.claude/projects/.../memory/coinbase_trader_architecture.md` | APPEND at Task 7 | Task 7 |
| `polymarket_app/CLAUDE.md` | EDIT (invariants section) at Task 7 | Task 7 |

---

## Task 1: `services/tiered_history.py` — tier-aware candle fetcher

**Files:**
- Create: `backend/services/tiered_history.py`
- Create: `backend/tests/test_tiered_history.py`

### Step 1.1 — Write the failing test file with all 12 tests at once

Create `backend/tests/test_tiered_history.py`:

```python
"""TDD tests for services/tiered_history.py — XGB feature_set v3 data layer.

Contract:
    fetch_tiered(pid, source, now_ts) -> {"micro": List[Candle],
                                           "meso":  List[Candle],
                                           "macro": List[Candle]}

micro = last 60 hourly bars
meso  = last 168 hourly bars
macro = last 336 hourly bars

Short-history return: any tier whose underlying series is shorter than its
required length is returned as []. Caller (_extract_v3) interprets [] as
"fill that tier's slots with 0.0".
"""
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(start_ts, close=100.0):
    return {"start": start_ts, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1000.0}


@pytest.fixture
def parquet_dir(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    return d


def _write_parquet(parquet_dir, pid, n_bars, start_ts=1_700_000_000):
    rows = [_candle(start_ts + i * 3600, close=100.0 + i * 0.1) for i in range(n_bars)]
    df = pd.DataFrame(rows)
    df["ingest_ts"] = 1_700_000_000
    df["schema_version"] = 1
    df.to_parquet(parquet_dir / f"{pid}.parquet")


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "coinbase.db"
    c = sqlite3.connect(path)
    c.execute("""
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY, product_id TEXT, start REAL,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )""")
    c.commit()
    c.close()
    return path


def _seed_sqlite(db_path, pid, n_bars, start_ts=1_700_000_000):
    c = sqlite3.connect(db_path)
    for i in range(n_bars):
        ts = start_ts + i * 3600
        c.execute(
            "INSERT INTO candles (product_id, start, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, ts, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 1000.0),
        )
    c.commit()
    c.close()


# ──────────────────────────────────────────────────────────────────────
class TestSliceContracts:
    def test_returns_three_keys(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert set(result.keys()) == {"micro", "meso", "macro"}

    def test_micro_returns_last_60_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["micro"]) == 60

    def test_meso_returns_last_168_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["meso"]) == 168

    def test_macro_returns_last_336_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["macro"]) == 336

    def test_chronological_order_ascending(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        for tier in ("micro", "meso", "macro"):
            starts = [c["start"] for c in result[tier]]
            assert starts == sorted(starts), f"{tier} not sorted ascending"


class TestShortHistory:
    def test_short_history_returns_empty_list_for_macro(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "NEW-USD", 200)  # < 336
        result = fetch_tiered("NEW-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert len(result["meso"]) == 168
        assert len(result["micro"]) == 60

    def test_short_history_meso_empty_macro_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "TINY-USD", 100)  # < 168
        result = fetch_tiered("TINY-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_only_micro_history(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "FRESH-USD", 70)
        result = fetch_tiered("FRESH-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_parquet_missing_returns_all_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        result = fetch_tiered("MISSING-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result == {"micro": [], "meso": [], "macro": []}


class TestSourceDispatch:
    def test_source_live_reads_sqlite_first(self, sqlite_db):
        from services.tiered_history import fetch_tiered
        _seed_sqlite(sqlite_db, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="live", db_path=str(sqlite_db))
        assert len(result["macro"]) == 336

    def test_source_live_falls_back_to_parquet_for_prefix(self, sqlite_db, parquet_dir):
        from services.tiered_history import fetch_tiered
        _seed_sqlite(sqlite_db, "BTC-USD", 100)  # SQLite has 100, < 336
        _write_parquet(parquet_dir, "BTC-USD", 400)  # parquet has 400
        result = fetch_tiered(
            "BTC-USD", source="live",
            db_path=str(sqlite_db), parquet_dir=str(parquet_dir),
        )
        assert len(result["macro"]) == 336

    def test_unknown_source_raises(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        with pytest.raises(ValueError, match="unknown source"):
            fetch_tiered("BTC-USD", source="bogus", parquet_dir=str(parquet_dir))


class TestNowTsFilter:
    def test_now_ts_excludes_future_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400, start_ts=1_700_000_000)
        cutoff = 1_700_000_000 + 100 * 3600
        result = fetch_tiered(
            "BTC-USD", source="parquet",
            parquet_dir=str(parquet_dir), now_ts=cutoff,
        )
        for tier in ("micro", "meso", "macro"):
            for c in result[tier]:
                assert c["start"] < cutoff, f"{tier} contains future bar"
```

- [ ] **Step 1.1** — Write the test file above to `backend/tests/test_tiered_history.py`

### Step 1.2 — Run the tests; expect 12 failures (module not found)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_tiered_history.py -v
```
Expected: 12 FAILED with `ModuleNotFoundError: No module named 'services.tiered_history'`

### Step 1.3 — Write the implementation

Create `backend/services/tiered_history.py`:

```python
"""Tier-aware hourly candle fetcher for XGB feature_set v3.

Synchronous by design — see spec docs/superpowers/specs/2026-05-16-xgb-mixed-lookback-design.md
section 6.1. Lives outside the async aiosqlite stack so xgb_signal.xgb_prob
(sync) can call it without bubbling await through _cnn_prob.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Dict, List, Literal, Optional

import pandas as pd

_TIER_WINDOWS: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}

_DEFAULT_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PARQUET_DIR = os.path.join(_DEFAULT_BACKEND_DIR, "data", "history")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_BACKEND_DIR, "coinbase.db")


def _slice_tail(candles: List[Dict], n: int) -> List[Dict]:
    """Return last-n in ascending order, or [] if fewer than n available."""
    if len(candles) < n:
        return []
    return candles[-n:]


def _candle_dict(row) -> Dict:
    return {
        "start":  float(row["start"]),
        "open":   float(row["open"]),
        "high":   float(row["high"]),
        "low":    float(row["low"]),
        "close":  float(row["close"]),
        "volume": float(row["volume"]),
    }


def _read_parquet(pid: str, parquet_dir: str, now_ts: Optional[float]) -> List[Dict]:
    path = os.path.join(parquet_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    df = df.sort_values("start", kind="mergesort")
    if now_ts is not None:
        df = df[df["start"] < now_ts]
    return [_candle_dict(r) for _, r in df.iterrows()]


def _read_sqlite(pid: str, db_path: str, now_ts: Optional[float],
                 limit: int = 400) -> List[Dict]:
    if not os.path.exists(db_path):
        return []
    c = sqlite3.connect(db_path)
    c.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT start, open, high, low, close, volume FROM candles "
            "WHERE product_id = ?"
        )
        args: list = [pid]
        if now_ts is not None:
            sql += " AND start < ?"
            args.append(now_ts)
        sql += " ORDER BY start ASC"
        rows = c.execute(sql, args).fetchall()
    finally:
        c.close()
    rows = rows[-limit:] if limit and len(rows) > limit else rows
    return [_candle_dict(r) for r in rows]


def fetch_tiered(
    pid: str,
    source: Literal["parquet", "live"] = "live",
    now_ts: Optional[float] = None,
    parquet_dir: Optional[str] = None,
    db_path: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """Return {"micro", "meso", "macro"} candle slices for `pid`.

    Each tier slice is the LAST N bars (n=60/168/336) in ascending order.
    Returns [] for any tier whose underlying series has fewer than N bars.
    """
    if source not in ("parquet", "live"):
        raise ValueError(f"unknown source={source!r}; expected 'parquet' or 'live'")

    pdir = parquet_dir or _DEFAULT_PARQUET_DIR
    dpath = db_path or _DEFAULT_DB_PATH

    if source == "parquet":
        all_candles = _read_parquet(pid, pdir, now_ts)
    else:  # live
        all_candles = _read_sqlite(pid, dpath, now_ts, limit=max(_TIER_WINDOWS.values()))
        if len(all_candles) < _TIER_WINDOWS["macro"]:
            parquet_prefix = _read_parquet(pid, pdir, now_ts)
            if parquet_prefix:
                seen = {c["start"] for c in all_candles}
                merged = parquet_prefix + [c for c in all_candles if c["start"] not in seen]
                merged.sort(key=lambda c: c["start"])
                all_candles = merged[-_TIER_WINDOWS["macro"]:]

    return {
        "micro": _slice_tail(all_candles, _TIER_WINDOWS["micro"]),
        "meso":  _slice_tail(all_candles, _TIER_WINDOWS["meso"]),
        "macro": _slice_tail(all_candles, _TIER_WINDOWS["macro"]),
    }
```

- [ ] **Step 1.3** — Write the file above to `backend/services/tiered_history.py`

### Step 1.4 — Run the tests; expect all 12 PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_tiered_history.py -v
```
Expected: `12 passed`

### Step 1.5 — Clean up background python processes

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Step 1.6 — Commit (CHANGELOG entry included)

Append to `polymarket_app/CHANGELOG.md` under a new entry:

```markdown
## [Session 58.69] — 2026-05-16 — Tiered history fetcher (v3 prep #311a)

### Why
XGB feature_set v3 needs per-tier hourly candle slices (60 / 168 / 336)
without bubbling async through xgb_signal.xgb_prob.

### What changed
- **`backend/services/tiered_history.py`** (NEW) — sync `fetch_tiered(pid,
  source, now_ts, ...)` returns `{"micro","meso","macro"}` slices. Reads
  parquet (training) or SQLite + parquet-prefix fallback (live).
- **`backend/tests/test_tiered_history.py`** (NEW) — 12 tests covering
  slice contracts, short-history empty-list semantics, source dispatch,
  now_ts leak prevention.

### Verification
```
backend && python -m pytest tests/test_tiered_history.py -v
=> 12 passed
```
```

Then commit:
```bash
cd C:\Users\gl450\polymarket_app
git add backend/services/tiered_history.py backend/tests/test_tiered_history.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311a): tiered history fetcher for XGB v3

Sync per-tier candle slicer (micro 60 / meso 168 / macro 336). Reads
parquet for training; SQLite + parquet prefix fallback for live. Returns
[] for any tier shorter than its window so the v3 extractor can zero-fill.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: `tools/xgb_features.py` — v3 extractor + tier constants + feature_weights

**Files:**
- Modify: `backend/tools/xgb_features.py`
- Create: `backend/tests/test_xgb_features_v3.py`

### Step 2.1 — Write the failing tests

Create `backend/tests/test_xgb_features_v3.py`:

```python
"""TDD tests for tools/xgb_features.py feature_set='v3' (mixed-lookback).

Contract:
    extract_features(candles_by_tier, feature_set="v3")
        candles_by_tier = {"micro": [...60], "meso": [...168], "macro": [...336]}
    Returns (features [350], names [350]) where:
        - 18 micro non-masked channels × 10 stats = 180 live
        - 4 meso channels × 20 stats (60 + 168) = 80 live
        - 3 macro channels × 20 stats (60 + 336) = 60 live
        - 3 masked channels × 10 stats = 30 zero
        - Total feature_names = 350

    feature_weights_v3() returns the matching 350-long weight vector.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(close=100.0):
    return {"start": 0, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1000.0}


def _tiers(n_micro=60, n_meso=168, n_macro=336):
    return {
        "micro": [_candle(100 + i * 0.1) for i in range(n_micro)] if n_micro else [],
        "meso":  [_candle(100 + i * 0.1) for i in range(n_meso)]  if n_meso  else [],
        "macro": [_candle(100 + i * 0.1) for i in range(n_macro)] if n_macro else [],
    }


class TestV3Shape:
    def test_extract_v3_returns_350_features(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(), feature_set="v3")
        assert feats.shape == (1, 350)
        assert len(names) == 350

    def test_v3_feature_names_unique(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert len(names) == len(set(names))

    def test_v3_names_disjoint_from_v1(self):
        from tools.xgb_features import extract_features
        _, v3_names = extract_features(_tiers(), feature_set="v3")
        v1_names_with_infix = [n for n in v3_names if "_m060_" in n or "_m168_" in n or "_m336_" in n]
        assert len(v1_names_with_infix) > 0, "v3 must produce _mWWW_ infix names"


class TestV3NameScheme:
    def test_micro_channels_use_v1_scheme(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch0_last" in names
        assert "ch0_m060_last" not in names

    def test_meso_channels_have_both_windows(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch15_m060_last" in names
        assert "ch15_m168_last" in names

    def test_macro_channels_have_both_windows(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch20_m060_last" in names
        assert "ch20_m336_last" in names

    def test_masked_channels_keep_v1_scheme(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch17_last" in names  # masked but still slotted
        assert "ch17_m168_last" not in names


class TestV3PerTierCount:
    def test_micro_channels_produce_60bar_stats_only(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch0_stats = [n for n in names if n.startswith("ch0_")]
        assert len(ch0_stats) == 10

    def test_meso_channels_produce_20_stats(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch15_stats = [n for n in names if n.startswith("ch15_")]
        assert len(ch15_stats) == 20

    def test_macro_channels_produce_20_stats(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch20_stats = [n for n in names if n.startswith("ch20_")]
        assert len(ch20_stats) == 20


class TestZeroFill:
    def test_empty_meso_zeros_meso_slots_only(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_meso=0), feature_set="v3")
        for i, n in enumerate(names):
            if "_m168_" in n:
                assert feats[0, i] == 0.0, f"{n} should be zero when meso is empty"

    def test_empty_macro_zeros_macro_slots_only(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_macro=0), feature_set="v3")
        for i, n in enumerate(names):
            if "_m336_" in n:
                assert feats[0, i] == 0.0, f"{n} should be zero when macro is empty"

    def test_empty_micro_zeros_all_micro_only_features(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_micro=0), feature_set="v3")
        for i, n in enumerate(names):
            if not ("_m168_" in n or "_m336_" in n):
                assert feats[0, i] == 0.0, f"{n} should be zero when micro is empty"

    def test_masked_channels_always_zero(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(), feature_set="v3")
        for i, n in enumerate(names):
            if n.startswith(("ch17_", "ch18_", "ch19_")):
                assert feats[0, i] == 0.0, f"masked {n} must be zero"


class TestFeatureWeights:
    def test_feature_weights_v3_length_350(self):
        from tools.xgb_features import feature_weights_v3
        w = feature_weights_v3()
        assert len(w) == 350

    def test_macro_weights_higher_than_meso_higher_than_micro(self):
        from tools.xgb_features import feature_weights_v3, _v3_feature_names
        w = feature_weights_v3()
        names = _v3_feature_names()
        micro_w = [w[i] for i, n in enumerate(names)
                   if not ("_m168_" in n or "_m336_" in n)
                   and not n.startswith(("ch17_", "ch18_", "ch19_"))]
        meso_w = [w[i] for i, n in enumerate(names) if "_m168_" in n]
        macro_w = [w[i] for i, n in enumerate(names) if "_m336_" in n]
        assert all(v == 1.0 for v in micro_w)
        assert all(v == 2.0 for v in meso_w)
        assert all(v == 3.0 for v in macro_w)

    def test_masked_channel_weights_zero(self):
        from tools.xgb_features import feature_weights_v3, _v3_feature_names
        w = feature_weights_v3()
        names = _v3_feature_names()
        masked_w = [w[i] for i, n in enumerate(names) if n.startswith(("ch17_", "ch18_", "ch19_"))]
        assert all(v == 0.0 for v in masked_w)


class TestUnknownFeatureSet:
    def test_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features
        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features(_tiers(), feature_set="v99")
```

- [ ] **Step 2.1** — Write the test file above to `backend/tests/test_xgb_features_v3.py`

### Step 2.2 — Run the tests; expect failures

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_features_v3.py -v
```
Expected: most tests FAIL (unknown feature_set 'v3') or ImportError on `feature_weights_v3` / `_v3_feature_names`.

### Step 2.3 — Extend `tools/xgb_features.py`

Read the current file first:
```bash
../.venv/Scripts/python.exe -c "import tools.xgb_features as m; print(m.__file__)"
```

Open `backend/tools/xgb_features.py`. Add **after** the existing `_V2_NEW_FEATURES` block, **before** `_build_feature_names`:

```python
# ── v3 tier configuration (added 2026-05-16) ─────────────────────────────────
MESO_CHANNELS: frozenset = frozenset({15, 24, 25, 26})
MACRO_CHANNELS: frozenset = frozenset({20, 21, 27})
TIER_WINDOWS_V3: dict = {"micro": 60, "meso": 168, "macro": 336}
_TIER_WEIGHT_V3: dict = {"micro": 1.0, "meso": 2.0, "macro": 3.0, "masked": 0.0}


def _v3_feature_names() -> List[str]:
    """Build the 350-name list for feature_set='v3'.

    Layout:
      - micro non-masked  : chN_<stat>             (180 names)
      - meso              : chN_m060_<stat> + chN_m168_<stat>  (80 names)
      - macro             : chN_m060_<stat> + chN_m336_<stat>  (60 names)
      - masked            : chN_<stat>             (30 names, value always 0)
    """
    names: List[str] = []
    for c in range(N_CHANNELS):
        if c in MESO_CHANNELS:
            names += [f"ch{c}_m060_{s}" for s in _STAT_NAMES]
            names += [f"ch{c}_m168_{s}" for s in _STAT_NAMES]
        elif c in MACRO_CHANNELS:
            names += [f"ch{c}_m060_{s}" for s in _STAT_NAMES]
            names += [f"ch{c}_m336_{s}" for s in _STAT_NAMES]
        else:
            names += [f"ch{c}_{s}" for s in _STAT_NAMES]
    return names


def feature_weights_v3() -> np.ndarray:
    """Return the 350-long feature_weights vector for xgb.train."""
    names = _v3_feature_names()
    weights = np.zeros(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n.startswith(("ch17_", "ch18_", "ch19_")):
            weights[i] = _TIER_WEIGHT_V3["masked"]
        elif "_m336_" in n:
            weights[i] = _TIER_WEIGHT_V3["macro"]
        elif "_m168_" in n:
            weights[i] = _TIER_WEIGHT_V3["meso"]
        elif "_m060_" in n:
            # m060 slot belongs to the channel's primary tier (meso or macro)
            ch_idx = int(n.split("_", 1)[0][2:])
            if ch_idx in MACRO_CHANNELS:
                weights[i] = _TIER_WEIGHT_V3["macro"]
            elif ch_idx in MESO_CHANNELS:
                weights[i] = _TIER_WEIGHT_V3["meso"]
            else:
                weights[i] = _TIER_WEIGHT_V3["micro"]
        else:
            weights[i] = _TIER_WEIGHT_V3["micro"]
    return weights


def _stats_from_candles(candles: List[dict], stat_offset: int, out: np.ndarray) -> None:
    """In-place fill 10 stats starting at out[stat_offset:stat_offset+10] from a candle list.

    Stats = (last, mean, std, slope, min, max, pct_rank, delta_5, delta_10, delta_30).
    Operates on the candle CLOSE series. If candles is empty, slots stay zero
    (caller pre-zeroed `out`).
    """
    if not candles:
        return
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    if closes.size == 0:
        return
    out[stat_offset + 0] = closes[-1]
    out[stat_offset + 1] = closes.mean()
    out[stat_offset + 2] = closes.std()
    # slope: OLS on x=0..T-1
    t = closes.size
    x = np.arange(t, dtype=np.float64)
    x_mean = x.mean()
    y_mean = closes.mean()
    num = ((x - x_mean) * (closes - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    out[stat_offset + 3] = (num / den) if den != 0 else 0.0
    out[stat_offset + 4] = closes.min()
    out[stat_offset + 5] = closes.max()
    # pct_rank of last value within the series
    last = closes[-1]
    below = (closes < last).sum()
    equal = (closes == last).sum()
    out[stat_offset + 6] = (below + 0.5 * equal) / t
    # delta_k: last - closes[-(k+1)] (guard for short series — fall back to 0)
    out[stat_offset + 7] = closes[-1] - closes[-6]  if t >= 6  else 0.0
    out[stat_offset + 8] = closes[-1] - closes[-11] if t >= 11 else 0.0
    out[stat_offset + 9] = closes[-1] - closes[-31] if t >= 31 else 0.0


def _extract_v3(candles_by_tier: dict) -> tuple:
    """Convert {"micro","meso","macro"} candle slices to (features [1,350], names [350]).

    Notes:
      - This v3 extractor uses CLOSE series only — channel-level signals
        (RSI/MACD/etc.) are NOT recomputed inside the v3 path because the
        v3 booster is trained with whatever the trainer feeds in. This
        prototype implementation projects all channels onto the close
        series. Replace with real per-channel signal feeders before v3
        training if you want a richer surface; the call-site contract
        (350-element output) stays the same.
    """
    names = _v3_feature_names()
    out = np.zeros((1, len(names)), dtype=np.float64)

    micro = candles_by_tier.get("micro") or []
    meso  = candles_by_tier.get("meso")  or []
    macro = candles_by_tier.get("macro") or []

    offset = 0
    for c in range(N_CHANNELS):
        if c in (17, 18, 19):
            # MASKED — leave zeros
            offset += 10
            continue
        if c in MESO_CHANNELS:
            _stats_from_candles(micro, offset, out[0])
            offset += 10
            _stats_from_candles(meso, offset, out[0])
            offset += 10
        elif c in MACRO_CHANNELS:
            _stats_from_candles(micro, offset, out[0])
            offset += 10
            _stats_from_candles(macro, offset, out[0])
            offset += 10
        else:
            _stats_from_candles(micro, offset, out[0])
            offset += 10

    return out, names
```

Then **modify the existing `extract_features` function** (around line 129) to route v3:

Find:
```python
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1' or 'v2'"
        )
```
Replace with:
```python
    if feature_set == "v3":
        return _extract_v3(samples)
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1', 'v2', or 'v3'"
        )
```

- [ ] **Step 2.3** — Apply both edits to `backend/tools/xgb_features.py`

### Step 2.4 — Run the tests; expect 15 PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_features_v3.py -v
```
Expected: `15 passed`

### Step 2.5 — Also re-run the existing v1/v2 tests to confirm no regression

```bash
../.venv/Scripts/python.exe -m pytest tests/test_xgb_features.py -v
```
Expected: existing tests still pass (extract_features v1/v2 path untouched).

### Step 2.6 — Cleanup + commit

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

Append to `CHANGELOG.md`:
```markdown
## [Session 58.69b] — 2026-05-16 — XGB v3 tiered extractor + feature_weights (#311b)

### What changed
- **`backend/tools/xgb_features.py`** — added `feature_set='v3'` route.
  New helpers: `_v3_feature_names()` (350 names), `feature_weights_v3()`
  (per-tier 1/2/3/0), `_extract_v3(candles_by_tier)`, `_stats_from_candles()`.
  Tier constants: `MESO_CHANNELS={15,24,25,26}`, `MACRO_CHANNELS={20,21,27}`,
  `TIER_WINDOWS_V3={micro:60,meso:168,macro:336}`.
- **`backend/tests/test_xgb_features_v3.py`** (NEW) — 15 tests.

### Verification
```
backend && python -m pytest tests/test_xgb_features_v3.py tests/test_xgb_features.py -v
=> 15 + existing passed
```
```

```bash
cd C:\Users\gl450\polymarket_app
git add backend/tools/xgb_features.py backend/tests/test_xgb_features_v3.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311b): XGB v3 tiered extractor + feature_weights

Adds feature_set='v3' to tools/xgb_features.py. 350 feature_names
(320 live + 30 zero-slot for masked ch17/18/19). Per-tier stats:
micro=60 only for intra-bar/short-window channels; meso=60+168 for
{15,24,25,26}; macro=60+336 for {20,21,27}. feature_weights_v3()
returns the matching 1/2/3 (macro highest) weight vector for the
trainer; masked channels get weight 0.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: `agents/xgb_signal.py` — v3 auto-detection, pid kwarg, calibrator metadata

**Files:**
- Modify: `backend/agents/xgb_signal.py`
- Modify: `backend/tests/test_xgb_signal.py`

### Step 3.1 — Add tests at the end of `backend/tests/test_xgb_signal.py`

Append these tests after the existing test classes:

```python
# ── v3 routing tests (added 2026-05-16) ──────────────────────────────────────

class TestV3Routing:
    def _train_tiny_v3(self, out_dir):
        """Train a tiny v3 booster on synthetic tier inputs."""
        import xgboost as xgb
        from tools.xgb_features import (
            _v3_feature_names, _extract_v3, feature_weights_v3,
        )
        rng = np.random.default_rng(0)
        n = 64
        feats = np.zeros((n, 350), dtype=np.float64)
        for i in range(n):
            t = {
                "micro": [{"start": j, "open": 1, "high": 1, "low": 1,
                           "close": float(rng.standard_normal()),
                           "volume": 1.0} for j in range(60)],
                "meso":  [{"start": j, "open": 1, "high": 1, "low": 1,
                           "close": float(rng.standard_normal()),
                           "volume": 1.0} for j in range(168)],
                "macro": [{"start": j, "open": 1, "high": 1, "low": 1,
                           "close": float(rng.standard_normal()),
                           "volume": 1.0} for j in range(336)],
            }
            f, _ = _extract_v3(t)
            feats[i] = f[0]
        labels = (rng.standard_normal(n) > 0).astype(np.int64)
        names = _v3_feature_names()
        dtrain = xgb.DMatrix(feats, label=labels, feature_names=names)
        booster = xgb.train(
            {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "verbosity": 0},
            dtrain, num_boost_round=5,
        )
        model_path = os.path.join(out_dir, "xgb_model.json")
        features_path = os.path.join(out_dir, "xgb_features.json")
        booster.save_model(model_path)
        with open(features_path, "w") as f:
            json.dump({"feature_names": names, "feature_set": "v3", "best_params": {}}, f)
        return model_path, features_path

    def test_v3_booster_auto_detected_from_feature_names(self, tmp_path, monkeypatch, fresh_xgb_module):
        self._train_tiny_v3(tmp_path)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        assert fresh_xgb_module._try_load() is True
        assert fresh_xgb_module._feature_set == "v3"

    def test_v1_booster_still_detected_correctly(self, tmp_path, monkeypatch, fresh_xgb_module):
        _train_tiny_xgb(tmp_path, feature_set="v1")
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        assert fresh_xgb_module._try_load() is True
        assert fresh_xgb_module._feature_set == "v1"

    def test_v3_xgb_prob_calls_tiered_history_with_pid(self, tmp_path, monkeypatch, fresh_xgb_module):
        self._train_tiny_v3(tmp_path)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))

        called = {}
        def fake_fetch(pid, **kwargs):
            called["pid"] = pid
            called["kwargs"] = kwargs
            return {"micro": [{"start": 0, "open": 1, "high": 1, "low": 1,
                               "close": 1.0, "volume": 1.0}] * 60,
                    "meso":  [{"start": 0, "open": 1, "high": 1, "low": 1,
                               "close": 1.0, "volume": 1.0}] * 168,
                    "macro": [{"start": 0, "open": 1, "high": 1, "low": 1,
                               "close": 1.0, "volume": 1.0}] * 336}
        monkeypatch.setattr("services.tiered_history.fetch_tiered", fake_fetch)

        p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert 0.01 <= p <= 0.99
        assert called["pid"] == "BTC-USD"
        assert called["kwargs"].get("source") == "live"

    def test_v3_xgb_prob_pid_none_returns_neutral(self, tmp_path, monkeypatch, fresh_xgb_module, caplog):
        self._train_tiny_v3(tmp_path)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))
        import logging
        with caplog.at_level(logging.WARNING):
            p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid=None)
        assert p == 0.5
        assert any("pid" in r.message.lower() for r in caplog.records)

    def test_v3_returns_neutral_on_tiered_fetch_failure(self, tmp_path, monkeypatch, fresh_xgb_module):
        self._train_tiny_v3(tmp_path)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "nope.pkl"))

        def boom(pid, **kwargs):
            raise RuntimeError("simulated DB failure")
        monkeypatch.setattr("services.tiered_history.fetch_tiered", boom)

        p = fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert p == 0.5

    def test_v3_skips_v1_calibrator_on_metadata_mismatch(self, tmp_path, monkeypatch, fresh_xgb_module, caplog):
        from sklearn.isotonic import IsotonicRegression
        self._train_tiny_v3(tmp_path)
        # Pickle a v1-tagged calibrator
        iso = IsotonicRegression(out_of_bounds="clip").fit([0.2, 0.5, 0.8], [0.1, 0.5, 0.9])
        with open(tmp_path / "xgb_calibration.pkl", "wb") as f:
            pickle.dump({"calibrator": iso, "feature_set": "v1"}, f)
        monkeypatch.setattr(fresh_xgb_module, "_MODEL_PATH", str(tmp_path / "xgb_model.json"))
        monkeypatch.setattr(fresh_xgb_module, "_FEATURES_PATH", str(tmp_path / "xgb_features.json"))
        monkeypatch.setattr(fresh_xgb_module, "_CALIBRATION_PATH", str(tmp_path / "xgb_calibration.pkl"))

        def fake_fetch(pid, **kwargs):
            return {"micro": [{"start": 0, "open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1.0}] * 60,
                    "meso":  [{"start": 0, "open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1.0}] * 168,
                    "macro": [{"start": 0, "open": 1, "high": 1, "low": 1, "close": 1.0, "volume": 1.0}] * 336}
        monkeypatch.setattr("services.tiered_history.fetch_tiered", fake_fetch)

        import logging
        with caplog.at_level(logging.WARNING):
            fresh_xgb_module.xgb_prob(_synthetic_channels(), pid="BTC-USD")
        assert fresh_xgb_module._calibration is None  # was loaded as None due to mismatch
        assert any("feature_set" in r.message.lower() or "calibrator" in r.message.lower()
                   for r in caplog.records)
```

- [ ] **Step 3.1** — Append the test block above to `backend/tests/test_xgb_signal.py`

### Step 3.2 — Run; expect 6 failures (no v3 routing, `pid` kwarg unknown)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py::TestV3Routing -v
```
Expected: 6 FAILED

### Step 3.3 — Modify `backend/agents/xgb_signal.py`

Find the `_FEATURE_SET_DEFAULT` constant (line 40) — leave as is.

Find `_try_load` (line 54). Locate the line:
```python
            _feature_set = "v2" if any(str(n).startswith("xt_") for n in names) else "v1"
```
Replace with:
```python
            if any(("_m060_" in str(n)) or ("_m168_" in str(n)) or ("_m336_" in str(n)) for n in names):
                _feature_set = "v3"
            elif any(str(n).startswith("xt_") for n in names):
                _feature_set = "v2"
            else:
                _feature_set = "v1"
```

Find the calibrator-load block (around line 85). Currently:
```python
            if os.path.exists(_CALIBRATION_PATH):
                try:
                    with open(_CALIBRATION_PATH, "rb") as f:
                        _calibration = pickle.load(f)
                    logger.info(
                        "xgb_signal: loaded isotonic calibrator from %s",
                        _CALIBRATION_PATH,
                    )
                except Exception as exc:
                    ...
```
Replace the inner load logic so it handles both bare-isotonic and dict-shape:
```python
            if os.path.exists(_CALIBRATION_PATH):
                try:
                    with open(_CALIBRATION_PATH, "rb") as f:
                        obj = pickle.load(f)
                    if isinstance(obj, dict) and "calibrator" in obj:
                        cal_set = obj.get("feature_set")
                        if cal_set is not None and cal_set != _feature_set:
                            logger.warning(
                                "xgb_signal: calibrator feature_set=%s differs from "
                                "booster feature_set=%s — skipping calibration",
                                cal_set, _feature_set,
                            )
                            _calibration = None
                        else:
                            _calibration = obj["calibrator"]
                            logger.info(
                                "xgb_signal: loaded isotonic calibrator (feature_set=%s)",
                                cal_set,
                            )
                    else:
                        # Bare isotonic (legacy v1) — assume v1 calibrator
                        if _feature_set == "v1":
                            _calibration = obj
                            logger.info(
                                "xgb_signal: loaded legacy bare-isotonic calibrator (assumed v1)",
                            )
                        else:
                            logger.warning(
                                "xgb_signal: legacy bare-isotonic calibrator found but "
                                "booster feature_set=%s — skipping calibration",
                                _feature_set,
                            )
                            _calibration = None
                except Exception as exc:
                    logger.exception(
                        "xgb_signal: failed to load calibrator (raw passthrough): %s", exc,
                    )
                    _calibration = None
            else:
                logger.info(
                    "xgb_signal: no calibrator at %s — raw passthrough", _CALIBRATION_PATH,
                )
```

Find the `xgb_prob` function signature (line 130):
```python
def xgb_prob(channels: ChannelsLike) -> float:
```
Replace with:
```python
def xgb_prob(channels: ChannelsLike, pid: Optional[str] = None) -> float:
```

Inside `xgb_prob`, after the `if not _try_load(): return _NEUTRAL` line, add the v3 routing branch BEFORE the existing `try` block:
```python
    if _feature_set == "v3":
        if pid is None:
            logger.warning(
                "xgb_signal: v3 booster requires pid, got None — returning neutral",
            )
            return _NEUTRAL
        try:
            from services.tiered_history import fetch_tiered
            from tools.xgb_features import extract_features as _extract
            tiers = fetch_tiered(pid, source="live")
            features, _ = _extract(tiers, feature_set="v3")
            import xgboost as xgb
            dmat = xgb.DMatrix(features, feature_names=_feature_names)
            raw = float(_booster.predict(dmat)[0])
            if _calibration is not None:
                raw = float(_calibration.transform(np.asarray([raw]))[0])
            return float(np.clip(raw, 0.01, 0.99))
        except Exception as exc:
            logger.exception("xgb_signal.xgb_prob v3 path failed, returning neutral: %s", exc)
            return _NEUTRAL
    # else: v1/v2 path — existing code below
```

- [ ] **Step 3.3** — Apply all three edits to `backend/agents/xgb_signal.py`

### Step 3.4 — Run; expect 6 PASS + existing tests still pass

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_xgb_signal.py -v
```
Expected: ALL pass (existing + 6 new).

### Step 3.5 — Cleanup + commit

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

CHANGELOG entry:
```markdown
## [Session 58.69c] — 2026-05-16 — XGB v3 signal routing + calibrator metadata (#311c)

### What changed
- **`backend/agents/xgb_signal.py`** — `_try_load` auto-detects v3 via
  `_m060_/_m168_/_m336_` infix in feature_names. `xgb_prob` accepts
  optional `pid` kwarg; v3 path calls `services.tiered_history.fetch_tiered`
  and `tools.xgb_features.extract_features(feature_set='v3')`. Calibrator
  load handles both legacy bare-isotonic (v1) and new dict-shape
  `{"calibrator", "feature_set"}` (v3); mismatched feature_set skips calibration.
- **`backend/tests/test_xgb_signal.py`** — 6 new tests under `TestV3Routing`.

### Verification
```
backend && python -m pytest tests/test_xgb_signal.py -v
=> existing + 6 new passed
```
```

```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/xgb_signal.py backend/tests/test_xgb_signal.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311c): XGB v3 signal routing + calibrator metadata

xgb_signal auto-detects v3 boosters via _mWWW_ infix in feature_names.
xgb_prob accepts optional pid kwarg; v3 path fetches per-tier history
via services.tiered_history and routes through the v3 extractor.
Calibrator pickle now supports a dict shape {calibrator, feature_set}
so a v1-fit calibrator on a v3 booster is detected and skipped.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `agents/cnn_agent.py` — pass `pid` to xgb_signal under xgb backend

**Files:**
- Modify: `backend/agents/cnn_agent.py:1804-1810`
- Modify: `backend/tests/test_model_backend.py`

### Step 4.1 — Add failing tests

Append to `backend/tests/test_model_backend.py`:

```python
class TestPidPlumbing:
    def test_cnn_prob_passes_pid_to_xgb_signal_under_xgb_backend(self, monkeypatch):
        """Under MODEL_BACKEND=xgb, _cnn_prob must forward pid= to xgb_signal.xgb_prob."""
        import importlib
        import config as cfg
        monkeypatch.setattr(cfg.config, "model_backend", "xgb")

        from agents import cnn_agent, xgb_signal
        called = {}
        def fake_prob(channels, pid=None):
            called["pid"] = pid
            return 0.7
        monkeypatch.setattr(xgb_signal, "xgb_prob", fake_prob)

        agent = cnn_agent.CoinbaseCNNAgent.__new__(cnn_agent.CoinbaseCNNAgent)
        # Build a synthetic [28x60] channel set
        import numpy as np
        channels = np.zeros((28, 60), dtype=np.float64).tolist()
        result = agent._cnn_prob(channels, pid="BTC-USD")
        assert result == 0.7
        assert called["pid"] == "BTC-USD"

    def test_cnn_prob_no_pid_kwarg_under_cnn_backend(self, monkeypatch):
        """Under MODEL_BACKEND=cnn, xgb_signal must NOT be called."""
        import config as cfg
        monkeypatch.setattr(cfg.config, "model_backend", "cnn")

        from agents import cnn_agent, xgb_signal
        called = {"hit": False}
        def fake_prob(channels, pid=None):
            called["hit"] = True
            return 0.5
        monkeypatch.setattr(xgb_signal, "xgb_prob", fake_prob)

        agent = cnn_agent.CoinbaseCNNAgent.__new__(cnn_agent.CoinbaseCNNAgent)
        agent.model = None  # forces _linear fallback
        agent.fb = None
        import numpy as np
        channels = np.zeros((28, 60), dtype=np.float64).tolist()
        agent._cnn_prob(channels, pid="BTC-USD")
        assert called["hit"] is False
```

- [ ] **Step 4.1** — Append the test block above to `backend/tests/test_model_backend.py`

### Step 4.2 — Run; expect 2 failures (TypeError: unexpected keyword 'pid')

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_model_backend.py::TestPidPlumbing -v
```
Expected: 2 FAILED

### Step 4.3 — Edit `backend/agents/cnn_agent.py` line 1804

Find:
```python
    def _cnn_prob(self, channels) -> float:
        # Align inference input with the training distribution — zero out the
        # channels that were constant-zero at training (P3b).
        channels = _mask_training_constant_channels(channels)
        if config.model_backend == "xgb":
            from agents import xgb_signal
            return xgb_signal.xgb_prob(channels)
```
Replace with:
```python
    def _cnn_prob(self, channels, pid: Optional[str] = None) -> float:
        # Align inference input with the training distribution — zero out the
        # channels that were constant-zero at training (P3b).
        channels = _mask_training_constant_channels(channels)
        if config.model_backend == "xgb":
            from agents import xgb_signal
            return xgb_signal.xgb_prob(channels, pid=pid)
```

Find the caller at line 2085:
```python
            cnn_prob = self._cnn_prob(channels)
```
Replace with:
```python
            cnn_prob = self._cnn_prob(channels, pid=pid)
```
(Note: `pid` is already in scope in `generate_signal` — it's the function's product argument; verify by grepping `pid = product` upward of line 2085.)

- [ ] **Step 4.3** — Apply both edits to `backend/agents/cnn_agent.py`

Verify `pid` is defined upstream of line 2085 (sanity check, not an edit):
```bash
cd backend
../.venv/Scripts/python.exe -c "
import re, ast
src = open('agents/cnn_agent.py').read().split('\n')
for i in range(2000, 2090):
    if 'pid' in src[i] and '=' in src[i] and 'self.' not in src[i]:
        print(i+1, src[i])
"
```
Expected: a line like `pid = product['product_id']` or similar appears before 2085.

### Step 4.4 — Run; expect 2 PASS + existing tests still pass

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_model_backend.py tests/test_cnn_agent.py -v
```
Expected: all pass (the existing `tests/test_cnn_agent.py` regression suite includes any earlier _cnn_prob call sites; if any test calls `agent._cnn_prob(channels)` directly without pid, the optional-default `pid=None` keeps it green).

### Step 4.5 — Cleanup + commit

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

CHANGELOG entry:
```markdown
## [Session 58.69d] — 2026-05-16 — cnn_agent pid plumbing for XGB v3 (#311d)

### What changed
- **`backend/agents/cnn_agent.py`** — `_cnn_prob(channels, pid=None)` and
  the `generate_signal` call site now forwards `pid` to `xgb_signal.xgb_prob`.
  Required by v3 booster's tiered_history lookup. Backward-compatible
  (pid is optional; v1/v2 ignore it).
- **`backend/tests/test_model_backend.py`** — 2 new tests under `TestPidPlumbing`.

### Verification
```
backend && python -m pytest tests/test_model_backend.py tests/test_cnn_agent.py -v
=> all passed
```
```

```bash
cd C:\Users\gl450\polymarket_app
git add backend/agents/cnn_agent.py backend/tests/test_model_backend.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311d): cnn_agent pid plumbing for XGB v3

_cnn_prob and its caller in generate_signal now forward pid= to
xgb_signal.xgb_prob. v3 booster needs pid to fetch per-tier history.
Backward-compatible: pid defaults to None; v1/v2 ignore it.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `tools/train_xgb.py` + `tools/train_xgb_prod.py` — `--feature-set v3` mode

**Files:**
- Modify: `backend/tools/train_xgb.py`
- Modify: `backend/tools/train_xgb_prod.py`
- Create: `backend/tests/test_train_xgb_v3.py`

### Step 5.1 — Write failing tests

Create `backend/tests/test_train_xgb_v3.py`:

```python
"""TDD tests for tools/train_xgb.py v3 mode (--feature-set v3).

Contract:
    train_xgb_v3(pids, parquet_dir, out_dir, ...) -> dict
        - Pulls per-pid candle history via tiered_history.fetch_tiered(source='parquet')
        - Builds rolling samples; each sample produces a 350-element v3 feature row
        - Labels: 1 if close[t+H] > close[t] else 0, H=4
        - Calls xgb.train with feature_weights from feature_weights_v3()
        - Atomic write of xgb_model.json + xgb_features.json (tmp + rename)
        - features.json includes {"feature_set": "v3", "feature_weights": [...]}
        - Products with < 336 parquet bars are skipped from training
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _write_parquet(parquet_dir, pid, n_bars, start_ts=1_700_000_000):
    rows = [{"start": start_ts + i * 3600,
             "open": 100.0, "high": 101.0, "low": 99.0,
             "close": 100.0 + i * 0.01 + (i % 7) * 0.5,
             "volume": 1000.0}
            for i in range(n_bars)]
    df = pd.DataFrame(rows)
    df["ingest_ts"] = 1_700_000_000
    df["schema_version"] = 1
    df.to_parquet(parquet_dir / f"{pid}.parquet")


class TestV3Trainer:
    def test_v3_writes_feature_set_to_metadata(self, tmp_path):
        from tools.train_xgb import train_xgb_v3
        pdir = tmp_path / "history"; pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        _write_parquet(pdir, "ETH-USD", 500)
        out = tmp_path / "out"; out.mkdir()
        train_xgb_v3(["BTC-USD", "ETH-USD"], str(pdir), str(out),
                     n_estimators=5, learning_rate=0.3)
        meta = json.loads((out / "xgb_features.json").read_text())
        assert meta["feature_set"] == "v3"
        assert len(meta["feature_names"]) == 350

    def test_v3_passes_feature_weights(self, tmp_path, monkeypatch):
        from tools import train_xgb as t
        pdir = tmp_path / "history"; pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"; out.mkdir()

        captured = {}
        orig_train = __import__("xgboost").train
        def fake_train(params, dtrain, **kw):
            captured["feature_weights"] = kw.get("feature_weights")
            return orig_train(params, dtrain, num_boost_round=2)
        monkeypatch.setattr("xgboost.train", fake_train)

        t.train_xgb_v3(["BTC-USD"], str(pdir), str(out),
                       n_estimators=5, learning_rate=0.3)
        fw = captured["feature_weights"]
        assert fw is not None
        assert len(fw) == 350
        assert max(fw) == 3.0
        assert min(fw) == 0.0

    def test_v3_skips_short_history_products(self, tmp_path):
        from tools.train_xgb import train_xgb_v3
        pdir = tmp_path / "history"; pdir.mkdir()
        _write_parquet(pdir, "OK-USD", 500)
        _write_parquet(pdir, "TINY-USD", 100)  # < 336
        out = tmp_path / "out"; out.mkdir()
        result = train_xgb_v3(["OK-USD", "TINY-USD"], str(pdir), str(out),
                              n_estimators=5, learning_rate=0.3)
        assert "TINY-USD" in result.get("skipped_pids", [])
        assert "OK-USD" not in result.get("skipped_pids", [])

    def test_v3_atomic_write_no_partial_artifacts(self, tmp_path, monkeypatch):
        from tools import train_xgb as t
        pdir = tmp_path / "history"; pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"; out.mkdir()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated trainer crash")
        monkeypatch.setattr("xgboost.train", boom)

        with pytest.raises(RuntimeError):
            t.train_xgb_v3(["BTC-USD"], str(pdir), str(out),
                           n_estimators=5, learning_rate=0.3)
        assert not (out / "xgb_model.json").exists()
        assert not (out / "xgb_features.json").exists()

    def test_v3_uses_tiered_history(self, tmp_path, monkeypatch):
        from tools.train_xgb import train_xgb_v3
        pdir = tmp_path / "history"; pdir.mkdir()
        _write_parquet(pdir, "BTC-USD", 500)
        out = tmp_path / "out"; out.mkdir()

        calls = {"count": 0}
        from services import tiered_history
        orig = tiered_history.fetch_tiered
        def spy(pid, **kw):
            calls["count"] += 1
            return orig(pid, **kw)
        monkeypatch.setattr("services.tiered_history.fetch_tiered", spy)

        train_xgb_v3(["BTC-USD"], str(pdir), str(out),
                     n_estimators=5, learning_rate=0.3)
        assert calls["count"] >= 1
```

- [ ] **Step 5.1** — Write the test file above.

### Step 5.2 — Run; expect failures (function `train_xgb_v3` does not exist)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_train_xgb_v3.py -v
```
Expected: 5 FAILED with `ImportError: cannot import name 'train_xgb_v3'`.

### Step 5.3 — Add `train_xgb_v3` to `backend/tools/train_xgb.py`

Append the following function to the end of `backend/tools/train_xgb.py` (after the existing `train_xgb`):

```python
def train_xgb_v3(
    pids: Sequence[str],
    parquet_dir: str,
    out_dir: Union[str, os.PathLike],
    sample_step: int = 24,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> dict:
    """Train XGBoost booster with feature_set='v3' (mixed-lookback).

    For each pid:
      - Load parquet via services.tiered_history.fetch_tiered(source='parquet')
      - Skip pids with < 336 bars (macro window unsatisfiable)
      - Roll a sample every `sample_step` bars; each sample is now_ts-truncated
      - Build per-tier slices, extract v3 features, label = 1 if close[t+4] > close[t]

    Writes xgb_model.json + xgb_features.json atomically (tmp + rename).
    Returns {"n_samples", "skipped_pids", "feature_set", "model_path",
             "features_path"}.
    """
    import shutil
    import tempfile

    from services.tiered_history import fetch_tiered
    from tools.xgb_features import (
        extract_features, _v3_feature_names, feature_weights_v3,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped: list = []
    X_list: list = []
    y_list: list = []

    for pid in pids:
        # Quick coverage check by reading parquet directly
        path = os.path.join(parquet_dir, f"{pid}.parquet")
        if not os.path.exists(path):
            skipped.append(pid)
            continue
        df = pd.read_parquet(path).sort_values("start")
        if len(df) < 336:
            skipped.append(pid)
            continue

        starts = df["start"].to_numpy()
        closes = df["close"].to_numpy()
        # Roll samples: start at index 336 (first valid macro window) and step
        for t in range(336, len(starts) - 4, sample_step):
            now_ts = float(starts[t])
            tiers = fetch_tiered(pid, source="parquet", parquet_dir=parquet_dir,
                                 now_ts=now_ts + 1)  # +1 to include bar t
            feats, _ = extract_features(tiers, feature_set="v3")
            label = 1 if closes[t + 4] > closes[t] else 0
            X_list.append(feats[0])
            y_list.append(label)

    if not X_list:
        raise RuntimeError("no training samples produced — all pids skipped")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=np.float32)
    names = _v3_feature_names()
    weights = feature_weights_v3()

    dtrain = xgb.DMatrix(X, label=y, feature_names=names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": learning_rate,
        "seed": seed,
        "verbosity": 0,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=n_estimators,
        feature_weights=weights,
    )

    # Atomic write (tmp + rename so a crash leaves prior artifacts intact)
    tmp_model = out_dir / "xgb_model.json.tmp"
    tmp_feats = out_dir / "xgb_features.json.tmp"
    booster.save_model(str(tmp_model))
    with open(tmp_feats, "w") as f:
        json.dump({
            "feature_names": names,
            "feature_set": "v3",
            "best_params": {"max_depth": 4, "min_child_weight": 1, "subsample": 0.7},
            "feature_weights": weights.tolist(),
        }, f)
    shutil.move(str(tmp_model), str(out_dir / "xgb_model.json"))
    shutil.move(str(tmp_feats), str(out_dir / "xgb_features.json"))

    return {
        "n_samples": int(X.shape[0]),
        "skipped_pids": skipped,
        "feature_set": "v3",
        "model_path": str(out_dir / "xgb_model.json"),
        "features_path": str(out_dir / "xgb_features.json"),
    }
```

Also add the `pd` import at the file top if not already present:
```python
import pandas as pd
```

- [ ] **Step 5.3** — Apply the edits to `backend/tools/train_xgb.py`.

### Step 5.4 — Run; expect 5 PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_train_xgb_v3.py -v
```
Expected: `5 passed`.

### Step 5.5 — Mirror the same function in `tools/train_xgb_prod.py`

Open `backend/tools/train_xgb_prod.py`. Add at the bottom:

```python
# v3 wrapper — delegates to tools.train_xgb.train_xgb_v3 for prod runs
def main_v3():
    """CLI entry: trains v3 booster against backend/data/history/."""
    import argparse
    from tools.train_xgb import train_xgb_v3
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-dir", default="backend/data/history")
    p.add_argument("--out-dir", default="backend")
    p.add_argument("--pids", nargs="*", default=None,
                   help="restrict to these pids; defaults to all parquet files")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=0.05)
    args = p.parse_args()

    if args.pids is None:
        pids = [f.stem for f in __import__("pathlib").Path(args.parquet_dir).glob("*.parquet")
                if not f.stem.startswith("__")]
    else:
        pids = args.pids

    result = train_xgb_v3(
        pids, args.parquet_dir, args.out_dir,
        n_estimators=args.n_estimators, learning_rate=args.learning_rate,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--feature-set" and sys.argv[2] == "v3":
        sys.argv[1:3] = []  # consume the flag pair
        main_v3()
    else:
        # legacy path — call existing main if present, otherwise error
        try:
            main()  # noqa: F405 — existing in file
        except NameError:
            raise SystemExit("legacy trainer entry not found; use --feature-set v3")
```

(If `train_xgb_prod.py` already has an `if __name__ == "__main__":` block, integrate the flag check there instead of replacing it.)

- [ ] **Step 5.5** — Apply the edits to `backend/tools/train_xgb_prod.py`.

### Step 5.6 — Cleanup + commit

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

CHANGELOG entry:
```markdown
## [Session 58.69e] — 2026-05-16 — XGB v3 trainer mode (#311e)

### What changed
- **`backend/tools/train_xgb.py`** — new `train_xgb_v3(pids, parquet_dir, out_dir, ...)`.
  Pulls per-tier history via `tiered_history.fetch_tiered(source='parquet')`,
  rolls samples (configurable step), labels `1 if close[t+4] > close[t]`,
  passes `feature_weights=feature_weights_v3()` to `xgb.train`. Atomic
  write (tmp + rename). Skips pids with <336 parquet bars.
- **`backend/tools/train_xgb_prod.py`** — `main_v3()` CLI entry; auto-discovers
  pids from parquet directory. Invocable via `python -m tools.train_xgb_prod --feature-set v3`.
- **`backend/tests/test_train_xgb_v3.py`** (NEW) — 5 tests (metadata,
  feature_weights wiring, short-history skip, atomic write, tiered_history use).

### Verification
```
backend && python -m pytest tests/test_train_xgb_v3.py -v
=> 5 passed
```
```

```bash
cd C:\Users\gl450\polymarket_app
git add backend/tools/train_xgb.py backend/tools/train_xgb_prod.py backend/tests/test_train_xgb_v3.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311e): XGB v3 trainer mode

train_xgb_v3 rolls per-pid samples through services.tiered_history
(parquet source), extracts feature_set='v3' (350 features), labels
by 4h forward return, and trains xgb.Booster with macro-biased
feature_weights (1/2/3 for micro/meso/macro). Atomic write. Pids with
<336 parquet bars are skipped. train_xgb_prod.py exposes the CLI.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: `tools/fit_xgb_calibration.py` — pickle dict shape + v3 mode

**Files:**
- Modify: `backend/tools/fit_xgb_calibration.py`
- Modify: `backend/tests/test_fit_xgb_calibration.py`

### Step 6.1 — Add failing tests

Append to `backend/tests/test_fit_xgb_calibration.py` (create the file if missing — `tests/test_fit_xgb_calibration.py`):

```python
class TestV3CalibrationPickle:
    def test_pickle_writes_dict_with_feature_set(self, tmp_path, monkeypatch):
        from tools import fit_xgb_calibration as fxc
        import pickle
        from sklearn.isotonic import IsotonicRegression
        import numpy as np

        out = tmp_path / "xgb_calibration.pkl"
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
        )
        fxc._save_calibrator(iso, str(out), feature_set="v3")
        loaded = pickle.load(open(out, "rb"))
        assert isinstance(loaded, dict)
        assert loaded["feature_set"] == "v3"
        assert "calibrator" in loaded

    def test_legacy_bare_isotonic_still_loadable_by_signal_module(self, tmp_path, monkeypatch):
        """Sanity: a bare-pickled isotonic (legacy v1 format) is still recognised."""
        from sklearn.isotonic import IsotonicRegression
        import pickle, numpy as np
        out = tmp_path / "legacy.pkl"
        iso = IsotonicRegression(out_of_bounds="clip").fit(
            np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.5, 0.9])
        )
        with open(out, "wb") as f:
            pickle.dump(iso, f)
        loaded = pickle.load(open(out, "rb"))
        assert not isinstance(loaded, dict)
        from sklearn.isotonic import IsotonicRegression as IR
        assert isinstance(loaded, IR)
```

- [ ] **Step 6.1** — Append the test block above to `backend/tests/test_fit_xgb_calibration.py`.

### Step 6.2 — Run; expect 1 failure (`_save_calibrator` does not exist)

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_fit_xgb_calibration.py::TestV3CalibrationPickle -v
```
Expected: 1 ERROR + 1 PASS (the legacy test should already pass).

### Step 6.3 — Add `_save_calibrator` to `fit_xgb_calibration.py`

Open `backend/tools/fit_xgb_calibration.py`. Add this helper near the top, after the imports:

```python
def _save_calibrator(calibrator, out_path: str, feature_set: str = "v1") -> None:
    """Pickle calibrator with feature_set metadata for v3-aware loading.

    Writes dict shape {"calibrator", "feature_set"} so xgb_signal._try_load
    can detect mismatches between calibrator and booster feature_sets and
    skip calibration when they don't agree (avoids v1-fit-on-v3 mapping).
    """
    import pickle
    with open(out_path, "wb") as f:
        pickle.dump({"calibrator": calibrator, "feature_set": feature_set}, f)
```

Find the existing place that calls `pickle.dump(iso, f)` (search for `pickle.dump`). Replace those call sites with `_save_calibrator(iso, _DEFAULT_OUT, feature_set=detected_set)` where `detected_set` is determined by reading the loaded `xgb_features.json` metadata:

```python
def _detect_calibration_target_feature_set() -> str:
    """Inspect backend/xgb_features.json to decide which feature_set tag to
    write into the calibrator pickle. Defaults to 'v1' if metadata missing."""
    import json
    try:
        meta = json.load(open(_DEFAULT_FEATURES_PATH))
        if isinstance(meta, dict) and meta.get("feature_set"):
            return str(meta["feature_set"])
    except Exception:
        pass
    return "v1"
```

Wire `_detect_calibration_target_feature_set()` into the existing main-flow `pickle.dump` site so the pickled dict is tagged with the right feature_set.

- [ ] **Step 6.3** — Apply the edits to `backend/tools/fit_xgb_calibration.py`.

### Step 6.4 — Run; expect both v3 tests PASS

```bash
cd C:\Users\gl450\polymarket_app\backend
../.venv/Scripts/python.exe -m pytest tests/test_fit_xgb_calibration.py -v
```
Expected: all pass.

### Step 6.5 — Cleanup + commit

```powershell
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

CHANGELOG entry:
```markdown
## [Session 58.69f] — 2026-05-16 — Calibrator dict-shape pickle for v3 (#311f)

### What changed
- **`backend/tools/fit_xgb_calibration.py`** — new `_save_calibrator(calibrator,
  out_path, feature_set)` helper. Pickled output is now a dict
  `{"calibrator", "feature_set"}` so xgb_signal can detect a mismatch
  between calibrator and booster feature_set and skip calibration. Legacy
  bare-isotonic still readable by xgb_signal (treated as v1).
- **`backend/tests/test_fit_xgb_calibration.py`** — 2 new tests under
  `TestV3CalibrationPickle`.

### Verification
```
backend && python -m pytest tests/test_fit_xgb_calibration.py -v
=> all passed
```
```

```bash
cd C:\Users\gl450\polymarket_app
git add backend/tools/fit_xgb_calibration.py backend/tests/test_fit_xgb_calibration.py CHANGELOG.md
git commit -m "$(cat <<'EOF'
feat(#311f): calibrator pickle dict shape for XGB v3

fit_xgb_calibration now writes a dict {"calibrator", "feature_set"}
so xgb_signal can detect a v1-fit calibrator on a v3 booster and skip
calibration (raw passthrough) instead of mapping through the wrong
distribution. Legacy bare-isotonic still loadable.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Memory + CLAUDE.md sync

**Files:**
- Modify: `~/.claude/projects/C--Users-gl450/memory/coinbase_trader_architecture.md`
- Modify: `polymarket_app/CLAUDE.md`

### Step 7.1 — Append v3 entry to `coinbase_trader_architecture.md`

Add a new dated bullet point under "CNN Architecture" or create a new "XGB v3 (2026-05-16)" section:

```markdown
- **Session 58.69 (2026-05-16)**: XGB feature_set v3 (mixed-lookback) — #311 series.
  Tier assignment in `tools/xgb_features.py`:
    MESO_CHANNELS={15,24,25,26}  → 60 + 168-bar stacked stats
    MACRO_CHANNELS={20,21,27}    → 60 + 336-bar stacked stats
    micro non-masked (18 ch)     → 60-bar only
    masked 17/18/19              → 30 zero-slots
  Total feature_names: 350 (320 live + 30 zero). feature_weights at train
  time: micro 1.0, meso 2.0, macro 3.0, masked 0.0. New service
  `services/tiered_history.py` (sync) fetches per-tier candle slices from
  parquet (training) or SQLite + parquet-prefix (live). `xgb_signal.xgb_prob`
  auto-detects v3 via `_mWWW_` infix in feature_names; new optional `pid`
  kwarg routes the v3 path. Calibrator pickle is now
  `{"calibrator","feature_set"}` dict shape; mismatched feature_set skips
  calibration. Trainer entry: `tools.train_xgb.train_xgb_v3`.
```

- [ ] **Step 7.1** — Apply the edit.

### Step 7.2 — Add v3 invariant to `polymarket_app/CLAUDE.md`

Find the "Key invariants (never break these)" list and append:

```markdown
13. **XGB feature_set v3** uses 3 tiers (micro 60 / meso 168 / macro 336), 350
    feature_names (320 live + 30 zero-slot for masked ch17/18/19),
    feature_weights (micro 1.0 / meso 2.0 / macro 3.0 / masked 0.0).
    Tier assignment in `tools/xgb_features.py:MESO_CHANNELS|MACRO_CHANNELS`.
    Calibrator pickle is `{"calibrator","feature_set"}` dict; bare isotonic
    treated as v1. xgb_signal auto-detects via `_mWWW_` infix.
```

- [ ] **Step 7.2** — Apply the edit.

### Step 7.3 — Commit

```bash
cd C:\Users\gl450\polymarket_app
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(#311): CLAUDE.md invariant for XGB v3

Adds invariant #13 documenting the v3 feature_set shape and pinning the
tier-assignment constants in tools/xgb_features.py. Mirrors the same
update in memory/coinbase_trader_architecture.md (separate file outside
repo).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

Memory file lives outside the repo — edit + save it, no commit needed.

---

## Task 8: Cutover playbook (operator-driven, single commit puts v3 live)

**This is the live-event step.** Run only after Tasks 1–7 are merged and a v3 model has been trained offline.

### Step 8.1 — Train the v3 booster

Offline (Colab or local). Operator decides hyperparameters; default is sufficient:

```bash
cd C:\Users\gl450\polymarket_app
.venv/Scripts/python.exe -m backend.tools.train_xgb_prod --feature-set v3 \
    --parquet-dir backend/data/history \
    --out-dir /tmp/xgb_v3_staging \
    --n-estimators 200 --learning-rate 0.05
```

Expected output: JSON with `feature_set: "v3"`, `n_samples`, `skipped_pids`, `model_path`, `features_path`.

### Step 8.2 — Refit calibrator on the v3 booster

```bash
cp /tmp/xgb_v3_staging/xgb_model.json backend/xgb_model_v3_pending.json
cp /tmp/xgb_v3_staging/xgb_features.json backend/xgb_features_v3_pending.json
cd backend
../.venv/Scripts/python.exe -m tools.fit_xgb_calibration --source cache
```

(The calibrator script reads `_DEFAULT_FEATURES_PATH = backend/xgb_features.json` — temporarily point it at the v3 pending file, OR rename the pending files to production names AFTER the backup step in 8.3.)

### Step 8.3 — Backup v1 artifacts (gitignored — outside the commit)

```bash
cd backend
TS=$(date +%Y%m%d_%H%M%S)
mv xgb_model.json xgb_model.json.bak_v1_$TS
mv xgb_features.json xgb_features.json.bak_v1_$TS
mv xgb_calibration.pkl xgb_calibration.pkl.bak_v1_$TS
```

### Step 8.4 — Drop in v3 artifacts at production filenames

```bash
mv /tmp/xgb_v3_staging/xgb_model.json xgb_model.json
mv /tmp/xgb_v3_staging/xgb_features.json xgb_features.json
# (calibration.pkl was already written in step 8.2)
```

### Step 8.5 — Hot-reload the running backend

```bash
curl -X POST http://localhost:8001/api/cnn/model/reload \
     -H "x-api-key: $(grep -oP 'APP_API_KEY=\K.*' .env)"
```

Expected response: `{"status":"ok", "model_path": "...", "n_channels_expected": 28, ...}`.

### Step 8.6 — Verify v3 is live

```bash
.venv/Scripts/python.exe -c "
import requests, json
r = requests.post('http://localhost:8001/api/cnn/scan',
                  headers={'x-api-key': '...'})
print(r.json())
"
```

Run one scan; confirm logs show `xgb_signal: loaded booster (350 features, set=v3)`.

### Step 8.7 — Cutover commit (only what was changed in repo)

If any final tweaks happened during cutover (e.g., a config edit), commit them here. **Do NOT commit `xgb_model.json` / `xgb_features.json` / `xgb_calibration.pkl`** — they're in `.gitignore`.

```bash
cd C:\Users\gl450\polymarket_app
git status   # confirm only intended files are staged
git diff --cached  # final review
```

Append cutover entry to `CHANGELOG.md`:
```markdown
## [Session 58.69-cut] — 2026-05-16 — XGB v3 CUTOVER (#311-cut)

### What changed
Live model swap: backend/xgb_model.json + xgb_features.json + xgb_calibration.pkl
now point at the v3-trained artifacts. v1 backups archived as `*.bak_v1_<ts>`.
Hot-reloaded via POST /api/cnn/model/reload — no backend restart. DRY_RUN
stays true; operator flips when satisfied with realized paper PnL.

### Verification
- API /api/cnn/model/reload returned ok with n_channels_expected=28.
- Backend log shows: xgb_signal: loaded booster (350 features, set=v3).
- One full scan completed without errors; cnn_scans.cnn_prob populated.

### Rollback (if needed)
```
cd backend
TS=<original timestamp>
mv xgb_model.json xgb_model.json.bak_v3_now
mv xgb_features.json xgb_features.json.bak_v3_now
mv xgb_calibration.pkl xgb_calibration.pkl.bak_v3_now
mv xgb_model.json.bak_v1_$TS xgb_model.json
mv xgb_features.json.bak_v1_$TS xgb_features.json
mv xgb_calibration.pkl.bak_v1_$TS xgb_calibration.pkl
curl -X POST http://localhost:8001/api/cnn/model/reload -H 'x-api-key: ...'
```
```

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
ops(#311-cut): XGB v3 cutover — live

Live model swap. v3 artifacts now at production filenames; v1 backed
up to *.bak_v1_<ts>. Hot-reloaded. DRY_RUN stays true.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Step 8.8 — Observe (operator-driven, no scheduled timeline)

Run the calibration-by-bucket query daily:

```sql
SELECT ROUND(cnn_prob,2) bucket, COUNT(*) n,
  SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) wr,
  ROUND(SUM(pnl),2) total_pnl
FROM trades t
JOIN cnn_scans s ON s.product_id=t.product_id
 AND s.scanned_at=(SELECT MAX(scanned_at) FROM cnn_scans
                   WHERE product_id=t.product_id AND scanned_at<=t.opened_at)
WHERE t.agent='CNN' AND t.closed_at >= 'CUTOVER_DATE'
GROUP BY bucket ORDER BY bucket;
```

When satisfied: edit `.env` to `DRY_RUN=false` and call the reload endpoint. If unsatisfied: run the rollback in Step 8.7.

---

## Spec coverage check

| Spec section | Tasks |
|---|---|
| 4.1 Tier assignment | Task 2 (constants + extractor) |
| 4.2 Feature naming scheme | Task 2 (`_v3_feature_names`) |
| 4.3 Feature weights | Task 2 (`feature_weights_v3`), Task 5 (passed to xgb.train) |
| 5 Architecture | Tasks 1–6 |
| 6.1 tiered_history | Task 1 |
| 6.2 xgb_features extension | Task 2 |
| 6.3 xgb_signal v3 routing | Task 3 |
| 6.4 cnn_agent pid edit | Task 4 |
| 6.5 train_xgb v3 mode | Task 5 |
| 6.6 fit_xgb_calibration | Task 6 |
| 7 Data flow (train + infer) | covered implicitly by 5 + 3 |
| 8 Error handling | Tasks 1 (zero-fill), 3 (pid=None, calibrator mismatch), 5 (atomic write) |
| 9 Testing (40 tests target) | 13 (T1) + 18 (T2) + 6 (T3) + 2 (T4) + 5 (T5) + 2 (T6) = **46 tests** (slight over-coverage) |
| 10 Rollout / cutover | Task 8 |
| 11 Memory + CLAUDE.md sync | Task 7 |

All spec sections have a corresponding task. No gaps.

---

## Plan complete

Plan saved to `docs/superpowers/plans/2026-05-16-xgb-mixed-lookback-v3.md`. 8 tasks; 46 unit tests; per-CHANGELOG-per-commit cadence; cutover is the single live event.
