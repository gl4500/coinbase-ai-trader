# Strategy-Discovery Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the curated 50-token universe into per-token parquet files containing 13 features (6 tokenomic + 7 trend) and 5 dynamic-exit horizon labels (1h/4h/24h/72h/168h), ready for Phase 3 mining.

**Architecture:** Four loosely-coupled pure-function modules under `backend/tools/strategy_discovery/`. `features.py` derives trend features from 1h OHLCV. `tokenomic_stamp.py` aligns daily CoinPaprika snapshots onto the hourly grid with T+1 causality. `labels.py` simulates the deployed dynamic-exit policy (SL 8%, ATR trail 6% floor, max-hold 168 bars, 1.2% fee) for each horizon. `build_phase2.py` orchestrates load → features → stamp → labels → write per pid in the universe.

**Tech Stack:** Python 3, pandas (DataFrame ops + `ewm`), numpy, pyarrow (parquet I/O). No HTTP, no DB, no model loading. Pytest with mocks only.

**Spec:** `docs/superpowers/specs/2026-05-23-strategy-discovery-phase2-design.md`

---

## File Map

| File | Purpose |
|---|---|
| `backend/tools/strategy_discovery/features.py` (NEW) | Trend features from 1h OHLCV |
| `backend/tools/strategy_discovery/tokenomic_stamp.py` (NEW) | T+1 daily-snapshot stamping |
| `backend/tools/strategy_discovery/labels.py` (NEW) | Per-horizon dynamic-exit simulation |
| `backend/tools/strategy_discovery/build_phase2.py` (NEW) | Orchestrator + CLI |
| `backend/tests/tools/strategy_discovery/test_features.py` (NEW) | 5 tests |
| `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py` (NEW) | 4 tests |
| `backend/tests/tools/strategy_discovery/test_labels.py` (NEW) | 7 tests |
| `backend/tests/tools/strategy_discovery/test_build_phase2.py` (NEW) | 3 tests |
| `backend/tests/tools/strategy_discovery/__init__.py` (NEW if missing) | pytest discovery marker |
| `CHANGELOG.md` (MODIFY, append entry) | Session log of Phase 2 implementation |
| `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` (MODIFY) | Add Phase 2 module section + spec/plan cross-references |

**Module boundary discipline (per `feedback_loose_coupling`):**

- `features.py` knows about pandas + numpy; nothing about tokenomics, labels, parquet, or pids.
- `tokenomic_stamp.py` knows about pandas + the `SupplySnapshot` dataclass; nothing about trend features or labels.
- `labels.py` knows about pandas + numpy + the OHLCV+ATR contract; nothing about parquet I/O or tokenomics.
- `build_phase2.py` is the only module that imports the other three and is the only module that touches the filesystem.

Tests for each module never import the other three modules.

---

## Task 1: Trend features (`features.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/features.py`
- Create: `backend/tests/tools/strategy_discovery/__init__.py` (empty file, pytest marker)
- Create: `backend/tests/tools/strategy_discovery/test_features.py`

**Scaffolding (write before any test):**

- [ ] **Step 1.0a: Ensure test package marker exists**

Create `backend/tests/tools/strategy_discovery/__init__.py` as a 0-byte file (only if it does not already exist). Confirm with:

```powershell
Test-Path backend\tests\tools\strategy_discovery\__init__.py
```

Expected: `True`.

- [ ] **Step 1.0b: Create skeleton `features.py`**

Create `backend/tools/strategy_discovery/features.py` with the module docstring + imports + the public `_TREND_COLUMNS` tuple only — no function bodies yet:

```python
"""Trend feature compute for the strategy-discovery rebuild (Phase 2).

Adds 7 trend columns to a 1h OHLCV DataFrame:
  - price_over_ema20 / 50 / 200 (close / EMA, scale-free ratio)
  - ret_1h_sign / ret_24h_sign / ret_7d_sign (numpy.sign of close-vs-past-close)
  - atr14_pct (Wilder ATR-14 divided by close)

Pure functions on pandas DataFrames. No I/O. No tokenomics. No labels.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

_TREND_COLUMNS: Tuple[str, ...] = (
    "price_over_ema20",
    "price_over_ema50",
    "price_over_ema200",
    "ret_1h_sign",
    "ret_24h_sign",
    "ret_7d_sign",
    "atr14_pct",
)
```

### Round 1 — `test_ema20_matches_pandas_ewm`

- [ ] **Step 1.1.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_features.py`:

```python
"""Tests for tools.strategy_discovery.features (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.features import (
    _TREND_COLUMNS,
    add_trend_features,
    first_valid_index,
)


def _synthetic_ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, size=n).cumsum()
    high = close + np.abs(rng.normal(0.0, 0.5, size=n))
    low = close - np.abs(rng.normal(0.0, 0.5, size=n))
    open_ = close + rng.normal(0.0, 0.2, size=n)
    ts = np.arange(n, dtype="int64") * 3_600_000
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close})


def test_ema20_matches_pandas_ewm():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    expected = df["close"] / df["close"].ewm(span=20, adjust=False).mean()
    np.testing.assert_allclose(
        out["price_over_ema20"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )
```

- [ ] **Step 1.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_ema20_matches_pandas_ewm -v
```

Expected: `ImportError` (cannot import `add_trend_features`) — function not yet implemented.

- [ ] **Step 1.1.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/features.py`:

```python
def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def add_trend_features(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Add the 7 trend feature columns to a 1h OHLCV DataFrame.

    Requires columns: ts, open, high, low, close. Returns a copy with 7 added
    columns. Rows inside the warm-up region (< 200 bars from the start) will
    have NaN in EMA200-dependent ratios; use first_valid_index() to skip them.
    """
    out = df_ohlcv.copy()
    close = out["close"]
    out["price_over_ema20"] = close / _ema(close, 20)
    return out


def first_valid_index(df: pd.DataFrame, min_warmup: int = 200) -> int:
    """First row index where all trend feature columns are finite."""
    cols = [c for c in _TREND_COLUMNS if c in df.columns]
    if not cols:
        return min(min_warmup, len(df))
    finite = df[cols].notna().all(axis=1) & np.isfinite(df[cols]).all(axis=1)
    n = len(df)
    for i in range(min_warmup, n):
        if bool(finite.iloc[i]):
            return i
    return n
```

- [ ] **Step 1.1.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_ema20_matches_pandas_ewm -v
```

Expected: `1 passed`.

- [ ] **Step 1.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/features.py backend/tests/tools/strategy_discovery/__init__.py backend/tests/tools/strategy_discovery/test_features.py
git commit -m "$(cat <<'EOF'
feat(phase2): add features.py scaffold + EMA20 ratio

Phase 2 strategy-discovery rebuild — first cut of trend features.
Implements add_trend_features() with price/EMA20 ratio only; remaining
6 trend columns follow in subsequent commits.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 2 — `test_ema_warmup_returns_finite_after_n_bars`

- [ ] **Step 1.2.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_features.py`:

```python
def test_ema_warmup_returns_finite_after_n_bars():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    # All three EMA ratios must be finite for every row at or after index 200.
    for col in ("price_over_ema20", "price_over_ema50", "price_over_ema200"):
        tail = out[col].iloc[200:]
        assert np.isfinite(tail.to_numpy()).all(), f"non-finite values in {col} after warmup"
```

- [ ] **Step 1.2.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_ema_warmup_returns_finite_after_n_bars -v
```

Expected: `KeyError: 'price_over_ema50'` — only EMA20 ratio exists so far.

- [ ] **Step 1.2.3: Implement the minimal code**

Update `add_trend_features` in `backend/tools/strategy_discovery/features.py` to add the EMA50 and EMA200 ratios. Replace the function body (everything from `out = df_ohlcv.copy()` to `return out`) with:

```python
    out = df_ohlcv.copy()
    close = out["close"]
    out["price_over_ema20"]  = close / _ema(close, 20)
    out["price_over_ema50"]  = close / _ema(close, 50)
    out["price_over_ema200"] = close / _ema(close, 200)
    return out
```

- [ ] **Step 1.2.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py -v
```

Expected: `2 passed`.

- [ ] **Step 1.2.5: Commit**

```bash
git add backend/tools/strategy_discovery/features.py backend/tests/tools/strategy_discovery/test_features.py
git commit -m "$(cat <<'EOF'
feat(phase2): add EMA50/EMA200 ratios + warmup pin test

Round 2 of features.py — extends add_trend_features with EMA50 + EMA200
ratios. Pin test asserts all three ratios are finite at row index >= 200.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 3 — `test_atr14_wilder_smoothing_formula`

- [ ] **Step 1.3.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_features.py`:

```python
def test_atr14_wilder_smoothing_formula():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    # Recompute Wilder ATR-14 from first principles and compare.
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing = ewm with alpha = 1/14, adjust=False.
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    expected = atr / df["close"]
    np.testing.assert_allclose(
        out["atr14_pct"].iloc[14:].to_numpy(),
        expected.iloc[14:].to_numpy(),
        rtol=1e-9,
    )
```

- [ ] **Step 1.3.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_atr14_wilder_smoothing_formula -v
```

Expected: `KeyError: 'atr14_pct'`.

- [ ] **Step 1.3.3: Implement the minimal code**

Append helper + extend `add_trend_features` in `backend/tools/strategy_discovery/features.py`. After `_ema` and before `add_trend_features`, insert:

```python
def _wilder_atr14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder ATR-14.

    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|).
    Wilder smoothing is equivalent to ewm(alpha=1/14, adjust=False).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
```

Then append two lines inside `add_trend_features`, immediately before `return out`:

```python
    out["atr14_pct"] = _wilder_atr14(out["high"], out["low"], close) / close
```

- [ ] **Step 1.3.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py -v
```

Expected: `3 passed`.

- [ ] **Step 1.3.5: Commit**

```bash
git add backend/tools/strategy_discovery/features.py backend/tests/tools/strategy_discovery/test_features.py
git commit -m "$(cat <<'EOF'
feat(phase2): add Wilder ATR-14 percentage feature

Round 3 of features.py — adds atr14_pct column using ewm(alpha=1/14)
equivalent to Wilder smoothing. Pin test recomputes ATR from first
principles.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 4 — `test_ret_sign_at_zero_returns_zero`

- [ ] **Step 1.4.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_features.py`:

```python
def test_ret_sign_at_zero_returns_zero():
    # Construct a series where close_t == close_{t-1} == close_{t-24} == close_{t-168}.
    n = 300
    df = _synthetic_ohlcv(n)
    df["close"] = 100.0  # all closes identical → sign of any diff is 0
    df["high"]  = 100.5
    df["low"]   = 99.5
    df["open"]  = 100.0
    out = add_trend_features(df)
    tail = out.iloc[200:]   # post-warmup
    assert (tail["ret_1h_sign"]  == 0).all()
    assert (tail["ret_24h_sign"] == 0).all()
    assert (tail["ret_7d_sign"]  == 0).all()
```

- [ ] **Step 1.4.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_ret_sign_at_zero_returns_zero -v
```

Expected: `KeyError: 'ret_1h_sign'`.

- [ ] **Step 1.4.3: Implement the minimal code**

Insert three lines inside `add_trend_features` in `backend/tools/strategy_discovery/features.py`, immediately after the `price_over_ema200` line and before the `atr14_pct` line:

```python
    out["ret_1h_sign"]  = np.sign(close - close.shift(1)).astype("float64")
    out["ret_24h_sign"] = np.sign(close - close.shift(24)).astype("float64")
    out["ret_7d_sign"]  = np.sign(close - close.shift(168)).astype("float64")
```

- [ ] **Step 1.4.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py -v
```

Expected: `4 passed`.

- [ ] **Step 1.4.5: Commit**

```bash
git add backend/tools/strategy_discovery/features.py backend/tests/tools/strategy_discovery/test_features.py
git commit -m "$(cat <<'EOF'
feat(phase2): add ret_1h/24h/7d sign features

Round 4 of features.py — three sign-only return features at 1h, 24h
and 7d lags. Pin test asserts zero output on constant-price input.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 5 — `test_first_valid_index_at_200`

- [ ] **Step 1.5.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_features.py`:

```python
def test_first_valid_index_at_200():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    idx = first_valid_index(out, min_warmup=200)
    assert idx == 200, f"expected first_valid_index == 200, got {idx}"
    # And every row from idx onward has finite values in all trend cols.
    tail = out.iloc[idx:]
    for col in _TREND_COLUMNS:
        assert np.isfinite(tail[col].to_numpy()).all(), f"non-finite in {col}"
```

- [ ] **Step 1.5.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py::test_first_valid_index_at_200 -v
```

Expected: PASS (the scaffolded `first_valid_index` from Step 1.0b should already satisfy this once all 7 features exist). If it FAILS, debug `first_valid_index` — the most likely cause is an off-by-one with `min_warmup`.

- [ ] **Step 1.5.3: Implementation (only if Step 1.5.2 failed)**

If the test failed, inspect the failure. The expected behaviour: with `min_warmup=200`, `first_valid_index` should return the smallest `i >= min_warmup` such that all 7 trend columns are finite at row `i`. The scaffolded implementation iterates `range(min_warmup, n)` and returns `i` on the first row where `finite.iloc[i]` is True. If returning `> 200`, check whether an earlier feature column is producing NaN past row 200 — it should not.

- [ ] **Step 1.5.4: Run the full module test file to confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_features.py -v
```

Expected: `5 passed`.

- [ ] **Step 1.5.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_features.py
git commit -m "$(cat <<'EOF'
test(phase2): pin first_valid_index at 200-bar warmup boundary

Round 5 of features.py — locks in the warmup contract that downstream
modules rely on (build_phase2 uses first_valid_index to drop the EMA200
warmup region before stamping + labeling).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Tokenomic stamping (`tokenomic_stamp.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/tokenomic_stamp.py`
- Create: `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py`

**Scaffolding (write before any test):**

- [ ] **Step 2.0: Create skeleton `tokenomic_stamp.py`**

Create `backend/tools/strategy_discovery/tokenomic_stamp.py`:

```python
"""T+1 tokenomic stamping for the strategy-discovery rebuild (Phase 2).

Aligns a daily CoinPaprika snapshot (one row per UTC day at 00:00:00 UTC)
onto an hourly OHLCV grid using the T+1 causality rule from the spec:
the daily row dated UTC-day D applies only to candidate hours in
UTC-day D+1 and later.

Adds 6 tokenomic columns to the input DataFrame:
  market_cap, fdv, fdv_over_mc, circ_over_total, vol_24h, vol_over_mc

Pure functions. No I/O. Imports only stdlib + pandas + numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_DAY_MS = 86_400_000

_TOKENOMIC_COLUMNS: Tuple[str, ...] = (
    "market_cap",
    "fdv",
    "fdv_over_mc",
    "circ_over_total",
    "vol_24h",
    "vol_over_mc",
)


@dataclass(frozen=True)
class SupplySnapshot:
    """Point-in-time supply reading from CoinPaprika /v1/tickers/{cp_id}."""
    pid: str
    circulating: float
    total: float
    max_supply: Optional[float]
```

### Round 1 — `test_t_plus_1_boundary_uses_yesterday_snapshot`

- [ ] **Step 2.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py`:

```python
"""Tests for tools.strategy_discovery.tokenomic_stamp (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.tokenomic_stamp import (
    SupplySnapshot,
    _TOKENOMIC_COLUMNS,
    stamp_tokenomic,
)

_DAY_MS = 86_400_000


def _hourly_ts(start_day_ms: int, n_hours: int) -> np.ndarray:
    return start_day_ms + np.arange(n_hours, dtype="int64") * 3_600_000


def _trivial_supply(pid: str = "FOO-USD") -> SupplySnapshot:
    return SupplySnapshot(pid=pid, circulating=1_000_000.0, total=2_000_000.0, max_supply=None)


def test_t_plus_1_boundary_uses_yesterday_snapshot():
    # Day D = 1_000 * _DAY_MS, snapshot on that day reports MC=100, vol=10.
    # Hourly candidates start on Day D+1 — the FIRST candidate at D+1 00:00
    # must read Day D's snapshot (T+1 rule).
    d0 = 1_000 * _DAY_MS
    d1 = d0 + _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0, d1],
        "market_cap": [100.0, 200.0],
        "volume_24h": [10.0,  20.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d1, 48),                # Day D+1 00:00 .. Day D+2 23:00
        "close": np.full(48, 5.0, dtype="float64"),
    })
    out = stamp_tokenomic(df_hourly, df_daily, _trivial_supply(), drop_on_missing_volume=False)
    # Day D+1 00:00 must read Day D's snapshot (MC=100, vol=10).
    assert out.loc[0, "market_cap"] == pytest.approx(100.0)
    assert out.loc[0, "vol_24h"]    == pytest.approx(10.0)
    # Day D+2 00:00 (index 24) must read Day D+1's snapshot (MC=200, vol=20).
    assert out.loc[24, "market_cap"] == pytest.approx(200.0)
    assert out.loc[24, "vol_24h"]    == pytest.approx(20.0)
```

- [ ] **Step 2.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py::test_t_plus_1_boundary_uses_yesterday_snapshot -v
```

Expected: `ImportError` (`stamp_tokenomic` not yet implemented).

- [ ] **Step 2.1.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/tokenomic_stamp.py`:

```python
def _stamp_daily_to_hourly(
    df_hourly: pd.DataFrame,
    df_daily: pd.DataFrame,
    *,
    value_cols: Tuple[str, ...],
) -> pd.DataFrame:
    """Forward-merge daily values onto hourly rows with a +1 day shift.

    df_daily['ts'] is UTC-day midnight in ms. Each row applies to candidate
    hours starting (ts + 1 day) — the T+1 rule.
    """
    daily = df_daily[["ts", *value_cols]].copy()
    daily["ts"] = daily["ts"].astype("int64") + _DAY_MS
    daily = daily.sort_values("ts").reset_index(drop=True)
    hourly = df_hourly[["ts"]].sort_values("ts").reset_index(drop=True)
    return pd.merge_asof(hourly, daily, on="ts", direction="backward")


def stamp_tokenomic(
    df_hourly: pd.DataFrame,
    df_daily_marketcap: pd.DataFrame,
    supply_row: SupplySnapshot,
    drop_on_missing_volume: bool = True,
) -> pd.DataFrame:
    """Add 6 tokenomic columns to df_hourly via T+1 stamping.

    df_hourly must contain: ts (epoch ms), close.
    df_daily_marketcap must contain: ts (UTC-day midnight epoch ms),
                                     market_cap, volume_24h.
    Returns a copy with 6 added columns. When drop_on_missing_volume=True,
    rows whose stamped vol_24h is NaN are dropped (per spec missing-data
    policy: volume staleness is unsafe to forward-fill).
    """
    stamped = _stamp_daily_to_hourly(
        df_hourly[["ts"]],
        df_daily_marketcap.rename(columns={"volume_24h": "vol_24h"}),
        value_cols=("market_cap", "vol_24h"),
    )
    out = df_hourly.copy().sort_values("ts").reset_index(drop=True)
    out["market_cap"] = stamped["market_cap"].to_numpy()
    out["vol_24h"]    = stamped["vol_24h"].to_numpy()
    out["fdv"]        = out["close"].astype("float64") * float(supply_row.total)
    out["fdv_over_mc"]    = out["fdv"]     / np.maximum(out["market_cap"], 1e-12)
    out["circ_over_total"] = float(supply_row.circulating) / max(float(supply_row.total), 1e-12)
    out["vol_over_mc"]    = out["vol_24h"] / np.maximum(out["market_cap"], 1e-12)
    if drop_on_missing_volume:
        out = out[out["vol_24h"].notna()].reset_index(drop=True)
    return out
```

- [ ] **Step 2.1.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py -v
```

Expected: `1 passed`.

- [ ] **Step 2.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/tokenomic_stamp.py backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py
git commit -m "$(cat <<'EOF'
feat(phase2): add tokenomic_stamp with T+1 daily-to-hourly merge

Phase 2 strategy-discovery rebuild — implements stamp_tokenomic with the
T+1 causality rule: snapshot dated UTC-day D applies to candidate hours
in UTC-day D+1 onward. Pin test exercises the boundary at D+1 00:00 UTC.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 2 — `test_forward_fill_supplies_carry_indefinitely`

- [ ] **Step 2.2.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py`:

```python
def test_forward_fill_supplies_carry_indefinitely():
    # Daily MC reported only on Day D=1000. Hourly grid spans Days D+1..D+5
    # (i.e. 4 days * 24 h = 96 hourly rows after the T+1 boundary). MC must
    # forward-fill across the whole window — slow-moving features have no
    # time cap.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],
        "market_cap": [100.0],
        "volume_24h": [10.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 96),
        "close": np.full(96, 5.0, dtype="float64"),
    })
    out = stamp_tokenomic(df_hourly, df_daily, _trivial_supply(), drop_on_missing_volume=False)
    assert out["market_cap"].notna().all()
    assert (out["market_cap"] == pytest.approx(100.0)).all()
```

- [ ] **Step 2.2.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py::test_forward_fill_supplies_carry_indefinitely -v
```

Expected: PASS — `pd.merge_asof(..., direction="backward")` already carries the last known daily row forward across all later hourly rows. No code change required.

- [ ] **Step 2.2.3: (Skipped — test passed)**

If Step 2.2.2 FAILED instead, inspect the failure mode. Most likely cause: the daily frame had only one row and the hourly start is exactly at `d0 + _DAY_MS` — `merge_asof` should still find it. Confirm `daily["ts"]` was shifted by `+_DAY_MS` inside `_stamp_daily_to_hourly`.

- [ ] **Step 2.2.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py -v
```

Expected: `2 passed`.

- [ ] **Step 2.2.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py
git commit -m "$(cat <<'EOF'
test(phase2): pin indefinite ffill of slow-moving tokenomic features

Round 2 of tokenomic_stamp.py — locks in the spec's missing-data policy:
MC and supply ratios carry forward indefinitely from the last known daily
snapshot.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 3 — `test_missing_volume_drops_candidate_row`

- [ ] **Step 2.3.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py`:

```python
def test_missing_volume_drops_candidate_row():
    # Daily snapshot exists for Day D but not for Day D+1. Hourly rows on
    # Day D+1 (which read Day D's snapshot — vol present) survive; hourly
    # rows on Day D+2 (which read Day D+1's snapshot — vol missing) drop.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],          # only Day D — Day D+1 is missing
        "market_cap": [100.0],
        "volume_24h": [10.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 48),     # 24h on D+1 + 24h on D+2
        "close": np.full(48, 5.0, dtype="float64"),
    })
    # With drop_on_missing_volume=True, ALL 48 rows survive — because vol
    # IS present (forward-filled from Day D for all 48 hours via merge_asof).
    # So we need to construct the missing-vol case differently: leave a
    # genuine gap by passing a daily frame where vol_24h is NaN on Day D+1.
    df_daily_with_gap = pd.DataFrame({
        "ts":         [d0,    d0 + _DAY_MS],
        "market_cap": [100.0, 200.0],
        "volume_24h": [10.0,  float("nan")],
    })
    out = stamp_tokenomic(df_hourly, df_daily_with_gap, _trivial_supply(), drop_on_missing_volume=True)
    # Day D+1 rows (24h) keep vol=10. Day D+2 rows would read NaN — dropped.
    assert len(out) == 24, f"expected 24 surviving rows, got {len(out)}"
    assert (out["vol_24h"] == pytest.approx(10.0)).all()
```

- [ ] **Step 2.3.2: Run the test to verify it passes (or fails)**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py::test_missing_volume_drops_candidate_row -v
```

Expected: PASS — the `drop_on_missing_volume` branch in `stamp_tokenomic` already filters via `out["vol_24h"].notna()`. If it FAILS, inspect whether the row count matches; if extra rows survived, the bug is likely that `merge_asof` carried forward the NaN (it shouldn't — `merge_asof` keeps the NaN as-is from the source frame).

- [ ] **Step 2.3.3: (Skipped — test passed)**

- [ ] **Step 2.3.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py -v
```

Expected: `3 passed`.

- [ ] **Step 2.3.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py
git commit -m "$(cat <<'EOF'
test(phase2): pin volume-missing drop policy

Round 3 of tokenomic_stamp.py — locks the spec's drop-on-missing-volume
rule: candidate rows whose stamped vol_24h is NaN are removed from the
output (stale volume mis-signals the hot-float state).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 4 — `test_fdv_derived_from_price_and_total_supply`

- [ ] **Step 2.4.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py`:

```python
def test_fdv_derived_from_price_and_total_supply():
    # FDV = close_t * supply.total at each row, and fdv_over_mc = fdv / market_cap.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],
        "market_cap": [50_000.0],   # 50k market_cap
        "volume_24h": [1_000.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 3),
        "close": np.array([2.0, 4.0, 8.0]),
    })
    supply = SupplySnapshot(pid="FOO-USD", circulating=10_000.0, total=25_000.0, max_supply=None)
    out = stamp_tokenomic(df_hourly, df_daily, supply, drop_on_missing_volume=False)
    np.testing.assert_allclose(out["fdv"].to_numpy(),
                                np.array([2.0, 4.0, 8.0]) * 25_000.0)
    np.testing.assert_allclose(out["fdv_over_mc"].to_numpy(),
                                (np.array([2.0, 4.0, 8.0]) * 25_000.0) / 50_000.0)
    # circ_over_total is a constant row-derived value: 10_000 / 25_000 = 0.4
    np.testing.assert_allclose(out["circ_over_total"].to_numpy(),
                                np.full(3, 0.4))
    np.testing.assert_allclose(out["vol_over_mc"].to_numpy(),
                                np.full(3, 1_000.0 / 50_000.0))
```

- [ ] **Step 2.4.2: Run the test to verify it passes (or fails)**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py::test_fdv_derived_from_price_and_total_supply -v
```

Expected: PASS — these are derived columns the implementation already computes. If it FAILS, double-check column ordering inside `stamp_tokenomic` — `fdv` must be set before `fdv_over_mc` references it, and `market_cap` must be set before either.

- [ ] **Step 2.4.3: (Skipped — test passed)**

- [ ] **Step 2.4.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_tokenomic_stamp.py -v
```

Expected: `4 passed`.

- [ ] **Step 2.4.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_tokenomic_stamp.py
git commit -m "$(cat <<'EOF'
test(phase2): pin FDV / circ_over_total / vol_over_mc formulas

Round 4 of tokenomic_stamp.py — locks the four derived columns' formulas
(FDV from live close × total supply, ratios as defined in the spec).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Dynamic-exit labels (`labels.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/labels.py`
- Create: `backend/tests/tools/strategy_discovery/test_labels.py`

**Scaffolding (write before any test):**

- [ ] **Step 3.0: Create skeleton `labels.py`**

Create `backend/tools/strategy_discovery/labels.py`:

```python
"""Dynamic-exit PnL labeling for the strategy-discovery rebuild (Phase 2).

For each candidate row (pid, t) and each horizon h, simulate the deployed
exit policy and report net PnL fraction after fees:

  - stop-loss at 8% drawdown vs entry (priority)
  - trail-stop at max(atr14_pct_t, 6%) drawdown vs running peak
  - max-hold cap at 168 bars (7d)
  - 1.2% retail round-trip fee subtracted from gross PnL

Mirrors agents/exit_watcher.on_price_tick and cnn_agent._check_risk_exits.
Pure functions. No I/O.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_DEFAULT_HORIZONS: Tuple[int, ...] = (1, 4, 24, 72, 168)
_DEFAULT_STOP_LOSS_PCT  = 0.08
_DEFAULT_ATR_TRAIL_FLOOR = 0.06
_DEFAULT_MAX_HOLD_BARS  = 168
_DEFAULT_ROUND_TRIP_FEE = 0.012
```

### Round 1 — `test_stop_loss_fires_at_8pct_drawdown`

- [ ] **Step 3.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
"""Tests for tools.strategy_discovery.labels (Phase 2)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.labels import (
    _DEFAULT_ATR_TRAIL_FLOOR,
    _DEFAULT_ROUND_TRIP_FEE,
    _DEFAULT_STOP_LOSS_PCT,
    simulate_dynamic_exit_labels,
)


def _frame(closes, highs=None, lows=None, atrs=None):
    n = len(closes)
    if highs is None:
        highs = list(closes)
    if lows is None:
        lows = list(closes)
    if atrs is None:
        atrs = [0.02] * n
    return pd.DataFrame({
        "ts":         np.arange(n, dtype="int64") * 3_600_000,
        "open":       np.array(closes, dtype="float64"),
        "high":       np.array(highs,  dtype="float64"),
        "low":        np.array(lows,   dtype="float64"),
        "close":      np.array(closes, dtype="float64"),
        "atr14_pct":  np.array(atrs,   dtype="float64"),
    })


def test_stop_loss_fires_at_8pct_drawdown():
    # Entry at close=100; bar 1 dips to low=91 (-9% from entry, beats SL=-8%).
    # Expected exit price = 100 * (1 - 0.08) = 92; label = (92/100 - 1) - 0.012 = -0.092.
    df = _frame(
        closes=[100.0, 95.0, 95.0, 95.0],
        highs=[100.0, 95.0, 95.0, 95.0],
        lows=[100.0, 91.0, 95.0, 95.0],
        atrs=[0.02, 0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[3])
    assert out.loc[0, "label_h3"] == pytest.approx(-0.092, abs=1e-9)
```

- [ ] **Step 3.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_stop_loss_fires_at_8pct_drawdown -v
```

Expected: `ImportError` — `simulate_dynamic_exit_labels` not implemented.

- [ ] **Step 3.1.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/labels.py`:

```python
def _simulate_one(
    entry_idx: int,
    horizon: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_pcts: np.ndarray,
    stop_loss_pct: float,
    atr_trail_floor: float,
    max_hold_bars: int,
    round_trip_fee: float,
) -> float:
    """Simulate one (entry, horizon) trade. Returns net PnL fraction or NaN."""
    n = len(closes)
    entry_price = float(closes[entry_idx])
    horizon_cap = min(horizon, max_hold_bars)
    last_idx = entry_idx + horizon_cap
    if last_idx >= n:
        return float("nan")
    peak = entry_price
    for s in range(1, horizon_cap + 1):
        i = entry_idx + s
        bar_low  = float(lows[i])
        bar_high = float(highs[i])
        # 1. Stop-loss check (priority — matches cnn_agent._check_risk_exits)
        if bar_low / entry_price - 1.0 <= -stop_loss_pct:
            exit_price = entry_price * (1.0 - stop_loss_pct)
            return (exit_price / entry_price - 1.0) - round_trip_fee
        # 2. Trail-stop check (ATR-based with floor)
        if bar_high > peak:
            peak = bar_high
        atr_now = float(atr_pcts[i])
        if not np.isfinite(atr_now):
            atr_now = atr_trail_floor
        atr_pct = max(atr_now, atr_trail_floor)
        if bar_low / peak - 1.0 <= -atr_pct:
            exit_price = peak * (1.0 - atr_pct)
            return (exit_price / entry_price - 1.0) - round_trip_fee
    # 3. Horizon reached without trigger — exit at last bar's close
    exit_price = float(closes[last_idx])
    return (exit_price / entry_price - 1.0) - round_trip_fee


def simulate_dynamic_exit_labels(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    stop_loss_pct: float = _DEFAULT_STOP_LOSS_PCT,
    atr_trail_floor: float = _DEFAULT_ATR_TRAIL_FLOOR,
    max_hold_bars: int = _DEFAULT_MAX_HOLD_BARS,
    round_trip_fee: float = _DEFAULT_ROUND_TRIP_FEE,
) -> pd.DataFrame:
    """Add label_h{h} columns per horizon by simulating the deployed exit policy.

    Requires columns: ts, open, high, low, close, atr14_pct. NaN labels when
    horizon would extend past the end of df.
    """
    horizons_list = list(horizons) if horizons is not None else list(_DEFAULT_HORIZONS)
    out = df.copy()
    closes   = out["close"].to_numpy(dtype="float64")
    highs    = out["high"].to_numpy(dtype="float64")
    lows     = out["low"].to_numpy(dtype="float64")
    atr_pcts = out["atr14_pct"].to_numpy(dtype="float64")
    n = len(out)
    for h in horizons_list:
        col = np.empty(n, dtype="float64")
        for i in range(n):
            col[i] = _simulate_one(
                i, h, closes, highs, lows, atr_pcts,
                stop_loss_pct, atr_trail_floor, max_hold_bars, round_trip_fee,
            )
        out[f"label_h{h}"] = col
    return out
```

- [ ] **Step 3.1.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `1 passed`.

- [ ] **Step 3.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/labels.py backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
feat(phase2): add dynamic-exit label simulator with SL trigger

Phase 2 strategy-discovery rebuild — first cut of labels.py. Simulates
the deployed exit policy at each (entry, horizon) and reports net PnL
fraction. Stop-loss (-8% drawdown) triggers conservative exit at SL
boundary price (matches WS exit checker fill semantics).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 2 — `test_trail_stop_fires_at_atr_floor`

- [ ] **Step 3.2.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_trail_stop_fires_at_atr_floor():
    # Entry at close=100. Bar 1: high=110 (new peak). Bar 2: low=103.4
    # → drawdown from peak = 103.4/110 - 1 = -0.06 (exactly the 6% floor).
    # ATR provided is 0.03 (below floor) → effective trail = 6% floor.
    # Trail exit price = peak * (1 - 0.06) = 110 * 0.94 = 103.4.
    # Net label = (103.4/100 - 1) - 0.012 = 0.034 - 0.012 = 0.022.
    df = _frame(
        closes=[100.0, 110.0, 103.4, 103.4],
        highs=[100.0, 110.0, 103.4, 103.4],
        lows=[100.0, 100.0, 103.4, 103.4],
        atrs=[0.03, 0.03, 0.03, 0.03],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[3])
    assert out.loc[0, "label_h3"] == pytest.approx(0.022, abs=1e-9)
```

- [ ] **Step 3.2.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_trail_stop_fires_at_atr_floor -v
```

Expected: PASS — the implementation from Round 1 already handles trail-stop. If it FAILS, inspect: most likely the trail condition (`bar_low / peak - 1.0 <= -atr_pct`) is using strict `<` instead of `<=`. The spec uses `<=`.

- [ ] **Step 3.2.3: (Skipped — test passed)**

- [ ] **Step 3.2.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `2 passed`.

- [ ] **Step 3.2.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin trail-stop at ATR 6% floor

Round 2 of labels.py — locks the spec's ATR trail-stop with 6% floor
(matches deployed _CNN_ATR_TRAIL_MIN).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 3 — `test_max_hold_cap_at_168_for_h168`

- [ ] **Step 3.3.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_max_hold_cap_at_168_for_h168():
    # Construct a 200-bar series where price drifts up linearly (no SL, no
    # trail triggers given small ATR). horizon=168 should exit at index 168
    # post-entry, NOT at any later bar.
    n = 200
    closes = list(np.linspace(100.0, 200.0, n))
    df = _frame(
        closes=closes,
        highs=[c + 0.01 for c in closes],          # tiny range → trail never fires
        lows= [c - 0.01 for c in closes],
        atrs=[0.0001] * n,                          # ATR floor (6%) governs; never fires
    )
    out = simulate_dynamic_exit_labels(df, horizons=[168])
    # Entry at index 0, horizon_cap = min(168, 168) = 168, exit at index 168.
    entry_close = closes[0]
    exit_close  = closes[168]
    expected = (exit_close / entry_close - 1.0) - _DEFAULT_ROUND_TRIP_FEE
    assert out.loc[0, "label_h168"] == pytest.approx(expected, abs=1e-9)
```

- [ ] **Step 3.3.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_max_hold_cap_at_168_for_h168 -v
```

Expected: PASS — implementation already caps `horizon_cap = min(horizon, max_hold_bars)`. If it FAILS, inspect the `last_idx` calculation; it should be `entry_idx + horizon_cap` exactly (not `+ horizon_cap - 1`).

- [ ] **Step 3.3.3: (Skipped — test passed)**

- [ ] **Step 3.3.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `3 passed`.

- [ ] **Step 3.3.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin max-hold cap at 168 bars

Round 3 of labels.py — locks horizon_cap = min(horizon, 168) so h=168
uses the live system's 7-day max-hold even when h is passed verbatim.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 4 — `test_horizon_reached_without_trigger_uses_close`

- [ ] **Step 3.4.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_horizon_reached_without_trigger_uses_close():
    # Entry at close=100. Bars 1..4 stay flat. horizon=4 reaches without
    # SL or trail triggers — exit at closes[4] = 100.
    # Net label = (100/100 - 1) - 0.012 = -0.012.
    df = _frame(
        closes=[100.0] * 6,
        highs=[100.0]  * 6,
        lows= [100.0]  * 6,
        atrs= [0.02]   * 6,
    )
    out = simulate_dynamic_exit_labels(df, horizons=[4])
    assert out.loc[0, "label_h4"] == pytest.approx(-0.012, abs=1e-9)
```

- [ ] **Step 3.4.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_horizon_reached_without_trigger_uses_close -v
```

Expected: PASS.

- [ ] **Step 3.4.3: (Skipped — test passed)**

- [ ] **Step 3.4.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `4 passed`.

- [ ] **Step 3.4.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin horizon-reached exit at close

Round 4 of labels.py — when neither SL nor trail trigger inside the
horizon window, exit at close[entry+horizon_cap].

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 5 — `test_stop_loss_priority_over_trail`

- [ ] **Step 3.5.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_stop_loss_priority_over_trail():
    # Entry at close=100. Bar 1: high=120 (new peak), low=91.
    # - SL trigger: low/entry - 1 = 91/100 - 1 = -0.09 <= -0.08 → SL fires.
    # - Trail trigger: low/peak - 1 = 91/120 - 1 = -0.2417 <= -max(atr, 0.06)=-0.06 → also fires.
    # Both trigger in the same bar — SL must win.
    # SL exit price = 100 * 0.92 = 92; label = -0.08 - 0.012 = -0.092.
    df = _frame(
        closes=[100.0, 95.0, 95.0],
        highs=[100.0, 120.0, 95.0],
        lows=[100.0, 91.0, 95.0],
        atrs=[0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[2])
    assert out.loc[0, "label_h2"] == pytest.approx(-0.092, abs=1e-9)
```

- [ ] **Step 3.5.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_stop_loss_priority_over_trail -v
```

Expected: PASS — the implementation checks SL before updating peak / checking trail, so SL wins when both would fire in the same bar. If FAILS, the bug is the order of the two checks inside `_simulate_one`.

- [ ] **Step 3.5.3: (Skipped — test passed)**

- [ ] **Step 3.5.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `5 passed`.

- [ ] **Step 3.5.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin SL priority over trail-stop in same bar

Round 5 of labels.py — locks the spec's invariant that stop-loss wins
when both exit triggers fire on the same bar (matches the live exit
checker ordering in agents/exit_watcher.on_price_tick).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 6 — `test_fee_subtracted_from_label`

- [ ] **Step 3.6.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_fee_subtracted_from_label():
    # Entry at 100, flat 5 bars → exit at close[5] = 100. Default fee = 0.012.
    df = _frame(
        closes=[100.0] * 6,
        highs=[100.0]  * 6,
        lows= [100.0]  * 6,
        atrs= [0.02]   * 6,
    )
    # With default fee
    out_default = simulate_dynamic_exit_labels(df, horizons=[5])
    assert out_default.loc[0, "label_h5"] == pytest.approx(-_DEFAULT_ROUND_TRIP_FEE, abs=1e-9)
    # With zero fee — gross PnL should be zero
    out_zero = simulate_dynamic_exit_labels(df, horizons=[5], round_trip_fee=0.0)
    assert out_zero.loc[0, "label_h5"] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 3.6.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_fee_subtracted_from_label -v
```

Expected: PASS.

- [ ] **Step 3.6.3: (Skipped — test passed)**

- [ ] **Step 3.6.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `6 passed`.

- [ ] **Step 3.6.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin 1.2% round-trip fee subtraction

Round 6 of labels.py — locks the spec's net-label = gross - fee
convention. Verifies the parameter override path too.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 7 — `test_insufficient_forward_bars_returns_nan`

- [ ] **Step 3.7.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_labels.py`:

```python
def test_insufficient_forward_bars_returns_nan():
    # Only 3 rows total; horizon=5 requires 5 forward bars → entry at index 0
    # has only 2 forward bars → NaN label. Horizons that fit (h=2) must NOT be NaN.
    df = _frame(
        closes=[100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0],
        lows= [100.0, 100.0, 100.0],
        atrs= [0.02, 0.02, 0.02],
    )
    out = simulate_dynamic_exit_labels(df, horizons=[2, 5])
    # h=2 fits at index 0 (entry+2 = 2 is in-bounds)
    assert not math.isnan(out.loc[0, "label_h2"])
    # h=5 does NOT fit (entry+5 = 5 out of bounds)
    assert math.isnan(out.loc[0, "label_h5"])
    # h=2 does NOT fit at index 2 (entry+2 = 4 out of bounds) — NaN expected
    assert math.isnan(out.loc[2, "label_h2"])
```

- [ ] **Step 3.7.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py::test_insufficient_forward_bars_returns_nan -v
```

Expected: PASS — `_simulate_one` returns NaN when `last_idx >= n`. If FAILS, inspect the boundary: `last_idx = entry_idx + horizon_cap`; with `n = 3` and `entry_idx = 0`, `horizon_cap = 5`, `last_idx = 5 >= 3` → NaN. With `entry_idx = 0`, `horizon_cap = 2`, `last_idx = 2 < 3` → simulates.

- [ ] **Step 3.7.3: (Skipped — test passed)**

- [ ] **Step 3.7.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_labels.py -v
```

Expected: `7 passed`.

- [ ] **Step 3.7.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_labels.py
git commit -m "$(cat <<'EOF'
test(phase2): pin NaN label when forward bars are insufficient

Round 7 of labels.py — locks the spec's edge case: when t + horizon_cap
exceeds the available history, the label is NaN (not silently truncated).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Orchestrator (`build_phase2.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/build_phase2.py`
- Create: `backend/tests/tools/strategy_discovery/test_build_phase2.py`

**Universe JSON shape (Phase 1 convention):** the curated universe is
`{cohort: [pid, ...]}`, e.g. `{"large": ["BTC-USD", ...], "mid": [...],
"high_fdv_ratio": [...], "low_turnover": [...]}`. See
`tools/strategy_discovery/build_universe_marketcap.py:universe_pids_from_curation`
for the canonical flattener.

**Input parquet schemas:**
- History parquet (`backend/data/history/{pid}.parquet`): column `start` is epoch seconds; OHLCV columns are `open, high, low, close, volume`.
- Marketcap parquet (`backend/data/marketcap/{pid}.parquet`): column `start` is epoch seconds (UTC-day midnight); value columns are `market_cap, volume_24h`.
- Supply snapshot parquet (`backend/data/supply/snapshot.parquet`): one row per pid with `pid, circulating, total, max_supply, ingest_ts, schema_version` (see `build_supply_snapshot._SCHEMA`).

**Scaffolding (write before any test):**

- [ ] **Step 4.0: Create skeleton `build_phase2.py`**

Create `backend/tools/strategy_discovery/build_phase2.py`:

```python
"""Phase 2 orchestrator — universe → per-token feature+label parquet.

Loads inputs (curated universe JSON, per-pid hourly OHLCV parquet, per-pid
daily CoinPaprika marketcap parquet, single supply-snapshot parquet),
applies features → tokenomic stamping → labels, and writes one parquet
per pid to backend/data/phase2/.

Run:
    cd backend && python -m tools.strategy_discovery.build_phase2 \\
        --universe ../docs/superpowers/specs/2026-05-23-universe-50.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.features import (  # noqa: E402
    _TREND_COLUMNS,
    add_trend_features,
    first_valid_index,
)
from tools.strategy_discovery.labels import (  # noqa: E402
    _DEFAULT_HORIZONS,
    simulate_dynamic_exit_labels,
)
from tools.strategy_discovery.tokenomic_stamp import (  # noqa: E402
    SupplySnapshot,
    _TOKENOMIC_COLUMNS,
    stamp_tokenomic,
)

_SCHEMA_VERSION = 1
_DEFAULT_HISTORY_DIR  = Path(BACKEND) / "data" / "history"
_DEFAULT_MARKETCAP_DIR = Path(BACKEND) / "data" / "marketcap"
_DEFAULT_SUPPLY_PATH  = Path(BACKEND) / "data" / "supply" / "snapshot.parquet"
_DEFAULT_OUTPUT_DIR   = Path(BACKEND) / "data" / "phase2"


@dataclass
class BuildResult:
    pid: str
    rows_written: int = 0
    rows_dropped_missing_volume: int = 0
    nan_label_counts: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
```

### Round 1 — `test_build_phase2_for_pid_writes_parquet`

- [ ] **Step 4.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_build_phase2.py`:

```python
"""Tests for tools.strategy_discovery.build_phase2 (Phase 2 orchestrator)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.build_phase2 import (
    BuildResult,
    build_phase2_for_pid,
    build_phase2_for_universe,
)

_HOUR_S = 3_600
_DAY_S  = 86_400


def _write_history_parquet(path: Path, n_hours: int = 400, start_day_s: int = 1_000 * _DAY_S):
    rng = np.random.default_rng(11)
    close = 100.0 + rng.normal(0.0, 0.5, size=n_hours).cumsum()
    df = pd.DataFrame({
        "start":  start_day_s + np.arange(n_hours, dtype="int64") * _HOUR_S,
        "open":   close,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": np.full(n_hours, 1_000.0),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_marketcap_parquet(path: Path, n_days: int = 20, start_day_s: int = 1_000 * _DAY_S):
    df = pd.DataFrame({
        "start":      start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
        "market_cap": np.full(n_days, 100_000.0),
        "volume_24h": np.full(n_days, 5_000.0),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_supply_snapshot(path: Path, pid: str = "FOO-USD"):
    schema = pa.schema([
        pa.field("pid",            pa.string()),
        pa.field("circulating",    pa.float64()),
        pa.field("total",          pa.float64()),
        pa.field("max_supply",     pa.float64()),
        pa.field("ingest_ts",      pa.int64()),
        pa.field("schema_version", pa.int32()),
    ])
    tbl = pa.table({
        "pid":            [pid],
        "circulating":    [1_000_000.0],
        "total":          [2_000_000.0],
        "max_supply":     [None],
        "ingest_ts":      [1_700_000_000],
        "schema_version": [1],
    }, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="snappy")


def test_build_phase2_for_pid_writes_parquet(tmp_path: Path):
    pid = "FOO-USD"
    history_dir   = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path   = tmp_path / "supply" / "snapshot.parquet"
    output_dir    = tmp_path / "phase2"
    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_marketcap_parquet(marketcap_dir / f"{pid}.parquet", n_days=20)
    _write_supply_snapshot(supply_path, pid=pid)

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)

    assert result.error is None, f"unexpected error: {result.error}"
    assert result.rows_written > 0
    assert (output_dir / f"{pid}.parquet").exists()

    out = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()
    # Must have all 13 features + 5 labels + identifiers + schema_version
    for col in ("ts", "pid",
                "market_cap", "fdv", "fdv_over_mc", "circ_over_total", "vol_24h", "vol_over_mc",
                "price_over_ema20", "price_over_ema50", "price_over_ema200",
                "ret_1h_sign", "ret_24h_sign", "ret_7d_sign", "atr14_pct",
                "label_h1", "label_h4", "label_h24", "label_h72", "label_h168",
                "schema_version"):
        assert col in out.columns, f"missing column {col}"
    assert (out["pid"] == pid).all()
    assert (out["schema_version"] == 1).all()
```

- [ ] **Step 4.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py::test_build_phase2_for_pid_writes_parquet -v
```

Expected: `ImportError` (`build_phase2_for_pid` not yet implemented).

- [ ] **Step 4.1.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/build_phase2.py`:

```python
def _load_supply_snapshot(supply_path: Path, pid: str) -> Optional[SupplySnapshot]:
    """Read one pid's row from the supply snapshot parquet. Returns None if absent."""
    if not supply_path.exists():
        return None
    tbl = pq.read_table(supply_path).to_pandas()
    row = tbl[tbl["pid"] == pid]
    if row.empty:
        return None
    r = row.iloc[0]
    max_supply = None if pd.isna(r["max_supply"]) else float(r["max_supply"])
    return SupplySnapshot(
        pid=pid,
        circulating=float(r["circulating"]),
        total=float(r["total"]),
        max_supply=max_supply,
    )


def _load_history_parquet(path: Path) -> pd.DataFrame:
    """Read history parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def _load_marketcap_parquet(path: Path) -> pd.DataFrame:
    """Read marketcap parquet, rename 'start' (epoch s) → 'ts' (epoch ms)."""
    df = pq.read_table(path).to_pandas()
    df = df.rename(columns={"start": "ts"})
    df["ts"] = df["ts"].astype("int64") * 1_000
    return df.sort_values("ts").reset_index(drop=True)


def build_phase2_for_pid(
    pid: str,
    history_dir: Path,
    marketcap_dir: Path,
    supply_path: Path,
    output_dir: Path,
) -> BuildResult:
    """End-to-end Phase 2 build for one pid. Writes output_dir/{pid}.parquet."""
    history_path   = Path(history_dir)   / f"{pid}.parquet"
    marketcap_path = Path(marketcap_dir) / f"{pid}.parquet"
    if not history_path.exists():
        return BuildResult(pid=pid, error=f"missing history: {history_path}")
    if not marketcap_path.exists():
        return BuildResult(pid=pid, error=f"missing marketcap: {marketcap_path}")
    supply = _load_supply_snapshot(Path(supply_path), pid)
    if supply is None:
        return BuildResult(pid=pid, error=f"missing supply: {pid}")

    df_hourly = _load_history_parquet(history_path)
    if len(df_hourly) < 200:
        return BuildResult(pid=pid, error=f"history too short ({len(df_hourly)} < 200 bars)")
    df_daily = _load_marketcap_parquet(marketcap_path)

    # features → drop warmup → stamp → labels
    df_feat = add_trend_features(df_hourly)
    cut = first_valid_index(df_feat, min_warmup=200)
    df_feat = df_feat.iloc[cut:].reset_index(drop=True)
    rows_pre_drop = len(df_feat)
    df_stamped = stamp_tokenomic(df_feat, df_daily, supply, drop_on_missing_volume=True)
    rows_dropped = rows_pre_drop - len(df_stamped)
    df_labeled = simulate_dynamic_exit_labels(df_stamped, horizons=list(_DEFAULT_HORIZONS))

    df_labeled["pid"] = pid
    df_labeled["schema_version"] = _SCHEMA_VERSION
    nan_counts = {
        f"label_h{h}": int(df_labeled[f"label_h{h}"].isna().sum())
        for h in _DEFAULT_HORIZONS
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{pid}.parquet"
    df_labeled.to_parquet(out_path, compression="snappy", index=False)
    return BuildResult(
        pid=pid,
        rows_written=len(df_labeled),
        rows_dropped_missing_volume=rows_dropped,
        nan_label_counts=nan_counts,
    )
```

- [ ] **Step 4.1.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py -v
```

Expected: `1 passed`.

- [ ] **Step 4.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/build_phase2.py backend/tests/tools/strategy_discovery/test_build_phase2.py
git commit -m "$(cat <<'EOF'
feat(phase2): add build_phase2_for_pid orchestrator

Phase 2 strategy-discovery rebuild — wires features.py + tokenomic_stamp.py
+ labels.py into a per-pid pipeline that writes a parquet to
backend/data/phase2/{pid}.parquet with all 13 features + 5 labels.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 2 — `test_build_phase2_for_universe_iterates_all_pids`

- [ ] **Step 4.2.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_build_phase2.py`:

```python
def test_build_phase2_for_universe_iterates_all_pids(tmp_path: Path):
    pids = ["FOO-USD", "BAR-USD", "BAZ-USD"]
    history_dir   = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path   = tmp_path / "supply" / "snapshot.parquet"
    output_dir    = tmp_path / "phase2"

    for p in pids:
        _write_history_parquet(history_dir / f"{p}.parquet", n_hours=400)
        _write_marketcap_parquet(marketcap_dir / f"{p}.parquet", n_days=20)
    # Universe JSON uses Phase 1 cohort layout: {cohort: [pids]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps({
        "large": ["FOO-USD"],
        "mid":   ["BAR-USD"],
        "high_fdv_ratio": ["BAZ-USD"],
        "low_turnover":   [],
    }), encoding="utf-8")

    # Single supply snapshot parquet for all three pids
    schema = pa.schema([
        pa.field("pid",            pa.string()),
        pa.field("circulating",    pa.float64()),
        pa.field("total",          pa.float64()),
        pa.field("max_supply",     pa.float64()),
        pa.field("ingest_ts",      pa.int64()),
        pa.field("schema_version", pa.int32()),
    ])
    tbl = pa.table({
        "pid":            pids,
        "circulating":    [1_000_000.0] * 3,
        "total":          [2_000_000.0] * 3,
        "max_supply":     [None] * 3,
        "ingest_ts":      [1_700_000_000] * 3,
        "schema_version": [1] * 3,
    }, schema=schema)
    supply_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, supply_path, compression="snappy")

    results = build_phase2_for_universe(
        universe_path,
        history_dir=history_dir,
        marketcap_dir=marketcap_dir,
        supply_path=supply_path,
        output_dir=output_dir,
    )
    assert len(results) == 3
    assert {r.pid for r in results} == set(pids)
    assert all(r.error is None for r in results)
    for p in pids:
        assert (output_dir / f"{p}.parquet").exists()
```

- [ ] **Step 4.2.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py::test_build_phase2_for_universe_iterates_all_pids -v
```

Expected: `ImportError` or `AttributeError` — `build_phase2_for_universe` not implemented.

- [ ] **Step 4.2.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/build_phase2.py`:

```python
def _pids_from_universe_json(universe_path: Path) -> List[str]:
    """Flatten {cohort: [pids]} into a deduplicated sorted pid list."""
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def build_phase2_for_universe(
    universe_path: Path,
    history_dir: Optional[Path] = None,
    marketcap_dir: Optional[Path] = None,
    supply_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> List[BuildResult]:
    """Iterate every pid in the universe JSON; collect per-pid BuildResults."""
    history_dir   = Path(history_dir)   if history_dir   else _DEFAULT_HISTORY_DIR
    marketcap_dir = Path(marketcap_dir) if marketcap_dir else _DEFAULT_MARKETCAP_DIR
    supply_path   = Path(supply_path)   if supply_path   else _DEFAULT_SUPPLY_PATH
    output_dir    = Path(output_dir)    if output_dir    else _DEFAULT_OUTPUT_DIR
    pids = _pids_from_universe_json(Path(universe_path))
    results: List[BuildResult] = []
    for pid in pids:
        results.append(build_phase2_for_pid(
            pid, history_dir, marketcap_dir, supply_path, output_dir,
        ))
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint — build Phase 2 parquet for an entire universe."""
    import argparse
    parser = argparse.ArgumentParser(description="Build Phase 2 features+labels for a universe.")
    parser.add_argument(
        "--universe",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--history-dir",   default=None)
    parser.add_argument("--marketcap-dir", default=None)
    parser.add_argument("--supply",        default=None)
    parser.add_argument("--output-dir",    default=None)
    args = parser.parse_args(argv)

    results = build_phase2_for_universe(
        Path(args.universe),
        history_dir   = Path(args.history_dir)   if args.history_dir   else None,
        marketcap_dir = Path(args.marketcap_dir) if args.marketcap_dir else None,
        supply_path   = Path(args.supply)        if args.supply        else None,
        output_dir    = Path(args.output_dir)    if args.output_dir    else None,
    )
    n_ok  = sum(1 for r in results if r.error is None)
    n_err = len(results) - n_ok
    print(f"  ok:    {n_ok}", flush=True)
    print(f"  error: {n_err}", flush=True)
    for r in results:
        if r.error:
            print(f"    [ERR] {r.pid}: {r.error}", flush=True)
        else:
            print(f"    {r.pid}: {r.rows_written:,} rows "
                  f"(dropped {r.rows_dropped_missing_volume:,} missing-vol)", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4.2.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py -v
```

Expected: `2 passed`.

- [ ] **Step 4.2.5: Commit**

```bash
git add backend/tools/strategy_discovery/build_phase2.py backend/tests/tools/strategy_discovery/test_build_phase2.py
git commit -m "$(cat <<'EOF'
feat(phase2): add universe orchestrator + CLI entrypoint

Round 2 of build_phase2.py — adds build_phase2_for_universe() that
iterates every pid in the curated universe JSON, plus a main() CLI that
operators can invoke once Phase 2 lands.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

### Round 3 — `test_build_result_reports_drop_counts`

- [ ] **Step 4.3.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_build_phase2.py`:

```python
def test_build_result_reports_drop_counts(tmp_path: Path):
    # Build marketcap parquet with a NaN volume_24h on day D+5 — should drop
    # 24 hourly rows from the output and report it in BuildResult.
    pid = "FOO-USD"
    history_dir   = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path   = tmp_path / "supply" / "snapshot.parquet"
    output_dir    = tmp_path / "phase2"

    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_supply_snapshot(supply_path, pid=pid)

    # Marketcap with a single NaN-volume day
    n_days = 20
    start_day_s = 1_000 * _DAY_S
    vols = np.full(n_days, 5_000.0)
    vols[5] = np.nan
    df = pd.DataFrame({
        "start":      start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
        "market_cap": np.full(n_days, 100_000.0),
        "volume_24h": vols,
    })
    marketcap_path = marketcap_dir / f"{pid}.parquet"
    marketcap_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), marketcap_path, compression="snappy")

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)
    assert result.error is None
    assert result.rows_dropped_missing_volume > 0
    # nan_label_counts should be a dict over all 5 horizons
    assert set(result.nan_label_counts.keys()) == {"label_h1", "label_h4", "label_h24", "label_h72", "label_h168"}
```

- [ ] **Step 4.3.2: Run the test to verify its outcome**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py::test_build_result_reports_drop_counts -v
```

Expected: PASS — `build_phase2_for_pid` already populates `rows_dropped_missing_volume` and `nan_label_counts`. If it FAILS, inspect the BuildResult fields and ensure the drop-count arithmetic happens *after* stamping (`rows_pre_drop - len(df_stamped)`).

- [ ] **Step 4.3.3: (Skipped — test passed)**

- [ ] **Step 4.3.4: Run the full module test file**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_build_phase2.py -v
```

Expected: `3 passed`.

- [ ] **Step 4.3.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_build_phase2.py
git commit -m "$(cat <<'EOF'
test(phase2): pin BuildResult drop/NaN telemetry fidelity

Round 3 of build_phase2.py — locks the operator-facing summary fields:
rows_dropped_missing_volume reflects the stamping drop count, and
nan_label_counts is populated for every horizon (zero counts included).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Full-suite green check + memory/CHANGELOG sync

This task does NOT add new code — it confirms the full backend test suite is green after Phase 2 and updates persistent memory + CHANGELOG per the sync rule (`feedback_sync_rule`).

**Files:**
- Modify: `CHANGELOG.md`
- Modify: `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`

- [ ] **Step 5.1: Run the full backend test suite once**

```bash
cd backend && python -m pytest tests/ -v --tb=short
```

Expected: all tests pass (970 pre-existing + 19 new = 989). If anything regresses, stop and investigate — Phase 2 modules import only pandas/numpy/pyarrow/stdlib, so a regression in unrelated tests would indicate a hook or environment issue, not a Phase 2 bug.

- [ ] **Step 5.2: Shell cleanup per CLAUDE.md**

```powershell
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

- [ ] **Step 5.3: Append CHANGELOG entry**

Open `C:\Users\gl450\polymarket_app\CHANGELOG.md` and prepend (or insert at the top of the most recent session block) a new entry:

```markdown
### Session — 2026-05-23 — Strategy-discovery Phase 2: feature compute + dynamic-exit labels

Implemented Phase 2 of the strategy-discovery rebuild per spec
`docs/superpowers/specs/2026-05-23-strategy-discovery-phase2-design.md`
and plan `docs/superpowers/plans/2026-05-23-strategy-discovery-phase2-implementation.md`.

**New modules (all under `backend/tools/strategy_discovery/`):**
- `features.py` — 7 trend features (EMA20/50/200 ratios, sign-only returns at 1h/24h/7d, Wilder ATR-14 percentage).
- `tokenomic_stamp.py` — T+1 daily-to-hourly merge for 6 tokenomic features (MC, FDV, FDV/MC, circ/total, vol_24h, vol/MC).
- `labels.py` — dynamic-exit simulator mirroring the deployed WS exit checker (SL 8%, ATR trail 6% floor, max-hold 168 bars, 1.2% net fee) across horizons {1, 4, 24, 72, 168}.
- `build_phase2.py` — orchestrator + CLI; writes one parquet per pid to `backend/data/phase2/{pid}.parquet`.

**Test surface added:** 19 new tests under `backend/tests/tools/strategy_discovery/`. Full suite green.

**Operator step (post-merge):**

```
cd backend && python -m tools.strategy_discovery.build_phase2 \
    --universe ../docs/superpowers/specs/2026-05-23-universe-50.json
```
```

- [ ] **Step 5.4: Update memory — `coinbase_trader_architecture.md`**

Open `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` and, inside the "Strategy-discovery rebuild" section that already documents Phase 1, append a new sub-section (do not duplicate the Phase 1 content):

```markdown

### Phase 2 (feature compute + dynamic-exit labels, 2026-05-23)

Phase 2 turns the curated 50-pid universe into per-token parquet files
with 13 features + 5 dynamic-exit horizon labels, ready for Phase 3 mining.

**New modules** (all in `backend/tools/strategy_discovery/`):
- `features.py` — `add_trend_features(df)` adds 7 trend cols
  (`price_over_ema20/50/200`, `ret_1h/24h/7d_sign`, `atr14_pct`).
  `first_valid_index(df, min_warmup=200)` is the warmup boundary helper.
- `tokenomic_stamp.py` — `stamp_tokenomic(df_hourly, df_daily, SupplySnapshot)`
  adds 6 tokenomic cols via T+1 merge. Drops rows with missing vol_24h by
  default; forward-fills slow-moving MC/supply ratios.
- `labels.py` — `simulate_dynamic_exit_labels(df, horizons=[1,4,24,72,168])`
  mirrors the deployed WS exit checker (SL 8% / ATR trail 6% floor /
  max-hold 168 bars / 1.2% fee).
- `build_phase2.py` — orchestrator + CLI; per-pid output at
  `backend/data/phase2/{pid}.parquet`.

**Output schema:** `ts, pid, market_cap, fdv, fdv_over_mc, circ_over_total,
vol_24h, vol_over_mc, price_over_ema20, price_over_ema50, price_over_ema200,
ret_1h_sign, ret_24h_sign, ret_7d_sign, atr14_pct, label_h1, label_h4,
label_h24, label_h72, label_h168, schema_version`.

**Tests:** 19 tests under `backend/tests/tools/strategy_discovery/test_{features,tokenomic_stamp,labels,build_phase2}.py`. All mock-only — no API, no DB, no live file I/O outside `tmp_path`.

**Operator runs** (post-merge):
```
cd backend && python -m tools.strategy_discovery.build_phase2 \
    --universe ../docs/superpowers/specs/2026-05-23-universe-50.json
```

**Status:** code-complete (Phase 2 plan = `2026-05-23-strategy-discovery-phase2-implementation.md`). Phase 3 (mining algorithm) is the next brainstorm round.

## See also
- [[xgb_post_scorecard_roadmap]] — operator picked bar-structure (off 4h+ time bars) before this rebuild round
- 2026-05-23-strategy-discovery-rebuild-brainstorm.md (spec)
- 2026-05-23-strategy-discovery-phase1-data-foundation.md (plan, complete)
- 2026-05-23-strategy-discovery-phase2-design.md (spec, this round)
- 2026-05-23-strategy-discovery-phase2-implementation.md (plan, this round)
```

- [ ] **Step 5.5: Commit memory + CHANGELOG**

```bash
git add CHANGELOG.md
git commit -m "$(cat <<'EOF'
docs: changelog entry for strategy-discovery Phase 2

Records the 4-module Phase 2 implementation + 19-test surface added in
this session. Memory file coinbase_trader_architecture.md updated
out-of-tree per the sync rule.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

The memory file lives outside the repo (`~/.claude/projects/...`) and is not git-tracked — Step 5.4's edit is persistent state, not a commit.

- [ ] **Step 5.6: Push all Phase 2 commits**

```bash
git push origin main
```

Expected: ~20 commits pushed (5 from features.py + 4 from tokenomic_stamp.py + 7 from labels.py + 3 from build_phase2.py + 1 changelog).

If push fails because parallel work landed on main, pull --rebase and retry (no Phase 2 commit touches files outside `backend/tools/strategy_discovery/`, `backend/tests/tools/strategy_discovery/`, or `CHANGELOG.md`, so conflicts should be rare).

---

## Self-Review

**1. Spec coverage** — every section of the spec maps to a task:
- Goal + Inputs + Output schema → Task 4 (build_phase2 writes the per-token parquet with the exact column set).
- Causality Contract (T+1 rule) → Task 2 Round 1 (`test_t_plus_1_boundary_uses_yesterday_snapshot`).
- Feature Definitions (Tokenomic 6) → Task 2 Round 4 (`test_fdv_derived_from_price_and_total_supply`).
- Feature Definitions (Trend 7) → Task 1 Rounds 1-4 (EMA20 ratio, EMA50/200 warmup, ATR formula, ret signs).
- Label Definitions (dynamic-exit simulation) → Task 3 Rounds 1-7 (SL, trail, max-hold, horizon-default, SL priority, fee, NaN edge).
- Missing-Data Policy → Task 2 Rounds 2 + 3 (ffill slow features, drop on missing volume).
- Module Structure (4 modules + interfaces) → Tasks 1-4 directly.
- Testing (19 tests listed in spec) → 5 + 4 + 7 + 3 = 19 tests in Tasks 1-4.
- Operator Integration → Task 4 Round 2 implements the CLI; Task 5 captures the operator command in CHANGELOG + memory.

**2. Placeholder scan** — every code step shows complete code blocks. No "TBD", "fill in", "add error handling", or "similar to Task N" references.

**3. Type consistency**:
- `SupplySnapshot` dataclass — defined in Task 2 Step 2.0; consumed in Tasks 2.x and Task 4 (import in build_phase2 scaffolding).
- `_TREND_COLUMNS`, `_TOKENOMIC_COLUMNS`, `_DEFAULT_HORIZONS` — declared in their respective module scaffolds (Steps 1.0b, 2.0, 3.0); imported by build_phase2 (Step 4.0).
- `BuildResult` — declared in Task 4 scaffolding; used in Task 4 Rounds 1-3.
- Function signatures match the spec's "Public API" section verbatim.
- Column names in tests match column names in implementations (`ts`, `close`, `high`, `low`, `atr14_pct`, `label_h{h}`, etc.).

No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-23-strategy-discovery-phase2-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance, then code quality) between each. Each of the 4 implementation tasks is independently scoped; Task 5 (memory/CHANGELOG sync) runs after all four land.

**2. Inline Execution** — execute tasks in this session, batching checkpoints after each task. Cheaper context-wise but no fresh-context review per task.

Which approach?
