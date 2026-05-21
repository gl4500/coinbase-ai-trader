# Dollar-Bar Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Backfill 1-minute candles for the scorecard's top-20 products and construct per-product dollar bars, materialized as parquet, for Sub-project 2 to consume.

**Architecture:** Two stages. Stage 1 extends the existing `services/history_backfill.py` (whose `_backfill_to_path` / `_fetch_range` are already granularity-agnostic — they do 1h and 5m today) with 1-minute support, driven by a thin operator CLI. Stage 2 is a new `tools/build_dollar_bars.py` that reads the 1m parquets, calibrates a per-product dollar threshold from the matching 1h parquet's bar count, walks the 1m candles into dollar bars, and writes them to `data/history/dollar/`.

**Tech Stack:** Python 3.11 (`.venv/Scripts/python.exe`), pyarrow, asyncio, httpx (Coinbase Advanced Trade API), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-20-dollar-bar-data-pipeline-design.md`.

**Conventions:**
- TDD red-green-refactor — write the failing test, run to confirm RED, implement to GREEN, commit.
- `.venv/Scripts/python.exe` for all test runs.
- Each commit includes the test file + the implementation file + a `CHANGELOG.md` sub-bullet, and is pushed immediately.
- The pre-commit hook runs the full regression suite when Python files are staged — commit only during a training/backend-idle window (`feedback_no_pytest_during_trading`).
- No live API calls in tests — the Coinbase fetch path is mocked.
- Pure functions with type hints, single responsibility.

---

## File Structure

**Modify:**
- `backend/services/history_backfill.py` — add 1-minute support: `_ONE_MINUTE_GRANULARITY`, `_ONE_MINUTE_BAR_SECS`, `_parquet_path_1m`, `load_1m_history`, `backfill_product_1m`. Mirrors the existing 5-minute functions.
- `backend/tests/test_history_backfill.py` — add tests for the 1-minute additions.
- `CHANGELOG.md` — one sub-bullet per task under a new Session section.

**Create:**
- `backend/tools/backfill_1m_candles.py` — operator CLI: resolve the top-20 products, compute each one's backfill depth from its 1h parquet, call `backfill_product_1m`.
- `backend/tests/test_backfill_1m_candles.py` — tests for the CLI's pure helpers.
- `backend/tools/build_dollar_bars.py` — Stage 2: dollar-value + threshold helpers, the construction core, the pure assembly function, and the I/O + CLI.
- `backend/tests/test_build_dollar_bars.py` — tests for Stage 2.

**Responsibilities:**
- `history_backfill.py` owns Coinbase candle fetching + parquet persistence at any granularity.
- `backfill_1m_candles.py` owns "which products, how deep" for the 1m pull — nothing else.
- `build_dollar_bars.py` owns the dollar-bar transform: calibration, construction, persistence.

---

## Task 1: 1-minute backfill support in history_backfill.py

**Files:**
- Modify: `backend/services/history_backfill.py`
- Test: `backend/tests/test_history_backfill.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_history_backfill.py`:

```python
import os
import pytest
from services import history_backfill as hb


def test_parquet_path_1m_uses_1m_subdir():
    p = hb._parquet_path_1m("BTC-USD")
    assert p.endswith(os.path.join("1m", "BTC-USD.parquet"))


def test_load_1m_history_missing_file_returns_empty():
    assert hb.load_1m_history("ZZZ-NONEXISTENT-USD") == []


@pytest.mark.asyncio
async def test_backfill_product_1m_delegates_with_one_minute_params(monkeypatch):
    captured = {}

    async def fake_backfill_to_path(pid, days, granularity, bar_secs, path):
        captured.update(pid=pid, days=days, granularity=granularity,
                        bar_secs=bar_secs, path=path)
        return {"product_id": pid, "new_bars": 0, "total_bars": 0, "oldest_ts": None}

    monkeypatch.setattr(hb, "_backfill_to_path", fake_backfill_to_path)
    await hb.backfill_product_1m("BTC-USD", days=3)
    assert captured["granularity"] == "ONE_MINUTE"
    assert captured["bar_secs"] == 60
    assert captured["days"] == 3
    assert captured["path"].endswith(os.path.join("1m", "BTC-USD.parquet"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_history_backfill.py -k "1m" -v`
Expected: FAIL with `AttributeError: module 'services.history_backfill' has no attribute '_parquet_path_1m'`.

- [ ] **Step 3: Add the 1-minute support**

In `backend/services/history_backfill.py`, after the `_FIVE_MINUTE_BAR_SECS` constant (line ~37) add:

```python
_ONE_MINUTE_GRANULARITY = "ONE_MINUTE"
_ONE_MINUTE_BAR_SECS    = 60           # seconds per 1-minute bar
```

After `_parquet_path_5m` (line ~62) add:

```python
def _parquet_path_1m(product_id: str) -> str:
    """1-minute candle parquet path — separate namespace under history/1m/."""
    safe = product_id.replace("/", "_")
    return os.path.join(_HISTORY_DIR, "1m", f"{safe}.parquet")
```

After `load_5m_history` (line ~158) add:

```python
def load_1m_history(product_id: str) -> List[Dict]:
    """Load all stored 1-minute candles for product_id. [] if no file."""
    return _load_from_path(_parquet_path_1m(product_id))
```

After `backfill_product_5m` (line ~282) add:

```python
async def backfill_product_1m(
    product_id: str,
    days: int = 7,
) -> Dict:
    """Backfill one product's 1-minute history. Same shape as hourly/5m.

    1m bars are 60s; at 300 bars/request that is 5h per request, so a long
    history is many paged requests — callers pass `days` explicitly.
    """
    return await _backfill_to_path(
        product_id, days, _ONE_MINUTE_GRANULARITY, _ONE_MINUTE_BAR_SECS,
        _parquet_path_1m(product_id),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_history_backfill.py -k "1m" -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/services/history_backfill.py backend/tests/test_history_backfill.py CHANGELOG.md
git commit -m "feat(history): 1-minute candle backfill support

Adds backfill_product_1m / load_1m_history / _parquet_path_1m, mirroring
the existing 5m functions. Reuses the granularity-agnostic _backfill_to_path.
First stage of the dollar-bar data pipeline (SP1).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

For the `CHANGELOG.md` edit in this commit, add this section directly under the top `---` divider:

```markdown
## Session 58.71n — Dollar-bar data pipeline (SP1) — 2026-05-20

**Spec:** `docs/superpowers/specs/2026-05-20-dollar-bar-data-pipeline-design.md`
**Plan:** `docs/superpowers/plans/2026-05-20-dollar-bar-data-pipeline.md`

Sub-project 1 of the off-the-clock XGB exploration: 1-minute backfill + per-product dollar bars for the scorecard's top-20 products.

- `services/history_backfill.py` — 1-minute backfill support (`backfill_product_1m`, `load_1m_history`, `_parquet_path_1m`), mirroring the 5m functions.
```

---

## Task 2: backfill_1m_candles.py operator CLI

**Files:**
- Create: `backend/tools/backfill_1m_candles.py`
- Test: `backend/tests/test_backfill_1m_candles.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_backfill_1m_candles.py
import pytest
from tools import backfill_1m_candles as b1m


def test_days_to_cover_computes_from_1h_first_ts(monkeypatch):
    # 1h history starting 10 days before now_ts
    now_ts = 10_000_000
    first_ts = now_ts - 10 * 86400
    monkeypatch.setattr(b1m, "load_history",
                        lambda pid: [{"start": first_ts}, {"start": now_ts}])
    assert b1m._days_to_cover("BTC-USD", now_ts) == 10


def test_days_to_cover_rounds_up_partial_day(monkeypatch):
    now_ts = 10_000_000
    first_ts = now_ts - (5 * 86400 + 100)  # 5 days + a bit
    monkeypatch.setattr(b1m, "load_history", lambda pid: [{"start": first_ts}])
    assert b1m._days_to_cover("BTC-USD", now_ts) == 6


def test_days_to_cover_no_parquet_returns_zero(monkeypatch):
    monkeypatch.setattr(b1m, "load_history", lambda pid: [])
    assert b1m._days_to_cover("BTC-USD", 10_000_000) == 0


def test_resolve_pids_from_explicit_arg():
    pids = b1m._resolve_pids("ignored.pt", "BTC-USD, ETH-USD ,SOL-USD")
    assert pids == ["BTC-USD", "ETH-USD", "SOL-USD"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_backfill_1m_candles.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.backfill_1m_candles'`.

- [ ] **Step 3: Write the implementation**

```python
# backend/tools/backfill_1m_candles.py
"""Operator CLI: backfill 1-minute candles for the scorecard's top-20 products.

Each product's depth is computed from its existing 1h parquet so the 1m
history covers the same calendar span. Long-running and network-heavy — run
offline; it makes only read-only Coinbase candle requests (it does not touch
the database or the live backend).
"""
from __future__ import annotations

import argparse
import asyncio
import math
import time

from services.history_backfill import backfill_product_1m, load_history


def _days_to_cover(pid: str, now_ts: int) -> int:
    """Days back needed for 1m history to reach the 1h parquet's first bar.

    Returns 0 when the product has no 1h parquet (nothing to calibrate against).
    """
    one_h = load_history(pid)
    if not one_h:
        return 0
    first_ts = int(one_h[0]["start"])
    return max(1, math.ceil((now_ts - first_ts) / 86400))


def _resolve_pids(cache_path: str, pids_arg: str | None) -> list[str]:
    """Explicit --pids list, else the survivorship-aware top-20 from the cache."""
    if pids_arg:
        return [p.strip() for p in pids_arg.split(",") if p.strip()]
    from tools._scorecard._cv_harness import top_n_pids_from_cache
    return list(top_n_pids_from_cache(cache_path))


async def _run(pids: list[str]) -> None:
    now_ts = int(time.time())
    for i, pid in enumerate(pids, 1):
        days = _days_to_cover(pid, now_ts)
        if days == 0:
            print(f"[{i}/{len(pids)}] {pid}: no 1h parquet — skip", flush=True)
            continue
        print(f"[{i}/{len(pids)}] {pid}: backfilling 1m, {days}d ...", flush=True)
        result = await backfill_product_1m(pid, days=days)
        print(f"    +{result['new_bars']} new | {result['total_bars']} total",
              flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill 1-minute candles for the top-20 products"
    )
    parser.add_argument("--cache", default="cnn_dataset_cache.pt",
                        help="cache for the survivorship-aware top-20 ranking")
    parser.add_argument("--pids", default=None,
                        help="comma-separated product ids (overrides --cache)")
    args = parser.parse_args()
    pids = _resolve_pids(args.cache, args.pids)
    print(f"1m backfill: {len(pids)} products", flush=True)
    asyncio.run(_run(pids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_backfill_1m_candles.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/backfill_1m_candles.py backend/tests/test_backfill_1m_candles.py CHANGELOG.md
git commit -m "feat(scorecard): 1m backfill operator CLI for top-20 products

backfill_1m_candles.py resolves the survivorship-aware top-20, computes
each product's backfill depth from its 1h parquet span, and runs
backfill_product_1m. Stage 1 of the dollar-bar data pipeline.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet to add under the Session 58.71n section:

```markdown
- `tools/backfill_1m_candles.py` — operator CLI driving the 1m backfill for the top-20 (depth per product from its 1h span).
```

---

## Task 3: dollar-value and threshold-calibration helpers

**Files:**
- Create: `backend/tools/build_dollar_bars.py`
- Test: `backend/tests/test_build_dollar_bars.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_build_dollar_bars.py
import pytest
from tools.build_dollar_bars import candle_dollar_value, calibrate_threshold


def test_candle_dollar_value_uses_typical_price():
    # typical price = (high + low + close) / 3 = (110 + 90 + 100) / 3 = 100
    c = {"start": 0, "open": 95.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "volume": 4.0}
    assert candle_dollar_value(c) == pytest.approx(400.0)  # 4 * 100


def test_calibrate_threshold_is_total_over_1h_count():
    # 3 candles, each dollar value 300 => total 900; 1h count 3 => threshold 300
    candles = [
        {"start": i, "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 3.0}
        for i in range(3)
    ]
    assert calibrate_threshold(candles, n_1h_bars=3) == pytest.approx(300.0)


def test_calibrate_threshold_rejects_nonpositive_bar_count():
    with pytest.raises(ValueError, match="n_1h_bars"):
        calibrate_threshold([], n_1h_bars=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.build_dollar_bars'`.

- [ ] **Step 3: Write the implementation**

```python
# backend/tools/build_dollar_bars.py
"""Construct per-product dollar bars from 1-minute candles.

Stage 2 of the dollar-bar data pipeline (SP1). A dollar bar closes when the
cumulative dollar volume (volume x typical price) of consecutive 1-minute
candles crosses a per-product threshold. The threshold is calibrated so each
product yields about the same number of bars as its existing 1h history.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "history")


def candle_dollar_value(candle: dict) -> float:
    """Dollar volume of one candle: volume x typical price ((H+L+C)/3)."""
    typical = (candle["high"] + candle["low"] + candle["close"]) / 3.0
    return candle["volume"] * typical


def calibrate_threshold(one_min_candles: list[dict], n_1h_bars: int) -> float:
    """Per-product dollar threshold = total dollar volume / 1h bar count.

    Raises:
        ValueError: if n_1h_bars is not positive.
    """
    if n_1h_bars <= 0:
        raise ValueError(f"n_1h_bars must be positive, got {n_1h_bars}")
    total = sum(candle_dollar_value(c) for c in one_min_candles)
    return total / n_1h_bars
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/build_dollar_bars.py backend/tests/test_build_dollar_bars.py CHANGELOG.md
git commit -m "feat(scorecard): dollar-value + threshold-calibration helpers

candle_dollar_value (volume x typical price) and calibrate_threshold
(total dollar volume / 1h bar count) for the dollar-bar pipeline.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under Session 58.71n:

```markdown
- `tools/build_dollar_bars.py` — `candle_dollar_value` + `calibrate_threshold` helpers.
```

---

## Task 4: dollar_bars_from_candles construction core

**Files:**
- Modify: `backend/tools/build_dollar_bars.py`
- Test: `backend/tests/test_build_dollar_bars.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_build_dollar_bars.py`:

```python
from tools.build_dollar_bars import dollar_bars_from_candles


def _flat_candle(start, price, vol):
    """A candle with open=high=low=close=price, so typical price == price."""
    return {"start": start, "open": price, "high": price, "low": price,
            "close": price, "volume": vol}


def test_dollar_bars_basic_boundaries():
    # 6 flat candles, dollar value 300 each; threshold 900 => a bar every 3.
    candles = [_flat_candle(i * 60, 100.0, 3.0) for i in range(6)]
    bars = dollar_bars_from_candles(candles, threshold=900.0)
    assert len(bars) == 2
    assert bars[0]["start"] == 0
    assert bars[0]["end"] == 120          # 3rd candle's start (i=2)
    assert bars[0]["n_candles"] == 3
    assert bars[0]["volume"] == pytest.approx(9.0)
    assert bars[0]["dollar_value"] == pytest.approx(900.0)
    assert bars[1]["start"] == 180


def test_dollar_bars_trailing_partial_dropped():
    # 7 candles => 2 full bars; the 7th (300 < 900) is an incomplete bar.
    candles = [_flat_candle(i * 60, 100.0, 3.0) for i in range(7)]
    bars = dollar_bars_from_candles(candles, threshold=900.0)
    assert len(bars) == 2


def test_dollar_bars_ohlc_aggregation():
    candles = [
        {"start": 0,   "open": 100.0, "high": 110.0, "low": 95.0,
         "close": 105.0, "volume": 10.0},
        {"start": 60,  "open": 105.0, "high": 120.0, "low": 100.0,
         "close": 115.0, "volume": 10.0},
        {"start": 120, "open": 115.0, "high": 118.0, "low": 90.0,
         "close": 92.0,  "volume": 10.0},
    ]
    # total dollar value ~3150; threshold 3000 => all 3 candles -> one bar.
    bars = dollar_bars_from_candles(candles, threshold=3000.0)
    assert len(bars) == 1
    assert bars[0]["open"] == 100.0
    assert bars[0]["high"] == 120.0
    assert bars[0]["low"] == 90.0
    assert bars[0]["close"] == 92.0
    assert bars[0]["n_candles"] == 3


def test_dollar_bars_empty_input():
    assert dollar_bars_from_candles([], threshold=100.0) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -k dollar_bars -v`
Expected: FAIL with `ImportError: cannot import name 'dollar_bars_from_candles'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/build_dollar_bars.py`:

```python
def dollar_bars_from_candles(candles: list[dict], threshold: float) -> list[dict]:
    """Walk time-ordered 1m candles into dollar bars.

    A bar closes on the candle whose inclusion makes cumulative dollar value
    reach `threshold`. A 1m candle is atomic — never split. The trailing
    partial bar (residual below threshold at series end) is dropped.

    Each output bar: start, end, open, high, low, close, volume,
    dollar_value, n_candles.
    """
    bars: list[dict] = []
    acc_dollar = 0.0
    acc_volume = 0.0
    bar_start = None
    bar_open = None
    bar_high = None
    bar_low = None
    n = 0

    for c in candles:
        if bar_start is None:
            bar_start = c["start"]
            bar_open = c["open"]
            bar_high = c["high"]
            bar_low = c["low"]
        else:
            bar_high = max(bar_high, c["high"])
            bar_low = min(bar_low, c["low"])
        acc_dollar += candle_dollar_value(c)
        acc_volume += c["volume"]
        n += 1

        if acc_dollar >= threshold:
            bars.append({
                "start": bar_start,
                "end": c["start"],
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": c["close"],
                "volume": acc_volume,
                "dollar_value": acc_dollar,
                "n_candles": n,
            })
            acc_dollar = 0.0
            acc_volume = 0.0
            bar_start = None
            bar_open = None
            bar_high = None
            bar_low = None
            n = 0

    return bars
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -k dollar_bars -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/build_dollar_bars.py backend/tests/test_build_dollar_bars.py CHANGELOG.md
git commit -m "feat(scorecard): dollar-bar construction core

dollar_bars_from_candles walks time-ordered 1m candles into dollar bars;
closes a bar when cumulative dollar value crosses the threshold, drops the
trailing partial.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under Session 58.71n:

```markdown
- `tools/build_dollar_bars.py` — `dollar_bars_from_candles` construction core (threshold-crossing boundaries, trailing partial dropped).
```

---

## Task 5: build_dollar_bars_for_candles pure assembly

**Files:**
- Modify: `backend/tools/build_dollar_bars.py`
- Test: `backend/tests/test_build_dollar_bars.py`

**Rationale:** A pure function that clips 1m candles to the 1h parquet's `[first_ts, last_ts]` window, calibrates the threshold on that clipped set, and constructs bars. Clipping in the assembly step keeps the threshold calibration coherent regardless of how much 1m data the backfill fetched.

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_build_dollar_bars.py`:

```python
from tools.build_dollar_bars import build_dollar_bars_for_candles


def test_build_assembly_clips_to_1h_window():
    # 1h window covers starts 100..400; 1m candles include out-of-window ones
    # on both sides (50 before, 500 after) that must be clipped out.
    one_h = [{"start": 100}, {"start": 400}]  # n_1h_bars = 2
    one_min = (
        [_flat_candle(50, 100.0, 3.0)]                               # before window
        + [_flat_candle(s, 100.0, 3.0) for s in range(100, 460, 60)]  # 100..400, in window
        + [_flat_candle(500, 100.0, 3.0)]                            # after window
    )
    bars = build_dollar_bars_for_candles(one_min, one_h)
    # Only the 6 candles with 100 <= start <= 400 count: dollar value 6*300=1800;
    # threshold = 1800 / 2 = 900 => 2 bars.
    assert len(bars) == 2
    assert all(100 <= b["start"] <= 400 for b in bars)
    assert all(100 <= b["end"] <= 400 for b in bars)


def test_build_assembly_empty_1h_returns_empty():
    assert build_dollar_bars_for_candles([_flat_candle(0, 100.0, 3.0)], []) == []


def test_build_assembly_no_1m_in_window_returns_empty():
    one_h = [{"start": 1000}, {"start": 2000}]
    one_min = [_flat_candle(0, 100.0, 3.0), _flat_candle(60, 100.0, 3.0)]
    assert build_dollar_bars_for_candles(one_min, one_h) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -k assembly -v`
Expected: FAIL with `ImportError: cannot import name 'build_dollar_bars_for_candles'`.

- [ ] **Step 3: Write the implementation**

Append to `backend/tools/build_dollar_bars.py`:

```python
def build_dollar_bars_for_candles(
    one_min_candles: list[dict],
    one_h_candles: list[dict],
) -> list[dict]:
    """Clip 1m candles to the 1h window, calibrate the threshold, build bars.

    `one_h_candles` and `one_min_candles` must be time-sorted (the
    history_backfill loaders return sorted lists). Returns [] when there is no
    1h history or no 1m candle falls inside its span.
    """
    if not one_h_candles:
        return []
    first_ts = int(one_h_candles[0]["start"])
    last_ts = int(one_h_candles[-1]["start"])
    clipped = [c for c in one_min_candles
               if first_ts <= int(c["start"]) <= last_ts]
    if not clipped:
        return []
    threshold = calibrate_threshold(clipped, len(one_h_candles))
    return dollar_bars_from_candles(clipped, threshold)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -k assembly -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/build_dollar_bars.py backend/tests/test_build_dollar_bars.py CHANGELOG.md
git commit -m "feat(scorecard): dollar-bar assembly with 1h-window clipping

build_dollar_bars_for_candles clips 1m candles to the 1h parquet span,
calibrates the threshold on that window, and constructs bars — keeping
calibration coherent regardless of backfill over-coverage.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under Session 58.71n:

```markdown
- `tools/build_dollar_bars.py` — `build_dollar_bars_for_candles` pure assembly (clip to 1h window + calibrate + construct).
```

---

## Task 6: build_dollar_bars.py persistence and CLI

**Files:**
- Modify: `backend/tools/build_dollar_bars.py`
- Test: `backend/tests/test_build_dollar_bars.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_build_dollar_bars.py`:

```python
from tools import build_dollar_bars as bdb


def test_dollar_parquet_path_uses_dollar_subdir():
    p = bdb._dollar_parquet_path("BTC-USD")
    assert p.endswith(os.path.join("dollar", "BTC-USD.parquet"))


def test_save_and_reload_dollar_bars_parquet_roundtrip(tmp_path):
    bars = [
        {"start": 0, "end": 120, "open": 100.0, "high": 110.0, "low": 90.0,
         "close": 105.0, "volume": 9.0, "dollar_value": 900.0, "n_candles": 3},
    ]
    path = str(tmp_path / "BTC-USD.parquet")
    bdb._save_dollar_bars(path, bars)
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pydict()
    assert rows["start"] == [0]
    assert rows["n_candles"] == [3]
    assert rows["dollar_value"][0] == pytest.approx(900.0)


def test_build_for_pid_writes_parquet(tmp_path, monkeypatch):
    one_h = [{"start": 0}, {"start": 600}]  # n_1h_bars = 2
    one_min = [
        {"start": s, "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 3.0}
        for s in range(0, 660, 60)  # 11 candles in [0, 600]
    ]
    monkeypatch.setattr(bdb, "load_1m_history", lambda pid: one_min)
    monkeypatch.setattr(bdb, "load_history", lambda pid: one_h)
    monkeypatch.setattr(bdb, "_dollar_parquet_path",
                        lambda pid: str(tmp_path / f"{pid}.parquet"))
    result = bdb.build_for_pid("BTC-USD")
    assert result["pid"] == "BTC-USD"
    assert result["n_bars"] > 0
    assert (tmp_path / "BTC-USD.parquet").exists()


def test_build_for_pid_no_data_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bdb, "load_1m_history", lambda pid: [])
    monkeypatch.setattr(bdb, "load_history", lambda pid: [])
    monkeypatch.setattr(bdb, "_dollar_parquet_path",
                        lambda pid: str(tmp_path / f"{pid}.parquet"))
    result = bdb.build_for_pid("BTC-USD")
    assert result["n_bars"] == 0
    assert not (tmp_path / "BTC-USD.parquet").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -k "parquet or build_for_pid" -v`
Expected: FAIL with `AttributeError: module 'tools.build_dollar_bars' has no attribute '_dollar_parquet_path'`.

- [ ] **Step 3: Write the implementation**

Add the import to the top of `backend/tools/build_dollar_bars.py` (with the other imports):

```python
from services.history_backfill import load_1m_history, load_history
```

Append to `backend/tools/build_dollar_bars.py`:

```python
_DOLLAR_SCHEMA = pa.schema([
    pa.field("start",        pa.int64()),
    pa.field("end",          pa.int64()),
    pa.field("open",         pa.float64()),
    pa.field("high",         pa.float64()),
    pa.field("low",          pa.float64()),
    pa.field("close",        pa.float64()),
    pa.field("volume",       pa.float64()),
    pa.field("dollar_value", pa.float64()),
    pa.field("n_candles",    pa.int64()),
])


def _dollar_parquet_path(product_id: str) -> str:
    """Dollar-bar parquet path — separate namespace under history/dollar/."""
    safe = product_id.replace("/", "_")
    return os.path.join(_HISTORY_DIR, "dollar", f"{safe}.parquet")


def _save_dollar_bars(path: str, bars: list[dict]) -> None:
    """Write dollar bars to a parquet file (overwrites)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.table(
        {
            "start":        [b["start"]        for b in bars],
            "end":          [b["end"]          for b in bars],
            "open":         [b["open"]         for b in bars],
            "high":         [b["high"]         for b in bars],
            "low":          [b["low"]          for b in bars],
            "close":        [b["close"]        for b in bars],
            "volume":       [b["volume"]       for b in bars],
            "dollar_value": [b["dollar_value"] for b in bars],
            "n_candles":    [b["n_candles"]    for b in bars],
        },
        schema=_DOLLAR_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def build_for_pid(product_id: str) -> dict:
    """Build and persist dollar bars for one product. {} parquet if no bars."""
    one_min = load_1m_history(product_id)
    one_h = load_history(product_id)
    bars = build_dollar_bars_for_candles(one_min, one_h)
    if bars:
        _save_dollar_bars(_dollar_parquet_path(product_id), bars)
    return {"pid": product_id, "n_bars": len(bars)}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build dollar bars for the top-20 products from 1m candles"
    )
    parser.add_argument("--cache", default="cnn_dataset_cache.pt",
                        help="cache for the survivorship-aware top-20 ranking")
    parser.add_argument("--pids", default=None,
                        help="comma-separated product ids (overrides --cache)")
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    else:
        from tools._scorecard._cv_harness import top_n_pids_from_cache
        pids = list(top_n_pids_from_cache(args.cache))

    print(f"build_dollar_bars: {len(pids)} products", flush=True)
    for i, pid in enumerate(pids, 1):
        result = build_for_pid(pid)
        print(f"[{i}/{len(pids)}] {pid}: {result['n_bars']} dollar bars",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_build_dollar_bars.py -v`
Expected: 14 passed (all Stage 2 tests).

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/build_dollar_bars.py backend/tests/test_build_dollar_bars.py CHANGELOG.md
git commit -m "feat(scorecard): dollar-bar persistence + CLI

_save_dollar_bars writes the dollar-bar parquet schema; build_for_pid
composes load -> assemble -> persist; main() runs the top-20. Completes
Stage 2 of the dollar-bar data pipeline (SP1).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under Session 58.71n:

```markdown
- `tools/build_dollar_bars.py` — `_save_dollar_bars` (parquet) + `build_for_pid` + `main` CLI; writes `data/history/dollar/<pid>.parquet`.
```

---

## Operator steps (after the plan is implemented)

These run the actual pipeline — they are not test steps and are performed by the operator, offline, during a quiet window:

1. `cd backend && ../.venv/Scripts/python.exe -m tools.backfill_1m_candles` — long-running 1m backfill for the top-20 (read-only Coinbase requests; safe alongside the live backend).
2. `cd backend && ../.venv/Scripts/python.exe -m tools.build_dollar_bars` — construct + persist dollar bars to `data/history/dollar/`.
3. Spot-check a `data/history/dollar/<pid>.parquet`: bar count ≈ the product's 1h bar count; `n_candles` varies (small in volatile periods, large in quiet ones).

Sub-project 2 (off-the-clock XGB track) then consumes `data/history/dollar/`.

---

## Done criteria

- [ ] All new tests pass: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_history_backfill.py tests/test_backfill_1m_candles.py tests/test_build_dollar_bars.py -v`
- [ ] Full suite green via the pre-commit hook on every commit.
- [ ] No new dependencies.
- [ ] `cnn_agent.py` not modified.
- [ ] All commits pushed to `feat/gpu-coord-mirror`.
- [ ] CHANGELOG.md has the Session 58.71n section with one sub-bullet per task.
- [ ] Memory: append a session-log note to `coinbase_trader_session_log.md` recording the SP1 pipeline ship.
