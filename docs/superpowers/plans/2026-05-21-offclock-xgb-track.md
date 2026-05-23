# Off-the-Clock XGB Track Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and scorecard-evaluate 16 XGB configs — 2 substrates (dollar bars / 1h time bars) × 2 label variants (direction / triple-barrier) × 4 horizons (k ∈ {4,24,72,168}) — to test whether activity-based bars beat fixed time bars.

**Architecture:** A parameterized harness `_scorecard/_offclock_harness.py` builds samples for any `(substrate, label_variant, horizon)` using the existing `extract_v4` feature extractor, runs 5-fold purged-WF OOF prediction with a fresh per-fold booster, and composes the input for `compute_scorecard`. A CLI `tools/offclock_sweep.py` loops the 16 configs and writes a results doc. Reuses `compute_scorecard`, `purged_walk_forward_splits`, `realized_log_returns_per_sample`, and `extract_v4` — no new modeling primitives.

**Tech Stack:** Python 3.11 (`.venv/Scripts/python.exe`), numpy, xgboost, pyarrow, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-21-offclock-xgb-track-design.md`.

**Conventions:**
- TDD red-green-refactor — write the failing test, run RED, implement to GREEN, commit.
- `.venv/Scripts/python.exe` for all test runs.
- Each commit includes the test file + implementation + a `CHANGELOG.md` sub-bullet, pushed immediately.
- The pre-commit hook runs the full regression suite when Python files are staged — commit only during a training/backend-idle window (`feedback_no_pytest_during_trading`).
- No live API calls / no live training in unit tests — synthetic data and mocked harness only.
- Pure functions with type hints, single responsibility.

---

## File Structure

**Create:**
- `backend/tools/_scorecard/_offclock_harness.py` — bar loaders, the two label functions, per-product sample building, pooling, OOF prediction, the per-config runner.
- `backend/tools/offclock_sweep.py` — CLI: loop the 16 configs, run `compute_scorecard` per config, write the results doc.

**Test:**
- `backend/tests/test_offclock_harness.py`
- `backend/tests/test_offclock_sweep.py`

**Modify:**
- `CHANGELOG.md` — one sub-bullet per task under a new Session section.

**Output (operator-run):**
- `docs/superpowers/specs/2026-05-21-offclock-sweep-results.md` — the 16-row results table.

**Responsibilities:** `_offclock_harness.py` owns everything from "raw bars" to "scorecard input dict". `offclock_sweep.py` owns the 16-config loop and the results-doc rendering. Both reuse the scorecard package's existing primitives.

---

## Task 1: Harness skeleton and bar loaders

**Files:**
- Create: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_offclock_harness.py
import os
import pytest
from tools._scorecard import _offclock_harness as och


def test_load_bars_rejects_unknown_substrate():
    with pytest.raises(ValueError, match="substrate"):
        och.load_bars("hourly", "BTC-USD")


def test_load_dollar_bars_missing_file_returns_empty():
    assert och.load_dollar_bars("ZZZ-NONEXISTENT-USD") == []


def test_load_bars_time_delegates_to_history(monkeypatch):
    sentinel = [{"start": 1, "open": 1.0, "high": 1.0, "low": 1.0,
                 "close": 1.0, "volume": 1.0}]
    monkeypatch.setattr(och, "load_history", lambda pid: sentinel)
    assert och.load_bars("time", "BTC-USD") is sentinel


def test_load_dollar_bars_roundtrip(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "start": [60, 0], "open": [2.0, 1.0], "high": [2.0, 1.0],
        "low": [2.0, 1.0], "close": [2.0, 1.0], "volume": [2.0, 1.0],
        "end": [119, 59], "dollar_value": [2.0, 1.0], "n_candles": [1, 1],
    })
    d = tmp_path / "dollar"
    d.mkdir()
    pq.write_table(table, str(d / "BTC-USD.parquet"))
    monkeypatch.setattr(och, "_HISTORY_DIR", str(tmp_path))
    bars = och.load_dollar_bars("BTC-USD")
    assert [b["start"] for b in bars] == [0, 60]   # sorted ascending
    assert bars[0]["close"] == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools._scorecard._offclock_harness'`.

- [ ] **Step 3: Write the implementation**

```python
# backend/tools/_scorecard/_offclock_harness.py
"""Off-the-clock XGB track: sample building + OOF prediction (SP2).

Builds XGB training samples on either dollar bars (data/history/dollar/) or
1h time bars (data/history/), with two label variants (direction,
triple-barrier) across a horizon sweep, and produces out-of-fold predictions
for the deployment scorecard. See 2026-05-21-offclock-xgb-track-design.md.
"""
from __future__ import annotations

import os

from services.history_backfill import load_history

_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "history")


def load_dollar_bars(pid: str) -> list[dict]:
    """Load a product's dollar bars from data/history/dollar/<pid>.parquet.

    Returns OHLCV+start bar dicts sorted by start; [] if the file is missing.
    """
    import pyarrow.parquet as pq

    safe = pid.replace("/", "_")
    path = os.path.join(_HISTORY_DIR, "dollar", f"{safe}.parquet")
    if not os.path.exists(path):
        return []
    rows = pq.read_table(path).to_pydict()
    n = len(rows["start"])
    bars = [
        {
            "start": int(rows["start"][i]),
            "open": float(rows["open"][i]),
            "high": float(rows["high"][i]),
            "low": float(rows["low"][i]),
            "close": float(rows["close"][i]),
            "volume": float(rows["volume"][i]),
        }
        for i in range(n)
    ]
    bars.sort(key=lambda b: b["start"])
    return bars


def load_bars(substrate: str, pid: str) -> list[dict]:
    """Load a product's bars for the substrate: 'dollar' or 'time'."""
    if substrate == "dollar":
        return load_dollar_bars(pid)
    if substrate == "time":
        return load_history(pid)
    raise ValueError(f"unknown substrate {substrate!r}; expected 'dollar' or 'time'")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): off-clock harness skeleton + bar loaders

load_dollar_bars / load_bars for the SP2 off-the-clock XGB track — reads
dollar bars from data/history/dollar/ or 1h time bars via history_backfill.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

For the `CHANGELOG.md` edit, add this section directly under the top `---` divider:

```markdown
## Session 58.71o — Off-the-clock XGB track (SP2) — 2026-05-21

**Spec:** `docs/superpowers/specs/2026-05-21-offclock-xgb-track-design.md`
**Plan:** `docs/superpowers/plans/2026-05-21-offclock-xgb-track.md`

Sub-project 2 of the off-the-clock XGB exploration: train + scorecard 16 configs (2 substrates × 2 label variants × 4 horizons).

- `tools/_scorecard/_offclock_harness.py` — bar loaders (`load_dollar_bars`, `load_bars`).
```

---

## Task 2: Direction label

**Files:**
- Modify: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests** — Append to `backend/tests/test_offclock_harness.py`:

```python
from tools._scorecard._offclock_harness import direction_label


def test_direction_label_up():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    label, exit_close = direction_label(closes, t=0, k=4)
    assert label == 1
    assert exit_close == 104.0


def test_direction_label_down():
    closes = [100.0, 99.0, 98.0, 97.0, 96.0]
    label, exit_close = direction_label(closes, t=0, k=4)
    assert label == 0
    assert exit_close == 96.0


def test_direction_label_flat_is_zero():
    closes = [100.0, 100.0, 100.0]
    label, exit_close = direction_label(closes, t=0, k=2)
    assert label == 0          # not strictly greater
    assert exit_close == 100.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k direction -v`
Expected: FAIL with `ImportError: cannot import name 'direction_label'`.

- [ ] **Step 3: Write the implementation** — Append to `backend/tools/_scorecard/_offclock_harness.py`:

```python
def direction_label(closes, t: int, k: int) -> tuple[int, float]:
    """k-bars-ahead direction label for entry bar t.

    Returns (label, exit_close): label is 1 if close[t+k] > close[t] else 0;
    exit_close is close[t+k]. The caller guarantees t + k < len(closes).
    """
    entry = closes[t]
    exit_close = float(closes[t + k])
    return (1 if exit_close > entry else 0), exit_close
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k direction -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): direction label for the off-clock track

direction_label: k-bars-ahead close-direction label + exit close.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/_scorecard/_offclock_harness.py` — `direction_label` (k-bars-ahead direction).
```

---

## Task 3: Triple-barrier label

**Files:**
- Modify: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests** — Append to `backend/tests/test_offclock_harness.py`:

```python
from tools._scorecard._offclock_harness import triple_barrier_label


def _ohlc(start, o, h, l, c):
    return {"start": start, "open": o, "high": h, "low": l, "close": c,
            "volume": 1.0}


def test_triple_barrier_upper_hit():
    # entry close 100; upper barrier 101. Bar 2 highs to 101.5 -> UP.
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 100.5, 99.8, 100.2),
        _ohlc(2, 100.2, 101.5, 100.1, 101.0),
        _ohlc(3, 101.0, 101.2, 100.9, 101.0),
        _ohlc(4, 101.0, 101.1, 100.8, 100.9),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(101.0)   # entry * 1.01


def test_triple_barrier_lower_hit():
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 100.1, 98.5, 99.0),     # low 98.5 <= 99.0 barrier
        _ohlc(2, 99.0, 99.2, 98.8, 99.0),
        _ohlc(3, 99.0, 99.1, 98.9, 99.0),
        _ohlc(4, 99.0, 99.1, 98.9, 99.0),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 0
    assert exit_close == pytest.approx(99.0)    # entry * 0.99


def test_triple_barrier_timeout_uses_close_direction():
    # neither barrier hit within k; close[t+k]=100.4 > entry -> label 1
    bars = [_ohlc(i, 100.0, 100.3, 99.8, 100.0 + 0.1 * i) for i in range(5)]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(100.4)


def test_triple_barrier_both_hit_close_breaks_tie():
    # bar 1 hits both barriers; close 101.0 >= entry -> UP
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 102.0, 98.0, 101.0),
        _ohlc(2, 101.0, 101.0, 101.0, 101.0),
        _ohlc(3, 101.0, 101.0, 101.0, 101.0),
        _ohlc(4, 101.0, 101.0, 101.0, 101.0),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(101.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k triple_barrier -v`
Expected: FAIL with `ImportError: cannot import name 'triple_barrier_label'`.

- [ ] **Step 3: Write the implementation** — Append to `backend/tools/_scorecard/_offclock_harness.py`.

First add this constant near the top of the file, after `_HISTORY_DIR`:

```python
_TB_BARRIER = 0.01  # +/-1% triple-barrier
```

Then append the function:

```python
def triple_barrier_label(bars, t: int, k: int) -> tuple[int, float]:
    """Triple-barrier label for entry bar t with a k-bar vertical timeout.

    Upper barrier = close[t] * 1.01, lower = close[t] * 0.99. Scans bars
    t+1 .. t+k. Returns (label, exit_close):
      - upper hit first   -> (1, upper)
      - lower hit first   -> (0, lower)
      - both in one bar   -> close-direction breaks the tie (close >= entry -> UP)
      - neither (timeout) -> (1 if close[t+k] > close[t] else 0, close[t+k])
    The caller guarantees t + k < len(bars).
    """
    entry = bars[t]["close"]
    upper = entry * (1.0 + _TB_BARRIER)
    lower = entry * (1.0 - _TB_BARRIER)
    for i in range(t + 1, t + k + 1):
        b = bars[i]
        hit_up = b["high"] >= upper
        hit_dn = b["low"] <= lower
        if hit_up and hit_dn:
            return (1, upper) if b["close"] >= entry else (0, lower)
        if hit_up:
            return 1, upper
        if hit_dn:
            return 0, lower
    exit_close = float(bars[t + k]["close"])
    return (1 if exit_close > entry else 0), exit_close
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k triple_barrier -v`
Expected: 4 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): triple-barrier label for the off-clock track

triple_barrier_label: +/-1% barriers, k-bar timeout, barrier-aware exit
price (UP/DOWN/timeout/tie).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/_scorecard/_offclock_harness.py` — `triple_barrier_label` (±1% barriers, k-bar timeout, barrier-aware exit).
```

---

## Task 4: Per-product sample building

**Files:**
- Modify: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests** — Append to `backend/tests/test_offclock_harness.py`:

```python
from tools._scorecard._offclock_harness import build_product_samples


def _rising_bars(n):
    """n bars with strictly rising close so direction labels are all 1."""
    return [
        {"start": i * 60, "open": 100.0 + 0.01 * i, "high": 100.0 + 0.01 * i,
         "low": 100.0 + 0.01 * i, "close": 100.0 + 0.01 * i, "volume": 1.0}
        for i in range(n)
    ]


def test_build_product_samples_shape_and_count():
    bars = _rising_bars(400)
    s = build_product_samples(bars, "direction", k=4, sample_step=24)
    # samples roll at t in range(336, 396, 24) -> 336, 360, 384 => 3 samples
    assert s["X"].shape == (3, 150)
    assert len(s["y"]) == 3
    assert list(s["y"]) == [1, 1, 1]            # rising closes
    assert s["entry_ts"][0] == 336 * 60


def test_build_product_samples_too_short_returns_empty():
    bars = _rising_bars(338)                    # < 336 + k + 1 for k=4
    s = build_product_samples(bars, "direction", k=4, sample_step=24)
    assert s["X"].shape == (0, 150)
    assert len(s["y"]) == 0


def test_build_product_samples_rejects_unknown_variant():
    bars = _rising_bars(400)
    with pytest.raises(ValueError, match="label_variant"):
        build_product_samples(bars, "regression", k=4, sample_step=24)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k build_product -v`
Expected: FAIL with `ImportError: cannot import name 'build_product_samples'`.

- [ ] **Step 3: Write the implementation** — Append to `backend/tools/_scorecard/_offclock_harness.py`:

```python
_MACRO_WINDOW = 336  # macro tier lookback (= TIER_WINDOWS_V4["macro"])


def build_product_samples(
    bars: list[dict],
    label_variant: str,
    k: int,
    sample_step: int,
) -> dict:
    """Build samples for one product's bar list.

    Rolls one sample every `sample_step` bars from index 336 (macro lookback)
    up to len(bars) - k. Each sample: extract_v4 features over the micro/meso/
    macro tiers, a label, and entry/exit close prices.

    Returns a dict of numpy arrays: X (N,150), y (N,), entry_close (N,),
    exit_close (N,), entry_ts (N,). Empty arrays if the product is too short.

    Raises:
        ValueError: if label_variant is not 'direction' or 'triple_barrier'.
    """
    import numpy as np
    from tools.xgb_v4_features import N_FEATURES_V4, extract_v4

    if label_variant not in ("direction", "triple_barrier"):
        raise ValueError(
            f"unknown label_variant {label_variant!r}; "
            "expected 'direction' or 'triple_barrier'"
        )

    empty = {
        "X": np.zeros((0, N_FEATURES_V4), dtype=np.float64),
        "y": np.zeros(0, dtype=np.int64),
        "entry_close": np.zeros(0, dtype=np.float64),
        "exit_close": np.zeros(0, dtype=np.float64),
        "entry_ts": np.zeros(0, dtype=np.int64),
    }
    n = len(bars)
    last_t = n - k
    if last_t <= _MACRO_WINDOW:
        return empty

    closes = [b["close"] for b in bars]
    feats, ys, ec, xc, ts = [], [], [], [], []
    for t in range(_MACRO_WINDOW, last_t, sample_step):
        tier_slices = {
            "micro": bars[t - 60:t],
            "meso": bars[t - 168:t],
            "macro": bars[t - 336:t],
        }
        f, _ = extract_v4(tier_slices)
        if label_variant == "direction":
            label, exit_close = direction_label(closes, t, k)
        else:
            label, exit_close = triple_barrier_label(bars, t, k)
        feats.append(f[0])
        ys.append(label)
        ec.append(float(closes[t]))
        xc.append(exit_close)
        ts.append(int(bars[t]["start"]))

    if not feats:
        return empty
    return {
        "X": np.stack(feats, axis=0),
        "y": np.array(ys, dtype=np.int64),
        "entry_close": np.array(ec, dtype=np.float64),
        "exit_close": np.array(xc, dtype=np.float64),
        "entry_ts": np.array(ts, dtype=np.int64),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k build_product -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): per-product sample building for the off-clock track

build_product_samples: rolls extract_v4 samples over micro/meso/macro tiers
with the chosen label variant, returns feature/label/return/timestamp arrays.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/_scorecard/_offclock_harness.py` — `build_product_samples` (extract_v4 over tiers + label per sample).
```

---

## Task 5: Pool samples across products

**Files:**
- Modify: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests** — Append to `backend/tests/test_offclock_harness.py`:

```python
from tools._scorecard._offclock_harness import pool_samples


def test_pool_samples_concatenates_and_sorts(monkeypatch):
    import numpy as np
    from tools._scorecard import _offclock_harness as h

    # two products: pid B's samples are chronologically before pid A's
    samples = {
        "A": {"X": np.ones((2, 150)), "y": np.array([1, 0]),
              "entry_close": np.array([10.0, 11.0]),
              "exit_close": np.array([10.5, 11.5]),
              "entry_ts": np.array([300, 400], dtype=np.int64)},
        "B": {"X": np.zeros((1, 150)), "y": np.array([1]),
              "entry_close": np.array([20.0]), "exit_close": np.array([21.0]),
              "entry_ts": np.array([100], dtype=np.int64)},
    }
    monkeypatch.setattr(h, "load_bars", lambda substrate, pid: ["bars-of", pid])
    monkeypatch.setattr(h, "build_product_samples",
                        lambda bars, lv, k, step: samples[bars[1]])
    pooled = pool_samples("dollar", "direction", k=4,
                          pids=["A", "B"], sample_step=24)
    # sorted by entry_ts ascending: 100 (B), 300 (A), 400 (A)
    assert list(pooled["entry_ts"]) == [100, 300, 400]
    assert pooled["X"].shape == (3, 150)


def test_pool_samples_raises_when_no_samples(monkeypatch):
    from tools._scorecard import _offclock_harness as h
    import numpy as np
    monkeypatch.setattr(h, "load_bars", lambda substrate, pid: [])
    monkeypatch.setattr(h, "build_product_samples",
                        lambda bars, lv, k, step: {
                            "X": np.zeros((0, 150)), "y": np.zeros(0),
                            "entry_close": np.zeros(0), "exit_close": np.zeros(0),
                            "entry_ts": np.zeros(0, dtype=np.int64)})
    with pytest.raises(RuntimeError, match="no samples"):
        pool_samples("dollar", "direction", k=4, pids=["A"], sample_step=24)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k pool -v`
Expected: FAIL with `ImportError: cannot import name 'pool_samples'`.

- [ ] **Step 3: Write the implementation** — Append to `backend/tools/_scorecard/_offclock_harness.py`:

```python
def pool_samples(
    substrate: str,
    label_variant: str,
    k: int,
    pids: list[str],
    sample_step: int,
) -> dict:
    """Build and pool samples across products, sorted by entry timestamp.

    Returns the same dict shape as build_product_samples, concatenated over
    all products that yielded at least one sample.

    Raises:
        RuntimeError: if no product yields any sample.
    """
    import numpy as np

    parts = []
    for pid in pids:
        bars = load_bars(substrate, pid)
        s = build_product_samples(bars, label_variant, k, sample_step)
        if len(s["y"]) > 0:
            parts.append(s)

    if not parts:
        raise RuntimeError(
            f"no samples for substrate={substrate!r} label_variant="
            f"{label_variant!r} k={k} — check data/history/ inputs exist"
        )

    pooled = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.argsort(pooled["entry_ts"], kind="stable")
    return {key: val[order] for key, val in pooled.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k pool -v`
Expected: 2 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): pool off-clock samples across products

pool_samples: builds per-product samples for the top-20, concatenates, and
sorts by entry timestamp for purged-WF CV.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/_scorecard/_offclock_harness.py` — `pool_samples` (concat top-20, sort by entry timestamp).
```

---

## Task 6: Out-of-fold prediction and per-config runner

**Files:**
- Modify: `backend/tools/_scorecard/_offclock_harness.py`
- Test: `backend/tests/test_offclock_harness.py`

- [ ] **Step 1: Write the failing tests** — Append to `backend/tests/test_offclock_harness.py`:

```python
from tools._scorecard._offclock_harness import oof_predict_offclock, run_config


def test_oof_predict_offclock_shapes():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 500
    X = rng.normal(size=(n, 150))
    y = rng.integers(0, 2, size=n)
    entry_ts = np.arange(n, dtype=np.int64) * 3600
    scores, fold_ids, spans = oof_predict_offclock(X, y, entry_ts)
    assert scores.shape == (n,)
    assert not np.isnan(scores).any()           # every sample gets an OOF score
    assert set(fold_ids.tolist()) == {0, 1, 2, 3, 4}
    assert set(spans.keys()) == {0, 1, 2, 3, 4}


def test_run_config_composes_pool_oof_returns(monkeypatch):
    import numpy as np
    from tools._scorecard import _offclock_harness as h

    pooled = {
        "X": np.ones((4, 150)), "y": np.array([1, 0, 1, 0]),
        "entry_close": np.array([10.0, 10.0, 10.0, 10.0]),
        "exit_close": np.array([11.0, 9.0, 11.0, 9.0]),
        "entry_ts": np.array([1, 2, 3, 4], dtype=np.int64),
    }
    monkeypatch.setattr(h, "pool_samples",
                        lambda sub, lv, k, pids, step: pooled)
    monkeypatch.setattr(h, "oof_predict_offclock",
                        lambda X, y, ts: (np.array([0.6, 0.4, 0.7, 0.3]),
                                          np.array([0, 0, 1, 1]),
                                          {0: 30.0, 1: 30.0}))
    out = run_config("dollar", "direction", k=4, pids=["A"], sample_step=24)
    assert out["scores"].shape == (4,)
    assert list(out["labels"]) == [1, 0, 1, 0]
    # returns = ln(exit/entry): ln(1.1), ln(0.9), ln(1.1), ln(0.9)
    assert out["returns"][0] == pytest.approx(np.log(1.1))
    assert out["fold_spans_days"] == {0: 30.0, 1: 30.0}
    assert out["n"] == 4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k "oof or run_config" -v`
Expected: FAIL with `ImportError: cannot import name 'oof_predict_offclock'`.

- [ ] **Step 3: Write the implementation** — Append to `backend/tools/_scorecard/_offclock_harness.py`:

```python
def oof_predict_offclock(X, y, entry_ts, n_folds: int = 5, embargo_hours: int = 4):
    """Out-of-fold predictions via 5-fold purged walk-forward CV.

    Trains a fresh booster per fold with the v4 production config
    (feature_weights_v4 on the DMatrix, subsample 0.7, colsample 0.8). A fold
    whose training rows are single-class falls back to a constant 0.5 score.

    Returns (scores, fold_ids, fold_spans_days).
    """
    import numpy as np
    import xgboost as xgb

    from tools.walk_forward import purged_walk_forward_splits
    from tools.xgb_v4_features import feature_names_v4, feature_weights_v4

    names = feature_names_v4()
    weights = feature_weights_v4()
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.8,
        "learning_rate": 0.05,
        "seed": 0,
        "verbosity": 0,
    }

    n = len(y)
    scores = np.full(n, np.nan)
    fold_ids = np.full(n, -1, dtype=int)
    fold_spans_days: dict[int, float] = {}

    splits = list(purged_walk_forward_splits(entry_ts, n_folds, embargo_hours))
    for f_idx, (train_idx, val_idx) in enumerate(splits):
        if len(np.unique(y[train_idx])) < 2:
            scores[val_idx] = 0.5
        else:
            d_tr = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=names)
            d_tr.set_info(feature_weights=weights)
            d_va = xgb.DMatrix(X[val_idx], feature_names=names)
            booster = xgb.train(params, d_tr, num_boost_round=200)
            scores[val_idx] = np.asarray(booster.predict(d_va), dtype=np.float64)
        fold_ids[val_idx] = f_idx
        span = (int(entry_ts[val_idx].max()) - int(entry_ts[val_idx].min())) / 86400.0
        fold_spans_days[f_idx] = float(span) if span > 0 else 1.0

    return scores, fold_ids, fold_spans_days


def run_config(
    substrate: str,
    label_variant: str,
    k: int,
    pids: list[str],
    sample_step: int,
) -> dict:
    """Build pooled samples, OOF-predict, compute realized returns.

    Returns the dict compute_scorecard consumes: scores, labels, returns,
    fold_ids, fold_spans_days, n.
    """
    from tools._returns import realized_log_returns_per_sample

    pooled = pool_samples(substrate, label_variant, k, pids, sample_step)
    scores, fold_ids, fold_spans_days = oof_predict_offclock(
        pooled["X"], pooled["y"], pooled["entry_ts"]
    )
    returns = realized_log_returns_per_sample(
        pooled["entry_close"], pooled["exit_close"]
    )
    return {
        "scores": scores,
        "labels": pooled["y"].astype(int),
        "returns": returns,
        "fold_ids": fold_ids,
        "fold_spans_days": fold_spans_days,
        "n": int(len(pooled["y"])),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py -k "oof or run_config" -v`
Expected: 2 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/_scorecard/_offclock_harness.py backend/tests/test_offclock_harness.py CHANGELOG.md
git commit -m "feat(scorecard): off-clock OOF prediction + per-config runner

oof_predict_offclock: 5-fold purged-WF, fresh per-fold v4-config booster.
run_config: pool -> OOF -> realized returns -> compute_scorecard input.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/_scorecard/_offclock_harness.py` — `oof_predict_offclock` (5-fold purged-WF) + `run_config` (per-config scorecard input).
```

---

## Task 7: Sweep CLI and results doc

**Files:**
- Create: `backend/tools/offclock_sweep.py`
- Test: `backend/tests/test_offclock_sweep.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_offclock_sweep.py
import pytest
from tools import offclock_sweep as sweep


def test_config_grid_is_16_configs():
    grid = sweep._config_grid()
    assert len(grid) == 16
    # 2 substrates x 2 label variants x 4 horizons, all distinct
    assert len(set(grid)) == 16
    assert ("dollar", "direction", 4) in grid
    assert ("time", "triple_barrier", 168) in grid


def test_render_results_doc_has_row_per_config():
    rows = [
        {"substrate": "dollar", "label_variant": "direction", "horizon": 4,
         "auc": 0.55, "n": 5000, "precision": True, "expected_return": True,
         "paper_sharpe": False, "ece": True, "recommended_tau": 0.6},
        {"substrate": "time", "label_variant": "direction", "horizon": 4,
         "auc": 0.51, "n": 5000, "precision": False, "expected_return": False,
         "paper_sharpe": False, "ece": True, "recommended_tau": float("nan")},
    ]
    doc = sweep._render_results_doc(rows)
    assert "# Off-the-Clock Sweep Results" in doc
    assert doc.count("| dollar ") >= 1
    assert doc.count("| time ") >= 1
    # the dollar-minus-time delta section pairs the matched configs
    assert "Dollar - time delta" in doc
    assert "| direction | 4 |" in doc


def test_render_results_doc_empty_rows():
    doc = sweep._render_results_doc([])
    assert "# Off-the-Clock Sweep Results" in doc
    assert "no configs" in doc.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.offclock_sweep'`.

- [ ] **Step 3: Write the implementation**

```python
# backend/tools/offclock_sweep.py
"""CLI: sweep the 16 off-the-clock XGB configs and scorecard each.

For each (substrate, label_variant, horizon) it builds samples, runs 5-fold
purged-WF OOF prediction, scores the result through compute_scorecard, and
writes a results table. See 2026-05-21-offclock-xgb-track-design.md.

Operator-run and offline: it trains 16 x 5 boosters. Requires
data/history/dollar/ to be populated (SP1's backfill + build steps).
"""
from __future__ import annotations

import argparse
import os
import traceback

SUBSTRATES = ("dollar", "time")
LABEL_VARIANTS = ("direction", "triple_barrier")
HORIZONS = (4, 24, 72, 168)

_DEFAULT_CACHE = "cnn_dataset_cache.pt"
_DEFAULT_SAMPLE_STEP = 24
_DEFAULT_OUT = os.path.join(
    "..", "docs", "superpowers", "specs", "2026-05-21-offclock-sweep-results.md"
)


def _config_grid() -> list[tuple[str, str, int]]:
    """All 16 (substrate, label_variant, horizon) configs."""
    return [
        (s, lv, k)
        for s in SUBSTRATES
        for lv in LABEL_VARIANTS
        for k in HORIZONS
    ]


def _gates_passed(row: dict) -> int:
    """Count of the 4 hard gates a config row passes."""
    return sum((row["precision"], row["expected_return"],
                row["paper_sharpe"], row["ece"]))


def _render_results_doc(rows: list[dict]) -> str:
    """Render the sweep results as a markdown doc: per-config table + the
    dollar-minus-time delta per (label_variant, horizon) cell."""
    lines = [
        "# Off-the-Clock Sweep Results",
        "",
        "Spec: `2026-05-21-offclock-xgb-track-design.md`. Each row is one XGB",
        "config: 5-fold purged-WF OOF, scored by the deployment scorecard.",
        "",
    ]
    if not rows:
        lines.append("_Sweep produced no configs (no samples / all failed)._")
        return "\n".join(lines)

    lines += [
        "## Per-config scorecard",
        "",
        "| substrate | label | horizon | n | AUC | precision | E[r] | Sharpe | ECE | rec_tau |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['substrate']} | {r['label_variant']} | {r['horizon']} | "
            f"{r['n']} | {r['auc']:.4f} | {r['precision']} | "
            f"{r['expected_return']} | {r['paper_sharpe']} | {r['ece']} | "
            f"{r['recommended_tau']} |"
        )

    n_pass = sum(1 for r in rows if _gates_passed(r) == 4)
    lines += ["", f"**{n_pass} of {len(rows)} configs pass all 4 hard gates.**", ""]

    # Dollar - time delta per (label_variant, horizon) cell — the clean A/B:
    # label and horizon are held fixed, so the delta isolates bar structure.
    by_key = {(r["substrate"], r["label_variant"], r["horizon"]): r for r in rows}
    lines += [
        "## Dollar - time delta",
        "",
        "Each row holds label + horizon fixed, so the delta isolates the",
        "bar-structure effect. Positive = dollar bars beat time bars.",
        "",
        "| label | horizon | dAUC | dGates |",
        "|---|---|---|---|",
    ]
    for lv in LABEL_VARIANTS:
        for k in HORIZONS:
            d = by_key.get(("dollar", lv, k))
            t = by_key.get(("time", lv, k))
            if d is None or t is None:
                continue
            d_auc = d["auc"] - t["auc"]
            d_gates = _gates_passed(d) - _gates_passed(t)
            lines.append(f"| {lv} | {k} | {d_auc:+.4f} | {d_gates:+d} |")
    return "\n".join(lines)


def _run_one(substrate: str, label_variant: str, k: int,
             pids: list[str], sample_step: int) -> dict:
    """Run + scorecard one config. Returns a results row."""
    from sklearn.metrics import roc_auc_score

    from tools._scorecard._offclock_harness import run_config
    from tools.scorecard import compute_scorecard

    data = run_config(substrate, label_variant, k, pids, sample_step)
    report = compute_scorecard(
        data["scores"], data["labels"], data["returns"],
        data["fold_ids"], data["fold_spans_days"],
    )
    try:
        auc = float(roc_auc_score(data["labels"], data["scores"]))
    except Exception:
        auc = float("nan")
    return {
        "substrate": substrate,
        "label_variant": label_variant,
        "horizon": k,
        "n": data["n"],
        "auc": auc,
        "precision": report.gates_passed["precision"],
        "expected_return": report.gates_passed["expected_return"],
        "paper_sharpe": report.gates_passed["paper_sharpe"],
        "ece": report.gates_passed["ece"],
        "recommended_tau": report.recommended_operating_tau,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sweep 16 off-the-clock XGB configs through the scorecard"
    )
    parser.add_argument("--cache", default=_DEFAULT_CACHE,
                        help="cache for the survivorship-aware top-20 ranking")
    parser.add_argument("--pids", default=None,
                        help="comma-separated product ids (overrides --cache)")
    parser.add_argument("--sample-step", type=int, default=_DEFAULT_SAMPLE_STEP,
                        help="roll one sample every N bars (default: 24)")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help="results doc path (default: the SP2 results spec)")
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    else:
        from tools._scorecard._cv_harness import top_n_pids_from_cache
        pids = list(top_n_pids_from_cache(args.cache))

    grid = _config_grid()
    print(f"offclock_sweep: {len(grid)} configs, {len(pids)} products", flush=True)
    rows: list[dict] = []
    for i, (substrate, label_variant, k) in enumerate(grid, 1):
        tag = f"{substrate}/{label_variant}/h{k}"
        print(f"[{i}/{len(grid)}] {tag} ...", flush=True)
        try:
            rows.append(_run_one(substrate, label_variant, k, pids, args.sample_step))
        except Exception as e:
            print(f"    SKIPPED {tag}: {e}", flush=True)
            traceback.print_exc()

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(_render_results_doc(rows))
    print(f"wrote {args.out} ({len(rows)} configs)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_sweep.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
cd C:/Users/gl450/polymarket_app
git add backend/tools/offclock_sweep.py backend/tests/test_offclock_sweep.py CHANGELOG.md
git commit -m "feat(scorecard): off-the-clock 16-config sweep CLI

offclock_sweep.py loops the 16 (substrate x label x horizon) configs,
scorecards each via compute_scorecard, and writes the results table.
Completes the SP2 off-the-clock XGB track.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push
```

CHANGELOG.md sub-bullet under the EXISTING Session 58.71o section:

```markdown
- `tools/offclock_sweep.py` — CLI sweeping the 16 configs through `compute_scorecard`; writes the results doc.
```

---

## Operator steps (after the plan is implemented)

Run offline, during a training/backend-idle window. Not test steps.

1. Confirm SP1's dollar bars exist: `data/history/dollar/<pid>.parquet` for the top-20 (run SP1's `backfill_1m_candles` + `build_dollar_bars` first if not).
2. `cd backend && ../.venv/Scripts/python.exe -m tools.offclock_sweep` — trains and scorecards all 16 configs (~16 × 5 boosters; minutes to tens of minutes depending on `--sample-step`).
3. Review `docs/superpowers/specs/2026-05-21-offclock-sweep-results.md`: compare each dollar-bar config against its matched time-bar config (same label variant + horizon). If dollar configs pass gates their time-bar twins fail, the "time bars are the ceiling" hypothesis holds.

---

## Done criteria

- [ ] All new tests pass: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/test_offclock_harness.py tests/test_offclock_sweep.py -v`
- [ ] Full suite green via the pre-commit hook on every commit.
- [ ] No new dependencies.
- [ ] `cnn_agent.py` not modified.
- [ ] All commits pushed to `feat/gpu-coord-mirror`.
- [ ] CHANGELOG.md has the Session 58.71o section with one sub-bullet per task.
- [ ] Memory: append a session-log note to `coinbase_trader_session_log.md` recording the SP2 track ship.
