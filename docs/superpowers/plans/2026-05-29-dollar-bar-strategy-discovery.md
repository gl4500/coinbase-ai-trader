# Dollar-Bar Strategy-Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the strategy-discovery Phase 2 → 3 → 4 pipeline on 1h-aggregated dollar bars instead of fixed 1h time bars, producing `data/phase4_dbar/scorecard.md` for direct A/B comparison against the 1h `scorecard.md` (verdict ABORT).

**Architecture:** Introduce one new module that aggregates the existing 1h OHLCV into matched-count dollar bars (`info_bars.aggregate_dollar_bars`) and one CLI that maps it across the universe (`build_info_bars`). Refactor `profit_split.build_next_eligible` from millisecond/timestamp math to pure bar-index arithmetic, drop `_MS_PER_BAR`. Every other Phase 2/3/4 module reuses unchanged via the existing `--history-dir`/`--phase2-dir`/`--phase3-dir`/`--output-dir` CLI flags. Output writes to parallel `*_dbar` namespaces; the 1h baseline is preserved.

**Tech Stack:** Python 3.11+, pandas, pyarrow, PyTorch (CUDA for Phase 3 mining only), pytest. No new dependencies.

---

## Spec reference

`docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md` (uncommitted; included in Task 1's first commit).

## Pre-flight: branch setup (one-time)

The shared working tree currently holds 5-day-stale foreign WIP (`backend/agents/cnn_agent.py`, `backend/agents/exit_watcher.py`, `backend/main.py`, `backend/agents/exit_thresholds.py`, three test files, several frontend components). It **must not** be committed or modified. A `git checkout` to a different branch in the shared tree would carry that WIP across branches; the safe pattern is an isolated worktree off `origin/main`.

- [ ] **Step 1: Create isolated worktree off `origin/main`**

```bash
cd C:/Users/gl450/polymarket_app
git fetch origin --quiet
git worktree add --track -b feat/dollar-bar-strategy-discovery .wt-dbar origin/main
cd .wt-dbar
git rev-parse --abbrev-ref HEAD   # → feat/dollar-bar-strategy-discovery
git status --short                # → clean
```

Expected: a clean checkout at `C:/Users/gl450/polymarket_app/.wt-dbar/` on a fresh branch, with no foreign WIP.

- [ ] **Step 2: Copy the design spec into the worktree**

```bash
# from the main working tree (where the spec was authored)
cp ../docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md \
   docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md
```

- [ ] **Step 3: Copy this plan into the worktree**

```bash
cp ../docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md \
   docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md
```

**All subsequent `cd`/path references in this plan are inside `.wt-dbar/`.** All commits use surgical pathspec (`git commit -- <explicit paths>`); never `git commit -a` / `-am`. Push with `git push -u origin feat/dollar-bar-strategy-discovery` on the first push.

## File structure

| File | Role |
|---|---|
| `backend/tools/strategy_discovery/info_bars.py` (NEW) | Pure function: aggregate a 1h OHLCV DataFrame into matched-count dollar bars. No I/O. |
| `backend/tools/strategy_discovery/build_info_bars.py` (NEW) | CLI: read universe JSON → load each pid's 1h parquet → `aggregate_dollar_bars` → write `data/history/dollar_1h/<pid>.parquet`. |
| `backend/tools/strategy_discovery/profit_split.py` (MODIFY) | `build_next_eligible(n_rows, horizon_bars)` returns `(idx + horizon).clamp_max(n)` on bar indices. Delete `_MS_PER_BAR`. |
| `backend/tools/strategy_discovery/mine_profiles.py` (MODIFY) | Line 275 call site: pass `n` (already defined line 268) instead of `ts_ms`. Drop the now-unused `ts_ms` tensor (line 272). |
| `backend/tools/strategy_discovery/_diag_mining_hang.py` (MODIFY) | Line 67 call site mirror update (untracked diagnostic; touch only the call site). |
| `backend/tests/tools/strategy_discovery/test_info_bars.py` (NEW) | Aggregation contract tests. |
| `backend/tests/tools/strategy_discovery/test_profit_split.py` (MODIFY) | Lines 60–63, 71–78, 89–92: replace `build_next_eligible(ts, …)` with `build_next_eligible(N, …)`; expected outputs unchanged (uniform 1h spacing gives identical bar-index result). |
| `backend/tests/tools/strategy_discovery/test_profit_tree.py` (MODIFY) | Two call sites: same `ts → N` substitution. |
| `docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md` | Committed in Task 1 (copied in Pre-flight). |
| `docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md` | This file; committed alongside Task 1. |

No other files change. `features.py`, `labels.py`, `tokenomic_stamp.py`, `purged_wf.py`, `build_phase2.py`, `mine_universe.py`, `build_phase4.py` reuse unchanged via existing CLI flags.

---

## Task 1: Spec/plan commit + `info_bars.aggregate_dollar_bars`

**Files:**
- Create: `backend/tools/strategy_discovery/info_bars.py`
- Create: `backend/tests/tools/strategy_discovery/test_info_bars.py`
- Commit: `docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md`
- Commit: `docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/tools/strategy_discovery/test_info_bars.py`:

```python
"""Tests for tools.strategy_discovery.info_bars (matched-count dollar bars)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.info_bars import aggregate_dollar_bars


def _mk_1h(starts, opens, highs, lows, closes, vols):
    return pd.DataFrame({
        "start":  np.asarray(starts,  dtype="int64"),
        "open":   np.asarray(opens,   dtype="float64"),
        "high":   np.asarray(highs,   dtype="float64"),
        "low":    np.asarray(lows,    dtype="float64"),
        "close":  np.asarray(closes,  dtype="float64"),
        "volume": np.asarray(vols,    dtype="float64"),
    })


def test_empty_input_returns_empty_with_full_schema():
    out = aggregate_dollar_bars(_mk_1h([], [], [], [], [], []))
    assert list(out.columns) == [
        "start", "end", "open", "high", "low", "close",
        "volume", "dollar_value", "n_1h",
    ]
    assert len(out) == 0


def test_zero_total_dollar_value_returns_empty():
    df = _mk_1h([1, 2, 3], [10, 10, 10], [10, 10, 10], [10, 10, 10], [10, 10, 10], [0, 0, 0])
    out = aggregate_dollar_bars(df)
    assert len(out) == 0


def test_emits_n1h_bars_when_dollar_value_is_flat():
    # Every 1h row carries identical dollar value → threshold = mean → 1 bar per row.
    df = _mk_1h(
        starts=[100, 200, 300, 400],
        opens=[1.0] * 4, highs=[1.0] * 4, lows=[1.0] * 4, closes=[1.0] * 4,
        vols=[10.0] * 4,
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 4
    assert out["start"].tolist() == [100, 200, 300, 400]
    assert out["end"].tolist() == [100, 200, 300, 400]
    assert out["n_1h"].tolist() == [1, 1, 1, 1]


def test_ohlc_integrity_when_two_rows_merge_into_one_bar():
    # Row 0 dollar_value=1, row 1 dollar_value=5 → threshold = 3.0; bar 1 closes on row 1.
    df = _mk_1h(
        starts=[10, 20],
        opens=[100.0, 105.0],
        highs=[110.0, 115.0],
        lows=[95.0, 100.0],
        closes=[105.0, 112.0],
        vols=[(1.0 / ((100 + 110 + 105) / 3.0)), (5.0 / ((105 + 115 + 112) / 3.0))],
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["start"] == 10
    assert row["end"] == 20
    assert row["open"] == pytest.approx(100.0)
    assert row["close"] == pytest.approx(112.0)
    assert row["high"] == pytest.approx(115.0)   # max over rows 0, 1
    assert row["low"] == pytest.approx(95.0)     # min over rows 0, 1
    assert row["n_1h"] == 2
    assert row["dollar_value"] == pytest.approx(6.0, abs=1e-9)


def test_residual_below_threshold_is_dropped():
    # Three rows, threshold = mean = (1 + 5 + 0.5) / 3 ≈ 2.167. Row 0 alone (1.0) < threshold;
    # rows 0+1 cumulative 6.0 ≥ threshold → bar closes on row 1. Row 2 (0.5) < threshold → dropped.
    df = _mk_1h(
        starts=[10, 20, 30],
        opens=[100.0, 100.0, 100.0],
        highs=[100.0, 100.0, 100.0],
        lows=[100.0, 100.0, 100.0],
        closes=[100.0, 100.0, 100.0],
        vols=[1.0 / 100.0, 5.0 / 100.0, 0.5 / 100.0],
    )
    out = aggregate_dollar_bars(df)
    assert len(out) == 1
    assert out.iloc[0]["end"] == 20
    assert out.iloc[0]["n_1h"] == 2


def test_volume_and_dollar_value_are_sums_over_merged_rows():
    df = _mk_1h(
        starts=[1, 2, 3],
        opens=[1.0] * 3, highs=[1.0] * 3, lows=[1.0] * 3, closes=[1.0] * 3,
        vols=[10.0, 20.0, 30.0],
    )
    out = aggregate_dollar_bars(df)
    # threshold = mean dollar_value = 20.0; each row's dv = its volume × 1.0.
    # Row 0 (10) < 20; rows 0+1 (30) ≥ 20 → bar 1 closes at row 1, sums vol=30, dv=30.
    # Row 2 (30) ≥ 20 → bar 2 closes at row 2, sums vol=30, dv=30.
    assert len(out) == 2
    assert out["volume"].tolist() == [pytest.approx(30.0), pytest.approx(30.0)]
    assert out["dollar_value"].tolist() == [pytest.approx(30.0), pytest.approx(30.0)]
    assert out["n_1h"].tolist() == [2, 1]


def test_start_field_is_monotonic_nondecreasing():
    rng = np.random.default_rng(7)
    n = 200
    starts = np.arange(n, dtype="int64") * 3600
    vols = rng.uniform(0.1, 10.0, size=n)
    df = _mk_1h(starts, [1.0] * n, [1.0] * n, [1.0] * n, [1.0] * n, vols)
    out = aggregate_dollar_bars(df)
    assert len(out) > 0
    arr = out["start"].to_numpy()
    assert np.all(np.diff(arr) >= 0)
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_info_bars.py -v
```

Expected: collection error or `ModuleNotFoundError: No module named 'tools.strategy_discovery.info_bars'`.

- [ ] **Step 3: Implement `info_bars.aggregate_dollar_bars`**

Create `backend/tools/strategy_discovery/info_bars.py`:

```python
"""Aggregate a 1h OHLCV DataFrame into matched-count dollar bars.

A dollar bar closes when the cumulative dollar value (volume x (H+L+C)/3) of
consecutive 1h rows crosses a threshold equal to total_dollar_value / n_1h_rows.
This makes the emitted bar count approximately equal to the source 1h-bar count,
holding sample size fixed and isolating the sampling clock as the only variable
that changes vs the 1h baseline.

Mirrors the accumulation contract of `tools.build_dollar_bars.dollar_bars_from_candles`,
fed 1h rows instead of 1m candles. Pure function, no I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_OUT_COLUMNS = (
    "start", "end", "open", "high", "low", "close",
    "volume", "dollar_value", "n_1h",
)


def aggregate_dollar_bars(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Aggregate time-ordered 1h OHLCV rows into matched-count dollar bars.

    Input: DataFrame with columns ``start`` (epoch seconds), ``open``, ``high``,
    ``low``, ``close``, ``volume``. Rows are assumed time-sorted.

    Output: DataFrame with columns ``start`` (first merged row's epoch s),
    ``end`` (closing merged row's epoch s), ``open`` (first merged row's open),
    ``high`` / ``low`` (max / min over merged rows), ``close`` (last merged row's
    close), ``volume`` / ``dollar_value`` (sums), ``n_1h`` (merged row count).

    The trailing sub-threshold residual is dropped. Returns an empty frame with
    the full schema when input is empty or total dollar value is non-positive.
    """
    if len(df_1h) == 0:
        return pd.DataFrame({c: [] for c in _OUT_COLUMNS})

    typical = (df_1h["high"] + df_1h["low"] + df_1h["close"]) / 3.0
    dv = (df_1h["volume"] * typical).to_numpy(dtype="float64")
    total = float(dv.sum())
    n_rows = len(df_1h)
    if total <= 0.0:
        return pd.DataFrame({c: [] for c in _OUT_COLUMNS})

    threshold = total / n_rows

    starts = df_1h["start"].to_numpy(dtype="int64")
    opens  = df_1h["open"].to_numpy(dtype="float64")
    highs  = df_1h["high"].to_numpy(dtype="float64")
    lows   = df_1h["low"].to_numpy(dtype="float64")
    closes = df_1h["close"].to_numpy(dtype="float64")
    vols   = df_1h["volume"].to_numpy(dtype="float64")

    bars: list[dict] = []
    acc_dv = 0.0
    acc_vol = 0.0
    bar_start = None
    bar_open = None
    bar_high = None
    bar_low = None
    n = 0

    for i in range(n_rows):
        if bar_start is None:
            bar_start = int(starts[i])
            bar_open = float(opens[i])
            bar_high = float(highs[i])
            bar_low = float(lows[i])
        else:
            if highs[i] > bar_high:
                bar_high = float(highs[i])
            if lows[i] < bar_low:
                bar_low = float(lows[i])
        acc_dv += float(dv[i])
        acc_vol += float(vols[i])
        n += 1

        if acc_dv >= threshold:
            bars.append({
                "start":        bar_start,
                "end":          int(starts[i]),
                "open":         bar_open,
                "high":         bar_high,
                "low":          bar_low,
                "close":        float(closes[i]),
                "volume":       acc_vol,
                "dollar_value": acc_dv,
                "n_1h":         n,
            })
            acc_dv = 0.0
            acc_vol = 0.0
            bar_start = None
            bar_open = None
            bar_high = None
            bar_low = None
            n = 0

    return pd.DataFrame(bars, columns=list(_OUT_COLUMNS))
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_info_bars.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit (spec + plan + new module + tests, surgical pathspec)**

```bash
git add -- \
  docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md \
  docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md \
  backend/tools/strategy_discovery/info_bars.py \
  backend/tests/tools/strategy_discovery/test_info_bars.py
git status --short                  # confirm ONLY those four paths are staged
git commit -m "$(cat <<'EOF'
feat(strategy-discovery): info_bars dollar-bar aggregation + design/plan docs

aggregate_dollar_bars merges time-ordered 1h OHLCV rows into matched-count
dollar bars (threshold = total dollar-value / n_1h_rows). Schema matches
build_dollar_bars (start, end, open, high, low, close, volume, dollar_value,
n_1h) so the existing Phase 2 history loader reads it unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- \
  docs/superpowers/specs/2026-05-29-dollar-bar-strategy-discovery-design.md \
  docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md \
  backend/tools/strategy_discovery/info_bars.py \
  backend/tests/tools/strategy_discovery/test_info_bars.py
git log -1 --stat                   # confirm only the four files landed
git push -u origin feat/dollar-bar-strategy-discovery
```

Pre-commit hook runs the full suite (~6 min). If it fails for reasons **unrelated** to the four files just changed, halt and notify the operator — do not `--no-verify` and do not stash the foreign WIP from the main working tree (it lives outside `.wt-dbar/`, so it shouldn't appear in the worktree's git status; if it does, investigate before continuing).

---

## Task 2: `build_info_bars` CLI

**Files:**
- Create: `backend/tools/strategy_discovery/build_info_bars.py`
- Create: `backend/tests/tools/strategy_discovery/test_build_info_bars.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/tools/strategy_discovery/test_build_info_bars.py`:

```python
"""Tests for the build_info_bars CLI orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.build_info_bars import build_info_bars_for_pid


def _write_history_parquet(path: Path, n_rows: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "start":  np.arange(n_rows, dtype="int64") * 3600,
        "open":   np.full(n_rows, 100.0),
        "high":   np.full(n_rows, 101.0),
        "low":    np.full(n_rows, 99.0),
        "close":  np.full(n_rows, 100.0),
        "volume": rng.uniform(1.0, 10.0, size=n_rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def test_build_for_pid_writes_parquet_matching_schema(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    _write_history_parquet(hist_dir / "TEST-USD.parquet", n_rows=80, seed=1)
    result = build_info_bars_for_pid("TEST-USD", hist_dir, out_dir)
    assert result["error"] is None
    out_path = out_dir / "TEST-USD.parquet"
    assert out_path.exists()
    cols = set(pq.read_table(out_path).column_names)
    assert {"start", "end", "open", "high", "low", "close",
            "volume", "dollar_value", "n_1h"} <= cols


def test_build_for_pid_missing_history_returns_error_no_file(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    hist_dir.mkdir()
    result = build_info_bars_for_pid("ABSENT-USD", hist_dir, out_dir)
    assert result["error"] == "missing history"
    assert not (out_dir / "ABSENT-USD.parquet").exists()


def test_build_for_pid_empty_aggregation_returns_error_no_file(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    # All-zero volume → degenerate; aggregate_dollar_bars returns empty.
    df = pd.DataFrame({
        "start":  np.arange(10, dtype="int64") * 3600,
        "open":   np.full(10, 100.0),
        "high":   np.full(10, 100.0),
        "low":    np.full(10, 100.0),
        "close":  np.full(10, 100.0),
        "volume": np.zeros(10),
    })
    (hist_dir).mkdir()
    df.to_parquet(hist_dir / "ZERO-USD.parquet", compression="snappy", index=False)
    result = build_info_bars_for_pid("ZERO-USD", hist_dir, out_dir)
    assert result["error"] == "empty aggregation"
    assert not (out_dir / "ZERO-USD.parquet").exists()
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_build_info_bars.py -v
```

Expected: `ModuleNotFoundError: No module named 'tools.strategy_discovery.build_info_bars'`.

- [ ] **Step 3: Implement `build_info_bars`**

Create `backend/tools/strategy_discovery/build_info_bars.py`:

```python
"""CLI: aggregate 1h history into matched-count dollar bars per universe pid.

Reads the universe JSON, loads each pid's 1h history parquet, runs
aggregate_dollar_bars, and writes data/history/dollar_1h/<pid>.parquet
in the same history schema (the existing Phase 2 loader reads it unchanged).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.info_bars import aggregate_dollar_bars  # noqa: E402

_DEFAULT_HISTORY_DIR = Path(BACKEND) / "data" / "history"
_DEFAULT_OUTPUT_DIR  = Path(BACKEND) / "data" / "history" / "dollar_1h"
_DEFAULT_UNIVERSE    = (Path(BACKEND).parent / "docs" / "superpowers"
                        / "specs" / "2026-05-23-universe-50.json")


def _pids_from_universe_json(universe_path: Path) -> List[str]:
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def build_info_bars_for_pid(pid: str, history_dir: Path, output_dir: Path) -> Dict:
    """Build matched-count dollar bars for one pid; write parquet on success."""
    src = Path(history_dir) / f"{pid}.parquet"
    if not src.exists():
        return {"pid": pid, "n_bars": 0, "n_1h": 0, "error": "missing history"}
    df = pq.read_table(src).to_pandas().sort_values("start").reset_index(drop=True)
    bars = aggregate_dollar_bars(df)
    if len(bars) == 0:
        return {"pid": pid, "n_bars": 0, "n_1h": int(len(df)),
                "error": "empty aggregation"}
    out_path = Path(output_dir) / f"{pid}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out_path, compression="snappy", index=False)
    return {"pid": pid, "n_bars": int(len(bars)), "n_1h": int(len(df)), "error": None}


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build matched-count dollar bars for a universe.")
    parser.add_argument("--universe",   default=str(_DEFAULT_UNIVERSE))
    parser.add_argument("--history-dir", default=str(_DEFAULT_HISTORY_DIR))
    parser.add_argument("--output-dir",  default=str(_DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    pids = _pids_from_universe_json(Path(args.universe))
    n_ok = 0
    n_err = 0
    for i, pid in enumerate(pids, 1):
        r = build_info_bars_for_pid(pid, Path(args.history_dir), Path(args.output_dir))
        if r["error"]:
            n_err += 1
            print(f"  [{i:3d}/{len(pids)}] {pid:14s}  ERROR: {r['error']}", flush=True)
        else:
            n_ok += 1
            print(f"  [{i:3d}/{len(pids)}] {pid:14s}  {r['n_bars']:6d} bars "
                  f"(from {r['n_1h']} 1h rows)", flush=True)
    print(f"  ok: {n_ok}  error: {n_err}", flush=True)
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_build_info_bars.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add -- \
  backend/tools/strategy_discovery/build_info_bars.py \
  backend/tests/tools/strategy_discovery/test_build_info_bars.py
git status --short
git commit -m "$(cat <<'EOF'
feat(strategy-discovery): build_info_bars CLI

Maps aggregate_dollar_bars over the universe JSON and writes per-pid parquets
to data/history/dollar_1h/, reusable by build_phase2 via its existing
--history-dir flag.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- \
  backend/tools/strategy_discovery/build_info_bars.py \
  backend/tests/tools/strategy_discovery/test_build_info_bars.py
git log -1 --stat
git push
```

---

## Task 3: `profit_split.build_next_eligible` bar-index refactor

**Files:**
- Modify: `backend/tools/strategy_discovery/profit_split.py` (lines 18 + 34–43)
- Modify: `backend/tests/tools/strategy_discovery/test_profit_split.py` (lines 60, 62, 71, 78, 89, 92)
- Modify: `backend/tests/tools/strategy_discovery/test_profit_tree.py` (two call sites)

- [ ] **Step 1: Rewrite the 5 test call sites first (TDD red)**

In `backend/tests/tools/strategy_discovery/test_profit_split.py`, replace the three test functions that call `build_next_eligible`:

```python
def test_concurrency_max_1_skips_overlapping_entry():
    labels = torch.tensor([1.0, 10.0, 100.0, 2.0, 50.0], dtype=torch.float64)
    next_eligible = build_next_eligible(5, horizon_bars=3)
    assert next_eligible.tolist() == [3, 4, 5, 5, 5]
    subset = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    total = walk_and_sum(subset, next_eligible, labels)
    assert total.item() == pytest.approx(3.0, abs=1e-12)


def test_split_metric_picks_higher_pnl_subgroup():
    N = 100
    horizon_bars = 1
    features = torch.zeros((N, 1), dtype=torch.float64)
    features[50:, 0] = 1.0
    labels = torch.zeros(N, dtype=torch.float64)
    labels[:50] = -0.02
    labels[50:] = 0.10
    next_eligible = build_next_eligible(N, horizon_bars=horizon_bars)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is not None
    assert result.feature == 0
    assert 0.0 < result.threshold < 1.0
    assert result.score == pytest.approx(5.0, abs=1e-9)


def test_no_profitable_split_returns_none():
    N = 30
    features = torch.linspace(0.0, 1.0, N, dtype=torch.float64).unsqueeze(1)
    labels = torch.full((N,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(N, horizon_bars=1)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is None
```

In `backend/tests/tools/strategy_discovery/test_profit_tree.py`, change the two existing call sites:

```python
# both occurrences:
next_eligible = build_next_eligible(len(ts), horizon_bars=1)
```

(leave the `ts` variable alone if it's used elsewhere in the test; otherwise inline `N` for clarity).

- [ ] **Step 2: Run failing tests**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_profit_split.py tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: failures with `TypeError: build_next_eligible() … received tensor` (the rewritten tests pass `int N`; the old implementation expects a tensor `ts_ms`).

- [ ] **Step 3: Refactor `build_next_eligible` to bar-index**

Edit `backend/tools/strategy_discovery/profit_split.py`. Replace lines 16–43 (the `_MS_PER_BAR` constant + the docstring + `build_next_eligible` body) with:

```python
import torch

# _MS_PER_BAR removed — bars are no longer assumed to be fixed-width in time.


@dataclass
class SplitResult:
    """Best-split outcome from one node's candidate scan.

    Treat as immutable — not frozen because the `left_mask` torch.Tensor is
    unhashable, so `frozen=True` would create a misleading hashability contract.
    """
    feature: int
    threshold: float
    left_mask: torch.Tensor   # (n,) bool — True for rows going to left subtree
    score: float              # the split_metric value (cum_pnl of better side)


def build_next_eligible(n_rows: int, horizon_bars: int) -> torch.Tensor:
    """Returns `next_eligible[i] = min(i + horizon_bars, n_rows)` on bar indices.

    Bar-index concurrency: an entry at row i opens a position whose minimum
    next-eligible entry is row i + horizon_bars (clamped to n). No timestamp
    arithmetic, no fixed-width-bar assumption.
    """
    n = int(n_rows)
    h = int(horizon_bars)
    return (torch.arange(n) + h).clamp_max(n)
```

(Keep `walk_and_sum`, `_quantile_thresholds`, and `best_split` unchanged — they consume `next_eligible` as indices already.)

Also update the module docstring's bar-width remark if present (search for any mention of "_MS_PER_BAR" or "1h bar" and remove).

- [ ] **Step 4: Run tests and verify they pass**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_profit_split.py tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: all profit_split + profit_tree tests pass (the bar-index outputs equal the prior ms-based outputs on the uniform 1h spacing used in tests, so expected lists unchanged).

- [ ] **Step 5: Commit**

```bash
git add -- \
  backend/tools/strategy_discovery/profit_split.py \
  backend/tests/tools/strategy_discovery/test_profit_split.py \
  backend/tests/tools/strategy_discovery/test_profit_tree.py
git status --short
git commit -m "$(cat <<'EOF'
refactor(strategy-discovery): profit_split.build_next_eligible to bar-index

Drops _MS_PER_BAR and the fixed-width 1h assumption. next_eligible is now
(idx + horizon_bars).clamp_max(n), pure bar-index arithmetic. Downstream
concurrency math in walk_and_sum/best_split already treats next_eligible as
indices, so behavior on uniform 1h spacing is identical (tests' expected
outputs unchanged).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- \
  backend/tools/strategy_discovery/profit_split.py \
  backend/tests/tools/strategy_discovery/test_profit_split.py \
  backend/tests/tools/strategy_discovery/test_profit_tree.py
git log -1 --stat
git push
```

---

## Task 4: Update `mine_profiles` + `_diag` call sites

**Files:**
- Modify: `backend/tools/strategy_discovery/mine_profiles.py` (lines 272 + 275)
- Modify: `backend/tools/strategy_discovery/_diag_mining_hang.py` (lines 64–67)

- [ ] **Step 1: Update `mine_profiles.py`**

In `backend/tools/strategy_discovery/mine_profiles.py`, replace lines 272–275:

```python
    labels = torch.tensor(df[label_col].to_numpy(dtype="float64"), device=dev)
    features = torch.tensor(df[list(_FEATURE_COLUMNS)].to_numpy(dtype="float64"), device=dev)
    next_eligible = build_next_eligible(n, horizon_bars=int(horizon))
```

(Delete the `ts_ms = torch.tensor(df["ts"].to_numpy(...), device=dev)` line — it was only consumed by `build_next_eligible`. Confirm `ts_ms` is not referenced elsewhere in the file: `grep -n ts_ms backend/tools/strategy_discovery/mine_profiles.py` should return nothing after the edit.)

- [ ] **Step 2: Update `_diag_mining_hang.py`**

In `backend/tools/strategy_discovery/_diag_mining_hang.py`, replace lines 64–67:

```python
    dev = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    labels = torch.tensor(df[label_col].to_numpy(dtype="float64"), device=dev)
    features = torch.tensor(df[list(FEATS)].to_numpy(dtype="float64"), device=dev)
    next_eligible = build_next_eligible(len(df), horizon_bars=H)
```

(Drop the `ts_ms` line for the same reason.)

- [ ] **Step 3: Run the strategy-discovery test suite to confirm nothing regressed**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/ -v
```

Expected: all strategy_discovery tests pass (includes the integration test for `mine_profiles_for_pid_horizon`).

- [ ] **Step 4: Commit**

```bash
git add -- \
  backend/tools/strategy_discovery/mine_profiles.py \
  backend/tools/strategy_discovery/_diag_mining_hang.py
git status --short
git commit -m "$(cat <<'EOF'
refactor(strategy-discovery): bar-index next_eligible call-site updates

Pass n (data length) to build_next_eligible instead of the ts_ms tensor.
Drop now-unused ts_ms allocations in mine_profiles_for_pid_horizon and the
diagnostic probe.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- \
  backend/tools/strategy_discovery/mine_profiles.py \
  backend/tools/strategy_discovery/_diag_mining_hang.py
git log -1 --stat
git push
```

---

## Task 5: End-to-end integration smoke test

**Files:**
- Create: `backend/tests/tools/strategy_discovery/test_dbar_integration.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_dbar_integration.py`:

```python
"""Integration smoke: info-bars → build_phase2 on a tiny fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tools.strategy_discovery.build_info_bars import build_info_bars_for_pid
from tools.strategy_discovery.build_phase2 import build_phase2_for_pid
from tools.strategy_discovery.tokenomic_stamp import _TOKENOMIC_COLUMNS


def _seed_1h_history(path: Path, n: int = 800, seed: int = 5) -> None:
    """Synthetic 1h OHLCV with realistic trend + volume variation."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    opens = np.concatenate([[100.0], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=n)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=n)))
    vols = rng.uniform(50.0, 500.0, size=n)
    df = pd.DataFrame({
        "start":  np.arange(n, dtype="int64") * 3600 + 1_700_000_000,
        "open":   opens, "high": highs, "low": lows, "close": closes,
        "volume": vols,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def _seed_marketcap(path: Path, n_days: int = 60, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "start":       np.arange(n_days, dtype="int64") * 86400 + 1_700_000_000,
        "market_cap":  rng.uniform(1e9, 5e9, size=n_days),
        "volume_24h":  rng.uniform(1e7, 5e7, size=n_days),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def _seed_supply(path: Path, pid: str) -> None:
    df = pd.DataFrame({
        "pid":         [pid],
        "circulating": [1_000_000_000.0],
        "total":       [1_500_000_000.0],
        "max_supply":  [2_000_000_000.0],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def test_info_bars_to_phase2_smoke(tmp_path: Path):
    pid = "TEST-USD"
    hist_1h     = tmp_path / "history"
    hist_dbar   = tmp_path / "history" / "dollar_1h"
    mc_dir      = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    phase2_dir  = tmp_path / "phase2_dbar"

    _seed_1h_history(hist_1h / f"{pid}.parquet", n=800)
    _seed_marketcap(mc_dir / f"{pid}.parquet", n_days=60)
    _seed_supply(supply_path, pid)

    r1 = build_info_bars_for_pid(pid, hist_1h, hist_dbar)
    assert r1["error"] is None
    assert r1["n_bars"] > 0

    r2 = build_phase2_for_pid(pid, hist_dbar, mc_dir, supply_path, phase2_dir)
    assert r2.error is None
    assert r2.rows_written > 0

    out = pq.read_table(phase2_dir / f"{pid}.parquet").to_pandas()
    # OHLCV + trend features
    for col in ("ts", "open", "high", "low", "close",
                "price_over_ema200", "atr14_pct", "ret_24h_sign"):
        assert col in out.columns
    # Tokenomic stamp columns
    for col in _TOKENOMIC_COLUMNS:
        assert col in out.columns
    # Labels finite for at least h24/h72/h168
    for h in (24, 72, 168):
        col = f"label_h{h}"
        assert col in out.columns
        # Most rows should have a finite label (tail rows are NaN by horizon)
        assert int(np.isfinite(out[col]).sum()) >= int(0.5 * len(out))
```

- [ ] **Step 2: Run and verify pass**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/tools/strategy_discovery/test_dbar_integration.py -v
```

Expected: 1 passed.

- [ ] **Step 3: Run the full backend suite to confirm no regression**

```bash
cd backend
../.venv/Scripts/python.exe -m pytest tests/ -q
```

Expected: full suite green.

- [ ] **Step 4: Commit**

```bash
git add -- backend/tests/tools/strategy_discovery/test_dbar_integration.py
git status --short
git commit -m "$(cat <<'EOF'
test(strategy-discovery): dollar-bar integration smoke (info_bars -> build_phase2)

Synthesizes 1h OHLCV + daily marketcap + supply snapshot in tmp_path, runs the
info_bars -> build_phase2 chain, and asserts the resulting phase2_dbar parquet
has OHLC + trend features + tokenomic stamp columns + finite labels at h24/72/168.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_dbar_integration.py
git log -1 --stat
git push
```

---

## Task 6: End-to-end production run

This task runs the actual pipeline. It is **operational**, not a code task — no commits. It is gated on:
- 8001 backend idle (per the operator's `no-pytest/commit-while-trading` rule) — the actual run does **not** trigger pytest, but the operator should confirm before kicking off the long Phase 3 mining (which heavily uses GPU + CPU and may contend with the live backend).
- CUDA device free.

- [ ] **Step 1: Build matched-count dollar bars for the universe**

```bash
cd backend
../.venv/Scripts/python.exe -u -m tools.strategy_discovery.build_info_bars \
  --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
  --history-dir data/history \
  --output-dir data/history/dollar_1h
```

Expected runtime: < 1 min. Output: ~50 parquets under `data/history/dollar_1h/`.

- [ ] **Step 2: Build Phase 2 from dollar bars**

```bash
cd backend
../.venv/Scripts/python.exe -u -m tools.strategy_discovery.build_phase2 \
  --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
  --history-dir data/history/dollar_1h \
  --output-dir  data/phase2_dbar
```

Expected: ~2 min. Output: `data/phase2_dbar/<pid>.parquet` per universe pid.

- [ ] **Step 3: Mine Phase 3 (the long pole)**

```bash
cd backend
../.venv/Scripts/python.exe -u -m tools.strategy_discovery.mine_universe \
  --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
  --phase2-dir data/phase2_dbar \
  --output-dir data/phase3_dbar \
  --device cuda --seed 42
```

Expected: ~20 hours on RTX 2060 (similar to the 1h baseline since bar count is matched). Per-task progress lines stream to stdout. Run via the background-task tooling so the harness notifies on completion.

- [ ] **Step 4: Build Phase 4 scorecard**

```bash
cd backend
../.venv/Scripts/python.exe -u -m tools.strategy_discovery.build_phase4 \
  --phase3-dir data/phase3_dbar \
  --phase2-dir data/phase2_dbar \
  --output-dir data/phase4_dbar \
  --seed 42
```

Expected: ~1–3 hr (compute-bound portfolio beam-search across 25-ish profiles × caps 3/4/5). Output: `data/phase4_dbar/scorecard.md`.

- [ ] **Step 5: Review the scorecard and compare to the 1h baseline**

```bash
diff -u backend/data/phase4/scorecard.md backend/data/phase4_dbar/scorecard.md | head -60
```

Verdict will be either "deploy at N={3|4|5}" or another documented ABORT. Either outcome answers the research question. Surface the verdict + per-cap raw-vs-deflated numbers to the operator.

---

## Self-review checklist (already run on this plan)

- **Spec coverage:** every "MODIFY"/"NEW" entry in the spec's file map has a corresponding task. The bar-count convention + max-hold caveat from the spec are documented but require no code changes, so no task is needed for them.
- **No placeholders:** all code blocks are complete; no "TBD", "TODO", "similar to Task N", "add error handling" phrasing.
- **Type consistency:** `aggregate_dollar_bars(df_1h)` returns the same 9-column schema used by `build_info_bars_for_pid` and asserted by `test_build_info_bars`; `build_next_eligible(n_rows, horizon_bars)` matches the signature used in `mine_profiles`, `_diag`, and the rewritten tests.

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-29-dollar-bar-strategy-discovery.md` (uncommitted). Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, two-stage review (spec compliance → code quality) between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batch with checkpoints for review.

Pick one when ready to start.
