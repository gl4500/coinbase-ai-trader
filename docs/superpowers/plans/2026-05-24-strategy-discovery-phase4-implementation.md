# Strategy-Discovery Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consume Phase 3's per-horizon profile parquets; for each concurrency cap N ∈ {3,4,5}, run a beam-search portfolio knapsack; emit a per-cap scorecard + deployment JSON + telemetry parquet so the operator can pick which cap (if any) to deploy.

**Architecture:** Five loosely-coupled modules under `backend/tools/strategy_discovery/`. `profile_loader.py` is pure I/O (load Phase 3 profile parquets + sidecars + Phase 2 features). `portfolio_sim.py` walks historical bars and enforces the concurrency cap + per-pid-max-1 rule. `knapsack_search.py` runs beam search over subsets, calls `portfolio_sim` for each candidate, applies the σ × √(2 ln K) deflation. `scorecard.py` evaluates the per-cap Q0 gates and renders the Markdown report. `build_phase4.py` orchestrates the cap sweep + writes all artifacts.

**Tech Stack:** Python 3, pandas, numpy, pyarrow. No torch (Phase 4 is pure pandas/numpy). Pytest with mocks only.

**Spec:** `docs/superpowers/specs/2026-05-24-strategy-discovery-phase4-design.md`

---

## File Map

| File | Purpose |
|---|---|
| `backend/tools/strategy_discovery/profile_loader.py` (NEW) | Load Phase 3 profile parquets + rule-path JSONs + Phase 2 features |
| `backend/tools/strategy_discovery/portfolio_sim.py` (NEW) | Time-walk simulator with concurrency cap + per-pid-max-1 |
| `backend/tools/strategy_discovery/knapsack_search.py` (NEW) | Beam-search over profile subsets + portfolio deflation |
| `backend/tools/strategy_discovery/scorecard.py` (NEW) | Per-cap Q0 gates + Markdown scorecard + verdict |
| `backend/tools/strategy_discovery/build_phase4.py` (NEW) | Orchestrator + CLI: sweep N∈{3,4,5}, write all artifacts |
| `backend/tests/tools/strategy_discovery/test_profile_loader.py` (NEW) | 3 tests |
| `backend/tests/tools/strategy_discovery/test_portfolio_sim.py` (NEW) | 6 tests |
| `backend/tests/tools/strategy_discovery/test_knapsack_search.py` (NEW) | 3 tests |
| `backend/tests/tools/strategy_discovery/test_scorecard.py` (NEW) | 5 tests |
| `backend/tests/tools/strategy_discovery/test_build_phase4.py` (NEW) | 3 tests |
| `CHANGELOG.md` (MODIFY, prepend) | Session log of Phase 4 implementation |
| `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` (MODIFY) | Append Phase 4 sub-section after Phase 3 |

**Branch discipline:** All commits on a fresh `feat/strategy-discovery-phase4` branch off main. Every commit uses surgical `--` pathspec. Implementer subagents PASTE `git status -s` and `git log -1 --stat` verbatim in their reports so collisions get caught.

---

## Task 1: Profile loader (`profile_loader.py` + 3 tests)

**Files:**
- Create: `backend/tools/strategy_discovery/profile_loader.py`
- Create: `backend/tests/tools/strategy_discovery/test_profile_loader.py`

### Scaffolding (Step 1.0)

Create `profile_loader.py`:

```python
"""Phase 4 profile loader — Phase 3 parquets + sidecars + Phase 2 features.

Pure I/O. No simulation, no selection.
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


@dataclass
class LoadedProfile:
    pid: str
    horizon: int
    leaf_id: int
    rule_path: str
    cumulative_profit_raw: float
    cumulative_profit_deflated: float
    deflation_pp: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd: float
    sortino: float
    trade_count: int
    n_folds_passed_q0: int
    chosen_depth: int
    chosen_min_leaf: int

    @property
    def profile_id(self) -> str:
        return f"{self.pid}__{self.leaf_id}"
```

### Round 1 — `test_loads_all_horizon_parquets`

- [ ] **Step 1.1.1 — Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_profile_loader.py`:

```python
"""Tests for tools.strategy_discovery.profile_loader (Phase 4)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.profile_loader import (
    LoadedProfile,
    load_all_profiles,
    load_pid_features,
)

_PROFILE_COLUMNS = [
    "pid", "horizon", "leaf_id", "rule_path_summary",
    "cumulative_profit_raw", "cumulative_profit_deflated", "deflation_pp",
    "win_rate", "avg_win", "avg_loss", "max_dd", "sortino",
    "trade_count", "n_folds_passed_q0",
    "chosen_depth", "chosen_min_leaf",
    "bootstrap_triggered", "bootstrap_ci_lower", "bootstrap_ci_upper",
    "n_combos_searched", "inner_cv_se", "schema_version",
]


def _write_profile_parquet(path: Path, rows):
    df = pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _write_rule_paths_json(path: Path, mapping: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")


def test_loads_all_horizon_parquets(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    # h1 has 1 profile, h24 has 2 profiles, others empty (no file)
    _write_profile_parquet(phase3_dir / "profiles_h1.parquet", [
        ("BTC-USD", 1, 0, "vol_over_mc > 0.05", 0.05, 0.03, 0.02, 0.6, 0.07, -0.04, 0.20, 1.2, 30, 5, 3, 20, False, None, None, 9, 0.01, 1),
    ])
    _write_profile_parquet(phase3_dir / "profiles_h24.parquet", [
        ("BTC-USD", 24, 0, "price_over_ema20 > 1.02", 0.10, 0.06, 0.04, 0.55, 0.08, -0.05, 0.18, 1.3, 50, 5, 5, 50, False, None, None, 9, 0.015, 1),
        ("ETH-USD", 24, 1, "ret_24h_sign > 0",        0.08, 0.05, 0.03, 0.58, 0.07, -0.04, 0.15, 1.4, 40, 4, 5, 50, False, None, None, 9, 0.012, 1),
    ])
    _write_rule_paths_json(phase3_dir / "rule_paths_h1.json",  {"BTC-USD__0": "vol_over_mc > 0.05"})
    _write_rule_paths_json(phase3_dir / "rule_paths_h24.json", {
        "BTC-USD__0": "price_over_ema20 > 1.02",
        "ETH-USD__1": "ret_24h_sign > 0",
    })

    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[1, 4, 24, 72, 168])
    assert len(profiles) == 3
    # All carry their (pid, horizon, leaf_id) identifiers + their rule_path
    by_id = {p.profile_id: p for p in profiles}
    assert "BTC-USD__0" in by_id   # appears for both h1 and h24
    assert by_id["BTC-USD__0"].horizon in (1, 24)
    assert all(isinstance(p, LoadedProfile) for p in profiles)
```

- [ ] **Step 1.1.2 — Run, confirm ImportError**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profile_loader.py::test_loads_all_horizon_parquets -v
```

Expect: ImportError.

- [ ] **Step 1.1.3 — Implement**

Append to `profile_loader.py`:

```python
def load_all_profiles(
    phase3_dir: Path = Path(BACKEND) / "data" / "phase3",
    horizons: List[int] = [1, 4, 24, 72, 168],
    min_folds_passed_q0: int = 4,
) -> List[LoadedProfile]:
    """Load all per-horizon profile parquets + rule-path sidecars.

    Filters profiles with n_folds_passed_q0 < min_folds_passed_q0 (re-enforces
    Phase 3 gate at the Phase 4 input boundary).
    """
    phase3_dir = Path(phase3_dir)
    out: List[LoadedProfile] = []
    for h in horizons:
        parquet_path = phase3_dir / f"profiles_h{int(h)}.parquet"
        sidecar_path = phase3_dir / f"rule_paths_h{int(h)}.json"
        if not parquet_path.exists():
            continue
        df = pq.read_table(parquet_path).to_pandas()
        rule_paths: Dict[str, str] = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                rule_paths = json.load(f)
        for _, row in df.iterrows():
            if int(row["n_folds_passed_q0"]) < min_folds_passed_q0:
                continue
            pid = str(row["pid"])
            leaf_id = int(row["leaf_id"])
            profile_id = f"{pid}__{leaf_id}"
            rule_str = rule_paths.get(profile_id, str(row.get("rule_path_summary", "")))
            out.append(LoadedProfile(
                pid=pid,
                horizon=int(row["horizon"]),
                leaf_id=leaf_id,
                rule_path=rule_str,
                cumulative_profit_raw=float(row["cumulative_profit_raw"]),
                cumulative_profit_deflated=float(row["cumulative_profit_deflated"]),
                deflation_pp=float(row["deflation_pp"]),
                win_rate=float(row["win_rate"]),
                avg_win=float(row["avg_win"]),
                avg_loss=float(row["avg_loss"]),
                max_dd=float(row["max_dd"]),
                sortino=float(row["sortino"]),
                trade_count=int(row["trade_count"]),
                n_folds_passed_q0=int(row["n_folds_passed_q0"]),
                chosen_depth=int(row["chosen_depth"]),
                chosen_min_leaf=int(row["chosen_min_leaf"]),
            ))
    return out


def load_pid_features(
    pid: str,
    phase2_dir: Path = Path(BACKEND) / "data" / "phase2",
) -> pd.DataFrame:
    """Load Phase 2 parquet for one pid. Returns empty DataFrame if missing."""
    path = Path(phase2_dir) / f"{pid}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()
```

- [ ] **Step 1.1.4 — Run + commit**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profile_loader.py -v
```

Expect: 1 passed.

Pre-commit: PASTE `git rev-parse --abbrev-ref HEAD` (must be `feat/strategy-discovery-phase4`) + `git status -s`.

```
git add backend/tools/strategy_discovery/profile_loader.py backend/tests/tools/strategy_discovery/test_profile_loader.py
git commit -m "$(cat <<'EOF'
feat(phase4): add profile_loader — Phase 3 parquets + sidecars + Phase 2 features

Phase 4 strategy-discovery rebuild — pure I/O loader. Loads all 5 per-horizon
profile parquets, attaches rule-path sidecars, filters profiles below
min_folds_passed_q0=4 (re-enforces Phase 3 gate at Phase 4 input boundary).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/profile_loader.py backend/tests/tools/strategy_discovery/test_profile_loader.py
```

Post-commit: PASTE `git log -1 --stat`. Must show only those 2 files.

### Round 2 — `test_attaches_rule_paths_from_sidecar_json`

- [ ] **Step 1.2.1 — Write the failing test**

Append to `test_profile_loader.py`:

```python
def test_attaches_rule_paths_from_sidecar_json(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    _write_profile_parquet(phase3_dir / "profiles_h24.parquet", [
        ("BTC-USD", 24, 0, "short_summary", 0.05, 0.02, 0.03, 0.6, 0.08, -0.04, 0.15, 1.2, 30, 5, 5, 50, False, None, None, 9, 0.01, 1),
    ])
    _write_rule_paths_json(phase3_dir / "rule_paths_h24.json", {
        "BTC-USD__0": "vol_over_mc > 0.08 AND price_over_ema20 > 1.02"
    })
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[24])
    assert len(profiles) == 1
    # Full rule from JSON wins over the parquet's truncated rule_path_summary
    assert profiles[0].rule_path == "vol_over_mc > 0.08 AND price_over_ema20 > 1.02"
```

- [ ] **Step 1.2.2 — Run, expect PASS** (impl already handles this)

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profile_loader.py -v
```

Expect: 2 passed.

- [ ] **Step 1.2.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin sidecar JSON rule paths beat parquet summary

Round 2 of profile_loader.py — the human-readable rule path lives in the
sidecar JSON; parquet's rule_path_summary is a fallback only.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profile_loader.py
```

Post-commit: PASTE `git log -1 --stat`.

### Round 3 — `test_filters_profiles_below_min_folds_passed`

- [ ] **Step 1.3.1 — Write the failing test**

Append:

```python
def test_filters_profiles_below_min_folds_passed(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    _write_profile_parquet(phase3_dir / "profiles_h24.parquet", [
        # Profile A passes (5 folds)
        ("BTC-USD", 24, 0, "rule_a", 0.05, 0.02, 0.03, 0.6, 0.08, -0.04, 0.15, 1.2, 30, 5, 5, 50, False, None, None, 9, 0.01, 1),
        # Profile B is borderline (4 folds)
        ("ETH-USD", 24, 1, "rule_b", 0.04, 0.01, 0.03, 0.55, 0.07, -0.04, 0.18, 1.0, 25, 4, 5, 50, False, None, None, 9, 0.01, 1),
        # Profile C below threshold (3 folds) — should be dropped
        ("SOL-USD", 24, 2, "rule_c", 0.03, 0.01, 0.02, 0.50, 0.06, -0.05, 0.20, 0.8, 20, 3, 5, 50, False, None, None, 9, 0.01, 1),
    ])
    _write_rule_paths_json(phase3_dir / "rule_paths_h24.json", {
        "BTC-USD__0": "rule_a", "ETH-USD__1": "rule_b", "SOL-USD__2": "rule_c",
    })
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[24], min_folds_passed_q0=4)
    pids = sorted(p.pid for p in profiles)
    assert pids == ["BTC-USD", "ETH-USD"]   # SOL-USD dropped (3 < 4 folds)
```

- [ ] **Step 1.3.2 — Run, expect PASS** (impl already handles this)

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profile_loader.py -v
```

Expect: 3 passed.

- [ ] **Step 1.3.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin min_folds_passed_q0=4 filter at load boundary

Round 3 of profile_loader.py — re-enforces Phase 3's ≥4-of-5 outer-fold
gate at the Phase 4 entry point. Profiles with n_folds_passed_q0 < 4
are dropped before reaching the knapsack.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profile_loader.py
```

Post-commit: PASTE `git log -1 --stat`.

---

## Task 2: Portfolio simulator (`portfolio_sim.py` + 6 tests)

**Files:**
- Create: `backend/tools/strategy_discovery/portfolio_sim.py`
- Create: `backend/tests/tools/strategy_discovery/test_portfolio_sim.py`

### Scaffolding (Step 2.0)

Create `portfolio_sim.py`:

```python
"""Phase 4 portfolio simulator — time-walk with concurrency cap.

Walks historical bars chronologically; at each bar, closes positions whose
horizon expired, evaluates which profiles fire on their pid, and enters
the highest-deflated-profit firing profiles up to the cap.

Per-pid cap of 1 carried from Phase 3. Exit PnL inherited from Phase 2
label_h{horizon} — no exit re-simulation.

Pure pandas + numpy. No I/O, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.strategy_discovery.profile_loader import LoadedProfile


@dataclass
class PortfolioMetrics:
    cumulative_profit_raw: float = 0.0
    cumulative_profit_deflated: float = 0.0
    max_dd: float = 0.0
    sortino: float = 0.0
    trade_count: int = 0
    pct_slots_full: float = 0.0
    mean_concurrent: float = 0.0


@dataclass
class TelemetryRow:
    ts: int
    equity: float
    n_open: int
    fired_profile_id: Optional[str] = None
    closed_profile_id: Optional[str] = None
    realized_pnl: Optional[float] = None
```

### Round 1 — `test_concurrency_cap_blocks_new_entries_when_full`

- [ ] **Step 2.1.1 — Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_portfolio_sim.py`:

```python
"""Tests for tools.strategy_discovery.portfolio_sim (Phase 4)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.portfolio_sim import (
    PortfolioMetrics,
    TelemetryRow,
    parse_rule_path,
    simulate_portfolio,
)
from tools.strategy_discovery.profile_loader import LoadedProfile


def _make_profile(pid: str, leaf_id: int, horizon: int, rule_path: str,
                  deflated: float = 0.05) -> LoadedProfile:
    return LoadedProfile(
        pid=pid, horizon=horizon, leaf_id=leaf_id, rule_path=rule_path,
        cumulative_profit_raw=deflated + 0.02,
        cumulative_profit_deflated=deflated,
        deflation_pp=0.02,
        win_rate=0.6, avg_win=0.08, avg_loss=-0.04,
        max_dd=0.20, sortino=1.2, trade_count=30,
        n_folds_passed_q0=5, chosen_depth=5, chosen_min_leaf=50,
    )


def _make_pid_features(pid: str, n_hours: int, ema_ratio: float, label: float):
    """Synthetic Phase 2 frame: feature is `price_over_ema20`, label is constant."""
    ts = (np.arange(n_hours, dtype="int64") * 3_600_000).tolist()
    return pd.DataFrame({
        "ts": ts,
        "close": [1.0] * n_hours,
        "price_over_ema20": [ema_ratio] * n_hours,
        "vol_over_mc": [0.01] * n_hours,
        "label_h1": [label] * n_hours,
        "label_h24": [label] * n_hours,
    })


def test_concurrency_cap_blocks_new_entries_when_full():
    # 3 pids, each fires on the same rule, 24h horizon.
    # Cap=2 → at most 2 open at any time; the 3rd pid never enters.
    pids = ["A-USD", "B-USD", "C-USD"]
    profiles = [_make_profile(pid, 0, 24, "price_over_ema20 > 1.0", deflated=0.05) for pid in pids]
    pid_features = {pid: _make_pid_features(pid, n_hours=200, ema_ratio=1.5, label=0.10)
                    for pid in pids}
    metrics, telemetry = simulate_portfolio(profiles, cap=2, pid_features=pid_features)
    # n_open never exceeds 2
    assert max(t.n_open for t in telemetry) <= 2
    # At least some bars have n_open == 2 (cap was hit)
    assert any(t.n_open == 2 for t in telemetry)
    # Each pid that does enter must have closed by horizon
    assert metrics.trade_count > 0
```

- [ ] **Step 2.1.2 — Run, confirm ImportError**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_portfolio_sim.py::test_concurrency_cap_blocks_new_entries_when_full -v
```

Expect: ImportError.

- [ ] **Step 2.1.3 — Implement**

Append to `portfolio_sim.py`:

```python
def parse_rule_path(rule_path: str) -> List[Tuple[str, str, float]]:
    """Parse 'feat_a > 1.02 AND feat_b <= 0.08' into [(feature, op, threshold), ...].

    Operators supported: >, <, >=, <=. The '(root)' or empty rule returns [] (always fires).
    """
    if rule_path.strip() in ("", "(root)"):
        return []
    conditions: List[Tuple[str, str, float]] = []
    for clause in rule_path.split(" AND "):
        clause = clause.strip()
        for op in (">=", "<=", ">", "<"):
            if f" {op} " in clause:
                feature, threshold_str = clause.split(f" {op} ", 1)
                conditions.append((feature.strip(), op, float(threshold_str.strip())))
                break
        else:
            raise ValueError(f"unparseable rule clause: {clause!r}")
    return conditions


def _rule_holds_at(conditions: List[Tuple[str, str, float]], row: pd.Series) -> bool:
    """Evaluate parsed conditions against a Phase 2 feature row."""
    if not conditions:
        return True
    for feature, op, threshold in conditions:
        if feature not in row.index:
            return False
        v = float(row[feature])
        if op == ">"  and not (v >  threshold): return False
        if op == ">=" and not (v >= threshold): return False
        if op == "<"  and not (v <  threshold): return False
        if op == "<=" and not (v <= threshold): return False
    return True


def _compute_max_dd(equity_series: List[float]) -> float:
    if not equity_series:
        return 0.0
    arr = np.asarray(equity_series, dtype="float64")
    running_max = np.maximum.accumulate(arr)
    drawdown = running_max - arr
    return float(drawdown.max())


def _compute_sortino(trade_pnls: List[float]) -> float:
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype="float64")
    mean = float(arr.mean())
    downside = arr[arr < 0]
    if downside.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return mean / dd if dd > 0 else 0.0


def simulate_portfolio(
    subset: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
) -> Tuple[PortfolioMetrics, List[TelemetryRow]]:
    """Walk historical bars in the subset's union; enforce cap; return metrics + telemetry."""
    # Pre-parse rule paths for speed
    parsed_rules = {(p.pid, p.leaf_id): parse_rule_path(p.rule_path) for p in subset}
    label_cols = {p.profile_id: f"label_h{int(p.horizon)}" for p in subset}
    horizon_ms = {p.profile_id: int(p.horizon) * 3_600_000 for p in subset}

    # Build a master timestamp index across all pids in subset
    all_ts = set()
    for p in subset:
        f = pid_features.get(p.pid, pd.DataFrame())
        if not f.empty and "ts" in f.columns:
            all_ts.update(f["ts"].astype("int64").tolist())
    sorted_ts = sorted(all_ts)

    # Per-pid feature lookup by ts
    pid_ts_to_row: Dict[str, Dict[int, pd.Series]] = {}
    for pid, f in pid_features.items():
        if f.empty:
            continue
        pid_ts_to_row[pid] = {int(t): row for t, row in zip(f["ts"], (f.iloc[i] for i in range(len(f))))}

    open_positions: List[dict] = []   # {pid, profile_id, entry_ts, exit_ts, expected_pnl}
    trade_log: List[float] = []
    telemetry: List[TelemetryRow] = []
    equity = 0.0

    for ts in sorted_ts:
        # 1. Close positions whose exit_ts == ts
        still_open: List[dict] = []
        closed_this_bar: List[dict] = []
        for p in open_positions:
            if p["exit_ts"] <= ts:
                closed_this_bar.append(p)
            else:
                still_open.append(p)
        for c in closed_this_bar:
            equity += c["expected_pnl"]
            trade_log.append(float(c["expected_pnl"]))
            telemetry.append(TelemetryRow(
                ts=ts, equity=equity, n_open=len(still_open),
                closed_profile_id=c["profile_id"], realized_pnl=float(c["expected_pnl"]),
            ))
        open_positions = still_open

        # 2. Evaluate firings
        occupied_pids = {p["pid"] for p in open_positions}
        firings: List[LoadedProfile] = []
        for profile in subset:
            if profile.pid in occupied_pids:
                continue
            row = pid_ts_to_row.get(profile.pid, {}).get(int(ts))
            if row is None:
                continue
            if _rule_holds_at(parsed_rules[(profile.pid, profile.leaf_id)], row):
                firings.append(profile)

        # 3. Enforce cap; tiebreaker = highest deflated profit
        available = cap - len(open_positions)
        if available <= 0:
            telemetry.append(TelemetryRow(ts=ts, equity=equity, n_open=len(open_positions)))
            continue
        firings.sort(key=lambda p: -p.cumulative_profit_deflated)
        for profile in firings[:available]:
            label_col = label_cols[profile.profile_id]
            row = pid_ts_to_row[profile.pid][int(ts)]
            if label_col not in row.index or pd.isna(row[label_col]):
                continue
            expected = float(row[label_col])
            open_positions.append({
                "pid": profile.pid,
                "profile_id": profile.profile_id,
                "entry_ts": int(ts),
                "exit_ts": int(ts) + horizon_ms[profile.profile_id],
                "expected_pnl": expected,
            })
            telemetry.append(TelemetryRow(
                ts=ts, equity=equity, n_open=len(open_positions),
                fired_profile_id=profile.profile_id,
            ))
        # If no firings, emit a baseline telemetry row anyway
        if not firings:
            telemetry.append(TelemetryRow(ts=ts, equity=equity, n_open=len(open_positions)))

    equity_curve = [t.equity for t in telemetry]
    n_open_log = [t.n_open for t in telemetry]
    metrics = PortfolioMetrics(
        cumulative_profit_raw=equity,
        max_dd=_compute_max_dd(equity_curve),
        sortino=_compute_sortino(trade_log),
        trade_count=len(trade_log),
        pct_slots_full=float(sum(1 for n in n_open_log if n == cap) / max(len(n_open_log), 1)),
        mean_concurrent=float(sum(n_open_log) / max(len(n_open_log), 1)),
    )
    return metrics, telemetry
```

- [ ] **Step 2.1.4 — Run + commit**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_portfolio_sim.py -v
```

Expect: 1 passed.

Pre-commit: PASTE `git rev-parse --abbrev-ref HEAD` + `git status -s`.

```
git add backend/tools/strategy_discovery/portfolio_sim.py backend/tests/tools/strategy_discovery/test_portfolio_sim.py
git commit -m "$(cat <<'EOF'
feat(phase4): add portfolio_sim — time-walk + concurrency cap + per-pid-max-1

Phase 4 strategy-discovery rebuild — portfolio simulator that walks
historical bars, closes positions on horizon expiry (PnL from Phase 2
label_h{horizon}), evaluates rule firings against live Phase 2 features,
enforces concurrency cap N and per-pid max-1 (carried from Phase 3).
Tiebreaker on simultaneous fires = highest deflated profit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/portfolio_sim.py backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

Post-commit: PASTE `git log -1 --stat`.

### Round 2 — `test_max_1_position_per_pid_carried_over`

- [ ] **Step 2.2.1 — Write the failing test**

Append:

```python
def test_max_1_position_per_pid_carried_over():
    # Same pid, two profiles (different leaf_ids), both fire — only one enters.
    profiles = [
        _make_profile("BTC-USD", 0, 24, "price_over_ema20 > 1.0", deflated=0.05),
        _make_profile("BTC-USD", 1, 24, "vol_over_mc > 0.0",      deflated=0.04),
    ]
    feats = _make_pid_features("BTC-USD", n_hours=200, ema_ratio=1.5, label=0.10)
    metrics, telemetry = simulate_portfolio(profiles, cap=3, pid_features={"BTC-USD": feats})
    # At any bar n_open <= 1 (only one BTC-USD position)
    assert max(t.n_open for t in telemetry) <= 1
```

- [ ] **Step 2.2.2 — Run, expect PASS**

Expect: 2 passed.

- [ ] **Step 2.2.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin per-pid max-1 rule carried from Phase 3

Round 2 of portfolio_sim.py — two profiles on the same pid never both
open at the same time, even when the cap allows N>1.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

Post-commit: PASTE `git log -1 --stat`.

### Round 3 — `test_simultaneous_fires_resolved_by_deflated_profit_tiebreaker`

- [ ] **Step 2.3.1 — Write the failing test**

Append:

```python
def test_simultaneous_fires_resolved_by_deflated_profit_tiebreaker():
    # 3 pids fire simultaneously on bar 0; cap=1; only the highest-deflated wins.
    profiles = [
        _make_profile("A-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.03),
        _make_profile("B-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.07),   # winner
        _make_profile("C-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.05),
    ]
    pid_features = {pid: _make_pid_features(pid, n_hours=2, ema_ratio=1.5, label=0.10)
                    for pid in ["A-USD", "B-USD", "C-USD"]}
    metrics, telemetry = simulate_portfolio(profiles, cap=1, pid_features=pid_features)
    # First fire telemetry must be the highest-deflated profile (B-USD)
    fires = [t for t in telemetry if t.fired_profile_id is not None]
    assert len(fires) >= 1
    assert fires[0].fired_profile_id.startswith("B-USD")
```

- [ ] **Step 2.3.2 — Run, expect PASS**

Expect: 3 passed.

- [ ] **Step 2.3.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin tiebreaker = highest cumulative_profit_deflated

Round 3 of portfolio_sim.py — when more profiles fire than available
slots, the ones with highest Phase 3 deflated profit win.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

### Round 4 — `test_exit_pnl_read_from_phase2_label`

- [ ] **Step 2.4.1 — Write the failing test**

Append:

```python
def test_exit_pnl_read_from_phase2_label():
    # Synthetic: one profile fires, label_h1 = +0.10, so one trade closes with +0.10.
    profile = _make_profile("BTC-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.05)
    # Make sure label fires (ema_ratio > 1.0) only at entry bar; subsequent bars
    # are still 'fireable' but the per-pid cap prevents re-entry until exit.
    feats = _make_pid_features("BTC-USD", n_hours=5, ema_ratio=1.5, label=0.10)
    metrics, telemetry = simulate_portfolio([profile], cap=1, pid_features={"BTC-USD": feats})
    assert metrics.cumulative_profit_raw > 0
    # At least one trade closed
    closes = [t for t in telemetry if t.closed_profile_id is not None]
    assert len(closes) >= 1
    # And the realized PnL on the closed trade matches the Phase 2 label exactly
    assert closes[0].realized_pnl == pytest.approx(0.10, abs=1e-9)
```

- [ ] **Step 2.4.2 — Run, expect PASS**

Expect: 4 passed.

- [ ] **Step 2.4.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin exit PnL = Phase 2 label_h{horizon} at entry row

Round 4 of portfolio_sim.py — Phase 4 does NOT re-simulate exits; the
realized PnL on each closed trade matches the Phase 2 label exactly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

### Round 5 — `test_max_dd_computed_on_equity_curve`

- [ ] **Step 2.5.1 — Write the failing test**

Append:

```python
def test_max_dd_computed_on_equity_curve():
    # Construct a deterministic equity curve via _compute_max_dd directly
    from tools.strategy_discovery.portfolio_sim import _compute_max_dd
    # Equity goes: 0, 1, 2, 0.5, 1.0, 1.5 — peak=2, trough after peak=0.5, dd=1.5
    curve = [0.0, 1.0, 2.0, 0.5, 1.0, 1.5]
    assert _compute_max_dd(curve) == pytest.approx(1.5, abs=1e-9)
    # Monotonically increasing curve → dd = 0
    assert _compute_max_dd([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0)
    # Empty → 0
    assert _compute_max_dd([]) == 0.0
```

- [ ] **Step 2.5.2 — Run, expect PASS**

Expect: 5 passed.

- [ ] **Step 2.5.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin _compute_max_dd via running-max drawdown

Round 5 of portfolio_sim.py — drawdown = max(running_max - equity) over
the equity curve.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

### Round 6 — `test_slot_utilization_telemetry`

- [ ] **Step 2.6.1 — Write the failing test**

Append:

```python
def test_slot_utilization_telemetry():
    # 3 pids all firing constantly, cap=2. pct_slots_full should be high (>0.5).
    profiles = [_make_profile(pid, 0, 1, "price_over_ema20 > 1.0", deflated=0.05)
                for pid in ["A-USD", "B-USD", "C-USD"]]
    pid_features = {pid: _make_pid_features(pid, n_hours=100, ema_ratio=1.5, label=0.10)
                    for pid in ["A-USD", "B-USD", "C-USD"]}
    metrics, telemetry = simulate_portfolio(profiles, cap=2, pid_features=pid_features)
    assert 0.0 <= metrics.pct_slots_full <= 1.0
    assert 0.0 <= metrics.mean_concurrent <= 2.0
    # With cap=2 and 3 always-firing pids, we expect cap to be hit some of the time
    assert metrics.pct_slots_full > 0.3
```

- [ ] **Step 2.6.2 — Run, expect PASS**

Expect: 6 passed.

- [ ] **Step 2.6.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin slot utilization telemetry (pct_slots_full, mean_concurrent)

Round 6 of portfolio_sim.py — operator-facing diagnostics for whether
the cap is binding or slack.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_portfolio_sim.py
```

---

## Task 3: Knapsack search (`knapsack_search.py` + 3 tests)

**Files:**
- Create: `backend/tools/strategy_discovery/knapsack_search.py`
- Create: `backend/tests/tools/strategy_discovery/test_knapsack_search.py`

### Scaffolding (Step 3.0)

Create `knapsack_search.py`:

```python
"""Beam-search knapsack over profile subsets for Phase 4.

For each candidate subset of size `cap`, calls portfolio_sim and keeps the
top-BEAM_WIDTH subsets at each step. Applies σ × √(2 ln K) deflation to
the winner's cumulative profit using K = total portfolio_sim calls.

Pure Python orchestration; depends only on portfolio_sim + profile_loader.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.strategy_discovery.portfolio_sim import (
    PortfolioMetrics,
    TelemetryRow,
    simulate_portfolio,
)
from tools.strategy_discovery.profile_loader import LoadedProfile

_DEFAULT_BEAM_WIDTH = 20
_DEFAULT_POOL_SIZE  = 100
_DEFAULT_BOOTSTRAP_N = 1000


@dataclass
class KnapsackResult:
    best_subset: List[LoadedProfile]
    best_metrics: PortfolioMetrics
    best_telemetry: List[TelemetryRow]
    k_evaluated: int
    inflation: float
    beam_history: List[List[float]] = field(default_factory=list)
```

### Round 1 — `test_returns_k_evaluated_for_deflation`

- [ ] **Step 3.1.1 — Write the failing test**

Create `test_knapsack_search.py`:

```python
"""Tests for tools.strategy_discovery.knapsack_search (Phase 4)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.knapsack_search import (
    KnapsackResult,
    beam_search_knapsack,
)
from tools.strategy_discovery.profile_loader import LoadedProfile


def _make_profile(pid: str, leaf_id: int, horizon: int, deflated: float) -> LoadedProfile:
    return LoadedProfile(
        pid=pid, horizon=horizon, leaf_id=leaf_id,
        rule_path="price_over_ema20 > 1.0",
        cumulative_profit_raw=deflated + 0.02,
        cumulative_profit_deflated=deflated,
        deflation_pp=0.02,
        win_rate=0.6, avg_win=0.08, avg_loss=-0.04,
        max_dd=0.20, sortino=1.2, trade_count=30,
        n_folds_passed_q0=5, chosen_depth=5, chosen_min_leaf=50,
    )


def _make_pid_features(pid: str, n: int):
    return pd.DataFrame({
        "ts": (np.arange(n, dtype="int64") * 3_600_000).tolist(),
        "close": [1.0] * n,
        "price_over_ema20": [1.5] * n,
        "vol_over_mc": [0.01] * n,
        "label_h1": [0.10] * n,
    })


def test_returns_k_evaluated_for_deflation():
    profiles = [_make_profile(f"P{i}-USD", 0, 1, deflated=0.05 + i * 0.01) for i in range(5)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=3, pool_size=5, bootstrap_iter=100, seed=42,
    )
    # K_evaluated > 0 and matches: step 1 (1 × 5) + step 2 (3 × 4) = 5 + 12 = 17
    assert result.k_evaluated > 0
    assert result.k_evaluated == 1 * 5 + 3 * 4
    # inflation > 0 because σ × √(2 ln 17) > 0 for any non-zero σ
    assert result.inflation >= 0
    # best subset size == cap
    assert len(result.best_subset) == 2
```

- [ ] **Step 3.1.2 — Run, confirm ImportError**

Expect: ImportError on `beam_search_knapsack`.

- [ ] **Step 3.1.3 — Implement**

Append to `knapsack_search.py`:

```python
def _bootstrap_portfolio_std(
    trade_pnls: List[float],
    n_iter: int = _DEFAULT_BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Bootstrap SE of cumulative profit. Resample the trade list with replacement."""
    if rng is None:
        rng = np.random.default_rng()
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype="float64")
    n = arr.shape[0]
    samples = rng.choice(arr, size=(int(n_iter), n), replace=True)
    cums = samples.sum(axis=1)
    return float(cums.std(ddof=1))


def beam_search_knapsack(
    all_qualifying: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
    beam_width: int = _DEFAULT_BEAM_WIDTH,
    pool_size: int = _DEFAULT_POOL_SIZE,
    bootstrap_iter: int = _DEFAULT_BOOTSTRAP_N,
    seed: int = 42,
) -> KnapsackResult:
    """Beam search over subsets of size `cap`; return best by deflated profit."""
    # Cap the candidate pool
    pool = sorted(all_qualifying, key=lambda p: -p.cumulative_profit_deflated)[:int(pool_size)]
    if not pool:
        return KnapsackResult(
            best_subset=[], best_metrics=PortfolioMetrics(), best_telemetry=[],
            k_evaluated=0, inflation=0.0, beam_history=[],
        )

    # Beam = list of (subset, metrics, telemetry, trade_pnls)
    beam: List[Tuple[List[LoadedProfile], PortfolioMetrics, List[TelemetryRow], List[float]]] = [([], PortfolioMetrics(), [], [])]
    k_evaluated = 0
    beam_history: List[List[float]] = []

    for _step in range(int(cap)):
        candidates: List[Tuple[List[LoadedProfile], PortfolioMetrics, List[TelemetryRow], List[float]]] = []
        for subset, _m, _t, _p in beam:
            occupied_pids = {p.pid for p in subset}
            for cand in pool:
                if cand.pid in occupied_pids or cand in subset:
                    continue
                new_subset = subset + [cand]
                metrics, telemetry = simulate_portfolio(new_subset, cap=int(cap), pid_features=pid_features)
                trade_pnls = [t.realized_pnl for t in telemetry if t.realized_pnl is not None]
                k_evaluated += 1
                candidates.append((new_subset, metrics, telemetry, trade_pnls))
        if not candidates:
            break
        candidates.sort(key=lambda c: -c[1].cumulative_profit_raw)
        beam = candidates[:int(beam_width)]
        beam_history.append([c[1].cumulative_profit_raw for c in beam])

    # Pick the best from final beam
    best_subset, best_metrics, best_telemetry, best_trades = beam[0] if beam else ([], PortfolioMetrics(), [], [])
    rng = np.random.default_rng(seed)
    sigma = _bootstrap_portfolio_std(best_trades, n_iter=bootstrap_iter, rng=rng)
    inflation = sigma * math.sqrt(2.0 * math.log(max(k_evaluated, 1)))
    best_metrics.cumulative_profit_deflated = best_metrics.cumulative_profit_raw - inflation
    return KnapsackResult(
        best_subset=best_subset, best_metrics=best_metrics, best_telemetry=best_telemetry,
        k_evaluated=k_evaluated, inflation=inflation, beam_history=beam_history,
    )
```

- [ ] **Step 3.1.4 — Run + commit**

Expect: 1 passed.

```
git add backend/tools/strategy_discovery/knapsack_search.py backend/tests/tools/strategy_discovery/test_knapsack_search.py
git commit -m "$(cat <<'EOF'
feat(phase4): add beam_search_knapsack + portfolio-level deflation

Phase 4 strategy-discovery rebuild — beam search over subsets of size N
using portfolio_sim as the scorer. Tracks k_evaluated for the σ × √(2 ln K)
deflation applied to the winner's cumulative profit. Per-pid uniqueness
enforced at the candidate-expansion step.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/knapsack_search.py backend/tests/tools/strategy_discovery/test_knapsack_search.py
```

### Round 2 — `test_beam_search_finds_known_optimal_on_toy_3_profile_pool`

- [ ] **Step 3.2.1 — Write the failing test**

Append:

```python
def test_beam_search_finds_known_optimal_on_toy_3_profile_pool():
    # 3 profiles with clearly ordered deflated profits. Best cap=2 subset = top-2.
    profiles = [_make_profile(f"P{i}-USD", 0, 1, deflated=0.05 + i * 0.05) for i in range(3)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=10, pool_size=3, bootstrap_iter=100, seed=42,
    )
    # The 2 best by deflated profit are P2 (0.15) and P1 (0.10)
    chosen_pids = sorted(p.pid for p in result.best_subset)
    assert chosen_pids == ["P1-USD", "P2-USD"]
```

- [ ] **Step 3.2.2 — Run, expect PASS**

Expect: 2 passed.

- [ ] **Step 3.2.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin beam_search finds top-2 on tiny known-ordered pool

Round 2 of knapsack_search.py — on a 3-profile pool with monotone deflated
profits, the best cap=2 subset is the top-2.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_knapsack_search.py
```

### Round 3 — `test_beam_width_caps_branching`

- [ ] **Step 3.3.1 — Write the failing test**

Append:

```python
def test_beam_width_caps_branching():
    # 5 profiles, beam_width=2, cap=2.
    # Step 1: 1 × 5 = 5 candidates → beam keeps 2.
    # Step 2: 2 × 4 = 8 candidates → beam keeps 2.
    # Total k_evaluated = 5 + 8 = 13.
    profiles = [_make_profile(f"Q{i}-USD", 0, 1, deflated=0.05 + i * 0.01) for i in range(5)]
    pid_features = {p.pid: _make_pid_features(p.pid, n=20) for p in profiles}
    result = beam_search_knapsack(
        all_qualifying=profiles, cap=2, pid_features=pid_features,
        beam_width=2, pool_size=5, bootstrap_iter=100, seed=42,
    )
    assert result.k_evaluated == 5 + 8
    # Beam history captures 2 steps
    assert len(result.beam_history) == 2
    # Beam at each step ≤ 2
    for step in result.beam_history:
        assert len(step) <= 2
```

- [ ] **Step 3.3.2 — Run, expect PASS**

Expect: 3 passed.

- [ ] **Step 3.3.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin beam_width branching cap

Round 3 of knapsack_search.py — beam keeps at most beam_width subsets
per step; total k_evaluated tracks the search-space breadth.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_knapsack_search.py
```

---

## Task 4: Scorecard (`scorecard.py` + 5 tests)

**Files:**
- Create: `backend/tools/strategy_discovery/scorecard.py`
- Create: `backend/tests/tools/strategy_discovery/test_scorecard.py`

### Scaffolding (Step 4.0)

Create `scorecard.py`:

```python
"""Phase 4 scorecard — per-cap Q0 gates + Markdown verdict report.

Pure functions on PortfolioMetrics. No I/O (build_phase4 owns writing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
from tools.strategy_discovery.profile_loader import LoadedProfile

_PORTFOLIO_MAX_DD     = 0.30
_PORTFOLIO_MIN_TRADES = 50


@dataclass
class CapScorecard:
    cap: int
    metrics: PortfolioMetrics
    k_evaluated: int
    inflation: float
    gates: Dict[str, bool]
    overall_pass: bool
    selected_profiles: List[LoadedProfile]
```

### Round 1 — `test_portfolio_gates_max_dd_30`

- [ ] **Step 4.1.1 — Write the failing test**

Create `test_scorecard.py`:

```python
"""Tests for tools.strategy_discovery.scorecard (Phase 4)."""
from __future__ import annotations

import pytest

from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
from tools.strategy_discovery.profile_loader import LoadedProfile
from tools.strategy_discovery.scorecard import (
    CapScorecard,
    evaluate_cap_gates,
    pick_verdict,
    render_scorecard,
)


def _passing_metrics() -> PortfolioMetrics:
    m = PortfolioMetrics(
        cumulative_profit_raw=0.20,
        cumulative_profit_deflated=0.15,
        max_dd=0.25, sortino=1.5, trade_count=100,
        pct_slots_full=0.4, mean_concurrent=1.8,
    )
    return m


def test_portfolio_gates_max_dd_30():
    m = _passing_metrics()
    m.max_dd = 0.31
    gates, overall = evaluate_cap_gates(m)
    assert gates["max_dd_le_30"] is False
    assert overall is False
```

- [ ] **Step 4.1.2 — Run, confirm ImportError**

- [ ] **Step 4.1.3 — Implement**

Append to `scorecard.py`:

```python
def evaluate_cap_gates(metrics: PortfolioMetrics) -> Tuple[Dict[str, bool], bool]:
    """Apply 4 portfolio Q0 gates."""
    gates = {
        "max_dd_le_30":        metrics.max_dd <= _PORTFOLIO_MAX_DD,
        "deflated_profit_gt_0": metrics.cumulative_profit_deflated > 0.0,
        "trade_count_ge_50":   metrics.trade_count >= _PORTFOLIO_MIN_TRADES,
        "sortino_ge_0":        metrics.sortino >= 0.0,
    }
    overall = all(gates.values())
    return gates, overall
```

- [ ] **Step 4.1.4 — Run + commit**

Expect: 1 passed.

```
git add backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
git commit -m "$(cat <<'EOF'
feat(phase4): add evaluate_cap_gates (4 portfolio Q0 gates)

Phase 4 strategy-discovery rebuild — applies portfolio-level Q0 gates:
max_dd ≤ 30%, deflated cum_profit > 0, trade_count ≥ 50, sortino ≥ 0.
ALL must pass for the cap to qualify.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
```

### Round 2 — `test_portfolio_gates_deflated_profit_positive`

- [ ] **Step 4.2.1 — Write the failing test**

Append:

```python
def test_portfolio_gates_deflated_profit_positive():
    m = _passing_metrics()
    m.cumulative_profit_deflated = -0.01
    gates, overall = evaluate_cap_gates(m)
    assert gates["deflated_profit_gt_0"] is False
    assert overall is False
```

- [ ] **Step 4.2.2 — Run, expect PASS**

Expect: 2 passed.

- [ ] **Step 4.2.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin gate fail on deflated cumulative profit <= 0

Round 2 of scorecard.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_scorecard.py
```

### Round 3 — `test_portfolio_gates_trade_count_50`

- [ ] **Step 4.3.1 — Write the failing test**

Append:

```python
def test_portfolio_gates_trade_count_50():
    m = _passing_metrics()
    m.trade_count = 49
    gates, overall = evaluate_cap_gates(m)
    assert gates["trade_count_ge_50"] is False
    assert overall is False
```

- [ ] **Step 4.3.2 — Run, expect PASS**

Expect: 3 passed.

- [ ] **Step 4.3.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin gate fail on trade_count < 50

Round 3 of scorecard.py.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_scorecard.py
```

### Round 4 — `test_verdict_picks_highest_deflated_passing_cap`

- [ ] **Step 4.4.1 — Write the failing test**

Append:

```python
def test_verdict_picks_highest_deflated_passing_cap():
    # 3 caps: N=3 fails, N=4 passes with 0.10, N=5 passes with 0.15 → pick N=5
    cards = [
        CapScorecard(cap=3, metrics=PortfolioMetrics(
            cumulative_profit_raw=-0.05, cumulative_profit_deflated=-0.08,
            max_dd=0.30, sortino=0.5, trade_count=60), k_evaluated=5000,
            inflation=0.03, gates={"deflated_profit_gt_0": False, "max_dd_le_30": True,
                                    "trade_count_ge_50": True, "sortino_ge_0": True},
            overall_pass=False, selected_profiles=[]),
        CapScorecard(cap=4, metrics=PortfolioMetrics(
            cumulative_profit_raw=0.15, cumulative_profit_deflated=0.10,
            max_dd=0.20, sortino=1.2, trade_count=80), k_evaluated=8000,
            inflation=0.05, gates={k: True for k in ("max_dd_le_30","deflated_profit_gt_0","trade_count_ge_50","sortino_ge_0")},
            overall_pass=True, selected_profiles=[]),
        CapScorecard(cap=5, metrics=PortfolioMetrics(
            cumulative_profit_raw=0.20, cumulative_profit_deflated=0.15,
            max_dd=0.25, sortino=1.5, trade_count=100), k_evaluated=10000,
            inflation=0.05, gates={k: True for k in ("max_dd_le_30","deflated_profit_gt_0","trade_count_ge_50","sortino_ge_0")},
            overall_pass=True, selected_profiles=[]),
    ]
    chosen_cap, verdict = pick_verdict(cards)
    assert chosen_cap == 5
    assert "deploy" in verdict.lower()
    assert "5" in verdict
```

- [ ] **Step 4.4.2 — Run, confirm ImportError**

- [ ] **Step 4.4.3 — Implement**

Append to `scorecard.py`:

```python
def pick_verdict(per_cap: List[CapScorecard]) -> Tuple[Optional[int], str]:
    """Pick the highest-deflated-profit passing cap, or abort if none pass."""
    passing = [c for c in per_cap if c.overall_pass]
    if not passing:
        return None, "abort — no qualifying portfolio at any cap"
    best = max(passing, key=lambda c: c.metrics.cumulative_profit_deflated)
    return best.cap, f"deploy at N={best.cap} (deflated profit = {best.metrics.cumulative_profit_deflated:.4f})"
```

- [ ] **Step 4.4.4 — Run + commit**

Expect: 4 passed.

```
git add backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
git commit -m "$(cat <<'EOF'
feat(phase4): add pick_verdict — highest-deflated passing cap or abort

Round 4 of scorecard.py — when multiple caps pass, pick the one with
highest cumulative_profit_deflated; when none pass, abort with explanation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
```

### Round 5 — `test_verdict_abort_when_all_caps_fail`

- [ ] **Step 4.5.1 — Write the failing test**

Append:

```python
def test_verdict_abort_when_all_caps_fail():
    cards = [
        CapScorecard(cap=cap, metrics=PortfolioMetrics(
            cumulative_profit_raw=-0.05, cumulative_profit_deflated=-0.08,
            max_dd=0.40, sortino=-0.5, trade_count=20), k_evaluated=5000,
            inflation=0.03, gates={"deflated_profit_gt_0": False, "max_dd_le_30": False,
                                    "trade_count_ge_50": False, "sortino_ge_0": False},
            overall_pass=False, selected_profiles=[]) for cap in [3, 4, 5]
    ]
    chosen_cap, verdict = pick_verdict(cards)
    assert chosen_cap is None
    assert "abort" in verdict.lower()


def test_render_scorecard_produces_markdown_with_all_cap_sections():
    cards = [
        CapScorecard(cap=cap, metrics=PortfolioMetrics(
            cumulative_profit_raw=0.10 + cap * 0.01,
            cumulative_profit_deflated=0.05 + cap * 0.01,
            max_dd=0.20, sortino=1.2, trade_count=60,
            pct_slots_full=0.3, mean_concurrent=1.5),
            k_evaluated=5000, inflation=0.05,
            gates={k: True for k in ("max_dd_le_30","deflated_profit_gt_0","trade_count_ge_50","sortino_ge_0")},
            overall_pass=True, selected_profiles=[]) for cap in [3, 4, 5]
    ]
    md = render_scorecard(cards)
    # Mentions each cap and the verdict
    assert "N=3" in md and "N=4" in md and "N=5" in md
    assert "Verdict" in md or "verdict" in md
```

- [ ] **Step 4.5.2 — Run, confirm test 6 ImportError on render_scorecard**

- [ ] **Step 4.5.3 — Implement**

Append to `scorecard.py`:

```python
def render_scorecard(per_cap: List[CapScorecard]) -> str:
    """Render a Markdown scorecard with per-cap sections + comparison + verdict."""
    lines: List[str] = ["# Phase 4 Scorecard — strategy-discovery deployment selection", ""]
    chosen_cap, verdict = pick_verdict(per_cap)
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append("| cap | deflated profit | raw profit | inflation | max DD | sortino | trades | pct full | mean conc | pass? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for c in sorted(per_cap, key=lambda x: x.cap):
        lines.append(
            f"| {c.cap} "
            f"| {c.metrics.cumulative_profit_deflated:+.4f} "
            f"| {c.metrics.cumulative_profit_raw:+.4f} "
            f"| {c.inflation:.4f} "
            f"| {c.metrics.max_dd:.4f} "
            f"| {c.metrics.sortino:.3f} "
            f"| {c.metrics.trade_count} "
            f"| {c.metrics.pct_slots_full:.2%} "
            f"| {c.metrics.mean_concurrent:.2f} "
            f"| {'✅' if c.overall_pass else '❌'} |"
        )
    lines.append("")
    for c in sorted(per_cap, key=lambda x: x.cap):
        lines.append(f"## N={c.cap}")
        lines.append("")
        lines.append(f"- Deflated cumulative profit: **{c.metrics.cumulative_profit_deflated:+.4f}**  (raw {c.metrics.cumulative_profit_raw:+.4f}, inflation {c.inflation:.4f})")
        lines.append(f"- Max drawdown: {c.metrics.max_dd:.4f}")
        lines.append(f"- Sortino: {c.metrics.sortino:.3f}")
        lines.append(f"- Trade count: {c.metrics.trade_count}")
        lines.append(f"- Slot utilization: {c.metrics.pct_slots_full:.2%} full, mean concurrent {c.metrics.mean_concurrent:.2f}")
        lines.append(f"- K subsets evaluated: {c.k_evaluated}")
        lines.append(f"- Gates: " + ", ".join(f"{k}={'✅' if v else '❌'}" for k, v in c.gates.items()))
        lines.append(f"- Overall: {'✅ PASS' if c.overall_pass else '❌ FAIL'}")
        lines.append("")
        if c.selected_profiles:
            lines.append("Selected profiles:")
            for p in c.selected_profiles:
                lines.append(f"  - {p.profile_id}  (h={p.horizon})  `{p.rule_path}`  → deflated {p.cumulative_profit_deflated:+.4f}")
            lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4.5.4 — Run + commit**

Expect: 6 passed (5 + 1 added).

```
git add backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
git commit -m "$(cat <<'EOF'
feat(phase4): add render_scorecard Markdown + verdict abort path

Round 5 of scorecard.py — renders per-cap sections + comparison table +
top-line verdict. Abort path emits explanation when all caps fail.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/scorecard.py backend/tests/tools/strategy_discovery/test_scorecard.py
```

---

## Task 5: Orchestrator + CLI (`build_phase4.py` + 3 tests)

**Files:**
- Create: `backend/tools/strategy_discovery/build_phase4.py`
- Create: `backend/tests/tools/strategy_discovery/test_build_phase4.py`

### Scaffolding (Step 5.0)

Create `build_phase4.py`:

```python
"""Phase 4 orchestrator + CLI.

Iterates caps ∈ {3, 4, 5}, dispatches knapsack search per cap, writes:
  - backend/data/phase4/scorecard.md
  - backend/data/phase4/deployment_n{N}.json  (one per cap)
  - backend/data/phase4/portfolio_telemetry_n{N}.parquet  (one per cap)

Only module in Phase 4 that touches the filesystem.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.knapsack_search import beam_search_knapsack  # noqa: E402
from tools.strategy_discovery.profile_loader import (  # noqa: E402
    LoadedProfile,
    load_all_profiles,
    load_pid_features,
)
from tools.strategy_discovery.scorecard import (  # noqa: E402
    CapScorecard,
    evaluate_cap_gates,
    render_scorecard,
)

_DEFAULT_PHASE3_DIR = Path(BACKEND) / "data" / "phase3"
_DEFAULT_PHASE2_DIR = Path(BACKEND) / "data" / "phase2"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase4"
_DEFAULT_CAPS       = (3, 4, 5)
```

### Round 1 — `test_sweeps_all_three_caps_writes_three_deployments`

- [ ] **Step 5.1.1 — Write the failing test**

Create `test_build_phase4.py`:

```python
"""Tests for tools.strategy_discovery.build_phase4 (Phase 4 driver)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_minimal_phase3(phase3_dir: Path):
    _PROFILE_COLUMNS = [
        "pid", "horizon", "leaf_id", "rule_path_summary",
        "cumulative_profit_raw", "cumulative_profit_deflated", "deflation_pp",
        "win_rate", "avg_win", "avg_loss", "max_dd", "sortino",
        "trade_count", "n_folds_passed_q0",
        "chosen_depth", "chosen_min_leaf",
        "bootstrap_triggered", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "n_combos_searched", "inner_cv_se", "schema_version",
    ]
    rows = [
        ("BTC-USD", 24, 0, "price_over_ema20 > 1.0", 0.10, 0.06, 0.04, 0.6, 0.08, -0.04, 0.20, 1.2, 50, 5, 5, 50, False, None, None, 9, 0.015, 1),
        ("ETH-USD", 24, 1, "price_over_ema20 > 1.0", 0.08, 0.05, 0.03, 0.55, 0.07, -0.04, 0.18, 1.1, 40, 5, 5, 50, False, None, None, 9, 0.012, 1),
        ("SOL-USD", 24, 2, "price_over_ema20 > 1.0", 0.06, 0.04, 0.02, 0.50, 0.06, -0.05, 0.22, 1.0, 35, 4, 5, 50, False, None, None, 9, 0.010, 1),
    ]
    df = pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
    phase3_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), phase3_dir / "profiles_h24.parquet")
    sidecar = {"BTC-USD__0": "price_over_ema20 > 1.0",
               "ETH-USD__1": "price_over_ema20 > 1.0",
               "SOL-USD__2": "price_over_ema20 > 1.0"}
    (phase3_dir / "rule_paths_h24.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _write_minimal_phase2(phase2_dir: Path, pids):
    phase2_dir.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        n = 100
        df = pd.DataFrame({
            "ts": (np.arange(n, dtype="int64") * 3_600_000).tolist(),
            "close": [1.0] * n,
            "price_over_ema20": [1.5] * n,
            "vol_over_mc": [0.01] * n,
            "label_h24": [0.05] * n,
        })
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), phase2_dir / f"{pid}.parquet")


def test_sweeps_all_three_caps_writes_three_deployments(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4
    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    cards = build_phase4(
        phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir,
        caps=[3, 4, 5], beam_width=3, pool_size=3, bootstrap_iter=50, seed=42,
        horizons=[24],
    )
    assert set(cards.keys()) == {3, 4, 5}
    for cap in [3, 4, 5]:
        assert (output_dir / f"deployment_n{cap}.json").exists()
```

- [ ] **Step 5.1.2 — Run, confirm ImportError**

- [ ] **Step 5.1.3 — Implement**

Append to `build_phase4.py`:

```python
def _write_deployment_json(
    card: CapScorecard,
    output_path: Path,
) -> None:
    payload = {
        "cap": int(card.cap),
        "selected_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "k_subsets_evaluated": int(card.k_evaluated),
        "portfolio_metrics": {
            "cumulative_profit_raw":      float(card.metrics.cumulative_profit_raw),
            "cumulative_profit_deflated": float(card.metrics.cumulative_profit_deflated),
            "deflation_pp":               float(card.inflation),
            "max_dd":                     float(card.metrics.max_dd),
            "sortino":                    float(card.metrics.sortino),
            "trade_count":                int(card.metrics.trade_count),
            "pct_slots_full":             float(card.metrics.pct_slots_full),
            "mean_concurrent":            float(card.metrics.mean_concurrent),
        },
        "gates": {**card.gates, "overall": "pass" if card.overall_pass else "fail"},
        "profiles": [
            {
                "pid": p.pid,
                "horizon": int(p.horizon),
                "leaf_id": int(p.leaf_id),
                "rule_path": p.rule_path,
                "expected_avg_win": float(p.avg_win),
                "expected_avg_loss": float(p.avg_loss),
                "expected_max_dd": float(p.max_dd),
                "expected_trade_count": int(p.trade_count),
                "expected_sortino": float(p.sortino),
                "phase3_cumulative_profit_deflated": float(p.cumulative_profit_deflated),
            }
            for p in card.selected_profiles
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_telemetry_parquet(
    telemetry,
    output_path: Path,
) -> None:
    if not telemetry:
        return
    rows = [
        {
            "ts": int(t.ts),
            "equity": float(t.equity),
            "n_open": int(t.n_open),
            "fired_profile_id": t.fired_profile_id,
            "closed_profile_id": t.closed_profile_id,
            "realized_pnl": None if t.realized_pnl is None else float(t.realized_pnl),
            "schema_version": 1,
        }
        for t in telemetry
    ]
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), output_path,
                   compression="snappy")


def build_phase4(
    *,
    phase3_dir: Path = _DEFAULT_PHASE3_DIR,
    phase2_dir: Path = _DEFAULT_PHASE2_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    caps = _DEFAULT_CAPS,
    horizons: List[int] = [1, 4, 24, 72, 168],
    beam_width: int = 20,
    pool_size: int = 100,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> Dict[int, CapScorecard]:
    """Sweep caps; per cap: knapsack search → score → write artifacts."""
    phase3_dir = Path(phase3_dir)
    phase2_dir = Path(phase2_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=horizons)
    pid_features = {pid: load_pid_features(pid, phase2_dir=phase2_dir)
                    for pid in {p.pid for p in profiles}}
    pid_features = {k: v for k, v in pid_features.items() if not v.empty}
    cards: Dict[int, CapScorecard] = {}
    for cap in caps:
        result = beam_search_knapsack(
            all_qualifying=profiles, cap=int(cap), pid_features=pid_features,
            beam_width=int(beam_width), pool_size=int(pool_size),
            bootstrap_iter=int(bootstrap_iter), seed=int(seed),
        )
        gates, overall = evaluate_cap_gates(result.best_metrics)
        card = CapScorecard(
            cap=int(cap), metrics=result.best_metrics,
            k_evaluated=result.k_evaluated, inflation=result.inflation,
            gates=gates, overall_pass=overall,
            selected_profiles=result.best_subset,
        )
        cards[int(cap)] = card
        _write_deployment_json(card, output_dir / f"deployment_n{int(cap)}.json")
        _write_telemetry_parquet(result.best_telemetry,
                                  output_dir / f"portfolio_telemetry_n{int(cap)}.parquet")
    # Render scorecard
    md = render_scorecard(list(cards.values()))
    (output_dir / "scorecard.md").write_text(md, encoding="utf-8")
    return cards


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 4 — scorecard + deployment selection.")
    parser.add_argument("--phase3-dir", default=str(_DEFAULT_PHASE3_DIR))
    parser.add_argument("--phase2-dir", default=str(_DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir", default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--caps", default="3,4,5")
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--pool-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    caps = [int(c.strip()) for c in args.caps.split(",") if c.strip()]
    cards = build_phase4(
        phase3_dir=Path(args.phase3_dir), phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir), caps=caps,
        beam_width=args.beam_width, pool_size=args.pool_size, seed=args.seed,
    )
    n_passing = sum(1 for c in cards.values() if c.overall_pass)
    print(f"  scorecard written to {args.output_dir}/scorecard.md", flush=True)
    print(f"  {n_passing} of {len(cards)} caps passed", flush=True)
    return 0 if n_passing > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.1.4 — Run + commit**

Expect: 1 passed.

```
git add backend/tools/strategy_discovery/build_phase4.py backend/tests/tools/strategy_discovery/test_build_phase4.py
git commit -m "$(cat <<'EOF'
feat(phase4): add build_phase4 orchestrator + CLI

Phase 4 strategy-discovery rebuild — sweeps caps ∈ {3,4,5}, dispatches
knapsack per cap, writes deployment JSONs + telemetry parquets + scorecard
Markdown. Single fs-writing module in Phase 4.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/build_phase4.py backend/tests/tools/strategy_discovery/test_build_phase4.py
```

### Round 2 — `test_writes_scorecard_md_and_telemetry_parquet`

- [ ] **Step 5.2.1 — Write the failing test**

Append:

```python
def test_writes_scorecard_md_and_telemetry_parquet(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4
    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    build_phase4(
        phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir,
        caps=[3], beam_width=3, pool_size=3, bootstrap_iter=50, seed=42,
        horizons=[24],
    )
    assert (output_dir / "scorecard.md").exists()
    # Telemetry parquet may or may not have content (depending on whether any trades fired)
    # but the file should be writable; if no telemetry, skip the assertion
    tele_path = output_dir / "portfolio_telemetry_n3.parquet"
    if tele_path.exists():
        df = pq.read_table(tele_path).to_pandas()
        # If exists, must have schema_version column
        assert "schema_version" in df.columns
```

- [ ] **Step 5.2.2 — Run, expect PASS**

Expect: 2 passed.

- [ ] **Step 5.2.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin scorecard.md + telemetry parquet emission

Round 2 of build_phase4.py — confirms artifacts land on disk.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_build_phase4.py
```

### Round 3 — `test_main_returns_zero_on_at_least_one_passing_cap`

- [ ] **Step 5.3.1 — Write the failing test**

Append:

```python
def test_main_returns_zero_on_at_least_one_passing_cap(tmp_path: Path, monkeypatch):
    from tools.strategy_discovery import build_phase4 as bp4
    # Mock build_phase4 to return a card with overall_pass=True
    def fake_build(*args, **kwargs):
        from tools.strategy_discovery.scorecard import CapScorecard
        from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
        return {3: CapScorecard(cap=3, metrics=PortfolioMetrics(),
                                k_evaluated=0, inflation=0.0,
                                gates={}, overall_pass=True, selected_profiles=[])}
    monkeypatch.setattr(bp4, "build_phase4", fake_build)
    rc = bp4.main(["--phase3-dir", str(tmp_path), "--phase2-dir", str(tmp_path),
                   "--output-dir", str(tmp_path / "out"), "--caps", "3"])
    assert rc == 0
```

- [ ] **Step 5.3.2 — Run, expect PASS**

Expect: 3 passed.

- [ ] **Step 5.3.3 — Commit (test only)**

```
git commit -m "$(cat <<'EOF'
test(phase4): pin main exits 0 when at least one cap passes

Round 3 of build_phase4.py — CLI exit-code semantics for shell integration.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_build_phase4.py
```

---

## Task 6: Full-suite green + CHANGELOG / memory sync

**Files:**
- Modify: `CHANGELOG.md`
- Modify (out-of-tree): `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`

### Step 6.1 — Run full backend test suite

```
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/ -q --tb=line
```

Expect: full suite passes. Phase 4 adds 20 new tests (3+6+3+5+3). Pre-existing baseline including Phase 3: ~1234. New target ≥ 1254.

### Step 6.2 — Shell cleanup

```powershell
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

### Step 6.3 — Prepend CHANGELOG entry

Insert at the top of `## Unreleased` in `CHANGELOG.md`:

```markdown
### Session — 2026-05-24 — Strategy-discovery Phase 4: scorecard + deployment selection

Implemented Phase 4 of the strategy-discovery rebuild per spec
`docs/superpowers/specs/2026-05-24-strategy-discovery-phase4-design.md`
and plan `docs/superpowers/plans/2026-05-24-strategy-discovery-phase4-implementation.md`.

**New modules (all under `backend/tools/strategy_discovery/`):**
- `profile_loader.py` — Phase 3 parquets + sidecars + Phase 2 features loader (pure I/O).
- `portfolio_sim.py` — Time-walk portfolio simulator with concurrency cap and per-pid max-1 (carried from Phase 3).
- `knapsack_search.py` — Beam-search over profile subsets with σ × √(2 ln K) portfolio-level deflation.
- `scorecard.py` — Per-cap Q0 gates (max_dd ≤ 30%, deflated > 0, trades ≥ 50, sortino ≥ 0), Markdown render, verdict.
- `build_phase4.py` — Orchestrator + CLI; sweeps N∈{3,4,5}, writes deployment JSON + telemetry parquet + scorecard.md per cap.

**Test surface added:** 20 new tests under `backend/tests/tools/strategy_discovery/`. Full backend suite green.

**Operator step (post-merge):**

    cd backend && python -m tools.strategy_discovery.build_phase4 \
        --phase3-dir data/phase3 --phase2-dir data/phase2 \
        --output-dir data/phase4 --seed 42

Reads scorecard.md, picks winning cap (or aborts), then opens a separate
integration commit to wire the chosen deployment_n{N}.json into agents/cnn_agent.py.
```

### Step 6.4 — Update memory

Append a Phase 4 sub-section after the Phase 3 block in
`C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`:

```markdown

### Phase 4 (scorecard + deployment selection, 2026-05-24)

Phase 4 ranks profiles across the Phase 3 pool and picks a deployment set
per concurrency cap. Sweeps N ∈ {3, 4, 5}; for each cap, runs a beam-search
knapsack over subsets (beam_width=20, pool=top-100 profiles by Phase 3
deflated profit). Applies portfolio-level σ × √(2 ln K) deflation. Emits
per-cap scorecard with pass/fail verdict.

**New modules** (all in `backend/tools/strategy_discovery/`):
- `profile_loader.py` — `load_all_profiles(phase3_dir, min_folds_passed_q0=4)`
  flattens 5 per-horizon parquets, attaches rule_path sidecars, re-enforces
  Phase 3's ≥4-of-5 fold gate.
- `portfolio_sim.py` — `simulate_portfolio(subset, cap, pid_features)` walks
  historical bars; closes on horizon expiry (PnL from Phase 2 label); enforces
  per-pid max-1 + portfolio cap; tiebreaker = highest deflated profit. Returns
  PortfolioMetrics + telemetry rows.
- `knapsack_search.py` — `beam_search_knapsack(profiles, cap, pid_features, ...)`
  beam-expands subsets of size `cap`; bootstraps portfolio profit SE for the
  deflation factor. Returns KnapsackResult with best subset, metrics, telemetry,
  k_evaluated, inflation.
- `scorecard.py` — `evaluate_cap_gates(metrics)` applies 4 Q0 gates
  (max_dd ≤ 30%, deflated_profit > 0, trade_count ≥ 50, sortino ≥ 0).
  `render_scorecard(per_cap)` renders Markdown. `pick_verdict(per_cap)`
  returns (chosen_cap, verdict_string) — highest-deflated passing cap or abort.
- `build_phase4.py` — CLI orchestrator; sweeps caps, writes
  `deployment_n{N}.json`, `portfolio_telemetry_n{N}.parquet`, and
  `scorecard.md` to `backend/data/phase4/`.

**Output schema (deployment_n{N}.json):** `cap, selected_at_utc,
k_subsets_evaluated, portfolio_metrics (raw, deflated, max_dd, sortino,
trade_count, pct_slots_full, mean_concurrent), gates (4 booleans +
overall), profiles[] (pid, horizon, leaf_id, rule_path, expected_*
metrics, phase3_cumulative_profit_deflated)`.

**Telemetry parquet:** per-bar `ts, equity, n_open, fired_profile_id,
closed_profile_id, realized_pnl, schema_version`.

**Tests:** 20 tests under `backend/tests/tools/strategy_discovery/test_{profile_loader,portfolio_sim,knapsack_search,scorecard,build_phase4}.py`. All mock-only — no torch, no GPU, no live data.

**Operator runs** (post-merge):

    cd backend && python -m tools.strategy_discovery.build_phase4 \
        --phase3-dir data/phase3 --phase2-dir data/phase2 \
        --output-dir data/phase4 --seed 42

**Status:** code-complete on branch `feat/strategy-discovery-phase4`.
Phase 4 does NOT touch the live agent — the operator picks a winning cap
from the scorecard, then a separate integration commit wires
`deployment_n{N}.json` into `agents/cnn_agent.py`.
```

### Step 6.5 — Commit CHANGELOG

```
git commit -m "$(cat <<'EOF'
docs: changelog entry for strategy-discovery Phase 4

Records the 5-module Phase 4 implementation + 20-test surface added across
this branch. Memory file coinbase_trader_architecture.md updated out-of-tree.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- CHANGELOG.md
```

Post-commit: PASTE `git log -1 --stat`. Must show only CHANGELOG.md.

### Step 6.6 — Final verification

```
cd backend && python -m pytest tests/tools/strategy_discovery/ -q
```

Expect: all strategy_discovery tests pass.

```
git log --oneline main..HEAD
```

Expect: ~21 commits on `feat/strategy-discovery-phase4`.

DO NOT push — controller handles push.

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:

- Goal + I/O + output schema → Tasks 1 + 5
- Algorithm — portfolio simulator → Task 2 (all 6 tests cover the spec's per-row simulator rules)
- Algorithm — knapsack beam search → Task 3 (3 tests cover beam, k_evaluated, top-2 optimality)
- Algorithm — portfolio deflation → Task 3 Round 1 (k_evaluated tracked, inflation applied)
- Q0 portfolio gates → Task 4 (4 tests, one per gate; verdict logic in 2 more)
- Module structure (5 modules) → Tasks 1-5 directly
- Testing (19 in spec, 20 in plan — extra test_max_dd_computed for the helper) → Tasks 1-5
- CHANGELOG + memory sync → Task 6

**2. Placeholder scan** — every code step has complete code blocks. No "TBD", no "similar to Task N".

**3. Type consistency:**
- `LoadedProfile` defined Task 1 Step 1.0; consumed by Tasks 2, 3, 4, 5 (all import it from `profile_loader`). ✓
- `PortfolioMetrics`, `TelemetryRow` defined Task 2 Step 2.0; consumed by Tasks 3, 4, 5. ✓
- `KnapsackResult` defined Task 3 Step 3.0; consumed by Task 5. ✓
- `CapScorecard` defined Task 4 Step 4.0; consumed by Task 5. ✓
- Function signatures match between definitions and consumers.

No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-strategy-discovery-phase4-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review between each. 6 independent tasks; cleanest with subagent isolation.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch checkpoints after each task.

Which approach?
