# Strategy-Discovery Rebuild — Phase 1: Data Foundation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the data layer the strategy-discovery rebuild depends on — an inventory of existing local data, a curated 50-pid universe spanning the tokenomic state space, and complete CoinPaprika tokenomic parquets (marketcap + supply) for those 50 pids.

**Architecture:** Read-only inventory pass first (data-first directive from the brainstorm); then extend the CoinPaprika service with a current-ticker snapshot endpoint for supply data; then a universe-curation script that consumes the inventory + supply data and applies Q5 criteria; then a bulk historical-marketcap backfill for the curated pids. Output deliverables are a Markdown inventory report, a Markdown universe rationale doc, a JSON pid-list sidecar, and parquet files under `backend/data/marketcap/` and `backend/data/supply/`.

**Tech Stack:** Python 3.x, pyarrow/pyarrow.parquet, httpx (CoinPaprika), pytest, pytest-asyncio. No new dependencies.

**Source spec:** `docs/superpowers/specs/2026-05-23-strategy-discovery-rebuild-brainstorm.md`

**Spec sections this plan implements:**
- "Data-first directive" → Task 1 (inventory)
- Q5 "diverse-sample ~50 products" curation criteria → Tasks 4-5 (curation)
- Q3 features 1-6 (tokenomic) source plumbing → Tasks 2-3 + Task 6 (supply + marketcap backfill)

**Phase scope (what this plan does NOT do):**
- No trend-feature computation (Phase 2)
- No labeling / mining / validation (Phases 2-3)
- No output doc / verdict (Phase 4)

---

## File structure (created or modified by this plan)

```
backend/tools/strategy_discovery/
  __init__.py                  # NEW package marker
  inventory.py                 # NEW (Task 1) — audits local data, writes report + JSON
  curate_universe.py           # NEW (Task 5) — applies Q5 criteria, picks 50 pids
  build_supply_snapshot.py     # NEW (Task 3) — CLI: fetch supply snapshots, write parquet
  build_universe_marketcap.py  # NEW (Task 6) — CLI: bulk marketcap backfill for the 50

backend/services/coinpaprika_marketcap.py  # MODIFY (Task 2) — add fetch_supply_snapshot()

backend/tests/tools/strategy_discovery/
  __init__.py                  # NEW
  test_inventory.py            # NEW (Task 1)
  test_curate_universe.py      # NEW (Task 5)
backend/tests/services/
  test_coinpaprika_supply.py   # NEW (Task 2)
backend/tests/tools/
  test_build_supply_snapshot.py        # NEW (Task 3)
  test_build_universe_marketcap.py     # NEW (Task 6)

backend/data/supply/                       # NEW directory
  snapshot.parquet             # written by Task 3 CLI

docs/superpowers/specs/
  2026-05-23-data-inventory-report.md  # NEW (Task 1 CLI output)
  2026-05-23-universe-50.md            # NEW (Task 5 CLI output)
  2026-05-23-universe-50.json          # NEW (Task 5 sidecar)

CHANGELOG.md                  # MODIFY (each task)
```

**Naming conventions followed:**
- `tools/<feature>/*.py` is established (cf. `tools/_scorecard/`, `tools/mc/`). The `strategy_discovery/` package is the home for all rebuild code through Phase 4.
- Bronze parquet schema mirrors `tools/build_marketcap_parquet.py:_SCHEMA` — `start` (epoch seconds, bar-aligned), domain columns, `ingest_ts`, `schema_version`.
- CLI scripts print progress and end with a one-line summary (cf. `tools/build_marketcap_parquet.py:main`).

---

## Reference: existing infrastructure this plan extends

- `backend/services/coinpaprika_marketcap.py` — provides `fetch_marketcap_history(pid, start_ms, end_ms) -> list[(ts_ms, market_cap, volume_24h)]`. Also exposes `_PRODUCT_TO_CP_ID` mapping (~28 entries).
- `backend/tools/build_marketcap_parquet.py` — writes per-pid bronze parquets at `backend/data/marketcap/<pid>.parquet`, schema `(start, market_cap, fdv, volume_24h, ingest_ts, schema_version)`. CLI: `--source coinpaprika --pids ... --start ... --end ...`. As of writing, only 3 pids have been backfilled (BTC, ETH, SOL).
- `backend/data/history/*.parquet` — 1h OHLCV per pid, schema `(start, open, high, low, close, volume)` with `start` as epoch seconds at hourly bars. ~220 files present.
- `backend/data/history/1m/` — declared by SP1 infrastructure but populated status TBD (Task 1 will check).

---

## Pre-flight (one-time setup; not a tracked task)

Before Task 1, confirm the dev environment can run tests:

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m pytest tests/ -k "test_coinpaprika_marketcap" -v
```

Expected: existing CoinPaprika tests pass. If they don't, fix that before starting Task 1 — the new tests inherit the same fixtures.

---

## Task 1: Local-data inventory script

Produces a fact-base of what data we already have locally, satisfying the data-first directive. Outputs a Markdown report for humans and a JSON sidecar for downstream tasks (Task 5 consumes the JSON).

**Files:**
- Create: `backend/tools/strategy_discovery/__init__.py` (empty package marker)
- Create: `backend/tools/strategy_discovery/inventory.py`
- Create: `backend/tests/tools/strategy_discovery/__init__.py` (empty)
- Create: `backend/tests/tools/strategy_discovery/test_inventory.py`
- Modify: `CHANGELOG.md` (add entry under "Unreleased")

- [ ] **Step 1: Create the package markers**

```python
# backend/tools/strategy_discovery/__init__.py
```

(empty file — package marker only)

```python
# backend/tests/tools/strategy_discovery/__init__.py
```

(empty file)

- [ ] **Step 2: Write the failing test for `scan_history_parquets`**

Create `backend/tests/tools/strategy_discovery/test_inventory.py`:

```python
"""Unit tests for tools/strategy_discovery/inventory.py."""
from __future__ import annotations

import json
import os
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery import inventory


def _write_ohlcv_parquet(path: str, starts: list[int]) -> None:
    """Helper: write an OHLCV parquet matching the history schema."""
    n = len(starts)
    table = pa.table({
        "start":  starts,
        "open":   [100.0] * n,
        "high":   [101.0] * n,
        "low":    [99.0] * n,
        "close":  [100.5] * n,
        "volume": [10.0] * n,
    })
    pq.write_table(table, path, compression="snappy")


def test_scan_history_parquets_returns_per_pid_coverage(tmp_path):
    """scan_history_parquets returns {pid: (first_ts, last_ts, n_rows)} for *.parquet files."""
    hdir = tmp_path / "history"
    hdir.mkdir()
    _write_ohlcv_parquet(str(hdir / "BTC-USD.parquet"), starts=[1_700_000_000, 1_700_003_600, 1_700_007_200])
    _write_ohlcv_parquet(str(hdir / "ETH-USD.parquet"), starts=[1_700_000_000, 1_700_003_600])

    out = inventory.scan_history_parquets(str(hdir))

    assert set(out.keys()) == {"BTC-USD", "ETH-USD"}
    btc = out["BTC-USD"]
    assert btc["first_ts"] == 1_700_000_000
    assert btc["last_ts"] == 1_700_007_200
    assert btc["n_rows"] == 3
    eth = out["ETH-USD"]
    assert eth["n_rows"] == 2


def test_scan_history_parquets_skips_macro_prefixed_files(tmp_path):
    """`__`-prefixed parquets (e.g. __MACRO__.parquet) are filtered out per CLAUDE.md invariant #8."""
    hdir = tmp_path / "history"
    hdir.mkdir()
    _write_ohlcv_parquet(str(hdir / "BTC-USD.parquet"), starts=[1_700_000_000])
    _write_ohlcv_parquet(str(hdir / "__MACRO__.parquet"), starts=[1_700_000_000])

    out = inventory.scan_history_parquets(str(hdir))

    assert set(out.keys()) == {"BTC-USD"}
    assert "__MACRO__" not in out


def test_scan_history_parquets_handles_missing_dir(tmp_path):
    """Missing directory returns empty dict, no exception."""
    out = inventory.scan_history_parquets(str(tmp_path / "does_not_exist"))
    assert out == {}
```

- [ ] **Step 3: Run the tests to verify they fail**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py -v
```

Expected: 3 failures with `ModuleNotFoundError: No module named 'tools.strategy_discovery.inventory'`.

- [ ] **Step 4: Implement `scan_history_parquets`**

Create `backend/tools/strategy_discovery/inventory.py`:

```python
"""Local-data inventory for the strategy-discovery rebuild (Phase 1).

Audits existing on-disk data BEFORE any new CoinPaprika API calls (per the
data-first directive in the 2026-05-23 brainstorm spec). Produces:

  - scan_history_parquets(dir)    -> {pid: {first_ts, last_ts, n_rows}}
  - scan_marketcap_parquets(dir)  -> {pid: {first_ts, last_ts, n_rows}}
  - scan_1m_dir(dir)              -> {pid: n_files} (returns {} when empty)
  - inventory_report(...)         -> Markdown string
  - run() / main()                CLI: write Markdown report + JSON sidecar

Pure stdlib + pyarrow. No HTTP calls. No model loading. No DB access.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import pyarrow.parquet as pq

# Directory layout (relative to repo root or BACKEND env)
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_HISTORY_DIR    = os.path.join(BACKEND_DIR, "data", "history")
DEFAULT_MARKETCAP_DIR  = os.path.join(BACKEND_DIR, "data", "marketcap")
DEFAULT_1M_DIR         = os.path.join(BACKEND_DIR, "data", "history", "1m")


def _is_pid_parquet(filename: str) -> bool:
    """A pid parquet is *.parquet that does NOT start with '__' (CLAUDE.md inv #8)."""
    return filename.endswith(".parquet") and not filename.startswith("__")


def _pid_from_filename(filename: str) -> str:
    """Strip .parquet suffix, restore '/' from '_' (matches build_marketcap_parquet)."""
    name = filename[:-len(".parquet")]
    return name  # we use '-' separators (e.g. BTC-USD), never '/', so no restore needed


def scan_history_parquets(directory: str) -> Dict[str, Dict[str, int]]:
    """Scan a directory of *.parquet OHLCV files and return per-pid coverage stats.

    Schema assumed: a 'start' column in epoch seconds. Files starting with '__'
    (e.g. __MACRO__.parquet) are skipped. Missing directory returns {}.
    """
    if not os.path.isdir(directory):
        return {}
    out: Dict[str, Dict[str, int]] = {}
    for entry in sorted(os.listdir(directory)):
        if not _is_pid_parquet(entry):
            continue
        path = os.path.join(directory, entry)
        try:
            table = pq.read_table(path, columns=["start"])
        except Exception:
            continue
        starts = table.column("start").to_pylist()
        if not starts:
            continue
        pid = _pid_from_filename(entry)
        out[pid] = {
            "first_ts": int(min(starts)),
            "last_ts":  int(max(starts)),
            "n_rows":   len(starts),
        }
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Add failing tests for `scan_marketcap_parquets` and `scan_1m_dir`**

Append to `backend/tests/tools/strategy_discovery/test_inventory.py`:

```python
def _write_marketcap_parquet(path: str, starts: list[int]) -> None:
    """Helper: write a marketcap parquet matching the bronze schema."""
    n = len(starts)
    table = pa.table({
        "start":          starts,
        "market_cap":     [1e9] * n,
        "fdv":            [1.5e9] * n,
        "volume_24h":     [1e7] * n,
        "ingest_ts":      [1_700_000_000] * n,
        "schema_version": [2] * n,
    })
    pq.write_table(table, path, compression="snappy")


def test_scan_marketcap_parquets_returns_per_pid_coverage(tmp_path):
    mdir = tmp_path / "marketcap"
    mdir.mkdir()
    _write_marketcap_parquet(str(mdir / "BTC-USD.parquet"), starts=[1_700_000_000, 1_700_086_400])

    out = inventory.scan_marketcap_parquets(str(mdir))

    assert set(out.keys()) == {"BTC-USD"}
    assert out["BTC-USD"]["n_rows"] == 2
    assert out["BTC-USD"]["first_ts"] == 1_700_000_000
    assert out["BTC-USD"]["last_ts"] == 1_700_086_400


def test_scan_1m_dir_counts_files_per_pid_subdir(tmp_path):
    """1m candles live under <1m_dir>/<pid>/*.parquet. Returns {pid: n_files}."""
    m1 = tmp_path / "1m"
    m1.mkdir()
    btc_dir = m1 / "BTC-USD"
    btc_dir.mkdir()
    (btc_dir / "2026-05.parquet").write_bytes(b"x")
    (btc_dir / "2026-04.parquet").write_bytes(b"x")
    eth_dir = m1 / "ETH-USD"
    eth_dir.mkdir()
    # eth_dir empty

    out = inventory.scan_1m_dir(str(m1))

    assert out == {"BTC-USD": 2, "ETH-USD": 0}


def test_scan_1m_dir_missing_returns_empty(tmp_path):
    """Missing 1m dir returns {}, not exception."""
    assert inventory.scan_1m_dir(str(tmp_path / "no_such_dir")) == {}
```

- [ ] **Step 7: Run the new tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py -v
```

Expected: 3 new failures with `AttributeError: module 'tools.strategy_discovery.inventory' has no attribute 'scan_marketcap_parquets'` / `scan_1m_dir`.

- [ ] **Step 8: Implement `scan_marketcap_parquets` and `scan_1m_dir`**

Append to `backend/tools/strategy_discovery/inventory.py`:

```python
def scan_marketcap_parquets(directory: str) -> Dict[str, Dict[str, int]]:
    """Same shape as scan_history_parquets but for marketcap bronze parquets."""
    return scan_history_parquets(directory)  # identical scan — both keyed on 'start'


def scan_1m_dir(directory: str) -> Dict[str, int]:
    """Scan a 1-minute candles directory laid out as <dir>/<pid>/*.parquet.

    Returns {pid: n_files} for each pid subdirectory. Pids with no files yet
    return n_files=0 (subdir exists but empty — SP1 infrastructure shipped
    but populated status varies). Missing directory returns {}.
    """
    if not os.path.isdir(directory):
        return {}
    out: Dict[str, int] = {}
    for entry in sorted(os.listdir(directory)):
        sub = os.path.join(directory, entry)
        if not os.path.isdir(sub):
            continue
        n = sum(1 for f in os.listdir(sub) if f.endswith(".parquet"))
        out[entry] = n
    return out
```

- [ ] **Step 9: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py -v
```

Expected: 6 passed.

- [ ] **Step 10: Add failing test for `inventory_report` (Markdown rendering)**

Append to `backend/tests/tools/strategy_discovery/test_inventory.py`:

```python
def test_inventory_report_contains_section_headers_and_pid_counts():
    """inventory_report renders Markdown with required sections and the totals."""
    history    = {"BTC-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 4200},
                  "ETH-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 4200}}
    marketcap  = {"BTC-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 175}}
    minute1    = {"BTC-USD": 12, "ETH-USD": 0}

    md = inventory.inventory_report(history=history, marketcap=marketcap, minute1=minute1)

    assert "# Local Data Inventory" in md
    assert "## 1-hour OHLCV (`backend/data/history/`)" in md
    assert "## CoinPaprika tokenomic (`backend/data/marketcap/`)" in md
    assert "## 1-minute OHLCV (`backend/data/history/1m/`)" in md
    # Pids appear in the tables
    assert "BTC-USD" in md
    assert "ETH-USD" in md
    # Counts surface in the summary line
    assert "2 pids" in md or "2  pids" in md or "Total: 2" in md  # accept any reasonable phrasing
```

- [ ] **Step 11: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py::test_inventory_report_contains_section_headers_and_pid_counts -v
```

Expected: 1 failure with `AttributeError: ... 'inventory_report'`.

- [ ] **Step 12: Implement `inventory_report`**

Append to `backend/tools/strategy_discovery/inventory.py`:

```python
def _ts_to_iso_date(ts: int) -> str:
    import datetime as _dt
    return _dt.datetime.fromtimestamp(int(ts), tz=_dt.timezone.utc).strftime("%Y-%m-%d")


def _days_span(first_ts: int, last_ts: int) -> int:
    return max(0, (int(last_ts) - int(first_ts)) // 86400)


def inventory_report(
    *,
    history:   Dict[str, Dict[str, int]],
    marketcap: Dict[str, Dict[str, int]],
    minute1:   Dict[str, int],
) -> str:
    """Render the three inventory dicts as a single Markdown report."""
    lines: list[str] = []
    lines.append("# Local Data Inventory")
    lines.append("")
    lines.append(f"Generated for the 2026-05-23 strategy-discovery rebuild data-first directive.")
    lines.append("")

    # --- 1-hour OHLCV ---
    lines.append("## 1-hour OHLCV (`backend/data/history/`)")
    lines.append("")
    lines.append(f"Total: {len(history)} pids")
    lines.append("")
    lines.append("| pid | first | last | days | rows |")
    lines.append("|---|---|---|---:|---:|")
    for pid in sorted(history.keys()):
        h = history[pid]
        days = _days_span(h["first_ts"], h["last_ts"])
        lines.append(
            f"| {pid} | {_ts_to_iso_date(h['first_ts'])} | "
            f"{_ts_to_iso_date(h['last_ts'])} | {days} | {h['n_rows']:,} |"
        )
    lines.append("")

    # --- Marketcap (CoinPaprika tokenomic) ---
    lines.append("## CoinPaprika tokenomic (`backend/data/marketcap/`)")
    lines.append("")
    lines.append(f"Total: {len(marketcap)} pids")
    lines.append("")
    if marketcap:
        lines.append("| pid | first | last | days | rows |")
        lines.append("|---|---|---|---:|---:|")
        for pid in sorted(marketcap.keys()):
            m = marketcap[pid]
            days = _days_span(m["first_ts"], m["last_ts"])
            lines.append(
                f"| {pid} | {_ts_to_iso_date(m['first_ts'])} | "
                f"{_ts_to_iso_date(m['last_ts'])} | {days} | {m['n_rows']:,} |"
            )
    else:
        lines.append("(none)")
    lines.append("")

    # --- 1-minute OHLCV ---
    lines.append("## 1-minute OHLCV (`backend/data/history/1m/`)")
    lines.append("")
    lines.append(f"Total: {len(minute1)} pid directories")
    lines.append("")
    if minute1:
        lines.append("| pid | n_files |")
        lines.append("|---|---:|")
        for pid in sorted(minute1.keys()):
            lines.append(f"| {pid} | {minute1[pid]} |")
    else:
        lines.append("(no 1m directory or empty)")
    lines.append("")

    return "\n".join(lines) + "\n"
```

- [ ] **Step 13: Run all inventory tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_inventory.py -v
```

Expected: 7 passed.

- [ ] **Step 14: Add the CLI entry point**

Append to `backend/tools/strategy_discovery/inventory.py`:

```python
def main() -> int:
    """CLI: write Markdown report + JSON sidecar to docs/superpowers/specs/.

    The JSON sidecar is the structured data Task 5 (universe curation) consumes.
    """
    import argparse, sys
    parser = argparse.ArgumentParser(description="Audit local data for strategy-discovery rebuild.")
    parser.add_argument("--history-dir", default=DEFAULT_HISTORY_DIR)
    parser.add_argument("--marketcap-dir", default=DEFAULT_MARKETCAP_DIR)
    parser.add_argument("--minute1-dir", default=DEFAULT_1M_DIR)
    parser.add_argument(
        "--out-md",
        default=os.path.join(BACKEND_DIR, "..", "docs", "superpowers", "specs",
                             "2026-05-23-data-inventory-report.md"),
    )
    parser.add_argument(
        "--out-json",
        default=os.path.join(BACKEND_DIR, "..", "docs", "superpowers", "specs",
                             "2026-05-23-data-inventory.json"),
    )
    args = parser.parse_args()

    history   = scan_history_parquets(args.history_dir)
    marketcap = scan_marketcap_parquets(args.marketcap_dir)
    minute1   = scan_1m_dir(args.minute1_dir)

    md = inventory_report(history=history, marketcap=marketcap, minute1=minute1)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"history": history, "marketcap": marketcap, "minute1": minute1}, f, indent=2)

    print(f"  history:   {len(history)} pids", flush=True)
    print(f"  marketcap: {len(marketcap)} pids", flush=True)
    print(f"  1m dirs:   {len(minute1)} pids", flush=True)
    print(f"  wrote: {args.out_md}", flush=True)
    print(f"  wrote: {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
```

- [ ] **Step 15: Run the CLI on the live data**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m tools.strategy_discovery.inventory
```

Expected: prints non-zero pid counts for history (~220) and marketcap (3), and writes both files. Open the Markdown report and skim it to confirm the format is sensible.

- [ ] **Step 16: Update CHANGELOG.md**

Append to the "Unreleased" section of `CHANGELOG.md`:

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 1** — local data inventory script (`tools/strategy_discovery/inventory.py`) writes a Markdown report + JSON sidecar of what 1h OHLCV / CoinPaprika tokenomic / 1m OHLCV data exists locally. Read-only; no API calls. Outputs feed universe curation (Task 5). Per the data-first directive.
```

- [ ] **Step 17: Commit**

```powershell
cd C:\Users\gl450\polymarket_app
git add backend/tools/strategy_discovery/__init__.py `
        backend/tools/strategy_discovery/inventory.py `
        backend/tests/tools/strategy_discovery/__init__.py `
        backend/tests/tools/strategy_discovery/test_inventory.py `
        docs/superpowers/specs/2026-05-23-data-inventory-report.md `
        docs/superpowers/specs/2026-05-23-data-inventory.json `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 1 — local data inventory script"
git push origin HEAD
```

---

## Task 2: CoinPaprika supply-snapshot fetcher

Extend `services/coinpaprika_marketcap.py` with a function that fetches `circulating_supply`, `total_supply`, and `max_supply` from CoinPaprika's `/v1/tickers/{cp_id}` endpoint. This is a snapshot (single value per pid), not a timeseries — supply changes slowly enough that a snapshot is acceptable per the brainstorm spec ("Derived from current supply snapshot").

**Files:**
- Modify: `backend/services/coinpaprika_marketcap.py` (add `fetch_supply_snapshot` and the URL constant)
- Create: `backend/tests/services/test_coinpaprika_supply.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/services/test_coinpaprika_supply.py`:

```python
"""Tests for the CoinPaprika current-ticker supply-snapshot fetcher."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from services.coinpaprika_marketcap import fetch_supply_snapshot


class _MockResp:
    def __init__(self, *, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
    def json(self):
        return self._payload


def _mock_client(resp: _MockResp):
    cm = AsyncMock()
    cm.__aenter__.return_value = AsyncMock(get=AsyncMock(return_value=resp))
    cm.__aexit__.return_value = None
    return cm


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_tuple_for_mapped_pid():
    """fetch_supply_snapshot('BTC-USD') returns (circulating, total, max_or_None)."""
    payload = {
        "id": "btc-bitcoin",
        "symbol": "BTC",
        "circulating_supply": 19_700_000.0,
        "total_supply":       19_700_000.0,
        "max_supply":         21_000_000.0,
        "quotes": {"USD": {"price": 70000.0, "market_cap": 1.4e12}},
    }
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=200, payload=payload)),
    ):
        result = await fetch_supply_snapshot("BTC-USD")

    assert result is not None
    circ, total, max_supply = result
    assert circ  == 19_700_000.0
    assert total == 19_700_000.0
    assert max_supply == 21_000_000.0


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_none_for_unmapped_pid():
    """Unmapped pid returns None without hitting the network."""
    result = await fetch_supply_snapshot("ZZZ-USD")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_returns_none_on_non_200():
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=404, payload={})),
    ):
        result = await fetch_supply_snapshot("BTC-USD")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_handles_missing_max_supply():
    """Some tokens (e.g. ETH) have null max_supply — return None in that slot, not 0."""
    payload = {
        "id": "eth-ethereum",
        "circulating_supply": 120_000_000.0,
        "total_supply":       120_000_000.0,
        "max_supply":         None,
    }
    with patch(
        "services.coinpaprika_marketcap.httpx.AsyncClient",
        return_value=_mock_client(_MockResp(status_code=200, payload=payload)),
    ):
        result = await fetch_supply_snapshot("ETH-USD")

    assert result is not None
    circ, total, max_supply = result
    assert circ == 120_000_000.0
    assert total == 120_000_000.0
    assert max_supply is None


@pytest.mark.asyncio
async def test_fetch_supply_snapshot_respects_disabled_env(monkeypatch):
    """COINPAPRIKA_DISABLED=1 short-circuits without HTTP call."""
    monkeypatch.setenv("COINPAPRIKA_DISABLED", "1")
    result = await fetch_supply_snapshot("BTC-USD")
    assert result is None
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m pytest tests/services/test_coinpaprika_supply.py -v
```

Expected: 5 failures with `ImportError: cannot import name 'fetch_supply_snapshot' from 'services.coinpaprika_marketcap'`.

- [ ] **Step 3: Implement `fetch_supply_snapshot`**

In `backend/services/coinpaprika_marketcap.py`, find the section after the `_HISTORY_URL` constant (~line 42) and add:

```python
_TICKER_URL  = f"{_BASE}/tickers/{{cp_id}}"
```

Then append a new function after `fetch_marketcap_history` (which ends around line 186):

```python
async def fetch_supply_snapshot(
    product_id: str,
) -> Optional[Tuple[float, float, Optional[float]]]:
    """Current-ticker supply snapshot for one Coinbase pid.

    Returns (circulating_supply, total_supply, max_supply_or_None) or None on
    any failure (unmapped pid, disabled, non-200, malformed body, missing
    circulating/total).

    max_supply is None for tokens with no fixed cap (e.g. ETH) — the
    distinction matters because it changes how FDV is interpreted downstream.

    Endpoint: GET /v1/tickers/{cp_id}  (no key, free tier).
    """
    if _is_disabled():
        return None

    cp_id = _coinbase_to_cp_id(product_id)
    if cp_id is None:
        return None

    url = _TICKER_URL.format(cp_id=cp_id)

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.get(url)
    except Exception as e:
        logger.warning(
            "coinpaprika_marketcap ticker HTTP error pid=%s: %r", product_id, e
        )
        return None

    if resp.status_code != 200:
        logger.warning(
            "coinpaprika_marketcap ticker non-200 pid=%s status=%d",
            product_id, resp.status_code,
        )
        return None

    try:
        body = resp.json()
    except Exception:
        return None

    if not isinstance(body, dict):
        return None

    try:
        circ  = float(body["circulating_supply"])
        total = float(body["total_supply"])
    except (KeyError, TypeError, ValueError):
        return None

    max_raw = body.get("max_supply")
    max_supply: Optional[float]
    if max_raw is None:
        max_supply = None
    else:
        try:
            max_supply = float(max_raw)
        except (TypeError, ValueError):
            max_supply = None

    return (circ, total, max_supply)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/services/test_coinpaprika_supply.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Update CHANGELOG.md**

Append to the "Unreleased" section:

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 2** — added `fetch_supply_snapshot(pid)` to `services/coinpaprika_marketcap.py`. Hits `/v1/tickers/{cp_id}` and returns `(circulating_supply, total_supply, max_supply_or_None)`. Enables FDV computation (`price × total_supply`) and circ/total ratio for the rebuild's tokenomic feature set.
```

- [ ] **Step 6: Commit**

```powershell
cd C:\Users\gl450\polymarket_app
git add backend/services/coinpaprika_marketcap.py `
        backend/tests/services/test_coinpaprika_supply.py `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 2 — CoinPaprika supply-snapshot fetcher"
git push origin HEAD
```

---

## Task 3: Supply-snapshot parquet writer + CLI

Persists supply snapshots to `backend/data/supply/snapshot.parquet` so downstream tasks (curation, feature build) read locally rather than re-fetching. Single combined parquet keyed by pid — one row per pid, current-snapshot semantics.

**Files:**
- Create: `backend/tools/strategy_discovery/build_supply_snapshot.py`
- Create: `backend/tests/tools/test_build_supply_snapshot.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test for the parquet read/write round-trip**

Create `backend/tests/tools/test_build_supply_snapshot.py`:

```python
"""Tests for supply-snapshot parquet writer."""
from __future__ import annotations

import os
from unittest.mock import AsyncMock, patch

import pytest

from tools.strategy_discovery import build_supply_snapshot as bss


def test_save_and_load_supply_snapshot_roundtrip(tmp_path):
    """save -> load returns the same rows."""
    rows = [
        {"pid": "BTC-USD", "circulating": 19_700_000.0, "total": 19_700_000.0,
         "max_supply":  21_000_000.0, "ingest_ts": 1_700_000_000},
        {"pid": "ETH-USD", "circulating": 120_000_000.0, "total": 120_000_000.0,
         "max_supply":  None,         "ingest_ts": 1_700_000_000},
    ]
    path = tmp_path / "snapshot.parquet"
    bss.save_snapshot(str(path), rows)
    loaded = bss.load_snapshot(str(path))

    assert len(loaded) == 2
    by_pid = {r["pid"]: r for r in loaded}
    assert by_pid["BTC-USD"]["circulating"] == 19_700_000.0
    assert by_pid["BTC-USD"]["max_supply"]  == 21_000_000.0
    assert by_pid["ETH-USD"]["max_supply"] is None


def test_load_snapshot_missing_returns_empty_list(tmp_path):
    assert bss.load_snapshot(str(tmp_path / "no_such.parquet")) == []


def test_save_snapshot_dedups_by_pid_last_wins(tmp_path):
    """If the same pid appears twice in input, the LAST row wins."""
    rows = [
        {"pid": "BTC-USD", "circulating": 1.0, "total": 1.0, "max_supply": 21_000_000.0, "ingest_ts": 100},
        {"pid": "BTC-USD", "circulating": 2.0, "total": 2.0, "max_supply": 21_000_000.0, "ingest_ts": 200},
    ]
    path = tmp_path / "snapshot.parquet"
    bss.save_snapshot(str(path), rows)
    loaded = bss.load_snapshot(str(path))
    assert len(loaded) == 1
    assert loaded[0]["circulating"] == 2.0
    assert loaded[0]["ingest_ts"] == 200


@pytest.mark.asyncio
async def test_fetch_and_persist_one_pid_merges_with_existing(tmp_path, monkeypatch):
    """fetch_and_persist appends a new pid's snapshot to the existing parquet."""
    path = tmp_path / "snapshot.parquet"
    # Seed with ETH already present.
    bss.save_snapshot(str(path), [
        {"pid": "ETH-USD", "circulating": 120e6, "total": 120e6, "max_supply": None, "ingest_ts": 100},
    ])

    async def _fake_fetch(pid):
        return (19_700_000.0, 19_700_000.0, 21_000_000.0)

    monkeypatch.setattr(bss, "fetch_supply_snapshot", _fake_fetch)
    monkeypatch.setattr(bss, "_now_ts", lambda: 200)

    await bss.fetch_and_persist(["BTC-USD"], parquet_path=str(path))

    loaded = bss.load_snapshot(str(path))
    by_pid = {r["pid"]: r for r in loaded}
    assert set(by_pid.keys()) == {"BTC-USD", "ETH-USD"}
    assert by_pid["BTC-USD"]["circulating"] == 19_700_000.0
    assert by_pid["BTC-USD"]["ingest_ts"] == 200
    # ETH untouched
    assert by_pid["ETH-USD"]["ingest_ts"] == 100
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m pytest tests/tools/test_build_supply_snapshot.py -v
```

Expected: 4 failures with `ModuleNotFoundError: No module named 'tools.strategy_discovery.build_supply_snapshot'`.

- [ ] **Step 3: Implement the module**

Create `backend/tools/strategy_discovery/build_supply_snapshot.py`:

```python
"""Supply-snapshot parquet writer + CLI for the strategy-discovery rebuild.

Hits CoinPaprika's /v1/tickers/{cp_id} endpoint (via
services.coinpaprika_marketcap.fetch_supply_snapshot) and persists one row per
pid to backend/data/supply/snapshot.parquet.

Schema:
  pid          : string   (Coinbase product id, e.g. "BTC-USD")
  circulating  : float64  (tokens in market)
  total        : float64  (tokens that will eventually exist; max may exceed)
  max_supply   : float64? (None for uncapped tokens like ETH; stored as null)
  ingest_ts    : int64    (wall-clock epoch seconds at write time)
  schema_version : int32  (currently 1)

On `pid`-collision dedup, the LAST row in input wins (mirrors
build_marketcap_parquet semantics).
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

# Ensure backend/ is importable when this file is run as a script.
BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services.coinpaprika_marketcap import fetch_supply_snapshot  # noqa: E402

_SCHEMA_VERSION = 1
_SCHEMA = pa.schema([
    pa.field("pid",            pa.string()),
    pa.field("circulating",    pa.float64()),
    pa.field("total",          pa.float64()),
    pa.field("max_supply",     pa.float64()),
    pa.field("ingest_ts",      pa.int64()),
    pa.field("schema_version", pa.int32()),
])

_DEFAULT_PATH = os.path.join(BACKEND, "data", "supply", "snapshot.parquet")


def _now_ts() -> int:
    return int(time.time())


def save_snapshot(path: str, rows: List[Dict]) -> None:
    """Write deduplicated supply-snapshot rows to a parquet path.

    Dedup: on pid-collision, the last row in input wins.
    `max_supply=None` is stored as a parquet null (pa.float64() is nullable).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    seen: Dict[str, Dict] = {}
    for r in rows:
        pid = str(r["pid"])
        seen[pid] = {
            "pid":            pid,
            "circulating":    float(r["circulating"]),
            "total":          float(r["total"]),
            "max_supply":     None if r.get("max_supply") is None else float(r["max_supply"]),
            "ingest_ts":      int(r.get("ingest_ts", _now_ts())),
            "schema_version": int(r.get("schema_version", _SCHEMA_VERSION)),
        }
    ordered = sorted(seen.values(), key=lambda r: r["pid"])
    table = pa.table(
        {
            "pid":            [r["pid"]            for r in ordered],
            "circulating":    [r["circulating"]    for r in ordered],
            "total":          [r["total"]          for r in ordered],
            "max_supply":     [r["max_supply"]     for r in ordered],
            "ingest_ts":      [r["ingest_ts"]      for r in ordered],
            "schema_version": [r["schema_version"] for r in ordered],
        },
        schema=_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def load_snapshot(path: str) -> List[Dict]:
    """Read supply-snapshot rows from a parquet path. Missing -> []."""
    if not os.path.exists(path):
        return []
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = len(cols["pid"])
    out: List[Dict] = []
    for i in range(n):
        out.append({
            "pid":            str(cols["pid"][i]),
            "circulating":    float(cols["circulating"][i]),
            "total":          float(cols["total"][i]),
            "max_supply":     None if cols["max_supply"][i] is None else float(cols["max_supply"][i]),
            "ingest_ts":      int(cols["ingest_ts"][i]),
            "schema_version": int(cols["schema_version"][i]),
        })
    return out


async def fetch_and_persist(
    pids: List[str],
    *,
    parquet_path: str = _DEFAULT_PATH,
    sleep_secs: float = 0.5,
) -> Dict[str, Optional[tuple]]:
    """Fetch supply snapshots for each pid and merge into the parquet at parquet_path.

    Returns {pid: (circ, total, max_or_None) | None} for caller diagnostics.
    Existing rows for pids not in `pids` are preserved.
    A `sleep_secs` delay between fetches keeps us friendly with the free tier
    (default 0.5s ~ 7,200 req/hour, well under CoinPaprika's 25k/day).
    """
    existing = load_snapshot(parquet_path)
    merged: Dict[str, Dict] = {r["pid"]: r for r in existing}
    results: Dict[str, Optional[tuple]] = {}
    now = _now_ts()

    for pid in pids:
        snap = await fetch_supply_snapshot(pid)
        results[pid] = snap
        if snap is None:
            continue
        circ, total, max_supply = snap
        merged[pid] = {
            "pid":            pid,
            "circulating":    circ,
            "total":          total,
            "max_supply":     max_supply,
            "ingest_ts":      now,
            "schema_version": _SCHEMA_VERSION,
        }
        await asyncio.sleep(sleep_secs)

    save_snapshot(parquet_path, list(merged.values()))
    return results


def main() -> int:
    """CLI: fetch supply snapshots for a comma-separated pid list.

    Run:
        cd backend && python -m tools.strategy_discovery.build_supply_snapshot \\
            --pids BTC-USD,ETH-USD,SOL-USD
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pids", required=True, help="comma-separated, e.g. BTC-USD,ETH-USD")
    parser.add_argument("--out", default=_DEFAULT_PATH)
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds between requests")
    args = parser.parse_args()

    pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    print(f"  fetching supply snapshots for {len(pids)} pids -> {args.out}", flush=True)

    results = asyncio.run(fetch_and_persist(pids, parquet_path=args.out, sleep_secs=args.sleep))

    ok    = [p for p, v in results.items() if v is not None]
    failed = [p for p, v in results.items() if v is None]
    print(f"  ok:     {len(ok)}", flush=True)
    print(f"  failed: {len(failed)}  -> {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/test_build_supply_snapshot.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Update CHANGELOG.md**

Append to "Unreleased":

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 3** — supply-snapshot parquet writer (`tools/strategy_discovery/build_supply_snapshot.py`). Persists `(pid, circulating, total, max_supply, ingest_ts)` rows to `backend/data/supply/snapshot.parquet`. CLI fetches via `fetch_supply_snapshot` with rate-limit-friendly sleep; merges with existing parquet (dedup by pid, last wins).
```

- [ ] **Step 6: Commit**

```powershell
cd C:\Users\gl450\polymarket_app
git add backend/tools/strategy_discovery/build_supply_snapshot.py `
        backend/tests/tools/test_build_supply_snapshot.py `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 3 — supply-snapshot parquet writer"
git push origin HEAD
```

---

## Task 4: Extend the Coinbase→CoinPaprika pid mapping for the candidate pool

The universe curation in Task 5 needs supply data for a CANDIDATE pool larger than 50 — say ~100-150 Coinbase USD spot pids with viable history. Today the static mapping in `services/coinpaprika_marketcap.py:_PRODUCT_TO_CP_ID` has ~28 entries, mostly large-caps. We need to broaden it.

**Approach:** add the mapping entries hand-curated from the Task 1 inventory output (the pids with the longest history are the highest-value mapping targets). This is a one-time data-entry task; no algorithmic discovery needed.

**How to derive the list to add:** open `docs/superpowers/specs/2026-05-23-data-inventory-report.md` (produced by Task 1) and pick the top ~120 pids by `days` column. Cross-reference each against `https://api.coinpaprika.com/v1/coins` (a one-time browser/curl lookup the implementer does manually) to get the `id` field. Add entries to the dict.

**Files:**
- Modify: `backend/services/coinpaprika_marketcap.py` (extend `_PRODUCT_TO_CP_ID`)
- Modify: `backend/tests/services/test_coinpaprika_marketcap.py` (add a count assertion)
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write a failing test that asserts the mapping has grown**

Open `backend/tests/services/test_coinpaprika_marketcap.py` and add:

```python
def test_product_to_cp_id_has_at_least_100_entries():
    """The mapping was extended in 2026-05-23 to support the strategy-discovery
    rebuild's 50-pid universe curation (needs a ~100+ candidate pool)."""
    from services.coinpaprika_marketcap import _PRODUCT_TO_CP_ID
    assert len(_PRODUCT_TO_CP_ID) >= 100, (
        f"_PRODUCT_TO_CP_ID has only {len(_PRODUCT_TO_CP_ID)} entries; "
        f"Phase 1 Task 4 requires ≥ 100 for universe curation candidate pool"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/services/test_coinpaprika_marketcap.py::test_product_to_cp_id_has_at_least_100_entries -v
```

Expected: 1 failure asserting `28 >= 100` is False.

- [ ] **Step 3: Build the candidate list**

Run the inventory CLI from Task 1 if you haven't recently:

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m tools.strategy_discovery.inventory
```

Then open `docs/superpowers/specs/2026-05-23-data-inventory-report.md` and pick the top ~100 pids by the `days` column (longest history first). For each one, look up the CoinPaprika id by visiting:

```
https://api.coinpaprika.com/v1/coins
```

This is a single JSON document (~5MB) listing all coins. Search it for the ticker (case-insensitive). The first entry whose `symbol` field matches and whose `is_active` is true is the canonical id.

Alternative one-shot lookup script (run from REPL, do NOT add as a production tool):

```python
import asyncio, httpx
async def find(symbols):
    async with httpx.AsyncClient(timeout=60) as c:
        coins = (await c.get("https://api.coinpaprika.com/v1/coins")).json()
    by_sym = {}
    for c in coins:
        if c.get("is_active"):
            by_sym.setdefault(c["symbol"].upper(), c["id"])  # first wins (rank order)
    return {s: by_sym.get(s.split("-")[0].upper()) for s in symbols}
print(asyncio.run(find(["DOGE-USD", "AVAX-USD"])))
```

- [ ] **Step 4: Edit `_PRODUCT_TO_CP_ID` to add the new entries**

In `backend/services/coinpaprika_marketcap.py`, extend the `_PRODUCT_TO_CP_ID` dict. Keep existing entries; add new ones alphabetically grouped. Example new entries (representative — actual list comes from Step 3):

```python
_PRODUCT_TO_CP_ID = {
    # ... existing 28 entries ...
    "MATIC-USD":   "matic-polygon",
    "UNI-USD":     "uni-uniswap",
    "LTC-USD":     "ltc-litecoin",
    "ATOM-USD":    "atom-cosmos",
    "NEAR-USD":    "near-near-protocol",
    # ... (continue until ≥ 100 entries total) ...
}
```

The exact set is determined by Step 3's lookup. Document any pid that can't be mapped (e.g. wrapped tokens, exchange-only listings) as a comment so future maintenance knows it was intentionally skipped.

- [ ] **Step 5: Run the test to verify it passes**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/services/test_coinpaprika_marketcap.py -v
```

Expected: the new test passes (mapping ≥ 100), plus existing CoinPaprika tests still pass.

- [ ] **Step 6: Update CHANGELOG.md**

Append to "Unreleased":

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 4** — extended `services/coinpaprika_marketcap._PRODUCT_TO_CP_ID` from 28 to ≥100 entries. Sourced from Task 1's inventory (longest-history pids first), looked up via CoinPaprika's public `/v1/coins` endpoint. Feeds the candidate pool that Task 5 curates to 50 pids.
```

- [ ] **Step 7: Commit**

```powershell
git add backend/services/coinpaprika_marketcap.py `
        backend/tests/services/test_coinpaprika_marketcap.py `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 4 — extend CoinPaprika pid mapping to 100+ entries"
git push origin HEAD
```

---

## Task 5: Universe curation — apply Q5 criteria, pick 50 pids

Consumes the Task 1 inventory + Task 3 supply snapshot + existing marketcap parquets, applies the Q5 criteria from the brainstorm spec, and outputs the 50-pid universe.

**Q5 criteria (from spec):**
- ~15 large-cap (top by market_cap)
- ~15 mid-cap (next tier by market_cap)
- ~10 high FDV/MC ratio (= total_supply / circulating_supply, ranking by descending ratio after excluding pids already picked for large/mid)
- ~10 low turnover ratio (= 24h_volume / market_cap, ascending after excluding already-picked)
- All filtered to require ≥ 6 months of 1h OHLCV AND CoinPaprika supply data

For pids without a marketcap parquet yet, fall back to the supply snapshot's `quotes.USD.market_cap` (we'd need to capture that in Task 2). Alternative: do Task 5 AFTER Task 6 (the bulk backfill). Cleanest order:

Actually, to keep the dependency tree linear, Task 5 will use:
1. Inventory JSON (Task 1) — for "≥ 6 months 1h OHLCV" filter
2. Supply snapshot parquet (Task 3) — for FDV/MC and circ/total ratios
3. The CURRENT market_cap, which we'll source from a one-time CoinPaprika `/v1/tickers/{cp_id}` snapshot during Task 5 itself (single call per candidate, cached in the supply snapshot parquet — see below).

**Decision:** Task 5 will EXTEND Task 3's supply snapshot to include a `current_mc` and `volume_24h` column, so we have everything we need to rank. We need to revisit Task 2 / Task 3 to ensure the snapshot includes these. To avoid refactoring Tasks 2-3, Task 5 adds a separate "current marketcap" parquet path.

**Simpler approach:** Use the existing `fetch_marketcap_history(pid, start=yesterday, end=today)` to get the most recent daily MC + volume row. Already implemented in `services/coinpaprika_marketcap.py`. No new endpoint needed.

**Files:**
- Create: `backend/tools/strategy_discovery/curate_universe.py`
- Create: `backend/tests/tools/strategy_discovery/test_curate_universe.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test for `rank_by_market_cap`**

Create `backend/tests/tools/strategy_discovery/test_curate_universe.py`:

```python
"""Tests for universe-curation logic."""
from __future__ import annotations

import pytest

from tools.strategy_discovery import curate_universe as cu


def test_rank_by_market_cap_descending():
    """rank_by_market_cap returns pids sorted by descending mc."""
    candidates = {
        "BTC-USD": {"market_cap": 1.4e12, "volume_24h": 5e10, "circulating": 19.7e6, "total": 19.7e6},
        "DOGE-USD": {"market_cap": 2e10,  "volume_24h": 1e9,  "circulating": 140e9,  "total": 140e9},
        "ETH-USD": {"market_cap": 4e11,  "volume_24h": 2e10, "circulating": 120e6,  "total": 120e6},
    }
    ranked = cu.rank_by_market_cap(candidates)
    assert ranked == ["BTC-USD", "ETH-USD", "DOGE-USD"]


def test_rank_by_fdv_mc_ratio_descending_excludes_already_picked():
    """rank_by_fdv_mc_ratio excludes the already-picked pids."""
    candidates = {
        "A": {"market_cap": 1e9, "volume_24h": 1e7, "circulating": 100.0, "total": 1000.0},  # ratio 10x
        "B": {"market_cap": 1e9, "volume_24h": 1e7, "circulating": 500.0, "total": 1000.0},  # ratio 2x
        "C": {"market_cap": 1e9, "volume_24h": 1e7, "circulating": 200.0, "total": 1000.0},  # ratio 5x
    }
    ranked = cu.rank_by_fdv_mc_ratio(candidates, exclude={"A"})
    assert ranked == ["C", "B"]


def test_rank_by_turnover_ascending_excludes_already_picked():
    """rank_by_turnover sorts ascending (LOW turnover wins)."""
    candidates = {
        "A": {"market_cap": 1e9, "volume_24h": 1e8, "circulating": 1.0, "total": 1.0},  # turnover 0.1
        "B": {"market_cap": 1e9, "volume_24h": 1e6, "circulating": 1.0, "total": 1.0},  # turnover 0.001
        "C": {"market_cap": 1e9, "volume_24h": 5e7, "circulating": 1.0, "total": 1.0},  # turnover 0.05
    }
    ranked = cu.rank_by_turnover(candidates, exclude={"B"})
    assert ranked == ["C", "A"]


def test_curate_universe_picks_target_counts_with_overlap_handling():
    """End-to-end: given 100 candidates, picks (up to) 15+15+10+10 distinct pids."""
    # Synthesize 100 candidates with varied attributes.
    candidates = {}
    for i in range(100):
        mc  = (100 - i) * 1e9          # descending mc by i
        vol = 1e6 if i < 20 else 1e8   # first 20 have unusually low turnover
        ratio_circ = 100.0 if i >= 80 else 500.0   # last 20 have high fdv/mc ratio
        candidates[f"P{i:03d}"] = {
            "market_cap": mc,
            "volume_24h": vol,
            "circulating": ratio_circ,
            "total":       1000.0,
        }

    picked = cu.curate_universe(
        candidates,
        n_large=15, n_mid=15, n_high_fdv_ratio=10, n_low_turnover=10,
    )

    assert len(picked) == 15 + 15 + 10 + 10  # 50 — no overlap forces a smaller count in this synthetic
    # large/mid are the top-30 by mc
    large_mid = set(picked["large"]) | set(picked["mid"])
    assert large_mid == {f"P{i:03d}" for i in range(30)}
    # high_fdv_ratio drawn from i>=80 cohort (after excluding large/mid; here disjoint)
    assert all(int(p[1:]) >= 80 for p in picked["high_fdv_ratio"])
    # low_turnover drawn from i<20 cohort, but i<15 went to large_mid first
    assert all(int(p[1:]) >= 15 and int(p[1:]) < 20 for p in picked["low_turnover"])


def test_curate_universe_respects_overlap_no_pid_listed_twice():
    """Same pid can't appear in two cohorts — overlap forces the second cohort to skip it."""
    # 5 candidates where the same pid is best in BOTH high-fdv-ratio AND low-turnover.
    candidates = {
        "A": {"market_cap": 5e9, "volume_24h": 1e6, "circulating": 100.0, "total": 1000.0},   # best in BOTH
        "B": {"market_cap": 4e9, "volume_24h": 1e7, "circulating": 200.0, "total": 1000.0},
        "C": {"market_cap": 3e9, "volume_24h": 1e8, "circulating": 500.0, "total": 1000.0},
        "D": {"market_cap": 2e9, "volume_24h": 1e9, "circulating": 700.0, "total": 1000.0},
        "E": {"market_cap": 1e9, "volume_24h": 1e10, "circulating": 900.0, "total": 1000.0},
    }
    picked = cu.curate_universe(
        candidates, n_large=1, n_mid=1, n_high_fdv_ratio=1, n_low_turnover=1,
    )

    all_picks = [p for cohort in picked.values() for p in cohort]
    assert len(all_picks) == len(set(all_picks))  # no duplicates
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_curate_universe.py -v
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'tools.strategy_discovery.curate_universe'`.

- [ ] **Step 3: Implement the ranking helpers + `curate_universe`**

Create `backend/tools/strategy_discovery/curate_universe.py`:

```python
"""Universe curation for the strategy-discovery rebuild (Phase 1 Q5).

Applies the brainstorm's Q5 criteria to a candidate pool:
  ~15 large-cap (top by market_cap)
  ~15 mid-cap (next by market_cap)
  ~10 high FDV/MC = total_supply / circulating_supply (highest dilution overhang)
  ~10 low turnover = volume_24h / market_cap (lowest velocity)
All filtered to require >= 6 months of 1h OHLCV AND CoinPaprika supply data.

Pure ranking logic + CLI. No HTTP calls inside the ranking functions —
candidates are passed in pre-assembled by the CLI.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Dict, Iterable, List, Set

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Type: candidates: Dict[pid, Dict[market_cap, volume_24h, circulating, total]]
CandidateMap = Dict[str, Dict[str, float]]


def rank_by_market_cap(candidates: CandidateMap, exclude: Set[str] = frozenset()) -> List[str]:
    pids = [p for p in candidates if p not in exclude]
    return sorted(pids, key=lambda p: candidates[p]["market_cap"], reverse=True)


def rank_by_fdv_mc_ratio(candidates: CandidateMap, exclude: Set[str] = frozenset()) -> List[str]:
    """Rank by total/circulating descending (high dilution overhang first)."""
    pids = [p for p in candidates if p not in exclude]
    return sorted(
        pids,
        key=lambda p: candidates[p]["total"] / max(candidates[p]["circulating"], 1e-12),
        reverse=True,
    )


def rank_by_turnover(candidates: CandidateMap, exclude: Set[str] = frozenset()) -> List[str]:
    """Rank by volume_24h / market_cap ascending (low velocity first)."""
    pids = [p for p in candidates if p not in exclude]
    return sorted(
        pids,
        key=lambda p: candidates[p]["volume_24h"] / max(candidates[p]["market_cap"], 1e-12),
    )


def curate_universe(
    candidates: CandidateMap,
    *,
    n_large: int = 15,
    n_mid: int = 15,
    n_high_fdv_ratio: int = 10,
    n_low_turnover: int = 10,
) -> Dict[str, List[str]]:
    """Apply Q5 criteria. Returns {'large', 'mid', 'high_fdv_ratio', 'low_turnover'}.

    Cohorts are filled in order: large, mid, high_fdv_ratio, low_turnover.
    Each cohort excludes pids already picked by earlier cohorts. No pid
    appears in two cohorts.
    """
    by_mc = rank_by_market_cap(candidates)
    large = by_mc[:n_large]
    mid   = by_mc[n_large:n_large + n_mid]
    picked: Set[str] = set(large) | set(mid)

    high_fdv = rank_by_fdv_mc_ratio(candidates, exclude=picked)[:n_high_fdv_ratio]
    picked.update(high_fdv)

    low_turnover = rank_by_turnover(candidates, exclude=picked)[:n_low_turnover]

    return {
        "large":          large,
        "mid":            mid,
        "high_fdv_ratio": high_fdv,
        "low_turnover":   low_turnover,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_curate_universe.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Add the CLI orchestrator (assembles candidates, runs curate, writes outputs)**

Append to `backend/tools/strategy_discovery/curate_universe.py`:

```python
def _assemble_candidates_from_local(
    *,
    inventory_json_path: str,
    supply_parquet_path: str,
    marketcap_dir: str,
    min_history_days: int = 180,  # 6 months
) -> CandidateMap:
    """Build the candidate pool from local files.

    A pid enters the pool when ALL hold:
      - inventory shows >= min_history_days of 1h OHLCV
      - supply snapshot has the pid (so circ/total/max are known)
      - marketcap parquet has at least one recent row (for current MC + volume)

    Pids missing any of these are skipped (logged to stdout by the CLI).
    """
    from tools.strategy_discovery.build_supply_snapshot import load_snapshot
    import pyarrow.parquet as pq

    with open(inventory_json_path, "r", encoding="utf-8") as f:
        inv = json.load(f)
    history_inv = inv.get("history", {})

    supply_rows = {r["pid"]: r for r in load_snapshot(supply_parquet_path)}

    out: CandidateMap = {}
    for pid, hist in history_inv.items():
        days = (int(hist["last_ts"]) - int(hist["first_ts"])) // 86400
        if days < min_history_days:
            continue
        if pid not in supply_rows:
            continue
        mc_path = os.path.join(marketcap_dir, f"{pid}.parquet")
        if not os.path.exists(mc_path):
            continue
        table = pq.read_table(mc_path, columns=["market_cap", "volume_24h"])
        cols = table.to_pydict()
        if not cols["market_cap"]:
            continue
        # use the last (most recent) row
        out[pid] = {
            "market_cap": float(cols["market_cap"][-1]),
            "volume_24h": float(cols["volume_24h"][-1]),
            "circulating": float(supply_rows[pid]["circulating"]),
            "total":       float(supply_rows[pid]["total"]),
        }
    return out


def _render_universe_md(picked: Dict[str, List[str]], candidates: CandidateMap) -> str:
    """Render the picked universe as a Markdown rationale doc."""
    lines = [
        "# Strategy-Discovery Universe (50 pids)",
        "",
        "Generated by `tools/strategy_discovery/curate_universe.py` per Q5 of the "
        "2026-05-23 brainstorm spec.",
        "",
        "## Criteria (locked from Q5)",
        "",
        "- ~15 large-cap (top by market_cap)",
        "- ~15 mid-cap (next by market_cap)",
        "- ~10 high FDV/MC = total / circulating (high dilution overhang)",
        "- ~10 low turnover = volume_24h / market_cap (low velocity)",
        "- All filtered to >= 6 months 1h OHLCV + CoinPaprika supply data",
        "",
    ]
    for cohort, pids in picked.items():
        lines.append(f"## {cohort}  ({len(pids)} pids)")
        lines.append("")
        lines.append("| pid | market_cap | volume_24h | total/circ | vol/MC |")
        lines.append("|---|---:|---:|---:|---:|")
        for pid in pids:
            c = candidates[pid]
            ratio = c["total"] / max(c["circulating"], 1e-12)
            turnover = c["volume_24h"] / max(c["market_cap"], 1e-12)
            lines.append(
                f"| {pid} | {c['market_cap']:,.0f} | {c['volume_24h']:,.0f} "
                f"| {ratio:.3f} | {turnover:.4f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """CLI: read local inventory + supply + marketcap, curate, write outputs.

    Run:
        cd backend && python -m tools.strategy_discovery.curate_universe
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory-json",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-data-inventory.json"),
    )
    parser.add_argument(
        "--supply-parquet",
        default=os.path.join(BACKEND, "data", "supply", "snapshot.parquet"),
    )
    parser.add_argument(
        "--marketcap-dir",
        default=os.path.join(BACKEND, "data", "marketcap"),
    )
    parser.add_argument(
        "--out-md",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.md"),
    )
    parser.add_argument(
        "--out-json",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--n-large", type=int, default=15)
    parser.add_argument("--n-mid", type=int, default=15)
    parser.add_argument("--n-high-fdv-ratio", type=int, default=10)
    parser.add_argument("--n-low-turnover", type=int, default=10)
    parser.add_argument("--min-history-days", type=int, default=180)
    args = parser.parse_args()

    print("  assembling candidates from local data...", flush=True)
    candidates = _assemble_candidates_from_local(
        inventory_json_path=args.inventory_json,
        supply_parquet_path=args.supply_parquet,
        marketcap_dir=args.marketcap_dir,
        min_history_days=args.min_history_days,
    )
    print(f"  candidate pool: {len(candidates)} pids", flush=True)

    picked = curate_universe(
        candidates,
        n_large=args.n_large,
        n_mid=args.n_mid,
        n_high_fdv_ratio=args.n_high_fdv_ratio,
        n_low_turnover=args.n_low_turnover,
    )
    n_total = sum(len(v) for v in picked.values())
    print(f"  picked: {n_total} pids across {len(picked)} cohorts", flush=True)

    md = _render_universe_md(picked, candidates)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(md)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(picked, f, indent=2)

    print(f"  wrote: {args.out_md}", flush=True)
    print(f"  wrote: {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Run the unit tests again to confirm nothing regressed**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/strategy_discovery/test_curate_universe.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Update CHANGELOG.md**

Append to "Unreleased":

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 5** — universe curation script (`tools/strategy_discovery/curate_universe.py`). Reads Task 1 inventory + Task 3 supply snapshot + existing marketcap parquets, applies Q5 criteria (15 large / 15 mid / 10 high FDV-MC / 10 low turnover, ≥6 months history), writes `docs/superpowers/specs/2026-05-23-universe-50.{md,json}`.
```

- [ ] **Step 8: Commit**

```powershell
git add backend/tools/strategy_discovery/curate_universe.py `
        backend/tests/tools/strategy_discovery/test_curate_universe.py `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 5 — universe curation"
git push origin HEAD
```

---

## Task 6: Bulk marketcap historical backfill for the curated 50

Now that we know which 50 pids the rebuild targets, run the historical backfill for any pid in that list that doesn't already have a complete marketcap parquet. This task is a thin wrapper over the existing `tools/build_marketcap_parquet.py` CLI — it reads the curated pid list and dispatches the backfill, skipping pids already covered.

**Files:**
- Create: `backend/tools/strategy_discovery/build_universe_marketcap.py`
- Create: `backend/tests/tools/test_build_universe_marketcap.py`
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/tools/test_build_universe_marketcap.py`:

```python
"""Tests for the universe marketcap backfill orchestrator."""
from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery import build_universe_marketcap as bum


def _write_mc_parquet(path: str, starts: list[int]) -> None:
    n = len(starts)
    table = pa.table({
        "start":          starts,
        "market_cap":     [1e9] * n,
        "fdv":            [1e9] * n,
        "volume_24h":     [1e7] * n,
        "ingest_ts":      [1_700_000_000] * n,
        "schema_version": [2] * n,
    })
    pq.write_table(table, path, compression="snappy")


def test_pids_needing_backfill_filters_already_complete(tmp_path):
    """A pid with a parquet covering >= min_days is skipped."""
    mcdir = tmp_path / "marketcap"
    mcdir.mkdir()
    # BTC: covers 200 days, ETH: covers 100 days, SOL: no parquet.
    _write_mc_parquet(str(mcdir / "BTC-USD.parquet"), starts=[1_700_000_000 + i * 86400 for i in range(0, 200)])
    _write_mc_parquet(str(mcdir / "ETH-USD.parquet"), starts=[1_700_000_000 + i * 86400 for i in range(0, 100)])

    needs = bum.pids_needing_backfill(
        pids=["BTC-USD", "ETH-USD", "SOL-USD"],
        marketcap_dir=str(mcdir),
        min_days=180,
    )
    assert needs == ["ETH-USD", "SOL-USD"]


def test_pids_needing_backfill_empty_dir(tmp_path):
    """All pids need backfill when the marketcap dir is empty."""
    mcdir = tmp_path / "marketcap"
    mcdir.mkdir()
    needs = bum.pids_needing_backfill(pids=["A", "B"], marketcap_dir=str(mcdir), min_days=180)
    assert sorted(needs) == ["A", "B"]


def test_universe_pids_from_curation_json(tmp_path):
    """universe_pids_from_curation flattens the cohort dict into a single deduplicated list."""
    p = tmp_path / "universe.json"
    p.write_text(json.dumps({
        "large":          ["BTC-USD", "ETH-USD"],
        "mid":            ["LINK-USD"],
        "high_fdv_ratio": ["AAA-USD"],
        "low_turnover":   ["BBB-USD"],
    }))
    pids = bum.universe_pids_from_curation(str(p))
    assert sorted(pids) == ["AAA-USD", "BBB-USD", "BTC-USD", "ETH-USD", "LINK-USD"]
```

- [ ] **Step 2: Run the tests to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/test_build_universe_marketcap.py -v
```

Expected: 3 failures with `ModuleNotFoundError: No module named 'tools.strategy_discovery.build_universe_marketcap'`.

- [ ] **Step 3: Implement the orchestrator**

Create `backend/tools/strategy_discovery/build_universe_marketcap.py`:

```python
"""Bulk marketcap historical backfill for the curated 50-pid universe.

Reads the universe JSON from Task 5, identifies pids whose existing marketcap
parquets DON'T cover at least `--min-days` of history, and dispatches
`tools.build_marketcap_parquet.fetch_marketcap_history -> save` for each
missing pid. Honors the data-first directive — never re-fetches a pid that
already has sufficient coverage.

Run:
    cd backend && python -m tools.strategy_discovery.build_universe_marketcap \\
        --universe-json ../docs/superpowers/specs/2026-05-23-universe-50.json \\
        --start 2025-05-23 --end 2026-05-23
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt
import json
import os
import sys
from typing import List

import pyarrow.parquet as pq

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services.coinpaprika_marketcap import fetch_marketcap_history  # noqa: E402
from tools.build_marketcap_parquet import (  # noqa: E402
    _save_marketcap_history,
    rows_from_history,
)

_MARKETCAP_DIR = os.path.join(BACKEND, "data", "marketcap")


def universe_pids_from_curation(curation_json_path: str) -> List[str]:
    """Flatten {cohort: [pids]} into a single deduplicated sorted list."""
    with open(curation_json_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)


def pids_needing_backfill(*, pids: List[str], marketcap_dir: str, min_days: int) -> List[str]:
    """Return the subset of pids whose existing marketcap parquet covers < min_days."""
    out: List[str] = []
    for pid in pids:
        path = os.path.join(marketcap_dir, f"{pid}.parquet")
        if not os.path.exists(path):
            out.append(pid)
            continue
        table = pq.read_table(path, columns=["start"])
        starts = table.column("start").to_pylist()
        if not starts:
            out.append(pid)
            continue
        days = (int(max(starts)) - int(min(starts))) // 86400
        if days < min_days:
            out.append(pid)
    return out


def _date_to_ms(s: str) -> int:
    d = _dt.datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_dt.timezone.utc)
    return int(d.timestamp() * 1000)


async def backfill_one(pid: str, *, start_ms: int, end_ms: int, marketcap_dir: str) -> int:
    """Fetch + save one pid. Returns rows written (0 on failure)."""
    history = await fetch_marketcap_history(pid, start_ms, end_ms)
    if not history:
        return 0
    rows = rows_from_history(history)
    path = os.path.join(marketcap_dir, f"{pid}.parquet")
    _save_marketcap_history(path, rows)
    return len(rows)


async def backfill_universe(
    *,
    pids: List[str],
    start_ms: int,
    end_ms: int,
    marketcap_dir: str = _MARKETCAP_DIR,
    sleep_secs: float = 0.5,
) -> dict:
    results: dict = {}
    for i, pid in enumerate(pids):
        n = await backfill_one(pid, start_ms=start_ms, end_ms=end_ms, marketcap_dir=marketcap_dir)
        results[pid] = n
        print(f"  [{i+1}/{len(pids)}] {pid}: {n} rows", flush=True)
        if i + 1 < len(pids):
            await asyncio.sleep(sleep_secs)
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe-json",
        default=os.path.join(BACKEND, "..", "docs", "superpowers", "specs",
                             "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--marketcap-dir", default=_MARKETCAP_DIR)
    parser.add_argument("--start", required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--end",   required=True, help="YYYY-MM-DD UTC")
    parser.add_argument("--min-days", type=int, default=180)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--force", action="store_true",
                        help="re-fetch even for pids with sufficient coverage")
    args = parser.parse_args()

    pids = universe_pids_from_curation(args.universe_json)
    print(f"  universe: {len(pids)} pids", flush=True)

    if args.force:
        needs = pids
    else:
        needs = pids_needing_backfill(
            pids=pids, marketcap_dir=args.marketcap_dir, min_days=args.min_days
        )
    print(f"  needing backfill: {len(needs)} pids", flush=True)

    if not needs:
        print("  nothing to do — all pids already covered", flush=True)
        return 0

    start_ms = _date_to_ms(args.start)
    end_ms   = _date_to_ms(args.end)
    print(f"  range: {args.start} -> {args.end}", flush=True)

    results = asyncio.run(backfill_universe(
        pids=needs,
        start_ms=start_ms,
        end_ms=end_ms,
        marketcap_dir=args.marketcap_dir,
        sleep_secs=args.sleep,
    ))

    succeeded = [p for p, n in results.items() if n > 0]
    failed    = [p for p, n in results.items() if n == 0]
    print(f"\n  succeeded: {len(succeeded)}", flush=True)
    print(f"  failed:    {len(failed)}  -> {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/tools/test_build_universe_marketcap.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Update CHANGELOG.md**

Append to "Unreleased":

```markdown
- **2026-05-23: Strategy-discovery rebuild Phase 1 Task 6** — bulk marketcap backfill orchestrator (`tools/strategy_discovery/build_universe_marketcap.py`). Reads Task 5's universe JSON, identifies pids needing backfill (< min-days coverage), dispatches `fetch_marketcap_history` for each, persists via `build_marketcap_parquet._save_marketcap_history`. Skips already-covered pids (data-first directive).
```

- [ ] **Step 6: Commit**

```powershell
git add backend/tools/strategy_discovery/build_universe_marketcap.py `
        backend/tests/tools/test_build_universe_marketcap.py `
        CHANGELOG.md
git commit -m "feat: strategy-discovery Phase 1 Task 6 — bulk marketcap backfill for universe"
git push origin HEAD
```

---

## Final integration run (after Task 6 commits)

This is not a coded task — it's the operator-driven run of the Phase 1 pipeline end-to-end. The subagent-driven-development workflow should pause for operator review here.

- [ ] **Step 1: Run the inventory CLI**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m tools.strategy_discovery.inventory
```

Open `docs/superpowers/specs/2026-05-23-data-inventory-report.md`. Confirm the pid counts look reasonable.

- [ ] **Step 2: Fetch supply snapshots for the candidate pool**

Pick a candidate-pool list of ~100 pids (from the Task 4 mapping + inventory's longest-history pids). Run:

```powershell
.\.venv\Scripts\python.exe -m tools.strategy_discovery.build_supply_snapshot `
    --pids BTC-USD,ETH-USD,SOL-USD,...
```

This will take ~50 seconds (0.5s/pid).

- [ ] **Step 3: Refresh recent marketcap for the candidate pool**

The curation script needs a current MC + volume per pid. Run `tools/build_marketcap_parquet.py` for any pid in the candidate pool that doesn't yet have a marketcap parquet (use a short window — last 30 days is enough for curation):

```powershell
.\.venv\Scripts\python.exe -m tools.build_marketcap_parquet `
    --source coinpaprika `
    --pids PID1,PID2,... `
    --start 2026-04-23 --end 2026-05-23
```

- [ ] **Step 4: Run the universe curation**

```powershell
.\.venv\Scripts\python.exe -m tools.strategy_discovery.curate_universe
```

Open `docs/superpowers/specs/2026-05-23-universe-50.md` and review the picks per cohort. If the picks look skewed (e.g. low-turnover cohort is dominated by stablecoins), iterate on the criteria via CLI flags (`--n-*` arguments). When happy, commit the curation outputs:

```powershell
cd C:\Users\gl450\polymarket_app
git add docs/superpowers/specs/2026-05-23-universe-50.md `
        docs/superpowers/specs/2026-05-23-universe-50.json `
        docs/superpowers/specs/2026-05-23-data-inventory-report.md `
        docs/superpowers/specs/2026-05-23-data-inventory.json `
        backend/data/supply/snapshot.parquet
git commit -m "data: strategy-discovery Phase 1 — initial inventory, supply snapshot, curated 50-pid universe"
git push origin HEAD
```

- [ ] **Step 5: Run the bulk historical backfill for the 50**

```powershell
cd C:\Users\gl450\polymarket_app\backend
.\.venv\Scripts\python.exe -m tools.strategy_discovery.build_universe_marketcap `
    --start 2025-05-23 --end 2026-05-23
```

This will run for ~25 seconds × N missing pids (where N is at most 47 if BTC/ETH/SOL are already covered). Watch the output for any pids that fail (likely candidates: unmapped CP ids — fall back to manual mapping in `_PRODUCT_TO_CP_ID`).

When complete, commit the newly written parquets:

```powershell
cd C:\Users\gl450\polymarket_app
git add backend/data/marketcap/
git commit -m "data: strategy-discovery Phase 1 — historical marketcap for the 50-pid universe"
git push origin HEAD
```

- [ ] **Step 6: Memory update**

Add a note to `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` referencing the new `tools/strategy_discovery/` package and the universe JSON. Format:

```markdown
- `tools/strategy_discovery/` — Phase 1 data foundation for the post-v3 rebuild (2026-05-23 brainstorm). Inventory + universe curation + bulk backfill scripts. Universe pid list at `docs/superpowers/specs/2026-05-23-universe-50.json`. Supply snapshot parquet at `backend/data/supply/snapshot.parquet`.
```

- [ ] **Step 7: Phase 1 review checkpoint**

Pause here and surface to the operator:

> "Phase 1 complete. Inventory shows N pids with ≥6 months history. Universe of 50 picked across 4 cohorts. Marketcap + supply data complete for all 50. Outputs: `docs/superpowers/specs/2026-05-23-{data-inventory-report.md,universe-50.md}`. Ready to draft Phase 2 (feature compute + labeling) — proceed?"

Wait for operator approval before invoking `superpowers:writing-plans` for Phase 2.

---

## What this plan does NOT do (deferred to subsequent plans)

- **Phase 2 — Feature + Label Layer:** trend feature builder (5 features from 1h OHLCV), tokenomic feature builder (6 features from marketcap + supply parquets), multi-horizon realized-PnL labeler at `{1h, 4h, 24h, 72h, 168h, 336h}`, feature × label table assembler.
- **Phase 3 — Mining + Validation:** concurrency-capped fixed-fraction simulator, custom profit-based split criterion, decision tree mining, Q0 gate evaluator per leaf, purged WF, bootstrap CI on per-profile metrics.
- **Phase 4 — Output:** profile-ranked output doc renderer, top-level CLI orchestrator, mining run + verdict commit.

Each subsequent phase becomes its own plan, drafted after the previous one ships.

---

## Self-review checklist (run before handing the plan off)

This is a sanity check the plan-writer ran after drafting; the implementer can ignore it.

**1. Spec coverage:**
- ✅ Data-first directive — Task 1
- ✅ Q5 universe criteria — Task 5
- ✅ Q5 ≥ 6 months history filter — Task 5 (`min_history_days=180`)
- ✅ Q5 supply data requirement — Task 2-3 (fetch + persist)
- ✅ Tokenomic backfill (extend from 3 to 50) — Task 6
- ✅ FDV computation precondition (need total_supply for daily price × total_supply) — Task 2 fetches total_supply
- ✅ Inventory targets enumerated in spec (history, marketcap, 1m, cache) — Task 1 covers history, marketcap, 1m. Cache (cnn_dataset_cache.pt) is intentionally NOT scanned by Task 1 — it's a .pt file requiring torch, and the spec notes its purpose is "survivorship-aware top-20 pid list (input to curation, not the universe itself)". Curation does not need it.
- ✅ Q3 feature 1-2 (market_cap, FDV) sourced from CoinPaprika daily + supply snapshot — Tasks 2, 6
- ✅ Q3 feature 3-4 (FDV/MC ratio, circ/total ratio) derivable from above — no separate task needed
- ✅ Q3 feature 5-6 (24h_volume, vol/MC) sourced from existing CoinPaprika daily — already covered by `fetch_marketcap_history`'s `volume_24h` field (Task 6 uses it)

**2. Placeholder scan:**
- No "TBD", "TODO", or "implement later" in code blocks
- No "Add appropriate error handling" hand-waves — every function has explicit None/empty-return paths
- No "Write tests for the above" without code — every test is shown

**3. Type consistency:**
- `scan_history_parquets` / `scan_marketcap_parquets` both return `Dict[str, Dict[str, int]]` with same keys `(first_ts, last_ts, n_rows)` — consistent
- `fetch_supply_snapshot` returns `Optional[Tuple[float, float, Optional[float]]]` — consistent across Tasks 2 and 3
- `CandidateMap = Dict[str, Dict[str, float]]` with keys `(market_cap, volume_24h, circulating, total)` — consistent in all curation helpers
- Parquet schema for supply: `(pid, circulating, total, max_supply, ingest_ts, schema_version)` — consistent in `save_snapshot` / `load_snapshot` / `_SCHEMA`
