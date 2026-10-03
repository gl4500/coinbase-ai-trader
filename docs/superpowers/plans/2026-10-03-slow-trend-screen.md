# BTC/ETH Slow-Trend Falsification Screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ONE preregistered historical screen of a frozen BTC/ETH SMA100 trend rule against cash,
buy-and-hold and a defined 52-week DCA at the account's verified fees, and emit exactly one verdict:
`KILL`, `INCONCLUSIVE` or `PASS_TO_FORWARD`.

**Architecture:** A self-contained research package, `backend/tools/slow_trend/`.
- **Pure layers** (`rule`, `sim`, `metrics`, `gates`, and `screen.evaluate`) take DataFrames and
  dicts, and never touch the network, the DB or the clock.
- **One I/O layer** (`daily_bars.fetch_daily`) keeps raw pages as evidence; `daily_bars.audit`
  validates those raw rows BEFORE anything is normalised.
- **One CLI** (`screen`) enforces the freeze. It refuses to overwrite a snapshot, requires a
  committed `snapshot.lock` and keeps an append-only run ledger.

Every tunable value lives in `prereg.py` and is pinned by a test.

**Tech Stack:** Python 3.11, pandas 3.0, numpy 2.4, pytest, existing `clients/coinbase_client`
(`_get`, `get_product`). No new dependencies. No DB access. No backend process involvement.

**Spec:** the **Preregistration** section below is the spec. It is the converged result of the
Claude–Codex debate of 2026-10-03, session-link messages:
- debate: `a3f4f64e`, `7eee3897`, `28e4c12a`, `2ed9d142`;
- plan review: `36830843` (B1–B6 blocking, N1–N5 non-blocking), all addressed in revision 2.

Background: `docs/specs/2026-09-27-strategy-evidence-and-decision.md` (branch
`docs/strategy-evidence-decision`) and the 58.90 ABSTAIN verdict.

## Preregistration (frozen BEFORE any rule is run on any data)

**Purpose.** This is an economic abandonment screen for ONE candidate, not a test of trend
following in general. `PASS_TO_FORWARD` earns only a forward paper run. No verdict authorises
funding, and none establishes superiority. Results at USD 1,000 do not automatically transfer to
a smaller bankroll.

**Verified inputs**
- Account fee tier: read-only `GET transaction_summary` on 2026-10-03 returned `pricing_tier=Intro`,
  `maker_fee_rate=0.005`, `taker_fee_rate=0.009`, 30-day volume 0. Applied as a constant
  **planning assumption**, not a reconstruction of historical fees.
- Product constraints (`base_increment`, `base_min_size`, `quote_min_size`) are read live at
  fetch time and recorded in the manifest. They are a planning assumption applied to historical
  prices, not historical rules. Missing or invalid constraints abort the fetch; there is no
  fallback value.

**Universe and rule**
- Universe: `BTC-USD`, `ETH-USD`, chosen a priori for liquidity and scope, NOT from ledger results.
- Signal at each completed UTC daily close `t`: `long` iff `close[t] > mean(close[t-99..t])`
  (arithmetic, 100 closes, including `t`). Equality is `flat`. A window containing any missing
  day yields NO decision; the current state is held.
- SMA length 100 is a discretionary research choice, not evidence of an optimal horizon. No other
  length is run, including no 20-week variant.
- Position: all-in or all-cash per sleeve. No resizing, shorts, stops, leverage or borrowing.
- Capital: USD 1,000 split into two independent sleeves of USD 500 (BTC, ETH). No transfers or
  rebalancing between sleeves. Cash earns 0.

**Period reset.** Decisions are computed with full historical warmup and then sliced to the
period. The held state is forward-filled ONLY within the period, starting FLAT. A long decision
made before the period never carries in.

**Execution semantics (state-targeting, not an order queue)**
- The state decided at close `t` is the target at the OPEN of `t+1` (primary) or `t+2`
  (sensitivity D1). Next-open execution is an **idealised proxy**: the Coinbase candle open is
  the first trade, not an executable quote. D1 is never selected because it performs better, and
  it is not a conservative bound.
- On every day with a valid open, the sleeve moves to that day's target. A target missed because
  the open is absent is superseded by later targets; it is not queued.
- A buy rejected by size limits (`units < base_min_size` or `notional < quote_min_size`) is
  counted as `skipped`, then retried at every later open while the target stays long.
- A sell whose proceeds would fall below `quote_min_size` is NOT executed: the holding is
  retained, `skipped_exits` is counted, and the sell is retried at later opens while the target
  stays flat. At the endpoint, a holding whose liquidation proceeds fall below `quote_min_size`
  is excluded from the terminal cash value (never converted into invented cash). It is recorded
  separately as `residual_units` and `residual_marked_value`, and flagged `unliquidatable`. That
  cash value is a LOWER BOUND, not an economic value, so it is NOT treated as conservative
  evidence. Any trend or buy-and-hold sleeve flagged in a gating period/scenario makes that
  comparison **affected**, and the verdict order below handles it explicitly. DCA flags are
  diagnostic only.
- Before the first observed close, a position is marked at its execution price, for marking only.

**Costs** (applied to actual entry/exit notionals; fee cash reserved; self-financing compounding)
- P0 (primary): taker 0.90% on entry AND exit, 0 bps price stress.
- S10 / S25: P0 plus an assumed adverse price stress of 10 / 25 bps per leg (buy at
  `open*(1+s)`, sell at `open*(1-s)`). These are assumed stresses, not measured spreads.
- SM: maker 0.50% entry, taker 0.90% exit. **Optimistic sensitivity only**; it never rescues
  a P0 failure.
- Every trading comparator pays the same scenario's fees and stress on every leg, including the
  terminal liquidation. Execution fees and the terminal liquidation fee are reported separately.

**Comparators** (same USD 1,000, same period start, same sleeve split)
- Cash: USD 1,000, return 0.
- Buy-and-hold: each sleeve targets long from the first day of the period, then holds.
- DCA-52: on the first 52 Mondays (UTC) on or after the period start, each sleeve spends a
  **fee-inclusive** cash tranche of `500/52` USD at that day's open (or at the next valid open),
  then holds. Unspent cash stays in equity. Reported, but **not** a gate.

**Data and periods**
- Source, fixed a priori: Coinbase Advanced daily candles (`granularity=ONE_DAY`, 300-day pages,
  within the documented 350 maximum). The request bound 2015-01-01 is not evidence of coverage;
  the earliest returned day is recorded.
- Raw rows are stored with their page number. The audit runs on RAW rows:
  - finite, positive OHLC with `high >= max(open, close)`, `low <= min(open, close)` and
    `volume >= 0`;
  - midnight-UTC alignment;
  - duplicates: an identical pagination copy is allowed and collapsed, while a **conflicting**
    duplicate makes the data inadequate.
- `common_first` is the later of the two products' first aligned days. Rows outside
  `[common_first, 2026-10-02]` are excluded BEFORE normalisation. They are unused and never
  audited, so they can neither pass nor fail the screen.
- The development audit window is `[common_first, 2025-04-13]` (warmup INCLUDED); the
  validation-block window is `[2025-04-14, 2026-10-02]`.
- The data is inadequate if any of the following hold:
  - any invalid row, misaligned row or conflicting duplicate in a window;
  - a product with no aligned rows at all, or `common_first` later than 2025-04-13 (no
    development coverage);
  - more than 3 missing days per product per window (missing days are never filled);
  - no valid close for both products on 2025-04-13 or on 2026-10-02 (the endpoint is never moved);
  - no first common valid SMA day on or before 2025-04-13.
- Development period: from the **first day on which BOTH products have a valid SMA decision**
  (derived from data coverage, never from P&L) through 2025-04-13.
- Validation block, named "previously observed retrospective validation block": 2025-04-14
  through 2026-10-02, the last complete UTC day. It is NOT a clean holdout. We chose this
  candidate after studying this period for other policies. Its warmup uses development closes.
  It is reported separately and never averaged with development.
- In-repo hourly `data/history/*.parquet` is used only for an informational overlap check. Its
  sha256 is recorded. A failure there is recorded as a diagnostic and never fails the run.
- Disclosed per product per period: stale-mark days and the longest stale run, plus
  suppressed-decision days.

**Metrics** (per period, per scenario, per strategy)
- **Portfolio:**
  - value and costs: terminal liquidation value, net return, execution fees, terminal fee,
    `total_fees`, `slippage_cost` (the assumed adverse adjustment, terminal included), and
    unliquidatable flags;
  - drawdown: max drawdown of marked equity **seeded with the pre-trade USD 1,000**;
  - activity: entries, exits, completed round trips (terminal liquidation is not a round trip),
    skipped buys and exits, `executed_turnover` (executed notional / initial; excludes the
    hypothetical terminal sale), mean sleeve exposure.
- **Per sleeve:** terminal value, net return, max drawdown, exposure, round trips, stale-mark days.
- **Weekly returns** use COMPLETE Monday–Sunday weeks only (Sunday close to Sunday close). The
  partial head and tail are reported separately as boundary returns.
- **Bootstrap:** the excess series is formed BEFORE resampling and requires identical ordered
  week indexes and finite values. Moving blocks of 8 weeks, 10,000 resamples, seed `20261003`;
  blocks of 4 and 13 are reported as sensitivity. When there are fewer weeks than the block
  length, the CI is reported as `None`. The CIs are exploratory and are NOT gates.

**Gates** (portfolio = sum of both sleeves)
- G1 (cash): terminal liquidation value > USD 1,000.
- G2 (risk vs buy-and-hold): trend net return ≥ buy-and-hold net return, OR trend max drawdown
  ≤ (2/3) × buy-and-hold max drawdown. The 2/3 ratio is an **explicitly arbitrary provisional
  utility gate**.

**Verdict, in this order, frozen**
1. Data inadequate → `INCONCLUSIVE` (reason `data`).
2. P0 fails G1 or G2 in an UNAFFECTED period (development OR validation block) → `KILL`
   (`primary_failed`). A block failure is never averaged away. Explicitly prioritised: an
   unaffected demonstrable failure decides even if the other period is affected.
3. P0 is affected in either period → `INCONCLUSIVE` (`terminal_size`). An affected comparison
   never produces `PASS_TO_FORWARD`, and an affected failure never produces `KILL`.
4. Any of S10, S25 or D1 is affected in either period → `INCONCLUSIVE` (`terminal_size`). If
   none is affected but any fails G1 or G2 → `INCONCLUSIVE` (`fragile`). SM is never consulted.
5. Validation block P0 trend has fewer than 4 completed round trips, pooled across sleeves →
   `INCONCLUSIVE` (`insufficient_transitions`). 4 is an **explicitly arbitrary administrative
   threshold**; one asset can meet it, and it gives no independent-regime assurance. Per-asset
   counts are reported.
6. Otherwise `PASS_TO_FORWARD`.

**Freeze mechanics**
1. `screen fetch` refuses to run if a snapshot manifest already exists. A new snapshot is a new
   preregistration, made by deliberately moving the old directory aside; old outcomes are never
   deleted.
2. `screen lock` writes the manifest's sha256 to `backend/tools/slow_trend/snapshot.lock`, which
   is committed before `run`.
3. `screen run` refuses when there are uncommitted changes under `backend/tools/slow_trend` or in
   `backend/clients/coinbase_client.py`, or when the lock does not match the manifest.
4. `experiment_id = sha256(prereg digest, manifest digest)` is stable across commits. Each
   attempt records its own HEAD, its parent attempt in the same experiment, and its mode.
5. An append-only `runs.jsonl` records `started`/`failed`/`completed`. Permissions are keyed by
   `experiment_id`, never by HEAD:
   - a completed experiment cannot run again except with `--replay`;
   - a replay is labelled `replay` and its report is kept alongside the first, never replacing
     it;
   - after a `failed` or unfinished `started` attempt, the next attempt is a `retry`, even on a
     new commit;
   - a DIFFERENT `experiment_id` while another experiment has already completed requires an
     explicit `--new-preregistration`, and is labelled `new_preregistration`.

## Global Constraints

- Read-only with respect to the app: no writes to `coinbase.db`, no backend restart, no orders.
  Network use is limited to public-data candle/product reads.
- **8001 is live paper trading.** Per operator rule, do NOT run `pytest`, and do NOT commit `.py`
  files (the pre-commit hook runs the full suite), without asking the operator first.
  Docs-only and lock-file commits skip the suite.
- Outputs go under `backend/data/research/slow_trend/`, which is gitignored and never committed.
  `snapshot.lock` is the only committed run artifact.
- Ruff: use the pinned 0.9.0 (`$CLAUDE_JOB_DIR/tmp/ruff090/Scripts/ruff.exe`) for `check` and
  `format --check`.
- Python: `.venv/Scripts/python.exe`, with tests run from `backend/`.
- Any change to `prereg.py` after a run is a new preregistration (a new commit), never a
  revision of a verdict.

## Review Focus

1. **Boundary carry-in:** a pre-period long decision followed by invalid in-period windows must
   NOT buy → `test_pre_period_long_does_not_carry_in` (Task 6).
2. **Lookahead through the execution shift:** a decision at close `t` must never trade at the
   open of `t` → `test_trade_never_executes_on_decision_day` (Task 3).
3. **Silent bad data:** a conflicting duplicate or a timestamped NaN/zero price must make the data
   inadequate, never a "present" day → `test_audit_flags_conflicting_duplicate`,
   `test_audit_flags_invalid_prices` (Task 1).
4. **Invented terminal price:** a missing endpoint close must yield `INCONCLUSIVE(data)`, not a
   stale liquidation → `test_missing_terminal_close_is_inadequate` (Task 6) and
   `test_terminal_close_missing_raises` (Task 3).
5. **Drawdown asymmetry:** an all-in buy at constant price must show the fee as drawdown →
   `test_constant_price_buy_hold_drawdown_is_entry_fee` (Task 4).

---

### Task 1: Preregistration constants + raw daily data layer

**Files:**
- Create: `backend/tools/slow_trend/__init__.py` (empty)
- Create: `backend/tools/slow_trend/prereg.py`
- Create: `backend/tools/slow_trend/daily_bars.py`
- Create: `backend/tests/tools/slow_trend/__init__.py` (empty)
- Test: `backend/tests/tools/slow_trend/test_prereg.py`
- Test: `backend/tests/tools/slow_trend/test_daily_bars.py`
- Modify: `.gitignore` (append `backend/data/research/`)

**Interfaces:**
- Produces:
  - `prereg.*` constants (below);
  - `daily_bars.fetch_daily(pid, start_ts, end_ts, getter, page_days=300) -> pd.DataFrame`
    (RAW rows, columns `start, open, high, low, close, volume, page`; no dedupe);
  - `daily_bars.audit(raw, first: str, last: str, max_missing: int) -> dict`
    (keys `missing_days`, `misaligned`, `invalid_rows`, `conflicting_duplicates`,
    `identical_copies`, `adequate`);
  - `daily_bars.first_day(raw) -> str | None` (first aligned day; `None` if there is none);
  - `daily_bars.window(raw, first, last) -> pd.DataFrame` (rows whose start is in the window);
  - `daily_bars.normalise(raw) -> pd.DataFrame` (collapses identical copies; raises
    `ValueError` on a conflicting duplicate);
  - `daily_bars.to_calendar(norm, first, last) -> pd.DataFrame` (daily index, `open, close`,
    NaN for missing);
  - `daily_bars.product_constraints(product: dict | None) -> dict`;
  - `daily_bars.hourly_overlap(hourly, daily_cal) -> dict`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_prereg.py
"""Pins every preregistered value. Changing one must be a visible, reviewed diff."""
from tools.slow_trend import prereg as P


def test_frozen_values():
    assert P.PRODUCTS == ("BTC-USD", "ETH-USD")
    assert P.SMA_LEN == 100
    assert P.INITIAL_USD == 1000.0 and P.SLEEVE_USD == 500.0
    assert P.TAKER_FEE == 0.009 and P.MAKER_FEE == 0.005
    assert P.DEV_END == "2025-04-13"
    assert P.BLOCK_START == "2025-04-14" and P.BLOCK_END == "2026-10-02"
    assert P.MAX_MISSING_DAYS == 3
    assert P.DCA_TRANCHES == 52
    assert P.G2_DRAWDOWN_RATIO == 2 / 3
    assert P.MIN_BLOCK_ROUND_TRIPS == 4
    assert P.BOOT_BLOCK_WEEKS == 8 and P.BOOT_SENS_WEEKS == (4, 13)
    assert P.BOOT_RESAMPLES == 10_000 and P.BOOT_SEED == 20261003
    assert P.FREEZE_PATHS == ("backend/tools/slow_trend", "backend/clients/coinbase_client.py")


def test_scenarios():
    s = {x.name: x for x in P.SCENARIOS}
    assert set(s) == {"P0", "S10", "S25", "D1", "SM"}
    assert (s["P0"].entry_fee, s["P0"].exit_fee, s["P0"].slip, s["P0"].delay) == (0.009, 0.009, 0.0, 0)
    assert s["S10"].slip == 0.0010 and s["S25"].slip == 0.0025
    assert s["D1"].delay == 1 and s["D1"].slip == 0.0
    assert s["SM"].entry_fee == 0.005 and s["SM"].exit_fee == 0.009
    assert P.GATING_SENSITIVITIES == ("S10", "S25", "D1")
```

```python
# backend/tests/tools/slow_trend/test_daily_bars.py
import asyncio

import numpy as np
import pandas as pd
import pytest

from tools.slow_trend import daily_bars as D

DAY = 86400
T0 = 1_700_006_400  # 2023-11-15 00:00 UTC


def _candles(starts):
    return [{"start": str(s), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}
            for s in starts]


def _raw(starts, **cols):
    n = len(starts)
    base = {"start": starts, "open": [1.0] * n, "high": [1.0] * n, "low": [1.0] * n,
            "close": [1.0] * n, "volume": [1.0] * n, "page": [0] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_fetch_keeps_raw_rows_with_page_numbers():
    async def getter(path, params):
        s, e = int(params["start"]), int(params["end"])
        return {"candles": _candles(range(s - s % DAY, e, DAY))}

    raw = asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 700 * DAY, getter, page_days=300))
    assert len(raw) == 700 and set(raw["page"]) == {0, 1, 2}


def test_fetch_rejects_malformed_response():
    async def getter(path, params):
        return {"candles": None}

    with pytest.raises(ValueError, match="malformed"):
        asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 10 * DAY, getter))


def test_fetch_rejects_candle_missing_fields():
    async def getter(path, params):
        return {"candles": [{"start": str(T0), "open": "1"}]}

    with pytest.raises(ValueError, match="missing"):
        asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 10 * DAY, getter))


def test_audit_counts_missing_and_misaligned():
    raw = _raw([T0, T0 + 2 * DAY + 3600])
    a = D.audit(raw, "2023-11-15", "2023-11-17", 3)
    assert a["misaligned"] == 1 and a["missing_days"] == ["2023-11-16", "2023-11-17"]
    assert a["adequate"] is False


def test_audit_allows_up_to_three_missing():
    starts = [T0 + i * DAY for i in range(10) if i not in (2, 5, 7)]
    a = D.audit(_raw(starts), "2023-11-15", "2023-11-24", 3)
    assert len(a["missing_days"]) == 3 and a["adequate"] is True


def test_audit_flags_conflicting_duplicate():
    raw = _raw([T0, T0], close=[1.0, 2.0], high=[1.0, 2.0])
    a = D.audit(raw, "2023-11-15", "2023-11-15", 3)
    assert a["conflicting_duplicates"] == 1 and a["adequate"] is False


def test_audit_allows_identical_page_boundary_copy():
    raw = _raw([T0, T0], page=[0, 1])
    a = D.audit(raw, "2023-11-15", "2023-11-15", 3)
    assert a["identical_copies"] == 1 and a["conflicting_duplicates"] == 0
    assert a["adequate"] is True


def test_audit_flags_invalid_prices():
    raw = _raw([T0, T0 + DAY, T0 + 2 * DAY, T0 + 3 * DAY],
               close=[np.nan, 1.0, 1.0, 1.0], open=[1.0, 0.0, 1.0, 1.0],
               high=[1.0, 1.0, 0.5, np.inf])
    a = D.audit(raw, "2023-11-15", "2023-11-18", 3)
    assert a["invalid_rows"] == 4 and a["adequate"] is False


def test_normalise_collapses_identical_and_rejects_conflict():
    assert len(D.normalise(_raw([T0, T0], page=[0, 1]))) == 1
    with pytest.raises(ValueError, match="conflicting"):
        D.normalise(_raw([T0, T0], close=[1.0, 2.0], high=[1.0, 2.0]))


def test_to_calendar_marks_missing_as_nan():
    cal = D.to_calendar(D.normalise(_raw([T0, T0 + 2 * DAY])), "2023-11-15", "2023-11-17")
    assert list(cal.index.strftime("%Y-%m-%d")) == ["2023-11-15", "2023-11-16", "2023-11-17"]
    assert cal.loc["2023-11-16"].isna().all()


def test_first_day_ignores_misaligned_rows():
    assert D.first_day(_raw([T0 - 3600, T0 + DAY])) == "2023-11-16"


def test_first_day_none_without_aligned_rows():
    assert D.first_day(_raw([T0 + 3600])) is None
    assert D.first_day(_raw([])) is None


def test_product_constraints_require_every_field():
    ok = {"base_increment": "0.00000001", "base_min_size": "0.00000001", "quote_min_size": "1"}
    assert D.product_constraints(ok) == {"base_increment": 1e-8, "base_min": 1e-8,
                                         "quote_min": 1.0}
    with pytest.raises(ValueError, match="quote_min_size"):
        D.product_constraints({k: v for k, v in ok.items() if k != "quote_min_size"})
    with pytest.raises(ValueError, match="base_increment"):
        D.product_constraints(dict(ok, base_increment="0"))
    with pytest.raises(ValueError, match="product"):
        D.product_constraints(None)


def test_hourly_overlap_uses_only_complete_days():
    hours = [T0 + h * 3600 for h in range(24 + 5)]
    hourly = pd.DataFrame({"start": hours, "close": [float(h) for h in range(29)]})
    cal = pd.DataFrame({"open": [0.0, 0.0], "close": [23.0, 99.0]},
                       index=pd.to_datetime(["2023-11-15", "2023-11-16"]))
    o = D.hourly_overlap(hourly, cal)
    assert o["days_compared"] == 1 and o["max_abs_rel_diff"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run to verify failure**

Run (from `backend/`, ONLY after the operator OKs pytest while 8001 is live):
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.slow_trend'`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/prereg.py
"""Frozen preregistration for the BTC/ETH SMA100 falsification screen (2026-10-03).

Every value here was fixed BEFORE any rule ran on any data. See
docs/superpowers/plans/2026-10-03-slow-trend-screen.md#preregistration. Changing a value is a
new preregistration (new commit), never a revision of an existing verdict.
"""

from dataclasses import dataclass

PRODUCTS = ("BTC-USD", "ETH-USD")
SMA_LEN = 100
INITIAL_USD = 1000.0
SLEEVE_USD = INITIAL_USD / len(PRODUCTS)

TAKER_FEE = 0.009  # verified Intro tier, transaction_summary 2026-10-03
MAKER_FEE = 0.005

FETCH_FROM = "2015-01-01"  # request bound only; actual first day is recorded
DEV_END = "2025-04-13"
BLOCK_START = "2025-04-14"
BLOCK_END = "2026-10-02"  # last complete UTC day at preregistration
MAX_MISSING_DAYS = 3

DCA_TRANCHES = 52
G2_DRAWDOWN_RATIO = 2 / 3  # arbitrary provisional utility gate, not statistical materiality
MIN_BLOCK_ROUND_TRIPS = 4  # arbitrary administrative threshold, not an evidence threshold

BOOT_BLOCK_WEEKS = 8
BOOT_SENS_WEEKS = (4, 13)
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20261003

FREEZE_PATHS = ("backend/tools/slow_trend", "backend/clients/coinbase_client.py")


@dataclass(frozen=True)
class Scenario:
    name: str
    entry_fee: float
    exit_fee: float
    slip: float  # adverse price adjustment per leg, fraction
    delay: int  # extra days between decision close and execution open


SCENARIOS = (
    Scenario("P0", TAKER_FEE, TAKER_FEE, 0.0, 0),
    Scenario("S10", TAKER_FEE, TAKER_FEE, 0.0010, 0),
    Scenario("S25", TAKER_FEE, TAKER_FEE, 0.0025, 0),
    Scenario("D1", TAKER_FEE, TAKER_FEE, 0.0, 1),
    Scenario("SM", MAKER_FEE, TAKER_FEE, 0.0, 0),  # optimistic; never consulted by gates
)
GATING_SENSITIVITIES = ("S10", "S25", "D1")
```

```python
# backend/tools/slow_trend/daily_bars.py
"""Coinbase daily candles. Raw rows are evidence: fetched with page numbers, audited BEFORE
normalisation, and never silently deduplicated or repaired."""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Dict, Optional

import numpy as np
import pandas as pd

DAY = 86400
FIELDS = ("start", "open", "high", "low", "close", "volume")
VALUES = list(FIELDS[1:])
Getter = Callable[[str, Dict[str, str]], Awaitable[Any]]


async def fetch_daily(pid: str, start_ts: int, end_ts: int, getter: Getter,
                      page_days: int = 300) -> pd.DataFrame:
    frames, end, page = [], end_ts, 0
    while end > start_ts:
        start = max(start_ts, end - page_days * DAY)
        data = await getter(f"/products/{pid}/candles",
                            {"start": str(start), "end": str(end), "granularity": "ONE_DAY"})
        if not isinstance(data, dict) or not isinstance(data.get("candles"), list):
            raise ValueError(f"{pid}: malformed candles response on page {page}")
        for c in data["candles"]:
            missing = [k for k in FIELDS if k not in c]
            if missing:
                raise ValueError(f"{pid}: candle missing {missing} on page {page}")
        df = pd.DataFrame(data["candles"], columns=list(FIELDS))
        df["page"] = page
        frames.append(df)
        end, page = start, page + 1
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.astype({"start": "int64", "page": "int64", **{k: float for k in VALUES}})
    return raw[(raw["start"] >= start_ts) & (raw["start"] < end_ts)].reset_index(drop=True)


def window(raw: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    lo = pd.Timestamp(first).timestamp()
    hi = pd.Timestamp(last).timestamp() + DAY
    return raw[(raw["start"] >= lo) & (raw["start"] < hi)]


def _valid_rows(r: pd.DataFrame) -> np.ndarray:
    v = r[VALUES].to_numpy(dtype=float)
    finite = np.isfinite(v).all(axis=1)
    o, h, lo, c, vol = (r[k].to_numpy(dtype=float) for k in VALUES)
    with np.errstate(invalid="ignore"):
        ok = ((o > 0) & (h > 0) & (lo > 0) & (c > 0) & (vol >= 0)
              & (h >= np.maximum(o, c)) & (lo <= np.minimum(o, c)))
    return finite & ok


def audit(raw: pd.DataFrame, first: str, last: str, max_missing: int) -> Dict[str, Any]:
    r = window(raw, first, last)
    aligned = (r["start"] % DAY == 0).to_numpy()
    valid = _valid_rows(r)
    content = r[list(FIELDS)]
    identical = int(content.duplicated().sum())
    distinct = content.drop_duplicates()
    conflicting = int(distinct["start"].duplicated().sum())
    present = set(pd.to_datetime(r["start"][aligned & valid], unit="s").dt.strftime("%Y-%m-%d"))
    days = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
    missing = [d for d in days if d not in present]
    invalid = int((~valid).sum())
    misaligned = int((~aligned).sum())
    adequate = (invalid == 0 and misaligned == 0 and conflicting == 0
                and len(missing) <= max_missing)
    return {"missing_days": missing, "misaligned": misaligned, "invalid_rows": invalid,
            "conflicting_duplicates": conflicting, "identical_copies": identical,
            "adequate": adequate}


def first_day(raw: pd.DataFrame) -> Optional[str]:
    aligned = raw["start"][raw["start"] % DAY == 0]
    if aligned.empty:
        return None
    return str(pd.to_datetime(aligned.min(), unit="s").date())


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    distinct = raw[list(FIELDS)].drop_duplicates()
    if distinct["start"].duplicated().any():
        raise ValueError("conflicting duplicate candles; audit must reject this data")
    return distinct.sort_values("start").reset_index(drop=True)


def to_calendar(norm: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    idx = pd.to_datetime(norm["start"], unit="s")
    out = pd.DataFrame({"open": norm["open"].to_numpy(), "close": norm["close"].to_numpy()},
                       index=idx)
    return out.reindex(pd.date_range(first, last, freq="D"))


def product_constraints(product: Optional[dict]) -> Dict[str, float]:
    if not product:
        raise ValueError("product metadata missing")
    out = {}
    for src, dst in (("base_increment", "base_increment"), ("base_min_size", "base_min"),
                     ("quote_min_size", "quote_min")):
        try:
            val = float(product[src])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"product constraint {src} missing or non-numeric") from None
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"product constraint {src} must be finite and > 0, got {val}")
        out[dst] = val
    return out


def hourly_overlap(hourly: pd.DataFrame, daily_cal: pd.DataFrame) -> Dict[str, Any]:
    """Informational only: complete-UTC-day hourly closes vs daily candle closes."""
    h = hourly.assign(day=pd.to_datetime(hourly["start"].astype("int64"), unit="s").dt.floor("D"))
    g = h.sort_values("start").groupby("day").agg(n=("close", "size"), close=("close", "last"))
    g = g[g["n"] == 24]
    joined = g.join(daily_cal["close"].rename("daily"), how="inner").dropna()
    rel = (joined["close"] / joined["daily"] - 1).abs()
    return {"days_compared": int(len(joined)),
            "max_abs_rel_diff": float(rel.max()) if len(rel) else None,
            "days_over_10bps": int((rel > 0.001).sum())}
```

Append to `.gitignore`:

```
# slow-trend research outputs (snapshots, ledger, reports; never committed)
backend/data/research/
```

- [ ] **Step 4: Run to verify pass.** Expected: 16 passed.

- [ ] **Step 5: Commit** (operator OK required while 8001 is live; the hook runs the full suite)

```bash
git rev-parse --abbrev-ref HEAD   # research/slow-trend-screen
git add .gitignore backend/tools/slow_trend backend/tests/tools/slow_trend
git commit -m "feat(research): slow-trend preregistration + raw daily candle audit layer" -- .gitignore backend/tools/slow_trend backend/tests/tools/slow_trend
git log -1 --stat && git push
```

### Task 2: The frozen rule

**Files:**
- Create: `backend/tools/slow_trend/rule.py`
- Test: `backend/tests/tools/slow_trend/test_rule.py`

**Interfaces:**
- Produces:
  - `rule.decisions(close: pd.Series, n: int) -> pd.Series` (1.0 long, 0.0 flat, NaN = no
    decision);
  - `rule.desired_state(dec: pd.Series) -> pd.Series` (bool; NaN holds the current state;
    starts FLAT at the first index of whatever slice it is given).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_rule.py
import numpy as np
import pandas as pd

from tools.slow_trend.rule import decisions, desired_state

IDX = pd.date_range("2024-01-01", periods=6, freq="D")


def test_long_above_flat_below():
    d = decisions(pd.Series([1, 2, 3, 2, 2, 5], index=IDX, dtype=float), 3)
    assert np.isnan(d.iloc[0]) and np.isnan(d.iloc[1])
    assert d.iloc[2] == 1.0 and d.iloc[3] == 0.0


def test_equality_is_flat():
    d = decisions(pd.Series([2.0] * 6, index=IDX), 3)
    assert (d.dropna() == 0.0).all()


def test_missing_day_holds_state():
    d = decisions(pd.Series([1, 2, 3, np.nan, 9, 10], index=IDX, dtype=float), 3)
    assert d.iloc[3:6].isna().all()
    s = desired_state(d)
    assert s.iloc[2] and s.iloc[3] and s.iloc[5]


def test_state_starts_flat_on_any_slice():
    d = pd.Series([1.0, np.nan, np.nan], index=IDX[:3])
    assert list(desired_state(d.iloc[1:])) == [False, False]
```

- [ ] **Step 2: Run to verify failure.**
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_rule.py -v`. Expected:
`ModuleNotFoundError ... tools.slow_trend.rule`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/rule.py
"""The frozen rule: long iff completed close > SMA(n) including that close; equality is flat."""

import numpy as np
import pandas as pd


def decisions(close: pd.Series, n: int) -> pd.Series:
    sma = close.rolling(n, min_periods=n).mean()
    out = pd.Series(np.where(close > sma, 1.0, 0.0), index=close.index)
    return out.where(sma.notna() & close.notna())


def desired_state(dec: pd.Series) -> pd.Series:
    """Hold the current state through NaN decisions; the slice starts FLAT.

    Callers slice `dec` to the period BEFORE calling this, so no pre-period state carries in.
    """
    return dec.ffill().fillna(0.0).astype(bool)
```

- [ ] **Step 4: Run to verify pass.** Expected: 4 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/rule.py backend/tests/tools/slow_trend/test_rule.py
git commit -m "feat(research): frozen SMA100 rule (equality flat, gaps hold state)" -- backend/tools/slow_trend/rule.py backend/tests/tools/slow_trend/test_rule.py
git log -1 --stat && git push
```

### Task 3: Sleeve simulator (trend, buy-and-hold, DCA)

**Files:**
- Create: `backend/tools/slow_trend/sim.py`
- Test: `backend/tests/tools/slow_trend/test_sim.py`

**Interfaces:**
- Consumes: the calendar frame (`open`, `close`, NaN for missing days).
- Produces:
  - `sim.Costs(entry_fee, exit_fee, slip)` and `sim.Product(base_increment, base_min, quote_min)`,
    both frozen dataclasses;
  - `sim.trend_target(state: pd.Series, delay: int) -> pd.Series`;
  - `sim.run_sleeve(bars, target, cash, costs, product) -> SleeveResult`;
  - `sim.run_dca(bars, cash, tranches, costs, product) -> SleeveResult`.
  - `SleeveResult` fields:
    - `equity` (marked at close), `initial`;
    - `terminal_value`, `exec_fees`, `terminal_fee`, `traded_notional` (executed only),
      `slippage_cost` (terminal included), `unliquidatable`, `residual_units`,
      `residual_marked_value`;
    - `entries`, `exits`, `round_trips`, `skipped`, `skipped_exits`;
    - `exposure`, `stale_mark_days`, `max_stale_run`.
  - Raises `ValueError("terminal close missing")` when the last bar has no close.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_sim.py
import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

IDX = pd.date_range("2024-01-01", periods=5, freq="D")
FREE = Costs(0.0, 0.0, 0.0)
FINE = Product(base_increment=1e-8, base_min=1e-8, quote_min=1.0)


def _bars(opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)], dtype=float)


def _t(*vals):
    return pd.Series(list(vals), index=IDX[: len(vals)])


def test_trade_never_executes_on_decision_day():
    state = _t(True, True, True, True, True)
    assert list(trend_target(state, 0)[:2]) == [False, True]
    assert list(trend_target(state, 1)[:3]) == [False, False, True]


def test_round_trip_cost_identity():
    r = run_sleeve(_bars([100] * 4, [100] * 4), _t(True, True, False, False), 500.0,
                   Costs(0.009, 0.009, 0.0), FINE)
    assert r.terminal_value == pytest.approx(500.0 * (1 - 0.009) / (1 + 0.009), rel=1e-6)
    assert (r.entries, r.exits, r.round_trips, r.terminal_fee) == (1, 1, 1, 0.0)


def test_slip_is_adverse_on_both_legs():
    r = run_sleeve(_bars([100] * 3, [100] * 3), _t(True, False, False), 500.0,
                   Costs(0.0, 0.0, 0.0025), FINE)
    assert r.terminal_value == pytest.approx(500.0 * 0.9975 / 1.0025, rel=1e-6)
    units = 500.0 / 100.25
    assert r.slippage_cost == pytest.approx(2 * units * 100 * 0.0025, rel=1e-6)
    assert r.terminal_value == pytest.approx(500.0 - r.slippage_cost, rel=1e-6)


def test_sell_below_quote_min_is_retained_and_endpoint_unliquidatable():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(_bars([2.0, 0.5, 0.5], [2.0, 0.5, 0.5]), _t(True, False, False), 2.5,
                   FREE, coarse)
    assert r.entries == 1 and r.exits == 0 and r.skipped_exits == 2
    assert r.unliquidatable is True
    assert r.terminal_value == pytest.approx(0.5)  # cash only: a LOWER BOUND, not a value
    assert (r.residual_units, r.residual_marked_value) == (1.0, 0.5)
    assert r.terminal_fee == 0.0


def test_open_position_marked_in_equity_and_liquidated_in_terminal_value():
    r = run_sleeve(_bars([100, 100], [100, 110]), _t(True, True), 500.0,
                   Costs(0.0, 0.009, 0.0), FINE)
    assert r.equity.iloc[-1] == pytest.approx(550.0)
    assert r.terminal_value == pytest.approx(550.0 * (1 - 0.009))
    assert r.terminal_fee == pytest.approx(550.0 * 0.009) and r.round_trips == 0


def test_costs_reconcile_with_terminal_value():
    c = Costs(0.009, 0.009, 0.0)
    r = run_sleeve(_bars([100] * 3, [100] * 3), _t(True, True, True), 500.0, c, FINE)
    assert r.terminal_value == pytest.approx(500.0 - r.exec_fees - r.terminal_fee, rel=1e-9)
    assert r.traded_notional == pytest.approx(500.0 / 1.009, rel=1e-6)


def test_terminal_close_missing_raises():
    with pytest.raises(ValueError, match="terminal close missing"):
        run_sleeve(_bars([100, 100], [100, np.nan]), _t(True, True), 500.0, FREE, FINE)


def test_missing_open_defers_and_cash_marks_through_gap():
    r = run_sleeve(_bars([100, np.nan, 100], [100, np.nan, 100]), _t(False, True, True),
                   500.0, FREE, FINE)
    assert r.entries == 1 and r.equity.iloc[1] == pytest.approx(500.0)
    assert r.stale_mark_days == 1 and r.max_stale_run == 1


def test_missed_target_is_superseded_not_queued():
    r = run_sleeve(_bars([100, np.nan, 100], [100, 100, 100]), _t(False, True, False),
                   500.0, FREE, FINE)
    assert r.entries == 0 and r.exits == 0


def test_rejected_buy_retried_at_next_open():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(_bars([600, 400], [600, 400]), _t(True, True), 500.0, FREE, coarse)
    assert r.skipped == 1 and r.entries == 1


def test_below_quote_min_is_skipped():
    r = run_sleeve(_bars([100], [100]), _t(True), 0.5, FREE, FINE)
    assert r.skipped == 1 and r.entries == 0


def test_units_floor_to_increment():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_sleeve(_bars([300, 300], [300, 300]), _t(True, True), 500.0, FREE, coarse)
    assert r.equity.iloc[-1] == pytest.approx(500.0)


def test_dca_tranches_are_fee_inclusive_and_mondays_only():
    idx = pd.date_range("2024-01-01", periods=15, freq="D")  # 2024-01-01 is a Monday
    bars = pd.DataFrame({"open": 10.0, "close": 10.0}, index=idx)
    r = run_dca(bars, 520.0, 52, Costs(0.009, 0.009, 0.0), FINE)
    assert r.entries == 3
    spent = 520.0 - (r.equity.iloc[-1] - r.traded_notional)
    assert spent == pytest.approx(3 * 10.0, rel=1e-6)  # 3 tranches of 520/52, fee inside
```

- [ ] **Step 2: Run to verify failure.**
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_sim.py -v`. Expected:
`ModuleNotFoundError ... tools.slow_trend.sim`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/sim.py
"""Self-financing single-asset sleeve simulator. Pure: no I/O, no clock.

State-targeting, not an order queue: on each day with a valid open the sleeve moves to that
day's target. A target missed on an absent open is superseded by later targets. A buy rejected
by size limits is retried at every later open while the target stays long. A sell whose proceeds
fall below quote_min is not executed (retained, counted, retried). Equity is marked at each close;
a stale close is carried for MARKING only, never as a terminal price. An endpoint holding that
cannot meet quote_min is valued at 0 and flagged, never converted into invented cash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Costs:
    entry_fee: float
    exit_fee: float
    slip: float


@dataclass(frozen=True)
class Product:
    base_increment: float
    base_min: float
    quote_min: float


@dataclass
class SleeveResult:
    equity: pd.Series
    initial: float
    terminal_value: float
    exec_fees: float
    terminal_fee: float
    traded_notional: float
    slippage_cost: float
    unliquidatable: bool
    residual_units: float
    residual_marked_value: float
    entries: int
    exits: int
    round_trips: int
    skipped: int
    skipped_exits: int
    exposure: float
    stale_mark_days: int
    max_stale_run: int


def trend_target(state: pd.Series, delay: int) -> pd.Series:
    return state.shift(1 + delay, fill_value=False).astype(bool)


class _Book:
    def __init__(self, cash: float, costs: Costs, product: Product):
        self.cash, self.units, self.costs, self.product = cash, 0.0, costs, product
        self.exec_fees = self.traded = self.slippage = 0.0
        self.entries = self.exits = self.skipped = self.skipped_exits = 0

    def buy(self, open_px: float, budget: float) -> bool:
        px = open_px * (1 + self.costs.slip)
        raw = min(budget, self.cash) / (1 + self.costs.entry_fee) / px
        units = math.floor(raw / self.product.base_increment) * self.product.base_increment
        notional = units * px
        if units < self.product.base_min or notional < self.product.quote_min:
            self.skipped += 1
            return False
        fee = notional * self.costs.entry_fee
        self.cash -= notional + fee
        self.units += units
        self.exec_fees += fee
        self.traded += notional
        self.slippage += units * open_px * self.costs.slip
        self.entries += 1
        return True

    def sell_all(self, open_px: float) -> bool:
        proceeds = self.units * open_px * (1 - self.costs.slip)
        if proceeds < self.product.quote_min:
            self.skipped_exits += 1  # holding retained; retried while the target stays flat
            return False
        fee = proceeds * self.costs.exit_fee
        self.cash += proceeds - fee
        self.exec_fees += fee
        self.traded += proceeds
        self.slippage += self.units * open_px * self.costs.slip
        self.units = 0.0
        self.exits += 1
        return True


class _Marks:
    def __init__(self):
        self.values, self.last_close = [], float("nan")
        self.held = self.stale = self.run = self.max_run = 0

    def mark(self, book: _Book, row, exec_px: float) -> None:
        if math.isnan(row["close"]):
            self.stale += 1
            self.run += 1
            self.max_run = max(self.max_run, self.run)
            if math.isnan(self.last_close) and book.units > 0:
                self.last_close = exec_px  # marking only, before any observed close
        else:
            self.last_close, self.run = row["close"], 0
        self.held += book.units > 0
        self.values.append(book.cash + (book.units * self.last_close if book.units > 0 else 0.0))


def _finish(book: _Book, marks: _Marks, bars: pd.DataFrame, initial: float) -> SleeveResult:
    last = bars["close"].iloc[-1]
    if math.isnan(last):
        raise ValueError("terminal close missing: endpoint is never moved")
    proceeds = book.units * last * (1 - book.costs.slip)
    unliquidatable = book.units > 0 and proceeds < book.product.quote_min
    residual_units, residual_marked = 0.0, 0.0
    if unliquidatable:  # never invent cash for a sale the size model forbids
        residual_units, residual_marked = book.units, book.units * last
        proceeds, terminal_slip = 0.0, 0.0
    else:
        terminal_slip = book.units * last * book.costs.slip
    terminal_fee = proceeds * book.costs.exit_fee
    return SleeveResult(
        equity=pd.Series(marks.values, index=bars.index), initial=initial,
        terminal_value=book.cash + proceeds - terminal_fee, exec_fees=book.exec_fees,
        terminal_fee=terminal_fee, traded_notional=book.traded,
        slippage_cost=book.slippage + terminal_slip, unliquidatable=bool(unliquidatable),
        residual_units=residual_units, residual_marked_value=residual_marked,
        entries=book.entries, exits=book.exits, round_trips=book.exits,
        skipped=book.skipped, skipped_exits=book.skipped_exits,
        exposure=marks.held / len(bars), stale_mark_days=marks.stale,
        max_stale_run=marks.max_run)


def run_sleeve(bars: pd.DataFrame, target: pd.Series, cash: float,
               costs: Costs, product: Product) -> SleeveResult:
    book, marks = _Book(cash, costs, product), _Marks()
    for day, row in bars.iterrows():
        want, px = bool(target.loc[day]), row["open"]
        if not math.isnan(px):
            if want and book.units == 0:
                book.buy(px, book.cash)
            elif not want and book.units > 0:
                book.sell_all(px)
        marks.mark(book, row, px)
    return _finish(book, marks, bars, cash)


def run_dca(bars: pd.DataFrame, cash: float, tranches: int,
            costs: Costs, product: Product) -> SleeveResult:
    """Fee-inclusive tranche of cash/tranches on each of the first `tranches` Mondays."""
    book, marks = _Book(cash, costs, product), _Marks()
    tranche, done, pending = cash / tranches, 0, False
    for day, row in bars.iterrows():
        if day.weekday() == 0 and done + pending < tranches:
            pending = True
        if pending and not math.isnan(row["open"]):
            book.buy(row["open"], tranche)
            pending, done = False, done + 1
        marks.mark(book, row, row["open"])
    return _finish(book, marks, bars, cash)
```

Notes for the implementer:
- `round_trips == exits` because every exit closes the only open position. A position still open
  at the end is liquidated only in `terminal_value` and is not a round trip.
- In `test_dca_tranches_are_fee_inclusive_and_mondays_only`, `equity - traded_notional` is the
  remaining cash (price is constant at 10), so the cash spent is exactly 3 × 10, fees included.

- [ ] **Step 4: Run to verify pass.** Expected: 13 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/sim.py backend/tests/tools/slow_trend/test_sim.py
git commit -m "feat(research): self-financing sleeve simulator (state-targeting, fee-reserved)" -- backend/tools/slow_trend/sim.py backend/tests/tools/slow_trend/test_sim.py
git log -1 --stat && git push
```

### Task 4: Metrics and paired block bootstrap

**Files:**
- Create: `backend/tools/slow_trend/metrics.py`
- Test: `backend/tests/tools/slow_trend/test_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.max_drawdown(equity: pd.Series, initial: float) -> float` (seeded with `initial`);
  - `metrics.weekly_returns(equity: pd.Series) -> pd.Series` (complete Mon–Sun weeks only);
  - `metrics.boundary_returns(equity: pd.Series, initial: float) -> dict` (`head`, `tail`);
  - `metrics.paired_block_ci(a, b, block, n, seed) -> dict`, with keys `mean_excess`, `lo`,
    `hi` and `n_weeks`. `lo`/`hi` are `None` when the series is shorter than the block; it
    raises on unaligned or non-finite input.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_metrics.py
import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.metrics import (boundary_returns, max_drawdown, paired_block_ci,
                                      weekly_returns)
from tools.slow_trend.sim import Costs, Product, run_sleeve


def test_max_drawdown_is_seeded_with_initial():
    eq = pd.Series([95, 120, 90, 130, 65], dtype=float)
    assert max_drawdown(eq, 100.0) == pytest.approx(0.5)
    assert max_drawdown(pd.Series([90.0, 80.0]), 100.0) == pytest.approx(0.2)


def test_constant_price_buy_hold_drawdown_is_entry_fee():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    bars = pd.DataFrame({"open": 100.0, "close": 100.0}, index=idx)
    r = run_sleeve(bars, pd.Series(True, index=idx), 500.0, Costs(0.009, 0.009, 0.0),
                   Product(1e-8, 1e-8, 1.0))
    assert max_drawdown(r.equity, 500.0) == pytest.approx(r.exec_fees / 500.0, rel=1e-6)


def test_weekly_returns_use_complete_weeks_only():
    idx = pd.date_range("2024-01-03", "2024-01-19", freq="D")  # Wed .. Fri
    eq = pd.Series(100.0, index=idx)
    eq.loc["2024-01-14":] = 110.0
    w = weekly_returns(eq)
    assert list(w.index.strftime("%Y-%m-%d")) == ["2024-01-14"]
    assert w.iloc[0] == pytest.approx(0.10)


def test_boundary_returns_report_partial_head_and_tail():
    idx = pd.date_range("2024-01-03", "2024-01-19", freq="D")
    eq = pd.Series(100.0, index=idx)
    eq.iloc[-1] = 121.0
    b = boundary_returns(eq, 100.0)
    assert b["head"] == pytest.approx(0.0) and b["tail"] == pytest.approx(0.21)


def test_identical_series_zero_excess():
    s = pd.Series(np.random.default_rng(1).normal(0, 0.05, 60))
    ci = paired_block_ci(s, s.copy(), block=8, n=500, seed=7)
    assert ci["lo"] == ci["hi"] == ci["mean_excess"] == 0.0


def test_ci_is_deterministic_for_seed():
    rng = np.random.default_rng(2)
    a, b = pd.Series(rng.normal(0.01, 0.05, 80)), pd.Series(rng.normal(0, 0.05, 80))
    assert paired_block_ci(a, b, 8, 300, 11) == paired_block_ci(a, b, 8, 300, 11)


def test_short_series_reports_no_ci_instead_of_crashing():
    ci = paired_block_ci(pd.Series([0.01] * 5), pd.Series([0.0] * 5), 13, 100, 0)
    assert ci["lo"] is None and ci["hi"] is None and ci["n_weeks"] == 5


def test_ci_rejects_misaligned_indexes():
    a = pd.Series([0.1, 0.2], index=[0, 1])
    with pytest.raises(ValueError, match="aligned"):
        paired_block_ci(a, pd.Series([0.1, 0.2], index=[1, 2]), 1, 10, 0)


def test_ci_rejects_non_finite():
    with pytest.raises(ValueError, match="finite"):
        paired_block_ci(pd.Series([0.1, np.nan]), pd.Series([0.1, 0.2]), 1, 10, 0)
```

- [ ] **Step 2: Run to verify failure.**
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_metrics.py -v`. Expected:
`ModuleNotFoundError ... tools.slow_trend.metrics`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/metrics.py
"""Drawdown, complete-week returns and a PAIRED moving-block bootstrap (exploratory, not a gate)."""

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series, initial: float) -> float:
    path = np.r_[initial, equity.to_numpy(dtype=float)]
    peak = np.maximum.accumulate(path)
    return float((1 - path / peak).max())


def _sundays(equity: pd.Series) -> pd.Series:
    return equity[equity.index.weekday == 6]


def weekly_returns(equity: pd.Series) -> pd.Series:
    """Sunday close to Sunday close: complete Mon-Sun weeks only."""
    return _sundays(equity).pct_change().dropna()


def boundary_returns(equity: pd.Series, initial: float) -> dict:
    s = _sundays(equity)
    if s.empty:
        return {"head": float(equity.iloc[-1] / initial - 1), "tail": None}
    return {"head": float(s.iloc[0] / initial - 1), "tail": float(equity.iloc[-1] / s.iloc[-1] - 1)}


def paired_block_ci(a: pd.Series, b: pd.Series, block: int, n: int, seed: int) -> dict:
    if not a.index.equals(b.index):
        raise ValueError("paired series must be aligned (identical ordered indexes)")
    d = (a.to_numpy(dtype=float) - b.to_numpy(dtype=float))
    if not np.isfinite(d).all():
        raise ValueError("paired series must be finite")
    m = len(d)
    if m < block:
        return {"mean_excess": float(d.mean()) if m else None, "lo": None, "hi": None,
                "n_weeks": m}
    rng = np.random.default_rng(seed)
    k = int(np.ceil(m / block))
    starts = rng.integers(0, m - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :m]
    means = d[idx].mean(axis=1)
    return {"mean_excess": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "n_weeks": m}
```

- [ ] **Step 4: Run to verify pass.** Expected: 9 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/metrics.py backend/tests/tools/slow_trend/test_metrics.py
git commit -m "feat(research): seeded drawdown, complete-week returns, paired bootstrap" -- backend/tools/slow_trend/metrics.py backend/tests/tools/slow_trend/test_metrics.py
git log -1 --stat && git push
```

### Task 5: Gates and verdict

**Files:**
- Create: `backend/tools/slow_trend/gates.py`
- Test: `backend/tests/tools/slow_trend/test_gates.py`

**Interfaces:**
- Produces:
  - `gates.passes(trend: dict, bh: dict, initial: float, ratio: float) -> dict` (`G1`, `G2`,
    `pass`);
  - `gates.affected(trend: dict, bh: dict) -> bool` (any `unliquidatable` sleeve in either);
  - `gates.verdict(data_ok: bool, results: dict | None, block_round_trips: int) -> dict`
    (`verdict`, `reason`). `results[period][scenario]` is a `passes` output plus an
    `"affected"` bool, for periods `"dev"` and `"block"`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_gates.py
from tools.slow_trend.gates import affected, passes, verdict

OK = {"G1": True, "G2": True, "pass": True, "affected": False}
BAD = {"G1": False, "G2": True, "pass": False, "affected": False}
OK_AFFECTED = dict(OK, affected=True)
BAD_AFFECTED = dict(BAD, affected=True)


def _side(*flags):
    return {"unliquidatable": list(flags)}


def test_affected_by_buy_hold_only_or_trend_only():
    assert affected(_side(False, False), _side(False, True)) is True   # buy-and-hold only
    assert affected(_side(True, False), _side(False, False)) is True   # trend only
    assert affected(_side(False, False), _side(False, False)) is False


def test_buy_hold_write_down_cannot_manufacture_a_pass():
    # near-boundary: trend would fail the return comparison but for a buy-and-hold write-down
    r = verdict(True, _res(block=OK_AFFECTED), 10)
    assert r == {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}


def test_affected_failure_is_not_a_kill():
    assert verdict(True, _res(dev=BAD_AFFECTED), 10) == {"verdict": "INCONCLUSIVE",
                                                         "reason": "terminal_size"}


def test_unaffected_failure_kills_even_if_other_period_affected():
    assert verdict(True, _res(dev=BAD, block=OK_AFFECTED), 10)["verdict"] == "KILL"


def test_affected_gating_sensitivity_is_terminal_size():
    assert verdict(True, _res(dev_D1=OK_AFFECTED), 10)["reason"] == "terminal_size"


def _res(dev=OK, block=OK, **over):
    base = {s: OK for s in ("P0", "S10", "S25", "D1", "SM")}
    r = {"dev": dict(base, P0=dev), "block": dict(base, P0=block)}
    for key, val in over.items():
        period, scen = key.split("_")
        r[period][scen] = val
    return r


def test_g1_strictly_above_initial():
    t = {"terminal_value": 1000.0, "net_return": 0.0, "max_drawdown": 0.1}
    assert passes(t, t, 1000.0, 2 / 3)["G1"] is False


def test_g2_return_or_drawdown():
    bh = {"terminal_value": 1500.0, "net_return": 0.5, "max_drawdown": 0.6}
    low_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.39}
    high_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.41}
    assert passes(low_dd, bh, 1000.0, 2 / 3)["G2"] is True
    assert passes(high_dd, bh, 1000.0, 2 / 3)["G2"] is False


def test_data_inadequate_first():
    assert verdict(False, None, 0) == {"verdict": "INCONCLUSIVE", "reason": "data"}


def test_block_failure_kills():
    assert verdict(True, _res(block=BAD), 10)["verdict"] == "KILL"


def test_fragile_when_gating_sensitivity_fails():
    assert verdict(True, _res(block_S25=BAD), 10) == {"verdict": "INCONCLUSIVE",
                                                      "reason": "fragile"}


def test_maker_sensitivity_never_consulted():
    assert verdict(True, _res(dev_SM=BAD), 10)["verdict"] == "PASS_TO_FORWARD"
    assert verdict(True, _res(dev=BAD, dev_SM=OK), 10)["verdict"] == "KILL"


def test_insufficient_transitions():
    assert verdict(True, _res(), 3) == {"verdict": "INCONCLUSIVE",
                                        "reason": "insufficient_transitions"}


def test_pass():
    assert verdict(True, _res(), 4) == {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
```

- [ ] **Step 2: Run to verify failure.**
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_gates.py -v`. Expected:
`ModuleNotFoundError ... tools.slow_trend.gates`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/gates.py
"""Preregistered gates and verdict order. KILL abandons THIS candidate; it is not a general
falsification of trend following."""

from tools.slow_trend import prereg as P


def passes(trend: dict, bh: dict, initial: float, ratio: float) -> dict:
    g1 = trend["terminal_value"] > initial
    g2 = (trend["net_return"] >= bh["net_return"]
          or trend["max_drawdown"] <= ratio * bh["max_drawdown"])
    return {"G1": bool(g1), "G2": bool(g2), "pass": bool(g1 and g2)}


def affected(trend: dict, bh: dict) -> bool:
    """A gating comparison is affected when a trend OR buy-and-hold sleeve ended holding units
    it could not sell: its terminal cash is a lower bound, not an economic value."""
    return bool(any(trend["unliquidatable"]) or any(bh["unliquidatable"]))


def verdict(data_ok: bool, results, block_round_trips: int) -> dict:
    if not data_ok:
        return {"verdict": "INCONCLUSIVE", "reason": "data"}
    periods = ("dev", "block")
    p0 = [results[p]["P0"] for p in periods]
    if any(not r["pass"] and not r["affected"] for r in p0):
        return {"verdict": "KILL", "reason": "primary_failed"}
    if any(r["affected"] for r in p0):
        return {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}
    sens = [results[p][s] for p in periods for s in P.GATING_SENSITIVITIES]
    if any(r["affected"] for r in sens):
        return {"verdict": "INCONCLUSIVE", "reason": "terminal_size"}
    if not all(r["pass"] for r in sens):
        return {"verdict": "INCONCLUSIVE", "reason": "fragile"}
    if block_round_trips < P.MIN_BLOCK_ROUND_TRIPS:
        return {"verdict": "INCONCLUSIVE", "reason": "insufficient_transitions"}
    return {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
```

- [ ] **Step 4: Run to verify pass.** Expected: 13 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/gates.py backend/tests/tools/slow_trend/test_gates.py
git commit -m "feat(research): preregistered gates and verdict order" -- backend/tools/slow_trend/gates.py backend/tests/tools/slow_trend/test_gates.py
git log -1 --stat && git push
```

### Task 6: Evaluation (pure) + freeze-enforcing CLI

**Files:**
- Create: `backend/tools/slow_trend/screen.py`
- Test: `backend/tests/tools/slow_trend/test_screen.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `screen.dev_start_from(cal: dict, n: int, last: str) -> str | None`;
  - `screen.run_period(cal, start, end, products: dict[str, Product]) -> dict`;
  - `screen.evaluate(raw: dict[str, pd.DataFrame], constraints: dict[str, dict]) -> dict`
    (pure; returns `verdict`, `data_checks`, `dev_start`, `results`, `diagnostics`);
  - `screen.ensure_new_snapshot(out: Path) -> None` (raises `FileExistsError`);
  - `screen.experiment_identity(prereg_sha, manifest_sha) -> str` (stable across commits);
  - `screen.run_mode(entries, experiment_id, replay=False, new_prereg=False) -> str`
    (`first|retry|replay|new_preregistration`; raises `RuntimeError`);
  - `screen.last_attempt(entries, experiment_id) -> int | None` (parent attempt);
  - `screen.safe_overlap(hourly_path, raw, first) -> dict` (`absent|ok|error`; never raises;
    hashes and parses the same captured bytes);
  - `screen.source_digest(paths, root=REPO) -> str` (recorded per attempt);
  - `screen.replay_label(entries, experiment_id, source_sha) -> (str, int)`
    (`replay` or `corrected_replay`, linked to the first completed attempt);
  - CLI `python -m tools.slow_trend.screen {fetch|lock|run [--replay] [--new-preregistration]}`.

- [ ] **Step 1: Write the failing tests** (synthetic data; no network, no git)

```python
# backend/tests/tools/slow_trend/test_screen.py
import numpy as np
import pandas as pd
import pytest

from tools.slow_trend import screen as S
from tools.slow_trend.sim import Product

FINE = Product(1e-8, 1e-8, 1.0)
PRODUCTS = {"BTC-USD": FINE, "ETH-USD": FINE}
CONSTRAINTS = {p: {"base_increment": 1e-8, "base_min": 1e-8, "quote_min": 1.0}
               for p in PRODUCTS}


def _cal(prices, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D")
    p = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"open": p.shift(1).fillna(p.iloc[0]), "close": p})


def _raw(first, last, drop=()):
    days = pd.date_range(first, last, freq="D")
    p = np.linspace(100, 300, len(days))
    epoch_s = ((days - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype("int64")
    df = pd.DataFrame({"start": epoch_s,
                       "open": np.r_[p[0], p[:-1]], "close": p, "volume": 1.0, "page": 0})
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    keep = ~days.strftime("%Y-%m-%d").isin(list(drop))
    return df[keep].reset_index(drop=True)


def test_period_runs_all_scenarios_and_comparators():
    up = np.linspace(100, 300, 400)
    out = S.run_period({"BTC-USD": _cal(up), "ETH-USD": _cal(up)}, "2020-05-01",
                       "2021-01-31", PRODUCTS)
    p0 = out["scenarios"]["P0"]
    assert set(out["scenarios"]) == {"P0", "S10", "S25", "D1", "SM"}
    assert {"trend", "buy_hold", "dca52", "cash", "passes", "ci_vs_bh", "ci_vs_dca52",
            "ci_vs_cash"} <= set(p0)
    assert p0["cash"]["terminal_value"] == 1000.0 and p0["passes"]["affected"] is False
    assert p0["trend"]["round_trips"] == 0
    assert set(p0["trend"]["per_sleeve"]) == {"BTC-USD", "ETH-USD"}


def test_pre_period_long_does_not_carry_in():
    up = np.linspace(100, 300, 400)
    btc, eth = _cal(up), _cal(up)
    for c in (btc, eth):
        c.loc["2020-04-30", ["open", "close"]] = np.nan  # invalidates the next 100 windows
    out = S.run_period({"BTC-USD": btc, "ETH-USD": eth}, "2020-05-01", "2020-06-30", PRODUCTS)
    assert out["scenarios"]["P0"]["trend"]["entries"] == 0
    assert out["diagnostics"]["BTC-USD"]["suppressed_decision_days"] == 61


def test_dev_start_is_first_common_valid_sma_day():
    btc = _cal(np.linspace(100, 300, 300))
    eth = _cal(np.linspace(100, 300, 290), start="2020-01-11")
    cal = {"BTC-USD": btc, "ETH-USD": eth.reindex(btc.index)}
    assert S.dev_start_from(cal, 100, "2020-12-31") == "2020-04-19"  # Jan 11 + 99 days
    btc.loc["2020-02-01", "close"] = np.nan
    assert S.dev_start_from(cal, 100, "2020-12-31") == "2020-05-11"  # first full window after gap


def test_dev_start_none_when_history_starts_too_late():
    cal = {p: _cal(np.linspace(100, 300, 50)) for p in PRODUCTS}
    assert S.dev_start_from(cal, 100, "2020-12-31") is None


def test_evaluate_end_to_end_on_monotone_rise_kills():
    raw = {p: _raw("2024-01-01", "2026-10-02") for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["dev_start"] == "2024-04-09"
    assert rep["results"]["dev"]["scenarios"]["P0"]["passes"] == {"G1": True, "G2": False,
                                                                 "pass": False,
                                                                 "affected": False}
    assert rep["verdict"] == {"verdict": "KILL", "reason": "primary_failed"}


def test_evaluate_empty_input_is_inadequate():
    raw = {p: _raw("2024-01-01", "2026-10-02").iloc[0:0] for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"]["reason"] == "data"
    assert rep["data_checks"]["inadequate_because"] == "no_aligned_rows"


def test_evaluate_all_misaligned_input_is_inadequate():
    raw = {p: _raw("2024-01-01", "2026-10-02").assign(start=lambda d: d["start"] + 3600)
           for p in PRODUCTS}
    assert S.evaluate(raw, CONSTRAINTS)["data_checks"]["inadequate_because"] == "no_aligned_rows"


def test_evaluate_coverage_starting_at_block_start_is_inadequate():
    raw = {p: _raw("2025-04-14", "2026-10-02") for p in PRODUCTS}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"]["reason"] == "data"
    assert rep["data_checks"]["inadequate_because"] == "no_development_coverage"


def test_unused_early_conflict_is_excluded_not_a_crash():
    early = _raw("2023-06-01", "2026-10-02")
    dup = early.iloc[[3]].copy()
    dup["close"] += 1.0
    dup["high"] += 1.0
    raw = {"BTC-USD": pd.concat([early, dup]), "ETH-USD": _raw("2024-01-01", "2026-10-02")}
    rep = S.evaluate(raw, CONSTRAINTS)  # BTC's 2023 rows precede common_first and are unused
    assert rep["data_checks"]["common_first"] == "2024-01-01"
    assert rep["results"] is not None


def test_malformed_hourly_overlap_is_a_diagnostic(tmp_path):
    bad = tmp_path / "BTC-USD.parquet"
    bad.write_text("not a parquet file")
    out = S.safe_overlap(bad, _raw("2024-01-01", "2024-03-01"), "2024-01-01")
    assert out["status"] == "error" and out["sha256"].startswith("sha256:")
    assert S.safe_overlap(tmp_path / "missing.parquet", _raw("2024-01-01", "2024-03-01"),
                          "2024-01-01") == {"status": "absent"}


def test_unreadable_hourly_input_is_a_diagnostic_without_a_hash(tmp_path):
    unreadable = tmp_path / "BTC-USD.parquet"
    unreadable.mkdir()  # exists, but read_bytes() raises
    out = S.safe_overlap(unreadable, _raw("2024-01-01", "2024-03-01"), "2024-01-01")
    assert out["status"] == "error" and out["sha256"] is None


def test_source_digest_tracks_content(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("x = 1\n")
    before = S.source_digest([f], root=tmp_path)
    assert before == S.source_digest([f], root=tmp_path)
    f.write_text("x = 2\n")
    assert S.source_digest([f], root=tmp_path) != before


def test_replay_label_distinguishes_corrected_computation():
    entries = [{"experiment_id": "E", "status": "started", "attempt": 1, "source_sha256": "s1"},
               {"experiment_id": "E", "status": "completed", "attempt": 1, "source_sha256": "s1"}]
    assert S.replay_label(entries, "E", "s1") == ("replay", 1)
    assert S.replay_label(entries, "E", "s2") == ("corrected_replay", 1)


def test_missing_terminal_close_is_inadequate():
    raw = {"BTC-USD": _raw("2024-01-01", "2026-10-02"),
           "ETH-USD": _raw("2024-01-01", "2026-10-02", drop=("2026-10-02",))}
    rep = S.evaluate(raw, CONSTRAINTS)
    assert rep["verdict"] == {"verdict": "INCONCLUSIVE", "reason": "data"}
    assert rep["data_checks"]["terminal"]["ETH-USD"]["2026-10-02"] is False


def test_conflicting_duplicate_is_inadequate_not_crash():
    btc = _raw("2024-01-01", "2026-10-02")
    dup = btc.iloc[[500]].copy()
    dup["close"] += 1.0
    dup["high"] += 1.0
    raw = {"BTC-USD": pd.concat([btc, dup]), "ETH-USD": _raw("2024-01-01", "2026-10-02")}
    assert S.evaluate(raw, CONSTRAINTS)["verdict"]["reason"] == "data"


def test_refuses_snapshot_overwrite(tmp_path):
    S.ensure_new_snapshot(tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(FileExistsError, match="new preregistration"):
        S.ensure_new_snapshot(tmp_path)


def test_experiment_identity_ignores_head_but_not_prereg_or_snapshot():
    base = S.experiment_identity("prereg", "snap")
    assert base == S.experiment_identity("prereg", "snap")
    assert len({base, S.experiment_identity("x", "snap"),
                S.experiment_identity("prereg", "x")}) == 3


def _e(exp, status, attempt=1, head="h1"):
    return {"experiment_id": exp, "status": status, "attempt": attempt, "head": head}


def test_first_attempt():
    assert S.run_mode([], "E") == "first"


def test_failure_then_new_commit_is_still_a_retry():
    entries = [_e("E", "started", 1, "h1"), _e("E", "failed", 1, "h1")]
    assert S.run_mode(entries, "E") == "retry"  # HEAD h2 is irrelevant to the permission
    assert S.last_attempt(entries, "E") == 1


def test_unfinished_start_is_a_retry():
    assert S.run_mode([_e("E", "started")], "E") == "retry"


def test_completion_then_new_commit_refuses_without_replay():
    entries = [_e("E", "started"), _e("E", "completed")]
    with pytest.raises(RuntimeError, match="replay"):
        S.run_mode(entries, "E")
    assert S.run_mode(entries, "E", replay=True) == "replay"


def test_new_preregistration_must_be_explicit():
    entries = [_e("E", "started"), _e("E", "completed")]
    with pytest.raises(RuntimeError, match="new-preregistration"):
        S.run_mode(entries, "F")
    assert S.run_mode(entries, "F", new_prereg=True) == "new_preregistration"
```

- [ ] **Step 2: Run to verify failure.**
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_screen.py -v`. Expected:
`ModuleNotFoundError ... tools.slow_trend.screen`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/screen.py
"""Preregistered slow-trend screen. See the plan's Preregistration section.

  python -m tools.slow_trend.screen fetch          # raw snapshot + manifest (refuses overwrite)
  python -m tools.slow_trend.screen lock           # write snapshot.lock (commit it before run)
  python -m tools.slow_trend.screen run [--replay] # frozen grid -> ledger + report
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tools.slow_trend import daily_bars as D
from tools.slow_trend import gates as G
from tools.slow_trend import metrics as M
from tools.slow_trend import prereg as P
from tools.slow_trend.rule import decisions, desired_state
from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

BACKEND = Path(__file__).resolve().parents[2]
REPO = BACKEND.parent
OUT = BACKEND / "data" / "research" / "slow_trend"
LOCK = Path(__file__).with_name("snapshot.lock")
BLOCKS = (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)


# ── pure evaluation ────────────────────────────────────────────────────────────


def dev_start_from(cal: dict, n: int, last: str) -> Optional[str]:
    ok = None
    for c in cal.values():
        v = decisions(c["close"], n).notna()
        ok = v if ok is None else (ok & v)
    days = ok[ok].index
    days = days[days <= pd.Timestamp(last)]
    return str(days[0].date()) if len(days) else None


def _summ(results: list, initial: float) -> dict:
    eq = sum(r.equity for r in results)
    tv = sum(r.terminal_value for r in results)
    ex = sum(r.exec_fees for r in results)
    tf = sum(r.terminal_fee for r in results)
    return {
        "terminal_value": tv, "net_return": tv / initial - 1,
        "max_drawdown": M.max_drawdown(eq, initial),
        "exec_fees": ex, "terminal_fee": tf, "total_fees": ex + tf,
        "slippage_cost": sum(r.slippage_cost for r in results),
        "unliquidatable": [r.unliquidatable for r in results],
        "residual_units": [r.residual_units for r in results],
        "residual_marked_value": sum(r.residual_marked_value for r in results),
        "entries": sum(r.entries for r in results), "exits": sum(r.exits for r in results),
        "round_trips": sum(r.round_trips for r in results),
        "skipped": sum(r.skipped for r in results),
        "skipped_exits": sum(r.skipped_exits for r in results),
        "executed_turnover": sum(r.traded_notional for r in results) / initial,
        "exposure_mean": float(np.mean([r.exposure for r in results])),
        "boundary_returns": M.boundary_returns(eq, initial),
        "per_sleeve": {pid: {"terminal_value": r.terminal_value,
                             "net_return": r.terminal_value / r.initial - 1,
                             "max_drawdown": M.max_drawdown(r.equity, r.initial),
                             "exposure": r.exposure, "round_trips": r.round_trips,
                             "stale_mark_days": r.stale_mark_days,
                             "max_stale_run": r.max_stale_run}
                       for pid, r in zip(P.PRODUCTS, results)},
        "_equity": eq,
    }


def run_period(cal: dict, start: str, end: str, products: dict) -> dict:
    diagnostics = {}
    for pid in P.PRODUCTS:
        dec = decisions(cal[pid]["close"], P.SMA_LEN).loc[start:end]
        diagnostics[pid] = {"suppressed_decision_days": int(dec.isna().sum())}
    out = {"start": start, "end": end, "diagnostics": diagnostics, "scenarios": {}}
    for sc in P.SCENARIOS:
        costs = Costs(sc.entry_fee, sc.exit_fee, sc.slip)
        legs = {"trend": [], "buy_hold": [], "dca52": []}
        for pid in P.PRODUCTS:
            full, prod = cal[pid], products[pid]
            bars = full.loc[start:end]
            # warmup from history, THEN slice, THEN hold state: the period starts flat
            dec = decisions(full["close"], P.SMA_LEN).loc[start:end]
            tgt = trend_target(desired_state(dec), sc.delay)
            legs["trend"].append(run_sleeve(bars, tgt, P.SLEEVE_USD, costs, prod))
            hold = pd.Series(True, index=bars.index)
            legs["buy_hold"].append(run_sleeve(bars, hold, P.SLEEVE_USD, costs, prod))
            legs["dca52"].append(run_dca(bars, P.SLEEVE_USD, P.DCA_TRANCHES, costs, prod))
        s = {k: _summ(v, P.INITIAL_USD) for k, v in legs.items()}
        s["cash"] = {"terminal_value": P.INITIAL_USD, "net_return": 0.0, "max_drawdown": 0.0}
        s["passes"] = {**G.passes(s["trend"], s["buy_hold"], P.INITIAL_USD,
                                  P.G2_DRAWDOWN_RATIO),
                       "affected": G.affected(s["trend"], s["buy_hold"])}
        wt = M.weekly_returns(s["trend"]["_equity"])
        for name, key in (("buy_hold", "bh"), ("dca52", "dca52")):
            wc = M.weekly_returns(s[name]["_equity"])
            s[f"ci_vs_{key}"] = {str(b): M.paired_block_ci(wt, wc, b, P.BOOT_RESAMPLES,
                                                           P.BOOT_SEED) for b in BLOCKS}
        s["ci_vs_cash"] = {str(b): M.paired_block_ci(wt, wt * 0.0, b, P.BOOT_RESAMPLES,
                                                     P.BOOT_SEED) for b in BLOCKS}
        for k in ("trend", "buy_hold", "dca52"):
            s[k].pop("_equity")
        out["scenarios"][sc.name] = s
    return out


def _inadequate(report: dict, why: str) -> dict:
    report["data_checks"]["inadequate_because"] = why
    report["verdict"] = G.verdict(False, None, 0)
    return report


def evaluate(raw: dict, constraints: dict) -> dict:
    report = {"data_checks": {"terminal": {}}, "dev_start": None, "results": None}
    firsts = {p: D.first_day(raw[p]) for p in P.PRODUCTS}
    report["data_checks"]["first_day"] = firsts
    if any(f is None for f in firsts.values()):
        return _inadequate(report, "no_aligned_rows")
    common_first = max(firsts.values())
    report["data_checks"]["common_first"] = common_first
    if common_first > P.DEV_END:
        return _inadequate(report, "no_development_coverage")
    windows = {"dev": (common_first, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    audits = {w: {p: D.audit(raw[p], a, b, P.MAX_MISSING_DAYS) for p in P.PRODUCTS}
              for w, (a, b) in windows.items()}
    report["data_checks"]["audits"] = audits
    if not all(v["adequate"] for w in audits.values() for v in w.values()):
        return _inadequate(report, "audit")
    # rows outside [common_first, BLOCK_END] are unused and excluded BEFORE normalisation
    cal = {p: D.to_calendar(D.normalise(D.window(raw[p], common_first, P.BLOCK_END)),
                            common_first, P.BLOCK_END) for p in P.PRODUCTS}
    checks = report["data_checks"]
    for p in P.PRODUCTS:
        checks["terminal"][p] = {d: bool(pd.notna(cal[p].loc[d, "close"]))
                                 for d in (P.DEV_END, P.BLOCK_END)}
    if not all(all(t.values()) for t in checks["terminal"].values()):
        return _inadequate(report, "terminal_close_missing")
    dev_start = dev_start_from(cal, P.SMA_LEN, P.DEV_END)
    report["dev_start"] = dev_start
    if dev_start is None:
        return _inadequate(report, "no_common_valid_sma_before_dev_end")
    products = {p: Product(**constraints[p]) for p in P.PRODUCTS}
    periods = {"dev": (dev_start, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    res = {n: run_period(cal, a, b, products) for n, (a, b) in periods.items()}
    passes = {n: {s: r["scenarios"][s]["passes"] for s in r["scenarios"]} for n, r in res.items()}
    rt = res["block"]["scenarios"]["P0"]["trend"]["round_trips"]
    report.update(periods=periods, results=res, verdict=G.verdict(True, passes, rt))
    return report


# ── freeze mechanics ───────────────────────────────────────────────────────────


def ensure_new_snapshot(out: Path) -> None:
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out} already holds a snapshot; a new snapshot is a new "
                              "preregistration - move the old directory aside deliberately")


def experiment_identity(prereg_sha: str, manifest_sha: str) -> str:
    """Stable across commits: HEAD is attempt provenance, never part of the permission key."""
    return hashlib.sha256(f"{prereg_sha}|{manifest_sha}".encode()).hexdigest()[:16]


def run_mode(entries: list, experiment_id: str, replay: bool = False,
             new_prereg: bool = False) -> str:
    mine = [e["status"] for e in entries if e.get("experiment_id") == experiment_id]
    if "completed" in mine:
        if replay:
            return "replay"
        raise RuntimeError("experiment already completed: rerun only as --replay")
    if mine:
        return "retry"  # an earlier attempt started or failed; a new commit does not reset it
    others_done = any(e["status"] == "completed" and e.get("experiment_id") != experiment_id
                      for e in entries)
    if others_done:
        if not new_prereg:
            raise RuntimeError("a different preregistration already completed; pass "
                               "--new-preregistration explicitly")
        return "new_preregistration"
    return "first"


def last_attempt(entries: list, experiment_id: str) -> Optional[int]:
    seqs = [e["attempt"] for e in entries
            if e.get("experiment_id") == experiment_id and e["status"] == "started"]
    return seqs[-1] if seqs else None


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(ledger: Path, entry: dict) -> None:
    with ledger.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


async def _fetch() -> None:
    from clients import coinbase_client as cc

    ensure_new_snapshot(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    start = int(pd.Timestamp(P.FETCH_FROM).timestamp())
    end = int(pd.Timestamp(P.BLOCK_END).timestamp()) + D.DAY
    manifest = {"fetched_at": _now(), "products": {}}
    for pid in P.PRODUCTS:
        raw = await D.fetch_daily(pid, start, end, cc._get)
        path = OUT / f"{pid}.raw.parquet"
        raw.to_parquet(path, index=False)
        manifest["products"][pid] = {
            "sha256": _sha(path), "rows": len(raw), "pages": int(raw["page"].max()) + 1,
            "first_day": D.first_day(raw),
            "constraints": D.product_constraints(await cc.get_product(pid))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _lock() -> None:
    LOCK.write_text(_sha(OUT / "manifest.json") + "\n")
    print(f"wrote {LOCK}; commit it before `run`")


def safe_overlap(hourly_path: Path, raw: pd.DataFrame, first: str) -> dict:
    """Informational only. Never raises: the bytes are read ONCE, then hashed and parsed from
    that same capture, and the error path does no file I/O."""
    if not hourly_path.exists():
        return {"status": "absent"}
    digest = None
    try:
        data = hourly_path.read_bytes()
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        cal = D.to_calendar(D.normalise(D.window(raw, first, P.BLOCK_END)), first, P.BLOCK_END)
        out = D.hourly_overlap(pd.read_parquet(io.BytesIO(data)), cal)
        return {"status": "ok", "sha256": digest, **out}
    except Exception as exc:
        return {"status": "error", "sha256": digest, "error": repr(exc)}


def source_digest(paths: list, root: Path = REPO) -> str:
    """Digest of the measurement source actually on disk, recorded per attempt."""
    h = hashlib.sha256()
    for path in sorted(paths, key=str):
        h.update(str(path.relative_to(root)).replace("\\", "/").encode() + b"\0")
        h.update(path.read_bytes() + b"\0")
    return "sha256:" + h.hexdigest()


def _source_files() -> list:
    return sorted(Path(__file__).parent.glob("*.py")) + [BACKEND / "clients" / "coinbase_client.py"]


def replay_label(entries: list, experiment_id: str, source_sha: str):
    """`replay` when the completed attempt ran the same source; otherwise `corrected_replay`,
    linked to that first completed attempt. Both reports are kept."""
    done = [e for e in entries
            if e.get("experiment_id") == experiment_id and e["status"] == "completed"]
    first = done[0]
    label = "replay" if first.get("source_sha256") == source_sha else "corrected_replay"
    return label, first.get("attempt")


def _run(replay: bool, new_prereg: bool) -> None:
    dirty = _git("status", "--porcelain", "--", *P.FREEZE_PATHS)
    if dirty:
        sys.exit(f"refusing to run: uncommitted changes\n{dirty}")
    manifest_sha = _sha(OUT / "manifest.json")
    if not LOCK.exists() or LOCK.read_text().strip() != manifest_sha:
        sys.exit("snapshot.lock missing or does not match the manifest")
    manifest = json.loads((OUT / "manifest.json").read_text())
    head, prereg_sha = _git("rev-parse", "HEAD"), _sha(Path(P.__file__))
    exp_id = experiment_identity(prereg_sha, manifest_sha)
    ledger = OUT / "runs.jsonl"
    entries = ([json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
               if ledger.exists() else [])
    mode = run_mode(entries, exp_id, replay=replay, new_prereg=new_prereg)
    src = source_digest(_source_files())
    corrects = None
    if mode == "replay":
        mode, corrects = replay_label(entries, exp_id, src)
    attempt = 1 + max((e.get("attempt", 0) for e in entries), default=0)
    base = {"experiment_id": exp_id, "attempt": attempt, "head": head, "mode": mode,
            "source_sha256": src, "replays_attempt": corrects}
    _append(ledger, {**base, "status": "started", "parent": last_attempt(entries, exp_id),
                     "at": _now()})
    try:
        raw, constraints = {}, {}
        for pid, m in manifest["products"].items():
            path = OUT / f"{pid}.raw.parquet"
            if _sha(path) != m["sha256"]:
                raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
            raw[pid], constraints[pid] = pd.read_parquet(path), m["constraints"]
        report = evaluate(raw, constraints)
    except Exception as exc:
        _append(ledger, {**base, "status": "failed", "error": repr(exc), "at": _now()})
        raise
    first = report["data_checks"].get("common_first")
    report["hourly_overlap_informational"] = (
        {pid: safe_overlap(BACKEND / "data" / "history" / f"{pid}.parquet", raw[pid], first)
         for pid in P.PRODUCTS} if first else {})
    report.update(experiment_id=exp_id, attempt=attempt, mode=mode, head=head,
                  source_sha256=src, replays_attempt=corrects,
                  prereg_sha256=prereg_sha, manifest_sha256=manifest_sha)
    name = f"report_{exp_id}_a{attempt}_{mode}.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=str))
    _append(ledger, {**base, "status": "completed", "report": name,
                     "verdict": report["verdict"], "at": _now()})
    print(json.dumps({"verdict": report["verdict"], "mode": mode, "report": name}, indent=2))


if __name__ == "__main__":
    cmd, flags = (sys.argv[1] if len(sys.argv) > 1 else ""), set(sys.argv[2:])
    if cmd == "fetch":
        from dotenv import load_dotenv

        load_dotenv(REPO / ".env")
        asyncio.run(_fetch())
    elif cmd == "lock":
        _lock()
    elif cmd == "run":
        _run(replay="--replay" in flags, new_prereg="--new-preregistration" in flags)
    else:
        sys.exit("usage: python -m tools.slow_trend.screen "
                 "{fetch|lock|run [--replay] [--new-preregistration]}")
```

Notes for the implementer:
- In `test_pre_period_long_does_not_carry_in`, the NaN on 2020-04-30 makes every window through
  2020-08-07 invalid. All 61 days of the May–June period are therefore suppressed, and with the
  B5 slicing the sleeve never enters. Before that fix it bought on 2020-05-02.
- In `test_evaluate_end_to_end_on_monotone_rise_kills`, the trend's first valid decision is on
  2024-04-09 (2024-01-01 + 99 days) and it buys at the 04-10 open, one day after buy-and-hold.
  `_raw` sets `open[t] = close[t-1]`, so each entry day's mark includes a positive intraday
  move. The seeded drawdowns are therefore NOT exactly the entry fee and NOT identical: the later
  entry earns a smaller percentage first-day gain, so its drawdown is slightly larger. Either
  way, the trend's return is lower and its drawdown is not ≤ 2/3 of buy-and-hold's, so G1
  passes and G2 fails. The test asserts both, so the `KILL` cannot come from another cause.

- [ ] **Step 4: Run to verify pass.** Expected: 22 passed. Then run the whole package with
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend -v` (expected 77 passed).
Finally, ruff 0.9.0 `check` and `format --check` on `backend/tools/slow_trend` and
`backend/tests/tools/slow_trend` must both be clean.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/screen.py backend/tests/tools/slow_trend/test_screen.py
git commit -m "feat(research): slow-trend evaluation + freeze-enforcing runner" -- backend/tools/slow_trend/screen.py backend/tests/tools/slow_trend/test_screen.py
git log -1 --stat && git push
```

### Task 7: Review gate, freeze, and the single run

- [ ] **Step 1:** Request Codex review of the full branch via the session link. Address findings
  through `superpowers:receiving-code-review`, with a failing test before each fix.
- [ ] **Step 2:** `cd backend && ../.venv/Scripts/python.exe -m tools.slow_trend.screen fetch`.
  Paste the manifest (first days, rows, pages, constraints) into the PR description.
- [ ] **Step 3:** `../.venv/Scripts/python.exe -m tools.slow_trend.screen lock`, then commit
  `backend/tools/slow_trend/snapshot.lock` (not `.py`, so the hook skips the suite) and push.
- [ ] **Step 4:** With a clean tree, run `../.venv/Scripts/python.exe -m tools.slow_trend.screen run`
  once. On failure, fix through a reviewed commit; the ledger records the retry.
- [ ] **Step 5:** Report to the operator:
  - the verdict and its reason;
  - P0 development and block tables against all comparators;
  - the sensitivity outcomes and the data checks;
  - the run_id and the report path.

  State plainly that `PASS_TO_FORWARD` does not authorise funding and that `KILL` abandons this
  candidate only. Log the result to CHANGELOG, memory and the win-factors ledger.
