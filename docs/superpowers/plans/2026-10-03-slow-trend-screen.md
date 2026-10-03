# BTC/ETH Slow-Trend Falsification Screen — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run ONE preregistered historical screen of a frozen BTC/ETH SMA100 trend rule against cash,
buy-and-hold and a defined 52-week DCA at the account's verified fees, and emit exactly one verdict:
`KILL`, `INCONCLUSIVE` or `PASS_TO_FORWARD`.

**Architecture:** A self-contained research package `backend/tools/slow_trend/`. Pure layers
(`rule`, `sim`, `metrics`, `gates`) take DataFrames/scalars and never touch network, DB or clock;
one I/O layer (`daily_bars`) fetches and validates Coinbase daily candles into a hashed snapshot;
one runner (`screen`) loads the snapshot, runs the frozen scenario grid and writes a JSON report.
Every decision that could be tuned lives in `prereg.py` and is pinned by a test, so changing a
frozen value is a visible, reviewed diff — never a silent rerun.

**Tech Stack:** Python 3.11, pandas 3.0, numpy 2.4, pytest, existing `clients/coinbase_client`
(authenticated `_get`). No new dependencies. No DB access. No backend process involvement.

**Spec:** this plan's **Preregistration** section is the spec. It is the converged result of the
Claude–Codex debate of 2026-10-03 (session-link messages `a3f4f64e`, `7eee3897`, `28e4c12a`,
`2ed9d142`). Background: `docs/specs/2026-09-27-strategy-evidence-and-decision.md` (branch
`docs/strategy-evidence-decision`) and the 58.90 ABSTAIN verdict.

## Preregistration (frozen BEFORE any rule is run on any data)

**Purpose.** This is an economic abandonment screen for ONE candidate, not a test of trend
following in general. `PASS_TO_FORWARD` earns only a forward paper run. No verdict authorises
funding, and none establishes superiority.

**Verified inputs**
- Account fee tier: read-only `GET transaction_summary` on 2026-10-03 returned `pricing_tier=Intro`,
  `maker_fee_rate=0.005`, `taker_fee_rate=0.009`, 30-day volume 0. These are applied as a
  constant **planning assumption**, not a reconstruction of historical fees.

**Universe and rule**
- Universe: `BTC-USD`, `ETH-USD`. Chosen a priori for liquidity and scope, NOT from ledger results.
- Signal at each completed UTC daily close `t`: `long` iff `close[t] > mean(close[t-99..t])`
  (arithmetic, 100 closes, including `t`). Equality is `flat`. A window containing any missing
  day yields NO decision; the previous state is held.
- SMA length 100 is a discretionary research choice, not evidence of an optimal horizon.
  No other length is run. No 20-week variant is run.
- Position: all-in or all-cash per sleeve. Buy only on a flat→long transition, sell only on
  long→flat. No resizing, shorts, stops, leverage or borrowing.
- Capital: USD 1,000 split into two independent sleeves of USD 500 (BTC, ETH). No transfers
  and no rebalancing between sleeves. Cash earns 0.

**Timing**
- Primary: decision at close of day `t`, execution at the OPEN of day `t+1`. Labelled an
  **idealised proxy**: Coinbase candle open is the first trade, not an executable quote.
- Sensitivity D1: one additional day of delay (execute at the open of `t+2`). Never selected
  because it performs better. It is not a conservative bound.
- If the scheduled execution day has no bar, execute at the next available open.

**Costs** (applied to actual entry/exit notionals; fee cash reserved; self-financing compounding)
- Primary P0: taker 0.90% on entry AND exit, 0 bps adverse price adjustment.
- S10 / S25: P0 plus an assumed adverse price stress of 10 / 25 bps per leg (buy at
  `open*(1+s)`, sell at `open*(1-s)`). These are assumed stresses, not measured spreads.
- SM: maker 0.50% entry, taker 0.90% exit. **Optimistic sensitivity only.** It can never rescue
  a P0 failure.
- Every trading comparator (buy-and-hold, DCA) pays the same scenario's fees and stress on every
  leg, including the final liquidation.
- Size rounding: units are floored to the product `base_increment`. Orders below
  `quote_min_size` are skipped and counted. Both values are recorded from the live product
  endpoint into the data manifest.

**Comparators** (same USD 1,000, same period start, same sleeve split)
- Cash: USD 1,000 held, return 0.
- Buy-and-hold: each sleeve buys fully at the first open of the period, then holds.
- DCA-52: 52 equal weekly tranches on the first 52 Mondays (UTC) on or after the period start,
  each buying `500/52` USD per sleeve at that day's open, then hold. Unspent cash stays in equity.
  Reported, but **not** a gate.
- All risky strategies are marked at each close (no liquidation cost) for equity and drawdown.
  A terminal liquidation value is reported at the last close, with exit fee and stress.

**Data and periods**
- Source, fixed a priori: Coinbase Advanced daily candles (`granularity=ONE_DAY`) for both
  products, fetched once into `backend/data/research/slow_trend/` with a sha256 manifest. The
  source is never switched by P&L.
- In-repo hourly `data/history/*.parquet` is used ONLY for an informational overlap check:
  closes of complete UTC days (24 hourly bars) versus daily candle closes.
- Development period: from the first day both SMAs are defined over common verified coverage,
  through **2025-04-13**.
- Validation block (named "previously observed retrospective validation block"): **2025-04-14
  through 2026-10-02**, the last complete UTC day. It is NOT a clean holdout. We chose this
  candidate after studying this period for other policies, so it carries indirect adaptation.
  Only untouched future data earns prospective status. Its SMA warmup uses development closes.
  Its results are reported separately and never averaged with development.
- Every strategy starts in cash on the first day of each period.
- Missing-day policy: per period, count calendar days with no bar. ≤ 3 are allowed: they are
  never filled, so affected SMA windows produce no decision. > 3 makes the data inadequate.
  Duplicate timestamps or non-midnight-UTC starts also make the data inadequate.

**Metrics** (per period, per scenario)
- Portfolio and per-sleeve: terminal liquidation value, net return, max drawdown of marked
  equity, exposure fraction, transitions (entries/exits), completed round trips, fees paid,
  skipped orders.
- Weekly (W-SUN) marked-equity returns. Trend-minus-comparator excess versus buy-and-hold, cash
  and DCA-52.
- Moving-block bootstrap of mean weekly excess: block 8 weeks, 10,000 resamples, seed
  `20261003`. Trend and comparator are resampled with the SAME blocks. BTC and ETH are never
  treated as independent replications. Blocks 4 and 13 are reported as sensitivity. The CIs
  are exploratory only and are NOT gates.

**Gates** (evaluated on the portfolio = sum of both sleeves)
- G1 (cash): terminal liquidation value > USD 1,000.
- G2 (risk vs buy-and-hold): trend net return ≥ buy-and-hold net return, OR trend max drawdown
  ≤ (2/3) × buy-and-hold max drawdown. The 2/3 threshold is an **explicitly arbitrary
  provisional utility gate**, not a claim of statistical materiality.

**Verdict, in this order, frozen**
1. Data inadequate in either period → `INCONCLUSIVE` (reason `data`).
2. P0 fails G1 or G2 in development OR in the validation block → `KILL`.
   A validation-block failure is never averaged away.
3. P0 passes both, but any of S10, S25 or D1 fails G1 or G2 in either period → `INCONCLUSIVE`
   (reason `fragile`). SM is reported and never consulted.
4. Validation block has fewer than 4 completed round trips across both sleeves under P0 →
   `INCONCLUSIVE` (reason `insufficient_transitions`).
5. Otherwise `PASS_TO_FORWARD`.

**Freeze mechanics:** `screen` refuses to run when `git status --porcelain backend/tools/slow_trend`
is non-empty. It records the HEAD SHA and the sha256 of `prereg.py` and of the data manifest in
the report. Any change to `prereg.py` after a run must be a new commit and is reported as a new
preregistration, never as a revision of a verdict.

## Global Constraints

- Read-only with respect to the app. No writes to `coinbase.db`, no backend restart, no orders.
  Network use is limited to public-data candle/product reads via `clients.coinbase_client._get`.
- **8001 is live paper trading.** Per operator rule, do NOT run `pytest` or commit `.py` files
  (the pre-commit hook runs the full suite) without asking the operator first. Docs-only commits
  skip the suite.
- Outputs go under `backend/data/research/slow_trend/`, which is gitignored and never committed.
- Use the pinned ruff 0.9.0 (`$CLAUDE_JOB_DIR/tmp/ruff090/Scripts/ruff.exe`) for `check` and
  `format --check`. PATH ruff 0.15.x formats differently.
- Python `.venv/Scripts/python.exe` from the repo root. Tests run from `backend/`.
- No value in `prereg.py` may be changed after the first `screen` run without a new commit,
  as stated in the Preregistration.

## Review Focus

1. **Lookahead through the execution shift:** a decision at close `t` must never trade at
   open `t` → `test_trade_never_executes_on_decision_day` (Task 3).
2. **Equality and NaN windows:** `close == sma` must be flat, and a missing day must hold
   state rather than fill → `test_equality_is_flat`, `test_missing_day_holds_state` (Task 2).
3. **Fee arithmetic drift:** a round trip at an unchanged price must lose exactly
   `1-(1-f_exit)/(1+f_entry)` → `test_round_trip_cost_identity` (Task 3).
4. **Bootstrap pairing:** trend and comparator must share sampled blocks. If you resample
   identical series you must get an exactly-zero excess CI →
   `test_identical_series_zero_excess` (Task 4).
5. **Validation failure cannot be averaged away:** dev pass + block fail must be `KILL` →
   `test_block_failure_kills` (Task 5).

---

### Task 1: Preregistration constants + daily data snapshot

**Files:**
- Create: `backend/tools/slow_trend/__init__.py` (empty)
- Create: `backend/tools/slow_trend/prereg.py`
- Create: `backend/tools/slow_trend/daily_bars.py`
- Create: `backend/tests/tools/slow_trend/__init__.py` (empty)
- Test: `backend/tests/tools/slow_trend/test_prereg.py`
- Test: `backend/tests/tools/slow_trend/test_daily_bars.py`
- Modify: `.gitignore`, appending `backend/data/research/`

**Interfaces:**
- Produces: `prereg.*` constants (below);
  `daily_bars.fetch_daily(pid: str, start_ts: int, end_ts: int, getter) -> pd.DataFrame`;
  `daily_bars.to_calendar(df: pd.DataFrame, first: str, last: str) -> pd.DataFrame`
  (index: tz-naive daily `DatetimeIndex`, columns `open, close`, NaN for missing days);
  `daily_bars.validate(df_raw: pd.DataFrame, first: str, last: str) -> dict` (keys
  `missing_days: list[str]`, `duplicates: int`, `misaligned: int`, `adequate: bool`);
  `daily_bars.hourly_overlap(hourly: pd.DataFrame, daily_cal: pd.DataFrame) -> dict`.

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

import pandas as pd
import pytest

from tools.slow_trend import daily_bars as D

DAY = 86400
T0 = 1_700_006_400  # 2023-11-15 00:00 UTC


def _candles(starts):
    return [{"start": str(s), "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}
            for s in starts]


def test_fetch_pages_backwards_and_dedupes():
    calls = []

    async def getter(path, params):
        calls.append((int(params["start"]), int(params["end"])))
        s, e = int(params["start"]), int(params["end"])
        return {"candles": _candles(range(s - s % DAY, e, DAY))}

    df = asyncio.run(D.fetch_daily("BTC-USD", T0, T0 + 700 * DAY, getter, page_days=300))
    assert len(df) == 700
    assert df["start"].is_monotonic_increasing and df["start"].is_unique
    assert len(calls) == 3


def test_to_calendar_marks_missing_as_nan():
    raw = pd.DataFrame({"start": [T0, T0 + 2 * DAY], "open": [1.0, 3.0], "close": [1.5, 3.5]})
    cal = D.to_calendar(raw, "2023-11-15", "2023-11-17")
    assert list(cal.index.strftime("%Y-%m-%d")) == ["2023-11-15", "2023-11-16", "2023-11-17"]
    assert cal.loc["2023-11-16"].isna().all()


def test_validate_counts_missing_duplicates_misaligned():
    raw = pd.DataFrame({"start": [T0, T0, T0 + 2 * DAY + 3600],
                        "open": [1.0, 1.0, 2.0], "close": [1.0, 1.0, 2.0]})
    v = D.validate(raw, "2023-11-15", "2023-11-17")
    assert v["duplicates"] == 1 and v["misaligned"] == 1
    assert v["missing_days"] == ["2023-11-16", "2023-11-17"]
    assert v["adequate"] is False


def test_validate_allows_up_to_three_missing():
    starts = [T0 + i * DAY for i in range(10) if i not in (2, 5, 7)]
    raw = pd.DataFrame({"start": starts, "open": 1.0, "close": 1.0})
    v = D.validate(raw, "2023-11-15", "2023-11-24")
    assert len(v["missing_days"]) == 3 and v["adequate"] is True


def test_hourly_overlap_uses_only_complete_days():
    hours = [T0 + h * 3600 for h in range(24 + 5)]  # one complete day + 5 hours
    hourly = pd.DataFrame({"start": hours, "close": [float(h) for h in range(29)]})
    cal = pd.DataFrame({"open": [0.0, 0.0], "close": [23.0, 99.0]},
                       index=pd.to_datetime(["2023-11-15", "2023-11-16"]))
    o = D.hourly_overlap(hourly, cal)
    assert o["days_compared"] == 1 and o["max_abs_rel_diff"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `backend/`, ONLY after the operator OKs pytest while 8001 is live):
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.slow_trend'`

- [ ] **Step 3: Write the implementation**

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

FETCH_FROM = "2015-01-01"
DEV_END = "2025-04-13"
BLOCK_START = "2025-04-14"
BLOCK_END = "2026-10-02"  # last complete UTC day at preregistration
MAX_MISSING_DAYS = 3

DCA_TRANCHES = 52
G2_DRAWDOWN_RATIO = 2 / 3  # arbitrary provisional utility gate, not statistical materiality
MIN_BLOCK_ROUND_TRIPS = 4

BOOT_BLOCK_WEEKS = 8
BOOT_SENS_WEEKS = (4, 13)
BOOT_RESAMPLES = 10_000
BOOT_SEED = 20261003


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
"""Coinbase daily candles: fetch (I/O), calendar alignment and validation (pure)."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

import pandas as pd

DAY = 86400
Getter = Callable[[str, Dict[str, str]], Awaitable[Any]]


async def fetch_daily(pid: str, start_ts: int, end_ts: int, getter: Getter,
                      page_days: int = 300) -> pd.DataFrame:
    """All ONE_DAY candles in [start_ts, end_ts), paging backwards; deduped, oldest first."""
    rows: list[dict] = []
    end = end_ts
    while end > start_ts:
        start = max(start_ts, end - page_days * DAY)
        data = await getter(f"/products/{pid}/candles",
                            {"start": str(start), "end": str(end), "granularity": "ONE_DAY"})
        rows.extend(data.get("candles", []))
        end = start
    df = pd.DataFrame(rows, columns=["start", "open", "high", "low", "close", "volume"])
    df = df.astype({"start": "int64", "open": float, "high": float, "low": float,
                    "close": float, "volume": float})
    df = df[(df["start"] >= start_ts) & (df["start"] < end_ts)]
    return df.drop_duplicates("start").sort_values("start").reset_index(drop=True)


def _days(first: str, last: str) -> pd.DatetimeIndex:
    return pd.date_range(first, last, freq="D")


def to_calendar(df: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    """One row per calendar day in [first, last]; days without a bar are NaN, never filled."""
    idx = pd.to_datetime(df["start"], unit="s")
    out = pd.DataFrame({"open": df["open"].to_numpy(), "close": df["close"].to_numpy()}, index=idx)
    out = out[~out.index.duplicated()]
    return out.reindex(_days(first, last))


def validate(df_raw: pd.DataFrame, first: str, last: str,
             max_missing: int = 3) -> Dict[str, Any]:
    starts = df_raw["start"].astype("int64")
    duplicates = int(starts.duplicated().sum())
    misaligned = int((starts % DAY != 0).sum())
    present = set(pd.to_datetime(starts[starts % DAY == 0], unit="s").strftime("%Y-%m-%d"))
    missing = [d for d in _days(first, last).strftime("%Y-%m-%d") if d not in present]
    adequate = duplicates == 0 and misaligned == 0 and len(missing) <= max_missing
    return {"missing_days": missing, "duplicates": duplicates, "misaligned": misaligned,
            "adequate": adequate}


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
# slow-trend research outputs (snapshots + reports; never committed)
backend/data/research/
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend -v`
Expected: 7 passed.

- [ ] **Step 5: Commit** (operator OK required while 8001 is live; the hook runs the full suite)

```bash
git add .gitignore backend/tools/slow_trend backend/tests/tools/slow_trend
git commit -m "feat(research): slow-trend screen preregistration + daily candle snapshot layer"
git push
```

### Task 2: The frozen rule

**Files:**
- Create: `backend/tools/slow_trend/rule.py`
- Test: `backend/tests/tools/slow_trend/test_rule.py`

**Interfaces:**
- Consumes: calendar frame from `daily_bars.to_calendar`.
- Produces: `rule.decisions(close: pd.Series, n: int) -> pd.Series` (float: 1.0 long,
  0.0 flat, NaN no decision); `rule.desired_state(decisions: pd.Series) -> pd.Series`
  (bool, NaN decisions hold the previous state, initial state flat).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_rule.py
import numpy as np
import pandas as pd

from tools.slow_trend.rule import decisions, desired_state

IDX = pd.date_range("2024-01-01", periods=6, freq="D")


def test_long_above_flat_at_or_below():
    close = pd.Series([1, 2, 3, 2, 2, 5], index=IDX, dtype=float)
    d = decisions(close, 3)
    assert np.isnan(d.iloc[0]) and np.isnan(d.iloc[1])  # warmup
    assert d.iloc[2] == 1.0  # 3 > mean(1,2,3)=2
    assert d.iloc[3] == 0.0  # 2 < mean(2,3,2)


def test_equality_is_flat():
    close = pd.Series([2, 2, 2, 2, 2, 2], index=IDX, dtype=float)
    assert (decisions(close, 3).dropna() == 0.0).all()


def test_missing_day_holds_state():
    close = pd.Series([1, 2, 3, np.nan, 9, 10], index=IDX, dtype=float)
    d = decisions(close, 3)
    assert d.iloc[3:6].isna().all()  # any window touching the gap has no decision
    s = desired_state(d)
    assert s.iloc[2] and s.iloc[3] and s.iloc[5]  # long held through the gap


def test_initial_state_is_flat():
    d = pd.Series([np.nan, np.nan, 0.0], index=IDX[:3])
    assert list(desired_state(d)) == [False, False, False]
```

- [ ] **Step 2: Run to verify failure**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_rule.py -v`
Expected: FAIL with `ModuleNotFoundError ... tools.slow_trend.rule`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/rule.py
"""The frozen rule: long iff completed close > SMA(n) including that close; equality is flat."""

import numpy as np
import pandas as pd


def decisions(close: pd.Series, n: int) -> pd.Series:
    """1.0 long, 0.0 flat, NaN when the n-day window is incomplete or touches a missing day."""
    sma = close.rolling(n, min_periods=n).mean()
    out = pd.Series(np.where(close > sma, 1.0, 0.0), index=close.index)
    return out.where(sma.notna() & close.notna())


def desired_state(dec: pd.Series) -> pd.Series:
    """Hold the previous state through NaN decisions; start flat."""
    return dec.ffill().fillna(0.0).astype(bool)
```

- [ ] **Step 4: Run to verify pass.** Expected: 4 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/rule.py backend/tests/tools/slow_trend/test_rule.py
git commit -m "feat(research): frozen SMA100 rule (equality flat, gaps hold state)"
git push
```

### Task 3: Sleeve simulator (trend, buy-and-hold, DCA)

**Files:**
- Create: `backend/tools/slow_trend/sim.py`
- Test: `backend/tests/tools/slow_trend/test_sim.py`

**Interfaces:**
- Consumes: `prereg.Scenario`; calendar frame (`open`, `close`, NaN for missing days);
  `rule.desired_state`.
- Produces: `sim.Costs(entry_fee, exit_fee, slip)`, a frozen dataclass;
  `sim.Product(base_increment: float, quote_min: float)`;
  `sim.run_sleeve(bars, target: pd.Series, cash: float, costs, product) -> SleeveResult`
  where `target[d]` is the holding state wanted at the OPEN of day `d`;
  `sim.trend_target(state: pd.Series, delay: int) -> pd.Series`;
  `sim.run_dca(bars, cash, tranches, costs, product) -> SleeveResult`;
  `SleeveResult` fields `equity: pd.Series` (marked at close), `terminal_value: float`
  (liquidated at the last close with exit fee + slip), `fees: float`, `entries: int`,
  `exits: int`, `round_trips: int`, `skipped: int`, `exposure: float`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_sim.py
import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

IDX = pd.date_range("2024-01-01", periods=5, freq="D")
FREE = Costs(0.0, 0.0, 0.0)
FINE = Product(base_increment=1e-8, quote_min=1.0)


def _bars(opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)], dtype=float)


def test_trade_never_executes_on_decision_day():
    state = pd.Series([True, True, True, True, True], index=IDX)
    tgt = trend_target(state, delay=0)
    assert not tgt.iloc[0] and tgt.iloc[1]
    assert not trend_target(state, delay=1).iloc[1] and trend_target(state, delay=1).iloc[2]


def test_round_trip_cost_identity():
    c = Costs(0.009, 0.009, 0.0)
    bars = _bars([100] * 4, [100] * 4)
    tgt = pd.Series([True, True, False, False], index=IDX[:4])
    r = run_sleeve(bars, tgt, 500.0, c, FINE)
    assert r.terminal_value == pytest.approx(500.0 * (1 - 0.009) / (1 + 0.009), rel=1e-6)
    assert (r.entries, r.exits, r.round_trips) == (1, 1, 1)


def test_slip_is_adverse_on_both_legs():
    c = Costs(0.0, 0.0, 0.0025)
    bars = _bars([100] * 3, [100] * 3)
    tgt = pd.Series([True, False, False], index=IDX[:3])
    r = run_sleeve(bars, tgt, 500.0, c, FINE)
    assert r.terminal_value == pytest.approx(500.0 * 0.9975 / 1.0025, rel=1e-6)


def test_open_position_is_liquidated_in_terminal_value_but_marked_in_equity():
    c = Costs(0.0, 0.009, 0.0)
    bars = _bars([100, 100], [100, 110])
    r = run_sleeve(bars, pd.Series([True, True], index=IDX[:2]), 500.0, c, FINE)
    assert r.equity.iloc[-1] == pytest.approx(550.0)
    assert r.terminal_value == pytest.approx(550.0 * (1 - 0.009))
    assert r.round_trips == 0


def test_missing_open_defers_execution_to_next_bar():
    bars = _bars([100, np.nan, 100], [100, np.nan, 100])
    tgt = pd.Series([False, True, True], index=IDX[:3])
    r = run_sleeve(bars, tgt, 500.0, FREE, FINE)
    assert r.entries == 1 and r.equity.iloc[1] == pytest.approx(500.0)  # cash marked through gap


def test_below_quote_min_is_skipped():
    r = run_sleeve(_bars([100], [100]), pd.Series([True], index=IDX[:1]), 0.5, FREE, FINE)
    assert r.skipped == 1 and r.entries == 0


def test_units_floor_to_increment():
    coarse = Product(base_increment=1.0, quote_min=1.0)
    r = run_sleeve(_bars([300, 300], [300, 300]),
                   pd.Series([True, True], index=IDX[:2]), 500.0, FREE, coarse)
    assert r.equity.iloc[-1] == pytest.approx(500.0)  # 1 unit @300 + 200 cash


def test_dca_buys_on_mondays_and_keeps_unspent_cash():
    idx = pd.date_range("2024-01-01", periods=15, freq="D")  # 2024-01-01 is a Monday
    bars = pd.DataFrame({"open": 10.0, "close": 10.0}, index=idx)
    r = run_dca(bars, 520.0, 52, FREE, FINE)
    assert r.entries == 3  # Mondays 01, 08, 15
    assert r.equity.iloc[-1] == pytest.approx(520.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_sim.py -v`
Expected: FAIL with `ModuleNotFoundError ... tools.slow_trend.sim`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/sim.py
"""Self-financing single-asset sleeve simulator. Pure: no I/O, no clock.

Execution happens at a day's OPEN (adverse slip applied); equity is marked at each CLOSE
(last valid close carried for MARKING ONLY). Fees are charged on actual notional and reserved
from cash, so a buy never overdraws.
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
    quote_min: float


@dataclass
class SleeveResult:
    equity: pd.Series
    terminal_value: float
    fees: float
    entries: int
    exits: int
    round_trips: int
    skipped: int
    exposure: float


def trend_target(state: pd.Series, delay: int) -> pd.Series:
    """State decided at close t is wanted at the open of t+1+delay; start flat."""
    return state.shift(1 + delay, fill_value=False).astype(bool)


class _Book:
    def __init__(self, cash: float, costs: Costs, product: Product):
        self.cash, self.units, self.costs, self.product = cash, 0.0, costs, product
        self.fees = 0.0
        self.entries = self.exits = self.skipped = 0

    def buy(self, open_px: float, budget: float) -> None:
        px = open_px * (1 + self.costs.slip)
        raw = min(budget, self.cash) / (1 + self.costs.entry_fee) / px
        units = math.floor(raw / self.product.base_increment) * self.product.base_increment
        notional = units * px
        if units <= 0 or notional < self.product.quote_min:
            self.skipped += 1
            return
        fee = notional * self.costs.entry_fee
        self.cash -= notional + fee
        self.units += units
        self.fees += fee
        self.entries += 1

    def sell_all(self, open_px: float) -> None:
        proceeds = self.units * open_px * (1 - self.costs.slip)
        fee = proceeds * self.costs.exit_fee
        self.cash += proceeds - fee
        self.fees += fee
        self.units = 0.0
        self.exits += 1

    def liquidation_value(self, close_px: float) -> float:
        proceeds = self.units * close_px * (1 - self.costs.slip)
        return self.cash + proceeds * (1 - self.costs.exit_fee)


def _finish(book: _Book, bars: pd.DataFrame, marks: list, held_days: int) -> SleeveResult:
    last_close = bars["close"].ffill().iloc[-1]
    return SleeveResult(
        equity=pd.Series(marks, index=bars.index),
        terminal_value=book.liquidation_value(last_close),
        fees=book.fees, entries=book.entries, exits=book.exits,
        round_trips=book.exits, skipped=book.skipped,
        exposure=held_days / len(bars),
    )


def run_sleeve(bars: pd.DataFrame, target: pd.Series, cash: float,
               costs: Costs, product: Product) -> SleeveResult:
    book = _Book(cash, costs, product)
    marks, held, last_close = [], 0, float("nan")
    for day, row in bars.iterrows():
        want = bool(target.loc[day])
        if not math.isnan(row["open"]):
            if want and book.units == 0:
                book.buy(row["open"], book.cash)
            elif not want and book.units > 0:
                book.sell_all(row["open"])
        if not math.isnan(row["close"]):
            last_close = row["close"]
        held += book.units > 0
        mark = book.cash + (book.units * last_close if book.units > 0 else 0.0)
        marks.append(mark)
    return _finish(book, bars, marks, held)


def run_dca(bars: pd.DataFrame, cash: float, tranches: int,
            costs: Costs, product: Product) -> SleeveResult:
    """Buy cash/tranches at the open of each of the first `tranches` Mondays, then hold."""
    book = _Book(cash, costs, product)
    tranche = cash / tranches
    done, marks, held, last_close = 0, [], 0, float("nan")
    pending = False
    for day, row in bars.iterrows():
        if day.weekday() == 0 and done + pending < tranches:
            pending = True
        if pending and not math.isnan(row["open"]):
            book.buy(row["open"], tranche)
            pending, done = False, done + 1
        if not math.isnan(row["close"]):
            last_close = row["close"]
        held += book.units > 0
        marks.append(book.cash + (book.units * last_close if book.units > 0 else 0.0))
    return _finish(book, bars, marks, held)
```

Note for the implementer: `round_trips == exits` holds because every exit closes the only open
position. An open position at the end counts as an entry, not a round trip.

- [ ] **Step 4: Run to verify pass.** Expected: 8 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/sim.py backend/tests/tools/slow_trend/test_sim.py
git commit -m "feat(research): self-financing sleeve simulator for trend, buy-hold and DCA"
git push
```

### Task 4: Metrics and paired block bootstrap

**Files:**
- Create: `backend/tools/slow_trend/metrics.py`
- Test: `backend/tests/tools/slow_trend/test_metrics.py`

**Interfaces:**
- Produces: `metrics.max_drawdown(equity: pd.Series) -> float` (positive fraction);
  `metrics.weekly_returns(equity: pd.Series) -> pd.Series` (W-SUN last, pct_change, dropna);
  `metrics.paired_block_ci(a: pd.Series, b: pd.Series, block: int, n: int, seed: int)
  -> dict` (keys `mean_excess`, `lo`, `hi`, `n_weeks`; 95% percentile CI of mean(a-b)
  where a and b are resampled with the SAME block starts).

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_metrics.py
import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.metrics import max_drawdown, paired_block_ci, weekly_returns


def test_max_drawdown():
    eq = pd.Series([100, 120, 90, 130, 65], dtype=float)
    assert max_drawdown(eq) == pytest.approx(0.5)


def test_weekly_returns_week_ends_sunday():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")  # Mon..Sun x2
    eq = pd.Series(np.r_[np.full(7, 100.0), np.full(7, 110.0)], index=idx)
    w = weekly_returns(eq)
    assert len(w) == 1 and w.iloc[0] == pytest.approx(0.10)


def test_identical_series_zero_excess():
    s = pd.Series(np.random.default_rng(1).normal(0, 0.05, 60))
    ci = paired_block_ci(s, s.copy(), block=8, n=500, seed=7)
    assert ci["lo"] == ci["hi"] == ci["mean_excess"] == 0.0


def test_ci_is_deterministic_for_seed():
    rng = np.random.default_rng(2)
    a, b = pd.Series(rng.normal(0.01, 0.05, 80)), pd.Series(rng.normal(0, 0.05, 80))
    assert paired_block_ci(a, b, 8, 300, 11) == paired_block_ci(a, b, 8, 300, 11)


def test_short_series_reports_no_ci_instead_of_crashing():
    ci = paired_block_ci(pd.Series([0.01] * 5), pd.Series([0.0] * 5), block=13, n=100, seed=0)
    assert ci["lo"] is None and ci["hi"] is None and ci["n_weeks"] == 5


def test_ci_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="aligned"):
        paired_block_ci(pd.Series([0.1, 0.2]), pd.Series([0.1]), 1, 10, 0)
```

- [ ] **Step 2: Run to verify failure**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError ... tools.slow_trend.metrics`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/metrics.py
"""Drawdown, weekly returns and a PAIRED moving-block bootstrap. Exploratory, never a gate."""

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    return float((1 - equity / equity.cummax()).max())


def weekly_returns(equity: pd.Series) -> pd.Series:
    return equity.resample("W-SUN").last().pct_change().dropna()


def paired_block_ci(a: pd.Series, b: pd.Series, block: int, n: int, seed: int) -> dict:
    if len(a) != len(b):
        raise ValueError("paired series must be aligned (equal length)")
    d = (a.to_numpy() - b.to_numpy()).astype(float)
    m = len(d)
    if m < block:  # too short to resample at this block length: report, never crash
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

The excess series is formed BEFORE resampling and the same index is used for both legs, so
the pairing is structural. That is what `test_identical_series_zero_excess` pins.

- [ ] **Step 4: Run to verify pass.** Expected: 6 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/metrics.py backend/tests/tools/slow_trend/test_metrics.py
git commit -m "feat(research): drawdown, weekly returns, paired block bootstrap"
git push
```

### Task 5: Gates and verdict

**Files:**
- Create: `backend/tools/slow_trend/gates.py`
- Test: `backend/tests/tools/slow_trend/test_gates.py`

**Interfaces:**
- Consumes: `prereg` constants.
- Produces: `gates.passes(trend: dict, bh: dict, initial: float, ratio: float) -> dict`
  where `trend`/`bh` hold `terminal_value`, `net_return`, `max_drawdown`; returns
  `{"G1": bool, "G2": bool, "pass": bool}`;
  `gates.verdict(data_ok: bool, results: dict, block_round_trips: int) -> dict` where
  `results[period][scenario] = passes(...)` output, periods `"dev"` and `"block"`; returns
  `{"verdict": str, "reason": str}`.

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/tools/slow_trend/test_gates.py
from tools.slow_trend.gates import passes, verdict

OK = {"G1": True, "G2": True, "pass": True}
BAD = {"G1": False, "G2": True, "pass": False}


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
    lower_ret_low_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.39}
    lower_ret_high_dd = {"terminal_value": 1100.0, "net_return": 0.1, "max_drawdown": 0.41}
    assert passes(lower_ret_low_dd, bh, 1000.0, 2 / 3)["G2"] is True
    assert passes(lower_ret_high_dd, bh, 1000.0, 2 / 3)["G2"] is False


def test_data_inadequate_first():
    assert verdict(False, _res(dev=BAD), 10) == {"verdict": "INCONCLUSIVE", "reason": "data"}


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

- [ ] **Step 2: Run to verify failure**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_gates.py -v`
Expected: FAIL with `ModuleNotFoundError ... tools.slow_trend.gates`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/gates.py
"""Preregistered gates and verdict order. KILL is abandonment of THIS candidate, not a
general falsification of trend following."""

from tools.slow_trend import prereg as P


def passes(trend: dict, bh: dict, initial: float, ratio: float) -> dict:
    g1 = trend["terminal_value"] > initial
    g2 = (trend["net_return"] >= bh["net_return"]
          or trend["max_drawdown"] <= ratio * bh["max_drawdown"])
    return {"G1": bool(g1), "G2": bool(g2), "pass": bool(g1 and g2)}


def verdict(data_ok: bool, results: dict, block_round_trips: int) -> dict:
    if not data_ok:
        return {"verdict": "INCONCLUSIVE", "reason": "data"}
    periods = ("dev", "block")
    if not all(results[p]["P0"]["pass"] for p in periods):
        return {"verdict": "KILL", "reason": "primary_failed"}
    if not all(results[p][s]["pass"] for p in periods for s in P.GATING_SENSITIVITIES):
        return {"verdict": "INCONCLUSIVE", "reason": "fragile"}
    if block_round_trips < P.MIN_BLOCK_ROUND_TRIPS:
        return {"verdict": "INCONCLUSIVE", "reason": "insufficient_transitions"}
    return {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
```

- [ ] **Step 4: Run to verify pass.** Expected: 8 passed.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/gates.py backend/tests/tools/slow_trend/test_gates.py
git commit -m "feat(research): preregistered gates and verdict order"
git push
```

### Task 6: Runner (fetch → validate → run grid → report)

**Files:**
- Create: `backend/tools/slow_trend/screen.py`
- Test: `backend/tests/tools/slow_trend/test_screen.py`

**Interfaces:**
- Consumes: everything above; `clients.coinbase_client._get` and `get_product` for real runs.
- Produces: `screen.run_period(cal: dict[str, pd.DataFrame], start: str, end: str,
  products: dict[str, Product]) -> dict` (per-scenario portfolio metrics + `passes`, the
  comparators, bootstrap CIs, and per-sleeve contribution);
  CLI `python -m tools.slow_trend.screen fetch` (writes snapshot + manifest) and
  `python -m tools.slow_trend.screen run` (refuses a dirty tree; writes
  `report_<UTC timestamp>.json`; prints the verdict).

- [ ] **Step 1: Write the failing tests** (synthetic data; no network)

```python
# backend/tests/tools/slow_trend/test_screen.py
import numpy as np
import pandas as pd

from tools.slow_trend.screen import run_period
from tools.slow_trend.sim import Product

FINE = Product(1e-8, 1.0)


def _cal(prices, start="2020-01-01"):
    idx = pd.date_range(start, periods=len(prices), freq="D")
    p = pd.Series(prices, index=idx, dtype=float)
    return pd.DataFrame({"open": p.shift(1).fillna(p.iloc[0]), "close": p})


def test_period_runs_all_scenarios_and_comparators():
    up = np.linspace(100, 300, 400)
    cal = {"BTC-USD": _cal(up), "ETH-USD": _cal(up)}
    out = run_period(cal, "2020-05-01", "2021-01-31", {"BTC-USD": FINE, "ETH-USD": FINE})
    assert set(out["scenarios"]) == {"P0", "S10", "S25", "D1", "SM"}
    p0 = out["scenarios"]["P0"]
    assert {"trend", "buy_hold", "dca52", "cash", "passes", "ci_vs_bh"} <= set(p0)
    assert p0["cash"]["terminal_value"] == 1000.0
    assert p0["trend"]["round_trips"] == 0  # monotone rise: one entry, never exits


def test_warmup_uses_pre_period_history():
    up = np.linspace(100, 300, 400)
    cal = {"BTC-USD": _cal(up), "ETH-USD": _cal(up)}
    out = run_period(cal, "2020-05-01", "2020-06-30", {"BTC-USD": FINE, "ETH-USD": FINE})
    # SMA is already defined on the first period day, so the trend enters on day 2
    assert out["scenarios"]["P0"]["trend"]["entries"] == 2  # one per sleeve
```

- [ ] **Step 2: Run to verify failure**

Run: `../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend/test_screen.py -v`
Expected: FAIL with `ModuleNotFoundError ... tools.slow_trend.screen`

- [ ] **Step 3: Implement**

```python
# backend/tools/slow_trend/screen.py
"""Runner for the preregistered slow-trend screen. See the plan's Preregistration section.

  python -m tools.slow_trend.screen fetch   # snapshot daily candles + manifest (network)
  python -m tools.slow_trend.screen run     # frozen grid -> report_<ts>.json + verdict
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from tools.slow_trend import daily_bars as D
from tools.slow_trend import gates as G
from tools.slow_trend import metrics as M
from tools.slow_trend import prereg as P
from tools.slow_trend.rule import decisions, desired_state
from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve, trend_target

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "data" / "research" / "slow_trend"


def _summ(results: list, initial: float) -> dict:
    eq = sum(r.equity for r in results)
    tv = sum(r.terminal_value for r in results)
    return {"terminal_value": tv, "net_return": tv / initial - 1,
            "max_drawdown": M.max_drawdown(eq), "fees": sum(r.fees for r in results),
            "entries": sum(r.entries for r in results),
            "round_trips": sum(r.round_trips for r in results),
            "skipped": sum(r.skipped for r in results),
            "exposure": [r.exposure for r in results],
            "per_sleeve_terminal": [r.terminal_value for r in results],
            "_equity": eq}


def run_period(cal: dict, start: str, end: str, products: dict) -> dict:
    out = {"start": start, "end": end, "scenarios": {}}
    for sc in P.SCENARIOS:
        costs = Costs(sc.entry_fee, sc.exit_fee, sc.slip)
        legs = {"trend": [], "buy_hold": [], "dca52": []}
        for pid in P.PRODUCTS:
            full = cal[pid]
            state = desired_state(decisions(full["close"], P.SMA_LEN)).loc[:end]
            bars = full.loc[start:end]
            tgt = trend_target(state.loc[start:], sc.delay)  # starts flat in-period
            legs["trend"].append(run_sleeve(bars, tgt, P.SLEEVE_USD, costs, products[pid]))
            hold = pd.Series(True, index=bars.index)
            legs["buy_hold"].append(run_sleeve(bars, hold, P.SLEEVE_USD, costs, products[pid]))
            legs["dca52"].append(run_dca(bars, P.SLEEVE_USD, P.DCA_TRANCHES, costs,
                                         products[pid]))
        s = {k: _summ(v, P.INITIAL_USD) for k, v in legs.items()}
        s["cash"] = {"terminal_value": P.INITIAL_USD, "net_return": 0.0, "max_drawdown": 0.0}
        s["passes"] = G.passes(s["trend"], s["buy_hold"], P.INITIAL_USD, P.G2_DRAWDOWN_RATIO)
        wt = M.weekly_returns(s["trend"]["_equity"])
        for name in ("buy_hold", "dca52"):
            wc = M.weekly_returns(s[name]["_equity"]).reindex(wt.index)
            s[f"ci_vs_{'bh' if name == 'buy_hold' else name}"] = {
                str(b): M.paired_block_ci(wt, wc, b, P.BOOT_RESAMPLES, P.BOOT_SEED)
                for b in (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)}
        s["ci_vs_cash"] = {str(b): M.paired_block_ci(wt, wt * 0, b, P.BOOT_RESAMPLES,
                                                     P.BOOT_SEED)
                           for b in (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)}
        for k in ("trend", "buy_hold", "dca52"):
            s[k].pop("_equity")
        out["scenarios"][sc.name] = s
    return out


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


async def _fetch() -> None:
    from clients import coinbase_client as cc

    OUT.mkdir(parents=True, exist_ok=True)
    start = int(pd.Timestamp(P.FETCH_FROM).timestamp())
    end = int(pd.Timestamp(P.BLOCK_END).timestamp()) + D.DAY
    manifest = {"fetched_at": datetime.now(timezone.utc).isoformat(), "products": {}}
    for pid in P.PRODUCTS:
        df = await D.fetch_daily(pid, start, end, cc._get)
        path = OUT / f"{pid}.parquet"
        df.to_parquet(path, index=False)
        prod = await cc.get_product(pid) or {}
        manifest["products"][pid] = {
            "sha256": _sha(path), "rows": len(df),
            "first": str(pd.to_datetime(df["start"].min(), unit="s").date()),
            "base_increment": float(prod["base_increment"]),
            "quote_min": float(prod.get("quote_min_size") or 1.0)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


def _dirty() -> bool:
    r = subprocess.run(["git", "status", "--porcelain", str(Path(__file__).parent)],
                       capture_output=True, text=True, check=True)
    return bool(r.stdout.strip())


def _run() -> None:
    if _dirty():
        sys.exit("refusing to run: backend/tools/slow_trend has uncommitted changes")
    manifest = json.loads((OUT / "manifest.json").read_text())
    raw, cal, products, data = {}, {}, {}, {}
    common_first = max(m["first"] for m in manifest["products"].values())
    for pid, m in manifest["products"].items():
        path = OUT / f"{pid}.parquet"
        if _sha(path) != m["sha256"]:
            sys.exit(f"snapshot {pid} does not match its manifest digest")
        raw[pid] = pd.read_parquet(path)
        cal[pid] = D.to_calendar(raw[pid], common_first, P.BLOCK_END)
        products[pid] = Product(m["base_increment"], m["quote_min"])
    dev_start = str((pd.Timestamp(common_first) + pd.Timedelta(days=P.SMA_LEN - 1)).date())
    periods = {"dev": (dev_start, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    for name, (a, b) in periods.items():
        data[name] = {pid: D.validate(raw[pid][(raw[pid]["start"] >= pd.Timestamp(a).timestamp())
                                               & (raw[pid]["start"] < pd.Timestamp(b).timestamp()
                                                  + D.DAY)], a, b, P.MAX_MISSING_DAYS)
                      for pid in P.PRODUCTS}
    data_ok = all(v["adequate"] for p in data.values() for v in p.values())
    overlap = {}
    for pid in P.PRODUCTS:
        hp = BACKEND / "data" / "history" / f"{pid}.parquet"
        if hp.exists():
            overlap[pid] = D.hourly_overlap(pd.read_parquet(hp), cal[pid])
    res = {name: run_period(cal, a, b, products) for name, (a, b) in periods.items()}
    passes = {n: {s: r["scenarios"][s]["passes"] for s in r["scenarios"]} for n, r in res.items()}
    v = G.verdict(data_ok, passes, res["block"]["scenarios"]["P0"]["trend"]["round_trips"])
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    report = {"verdict": v, "head": head, "prereg_sha256": _sha(Path(P.__file__)),
              "manifest_sha256": _sha(OUT / "manifest.json"), "periods": periods,
              "data_checks": data, "hourly_overlap_informational": overlap, "results": res}
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (OUT / f"report_{ts}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": v, "report": f"report_{ts}.json"}, indent=2))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "fetch":
        from dotenv import load_dotenv

        load_dotenv(BACKEND.parent / ".env")
        asyncio.run(_fetch())
    elif cmd == "run":
        _run()
    else:
        sys.exit("usage: python -m tools.slow_trend.screen {fetch|run}")
```

- [ ] **Step 4: Run to verify pass.** Expected: 2 passed. Then run the whole package:
`../.venv/Scripts/python.exe -m pytest tests/tools/slow_trend -v`, expected 35 passed. Then
run `ruff090 check backend/tools/slow_trend backend/tests/tools/slow_trend` and
`ruff090 format --check` on the same paths; both must be clean.

- [ ] **Step 5: Commit** (operator OK required)

```bash
git add backend/tools/slow_trend/screen.py backend/tests/tools/slow_trend/test_screen.py
git commit -m "feat(research): slow-trend screen runner (fetch, frozen grid, report)"
git push
```

### Task 7: Review gate, then the single run

- [ ] **Step 1:** Request Codex review of the full branch via session link. Fix findings,
  as the TDD steps above direct.
- [ ] **Step 2:** `cd backend && ../.venv/Scripts/python.exe -m tools.slow_trend.screen fetch`.
  Record the manifest (first dates, row counts, increments) in the PR description BEFORE `run`.
- [ ] **Step 3:** With a clean tree, run
  `../.venv/Scripts/python.exe -m tools.slow_trend.screen run` exactly ONCE. Never rerun
  after a code or prereg change without declaring a new preregistration.
- [ ] **Step 4:** Report the verdict to the operator with: the verdict and reason; P0 dev and
  block tables against all comparators; the sensitivity outcomes; and the data checks. State
  plainly that `PASS_TO_FORWARD` does not authorise funding and `KILL` abandons this candidate
  only. Log it to CHANGELOG, memory, and the win-factors ledger.
