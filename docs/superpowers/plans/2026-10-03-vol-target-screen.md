# Volatility-Target Screen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A one-shot, preregistered abandonment screen for volatility-targeted BTC/ETH holding,
run once on the SAME locked daily snapshot as the SMA100 screen. It yields a KILL / INCONCLUSIVE
/ PASS_TO_FORWARD verdict.

**Architecture:** A new package, `backend/tools/vol_target/`, stacked on the frozen
`tools/slow_trend` package. It reuses that package's bars, audits, metrics, gates, comparators
(buy-and-hold, DCA-52) and freeze mechanics. New pieces:
- a frozen preregistration;
- a weekly volatility signal;
- a self-financing fractional-weight sleeve simulator;
- a verdict wrapper whose coverage gate replaces the binary round-trip gate;
- its own OUT, lock and run ledger.

The locked snapshot is IMPORTED (copied and verified against the slow-trend lock), never
refetched.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No network at run time.

**Spec:** the preregistration is two session-link messages, archived verbatim in
`C:\Users\gl450\analysis_archive\vol_target_prereg_2026-10-03\`:
- `c4c6827b`: Claude's proposal;
- `52c7e2ee`: Codex's amendments.

Where they differ, the amendment governs. The binding text is restated in **Preregistration**
below.

## Preregistration (frozen; every value lives in `vol_target/prereg.py`)

### Data and capital

**Snapshot:** the slow-trend raw daily snapshot. Its manifest digest is
`sha256:5215b520ba2d25b78897cbe0b5c3024547f10b353454e3cc3a89f09f0b71320c`; it is copied, never
refetched, and the manifest is unchanged.

**Sleeves:**
- two independent USD 500 sleeves, BTC-USD and ETH-USD;
- zero cash yield;
- fees and product rules are current planning assumptions, not reconstructed historical rules.

### Signal

`σ̂_d` = the sample standard deviation (ddof=1) of the 20 daily log close-to-close returns from
the 21 consecutive finite, positive daily closes ending at day `d`, × √365.
- No filled gaps and no compressed window: a missing or invalid close anywhere in the window
  makes `σ̂_d` invalid.
- This is a noisy volatility forecast, not a tail-risk bound.

### Target weight

`w = min(1, 0.50 / σ̂)`, the same for each asset sleeve.
- `σ̂ == 0` → `w = 1`.
- A non-finite or invalid `σ̂` → no order. The sleeve holds its UNITS and CASH, and its marked
  weight drifts.
- 50% is a discretionary risk budget frozen before evaluation. It is not an optimum, and no
  40/60 variants are run.
- Sizing each asset independently does NOT target 50% volatility for the combined portfolio.

### Schedule

**Decisions:**
- One decision per week, on the completed Sunday-labelled UTC daily bar.
- Eligibility: `|w_target − w_held| > 0.10`, strictly and in absolute terms. `w_held` is
  `units × Sunday close / (cash + units × Sunday close)`, per sleeve.

**Execution:**
- P0 executes at Monday's open; D1 at Tuesday's open.
- The order is a TARGET-WEIGHT instruction frozen at Sunday: no newer closes enter it, and there
  is no second deadband test at the open.
- Quantities are solved at the execution open from the then-current cash and units, fees and the
  adverse execution price.
- A missing execution open, or a rebalance rejected by size rules, is counted and expires: there
  is no daily retry.

### Periods

**Dates:**
- dev is 2016-08-30..2025-04-13; block is 2025-04-14..2026-10-02.
- Features use the preceding history. Each sleeve resets to cash at each period start.

**Which decisions count:** a scheduled decision counts when its execution day lies in
[start, end] and its decision Sunday is ≥ start − 1 day.

| Period | Scenario | First execution |
|---|---|---|
| dev | P0 | Mon 2016-09-05 |
| dev | D1 | Tue 2016-09-06 |
| block | P0 | Mon 2025-04-14 (initialised from Sunday 2025-04-13) |
| block | D1 | Tue 2025-04-15 |

The calendar-phase difference from day-1 buy-and-hold is part of the frozen policy.

### Accounting

**The simulator:**
- Self-financing: fees reduce cash, and only the CHANGE in units is charged.
- Target holdings are solved with fee-inclusive cash, and the target weight is defined on
  OPEN-price marks.
- Quantities are rounded DOWN to `base_increment` and must then satisfy `base_min` and
  `quote_min`. It can never overspend or oversell.
- A skipped order leaves holdings intact. Weights are recomputed from EXECUTED holdings.

**Terminal liquidation** happens at the last close with slip and the exit fee. It validates the
full size contract (`units ≥ base_min` AND `proceeds ≥ quote_min`); an unsellable residual is
marked `unliquidatable`, and no cash is invented.

**Scenarios:** P0, S10, S25, D1 and SM, as in slow-trend.
- SM is a maker-buy / taker-sell, fee-only, optimistic diagnostic. It never gates, and
  immediate next-open execution does not establish maker fills.

### Gates and verdict

The gates and verdict order are unchanged from slow-trend.
- **G1:** terminal value > cash.
- **G2:** net return ≥ buy-and-hold OR max drawdown ≤ 2/3 × buy-and-hold.
- Both must hold in BOTH periods under P0 and the gating sensitivities S10, S25 and D1.
- The `affected` / `terminal_size` precedence is unchanged.

The ONLY change replaces the binary `MIN_BLOCK_ROUND_TRIPS` (a partial reduction is not a round
trip). PASS additionally requires **≥ 4 valid scheduled weekly decisions PER sleeve in the block
under P0**, excluding terminal liquidation.
- This is an administrative coverage gate, not a power threshold.
- It is implemented in a vol-target verdict wrapper, not passed through `block_round_trips`.

### Diagnostics (never gates)

- exposure, as the time-average marked asset fraction;
- realised volatility per sleeve and for the portfolio;
- requested and executed weights, and the fraction of decisions capped at 1;
- valid and suppressed decisions;
- eligible, executed and size-skipped rebalances, and missed opens;
- buy and sell counts, fees, turnover, and stale-mark days.

**A NON-GATING fixed 50%-asset / 50%-cash diagnostic** uses the same weekly / deadband /
execution convention. It separates volatility timing from simply holding less crypto. It is
never tuned, and never gates.

A lower drawdown is utility at the policy's actual exposure, not timing alpha.

### Status

**The block is not a clean holdout.** It is a previously observed retrospective evaluation,
now reused for a SECOND candidate in this screen family; the ledger records the sequence and the
prior SMA KILL.

**What a verdict can and cannot mean:**
- PASS establishes utility under the assumed simulator only. It does not establish superior
  prediction or funding readiness; a PASS leads only to a forward paper experiment.
- The old SMA verdict and report are never altered.

**Freeze:** the new package AND every reused `slow_trend` dependency, plus
`clients/coinbase_client.py`.

## Global Constraints

- The run is OFFLINE: no network in `evaluate` or `run`. `import` copies local files only.
- One run. `run` refuses a second completed run unless given `--replay`, with the same
  first/retry/replay/new_preregistration modes as slow-trend.
- OUT is `backend/data/research/vol_target/` (gitignored, as for slow-trend). The lock is
  `backend/tools/vol_target/snapshot.lock` and must equal `SOURCE_SNAPSHOT_LOCK`.
- No change to any file under `backend/tools/slow_trend/`.
- Ruff 0.9.0 (pinned) for format and check. TDD per task. `git commit -- <paths>` only.

## Review Focus

1. **Deadband equality:** a weight change of exactly 0.10 must NOT trade (strict `>`).
2. **D1 must execute the SUNDAY target.** It must not re-decide on Monday's close, even when
   Monday's σ̂ would differ.
3. **An overnight gap:** the quantity is solved from Monday's OPEN, so the post-trade weight
   marked at the open equals the target net of fees, not the Sunday-close weight.
4. **A missing Monday open expires the instruction.** It is not retried on Tuesday, and it is
   counted as `missed_open`.
5. **Terminal dust:** units below `base_min` at the end → `unliquidatable` and affected, and
   `terminal_value` excludes the residual.

Each item is pinned by a named test in Task 2.

---

### Task 1: Preregistration and weekly signal

**Files:**
- Create: `backend/tools/vol_target/__init__.py` (empty), `backend/tools/vol_target/prereg.py`,
  `backend/tools/vol_target/signal.py`
- Test: `backend/tests/tools/vol_target/__init__.py` (empty),
  `backend/tests/tools/vol_target/test_signal.py`

**Interfaces:**
- Produces:
  - `annualised_sigma(close: pd.Series) -> pd.Series`;
  - `target_weight(sigma: pd.Series) -> pd.Series`;
  - `schedule(close: pd.Series, start: str, end: str, delay: int) -> pd.DataFrame`, indexed by
    decision Sunday with columns `execute` (Timestamp) and `target` (float, NaN when invalid).

- [ ] **Step 1: Write the failing tests** (`test_signal.py`)

```python
import math

import numpy as np
import pandas as pd
import pytest

from tools.vol_target import prereg as P
from tools.vol_target.signal import annualised_sigma, schedule, target_weight

IDX = pd.date_range("2024-01-01", periods=60, freq="D")  # 2024-01-01 is a Monday


def _close(vals=None):
    vals = vals if vals is not None else 100 * np.exp(0.01 * np.sin(np.arange(60)))
    return pd.Series(vals, index=IDX[: len(vals)], dtype=float)


def test_sigma_is_sample_std_of_20_log_returns_annualised():
    c = _close()
    r = np.log(c.to_numpy())[1:] - np.log(c.to_numpy())[:-1]
    expected = np.std(r[-20:], ddof=1) * math.sqrt(365)
    assert annualised_sigma(c).iloc[-1] == pytest.approx(expected, rel=1e-12)


def test_sigma_needs_21_consecutive_valid_closes():
    s = annualised_sigma(_close())
    assert s.iloc[:20].isna().all() and np.isfinite(s.iloc[20])


@pytest.mark.parametrize("bad", [np.nan, 0.0, -1.0])
def test_any_invalid_close_in_the_window_invalidates_sigma(bad):
    v = _close().to_numpy().copy()
    v[40] = bad
    s = annualised_sigma(_close(v))
    assert s.iloc[40:61].isna().all()  # every window containing day 40 (through day 60)
    assert np.isfinite(s.iloc[39])


def test_target_weight_caps_at_one_and_scales_down():
    w = target_weight(pd.Series([0.25, 0.5, 1.0, 2.0]))
    assert list(w) == [1.0, 1.0, 0.5, 0.25]


def test_zero_sigma_is_full_weight_and_invalid_sigma_is_no_target():
    w = target_weight(pd.Series([0.0, np.nan, np.inf, -0.1]))
    assert w.iloc[0] == 1.0 and w.iloc[1:].isna().all()


def test_schedule_tuesday_start_waits_for_first_monday():
    c = _close()
    sch = schedule(c, "2024-01-30", "2024-02-28", delay=0)  # 2024-01-30 is a Tuesday
    assert sch["execute"].iloc[0] == pd.Timestamp("2024-02-05")  # Monday
    assert sch.index[0] == pd.Timestamp("2024-02-04")  # its Sunday decision


def test_schedule_monday_start_uses_the_preceding_sunday():
    sch = schedule(_close(), "2024-01-29", "2024-02-28", delay=0)  # Monday start
    assert sch.index[0] == pd.Timestamp("2024-01-28")
    assert sch["execute"].iloc[0] == pd.Timestamp("2024-01-29")


def test_schedule_d1_executes_tuesday_with_the_sunday_target():
    c = _close()
    p0 = schedule(c, "2024-01-29", "2024-02-28", delay=0)
    d1 = schedule(c, "2024-01-29", "2024-02-28", delay=1)
    assert d1["execute"].iloc[0] == pd.Timestamp("2024-01-30")
    assert d1["target"].iloc[0] == p0["target"].iloc[0]  # frozen at Sunday, not re-decided


def test_schedule_excludes_executions_after_the_period_end():
    sch = schedule(_close(), "2024-01-29", "2024-02-25", delay=0)  # ends on a Sunday
    assert sch["execute"].max() <= pd.Timestamp("2024-02-25")


def test_prereg_freezes_the_agreed_values():
    assert (P.VOL_RETURNS, P.ANNUALISATION, P.SIGMA_TARGET, P.DEADBAND) == (20, 365, 0.5, 0.10)
    assert (P.DEV_START, P.DEV_END, P.BLOCK_START, P.BLOCK_END) == (
        "2016-08-30", "2025-04-13", "2025-04-14", "2026-10-02",
    )
    assert P.MIN_BLOCK_VALID_DECISIONS == 4
    assert P.SOURCE_SNAPSHOT_LOCK.startswith("sha256:5215b520ba2d25b7")
```

- [ ] **Step 2: Run them.** Expected: FAIL, `ModuleNotFoundError: tools.vol_target`.

Run: `cd backend && ../.venv/Scripts/python.exe -m pytest tests/tools/vol_target/test_signal.py -q`

- [ ] **Step 3: Implement** `prereg.py`

```python
"""Frozen preregistration for the BTC/ETH volatility-target screen (2026-10-03).

Every value was fixed BEFORE any rule ran on any data, in the Claude/Codex debate archived at
C:\\Users\\gl450\\analysis_archive\\vol_target_prereg_2026-10-03 (c4c6827b + amendments 52c7e2ee).
Shared values are imported from the frozen slow_trend preregistration so the two screens cannot
drift apart. Changing a value is a new preregistration, never a revision of a verdict.
"""

from tools.slow_trend import prereg as S

PRODUCTS = S.PRODUCTS
INITIAL_USD = S.INITIAL_USD
SLEEVE_USD = S.SLEEVE_USD
SCENARIOS = S.SCENARIOS
GATING_SENSITIVITIES = S.GATING_SENSITIVITIES
G2_DRAWDOWN_RATIO = S.G2_DRAWDOWN_RATIO
DCA_TRANCHES = S.DCA_TRANCHES
MAX_MISSING_DAYS = S.MAX_MISSING_DAYS
BOOT_BLOCK_WEEKS, BOOT_SENS_WEEKS = S.BOOT_BLOCK_WEEKS, S.BOOT_SENS_WEEKS
BOOT_RESAMPLES, BOOT_SEED = S.BOOT_RESAMPLES, S.BOOT_SEED

DEV_START = "2016-08-30"  # the SMA screen's evaluated dev start, frozen for comparability
DEV_END, BLOCK_START, BLOCK_END = S.DEV_END, S.BLOCK_START, S.BLOCK_END

VOL_RETURNS = 20  # log returns -> 21 consecutive valid daily closes
ANNUALISATION = 365
SIGMA_TARGET = 0.50  # per asset sleeve; discretionary risk budget, not an optimum
DEADBAND = 0.10  # absolute weight; trade only when |target - held| > DEADBAND (strict)
DECISION_WEEKDAY = 6  # Sunday-labelled completed UTC daily bar
FIXED_DIAG_WEIGHT = 0.5  # NON-GATING diagnostic only
MIN_BLOCK_VALID_DECISIONS = 4  # per sleeve, P0, block; administrative coverage, not power

SOURCE_SNAPSHOT_LOCK = "sha256:5215b520ba2d25b78897cbe0b5c3024547f10b353454e3cc3a89f09f0b71320c"
FREEZE_PATHS = (
    "backend/tools/vol_target",
    "backend/tools/slow_trend",
    "backend/clients/coinbase_client.py",
)
```

`signal.py`:

```python
"""Weekly volatility-target signal. Pure: no I/O, no clock."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from tools.vol_target import prereg as P


def annualised_sigma(close: pd.Series) -> pd.Series:
    """`close` must be on a full daily calendar (missing days are NaN rows). A window with any
    missing, non-finite or non-positive close yields NaN: no filled or compressed windows."""
    c = close.astype(float).where(close > 0)
    r = np.log(c).diff()
    n = P.VOL_RETURNS
    return r.rolling(n, min_periods=n).std(ddof=1) * math.sqrt(P.ANNUALISATION)


def target_weight(sigma: pd.Series) -> pd.Series:
    s = sigma.astype(float)
    w = pd.Series(np.nan, index=s.index)
    finite = np.isfinite(s)
    w[finite & (s == 0)] = 1.0
    pos = finite & (s > 0)
    w[pos] = np.minimum(1.0, P.SIGMA_TARGET / s[pos])
    return w


def schedule(close: pd.Series, start: str, end: str, delay: int) -> pd.DataFrame:
    """One row per scheduled weekly decision whose execution day (Sunday + 1 + delay) lies in
    [start, end] and whose Sunday is >= start - 1 day. Uses only history up to that Sunday."""
    w = target_weight(annualised_sigma(close))
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for d in close.index[close.index.weekday == P.DECISION_WEEKDAY]:
        ex = d + pd.Timedelta(days=1 + delay)
        if d >= lo - pd.Timedelta(days=1) and lo <= ex <= hi:
            rows.append((d, ex, float(w.loc[d])))
    out = pd.DataFrame(rows, columns=["decision", "execute", "target"])
    return out.set_index("decision")
```

- [ ] **Step 4: Run the tests.** Expected: PASS, 10/10.
- [ ] **Step 5: Commit**: `git commit -m "feat(research): vol-target prereg + weekly signal" -- backend/tools/vol_target backend/tests/tools/vol_target`

### Task 2: Self-financing fractional-weight sleeve simulator

**Files:**
- Create: `backend/tools/vol_target/fsim.py`
- Test: `backend/tests/tools/vol_target/test_fsim.py`

**Interfaces:**
- Consumes: `schedule(...)` DataFrame (Task 1); `tools.slow_trend.sim.Costs`, `Product`.
- Produces:
  - `solve_units(cash, units, open_px, w, costs) -> float`;
  - `run_weight_sleeve(bars, sched, cash, costs, product, deadband) -> WeightResult`. The
    dataclass carries:
    - from slow-trend: `equity`, `initial`, `terminal_value`, `exec_fees`, `terminal_fee`,
      `traded_notional`, `slippage_cost`, `unliquidatable`, `residual_units`,
      `residual_marked_value`;
    - new: `buys`, `sells`, `size_skipped`, `missed_open`, `within_deadband`,
      `invalid_decisions`, `valid_decisions`, `executed`, `requested`, `executed_weights`,
      `capped`, `exposure`;
    - from slow-trend: `stale_mark_days`, `max_stale_run`.

- [ ] **Step 1: Write the failing tests** (`test_fsim.py`)

```python
import math

import numpy as np
import pandas as pd
import pytest

from tools.slow_trend.sim import Costs, Product
from tools.vol_target.fsim import run_weight_sleeve, solve_units

FREE = Costs(0.0, 0.0, 0.0)
TAKER = Costs(0.009, 0.009, 0.0)
FINE = Product(base_increment=1e-8, base_min=1e-8, quote_min=1.0)
IDX = pd.date_range("2024-01-07", periods=10, freq="D")  # Sunday 01-07 .. Tuesday 01-16


def _bars(opens, closes):
    return pd.DataFrame({"open": opens, "close": closes}, index=IDX[: len(opens)], dtype=float)


def _sched(*rows):  # (decision_day_index, execute_day_index, target)
    return pd.DataFrame(
        [(IDX[d], IDX[e], t) for d, e, t in rows], columns=["decision", "execute", "target"]
    ).set_index("decision")


def _marked_weight(cash, units, px):
    return units * px / (cash + units * px)


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_solve_units_hits_target_weight_net_of_fees_at_the_open(w):
    cash, units, px = 300.0, 2.0, 100.0
    u = solve_units(cash, units, px, w, TAKER)
    du = u - units
    fee = abs(du) * px * 0.009
    new_cash = cash - du * px - fee
    assert new_cash >= -1e-9
    assert _marked_weight(new_cash, u, px) == pytest.approx(w, abs=1e-9)


def test_full_weight_spends_all_cash_zero_weight_sells_all():
    assert solve_units(500.0, 0.0, 100.0, 1.0, TAKER) == pytest.approx(500 / (100 * 1.009))
    assert solve_units(0.0, 3.0, 100.0, 0.0, TAKER) == 0.0


def test_deadband_equality_does_not_trade():
    # Sunday decision: held 0 (cash), target exactly 0.10 -> |0.10 - 0| is NOT > 0.10
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 0.10)), 500.0, FREE, FINE, 0.10)
    assert (r.buys, r.within_deadband, r.executed) == (0, 1, 0)


def test_just_above_deadband_trades():
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 0.1001)), 500.0, FREE, FINE, 0.10)
    assert r.buys == 1 and r.executed_weights[0] == pytest.approx(0.1001, abs=1e-6)


def test_overnight_gap_solves_quantity_at_the_monday_open():
    # Sunday close 100, Monday open 200: target 0.5 must be met at the 200 open
    r = run_weight_sleeve(_bars([100, 200, 200], [100, 200, 200]), _sched((0, 1, 0.5)), 500.0, FREE, FINE, 0.10)
    assert r.executed_weights[0] == pytest.approx(0.5, abs=1e-6)
    assert r.equity.iloc[1] == pytest.approx(500.0, rel=1e-6)  # no gain: bought at 200


def test_d1_executes_the_frozen_sunday_target_on_tuesday():
    r = run_weight_sleeve(_bars([100] * 4, [100] * 4), _sched((0, 2, 0.6)), 500.0, FREE, FINE, 0.10)
    assert r.equity.index[2] == IDX[2] and r.buys == 1
    assert r.requested == [0.6] and r.executed_weights[0] == pytest.approx(0.6, abs=1e-6)


def test_missing_execution_open_expires_without_retry():
    r = run_weight_sleeve(_bars([100, np.nan, 100, 100], [100] * 4), _sched((0, 1, 0.8)), 500.0, FREE, FINE, 0.10)
    assert (r.missed_open, r.buys) == (1, 0)
    assert r.terminal_value == pytest.approx(500.0)


def test_invalid_decision_holds_units_and_cash():
    s = _sched((0, 1, 0.8), (7, 8, np.nan))
    r = run_weight_sleeve(_bars([100] * 10, [100] * 7 + [150] * 3), s, 500.0, FREE, FINE, 0.10)
    assert (r.buys, r.sells, r.invalid_decisions, r.valid_decisions) == (1, 0, 1, 1)


def test_size_rejected_rebalance_is_skipped_and_holdings_unchanged():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 0.15)), 500.0, FREE, coarse, 0.10)
    assert (r.size_skipped, r.buys) == (1, 0)  # 0.75 units rounds down to 0
    assert r.terminal_value == pytest.approx(500.0)


def test_never_overspends_with_fees_and_rounding():
    coarse = Product(base_increment=0.01, base_min=0.01, quote_min=1.0)
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 1.0)), 500.0, TAKER, coarse, 0.10)
    assert r.equity.min() > 0 and r.exec_fees > 0
    assert r.executed_weights[0] <= 1.0


def test_weights_are_recomputed_from_executed_rounded_holdings():
    coarse = Product(base_increment=1.0, base_min=1.0, quote_min=1.0)
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 0.55)), 1000.0, FREE, coarse, 0.10)
    assert r.requested == [0.55] and r.executed_weights == [pytest.approx(0.5)]


def test_terminal_dust_is_unliquidatable_and_never_cash():
    tiny = Product(base_increment=1e-8, base_min=1e-8, quote_min=50.0)
    r = run_weight_sleeve(_bars([100] * 3, [100] * 3), _sched((0, 1, 0.12)), 500.0, FREE, tiny, 0.10)
    # 0.12*500 = 60 bought (>= quote_min); price unchanged so proceeds 60 >= 50 -> liquidatable
    assert not r.unliquidatable
    crash = run_weight_sleeve(_bars([100, 100, 10], [100, 100, 10]), _sched((0, 1, 0.12)), 500.0, FREE, tiny, 0.10)
    assert crash.unliquidatable and crash.residual_units > 0
    assert crash.terminal_value == pytest.approx(500.0 - 60.0)  # residual excluded


def test_exposure_is_time_average_marked_asset_fraction():
    r = run_weight_sleeve(_bars([100] * 4, [100] * 4), _sched((0, 1, 0.5)), 500.0, FREE, FINE, 0.10)
    assert r.exposure == pytest.approx((0 + 0.5 + 0.5 + 0.5) / 4, abs=1e-6)


def test_self_financing_round_trip_cost_identity():
    s = _sched((0, 1, 1.0), (7, 8, 0.0))
    r = run_weight_sleeve(_bars([100] * 10, [100] * 10), s, 500.0, TAKER, FINE, 0.10)
    assert r.terminal_value == pytest.approx(500.0 * (1 - 0.009) / (1 + 0.009), rel=1e-6)
    assert (r.buys, r.sells) == (1, 1)
```

- [ ] **Step 2: Run them.** Expected: FAIL, `ModuleNotFoundError: tools.vol_target.fsim`.
- [ ] **Step 3: Implement** `fsim.py`

```python
"""Self-financing fractional-weight sleeve. Pure: no I/O, no clock.

Weekly target-weight instructions are decided at a Sunday close (deadband on the weight marked
at that close) and executed at the scheduled open. Units are an integer count of base_increment
ticks so rounding never drifts. A missed open or a size-rejected rebalance expires (no retry).
Equity is marked at each close; a stale close is carried for MARKING only. A terminal holding
that fails base_min or quote_min is flagged unliquidatable and valued at 0, never invented cash.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from tools.slow_trend.sim import Costs, Product

_EPS = 1e-9


def solve_units(cash: float, units: float, open_px: float, w: float, costs: Costs) -> float:
    """Units after a fee-inclusive trade such that units*open/(cash'+units*open) == w."""
    pb = open_px * (1 + costs.slip) * (1 + costs.entry_fee)
    up = w * (cash + units * pb) / (open_px * (1 - w) + w * pb)
    if up > units:
        return up
    ps = open_px * (1 - costs.slip) * (1 - costs.exit_fee)
    down = w * (cash + units * ps) / (open_px * (1 - w) + w * ps)
    return min(down, units)


@dataclass
class WeightResult:
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
    buys: int
    sells: int
    size_skipped: int
    missed_open: int
    within_deadband: int
    invalid_decisions: int
    valid_decisions: int
    executed: int
    requested: list = field(default_factory=list)
    executed_weights: list = field(default_factory=list)
    capped: int = 0
    exposure: float = 0.0
    stale_mark_days: int = 0
    max_stale_run: int = 0


class _Book:
    def __init__(self, cash: float, costs: Costs, product: Product):
        self.cash, self.ticks, self.costs, self.p = cash, 0, costs, product
        self.exec_fees = self.traded = self.slippage = 0.0
        self.buys = self.sells = self.size_skipped = 0

    @property
    def units(self) -> float:
        return self.ticks * self.p.base_increment

    def rebalance(self, open_px: float, w: float) -> bool:
        inc, c = self.p.base_increment, self.costs
        want = solve_units(self.cash, self.units, open_px, w, c) / inc
        if want > self.ticks + _EPS:
            px = open_px * (1 + c.slip)
            afford = math.floor(self.cash / (px * (1 + c.entry_fee)) / inc + _EPS)
            d = min(math.floor(want - self.ticks + _EPS), afford)
            du, notional = d * inc, d * inc * px
            if d <= 0 or du < self.p.base_min or notional < self.p.quote_min:
                self.size_skipped += 1
                return False
            fee = notional * c.entry_fee
            self.cash -= notional + fee
            self.ticks += d
            self.buys += 1
        elif want < self.ticks - _EPS:
            px = open_px * (1 - c.slip)
            d = min(math.floor(self.ticks - want + _EPS), self.ticks)
            du, notional = d * inc, d * inc * px
            if d <= 0 or du < self.p.base_min or notional < self.p.quote_min:
                self.size_skipped += 1
                return False
            fee = notional * c.exit_fee
            self.cash += notional - fee
            self.ticks -= d
            self.sells += 1
        else:
            return False
        self.exec_fees += fee
        self.traded += notional
        self.slippage += du * open_px * c.slip
        return True

    def weight(self, px: float) -> float:
        value = self.units * px
        total = self.cash + value
        return value / total if total > 0 else 0.0


def run_weight_sleeve(
    bars: pd.DataFrame,
    sched: pd.DataFrame,
    cash: float,
    costs: Costs,
    product: Product,
    deadband: float,
) -> WeightResult:
    book = _Book(cash, costs, product)
    pending: dict = {}
    n = dict(missed=0, within=0, invalid=0, valid=0, executed=0, capped=0)
    requested, executed_w, values, fractions = [], [], [], []
    last_close, stale = float("nan"), [0, 0, 0]  # count, run, max_run

    def decide(day, held_weight):
        if day not in sched.index:
            return
        row = sched.loc[day]
        tgt = row["target"]
        if not (math.isfinite(tgt) and math.isfinite(held_weight)):  # missing Sunday close too
            n["invalid"] += 1
            return
        n["valid"] += 1
        n["capped"] += tgt >= 1.0
        if abs(tgt - held_weight) > deadband:
            pending[row["execute"]] = tgt
        else:
            n["within"] += 1

    first = bars.index[0]
    for d in sched.index[sched.index < first]:  # initialisation Sunday before the period
        decide(d, 0.0)
    for day, row in bars.iterrows():
        px = row["open"]
        if day in pending:
            tgt = pending.pop(day)
            if math.isnan(px):
                n["missed"] += 1
            else:
                requested.append(tgt)
                if book.rebalance(px, tgt):
                    n["executed"] += 1
                    executed_w.append(book.weight(px))
        if math.isnan(row["close"]):
            stale[0] += 1
            stale[1] += 1
            stale[2] = max(stale[2], stale[1])
            if math.isnan(last_close) and book.ticks > 0:
                last_close = px
        else:
            last_close, stale[1] = row["close"], 0
        mark = last_close if book.ticks > 0 else 0.0
        values.append(book.cash + book.units * mark)
        fractions.append(book.weight(mark) if book.ticks > 0 else 0.0)
        close = row["close"]
        decide(day, float("nan") if math.isnan(close) else book.weight(close))
    return _finish(book, bars, cash, values, fractions, n, requested, executed_w, stale)


def _finish(book, bars, cash, values, fractions, n, requested, executed_w, stale) -> WeightResult:
    last = bars["close"].iloc[-1]
    if math.isnan(last):
        raise ValueError("terminal close missing: endpoint is never moved")
    p, c = book.p, book.costs
    proceeds = book.units * last * (1 - c.slip)
    unliq = book.ticks > 0 and (book.units < p.base_min or proceeds < p.quote_min)
    residual_units = residual_marked = 0.0
    terminal_slip = book.units * last * c.slip
    if unliq:
        residual_units, residual_marked = book.units, book.units * last
        proceeds, terminal_slip = 0.0, 0.0
    terminal_fee = proceeds * c.exit_fee
    return WeightResult(
        equity=pd.Series(values, index=bars.index),
        initial=cash,
        terminal_value=book.cash + proceeds - terminal_fee,
        exec_fees=book.exec_fees,
        terminal_fee=terminal_fee,
        traded_notional=book.traded,
        slippage_cost=book.slippage + terminal_slip,
        unliquidatable=bool(unliq),
        residual_units=residual_units,
        residual_marked_value=residual_marked,
        buys=book.buys,
        sells=book.sells,
        size_skipped=book.size_skipped,
        missed_open=n["missed"],
        within_deadband=n["within"],
        invalid_decisions=n["invalid"],
        valid_decisions=n["valid"],
        executed=n["executed"],
        requested=requested,
        executed_weights=executed_w,
        capped=int(n["capped"]),
        exposure=float(sum(fractions) / len(fractions)),
        stale_mark_days=stale[0],
        max_stale_run=stale[2],
    )
```

Note on the decision timing: `decide` runs AFTER the day's close mark. So a Sunday inside the
period is decided on its own close, and its Monday/Tuesday execution is picked up on a later
loop iteration. A pre-period initialisation Sunday is decided with `w_held = 0` (the sleeve
starts as cash).

- [ ] **Step 4: Run the tests.** Expected: PASS, 16/16 (including the parametrised cases).
  Then break each Review Focus behaviour on purpose and confirm its test goes RED:
  - `>` → `>=` (deadband);
  - executing at the Sunday close instead of the open;
  - retrying on the next valid open;
  - dropping the `base_min` terminal check.
- [ ] **Step 5: Commit**: `git commit -m "feat(research): vol-target fractional self-financing sleeve" -- backend/tools/vol_target/fsim.py backend/tests/tools/vol_target/test_fsim.py`

### Task 3: Verdict wrapper with the coverage gate

**Files:**
- Create: `backend/tools/vol_target/verdict.py`
- Test: `backend/tests/tools/vol_target/test_verdict.py`

**Interfaces:**
- Produces: `verdict(data_ok: bool, results, block_valid_decisions: dict) -> dict`.
  - `results[period][scenario]` is `{"pass": bool, "affected": bool}`.
  - `block_valid_decisions` is `{pid: int}`, under P0.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from tools.slow_trend import gates as G
from tools.vol_target.verdict import verdict

OK = {"pass": True, "affected": False}
FAIL = {"pass": False, "affected": False}
AFF = {"pass": False, "affected": True}
SC = ("P0", "S10", "S25", "D1", "SM")
COVER = {"BTC-USD": 50, "ETH-USD": 50}


def _res(**over):
    r = {p: {s: dict(OK) for s in SC} for p in ("dev", "block")}
    for key, v in over.items():
        p, s = key.split("_")
        r[p][s] = v
    return r


@pytest.mark.parametrize(
    "res,expected",
    [
        (_res(dev_P0=FAIL), ("KILL", "primary_failed")),
        (_res(block_P0=AFF), ("INCONCLUSIVE", "terminal_size")),
        (_res(block_S25=AFF), ("INCONCLUSIVE", "terminal_size")),
        (_res(dev_D1=FAIL), ("INCONCLUSIVE", "fragile")),
        (_res(block_SM=FAIL), ("PASS_TO_FORWARD", "all_gates")),  # SM never gates
        (_res(), ("PASS_TO_FORWARD", "all_gates")),
    ],
)
def test_shared_branches_match_the_slow_trend_order(res, expected):
    v = verdict(True, res, COVER)
    assert (v["verdict"], v["reason"]) == expected
    assert v == G.verdict(True, res, 10**6)  # identical wherever coverage is met


def test_data_failure_first():
    assert verdict(False, None, {})["reason"] == "data"


def test_coverage_is_per_sleeve_and_last():
    v = verdict(True, _res(), {"BTC-USD": 50, "ETH-USD": 3})
    assert (v["verdict"], v["reason"]) == ("INCONCLUSIVE", "insufficient_coverage")
    assert verdict(True, _res(dev_P0=FAIL), {"BTC-USD": 0, "ETH-USD": 0})["verdict"] == "KILL"
```

- [ ] **Step 2: Run them.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement**

```python
"""Vol-target verdict: slow_trend's preregistered order, with the binary round-trip activity
gate replaced by per-sleeve weekly decision coverage (a partial reduction is not a round trip).
Re-stated rather than passing a decision count into slow_trend's `block_round_trips` argument."""

from tools.vol_target import prereg as P


def verdict(data_ok: bool, results, block_valid_decisions: dict) -> dict:
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
    if any(block_valid_decisions.get(p, 0) < P.MIN_BLOCK_VALID_DECISIONS for p in P.PRODUCTS):
        return {"verdict": "INCONCLUSIVE", "reason": "insufficient_coverage"}
    return {"verdict": "PASS_TO_FORWARD", "reason": "all_gates"}
```

- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Commit** (`verdict.py` and `test_verdict.py`).

### Task 4: Evaluation and freeze mechanics (`screen.py`)

**Files:**
- Create: `backend/tools/vol_target/screen.py`
- Test: `backend/tests/tools/vol_target/test_screen.py`

**Interfaces:**
- Consumes:
  - from Tasks 1–3: `schedule`, `run_weight_sleeve`, `verdict`;
  - from `tools.slow_trend`:
    - `daily_bars` (`D`): `first_day`, `audit`, `window`, `normalise`, `to_calendar`;
    - `sim`: `Costs`, `Product`, `run_sleeve`, `run_dca`;
    - `metrics` (`M`);
    - `gates`: `passes`, `affected`;
    - `screen` (`S`): `_summ` (comparators), `experiment_identity`, `run_mode`,
      `last_attempt`, `replay_label`, `source_digest`, `text_digest`, `ensure_new_snapshot`,
      `_append`, `_git`, `_now`, `_sha`.
- Produces:
  - `evaluate(raw: dict, constraints: dict) -> dict` (a report);
  - `import_snapshot(src: Path, out: Path) -> dict`;
  - CLI: `python -m tools.vol_target.screen {import <slow_trend_out>|lock|run [--replay] [--new-preregistration]}`.

- [ ] **Step 1: Write the failing tests**

```python
import json
import math

import numpy as np
import pandas as pd
import pytest

from tools.vol_target import prereg as P
from tools.vol_target import screen as V

DAY = 86400


def _raw(first="2016-05-18", last="2026-10-02", drop=(), seed=0):
    rng = np.random.default_rng(seed)
    days = pd.date_range(first, last, freq="D")
    days = days[~days.isin(pd.to_datetime(list(drop)))]
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(days))))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame({
        "start": (days.astype("int64") // 10**9).astype("int64"),
        "open": open_, "high": np.maximum(open_, close) * 1.01,
        "low": np.minimum(open_, close) * 0.99, "close": close,
        "volume": 1.0, "page": 0,
    })


CONS = {p: {"base_increment": 1e-8, "base_min": 1e-8, "quote_min": 1.0} for p in P.PRODUCTS}


@pytest.fixture(scope="module")
def report():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V.P, "BOOT_RESAMPLES", 200)  # speed only; the bootstrap never gates
        return V.evaluate({p: _raw(seed=i) for i, p in enumerate(P.PRODUCTS)}, CONS)


def test_evaluate_runs_both_periods_every_scenario(report):
    assert report["periods"]["dev"] == ("2016-08-30", "2025-04-13")
    assert set(report["results"]["block"]["scenarios"]) == {s.name for s in P.SCENARIOS}
    assert report["verdict"]["verdict"] in {"KILL", "INCONCLUSIVE", "PASS_TO_FORWARD"}


def test_first_executions_follow_the_frozen_calendar(report):
    d = report["results"]
    assert d["dev"]["scenarios"]["P0"]["calendar"]["first_execute"] == "2016-09-05"
    assert d["dev"]["scenarios"]["D1"]["calendar"]["first_execute"] == "2016-09-06"
    assert d["block"]["scenarios"]["P0"]["calendar"]["first_execute"] == "2025-04-14"
    assert d["block"]["scenarios"]["D1"]["calendar"]["first_execute"] == "2025-04-15"


def test_fixed_diagnostic_never_gates(report):
    s = report["results"]["block"]["scenarios"]["P0"]
    assert "fixed50" in s and "passes" in s and "fixed50" not in json.dumps(s["passes"])


def test_coverage_counts_valid_p0_block_decisions_per_sleeve(report):
    cov = report["block_valid_decisions"]
    assert set(cov) == set(P.PRODUCTS) and all(v >= 70 for v in cov.values())


def test_a_data_gap_beyond_tolerance_is_inconclusive_data():
    gap = [f"2020-01-{d:02d}" for d in range(1, 10)]
    r = V.evaluate({p: _raw(drop=gap) for p in P.PRODUCTS}, CONS)
    assert r["verdict"]["reason"] == "data"


def test_import_refuses_a_snapshot_whose_manifest_is_not_the_locked_one(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "manifest.json").write_text("{}")
    with pytest.raises(RuntimeError, match="lock"):
        V.import_snapshot(src, tmp_path / "out")


def test_import_verifies_every_raw_file_and_copies_bytes(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    (src / "BTC-USD.raw.parquet").write_bytes(b"btc")
    (src / "ETH-USD.raw.parquet").write_bytes(b"eth")
    man = {"products": {
        "BTC-USD": {"sha256": V.S._sha(src / "BTC-USD.raw.parquet")},
        "ETH-USD": {"sha256": "sha256:" + "0" * 64},
    }}
    (src / "manifest.json").write_text(json.dumps(man))
    monkeypatch.setattr(V.P, "SOURCE_SNAPSHOT_LOCK", V.S._sha(src / "manifest.json"))
    with pytest.raises(RuntimeError, match="ETH-USD"):
        V.import_snapshot(src, out)
    assert not (out / "manifest.json").exists()  # nothing published on failure
```

- [ ] **Step 2: Run them.** Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `screen.py`

```python
"""Preregistered volatility-target screen. See the plan's Preregistration section.

python -m tools.vol_target.screen import <slow_trend_out>   # copy + verify the locked snapshot
python -m tools.vol_target.screen lock                      # write snapshot.lock (commit it)
python -m tools.vol_target.screen run [--replay] [--new-preregistration]
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tools.slow_trend import daily_bars as D
from tools.slow_trend import gates as G
from tools.slow_trend import metrics as M
from tools.slow_trend import screen as S
from tools.slow_trend.sim import Costs, Product, run_dca, run_sleeve
from tools.vol_target import prereg as P
from tools.vol_target.fsim import run_weight_sleeve
from tools.vol_target.signal import schedule
from tools.vol_target.verdict import verdict

BACKEND = Path(__file__).resolve().parents[2]
OUT = BACKEND / "data" / "research" / "vol_target"
LOCK = Path(__file__).with_name("snapshot.lock")
BLOCKS = (P.BOOT_BLOCK_WEEKS, *P.BOOT_SENS_WEEKS)
RAW = tuple(f"{p}.raw.parquet" for p in P.PRODUCTS)


def _realised_vol(equity: pd.Series) -> float:
    r = np.log(equity[equity > 0]).diff().dropna()
    return float(r.std(ddof=1) * math.sqrt(P.ANNUALISATION)) if len(r) > 1 else float("nan")


def _summ_w(results: list) -> dict:
    eq = sum(r.equity for r in results)
    tv = sum(r.terminal_value for r in results)
    ex, tf = sum(r.exec_fees for r in results), sum(r.terminal_fee for r in results)
    return {
        "terminal_value": tv,
        "net_return": tv / P.INITIAL_USD - 1,
        "max_drawdown": M.max_drawdown(eq, P.INITIAL_USD),
        "total_fees": ex + tf,
        "slippage_cost": sum(r.slippage_cost for r in results),
        "unliquidatable": [r.unliquidatable for r in results],
        "residual_marked_value": sum(r.residual_marked_value for r in results),
        "executed_turnover": sum(r.traded_notional for r in results) / P.INITIAL_USD,
        "realised_vol": _realised_vol(eq),
        "boundary_returns": M.boundary_returns(eq, P.INITIAL_USD),
        "per_sleeve": {
            pid: {
                "terminal_value": r.terminal_value,
                "net_return": r.terminal_value / r.initial - 1,
                "max_drawdown": M.max_drawdown(r.equity, r.initial),
                "realised_vol": _realised_vol(r.equity),
                "exposure": r.exposure,
                "buys": r.buys,
                "sells": r.sells,
                "size_skipped": r.size_skipped,
                "missed_open": r.missed_open,
                "within_deadband": r.within_deadband,
                "valid_decisions": r.valid_decisions,
                "invalid_decisions": r.invalid_decisions,
                "executed": r.executed,
                "capped_fraction": r.capped / r.valid_decisions if r.valid_decisions else None,
                "requested_weights": r.requested,
                "executed_weights": r.executed_weights,
                "stale_mark_days": r.stale_mark_days,
                "max_stale_run": r.max_stale_run,
            }
            for pid, r in zip(P.PRODUCTS, results, strict=True)
        },
        "_equity": eq,
    }


def run_period(cal: dict, start: str, end: str, products: dict) -> dict:
    out = {"start": start, "end": end, "scenarios": {}}
    for sc in P.SCENARIOS:
        costs = Costs(sc.entry_fee, sc.exit_fee, sc.slip)
        legs = {"vol_target": [], "fixed50": [], "buy_hold": [], "dca52": []}
        first_exec = []
        for pid in P.PRODUCTS:
            bars, prod = cal[pid].loc[start:end], products[pid]
            sch = schedule(cal[pid]["close"], start, end, sc.delay)
            first_exec.append(sch["execute"].min())
            fixed = sch.assign(target=P.FIXED_DIAG_WEIGHT)
            legs["vol_target"].append(run_weight_sleeve(bars, sch, P.SLEEVE_USD, costs, prod, P.DEADBAND))
            legs["fixed50"].append(run_weight_sleeve(bars, fixed, P.SLEEVE_USD, costs, prod, P.DEADBAND))
            hold = pd.Series(True, index=bars.index)
            legs["buy_hold"].append(run_sleeve(bars, hold, P.SLEEVE_USD, costs, prod))
            legs["dca52"].append(run_dca(bars, P.SLEEVE_USD, P.DCA_TRANCHES, costs, prod))
        s = {k: _summ_w(legs[k]) for k in ("vol_target", "fixed50")}
        s.update({k: S._summ(legs[k], P.INITIAL_USD) for k in ("buy_hold", "dca52")})
        s["cash"] = {"terminal_value": P.INITIAL_USD, "net_return": 0.0, "max_drawdown": 0.0}
        s["calendar"] = {"first_execute": str(min(first_exec).date())}
        s["passes"] = {
            **G.passes(s["vol_target"], s["buy_hold"], P.INITIAL_USD, P.G2_DRAWDOWN_RATIO),
            "affected": G.affected(s["vol_target"], s["buy_hold"]),
        }
        wt = M.weekly_returns(s["vol_target"]["_equity"])
        for name, key in (("buy_hold", "bh"), ("dca52", "dca52"), ("fixed50", "fixed50")):
            wc = M.weekly_returns(s[name]["_equity"])
            s[f"ci_vs_{key}"] = {
                str(b): M.paired_block_ci(wt, wc, b, P.BOOT_RESAMPLES, P.BOOT_SEED) for b in BLOCKS
            }
        for k in ("vol_target", "fixed50", "buy_hold", "dca52"):
            s[k].pop("_equity")
        out["scenarios"][sc.name] = s
    return out


def _inadequate(report: dict, why: str) -> dict:
    report["data_checks"]["inadequate_because"] = why
    report["verdict"] = verdict(False, None, {})
    return report


def evaluate(raw: dict, constraints: dict) -> dict:
    report = {"data_checks": {"terminal": {}}, "results": None}
    firsts = {p: D.first_day(raw[p]) for p in P.PRODUCTS}
    report["data_checks"]["first_day"] = firsts
    if any(f is None for f in firsts.values()):
        return _inadequate(report, "no_aligned_rows")
    common_first = max(firsts.values())
    report["data_checks"]["common_first"] = common_first
    if common_first > P.DEV_START:
        return _inadequate(report, "no_development_coverage")
    windows = {"dev": (P.DEV_START, P.DEV_END), "block": (P.BLOCK_START, P.BLOCK_END)}
    audits = {
        w: {p: D.audit(raw[p], a, b, P.MAX_MISSING_DAYS) for p in P.PRODUCTS}
        for w, (a, b) in windows.items()
    }
    report["data_checks"]["audits"] = audits
    if not all(v["adequate"] for w in audits.values() for v in w.values()):
        return _inadequate(report, "audit")
    cal = {
        p: D.to_calendar(
            D.normalise(D.window(raw[p], common_first, P.BLOCK_END)), common_first, P.BLOCK_END
        )
        for p in P.PRODUCTS
    }
    for p in P.PRODUCTS:
        report["data_checks"]["terminal"][p] = {
            d: bool(pd.notna(cal[p].loc[d, "close"])) for d in (P.DEV_END, P.BLOCK_END)
        }
    if not all(all(t.values()) for t in report["data_checks"]["terminal"].values()):
        return _inadequate(report, "terminal_close_missing")
    products = {p: Product(**constraints[p]) for p in P.PRODUCTS}
    res = {n: run_period(cal, a, b, products) for n, (a, b) in windows.items()}
    passes = {n: {s: r["scenarios"][s]["passes"] for s in r["scenarios"]} for n, r in res.items()}
    p0 = res["block"]["scenarios"]["P0"]["vol_target"]["per_sleeve"]
    coverage = {p: p0[p]["valid_decisions"] for p in P.PRODUCTS}
    report.update(
        periods=windows,
        results=res,
        block_valid_decisions=coverage,
        verdict=verdict(True, passes, coverage),
        prior_in_family=["slow_trend SMA100: KILL primary_failed (report_2252e434a40e11f6_a1_first)"],
    )
    return report


def import_snapshot(src: Path, out: Path) -> dict:
    """Copy the slow-trend LOCKED snapshot; verify the manifest against the frozen lock and
    every raw file against the manifest BEFORE publishing anything."""
    if S._sha(src / "manifest.json") != P.SOURCE_SNAPSHOT_LOCK:
        raise RuntimeError("source manifest does not match SOURCE_SNAPSHOT_LOCK")
    manifest = json.loads((src / "manifest.json").read_text())
    for pid in P.PRODUCTS:
        if S._sha(src / f"{pid}.raw.parquet") != manifest["products"][pid]["sha256"]:
            raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
    S.ensure_new_snapshot(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in RAW:
        shutil.copyfile(src / name, out / name)
    shutil.copyfile(src / "manifest.json", out / "manifest.json")  # last: publication marker
    return manifest


def _source_files() -> list:
    here = Path(__file__).parent
    return (
        sorted(here.glob("*.py"))
        + sorted((here.parent / "slow_trend").glob("*.py"))
        + [BACKEND / "clients" / "coinbase_client.py"]
    )


def _lock() -> None:
    sha = S._sha(OUT / "manifest.json")
    if sha != P.SOURCE_SNAPSHOT_LOCK:
        sys.exit("imported manifest is not the locked slow-trend snapshot")
    LOCK.write_text(sha + "\n")
    print(f"wrote {LOCK}; commit it before `run`")


def _run(replay: bool, new_prereg: bool) -> None:
    dirty = S._git("status", "--porcelain", "--", *P.FREEZE_PATHS)
    if dirty:
        sys.exit(f"refusing to run: uncommitted changes\n{dirty}")
    manifest_sha = S._sha(OUT / "manifest.json")
    if not LOCK.exists() or LOCK.read_text().strip() != manifest_sha:
        sys.exit("snapshot.lock missing or does not match the manifest")
    manifest = json.loads((OUT / "manifest.json").read_text())
    head, prereg_sha = S._git("rev-parse", "HEAD"), S.text_digest(Path(P.__file__))
    exp_id = S.experiment_identity(prereg_sha, manifest_sha)
    ledger = OUT / "runs.jsonl"
    entries = (
        [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
        if ledger.exists()
        else []
    )
    mode = S.run_mode(entries, exp_id, replay=replay, new_prereg=new_prereg)
    src = S.source_digest(_source_files())
    corrects = None
    if mode == "replay":
        mode, corrects = S.replay_label(entries, exp_id, src)
    attempt = 1 + max((e.get("attempt", 0) for e in entries), default=0)
    base = {"experiment_id": exp_id, "attempt": attempt, "head": head, "mode": mode,
            "source_sha256": src, "replays_attempt": corrects}
    S._append(ledger, {**base, "status": "started",
                       "parent": S.last_attempt(entries, exp_id), "at": S._now()})
    try:
        raw, constraints = {}, {}
        for pid, m in manifest["products"].items():
            path = OUT / f"{pid}.raw.parquet"
            if S._sha(path) != m["sha256"]:
                raise RuntimeError(f"snapshot {pid} does not match its manifest digest")
            raw[pid], constraints[pid] = pd.read_parquet(path), m["constraints"]
        report = evaluate(raw, constraints)
    except Exception as exc:
        S._append(ledger, {**base, "status": "failed", "error": repr(exc), "at": S._now()})
        raise
    report.update(experiment_id=exp_id, attempt=attempt, mode=mode, head=head,
                  source_sha256=src, replays_attempt=corrects,
                  prereg_sha256=prereg_sha, manifest_sha256=manifest_sha)
    name = f"report_{exp_id}_a{attempt}_{mode}.json"
    (OUT / name).write_text(json.dumps(report, indent=2, default=str))
    S._append(ledger, {**base, "status": "completed", "report": name,
                       "verdict": report["verdict"], "at": S._now()})
    print(json.dumps({"verdict": report["verdict"], "mode": mode, "report": name}, indent=2))


if __name__ == "__main__":
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    if cmd == "import" and args:
        print(json.dumps(import_snapshot(Path(args[0]), OUT), indent=2))
    elif cmd == "lock":
        _lock()
    elif cmd == "run":
        flags = set(args)
        _run(replay="--replay" in flags, new_prereg="--new-preregistration" in flags)
    else:
        sys.exit("usage: python -m tools.vol_target.screen "
                 "{import <slow_trend_out>|lock|run [--replay] [--new-preregistration]}")
```

- [ ] **Step 4: Run the tests.** Expected: PASS. Then run the whole `tests/tools/vol_target`
  and `tests/tools/slow_trend` directories (the frozen package is unchanged). Expected: all PASS.
- [ ] **Step 5: Commit** (`screen.py` and `test_screen.py`); `.gitignore` already covers
  `backend/data/research/`. Verify that with `git check-ignore backend/data/research/vol_target/x`.

### Task 5: Import, lock, single run, record

- [ ] **Step 1: Codex reviews this plan** (BLOCKING / NON-BLOCKING) before any import, lock or
  run. Apply findings under TDD.
- [ ] **Step 2: Import**:
  `cd backend && ../.venv/Scripts/python.exe -m tools.vol_target.screen import C:/Users/gl450/polymarket_app/.claude/worktrees/slow-trend-screen/backend/data/research/slow_trend`
  Expected: the manifest is printed, and `OUT` holds 2 raw files plus the manifest.
- [ ] **Step 3: Lock**: `... -m tools.vol_target.screen lock`, then commit
  `backend/tools/vol_target/snapshot.lock`. Its content equals `SOURCE_SNAPSHOT_LOCK`.
- [ ] **Step 4: Run ONCE**: `... -m tools.vol_target.screen run`. Expected: mode `first`, a
  report path and a verdict.
- [ ] **Step 5: Record** the verdict in the CHANGELOG with:
  - the experiment id;
  - the per-period P0 numbers versus buy-and-hold;
  - the fixed-50 diagnostic, labelled non-gating;
  - the coverage;
  - the status caveats (a previously observed block; the second candidate in this family).

  Send the report to Codex for an independent read. Commit and push.
