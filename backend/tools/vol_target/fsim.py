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
_DEADBAND_DECIMALS = 12  # frozen: a difference equal to the deadband within 1e-12 does not trade


def eligible(target: float, held: float, deadband: float) -> bool:
    """Strict |target - held| > deadband, with binary-float noise removed (0.4 - 0.3 is not
    > 0.10 here)."""
    return round(abs(target - held), _DEADBAND_DECIMALS) > deadband


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
    decision_log: list = field(default_factory=list)
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
    log, executed_w, values, fractions = [], [], [], []
    last_close, stale = float("nan"), [0, 0, 0]  # count, run, max_run

    def decide(day, held_weight):
        if day not in sched.index:
            return
        row = sched.loc[day]
        tgt = row["target"]
        entry = {"decision": str(day.date()), "execute": str(row["execute"].date())}
        log.append(entry)
        if not (math.isfinite(tgt) and math.isfinite(held_weight)):  # missing Sunday close too
            n["invalid"] += 1
            entry.update(target=None, held=None, outcome="invalid")
            return
        n["valid"] += 1
        n["capped"] += tgt >= 1.0
        entry.update(target=tgt, held=held_weight)
        if eligible(tgt, held_weight, deadband):
            pending[row["execute"]] = (tgt, entry)
            entry["outcome"] = "eligible"
        else:
            n["within"] += 1
            entry["outcome"] = "within_deadband"

    first = bars.index[0]
    for d in sched.index[sched.index < first]:  # initialisation Sunday before the period
        decide(d, 0.0 if math.isfinite(sched.loc[d, "decision_close"]) else float("nan"))
    for day, row in bars.iterrows():
        px = row["open"]
        if day in pending:
            tgt, entry = pending.pop(day)
            if math.isnan(px):
                n["missed"] += 1
                entry["execution"] = "missed_open"
            else:
                skipped = book.size_skipped
                if book.rebalance(px, tgt):
                    n["executed"] += 1
                    executed_w.append(book.weight(px))
                    entry.update(execution="executed", executed_weight=executed_w[-1])
                else:
                    gone = book.size_skipped > skipped
                    entry["execution"] = "size_skipped" if gone else "no_change"
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
    return _finish(book, bars, cash, values, fractions, n, log, executed_w, stale)


def _finish(book, bars, cash, values, fractions, n, log, executed_w, stale) -> WeightResult:
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
        decision_log=log,
        executed_weights=executed_w,
        capped=int(n["capped"]),
        exposure=float(sum(fractions) / len(fractions)),
        stale_mark_days=stale[0],
        max_stale_run=stale[2],
    )
