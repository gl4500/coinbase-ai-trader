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
        equity=pd.Series(marks.values, index=bars.index),
        initial=initial,
        terminal_value=book.cash + proceeds - terminal_fee,
        exec_fees=book.exec_fees,
        terminal_fee=terminal_fee,
        traded_notional=book.traded,
        slippage_cost=book.slippage + terminal_slip,
        unliquidatable=bool(unliquidatable),
        residual_units=residual_units,
        residual_marked_value=residual_marked,
        entries=book.entries,
        exits=book.exits,
        round_trips=book.exits,
        skipped=book.skipped,
        skipped_exits=book.skipped_exits,
        exposure=marks.held / len(bars),
        stale_mark_days=marks.stale,
        max_stale_run=marks.max_run,
    )


def run_sleeve(
    bars: pd.DataFrame, target: pd.Series, cash: float, costs: Costs, product: Product
) -> SleeveResult:
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


def run_dca(
    bars: pd.DataFrame, cash: float, tranches: int, costs: Costs, product: Product
) -> SleeveResult:
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
