"""Replay a held position with the STOP_LOSS rung removed. Diagnostic only.

The live exit ladder (cnn_agent._check_risk_exits) checks in this order:

    STOP_LOSS  ->  MODEL_DOWN  ->  TRAIL_STOP  ->  MAX_HOLD

This runs the same ladder minus STOP_LOSS, over real candles, to measure what the stop
actually prevented. It answers "would this position have recovered?" from price history
rather than from the shape of the ledger, because the stop and the trail act on the SAME
positions: the realised stop losses are not recoverable simply by deleting the column.

WHAT THIS CANNOT DO, and none of it is a detail:

  * MODEL_DOWN is NOT replayed. It depends on `p_down` cached on the position by
    generate_signal at scan time, which is not recorded in `trades`. Its absence biases
    every result toward holding LONGER than reality would have, so counterfactual gains
    are overstated and counterfactual losses understated.
  * Opportunity cost is ignored. A position held longer ties up capital that could have
    funded another entry, and a per-position replay cannot see that. This is a per-trade
    measurement, never a portfolio result.
  * Intrabar order is unknowable from OHLC. Both orderings are enumerated; where they
    disagree, the disagreement IS the finding.

The trail comes from production via `_compute_exit_threshold`. Reimplementing it here
would measure the reimplementation.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import List, Optional, Sequence

from agents.exit_thresholds import _compute_exit_threshold

ORDERINGS = ("high_first", "low_first")


@dataclass(frozen=True)
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class CounterfactualExit:
    exit_reason: str
    exit_price: float
    exit_pnl_pct: float
    peak_pnl_pct: float
    bars_held: int
    ordering: str


def _checked_positive_int(value, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{field} must be an integer without coercion, got {value!r}")
    if int(value) <= 0:
        raise ValueError(f"{field} must be strictly positive, got {value!r}")
    return int(value)


def _validated_bars(bars: Sequence[Bar]) -> List[Bar]:
    rows = list(bars)
    if not rows:
        raise ValueError("replay needs at least one bar of forward history")
    for i, b in enumerate(rows):
        values = (b.open, b.high, b.low, b.close)
        if not all(isinstance(v, numbers.Real) and math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"bar {i}: open/high/low/close must be finite and positive")
        # Relations, not just magnitudes: a bar whose low exceeds its high, or whose open
        # or close sits outside [low, high], describes no traversable path.
        if b.low > b.high:
            raise ValueError(f"bar {i}: low must not exceed high")
        if not (b.low <= b.open <= b.high) or not (b.low <= b.close <= b.high):
            raise ValueError(f"bar {i}: open and close must lie within [low, high]")
    return rows


def replay_without_stop(
    bars: Sequence[Bar],
    *,
    entry_price: float,
    max_hold_bars: int,
    ordering: str,
    position_dollars: Optional[float] = None,
    total_capital: Optional[float] = None,
) -> CounterfactualExit:
    """Walk bars forward applying trail-then-max-hold, with no stop.

    `ordering` decides which extreme of a bar is visited first. It matters because a bar
    can both set a new peak and fall through the resulting trail; whether the trail fired
    on that bar or the next is not recoverable from OHLC.
    """
    if ordering not in ORDERINGS:
        raise ValueError(f"ordering must be one of {ORDERINGS}, got {ordering!r}")
    if (
        not isinstance(entry_price, numbers.Real)
        or not math.isfinite(entry_price)
        or entry_price <= 0
    ):
        raise ValueError(f"entry_price must be finite and positive, got {entry_price!r}")
    max_hold_bars = _checked_positive_int(max_hold_bars, "max_hold_bars")
    rows = _validated_bars(bars)

    def pnl(price: float) -> float:
        return price / entry_price - 1.0

    peak = 0.0  # peak_pnl_pct ratchets upward only, as the live book does

    for index, bar in enumerate(rows[:max_hold_bars], start=1):
        # The level that EXISTED at this bar's open, from the peak carried in. A level
        # created later by this bar's own high cannot have been gapped through at the
        # open, because it did not exist yet. Getting this wrong turns an ordinary
        # give-back into a fabricated gap fill at the open -- the exact defect Codex
        # found in the ATR probe (cb898726), reproduced here and fixed the same way.
        threshold_at_open = _compute_exit_threshold(
            peak_pnl_pct=peak,
            position_dollars=position_dollars,
            total_capital=total_capital,
        )
        level_at_open = (
            entry_price * (1.0 + threshold_at_open) if math.isfinite(threshold_at_open) else None
        )

        extremes = (bar.high, bar.low) if ordering == "high_first" else (bar.low, bar.high)
        for price in extremes:
            candidate = pnl(price)
            if candidate > peak:
                peak = candidate
            # The trail is only armed once green; below that production returns -inf and
            # ONLY the (removed) stop could have fired.
            threshold = _compute_exit_threshold(
                peak_pnl_pct=peak,
                position_dollars=position_dollars,
                total_capital=total_capital,
            )
            if candidate < threshold:
                # Fill AT the level the position crossed, not at the extreme the bar
                # happened to reach. Production fires the moment price crosses the
                # threshold; filling at the low would charge the position the whole rest
                # of the bar's excursion and systematically overstate the loss.
                # The exception is a bar that OPENED below the level -- then the level was
                # never available and the open is the first realisable price.
                level_price = entry_price * (1.0 + threshold)
                # Only a level that existed AT THE OPEN can have been gapped through.
                gapped = level_at_open is not None and bar.open <= level_at_open
                fill = bar.open if gapped else level_price
                return CounterfactualExit(
                    exit_reason="TRAIL_STOP",
                    exit_price=fill,
                    exit_pnl_pct=pnl(fill),
                    peak_pnl_pct=peak,
                    bars_held=index,
                    ordering=ordering,
                )
        closing = pnl(bar.close)
        if closing > peak:
            peak = closing
        threshold = _compute_exit_threshold(
            peak_pnl_pct=peak,
            position_dollars=position_dollars,
            total_capital=total_capital,
        )
        if closing < threshold:
            return CounterfactualExit(
                exit_reason="TRAIL_STOP",
                exit_price=bar.close,
                exit_pnl_pct=closing,
                peak_pnl_pct=peak,
                bars_held=index,
                ordering=ordering,
            )

    held = rows[:max_hold_bars][-1]
    return CounterfactualExit(
        exit_reason="MAX_HOLD",
        exit_price=held.close,
        exit_pnl_pct=pnl(held.close),
        peak_pnl_pct=peak,
        bars_held=min(len(rows), max_hold_bars),
        ordering=ordering,
    )
