"""Offline diagnostic for the ATR-causality proposal. Measures; changes nothing.

`docs/specs/2026-09-27-atr-causality-repair-proposal.md` proposes three independent changes to the
exit simulation. This module re-implements the exit walk with a knob for each, so their effects can
be measured SEPARATELY rather than as one combined diff that cannot be attributed to a cause.

Nothing here is production label semantics. `labels.py` is untouched, no artifact is regenerated,
and the published v2 records keep their meaning. This is the evidence-gathering step the proposal
asks for before any of it is decided.

**The re-implementation is the risk, and the anchor is the answer.** A probe that has drifted from
`labels._simulate_one` measures itself, and would report differences that do not exist. So
`test_the_legacy_variant_reproduces_the_production_simulation_bit_exactly` compares the LEGACY
variant against production via `float.hex()` across stop, trail and horizon exits. Every number
this module produces is worth exactly as much as that test.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

_ORDERINGS = ("high_before_low", "low_before_high")


@dataclass(frozen=True)
class VariantSpec:
    """One combination of the proposed changes.

    `atr_lag_bars`: 0 reproduces production; 1 is the proposed repair -- the threshold for the bar
    being traded through comes from the last COMPLETED bar.
    `ordering`: which intrabar sequence to assume. Neither is conservative (proposal §4.1).
    `gap_fill`: when True a triggered exit fills at `min(level, open)` rather than at the level,
    because a bar that opened beyond the level could not have filled there.
    """

    name: str
    atr_lag_bars: int = 0
    ordering: str = "high_before_low"
    gap_fill: bool = False


LEGACY = VariantSpec("legacy")
LAG_ONLY = VariantSpec("lag_only", atr_lag_bars=1)
ORDERING_ONLY = VariantSpec("ordering_only", ordering="low_before_high")
GAP_ONLY = VariantSpec("gap_only", gap_fill=True)
COMBINED = VariantSpec("combined", atr_lag_bars=1, ordering="low_before_high", gap_fill=True)

ALL_VARIANTS = (LEGACY, LAG_ONLY, ORDERING_ONLY, GAP_ONLY, COMBINED)


@dataclass(frozen=True)
class VariantResult:
    """One simulated trade. `pnl` is NaN when the horizon does not fit, as in production."""

    pnl: float
    bars_held: Optional[int]
    exit_kind: Optional[str]
    exit_price: Optional[float]
    gapped: bool = False


@dataclass(frozen=True)
class BracketReport:
    """Both orderings for one trade, plus the bound over the two.

    `lower_bound_pnl` bounds the two ENUMERATED orderings. It is NOT a bound over all intrabar
    paths -- a bar may visit its low, recover, set its high and fall back, touching a level
    neither ordering triggers. Proposal §4.2 states that limit; this field must not be described
    as a worst case.

    It is also a DIAGNOSTIC, not an approved production label policy. Whether a published label
    should be this bound is operator question 1 in the proposal, still open.
    """

    high_first: VariantResult
    low_first: VariantResult
    agree: bool
    lower_bound_pnl: float
    # WHICH branch the lower bound came from, and that branch's own result. Codex 2cc2feec: a
    # smaller-PnL path must carry its own exit, timing and occupancy -- pairing one branch's PnL
    # with the other's metadata would invent a trade that neither ordering produces.
    lower_bound_ordering: str
    lower_bound_result: VariantResult


def _checked_spec(spec: VariantSpec) -> VariantSpec:
    lag = spec.atr_lag_bars
    if isinstance(lag, bool) or not isinstance(lag, numbers.Integral) or int(lag) < 0:
        raise ValueError(
            f"atr_lag_bars must be a non-negative integer without coercion, got {lag!r}"
        )
    if spec.ordering not in _ORDERINGS:
        raise ValueError(f"ordering must be one of {list(_ORDERINGS)}, got {spec.ordering!r}")
    if type(spec.gap_fill) is not bool:
        raise ValueError("gap_fill must be an actual bool")
    return spec


def _threshold(atr_pcts: Sequence[float], index: int, lag: int, floor: float) -> float:
    """The trail threshold for the bar at `index`, from `lag` bars earlier.

    Clamped at 0 so an early bar cannot read before the array; the non-finite fallback to the
    floor is production behaviour, preserved deliberately (proposal §3.1) even though the SET of
    rows that take it shifts with the lag.
    """
    source = index - int(lag)
    if source < 0:
        source = 0
    value = float(atr_pcts[source])
    if not math.isfinite(value):
        value = floor
    return max(value, floor)


def simulate_variant(
    *,
    entry_idx: int,
    horizon: int,
    opens: Sequence[float],
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    atr_pcts: Sequence[float],
    config: Mapping[str, float],
    spec: VariantSpec,
) -> VariantResult:
    """One trade under one variant. Mirrors `labels._simulate_one` when `spec is LEGACY`."""
    spec = _checked_spec(spec)
    stop_loss_pct = float(config["stop_loss_pct"])
    atr_trail_floor = float(config["atr_trail_floor"])
    max_hold_bars = int(config["max_hold_bars"])
    round_trip_fee = float(config["round_trip_fee"])

    n = len(closes)
    entry_price = float(closes[entry_idx])
    horizon_cap = min(int(horizon), max_hold_bars)
    last_idx = entry_idx + horizon_cap
    if last_idx >= n:
        # Production precheck: a row whose full horizon does not fit is unavailable even when a
        # stop would have fired on its first bar.
        return VariantResult(float("nan"), None, None, None)

    peak = entry_price
    for step in range(1, horizon_cap + 1):
        index = entry_idx + step
        bar_low = float(lows[index])
        bar_high = float(highs[index])
        bar_open = float(opens[index])

        # 1. Stop-loss first, matching the live exit ladder.
        if bar_low / entry_price - 1.0 <= -stop_loss_pct:
            # The stop level is a constant of the entry, so it was in force at the open.
            level = entry_price * (1.0 - stop_loss_pct)
            price, gapped = _fill(level, level, bar_open, spec.gap_fill)
            return VariantResult(
                (price / entry_price - 1.0) - round_trip_fee, step, "stop", price, gapped
            )

        # 2. Trail. The ordering decides whether THIS bar's high may raise the peak before its
        #    low is tested against it -- which changes both whether an exit fires and, through
        #    the peak, at what level.
        threshold = _threshold(atr_pcts, index, spec.atr_lag_bars, atr_trail_floor)
        # The level in force when this bar opened, before its own high can lift the peak.
        level_at_open = peak * (1.0 - threshold)
        if spec.ordering == "high_before_low":
            if bar_high > peak:
                peak = bar_high
            triggered = bar_low / peak - 1.0 <= -threshold
        else:
            triggered = bar_low / peak - 1.0 <= -threshold
        if triggered:
            level = peak * (1.0 - threshold)
            price, gapped = _fill(level, level_at_open, bar_open, spec.gap_fill)
            return VariantResult(
                (price / entry_price - 1.0) - round_trip_fee, step, "trail", price, gapped
            )
        if spec.ordering == "low_before_high" and bar_high > peak:
            peak = bar_high

    # 3. Horizon reached untriggered: the close of a known bar, so no gap question arises.
    exit_price = float(closes[last_idx])
    return VariantResult(
        (exit_price / entry_price - 1.0) - round_trip_fee, horizon_cap, "horizon", exit_price
    )


def _fill(level: float, level_at_open: float, bar_open: float, gap_fill: bool) -> tuple:
    """The fill price, and whether the bar had already gapped through its level at the open.

    `level_at_open` is the exit level that EXISTED when the bar opened; `level` is the one that
    actually triggered. They differ on the trail branch under `high_before_low`, where this bar's
    own high raises the peak and so lifts the level mid-bar.

    A gap means the price was already beyond a level that was in force at the open. Testing the
    open against a level created later in the same bar is not a gap (Codex cb898726): with prior
    peak 100, open 100, high 120, low 95 and a 6% floor, the level at the open was 94 -- the open
    never gapped through it -- yet the triggering level of 112.8 is above the open, and the
    earlier version reported a fabricated fill at 100.0 (pnl 0.0000) instead of 112.8 (+0.1280).
    """
    if not gap_fill:
        return level, False
    if bar_open < level_at_open:
        return bar_open, True
    return level, False


def bracket_orderings(
    *,
    entry_idx: int,
    horizon: int,
    opens: Sequence[float],
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    atr_pcts: Sequence[float],
    config: Mapping[str, float],
    atr_lag_bars: int = 1,
    gap_fill: bool = False,
) -> BracketReport:
    """Run both orderings for one trade and report whether they agree.

    Agreement means the same exit kind, bar and value. Disagreement is the measurable form of the
    ambiguity, and §4.2 makes the published label the lower bound of the two with a blocker.
    """
    shared = dict(
        entry_idx=entry_idx,
        horizon=horizon,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atr_pcts,
        config=config,
    )
    high_first = simulate_variant(
        spec=VariantSpec("bracket_high", atr_lag_bars=atr_lag_bars, gap_fill=gap_fill), **shared
    )
    low_first = simulate_variant(
        spec=VariantSpec(
            "bracket_low",
            atr_lag_bars=atr_lag_bars,
            ordering="low_before_high",
            gap_fill=gap_fill,
        ),
        **shared,
    )
    both_nan = high_first.pnl != high_first.pnl and low_first.pnl != low_first.pnl
    if both_nan:
        return BracketReport(
            high_first, low_first, True, float("nan"), "high_before_low", high_first
        )
    agree = (
        high_first.exit_kind == low_first.exit_kind
        and high_first.bars_held == low_first.bars_held
        and high_first.pnl == low_first.pnl
    )
    if low_first.pnl < high_first.pnl:
        worse, worse_name = low_first, "low_before_high"
    else:
        worse, worse_name = high_first, "high_before_low"
    return BracketReport(high_first, low_first, agree, worse.pnl, worse_name, worse)
