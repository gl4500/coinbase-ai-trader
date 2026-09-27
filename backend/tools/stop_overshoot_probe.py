"""Split an 8% stop's realised overshoot into what no execution path could avoid and what
some path might have.

51 live CNN `STOP_LOSS` exits realise mean -9.494% against a configured 0.08; the tick path
(`WS_STOP_LOSS`) realises -8.323%. The ~1.17-point difference has two candidate causes with
opposite remedies -- a gap through the level (routing cannot help; the nominal stop is simply
not achievable on that instrument) versus a crossing inside the bar (routing might).

The discriminator is the exit bar's OPEN. That is the one quantity here that is OBSERVED
rather than inferred: the within-bar path is unknowable from OHLC, which is why nothing below
depends on it. `tests/tools/test_stop_overshoot_probe.py` carries the full limitation list;
the load-bearing one is that `best_achievable_fill` for an intrabar cross assumes a fill at
the level and therefore yields an UPPER BOUND on what routing could recover, not an estimate.

This module is pure. It reads no database, no file and no clock, so the report layer can be
audited separately from the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.cnn_agent import _CNN_STOP_LOSS_PCT

# Imported rather than restated. A probe with its own copy of the threshold keeps agreeing
# with itself after production moves, which is the failure mode that makes probes useless.
CONFIGURED_STOP_PCT: float = _CNN_STOP_LOSS_PCT

GAP_AT_OPEN = "GAP_AT_OPEN"
INTRABAR_CROSS = "INTRABAR_CROSS"
LEVEL_NOT_REACHED = "LEVEL_NOT_REACHED"


@dataclass(frozen=True)
class StopFill:
    """One stop exit, decomposed.

    `unavoidable_pts` and `attributable_pts` are percentage points of the ENTRY price and sum
    to the whole overshoot past the stop, so the split can never quietly lose or invent
    points. Both are <= 0 by construction for a real overshoot; 0 means no overshoot in that
    column rather than missing data. `None` marks a case the exit bar cannot speak to.
    """

    category: str
    stop_level: float
    best_achievable_fill: Optional[float]
    unavoidable_pts: Optional[float]
    attributable_pts: Optional[float]


def classify_stop_fill(
    entry_price: float,
    bar_open: float,
    bar_low: float,
    exit_price: float,
    stop_pct: float = CONFIGURED_STOP_PCT,
) -> StopFill:
    """Classify one stop exit against the bar in which it was recorded.

    An open at or below the stop level is a GAP: a touch at the open is not a miss, and no
    faster path improves on it, so the boundary belongs with the unavoidable cases.
    """
    if not entry_price > 0 or not bar_open > 0 or not bar_low > 0 or not exit_price > 0:
        raise ValueError("prices must be positive")
    if bar_low > bar_open:
        raise ValueError("incoherent bar: low above open")
    if not 0 < stop_pct < 1:
        raise ValueError("stop_pct must lie in (0, 1)")

    stop_level = entry_price * (1.0 - stop_pct)

    def pts(from_price: float, to_price: float) -> float:
        return (to_price - from_price) / entry_price * 100.0

    if bar_open <= stop_level:
        # The level was already gone when the bar began; the open is the best outcome
        # available to any path, and whatever separates the open from the recorded fill is
        # not explained by the gap.
        return StopFill(
            category=GAP_AT_OPEN,
            stop_level=stop_level,
            best_achievable_fill=bar_open,
            unavoidable_pts=pts(stop_level, bar_open),
            attributable_pts=min(0.0, pts(bar_open, exit_price)),
        )

    if bar_low <= stop_level:
        return StopFill(
            category=INTRABAR_CROSS,
            stop_level=stop_level,
            best_achievable_fill=stop_level,
            unavoidable_pts=0.0,
            attributable_pts=min(0.0, pts(stop_level, exit_price)),
        )

    # The bar never touched the level, so the crossing belongs to another bar. Saying
    # anything else here would invent an opportunity inside a bar that never offered one.
    return StopFill(
        category=LEVEL_NOT_REACHED,
        stop_level=stop_level,
        best_achievable_fill=None,
        unavoidable_pts=None,
        attributable_pts=None,
    )
