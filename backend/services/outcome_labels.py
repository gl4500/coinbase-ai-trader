"""Outcome-label contract, version 2 — pure label math, no I/O.

Full specification: `docs/specs/2026-09-26-outcome-label-contract.md`.

Version 1 (`OutcomeTracker.check_pending`) resolved an overdue signal with
whatever price was available when the resolver happened to run, which the
2026-09-26 strategy audit measured at a mean 45.75 h after the nominal 4 h
horizon. Version 2 measures the label at a *defined target time* from completed
hourly candles, and reports UNAVAILABLE rather than substituting a later price.

Everything here is deterministic and side-effect free so the timing rules can be
tested without a database, a clock, or the exchange.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

LABEL_VERSION = 2

BAR_SECS = 3600
H_BARS = 4

WIN_THRESHOLD = 0.005
LOSS_THRESHOLD = 0.005

MAX_RESOLVE_ATTEMPTS = 5
UNAVAILABLE_GRACE_SECS = 7 * 86400

SCORING_OUTCOMES = ("WIN", "LOSS", "NEUTRAL")

PRICE_SOURCE_LOCAL_CANDLES = "local_candles"

# A stored bar is usable only if the fetch that wrote it STARTED at least this
# long after the bar closed. A policy, not a measurement: the 2026-11-07 settle
# run (public endpoint, one weekend evening) saw closed bars still change up to
# (+60.6 s, +120.5 s] after close, so 300 s adds headroom over that one tail.
# It assumes the response was a fresh aggregate, which no request time can prove.
FINALITY_MARGIN_SECS = 300
FINALITY_BASIS = (
    f"request_start>=close+{FINALITY_MARGIN_SECS}s;assumes_fresh_aggregate;margin_unverified"
)


@dataclass(frozen=True)
class Resolution:
    """Outcome of one resolution attempt.

    status is RESOLVED (a label was produced), PENDING (retry later) or
    UNAVAILABLE (terminal, non-scoring). `reason` is set for everything except
    a clean RESOLVED.
    """

    status: str
    outcome: Optional[str] = None
    signed_return: Optional[float] = None
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    entry_candle_start: Optional[int] = None
    exit_candle_start: Optional[int] = None
    target_time: Optional[int] = None
    price_observed_at: Optional[int] = None
    price_source: Optional[str] = None
    finality_basis: Optional[str] = None
    reason: Optional[str] = None


def entry_candle_start(signal_time: float, bar_secs: int = BAR_SECS) -> int:
    """Bucket-open of the first bar starting strictly after `signal_time`.

    Strictly after: a signal stamped exactly on a boundary cannot be assumed to
    precede the trades inside that bar, so it takes the following one.
    """
    return int(signal_time // bar_secs) * bar_secs + bar_secs


def exit_candle_start(entry_start: int, h_bars: int = H_BARS, bar_secs: int = BAR_SECS) -> int:
    """Bucket-open of the bar whose close is the target price."""
    return entry_start + (h_bars - 1) * bar_secs


def target_time(entry_start: int, h_bars: int = H_BARS, bar_secs: int = BAR_SECS) -> int:
    """Instant the exit bar closes — when the label becomes measurable."""
    return entry_start + h_bars * bar_secs


def signed_return(entry_price: float, target_price: float, side: str) -> float:
    """Return from the signal's point of view, as a decimal fraction.

    A SELL is scored as a short: it is correct when price falls.
    """
    raw = (target_price - entry_price) / entry_price
    return -raw if side == "SELL" else raw


def classify(value: float) -> str:
    """WIN / LOSS / NEUTRAL against the inclusive +/-0.5% dead zone."""
    if value > WIN_THRESHOLD:
        return "WIN"
    if value < -LOSS_THRESHOLD:
        return "LOSS"
    return "NEUTRAL"


def _is_complete(candle_start: int, now: float, bar_secs: int = BAR_SECS) -> bool:
    """A bar stamped S covers [S, S+bar) and is final only at S+bar."""
    return now >= candle_start + bar_secs


def _valid_time(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def is_final(candle: Mapping, candle_start: int, now: float, bar_secs: int = BAR_SECS) -> bool:
    """Whether a stored bar may be used for a label at `now` (policy, see FINALITY_BASIS).

    The fetch that wrote it must have STARTED at least FINALITY_MARGIN_SECS after
    the bar closed, and not after `now`: a replay at `now` cannot use a fetch that
    had not begun. A missing or invalid vintage (legacy rows) is never final.
    """
    if not _is_complete(candle_start, now, bar_secs):
        return False
    requested_at = candle.get("requested_at")
    if not _valid_time(requested_at):
        return False
    return candle_start + bar_secs + FINALITY_MARGIN_SECS <= requested_at <= now


def _give_up(now: float, target: int, attempts: int) -> bool:
    return attempts + 1 >= MAX_RESOLVE_ATTEMPTS or now > target + UNAVAILABLE_GRACE_SECS


def resolve(
    signal_time: float,
    side: str,
    candles: Mapping[int, Dict],
    now: float,
    attempts: int = 0,
    h_bars: int = H_BARS,
    bar_secs: int = BAR_SECS,
) -> Resolution:
    """Attempt to label one signal.

    `candles` maps bucket-open epoch -> candle dict (`open`/`close` required).
    Only completed bars are consulted, and only the two bars the contract names:
    a missing target bar is never replaced by a later price.
    """
    entry_start = entry_candle_start(signal_time, bar_secs)
    exit_start = exit_candle_start(entry_start, h_bars, bar_secs)
    target = target_time(entry_start, h_bars, bar_secs)

    frame = dict(
        entry_candle_start=entry_start,
        exit_candle_start=exit_start,
        target_time=target,
    )

    if now < target:
        return Resolution(status="PENDING", reason="not_matured", **frame)

    def _missing(reason: str) -> Resolution:
        status = "UNAVAILABLE" if _give_up(now, target, attempts) else "PENDING"
        return Resolution(status=status, reason=reason, **frame)

    entry_candle = candles.get(entry_start)
    if entry_candle is None or not _is_complete(entry_start, now, bar_secs):
        return _missing("missing_entry_candle")
    if not is_final(entry_candle, entry_start, now, bar_secs):
        return _missing("unverified_entry_vintage")

    exit_candle = candles.get(exit_start)
    if exit_candle is None or not _is_complete(exit_start, now, bar_secs):
        return _missing("missing_exit_candle")
    if not is_final(exit_candle, exit_start, now, bar_secs):
        return _missing("unverified_exit_vintage")

    entry_price = float(entry_candle["open"])
    if entry_price <= 0:
        # Deterministic: retrying cannot make a non-positive price valid.
        return Resolution(status="UNAVAILABLE", reason="invalid_entry_price", **frame)

    target_price = float(exit_candle["close"])
    value = signed_return(entry_price, target_price, side)

    return Resolution(
        status="RESOLVED",
        outcome=classify(value),
        signed_return=value,
        entry_price=entry_price,
        target_price=target_price,
        price_observed_at=target,
        price_source=PRICE_SOURCE_LOCAL_CANDLES,
        finality_basis=FINALITY_BASIS,
        **frame,
    )
