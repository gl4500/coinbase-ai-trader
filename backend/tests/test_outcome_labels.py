"""Label-version-2 outcome contract.

Spec: docs/specs/2026-09-26-outcome-label-contract.md
"""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from services import outcome_labels as ol  # noqa: E402

HOUR = 3600
# 2026-01-01 00:00:00 UTC, an exact hour boundary.
_BOUNDARY = 1_767_225_600


def _candle(start, open_, close):
    return {
        "start_time": start,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "volume": 1.0,
    }


def _book(*candles):
    return {c["start_time"]: c for c in candles}


# ── Contract constants ────────────────────────────────────────────────────────


def test_contract_constants():
    assert ol.LABEL_VERSION == 2
    assert ol.H_BARS == 4
    assert ol.BAR_SECS == HOUR
    assert ol.WIN_THRESHOLD == 0.005
    assert ol.MAX_RESOLVE_ATTEMPTS == 5
    assert ol.UNAVAILABLE_GRACE_SECS == 7 * 86400


# ── Candle selection and boundary behaviour ───────────────────────────────────


def test_entry_candle_is_first_bar_starting_strictly_after_signal():
    # Mid-bar signal at 00:25 -> entry bar opens 01:00.
    assert ol.entry_candle_start(_BOUNDARY + 25 * 60) == _BOUNDARY + HOUR


def test_signal_exactly_on_boundary_takes_the_next_bar_not_that_bar():
    """A signal stamped exactly at T must not claim bar T: we cannot assume it
    preceded the trades inside that bar."""
    assert ol.entry_candle_start(_BOUNDARY) == _BOUNDARY + HOUR


def test_exit_bar_is_three_bars_after_entry_and_target_is_its_close():
    entry = ol.entry_candle_start(_BOUNDARY)
    assert ol.exit_candle_start(entry) == entry + 3 * HOUR
    assert ol.target_time(entry) == entry + 4 * HOUR


def test_entry_open_to_exit_close_spans_exactly_the_horizon():
    entry = ol.entry_candle_start(_BOUNDARY + 1)
    assert ol.target_time(entry) - entry == ol.H_BARS * HOUR


# ── Direction and classification ──────────────────────────────────────────────


def test_buy_return_is_unsigned_and_sell_is_inverted():
    assert ol.signed_return(100.0, 101.0, "BUY") == 0.01
    assert ol.signed_return(100.0, 101.0, "SELL") == -0.01
    assert ol.signed_return(100.0, 99.0, "SELL") == 0.01


def test_classify_thresholds_and_dead_zone():
    assert ol.classify(0.0051) == "WIN"
    assert ol.classify(-0.0051) == "LOSS"
    assert ol.classify(0.0) == "NEUTRAL"
    # Exactly on the boundary is NEUTRAL — the band is inclusive.
    assert ol.classify(0.005) == "NEUTRAL"
    assert ol.classify(-0.005) == "NEUTRAL"


def test_sell_signal_wins_when_price_falls():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry, 100.0, 100.0), _candle(entry + 3 * HOUR, 98.0, 98.0))
    r = ol.resolve(
        signal_time=_BOUNDARY, side="SELL", candles=book, now=ol.target_time(entry), attempts=0
    )
    assert r.status == "RESOLVED"
    assert r.outcome == "WIN"
    assert r.signed_return == 0.02


# ── The bug this contract exists to fix ───────────────────────────────────────


def test_delayed_processing_still_uses_the_target_time_price():
    """Processing 200 hours late must produce the same label as processing on
    time. This is the audit's 45.75h-mean-delay defect."""
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(
        _candle(entry, 100.0, 100.0),
        _candle(entry + 3 * HOUR, 101.0, 102.0),  # the target bar
        _candle(entry + 200 * HOUR, 500.0, 500.0),  # a much later bar
    )
    on_time = ol.resolve(
        signal_time=_BOUNDARY, side="BUY", candles=book, now=ol.target_time(entry), attempts=0
    )
    very_late = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=book,
        now=ol.target_time(entry) + 200 * HOUR,
        attempts=0,
    )
    assert on_time.outcome == very_late.outcome == "WIN"
    assert on_time.target_price == very_late.target_price == 102.0
    assert very_late.price_observed_at == ol.target_time(entry)
    # The 500.0 bar must never leak in.
    assert very_late.target_price != 500.0


def test_never_falls_back_to_a_later_price_when_target_bar_is_missing():
    """Missing target bar must not be papered over with any other price."""
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(
        _candle(entry, 100.0, 100.0),
        _candle(entry + 50 * HOUR, 130.0, 130.0),  # later bar exists
    )  # target bar absent
    r = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=book,
        now=ol.target_time(entry) + 50 * HOUR,
        attempts=0,
    )
    assert r.status == "PENDING"
    assert r.outcome is None
    assert r.target_price is None
    assert r.reason == "missing_exit_candle"


# ── Incomplete candles ────────────────────────────────────────────────────────


def test_exit_bar_present_but_not_yet_closed_is_not_resolvable():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry, 100.0, 100.0), _candle(entry + 3 * HOUR, 101.0, 101.0))
    # One second before the exit bar closes.
    r = ol.resolve(
        signal_time=_BOUNDARY, side="BUY", candles=book, now=ol.target_time(entry) - 1, attempts=0
    )
    assert r.status == "PENDING"
    assert r.reason == "not_matured"


def test_resolvable_at_the_exact_instant_the_exit_bar_closes():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry, 100.0, 100.0), _candle(entry + 3 * HOUR, 101.0, 101.0))
    r = ol.resolve(
        signal_time=_BOUNDARY, side="BUY", candles=book, now=ol.target_time(entry), attempts=0
    )
    assert r.status == "RESOLVED"


# ── Missing data, retries, terminal states ────────────────────────────────────


def test_missing_entry_candle_is_pending_then_unavailable_at_max_attempts():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry + 3 * HOUR, 101.0, 101.0))  # no entry bar
    now = ol.target_time(entry)

    early = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=now, attempts=0)
    assert early.status == "PENDING"
    assert early.reason == "missing_entry_candle"

    exhausted = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=book,
        now=now,
        attempts=ol.MAX_RESOLVE_ATTEMPTS - 1,
    )
    assert exhausted.status == "UNAVAILABLE"
    assert exhausted.reason == "missing_entry_candle"


def test_grace_window_expiry_makes_it_unavailable_even_with_attempts_left():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = {}
    past_grace = ol.target_time(entry) + ol.UNAVAILABLE_GRACE_SECS + 1
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=past_grace, attempts=0)
    assert r.status == "UNAVAILABLE"


def test_invalid_entry_price_is_terminal_immediately():
    """Retrying cannot fix a non-positive entry price."""
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry, 0.0, 0.0), _candle(entry + 3 * HOUR, 101.0, 101.0))
    r = ol.resolve(
        signal_time=_BOUNDARY, side="BUY", candles=book, now=ol.target_time(entry), attempts=0
    )
    assert r.status == "UNAVAILABLE"
    assert r.reason == "invalid_entry_price"


def test_unavailable_is_not_a_scoring_outcome():
    assert "UNAVAILABLE" not in ol.SCORING_OUTCOMES
    assert set(ol.SCORING_OUTCOMES) == {"WIN", "LOSS", "NEUTRAL"}


# ── Determinism / idempotency of the pure layer ───────────────────────────────


def test_resolution_is_deterministic_across_repeated_calls():
    entry = ol.entry_candle_start(_BOUNDARY)
    book = _book(_candle(entry, 100.0, 100.0), _candle(entry + 3 * HOUR, 101.0, 107.0))
    kwargs = dict(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=book,
        now=ol.target_time(entry) + 5 * HOUR,
        attempts=0,
    )
    first = ol.resolve(**kwargs)
    second = ol.resolve(**kwargs)
    assert first == second
