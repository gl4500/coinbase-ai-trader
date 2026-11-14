"""Label-version-2 outcome contract.

Spec: docs/specs/2026-09-26-outcome-label-contract.md
"""

import os
import sys

import pytest

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
        # Fetched once the bar had settled under the vintage gate
        # (test_candle_vintage.py owns the gate's own behaviour).
        "requested_at": start + HOUR + ol.FINALITY_MARGIN_SECS,
    }


def _settled(entry):
    """Earliest instant the exit bar can be labelled: its close plus the margin."""
    return ol.target_time(entry) + ol.FINALITY_MARGIN_SECS


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
        signal_time=_BOUNDARY, side="SELL", candles=book, now=_settled(entry), attempts=0
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
        signal_time=_BOUNDARY, side="BUY", candles=book, now=_settled(entry), attempts=0
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


def test_not_resolvable_at_the_close_only_once_a_settled_fetch_has_started():
    """Clock maturity alone no longer suffices: a fetch at the close may hold a
    provisional bar. The earliest label is close + FINALITY_MARGIN_SECS."""
    entry = ol.entry_candle_start(_BOUNDARY)
    exit_bar = _candle(entry + 3 * HOUR, 101.0, 101.0)
    at_close = dict(exit_bar, requested_at=ol.target_time(entry))
    book = _book(_candle(entry, 100.0, 100.0), at_close)
    r = ol.resolve(
        signal_time=_BOUNDARY, side="BUY", candles=book, now=ol.target_time(entry), attempts=0
    )
    assert (r.status, r.reason) == ("PENDING", "unverified_exit_vintage")

    book = _book(_candle(entry, 100.0, 100.0), exit_bar)
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_settled(entry), attempts=0)
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
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_settled(entry), attempts=0)
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


@pytest.mark.parametrize("field,start", [("open", 1767229200), ("close", 1767240000)])
@pytest.mark.parametrize("bad_price", [None, "invalid", float("nan"), float("inf"), -1, 0])
def test_invalid_candle_prices_never_become_scoring_labels(field, start, bad_price):
    book = {
        1767229200: {"open": 100.0, "requested_at": 1767229200 + HOUR + ol.FINALITY_MARGIN_SECS},
        1767240000: {"close": 102.0, "requested_at": 1767240000 + HOUR + ol.FINALITY_MARGIN_SECS},
    }
    book[start][field] = bad_price
    result = ol.resolve(1767225600, "BUY", book, now=1767243600 + ol.FINALITY_MARGIN_SECS)
    assert result.status == "UNAVAILABLE"
    assert result.outcome is None
    assert result.signed_return is None


def test_overflowing_return_never_becomes_a_win():
    book = {
        1767229200: {"open": 1e-300, "requested_at": 1767229200 + HOUR + ol.FINALITY_MARGIN_SECS},
        1767240000: {"close": 1e300, "requested_at": 1767240000 + HOUR + ol.FINALITY_MARGIN_SECS},
    }
    result = ol.resolve(1767225600, "BUY", book, now=1767243600 + ol.FINALITY_MARGIN_SECS)
    assert result.status == "UNAVAILABLE"
    assert result.outcome is None


# ── Per-label request-start audit metadata (WS1) ──────────────────────────────
# These record which fetch's request START justified each bar under the
# finality policy. They are policy audit metadata, not proof of finality and
# not a unique identity of the response content.


def _requested(candle, at):
    return {**candle, "requested_at": at}


def test_resolved_label_records_the_request_start_of_each_bar_used():
    entry = ol.entry_candle_start(_BOUNDARY)
    exit_ = ol.exit_candle_start(entry)
    entry_req = entry + HOUR + ol.FINALITY_MARGIN_SECS + 11.5
    exit_req = _settled(entry) + 37.25
    book = _book(
        _requested(_candle(entry, 100.0, 100.0), entry_req),
        _requested(_candle(exit_, 101.0, 102.0), exit_req),
    )
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=exit_req + 1)
    assert r.status == "RESOLVED"
    assert r.entry_requested_at == entry_req
    assert r.exit_requested_at == exit_req
    assert r.entry_requested_at != r.exit_requested_at
    assert r.finality_basis == ol.FINALITY_BASIS


def test_recorded_request_starts_satisfy_the_finality_policy():
    entry = ol.entry_candle_start(_BOUNDARY)
    exit_ = ol.exit_candle_start(entry)
    now = _settled(entry) + 500
    book = _book(
        _requested(_candle(entry, 100.0, 100.0), entry + HOUR + ol.FINALITY_MARGIN_SECS),
        _requested(_candle(exit_, 101.0, 102.0), now),
    )
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=now)
    assert r.status == "RESOLVED"
    for start, at in ((entry, r.entry_requested_at), (exit_, r.exit_requested_at)):
        assert start + HOUR + ol.FINALITY_MARGIN_SECS <= at <= now


def test_single_bar_horizon_records_that_bar_for_both_ends():
    entry = ol.entry_candle_start(_BOUNDARY)
    req = entry + HOUR + ol.FINALITY_MARGIN_SECS + 3
    book = _book(_requested(_candle(entry, 100.0, 102.0), req))
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=req, h_bars=1)
    assert r.status == "RESOLVED"
    assert r.entry_requested_at == r.exit_requested_at == req


def test_unresolved_results_carry_no_request_starts():
    entry = ol.entry_candle_start(_BOUNDARY)
    exit_ = ol.exit_candle_start(entry)
    entry_only = _book(_candle(entry, 100.0, 100.0))
    bad_price = _book(_candle(entry, 0.0, 100.0), _candle(exit_, 101.0, 102.0))
    results = [
        ol.resolve(_BOUNDARY, "BUY", entry_only, now=_settled(entry) - 1),  # not matured
        ol.resolve(_BOUNDARY, "BUY", entry_only, now=_settled(entry)),  # PENDING
        ol.resolve(_BOUNDARY, "BUY", entry_only, now=_settled(entry), attempts=99),
        ol.resolve(_BOUNDARY, "BUY", bad_price, now=_settled(entry)),  # invalid price
    ]
    assert [r.status for r in results] == ["PENDING", "PENDING", "UNAVAILABLE", "UNAVAILABLE"]
    for r in results:
        assert r.entry_requested_at is None
        assert r.exit_requested_at is None
