"""Why does an 8% stop realise -9.49%?

51 live CNN `STOP_LOSS` exits average -9.494% against a configured 0.08, worst -13.94%.
The tick path (`WS_STOP_LOSS`, n=10) averages -8.323% with a best of -8.01%, so roughly 1.17
points separate the two paths. Two explanations have completely different remedies:

  GAP_AT_OPEN     the bar opened already below the stop. No execution speed helps; the
                  overshoot is a property of the instrument and an 8% stop is not
                  achievable on it. Remedy: document the real distribution, not a faster path.
  INTRABAR_CROSS  the bar opened above the stop and traded through it. A faster path could
                  in principle have filled nearer the level. Remedy: routing.

The discriminator is the exit bar's OPEN, which is observed rather than inferred -- unlike
the within-bar path, which is unknowable from OHLC. That is deliberately the only thing the
classification rests on.

What this CANNOT establish, stated here so no caller has to rediscover it:
  * `best_achievable_fill` for an intrabar cross assumes a fill AT the level. It ignores
    slippage and available size, so recoverable amounts derived from it are an UPPER BOUND,
    not an expectation.
  * bars are hourly; the live tick path is far finer. Crossing within an hour says a faster
    path had an opportunity, not that it would have taken it at the level.
  * a `LEVEL_NOT_REACHED` verdict does not mean the stop was wrong -- it means the crossing
    is not in the bar where the exit was RECORDED, which is itself evidence of latency
    spanning bars rather than within one.
  * the 8% level is knowable in advance, which is what makes this measurable at all. The
    same method does NOT transfer to the ATR trail, whose level moves with the peak.
"""

from __future__ import annotations

import pytest

from tools.stop_overshoot_probe import CONFIGURED_STOP_PCT, classify_stop_fill


def test_a_bar_that_opens_below_the_stop_is_a_gap_no_speed_could_beat():
    """Entry 100, stop 92, bar opens at 88. The best any path achieves is the open."""
    got = classify_stop_fill(entry_price=100.0, bar_open=88.0, bar_low=85.0, exit_price=88.0)
    assert got.category == "GAP_AT_OPEN"
    assert got.stop_level == pytest.approx(92.0)
    assert got.best_achievable_fill == pytest.approx(88.0)
    # the 4 points below the stop are unavoidable, and nothing is attributable to latency
    assert got.unavoidable_pts == pytest.approx(-4.0)
    assert got.attributable_pts == pytest.approx(0.0)


def test_a_bar_that_opens_above_and_trades_through_is_an_intrabar_cross():
    """Entry 100, stop 92, bar opens 99 and dips to 90: a faster path had an opportunity."""
    got = classify_stop_fill(entry_price=100.0, bar_open=99.0, bar_low=90.0, exit_price=90.5)
    assert got.category == "INTRABAR_CROSS"
    assert got.best_achievable_fill == pytest.approx(92.0)
    assert got.unavoidable_pts == pytest.approx(0.0)
    assert got.attributable_pts == pytest.approx(-1.5)


def test_a_level_never_reached_in_the_exit_bar_is_not_silently_called_a_cross():
    """If the bar never touched 92, the crossing happened in some other bar. Calling that an
    intrabar cross would invent an opportunity inside a bar that never offered one."""
    got = classify_stop_fill(entry_price=100.0, bar_open=99.0, bar_low=95.0, exit_price=91.0)
    assert got.category == "LEVEL_NOT_REACHED"
    assert got.best_achievable_fill is None
    assert got.attributable_pts is None


def test_an_exit_at_or_better_than_the_level_attributes_nothing():
    """A fill at the level is the ideal outcome; the measure must not read as a negative."""
    got = classify_stop_fill(entry_price=100.0, bar_open=99.0, bar_low=90.0, exit_price=92.0)
    assert got.category == "INTRABAR_CROSS"
    assert got.attributable_pts == pytest.approx(0.0)


def test_a_gap_is_decided_by_the_open_alone_not_by_where_it_filled():
    """The open is the observed quantity. A gapped bar stays gapped even when the recorded
    fill is far from it -- that residual is attributable, and must not be hidden inside the
    unavoidable column."""
    got = classify_stop_fill(entry_price=100.0, bar_open=88.0, bar_low=80.0, exit_price=82.0)
    assert got.category == "GAP_AT_OPEN"
    assert got.unavoidable_pts == pytest.approx(-4.0)
    assert got.attributable_pts == pytest.approx(-6.0)


def test_the_two_columns_reconstruct_the_realised_overshoot_exactly():
    """Non-vacuity for the split: unavoidable + attributable must equal the whole overshoot
    past the stop, or the decomposition is losing or inventing points."""
    for bar_open, bar_low, exit_price in [
        (88.0, 80.0, 82.0),
        (99.0, 90.0, 90.5),
        (92.0, 91.0, 91.5),
    ]:
        got = classify_stop_fill(
            entry_price=100.0, bar_open=bar_open, bar_low=bar_low, exit_price=exit_price
        )
        overshoot = (exit_price - got.stop_level) / 100.0 * 100.0
        assert got.unavoidable_pts + got.attributable_pts == pytest.approx(overshoot)


def test_the_boundary_case_of_opening_exactly_on_the_stop_counts_as_a_gap():
    """An open exactly at the level is a touch, not a miss: no faster path improves on it,
    so it belongs with the unavoidable cases. Codex made the same point about inclusive
    trigger semantics on the ATR probe, and the same reasoning applies here."""
    got = classify_stop_fill(entry_price=100.0, bar_open=92.0, bar_low=85.0, exit_price=92.0)
    assert got.category == "GAP_AT_OPEN"
    assert got.unavoidable_pts == pytest.approx(0.0)


def test_the_configured_stop_matches_production_rather_than_a_local_constant():
    """A probe that hardcodes its own 8% would keep agreeing with itself after production
    changed. Invariant #3 fixes this at 0.08 for the $50k capital-at-risk math."""
    from agents.cnn_agent import _CNN_STOP_LOSS_PCT

    assert CONFIGURED_STOP_PCT == _CNN_STOP_LOSS_PCT


def test_incoherent_bars_are_rejected_rather_than_scored():
    """low above open is not a bar. Scoring it would produce a confident wrong verdict."""
    with pytest.raises(ValueError):
        classify_stop_fill(entry_price=100.0, bar_open=95.0, bar_low=97.0, exit_price=92.0)
    with pytest.raises(ValueError):
        classify_stop_fill(entry_price=0.0, bar_open=95.0, bar_low=90.0, exit_price=92.0)
