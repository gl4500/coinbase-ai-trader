"""What did the stop actually prevent?

The live ladder checks STOP_LOSS first, then MODEL_DOWN, then the trail, then MAX_HOLD.
This replays the SAME ladder with the stop rung removed, over real candles, so the
question "would this position have recovered?" is answered by price history rather than
by intuition.

Three disciplines carried over from the ATR work, because the same traps apply:

  * The trail rule is IMPORTED from production (`_compute_exit_threshold`), never
    reimplemented. A counterfactual against a rule I invented would measure my invention.
  * Intrabar ordering is enumerated, not assumed. Within one hourly bar the high and the
    low both occur; which came first decides whether the trail fired before the stop
    would have. Both orderings are reported and disagreement is surfaced.
  * A position that never goes green has NO trail (`peak_pnl_pct <= 0` returns -inf), so
    removing the stop leaves only MAX_HOLD. That is the whole mechanism under test.
"""

from __future__ import annotations

import pytest

from tools.exit_counterfactual import Bar, replay_without_stop

_H = 3_600_000


def _bars(rows):
    """rows: (open, high, low, close) per hour from the entry bar onward."""
    return [Bar(ts=i * _H, open=o, high=h, low=lo, close=c) for i, (o, h, lo, c) in enumerate(rows)]


def test_a_position_that_never_goes_green_has_no_trail_and_runs_to_max_hold():
    """The crux. _compute_exit_threshold returns -inf until peak_pnl_pct > 0, so with the
    stop removed nothing else can fire. If this ever exits early, the replay is not using
    the production rule."""
    bars = _bars([(100, 100, 95, 96)] + [(96, 97, 90, 92)] * 5)
    out = replay_without_stop(bars, entry_price=100.0, max_hold_bars=6, ordering="low_first")

    assert out.exit_reason == "MAX_HOLD"
    assert out.bars_held == 6
    assert out.peak_pnl_pct <= 0.0


def test_recovery_into_green_engages_the_trail_and_exits_on_giveback():
    """Once green the trail engages. With a peak of +20% the giveback is
    max(0.20*0.10, 2*0.006) = 0.02, so the exit threshold is +18%."""
    bars = _bars(
        [
            (100, 100, 92, 95),  # dips, never green
            (95, 120, 95, 119),  # peak +20%
            (119, 119, 110, 111),  # falls through +18%
        ]
    )
    out = replay_without_stop(bars, entry_price=100.0, max_hold_bars=10, ordering="high_first")

    assert out.exit_reason == "TRAIL_STOP"
    assert out.peak_pnl_pct == pytest.approx(0.20)
    assert out.exit_pnl_pct > 0.0, "a trail exit from a +20% peak must not be a loss"


def test_intrabar_ordering_is_enumerated_because_it_changes_the_answer():
    """One bar makes a new high AND falls below the resulting trail. Whether the trail
    fires this bar or the next depends on path order, which the data cannot tell us."""
    bars = _bars(
        [
            (100, 130, 100, 105),  # +30% high and a close back at +5%, same bar
            (105, 106, 104, 105),
        ]
    )
    high_first = replay_without_stop(
        bars, entry_price=100.0, max_hold_bars=5, ordering="high_first"
    )
    low_first = replay_without_stop(bars, entry_price=100.0, max_hold_bars=5, ordering="low_first")

    # high_first sets peak +30% then gives back through +27%; low_first sees the low
    # before the peak exists, so the trail cannot have fired on the low.
    assert high_first.exit_reason == "TRAIL_STOP"
    assert high_first.bars_held == 1
    assert low_first.bars_held >= 1
    assert (
        high_first.exit_pnl_pct != low_first.exit_pnl_pct
        or high_first.bars_held != low_first.bars_held
    )


def test_the_stop_is_genuinely_absent_from_the_replay():
    """Falsification: an 8% stop would have fired on bar 1. If this returns a ~-8% exit,
    the rung was not removed and every downstream number is meaningless."""
    bars = _bars([(100, 100, 80, 82)] + [(82, 83, 81, 82)] * 3)
    out = replay_without_stop(bars, entry_price=100.0, max_hold_bars=4, ordering="low_first")

    assert out.exit_reason == "MAX_HOLD"
    assert out.exit_pnl_pct == pytest.approx(-0.18, abs=0.01)


def test_empty_or_short_history_is_refused_rather_than_silently_truncated():
    """A replay over two bars is not evidence about a seven-day hold. Say so."""
    with pytest.raises(ValueError, match="at least one bar"):
        replay_without_stop([], entry_price=100.0, max_hold_bars=5, ordering="low_first")
    with pytest.raises(ValueError, match="ordering"):
        replay_without_stop(
            _bars([(1, 1, 1, 1)]), entry_price=100.0, max_hold_bars=5, ordering="sideways"
        )
    for bad in (0, -1, True):
        with pytest.raises(ValueError):
            replay_without_stop(
                _bars([(1, 1, 1, 1)]), entry_price=100.0, max_hold_bars=bad, ordering="low_first"
            )
    with pytest.raises(ValueError):
        replay_without_stop(
            _bars([(1, 1, 1, 1)]), entry_price=0.0, max_hold_bars=5, ordering="low_first"
        )


def test_incoherent_bars_are_refused():
    """A bar whose low exceeds its high describes no traversable path; the same guard the
    ATR probe needed after my own fixtures turned out incoherent."""
    with pytest.raises(ValueError, match="low"):
        replay_without_stop(
            _bars([(100, 90, 110, 100)]), entry_price=100.0, max_hold_bars=5, ordering="low_first"
        )


def test_the_replay_uses_production_exit_thresholds_not_a_local_copy(monkeypatch):
    """Non-vacuity: if the module reimplemented the trail, patching production would not
    change the result."""
    import tools.exit_counterfactual as mod

    calls = []
    real = mod._compute_exit_threshold

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(mod, "_compute_exit_threshold", spy)
    bars = _bars([(100, 115, 100, 114), (114, 115, 100, 101)])
    replay_without_stop(bars, entry_price=100.0, max_hold_bars=5, ordering="high_first")

    assert calls, "production _compute_exit_threshold was never consulted"
    assert all("peak_pnl_pct" in c for c in calls)
