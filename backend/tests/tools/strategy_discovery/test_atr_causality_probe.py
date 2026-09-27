"""The offline ATR-causality probe: variants of the exit simulation, measured in isolation.

The probe re-implements the exit walk with three knobs (ATR lag, intrabar ordering, gap fill) so
each proposed change can be measured on its own. That re-implementation is the whole risk: a
probe that has quietly drifted from `labels._simulate_one` measures ITSELF, not the production
label, and would report differences that do not exist.

So the anchor test comes first and is the one that matters most: the LEGACY variant must
reproduce `labels._simulate_one` **bit-exactly** via `float.hex()`, on a frame containing stop,
trail and horizon exits. Every other number the probe produces is only as trustworthy as that.

Nothing here changes production label semantics. The probe is additive and read-only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.atr_causality_probe import (
    COMBINED,
    GAP_ONLY,
    LAG_ONLY,
    LEGACY,
    ORDERING_ONLY,
    VariantSpec,
    bracket_orderings,
    simulate_variant,
)

_CONFIG = {
    "stop_loss_pct": 0.08,
    "atr_trail_floor": 0.06,
    "max_hold_bars": 168,
    "round_trip_fee": 0.012,
}
# Fee-free, so the counterexample's arithmetic is the exit price and nothing else.
_NO_FEE = dict(_CONFIG, round_trip_fee=0.0)


def _arrays(bars, *, entry_close=100.0, atr=0.06):
    """(opens, closes, highs, lows, atr_pcts) for an entry bar plus `bars` of (high, low, close).

    Row 0 is the entry bar: the trade enters at its close, so its own high and low never matter.
    """
    opens = [entry_close]
    closes = [entry_close]
    highs = [entry_close]
    lows = [entry_close]
    for high, low, close in bars:
        opens.append(close)  # overwritten by callers that care about the open
        closes.append(close)
        highs.append(high)
        lows.append(low)
    n = len(closes)
    atrs = [atr] * n if not isinstance(atr, (list, tuple)) else list(atr)
    return (
        np.array(opens, dtype="float64"),
        np.array(closes, dtype="float64"),
        np.array(highs, dtype="float64"),
        np.array(lows, dtype="float64"),
        np.array(atrs, dtype="float64"),
    )


def _production_frame():
    """A frame that exercises all three exit kinds, for the bit-exact anchor."""
    rng = np.random.default_rng(5)
    n = 400
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.02, size=n))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.015, size=n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.015, size=n)))
    return pd.DataFrame(
        {
            "ts": (np.arange(n, dtype="int64") * 3_600_000),
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "atr14_pct": np.full(n, 0.05),
        }
    )


def test_the_legacy_variant_reproduces_the_production_simulation_bit_exactly():
    """THE anchor. Without this the probe measures its own re-implementation.

    `float.hex()` rather than a tolerance: a relative tolerance would hide exactly the small
    arithmetic drift a re-implementation is prone to, and the point of the probe is to attribute
    differences to a named cause.
    """
    from tools.strategy_discovery.labels import _simulate_one

    frame = _production_frame()
    closes = frame["close"].to_numpy(dtype="float64")
    highs = frame["high"].to_numpy(dtype="float64")
    lows = frame["low"].to_numpy(dtype="float64")
    atrs = frame["atr14_pct"].to_numpy(dtype="float64")
    opens = frame["open"].to_numpy(dtype="float64")

    kinds = set()
    compared = 0
    for entry_idx in range(0, 300):
        for horizon in (1, 4, 24):
            expected = _simulate_one(
                entry_idx=entry_idx,
                horizon=horizon,
                closes=closes,
                highs=highs,
                lows=lows,
                atr_pcts=atrs,
                stop_loss_pct=_CONFIG["stop_loss_pct"],
                atr_trail_floor=_CONFIG["atr_trail_floor"],
                max_hold_bars=_CONFIG["max_hold_bars"],
                round_trip_fee=_CONFIG["round_trip_fee"],
            )
            actual = simulate_variant(
                entry_idx=entry_idx,
                horizon=horizon,
                opens=opens,
                closes=closes,
                highs=highs,
                lows=lows,
                atr_pcts=atrs,
                config=_CONFIG,
                spec=LEGACY,
            )
            if expected.pnl != expected.pnl:  # NaN: no room for the horizon
                assert actual.pnl != actual.pnl
                continue
            assert actual.pnl.hex() == expected.pnl.hex(), (
                f"row {entry_idx} h{horizon}: probe {actual.pnl!r} vs production {expected.pnl!r}"
            )
            # Production names it `exit_offset`; the probe names it `bars_held`. Same quantity,
            # compared explicitly so the rename cannot hide a difference.
            assert actual.bars_held == expected.exit_offset
            assert actual.exit_kind == expected.exit_kind
            kinds.add(expected.exit_kind)
            compared += 1

    assert compared > 500, f"only {compared} records compared; the anchor needs real coverage"
    assert kinds == {"stop", "trail", "horizon"}, (
        f"anchor exercised only {sorted(kinds)}; a variant that broke an unexercised branch "
        f"would slip through"
    )


def test_the_dominance_counterexample_is_reproduced_by_the_probe():
    """Codex c3c5acc8, fee-free so the numbers are the exit prices.

    high_before_low trails out at bar 1 (112.80); low_before_high survives and reaches the
    horizon close (145.00). The regression for a claim I got wrong.
    """
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0), (150.0, 114.0, 145.0)])

    first = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=VariantSpec("b1", ordering="high_before_low"),
    )
    second = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=VariantSpec("b2", ordering="low_before_high"),
    )

    assert first.exit_kind == "trail"
    assert first.pnl == pytest.approx(0.1280)
    assert second.exit_kind == "horizon"
    assert second.pnl == pytest.approx(0.4500)
    assert second.pnl > first.pnl, (
        "the whole point: low_before_high is NOT dominated, so neither ordering is conservative"
    )


def test_a_horizon_record_can_itself_be_ordering_dependent():
    """Codex df49b175: survival TO the horizon depends on the ordering, so a horizon record is
    not exempt from the ordering blocker."""
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0), (150.0, 114.0, 145.0)])
    report = bracket_orderings(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        atr_lag_bars=0,
    )
    assert report.agree is False
    assert {report.high_first.exit_kind, report.low_first.exit_kind} == {"trail", "horizon"}
    assert report.lower_bound_pnl == pytest.approx(0.1280)
    assert report.lower_bound_pnl <= report.high_first.pnl
    assert report.lower_bound_pnl <= report.low_first.pnl


def test_the_lag_changes_the_exit_when_the_atr_straddles_the_trigger():
    """A falling ATR: contemporaneous is below the drop, the previous bar's is above it."""
    # bar 1 drops 7% from the peak. atr at bar 1 is 0.05 (fires), at bar 0 is 0.09 (does not).
    opens, closes, highs, lows, atrs = _arrays(
        [(100.0, 93.0, 94.0), (100.0, 99.0, 99.5)], atr=[0.09, 0.05, 0.05]
    )
    legacy = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=LEGACY,
    )
    lagged = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=LAG_ONLY,
    )
    assert legacy.exit_kind == "trail", "fixture is vacuous: the contemporaneous ATR must fire"
    assert lagged.exit_kind != "trail", "the lagged ATR (0.09) must not fire on a 7% drop"
    assert lagged.pnl != legacy.pnl


def test_the_lag_changes_nothing_when_the_atr_is_constant():
    """Isolation: attributing an effect to the lag requires it to do nothing when it cannot."""
    frame = _production_frame()  # constant atr14_pct
    args = dict(
        opens=frame["open"].to_numpy(dtype="float64"),
        closes=frame["close"].to_numpy(dtype="float64"),
        highs=frame["high"].to_numpy(dtype="float64"),
        lows=frame["low"].to_numpy(dtype="float64"),
        atr_pcts=frame["atr14_pct"].to_numpy(dtype="float64"),
        config=_CONFIG,
    )
    differing = 0
    for entry_idx in range(0, 200):
        legacy = simulate_variant(entry_idx=entry_idx, horizon=24, spec=LEGACY, **args)
        lagged = simulate_variant(entry_idx=entry_idx, horizon=24, spec=LAG_ONLY, **args)
        if legacy.pnl == legacy.pnl and lagged.pnl != legacy.pnl:
            differing += 1
    assert differing == 0, f"{differing} records changed under a constant ATR"


def test_the_gap_variant_fills_at_the_open_when_the_bar_gapped_through_the_stop():
    """The current rule fills AT the stop level even when the bar opened below it, which reports
    a better price than the data supports."""
    opens, closes, highs, lows, atrs = _arrays([(93.0, 85.0, 88.0)])
    opens[1] = 90.0  # already below the 92.0 stop level at the open
    legacy = simulate_variant(
        entry_idx=0,
        horizon=1,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=LEGACY,
    )
    gapped = simulate_variant(
        entry_idx=0,
        horizon=1,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=GAP_ONLY,
    )
    assert legacy.exit_kind == "stop"
    assert legacy.pnl == pytest.approx(-0.08), "legacy fills at the level"
    assert gapped.exit_kind == "stop"
    assert gapped.gapped is True
    assert gapped.pnl == pytest.approx(-0.10), "gapped fill is the open, 90.0, not the level 92.0"
    assert gapped.pnl < legacy.pnl


def test_the_gap_variant_changes_nothing_when_no_bar_gaps_through():
    """Isolation, the other half."""
    opens, closes, highs, lows, atrs = _arrays([(93.0, 85.0, 88.0)])
    opens[1] = 99.0  # opens above the stop level
    legacy = simulate_variant(
        entry_idx=0,
        horizon=1,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=LEGACY,
    )
    gapped = simulate_variant(
        entry_idx=0,
        horizon=1,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=GAP_ONLY,
    )
    assert gapped.pnl.hex() == legacy.pnl.hex()
    assert gapped.gapped is False


def test_the_variant_registry_names_every_isolated_change_and_the_combination():
    """The probe must report variants separately, so an effect can be attributed to a cause."""
    assert LEGACY.atr_lag_bars == 0 and LEGACY.ordering == "high_before_low"
    assert LEGACY.gap_fill is False
    assert LAG_ONLY.atr_lag_bars == 1 and LAG_ONLY.gap_fill is False
    assert ORDERING_ONLY.ordering == "low_before_high" and ORDERING_ONLY.atr_lag_bars == 0
    assert GAP_ONLY.gap_fill is True and GAP_ONLY.atr_lag_bars == 0
    assert COMBINED.atr_lag_bars == 1 and COMBINED.gap_fill is True
    assert len({LEGACY.name, LAG_ONLY.name, ORDERING_ONLY.name, GAP_ONLY.name, COMBINED.name}) == 5


def test_an_unknown_ordering_is_refused_rather_than_defaulted():
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0)])
    with pytest.raises(ValueError, match="ordering"):
        simulate_variant(
            entry_idx=0,
            horizon=1,
            opens=opens,
            closes=closes,
            highs=highs,
            lows=lows,
            atr_pcts=atrs,
            config=_CONFIG,
            spec=VariantSpec("bad", ordering="whenever"),
        )


def test_a_negative_or_non_integral_lag_is_refused():
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0)])
    for bad in (-1, 1.0, True):
        with pytest.raises(ValueError, match="atr_lag_bars"):
            simulate_variant(
                entry_idx=0,
                horizon=1,
                opens=opens,
                closes=closes,
                highs=highs,
                lows=lows,
                atr_pcts=atrs,
                config=_CONFIG,
                spec=VariantSpec("bad", atr_lag_bars=bad),
            )


def test_the_lower_bound_carries_its_own_branch_metadata():
    """Codex 2cc2feec. A smaller-PnL path must carry ITS OWN exit, timing and occupancy.

    Pairing one ordering's PnL with the other's exit kind or bar would describe a trade neither
    ordering produces -- a fabricated record, which is worse than either branch alone. In the
    counterexample the lower bound is the high_before_low trail at bar 1, so the reported branch
    must be that one and its result must be that object, not the horizon survivor's.
    """
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0), (150.0, 114.0, 145.0)])
    report = bracket_orderings(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        atr_lag_bars=0,
    )
    assert report.lower_bound_ordering == "high_before_low"
    assert report.lower_bound_result is report.high_first
    assert report.lower_bound_result.pnl == report.lower_bound_pnl
    assert report.lower_bound_result.exit_kind == "trail"
    assert report.lower_bound_result.bars_held == 1
    # And the discriminating negative: it must NOT have taken the survivor's metadata.
    assert report.lower_bound_result.exit_kind != report.low_first.exit_kind
    assert report.low_first.exit_kind == "horizon"


def test_both_branches_are_preserved_even_when_they_agree():
    """Review needs both paths regardless, so agreement must not collapse them."""
    opens, closes, highs, lows, atrs = _arrays([(100.5, 99.5, 100.0), (100.5, 99.5, 100.0)])
    report = bracket_orderings(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        atr_lag_bars=0,
    )
    assert report.agree is True
    assert report.high_first is not None and report.low_first is not None
    assert report.lower_bound_result in (report.high_first, report.low_first)
    assert report.high_first.pnl == report.low_first.pnl


def test_a_level_created_mid_bar_is_not_a_gap_through():
    """Codex cb898726, a real defect in the first version of the probe.

    With prior peak 100, open 100, high 120, low 95 and a 6% floor, the level in force at the
    open was 94 and the open never gapped through it. The triggering level of 112.8 only came
    into being once this bar's high raised the peak. The earlier version compared the open
    against that later level and reported a fabricated fill at 100.0 (pnl 0.0000) instead of
    112.8 (+0.1280) -- WORSE than either honest reading, which is its own kind of wrong.
    """
    opens, closes, highs, lows, atrs = _arrays([(120.0, 95.0, 115.0)])
    opens[1] = 100.0
    args = dict(
        entry_idx=0,
        horizon=1,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
    )
    legacy = simulate_variant(spec=LEGACY, **args)
    gapped = simulate_variant(spec=GAP_ONLY, **args)

    assert legacy.exit_kind == "trail" and legacy.exit_price == pytest.approx(112.8)
    assert gapped.gapped is False, "no level in force at the open was breached"
    assert gapped.exit_price == pytest.approx(112.8)
    assert gapped.pnl.hex() == legacy.pnl.hex(), (
        "the gap variant must be bit-identical here; it differed only because it treated a "
        "mid-bar level as if it had existed at the open"
    )


def test_a_trail_level_in_force_at_the_open_still_registers_a_gap():
    """The other half: when the level DID exist at the open and the bar opened below it, the gap
    is real and the fill is the open. Without this the fix above could have disabled the
    diagnostic entirely and still passed."""
    # Bar 1 lifts the peak to 120 (level 112.8) and survives. Bar 2 opens at 105, already below
    # that pre-existing level, so it is a genuine gap-through.
    opens, closes, highs, lows, atrs = _arrays([(120.0, 115.0, 118.0), (118.0, 100.0, 104.0)])
    opens[1] = 116.0
    opens[2] = 105.0
    args = dict(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
    )
    legacy = simulate_variant(spec=LEGACY, **args)
    gapped = simulate_variant(spec=GAP_ONLY, **args)

    assert legacy.exit_kind == "trail" and legacy.exit_price == pytest.approx(112.8)
    assert gapped.gapped is True, "the 112.8 level was in force at bar 2's open of 105"
    assert gapped.exit_price == pytest.approx(105.0)
    assert gapped.pnl < legacy.pnl


def test_an_opening_breach_beats_a_later_stop_in_the_same_bar():
    """Codex c6c670ee. A level already breached at the open is a TIMED event; the full-bar extrema
    are not.

    Bar 1 lifts the peak to 120 (trail level 112.8) and survives. Bar 2 OPENS at 105 -- already
    through that level -- and only later falls to 90, below the 92 stop. The earlier version
    reported stop @ 92 (-0.0800), an exit that could not have happened because the position was
    already out at 105. Legacy keeps the full-bar stop-first behaviour unchanged.
    """
    opens = np.array([100.0, 116.0, 105.0], dtype="float64")
    closes = np.array([100.0, 118.0, 95.0], dtype="float64")
    highs = np.array([100.0, 120.0, 106.0], dtype="float64")
    lows = np.array([100.0, 115.0, 90.0], dtype="float64")
    atrs = np.array([0.06, 0.06, 0.06], dtype="float64")
    args = dict(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
    )
    legacy = simulate_variant(spec=LEGACY, **args)
    gapped = simulate_variant(spec=GAP_ONLY, **args)

    assert legacy.exit_kind == "stop" and legacy.exit_price == pytest.approx(92.0), (
        "legacy must be untouched by the opening-event rule"
    )
    assert gapped.exit_kind == "trail", "the trail level was already through at the open"
    assert gapped.exit_price == pytest.approx(105.0)
    assert gapped.gapped is True
    assert gapped.bars_held == 2
    assert gapped.pnl == pytest.approx(0.05)
    assert gapped.pnl > legacy.pnl, (
        "here the honest answer is BETTER than legacy -- the gap variant is not a one-way "
        "pessimism knob, which is itself worth knowing"
    )


def test_a_stop_breached_at_the_open_keeps_priority_over_a_trail_also_breached():
    """Both levels through at the open: the stop wins, matching the live exit ladder."""
    opens = np.array([100.0, 116.0, 80.0], dtype="float64")
    closes = np.array([100.0, 118.0, 82.0], dtype="float64")
    highs = np.array([100.0, 120.0, 83.0], dtype="float64")
    lows = np.array([100.0, 115.0, 79.0], dtype="float64")
    atrs = np.array([0.06, 0.06, 0.06], dtype="float64")
    gapped = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=GAP_ONLY,
    )
    assert gapped.exit_kind == "stop"
    assert gapped.exit_price == pytest.approx(80.0)
    assert gapped.gapped is True


def test_low_before_high_is_a_delayed_update_policy_not_a_literal_path():
    """Codex cb898726. Pins the limitation instead of leaving it implicit.

    Bar 1: low 96 clears the level in force at the open (94), then the high lifts the peak to 120
    so the level becomes 112.8 -- and the close at 100 is BELOW it. A literal O-L-H-C traversal
    would exit on that descent, within bar 1. This policy defers the raised peak to the next bar,
    so the exit lands on bar 2 instead. Asserted so the choice is visible and cannot drift.
    """
    opens = np.array([100.0, 100.0, 100.0], dtype="float64")
    closes = np.array([100.0, 100.0, 100.0], dtype="float64")
    highs = np.array([100.0, 120.0, 100.0], dtype="float64")
    lows = np.array([100.0, 96.0, 100.0], dtype="float64")
    atrs = np.array([0.06, 0.06, 0.06], dtype="float64")
    result = simulate_variant(
        entry_idx=0,
        horizon=2,
        opens=opens,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atrs,
        config=_NO_FEE,
        spec=VariantSpec("lbh", ordering="low_before_high"),
    )
    assert result.exit_kind == "trail"
    assert result.bars_held == 2, (
        "delayed update: the peak raised by bar 1's high only bites from bar 2. A literal "
        "O-L-H-C model would have exited during bar 1 on the high-to-close descent."
    )
    assert result.exit_price == pytest.approx(112.8)
