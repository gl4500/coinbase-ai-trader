"""The replay's ordered instants. Contract §9.1 and §9.5.

Entry decisions become available at a bar's CLOSE, because the features a rule reads are
close-derived. Exits land at `exit_observable_at`, also a bar close. The pre-integration
loop iterated raw bar STARTS and did closing, entry evaluation and occupancy sampling at
each one, which dated every entry a bar early and left an exit on the final bar unreachable.

This module deliberately emits NO per-position events. An earlier design did, and it would
have replaced the existing `-cumulative_profit_deflated` ranking with alphabetical product
order under a shared cap, and realized endpoints that were never entered. Ranking, cap and
the per-product constraint stay in `portfolio_sim`, which processes each instant as a batch.
"""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.replay_timeline import (  # noqa: E402
    bar_availability_instants,
    close_checkpoints,
    decision_instants,
    ordered_instants,
)

_BAR = 3_600_000


# ── decision instants are bar CLOSES, over participating products only ───────


def test_decision_instants_are_bar_closes_not_bar_starts():
    """Features are close-derived, so a bar's rule may only fire at that bar's CLOSE.
    Evaluating at the start uses information that does not exist yet and dates every entry
    one bar early."""
    assert decision_instants({"BTC-USD": [0, _BAR]}, bar_duration_ms=_BAR) == [
        _BAR,
        2 * _BAR,
    ]


def test_decision_instants_are_the_unique_union_across_products():
    """§9.5: one sample per unique master availability timestamp, matching the existing
    union-of-product-times convention -- not one per product row, which would change the
    denominator of pct_slots_full and mean_concurrent for a multi-product subset."""
    instants = decision_instants(
        {"BTC-USD": [0, _BAR], "ETH-USD": [0, 86_400_000]}, bar_duration_ms=_BAR
    )
    assert instants == sorted(set(instants))
    assert instants.count(_BAR) == 1
    assert instants == [_BAR, 2 * _BAR, 86_400_000 + _BAR]


def test_only_participating_products_contribute_instants():
    """Codex 0c0a6731: including every supplied product would let an UNRELATED input change
    the metric denominator. A product with no profile in the subset is not participating."""
    grid = {"BTC-USD": [0, _BAR], "DOGE-USD": [0, _BAR, 2 * _BAR, 3 * _BAR]}
    everyone = decision_instants(grid, bar_duration_ms=_BAR)
    participating = decision_instants(grid, bar_duration_ms=_BAR, participating=["BTC-USD"])
    assert participating == [_BAR, 2 * _BAR]
    assert len(participating) < len(everyone)


def test_a_participating_product_absent_from_the_grid_is_refused():
    """Silently contributing nothing would understate the denominator without any signal."""
    with pytest.raises(ValueError, match="ETH-USD"):
        decision_instants({"BTC-USD": [0]}, bar_duration_ms=_BAR, participating=["ETH-USD"])


def test_bar_availability_instants_maps_each_row_to_its_own_close():
    """The per-product lookup a consumer needs to pair an instant with the row that becomes
    decidable at it, and to cross-check each record's `entry_available_at`."""
    assert bar_availability_instants([0, _BAR, 3 * _BAR], [0, 1, 2], bar_duration_ms=_BAR) == {
        _BAR: 0,
        2 * _BAR: 1,
        4 * _BAR: 2,
    }


def test_bar_availability_instants_requires_source_ids_rather_than_enumerating():
    """Codex baa57071. On a FILTERED frame a position is not a source row id, and an
    endpoint's `entry_row_id` is a source row id -- so enumerating here would pair an instant
    with the wrong row the moment a caller passed a filtered sequence. The gap-carrying case
    is the whole point: retained rows 0, 3, 7 must map to 0, 3, 7, never to 0, 1, 2."""
    assert bar_availability_instants([0, 3 * _BAR, 7 * _BAR], [0, 3, 7], bar_duration_ms=_BAR) == {
        _BAR: 0,
        4 * _BAR: 3,
        8 * _BAR: 7,
    }


def test_mismatched_row_ids_and_bar_starts_are_refused():
    with pytest.raises(ValueError, match="describe the same rows"):
        bar_availability_instants([0, _BAR], [0], bar_duration_ms=_BAR)


@pytest.mark.parametrize("bad", [[0, 0], [3, 0], [0, True], [0, 1.5], [-1, 0]])
def test_malformed_source_row_ids_are_refused(bad):
    starts = [i * _BAR for i in range(len(bad))]
    with pytest.raises(ValueError):
        bar_availability_instants(starts, bad, bar_duration_ms=_BAR)


# ── checkpoints exist so a due position can be examined, and nothing more ────


def test_a_checkpoint_after_the_last_decision_instant_is_kept():
    """The dropped-PnL case. An exit on the final bar is observable at
    `last_start + bar_duration`, later than every decision instant in a one-bar frame, so
    without its own instant that position is never examined and its PnL vanishes."""
    ordered = ordered_instants(
        decision_instants({"BTC-USD": [0]}, bar_duration_ms=_BAR),
        close_checkpoints([2 * _BAR]),
    )
    assert ordered[-1] == (2 * _BAR, False)


def test_a_checkpoint_coinciding_with_a_decision_yields_one_decision_instant():
    """At a single instant the replay closes, then opens, then samples ONCE. Two entries
    would sample occupancy twice and move the metric denominator."""
    ordered = ordered_instants(
        decision_instants({"BTC-USD": [0]}, bar_duration_ms=_BAR), close_checkpoints([_BAR])
    )
    assert ordered == [(_BAR, True)]


def test_checkpoints_do_not_add_sampling_points():
    """§9.5, the metric-drift guard: only decision instants are sampled, so adding
    examination instants cannot move pct_slots_full or mean_concurrent."""
    decisions = decision_instants({"BTC-USD": [0, _BAR, 2 * _BAR]}, bar_duration_ms=_BAR)
    ordered = ordered_instants(decisions, close_checkpoints([1, 2, 3, 4, 5]))
    assert sum(1 for _, is_decision in ordered if is_decision) == len(decisions) == 3
    assert len(ordered) > len(decisions), "the fixture really does add extra instants"


def test_a_checkpoint_does_not_wait_for_another_products_calendar():
    """Without its own instant, a BTC exit would be examined only when some product happens
    to supply a later timestamp -- which on a daily/hourly mix can be a day away."""
    ordered = ordered_instants(
        decision_instants({"BTC-USD": [0], "ETH-USD": [0, 86_400_000]}, bar_duration_ms=_BAR),
        close_checkpoints([2 * _BAR]),
    )
    examined = [instant for instant, _ in ordered if instant == 2 * _BAR]
    assert examined == [2 * _BAR]
    assert min(i for i, _ in ordered if i > _BAR) == 2 * _BAR


def test_duplicate_checkpoint_instants_collapse_to_one():
    """Two positions due at the same instant are both closed by ONE examination -- closing
    is driven by the open positions, not by a count of events (Codex baa57071)."""
    assert close_checkpoints([_BAR, _BAR, 2 * _BAR]) == [_BAR, 2 * _BAR]


# ── timestamps are validated, never coerced ──────────────────────────────────


@pytest.mark.parametrize("bad", [1.5, True, False])
def test_a_non_integral_or_boolean_bar_start_is_refused(bad):
    """`int()` truncates. A fractional timestamp becomes a plausible bar start that no
    longer describes the frame -- the producer's defect, repeated one layer along."""
    with pytest.raises(ValueError):
        decision_instants({"BTC-USD": [0, bad]}, bar_duration_ms=_BAR)


@pytest.mark.parametrize("bad", [1.5, True, 0, -_BAR])
def test_a_malformed_bar_duration_is_refused(bad):
    with pytest.raises(ValueError, match="bar_duration_ms"):
        decision_instants({"BTC-USD": [0]}, bar_duration_ms=bad)


def test_out_of_order_or_duplicate_bar_starts_are_refused():
    for bad in ([_BAR, 0], [0, 0]):
        with pytest.raises(ValueError, match="strictly increasing"):
            decision_instants({"BTC-USD": bad}, bar_duration_ms=_BAR)


@pytest.mark.parametrize("bad", [1.5, True])
def test_a_malformed_checkpoint_instant_is_refused(bad):
    with pytest.raises(ValueError):
        close_checkpoints([bad])


def test_an_empty_grid_produces_no_instants_rather_than_failing():
    """A subset whose products all lack features is empty, not malformed -- the caller
    reports that, and it must not look like a crash."""
    assert decision_instants({}, bar_duration_ms=_BAR) == []
    assert ordered_instants([], close_checkpoints([])) == []


def test_several_simultaneous_exits_at_a_decision_instant_sample_once():
    """Codex 854dba05, the combined case. Three positions due at the same instant, and that
    instant is also a decision instant: the timeline must yield ONE entry marked as a
    decision, so the replay closes all three, then opens, then samples occupancy a single
    time. Any other shape moves the metric denominator."""
    decisions = decision_instants({"BTC-USD": [0, _BAR, 2 * _BAR]}, bar_duration_ms=_BAR)
    ordered = ordered_instants(decisions, close_checkpoints([2 * _BAR] * 3))
    assert ordered == [(_BAR, True), (2 * _BAR, True), (3 * _BAR, True)]
    assert sum(1 for instant, _ in ordered if instant == 2 * _BAR) == 1
    assert sum(1 for _, is_decision in ordered if is_decision) == len(decisions)


def test_several_simultaneous_exits_after_the_last_decision_instant_sample_zero_times():
    """The same collapse where the instant is NOT a decision: one inspection, and it adds no
    sampling point at all."""
    decisions = decision_instants({"BTC-USD": [0]}, bar_duration_ms=_BAR)
    ordered = ordered_instants(decisions, close_checkpoints([5 * _BAR, 5 * _BAR]))
    assert ordered == [(_BAR, True), (5 * _BAR, False)]
    assert sum(1 for _, is_decision in ordered if is_decision) == 1


# ── strict types: validate BEFORE conversion, not by round-tripping ───────────
#
# Codex 324f50ae, all three reproduced before fixing. `_whole` did `int(value)` and then
# compared, which ACCEPTS integral floats, and it rejected only Python `bool` -- so
# `bar_duration_ms=1.0` gave [1], `np.bool_(True)` became 1, and `ordered_instants` validated
# nothing at all. My own handoff had claimed "no int() coercion anywhere", which was false:
# coercing and then checking the round trip is still coercion, it just happens to catch
# fractions. The property wanted is "this IS an integer", which is `numbers.Integral`.

import numpy as np  # noqa: E402


@pytest.mark.parametrize("integral_float", [1.0, 3_600_000.0, 0.0])
def test_an_integral_float_is_not_an_integer(integral_float):
    """`1.0` is not a timestamp or a duration. Round-tripping through int() accepted it."""
    with pytest.raises(ValueError):
        decision_instants({"A": [0]}, bar_duration_ms=integral_float)
    with pytest.raises(ValueError):
        close_checkpoints([integral_float])
    with pytest.raises(ValueError):
        decision_instants({"A": [0, integral_float + _BAR]}, bar_duration_ms=_BAR)


@pytest.mark.parametrize("numpy_bool", [np.bool_(True), np.bool_(False)])
def test_a_numpy_bool_is_rejected_like_a_python_bool(numpy_bool):
    """`np.bool_` is not a `bool` subclass, so an isinstance(bool) guard misses it entirely
    and it converts happily to 1 or 0."""
    with pytest.raises(ValueError, match="bool"):
        close_checkpoints([numpy_bool])
    with pytest.raises(ValueError, match="bool"):
        decision_instants({"A": [0, numpy_bool]}, bar_duration_ms=_BAR)


@pytest.mark.parametrize(
    "value", [np.int64(2 * 3_600_000), np.int32(3_600_000), np.uint32(3_600_000)]
)
def test_numpy_integer_scalars_are_accepted(value):
    """Frames hand out numpy scalars, so rejecting them would make the module unusable on
    real input. Integral means integral, whatever the container."""
    assert close_checkpoints([value]) == [int(value)]
    assert decision_instants({"A": [0]}, bar_duration_ms=value) == [int(value)]


def test_ordered_instants_validates_both_of_its_inputs():
    """It validated NEITHER: `ordered_instants([0.5], [True])` returned
    `[(0.5, True), (True, False)]`, so a fraction and a bool both survived into the output --
    and `True` would then collide with instant 1 in any downstream set or dict."""
    with pytest.raises(ValueError):
        ordered_instants([0.5], [])
    with pytest.raises(ValueError, match="bool"):
        ordered_instants([], [True])
    with pytest.raises(ValueError, match="bool"):
        ordered_instants([np.bool_(True)], [])


def test_a_bool_cannot_collide_with_an_instant_in_the_ordered_output():
    """The collision is the real hazard: `True == 1`, so an unvalidated bool would either
    vanish into an existing instant or masquerade as one."""
    with pytest.raises(ValueError, match="bool"):
        ordered_instants([1], [True])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_nonfinite_instant_is_rejected(value):
    """`int(nan)` raises ValueError and `int(inf)` raises OverflowError, so without an
    explicit check the failure mode depends on which one arrives."""
    with pytest.raises(ValueError):
        close_checkpoints([value])
    with pytest.raises(ValueError):
        ordered_instants([value], [])


@pytest.mark.parametrize("value", ["3600000", None, [1], {"a": 1}])
def test_a_non_numeric_instant_is_rejected(value):
    with pytest.raises(ValueError):
        close_checkpoints([value])
