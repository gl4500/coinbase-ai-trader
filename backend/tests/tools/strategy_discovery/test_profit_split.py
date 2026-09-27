"""Tests for tools.strategy_discovery.profit_split (Phase 3)."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from tools.strategy_discovery.profit_split import (
    best_split,
    build_next_eligible,
    walk_and_sum,
)


def _naive_walk_and_sum_py(
    indices: List[int],
    next_eligible: List[int],
    labels: List[float],
) -> float:
    """Reference: walk indices in order, only enter if not already in a trade."""
    open_until = -1  # exclusive
    total = 0.0
    for i in indices:
        if i < open_until:
            continue
        total += labels[i]
        open_until = next_eligible[i]
    return total


def test_walk_and_sum_matches_naive_python_reference():
    rng = np.random.default_rng(13)
    N = 500
    labels = rng.normal(0.0, 0.05, size=N).astype("float64")
    horizon_bars = 24
    next_eligible = np.minimum(np.arange(N) + horizon_bars, N).astype("int64")
    B = 7
    subsets = []
    for _ in range(B):
        size = rng.integers(50, 200)
        chosen = sorted(rng.choice(N, size=size, replace=False).tolist())
        subsets.append(chosen)
    max_k = max(len(s) for s in subsets)
    subset_idx = torch.full((B, max_k), -1, dtype=torch.int64)
    for b, s in enumerate(subsets):
        subset_idx[b, : len(s)] = torch.tensor(s, dtype=torch.int64)
    out = walk_and_sum(
        subset_idx,
        torch.from_numpy(next_eligible),
        torch.from_numpy(labels),
    )
    expected = [_naive_walk_and_sum_py(s, next_eligible.tolist(), labels.tolist()) for s in subsets]
    np.testing.assert_allclose(out.cpu().numpy(), np.array(expected), rtol=1e-9, atol=1e-12)


def test_concurrency_max_1_skips_overlapping_entry():
    ts = torch.arange(5, dtype=torch.int64) * 3_600_000
    labels = torch.tensor([1.0, 10.0, 100.0, 2.0, 50.0], dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=3)
    assert next_eligible.tolist() == [3, 4, 5, 5, 5]
    subset = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    total = walk_and_sum(subset, next_eligible, labels)
    assert total.item() == pytest.approx(3.0, abs=1e-12)


def test_split_metric_picks_higher_pnl_subgroup():
    N = 100
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    horizon_bars = 1
    features = torch.zeros((N, 1), dtype=torch.float64)
    features[50:, 0] = 1.0
    labels = torch.zeros(N, dtype=torch.float64)
    labels[:50] = -0.02
    labels[50:] = 0.10
    next_eligible = build_next_eligible(ts, horizon_bars=horizon_bars)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is not None
    assert result.feature == 0
    assert 0.0 < result.threshold < 1.0
    assert result.score == pytest.approx(5.0, abs=1e-9)


def test_no_profitable_split_returns_none():
    N = 30
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    features = torch.linspace(0.0, 1.0, N, dtype=torch.float64).unsqueeze(1)
    labels = torch.full((N,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is None


# ── next_eligible read from published endpoints instead of a wall clock ───────
#
# Contract §9.1. `build_next_eligible` measures the horizon in WALL-CLOCK milliseconds
# while `walk_and_sum` compares ROW POSITIONS, and the label it gates was computed from row
# offsets with early stops. The two agree only on contiguous bars where every exit reached
# the nominal horizon. `build_next_eligible` is deliberately kept: it is the baseline the
# equivalence test below compares against, and deleting it would remove the only evidence
# that the endpoint path is a generalisation rather than a different algorithm.

import dataclasses  # noqa: E402

from tools.strategy_discovery.endpoint_consumers import (  # noqa: E402
    _VALIDATED_BY_LOADER,
    ValidatedEndpoints,
)
from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER, LabelEndpoint  # noqa: E402
from tools.strategy_discovery.profit_split import (  # noqa: E402
    build_next_eligible_from_endpoints,
)

_EP_BAR = 3_600_000


def _ep_template(horizon):
    return LabelEndpoint(
        product_id="BTC-USD",
        horizon=horizon,
        data_id="sha256:fixture",
        label_version="label_endpoint_v1",
        cost_version="round_trip_fee_v1",
        config_id="sha256:fixture",
        label_value=0.01,
        entry_row_id=0,
        exit_row_id=1,
        bars_held=1,
        max_hold_bars=168,
        entry_bar_start=0,
        exit_bar_start=_EP_BAR,
        bar_duration_ms=_EP_BAR,
        entry_available_at=_EP_BAR,
        exit_observable_at=2 * _EP_BAR,
        exit_kind="horizon",
        exit_price_basis="bar_close",
        intrabar_timing_known=True,
        intrabar_order_assumption=None,
        blockers=(CAUSALITY_BLOCKER,),
    )


_EP_BASIS = {
    "stop": "assumed_stop_level",
    "trail": "assumed_trail_level",
    "horizon": "bar_close",
}


def _validated(pairs, *, horizon, starts, exit_kind="horizon"):
    """Hand-specified (entry, exit) pairs, wrapped as a validated set.

    Uses the module-private construction token DELIBERATELY: these pairs exercise
    clock-versus-endpoint divergence, and no real simulation emits a chosen pair on demand.
    The gate that makes this explicit is covered in test_endpoint_consumers.py.

    Records are kept SEMANTICALLY POSSIBLE rather than merely well-typed. `exit_kind` drives
    `exit_price_basis`, `intrabar_timing_known` and `intrabar_order_assumption` together,
    because an early exit claiming horizon/bar_close is a record the real validator rejects,
    and a fixture the validator would reject proves nothing about behaviour on real records.
    Every pair must also satisfy entry < exit: clamping an exit to the final row produced
    entry == exit, a zero-bar hold that cannot occur.
    """
    assert all(entry < exit_row for entry, exit_row in pairs), (
        "a fixture with entry == exit is not a record the producer can emit"
    )
    assert all(exit_row < len(starts) for _, exit_row in pairs), (
        "an exit past the source frame has no bar start"
    )
    template = _ep_template(horizon)

    records = tuple(
        dataclasses.replace(
            template,
            entry_row_id=entry,
            exit_row_id=exit_row,
            bars_held=exit_row - entry,
            entry_bar_start=starts[entry],
            exit_bar_start=starts[exit_row],
            entry_available_at=starts[entry] + _EP_BAR,
            exit_observable_at=starts[exit_row] + _EP_BAR,
            exit_kind=exit_kind,
            exit_price_basis=_EP_BASIS[exit_kind],
            # an intrabar stop or trail has no known within-bar time; only a horizon exit
            # lands at a bar close
            intrabar_timing_known=(exit_kind == "horizon"),
            intrabar_order_assumption=("high_before_low" if exit_kind == "trail" else None),
        )
        for entry, exit_row in pairs
    )
    return ValidatedEndpoints(
        records=records,
        data_id="sha256:fixture",
        bar_duration_ms=_EP_BAR,
        exit_config={"max_hold_bars": 168},
        token=_VALIDATED_BY_LOADER,
    )


def test_on_contiguous_bars_with_no_early_exit_the_two_agree_exactly():
    """THE equivalence gate. Where the clock was right, the endpoints must reproduce it
    element for element -- not approximately, not mostly. If this fails, the endpoint path
    is a different algorithm rather than a generalisation, and the work stops for review."""
    source_rows, horizon = 12, 3
    starts = [i * _EP_BAR for i in range(source_rows)]
    # RETAINED rows are those that actually have a label: entry i needs exit i+horizon to
    # exist, so rows whose horizon runs past the end get no endpoint (part 1 counts them as
    # `insufficient_horizon`) and are filtered out before either consumer sees them. An
    # earlier version of this fixture handed endpoints to those tail rows and the
    # equivalence assertion failed -- correctly, because it was comparing a frame the
    # producer would never emit.
    retained = list(range(source_rows - horizon))
    ts_ms = torch.tensor([starts[r] for r in retained], dtype=torch.int64)
    validated = _validated([(r, r + horizon) for r in retained], horizon=horizon, starts=starts)
    from_clock = build_next_eligible(ts_ms, horizon_bars=horizon)
    from_endpoints = build_next_eligible_from_endpoints(validated, retained, horizon=horizon)
    assert from_endpoints.dtype == from_clock.dtype
    assert from_endpoints.shape == from_clock.shape
    assert torch.equal(from_endpoints, from_clock)


def test_an_early_exit_reopens_the_slot_sooner_than_the_clock_believed():
    """A stop at bar 1 of a 3-bar horizon. The clock holds the slot for 3 bars; the
    endpoint knows the trade was over at bar 1, so the clock UNDERSTATES capacity."""
    source_rows, horizon = 10, 3
    starts = [i * _EP_BAR for i in range(source_rows)]
    # only rows that could carry a horizon-3 label are retained
    retained = list(range(source_rows - horizon))
    ts_ms = torch.tensor([starts[r] for r in retained], dtype=torch.int64)
    validated = _validated(
        [(r, r + 1) for r in retained], horizon=horizon, starts=starts, exit_kind="stop"
    )
    from_clock = build_next_eligible(ts_ms, horizon_bars=horizon)
    from_endpoints = build_next_eligible_from_endpoints(validated, retained, horizon=horizon)
    assert bool((from_endpoints <= from_clock).all())
    assert bool((from_endpoints < from_clock).any())
    assert int(from_endpoints[0]) == 1 and int(from_clock[0]) == 3


def test_on_gapped_bars_the_clock_reopens_the_slot_while_the_label_is_still_open():
    """The double-counting direction, the one that inflates profit. With one bar of hole
    between rows and a 2-bar horizon, the clock admits the very next row while the label
    still runs two, so two positions occupy one leaf."""
    source_rows, horizon = 8, 2
    starts = [i * 2 * _EP_BAR for i in range(source_rows)]
    retained = list(range(source_rows - horizon))
    ts_ms = torch.tensor([starts[r] for r in retained], dtype=torch.int64)
    validated = _validated([(r, r + horizon) for r in retained], horizon=horizon, starts=starts)
    from_clock = build_next_eligible(ts_ms, horizon_bars=horizon)
    from_endpoints = build_next_eligible_from_endpoints(validated, retained, horizon=horizon)
    # exact vectors, not an inequality: the clock advances one row per position while the
    # label still runs two
    assert from_clock.tolist() == [1, 2, 3, 4, 5, 6]
    assert from_endpoints.tolist() == [2, 3, 4, 5, 6, 6]


def test_walk_and_sum_selects_fewer_trades_than_the_clock_did_on_gapped_bars():
    """The double-count, measured exactly rather than asserted to be positive.

    An assertion of `> 0` would have passed either way. With one bar of hole between rows
    and a 2-bar horizon, the endpoint vector admits 3 non-overlapping trades while the wall
    clock admits 6 -- it reopens the slot on the very next row while the label is still
    running. Same labels, same subset, twice the trades. This also confirms the vector is
    consumable by `walk_and_sum` unchanged, which is the point of matching its dtype and
    terminal convention.
    """
    source_rows, horizon = 8, 2
    starts = [i * 2 * _EP_BAR for i in range(source_rows)]
    retained = list(range(source_rows - horizon))
    ts_ms = torch.tensor([starts[r] for r in retained], dtype=torch.int64)
    validated = _validated([(r, r + horizon) for r in retained], horizon=horizon, starts=starts)

    labels = torch.tensor([0.1] * len(retained), dtype=torch.float64)
    subset = torch.arange(len(retained), dtype=torch.int64).unsqueeze(0)

    from_endpoints = build_next_eligible_from_endpoints(validated, retained, horizon=horizon)
    from_clock = build_next_eligible(ts_ms, horizon_bars=horizon)
    endpoint_total = float(walk_and_sum(subset, from_endpoints, labels)[0])
    clock_total = float(walk_and_sum(subset, from_clock, labels)[0])

    assert endpoint_total == pytest.approx(0.3), "rows 0, 2 and 4 fire; the rest overlap"
    assert clock_total == pytest.approx(0.6), "the clock fires every row, twice as many"
    assert clock_total > endpoint_total


def test_it_refuses_endpoints_that_were_never_validated():
    with pytest.raises(TypeError, match="ValidatedEndpoints"):
        build_next_eligible_from_endpoints([_ep_template(2)], [0, 1], horizon=2)
