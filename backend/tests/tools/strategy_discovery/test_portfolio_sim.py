"""Tests for tools.strategy_discovery.portfolio_sim (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.tools.strategy_discovery.rule_fixtures import machine_rule_fixture
from tools.strategy_discovery.portfolio_sim import (
    simulate_portfolio,
)
from tools.strategy_discovery.profile_loader import LoadedProfile


def _make_profile(
    pid: str, leaf_id: int, horizon: int, rule_path: str, deflated: float = 0.05
) -> LoadedProfile:
    return LoadedProfile(
        pid=pid,
        horizon=horizon,
        leaf_id=leaf_id,
        rule_path=rule_path,
        machine_rule=machine_rule_fixture(rule_path),
        cumulative_profit_raw=deflated + 0.02,
        cumulative_profit_deflated=deflated,
        deflation_pp=0.02,
        win_rate=0.6,
        avg_win=0.08,
        avg_loss=-0.04,
        max_dd=0.20,
        sortino=1.2,
        trade_count=30,
        n_folds_passed_q0=5,
        chosen_depth=5,
        chosen_min_leaf=50,
    )


_LEGACY_BAR = 3_600_000


def _make_pid_features(pid: str, n_hours: int, ema_ratio: float, label: float):
    """Synthetic Phase 2 frame, shaped as the producer would actually write it.

    Two corrections over the original: every row used to carry a finite label INCLUDING the
    last, which the producer cannot do -- a label needs its exit row to exist, so the final
    `h` rows of each horizon are NaN. And `source_row_id`, `high`, `low` and `atr14_pct` are
    present, because row identity and frame binding both need them.
    """
    ts = (np.arange(n_hours, dtype="int64") * _LEGACY_BAR).tolist()

    def _labels(horizon):
        return [float(label) if row + horizon < n_hours else float("nan") for row in range(n_hours)]

    return pd.DataFrame(
        {
            "ts": ts,
            "source_row_id": list(range(n_hours)),
            "close": [1.0] * n_hours,
            "high": [1.0] * n_hours,
            "low": [1.0] * n_hours,
            "atr14_pct": [0.06] * n_hours,
            "price_over_ema20": [ema_ratio] * n_hours,
            "vol_over_mc": [0.01] * n_hours,
            "label_h1": _labels(1),
            "label_h24": _labels(24),
        }
    )


_LEGACY_EXIT_CONFIG = {
    "stop_loss_pct": 0.08,
    "atr_trail_floor": 0.06,
    "max_hold_bars": 168,
    "round_trip_fee": 0.012,
}


def _legacy_endpoints(pid: str, frame, horizons):
    """Validated endpoints for every finite-label row of a legacy fixture frame.

    Goes through the private construction token rather than the loader, because these frames
    are synthesised rather than published -- but it computes a REAL `data_id` and a real
    fingerprint from the frame, so the binding checks exercise genuine values.
    """
    from tools.strategy_discovery.endpoint_consumers import (
        _VALIDATED_BY_LOADER,
        ValidatedEndpoints,
        frame_fingerprint,
    )
    from tools.strategy_discovery.endpoint_dataset import build_data_id
    from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER, LabelEndpoint

    starts = frame["ts"].tolist()
    data_id = build_data_id(
        product_id=pid,
        bar_duration_ms=_LEGACY_BAR,
        timestamps=starts,
        closes=frame["close"].tolist(),
        highs=frame["high"].tolist(),
        lows=frame["low"].tolist(),
        atr_pcts=frame["atr14_pct"].tolist(),
        feature_recipe="atr14_pct_wilder_v1",
        config=_LEGACY_EXIT_CONFIG,
    )
    # Records for EVERY horizon the subset uses on this product. One pid can carry profiles at
    # several horizons, and a set holding only one of them leaves the others uncovered -- which
    # is what the horizon-collision case exercises.
    records = []
    for horizon in sorted({int(h) for h in horizons}):
        column = f"label_h{horizon}"
        for row_id, value in zip(
            frame["source_row_id"].tolist(), frame[column].tolist(), strict=True
        ):
            if value != value:  # NaN tail carries no endpoint, as the producer leaves it
                continue
            exit_row = int(row_id) + horizon
            records.append(
                LabelEndpoint(
                    product_id=pid,
                    horizon=horizon,
                    data_id=data_id,
                    label_version="label_endpoint_v1",
                    cost_version="round_trip_fee_v1",
                    config_id=data_id,
                    label_value=float(value),
                    entry_row_id=int(row_id),
                    exit_row_id=exit_row,
                    bars_held=horizon,
                    max_hold_bars=168,
                    entry_bar_start=starts[int(row_id)],
                    exit_bar_start=starts[exit_row],
                    bar_duration_ms=_LEGACY_BAR,
                    entry_available_at=starts[int(row_id)] + _LEGACY_BAR,
                    exit_observable_at=starts[exit_row] + _LEGACY_BAR,
                    exit_kind="horizon",
                    exit_price_basis="bar_close",
                    intrabar_timing_known=True,
                    intrabar_order_assumption=None,
                    blockers=(CAUSALITY_BLOCKER,),
                )
            )
    return ValidatedEndpoints(
        records=tuple(records),
        data_id=data_id,
        bar_duration_ms=_LEGACY_BAR,
        exit_config=_LEGACY_EXIT_CONFIG,
        feature_recipe="atr14_pct_wilder_v1",
        frame_fingerprint=frame_fingerprint(frame),
        token=_VALIDATED_BY_LOADER,
    )


def _legacy_run(profiles, cap, pid_features):
    """`simulate_portfolio` with endpoints built from each profile's own frame and horizon."""
    horizons_by_pid = {}
    for profile in profiles:
        if profile.pid in pid_features and not pid_features[profile.pid].empty:
            horizons_by_pid.setdefault(profile.pid, set()).add(int(profile.horizon))
    endpoints = {
        pid: _legacy_endpoints(pid, pid_features[pid], horizons)
        for pid, horizons in horizons_by_pid.items()
    }
    return simulate_portfolio(
        profiles,
        cap=cap,
        pid_features=pid_features,
        endpoints_by_pid=endpoints,
        bar_duration_ms=_LEGACY_BAR,
    )


def test_concurrency_cap_blocks_new_entries_when_full():
    # 3 pids, each fires on the same rule, 24h horizon.
    # Cap=2 → at most 2 open at any time; the 3rd pid never enters.
    pids = ["A-USD", "B-USD", "C-USD"]
    profiles = [_make_profile(pid, 0, 24, "price_over_ema20 > 1.0", deflated=0.05) for pid in pids]
    pid_features = {
        pid: _make_pid_features(pid, n_hours=200, ema_ratio=1.5, label=0.10) for pid in pids
    }
    metrics, telemetry = _legacy_run(profiles, 2, pid_features)
    # n_open never exceeds 2
    assert max(t.n_open for t in telemetry) <= 2
    # At least some bars have n_open == 2 (cap was hit)
    assert any(t.n_open == 2 for t in telemetry)
    # Each pid that does enter must have closed by horizon
    assert metrics.trade_count > 0


def test_max_1_position_per_pid_carried_over():
    # Same pid, two profiles (different leaf_ids), both fire — only one enters.
    profiles = [
        _make_profile("BTC-USD", 0, 24, "price_over_ema20 > 1.0", deflated=0.05),
        _make_profile("BTC-USD", 1, 24, "vol_over_mc > 0.0", deflated=0.04),
    ]
    feats = _make_pid_features("BTC-USD", n_hours=200, ema_ratio=1.5, label=0.10)
    metrics, telemetry = _legacy_run(profiles, 3, {"BTC-USD": feats})
    # At any bar n_open <= 1 (only one BTC-USD position)
    assert max(t.n_open for t in telemetry) <= 1


def test_simultaneous_fires_resolved_by_deflated_profit_tiebreaker():
    # 3 pids fire simultaneously on bar 0; cap=1; only the highest-deflated wins.
    profiles = [
        _make_profile("A-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.03),
        _make_profile("B-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.07),  # winner
        _make_profile("C-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.05),
    ]
    pid_features = {
        pid: _make_pid_features(pid, n_hours=2, ema_ratio=1.5, label=0.10)
        for pid in ["A-USD", "B-USD", "C-USD"]
    }
    metrics, telemetry = _legacy_run(profiles, 1, pid_features)
    # First fire telemetry must be the highest-deflated profile (B-USD)
    fires = [t for t in telemetry if t.fired_profile_id is not None]
    assert len(fires) >= 1
    assert fires[0].fired_profile_id.startswith("B-USD")


def test_exit_pnl_read_from_phase2_label():
    # Synthetic: one profile fires, label_h1 = +0.10, so one trade closes with +0.10.
    profile = _make_profile("BTC-USD", 0, 1, "price_over_ema20 > 1.0", deflated=0.05)
    # Make sure label fires (ema_ratio > 1.0) only at entry bar; subsequent bars
    # are still 'fireable' but the per-pid cap prevents re-entry until exit.
    feats = _make_pid_features("BTC-USD", n_hours=5, ema_ratio=1.5, label=0.10)
    metrics, telemetry = _legacy_run([profile], 1, {"BTC-USD": feats})
    assert metrics.cumulative_profit_raw > 0
    # At least one trade closed
    closes = [t for t in telemetry if t.closed_profile_id is not None]
    assert len(closes) >= 1
    # And the realized PnL on the closed trade matches the Phase 2 label exactly
    assert closes[0].realized_pnl == pytest.approx(0.10, abs=1e-9)


def test_max_dd_computed_on_equity_curve():
    # Construct a deterministic equity curve via _compute_max_dd directly
    from tools.strategy_discovery.portfolio_sim import _compute_max_dd

    # Equity goes: 0, 1, 2, 0.5, 1.0, 1.5 — peak=2, trough after peak=0.5, dd=1.5
    curve = [0.0, 1.0, 2.0, 0.5, 1.0, 1.5]
    assert _compute_max_dd(curve) == pytest.approx(1.5, abs=1e-9)
    # Monotonically increasing curve → dd = 0
    assert _compute_max_dd([0.0, 1.0, 2.0, 3.0]) == pytest.approx(0.0)
    # Empty → 0
    assert _compute_max_dd([]) == 0.0


def test_slot_utilization_telemetry():
    # 3 pids all firing constantly, cap=2. pct_slots_full should be high (>0.5).
    profiles = [
        _make_profile(pid, 0, 1, "price_over_ema20 > 1.0", deflated=0.05)
        for pid in ["A-USD", "B-USD", "C-USD"]
    ]
    pid_features = {
        pid: _make_pid_features(pid, n_hours=100, ema_ratio=1.5, label=0.10)
        for pid in ["A-USD", "B-USD", "C-USD"]
    }
    metrics, telemetry = _legacy_run(profiles, 2, pid_features)
    assert 0.0 <= metrics.pct_slots_full <= 1.0
    assert 0.0 <= metrics.mean_concurrent <= 2.0
    # With cap=2 and 3 always-firing pids, we expect cap to be hit some of the time
    assert metrics.pct_slots_full > 0.3


@pytest.mark.parametrize("horizon", [1, 24])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("different_rules", [False, True])
def test_horizon_collision_preserves_selected_rule_label_and_exit(
    horizon, reverse, different_rules
):
    selected = _make_profile("BTC-USD", 0, horizon, "price_over_ema20 > 1.0", deflated=0.2)
    other = _make_profile(
        "BTC-USD",
        0,
        24 if horizon == 1 else 1,
        "price_over_ema20 > 2.0" if different_rules else "price_over_ema20 > 1.0",
        deflated=0.1,
    )
    profiles = [selected, other]
    if reverse:
        profiles.reverse()
    features = _make_pid_features("BTC-USD", n_hours=26, ema_ratio=0.0, label=0.0)
    features.loc[0, "price_over_ema20"] = 1.5
    # Overwrite the labels but PRESERVE the NaN tail: a row can only carry a label when its
    # exit row exists, so blanket assignment would invent labels the producer cannot emit.
    for column, value, h in (("label_h1", 0.1, 1), ("label_h24", -0.2, 24)):
        features[column] = [
            value if row + h < len(features) else float("nan") for row in range(len(features))
        ]
    metrics, telemetry = _legacy_run(profiles, 1, {"BTC-USD": features})
    closes = [row for row in telemetry if row.closed_profile_id is not None]
    # DERIVED, not observed (Codex's oracle 5ad67efc): only row 0 clears the rule, since
    # ema_ratio is 0.0 everywhere else, so the count stays 1. The close instant moves from
    # horizon*bar to (horizon+1)*bar, because the exit is now the exit bar's CLOSE
    # (exit_observable_at = exit_bar_start + bar_duration) rather than its start.
    assert metrics.trade_count == 1
    assert closes[0].ts == (horizon + 1) * 3_600_000
    assert closes[0].realized_pnl == pytest.approx(0.1 if horizon == 1 else -0.2)
    assert closes[0].closed_profile_id == selected.profile_id


@pytest.mark.parametrize("rule", ["", "   ", None])
def test_unresolved_rule_cannot_be_simulated_as_always_fire(rule):
    profile = _make_profile("BTC-USD", 0, 1, rule)
    feats = _make_pid_features("BTC-USD", 3, 1.5, 0.1)
    with pytest.raises(ValueError, match="rule"):
        _legacy_run([profile], 1, {"BTC-USD": feats})


def test_duplicate_profiles_do_not_overwrite_simulation_rules():
    first = _make_profile("BTC-USD", 0, 1, "price_over_ema20 > 1.0")
    second = _make_profile("BTC-USD", 0, 1, "price_over_ema20 > 2.0")
    feats = _make_pid_features("BTC-USD", 3, 1.5, 0.1)
    with pytest.raises(ValueError, match="duplicate"):
        _legacy_run([first, second], 1, {"BTC-USD": feats})


def test_explicit_root_can_trade_without_conditions():
    profile = _make_profile("BTC-USD", 0, 1, "(root)")
    feats = _make_pid_features("BTC-USD", 3, 0.0, 0.1)
    metrics, _ = _legacy_run([profile], 1, {"BTC-USD": feats})
    assert metrics.trade_count > 0


def test_simulation_uses_exact_rule_not_rounded_display():
    from tools.strategy_discovery.profit_tree import TreeNode
    from tools.strategy_discovery.rule_contract import encode_leaf_rule

    profile = _make_profile("BTC-USD", 0, 1, "price_over_ema20 <= 1.02")
    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    profile.machine_rule = encode_leaf_rule(tree, 0, ["price_over_ema20"])
    features = _make_pid_features("BTC-USD", 3, 1.023, 0.1)
    metrics, _ = _legacy_run([profile], 1, {"BTC-USD": features})
    assert metrics.trade_count > 0


def test_simulator_rejects_gaps_before_realizing_future_label_returns():
    from tools.strategy_discovery.labels import simulate_dynamic_exit_labels

    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    frame = pd.DataFrame(
        {
            "ts": np.arange(5, dtype="int64") * 7_200_000,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "atr14_pct": 0.06,
        }
    )
    frame = simulate_dynamic_exit_labels(frame, horizons=[2])
    frame["source_row_id"] = list(range(len(frame)))
    profile = _make_profile("GAP-USD", 0, 2, "(root)")
    # This return requires row 2 (hour 4), but old replay realized it at hour 2.
    assert frame.loc[0, "label_h2"] == pytest.approx(0.008)
    with pytest.raises(ValueError, match="contiguous hourly"):
        _legacy_run([profile], 1, {"GAP-USD": frame})


# ── the replay runs on endpoint instants, not a wall clock ───────────────────
#
# Contract §9.1. An ELIGIBILITY BOUNDARY is a position; an ACCOUNTING TIME is an instant. The
# single wall-clock `exit_ts` this replaces played both roles, which is why one defect
# produced errors in OPPOSITE directions.
#
# Every fixture below gives an endpoint to EVERY finite-label row (Codex 26837c63): weakening
# a fail-closed check to accommodate a thin fixture would hide exactly what these tests exist
# to find.

import dataclasses  # noqa: E402

from tools.strategy_discovery.endpoint_consumers import (  # noqa: E402
    _VALIDATED_BY_LOADER,
    MissingEndpoint,
    ValidatedEndpoints,
)
from tools.strategy_discovery.endpoint_records import (  # noqa: E402
    CAUSALITY_BLOCKER,
    LabelEndpoint,
)

_WBAR = 3_600_000
_WBASIS = {
    "stop": "assumed_stop_level",
    "trail": "assumed_trail_level",
    "horizon": "bar_close",
}


def _w_endpoint(
    pid,
    starts,
    entry,
    exit_row,
    *,
    horizon,
    label,
    exit_kind="horizon",
    data_id="sha256:wiring-fixture",
):
    """A LabelEndpoint consistent with the frame, using the module's real enum values."""
    assert entry < exit_row, "entry == exit is not a record the producer can emit"
    return LabelEndpoint(
        product_id=pid,
        horizon=horizon,
        data_id=data_id,
        label_version="label_endpoint_v1",
        cost_version="round_trip_fee_v1",
        config_id=data_id,
        label_value=float(label),
        entry_row_id=entry,
        exit_row_id=exit_row,
        bars_held=exit_row - entry,
        max_hold_bars=168,
        entry_bar_start=starts[entry],
        exit_bar_start=starts[exit_row],
        bar_duration_ms=_WBAR,
        entry_available_at=starts[entry] + _WBAR,
        exit_observable_at=starts[exit_row] + _WBAR,
        exit_kind=exit_kind,
        exit_price_basis=_WBASIS[exit_kind],
        intrabar_timing_known=(exit_kind == "horizon"),
        intrabar_order_assumption=("high_before_low" if exit_kind == "trail" else None),
        blockers=(CAUSALITY_BLOCKER,),
    )


def _wire(pid, bar_hours, *, horizon, label=0.1, exit_offset=None, ema=1.5, exit_kind="horizon"):
    """Frame plus a validated endpoint for EVERY finite-label row.

    A row carries a label only when its exit row exists, so the tail rows are NaN and have no
    endpoint -- which is what the producer does. `exit_offset` defaults to the nominal horizon;
    a smaller value models an early exit.
    """
    starts, clock = [], 0
    for gap in bar_hours:
        starts.append(clock)
        clock += int(gap) * _WBAR
    n = len(starts)
    offset = horizon if exit_offset is None else exit_offset
    labelled = [r for r in range(n) if r + horizon < n]
    frame = pd.DataFrame(
        {
            "ts": starts,
            "source_row_id": list(range(n)),
            "close": [1.0] * n,
            "price_over_ema20": [ema] * n,
            "vol_over_mc": [0.01] * n,
            f"label_h{horizon}": [
                float(label) if r in set(labelled) else float("nan") for r in range(n)
            ],
        }
    )
    # The replay RECOMPUTES the frame's identity, so the fixture must describe its own frame
    # rather than quote a placeholder. Computing it here is what makes the mismatch test below
    # meaningful: mutate the frame and the recompute no longer agrees.
    from tools.strategy_discovery.endpoint_dataset import build_data_id

    exit_config = {
        "stop_loss_pct": 0.08,
        "atr_trail_floor": 0.06,
        "max_hold_bars": 168,
        "round_trip_fee": 0.012,
    }
    frame["high"] = [1.0] * n
    frame["low"] = [1.0] * n
    frame["atr14_pct"] = [0.06] * n
    data_id = build_data_id(
        product_id=pid,
        bar_duration_ms=_WBAR,
        timestamps=starts,
        closes=frame["close"].tolist(),
        highs=frame["high"].tolist(),
        lows=frame["low"].tolist(),
        atr_pcts=frame["atr14_pct"].tolist(),
        feature_recipe="atr14_pct_wilder_v1",
        config=exit_config,
    )
    records = tuple(
        _w_endpoint(
            pid,
            starts,
            r,
            r + offset,
            horizon=horizon,
            label=label,
            exit_kind=exit_kind,
            data_id=data_id,
        )
        for r in labelled
    )
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint

    validated = ValidatedEndpoints(
        records=records,
        data_id=data_id,
        bar_duration_ms=_WBAR,
        exit_config=exit_config,
        feature_recipe="atr14_pct_wilder_v1",
        # the REAL fingerprint of this frame, so the mismatch tests below are meaningful
        frame_fingerprint=frame_fingerprint(frame),
        token=_VALIDATED_BY_LOADER,
    )
    return frame, validated


def _w_run(pid, frame, validated, *, horizon, cap=1, deflated=0.05):
    profile = _make_profile(pid, 0, horizon, "price_over_ema20 > 1.0", deflated=deflated)
    return simulate_portfolio(
        [profile],
        cap=cap,
        pid_features={pid: frame},
        endpoints_by_pid={pid: validated},
        bar_duration_ms=_WBAR,
    )


def _w_realized(telemetry):
    return [t for t in telemetry if t.realized_pnl is not None]


def _w_fired(telemetry):
    return [t for t in telemetry if t.fired_profile_id is not None]


def test_entries_are_decided_at_the_bar_close_not_the_bar_start():
    """Features are close-derived, so firing at a bar START uses information that does not
    exist yet and dates every entry one bar early."""
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    _, telemetry = _w_run("A-USD", frame, validated, horizon=2)
    fired = _w_fired(telemetry)
    assert fired, "the rule must fire somewhere"
    assert fired[0].ts == int(frame["ts"].iloc[0]) + _WBAR


def test_an_endpoint_whose_entry_instant_contradicts_the_grid_is_refused():
    """Cross-check, not an assumption: a record's `entry_available_at` must equal
    `row.ts + bar_duration_ms`, or the frame and the endpoints describe different bars."""
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    broken = dataclasses.replace(validated.records[0], entry_available_at=1)
    shifted = ValidatedEndpoints(
        records=(broken,) + validated.records[1:],
        data_id=validated.data_id,
        bar_duration_ms=_WBAR,
        exit_config=dict(validated.exit_config),
        feature_recipe=validated.feature_recipe,
        # the SAME frame binding, so this test fails on the entry instant rather than tripping
        # the frame check first
        frame_fingerprint=validated.frame_fingerprint,
        token=_VALIDATED_BY_LOADER,
    )
    with pytest.raises(MissingEndpoint, match="entry_available_at"):
        _w_run("A-USD", frame, shifted, horizon=2)


def test_a_position_exiting_on_the_final_source_bar_is_still_realized():
    """The dropped-PnL case end to end. The final bar's close is later than every decision
    instant, so a condition tested only at decision instants never fires."""
    frame, validated = _wire("A-USD", [1, 1, 1], horizon=2)
    metrics, telemetry = _w_run("A-USD", frame, validated, horizon=2)
    assert metrics.trade_count == 1
    assert _w_realized(telemetry)[0].ts == int(frame["ts"].iloc[2]) + _WBAR


def test_an_early_exit_realizes_pnl_at_its_real_time_and_frees_the_slot():
    """A stop at bar 1 of a 3-bar horizon. The clock held the slot to bar 3 and put the PnL
    on the curve there; both were wrong, in the same direction."""
    frame, validated = _wire("A-USD", [1] * 8, horizon=3, exit_offset=1, exit_kind="stop")
    _, telemetry = _w_run("A-USD", frame, validated, horizon=3)
    realized = _w_realized(telemetry)
    assert realized[0].ts == int(frame["ts"].iloc[1]) + _WBAR
    assert len(_w_fired(telemetry)) > 1, "the freed slot is reused, which the clock forbade"


def test_gapped_bars_are_still_refused_by_the_existing_contiguity_guard():
    """The gap direction is CONTAINED, not exercised -- and that boundary is worth a test.

    PR #72 rejects a non-contiguous hourly frame outright, with the message "label exit
    provenance required for gaps". Endpoints are now exactly that provenance, so relaxing the
    guard is the natural endgame -- but it is a deliberate behaviour change to a guard another
    session added, so it stays until that is decided, and this asserts it still fires.

    The consequence to be honest about: the inflating direction of the defect (a gap realizing
    PnL before the label window closed and freeing the slot for an overlapping entry) cannot be
    demonstrated end-to-end in the replay while the guard stands. It IS demonstrated on the
    eligibility side, where `test_walk_and_sum_selects_fewer_trades_than_the_clock_did_on_gapped_bars`
    shows 0.3 against 0.6 -- twice the trades from identical data.
    """
    frame, validated = _wire("A-USD", [2] * 8, horizon=2)
    with pytest.raises(ValueError, match="contiguous hourly"):
        _w_run("A-USD", frame, validated, horizon=2)


def test_the_exit_instant_may_itself_fire_a_new_entry():
    """Half-open `[entry, exit)` -- §3a tie-order. Close-before-open at a coincident clock."""
    frame, validated = _wire("A-USD", [1] * 8, horizon=2)
    _, telemetry = _w_run("A-USD", frame, validated, horizon=2)
    exit_instant = int(frame["ts"].iloc[2]) + _WBAR
    assert any(t.ts == exit_instant for t in _w_realized(telemetry))
    assert any(t.ts == exit_instant for t in _w_fired(telemetry))


def test_the_higher_deflated_profile_wins_a_contested_slot_not_the_earlier_name():
    """The ranking-preservation test. Two products fire at the same instant with cap=1: the
    higher `cumulative_profit_deflated` must win regardless of product name order, which an
    event-per-position timeline would have silently replaced with alphabetical order."""
    frame_a, validated_a = _wire("A-USD", [1] * 6, horizon=2)
    frame_z, validated_z = _wire("Z-USD", [1] * 6, horizon=2)
    _, telemetry = simulate_portfolio(
        [
            _make_profile("A-USD", 0, 2, "price_over_ema20 > 1.0", deflated=0.01),
            _make_profile("Z-USD", 0, 2, "price_over_ema20 > 1.0", deflated=0.09),
        ],
        cap=1,
        pid_features={"A-USD": frame_a, "Z-USD": frame_z},
        endpoints_by_pid={"A-USD": validated_a, "Z-USD": validated_z},
        bar_duration_ms=_WBAR,
    )
    assert _w_fired(telemetry)[0].fired_profile_id.startswith("Z-USD")


def test_two_positions_due_at_the_same_instant_both_close_exactly_once():
    """One inspection closes every due position: neither is missed, neither double-counted."""
    frame_a, validated_a = _wire("A-USD", [1, 1, 1], horizon=2)
    frame_z, validated_z = _wire("Z-USD", [1, 1, 1], horizon=2)
    metrics, telemetry = simulate_portfolio(
        [
            _make_profile("A-USD", 0, 2, "price_over_ema20 > 1.0"),
            _make_profile("Z-USD", 0, 2, "price_over_ema20 > 1.0"),
        ],
        cap=2,
        pid_features={"A-USD": frame_a, "Z-USD": frame_z},
        endpoints_by_pid={"A-USD": validated_a, "Z-USD": validated_z},
        bar_duration_ms=_WBAR,
    )
    assert metrics.trade_count == 2
    assert len(_w_realized(telemetry)) == 2


def test_a_candidate_skipped_by_the_cap_never_realizes_pnl():
    """A checkpoint exists for every candidate accounting time, so a position that was never
    opened must not be realized by one.

    Codex 3c639b82: asserting only `trade_count == 1` cannot tell WHICH candidate was
    realized, so the labels differ per product and the realized value is checked exactly.
    Z-USD wins the contested slot on deflated profit, so 0.5 must appear and 0.1 must not.
    """
    frame_a, validated_a = _wire("A-USD", [1, 1, 1], horizon=2, label=0.1)
    frame_z, validated_z = _wire("Z-USD", [1, 1, 1], horizon=2, label=0.5)
    metrics, telemetry = simulate_portfolio(
        [
            _make_profile("A-USD", 0, 2, "price_over_ema20 > 1.0", deflated=0.01),
            _make_profile("Z-USD", 0, 2, "price_over_ema20 > 1.0", deflated=0.09),
        ],
        cap=1,
        pid_features={"A-USD": frame_a, "Z-USD": frame_z},
        endpoints_by_pid={"A-USD": validated_a, "Z-USD": validated_z},
        bar_duration_ms=_WBAR,
    )
    realized = _w_realized(telemetry)
    assert metrics.trade_count == 1, "the skipped candidate's checkpoint realized nothing"
    assert [row.realized_pnl for row in realized] == pytest.approx([0.5])
    assert all(row.realized_pnl != pytest.approx(0.1) for row in realized), (
        "A-USD lost the contested slot, so its label must never be realized"
    )
    assert realized[0].closed_profile_id.startswith("Z-USD")
    assert metrics.cumulative_profit_raw == pytest.approx(0.5)


def test_a_missing_endpoint_raises_rather_than_guessing_an_exit():
    """§9.3: never fall back to horizon arithmetic; a silent fallback is what the contract
    removes."""
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    empty = ValidatedEndpoints(
        records=(),
        data_id=validated.data_id,
        bar_duration_ms=_WBAR,
        exit_config=dict(validated.exit_config),
        feature_recipe=validated.feature_recipe,
        frame_fingerprint=validated.frame_fingerprint,
        token=_VALIDATED_BY_LOADER,
    )
    with pytest.raises(MissingEndpoint):
        _w_run("A-USD", frame, empty, horizon=2)


def test_raw_endpoints_are_refused_by_the_replay():
    """A consumer may not hand the replay records nobody validated against the frame."""
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    with pytest.raises(TypeError, match="ValidatedEndpoints"):
        _w_run("A-USD", frame, list(validated.records), horizon=2)


def test_the_slot_metric_denominator_counts_decision_instants_only():
    """Codex 3c639b82: an earlier version ran the SAME call twice and compared, which proves
    nothing. This patches the checkpoint provider to inject instants that belong to no
    position, leaving the candidates, labels and decision instants untouched -- so only the
    irrelevant checkpoints vary.

    Contract §9.5: checkpoint-only instants exist so a due position can be examined, and must
    never become sampling points, or both occupancy metrics shift under an input that changed
    nothing about the trading.
    """
    from unittest.mock import patch

    import tools.strategy_discovery.portfolio_sim as ps

    frame, validated = _wire("A-USD", [1] * 8, horizon=2)
    baseline, _ = _w_run("A-USD", frame, validated, horizon=2)

    real = ps.close_checkpoints
    # instants far outside the frame, so no open position can ever be due at them
    spurious = [10_000 * _WBAR + i * _WBAR for i in range(25)]

    def _with_spurious(instants):
        return real(list(instants) + spurious)

    with patch.object(ps, "close_checkpoints", side_effect=_with_spurious):
        perturbed, _ = _w_run("A-USD", frame, validated, horizon=2)

    assert perturbed.pct_slots_full == baseline.pct_slots_full
    assert perturbed.mean_concurrent == baseline.mean_concurrent
    assert perturbed.trade_count == baseline.trade_count
    assert perturbed.cumulative_profit_raw == pytest.approx(baseline.cumulative_profit_raw)


def test_bar_duration_and_endpoints_are_required_keywords():
    """Neither may be defaulted: a caller that omitted one would silently keep the old
    wall-clock behaviour."""
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    profile = _make_profile("A-USD", 0, 2, "price_over_ema20 > 1.0")
    with pytest.raises(TypeError):
        simulate_portfolio([profile], cap=1, pid_features={"A-USD": frame})


def test_a_frame_the_records_were_not_validated_against_is_refused():
    """Codex 031559c7, the binding that a construction gate cannot provide.

    Frame B keeps every row id and timestamp but changes a feature the rule reads. The records
    are legitimately validated -- against frame A -- and both are ordinary public values, so
    nothing but recomputing the frame's identity catches it. Without the check the replay would
    select trades on B and realize A's outcomes.
    """
    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    mutated = frame.copy()
    mutated.loc[0, "close"] = float(mutated.loc[0, "close"]) + 1.0
    assert mutated["ts"].tolist() == frame["ts"].tolist()
    assert mutated["source_row_id"].tolist() == frame["source_row_id"].tolist()
    with pytest.raises(MissingEndpoint, match="validated against a different frame"):
        _w_run("A-USD", mutated, validated, horizon=2)


def test_a_frame_differing_only_in_a_rule_feature_is_refused():
    """Codex 5ad67efc: the case `data_id` structurally cannot catch.

    `build_data_id` hashes ts, close, high, low, atr14_pct and the config -- and nothing else.
    So a frame with identical clocks, identical OHLC, identical ATR and a DIFFERENT
    `price_over_ema20` has the SAME data_id, while firing different rules. The replay would
    select trades that frame A's records never described.

    This is what the full-content fingerprint is for, and asserting the two data_ids are equal
    is the part that makes the test prove something.
    """
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint
    from tools.strategy_discovery.endpoint_dataset import build_data_id

    frame, validated = _wire("A-USD", [1] * 6, horizon=2)
    mutated = frame.copy()
    mutated["price_over_ema20"] = [0.5] * len(mutated)  # would fire a different rule

    def _data_id(f):
        return build_data_id(
            product_id="A-USD",
            bar_duration_ms=_WBAR,
            timestamps=f["ts"].tolist(),
            closes=f["close"].tolist(),
            highs=f["high"].tolist(),
            lows=f["low"].tolist(),
            atr_pcts=f["atr14_pct"].tolist(),
            feature_recipe="atr14_pct_wilder_v1",
            config=dict(validated.exit_config),
        )

    assert _data_id(mutated) == _data_id(frame), (
        "the premise: data_id does not cover rule feature columns, so it cannot catch this"
    )
    assert frame_fingerprint(mutated) != frame_fingerprint(frame)

    with pytest.raises(MissingEndpoint, match="fingerprint"):
        _w_run("A-USD", mutated, validated, horizon=2)
