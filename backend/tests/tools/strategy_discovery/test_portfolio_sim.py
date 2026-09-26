"""Tests for tools.strategy_discovery.portfolio_sim (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def _make_pid_features(pid: str, n_hours: int, ema_ratio: float, label: float):
    """Synthetic Phase 2 frame: feature is `price_over_ema20`, label is constant."""
    ts = (np.arange(n_hours, dtype="int64") * 3_600_000).tolist()
    return pd.DataFrame(
        {
            "ts": ts,
            "close": [1.0] * n_hours,
            "price_over_ema20": [ema_ratio] * n_hours,
            "vol_over_mc": [0.01] * n_hours,
            "label_h1": [label] * n_hours,
            "label_h24": [label] * n_hours,
        }
    )


def test_concurrency_cap_blocks_new_entries_when_full():
    # 3 pids, each fires on the same rule, 24h horizon.
    # Cap=2 → at most 2 open at any time; the 3rd pid never enters.
    pids = ["A-USD", "B-USD", "C-USD"]
    profiles = [_make_profile(pid, 0, 24, "price_over_ema20 > 1.0", deflated=0.05) for pid in pids]
    pid_features = {
        pid: _make_pid_features(pid, n_hours=200, ema_ratio=1.5, label=0.10) for pid in pids
    }
    metrics, telemetry = simulate_portfolio(profiles, cap=2, pid_features=pid_features)
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
    metrics, telemetry = simulate_portfolio(profiles, cap=3, pid_features={"BTC-USD": feats})
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
    metrics, telemetry = simulate_portfolio(profiles, cap=1, pid_features=pid_features)
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
    metrics, telemetry = simulate_portfolio([profile], cap=1, pid_features={"BTC-USD": feats})
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
    metrics, telemetry = simulate_portfolio(profiles, cap=2, pid_features=pid_features)
    assert 0.0 <= metrics.pct_slots_full <= 1.0
    assert 0.0 <= metrics.mean_concurrent <= 2.0
    # With cap=2 and 3 always-firing pids, we expect cap to be hit some of the time
    assert metrics.pct_slots_full > 0.3
