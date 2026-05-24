"""Tests for tools.strategy_discovery.tokenomic_stamp (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.tokenomic_stamp import (
    SupplySnapshot,
    _TOKENOMIC_COLUMNS,
    stamp_tokenomic,
)

_DAY_MS = 86_400_000


def _hourly_ts(start_day_ms: int, n_hours: int) -> np.ndarray:
    return start_day_ms + np.arange(n_hours, dtype="int64") * 3_600_000


def _trivial_supply(pid: str = "FOO-USD") -> SupplySnapshot:
    return SupplySnapshot(pid=pid, circulating=1_000_000.0, total=2_000_000.0, max_supply=None)


def test_t_plus_1_boundary_uses_yesterday_snapshot():
    # Day D = 1_000 * _DAY_MS, snapshot on that day reports MC=100, vol=10.
    # Hourly candidates start on Day D+1 — the FIRST candidate at D+1 00:00
    # must read Day D's snapshot (T+1 rule).
    d0 = 1_000 * _DAY_MS
    d1 = d0 + _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0, d1],
        "market_cap": [100.0, 200.0],
        "volume_24h": [10.0,  20.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d1, 48),                # Day D+1 00:00 .. Day D+2 23:00
        "close": np.full(48, 5.0, dtype="float64"),
    })
    out = stamp_tokenomic(df_hourly, df_daily, _trivial_supply(), drop_on_missing_volume=False)
    # Day D+1 00:00 must read Day D's snapshot (MC=100, vol=10).
    assert out.loc[0, "market_cap"] == pytest.approx(100.0)
    assert out.loc[0, "vol_24h"]    == pytest.approx(10.0)
    # Day D+2 00:00 (index 24) must read Day D+1's snapshot (MC=200, vol=20).
    assert out.loc[24, "market_cap"] == pytest.approx(200.0)
    assert out.loc[24, "vol_24h"]    == pytest.approx(20.0)


def test_forward_fill_supplies_carry_indefinitely():
    # Daily MC reported only on Day D=1000. Hourly grid spans Days D+1..D+5
    # (i.e. 4 days * 24 h = 96 hourly rows after the T+1 boundary). MC must
    # forward-fill across the whole window — slow-moving features have no
    # time cap.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],
        "market_cap": [100.0],
        "volume_24h": [10.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 96),
        "close": np.full(96, 5.0, dtype="float64"),
    })
    out = stamp_tokenomic(df_hourly, df_daily, _trivial_supply(), drop_on_missing_volume=False)
    assert out["market_cap"].notna().all()
    np.testing.assert_allclose(out["market_cap"].to_numpy(), 100.0)


def test_missing_volume_drops_candidate_row():
    # Daily snapshot exists for Day D but not for Day D+1. Hourly rows on
    # Day D+1 (which read Day D's snapshot — vol present) survive; hourly
    # rows on Day D+2 (which read Day D+1's snapshot — vol missing) drop.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],          # only Day D — Day D+1 is missing
        "market_cap": [100.0],
        "volume_24h": [10.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 48),     # 24h on D+1 + 24h on D+2
        "close": np.full(48, 5.0, dtype="float64"),
    })
    # With drop_on_missing_volume=True, ALL 48 rows survive — because vol
    # IS present (forward-filled from Day D for all 48 hours via merge_asof).
    # So we need to construct the missing-vol case differently: leave a
    # genuine gap by passing a daily frame where vol_24h is NaN on Day D+1.
    df_daily_with_gap = pd.DataFrame({
        "ts":         [d0,    d0 + _DAY_MS],
        "market_cap": [100.0, 200.0],
        "volume_24h": [10.0,  float("nan")],
    })
    out = stamp_tokenomic(df_hourly, df_daily_with_gap, _trivial_supply(), drop_on_missing_volume=True)
    # Day D+1 rows (24h) keep vol=10. Day D+2 rows would read NaN — dropped.
    assert len(out) == 24, f"expected 24 surviving rows, got {len(out)}"
    np.testing.assert_allclose(out["vol_24h"].to_numpy(), 10.0)


def test_fdv_derived_from_price_and_total_supply():
    # FDV = close_t * supply.total at each row, and fdv_over_mc = fdv / market_cap.
    d0 = 1_000 * _DAY_MS
    df_daily = pd.DataFrame({
        "ts":         [d0],
        "market_cap": [50_000.0],   # 50k market_cap
        "volume_24h": [1_000.0],
    })
    df_hourly = pd.DataFrame({
        "ts":    _hourly_ts(d0 + _DAY_MS, 3),
        "close": np.array([2.0, 4.0, 8.0]),
    })
    supply = SupplySnapshot(pid="FOO-USD", circulating=10_000.0, total=25_000.0, max_supply=None)
    out = stamp_tokenomic(df_hourly, df_daily, supply, drop_on_missing_volume=False)
    np.testing.assert_allclose(out["fdv"].to_numpy(),
                                np.array([2.0, 4.0, 8.0]) * 25_000.0)
    np.testing.assert_allclose(out["fdv_over_mc"].to_numpy(),
                                (np.array([2.0, 4.0, 8.0]) * 25_000.0) / 50_000.0)
    # circ_over_total is a constant row-derived value: 10_000 / 25_000 = 0.4
    np.testing.assert_allclose(out["circ_over_total"].to_numpy(),
                                np.full(3, 0.4))
    np.testing.assert_allclose(out["vol_over_mc"].to_numpy(),
                                np.full(3, 1_000.0 / 50_000.0))
