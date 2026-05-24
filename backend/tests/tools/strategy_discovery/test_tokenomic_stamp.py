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
