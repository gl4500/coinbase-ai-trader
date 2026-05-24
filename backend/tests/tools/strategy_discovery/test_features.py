"""Tests for tools.strategy_discovery.features (Phase 2)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.strategy_discovery.features import (
    _TREND_COLUMNS,
    add_trend_features,
    first_valid_index,
)


def _synthetic_ohlcv(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 + rng.normal(0.0, 1.0, size=n).cumsum()
    high = close + np.abs(rng.normal(0.0, 0.5, size=n))
    low = close - np.abs(rng.normal(0.0, 0.5, size=n))
    open_ = close + rng.normal(0.0, 0.2, size=n)
    ts = np.arange(n, dtype="int64") * 3_600_000
    return pd.DataFrame({"ts": ts, "open": open_, "high": high, "low": low, "close": close})


def test_ema20_matches_pandas_ewm():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    expected = df["close"] / df["close"].ewm(span=20, adjust=False).mean()
    np.testing.assert_allclose(
        out["price_over_ema20"].to_numpy(),
        expected.to_numpy(),
        rtol=1e-9,
    )


def test_ema_warmup_returns_finite_after_n_bars():
    df = _synthetic_ohlcv(400)
    out = add_trend_features(df)
    # All three EMA ratios must be finite for every row at or after index 200.
    for col in ("price_over_ema20", "price_over_ema50", "price_over_ema200"):
        tail = out[col].iloc[200:]
        assert np.isfinite(tail.to_numpy()).all(), f"non-finite values in {col} after warmup"
