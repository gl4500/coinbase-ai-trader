"""Unit tests for backend/tools/xgb_v4_features.py — XGB v4 OHLCV-5 extractor.

5 channels (open/high/low/close/volume) x 3 tiers (micro/meso/macro)
x 10 stats = 150 features. Pure functions, no module state.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools import xgb_v4_features as v4  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────


class TestConstants:
    def test_channel_fields_order(self):
        assert v4._CHANNEL_FIELDS == ("open", "high", "low", "close", "volume")

    def test_n_channels_derived(self):
        assert v4.N_CHANNELS_V4 == 5
        assert v4.N_CHANNELS_V4 == len(v4._CHANNEL_FIELDS)

    def test_tier_windows(self):
        assert v4.TIER_WINDOWS_V4 == {"micro": 60, "meso": 168, "macro": 336}

    def test_tier_weights(self):
        assert v4.TIER_WEIGHTS_V4 == {"micro": 1.0, "meso": 2.0, "macro": 3.0}

    def test_stat_names_order(self):
        assert v4._STAT_NAMES_V4 == (
            "last",
            "mean",
            "std",
            "slope",
            "min",
            "max",
            "pct_rank",
            "dlt5",
            "dlt10",
            "dlt30",
        )

    def test_n_features_derived(self):
        assert v4.N_FEATURES_V4 == 150
        assert v4.N_FEATURES_V4 == v4.N_CHANNELS_V4 * v4.N_TIERS_V4 * v4.N_STATS_V4


# ── _slope ────────────────────────────────────────────────────────────────


class TestSlope:
    def test_empty_returns_zero(self):
        assert v4._slope(np.array([], dtype=np.float64)) == 0.0

    def test_single_returns_zero(self):
        assert v4._slope(np.array([5.0])) == 0.0

    def test_constant_returns_zero(self):
        assert v4._slope(np.array([3.0, 3.0, 3.0, 3.0])) == 0.0

    def test_linear_up_returns_one(self):
        # y = x: slope = 1.0
        assert v4._slope(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_linear_down_returns_negative(self):
        assert v4._slope(np.array([3.0, 2.0, 1.0, 0.0])) == pytest.approx(-1.0)


# ── _pct_rank ─────────────────────────────────────────────────────────────


class TestPctRank:
    def test_empty_returns_zero(self):
        assert v4._pct_rank(np.array([], dtype=np.float64)) == 0.0

    def test_single_returns_zero(self):
        assert v4._pct_rank(np.array([5.0])) == 0.0

    def test_last_is_max(self):
        # 4 values [1,2,3,10], last=10 is greater than 3/4 others
        # Convention: (below + 0.5*equal) / n
        out = v4._pct_rank(np.array([1.0, 2.0, 3.0, 10.0]))
        assert out == pytest.approx((3 + 0.5 * 1) / 4)

    def test_last_is_min(self):
        # last=1 is less than 3/4 others
        out = v4._pct_rank(np.array([10.0, 5.0, 2.0, 1.0]))
        assert out == pytest.approx((0 + 0.5 * 1) / 4)


# ── _delta_at ─────────────────────────────────────────────────────────────


class TestDeltaAt:
    def test_too_short_returns_zero(self):
        assert v4._delta_at(np.array([1.0, 2.0]), lookback=5) == 0.0

    def test_exact_lookback(self):
        # values[-1] - values[-1 - 5] for lookback=5
        # array of 6: values[-1]=v[5], values[-6]=v[0]
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 10.0])
        assert v4._delta_at(v, lookback=5) == 10.0 - 1.0

    def test_longer_than_needed(self):
        v = np.arange(20, dtype=np.float64)
        # last is 19; lookback=5 -> 19 - v[14] = 19 - 14 = 5
        assert v4._delta_at(v, lookback=5) == 5.0


# ── _extract_field ────────────────────────────────────────────────────────


class TestExtractField:
    def test_empty_candles_returns_empty(self):
        out = v4._extract_field([], "close")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        assert out.size == 0

    def test_extracts_one_column(self):
        candles = [
            {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
            {"open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200.0},
        ]
        assert list(v4._extract_field(candles, "close")) == [1.5, 2.0]
        assert list(v4._extract_field(candles, "volume")) == [100.0, 200.0]


# ── _compute_stats ────────────────────────────────────────────────────────


class TestComputeStats:
    def test_empty_returns_ten_zeros(self):
        out = v4._compute_stats(np.array([], dtype=np.float64))
        assert out.shape == (10,)
        assert (out == 0.0).all()

    def test_known_series(self):
        # Series 1..10
        v = np.arange(1, 11, dtype=np.float64)
        out = v4._compute_stats(v)
        assert out.shape == (10,)
        # _STAT_NAMES_V4 order: last, mean, std, slope, min, max, pct_rank, dlt5, dlt10, dlt30
        assert out[0] == 10.0  # last
        assert out[1] == pytest.approx(5.5)  # mean
        assert out[2] == pytest.approx(v.std())  # std
        assert out[3] == pytest.approx(1.0)  # slope
        assert out[4] == 1.0  # min
        assert out[5] == 10.0  # max
        assert out[6] == pytest.approx((9 + 0.5) / 10)  # pct_rank
        assert out[7] == 10.0 - 5.0  # dlt5 = v[-1]-v[-6]
        assert out[8] == 0.0  # dlt10 needs len>=11
        assert out[9] == 0.0  # dlt30 needs len>=31


# ── feature_names_v4 ──────────────────────────────────────────────────────


class TestFeatureNames:
    def test_returns_150_names(self):
        names = v4.feature_names_v4()
        assert len(names) == 150

    def test_layout_channel_then_tier_then_stat(self):
        names = v4.feature_names_v4()
        # First 30 should be channel 0 (open) across 3 tiers
        assert names[0] == "ch0_micro_last"
        assert names[1] == "ch0_micro_mean"
        assert names[9] == "ch0_micro_dlt30"
        assert names[10] == "ch0_meso_last"
        assert names[20] == "ch0_macro_last"
        assert names[29] == "ch0_macro_dlt30"
        assert names[30] == "ch1_micro_last"  # channel 1 starts here
        assert names[149] == "ch4_macro_dlt30"  # final

    def test_unique(self):
        names = v4.feature_names_v4()
        assert len(set(names)) == len(names)


# ── feature_weights_v4 ────────────────────────────────────────────────────


class TestFeatureWeights:
    def test_length_matches_names(self):
        assert len(v4.feature_weights_v4()) == len(v4.feature_names_v4())

    def test_dtype_float64(self):
        assert v4.feature_weights_v4().dtype == np.float64

    def test_tier_weight_values(self):
        names = v4.feature_names_v4()
        weights = v4.feature_weights_v4()
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert weights[i] == 1.0, f"{name} should be micro=1.0"
            elif "_meso_" in name:
                assert weights[i] == 2.0, f"{name} should be meso=2.0"
            elif "_macro_" in name:
                assert weights[i] == 3.0, f"{name} should be macro=3.0"


# ── extract_v4 ────────────────────────────────────────────────────────────


class TestExtractV4:
    def _make_candle(self, c: int) -> Dict[str, float]:
        # Distinct values per field so we can detect mis-routing
        return {
            "open": c * 1.0,
            "high": c * 1.0 + 0.5,
            "low": c * 1.0 - 0.5,
            "close": c * 1.0 + 0.25,
            "volume": c * 10.0,
        }

    def _make_tier(self, n: int) -> List[Dict[str, float]]:
        return [self._make_candle(i + 1) for i in range(n)]

    def test_shape_and_names(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        assert features.shape == (1, 150)
        assert features.dtype == np.float64
        assert names == v4.feature_names_v4()

    def test_channel_3_reads_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # ch3_micro_last corresponds to close of last micro candle.
        # Last micro candle is _make_candle(60): close = 60.25
        idx = names.index("ch3_micro_last")
        assert features[0, idx] == 60.25

    def test_channel_4_reads_volume_not_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # ch4_micro_last: volume of last micro candle = 60 * 10 = 600
        idx_v = names.index("ch4_micro_last")
        idx_c = names.index("ch3_micro_last")
        assert features[0, idx_v] == 600.0
        assert features[0, idx_c] == 60.25
        # Distinct from close — proves volume routing works.
        assert features[0, idx_v] != features[0, idx_c]

    def test_empty_tier_zeros_its_slots(self):
        candles_by_tier = {
            "micro": [],  # empty
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v4.extract_v4(candles_by_tier)
        # All micro slots (50 total) should be zero
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0, f"{name} should be zero (empty tier)"

    def test_missing_tier_key_zeros_slots(self):
        candles_by_tier = {
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }  # no "micro" key
        features, names = v4.extract_v4(candles_by_tier)
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0

    def test_missing_ohlcv_field_raises(self):
        bad = [{"open": 1.0, "high": 2.0, "low": 0.5}]  # missing close/volume
        with pytest.raises(KeyError):
            v4.extract_v4({"micro": bad, "meso": [], "macro": []})

    def test_determinism(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        f1, _ = v4.extract_v4(candles_by_tier)
        f2, _ = v4.extract_v4(candles_by_tier)
        assert (f1 == f2).all()


# ── Dispatcher integration (xgb_features.extract_features) ────────────────


class TestDispatcherV4Branch:
    def test_extract_features_v4_routes_to_v4(self):
        from tools.xgb_features import extract_features

        candles_by_tier = {
            "micro": [{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}] * 60,
            "meso": [{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}] * 168,
            "macro": [{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}] * 336,
        }
        features, names = extract_features(candles_by_tier, feature_set="v4")
        assert features.shape == (1, 150)
        assert len(names) == 150
        assert names[0] == "ch0_micro_last"

    def test_extract_features_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features

        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features({}, feature_set="v99")
