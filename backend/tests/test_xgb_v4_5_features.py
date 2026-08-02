"""Unit tests for backend/tools/xgb_v4_5_features.py — v4.5 7-channel extractor.

5 OHLCV channels + 2 BB channels (bb_position, bb_width) × 3 tiers
× 10 stats = 210 features. Pure functions, no module state.
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

from tools import xgb_v4_5_features as v45  # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    def test_channel_names_order(self):
        assert v45._CHANNEL_NAMES == (
            "open",
            "high",
            "low",
            "close",
            "volume",
            "bb_position",
            "bb_width",
        )

    def test_n_channels_derived(self):
        assert v45.N_CHANNELS_V45 == 7
        assert v45.N_CHANNELS_V45 == len(v45._CHANNEL_NAMES)

    def test_bb_params(self):
        assert v45.BB_PERIOD == 20
        assert v45.BB_MULT == 2.0

    def test_tier_windows(self):
        assert v45.TIER_WINDOWS_V45 == {"micro": 60, "meso": 168, "macro": 336}

    def test_n_features_derived(self):
        assert v45.N_FEATURES_V45 == 210
        assert v45.N_FEATURES_V45 == v45.N_CHANNELS_V45 * v45.N_TIERS_V45 * v45.N_STATS_V45


# ═══════════════════════════════════════════════════════════════════════════
# BB helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestBollingerHelpers:
    def test_bb_position_empty(self):
        out = v45._compute_bb_position(np.array([], dtype=np.float64))
        assert out.shape == (0,)

    def test_bb_position_pre_period_fallback(self):
        # Fewer than 20 bars: each bar gets 0.5 (mid) fallback
        out = v45._compute_bb_position(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert (out == 0.5).all()

    def test_bb_position_clamped_0_1(self):
        # 30 bars of constant 100 ⇒ mean=100, std=0, bw=0 ⇒ pos fallback 0.5
        closes = np.full(30, 100.0)
        out = v45._compute_bb_position(closes)
        # Pre-period (first 19) get 0.5
        assert (out[:19] == 0.5).all()
        # Post-period get 0.5 because zero std ⇒ no spread
        assert (out[19:] == 0.5).all()

    def test_bb_position_known_values(self):
        # 25 bars rising linearly: 80..104
        closes = np.arange(80, 105, dtype=np.float64)
        out = v45._compute_bb_position(closes)
        # Last bar: pos should be > 0.5 (close above mean)
        assert out[-1] > 0.5
        assert 0.0 <= out[-1] <= 1.0

    def test_bb_width_empty(self):
        out = v45._compute_bb_width(np.array([], dtype=np.float64))
        assert out.shape == (0,)

    def test_bb_width_pre_period_fallback(self):
        out = v45._compute_bb_width(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        assert (out == 0.0).all()

    def test_bb_width_formula(self):
        # Rising series so std > 0
        closes = np.arange(80, 110, dtype=np.float64)
        out = v45._compute_bb_width(closes)
        # Width = (upper - lower) / mean = (4 * std) / mean. Sanity check > 0
        assert out[-1] > 0.0


# ═══════════════════════════════════════════════════════════════════════════
# _compute_stats / _slope / _pct_rank / _delta_at
# ═══════════════════════════════════════════════════════════════════════════


class TestStatHelpers:
    def test_compute_stats_known_series(self):
        v = np.arange(1, 11, dtype=np.float64)
        out = v45._compute_stats(v)
        assert out.shape == (10,)
        # last, mean, std, slope, min, max, pct_rank, dlt5, dlt10, dlt30
        assert out[0] == 10.0
        assert out[1] == pytest.approx(5.5)
        assert out[3] == pytest.approx(1.0)
        assert out[4] == 1.0
        assert out[5] == 10.0
        assert out[7] == 10.0 - 5.0

    def test_slope_linear(self):
        assert v45._slope(np.array([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_pct_rank_empty_zero(self):
        assert v45._pct_rank(np.array([], dtype=np.float64)) == 0.0

    def test_delta_at_too_short(self):
        assert v45._delta_at(np.array([1.0, 2.0]), lookback=5) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# feature_names_v4_5 + feature_weights_v4_5
# ═══════════════════════════════════════════════════════════════════════════


class TestFeatureNames:
    def test_returns_210_names(self):
        names = v45.feature_names_v4_5()
        assert len(names) == 210

    def test_layout_channel_then_tier_then_stat(self):
        names = v45.feature_names_v4_5()
        assert names[0] == "ch0_micro_last"
        assert names[10] == "ch0_meso_last"
        assert names[20] == "ch0_macro_last"
        assert names[30] == "ch1_micro_last"  # next channel
        # Channel 5 = bb_position, channel 6 = bb_width
        assert names[150] == "ch5_micro_last"
        assert names[180] == "ch6_micro_last"
        assert names[209] == "ch6_macro_dlt30"

    def test_unique(self):
        names = v45.feature_names_v4_5()
        assert len(set(names)) == len(names)


class TestFeatureWeights:
    def test_length_210(self):
        assert len(v45.feature_weights_v4_5()) == 210

    def test_tier_weights(self):
        names = v45.feature_names_v4_5()
        weights = v45.feature_weights_v4_5()
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert weights[i] == 1.0
            elif "_meso_" in name:
                assert weights[i] == 2.0
            elif "_macro_" in name:
                assert weights[i] == 3.0


# ═══════════════════════════════════════════════════════════════════════════
# extract_v4_5 (the main public extractor)
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractV4_5:
    def _make_candle(self, c: int) -> Dict[str, float]:
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
        features, names = v45.extract_v4_5(candles_by_tier)
        assert features.shape == (1, 210)
        assert features.dtype == np.float64
        assert names == v45.feature_names_v4_5()

    def test_channel_3_reads_close(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch3_micro_last")
        # Last micro candle = _make_candle(60): close = 60.25
        assert features[0, idx] == 60.25

    def test_channel_5_is_bb_position(self):
        # Rising linear closes ⇒ bb_position monotonic toward upper band
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch5_micro_last")
        # bb_position is in [0, 1], not raw OHLCV value
        assert 0.0 <= features[0, idx] <= 1.0

    def test_channel_6_is_bb_width(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        idx = names.index("ch6_micro_last")
        # bb_width >= 0 (zero-spread only when std=0)
        assert features[0, idx] >= 0.0

    def test_empty_tier_zeros_its_slots(self):
        candles_by_tier = {
            "micro": [],
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        features, names = v45.extract_v4_5(candles_by_tier)
        # 7 channels × 10 stats = 70 micro slots, all zero
        for i, name in enumerate(names):
            if "_micro_" in name:
                assert features[0, i] == 0.0, f"{name} should be zero"

    def test_missing_ohlcv_field_raises(self):
        bad = [{"open": 1.0, "high": 2.0, "low": 0.5}]  # missing close/volume
        with pytest.raises(KeyError):
            v45.extract_v4_5({"micro": bad, "meso": [], "macro": []})

    def test_determinism(self):
        candles_by_tier = {
            "micro": self._make_tier(60),
            "meso": self._make_tier(168),
            "macro": self._make_tier(336),
        }
        f1, _ = v45.extract_v4_5(candles_by_tier)
        f2, _ = v45.extract_v4_5(candles_by_tier)
        assert (f1 == f2).all()


# ── Dispatcher integration (xgb_features.extract_features) ────────────────


class TestDispatcherV4_5Branch:
    def test_extract_features_v4_5_routes(self):
        from tools.xgb_features import extract_features

        candles = [{"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}]
        cbt = {
            "micro": candles * 60,
            "meso": candles * 168,
            "macro": candles * 336,
        }
        features, names = extract_features(cbt, feature_set="v4_5")
        assert features.shape == (1, 210)
        assert len(names) == 210
        assert names[0] == "ch0_micro_last"

    def test_extract_features_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features

        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features({}, feature_set="v99")
