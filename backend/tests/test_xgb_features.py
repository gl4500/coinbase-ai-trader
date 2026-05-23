"""TDD tests for tools/xgb_features.py — tabular feature extractor for XGBoost.

Converts CNN's [28 channels x 60 timesteps] samples into ~280 tabular
features for XGBoost. Honors _TRAINING_CONSTANT_CHANNELS = {17, 18, 19}
mask: features for those channels must be zero so XGBoost cannot exploit a
signal that inference (which masks them) doesn't have.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Constants must match agents/cnn_agent.py
N_CHANNELS = 28
SEQ_LEN = 60
MASKED_CHANNELS = frozenset({17, 18, 19})

# Per-channel stats: last, mean, std, slope, min, max,
# percentile_rank, delta_5, delta_10, delta_30 = 10
STATS_PER_CHANNEL = 10
EXPECTED_N_FEATURES = N_CHANNELS * STATS_PER_CHANNEL  # 280


def _sample_zeros() -> np.ndarray:
    return np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)


def _sample_random(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_CHANNELS, SEQ_LEN)).astype(np.float32)


# ── Channel-count sync with cnn_agent ───────────────────────────────────

class TestChannelCountSyncWithCnnAgent:
    """xgb_features.N_CHANNELS must always match agents.cnn_agent.N_CHANNELS.

    A drift between the two silently neutralizes XGB at inference: with
    cnn_agent.N=28 and xgb_features.N=27, FeatureBuilder.build emits
    [N,28,T] but extract_features raises ValueError on shape mismatch,
    which agents/xgb_signal.py catches and returns _NEUTRAL=0.5 — every
    XGB signal becomes neutral and the live $50k system silently degrades.
    """

    def test_n_channels_matches_cnn_agent(self):
        from tools.xgb_features import N_CHANNELS as XGB_N
        from agents.cnn_agent import N_CHANNELS as CNN_N
        assert XGB_N == CNN_N, (
            f"xgb_features.N_CHANNELS={XGB_N} != cnn_agent.N_CHANNELS={CNN_N}; "
            f"drift would silently neutralize XGB on shape mismatch."
        )

    def test_extract_features_accepts_full_channel_sample(self):
        from tools.xgb_features import extract_features, N_CHANNELS as XGB_N
        from agents.cnn_agent import N_CHANNELS as CNN_N
        sample = np.zeros((1, CNN_N, SEQ_LEN), dtype=np.float32)
        features, names = extract_features(sample)
        assert features.shape[1] == XGB_N * 10, (
            f"expected {XGB_N * 10} features, got {features.shape[1]}"
        )


# ── Module shape & signature ────────────────────────────────────────────

class TestSignature:

    def test_extract_features_exists(self):
        from tools.xgb_features import extract_features
        assert callable(extract_features)

    def test_returns_array_and_names(self):
        from tools.xgb_features import extract_features
        sample = _sample_zeros()[None, :, :]
        features, names = extract_features(sample)
        assert isinstance(features, np.ndarray)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_module_exports_masked_channels_constant(self):
        from tools.xgb_features import MASKED_CHANNELS as MC
        assert MC == MASKED_CHANNELS, (
            "MASKED_CHANNELS must mirror agents/cnn_agent.py "
            "_TRAINING_CONSTANT_CHANNELS exactly"
        )


# ── Feature shape ───────────────────────────────────────────────────────

class TestShape:

    def test_single_sample_returns_correct_n_features(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=0)[None, :, :]
        features, names = extract_features(sample)
        assert features.shape == (1, EXPECTED_N_FEATURES)
        assert len(names) == EXPECTED_N_FEATURES

    def test_batch_of_samples_returns_correct_shape(self):
        from tools.xgb_features import extract_features
        batch = np.stack([_sample_random(seed=i) for i in range(8)])
        features, _ = extract_features(batch)
        assert features.shape == (8, EXPECTED_N_FEATURES)

    def test_feature_names_are_unique(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_sample_zeros()[None, :, :])
        assert len(set(names)) == len(names)


# ── No NaN/Inf ──────────────────────────────────────────────────────────

class TestFinite:

    def test_no_nan_on_zeros(self):
        from tools.xgb_features import extract_features
        features, _ = extract_features(_sample_zeros()[None, :, :])
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()

    def test_no_nan_on_constant_channel(self):
        """All-constant input has std=0 — pct_rank, slope must not divide by zero."""
        from tools.xgb_features import extract_features
        sample = np.full((N_CHANNELS, SEQ_LEN), 0.5, dtype=np.float32)
        features, _ = extract_features(sample[None, :, :])
        assert not np.isnan(features).any()
        assert not np.isinf(features).any()


# ── Determinism ─────────────────────────────────────────────────────────

class TestDeterminism:

    def test_same_input_same_output(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=42)[None, :, :]
        f1, _ = extract_features(sample)
        f2, _ = extract_features(sample)
        np.testing.assert_array_equal(f1, f2)


# ── Mask honored ────────────────────────────────────────────────────────

class TestMask:

    def test_masked_channel_features_are_zero(self):
        """Channels 17, 18, 19 produce all-zero features so XGBoost cannot
        train on a signal that inference (post _zero_mask_channels) zeroes."""
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=7)[None, :, :]
        features, names = extract_features(sample)
        masked_idx = [
            i for i, n in enumerate(names)
            if any(n.startswith(f"ch{m}_") for m in MASKED_CHANNELS)
        ]
        assert len(masked_idx) == len(MASKED_CHANNELS) * STATS_PER_CHANNEL
        np.testing.assert_array_equal(
            features[0, masked_idx], np.zeros(len(masked_idx))
        )

    def test_non_masked_channels_have_nonzero_features(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=11)[None, :, :]
        features, names = extract_features(sample)
        non_masked_idx = [
            i for i, n in enumerate(names)
            if not any(n.startswith(f"ch{m}_") for m in MASKED_CHANNELS)
        ]
        assert features[0, non_masked_idx].any(), (
            "Non-masked channels must produce some non-zero features on random input"
        )


# ── Ablation-driven anti-signal drop list ───────────────────────────────

# Distinct from MASKED_CHANNELS (which protects against CNN train/serve
# skew on {17, 18, 19}). XGB_DROP_CHANNELS is ablation-discovered: dropping
# these from XGB lifts pooled AUC. Keep separate so the CNN mask stays
# minimal and the rationales don't get conflated.
XGB_DROP_CHANNELS = frozenset({21, 24})


class TestXGBDropChannels:

    def test_module_exports_xgb_drop_channels_constant(self):
        from tools.xgb_features import XGB_DROP_CHANNELS as X
        assert X == XGB_DROP_CHANNELS, (
            "XGB_DROP_CHANNELS must be {21, 24} per #146 ablation findings: "
            "ch21 btc_corr_20 (+0.0013) and ch24 ivrv_20 (+0.0010) are "
            "anti-signal in pooled XGB"
        )

    def test_drop_channels_disjoint_from_masked(self):
        """The two sets exist for different reasons and must not overlap —
        if a channel is CNN-masked it's already zero; double-listing hides bugs."""
        from tools.xgb_features import MASKED_CHANNELS as M, XGB_DROP_CHANNELS as X
        assert M.isdisjoint(X)

    def test_drop_channel_features_are_zero(self):
        """Channels in XGB_DROP_CHANNELS must produce all-zero stats."""
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=23)[None, :, :]
        features, names = extract_features(sample)
        drop_idx = [
            i for i, n in enumerate(names)
            if any(n.startswith(f"ch{c}_") for c in XGB_DROP_CHANNELS)
        ]
        assert len(drop_idx) == len(XGB_DROP_CHANNELS) * STATS_PER_CHANNEL
        np.testing.assert_array_equal(
            features[0, drop_idx], np.zeros(len(drop_idx))
        )


# ── Feature semantics ───────────────────────────────────────────────────

class TestSemantics:

    def test_last_feature_is_last_timestep(self):
        from tools.xgb_features import extract_features
        sample = np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)
        sample[0, -1] = 0.7
        features, names = extract_features(sample[None, :, :])
        idx = names.index("ch0_last")
        assert features[0, idx] == pytest.approx(0.7, abs=1e-6)

    def test_mean_feature_matches_numpy_mean(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=3)
        expected = float(sample[0].mean())
        features, names = extract_features(sample[None, :, :])
        idx = names.index("ch0_mean")
        assert features[0, idx] == pytest.approx(expected, abs=1e-5)

    def test_slope_positive_for_rising_series(self):
        from tools.xgb_features import extract_features
        sample = np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)
        sample[0, :] = np.linspace(0.0, 1.0, SEQ_LEN)
        features, names = extract_features(sample[None, :, :])
        idx = names.index("ch0_slope")
        assert features[0, idx] > 0

    def test_slope_negative_for_falling_series(self):
        from tools.xgb_features import extract_features
        sample = np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)
        sample[0, :] = np.linspace(1.0, 0.0, SEQ_LEN)
        features, names = extract_features(sample[None, :, :])
        idx = names.index("ch0_slope")
        assert features[0, idx] < 0

    def test_min_max_match_numpy(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=5)
        features, names = extract_features(sample[None, :, :])
        i_min = names.index("ch0_min")
        i_max = names.index("ch0_max")
        assert features[0, i_min] == pytest.approx(float(sample[0].min()), abs=1e-5)
        assert features[0, i_max] == pytest.approx(float(sample[0].max()), abs=1e-5)

    def test_delta_5_is_last_minus_5_back(self):
        from tools.xgb_features import extract_features
        sample = np.zeros((N_CHANNELS, SEQ_LEN), dtype=np.float32)
        sample[0, -1] = 0.9
        sample[0, -6] = 0.2  # 5 steps back from last
        features, names = extract_features(sample[None, :, :])
        idx = names.index("ch0_delta_5")
        assert features[0, idx] == pytest.approx(0.7, abs=1e-6)


# ── feature_set="v2": Tier-1 cross-channel/temporal additions ──────────
#
# v2 = v1 (270 per-channel stats) + 10 cross-channel / multi-horizon /
# crossover features the doc flags as high-cash-flow but absent from v1.
#
# New columns (in order):
#   xt_vol_regime_ratio   ch24[-1] / (ch25[-1] + 1e-8)        (RV20/RV60)
#   xt_vol_of_vol         std(ch24)
#   xt_ret_full           ch0[-1] - ch0[0]                    (60h move)
#   xt_ret_skew           scipy-style skew of diff(ch0)
#   xt_ret_kurt           Fisher kurtosis of diff(ch0)
#   xt_rsi_below_30       1.0 if ch4[-1] < 0.3 else 0.0
#   xt_rsi_above_70       1.0 if ch4[-1] > 0.7 else 0.0
#   xt_rsi_cross_up_3     1.0 if ch4[-3]<0.3 and ch4[-1]>=0.3 else 0.0
#   xt_macd_sign_change_3 1.0 if sign(ch5[-1]) != sign(ch5[-3]) else 0.0
#   xt_funding_x_trend    ch20[-1] * sign(slope(ch0))

V2_NEW_FEATURES = (
    "xt_vol_regime_ratio",
    "xt_vol_of_vol",
    "xt_ret_full",
    "xt_ret_skew",
    "xt_ret_kurt",
    "xt_rsi_below_30",
    "xt_rsi_above_70",
    "xt_rsi_cross_up_3",
    "xt_macd_sign_change_3",
    "xt_funding_x_trend",
)
V2_EXPECTED_N_FEATURES = EXPECTED_N_FEATURES + len(V2_NEW_FEATURES)


class TestV2Shape:

    def test_v1_default_is_unchanged(self):
        """Calling without feature_set must still return 270 cols (back-compat)."""
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=1)[None, :, :]
        f, names = extract_features(sample)
        assert f.shape == (1, EXPECTED_N_FEATURES)
        assert len(names) == EXPECTED_N_FEATURES

    def test_v2_returns_280_columns(self):
        from tools.xgb_features import extract_features
        sample = _sample_random(seed=2)[None, :, :]
        f, names = extract_features(sample, feature_set="v2")
        assert f.shape == (1, V2_EXPECTED_N_FEATURES)
        assert len(names) == V2_EXPECTED_N_FEATURES

    def test_v2_appends_new_names_after_v1_block(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_sample_zeros()[None, :, :], feature_set="v2")
        # First 270 names must match v1 ordering
        _, v1_names = extract_features(_sample_zeros()[None, :, :])
        assert names[:EXPECTED_N_FEATURES] == v1_names
        assert tuple(names[EXPECTED_N_FEATURES:]) == V2_NEW_FEATURES

    def test_v2_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features
        with pytest.raises(ValueError):
            extract_features(_sample_zeros()[None, :, :], feature_set="v99")


class TestV2Finite:

    def test_v2_no_nan_on_zeros(self):
        from tools.xgb_features import extract_features
        f, _ = extract_features(_sample_zeros()[None, :, :], feature_set="v2")
        assert not np.isnan(f).any()
        assert not np.isinf(f).any()

    def test_v2_no_nan_on_constant(self):
        from tools.xgb_features import extract_features
        sample = np.full((N_CHANNELS, SEQ_LEN), 0.5, dtype=np.float32)
        f, _ = extract_features(sample[None, :, :], feature_set="v2")
        assert not np.isnan(f).any()
        assert not np.isinf(f).any()


class TestV2Semantics:

    def test_vol_regime_ratio_is_ch24_over_ch25(self):
        from tools.xgb_features import extract_features
        sample = _sample_zeros()
        sample[24, -1] = 0.4
        sample[25, -1] = 0.2
        f, names = extract_features(sample[None, :, :], feature_set="v2")
        idx = names.index("xt_vol_regime_ratio")
        assert f[0, idx] == pytest.approx(2.0, rel=1e-3)

    def test_ret_full_is_last_minus_first(self):
        from tools.xgb_features import extract_features
        sample = _sample_zeros()
        sample[0, 0] = 0.1
        sample[0, -1] = 0.7
        f, names = extract_features(sample[None, :, :], feature_set="v2")
        idx = names.index("xt_ret_full")
        assert f[0, idx] == pytest.approx(0.6, abs=1e-6)

    def test_rsi_below_30_binary(self):
        from tools.xgb_features import extract_features
        s_low = _sample_zeros()
        s_low[4, -1] = 0.25
        s_high = _sample_zeros()
        s_high[4, -1] = 0.55
        f_lo, names = extract_features(s_low[None, :, :], feature_set="v2")
        f_hi, _ = extract_features(s_high[None, :, :], feature_set="v2")
        idx = names.index("xt_rsi_below_30")
        assert f_lo[0, idx] == 1.0
        assert f_hi[0, idx] == 0.0

    def test_rsi_above_70_binary(self):
        from tools.xgb_features import extract_features
        s_hi = _sample_zeros()
        s_hi[4, -1] = 0.75
        f, names = extract_features(s_hi[None, :, :], feature_set="v2")
        idx = names.index("xt_rsi_above_70")
        assert f[0, idx] == 1.0

    def test_rsi_cross_up_3_fires_when_crossing_above_30(self):
        from tools.xgb_features import extract_features
        sample = _sample_zeros()
        sample[4, -3] = 0.25  # was below 0.3
        sample[4, -1] = 0.35  # now above 0.3
        f, names = extract_features(sample[None, :, :], feature_set="v2")
        idx = names.index("xt_rsi_cross_up_3")
        assert f[0, idx] == 1.0

    def test_rsi_cross_up_3_zero_when_already_above(self):
        from tools.xgb_features import extract_features
        sample = _sample_zeros()
        sample[4, -3] = 0.5
        sample[4, -1] = 0.6
        f, names = extract_features(sample[None, :, :], feature_set="v2")
        idx = names.index("xt_rsi_cross_up_3")
        assert f[0, idx] == 0.0

    def test_macd_sign_change_3(self):
        from tools.xgb_features import extract_features
        flip = _sample_zeros()
        flip[5, -3] = -0.05
        flip[5, -1] = 0.03
        same = _sample_zeros()
        same[5, -3] = 0.02
        same[5, -1] = 0.04
        f1, names = extract_features(flip[None, :, :], feature_set="v2")
        f2, _ = extract_features(same[None, :, :], feature_set="v2")
        idx = names.index("xt_macd_sign_change_3")
        assert f1[0, idx] == 1.0
        assert f2[0, idx] == 0.0

    def test_funding_x_trend_sign_matches_trend(self):
        """Positive funding * uptrend slope -> positive."""
        from tools.xgb_features import extract_features
        sample = _sample_zeros()
        sample[20, -1] = 0.001                       # positive funding
        sample[0, :] = np.linspace(0.0, 1.0, SEQ_LEN)  # rising
        f, names = extract_features(sample[None, :, :], feature_set="v2")
        idx = names.index("xt_funding_x_trend")
        assert f[0, idx] > 0


class TestV2MaskInvariant:

    def test_v2_does_not_recompute_masked_channels(self):
        """v2 add-ons must not source signal from MASKED_CHANNELS (17/18/19)
        beyond what v1 does (which is zero them out)."""
        from tools.xgb_features import extract_features
        rng = np.random.default_rng(123)
        s_a = rng.standard_normal((N_CHANNELS, SEQ_LEN)).astype(np.float32)
        s_b = s_a.copy()
        # Perturb only masked channels — v2 outputs must be identical
        for ch in MASKED_CHANNELS:
            s_b[ch] = rng.standard_normal(SEQ_LEN).astype(np.float32)
        f_a, _ = extract_features(s_a[None, :, :], feature_set="v2")
        f_b, _ = extract_features(s_b[None, :, :], feature_set="v2")
        np.testing.assert_array_equal(f_a, f_b)
