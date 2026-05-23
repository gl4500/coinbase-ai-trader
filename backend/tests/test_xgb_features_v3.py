"""TDD tests for tools/xgb_features.py feature_set='v3' (mixed-lookback).

Contract:
    extract_features(candles_by_tier, feature_set="v3")
        candles_by_tier = {"micro": [...60], "meso": [...168], "macro": [...336]}
    Returns (features [350], names [350]) where:
        - 18 micro non-masked channels x 10 stats = 180 live
        - 4 meso channels x 20 stats (60 + 168) = 80 live
        - 3 macro channels x 20 stats (60 + 336) = 60 live
        - 3 masked channels x 10 stats = 30 zero
        - Total feature_names = 350

    feature_weights_v3() returns the matching 350-long weight vector.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(close=100.0):
    return {"start": 0, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1000.0}


def _tiers(n_micro=60, n_meso=168, n_macro=336):
    return {
        "micro": [_candle(100 + i * 0.1) for i in range(n_micro)] if n_micro else [],
        "meso":  [_candle(100 + i * 0.1) for i in range(n_meso)]  if n_meso  else [],
        "macro": [_candle(100 + i * 0.1) for i in range(n_macro)] if n_macro else [],
    }


class TestV3Shape:
    def test_extract_v3_returns_350_features(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(), feature_set="v3")
        assert feats.shape == (1, 350)
        assert len(names) == 350

    def test_v3_feature_names_unique(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert len(names) == len(set(names))

    def test_v3_names_disjoint_from_v1(self):
        from tools.xgb_features import extract_features
        _, v3_names = extract_features(_tiers(), feature_set="v3")
        v1_names_with_infix = [n for n in v3_names
                               if "_m060_" in n or "_m168_" in n or "_m336_" in n]
        assert len(v1_names_with_infix) > 0, "v3 must produce _mWWW_ infix names"


class TestV3NameScheme:
    def test_micro_channels_use_v1_scheme(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch0_last" in names
        assert "ch0_m060_last" not in names

    def test_meso_channels_have_both_windows(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch15_m060_last" in names
        assert "ch15_m168_last" in names

    def test_macro_channels_have_both_windows(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch20_m060_last" in names
        assert "ch20_m336_last" in names

    def test_masked_channels_keep_v1_scheme(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        assert "ch17_last" in names  # masked but still slotted
        assert "ch17_m168_last" not in names


class TestV3PerTierCount:
    def test_micro_channels_produce_60bar_stats_only(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch0_stats = [n for n in names if n.startswith("ch0_")]
        assert len(ch0_stats) == 10

    def test_meso_channels_produce_20_stats(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch15_stats = [n for n in names if n.startswith("ch15_")]
        assert len(ch15_stats) == 20

    def test_macro_channels_produce_20_stats(self):
        from tools.xgb_features import extract_features
        _, names = extract_features(_tiers(), feature_set="v3")
        ch20_stats = [n for n in names if n.startswith("ch20_")]
        assert len(ch20_stats) == 20


class TestZeroFill:
    def test_empty_meso_zeros_meso_slots_only(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_meso=0), feature_set="v3")
        for i, n in enumerate(names):
            if "_m168_" in n:
                assert feats[0, i] == 0.0, f"{n} should be zero when meso is empty"

    def test_empty_macro_zeros_macro_slots_only(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_macro=0), feature_set="v3")
        for i, n in enumerate(names):
            if "_m336_" in n:
                assert feats[0, i] == 0.0, f"{n} should be zero when macro is empty"

    def test_empty_micro_zeros_all_micro_only_features(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(n_micro=0), feature_set="v3")
        for i, n in enumerate(names):
            if not ("_m168_" in n or "_m336_" in n):
                assert feats[0, i] == 0.0, f"{n} should be zero when micro is empty"

    def test_masked_channels_always_zero(self):
        from tools.xgb_features import extract_features
        feats, names = extract_features(_tiers(), feature_set="v3")
        for i, n in enumerate(names):
            if n.startswith(("ch17_", "ch18_", "ch19_")):
                assert feats[0, i] == 0.0, f"masked {n} must be zero"


class TestFeatureWeights:
    def test_feature_weights_v3_length_350(self):
        from tools.xgb_features import feature_weights_v3
        w = feature_weights_v3()
        assert len(w) == 350

    def test_macro_weights_higher_than_meso_higher_than_micro(self):
        from tools.xgb_features import (
            feature_weights_v3, _v3_feature_names,
            MESO_CHANNELS, MACRO_CHANNELS,
        )
        w = feature_weights_v3()
        names = _v3_feature_names()
        # True micro-only: not masked, not a meso/macro channel slot.
        # Meso/macro channels' m060 baseline slots inherit their tier weight per spec 4.3.
        micro_w = []
        meso_w = []
        macro_w = []
        for i, n in enumerate(names):
            ch_idx = int(n.split("_", 1)[0][2:])
            if ch_idx in (17, 18, 19):
                continue
            if ch_idx in MACRO_CHANNELS:
                macro_w.append(w[i])
            elif ch_idx in MESO_CHANNELS:
                meso_w.append(w[i])
            else:
                micro_w.append(w[i])
        assert all(v == 1.0 for v in micro_w), f"micro weights not all 1.0: {set(micro_w)}"
        assert all(v == 2.0 for v in meso_w),  f"meso weights not all 2.0: {set(meso_w)}"
        assert all(v == 3.0 for v in macro_w), f"macro weights not all 3.0: {set(macro_w)}"

    def test_masked_channel_weights_zero(self):
        from tools.xgb_features import feature_weights_v3, _v3_feature_names
        w = feature_weights_v3()
        names = _v3_feature_names()
        masked_w = [w[i] for i, n in enumerate(names)
                    if n.startswith(("ch17_", "ch18_", "ch19_"))]
        assert all(v == 0.0 for v in masked_w)


class TestUnknownFeatureSet:
    def test_unknown_feature_set_raises(self):
        from tools.xgb_features import extract_features
        with pytest.raises(ValueError, match="unknown feature_set"):
            extract_features(_tiers(), feature_set="v99")
