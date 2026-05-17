"""Tabular feature extractor: CNN [28 ch x 60 t] samples -> XGBoost features.

Phase 1 of the CNN -> XGBoost transition (see
docs/superpowers/plans/2026-05-02-cnn-to-xgboost-transition.md).

v1 (default): per non-masked channel produces 10 stats: last, mean, std,
slope, min, max, percentile_rank, delta_5, delta_10, delta_30. Masked
channels (17, 18, 19) produce zeros so XGBoost cannot exploit a signal that
inference, after _zero_mask_channels, doesn't have.

v2: v1 + 10 cross-channel/temporal features the doc flags as high-cash-flow
but absent from per-channel stats — vol regime ratio, vol of vol, full-window
return, return skew/kurt, RSI threshold/cross binaries, MACD sign-change,
funding x trend interaction. None source from MASKED_CHANNELS.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

N_CHANNELS = 28
SEQ_LEN = 60

# Mirrors agents/cnn_agent.py _TRAINING_CONSTANT_CHANNELS. If that set
# changes (e.g. a new channel becomes inference-masked), update both.
MASKED_CHANNELS: frozenset = frozenset({17, 18, 19})

# Ablation-driven anti-signal drop list (#146). Distinct purpose from
# MASKED_CHANNELS — these channels are visible to the CNN at inference,
# but the pooled-top-20 ablation showed dropping them lifts XGB AUC:
#   ch21 btc_corr_20  Δ +0.0013
#   ch24 ivrv_20      Δ +0.0010
# Total ~+0.002 free lift on the 0.5129 baseline. Disjoint from MASKED_CHANNELS.
XGB_DROP_CHANNELS: frozenset = frozenset({21, 24})

_STAT_NAMES: tuple = (
    "last", "mean", "std", "slope", "min", "max",
    "pct_rank", "delta_5", "delta_10", "delta_30",
)

_V2_NEW_FEATURES: tuple = (
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


def _build_feature_names() -> List[str]:
    return [f"ch{c}_{s}" for c in range(N_CHANNELS) for s in _STAT_NAMES]


def _slope(series: np.ndarray) -> np.ndarray:
    """Per-row OLS slope of `series` (shape [N, T]) on x = [0..T-1].

    Returns shape [N]. Constant rows (var(x)=0 doesn't apply since x is
    fixed; var(y) handled by zero numerator) yield 0.
    """
    n, t = series.shape
    x = np.arange(t, dtype=np.float64)
    x_mean = x.mean()
    y_mean = series.mean(axis=1, keepdims=True)
    num = ((x - x_mean) * (series - y_mean)).sum(axis=1)
    den = ((x - x_mean) ** 2).sum()
    if den == 0:
        return np.zeros(n, dtype=np.float64)
    return num / den


def _percentile_rank(series: np.ndarray) -> np.ndarray:
    """Rank of last value within row, normalized to [0, 1]. Constant row -> 0.5."""
    last = series[:, -1:]
    below = (series < last).sum(axis=1)
    equal = (series == last).sum(axis=1)
    t = series.shape[1]
    rank = (below + 0.5 * equal) / t
    return rank


def _v2_addons(s64: np.ndarray) -> np.ndarray:
    """Compute the 10 v2 cross-channel/temporal addons from [N, 28, 60].

    Returns [N, 10] aligned with _V2_NEW_FEATURES order. None of the
    inputs read from MASKED_CHANNELS (17, 18, 19).
    """
    n = s64.shape[0]
    out = np.zeros((n, len(_V2_NEW_FEATURES)), dtype=np.float64)

    ch0 = s64[:, 0, :]
    ch4 = s64[:, 4, :]
    ch5 = s64[:, 5, :]
    ch20 = s64[:, 20, :]
    ch24 = s64[:, 24, :]
    ch25 = s64[:, 25, :]
    eps = 1e-8

    out[:, 0] = ch24[:, -1] / (ch25[:, -1] + eps)
    out[:, 1] = ch24.std(axis=1)
    out[:, 2] = ch0[:, -1] - ch0[:, 0]

    diffs = np.diff(ch0, axis=1)
    d_mean = diffs.mean(axis=1, keepdims=True)
    d_std = diffs.std(axis=1) + eps
    centered = diffs - d_mean
    m3 = (centered ** 3).mean(axis=1)
    m4 = (centered ** 4).mean(axis=1)
    out[:, 3] = m3 / (d_std ** 3)
    out[:, 4] = m4 / (d_std ** 4) - 3.0

    out[:, 5] = (ch4[:, -1] < 0.3).astype(np.float64)
    out[:, 6] = (ch4[:, -1] > 0.7).astype(np.float64)
    out[:, 7] = ((ch4[:, -3] < 0.3) & (ch4[:, -1] >= 0.3)).astype(np.float64)

    sign_now = np.sign(ch5[:, -1])
    sign_prev = np.sign(ch5[:, -3])
    out[:, 8] = (sign_now != sign_prev).astype(np.float64)

    out[:, 9] = ch20[:, -1] * np.sign(_slope(ch0))
    return out


# ── v3 tier configuration (added 2026-05-16, #311b) ────────────────────────
MESO_CHANNELS: frozenset = frozenset({15, 24, 25, 26})
MACRO_CHANNELS: frozenset = frozenset({20, 21, 27})
TIER_WINDOWS_V3: dict = {"micro": 60, "meso": 168, "macro": 336}
_TIER_WEIGHT_V3: dict = {"micro": 1.0, "meso": 2.0, "macro": 3.0, "masked": 0.0}


def _v3_feature_names() -> List[str]:
    """Build the 350-name list for feature_set='v3'.

    Layout:
      - micro non-masked  : chN_<stat>             (180 names)
      - meso              : chN_m060_<stat> + chN_m168_<stat>  (80 names)
      - macro             : chN_m060_<stat> + chN_m336_<stat>  (60 names)
      - masked            : chN_<stat>             (30 names, value always 0)
    """
    names: List[str] = []
    for c in range(N_CHANNELS):
        if c in MESO_CHANNELS:
            names += [f"ch{c}_m060_{s}" for s in _STAT_NAMES]
            names += [f"ch{c}_m168_{s}" for s in _STAT_NAMES]
        elif c in MACRO_CHANNELS:
            names += [f"ch{c}_m060_{s}" for s in _STAT_NAMES]
            names += [f"ch{c}_m336_{s}" for s in _STAT_NAMES]
        else:
            names += [f"ch{c}_{s}" for s in _STAT_NAMES]
    return names


def feature_weights_v3() -> np.ndarray:
    """Return the 350-long feature_weights vector for xgb.train."""
    names = _v3_feature_names()
    weights = np.zeros(len(names), dtype=np.float64)
    for i, n in enumerate(names):
        if n.startswith(("ch17_", "ch18_", "ch19_")):
            weights[i] = _TIER_WEIGHT_V3["masked"]
        elif "_m336_" in n:
            weights[i] = _TIER_WEIGHT_V3["macro"]
        elif "_m168_" in n:
            weights[i] = _TIER_WEIGHT_V3["meso"]
        elif "_m060_" in n:
            # m060 slot belongs to the channel's primary tier (meso or macro)
            ch_idx = int(n.split("_", 1)[0][2:])
            if ch_idx in MACRO_CHANNELS:
                weights[i] = _TIER_WEIGHT_V3["macro"]
            elif ch_idx in MESO_CHANNELS:
                weights[i] = _TIER_WEIGHT_V3["meso"]
            else:
                weights[i] = _TIER_WEIGHT_V3["micro"]
        else:
            weights[i] = _TIER_WEIGHT_V3["micro"]
    return weights


def _stats_from_candles(candles: list, stat_offset: int, out: np.ndarray) -> None:
    """In-place fill 10 stats starting at out[stat_offset:stat_offset+10] from
    a candle list. Operates on the CLOSE series. If candles is empty, slots
    stay zero (caller pre-zeroed `out`).

    Stats = (last, mean, std, slope, min, max, pct_rank, delta_5, delta_10, delta_30).
    """
    if not candles:
        return
    closes = np.array([c["close"] for c in candles], dtype=np.float64)
    if closes.size == 0:
        return
    out[stat_offset + 0] = closes[-1]
    out[stat_offset + 1] = closes.mean()
    out[stat_offset + 2] = closes.std()
    t = closes.size
    x = np.arange(t, dtype=np.float64)
    x_mean = x.mean()
    y_mean = closes.mean()
    num = ((x - x_mean) * (closes - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    out[stat_offset + 3] = (num / den) if den != 0 else 0.0
    out[stat_offset + 4] = closes.min()
    out[stat_offset + 5] = closes.max()
    last = closes[-1]
    below = (closes < last).sum()
    equal = (closes == last).sum()
    out[stat_offset + 6] = (below + 0.5 * equal) / t
    out[stat_offset + 7] = closes[-1] - closes[-6]  if t >= 6  else 0.0
    out[stat_offset + 8] = closes[-1] - closes[-11] if t >= 11 else 0.0
    out[stat_offset + 9] = closes[-1] - closes[-31] if t >= 31 else 0.0


def _extract_v3(candles_by_tier: dict) -> Tuple[np.ndarray, List[str]]:
    """Convert {"micro","meso","macro"} candle slices to (features [1,350], names [350]).

    This v3 extractor uses CLOSE series only — channel-level signals
    (RSI/MACD/etc.) are NOT recomputed here. The v3 booster is trained with
    whatever the trainer feeds in via this function; replace with real
    per-channel signal feeders before v3 training if you want richer surface.
    The call-site contract (350-element output) stays the same.
    """
    names = _v3_feature_names()
    out = np.zeros((1, len(names)), dtype=np.float64)

    micro = candles_by_tier.get("micro") or []
    meso  = candles_by_tier.get("meso")  or []
    macro = candles_by_tier.get("macro") or []

    offset = 0
    for c in range(N_CHANNELS):
        if c in (17, 18, 19):
            offset += 10
            continue
        if c in MESO_CHANNELS:
            _stats_from_candles(micro, offset, out[0])
            offset += 10
            _stats_from_candles(meso, offset, out[0])
            offset += 10
        elif c in MACRO_CHANNELS:
            _stats_from_candles(micro, offset, out[0])
            offset += 10
            _stats_from_candles(macro, offset, out[0])
            offset += 10
        else:
            _stats_from_candles(micro, offset, out[0])
            offset += 10

    return out, names


def extract_features(
    samples, feature_set: str = "v1"
) -> Tuple[np.ndarray, List[str]]:
    """Convert a batch of [N, 28, 60] samples to tabular features.

    feature_set:
        "v1" (default): 270 per-channel stats (back-compat).
        "v2": v1 + 10 cross-channel/temporal addons (_V2_NEW_FEATURES).
        "v3": tiered mixed-lookback — 350 features, samples arg becomes
              {"micro","meso","macro"} candle-list dict (#311b).
        "v4": OHLCV-5 channels x 3 tiers x 10 stats = 150 features,
              samples arg is {"micro","meso","macro"} candle-list dict
              (#xgb-v4 / Step B.1).

    Returns (features, feature_names) where features is float64 with no
    NaN/Inf, and feature_names matches the column order of features.
    """
    if feature_set == "v4":
        from tools.xgb_v4_features import extract_v4
        return extract_v4(samples)
    if feature_set == "v3":
        return _extract_v3(samples)
    if feature_set not in ("v1", "v2"):
        raise ValueError(
            f"unknown feature_set={feature_set!r}; expected 'v1', 'v2', 'v3', or 'v4'"
        )
    if samples.ndim != 3 or samples.shape[1] != N_CHANNELS or samples.shape[2] != SEQ_LEN:
        raise ValueError(
            f"expected shape [N, {N_CHANNELS}, {SEQ_LEN}], got {samples.shape}"
        )

    n = samples.shape[0]
    out = np.zeros((n, N_CHANNELS * len(_STAT_NAMES)), dtype=np.float64)

    s64 = samples.astype(np.float64)

    for c in range(N_CHANNELS):
        base = c * len(_STAT_NAMES)
        if c in MASKED_CHANNELS or c in XGB_DROP_CHANNELS:
            # MASKED_CHANNELS: signal is zeroed at inference -> would cause skew.
            # XGB_DROP_CHANNELS: ablation (#146) showed these are anti-signal.
            continue
        ch = s64[:, c, :]              # [N, T]
        out[:, base + 0] = ch[:, -1]   # last
        out[:, base + 1] = ch.mean(axis=1)
        out[:, base + 2] = ch.std(axis=1)
        out[:, base + 3] = _slope(ch)
        out[:, base + 4] = ch.min(axis=1)
        out[:, base + 5] = ch.max(axis=1)
        out[:, base + 6] = _percentile_rank(ch)
        out[:, base + 7] = ch[:, -1] - ch[:, -6]   # delta_5
        out[:, base + 8] = ch[:, -1] - ch[:, -11]  # delta_10
        out[:, base + 9] = ch[:, -1] - ch[:, -31]  # delta_30

    names = _build_feature_names()

    if feature_set == "v2":
        addons = _v2_addons(s64)
        addons = np.nan_to_num(addons, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.concatenate([out, addons], axis=1)
        names = names + list(_V2_NEW_FEATURES)

    return out, names
