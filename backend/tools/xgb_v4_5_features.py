"""XGB v4.5 7-channel feature extractor.

5 OHLCV channels (open/high/low/close/volume) + 2 Bollinger channels
(bb_position, bb_width) × 3 tiers (micro/meso/macro) × 10 stats = 210 features.

Per feedback_python_clean_functions: type hints, pure data-in/data-out helpers,
derived constants, no in-place buffer mutation.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Configuration constants
# ═══════════════════════════════════════════════════════════════════════════
_OHLCV_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
_BB_CHANNELS: Tuple[str, ...] = ("bb_position", "bb_width")
_CHANNEL_NAMES: Tuple[str, ...] = _OHLCV_FIELDS + _BB_CHANNELS
N_CHANNELS_V45: int = len(_CHANNEL_NAMES)  # = 7

TIER_WINDOWS_V45: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V45: Dict[str, float] = {"micro": 1.0, "meso": 2.0, "macro": 3.0}
_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")

BB_PERIOD: int = 20
BB_MULT: float = 2.0

_STAT_NAMES_V45: Tuple[str, ...] = (
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
N_STATS_V45: int = len(_STAT_NAMES_V45)  # = 10
N_TIERS_V45: int = len(TIER_WINDOWS_V45)  # = 3
N_FEATURES_V45: int = N_CHANNELS_V45 * N_TIERS_V45 * N_STATS_V45  # = 210


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def feature_names_v4_5() -> List[str]:
    """Return 210 feature names in stable column order.

    Layout: ch{0..6}_{micro|meso|macro}_{stat}, channel-major -> tier-major
    -> stat-major. Channels 0..4 = OHLCV, channel 5 = bb_position, channel 6
    = bb_width.
    """
    names: List[str] = []
    for c in range(N_CHANNELS_V45):
        for tier in _TIER_ORDER:
            for stat in _STAT_NAMES_V45:
                names.append(f"ch{c}_{tier}_{stat}")
    return names


def feature_weights_v4_5() -> np.ndarray:
    """Return 210-long float64 weight vector aligned with feature_names_v4_5().

    Per-tier weights: micro=1.0, meso=2.0, macro=3.0. Same weight for all
    10 stats within one (channel, tier) group.
    """
    weights = np.zeros(N_FEATURES_V45, dtype=np.float64)
    i = 0
    for _c in range(N_CHANNELS_V45):
        for tier in _TIER_ORDER:
            w = TIER_WEIGHTS_V45[tier]
            for _s in range(N_STATS_V45):
                weights[i] = w
                i += 1
    return weights


def extract_v4_5(
    candles_by_tier: Dict[str, Sequence[Dict[str, float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract 210 features from tier-keyed OHLCV candle lists.

    For BB channels (ch5, ch6): each bar in the tier slice gets its
    bb_position and bb_width computed from a trailing BB_PERIOD-bar window
    ending at that bar. Bars with fewer than BB_PERIOD preceding bars get
    (0.5, 0.0) fallback.

    Args:
        candles_by_tier: {"micro": [...], "meso": [...], "macro": [...]} where
            each entry is a candle dict with at minimum the OHLCV keys.

    Returns:
        (features, names) where features is shape (1, 210) float64 and
        names is len-210 list matching feature_names_v4_5().

    Missing/empty tier -> the 70 slots for that tier are zero.
    Missing OHLCV field in a candle -> raises KeyError (input contract).
    """
    out = np.zeros((1, N_FEATURES_V45), dtype=np.float64)
    names = feature_names_v4_5()
    slot = 0
    for c in range(N_CHANNELS_V45):
        field = _CHANNEL_NAMES[c]
        for tier in _TIER_ORDER:
            tier_candles = candles_by_tier.get(tier) or []
            if field in _OHLCV_FIELDS:
                values = _extract_ohlcv_field(tier_candles, field)
            elif field == "bb_position":
                closes = _extract_ohlcv_field(tier_candles, "close")
                values = _compute_bb_position(closes)
            elif field == "bb_width":
                closes = _extract_ohlcv_field(tier_candles, "close")
                values = _compute_bb_width(closes)
            else:  # unreachable; defensive
                values = np.array([], dtype=np.float64)
            stats = _compute_stats(values)
            out[0, slot : slot + N_STATS_V45] = stats
            slot += N_STATS_V45
    return out, names


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers (pure functions, one responsibility each)
# ═══════════════════════════════════════════════════════════════════════════


def _extract_ohlcv_field(
    candles: Sequence[Dict[str, float]],
    field: str,
) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray.

    Empty input -> empty ndarray. Missing field key -> KeyError (input contract).
    """
    if not candles:
        return np.array([], dtype=np.float64)
    return np.array([candle[field] for candle in candles], dtype=np.float64)


def _compute_bb_position(
    closes: np.ndarray,
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> np.ndarray:
    """Bollinger position in [0, 1] at each bar.

    For bar i: pos = (close[i] - lower) / (upper - lower) where upper/lower
    are computed from the trailing `period` bars ending at bar i. Bars with
    fewer than `period` prior bars get 0.5 (mid) fallback. Bars where the
    band has zero spread (std == 0) also get 0.5.
    """
    n = closes.size
    out = np.full(n, 0.5, dtype=np.float64)
    if n < period:
        return out
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = window.mean()
        std = window.std()
        if std == 0.0:
            out[i] = 0.5
            continue
        upper = mean + mult * std
        lower = mean - mult * std
        bw = upper - lower
        if bw <= 0.0:
            out[i] = 0.5
            continue
        pos = (closes[i] - lower) / bw
        out[i] = max(0.0, min(1.0, pos))
    return out


def _compute_bb_width(
    closes: np.ndarray,
    period: int = BB_PERIOD,
    mult: float = BB_MULT,
) -> np.ndarray:
    """(upper - lower) / mean at each bar.

    Pre-period bars (fewer than `period` prior bars) get 0.0 fallback. Bars
    where mean == 0 also get 0.0.
    """
    n = closes.size
    out = np.zeros(n, dtype=np.float64)
    if n < period:
        return out
    for i in range(period - 1, n):
        window = closes[i - period + 1 : i + 1]
        mean = window.mean()
        std = window.std()
        if mean == 0.0:
            out[i] = 0.0
            continue
        out[i] = (2.0 * mult * std) / mean
    return out


def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Return shape-(10,) stats in fixed _STAT_NAMES_V45 order.

    Empty input -> all zeros. No in-place mutation of any caller buffer.
    """
    out = np.zeros(N_STATS_V45, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[-1])  # last
    out[1] = float(values.mean())  # mean
    out[2] = float(values.std())  # std
    out[3] = _slope(values)  # slope
    out[4] = float(values.min())  # min
    out[5] = float(values.max())  # max
    out[6] = _pct_rank(values)  # pct_rank
    out[7] = _delta_at(values, lookback=5)
    out[8] = _delta_at(values, lookback=10)
    out[9] = _delta_at(values, lookback=30)
    return out


def _slope(values: np.ndarray) -> float:
    """OLS slope of values vs index 0..len-1. 0.0 for n<2 or zero variance."""
    n = values.size
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    x_mean = x.mean()
    y_mean = values.mean()
    num = ((x - x_mean) * (values - y_mean)).sum()
    den = ((x - x_mean) ** 2).sum()
    if den == 0.0:
        return 0.0
    return float(num / den)


def _pct_rank(values: np.ndarray) -> float:
    """Percentile rank of last value within the series. 0.0 if empty/single."""
    n = values.size
    if n < 2:
        return 0.0
    last = values[-1]
    below = (values < last).sum()
    equal = (values == last).sum()
    return float((below + 0.5 * equal) / n)


def _delta_at(values: np.ndarray, lookback: int) -> float:
    """values[-1] - values[-1-lookback], or 0.0 if series too short."""
    if values.size < lookback + 1:
        return 0.0
    return float(values[-1] - values[-1 - lookback])
