"""XGB v4 OHLCV-5 feature extractor.

5 channels (open/high/low/close/volume) x 3 tiers (micro/meso/macro)
x 10 stats = 150 features. Pure functions, no mutable module state.

Per feedback_python_clean_functions: type hints on every signature,
pure data-in/data-out helpers, derived constants, no in-place buffer
mutation (contrast v3's _stats_from_candles(candles, stat_offset, out)).
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ── Configuration constants ────────────────────────────────────────────────
_CHANNEL_FIELDS: Tuple[str, ...] = ("open", "high", "low", "close", "volume")
N_CHANNELS_V4: int = len(_CHANNEL_FIELDS)

TIER_WINDOWS_V4: Dict[str, int] = {"micro": 60, "meso": 168, "macro": 336}
TIER_WEIGHTS_V4: Dict[str, float] = {"micro": 1.0, "meso": 2.0, "macro": 3.0}
_TIER_ORDER: Tuple[str, ...] = ("micro", "meso", "macro")

_STAT_NAMES_V4: Tuple[str, ...] = (
    "last", "mean", "std", "slope",
    "min", "max", "pct_rank",
    "dlt5", "dlt10", "dlt30",
)
N_STATS_V4: int = len(_STAT_NAMES_V4)
N_TIERS_V4: int = len(TIER_WINDOWS_V4)
N_FEATURES_V4: int = N_CHANNELS_V4 * N_TIERS_V4 * N_STATS_V4  # = 150


# ── Public API ─────────────────────────────────────────────────────────────

def feature_names_v4() -> List[str]:
    """Return 150 feature names in stable column order.

    Layout: ch{0..4}_{micro|meso|macro}_{stat}, ordered
    channel-major -> tier-major -> stat-major.
    """
    names: List[str] = []
    for c in range(N_CHANNELS_V4):
        for tier in _TIER_ORDER:
            for stat in _STAT_NAMES_V4:
                names.append(f"ch{c}_{tier}_{stat}")
    return names


def feature_weights_v4() -> np.ndarray:
    """Return 150-long float64 weight vector aligned with feature_names_v4().

    Per-tier weights: micro 1.0, meso 2.0, macro 3.0. Same weight for all
    10 stats within one (channel, tier) group.
    """
    weights = np.zeros(N_FEATURES_V4, dtype=np.float64)
    i = 0
    for _c in range(N_CHANNELS_V4):
        for tier in _TIER_ORDER:
            w = TIER_WEIGHTS_V4[tier]
            for _s in range(N_STATS_V4):
                weights[i] = w
                i += 1
    return weights


def extract_v4(
    candles_by_tier: Dict[str, Sequence[Dict[str, float]]],
) -> Tuple[np.ndarray, List[str]]:
    """Extract 150 features from tier-keyed OHLCV candle lists.

    Args:
        candles_by_tier: {"micro": [...], "meso": [...], "macro": [...]}
            where each entry is a candle dict with at minimum the keys
            ("open","high","low","close","volume").

    Returns:
        (features, names) where features is shape (1, 150) float64 and
        names is len-150 list matching feature_names_v4().

    Missing/empty tier -> the 50 slots for that tier are zero.
    Missing OHLCV field in a candle -> raises KeyError (input contract).
    """
    out = np.zeros((1, N_FEATURES_V4), dtype=np.float64)
    names = feature_names_v4()
    slot = 0
    for c in range(N_CHANNELS_V4):
        field = _CHANNEL_FIELDS[c]
        for tier in _TIER_ORDER:
            tier_candles = candles_by_tier.get(tier) or []
            values = _extract_field(tier_candles, field)
            stats = _compute_stats(values)
            out[0, slot:slot + N_STATS_V4] = stats
            slot += N_STATS_V4
    return out, names


# ── Internal helpers (pure functions, one responsibility each) ─────────────

def _extract_field(
    candles: Sequence[Dict[str, float]],
    field: str,
) -> np.ndarray:
    """Extract one OHLCV column as float64 ndarray.

    Empty input -> empty ndarray (caller assembles into the zero slot).
    Missing field key in any candle raises KeyError (input contract).
    """
    if not candles:
        return np.array([], dtype=np.float64)
    return np.array([candle[field] for candle in candles], dtype=np.float64)


def _compute_stats(values: np.ndarray) -> np.ndarray:
    """Return shape-(10,) stats in fixed _STAT_NAMES_V4 order.

    Empty input -> all zeros. No in-place mutation of any caller buffer.
    """
    out = np.zeros(N_STATS_V4, dtype=np.float64)
    if values.size == 0:
        return out
    out[0] = float(values[-1])             # last
    out[1] = float(values.mean())          # mean
    out[2] = float(values.std())           # std
    out[3] = _slope(values)                # slope
    out[4] = float(values.min())           # min
    out[5] = float(values.max())           # max
    out[6] = _pct_rank(values)             # pct_rank
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
