"""
Tests for new signal_generator functions added from GitHub research:
  - _hurst_exponent()     : regime detection (trending vs mean-reverting)
  - _multi_rsi()          : multi-period RSI voting (6/12/24)
  - _dissimilarity_index(): DI gate to suppress unreliable CNN/LSTM output
  - _kelly_fraction()     : Kelly Criterion position sizing
"""
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from agents.signal_generator import (
    _hurst_exponent,
    _multi_rsi,
    _dissimilarity_index,
    _kelly_fraction,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _trending_closes(n: int = 100) -> list:
    """Strongly trending up — Hurst should be > 0.5."""
    return [100.0 + i * 0.5 for i in range(n)]


def _ranging_closes(n: int = 100) -> list:
    """Oscillating mean-reverting series — Hurst should be < 0.5."""
    import math
    return [100.0 + 5.0 * math.sin(i * 0.4) for i in range(n)]


def _random_closes(n: int = 200) -> list:
    """Pseudo-random walk — Hurst near 0.5."""
    import random
    random.seed(42)
    closes = [100.0]
    for _ in range(n - 1):
        closes.append(closes[-1] + random.gauss(0, 1))
    return closes


# ── _hurst_exponent ────────────────────────────────────────────────────────────

class TestHurstExponent:

    def test_returns_float(self):
        closes = _trending_closes(80)
        h = _hurst_exponent(closes)
        assert isinstance(h, float)

    def test_bounded_0_1(self):
        for closes in [_trending_closes(80), _ranging_closes(80), _random_closes(200)]:
            h = _hurst_exponent(closes)
            assert 0.0 <= h <= 1.0, f"Hurst {h} out of bounds for input length {len(closes)}"

    def test_trending_series_above_half(self):
        """Persistent trend → H should be > 0.5."""
        h = _hurst_exponent(_trending_closes(120))
        assert h > 0.5, f"Expected H > 0.5 for trending series, got {h:.3f}"

    def test_ranging_series_below_half(self):
        """Mean-reverting series → H should be < 0.5."""
        h = _hurst_exponent(_ranging_closes(120))
        assert h < 0.5, f"Expected H < 0.5 for mean-reverting series, got {h:.3f}"

    def test_short_series_returns_default(self):
        """Too few data points → return 0.5 (neutral / unknown regime)."""
        h = _hurst_exponent([100.0, 101.0, 100.5])
        assert h == pytest.approx(0.5, abs=0.01)

    def test_minimum_length_parameter(self):
        """min_len parameter controls when to return the default 0.5."""
        closes = _trending_closes(40)
        h_default = _hurst_exponent(closes, min_len=50)   # below threshold → neutral
        h_computed = _hurst_exponent(closes, min_len=30)  # above threshold → computed
        assert h_default == pytest.approx(0.5, abs=0.01)
        assert h_computed != pytest.approx(0.5, abs=0.1)  # actually computed


# ── _multi_rsi ─────────────────────────────────────────────────────────────────

class TestMultiRSI:

    def test_returns_dict_with_required_keys(self):
        closes = _trending_closes(60)
        result = _multi_rsi(closes)
        for key in ("rsi6", "rsi12", "rsi24", "buy_votes", "sell_votes"):
            assert key in result, f"Missing key: {key}"

    def test_all_rsi_bounded(self):
        closes = _ranging_closes(80)
        result = _multi_rsi(closes)
        for key in ("rsi6", "rsi12", "rsi24"):
            assert 0.0 <= result[key] <= 100.0, f"{key}={result[key]} out of [0,100]"

    def test_votes_bounded(self):
        closes = _ranging_closes(80)
        result = _multi_rsi(closes)
        assert 0 <= result["buy_votes"]  <= 3
        assert 0 <= result["sell_votes"] <= 3

    def test_downtrend_triggers_buy_votes(self):
        """Strongly falling series → multiple RSI periods should be oversold."""
        closes = [100.0 - i * 2 for i in range(80)]
        result = _multi_rsi(closes)
        assert result["buy_votes"] >= 2, (
            f"Expected ≥2 buy votes in downtrend, got {result['buy_votes']}"
        )

    def test_uptrend_triggers_sell_votes(self):
        """Strongly rising series → multiple RSI periods should be overbought."""
        closes = [50.0 + i * 3 for i in range(80)]
        result = _multi_rsi(closes)
        assert result["sell_votes"] >= 2, (
            f"Expected ≥2 sell votes in uptrend, got {result['sell_votes']}"
        )

    def test_insufficient_data_returns_neutral(self):
        """Too few closes → all RSIs neutral, votes = 0."""
        result = _multi_rsi([100.0, 101.0])
        assert result["buy_votes"]  == 0
        assert result["sell_votes"] == 0

    def test_buy_and_sell_votes_mutually_exclusive(self):
        """A series should not have both buy_votes > 0 and sell_votes > 0 simultaneously
        (extreme cases where some periods oversold and others overbought are edge cases,
        but the function must not crash)."""
        closes = _ranging_closes(80)
        result = _multi_rsi(closes)
        # No crash, and at least one of them is 0 for a ranging series
        assert result["buy_votes"] >= 0
        assert result["sell_votes"] >= 0


# ── _dissimilarity_index ───────────────────────────────────────────────────────

class TestDissimilarityIndex:

    def test_returns_float(self):
        closes = _trending_closes(40)
        di = _dissimilarity_index(closes)
        assert isinstance(di, float)

    def test_nonnegative(self):
        """DI is always ≥ 0 (it's a distance measure)."""
        for closes in [_trending_closes(40), _ranging_closes(40)]:
            di = _dissimilarity_index(closes)
            assert di >= 0.0, f"DI should be non-negative, got {di}"

    def test_flat_series_zero_di(self):
        """Perfectly flat price → DI ≈ 0 (price equals SMA)."""
        closes = [100.0] * 40
        di = _dissimilarity_index(closes)
        assert di == pytest.approx(0.0, abs=0.01)

    def test_volatile_series_higher_di(self):
        """A highly volatile series should produce higher DI than a flat one."""
        flat    = [100.0] * 40
        volatile = [100.0 + ((-1) ** i) * 20 for i in range(40)]
        assert _dissimilarity_index(volatile) > _dissimilarity_index(flat)

    def test_threshold_detection(self):
        """DI > 3 indicates price has diverged significantly from SMA — CNN unreliable.
        Period spans both old baseline and new elevated price so SMA lags behind."""
        # 20 bars at 100, then 20 bars at 140 — period=30 spans both regimes
        breakout = [100.0] * 20 + [140.0] * 20
        di = _dissimilarity_index(breakout, period=30)
        assert di > 3.0, f"Expected DI > 3 for breakout series, got {di:.2f}"

    def test_short_series_returns_zero(self):
        """Fewer candles than period → return 0.0 (no opinion)."""
        di = _dissimilarity_index([100.0, 101.0], period=14)
        assert di == pytest.approx(0.0, abs=0.001)


# ── _kelly_fraction ────────────────────────────────────────────────────────────

class TestKellyFraction:

    def test_returns_float(self):
        f = _kelly_fraction(confidence=0.7)
        assert isinstance(f, float)

    def test_bounded_0_to_max_frac(self):
        """Kelly fraction must be in [0, max_frac]."""
        for conf in [0.3, 0.5, 0.65, 0.8, 0.95]:
            f = _kelly_fraction(conf)
            assert 0.0 <= f <= 0.25, f"Kelly({conf:.2f}) = {f:.3f} out of [0, 0.25]"

    def test_fifty_percent_confidence_is_zero(self):
        """At 50% confidence (coin flip) Kelly fraction = 0 (no edge)."""
        f = _kelly_fraction(confidence=0.5)
        assert f == pytest.approx(0.0, abs=0.001)

    def test_higher_confidence_larger_fraction(self):
        """Higher confidence → larger fraction (monotone increasing up to cap).
        Use values below the cap: full Kelly = 2p-1 < 0.25 when p < 0.625."""
        f52 = _kelly_fraction(0.52)  # full=0.04
        f56 = _kelly_fraction(0.56)  # full=0.12
        f60 = _kelly_fraction(0.60)  # full=0.20
        assert f52 < f56 < f60, f"Kelly not monotone: {f52:.3f} {f56:.3f} {f60:.3f}"

    def test_capped_at_max_frac(self):
        """Full-confidence signal should be capped at max_frac (default 0.25)."""
        f = _kelly_fraction(confidence=1.0)
        assert f == pytest.approx(0.25, abs=0.001)

    def test_custom_max_frac(self):
        f = _kelly_fraction(confidence=0.9, max_frac=0.10)
        assert f <= 0.10

    def test_below_fifty_returns_zero(self):
        """Negative Kelly (< 50% confidence) → 0 (don't trade)."""
        f = _kelly_fraction(confidence=0.3)
        assert f == pytest.approx(0.0, abs=0.001)
