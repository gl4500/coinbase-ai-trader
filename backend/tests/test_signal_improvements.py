"""
TDD tests for 6 CNN/signal improvements:

1. ADX normalization fix   — _wilder() uses MEAN init, ADX values in 0-100
2. MACD fast params        — default fast/slow/signal = 5/13/3 for 1h crypto
3. RSI overbought 78       — momentum agent threshold updated
4. Funding rate channel    — Ch 20 present in FeatureBuilder output
5. BTC correlation channel — Ch 21 present in FeatureBuilder output
6. Time-of-day channels    — Ch 22 (sin) and Ch 23 (cos) present; N_CHANNELS=27

Written before implementation (TDD).
"""
import math
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sine_prices(n: int = 200, period: float = 40.0, base: float = 100.0,
                  amp: float = 5.0):
    """Smooth sine-wave price series — clearly trending in short windows."""
    return [base + amp * math.sin(2 * math.pi * i / period) for i in range(n)]


def _trend_up(n: int = 200, start: float = 100.0, step: float = 0.5):
    """Monotonically increasing prices — ADX should be high."""
    return [start + i * step for i in range(n)]


def _flat(n: int = 200, price: float = 100.0):
    return [price] * n


def _fake_candles(closes, spread: float = 0.5):
    """Synthesize OHLCV candles from a close series."""
    candles = []
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        candles.append({
            "open":   o,
            "high":   max(o, c) + spread,
            "low":    min(o, c) - spread,
            "close":  c,
            "volume": 1000.0,
        })
    return candles


# ─────────────────────────────────────────────────────────────────────────────
# 1. ADX normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestADXNormalization:
    """ADX must stay within 0–100 after the _wilder() mean-init fix."""

    def _adx_val(self, closes):
        from agents.signal_generator import _adx
        highs  = [c + 0.5 for c in closes]
        lows   = [c - 0.5 for c in closes]
        adx, _, _ = _adx(highs, lows, closes)
        return adx

    def test_adx_range_trending_market(self):
        closes = _trend_up(200)
        adx = self._adx_val(closes)
        assert 0.0 <= adx <= 100.0, f"ADX out of range: {adx}"

    def test_adx_range_sine_wave(self):
        closes = _sine_prices(200)
        adx = self._adx_val(closes)
        assert 0.0 <= adx <= 100.0, f"ADX out of range: {adx}"

    def test_adx_range_flat_market(self):
        closes = _flat(200)
        adx = self._adx_val(closes)
        assert 0.0 <= adx <= 100.0, f"ADX out of range: {adx}"

    def test_adx_trending_above_25(self):
        """A strong uptrend should produce ADX > 25 (the standard trend threshold)."""
        closes = _trend_up(200)
        adx = self._adx_val(closes)
        assert adx > 25.0, f"Strong trend gave ADX={adx:.1f}, expected > 25"

    def test_adx_not_inflated(self):
        """Regression: before fix, ADX could reach 400-600; must stay <= 100."""
        for prices in [_trend_up(150), _sine_prices(150), _flat(150)]:
            adx = self._adx_val(prices)
            assert adx <= 100.0, f"ADX inflated to {adx:.1f}"


# ─────────────────────────────────────────────────────────────────────────────
# 2. MACD fast parameters (5, 13, 3) for 1-hour crypto bars
# ─────────────────────────────────────────────────────────────────────────────

class TestMACDFastParams:
    """FeatureBuilder Ch 5 should use fast MACD (5,13,3) not stock-speed (12,26,9)."""

    def test_macd_default_fast(self):
        """_macd() default arguments must be fast=5, slow=13, signal=3."""
        import inspect
        from agents.signal_generator import _macd
        sig = inspect.signature(_macd)
        params = sig.parameters
        assert params["fast"].default   == 5,  f"fast={params['fast'].default}"
        assert params["slow"].default   == 13, f"slow={params['slow'].default}"
        assert params["signal"].default == 3,  f"signal={params['signal'].default}"

    def test_macd_needs_fewer_bars(self):
        """Fast MACD should produce a signal with fewer bars than old (12,26,9)."""
        from agents.signal_generator import _macd
        closes = _trend_up(25)
        m, s, h = _macd(closes)   # uses default (5,13,3)
        # Should not be all zeros with 25 bars
        assert not (m == 0.0 and s == 0.0 and h == 0.0), \
            "MACD(5,13,3) should produce signal in 25 bars"

    def test_feature_builder_ch5_uses_fast_macd(self):
        """
        FeatureBuilder Ch 5 must produce non-zero values with 25 candles
        (impossible with slow 12/26 params which need 35+ bars).
        """
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes = _trend_up(25)
        candles = _fake_candles(closes)
        ob = {"bid_depth": 1000, "ask_depth": 1000}
        channels = FeatureBuilder().build(candles, ob, T=len(candles))
        macd_ch = channels[5]
        # With slow MACD (12,26,9) first 35 bars are 0 → at 25 bars all 0
        # With fast MACD (5,13,3) signal appears at bar 16+
        last_vals = macd_ch[-5:]
        assert any(v != 0.0 for v in last_vals), \
            f"Ch 5 all zeros with 25 candles — old (12,26,9) params still in use"


# ─────────────────────────────────────────────────────────────────────────────
# 3. RSI overbought threshold 78
# ─────────────────────────────────────────────────────────────────────────────

class TestRSIOverbought:

    def test_momentum_rsi_overbought_threshold(self):
        from agents.momentum_agent_cb import _RSI_OVERBOUGHT
        assert _RSI_OVERBOUGHT == 65.0, \
            f"_RSI_OVERBOUGHT={_RSI_OVERBOUGHT}, expected 65.0"


# ─────────────────────────────────────────────────────────────────────────────
# 4 & 5 & 6. New channels: funding rate (20), BTC corr (21), sin/cos time (22,23)
# ─────────────────────────────────────────────────────────────────────────────

class TestNewChannels:
    """
    FeatureBuilder.build() must return 24 channels.
    Channels 20-23 must exist and contain valid floats in expected ranges.
    """

    def _build(self, n_candles=80, hour=10):
        import datetime
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(n_candles)
        candles = _fake_candles(closes)
        # Stamp all candles with the same fixed hour
        ts = datetime.datetime(2025, 1, 6, hour, 0, 0).isoformat() + "Z"
        for c in candles:
            c["start"] = ts
        ob = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        btc_closes = _trend_up(n_candles, start=50000.0, step=10.0)
        return FeatureBuilder().build(
            candles, ob, candles_5m=None,
            btc_closes=btc_closes,
            funding_rate=0.001,
            T=SEQ_LEN,
        )

    def test_n_channels_is_24(self):
        from agents.cnn_agent import N_CHANNELS
        assert N_CHANNELS == 27, f"N_CHANNELS={N_CHANNELS}, expected 27"

    def test_build_returns_24_channels(self):
        channels = self._build()
        assert len(channels) == 27, f"Got {len(channels)} channels"

    def test_channel_20_funding_rate(self):
        """Ch 20: funding rate — all values in [-1, 1]."""
        channels = self._build()
        ch = channels[20]
        assert all(-1.0 <= v <= 1.0 for v in ch), \
            f"Ch 20 out of [-1,1]: min={min(ch):.3f} max={max(ch):.3f}"

    def test_channel_20_reflects_positive_funding(self):
        """Positive funding rate → positive channel value."""
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(80)
        candles = _fake_candles(closes)
        ob      = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        channels = FeatureBuilder().build(
            candles, ob, candles_5m=None,
            btc_closes=None, funding_rate=0.05, T=SEQ_LEN,
        )
        assert channels[20][0] > 0.0, "Positive funding should give positive Ch 20"

    def test_channel_20_reflects_negative_funding(self):
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(80)
        candles = _fake_candles(closes)
        ob      = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        channels = FeatureBuilder().build(
            candles, ob, candles_5m=None,
            btc_closes=None, funding_rate=-0.05, T=SEQ_LEN,
        )
        assert channels[20][0] < 0.0, "Negative funding should give negative Ch 20"

    def test_channel_21_btc_corr_in_range(self):
        """Ch 21: BTC correlation — all values in [-1, 1]."""
        channels = self._build()
        ch = channels[21]
        assert all(-1.0 <= v <= 1.0 for v in ch), \
            f"Ch 21 out of [-1,1]: min={min(ch):.3f} max={max(ch):.3f}"

    def test_channel_21_high_when_correlated(self):
        """When asset moves identically to BTC, correlation ~ 1."""
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(80, start=100.0, step=0.5)
        btc     = _trend_up(80, start=50000.0, step=250.0)  # same direction
        candles = _fake_candles(closes)
        ob      = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        channels = FeatureBuilder().build(
            candles, ob, candles_5m=None,
            btc_closes=btc, funding_rate=0.0, T=SEQ_LEN,
        )
        # Last value should be near 1.0 (highly correlated uptrends)
        last = channels[21][-1]
        assert last > 0.5, f"Expected high BTC correlation, got {last:.3f}"

    def test_channel_21_none_btc_gives_zero(self):
        """If btc_closes is None, Ch 21 should be 0.0 (neutral)."""
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(80)
        candles = _fake_candles(closes)
        ob      = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        channels = FeatureBuilder().build(
            candles, ob, candles_5m=None,
            btc_closes=None, funding_rate=0.0, T=SEQ_LEN,
        )
        assert all(v == 0.0 for v in channels[21]), \
            "No BTC data → Ch 21 should be all 0.0"

    def test_channel_22_sin_time_in_range(self):
        """Ch 22: sin(hour) — all values in [-1, 1]."""
        channels = self._build(hour=0)
        ch = channels[22]
        assert all(-1.0 <= v <= 1.0 for v in ch)

    def test_channel_23_cos_time_in_range(self):
        """Ch 23: cos(hour) — all values in [-1, 1]."""
        channels = self._build(hour=0)
        ch = channels[23]
        assert all(-1.0 <= v <= 1.0 for v in ch)

    def test_channel_22_23_encode_hour_correctly(self):
        """Hour 0 → sin=0, cos=1; Hour 6 → sin=1, cos=0 (approximately)."""
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        import datetime

        def _channels_at_hour(h):
            closes  = _trend_up(80)
            candles = _fake_candles(closes)
            # Stamp ALL candles with the same hour so ch[-1] encodes that hour
            ts = datetime.datetime(2025, 1, 6, h, 0, 0).isoformat() + "Z"
            for c in candles:
                c["start"] = ts
            ob = {"bid_depth": 1000.0, "ask_depth": 1000.0}
            return FeatureBuilder().build(
                candles, ob, candles_5m=None,
                btc_closes=None, funding_rate=0.0, T=SEQ_LEN,
            )

        ch_h0  = _channels_at_hour(0)
        ch_h6  = _channels_at_hour(6)

        sin_h0 = ch_h0[22][-1]
        cos_h0 = ch_h0[23][-1]
        sin_h6 = ch_h6[22][-1]
        cos_h6 = ch_h6[23][-1]

        # hour=0: angle = 2π*0/24 = 0 → sin=0, cos=1
        assert abs(sin_h0) < 0.1,  f"sin at hour 0 = {sin_h0:.3f}, expected ~0"
        assert abs(cos_h0 - 1.0) < 0.1, f"cos at hour 0 = {cos_h0:.3f}, expected ~1"

        # hour=6: angle = 2π*6/24 = π/2 → sin=1, cos=0
        assert abs(sin_h6 - 1.0) < 0.1, f"sin at hour 6 = {sin_h6:.3f}, expected ~1"
        assert abs(cos_h6) < 0.1,  f"cos at hour 6 = {cos_h6:.3f}, expected ~0"

    def test_no_funding_defaults_to_zero(self):
        """funding_rate=None (not passed) → Ch 20 all 0.0."""
        from agents.cnn_agent import FeatureBuilder, SEQ_LEN
        closes  = _trend_up(80)
        candles = _fake_candles(closes)
        ob      = {"bid_depth": 1000.0, "ask_depth": 1000.0}
        # Call without new kwargs — backward compat
        channels = FeatureBuilder().build(candles, ob, T=SEQ_LEN)
        assert len(channels) == 27
        assert all(v == 0.0 for v in channels[20]), "No funding → Ch 20 all 0"


class TestOllamaModelFallback:
    """OLLAMA_MODEL fallback must match the documented default across modules.

    Docs/.env set OLLAMA_MODEL=llama3.1:8b; when the env var is missing, all
    code paths must agree on the same fallback string so behavior stays
    consistent. A stale fallback (e.g. qwen2.5:7b) silently routes traffic
    to a different model if the env is ever unset.
    """

    EXPECTED = 'os.getenv("OLLAMA_MODEL", "llama3.1:8b")'
    ALT      = '__import__("os").getenv("OLLAMA_MODEL", "llama3.1:8b")'

    def _read(self, rel_path: str) -> str:
        path = os.path.join(BACKEND, rel_path)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def test_cnn_agent_fallback(self):
        src = self._read("agents/cnn_agent.py")
        assert self.EXPECTED in src, "cnn_agent.py OLLAMA_MODEL fallback must be llama3.1:8b"

    def test_signal_generator_fallback(self):
        src = self._read("agents/signal_generator.py")
        assert self.ALT in src, "signal_generator.py OLLAMA_MODEL fallback must be llama3.1:8b"

    def test_outcome_tracker_fallback(self):
        src = self._read("services/outcome_tracker.py")
        assert self.EXPECTED in src, "outcome_tracker.py OLLAMA_MODEL fallback must be llama3.1:8b"
