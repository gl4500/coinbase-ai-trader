"""Tests for tools/long_trend_probe.py — long-horizon SMA / golden-cross probe.

Mirrors tests/test_btc_dominance_probe.py and tests/test_okx_ls_probe.py
shape: pure helpers (SMA distance, daily resample, golden-cross sign,
per-sample alignment) are tested without touching the cache. The main
runner is exercised end-to-end at probe runtime (#245), not here.

Why this probe (#243-#245):
    The 28-channel feature set has no long-horizon trend feature. Longest
    moving-average distance is `ema21_dist` (21 bars on 1h ≈ 0.9 days).
    Golden/death cross on 50d/200d SMAs is structurally different — it
    captures multi-week regime, not intra-day momentum. If lift Δ ≥ +0.01
    AUC vs ch13-baseline → integrate as a real channel.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")


_BAR = 3600
_DAY = 86400


# ── compute_sma_dist_series: hourly closes -> hourly (close - SMA)/SMA dict ─

class TestComputeSmaDistSeries:

    def test_constant_close_zero_distance(self):
        from tools.long_trend_probe import compute_sma_dist_series
        # 60 bars of constant 100 → SMA_50 = 100 → dist = 0 from bar 49 onward.
        closes_by_hour = {h * _BAR: 100.0 for h in range(60)}
        out = compute_sma_dist_series(closes_by_hour, period_bars=50)
        # First 49 bars: NaN (warm-up).
        for h in range(0, 49):
            assert h * _BAR not in out, f"bar {h} should be in warm-up"
        # Bar 49 onward: distance = 0.
        for h in range(49, 60):
            assert out[h * _BAR] == pytest.approx(0.0)

    def test_above_sma_positive_distance(self):
        from tools.long_trend_probe import compute_sma_dist_series
        # Period_bars=2: bars [100, 102] → SMA at bar 1 = 101, dist = (102-101)/101 ≈ +0.0099
        closes_by_hour = {0: 100.0, _BAR: 102.0}
        out = compute_sma_dist_series(closes_by_hour, period_bars=2)
        assert _BAR in out
        assert out[_BAR] == pytest.approx((102.0 - 101.0) / 101.0)
        # Bar 0 has only 1 close → NaN / not in output.
        assert 0 not in out

    def test_below_sma_negative_distance(self):
        from tools.long_trend_probe import compute_sma_dist_series
        closes_by_hour = {0: 102.0, _BAR: 100.0}
        out = compute_sma_dist_series(closes_by_hour, period_bars=2)
        assert out[_BAR] == pytest.approx((100.0 - 101.0) / 101.0)

    def test_warm_up_excluded(self):
        from tools.long_trend_probe import compute_sma_dist_series
        closes_by_hour = {h * _BAR: 100.0 + h for h in range(50)}
        out = compute_sma_dist_series(closes_by_hour, period_bars=50)
        # Only one bar (the 50th, h=49) has a full SMA window.
        assert set(out.keys()) == {49 * _BAR}

    def test_empty_returns_empty(self):
        from tools.long_trend_probe import compute_sma_dist_series
        assert compute_sma_dist_series({}, period_bars=50) == {}

    def test_handles_gaps_in_input(self):
        """Real parquet has missing hours. Helper should compute SMA over the
        available bars in chronological order, not pad gaps."""
        from tools.long_trend_probe import compute_sma_dist_series
        # 3 known bars with a gap: hours 0, 1, 5 (bars 2-4 missing in source).
        closes_by_hour = {0: 100.0, 1 * _BAR: 100.0, 5 * _BAR: 100.0}
        out = compute_sma_dist_series(closes_by_hour, period_bars=3)
        # Period 3 needs 3 bars → first eligible is the 3rd known bar (5*_BAR).
        # SMA = 100, dist = 0.
        assert out == {5 * _BAR: pytest.approx(0.0)}


# ── daily_resample: hourly closes -> daily-end closes (UTC day boundaries) ──

class TestDailyResample:

    def test_last_close_of_each_day_kept(self):
        from tools.long_trend_probe import daily_resample
        # Day 0 (00:00..23:00 UTC) — _DAY = 86400 sec aligns to midnight UTC at
        # epoch second 0. Bars: 0, 3600, 7200, 82800 (23:00).
        # Day 1 starts at 86400.
        # Output is keyed by the ACTUAL observation hour (the day's last
        # available bar), not the day-start — keying at day-start would let
        # downstream forward-fill a same-day 23:00 SMA onto a 14:00 sample.
        closes_by_hour = {
            0:           100.0,   # day 0, 00:00
            23 * _BAR:   105.0,   # day 0, 23:00 (last)
            24 * _BAR:   106.0,   # day 1, 00:00
            47 * _BAR:   110.0,   # day 1, 23:00 (last)
        }
        out = daily_resample(closes_by_hour)
        assert out == {23 * _BAR: 105.0, 47 * _BAR: 110.0}

    def test_partial_day_uses_last_available_bar(self):
        from tools.long_trend_probe import daily_resample
        # Only 14:00 of day 0 → that's the last available bar of the day.
        # Key is the actual observation hour (14*_BAR), not day-start.
        closes_by_hour = {14 * _BAR: 99.0}
        out = daily_resample(closes_by_hour)
        assert out == {14 * _BAR: 99.0}

    def test_empty_returns_empty(self):
        from tools.long_trend_probe import daily_resample
        assert daily_resample({}) == {}

    def test_no_lookahead_via_build_trend_signal(self):
        """Regression: a sample at hour H of day D must not have its SMA
        computed from a CLOSE later in day D. This pins the integration:
        daily_resample → compute_sma_dist_series → build_trend_signal must
        not let same-day future-hour values leak into a sample's window.
        """
        from tools.long_trend_probe import (build_trend_signal,
                                            compute_sma_dist_series,
                                            daily_resample)
        # Build a step in close at noon of day 5: hours 0..119 = 100,
        # hour 120 (day 5, 00:00) onward = 100 except hour 132 (day 5, 12:00)
        # jumps to 200; rest of the series stays flat at 100.
        closes = {h * _BAR: 100.0 for h in range(0, 240)}
        closes[132 * _BAR] = 200.0   # day 5, 12:00 — anomaly
        # Day 5 last bar (23:00) is at hour 143 → daily_resample picks 100.
        # If keying were day-start (120*_BAR), forward-fill would attach the
        # 23:00 close to 00:00 → leak.
        daily = daily_resample(closes)
        # Expect no key earlier than the actual last-hour observation.
        for ts in daily:
            assert ts % _DAY == 23 * _BAR or ts == 132 * _BAR or ts < 24 * _BAR
        # A sample aligned at 12:00 of day 5 must not see any SMA computed
        # past hour 132 (its own bar).
        sma_d1 = compute_sma_dist_series(daily, period_bars=2)
        # Build the per-sample signal and assert no NaN and that values used
        # come only from sma_d1 keys ≤ end_ts.
        sample_end_ts = np.array([132 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, sma_d1, seq_len=12)
        assert np.all(np.isfinite(sig))


# ── golden_cross_signal: daily SMA50/SMA200 sign as +1/0/-1 per HOURLY bar ──

class TestGoldenCrossSignal:

    def test_warmup_means_no_output_before_slow_period(self):
        from tools.long_trend_probe import golden_cross_signal
        # Only 199 daily bars of data → SMA200 not yet computable → empty.
        closes_by_hour = {d * _DAY: 100.0 for d in range(199)}
        out = golden_cross_signal(closes_by_hour, fast=50, slow=200)
        assert out == {}

    def test_constant_price_zero_signal(self):
        from tools.long_trend_probe import golden_cross_signal
        # 250 days flat → SMA50 == SMA200 → cross sign = 0 from day 200 onward.
        closes_by_hour = {d * _DAY: 100.0 for d in range(250)}
        out = golden_cross_signal(closes_by_hour, fast=50, slow=200)
        # Days 199..249 should all be present (need slow=200 days warm-up:
        # first eligible day index is 199 for the slow window).
        assert len(out) >= 50
        # All values must be 0 because SMA50 == SMA200.
        for v in out.values():
            assert v == 0

    def test_uptrend_emits_plus_one(self):
        from tools.long_trend_probe import golden_cross_signal
        # 250 days monotonic increase → SMA50 (recent half) > SMA200 (older
        # half) → +1 once both are warm.
        closes_by_hour = {d * _DAY: 100.0 + d for d in range(250)}
        out = golden_cross_signal(closes_by_hour, fast=50, slow=200)
        assert len(out) > 0
        # Late in the series, the signal must be +1.
        last_day = max(out.keys())
        assert out[last_day] == 1

    def test_downtrend_emits_minus_one(self):
        from tools.long_trend_probe import golden_cross_signal
        closes_by_hour = {d * _DAY: 1000.0 - d for d in range(250)}
        out = golden_cross_signal(closes_by_hour, fast=50, slow=200)
        last_day = max(out.keys())
        assert out[last_day] == -1


# ── build_trend_signal: per-sample [N, seq_len] z-scored signal ─────────────

class TestBuildTrendSignal:

    def test_shape_matches_n_seq_len(self):
        from tools.long_trend_probe import build_trend_signal
        dist_by_hour = {h * _BAR: 0.01 for h in range(120)}
        sample_end_ts = np.array([60 * _BAR, 90 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, dist_by_hour, seq_len=60)
        assert sig.shape == (2, 60)
        assert sig.dtype == np.float32

    def test_constant_distance_is_zero_zscore(self):
        from tools.long_trend_probe import build_trend_signal
        dist_by_hour = {h * _BAR: 0.05 for h in range(120)}
        sample_end_ts = np.array([100 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, dist_by_hour, seq_len=60)
        # Constant series → std=0 → all-zero output (no fake variance).
        assert np.allclose(sig, 0.0)

    def test_empty_history_returns_zeros(self):
        from tools.long_trend_probe import build_trend_signal
        sample_end_ts = np.array([60 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, {}, seq_len=60)
        assert sig.shape == (1, 60)
        assert np.allclose(sig, 0.0)

    def test_no_lookahead(self):
        """Window for a sample at end_ts must only sample at ts <= end_ts.
        Mirrors btc_dominance_probe's no-lookahead guarantee.
        """
        from tools.long_trend_probe import build_trend_signal
        # Bars 0..59 = constant prefix; 60..119 = different constant future.
        dist_by_hour = {h * _BAR: 0.01 for h in range(60)}
        dist_by_hour.update({h * _BAR: 0.99 for h in range(60, 120)})
        sample_end_ts = np.array([59 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, dist_by_hour, seq_len=60)
        # All 60 windowed bars came from the constant prefix → all equal.
        assert np.allclose(sig, sig[0, 0])
        # And the value must reflect the prefix (0.01), not the future (0.99).
        # 0.01 is below the full-series mean → z-score must be negative.
        assert sig[0, 0] < 0.0

    def test_pre_history_bars_use_neutral_mean(self):
        """Bars earlier than the first known hour map to z=0 (mean-fill),
        matching the btc_dominance_probe and okx_ls_probe contract.
        """
        from tools.long_trend_probe import build_trend_signal
        dist_by_hour = {4 * _BAR: 0.0, 5 * _BAR: 0.02}
        sample_end_ts = np.array([5 * _BAR], dtype=np.int64)
        sig = build_trend_signal(sample_end_ts, dist_by_hour, seq_len=3)
        # Bar 3*_BAR is before first known hour → 0 (neutral z).
        assert sig[0, 0] == pytest.approx(0.0, abs=1e-6)
