"""TDD tests for silver-layer OHLCV anomaly flagger (#167).

Standalone analytics primitive that scans a bronze candle list and
flags bars violating cheap, defensible sanity rules:

  - OHLC inconsistency (high < low, open or close outside [low,high])
  - Zero-volume tail (run of consecutive zero-volume bars)
  - Return z-score outlier (|log-return| > k * rolling_std)
  - Volume spike (volume > k * rolling median)

The flagger returns a list of anomaly records — one per offending
bar with `start`, `kind`, and a short `detail` string — so the caller
(audit script, dashboard) can present them. No I/O, no external deps
beyond numpy. Cross-exchange reconciliation is a separate follow-up.
"""

from __future__ import annotations

import os
import sys

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(start, o=100.0, h=101.0, lo=99.0, c=100.5, v=10.0):
    return {"start": start, "open": o, "high": h, "low": lo, "close": c, "volume": v}


class TestOHLCConsistency:
    def test_high_below_low_flagged(self):
        from tools.anomaly_flagger import flag_ohlc_consistency

        bars = [_candle(1, h=98.0, lo=100.0)]  # high < low
        out = flag_ohlc_consistency(bars)
        assert len(out) == 1
        assert out[0]["start"] == 1
        assert out[0]["kind"] == "ohlc_inconsistent"

    def test_open_above_high_flagged(self):
        from tools.anomaly_flagger import flag_ohlc_consistency

        bars = [_candle(2, o=200.0, h=101.0, lo=99.0, c=100.5)]
        out = flag_ohlc_consistency(bars)
        assert len(out) == 1
        assert out[0]["kind"] == "ohlc_inconsistent"

    def test_close_below_low_flagged(self):
        from tools.anomaly_flagger import flag_ohlc_consistency

        bars = [_candle(3, o=100.0, h=101.0, lo=99.0, c=50.0)]
        out = flag_ohlc_consistency(bars)
        assert len(out) == 1

    def test_consistent_bars_pass(self):
        from tools.anomaly_flagger import flag_ohlc_consistency

        bars = [_candle(i) for i in range(5)]
        assert flag_ohlc_consistency(bars) == []


class TestZeroVolumeRuns:
    def test_long_zero_volume_run_flagged(self):
        from tools.anomaly_flagger import flag_zero_volume_runs

        bars = [_candle(i, v=0.0) for i in range(8)]
        out = flag_zero_volume_runs(bars, min_run=5)
        assert len(out) >= 1
        assert all(r["kind"] == "zero_volume_run" for r in out)

    def test_short_zero_run_does_not_flag(self):
        from tools.anomaly_flagger import flag_zero_volume_runs

        bars = [_candle(i, v=0.0) for i in range(3)] + [_candle(99, v=10.0)]
        out = flag_zero_volume_runs(bars, min_run=5)
        assert out == []

    def test_isolated_zero_does_not_flag(self):
        from tools.anomaly_flagger import flag_zero_volume_runs

        bars = [
            _candle(1, v=10.0),
            _candle(2, v=0.0),
            _candle(3, v=10.0),
        ]
        assert flag_zero_volume_runs(bars, min_run=5) == []


class TestReturnZOutliers:
    def test_giant_jump_flagged(self):
        from tools.anomaly_flagger import flag_return_z_outliers

        # 100 quiet bars with +/-0.001 returns, then a single +50% jump
        rng = np.random.default_rng(0)
        prices = [100.0]
        for _ in range(99):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.001)))
        prices.append(prices[-1] * 1.5)  # huge jump
        bars = [_candle(i, c=p) for i, p in enumerate(prices)]
        out = flag_return_z_outliers(bars, window=30, k=4.0)
        starts = [r["start"] for r in out]
        assert (len(prices) - 1) in starts
        assert all(r["kind"] == "return_z_outlier" for r in out)

    def test_quiet_series_no_outliers(self):
        from tools.anomaly_flagger import flag_return_z_outliers

        # Real "quiet" data is a low-vol random walk, not deterministic
        # linear prices (those have near-zero rolling stdev → spurious
        # z-scores). Use seeded log-normal noise.
        rng = np.random.default_rng(42)
        prices = [100.0]
        for _ in range(150):
            prices.append(prices[-1] * (1 + rng.normal(0, 0.001)))
        bars = [_candle(i, c=p) for i, p in enumerate(prices)]
        out = flag_return_z_outliers(bars, window=30, k=4.0)
        assert out == []


class TestVolumeSpikes:
    def test_100x_spike_flagged(self):
        from tools.anomaly_flagger import flag_volume_spikes

        bars = [_candle(i, v=10.0) for i in range(50)]
        bars.append(_candle(50, v=10000.0))
        out = flag_volume_spikes(bars, window=20, k=10.0)
        assert any(r["start"] == 50 and r["kind"] == "volume_spike" for r in out)

    def test_steady_volume_passes(self):
        from tools.anomaly_flagger import flag_volume_spikes

        bars = [_candle(i, v=10.0 + 0.1 * i) for i in range(60)]
        out = flag_volume_spikes(bars, window=20, k=10.0)
        assert out == []


class TestRunAll:
    def test_returns_combined_report(self):
        from tools.anomaly_flagger import scan_bars

        # One inconsistent bar + one giant return jump
        bars = [_candle(i) for i in range(50)]
        bars[10] = _candle(10, h=98.0, lo=100.0)  # ohlc bad
        report = scan_bars(bars)
        kinds = {r["kind"] for r in report["anomalies"]}
        assert "ohlc_inconsistent" in kinds
        assert "n_bars" in report
        assert report["n_bars"] == 50

    def test_clean_bars_empty_anomalies(self):
        from tools.anomaly_flagger import scan_bars

        bars = [_candle(i) for i in range(100)]
        report = scan_bars(bars)
        assert report["anomalies"] == []
