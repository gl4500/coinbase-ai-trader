"""TDD tests for inference-time feature-freshness gate (#169).

A live inference window is (n_channels, seq_len). A channel that hasn't
moved in K trailing bars is suspect — feed paused, geo-blocked, broker
hiccup. The gate detects per-channel trailing-flat runs against caller-
supplied bar budgets and reports a pass/fail flag plus the offending
channels so the caller (cnn_agent / xgb_signal) can choose to block,
score-with-warning, or fall back.

Pure-numpy first pass; no external state, no I/O.
"""

from __future__ import annotations

import os
import sys

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestTrailingFlatBars:
    def test_zero_when_last_bar_differs(self):
        from tools.freshness_gate import _trailing_flat_bars

        ch = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _trailing_flat_bars(ch) == 0

    def test_counts_run_of_repeated_tail(self):
        from tools.freshness_gate import _trailing_flat_bars

        ch = np.array([1.0, 2.0, 3.0, 5.0, 5.0, 5.0])
        # last 3 bars are identical → 2 trailing flat (steps without change)
        assert _trailing_flat_bars(ch) == 2

    def test_all_constant_returns_n_minus_1(self):
        from tools.freshness_gate import _trailing_flat_bars

        ch = np.full(10, 0.7)
        assert _trailing_flat_bars(ch) == 9

    def test_short_channel_does_not_crash(self):
        from tools.freshness_gate import _trailing_flat_bars

        assert _trailing_flat_bars(np.array([1.0])) == 0
        assert _trailing_flat_bars(np.array([])) == 0


class TestEvaluateFreshness:
    def test_fresh_window_passes(self):
        from tools.freshness_gate import evaluate_freshness

        rng = np.random.default_rng(0)
        # 28 channels, 60 bars, all wiggling → nothing trailing-flat
        window = rng.normal(0.0, 1.0, size=(28, 60))
        out = evaluate_freshness(window, max_flat_bars=5)
        assert out["fresh"] is True
        assert out["stale_channels"] == []

    def test_stale_channel_flagged(self):
        from tools.freshness_gate import evaluate_freshness

        rng = np.random.default_rng(1)
        window = rng.normal(0.0, 1.0, size=(4, 60))
        # Freeze ch=2's last 10 bars to a constant
        window[2, -10:] = 0.42
        out = evaluate_freshness(window, max_flat_bars=5)
        assert out["fresh"] is False
        assert 2 in out["stale_channels"]
        # Other channels not flagged
        assert 0 not in out["stale_channels"]

    def test_per_channel_overrides_respected(self):
        from tools.freshness_gate import evaluate_freshness

        rng = np.random.default_rng(2)
        window = rng.normal(0.0, 1.0, size=(4, 60))
        # Channel 1: a slow 1h-cadence feed that legitimately repeats 11 bars
        # at 5m cadence between updates — caller raises its budget
        window[1, -11:] = 0.5
        out = evaluate_freshness(
            window,
            max_flat_bars=5,
            per_channel_max={1: 12},
        )
        assert 1 not in out["stale_channels"]

    def test_report_contains_per_channel_flat_counts(self):
        from tools.freshness_gate import evaluate_freshness

        rng = np.random.default_rng(3)
        window = rng.normal(0.0, 1.0, size=(3, 30))
        window[0, -7:] = 1.0
        out = evaluate_freshness(window, max_flat_bars=5)
        assert "channel_flat_bars" in out
        assert out["channel_flat_bars"][0] >= 6
        assert len(out["channel_flat_bars"]) == 3

    def test_threshold_boundary_exactly_max_is_ok(self):
        from tools.freshness_gate import evaluate_freshness

        window = np.zeros((1, 20))
        window[0] = np.arange(20, dtype=np.float64)
        # Repeat the last value once — flat=1 — at threshold=1 → not stale
        window[0, -1] = window[0, -2]
        out = evaluate_freshness(window, max_flat_bars=1)
        assert out["fresh"] is True

    def test_above_threshold_is_stale(self):
        from tools.freshness_gate import evaluate_freshness

        window = np.zeros((1, 20))
        window[0] = np.arange(20, dtype=np.float64)
        window[0, -3:] = window[0, -4]  # 3 trailing flat
        out = evaluate_freshness(window, max_flat_bars=2)
        assert out["fresh"] is False
        assert 0 in out["stale_channels"]


class TestIgnoredChannels:
    def test_constant_channels_ignored(self):
        from tools.freshness_gate import evaluate_freshness

        # Ch 11 is permanently zero in prod (geo-blocked feed) — caller passes
        # ignore=[11]; the gate should never flag it however flat it is.
        window = np.zeros((12, 30))
        window[11] = 0.0  # all-zero column
        rng = np.random.default_rng(4)
        for c in range(11):
            window[c] = rng.normal(0.0, 1.0, size=30)
        out = evaluate_freshness(
            window,
            max_flat_bars=5,
            ignore_channels=[11],
        )
        # Channel 11 was 100% flat but excluded
        assert 11 not in out["stale_channels"]
        assert out["fresh"] is True
