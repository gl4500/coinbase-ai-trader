"""Tests for tools/okx_ls_probe.py — OKX long/short ratio single-add probe.

Mirrors tests/test_btc_dominance_probe.py shape: pure helpers (signal
construction, z-scoring, alignment) are tested without touching the cache or
hitting OKX. The main runner is exercised end-to-end at probe runtime
(#235e), not in this test file.
"""
import os
import sys
from typing import Dict

import numpy as np

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from tools import okx_ls_probe as probe  # noqa: E402


_BAR = 3600


# ── build_ls_signal: per-pid z-scored series aligned to bar grid ────────────

class TestBuildLSSignal:

    def test_empty_history_returns_zeros(self):
        sample_end_ts = np.array([1_700_000_000, 1_700_003_600], dtype=np.int64)
        sig = probe.build_ls_signal(sample_end_ts, ls_history={}, seq_len=4)
        assert sig.shape == (2, 4)
        assert np.allclose(sig, 0.0)
        assert sig.dtype == np.float32

    def test_single_value_history_returns_zeros(self):
        """One sample → std=0 → fallback z=0 (avoid div-by-zero)."""
        sample_end_ts = np.array([1_700_000_000], dtype=np.int64)
        history = {1_700_000_000: 1.5}
        sig = probe.build_ls_signal(sample_end_ts, ls_history=history, seq_len=2)
        assert np.allclose(sig, 0.0)

    def test_two_value_history_z_scores_correctly(self):
        sample_end_ts = np.array([2 * _BAR], dtype=np.int64)
        history = {1 * _BAR: 1.0, 2 * _BAR: 3.0}
        sig = probe.build_ls_signal(sample_end_ts, ls_history=history, seq_len=2)
        # mean=2, std=1 → 1.0 → -1.0  and  3.0 → +1.0
        assert sig.shape == (1, 2)
        assert np.isclose(sig[0, 0], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 1], +1.0, atol=1e-6)

    def test_forward_fills_missing_hours(self):
        """Bars between known hours should carry the prior value forward,
        matching the btc_dominance_probe contract."""
        sample_end_ts = np.array([3 * _BAR], dtype=np.int64)
        # Hours 1, 3 known; hour 2 missing → should forward-fill from 1.
        history = {1 * _BAR: 1.0, 3 * _BAR: 3.0}
        sig = probe.build_ls_signal(sample_end_ts, ls_history=history, seq_len=3)
        # mean=2, std=1 over the 2 known values.
        # Bar at 1*BAR: 1.0 → -1.0
        # Bar at 2*BAR: ffill 1.0 → -1.0
        # Bar at 3*BAR: 3.0 → +1.0
        assert np.isclose(sig[0, 0], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 1], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 2], +1.0, atol=1e-6)

    def test_pre_history_bars_are_neutral_zero(self):
        """Bars earlier than the first known hour should be z=0 (mean-fill)."""
        sample_end_ts = np.array([5 * _BAR], dtype=np.int64)
        history = {4 * _BAR: 1.0, 5 * _BAR: 3.0}
        sig = probe.build_ls_signal(sample_end_ts, ls_history=history, seq_len=3)
        # Bar 3*BAR is before first known hour → 0
        # Bar 4*BAR: 1.0 → -1.0
        # Bar 5*BAR: 3.0 → +1.0
        assert np.isclose(sig[0, 0], 0.0, atol=1e-6)
        assert np.isclose(sig[0, 1], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 2], +1.0, atol=1e-6)


# ── ls_history_to_bar_grid: convert ms-stamped fetcher output to bar dict ───

class TestLSHistoryToBarGrid:
    # Use hour-aligned timestamps for fixtures: 472_222 * 3600 = 1_699_999_200,
    # 472_223 * 3600 = 1_700_002_800.
    _HOUR_A_SECS = 472_222 * 3600
    _HOUR_B_SECS = 472_223 * 3600
    _HOUR_A_MS = _HOUR_A_SECS * 1000
    _HOUR_B_MS = _HOUR_B_SECS * 1000

    def test_converts_ms_tuples_to_bar_aligned_seconds(self):
        rows = [
            (self._HOUR_A_MS, 1.5),
            (self._HOUR_B_MS, 2.0),
        ]
        out = probe.ls_history_to_bar_grid(rows)
        assert out == {self._HOUR_A_SECS: 1.5, self._HOUR_B_SECS: 2.0}

    def test_skips_off_grid_timestamps(self):
        """Only timestamps that are exact bar boundaries are kept; OKX
        sometimes returns slightly off-grid stamps (e.g. partial bars)."""
        off_grid_ms = self._HOUR_A_MS + 1_000   # +1s off grid
        rows = [
            (self._HOUR_A_MS, 1.5),         # on grid
            (off_grid_ms, 9.9),             # +1s off grid
            (self._HOUR_B_MS, 2.0),         # on grid
        ]
        out = probe.ls_history_to_bar_grid(rows)
        assert self._HOUR_A_SECS in out
        assert self._HOUR_B_SECS in out
        assert (off_grid_ms // 1000) not in out

    def test_empty_input_returns_empty_dict(self):
        assert probe.ls_history_to_bar_grid([]) == {}


# ── per-pid signal coverage: zero rows → all-zero sig (no fake variance) ────

class TestPerPidSignalCoverage:

    def test_no_coverage_pids_get_all_zero_signal(self):
        sample_end_ts = np.array([_BAR, 2 * _BAR], dtype=np.int64)
        sig = probe.build_ls_signal(sample_end_ts, {}, seq_len=4)
        # If a pid returns no L/S history, its signal should be all zeros so
        # the XGB feature doesn't spuriously fire on noise.
        assert (sig == 0.0).all()
