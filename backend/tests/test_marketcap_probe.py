"""Tests for tools/marketcap_probe.py — CoinGecko marketcap single-add probe.

Mirrors tests/test_okx_ls_probe.py shape: pure helpers (signal construction,
z-scoring, hour-grid alignment) are tested without touching the cache or
hitting CoinGecko. The main runner is exercised end-to-end at probe runtime
(#260f), not in this test file.

Probe candidate (#260): log(market_cap) per pid, forward-filled to the
hourly bar grid with a strict-causal 1-day lag, z-scored per pid, and
substituted for ch13 (obv_slope, lowest-importance per #146 ablation).
"""
import os
import sys
from typing import Dict

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


# ── Constants pinned by spec (#260) ─────────────────────────────────────────

class TestProbeConstants:

    def test_target_channel_is_ch13(self):
        """Spec §'Target channel': ch13 (obv_slope) — lowest-importance per
        #146 ablation, same channel that #235 (OKX L/S) targeted."""
        from tools.marketcap_probe import _TARGET_CHANNEL
        assert _TARGET_CHANNEL == 13

    def test_lag_secs_one_day(self):
        """CoinGecko stamps daily marketcap at end-of-day UTC. Same-day data
        at sample t leaks future close into a feature read at t. 1-day lag
        is strict-causal."""
        from tools.marketcap_probe import _LAG_SECS
        assert _LAG_SECS == _DAY

    def test_bar_secs_hourly(self):
        from tools.marketcap_probe import _BAR_SECS
        assert _BAR_SECS == _BAR

    def test_seq_len_60(self):
        """Mirrors btc_leadlag_probe / okx_ls_probe convention so the
        replacement signal is shape-compatible with the existing 60-bar
        feature window."""
        from tools.marketcap_probe import _SEQ_LEN
        assert _SEQ_LEN == 60


# ── marketcap_rows_to_log_grid: ms-stamped (ts, mc) → bar-aligned log-mc ────

class TestMarketcapRowsToLogGrid:

    def test_applies_log_to_marketcap(self):
        """log(market_cap) is the candidate — magnitude varies over orders of
        magnitude across pids (BTC ~$1.5T vs MOODENG ~$200M), so log
        normalizes the dynamic range before z-scoring."""
        from tools.marketcap_probe import marketcap_rows_to_log_grid
        # Two daily rows (ms-stamped).
        rows = [
            (1_700_000_000_000, 1e9),     # ts at h=472_222 in seconds
            (1_700_000_000_000 + _DAY * 1000, 2e9),
        ]
        # Hour grid that's the same as one of the row ts + lag.
        hour_grid_ms = [
            (1_700_000_000_000 + _DAY * 1000),       # at row[0].ts + lag
            (1_700_000_000_000 + 2 * _DAY * 1000),   # at row[1].ts + lag
        ]
        out = marketcap_rows_to_log_grid(
            rows, hour_grid_ms=hour_grid_ms, lag_secs=_DAY,
        )
        # Keys are seconds (bar-aligned), values are ln(market_cap).
        assert isinstance(out, dict)
        assert all(isinstance(k, int) for k in out.keys())
        # ln(1e9) ≈ 20.72; ln(2e9) ≈ 21.42
        for v in out.values():
            assert isinstance(v, float)
            assert 15.0 < v < 25.0

    def test_lag_is_applied(self):
        """Default 1-day lag means a row stamped at day D is only available at
        sample times >= day D+1. A grid point exactly at the row's own ts
        must NOT see the row's value."""
        from tools.marketcap_probe import marketcap_rows_to_log_grid
        row_ms = 1_700_000_000_000
        rows = [(row_ms, 1e9)]
        # Grid point exactly at the row ts: cutoff = ts - lag, no row qualifies.
        out = marketcap_rows_to_log_grid(
            rows, hour_grid_ms=[row_ms], lag_secs=_DAY,
        )
        assert out == {}

    def test_zero_lag_allows_same_ts_match(self):
        from tools.marketcap_probe import marketcap_rows_to_log_grid
        row_ms = 1_700_000_000_000
        rows = [(row_ms, 1e9)]
        out = marketcap_rows_to_log_grid(
            rows, hour_grid_ms=[row_ms], lag_secs=0,
        )
        # Single grid point, single row, zero lag: row.ts <= ts → applies.
        assert len(out) == 1
        bar_secs = row_ms // 1000
        assert bar_secs in out
        assert np.isclose(out[bar_secs], np.log(1e9), atol=1e-6)

    def test_empty_rows_returns_empty(self):
        from tools.marketcap_probe import marketcap_rows_to_log_grid
        out = marketcap_rows_to_log_grid(
            [], hour_grid_ms=[1_700_000_000_000], lag_secs=_DAY,
        )
        assert out == {}

    def test_non_positive_marketcap_skipped(self):
        """log undefined for mc <= 0. CoinGecko occasionally returns 0 for
        pre-launch coins or stale rows — those must be filtered."""
        from tools.marketcap_probe import marketcap_rows_to_log_grid
        row_a = 1_700_000_000_000
        row_b = row_a + _DAY * 1000
        rows = [
            (row_a, 0.0),     # invalid — log undefined
            (row_b, 1e9),     # valid
        ]
        grid = [row_a + 2 * _DAY * 1000]   # past row_b + lag
        out = marketcap_rows_to_log_grid(
            rows, hour_grid_ms=grid, lag_secs=_DAY,
        )
        # Only row_b is valid; the grid point should pick up its log value.
        assert len(out) == 1
        bar_secs = grid[0] // 1000
        assert bar_secs in out
        assert np.isclose(out[bar_secs], np.log(1e9), atol=1e-6)


# ── build_marketcap_signal: per-pid z-scored series aligned to bar grid ─────

class TestBuildMarketcapSignal:

    def test_empty_history_returns_zeros(self):
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([1_700_000_000, 1_700_003_600], dtype=np.int64)
        sig = build_marketcap_signal(sample_end_ts, mc_history={}, seq_len=4)
        assert sig.shape == (2, 4)
        assert np.allclose(sig, 0.0)
        assert sig.dtype == np.float32

    def test_single_value_history_returns_zeros(self):
        """One sample → std=0 → fallback z=0 (avoid div-by-zero)."""
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([1_700_000_000], dtype=np.int64)
        history = {1_700_000_000: 20.5}
        sig = build_marketcap_signal(sample_end_ts, mc_history=history, seq_len=2)
        assert np.allclose(sig, 0.0)

    def test_two_value_history_z_scores_correctly(self):
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([2 * _BAR], dtype=np.int64)
        history = {1 * _BAR: 20.0, 2 * _BAR: 22.0}
        sig = build_marketcap_signal(sample_end_ts, mc_history=history, seq_len=2)
        # mean=21, std=1 → 20.0 → -1.0  and  22.0 → +1.0
        assert sig.shape == (1, 2)
        assert np.isclose(sig[0, 0], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 1], +1.0, atol=1e-6)

    def test_forward_fills_missing_hours(self):
        """Marketcap only updates daily, so most hourly bars between known
        timestamps are missing and must forward-fill from the prior known
        value (matching the OKX L/S probe contract)."""
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([3 * _BAR], dtype=np.int64)
        # Hours 1, 3 known; hour 2 missing → should forward-fill from 1.
        history = {1 * _BAR: 20.0, 3 * _BAR: 22.0}
        sig = build_marketcap_signal(sample_end_ts, mc_history=history, seq_len=3)
        # mean=21, std=1 over the 2 known values.
        # Bar at 1*BAR: 20.0 → -1.0
        # Bar at 2*BAR: ffill 20.0 → -1.0
        # Bar at 3*BAR: 22.0 → +1.0
        assert np.isclose(sig[0, 0], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 1], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 2], +1.0, atol=1e-6)

    def test_pre_history_bars_are_neutral_zero(self):
        """Bars earlier than the first known hour should be z=0 (mean-fill)."""
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([5 * _BAR], dtype=np.int64)
        history = {4 * _BAR: 20.0, 5 * _BAR: 22.0}
        sig = build_marketcap_signal(sample_end_ts, mc_history=history, seq_len=3)
        # Bar 3*BAR is before first known hour → 0
        # Bar 4*BAR: 20.0 → -1.0
        # Bar 5*BAR: 22.0 → +1.0
        assert np.isclose(sig[0, 0], 0.0, atol=1e-6)
        assert np.isclose(sig[0, 1], -1.0, atol=1e-6)
        assert np.isclose(sig[0, 2], +1.0, atol=1e-6)


# ── per-pid signal coverage: empty pid → all-zero (no fake variance) ────────

class TestPerPidSignalCoverage:

    def test_no_coverage_pids_get_all_zero_signal(self):
        from tools.marketcap_probe import build_marketcap_signal
        sample_end_ts = np.array([_BAR, 2 * _BAR], dtype=np.int64)
        sig = build_marketcap_signal(sample_end_ts, {}, seq_len=4)
        # If a pid has no marketcap history (CoinGecko slug missing or
        # COINGECKO_DISABLED), its signal must be all zeros so the XGB
        # feature doesn't spuriously fire on noise.
        assert (sig == 0.0).all()


# ── --source flag dispatch (#285) ───────────────────────────────────────────

class TestSourceDispatch:
    """`--source cmc|coingecko|both` flag routes history fetches through the
    matching provider. coingecko (default) hits the bronze cache from #284.
    coinpaprika hits the no-key fallback service. `both` runs the probe twice
    for A/B comparison.
    """

    @pytest.mark.asyncio
    async def test_default_source_is_coingecko(self, monkeypatch, tmp_path):
        """Omitting --source uses coingecko via marketcap_history_cache."""
        from tools import marketcap_probe as mp

        called = {"src": None}

        async def _spy(pid, start_ms, end_ms, parquet_dir=None, refresh_secs=86400):
            called["src"] = "coingecko"
            return [(1_700_000_000_000, 1e9)]

        # Patch the cached fetcher so we observe the routing.
        monkeypatch.setattr(
            "services.marketcap_history_cache.fetch_marketcap_history_cached",
            _spy,
        )

        await mp._fetch_marketcap_for_pids(
            ["BTC-USD"], 1_700_000_000_000, 1_700_000_500_000,
            hour_grid_ms=[1_700_000_500_000],
            source="coingecko",
            parquet_dir=str(tmp_path),
        )
        assert called["src"] == "coingecko"

    @pytest.mark.asyncio
    async def test_coinpaprika_source_dispatches_to_paprika(
        self, monkeypatch, tmp_path
    ):
        from tools import marketcap_probe as mp

        called = {"src": None}

        async def _spy(pid, start_ms, end_ms):
            called["src"] = "coinpaprika"
            return [(1_700_000_000_000, 1e9)]

        # The coinpaprika branch must call services.coinpaprika_marketcap
        # .fetch_marketcap_history directly (no cache wrapper; that one is
        # CoinGecko-specific for now).
        monkeypatch.setattr(
            "services.coinpaprika_marketcap.fetch_marketcap_history", _spy
        )

        await mp._fetch_marketcap_for_pids(
            ["BTC-USD"], 1_700_000_000_000, 1_700_000_500_000,
            hour_grid_ms=[1_700_000_500_000],
            source="coinpaprika",
            parquet_dir=str(tmp_path),
        )
        assert called["src"] == "coinpaprika"

    @pytest.mark.asyncio
    async def test_unknown_source_raises_value_error(self, tmp_path):
        from tools import marketcap_probe as mp
        with pytest.raises(ValueError):
            await mp._fetch_marketcap_for_pids(
                ["BTC-USD"], 1_700_000_000_000, 1_700_000_500_000,
                hour_grid_ms=[1_700_000_500_000],
                source="bogus",
                parquet_dir=str(tmp_path),
            )

    def test_cli_accepts_source_flag(self):
        """`--source` is a recognized CLI flag with the three documented
        choices."""
        from tools.marketcap_probe import _build_argparser
        parser = _build_argparser()
        ns = parser.parse_args(["--source", "coinpaprika"])
        assert ns.source == "coinpaprika"
        ns = parser.parse_args(["--source", "both"])
        assert ns.source == "both"
        ns = parser.parse_args([])
        assert ns.source == "coingecko"
        with pytest.raises(SystemExit):
            parser.parse_args(["--source", "bogus"])
