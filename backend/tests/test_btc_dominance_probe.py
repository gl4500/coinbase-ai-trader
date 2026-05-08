"""TDD tests for tools/btc_dominance_probe.py — BTC USD-volume share signal.

Helper computes per-hour BTC USD volume share among the top-N USD-quoted pids
on Coinbase (USD vol = native_vol * close). Probe swaps that signal in for
ch13 (obv_slope, marginal noise per #146) and runs the standard purged-CV
Δ-AUC harness.

Decision rule (#156): Δ ≥ +0.01 → integrate as a real channel.

Data source: parquet history we already backfill (no external API needed).
This was the unblocker for #156 — earlier scopings assumed CoinGecko/CMC
historical, but BTC USD-vol share is a faithful dominance proxy computable
from local data, and Coinbase USD-quoted volume is what actually matters
for our deployment universe.
"""
import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


class TestComputeBtcUsdVolumeShare:
    """Pure-function helper: maps {pid: (ts, close, volume)} -> {hour_ts: share}."""

    def test_btc_alone_is_full_share(self):
        from tools.btc_dominance_probe import compute_btc_usd_volume_share
        ts = np.array([3600, 7200, 10800], dtype=np.int64)
        data = {"BTC-USD": (ts, np.full(3, 100.0), np.full(3, 1.0))}
        out = compute_btc_usd_volume_share(data)
        assert out == {3600: 1.0, 7200: 1.0, 10800: 1.0}

    def test_btc_half_when_eth_matches(self):
        from tools.btc_dominance_probe import compute_btc_usd_volume_share
        ts = np.array([3600], dtype=np.int64)
        # BTC: 1 unit * $100 = $100 USD vol; ETH: 100 units * $1 = $100 USD vol
        data = {
            "BTC-USD": (ts, np.array([100.0]), np.array([1.0])),
            "ETH-USD": (ts, np.array([1.0]),   np.array([100.0])),
        }
        out = compute_btc_usd_volume_share(data)
        assert out[3600] == pytest.approx(0.5)

    def test_btc_dominant_when_usd_vol_higher(self):
        from tools.btc_dominance_probe import compute_btc_usd_volume_share
        ts = np.array([3600], dtype=np.int64)
        # BTC: 80M USD; ETH: 10M USD; SOL: 10M USD → BTC = 80%
        data = {
            "BTC-USD": (ts, np.array([80_000_000.0]), np.array([1.0])),
            "ETH-USD": (ts, np.array([10_000_000.0]), np.array([1.0])),
            "SOL-USD": (ts, np.array([10_000_000.0]), np.array([1.0])),
        }
        out = compute_btc_usd_volume_share(data)
        assert out[3600] == pytest.approx(0.8)

    def test_no_btc_returns_empty(self):
        """When BTC-USD missing, no share can be computed."""
        from tools.btc_dominance_probe import compute_btc_usd_volume_share
        ts = np.array([3600], dtype=np.int64)
        data = {"ETH-USD": (ts, np.array([100.0]), np.array([1.0]))}
        out = compute_btc_usd_volume_share(data)
        assert out == {}

    def test_only_aligns_hours_present_in_btc(self):
        """A bar appears in output iff BTC has data for it. Other pids fill in."""
        from tools.btc_dominance_probe import compute_btc_usd_volume_share
        btc_ts = np.array([3600, 7200], dtype=np.int64)
        eth_ts = np.array([7200, 10800], dtype=np.int64)
        data = {
            "BTC-USD": (btc_ts, np.full(2, 100.0), np.full(2, 1.0)),
            "ETH-USD": (eth_ts, np.full(2, 1.0),   np.full(2, 100.0)),
        }
        out = compute_btc_usd_volume_share(data)
        # 3600: BTC alone → 1.0
        # 7200: BTC=$100, ETH=$100 → 0.5
        # 10800: not in BTC → not in output
        assert set(out.keys()) == {3600, 7200}
        assert out[3600] == pytest.approx(1.0)
        assert out[7200] == pytest.approx(0.5)


class TestBuildBtcDomSignal:
    """Per-sample [N, seq_len] alignment helper. Output is z-scored over
    the entire share series so the channel is unit-variance like the others.
    """

    def test_shape_matches_n_seq_len(self):
        from tools.btc_dominance_probe import build_btc_dom_signal
        share_by_hour = {h * 3600: 0.5 for h in range(120)}
        sample_end_ts = np.array([59 * 3600, 60 * 3600, 61 * 3600], dtype=np.int64)
        sig = build_btc_dom_signal(sample_end_ts, share_by_hour, seq_len=60)
        assert sig.shape == (3, 60)

    def test_constant_share_is_zero_zscore(self):
        from tools.btc_dominance_probe import build_btc_dom_signal
        share_by_hour = {h * 3600: 0.5 for h in range(120)}
        sample_end_ts = np.array([100 * 3600], dtype=np.int64)
        sig = build_btc_dom_signal(sample_end_ts, share_by_hour, seq_len=60)
        # constant series → std=0, helper falls back to sigma=1 → all zeros
        assert np.allclose(sig, 0.0)

    def test_missing_hours_use_forward_fill_then_neutral(self):
        from tools.btc_dominance_probe import build_btc_dom_signal
        # Only hour 0 and hour 50 populated; gaps forward-filled
        share_by_hour = {0: 0.4, 50 * 3600: 0.6}
        sample_end_ts = np.array([60 * 3600], dtype=np.int64)
        sig = build_btc_dom_signal(sample_end_ts, share_by_hour, seq_len=60)
        # No NaN
        assert not np.isnan(sig).any()
        # Some variance (z-scored, std > 0 in the original series)
        assert sig.shape == (1, 60)

    def test_no_lookahead(self):
        """Window for a sample at end_ts must only use shares at ts <= end_ts.

        Z-score normalization uses the full series stats (matches the OI
        probe pattern — acceptable for offline AUC evaluation), so when the
        prefix is constant 0.5 and the future is 0.99, the windowed bars all
        share the same VALUE. Therefore all 60 z-scores must be equal —
        proving the window only sampled the prefix, not the future.
        """
        from tools.btc_dominance_probe import build_btc_dom_signal
        share_by_hour = {h * 3600: 0.5 for h in range(60)}
        share_by_hour.update({h * 3600: 0.99 for h in range(60, 120)})
        sample_end_ts = np.array([59 * 3600], dtype=np.int64)
        sig = build_btc_dom_signal(sample_end_ts, share_by_hour, seq_len=60)
        # All 60 windowed bars came from the constant prefix → all equal.
        assert np.allclose(sig, sig[0, 0])
        # And the value is the z-score of 0.5 (the only prefix value), not 0.99.
        # 0.5 is below the full-series mean → z-score must be negative.
        assert sig[0, 0] < 0.0

    def test_empty_share_returns_zeros(self):
        from tools.btc_dominance_probe import build_btc_dom_signal
        sample_end_ts = np.array([60 * 3600], dtype=np.int64)
        sig = build_btc_dom_signal(sample_end_ts, {}, seq_len=60)
        assert sig.shape == (1, 60)
        assert np.allclose(sig, 0.0)


class TestDominanceBasket:
    """Regression for #156 first-run bug: pooled top-20 by sample count is
    dominated by alts/memes and didn't contain BTC-USD, so the proxy short-
    circuited to {} (0 hours covered, 0% per-sample coverage). The basket
    used to compute the dominance series must be decoupled from the sample
    universe and always include BTC-USD.
    """

    def test_basket_constant_includes_btc(self):
        from tools.btc_dominance_probe import _DOMINANCE_BASKET
        assert "BTC-USD" in _DOMINANCE_BASKET

    def test_basket_constant_includes_majors(self):
        """Basket must span enough liquidity to make the share number stable.
        Without ETH/SOL etc. the denominator collapses and the share saturates."""
        from tools.btc_dominance_probe import _DOMINANCE_BASKET
        for pid in ("BTC-USD", "ETH-USD", "SOL-USD"):
            assert pid in _DOMINANCE_BASKET, pid

    def test_basket_loader_skips_missing_pids(self, tmp_path):
        """build_pid_history_from_basket must silently skip pids whose
        parquet file doesn't exist (e.g. delisted pid in the basket constant)."""
        import pandas as pd
        from tools.btc_dominance_probe import build_pid_history_from_basket

        # Create one valid parquet
        ts = np.array([3600, 7200], dtype=np.int64)
        df = pd.DataFrame({
            "start": ts, "open": [100.0, 101.0], "high": [101.0, 102.0],
            "low": [99.0, 100.0], "close": [100.5, 101.5],
            "volume": [1.0, 2.0], "ingest_ts": [0, 0], "schema_version": [1, 1],
        })
        df.to_parquet(tmp_path / "BTC-USD.parquet")

        out = build_pid_history_from_basket(
            ["BTC-USD", "DOES-NOT-EXIST-USD"],
            history_dir=str(tmp_path),
        )
        assert "BTC-USD" in out
        assert "DOES-NOT-EXIST-USD" not in out
        # Sanity: the loaded arrays match what we wrote
        out_ts, out_close, out_vol = out["BTC-USD"]
        assert list(out_ts) == [3600, 7200]
        assert list(out_close) == [100.5, 101.5]
