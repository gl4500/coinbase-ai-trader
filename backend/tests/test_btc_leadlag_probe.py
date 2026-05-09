"""Tests for tools/btc_leadlag_probe.py — BTC→altcoin lead-lag probe.

Mirrors tests/test_btc_dominance_probe.py and tests/test_long_trend_probe.py
shape: pure helpers (lagged returns, rolling β, β-residual, alignment) are
tested without touching the cache. The main runner is exercised end-to-end
at probe runtime (#248), not here.

Why this probe (#246-#248):
    Existing Ch 21 `btc_corr_20` is *contemporaneous* (rolling 20-bar
    correlation alt_ret ↔ btc_ret at the same t). Lead-lag is *temporal*:
    does BTC's move at t-k predict alt's move at t? β-residual strips out
    the current-bar BTC influence to expose idiosyncratic alt motion.
    Together these are structurally different from the 4 failed positioning
    probes (MFI-rank, log10-vol-rank, BTC-dominance, OKX L/S).

Decision rule (#248): any candidate Δ ≥ +0.01 vs ch13 baseline → integrate.
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


# ── log_returns: hourly closes -> hourly log returns ─────────────────────────

class TestLogReturns:

    def test_constant_close_zero_return(self):
        from tools.btc_leadlag_probe import log_returns
        closes = {0: 100.0, _BAR: 100.0, 2 * _BAR: 100.0}
        out = log_returns(closes)
        # First bar is t-1 unavailable → not in output; later bars are 0.
        assert 0 not in out
        assert out[_BAR] == pytest.approx(0.0)
        assert out[2 * _BAR] == pytest.approx(0.0)

    def test_doubling_returns_ln_2(self):
        from tools.btc_leadlag_probe import log_returns
        closes = {0: 100.0, _BAR: 200.0}
        out = log_returns(closes)
        assert out[_BAR] == pytest.approx(np.log(2.0))

    def test_empty_returns_empty(self):
        from tools.btc_leadlag_probe import log_returns
        assert log_returns({}) == {}

    def test_handles_gaps_uses_chronological_neighbor(self):
        """When hours are missing, the return is computed against the prior
        AVAILABLE close, not a forward-filled phantom bar — keeps each
        return one observation, not synthetic."""
        from tools.btc_leadlag_probe import log_returns
        closes = {0: 100.0, 5 * _BAR: 110.0}
        out = log_returns(closes)
        assert out[5 * _BAR] == pytest.approx(np.log(110.0 / 100.0))


# ── lag_dict: shift {ts: v} forward by k bars (hourly grid) ────────────────

class TestLagDict:

    def test_lag_zero_is_identity(self):
        from tools.btc_leadlag_probe import lag_dict
        d = {0: 1.0, _BAR: 2.0}
        assert lag_dict(d, lag_bars=0) == d

    def test_lag_one_shifts_by_one_bar(self):
        from tools.btc_leadlag_probe import lag_dict
        # Value at t=0 is "from t-1" when lagged by 1. Output keys = input
        # keys + 1 bar — i.e. the value originally at t=0 reappears at t=1.
        d = {0: 5.0, _BAR: 7.0}
        out = lag_dict(d, lag_bars=1)
        assert out == {_BAR: 5.0, 2 * _BAR: 7.0}

    def test_lag_keeps_dtype_float(self):
        from tools.btc_leadlag_probe import lag_dict
        d = {0: 1.0}
        out = lag_dict(d, lag_bars=2)
        assert isinstance(next(iter(out.values())), float)


# ── align_pair: produce (alt_ret_arr, btc_ret_arr) on common timestamps ────

class TestAlignPair:

    def test_aligns_only_common_keys(self):
        from tools.btc_leadlag_probe import align_pair
        alt = {0: 0.1, _BAR: 0.2, 2 * _BAR: 0.3}
        btc = {_BAR: 0.5, 2 * _BAR: 0.6, 3 * _BAR: 0.7}
        ts, alt_arr, btc_arr = align_pair(alt, btc)
        assert list(ts) == [_BAR, 2 * _BAR]
        assert list(alt_arr) == [pytest.approx(0.2), pytest.approx(0.3)]
        assert list(btc_arr) == [pytest.approx(0.5), pytest.approx(0.6)]

    def test_empty_intersection_returns_empty(self):
        from tools.btc_leadlag_probe import align_pair
        alt = {0: 0.1}
        btc = {_BAR: 0.5}
        ts, alt_arr, btc_arr = align_pair(alt, btc)
        assert len(ts) == 0
        assert len(alt_arr) == 0
        assert len(btc_arr) == 0


# ── rolling_beta: per-bar rolling OLS β of alt on btc ───────────────────────

class TestRollingBeta:

    def test_perfect_unit_beta_when_alt_eq_btc(self):
        from tools.btc_leadlag_probe import rolling_beta
        # alt_ret = btc_ret exactly → β should be ~1 across the window.
        rng = np.random.RandomState(0)
        btc = rng.normal(0, 0.01, 100)
        alt = btc.copy()
        ts = np.arange(100, dtype=np.int64) * _BAR
        out = rolling_beta(ts, alt, btc, window=20)
        # Only bars with full 20-bar window are emitted.
        for k, v in out.items():
            assert v == pytest.approx(1.0, abs=1e-6)

    def test_zero_beta_when_alt_uncorrelated(self):
        from tools.btc_leadlag_probe import rolling_beta
        rng = np.random.RandomState(0)
        btc = rng.normal(0, 0.01, 1000)
        alt = rng.normal(0, 0.01, 1000)   # independent
        ts = np.arange(1000, dtype=np.int64) * _BAR
        out = rolling_beta(ts, alt, btc, window=200)
        # Mean β across all output windows should be near 0 (large sample).
        vals = np.array(list(out.values()))
        assert abs(vals.mean()) < 0.1

    def test_warmup_excluded(self):
        from tools.btc_leadlag_probe import rolling_beta
        ts = np.arange(50, dtype=np.int64) * _BAR
        alt = np.ones(50)
        btc = np.ones(50)
        out = rolling_beta(ts, alt, btc, window=20)
        # First 19 bars are warm-up; bar 19 onward is eligible.
        for i in range(0, 19):
            assert int(ts[i]) not in out
        assert int(ts[19]) in out

    def test_zero_btc_variance_skipped(self):
        from tools.btc_leadlag_probe import rolling_beta
        ts = np.arange(40, dtype=np.int64) * _BAR
        alt = np.linspace(0, 1, 40)
        btc = np.zeros(40)   # var=0 → β undefined
        out = rolling_beta(ts, alt, btc, window=20)
        # Helper must skip undefined windows rather than crash.
        # All values must be finite (no NaN, no inf).
        for v in out.values():
            assert np.isfinite(v)


# ── beta_residual: alt_ret minus β·btc_ret prediction (current-bar) ─────────

class TestBetaResidual:

    def test_zero_when_alt_perfectly_explained_by_btc(self):
        from tools.btc_leadlag_probe import beta_residual
        rng = np.random.RandomState(0)
        btc = rng.normal(0, 0.01, 100)
        alt = btc.copy()   # β=1, residual=0
        ts = np.arange(100, dtype=np.int64) * _BAR
        out = beta_residual(ts, alt, btc, window=20)
        for v in out.values():
            assert abs(v) < 1e-6

    def test_residual_picks_up_idiosyncratic_move(self):
        from tools.btc_leadlag_probe import beta_residual
        rng = np.random.RandomState(0)
        btc = rng.normal(0, 0.01, 100)
        idio = rng.normal(0, 0.01, 100)
        alt = btc + idio
        ts = np.arange(100, dtype=np.int64) * _BAR
        out = beta_residual(ts, alt, btc, window=20)
        # Residuals should correlate with the idiosyncratic component
        # (after warm-up). Take the eligible bars.
        eligible_ts = sorted(out.keys())
        eligible_idx = [int(t) // _BAR for t in eligible_ts]
        idio_eligible = idio[eligible_idx]
        resid_eligible = np.array([out[t] for t in eligible_ts])
        corr = np.corrcoef(resid_eligible, idio_eligible)[0, 1]
        assert corr > 0.7   # strong (β stripping leaves idio mostly intact)


# ── build_leadlag_signal: per-sample [N, seq_len] z-scored signal ──────────

class TestBuildLeadlagSignal:

    def test_shape_matches_n_seq_len(self):
        from tools.btc_leadlag_probe import build_leadlag_signal
        value_by_ts = {h * _BAR: 0.001 * h for h in range(120)}
        sample_end_ts = np.array([60 * _BAR, 90 * _BAR], dtype=np.int64)
        sig = build_leadlag_signal(sample_end_ts, value_by_ts, seq_len=60)
        assert sig.shape == (2, 60)
        assert sig.dtype == np.float32

    def test_constant_input_zero_zscore(self):
        from tools.btc_leadlag_probe import build_leadlag_signal
        value_by_ts = {h * _BAR: 0.5 for h in range(120)}
        sample_end_ts = np.array([60 * _BAR], dtype=np.int64)
        sig = build_leadlag_signal(sample_end_ts, value_by_ts, seq_len=60)
        assert np.allclose(sig, 0.0)

    def test_empty_returns_zeros(self):
        from tools.btc_leadlag_probe import build_leadlag_signal
        sample_end_ts = np.array([60 * _BAR], dtype=np.int64)
        sig = build_leadlag_signal(sample_end_ts, {}, seq_len=60)
        assert np.allclose(sig, 0.0)

    def test_no_lookahead(self):
        """Window for a sample at end_ts must only sample at ts <= end_ts.
        Same guarantee as build_trend_signal / build_btc_dom_signal."""
        from tools.btc_leadlag_probe import build_leadlag_signal
        value_by_ts = {h * _BAR: 0.01 for h in range(60)}
        value_by_ts.update({h * _BAR: 0.99 for h in range(60, 120)})
        sample_end_ts = np.array([59 * _BAR], dtype=np.int64)
        sig = build_leadlag_signal(sample_end_ts, value_by_ts, seq_len=60)
        # Prefix is constant → all 60 windowed bars should be equal.
        assert np.allclose(sig, sig[0, 0])
        # And below the full-series mean → negative z.
        assert sig[0, 0] < 0.0


# ── BTC pid skip: lead-lag must not include BTC vs itself ──────────────────

class TestBtcPidSkip:
    """Regression for the design intent: BTC's lag-of-itself is just
    autocorrelation, not lead-lag, so the probe runner must skip it."""

    def test_constant_skip_set_includes_btc(self):
        from tools.btc_leadlag_probe import _BTC_PID
        assert _BTC_PID == "BTC-USD"
