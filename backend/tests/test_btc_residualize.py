"""Tests for tools/btc_residualize.py — BTC-residualization helpers.

Why these helpers (#253):
    Empirical history shows explicit BTC features have all failed (#149 Ch 21
    btc_corr dropped as anti-signal; #156 BTC-dominance failed; #246-#248 BTC
    lead-lag at lags 1/3/6/12/24 all 5 failed). User's domain intuition
    (BTC leads alts) is sound, but the channels we tried each carry the BTC
    move *into* the input rather than *out* of it. β-residualization inverts
    that:
        r_alt[t] = β[t] · r_btc[t] + ε[t]
        ε[t] = r_alt[t] - β[t] · r_btc[t]   ← the alt-specific component
    The residual ε is what's left of the alt's move once the BTC-correlated
    part is stripped away. That's the candidate alpha for Ch 9 (1-bar price
    change) under #253c.

Causality: β[t] is fit on [t-W, t-1] strictly — it does NOT see r_btc[t] or
r_alt[t] at fit time. A lookahead test pins this invariant.

Pure-array helpers (no I/O, no dicts) so they can be applied directly inside
FeatureBuilder.build (#253c) on the 5m grid with W=288 (24h).
"""

import os
import sys

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")


# ── compute_rolling_beta ───────────────────────────────────────────────────


class TestComputeRollingBeta:
    """β[t] = Cov(alt[t-W:t], btc[t-W:t]) / Var(btc[t-W:t]); strict — excludes t."""

    def test_recovers_known_beta(self):
        from tools.btc_residualize import compute_rolling_beta

        rng = np.random.RandomState(0)
        n, W = 500, 50
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = 2.0 * btc  # β ≡ 2 exactly
        beta = compute_rolling_beta(alt, btc, window=W)
        assert beta.shape == (n,)
        # Eligible bars (t >= W) should be ~2.0.
        for v in beta[W:]:
            assert v == pytest.approx(2.0, abs=1e-6)

    def test_warmup_is_nan(self):
        from tools.btc_residualize import compute_rolling_beta

        n, W = 100, 30
        rng = np.random.RandomState(1)
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        beta = compute_rolling_beta(alt, btc, window=W)
        # First W entries (indices 0..W-1) cannot be computed without
        # peeking at t — they must be NaN.
        assert np.all(np.isnan(beta[:W]))
        # And bar W onward must be finite.
        assert np.all(np.isfinite(beta[W:]))

    def test_strict_causality_no_lookahead(self):
        """β[t] must depend ONLY on alt[t-W:t] and btc[t-W:t]. Mutating
        alt[t] or btc[t] (or any later index) MUST NOT change β[t]."""
        from tools.btc_residualize import compute_rolling_beta

        rng = np.random.RandomState(2)
        n, W = 200, 40
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        beta_before = compute_rolling_beta(alt, btc, window=W)
        # Mutate t and beyond.
        t = 100
        alt2 = alt.copy()
        btc2 = btc.copy()
        alt2[t:] = 999.0
        btc2[t:] = -999.0
        beta_after = compute_rolling_beta(alt2, btc2, window=W)
        # β at index t (and any earlier eligible index) must be unchanged.
        for i in range(W, t + 1):
            assert beta_after[i] == pytest.approx(beta_before[i], abs=1e-12), (
                f"β[{i}] changed after mutating index {t}+ — lookahead leak"
            )

    def test_zero_btc_variance_returns_finite_zero(self):
        """If btc has zero variance in the window, β is undefined. Helper
        must emit 0.0 (not NaN, not inf) so downstream consumers don't see
        garbage. (Channel must remain finite under #253c spec.)"""
        from tools.btc_residualize import compute_rolling_beta

        n, W = 100, 30
        alt = np.linspace(0.0, 1.0, n).astype(np.float64)
        btc = np.zeros(n, dtype=np.float64)  # var=0 in every window
        beta = compute_rolling_beta(alt, btc, window=W)
        # Warm-up still NaN; eligible bars must be finite (0.0 fallback).
        assert np.all(np.isnan(beta[:W]))
        assert np.all(np.isfinite(beta[W:]))
        for v in beta[W:]:
            assert v == pytest.approx(0.0, abs=1e-12)

    def test_zero_beta_when_uncorrelated(self):
        from tools.btc_residualize import compute_rolling_beta

        rng = np.random.RandomState(3)
        n, W = 2000, 200
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)  # independent
        beta = compute_rolling_beta(alt, btc, window=W)
        # Mean β over eligible bars should be near zero (large sample).
        eligible = beta[W:]
        assert abs(float(eligible.mean())) < 0.1

    def test_input_shape_preserved(self):
        from tools.btc_residualize import compute_rolling_beta

        n, W = 50, 20
        rng = np.random.RandomState(4)
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        beta = compute_rolling_beta(alt, btc, window=W)
        assert beta.shape == (n,)
        assert beta.dtype == np.float64

    def test_short_input_all_nan(self):
        """If the array is shorter than the window, every entry is NaN —
        no eligible bar exists."""
        from tools.btc_residualize import compute_rolling_beta

        n, W = 10, 30
        rng = np.random.RandomState(5)
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        beta = compute_rolling_beta(alt, btc, window=W)
        assert beta.shape == (n,)
        assert np.all(np.isnan(beta))

    def test_mismatched_lengths_raises(self):
        from tools.btc_residualize import compute_rolling_beta

        with pytest.raises(ValueError):
            compute_rolling_beta(
                np.zeros(50, dtype=np.float64),
                np.zeros(60, dtype=np.float64),
                window=20,
            )

    def test_invalid_window_raises(self):
        from tools.btc_residualize import compute_rolling_beta

        arr = np.zeros(50, dtype=np.float64)
        with pytest.raises(ValueError):
            compute_rolling_beta(arr, arr, window=0)
        with pytest.raises(ValueError):
            compute_rolling_beta(arr, arr, window=-5)


# ── residualize_returns ────────────────────────────────────────────────────


class TestResidualizeReturns:
    """ε[t] = alt[t] - β[t] · btc[t] where β[t] from [t-W, t-1] strictly."""

    def test_zero_residual_when_alt_eq_beta_btc(self):
        """alt = 2·btc exactly → β=2, residual=0 (after warm-up)."""
        from tools.btc_residualize import residualize_returns

        rng = np.random.RandomState(0)
        n, W = 500, 50
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = 2.0 * btc
        eps = residualize_returns(alt, btc, window=W)
        assert eps.shape == (n,)
        for v in eps[W:]:
            assert abs(v) < 1e-6

    def test_residual_recovers_idiosyncratic_component(self):
        """alt = β·btc + idio → residual should correlate strongly with idio."""
        from tools.btc_residualize import residualize_returns

        rng = np.random.RandomState(1)
        n, W = 1000, 100
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        idio = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = 1.5 * btc + idio
        eps = residualize_returns(alt, btc, window=W)
        eligible = ~np.isnan(eps)
        corr = np.corrcoef(eps[eligible], idio[eligible])[0, 1]
        assert corr > 0.7

    def test_warmup_is_nan(self):
        from tools.btc_residualize import residualize_returns

        n, W = 100, 30
        rng = np.random.RandomState(2)
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        eps = residualize_returns(alt, btc, window=W)
        assert np.all(np.isnan(eps[:W]))
        assert np.all(np.isfinite(eps[W:]))

    def test_strict_causality_no_lookahead(self):
        """ε[t] must depend on alt[t], btc[t], and the past — but not on
        any value at indices > t. Mutating future values of alt or btc must
        not change any ε[i] for i ≤ t."""
        from tools.btc_residualize import residualize_returns

        rng = np.random.RandomState(3)
        n, W = 200, 40
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        eps_before = residualize_returns(alt, btc, window=W)
        t = 120
        alt2 = alt.copy()
        btc2 = btc.copy()
        alt2[t + 1 :] = 999.0
        btc2[t + 1 :] = -999.0
        eps_after = residualize_returns(alt2, btc2, window=W)
        for i in range(W, t + 1):
            assert eps_after[i] == pytest.approx(eps_before[i], abs=1e-12), (
                f"ε[{i}] changed after mutating index {t + 1}+ — lookahead leak"
            )

    def test_btc_self_residual_is_zero(self):
        """When alt == btc, β=1 for every window and residual=0. This is
        the documented behaviour for the BTC-USD pid itself in the probe
        (#253c spec → 'BTC-USD self → passthrough'); upstream callers can
        choose to skip the residual lookup for that pid, but the helper
        itself returns zeros — never NaN, never raises."""
        from tools.btc_residualize import residualize_returns

        rng = np.random.RandomState(4)
        n, W = 200, 50
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        eps = residualize_returns(btc, btc, window=W)
        assert eps.shape == (n,)
        # Warm-up NaN, then eligible bars zero.
        assert np.all(np.isnan(eps[:W]))
        for v in eps[W:]:
            assert abs(v) < 1e-6

    def test_zero_btc_variance_passes_through_alt(self):
        """When btc has no variance in the window, β=0 (per
        compute_rolling_beta fallback) → residual = alt - 0·btc = alt[t]."""
        from tools.btc_residualize import residualize_returns

        n, W = 100, 30
        alt = np.linspace(0.0, 1.0, n).astype(np.float64)
        btc = np.zeros(n, dtype=np.float64)
        eps = residualize_returns(alt, btc, window=W)
        assert np.all(np.isnan(eps[:W]))
        for i in range(W, n):
            assert eps[i] == pytest.approx(alt[i], abs=1e-12)

    def test_input_shape_preserved(self):
        from tools.btc_residualize import residualize_returns

        n, W = 50, 20
        rng = np.random.RandomState(5)
        btc = rng.normal(0.0, 0.01, n).astype(np.float64)
        alt = rng.normal(0.0, 0.01, n).astype(np.float64)
        eps = residualize_returns(alt, btc, window=W)
        assert eps.shape == (n,)
        assert eps.dtype == np.float64

    def test_mismatched_lengths_raises(self):
        from tools.btc_residualize import residualize_returns

        with pytest.raises(ValueError):
            residualize_returns(
                np.zeros(50, dtype=np.float64),
                np.zeros(60, dtype=np.float64),
                window=20,
            )

    def test_invalid_window_raises(self):
        from tools.btc_residualize import residualize_returns

        arr = np.zeros(50, dtype=np.float64)
        with pytest.raises(ValueError):
            residualize_returns(arr, arr, window=0)
