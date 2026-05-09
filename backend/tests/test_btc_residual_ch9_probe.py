"""Tests for tools/btc_residual_ch9_probe.py — Ch 9 β-residual single-add probe.

The probe runner is thin glue around three already-tested pieces:
- log_returns / align_pair / build_leadlag_signal (from btc_leadlag_probe — covered)
- residualize_returns (from btc_residualize — covered in test_btc_residualize.py)
- run_replace (from channel_replace — covered in #147)

These tests pin the integration contract: target channel, β window,
BTC passthrough behaviour, and that the probe does in fact route through
`tools.btc_residualize.residualize_returns` (not a re-implementation).
"""
import os
import sys
from unittest.mock import patch

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


# ── Constants pinned by spec (#253a) ───────────────────────────────────────

class TestProbeConstants:

    def test_target_channel_is_ch9(self):
        """Spec §'Target channel': Ch 9 (1-bar price change) only.
        Wrong channel = wrong probe."""
        from tools.btc_residual_ch9_probe import _TARGET_CHANNEL
        assert _TARGET_CHANNEL == 9

    def test_beta_window_24h(self):
        """Spec §'β window': 24h of context. On the hourly grid this is
        W=24. (Spec said '288 bars (24h on 5m candles)'; the parquet
        history this probe consumes is hourly per #246 convention, so 24h
        maps to W=24 here. Documented in the probe header.)"""
        from tools.btc_residual_ch9_probe import _BETA_WINDOW
        assert _BETA_WINDOW == 24

    def test_btc_pid_constant_matches_leadlag(self):
        """The BTC-skip rule must use the same pid string as the leadlag
        probe — otherwise the residualize-self-passthrough never fires."""
        from tools.btc_residual_ch9_probe import _BTC_PID
        from tools.btc_leadlag_probe import _BTC_PID as LEADLAG_BTC
        assert _BTC_PID == LEADLAG_BTC == "BTC-USD"


# ── _build_residual_signal_for_pid: per-pid {ts: ε_t} dict ────────────────

class TestBuildResidualSignalForPid:

    def test_btc_pid_returns_empty(self):
        """BTC vs itself residualizes to ~0 at every bar (β=1, ε=0). The
        signal carries no information, so the probe drops the BTC pid
        rather than emit a flat-zero channel that would skew z-scoring."""
        from tools.btc_residual_ch9_probe import _build_residual_signal_for_pid
        closes = {h * _BAR: 100.0 + h * 0.5 for h in range(100)}
        out = _build_residual_signal_for_pid(
            "BTC-USD", alt_closes=closes, btc_closes=closes,
        )
        assert out == {}

    def test_emits_dict_keyed_by_ts(self):
        from tools.btc_residual_ch9_probe import _build_residual_signal_for_pid
        rng = np.random.RandomState(0)
        btc_closes = {h * _BAR: 100.0 * float(np.exp(np.cumsum(
            rng.normal(0, 0.01, 100))[h])) for h in range(100)}
        # Build alt as a derivative of BTC + idiosyncratic noise.
        idio = rng.normal(0, 0.005, 100)
        alt_closes = {
            h * _BAR: btc_closes[h * _BAR] * float(np.exp(idio[h]))
            for h in range(100)
        }
        out = _build_residual_signal_for_pid(
            "ETH-USD", alt_closes=alt_closes, btc_closes=btc_closes,
        )
        assert isinstance(out, dict)
        # Keys are ints (epoch seconds), values are floats.
        for k, v in out.items():
            assert isinstance(k, int)
            assert isinstance(v, float)
            assert np.isfinite(v)

    def test_warmup_excluded_from_output(self):
        """First `_BETA_WINDOW` aligned bars produce NaN β → must be filtered
        out before the dict is returned (downstream z-scoring would propagate
        NaN otherwise)."""
        from tools.btc_residual_ch9_probe import (
            _build_residual_signal_for_pid, _BETA_WINDOW,
        )
        closes_btc = {h * _BAR: 100.0 + h for h in range(50)}
        closes_alt = {h * _BAR: 50.0 + 2 * h for h in range(50)}
        out = _build_residual_signal_for_pid(
            "ETH-USD", alt_closes=closes_alt, btc_closes=closes_btc,
        )
        # Aligned hourly returns start at h=1 (50 closes -> 49 returns at
        # ts = h*_BAR for h in 1..49). Output keys are a subset of those,
        # excluding the first _BETA_WINDOW warm-up entries.
        assert all(np.isfinite(v) for v in out.values())
        # Number of finite outputs is at most (49 - _BETA_WINDOW).
        assert len(out) <= 49 - _BETA_WINDOW

    def test_routes_through_residualize_returns(self):
        """The probe MUST call residualize_returns (the tested helper),
        not re-implement the math. Patching the helper to return zeros
        should produce an all-zero residual dict."""
        from tools.btc_residual_ch9_probe import _build_residual_signal_for_pid
        closes_btc = {h * _BAR: 100.0 + h for h in range(80)}
        closes_alt = {h * _BAR: 50.0 + 2 * h for h in range(80)}
        with patch(
            "tools.btc_residual_ch9_probe.residualize_returns",
            return_value=np.zeros(79, dtype=np.float64),
        ) as mock_resid:
            out = _build_residual_signal_for_pid(
                "ETH-USD", alt_closes=closes_alt, btc_closes=closes_btc,
            )
        # Helper MUST have been called.
        assert mock_resid.called
        # Output dict values are all 0.0 (mock returned zeros).
        assert all(v == 0.0 for v in out.values())

    def test_short_history_returns_empty(self):
        """If the aligned history is shorter than the β window, no eligible
        bar exists → output is empty (not a partial dict, not a crash)."""
        from tools.btc_residual_ch9_probe import (
            _build_residual_signal_for_pid, _BETA_WINDOW,
        )
        # Only 5 bars — far less than _BETA_WINDOW.
        n = 5
        closes_btc = {h * _BAR: 100.0 + h for h in range(n)}
        closes_alt = {h * _BAR: 50.0 + 2 * h for h in range(n)}
        out = _build_residual_signal_for_pid(
            "ETH-USD", alt_closes=closes_alt, btc_closes=closes_btc,
        )
        assert out == {}

    def test_residual_strips_btc_component(self):
        """alt_close = β·btc_close (multiplicatively), with a small
        idiosyncratic component → residual should be dominated by the idio
        piece, not the BTC piece. Strong correlation between residual and
        idio_returns confirms the math is wired correctly."""
        from tools.btc_residual_ch9_probe import _build_residual_signal_for_pid
        rng = np.random.RandomState(1)
        n = 300
        btc_log_rets = rng.normal(0, 0.01, n)
        idio = rng.normal(0, 0.005, n)
        # alt = exp(1.5 · btc_log_ret + idio)  → log_ret_alt = 1.5·log_ret_btc + idio
        btc_levels = 100.0 * np.exp(np.cumsum(btc_log_rets))
        alt_log_rets = 1.5 * btc_log_rets + idio
        alt_levels = 100.0 * np.exp(np.cumsum(alt_log_rets))
        btc_closes = {h * _BAR: float(btc_levels[h]) for h in range(n)}
        alt_closes = {h * _BAR: float(alt_levels[h]) for h in range(n)}

        out = _build_residual_signal_for_pid(
            "ETH-USD", alt_closes=alt_closes, btc_closes=btc_closes,
        )
        assert len(out) > 50   # substantial coverage

        # Compare residual against idio at the matching ts.
        # idio[h] is the per-bar idiosyncratic component at ts = h*_BAR.
        ts_sorted = sorted(out.keys())
        residuals = np.array([out[t] for t in ts_sorted])
        idio_eligible = np.array([idio[t // _BAR] for t in ts_sorted])
        corr = float(np.corrcoef(residuals, idio_eligible)[0, 1])
        assert corr > 0.7
