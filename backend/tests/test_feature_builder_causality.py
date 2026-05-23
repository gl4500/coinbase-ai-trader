"""Causality tests for FeatureBuilder.build.

Property under test: for any non-masked channel c and any candle index k,
the channel value at candle k must depend ONLY on candles[0..k]. Equivalently:
building features twice — once on the full series and once on candles[:k+1] —
must produce the same channel-c value at candle k.

This is the lookahead test from docs/crypto_feature_engineering_pipeline.md §4
("If you do nothing else from this guide, build the lookahead test").

Currently expected RED on Ch 15 (ADX): cnn_agent.py:1306-1307 computes
adx_val once on the full candle series and broadcasts that single value
across all 60 timesteps, so windowed-vs-full ADX values diverge.
"""
import math
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

from agents.cnn_agent import FeatureBuilder, SEQ_LEN  # noqa: E402


def _sinusoidal_candles(n: int, start: float = 50_000.0) -> list:
    """Sinusoidal OHLCV — varied highs/lows so ADX is non-degenerate."""
    out = []
    for i in range(n):
        c = start + 500.0 * math.sin(i / 5.0)
        out.append({
            "open":   c - 50.0,
            "high":   c + 100.0,
            "low":    c - 100.0,
            "close":  c,
            "volume": 10_000.0 + i * 100.0,
            "start":  1_700_000_000 + i * 3600,
        })
    return out


def _ch_value_for_candle(channels, ch_idx: int, candle_k: int, n_total: int):
    """Map candle index k → output position in a build call with n_total candles.

    FeatureBuilder pads the per-bar series to length SEQ_LEN by keeping the
    last SEQ_LEN entries. So the value for candle k lands at output position
    SEQ_LEN - 1 - (n_total - 1 - k) = SEQ_LEN - n_total + k.
    Requires n_total >= SEQ_LEN and the candle k to be in the last SEQ_LEN
    bars (else it was padded out).
    """
    pos = SEQ_LEN - n_total + candle_k
    assert 0 <= pos < SEQ_LEN, (
        f"candle {candle_k} not in window for n_total={n_total} (pos={pos})"
    )
    return channels[ch_idx][pos]


class TestCh15ADXCausality:
    """Ch 15 (ADX regime) — currently broadcasts a single value computed on
    the full window; should be per-bar causal."""

    def setup_method(self):
        self.fb = FeatureBuilder()

    def test_ch15_terminal_value_independent_of_future_bars(self):
        """Build on candles[:80] and candles[:120]; Ch 15 value for the
        terminal candle of the shorter series (candle 79) must be identical
        in both calls."""
        full = _sinusoidal_candles(120)
        window = full[:80]

        ch_full = self.fb.build(full, {})
        ch_window = self.fb.build(window, {})

        full_val = _ch_value_for_candle(ch_full, 15, candle_k=79, n_total=120)
        window_val = _ch_value_for_candle(ch_window, 15, candle_k=79, n_total=80)

        assert full_val == pytest.approx(window_val, abs=1e-9), (
            f"Ch 15 leaks future info: full[candle 79]={full_val}, "
            f"window[candle 79]={window_val}. Diff={full_val - window_val:.6g}"
        )

    def test_ch15_varies_across_timesteps(self):
        """A per-bar causal ADX should produce different values for
        materially different bars within a single build call. The current
        broadcast impl produces a constant series — test fails until fixed."""
        candles = _sinusoidal_candles(120)
        channels = self.fb.build(candles, {})

        ch15 = channels[15]
        assert len(set(round(v, 6) for v in ch15)) > 1, (
            f"Ch 15 is constant across all {SEQ_LEN} timesteps — broadcast "
            f"impl detected. Sample={ch15[0]}, all values equal."
        )


# Channels exempted from the strict windowed-equality property below.
# - 17, 18, 19: 5m fast channels — broadcast by design AND in
#   _TRAINING_CONSTANT_CHANNELS (zeroed at both train and infer per CLAUDE.md
#   invariant 11), so any leakage they contain never reaches the model.
# - 20: funding rate — scalar broadcast across timesteps; not derived from
#   the candle series, so the windowed-equality test does not apply.
# - 21: BTC return correlation — depends on the optional `btc_closes` kwarg,
#   not the candles list under test. Tested separately when that input
#   is wired.
# - 24, 25: realized-vol windows — when called without `closes_ext`, fall
#   back to the input candles. Caller-provided `closes_ext` is what gives
#   them per-bar causality across sample boundaries; the harness here builds
#   without `closes_ext` so leakage there is a known fallback artifact, not
#   a production bug. Covered by dedicated tests in TestRvPrefixLookback.
_LOOKAHEAD_EXEMPT = frozenset({17, 18, 19, 20, 21, 24, 25})


class TestAllChannelsLookahead:
    """Generic windowed-equality lookahead harness for all 27 FeatureBuilder
    channels (#161, post-#157 follow-up).

    Property: for any non-exempt channel c and any candle k inside the
    SEQ_LEN window of both builds, the channel value at candle k must be
    identical when built on candles[:80] vs candles[:120]. Failure
    indicates the channel uses bars > k to compute the value at k —
    a strict-causality leak (the same class as the Ch 15 ADX broadcast).

    Exempt channels (see _LOOKAHEAD_EXEMPT above) are skipped with a clear
    reason. Any RED here goes onto the per-channel fix backlog.
    """

    CANDLE_K = 79  # terminal of the shorter (80-bar) series
    N_FULL = 120
    N_WINDOW = 80

    def setup_method(self):
        self.fb = FeatureBuilder()
        full = _sinusoidal_candles(self.N_FULL)
        self.ch_full = self.fb.build(full, {})
        self.ch_window = self.fb.build(full[: self.N_WINDOW], {})

    # Channels with known strict-causality issues, kept as xfail with an
    # explicit reason and the task ID for the fix decision. New failures
    # should NOT land here without a corresponding task — see #171.
    _KNOWN_LEAKY = {
        0: "Ch 0 norm_c uses min/max(closes) over full input window — "
           "every position's normalized value depends on the global min/max "
           "across the entire input. Strict-causality leak. See task #171 "
           "for the fix-vs-accept decision (per-bar expanding min/max would "
           "change normalization scale and require retrain).",
    }

    @pytest.mark.parametrize("ch_idx", [c for c in range(27) if c not in _LOOKAHEAD_EXEMPT])
    def test_channel_terminal_value_windowed_equality(self, ch_idx, request):
        if ch_idx in self._KNOWN_LEAKY:
            request.node.add_marker(
                pytest.mark.xfail(
                    reason=self._KNOWN_LEAKY[ch_idx], strict=True
                )
            )
        full_val = _ch_value_for_candle(
            self.ch_full, ch_idx, candle_k=self.CANDLE_K, n_total=self.N_FULL
        )
        window_val = _ch_value_for_candle(
            self.ch_window, ch_idx, candle_k=self.CANDLE_K, n_total=self.N_WINDOW
        )

        assert full_val == pytest.approx(window_val, abs=1e-9), (
            f"Ch {ch_idx} leaks future info: "
            f"full[candle {self.CANDLE_K}]={full_val}, "
            f"window[candle {self.CANDLE_K}]={window_val}. "
            f"Diff={full_val - window_val:.6g}"
        )
