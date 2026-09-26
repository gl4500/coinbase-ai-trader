import os
import sys

_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import pytest

from services.regime.macro_regime import evaluate, macro_mult, mvrv_prior


class TestMvrvPrior:
    def test_cheap_zone_max_leanin(self):
        assert mvrv_prior(0.5) == 1.25
        assert mvrv_prior(0.8) == 1.25

    def test_extended_zone_trim(self):
        assert mvrv_prior(4.0) == 0.85
        assert mvrv_prior(3.5) == 0.85

    def test_interpolation_midpoints(self):
        assert mvrv_prior(1.15) == pytest.approx(1.15)  # halfway 0.8->1.5 : 1.25->1.05
        assert mvrv_prior(2.25) == pytest.approx(1.00)  # halfway 1.5->3.0 : 1.05->0.95

    def test_none_is_neutral(self):
        assert mvrv_prior(None) == 1.0


class TestMacroMult:
    def test_decoupled_ignores_macro(self):
        # corr 0 -> w=0 -> macro ignored regardless of risk
        assert macro_mult(0.0, -1.0) == 1.0
        assert macro_mult(-0.3, 1.0) == 1.0  # negative corr also gates to 0

    def test_coupled_riskoff_reduces(self):
        assert macro_mult(0.6, -0.8) == pytest.approx(1 - 0.3 * 0.6 * 0.8)

    def test_coupled_riskon_boosts(self):
        assert macro_mult(0.5, 1.0) == pytest.approx(1 + 0.3 * 0.5 * 1.0)

    def test_none_is_neutral(self):
        assert macro_mult(None, 0.5) == 1.0
        assert macro_mult(0.5, None) == 1.0


class TestEvaluate:
    def test_worked_example_2022_coupled_riskoff(self):
        rs = evaluate(date="2022-06-30", mvrv=2.5, corr_spx_90d=0.6, macro_risk_raw=-0.8)
        assert rs.macro_mult == pytest.approx(0.856, abs=1e-3)
        assert rs.exposure_scalar < 1.0
        assert rs.confidence == 1.0

    def test_worked_example_nov2025_decoupled(self):
        rs = evaluate(date="2025-11-30", mvrv=1.0, corr_spx_90d=0.0, macro_risk_raw=-1.0)
        assert rs.macro_mult == 1.0  # macro ignored
        assert rs.exposure_scalar > 1.0  # MVRV prior leads
        assert rs.exposure_scalar == pytest.approx(min(mvrv_prior(1.0), 1.25))

    def test_cycle_bottom_leans_in(self):
        rs = evaluate(date="2018-12-15", mvrv=0.7, corr_spx_90d=0.3, macro_risk_raw=-0.5)
        assert rs.mvrv_prior == 1.25

    def test_clamp_upper(self):
        rs = evaluate(date="2020-01-01", mvrv=0.5, corr_spx_90d=0.6, macro_risk_raw=1.0)
        assert rs.exposure_scalar == 1.25  # 1.25 * 1.18 clamped to 1.25

    def test_clamp_lower(self):
        # force below floor via extreme (defensive) inputs
        rs = evaluate(date="2020-01-01", mvrv=3.5, corr_spx_90d=1.0, macro_risk_raw=-1.0)
        assert rs.exposure_scalar >= 0.4

    def test_all_missing_is_neutral(self):
        rs = evaluate(date="2020-01-01", mvrv=None, corr_spx_90d=None, macro_risk_raw=None)
        assert rs.exposure_scalar == 1.0
        assert rs.confidence == 0.0

    def test_staleness_treated_as_missing(self):
        rs = evaluate(
            date="2020-01-01", mvrv=0.5, corr_spx_90d=0.6, macro_risk_raw=1.0, mvrv_age_days=10
        )
        assert rs.mvrv_prior == 1.0  # stale MVRV -> neutral factor
        assert rs.confidence == pytest.approx(0.5)

    def test_never_raises_on_garbage(self):
        rs = evaluate(
            date="x", mvrv=float("nan"), corr_spx_90d=float("nan"), macro_risk_raw=float("nan")
        )
        assert 0.4 <= rs.exposure_scalar <= 1.25
