"""
Unit tests for the Kelly Criterion position sizer.
These are pure-math tests — no DB, no network, no mocking needed.
"""
import pytest
from agents.position_sizer import PositionSizer


@pytest.fixture
def sizer():
    return PositionSizer(
        kelly_fraction=0.25,
        max_position_usdc=500.0,
        max_total_exposure=5000.0,
        min_edge=0.08,
    )


# ── Edge filtering ────────────────────────────────────────────────────────────

class TestEdgeFilter:
    def test_below_min_edge_returns_zero(self, sizer):
        result = sizer.kelly_size(model_prob=0.55, market_price=0.50, bankroll=1000)
        assert result["recommended_usdc"] == 0.0
        assert result["skip_reason"] is not None

    def test_at_min_edge_threshold_trades(self, sizer):
        # Edge = 0.10 (clearly above min_edge=0.08), market_price = 0.50
        result = sizer.kelly_size(model_prob=0.60, market_price=0.50, bankroll=1000)
        assert result["recommended_usdc"] > 0.0
        assert result["skip_reason"] is None

    def test_negative_edge_no_trade(self, sizer):
        # Model says 0.40 but market says 0.60 → edge is −0.20 (wrong direction)
        result = sizer.kelly_size(model_prob=0.40, market_price=0.60, bankroll=1000)
        assert result["recommended_usdc"] == 0.0

    def test_extreme_market_price_near_zero_skipped(self, sizer):
        result = sizer.kelly_size(model_prob=0.99, market_price=0.005, bankroll=1000)
        assert result["recommended_usdc"] == 0.0

    def test_extreme_market_price_near_one_skipped(self, sizer):
        result = sizer.kelly_size(model_prob=0.90, market_price=0.995, bankroll=1000)
        assert result["recommended_usdc"] == 0.0


# ── Kelly formula correctness ─────────────────────────────────────────────────

class TestKellyFormula:
    def test_full_kelly_calculation(self, sizer):
        # edge = 0.70 - 0.50 = 0.20; full_kelly = 0.20 / 0.50 = 0.40
        result = sizer.kelly_size(model_prob=0.70, market_price=0.50, bankroll=1000)
        assert abs(result["full_kelly_pct"] - 0.40) < 0.001

    def test_fractional_kelly_is_quarter_of_full(self, sizer):
        result = sizer.kelly_size(model_prob=0.70, market_price=0.50, bankroll=1000)
        assert abs(result["frac_kelly_pct"] - result["full_kelly_pct"] * 0.25) < 0.001

    def test_edge_reported_correctly(self, sizer):
        result = sizer.kelly_size(model_prob=0.70, market_price=0.50, bankroll=1000)
        assert abs(result["edge"] - 0.20) < 0.001

    def test_size_scales_with_bankroll(self, sizer):
        r1 = sizer.kelly_size(model_prob=0.70, market_price=0.50, bankroll=1000)
        r2 = sizer.kelly_size(model_prob=0.70, market_price=0.50, bankroll=2000)
        assert r2["recommended_usdc"] > r1["recommended_usdc"]


# ── Position caps ─────────────────────────────────────────────────────────────

class TestPositionCaps:
    def test_max_position_usdc_cap(self, sizer):
        # Very large bankroll should still cap at max_position_usdc=500
        result = sizer.kelly_size(model_prob=0.90, market_price=0.50, bankroll=1_000_000)
        assert result["recommended_usdc"] <= 500.0

    def test_exposure_cap_limits_trade(self, sizer):
        # Already at max exposure → no room left
        result = sizer.kelly_size(
            model_prob=0.80, market_price=0.50, bankroll=1000,
            current_exposure=5000.0,
        )
        assert result["recommended_usdc"] == 0.0

    def test_partial_exposure_reduces_size(self, sizer):
        full = sizer.kelly_size(model_prob=0.80, market_price=0.50, bankroll=10000, current_exposure=0)
        partial = sizer.kelly_size(model_prob=0.80, market_price=0.50, bankroll=10000, current_exposure=4000)
        assert partial["recommended_usdc"] <= full["recommended_usdc"]


# ── Arb sizer ─────────────────────────────────────────────────────────────────

class TestArbSizer:
    def test_arb_size_positive(self, sizer):
        result = sizer.arb_size(yes_price=0.48, no_price=0.48, bankroll=1000)
        assert result["recommended_usdc"] > 0.0

    def test_arb_roi_computed(self, sizer):
        # sum = 0.96, discount = 0.04, roi = 0.04/0.96 * 100 ≈ 4.17%
        result = sizer.arb_size(yes_price=0.48, no_price=0.48, bankroll=10000)
        assert abs(result["roi_pct"] - 4.17) < 0.1

    def test_arb_legs_sum_to_total(self, sizer):
        result = sizer.arb_size(yes_price=0.45, no_price=0.50, bankroll=10000)
        # yes_leg + no_leg should equal recommended_usdc (within rounding)
        assert abs(result["yes_leg_usdc"] + result["no_leg_usdc"] - result["recommended_usdc"]) < 0.02

    def test_arb_capped_at_exposure_limit(self, sizer):
        result = sizer.arb_size(yes_price=0.45, no_price=0.50, bankroll=10000, current_exposure=5000.0)
        assert result["recommended_usdc"] == 0.0
