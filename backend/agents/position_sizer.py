"""
Kelly Criterion Position Sizer
───────────────────────────────
Implements fractional Kelly for prediction market binary outcomes.

Full Kelly formula for prediction markets:
  f* = (p - q) / (b)
where:
  p = model probability of YES
  q = 1 - p
  b = odds (1/price - 1)  — net profit per $1 wagered if correct

Simplified for convenience:
  f* = (model_prob - market_price) / (1 - market_price)

We use Quarter Kelly (25%) by default to reduce variance dramatically:
  - Full Kelly:    33% chance of halving bankroll
  - Quarter Kelly: 4%  chance of halving bankroll
"""
import logging
from typing import Dict, Optional

from config import config

logger = logging.getLogger(__name__)


class PositionSizer:
    def __init__(self,
                 kelly_fraction: float = None,
                 max_position_usdc: float = None,
                 max_total_exposure: float = None,
                 min_edge: float = None):
        self.kelly_fraction     = kelly_fraction or config.kelly_fraction
        self.max_position_usdc  = max_position_usdc or config.max_position_usdc
        self.max_total_exposure = max_total_exposure or config.max_total_exposure
        self.min_edge           = min_edge or config.min_edge

    def kelly_size(
        self,
        model_prob: float,
        market_price: float,
        bankroll: float,
        current_exposure: float = 0.0,
    ) -> Dict:
        """
        Compute fractional Kelly position size.

        Returns a dict with:
          recommended_usdc: float — position size in USDC
          full_kelly_pct:   float — full Kelly fraction
          frac_kelly_pct:   float — fractional Kelly fraction
          edge:             float — model_prob - market_price
          skip_reason:      str | None — reason if we should not trade
        """
        edge = model_prob - market_price

        # Reject below minimum edge threshold
        if abs(edge) < self.min_edge:
            return {
                "recommended_usdc": 0.0,
                "full_kelly_pct": 0.0,
                "frac_kelly_pct": 0.0,
                "edge": round(edge, 4),
                "skip_reason": f"Edge {edge:.3f} below min {self.min_edge}",
            }

        # Reject if market price is at extremes (near 0 or 1 — Kelly breaks down)
        if market_price <= 0.01 or market_price >= 0.99:
            return {
                "recommended_usdc": 0.0,
                "full_kelly_pct": 0.0,
                "frac_kelly_pct": 0.0,
                "edge": round(edge, 4),
                "skip_reason": f"Market price {market_price} too extreme for Kelly",
            }

        # Full Kelly fraction
        full_kelly = edge / (1.0 - market_price)

        # Negative Kelly = don't trade in that direction
        if full_kelly <= 0:
            return {
                "recommended_usdc": 0.0,
                "full_kelly_pct": round(full_kelly, 4),
                "frac_kelly_pct": 0.0,
                "edge": round(edge, 4),
                "skip_reason": "Negative Kelly — model and direction disagree",
            }

        frac_kelly = full_kelly * self.kelly_fraction
        raw_size   = frac_kelly * bankroll

        # Hard caps
        remaining_capacity = max(0.0, self.max_total_exposure - current_exposure)
        capped_size = min(raw_size, self.max_position_usdc, remaining_capacity)

        skip_reason = None
        if capped_size < 1.0:
            skip_reason = "Position size below $1 minimum after caps"

        return {
            "recommended_usdc": round(max(0.0, capped_size), 2),
            "full_kelly_pct":   round(full_kelly, 4),
            "frac_kelly_pct":   round(frac_kelly, 4),
            "edge":             round(edge, 4),
            "skip_reason":      skip_reason,
        }

    def arb_size(
        self,
        yes_price: float,
        no_price: float,
        bankroll: float,
        current_exposure: float = 0.0,
    ) -> Dict:
        """
        Position size for a bundle arb trade (buy both YES and NO).
        Risk is effectively zero (locked-in profit), so we can be more aggressive.
        Size = min(max_position * 2, bankroll * 0.10, remaining_capacity)
        """
        sum_price = yes_price + no_price
        discount  = 1.0 - sum_price
        roi_pct   = (discount / sum_price) * 100 if sum_price > 0 else 0

        # For arb, use 10% of bankroll capped at 2x normal max
        aggressive_max = min(self.max_position_usdc * 2, bankroll * 0.10)
        remaining_capacity = max(0.0, self.max_total_exposure - current_exposure)
        size = min(aggressive_max, remaining_capacity)

        return {
            "recommended_usdc": round(size, 2),   # per leg (buy both YES and NO)
            "yes_leg_usdc":     round(size * yes_price / sum_price, 2),
            "no_leg_usdc":      round(size * no_price  / sum_price, 2),
            "roi_pct":          round(roi_pct, 2),
            "locked_profit":    round(discount * size, 2),
        }
