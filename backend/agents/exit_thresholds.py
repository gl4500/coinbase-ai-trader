"""Pure-function exit-threshold helpers for the CNN agent's risk exits.

Used by both the scan-loop (cnn_agent._check_risk_exits) and the WS tick
handler (exit_watcher.on_price_tick) to compute the exit threshold in
PnL-percent terms.

Design (#46, B2 — operator chose gain-proportional + model-driven):

  Layer 1 (PROPORTIONAL TRAIL): exit when current_pnl_pct drops below
    peak_pnl_pct - giveback, where:
      giveback = max(peak_pnl_pct * GIVEBACK_FRAC, 2 * FEE_RATE)
    The giveback is proportional to the peak gain (give back X% of what
    you've earned at peak), with a fee-aware minimum so exits still cover
    Coinbase round-trip fees on small gains.

  Layer 2 (CAPITAL-RELATIVE DOLLAR-CAP): for positions larger than
    max(LARGE_POSITION_FLOOR, total_capital * LARGE_POSITION_FRAC), tighten
    the giveback to the smaller of (MAX_DOLLAR_GIVEBACK_FRAC of position $,
    MAX_LOSS_FRAC_OF_CAPITAL of total capital / position $).

  Model-driven exit (MODEL_DOWN) is NOT in this module — it lives in
  cnn_agent + exit_watcher, which read pos['p_down'] cached by
  generate_signal. They override _compute_exit_threshold whenever
  p_down > P_DOWN_EXIT_THRESHOLD, firing immediately regardless of trail.

Position exits when current_pnl_pct < _compute_exit_threshold(...).
"""
from __future__ import annotations
from typing import Optional


# ── Configuration constants ────────────────────────────────────────────────

FEE_RATE: float                  = 0.006   # Coinbase taker (round-trip basis = 2 * 0.006 = 1.2%)
GIVEBACK_FRAC: float             = 0.10    # give back 10% of peak gain (gain-proportional trail)
LARGE_POSITION_FRAC: float       = 0.05    # 5% of capital = "large" position
LARGE_POSITION_FLOOR: float      = 200.0   # USD; absolute floor for small-capital regime
MAX_DOLLAR_GIVEBACK_FRAC: float  = 0.02    # 2% of position $ for large positions
MAX_LOSS_FRAC_OF_CAPITAL: float  = 0.005   # 0.5% of total capital per trail-fire


__all__ = [
    "FEE_RATE",
    "GIVEBACK_FRAC",
    "LARGE_POSITION_FRAC",
    "LARGE_POSITION_FLOOR",
    "MAX_DOLLAR_GIVEBACK_FRAC",
    "MAX_LOSS_FRAC_OF_CAPITAL",
    "_proportional_giveback",
    "_large_position_threshold",
    "_dollar_cap_floor",
    "_compute_exit_threshold",
]


# ── Layer 1: gain-proportional trail with fee floor ────────────────────────

def _proportional_giveback(peak_pnl_pct: float) -> float:
    """Giveback in PnL percent terms — proportional to peak gain, with fee floor.

    Returns the % giveback (a positive number; subtract from peak_pnl_pct
    to get the exit threshold).

    For peak_pnl_pct <= 0: returns 0.0 (no trail engages until green).

    Examples (with GIVEBACK_FRAC=0.10, FEE_RATE=0.006):
      peak +1%:  giveback = max(0.001, 0.012) = 0.012 (fee floor dominates)
      peak +5%:  giveback = max(0.005, 0.012) = 0.012 (fee floor)
      peak +15%: giveback = max(0.015, 0.012) = 0.015 (proportional wins)
      peak +50%: giveback = max(0.050, 0.012) = 0.050 (proportional)
    """
    if peak_pnl_pct <= 0:
        return 0.0
    return max(peak_pnl_pct * GIVEBACK_FRAC, 2 * FEE_RATE)


# ── Layer 2: Capital-relative dollar-cap on large positions ─────────────────

def _large_position_threshold(total_capital: float) -> float:
    """Capital-relative cutoff for the dollar-cap layer.

    A position is 'large' if it's > LARGE_POSITION_FRAC of total capital,
    with LARGE_POSITION_FLOOR as the absolute minimum so the rule remains
    meaningful at small capital.
    """
    return max(LARGE_POSITION_FLOOR, total_capital * LARGE_POSITION_FRAC)


def _dollar_cap_floor(
    peak_pnl_pct: float,
    position_dollars: float,
    total_capital: float,
) -> Optional[float]:
    """Capital-relative dollar-cap on large positions.

    Caps giveback at the TIGHTER of:
      - MAX_DOLLAR_GIVEBACK_FRAC of position $ (scale-invariant %-cap)
      - MAX_LOSS_FRAC_OF_CAPITAL of total capital (portfolio-impact based)

    Returns the minimum exit threshold (in PnL terms) or None if the
    position is not 'large' relative to capital.
    """
    threshold = _large_position_threshold(total_capital)
    if position_dollars <= threshold:
        return None
    pct_cap = MAX_DOLLAR_GIVEBACK_FRAC
    cap_cap = (total_capital * MAX_LOSS_FRAC_OF_CAPITAL) / position_dollars
    giveback = min(pct_cap, cap_cap)
    return peak_pnl_pct - giveback


# ── Orchestrator ─────────────────────────────────────────────────────────────

def _compute_exit_threshold(
    *,
    peak_pnl_pct: float,
    atr_pct: float = 0.06,   # accepted for back-compat; not used in B2 design
    position_dollars: Optional[float] = None,
    total_capital: Optional[float] = None,
) -> float:
    """Combined exit threshold in PnL terms.

    Position exits when current_pnl_pct < returned value.

    Combines:
      Layer 1: gain-proportional trail with fee-aware floor
      Layer 2: capital-relative dollar-cap on large positions (requires both
               position_dollars AND total_capital; skipped if either omitted)

    Returns the max of all engaged layers - the TIGHTEST exit threshold.

    When peak_pnl_pct <= 0, returns -float('inf') so only stop_loss fires
    on never-green positions (no trail protection yet — wait for green).
    """
    if peak_pnl_pct <= 0:
        return float("-inf")

    giveback = _proportional_giveback(peak_pnl_pct)
    threshold = peak_pnl_pct - giveback

    if position_dollars and total_capital:
        dollar_floor = _dollar_cap_floor(
            peak_pnl_pct, position_dollars, total_capital,
        )
        if dollar_floor is not None:
            threshold = max(threshold, dollar_floor)

    return threshold
