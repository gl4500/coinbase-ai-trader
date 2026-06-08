"""Tests for backend/agents/exit_thresholds.py (B2 design).

Design: gain-proportional trail (Layer 1) + capital-relative dollar-cap
(Layer 2). Model-driven exit (p_down > threshold) lives outside this
module — tested in test_cnn_risk_exits.py + test_exit_watcher.py.
"""
from __future__ import annotations
import os
import sys

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from agents.exit_thresholds import (
    FEE_RATE,
    GIVEBACK_FRAC,
    LARGE_POSITION_FRAC,
    LARGE_POSITION_FLOOR,
    MAX_DOLLAR_GIVEBACK_FRAC,
    MAX_LOSS_FRAC_OF_CAPITAL,
    _proportional_giveback,
    _large_position_threshold,
    _dollar_cap_floor,
    _compute_exit_threshold,
)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: _proportional_giveback (gain-proportional trail)
# ═══════════════════════════════════════════════════════════════════════════

def test_proportional_giveback_returns_zero_when_never_green():
    # peak_pnl <= 0 means position never went above entry → no trail engagement
    assert _proportional_giveback(peak_pnl_pct=0.0) == 0.0
    assert _proportional_giveback(peak_pnl_pct=-0.01) == 0.0
    assert _proportional_giveback(peak_pnl_pct=-0.10) == 0.0


def test_proportional_giveback_fee_floor_dominates_at_small_peaks():
    # At peak_pnl 5%, 10%*5% = 0.5%; fee floor (1.2%) wins
    assert _proportional_giveback(peak_pnl_pct=0.05) == pytest.approx(2 * FEE_RATE)
    assert _proportional_giveback(peak_pnl_pct=0.10) == pytest.approx(2 * FEE_RATE)


def test_proportional_giveback_proportional_wins_at_large_peaks():
    # At peak_pnl 15%, 10%*15% = 1.5%; that beats fee floor 1.2%
    assert _proportional_giveback(peak_pnl_pct=0.15) == pytest.approx(0.015)
    # At peak_pnl 50%, 10%*50% = 5%; way above fee floor
    assert _proportional_giveback(peak_pnl_pct=0.50) == pytest.approx(0.050)


def test_proportional_giveback_crossover_at_12pct_peak():
    # 10%*12% = 1.2% — exactly equals fee floor
    assert _proportional_giveback(peak_pnl_pct=0.12) == pytest.approx(2 * FEE_RATE)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: _large_position_threshold
# ═══════════════════════════════════════════════════════════════════════════

def test_large_position_threshold_floor_at_small_capital():
    assert _large_position_threshold(total_capital=100.0) == LARGE_POSITION_FLOOR
    assert _large_position_threshold(total_capital=1000.0) == LARGE_POSITION_FLOOR
    assert _large_position_threshold(total_capital=3999.0) == LARGE_POSITION_FLOOR


def test_large_position_threshold_scales_above_floor():
    assert _large_position_threshold(total_capital=4000.0) == pytest.approx(200.0)
    assert _large_position_threshold(total_capital=10_000.0) == pytest.approx(500.0)
    assert _large_position_threshold(total_capital=100_000.0) == pytest.approx(5000.0)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: _dollar_cap_floor
# ═══════════════════════════════════════════════════════════════════════════

def test_dollar_cap_floor_skips_small_positions():
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=50.0,
                             total_capital=1000.0) is None
    # Exactly at threshold — not strictly > → skip
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=200.0,
                             total_capital=1000.0) is None


def test_dollar_cap_floor_engages_for_large_positions():
    # position $250 at capital $1k: threshold=$200, position > threshold
    # pct_cap = 2%, cap_cap = $5/$250 = 2%, min = 2%
    # exit threshold = 0.05 - 0.02 = 0.03
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=250.0,
                             total_capital=1000.0) == pytest.approx(0.03)


def test_dollar_cap_floor_capital_loss_cap_tighter_at_concentrated_position():
    # position $500 / $1k capital = 50% concentration
    # pct_cap = 2%, cap_cap = $5/$500 = 1%, min = 1% (capital wins)
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=1000.0) == pytest.approx(0.04)


def test_dollar_cap_floor_threshold_scales_with_capital():
    # Same $500 position; engagement varies with capital level
    # capital $1k:   threshold=$200 → engages
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=1000.0) is not None
    # capital $10k:  threshold=$500, position EQ threshold → skips
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=10_000.0) is None
    # capital $100k: threshold=$5k, position $500 << threshold → skips
    assert _dollar_cap_floor(peak_pnl_pct=0.05, position_dollars=500.0,
                             total_capital=100_000.0) is None


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator: _compute_exit_threshold
# ═══════════════════════════════════════════════════════════════════════════

def test_compute_exit_threshold_returns_neg_inf_when_never_green():
    # No protection until position has been green at some point
    result = _compute_exit_threshold(peak_pnl_pct=0.0, atr_pct=0.06)
    assert result == float("-inf")
    result = _compute_exit_threshold(peak_pnl_pct=-0.03, atr_pct=0.06)
    assert result == float("-inf")


def test_compute_exit_threshold_fee_floor_at_small_peak():
    # peak +5%, giveback = max(0.5%, 1.2%) = 1.2%
    # threshold = 5% - 1.2% = 3.8%
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06)
    assert result == pytest.approx(0.038)


def test_compute_exit_threshold_proportional_at_large_peak():
    # peak +20%, giveback = max(2.0%, 1.2%) = 2.0% (proportional wins)
    # threshold = 20% - 2.0% = 18.0%
    result = _compute_exit_threshold(peak_pnl_pct=0.20, atr_pct=0.06)
    assert result == pytest.approx(0.18)


def test_compute_exit_threshold_layer2_tightens_over_layer1():
    # peak +5%, atr unused, position $500, capital $1k
    # Layer 1: max(0.5%, 1.2%) = 1.2% giveback → threshold = 5% - 1.2% = 3.8%
    # Layer 2: pct_cap=2%, cap_cap=$5/$500=1%; min=1%; floor = 5% - 1% = 4%
    # max(3.8%, 4.0%) = 4.0% → Layer 2 wins (tightens)
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                     position_dollars=500.0, total_capital=1000.0)
    assert result == pytest.approx(0.04)


def test_compute_exit_threshold_layer2_skipped_when_capital_omitted():
    # No total_capital → Layer 2 skipped
    # Just Layer 1: 5% - 1.2% = 3.8%
    result = _compute_exit_threshold(peak_pnl_pct=0.05, atr_pct=0.06,
                                     position_dollars=500.0)
    assert result == pytest.approx(0.038)


def test_compute_exit_threshold_worked_example_dash():
    # DASH-like: peak_pnl 7.8%, pos $58, capital $1k
    # Layer 1: max(0.78%, 1.2%) = 1.2% giveback → 7.8% - 1.2% = 6.6%
    # Layer 2: position $58 < threshold $200 → skip
    # exit at +6.6% PnL
    result = _compute_exit_threshold(peak_pnl_pct=0.078, atr_pct=0.06,
                                     position_dollars=58.0, total_capital=1000.0)
    assert result == pytest.approx(0.066)


def test_compute_exit_threshold_worked_example_paxg_never_green():
    # PAXG: peak_pnl=0 (entry == peak)
    # → -infinity (only stop_loss fires)
    result = _compute_exit_threshold(peak_pnl_pct=0.0, atr_pct=0.04,
                                     position_dollars=452.0, total_capital=1000.0)
    assert result == float("-inf")


def test_compute_exit_threshold_eth_locks_meaningful_gain():
    # ETH-like: peak 4.3%, pos $103, capital $1k
    # Layer 1: max(0.43%, 1.2%) = 1.2% → 4.3% - 1.2% = 3.1%
    # Layer 2: position $103 < threshold $200 → skip
    # exit at +3.1% PnL — protects most of the peak gain
    result = _compute_exit_threshold(peak_pnl_pct=0.043, atr_pct=0.06,
                                     position_dollars=103.0, total_capital=1000.0)
    assert result == pytest.approx(0.031)
