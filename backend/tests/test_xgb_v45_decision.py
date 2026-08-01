"""Tests for v4.5 indep_thresholds decision rule.

Mirrors the rule in backend/tools/v4_5_horizon_compare.py:138 exactly:
  BUY  when p_up   > thresh_up   AND p_up   >= p_down (tie -> BUY)
  SELL when p_down > thresh_down AND p_down >  p_up  (strict)
"""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from agents.cnn_agent import _indep_thresholds_decision

THRESH = 0.50  # matches horizon_compare default


class TestIndepThresholdsDecision:
    def test_indep_strong_up(self):
        side, strength = _indep_thresholds_decision(0.10, 0.10, 0.80, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.800

    def test_indep_strong_down(self):
        side, strength = _indep_thresholds_decision(0.80, 0.10, 0.10, THRESH, THRESH)
        assert side == "SELL"
        assert strength == 0.800

    def test_indep_neutral_dominant(self):
        side, strength = _indep_thresholds_decision(0.25, 0.50, 0.25, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_both_below_threshold(self):
        side, strength = _indep_thresholds_decision(0.49, 0.02, 0.49, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_p_up_at_exact_threshold_holds(self):
        # Strict-greater rule: p_up == 0.50 is NOT > 0.50
        side, strength = _indep_thresholds_decision(0.49, 0.01, 0.50, THRESH, THRESH)
        assert side == "HOLD"
        assert strength == 0.0

    def test_indep_buy_wins_tie_when_both_above_threshold(self):
        # Asymmetric tie: BUY's >= accepts, SELL's > rejects
        side, strength = _indep_thresholds_decision(0.51, 0.00, 0.51, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.510

    def test_indep_buy_when_p_up_marginally_exceeds(self):
        side, strength = _indep_thresholds_decision(0.49, 0.00, 0.51, THRESH, THRESH)
        assert side == "BUY"
        assert strength == 0.510

    def test_indep_sell_when_p_down_marginally_exceeds(self):
        side, strength = _indep_thresholds_decision(0.51, 0.00, 0.49, THRESH, THRESH)
        assert side == "SELL"
        assert strength == 0.510
