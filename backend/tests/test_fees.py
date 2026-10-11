"""Tests for services/fees.py — pure paper-fee arithmetic."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.fees import (  # noqa: E402
    PAPER_MAKER_FEE_RATE,
    PAPER_TAKER_FEE_RATE,
    round_trip_fee,
)


def test_rates_match_verified_intro_tier():
    # Verified 2026-10-03 via read-only GET /api/v3/brokerage/transaction_summary.
    assert PAPER_TAKER_FEE_RATE == 0.009
    assert PAPER_MAKER_FEE_RATE == 0.005


def test_round_trip_fee_charges_each_side_on_its_own_notional():
    # $100 in, $110 out, 0.9% per side: 0.90 + 0.99
    assert round_trip_fee(100.0, 110.0, 0.009) == pytest.approx(1.89)


def test_round_trip_fee_zero_rate_is_zero():
    assert round_trip_fee(100.0, 90.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "usd_open,usd_close,rate",
    [(-1.0, 10.0, 0.009), (10.0, -1.0, 0.009), (10.0, 10.0, -0.001), (10.0, 10.0, 1.0)],
)
def test_round_trip_fee_rejects_impossible_inputs(usd_open, usd_close, rate):
    with pytest.raises(ValueError):
        round_trip_fee(usd_open, usd_close, rate)
