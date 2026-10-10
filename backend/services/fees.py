"""Paper-fee arithmetic for net-of-fee trade accounting.

The paper book fills at the last-trade price and charges no fee, so gross `trades.pnl` overstates
what the same trades would have earned. These rates are the account's Coinbase US Intro tier,
verified 2026-10-03 via read-only GET /api/v3/brokerage/transaction_summary. Paper fills are
market fills, so the taker rate is the default.

Accounting only: nothing here changes sizing, exits or the paper balance.
"""

from __future__ import annotations

PAPER_TAKER_FEE_RATE: float = 0.009
PAPER_MAKER_FEE_RATE: float = 0.005


def round_trip_fee(usd_open: float, usd_close: float, rate: float) -> float:
    """Dollar fee for one entry and one exit, each charged on its own notional."""
    if usd_open < 0 or usd_close < 0:
        raise ValueError(f"notional must be non-negative: open={usd_open} close={usd_close}")
    if not 0.0 <= rate < 1.0:
        raise ValueError(f"fee rate must be in [0, 1): {rate}")
    return (usd_open + usd_close) * rate
