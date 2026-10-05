"""Dated-futures carry check (hold spot, short the dated future to expiry). Pure: no I/O.

Uses EXECUTABLE sides: buy spot at the ask, sell the future at the bid. Capital per contract =
spot outlay incl. maker fee + short margin on FUTURES notional + a rally reserve
(r * futures notional * (1 + margin)) + futures fees both sides. The premium must beat all
costs AND the cash yield forgone on that whole capital. The spot exit fee at expiry is estimated
at the entry price. Fee and cash-rate defaults are illustrative assumptions (Coinbase's
advertised futures minimum; a T-bill-like rate), not this account's verified schedule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CarryParams:
    capital: float = 1000.0  # operator's current capital; update as contributions grow
    spot_maker_fee: float = 0.005  # Coinbase Intro maker (verified 2026-10-03)
    fut_fee_rate: float = 0.0002  # advertised "as low as 0.02%" — ASSUMPTION
    fut_fee_min: float = 0.15  # advertised $0.15 minimum per contract — ASSUMPTION
    cash_rate: float = 0.0417  # ~3m Treasury CMT, Oct 2026 — ASSUMPTION
    reserve_rally: float = 0.30  # illustrative stress reserve, not a loss bound
    min_excess_annual: float = 0.01  # alert only if >= 1%/yr above cash on all capital
    min_days: float = 7.0  # ignore contracts about to expire


def fut_fee(notional: float, p: CarryParams) -> float:
    return max(p.fut_fee_rate * notional, p.fut_fee_min)


def evaluate_dated(
    *,
    spot_ask: float,
    fut_bid: float,
    contract_size: float,
    short_margin: float,
    days_to_expiry: float,
    p: CarryParams,
) -> dict:
    vals = (spot_ask, fut_bid, contract_size, short_margin, days_to_expiry)
    valid = (
        all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
        and min(spot_ask, fut_bid, contract_size, days_to_expiry) > 0
    )
    if not valid:
        return {"valid": False, "alert": False}
    ns, nf = contract_size * spot_ask, contract_size * fut_bid
    capital = ns * (1 + p.spot_maker_fee) + short_margin * nf
    capital += p.reserve_rally * nf * (1 + short_margin) + 2 * fut_fee(nf, p)
    premium = (fut_bid - spot_ask) * contract_size
    costs = 2 * p.spot_maker_fee * ns + 2 * fut_fee(nf, p)
    forgone = capital * p.cash_rate * days_to_expiry / 365
    excess = premium - costs - forgone
    excess_annual = excess / capital * 365 / days_to_expiry
    contracts = int(p.capital // capital)
    alert = contracts >= 1 and days_to_expiry >= p.min_days and excess_annual >= p.min_excess_annual
    return {
        "valid": True,
        "capital_per_contract": capital,
        "contracts_affordable": contracts,
        "premium": premium,
        "basis_annual": (fut_bid / spot_ask - 1) * 365 / days_to_expiry,
        "costs": costs,
        "cash_forgone": forgone,
        "excess": excess,
        "excess_annual": excess_annual,
        "alert": bool(alert),
    }
