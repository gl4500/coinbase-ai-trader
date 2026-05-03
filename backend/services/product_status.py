"""Tiered blacklist evaluator for per-product trading status (#120).

Three-tier ladder driven by per-trade Sharpe over a rolling window of
the most recent N closed trades:

    Active     — full-size trades
    Probation  — half-size trades (real money, reduced risk)
    Suspended  — paper-trade only (signals logged, no execution)

Iron rule: a product cannot skip tiers in either direction. Suspended
must recover to Probation before returning to Active. This guards the
"what if a blacklisted loser becomes a winner?" case — recovery happens
gradually under observation rather than as a coin-flip.
"""
import statistics
from collections.abc import Sequence

STATUS_ACTIVE = "active"
STATUS_PROBATION = "probation"
STATUS_SUSPENDED = "suspended"

MIN_TRADES_FOR_REVIEW = 10
SHARPE_DEMOTE = -0.5
SHARPE_PROMOTE = +0.2

_MIN_STDEV = 0.005


def _per_trade_sharpe(pnl_pcts: Sequence[float]) -> tuple[float | None, float]:
    if len(pnl_pcts) < 2:
        return None, 0.0
    sd = statistics.stdev(pnl_pcts)
    if sd < _MIN_STDEV:
        return None, sd
    return statistics.fmean(pnl_pcts) / sd, sd


def compute_status(trades: Sequence[dict], current: str) -> tuple[str, str]:
    """Evaluate next status given the recent closed trades and current tier.

    Returns (new_status, reason). When no transition is warranted, returns
    the current status with a hold reason.
    """
    n = len(trades)
    if n < MIN_TRADES_FOR_REVIEW:
        return current, f"hold: only {n} trades (need {MIN_TRADES_FOR_REVIEW})"

    pnl = [float(t["pnl_pct"]) for t in trades]
    sharpe, sd = _per_trade_sharpe(pnl)
    if sharpe is None:
        return current, f"hold: per-trade stdev {sd:.5f} below {_MIN_STDEV} threshold"

    if sharpe <= SHARPE_DEMOTE:
        if current == STATUS_ACTIVE:
            return STATUS_PROBATION, f"demote: per-trade sharpe={sharpe:.3f}"
        if current == STATUS_PROBATION:
            return STATUS_SUSPENDED, f"demote: per-trade sharpe={sharpe:.3f}"
        return current, "hold: already suspended"

    if sharpe >= SHARPE_PROMOTE:
        if current == STATUS_SUSPENDED:
            return STATUS_PROBATION, f"promote: per-trade sharpe={sharpe:.3f}"
        if current == STATUS_PROBATION:
            return STATUS_ACTIVE, f"promote: per-trade sharpe={sharpe:.3f}"
        return current, "hold: already active"

    return current, f"hold: per-trade sharpe={sharpe:.3f}"
