"""ScorecardReport: dataclass aggregating all per-tau x per-tier rows + scalars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ScorecardReport:
    """Container for one model variant's scorecard.

    per_tau_rows: list of dicts, one per tau in tau_grid. Each row has:
        tau (float), precision (float|NaN), n_fired (int),
        e_return_{tier} (float|NaN) for each fee tier,
        sharpe_mean_{tier} (float|NaN), sharpe_std_{tier} (float|NaN).
    ece: scalar in [0, 1].
    recommended_operating_tau: tau with max precision among rows with
        n_fired >= floor AND positive E[r] at gate_tier. NaN if no row qualifies.
    gates_passed: dict[str, bool] for {precision, expected_return, paper_sharpe, ece}.
    """

    per_tau_rows: list[dict[str, Any]]
    ece: float
    recommended_operating_tau: float
    gates_passed: dict[str, bool]
    pos_rate: float
    gate_tier: str
