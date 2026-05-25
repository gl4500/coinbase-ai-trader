"""Phase 4 scorecard — per-cap Q0 gates + Markdown verdict report.

Pure functions on PortfolioMetrics. No I/O (build_phase4 owns writing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
from tools.strategy_discovery.profile_loader import LoadedProfile

_PORTFOLIO_MAX_DD     = 0.30
_PORTFOLIO_MIN_TRADES = 50


@dataclass
class CapScorecard:
    cap: int
    metrics: PortfolioMetrics
    k_evaluated: int
    inflation: float
    gates: Dict[str, bool]
    overall_pass: bool
    selected_profiles: List[LoadedProfile]


def evaluate_cap_gates(metrics: PortfolioMetrics) -> Tuple[Dict[str, bool], bool]:
    """Apply 4 portfolio Q0 gates."""
    gates = {
        "max_dd_le_30":        metrics.max_dd <= _PORTFOLIO_MAX_DD,
        "deflated_profit_gt_0": metrics.cumulative_profit_deflated > 0.0,
        "trade_count_ge_50":   metrics.trade_count >= _PORTFOLIO_MIN_TRADES,
        "sortino_ge_0":        metrics.sortino >= 0.0,
    }
    overall = all(gates.values())
    return gates, overall


def pick_verdict(per_cap: List[CapScorecard]) -> Tuple[Optional[int], str]:
    """Pick the highest-deflated-profit passing cap, or abort if none pass."""
    passing = [c for c in per_cap if c.overall_pass]
    if not passing:
        return None, "abort — no qualifying portfolio at any cap"
    best = max(passing, key=lambda c: c.metrics.cumulative_profit_deflated)
    return best.cap, f"deploy at N={best.cap} (deflated profit = {best.metrics.cumulative_profit_deflated:.4f})"


def render_scorecard(per_cap: List[CapScorecard]) -> str:
    """Render a Markdown scorecard with per-cap sections + comparison + verdict."""
    lines: List[str] = ["# Phase 4 Scorecard — strategy-discovery deployment selection", ""]
    chosen_cap, verdict = pick_verdict(per_cap)
    lines.append(f"**Verdict:** {verdict}")
    lines.append("")
    lines.append("## Comparison table")
    lines.append("")
    lines.append("| cap | deflated profit | raw profit | inflation | max DD | sortino | trades | pct full | mean conc | pass? |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|")
    for c in sorted(per_cap, key=lambda x: x.cap):
        lines.append(
            f"| {c.cap} "
            f"| {c.metrics.cumulative_profit_deflated:+.4f} "
            f"| {c.metrics.cumulative_profit_raw:+.4f} "
            f"| {c.inflation:.4f} "
            f"| {c.metrics.max_dd:.4f} "
            f"| {c.metrics.sortino:.3f} "
            f"| {c.metrics.trade_count} "
            f"| {c.metrics.pct_slots_full:.2%} "
            f"| {c.metrics.mean_concurrent:.2f} "
            f"| {'✅' if c.overall_pass else '❌'} |"
        )
    lines.append("")
    for c in sorted(per_cap, key=lambda x: x.cap):
        lines.append(f"## N={c.cap}")
        lines.append("")
        lines.append(f"- Deflated cumulative profit: **{c.metrics.cumulative_profit_deflated:+.4f}**  (raw {c.metrics.cumulative_profit_raw:+.4f}, inflation {c.inflation:.4f})")
        lines.append(f"- Max drawdown: {c.metrics.max_dd:.4f}")
        lines.append(f"- Sortino: {c.metrics.sortino:.3f}")
        lines.append(f"- Trade count: {c.metrics.trade_count}")
        lines.append(f"- Slot utilization: {c.metrics.pct_slots_full:.2%} full, mean concurrent {c.metrics.mean_concurrent:.2f}")
        lines.append(f"- K subsets evaluated: {c.k_evaluated}")
        lines.append(f"- Gates: " + ", ".join(f"{k}={'✅' if v else '❌'}" for k, v in c.gates.items()))
        lines.append(f"- Overall: {'✅ PASS' if c.overall_pass else '❌ FAIL'}")
        lines.append("")
        if c.selected_profiles:
            lines.append("Selected profiles:")
            for p in c.selected_profiles:
                lines.append(f"  - {p.profile_id}  (h={p.horizon})  `{p.rule_path}`  → deflated {p.cumulative_profit_deflated:+.4f}")
            lines.append("")
    return "\n".join(lines)
