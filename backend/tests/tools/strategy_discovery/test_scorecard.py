"""Tests for tools.strategy_discovery.scorecard (Phase 4)."""

from __future__ import annotations

from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
from tools.strategy_discovery.scorecard import (
    CapScorecard,
    evaluate_cap_gates,
    pick_verdict,
    render_scorecard,
)


def _passing_metrics() -> PortfolioMetrics:
    m = PortfolioMetrics(
        cumulative_profit_raw=0.20,
        cumulative_profit_deflated=0.15,
        max_dd=0.25,
        sortino=1.5,
        trade_count=100,
        pct_slots_full=0.4,
        mean_concurrent=1.8,
    )
    return m


def test_portfolio_gates_max_dd_30():
    m = _passing_metrics()
    m.max_dd = 0.31
    gates, overall = evaluate_cap_gates(m)
    assert gates["max_dd_le_30"] is False
    assert overall is False


def test_portfolio_gates_deflated_profit_positive():
    m = _passing_metrics()
    m.cumulative_profit_deflated = -0.01
    gates, overall = evaluate_cap_gates(m)
    assert gates["deflated_profit_gt_0"] is False
    assert overall is False


def test_portfolio_gates_trade_count_50():
    m = _passing_metrics()
    m.trade_count = 49
    gates, overall = evaluate_cap_gates(m)
    assert gates["trade_count_ge_50"] is False
    assert overall is False


def test_verdict_picks_highest_deflated_passing_cap():
    # 3 caps: N=3 fails, N=4 passes with 0.10, N=5 passes with 0.15 → pick N=5
    cards = [
        CapScorecard(
            cap=3,
            metrics=PortfolioMetrics(
                cumulative_profit_raw=-0.05,
                cumulative_profit_deflated=-0.08,
                max_dd=0.30,
                sortino=0.5,
                trade_count=60,
            ),
            k_evaluated=5000,
            inflation=0.03,
            gates={
                "deflated_profit_gt_0": False,
                "max_dd_le_30": True,
                "trade_count_ge_50": True,
                "sortino_ge_0": True,
            },
            overall_pass=False,
            selected_profiles=[],
        ),
        CapScorecard(
            cap=4,
            metrics=PortfolioMetrics(
                cumulative_profit_raw=0.15,
                cumulative_profit_deflated=0.10,
                max_dd=0.20,
                sortino=1.2,
                trade_count=80,
            ),
            k_evaluated=8000,
            inflation=0.05,
            gates={
                k: True
                for k in (
                    "max_dd_le_30",
                    "deflated_profit_gt_0",
                    "trade_count_ge_50",
                    "sortino_ge_0",
                )
            },
            overall_pass=True,
            selected_profiles=[],
        ),
        CapScorecard(
            cap=5,
            metrics=PortfolioMetrics(
                cumulative_profit_raw=0.20,
                cumulative_profit_deflated=0.15,
                max_dd=0.25,
                sortino=1.5,
                trade_count=100,
            ),
            k_evaluated=10000,
            inflation=0.05,
            gates={
                k: True
                for k in (
                    "max_dd_le_30",
                    "deflated_profit_gt_0",
                    "trade_count_ge_50",
                    "sortino_ge_0",
                )
            },
            overall_pass=True,
            selected_profiles=[],
        ),
    ]
    chosen_cap, verdict = pick_verdict(cards)
    assert chosen_cap == 5
    assert "deploy" in verdict.lower()
    assert "5" in verdict


def test_verdict_abort_when_all_caps_fail():
    cards = [
        CapScorecard(
            cap=cap,
            metrics=PortfolioMetrics(
                cumulative_profit_raw=-0.05,
                cumulative_profit_deflated=-0.08,
                max_dd=0.40,
                sortino=-0.5,
                trade_count=20,
            ),
            k_evaluated=5000,
            inflation=0.03,
            gates={
                "deflated_profit_gt_0": False,
                "max_dd_le_30": False,
                "trade_count_ge_50": False,
                "sortino_ge_0": False,
            },
            overall_pass=False,
            selected_profiles=[],
        )
        for cap in [3, 4, 5]
    ]
    chosen_cap, verdict = pick_verdict(cards)
    assert chosen_cap is None
    assert "abort" in verdict.lower()


def test_render_scorecard_produces_markdown_with_all_cap_sections():
    cards = [
        CapScorecard(
            cap=cap,
            metrics=PortfolioMetrics(
                cumulative_profit_raw=0.10 + cap * 0.01,
                cumulative_profit_deflated=0.05 + cap * 0.01,
                max_dd=0.20,
                sortino=1.2,
                trade_count=60,
                pct_slots_full=0.3,
                mean_concurrent=1.5,
            ),
            k_evaluated=5000,
            inflation=0.05,
            gates={
                k: True
                for k in (
                    "max_dd_le_30",
                    "deflated_profit_gt_0",
                    "trade_count_ge_50",
                    "sortino_ge_0",
                )
            },
            overall_pass=True,
            selected_profiles=[],
        )
        for cap in [3, 4, 5]
    ]
    md = render_scorecard(cards)
    # Mentions each cap and the verdict
    assert "N=3" in md and "N=4" in md and "N=5" in md
    assert "Verdict" in md or "verdict" in md
