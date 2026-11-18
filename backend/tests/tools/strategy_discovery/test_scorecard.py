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
    assert "research candidate" in verdict.lower()
    assert "deployment blocked" in verdict.lower()
    assert "deploy at" not in verdict.lower()
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


def test_empty_scorecard_still_discloses_missing_validation():
    md = render_scorecard([])
    assert "Deployment blocked" in md
    assert "untouched holdout" in md.lower()
    assert "research" in md.lower()


def _card_with_exact_rule():
    from tests.tools.strategy_discovery.rule_fixtures import machine_rule_fixture
    from tests.tools.strategy_discovery.test_portfolio_sim import _make_profile

    profile = _make_profile("BTC-USD", 0, 24, "price_over_ema20 > 1.02")
    profile.machine_rule = machine_rule_fixture("price_over_ema20 > 1.0249")
    profile.rule_digest = "a" * 64
    return CapScorecard(3, _passing_metrics(), 10, 0.05, {}, True, [profile])


def test_markdown_identifies_exact_rule_separately_from_rounded_display():
    import json

    card = _card_with_exact_rule()
    profile = card.selected_profiles[0]
    md = render_scorecard([card])
    assert "Rounded display summary (not executable):" in md
    assert "Exact machine rule (NaN routes right):" in md
    assert json.dumps(profile.machine_rule, sort_keys=True) in md
    assert profile.rule_digest in md
    assert "Group search metrics (not representative-policy performance)" in md
    assert "Deployment blocked" in md


def test_json_preserves_exact_rule_and_scopes_group_metrics(tmp_path):
    import json

    from tools.strategy_discovery.build_phase4 import _write_deployment_json

    card = _card_with_exact_rule()
    profile = card.selected_profiles[0]
    path = tmp_path / "research.json"
    _write_deployment_json(card, path)
    payload = json.loads(path.read_text())
    row = payload["profiles"][0]
    assert payload["schema_version"] == 2
    assert payload["deployment_eligible"] is False
    assert row["profile_id"] == profile.profile_id
    assert row["machine_rule"] == profile.machine_rule
    assert row["rule_digest"] == profile.rule_digest
    assert row["rounded_display_summary"] == "price_over_ema20 > 1.02"
    assert "rule_path" not in row
    assert not any(key.startswith("expected_") for key in row)
    assert row["group_search_metrics"]["scope"] == "qualifying_leaves_not_representative_policy"
    assert row["group_search_metrics"]["avg_win"] == profile.avg_win
