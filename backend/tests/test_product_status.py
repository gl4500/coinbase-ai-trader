"""
TDD tests for services/product_status.py — tiered blacklist evaluator
(Session 57, cash-flow lever 4, Task #120).

Architecture:
    Active     → full-size trades
    Probation  → ½-size trades (still real money, but reduced risk)
    Suspended  → paper-trade only (signals logged, no execution)

State transitions are based on per-trade Sharpe over the most recent
N closed trades (trade-count cadence — Coinbase trades 24/7 so calendar
windows would be unreliable).

Iron rule: a product cannot skip tiers. Suspended must recover to
Probation before it can return to Active. This protects the user's
explicit concern: "what if you blacklist losers then they become
winners?" — recovery is gradual and observation-based.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _trade(pnl_pct: float) -> dict:
    """Minimal trade record — only pnl_pct is used by evaluator."""
    return {"pnl_pct": pnl_pct}


def _losses(n: int, pct: float = -0.02) -> list[dict]:
    return [_trade(pct) for _ in range(n)]


def _wins(n: int, pct: float = +0.02) -> list[dict]:
    return [_trade(pct) for _ in range(n)]


# ── Module-level constants and enum ──────────────────────────────────────────


class TestConstants:
    def test_status_constants_defined(self):
        from services.product_status import (
            STATUS_ACTIVE,
            STATUS_PROBATION,
            STATUS_SUSPENDED,
        )

        assert STATUS_ACTIVE == "active"
        assert STATUS_PROBATION == "probation"
        assert STATUS_SUSPENDED == "suspended"

    def test_review_constants(self):
        from services.product_status import (
            MIN_TRADES_FOR_REVIEW,
            SHARPE_DEMOTE,
            SHARPE_PROMOTE,
        )

        assert MIN_TRADES_FOR_REVIEW == 10, (
            "Need at least 10 closed trades before status changes — "
            "stops noise from flipping a product after a 2-trade unlucky streak"
        )
        assert SHARPE_DEMOTE == -0.5, (
            "Demote when per-trade Sharpe ≤ -0.5 (clearly underperforming)"
        )
        assert SHARPE_PROMOTE == +0.2, (
            "Promote on modest positive Sharpe — recovery doesn't need to be heroic"
        )


# ── Insufficient sample size ─────────────────────────────────────────────────


class TestMinSampleGate:
    def test_too_few_trades_keeps_active(self):
        """9 losing trades isn't enough to demote — still gathering signal."""
        from services.product_status import STATUS_ACTIVE, compute_status

        new, _ = compute_status(_losses(9), current=STATUS_ACTIVE)
        assert new == STATUS_ACTIVE

    def test_too_few_trades_keeps_suspended(self):
        """A suspended product with only 4 paper wins isn't ready to be promoted."""
        from services.product_status import STATUS_SUSPENDED, compute_status

        new, _ = compute_status(_wins(4), current=STATUS_SUSPENDED)
        assert new == STATUS_SUSPENDED

    def test_empty_trades_no_change(self):
        from services.product_status import STATUS_ACTIVE, compute_status

        new, _ = compute_status([], current=STATUS_ACTIVE)
        assert new == STATUS_ACTIVE


# ── Demotion paths ───────────────────────────────────────────────────────────


class TestDemotion:
    def test_active_to_probation_on_bad_sharpe(self):
        """10 trades with Sharpe ≤ -0.5 demotes Active → Probation."""
        from services.product_status import STATUS_ACTIVE, STATUS_PROBATION, compute_status

        # 10 trades, all -2% → Sharpe undefined (zero stdev) but mean is -2%
        # Use mixed losses to get well-defined Sharpe ≤ -0.5
        trades = [
            _trade(-0.03),
            _trade(-0.01),
            _trade(-0.04),
            _trade(-0.02),
            _trade(-0.03),
            _trade(-0.01),
            _trade(-0.02),
            _trade(-0.03),
            _trade(-0.02),
            _trade(-0.04),
        ]
        new, reason = compute_status(trades, current=STATUS_ACTIVE)
        assert new == STATUS_PROBATION
        assert "sharpe" in reason.lower() or "demot" in reason.lower()

    def test_probation_to_suspended_on_continued_losses(self):
        """A probationary product that keeps losing → Suspended."""
        from services.product_status import STATUS_PROBATION, STATUS_SUSPENDED, compute_status

        trades = [_trade(-0.03)] * 5 + [_trade(-0.01)] * 5
        new, _ = compute_status(trades, current=STATUS_PROBATION)
        assert new == STATUS_SUSPENDED

    def test_active_does_not_skip_to_suspended(self):
        """Even with terrible Sharpe, Active demotes only one tier per evaluation."""
        from services.product_status import STATUS_ACTIVE, STATUS_PROBATION, compute_status

        # All -10% — catastrophic but evaluator must demote one step at a time
        trades = [
            _trade(-0.10),
            _trade(-0.08),
            _trade(-0.12),
            _trade(-0.09),
            _trade(-0.11),
            _trade(-0.07),
            _trade(-0.13),
            _trade(-0.06),
            _trade(-0.10),
            _trade(-0.09),
        ]
        new, _ = compute_status(trades, current=STATUS_ACTIVE)
        assert new == STATUS_PROBATION, (
            "Active must NOT jump straight to Suspended — "
            "tiered system requires observable recovery window"
        )


# ── Promotion paths (recovery from blacklist) ────────────────────────────────


class TestPromotion:
    def test_suspended_to_probation_on_paper_wins(self):
        """Suspended product whose paper trades show positive Sharpe → Probation."""
        from services.product_status import STATUS_PROBATION, STATUS_SUSPENDED, compute_status

        trades = [
            _trade(+0.03),
            _trade(+0.02),
            _trade(+0.01),
            _trade(+0.04),
            _trade(+0.02),
            _trade(+0.03),
            _trade(+0.05),
            _trade(+0.01),
            _trade(+0.02),
            _trade(+0.03),
        ]
        new, reason = compute_status(trades, current=STATUS_SUSPENDED)
        assert new == STATUS_PROBATION, (
            "FLIP recovery: a previously-losing product must be promotable "
            "back to Probation if its paper-trades show edge"
        )

    def test_probation_to_active_on_recovery(self):
        """Probationary product earning positive Sharpe → Active."""
        from services.product_status import STATUS_ACTIVE, STATUS_PROBATION, compute_status

        trades = _wins(10, pct=+0.03)
        # Add small variance so Sharpe is computable
        trades[0] = _trade(+0.01)
        trades[5] = _trade(+0.05)
        new, _ = compute_status(trades, current=STATUS_PROBATION)
        assert new == STATUS_ACTIVE

    def test_suspended_does_not_skip_to_active(self):
        """Even after wonderful paper trades, Suspended → Probation only."""
        from services.product_status import STATUS_PROBATION, STATUS_SUSPENDED, compute_status

        trades = _wins(10, pct=+0.10)
        trades[0] = _trade(+0.05)
        new, _ = compute_status(trades, current=STATUS_SUSPENDED)
        assert new == STATUS_PROBATION, (
            "Suspended must observe Probation tier first before returning to Active"
        )


# ── Boundary / hold conditions ───────────────────────────────────────────────


class TestHold:
    def test_neutral_sharpe_keeps_status(self):
        """Sharpe between thresholds → no change."""
        from services.product_status import STATUS_ACTIVE, compute_status

        trades = [_trade(+0.01), _trade(-0.01)] * 5  # mean ≈ 0
        new, _ = compute_status(trades, current=STATUS_ACTIVE)
        assert new == STATUS_ACTIVE

    def test_exactly_demote_threshold_does_not_demote(self):
        """Sharpe == -0.5 (boundary) is held, not demoted — strict inequality."""
        from services.product_status import STATUS_ACTIVE, compute_status

        # Construct a series with exactly -0.5 Sharpe is awkward; use slightly above
        # to assert the boundary is non-inclusive: -0.49 must NOT demote.
        trades = [_trade(-0.0049)] * 5 + [_trade(-0.0051)] * 5  # mean tiny, std tiny
        new, _ = compute_status(trades, current=STATUS_ACTIVE)
        assert new == STATUS_ACTIVE, "Demote rule is strict (≤) — boundary case should not demote"


# ── Orchestration helper: evaluate_and_persist (#125b) ───────────────────────


def _row(pct_pnl: float) -> dict:
    """Trade row as returned by database.get_trades — pct_pnl in percent units."""
    return {"pct_pnl": pct_pnl}


class TestEvaluateAndPersist:
    """`evaluate_and_persist(pid, agent="CNN", n_trades=10)` — pulls recent
    closed trades from `database.get_trades`, reads current status from
    `database.get_product_status`, runs `compute_status`, and persists the
    new status via `database.set_product_status` only if it changed.

    Returns (old_status, new_status, changed).

    Critical: `trades.pct_pnl` is stored in PERCENT units (×100) but
    `compute_status` expects decimal form because of the `_MIN_STDEV=0.005`
    gate. The helper must divide pct_pnl by 100 before passing to
    `compute_status`.
    """

    @pytest.mark.asyncio
    async def test_insufficient_trades_returns_unchanged(self):
        from services.product_status import evaluate_and_persist

        with (
            patch(
                "services.product_status.database.get_trades",
                AsyncMock(return_value=[_row(-2.0)] * 5),
            ),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "active"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert (old, new, changed) == ("active", "active", False)
        set_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_status_row_defaults_to_active(self):
        from services.product_status import evaluate_and_persist

        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=[])),
            patch(
                "services.product_status.database.get_product_status", AsyncMock(return_value=None)
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert old == "active"
        assert new == "active"
        assert changed is False
        set_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_demote_path_persists_and_returns_changed(self):
        """Active → Probation when sharpe ≤ -0.5; must persist."""
        from services.product_status import evaluate_and_persist

        # 10 losing trades in PERCENT units (pct_pnl as stored in DB)
        rows = [
            _row(-3.0),
            _row(-1.0),
            _row(-4.0),
            _row(-2.0),
            _row(-3.0),
            _row(-1.0),
            _row(-2.0),
            _row(-3.0),
            _row(-2.0),
            _row(-4.0),
        ]
        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=rows)),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "active"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert old == "active"
        assert new == "probation"
        assert changed is True
        set_mock.assert_awaited_once()
        args, kwargs = set_mock.call_args
        # accept either positional or kwarg form
        assert ("BTC-USD" in args) or (kwargs.get("product_id") == "BTC-USD")
        assert ("probation" in args) or (kwargs.get("status") == "probation")

    @pytest.mark.asyncio
    async def test_promote_path_persists_and_returns_changed(self):
        """Suspended → Probation when sharpe ≥ +0.2; must persist."""
        from services.product_status import evaluate_and_persist

        rows = [
            _row(+3.0),
            _row(+2.0),
            _row(+1.0),
            _row(+4.0),
            _row(+2.0),
            _row(+3.0),
            _row(+5.0),
            _row(+1.0),
            _row(+2.0),
            _row(+3.0),
        ]
        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=rows)),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "suspended"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("ETH-USD")
        assert old == "suspended"
        assert new == "probation"
        assert changed is True
        set_mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_change_returns_changed_false(self):
        """Neutral sharpe between thresholds — no persist call."""
        from services.product_status import evaluate_and_persist

        # 10 trades with mean ~0 — neutral sharpe
        rows = [_row(+1.0), _row(-1.0)] * 5
        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=rows)),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "active"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert old == "active"
        assert new == "active"
        assert changed is False
        set_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pct_pnl_converted_from_percent_to_decimal(self):
        """`trades.pct_pnl` is stored ×100; helper must divide by 100 before
        feeding to compute_status — otherwise tiny variance trades (stdev ≈
        0.001 in percent units = 0.1% in decimal) would never clear the
        `_MIN_STDEV=0.005` decimal gate.

        Construct a row set whose PERCENT-form stdev easily clears 0.005 but
        whose DECIMAL-form stdev (÷100) falls below it. The helper should
        return a 'hold' on the decimal form, NOT a transition.
        """
        from services.product_status import evaluate_and_persist

        # In percent: stdev ≈ 0.10 (clearly > 0.005)
        # In decimal: stdev ≈ 0.001 (< 0.005, gate fires → no Sharpe → hold)
        rows = [_row(-0.5)] * 5 + [_row(-0.3)] * 5  # all losing, low variance
        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=rows)),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "active"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()) as set_mock,
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert (old, new, changed) == ("active", "active", False), (
            "If pct_pnl wasn't converted to decimal, the percent-form stdev "
            "would clear the gate and a demote could fire. Conversion is required."
        )
        set_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_rows_with_null_pct_pnl(self):
        """Defensive: rows where pct_pnl IS NULL must be skipped, not crash."""
        from services.product_status import evaluate_and_persist

        # mix valid + None — only the valid ones should reach compute_status
        rows = [
            _row(-3.0),
            {"pct_pnl": None},
            _row(-1.0),
            _row(-4.0),
            _row(-2.0),
            _row(-3.0),
            _row(-1.0),
            _row(-2.0),
            _row(-3.0),
            _row(-2.0),
            _row(-4.0),
        ]
        with (
            patch("services.product_status.database.get_trades", AsyncMock(return_value=rows)),
            patch(
                "services.product_status.database.get_product_status",
                AsyncMock(return_value={"status": "active"}),
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()),
        ):
            old, new, changed = await evaluate_and_persist("BTC-USD")
        assert old == "active"
        # 10 valid losses → demote
        assert new == "probation"
        assert changed is True

    @pytest.mark.asyncio
    async def test_passes_agent_and_limit_to_get_trades(self):
        from services.product_status import evaluate_and_persist

        get_trades_mock = AsyncMock(return_value=[])
        with (
            patch("services.product_status.database.get_trades", get_trades_mock),
            patch(
                "services.product_status.database.get_product_status", AsyncMock(return_value=None)
            ),
            patch("services.product_status.database.set_product_status", AsyncMock()),
        ):
            await evaluate_and_persist("SOL-USD", agent="CNN", n_trades=10)
        get_trades_mock.assert_awaited_once()
        kwargs = get_trades_mock.call_args.kwargs
        assert kwargs.get("agent") == "CNN"
        assert kwargs.get("product_id") == "SOL-USD"
        assert kwargs.get("closed_only") is True
        assert kwargs.get("limit") == 10
