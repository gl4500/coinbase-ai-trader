"""
TDD tests for CNN risk management layer:
  - Hard stop-loss at -8%
  - Max hold time exit at 48 hours
  - Win/loss tracking on _CNNBook

Written before implementation (tests will fail until code is added).
"""

import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

from agents.cnn_agent import (
    _CNN_ATR_TRAIL_MAX,
    _CNN_ATR_TRAIL_MIN,
    _CNN_MAX_HOLD_SECS,
    _CNN_STOP_LOSS_PCT,
    CoinbaseCNNAgent,
    _CNNBook,
)

# ── helpers ───────────────────────────────────────────────────────────────────


def _book_with_position(
    pid: str,
    avg_price: float,
    size: float = 10.0,
    entry_time: float = None,
    peak_price: float = None,
) -> _CNNBook:
    """Return a _CNNBook pre-loaded with one open position (no DB calls)."""
    book = _CNNBook()
    book.balance = 1000.0
    book.positions[pid] = {
        "size": size,
        "avg_price": avg_price,
        "entry_time": entry_time or time.time(),
        "peak_price": peak_price or avg_price,
    }
    return book


# ── _CNNBook win/loss tracking ─────────────────────────────────────────────────


class TestCNNBookWinLossTracking:
    @pytest.mark.asyncio
    async def test_winning_sell_increments_wins(self):
        """Selling above avg_price must increment book.wins by 1."""
        book = _book_with_position("BTC-USD", avg_price=50_000.0, size=0.01)
        with (
            patch("agents.cnn_agent.database.close_trade", new=AsyncMock()),
            patch("agents.cnn_agent.database.save_agent_state", new=AsyncMock()),
        ):
            await book.sell("BTC-USD", price=55_000.0)  # +10% → win

        assert book.wins == 1, f"Expected wins=1, got {book.wins}"
        assert book.losses == 0

    @pytest.mark.asyncio
    async def test_losing_sell_increments_losses(self):
        """Selling below avg_price must increment book.losses by 1."""
        book = _book_with_position("ETH-USD", avg_price=3_000.0, size=0.1)
        with (
            patch("agents.cnn_agent.database.close_trade", new=AsyncMock()),
            patch("agents.cnn_agent.database.save_agent_state", new=AsyncMock()),
        ):
            await book.sell("ETH-USD", price=2_700.0)  # -10% → loss

        assert book.losses == 1, f"Expected losses=1, got {book.losses}"
        assert book.wins == 0

    @pytest.mark.asyncio
    async def test_win_rate_property(self):
        """win_rate = wins / (wins + losses), returns 0.0 when no trades."""
        book = _CNNBook()
        assert book.win_rate == 0.0, "Empty book should have win_rate=0.0"

        book.wins = 3
        book.losses = 1
        assert abs(book.win_rate - 0.75) < 0.001

    @pytest.mark.asyncio
    async def test_expectancy_property(self):
        """expectancy = win_rate * avg_win_pct - loss_rate * avg_loss_pct."""
        book = _CNNBook()
        # Simulate 4 trades: 3 wins at +2%, 1 loss at -1%
        book.wins = 3
        book.losses = 1
        book._sum_win_pct = 6.0  # 3 × 2%
        book._sum_loss_pct = 1.0  # 1 × 1%
        # expectancy = 0.75 * 2.0 - 0.25 * 1.0 = 1.25%
        assert abs(book.expectancy - 1.25) < 0.01

    @pytest.mark.asyncio
    async def test_multiple_sells_accumulate(self):
        """Multiple sells tracked correctly across wins and losses."""
        book = _CNNBook()
        book.balance = 1000.0
        prices = [
            ("A-USD", 100.0, 110.0),
            ("B-USD", 100.0, 90.0),
            ("C-USD", 100.0, 115.0),
            ("D-USD", 100.0, 85.0),
        ]
        for pid, entry, exit_p in prices:
            book.positions[pid] = {"size": 1.0, "avg_price": entry, "entry_time": time.time()}
        with (
            patch("agents.cnn_agent.database.close_trade", new=AsyncMock()),
            patch("agents.cnn_agent.database.save_agent_state", new=AsyncMock()),
        ):
            for pid, _, exit_p in prices:
                await book.sell(pid, exit_p)

        assert book.wins == 2
        assert book.losses == 2
        assert abs(book.win_rate - 0.50) < 0.001


# ── Stop-loss exit ─────────────────────────────────────────────────────────────


class TestCNNStopLoss:
    @pytest.mark.asyncio
    async def test_stop_loss_fires_at_8pct_loss(self):
        """Position down 8.1% → _check_risk_exits must close it."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        current = entry * (1 - 0.081)  # -8.1% → below -8% threshold
        agent.book = _book_with_position("XRP-USD", avg_price=entry)

        sell_mock = AsyncMock(return_value=-8.1)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        assert sell_mock.call_args[0][0] == "XRP-USD"
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "STOP" in trigger.upper(), f"Expected STOP trigger, got: {trigger}"

    @pytest.mark.asyncio
    async def test_stop_loss_does_not_fire_at_5pct_loss(self):
        """Position down 5% from entry but only 1% from peak → no exit triggered."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        current = entry * (1 - 0.05)  # -5% from entry → above -8% hard stop
        # peak_price just above current so pct_from_peak ≈ -1%, below 3% ATR floor
        agent.book = _book_with_position(
            "SOL-USD", avg_price=entry, size=0.01, peak_price=current * 1.01
        )

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_loss_does_not_fire_on_profitable_position(self):
        """Profitable position must not be stop-loss exited."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        current = entry * 1.05  # +5%
        agent.book = _book_with_position("ETH-USD", avg_price=entry)

        sell_mock = AsyncMock(return_value=50.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_loss_constant_is_8pct(self):
        """_CNN_STOP_LOSS_PCT must equal 0.08 (required for $50k math)."""
        assert _CNN_STOP_LOSS_PCT == 0.08, (
            f"Stop loss is {_CNN_STOP_LOSS_PCT:.2%} — should be 8% "
            "to match the capital-at-risk analysis."
        )


# ── Max hold time exit ─────────────────────────────────────────────────────────


class TestCNNMaxHoldTime:
    @pytest.mark.asyncio
    async def test_max_hold_fires_at_49_hours(self):
        """Position held beyond max-hold window → _check_risk_exits must close it."""
        agent = CoinbaseCNNAgent()
        entry = 500.0
        old_entry = time.time() - (_CNN_MAX_HOLD_SECS + 3600)  # 1h past limit
        agent.book = _book_with_position("DOT-USD", avg_price=entry, entry_time=old_entry)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = entry  # flat price, not a stop-loss
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "HOLD" in trigger.upper() or "TIME" in trigger.upper(), (
            f"Expected MAX_HOLD/TIME trigger, got: {trigger}"
        )

    @pytest.mark.asyncio
    async def test_max_hold_does_not_fire_at_47_hours(self):
        """Position held 47h → well within 7-day window, must not exit."""
        agent = CoinbaseCNNAgent()
        entry = 500.0
        recent_entry = time.time() - (47 * 3600)
        agent.book = _book_with_position("AVAX-USD", avg_price=entry, entry_time=recent_entry)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = entry
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_hold_constant_is_7_days(self):
        """_CNN_MAX_HOLD_SECS must equal 7 * 24 * 3600 (trailing stop replaced 48h limit)."""
        assert _CNN_MAX_HOLD_SECS == 7 * 24 * 3600, (
            f"Max hold is {_CNN_MAX_HOLD_SECS / 3600:.0f}h — expected 168h (7 days)."
        )

    @pytest.mark.asyncio
    async def test_stop_loss_takes_priority_over_max_hold(self):
        """When both conditions are true, stop-loss trigger is used (tighter risk)."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        old_entry = time.time() - (50 * 3600)  # 50h old
        current = entry * (1 - 0.10)  # also -10% loss
        agent.book = _book_with_position("LINK-USD", avg_price=entry, entry_time=old_entry)

        sell_mock = AsyncMock(return_value=-100.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "STOP" in trigger.upper(), (
            f"Stop-loss should take priority over max-hold, got trigger: {trigger}"
        )

    @pytest.mark.asyncio
    async def test_no_price_skips_exit(self):
        """If WS has no price and REST fallback unavailable, position is not touched."""
        agent = CoinbaseCNNAgent()
        agent.book = _book_with_position(
            "RARE-USD", avg_price=1.0, entry_time=time.time() - (49 * 3600)
        )
        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = None  # no WS price
        agent.ws = ws_mock

        with (
            patch.object(agent.book, "sell", sell_mock),
            patch(
                "agents.cnn_agent.coinbase_client.get_product",
                new=AsyncMock(side_effect=Exception("no data")),
            ),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()


# ── ATR trail-stop floor (Session 57 cash-flow phase) ─────────────────────────


class TestTrailFloor:
    """Floor on ATR trailing stop must be 6%, not 3%.

    Why: 3% floor was triggering TRAIL_STOP on routine intra-day chop,
    locking in losses before mean-reversion could play out. Bumping the
    floor to 6% gives positions room to breathe in low-volatility regimes
    while still capping downside at hard STOP_LOSS=8%.
    """

    @pytest.mark.asyncio
    async def test_trail_floor_constant_is_6pct(self):
        """_CNN_ATR_TRAIL_MIN must equal 0.06 (cash-flow lever 3)."""
        assert _CNN_ATR_TRAIL_MIN == 0.06, (
            f"Trail floor is {_CNN_ATR_TRAIL_MIN:.2%} — should be 6% "
            "to stop premature trail-stop exits in low-ATR regimes."
        )

    @pytest.mark.asyncio
    async def test_4pct_drawdown_does_not_trigger_trail_stop(self):
        """Position 4% off peak must NOT exit when ATR floor is 6%."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        peak = 1000.0
        current = peak * (1 - 0.04)  # -4% from peak → above 6% floor
        agent.book = _book_with_position("OP-USD", avg_price=entry, size=0.01, peak_price=peak)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with (
            patch.object(agent.book, "sell", sell_mock),
            patch(
                "agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])
            ),  # forces floor fallback
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_65pct_drawdown_triggers_trail_stop(self):
        """Position with peak above entry, then 6.5% pullback fires trail.
        Under B2 design, peak must be above entry for trail to engage."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        peak = 1100.0  # peak_pnl 10% → threshold = 10 - 1.2 = 8.8%
        current = 1080.0  # pnl 8% < threshold 8.8% → fire
        agent.book = _book_with_position("ARB-USD", avg_price=entry, size=0.01, peak_price=peak)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with (
            patch.object(agent.book, "sell", sell_mock),
            patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "TRAIL" in trigger.upper(), f"Expected TRAIL_STOP trigger, got: {trigger}"


# ── WS exit handler contract: trail_pct cache write ────────────────────────────


@pytest.mark.asyncio
async def test_check_risk_exits_writes_trail_pct_to_position():
    """Scan loop caches the computed trail_pct on pos['trail_pct'] so the WS
    exit handler can read it without recomputing ATR per tick. Contract
    between scan loop and WS handler (see agents/exit_watcher.on_price_tick).
    """
    agent = CoinbaseCNNAgent()
    agent.book = _book_with_position("BTC-USD", avg_price=100.0, size=0.01, peak_price=105.0)

    ws_mock = MagicMock()
    ws_mock.get_price.return_value = 104.0  # between trail and stop-loss → no exit
    agent.ws = ws_mock

    # 20 candles with deterministic non-zero ATR (high-low range = 2.0)
    fake_candles = [{"high": 100.0, "low": 98.0, "close": 99.0} for _ in range(20)]

    with (
        patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=fake_candles)),
        patch.object(agent.book, "sell", new=AsyncMock()),
    ):
        await agent._check_risk_exits()

    pos = agent.book.positions["BTC-USD"]
    assert "trail_pct" in pos, "scan loop must write trail_pct for WS handler"
    assert _CNN_ATR_TRAIL_MIN <= pos["trail_pct"] <= _CNN_ATR_TRAIL_MAX


# ═══════════════════════════════════════════════════════════════════════════
# Task #46 — PnL-anchored trail behavior in _check_risk_exits
# ═══════════════════════════════════════════════════════════════════════════


class TestCNNRiskExitsPnLAnchored:
    """_check_risk_exits dispatches through _compute_exit_threshold (task #46).
    Position state gains peak_pnl_pct + position_dollars on each scan."""

    @pytest.mark.asyncio
    async def test_check_risk_exits_writes_peak_pnl_pct_and_position_dollars(self):
        """After one scan, position should have peak_pnl_pct + position_dollars set."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        current = 1080.0
        agent.book = _book_with_position("DASH-USD", avg_price=entry, peak_price=current)
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        fake_candles = [
            {"high": 1010 + i, "low": 990 + i, "close": 1000 + i, "start_time": i}
            for i in range(20)
        ]

        with (
            patch(
                "agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=fake_candles)
            ),
            patch.object(agent.book, "sell", new=AsyncMock()),
        ):
            await agent._check_risk_exits()

        pos = agent.book.positions["DASH-USD"]
        assert "peak_pnl_pct" in pos
        # peak_pnl_pct seeded from ratcheted peak_price (1080)
        assert pos["peak_pnl_pct"] == pytest.approx(0.08)
        assert "position_dollars" in pos
        # size from _book_with_position default is 10.0; position_dollars = size * current_price
        assert pos["position_dollars"] == pytest.approx(10.0 * current)

    @pytest.mark.asyncio
    async def test_check_risk_exits_uses_pnl_anchored_threshold(self):
        """DASH-like setup. Price $43.30 (pnl +0.77%) below the threshold
        (baseline 1.8% from peak_pnl 7.8% − atr 6%, or break-even 1.2% — either
        way fires). Verifies the new PnL-anchored trail dispatches sell."""
        agent = CoinbaseCNNAgent()
        entry = 42.97
        peak = 46.32
        current = 43.30  # pnl ≈ +0.77%
        agent.book = _book_with_position(
            "DASH-USD",
            avg_price=entry,
            size=1.35,
            peak_price=peak,
        )
        agent.book.balance = 942.0  # → total capital after position dollars ≈ $1k
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        fake_candles = [
            {"high": entry + 1, "low": entry - 1, "close": entry, "start_time": i}
            for i in range(20)
        ]
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch(
                "agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=fake_candles)
            ),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert trigger == "TRAIL_STOP", f"Expected TRAIL_STOP, got: {trigger}"

    @pytest.mark.asyncio
    async def test_check_risk_exits_fires_model_down_when_p_down_above_threshold(self):
        """B2 MODEL_DOWN: when v4.5 says p_down > 0.55 (cached on position),
        force exit regardless of price/trail state."""
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD

        agent = CoinbaseCNNAgent()
        entry = 1000.0
        agent.book = _book_with_position("ABC-USD", avg_price=entry, size=0.01)
        # Cache a strong DOWN signal — fresh ts (task #80)
        agent.book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05  # 0.60
        agent.book.positions["ABC-USD"]["p_down_ts_ms"] = int(time.time() * 1000)

        ws_mock = MagicMock()
        ws_mock.get_price.return_value = entry * 1.01  # +1% PnL — trail wouldn't fire
        agent.ws = ws_mock
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert trigger == "MODEL_DOWN", f"Expected MODEL_DOWN, got: {trigger}"

    @pytest.mark.asyncio
    async def test_check_risk_exits_does_not_fire_model_down_below_threshold(self):
        """B2 MODEL_DOWN: p_down at 0.50 (below 0.55 threshold) → no model exit."""
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD

        agent = CoinbaseCNNAgent()
        entry = 1000.0
        peak = 1100.0  # +10% peak, well above trail threshold
        agent.book = _book_with_position("ABC-USD", avg_price=entry, size=0.01, peak_price=peak)
        # p_down just below threshold — fresh ts (task #80)
        agent.book.positions["ABC-USD"]["p_down"] = _P_DOWN_EXIT_THRESHOLD - 0.05
        agent.book.positions["ABC-USD"]["p_down_ts_ms"] = int(time.time() * 1000)

        ws_mock = MagicMock()
        ws_mock.get_price.return_value = peak * 0.99  # near peak, won't trail-fire
        agent.ws = ws_mock
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_risk_exits_does_not_fire_model_down_when_p_down_stale(self):
        """Task #80: p_down > threshold but p_down_ts_ms > _P_DOWN_STALE_MS old → no fire.
        Market may have reversed since last scan; cached p_down is no longer trustworthy."""
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD, _P_DOWN_STALE_MS

        agent = CoinbaseCNNAgent()
        entry = 1000.0
        peak = 1100.0
        agent.book = _book_with_position("ABC-USD", avg_price=entry, size=0.01, peak_price=peak)
        pos = agent.book.positions["ABC-USD"]
        pos["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05  # would fire if fresh
        pos["p_down_ts_ms"] = int(time.time() * 1000) - _P_DOWN_STALE_MS - 1000  # 1s past stale

        ws_mock = MagicMock()
        ws_mock.get_price.return_value = peak * 0.99  # near peak, no trail fire
        agent.ws = ws_mock
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_risk_exits_does_not_fire_model_down_when_ts_missing(self):
        """Task #80: legacy position with p_down but no p_down_ts_ms → treated as stale, no fire.
        Backward compat: positions migrated before the staleness gate must not fire spuriously."""
        from agents.cnn_agent import _P_DOWN_EXIT_THRESHOLD

        agent = CoinbaseCNNAgent()
        entry = 1000.0
        peak = 1100.0
        agent.book = _book_with_position("ABC-USD", avg_price=entry, size=0.01, peak_price=peak)
        pos = agent.book.positions["ABC-USD"]
        pos["p_down"] = _P_DOWN_EXIT_THRESHOLD + 0.05  # no ts_ms set

        ws_mock = MagicMock()
        ws_mock.get_price.return_value = peak * 0.99
        agent.ws = ws_mock
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch("agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=[])),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_check_risk_exits_does_not_fire_above_threshold(self):
        """B2: DASH peak +7.8% → exit threshold = 7.8 - 1.2 = 6.6%. Current
        pnl +7.5% > 6.6% threshold → no fire."""
        agent = CoinbaseCNNAgent()
        entry = 42.97
        peak = 46.32
        current = 46.20  # pnl ≈ +7.5% > 6.6% threshold
        agent.book = _book_with_position(
            "DASH-USD",
            avg_price=entry,
            size=1.35,
            peak_price=peak,
        )
        agent.book.balance = 942.0
        ws_mock = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        fake_candles = [
            {"high": entry + 1, "low": entry - 1, "close": entry, "start_time": i}
            for i in range(20)
        ]
        sell_mock = AsyncMock(return_value=0.0)

        with (
            patch(
                "agents.cnn_agent.database.get_candles", new=AsyncMock(return_value=fake_candles)
            ),
            patch.object(agent.book, "sell", sell_mock),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()
