"""
TDD tests for CNN risk management layer:
  - Hard stop-loss at -8%
  - Max hold time exit at 48 hours
  - Win/loss tracking on _CNNBook

Written before implementation (tests will fail until code is added).
"""
import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "qwen2.5:7b")

from agents.cnn_agent import CoinbaseCNNAgent, _CNNBook, _CNN_STOP_LOSS_PCT, _CNN_MAX_HOLD_SECS


# ── helpers ───────────────────────────────────────────────────────────────────

def _book_with_position(pid: str, avg_price: float, size: float = 10.0,
                         entry_time: float = None) -> _CNNBook:
    """Return a _CNNBook pre-loaded with one open position (no DB calls)."""
    book = _CNNBook()
    book.balance = 1000.0
    book.positions[pid] = {
        "size":       size,
        "avg_price":  avg_price,
        "entry_time": entry_time or time.time(),
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

        book.wins   = 3
        book.losses = 1
        assert abs(book.win_rate - 0.75) < 0.001

    @pytest.mark.asyncio
    async def test_expectancy_property(self):
        """expectancy = win_rate * avg_win_pct - loss_rate * avg_loss_pct."""
        book = _CNNBook()
        # Simulate 4 trades: 3 wins at +2%, 1 loss at -1%
        book.wins         = 3
        book.losses       = 1
        book._sum_win_pct  = 6.0   # 3 × 2%
        book._sum_loss_pct = 1.0   # 1 × 1%
        # expectancy = 0.75 * 2.0 - 0.25 * 1.0 = 1.25%
        assert abs(book.expectancy - 1.25) < 0.01

    @pytest.mark.asyncio
    async def test_multiple_sells_accumulate(self):
        """Multiple sells tracked correctly across wins and losses."""
        book = _CNNBook()
        book.balance = 1000.0
        prices = [("A-USD", 100.0, 110.0), ("B-USD", 100.0, 90.0),
                  ("C-USD", 100.0, 115.0), ("D-USD", 100.0, 85.0)]
        for pid, entry, exit_p in prices:
            book.positions[pid] = {"size": 1.0, "avg_price": entry,
                                    "entry_time": time.time()}
        with (
            patch("agents.cnn_agent.database.close_trade", new=AsyncMock()),
            patch("agents.cnn_agent.database.save_agent_state", new=AsyncMock()),
        ):
            for pid, _, exit_p in prices:
                await book.sell(pid, exit_p)

        assert book.wins   == 2
        assert book.losses == 2
        assert abs(book.win_rate - 0.50) < 0.001


# ── Stop-loss exit ─────────────────────────────────────────────────────────────

class TestCNNStopLoss:

    @pytest.mark.asyncio
    async def test_stop_loss_fires_at_8pct_loss(self):
        """Position down 8.1% → _check_risk_exits must close it."""
        agent = CoinbaseCNNAgent()
        entry = 1000.0
        current = entry * (1 - 0.081)   # -8.1% → below -8% threshold
        agent.book = _book_with_position("XRP-USD", avg_price=entry)

        sell_mock = AsyncMock(return_value=-8.1)
        ws_mock   = MagicMock()
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
        """Position down 5% (below threshold) → no exit."""
        agent = CoinbaseCNNAgent()
        entry   = 1000.0
        current = entry * (1 - 0.05)   # -5% → above -8% threshold
        agent.book = _book_with_position("SOL-USD", avg_price=entry)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock   = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_loss_does_not_fire_on_profitable_position(self):
        """Profitable position must not be stop-loss exited."""
        agent = CoinbaseCNNAgent()
        entry   = 1000.0
        current = entry * 1.05   # +5%
        agent.book = _book_with_position("ETH-USD", avg_price=entry)

        sell_mock = AsyncMock(return_value=50.0)
        ws_mock   = MagicMock()
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
        """Position held 49h → _check_risk_exits must close it."""
        agent     = CoinbaseCNNAgent()
        entry     = 500.0
        old_entry = time.time() - (49 * 3600)   # 49h ago
        agent.book = _book_with_position("DOT-USD", avg_price=entry,
                                          entry_time=old_entry)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock   = MagicMock()
        ws_mock.get_price.return_value = entry   # flat price, not a stop-loss
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "HOLD" in trigger.upper() or "TIME" in trigger.upper(), \
            f"Expected MAX_HOLD/TIME trigger, got: {trigger}"

    @pytest.mark.asyncio
    async def test_max_hold_does_not_fire_at_47_hours(self):
        """Position held 47h → still within window, must not exit."""
        agent     = CoinbaseCNNAgent()
        entry     = 500.0
        recent_entry = time.time() - (47 * 3600)
        agent.book = _book_with_position("AVAX-USD", avg_price=entry,
                                          entry_time=recent_entry)

        sell_mock = AsyncMock(return_value=0.0)
        ws_mock   = MagicMock()
        ws_mock.get_price.return_value = entry
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_max_hold_constant_is_48_hours(self):
        """_CNN_MAX_HOLD_SECS must equal 48 * 3600."""
        assert _CNN_MAX_HOLD_SECS == 48 * 3600, (
            f"Max hold is {_CNN_MAX_HOLD_SECS/3600:.0f}h — expected 48h."
        )

    @pytest.mark.asyncio
    async def test_stop_loss_takes_priority_over_max_hold(self):
        """When both conditions are true, stop-loss trigger is used (tighter risk)."""
        agent     = CoinbaseCNNAgent()
        entry     = 1000.0
        old_entry = time.time() - (50 * 3600)          # 50h old
        current   = entry * (1 - 0.10)                 # also -10% loss
        agent.book = _book_with_position("LINK-USD", avg_price=entry,
                                          entry_time=old_entry)

        sell_mock = AsyncMock(return_value=-100.0)
        ws_mock   = MagicMock()
        ws_mock.get_price.return_value = current
        agent.ws = ws_mock

        with patch.object(agent.book, "sell", sell_mock):
            await agent._check_risk_exits()

        sell_mock.assert_called_once()
        trigger = sell_mock.call_args[1].get("trigger") or sell_mock.call_args[0][2]
        assert "STOP" in trigger.upper(), \
            f"Stop-loss should take priority over max-hold, got trigger: {trigger}"

    @pytest.mark.asyncio
    async def test_no_price_skips_exit(self):
        """If WS has no price and REST fallback unavailable, position is not touched."""
        agent = CoinbaseCNNAgent()
        agent.book = _book_with_position("RARE-USD", avg_price=1.0,
                                          entry_time=time.time() - (49 * 3600))
        sell_mock = AsyncMock(return_value=0.0)
        ws_mock   = MagicMock()
        ws_mock.get_price.return_value = None   # no WS price
        agent.ws = ws_mock

        with (
            patch.object(agent.book, "sell", sell_mock),
            patch("agents.cnn_agent.coinbase_client.get_product",
                  new=AsyncMock(side_effect=Exception("no data"))),
        ):
            await agent._check_risk_exits()

        sell_mock.assert_not_called()
