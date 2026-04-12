"""
Order Executor — Coinbase Advanced Trade
─────────────────────────────────────────
Places, tracks, and manages orders on Coinbase.

Flow:
  1. Receive signal dict
  2. Drawdown circuit breaker check (daily / weekly)
  3. Pre-flight: check credentials, USD balance, exposure limit
  4. ATR-based position sizing (when signal carries atr)
  5. Place limit order at signal price (or market order)
  6. Record to database
  7. Return result dict
"""
import asyncio
import logging
import time
import uuid
from typing import Dict, Optional

import database
from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)


_DRY_RUN_BALANCE = 1_000.0   # simulated USD starting balance for dry-run

_SECS_PER_DAY  = 86_400
_SECS_PER_WEEK = 604_800


class OrderExecutor:
    def __init__(self, dry_run: bool = True):
        self.dry_run          = dry_run
        self._dry_run_balance = _DRY_RUN_BALANCE

        # ── Drawdown tracking ──────────────────────────────────────────────────
        self._day_start_balance:  Optional[float] = None
        self._week_start_balance: Optional[float] = None
        self._day_start_ts:       float            = time.time()
        self._week_start_ts:      float            = time.time()
        self._halted:             bool             = False
        self._halt_reason:        str              = ""

        if dry_run:
            self._day_start_balance  = _DRY_RUN_BALANCE
            self._week_start_balance = _DRY_RUN_BALANCE
            logger.info(
                f"OrderExecutor: DRY-RUN mode — simulated balance ${_DRY_RUN_BALANCE:,.2f} USD"
            )

    # ── Drawdown circuit breaker ───────────────────────────────────────────────

    async def _current_balance(self) -> float:
        if self.dry_run:
            return self._dry_run_balance
        return await coinbase_client.get_usd_balance()

    async def _reset_windows_if_due(self, balance: float) -> None:
        now = time.time()
        if now - self._day_start_ts >= _SECS_PER_DAY:
            self._day_start_balance = balance
            self._day_start_ts      = now
            logger.info(f"Drawdown: daily window reset — baseline ${balance:,.2f}")
        if now - self._week_start_ts >= _SECS_PER_WEEK:
            self._week_start_balance = balance
            self._week_start_ts      = now
            logger.info(f"Drawdown: weekly window reset — baseline ${balance:,.2f}")

    async def _check_drawdown(self) -> Optional[str]:
        """Return halt reason string if circuit breaker is tripped, else None."""
        if self._halted:
            return f"Trading halted: {self._halt_reason}"

        balance = await self._current_balance()

        # Seed baselines on first call
        if self._day_start_balance is None:
            self._day_start_balance  = balance
            self._week_start_balance = balance

        await self._reset_windows_if_due(balance)

        day_dd  = (self._day_start_balance  - balance) / max(self._day_start_balance,  1)
        week_dd = (self._week_start_balance - balance) / max(self._week_start_balance, 1)

        if day_dd >= config.daily_drawdown_limit:
            reason = (f"Daily drawdown {day_dd:.1%} ≥ limit {config.daily_drawdown_limit:.1%} "
                      f"— halting until window resets (~24 h)")
            logger.warning(reason)
            self._halted      = True
            self._halt_reason = reason
            return reason

        if week_dd >= config.weekly_drawdown_limit:
            reason = (f"Weekly drawdown {week_dd:.1%} ≥ limit {config.weekly_drawdown_limit:.1%} "
                      f"— halting until window resets (~7 d)")
            logger.warning(reason)
            self._halted      = True
            self._halt_reason = reason
            return reason

        # Auto-clear halt once the window resets (handled by _reset_windows_if_due)
        if self._halted:
            self._halted = False
            logger.info("Drawdown circuit breaker cleared — windows reset")

        return None

    # ── ATR-based position sizing ──────────────────────────────────────────────

    async def _size_from_atr(self, atr: float) -> float:
        """
        Risk exactly `atr_risk_pct` of current account balance per trade,
        using a stop distance of `atr_multiplier × ATR`.

        quote_size = min(
            (balance × atr_risk_pct) / (atr × atr_multiplier),
            max_position_usd
        )
        """
        if atr <= 0:
            return config.max_position_usd
        balance   = await self._current_balance()
        risk_usd  = balance * config.atr_risk_pct
        stop_dist = atr * config.atr_multiplier
        size      = risk_usd / stop_dist
        return min(size, config.max_position_usd)

    # ── Pre-flight ─────────────────────────────────────────────────────────────

    async def _preflight(self, quote_size: float) -> Optional[str]:
        if self.dry_run:
            if self._dry_run_balance < quote_size:
                return (f"DRY-RUN: Simulated balance ${self._dry_run_balance:.2f} "
                        f"< ${quote_size:.2f}")
            positions = await database.get_positions()
            exposure  = sum(p.get("current_value", 0) for p in positions)
            if exposure + quote_size > config.max_total_exposure:
                return (f"Exposure cap: ${exposure:.2f} + ${quote_size:.2f} "
                        f"> ${config.max_total_exposure:.2f}")
            return None

        if not config.has_credentials:
            return "No Coinbase API credentials configured"
        balance = await coinbase_client.get_usd_balance()
        if balance < quote_size:
            return f"Insufficient USD: ${balance:.2f} < ${quote_size:.2f}"
        positions = await database.get_positions()
        exposure  = sum(p.get("current_value", 0) for p in positions)
        if exposure + quote_size > config.max_total_exposure:
            return (f"Exposure cap: ${exposure:.2f} + ${quote_size:.2f} "
                    f"> ${config.max_total_exposure:.2f}")
        return None

    # ── Execute limit signal ───────────────────────────────────────────────────

    async def execute_signal(self, signal: Dict) -> Dict:
        """
        Execute a signal from SignalGenerator or CNNAgent.
        signal must have: product_id, side, price
        Optional: quote_size (overridden by ATR sizing when atr is present)
        """
        # 1 — Drawdown gate
        dd_err = await self._check_drawdown()
        if dd_err:
            return {"success": False, "reason": dd_err}

        pid   = signal["product_id"]
        side  = signal["side"].upper()   # BUY or SELL
        price = signal["price"]

        # 2 — ATR-based sizing (preferred) or fallback to signal/config default
        atr = signal.get("atr", 0.0) or 0.0
        if atr > 0:
            quote_size = await self._size_from_atr(atr)
            logger.debug(f"ATR sizing: ATR={atr:.4f} → ${quote_size:.2f} for {pid}")
        else:
            quote_size = signal.get("quote_size", config.max_position_usd)

        if quote_size < 1.0:
            return {"success": False, "reason": "Position below $1 minimum"}

        # 3 — Pre-flight balance / exposure
        error = await self._preflight(quote_size)
        if error:
            return {"success": False, "reason": error}

        # Convert USD → base currency size
        base_size = round(quote_size / price, 8)

        if self.dry_run:
            fake_id = f"DRY_{pid}_{side}_{uuid.uuid4().hex[:6]}"
            if side == "BUY":
                self._dry_run_balance -= quote_size
            logger.info(
                f"DRY-RUN: {side} {base_size} {pid} @ ${price:,.4f} (${quote_size:.2f})"
                f" | simulated balance: ${self._dry_run_balance:,.2f}"
            )
            await database.save_order({
                "order_id":   fake_id,
                "product_id": pid,
                "side":       side,
                "order_type": "LIMIT",
                "price":      price,
                "base_size":  base_size,
                "quote_size": quote_size,
                "status":     "dry_run",
                "strategy":   signal.get("signal_type", "TA"),
            })
            if signal.get("id"):
                await database.mark_signal_acted(signal["id"], fake_id)
            return {"success": True, "order_id": fake_id, "dry_run": True,
                    "simulated_balance": self._dry_run_balance}

        for attempt in range(3):
            try:
                resp     = await coinbase_client.place_limit_order(pid, side, base_size, price)
                order    = resp.get("success_response", resp)
                order_id = order.get("order_id") or order.get("client_order_id", "unknown")
                status   = "live" if resp.get("success") else "failed"

                await database.save_order({
                    "order_id":   order_id,
                    "product_id": pid,
                    "side":       side,
                    "order_type": "LIMIT",
                    "price":      price,
                    "base_size":  base_size,
                    "quote_size": quote_size,
                    "status":     status,
                    "strategy":   signal.get("signal_type", "TA"),
                })
                if signal.get("id"):
                    await database.mark_signal_acted(signal["id"], order_id)

                logger.info(f"ORDER: {side} {base_size} {pid} @ ${price:,.4f} → {order_id}")
                return {"success": True, "order_id": order_id, "status": status}

            except Exception as e:
                logger.error(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return {"success": False, "reason": "Order failed after 3 attempts"}

    # ── Execute market order ───────────────────────────────────────────────────

    async def execute_market_order(self, product_id: str, side: str,
                                    quote_size: float) -> Dict:
        """Immediate market order — pays spread, fills instantly."""
        # Drawdown gate
        dd_err = await self._check_drawdown()
        if dd_err:
            return {"success": False, "reason": dd_err}

        if self.dry_run:
            error = await self._preflight(quote_size)
            if error:
                return {"success": False, "reason": error}
            fake_id = f"DRY_MKT_{product_id}_{uuid.uuid4().hex[:6]}"
            if side.upper() == "BUY":
                self._dry_run_balance -= quote_size
            logger.info(
                f"DRY-RUN MARKET: {side} ${quote_size:.2f} of {product_id}"
                f" | simulated balance: ${self._dry_run_balance:,.2f}"
            )
            await database.save_order({
                "order_id":   fake_id,
                "product_id": product_id,
                "side":       side.upper(),
                "order_type": "MARKET",
                "quote_size": quote_size,
                "status":     "dry_run",
                "strategy":   "MANUAL_MARKET",
            })
            return {"success": True, "order_id": fake_id, "dry_run": True,
                    "simulated_balance": self._dry_run_balance}

        error = await self._preflight(quote_size)
        if error:
            return {"success": False, "reason": error}

        try:
            resp     = await coinbase_client.place_market_order(product_id, side, quote_size)
            order    = resp.get("success_response", resp)
            order_id = order.get("order_id", "unknown")
            await database.save_order({
                "order_id":   order_id,
                "product_id": product_id,
                "side":       side.upper(),
                "order_type": "MARKET",
                "quote_size": quote_size,
                "status":     "live",
                "strategy":   "MANUAL_MARKET",
            })
            return {"success": True, "order_id": order_id}
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return {"success": False, "reason": str(e)}

    # ── Cancel ─────────────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str) -> Dict:
        if self.dry_run:
            await database.update_order_status(order_id, "canceled")
            return {"success": True, "dry_run": True}
        try:
            resp = await coinbase_client.cancel_orders([order_id])
            await database.update_order_status(order_id, "canceled")
            logger.info(f"Canceled order {order_id}")
            return {"success": True, "response": resp}
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return {"success": False, "reason": str(e)}

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def drawdown_status(self) -> Dict:
        """Return current drawdown state for monitoring."""
        now = time.time()
        return {
            "halted":             self._halted,
            "halt_reason":        self._halt_reason,
            "day_start_balance":  self._day_start_balance,
            "week_start_balance": self._week_start_balance,
            "day_elapsed_pct":    min((now - self._day_start_ts)  / _SECS_PER_DAY,  1.0),
            "week_elapsed_pct":   min((now - self._week_start_ts) / _SECS_PER_WEEK, 1.0),
            "daily_limit":        config.daily_drawdown_limit,
            "weekly_limit":       config.weekly_drawdown_limit,
        }
