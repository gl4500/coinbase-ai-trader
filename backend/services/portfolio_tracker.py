"""
Portfolio Tracker
──────────────────
Syncs open positions with Coinbase account balances.
Computes unrealised P&L using live WS prices.
Runs every 60 seconds as a background task.
"""
import asyncio
import logging
from typing import Dict, Optional

import database
from clients import coinbase_client
from config import config
from services.ws_subscriber import CoinbaseWSSubscriber

logger = logging.getLogger(__name__)


class PortfolioTracker:
    def __init__(self, ws_subscriber: Optional[CoinbaseWSSubscriber] = None):
        self.ws  = ws_subscriber
        self.summary: Dict = {
            "open_positions": 0,
            "total_value":    0.0,
            "total_cost":     0.0,
            "total_pnl":      0.0,
            "pct_pnl":        0.0,
        }

    async def sync(self) -> Dict:
        """Pull balances from Coinbase, update positions and P&L in DB."""
        try:
            accounts = await coinbase_client.get_accounts()
        except Exception as e:
            logger.warning(f"Portfolio sync failed — Coinbase accounts error: {e}")
            self.summary = await database.get_portfolio_summary()
            return self.summary

        for acct in accounts:
            currency = acct["currency"]
            if currency in ("USD", "USDC", "USDT"):
                continue   # skip fiat/stable — not a position

            product_id = f"{currency}-USD"
            size       = acct["available"] + acct.get("hold", 0)
            if size <= 0:
                await database.delete_position(product_id)
                continue

            # Get current price from WS state or DB
            price: Optional[float] = None
            if self.ws:
                price = self.ws.get_price(product_id)
            if price is None:
                prod = await database.get_product(product_id)
                price = prod.get("price") if prod else None
            if price is None:
                continue

            # Reconstruct position from DB for cost basis
            existing = await database.get_product(product_id)
            avg_price = price  # fallback: assume bought at current price (no P&L)

            # Try to get avg_price from our order history
            orders = await database.get_orders()
            product_orders = [
                o for o in orders
                if o["product_id"] == product_id
                and o["side"] == "BUY"
                and o.get("avg_fill_price")
            ]
            if product_orders:
                total_spent = sum(
                    (o.get("avg_fill_price", 0) * o.get("filled_size", 0))
                    for o in product_orders
                )
                total_size  = sum(o.get("filled_size", 0) for o in product_orders)
                if total_size > 0:
                    avg_price = total_spent / total_size

            initial_value = size * avg_price
            current_value = size * price
            cash_pnl      = current_value - initial_value
            pct_pnl       = (cash_pnl / initial_value * 100) if initial_value > 0 else 0.0

            await database.upsert_position({
                "product_id":    product_id,
                "base_currency": currency,
                "side":          "BUY",
                "size":          size,
                "avg_price":     avg_price,
                "current_price": price,
                "initial_value": round(initial_value, 2),
                "current_value": round(current_value, 2),
                "cash_pnl":      round(cash_pnl, 2),
                "pct_pnl":       round(pct_pnl, 2),
            })

        self.summary = await database.get_portfolio_summary()
        return self.summary

    async def run_loop(self, interval: int = 60) -> None:
        logger.info("Portfolio tracker started")
        while True:
            try:
                await self.sync()
            except asyncio.CancelledError:
                logger.info("Portfolio tracker cancelled")
                return
            except Exception as e:
                logger.error(f"Portfolio tracker error: {e}")
            await asyncio.sleep(interval)
