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
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

import database
from clients import coinbase_client
from config import config

logger = logging.getLogger(__name__)


_DRY_RUN_BALANCE = 1_000.0  # simulated USD starting balance for dry-run

_SECS_PER_DAY = 86_400
_SECS_PER_WEEK = 604_800


def _maker_price(side: str, bid: float, ask: float) -> float:
    """Post-only price for a maker entry: bid for BUY, ask for SELL.

    Posting AT the touch rests in the book and gets a maker fill once a
    taker on the other side hits it. Crossing the spread (e.g. BUY at ask)
    would auto-cancel the post_only order on Coinbase.
    """
    return bid if side.upper() == "BUY" else ask


# An exchange may name an order, refuse it, or answer ambiguously. Those are
# three outcomes, not two, and collapsing the third into either of the others is
# the defect this module was repaired for: an ambiguous answer means an order MAY
# exist, so it can neither be reported as success nor retried.
_PLACEHOLDER_ORDER_IDS = frozenset({"unknown", "none", "null", "n/a", "na", "-", "?"})


def _usable_order_id(value: Any) -> Optional[str]:
    """Return a real exchange identifier, or None.

    `"unknown"` is an error path wearing an identifier's clothes — it was the
    literal default here, persisted to the orders table as though it named
    something. A non-string, blank, or placeholder value identifies nothing.
    """
    if not isinstance(value, str):
        return None
    ident = value.strip()
    if not ident or ident.lower() in _PLACEHOLDER_ORDER_IDS:
        return None
    return ident


def _accepted_placement(resp: Any) -> Optional[str]:
    """The order id of an AFFIRMATIVELY accepted placement, else None.

    Requires the response to say `success is True` — an actual bool, since `1`
    and `"true"` are not the exchange agreeing — and to name the order in its
    success payload. Anything else returns None and the caller must decide
    between rejection and ambiguity via `_explicit_rejection`.
    """
    if not isinstance(resp, dict) or resp.get("success") is not True:
        return None
    order = resp.get("success_response")
    if not isinstance(order, dict):
        return None
    return _usable_order_id(order.get("order_id"))


def _explicit_rejection(resp: Any) -> bool:
    """True only when the exchange affirmatively said no.

    A rejection is safe: nothing was placed. Silence, a malformed body, or a
    success flag with no usable id are NOT rejections — an order may exist — so
    they must never be reported as a clean failure or retried.
    """
    return isinstance(resp, dict) and resp.get("success") is False


async def _persist_accepted_order(row: Dict, signal_id: Any = None) -> Optional[str]:
    """Record an ACCEPTED placement. Returns None on success, else a failure reason.

    Once a placement is accepted a real order exists at the exchange, so the
    accepted identifier is the most valuable thing the caller can be given. An
    unguarded write discarded it: the exception propagated and took the id with
    it, leaving an order nothing names. Reporting "persistence failed, here is
    the id" is strictly better than raising, because the id is what makes the
    exposure reconcilable.
    """
    order_id = row["order_id"]
    try:
        await database.save_order(row)
    except Exception as e:
        logger.error("Order %s was accepted but could not be recorded: %s", order_id, e)
        return f"Order was accepted but could not be recorded: {e}"
    if signal_id:
        try:
            await database.mark_signal_acted(signal_id, order_id)
        except Exception as e:
            logger.error("Order %s recorded but signal link failed: %s", order_id, e)
            return f"Order was accepted but the signal link failed: {e}"
    return None


def _acknowledged_cancel(resp: Any, order_id: str) -> bool:
    """True when the exchange acknowledged the cancel request for THIS order.

    Acknowledgement is not terminality: it says the request was received, not
    that the order left the book. `_confirmed_unfilled_cancel` decides that,
    from a follow-up snapshot.
    """
    if not isinstance(resp, dict):
        return False
    results = resp.get("results")
    if not isinstance(results, list):
        return False
    matches = [
        r
        for r in results
        if isinstance(r, dict) and _usable_order_id(r.get("order_id")) == order_id
    ]
    return len(matches) == 1 and matches[0].get("success") is True


def _confirmed_unfilled_cancel(order: Dict, order_id: str) -> bool:
    if (
        order.get("order_id") != order_id
        or order.get("status") not in {"CANCELLED", "CANCELED"}
        or order.get("pending_cancel") is not False
    ):
        return False
    try:
        sizes = [Decimal(str(order[key])) for key in ("filled_size", "filled_value")]
    except (KeyError, InvalidOperation, ValueError):
        return False
    return all(size.is_finite() and size == 0 for size in sizes)


class OrderExecutor:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._dry_run_balance = _DRY_RUN_BALANCE

        # ── Drawdown tracking ──────────────────────────────────────────────────
        self._day_start_balance: Optional[float] = None
        self._week_start_balance: Optional[float] = None
        self._day_start_ts: float = time.time()
        self._week_start_ts: float = time.time()
        self._halted: bool = False
        self._halt_reason: str = ""

        if dry_run:
            self._day_start_balance = _DRY_RUN_BALANCE
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
            self._day_start_ts = now
            logger.info(f"Drawdown: daily window reset — baseline ${balance:,.2f}")
        if now - self._week_start_ts >= _SECS_PER_WEEK:
            self._week_start_balance = balance
            self._week_start_ts = now
            logger.info(f"Drawdown: weekly window reset — baseline ${balance:,.2f}")

    async def _check_drawdown(self) -> Optional[str]:
        """Return halt reason string if circuit breaker is tripped, else None."""
        if self._halted:
            return f"Trading halted: {self._halt_reason}"

        balance = await self._current_balance()

        # Seed baselines on first call
        if self._day_start_balance is None:
            self._day_start_balance = balance
            self._week_start_balance = balance

        await self._reset_windows_if_due(balance)

        day_dd = (self._day_start_balance - balance) / max(self._day_start_balance, 1)
        week_dd = (self._week_start_balance - balance) / max(self._week_start_balance, 1)

        if day_dd >= config.daily_drawdown_limit:
            reason = (
                f"Daily drawdown {day_dd:.1%} ≥ limit {config.daily_drawdown_limit:.1%} "
                f"— halting until window resets (~24 h)"
            )
            logger.warning(reason)
            self._halted = True
            self._halt_reason = reason
            return reason

        if week_dd >= config.weekly_drawdown_limit:
            reason = (
                f"Weekly drawdown {week_dd:.1%} ≥ limit {config.weekly_drawdown_limit:.1%} "
                f"— halting until window resets (~7 d)"
            )
            logger.warning(reason)
            self._halted = True
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
        balance = await self._current_balance()
        risk_usd = balance * config.atr_risk_pct
        stop_dist = atr * config.atr_multiplier
        size = risk_usd / stop_dist
        return min(size, config.max_position_usd)

    # ── Pre-flight ─────────────────────────────────────────────────────────────

    async def _preflight(self, quote_size: float) -> Optional[str]:
        if self.dry_run:
            if self._dry_run_balance < quote_size:
                return (
                    f"DRY-RUN: Simulated balance ${self._dry_run_balance:.2f} < ${quote_size:.2f}"
                )
            positions = await database.get_positions()
            exposure = sum(p.get("current_value", 0) for p in positions)
            if exposure + quote_size > config.max_total_exposure:
                return (
                    f"Exposure cap: ${exposure:.2f} + ${quote_size:.2f} "
                    f"> ${config.max_total_exposure:.2f}"
                )
            return None

        if not config.has_credentials:
            return "No Coinbase API credentials configured"
        balance = await coinbase_client.get_usd_balance()
        if balance < quote_size:
            return f"Insufficient USD: ${balance:.2f} < ${quote_size:.2f}"
        positions = await database.get_positions()
        exposure = sum(p.get("current_value", 0) for p in positions)
        if exposure + quote_size > config.max_total_exposure:
            return (
                f"Exposure cap: ${exposure:.2f} + ${quote_size:.2f} "
                f"> ${config.max_total_exposure:.2f}"
            )
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

        pid = signal["product_id"]
        side = signal["side"].upper()  # BUY or SELL
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
            await database.save_order(
                {
                    "order_id": fake_id,
                    "product_id": pid,
                    "side": side,
                    "order_type": "LIMIT",
                    "price": price,
                    "base_size": base_size,
                    "quote_size": quote_size,
                    "status": "dry_run",
                    "strategy": signal.get("signal_type", "TA"),
                }
            )
            if signal.get("id"):
                await database.mark_signal_acted(signal["id"], fake_id)
            return {
                "success": True,
                "order_id": fake_id,
                "dry_run": True,
                "simulated_balance": self._dry_run_balance,
            }

        # NO RETRY LOOP. An exception can be raised after the request reached the
        # exchange, so a retry may place a SECOND order for the same signal. The
        # previous three-attempt loop treated every failure as "nothing
        # happened"; an unconfirmed placement is reconciliation work, not a
        # reason to submit again.
        try:
            resp = await coinbase_client.place_limit_order(pid, side, base_size, price)
        except Exception as e:
            logger.error("Order placement for %s is unconfirmed, not resubmitting: %s", pid, e)
            return {
                "success": False,
                "reconciliation_required": True,
                "reason": f"Order placement unconfirmed: {e}",
            }

        order_id = _accepted_placement(resp)
        if order_id is None:
            if _explicit_rejection(resp):
                logger.warning("Exchange rejected the %s order for %s", side, pid)
                return {"success": False, "reason": "Exchange rejected the order"}
            logger.error(
                "Order placement for %s was neither accepted nor rejected; "
                "an order may exist and no replacement will be submitted",
                pid,
            )
            return {
                "success": False,
                "reconciliation_required": True,
                "reason": "Order placement was neither accepted nor rejected",
            }

        failure = await _persist_accepted_order(
            {
                "order_id": order_id,
                "product_id": pid,
                "side": side,
                "order_type": "LIMIT",
                "price": price,
                "base_size": base_size,
                "quote_size": quote_size,
                "status": "live",
                "strategy": signal.get("signal_type", "TA"),
            },
            signal.get("id"),
        )
        if failure:
            # The order exists. Hand back its id rather than an exception, and
            # place nothing else.
            return {
                "success": False,
                "order_id": order_id,
                "reconciliation_required": True,
                "reason": failure,
            }

        logger.info(f"ORDER: {side} {base_size} {pid} @ ${price:,.4f} → {order_id}")
        return {"success": True, "order_id": order_id, "status": "live"}

    # ── Maker (post-only LIMIT) entry with timeout fallback ────────────────────

    async def _wait_for_fill(
        self,
        product_id: str,
        order_id: str,
        timeout_secs: float,
    ) -> bool:
        """Poll get_orders for `order_id` until status==FILLED or deadline.
        Returns True on fill, False on timeout."""
        deadline = time.time() + max(0.0, float(timeout_secs))
        while True:
            try:
                orders = await coinbase_client.get_orders(product_id=product_id)
            except Exception as e:
                logger.warning(f"_wait_for_fill: get_orders failed: {e}")
                orders = []
            for o in orders:
                if o.get("order_id") == order_id and str(o.get("status", "")).upper() == "FILLED":
                    return True
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(0.5, max(0.01, remaining / 4)))

    async def execute_maker_signal(
        self,
        signal: Dict,
        timeout_secs: float = 30.0,
    ) -> Dict:
        """Maker order with a reconciled, zero-fill-only market replacement.

        Cuts the entry leg from taker (~0.60% on tier 0) to maker (~0.20%) by
        posting at the bid (BUY) / ask (SELL). If the resting limit doesn't
        fill within `timeout_secs`, request cancellation and reconcile its final
        state. Only a confirmed zero-fill cancellation permits a market replacement.
        Partial or unknown fills require reconciliation instead of an automatic top-up. Purely additive — no caller is migrated until the
        user opts in.

        Signal must have: product_id, side, bid, ask. Optional: atr,
        quote_size, signal_type, id.
        """
        # 1 — Drawdown gate
        dd_err = await self._check_drawdown()
        if dd_err:
            return {"success": False, "reason": dd_err}

        pid = signal["product_id"]
        side = signal["side"].upper()
        bid = float(signal.get("bid", 0.0) or 0.0)
        ask = float(signal.get("ask", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            return {"success": False, "reason": "Maker path requires bid + ask quotes"}

        maker_price = _maker_price(side, bid, ask)

        # 2 — ATR sizing or signal default
        atr = signal.get("atr", 0.0) or 0.0
        if atr > 0:
            quote_size = await self._size_from_atr(atr)
        else:
            quote_size = signal.get("quote_size", config.max_position_usd)

        if quote_size < 1.0:
            return {"success": False, "reason": "Position below $1 minimum"}

        # 3 — Pre-flight
        error = await self._preflight(quote_size)
        if error:
            return {"success": False, "reason": error}

        base_size = round(quote_size / maker_price, 8)

        # 4 — Dry-run short-circuit
        if self.dry_run:
            fake_id = f"DRY_MAKER_{pid}_{side}_{uuid.uuid4().hex[:6]}"
            if side == "BUY":
                self._dry_run_balance -= quote_size
            logger.info(
                f"DRY-RUN MAKER: {side} {base_size} {pid} @ ${maker_price:,.4f}"
                f" (quote ${quote_size:.2f}) | sim balance: ${self._dry_run_balance:,.2f}"
            )
            await database.save_order(
                {
                    "order_id": fake_id,
                    "product_id": pid,
                    "side": side,
                    "order_type": "LIMIT_MAKER",
                    "price": maker_price,
                    "base_size": base_size,
                    "quote_size": quote_size,
                    "status": "dry_run",
                    "strategy": signal.get("signal_type", "TA"),
                }
            )
            if signal.get("id"):
                await database.mark_signal_acted(signal["id"], fake_id)
            return {
                "success": True,
                "order_id": fake_id,
                "fill_mode": "MAKER",
                "dry_run": True,
                "simulated_balance": self._dry_run_balance,
            }

        # 5 — Live: place post-only LIMIT at maker price
        try:
            resp = await coinbase_client.place_limit_order(
                pid,
                side,
                base_size,
                maker_price,
                post_only=True,
            )
        except Exception as e:
            logger.error(f"Maker LIMIT placement failed: {e}")
            return {
                "success": False,
                "reason": f"Limit placement failed: {e}",
                "reconciliation_required": True,
            }

        order_id = _accepted_placement(resp)
        if order_id is None:
            return {
                "success": False,
                "reason": "Maker placement was rejected or unconfirmed",
                "reconciliation_required": not _explicit_rejection(resp),
            }
        failure = await _persist_accepted_order(
            {
                "order_id": order_id,
                "product_id": pid,
                "side": side,
                "order_type": "LIMIT_MAKER",
                "price": maker_price,
                "base_size": base_size,
                "quote_size": quote_size,
                "status": "live",
                "strategy": signal.get("signal_type", "TA"),
            },
            signal.get("id"),
        )
        if failure:
            # Stop before the fill poll and the market fallback: replacing an
            # order we could not record is how one signal becomes two positions.
            return {
                "success": False,
                "order_id": order_id,
                "maker_order_id": order_id,
                "fill_mode": "RECONCILIATION_REQUIRED",
                "reconciliation_required": True,
                "reason": failure,
            }
        logger.info(f"MAKER ORDER: {side} {base_size} {pid} @ ${maker_price:,.4f} → {order_id}")

        # 6 — Poll for fill within timeout
        filled = await self._wait_for_fill(pid, order_id, timeout_secs)
        if filled:
            return {"success": True, "order_id": order_id, "fill_mode": "MAKER"}

        # A cancel response acknowledges the request, not a final zero-fill state.
        unresolved = {
            "success": False,
            "order_id": order_id,
            "maker_order_id": order_id,
            "fill_mode": "RECONCILIATION_REQUIRED",
            "reconciliation_required": True,
        }
        try:
            cancel = await coinbase_client.cancel_orders([order_id])
            if not _acknowledged_cancel(cancel, order_id):
                return {**unresolved, "reason": "Maker cancellation was not acknowledged"}
            final = await coinbase_client.get_order(order_id)
            # Identity BEFORE evidence, as in cancel_order: a snapshot naming a
            # different order says nothing about this one, and attaching its
            # fills here would cite another order's quantities as this order's
            # reconciliation evidence.
            if not isinstance(final, dict) or _usable_order_id(final.get("order_id")) != order_id:
                logger.error(
                    "Maker cancellation status for %s named a different order; evidence discarded",
                    order_id,
                )
                return {**unresolved, "reason": "Maker cancellation status named a different order"}
            if final.get("status") == "FILLED":
                await database.update_order_status(order_id, "filled")
                return {"success": True, "order_id": order_id, "fill_mode": "MAKER"}
            if not _confirmed_unfilled_cancel(final, order_id):
                logger.error(
                    "Maker order %s requires reconciliation; no replacement submitted", order_id
                )
                return {
                    **unresolved,
                    "reason": "Maker cancellation/fills require reconciliation",
                    "filled_size": final.get("filled_size"),
                    "filled_value": final.get("filled_value"),
                }
            await database.update_order_status(order_id, "canceled")
        except Exception as e:
            logger.error("Maker cancellation reconciliation failed for %s: %s", order_id, e)
            return {**unresolved, "reason": f"Maker cancellation reconciliation failed: {e}"}

        mkt_id = None
        try:
            if side == "SELL":
                mkt_resp = await coinbase_client.place_market_order(pid, side, base_size=base_size)
            else:
                mkt_resp = await coinbase_client.place_market_order(pid, side, quote_size)
            mkt_id = _accepted_placement(mkt_resp)
            if mkt_id is None:
                return {**unresolved, "reason": "Market replacement rejected or unconfirmed"}
            await database.save_order(
                {
                    "order_id": mkt_id,
                    "product_id": pid,
                    "side": side,
                    "order_type": "MARKET",
                    "base_size": base_size if side == "SELL" else None,
                    "quote_size": quote_size,
                    "status": "live",
                    "strategy": signal.get("signal_type", "TA") + "_TAKER_FALLBACK",
                }
            )
            return {
                "success": True,
                "order_id": mkt_id,
                "maker_order_id": order_id,
                "fill_mode": "TAKER_FALLBACK",
                "status": "submitted",
            }
        except Exception as e:
            logger.error(f"Market fallback failed: {e}")
            return {
                **unresolved,
                "order_id": mkt_id or order_id,
                "reason": f"Maker timed out and market fallback failed: {e}",
            }

    # ── Execute market order ───────────────────────────────────────────────────

    async def execute_market_order(self, product_id: str, side: str, quote_size: float) -> Dict:
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
            await database.save_order(
                {
                    "order_id": fake_id,
                    "product_id": product_id,
                    "side": side.upper(),
                    "order_type": "MARKET",
                    "quote_size": quote_size,
                    "status": "dry_run",
                    "strategy": "MANUAL_MARKET",
                }
            )
            return {
                "success": True,
                "order_id": fake_id,
                "dry_run": True,
                "simulated_balance": self._dry_run_balance,
            }

        error = await self._preflight(quote_size)
        if error:
            return {"success": False, "reason": error}

        try:
            resp = await coinbase_client.place_market_order(product_id, side, quote_size)
        except Exception as e:
            logger.error("Market order for %s is unconfirmed: %s", product_id, e)
            return {
                "success": False,
                "reconciliation_required": True,
                "reason": f"Market order unconfirmed: {e}",
            }

        order_id = _accepted_placement(resp)
        if order_id is None:
            if _explicit_rejection(resp):
                logger.warning("Exchange rejected the %s market order for %s", side, product_id)
                return {"success": False, "reason": "Exchange rejected the order"}
            # The old code read the FULL response as the order payload, so a
            # failure body produced order_id "unknown" with status "live" and a
            # success result. Nothing is persisted unless the exchange named it.
            logger.error(
                "Market order for %s was neither accepted nor rejected; an order may exist",
                product_id,
            )
            return {
                "success": False,
                "reconciliation_required": True,
                "reason": "Market order was neither accepted nor rejected",
            }

        failure = await _persist_accepted_order(
            {
                "order_id": order_id,
                "product_id": product_id,
                "side": side.upper(),
                "order_type": "MARKET",
                "quote_size": quote_size,
                "status": "live",
                "strategy": "MANUAL_MARKET",
            }
        )
        if failure:
            return {
                "success": False,
                "order_id": order_id,
                "reconciliation_required": True,
                "reason": failure,
            }
        return {"success": True, "order_id": order_id}

    # ── Cancel ─────────────────────────────────────────────────────────────────

    async def cancel_order(self, order_id: str) -> Dict:
        """Cancel an order, recording "canceled" only on TERMINAL confirmation.

        Three states must stay distinct (lifecycle contract §4): the request was
        acknowledged, the order is terminally cancelled with no fills, and the
        order reached some other terminal state. The previous implementation
        persisted "canceled" whenever no exception was raised — it never read the
        per-order results, so a refused cancel was recorded as a completed one
        and a partially filled order had its fill erased.
        """
        if self.dry_run:
            await database.update_order_status(order_id, "canceled")
            return {"success": True, "dry_run": True}

        unresolved = {
            "success": False,
            "order_id": order_id,
            "reconciliation_required": True,
        }

        try:
            resp = await coinbase_client.cancel_orders([order_id])
        except Exception as e:
            logger.error("Cancel request for %s failed: %s", order_id, e)
            return {**unresolved, "reason": f"Cancel request failed: {e}"}

        if not _acknowledged_cancel(resp, order_id):
            # Not acknowledged for this exact order: it may still be live, or it
            # may already be terminal. Either way the state is unknown.
            logger.error("Cancellation of %s was not acknowledged for that order", order_id)
            return {**unresolved, "reason": "Cancellation was not acknowledged", "response": resp}

        try:
            final = await coinbase_client.get_order(order_id)
        except Exception as e:
            logger.error("Cancellation of %s acknowledged but unverifiable: %s", order_id, e)
            return {**unresolved, "reason": f"Cancellation status unavailable: {e}"}

        if not isinstance(final, dict):
            return {**unresolved, "reason": "Cancellation status was malformed"}

        # Identity FIRST. A snapshot naming another order is not weak evidence
        # about this one, it is evidence about something else: attaching its
        # status or fills here would manufacture a reconciliation record citing
        # quantities that belong to a different order.
        if _usable_order_id(final.get("order_id")) != order_id:
            logger.error(
                "Cancellation status for %s named a different order; evidence discarded",
                order_id,
            )
            return {**unresolved, "reason": "Cancellation status named a different order"}

        fills = {
            "filled_size": final.get("filled_size"),
            "filled_value": final.get("filled_value"),
        }
        status = str(final.get("status", "")).upper()

        if status == "FILLED":
            # The cancel lost the race. The fill is the truth; recording a
            # cancellation here would erase a real position.
            await database.update_order_status(order_id, "filled")
            logger.warning("Cancellation of %s lost the race to a fill", order_id)
            return {
                **unresolved,
                "status": "FILLED",
                "reason": "Cancellation lost the race to a fill",
                **fills,
            }

        if not _confirmed_unfilled_cancel(final, order_id):
            # Includes the partial fill: a settled partial is not a zero-fill
            # cancellation, and its fill must survive in the result.
            logger.error("Cancellation of %s is not terminally confirmed", order_id)
            return {
                **unresolved,
                "status": status or None,
                "reason": "Cancellation is not terminally confirmed",
                **fills,
            }

        await database.update_order_status(order_id, "canceled")
        logger.info("Canceled order %s (terminally confirmed, zero fills)", order_id)
        return {"success": True, "order_id": order_id, "status": "CANCELLED", "response": resp}

    # ── Status ─────────────────────────────────────────────────────────────────

    @property
    def drawdown_status(self) -> Dict:
        """Return current drawdown state for monitoring."""
        now = time.time()
        return {
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "day_start_balance": self._day_start_balance,
            "week_start_balance": self._week_start_balance,
            "day_elapsed_pct": min((now - self._day_start_ts) / _SECS_PER_DAY, 1.0),
            "week_elapsed_pct": min((now - self._week_start_ts) / _SECS_PER_WEEK, 1.0),
            "daily_limit": config.daily_drawdown_limit,
            "weekly_limit": config.weekly_drawdown_limit,
        }
