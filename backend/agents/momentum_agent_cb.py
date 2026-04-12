"""
MomentumAgent — Coinbase Trader
──────────────────────────────────────────────────────────────────────────────
Ported from trading_app/backend/agents/momentum_agent.py.
No pandas dependency — pure Python list arithmetic.

Scoring:
  5d ROC   up to 0.30   (short momentum, highest weight)
  10d ROC  up to 0.25   (mid momentum)
  20d ROC  up to 0.15   (long trend direction)
  Trend consistency  0.15   (fraction of days that closed higher)
  Momentum acceleration  0.10
  Volume ratio  0.05

BUY  threshold : score >= 0.30  AND  VW-momentum > 0
SELL threshold : score >= 0.30  OR   5d ROC < -2 %
Trailing stop : 3 % from high-water mark

Dry-run balance: $1,000  (mirrors CNN/OrderExecutor)
Decisions saved to `agent_decisions` table for CNN context.
"""
import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import database
from clients import coinbase_client
from agents.signal_generator import _adx, _mfi
from services.outcome_tracker import get_tracker

logger = logging.getLogger(__name__)

_DRY_RUN_BALANCE   = 1_000.0
_BUY_THRESHOLD     = 0.30
_SELL_THRESHOLD    = 0.30
_TRAILING_STOP     = 0.03    # 3 % trailing stop from high-water mark
_HARD_STOP_LOSS    = 0.05    # 5 % hard stop below avg entry price (real-time)
_TAKE_PROFIT       = 0.20    # 20 % take-profit above avg entry price (real-time)
_MOMENTUM_SHORT    = 5       # 5-bar ROC
_MOMENTUM_MID      = 10      # 10-bar ROC
_MOMENTUM_LONG     = 20      # 20-bar ROC
_MOMENTUM_THRESH   = 0.02    # 2 % minimum to count as "positive" momentum
_MAX_POSITION_FRAC = 0.15
_SCAN_INTERVAL     = 60      # 1 min — scan-based momentum scoring


# ── Momentum helpers ──────────────────────────────────────────────────────────

def _roc(closes: List[float], period: int) -> float:
    """Rate of change: (close[-1] - close[-period-1]) / close[-period-1]."""
    if len(closes) <= period:
        return 0.0
    prev = closes[-(period + 1)]
    return (closes[-1] - prev) / max(prev, 1e-9)


def _sma(values: List[float], period: int) -> float:
    w = values[-period:] if len(values) >= period else values
    return sum(w) / max(len(w), 1)


def _trend_consistency(closes: List[float], period: int = 20) -> float:
    """Fraction of last `period` bars that closed above the prior bar."""
    w = closes[-(period + 1):]
    if len(w) < 2:
        return 0.5
    pos = sum(1 for i in range(1, len(w)) if w[i] > w[i - 1])
    return pos / (len(w) - 1)


def _vw_momentum(closes: List[float], volumes: List[float], period: int = 5) -> float:
    """Volume-weighted average of per-bar returns over last `period` bars."""
    c = closes[-(period + 1):]
    v = volumes[-(period + 1):]
    n = min(len(c), len(v))
    if n < 2:
        return 0.0
    returns = [(c[i] - c[i - 1]) / max(c[i - 1], 1e-9) for i in range(1, n)]
    vols    = v[1:n]
    total_v = sum(vols)
    if total_v <= 0:
        return sum(returns) / len(returns)
    return sum(r * vw / total_v for r, vw in zip(returns, vols))


def _volume_ratio(volumes: List[float], recent: int = 5) -> float:
    if len(volumes) < recent + 1:
        return 1.0
    avg_recent = sum(volumes[-recent:]) / recent
    avg_all    = sum(volumes) / len(volumes)
    return avg_recent / max(avg_all, 1e-9)


# ── Dry-run book ──────────────────────────────────────────────────────────────

class _Book:
    def __init__(self, agent_name: str, balance: float = _DRY_RUN_BALANCE):
        self._agent      = agent_name
        self.balance     = balance
        self.positions: Dict[str, Dict] = {}
        self.realized_pnl = 0.0

    async def load(self, high_water_ref: Dict) -> None:
        """Restore state from DB, also repopulate high_water_ref dict."""
        state = await database.load_agent_state(self._agent)
        if state:
            self.balance      = state["balance"]
            self.realized_pnl = state["realized_pnl"]
            self.positions    = state["positions"]
            high_water_ref.update(state["high_water"])
            logger.info(
                f"{self._agent} state restored | balance=${self.balance:.2f} | "
                f"pnl=${self.realized_pnl:+.2f} | positions={len(self.positions)} | "
                f"high_water={len(high_water_ref)}"
            )
        else:
            logger.info(f"{self._agent} no saved state — starting fresh at ${self.balance:.2f}")

    async def _save(self, high_water: Dict) -> None:
        await database.save_agent_state(
            self._agent, self.balance, self.realized_pnl, self.positions, high_water
        )

    async def buy(self, pid: str, price: float, frac: float,
                  high_water: Dict, trigger: str = "SCAN") -> Tuple[float, float]:
        spend = min(self.balance * frac, self.balance * 0.95)
        if spend < 1.0 or price <= 0:
            return 0.0, 0.0
        size = spend / price
        if pid in self.positions:
            pos = self.positions[pid]
            tot = pos["size"] + size
            pos["avg_price"] = (pos["avg_price"] * pos["size"] + price * size) / tot
            pos["size"] = tot
        else:
            self.positions[pid] = {"size": size, "avg_price": price}
        self.balance -= spend
        await self._save(high_water)
        await database.open_trade(
            agent=self._agent, product_id=pid, entry_price=price,
            size=size, usd_open=spend, trigger_open=trigger,
            balance_after=self.balance,
        )
        return spend, size

    async def sell(self, pid: str, price: float, high_water: Dict,
                   trigger: str = "SCAN") -> float:
        if pid not in self.positions:
            return 0.0
        pos = self.positions.pop(pid)
        proceeds = pos["size"] * price
        pnl = proceeds - pos["size"] * pos["avg_price"]
        self.balance += proceeds
        self.realized_pnl += pnl
        await self._save(high_water)
        await database.close_trade(
            agent=self._agent, product_id=pid, exit_price=price,
            size=pos["size"], pnl=pnl, trigger_close=trigger,
            balance_after=self.balance,
        )
        return pnl

    def has_position(self, pid: str) -> bool:
        return pid in self.positions


# ── MomentumAgent ─────────────────────────────────────────────────────────────

class MomentumAgentCB:
    """
    Momentum-based trading agent for the Coinbase Trader.
    Tracks high-water marks per product and applies a 3 % trailing stop.
    Saves every decision to `agent_decisions` for CNN context injection.
    Accepts an optional ws_subscriber to read live tick prices.
    """

    def __init__(self, ws_subscriber=None):
        self.book = _Book("MOMENTUM")
        self.ws   = ws_subscriber   # CoinbaseWSSubscriber — live tick prices
        self._high_water: Dict[str, float] = {}  # pid → highest price since entry
        self._tick_locks: Dict[str, asyncio.Lock] = {}   # per-product tick lock
        self._score_cache: Dict[str, Dict] = {}          # pid → last scan scores
        self.scan_count   = 0
        self.signals_buy  = 0
        self.signals_sell = 0
        self.last_scan_at: Optional[float] = None
        logger.info(f"MomentumAgentCB ready | live_prices={'yes' if ws_subscriber else 'no (DB only)'}")

    async def start(self) -> None:
        """Load persisted state and register real-time price handler."""
        await self.book.load(self._high_water)
        if self.ws:
            self.ws.register_price_handler(self.on_price_tick)
            logger.info("MomentumAgentCB registered real-time price handler")

    def _get_lock(self, pid: str) -> asyncio.Lock:
        if pid not in self._tick_locks:
            self._tick_locks[pid] = asyncio.Lock()
        return self._tick_locks[pid]

    async def on_price_tick(self, pid: str, price: float) -> None:
        """
        Fires on every WS tick for pid.
        Checks trailing stop, hard stop-loss, take-profit, and cached momentum
        signals — executes immediately without waiting for the next scan cycle.
        """
        if not self.book.has_position(pid):
            # No position — check if we have a cached buy signal
            sc = self._score_cache.get(pid)
            if sc and sc["buy_score"] >= _BUY_THRESHOLD and sc["vw_mom"] > 0:
                lock = self._get_lock(pid)
                if lock.locked():
                    return
                async with lock:
                    if self.book.has_position(pid):  # re-check inside lock
                        return
                    frac = _MAX_POSITION_FRAC * sc["buy_score"]
                    self._high_water[pid] = price
                    spent, _ = await self.book.buy(pid, price, frac, self._high_water, trigger="TICK_SIGNAL")
                    if spent > 0:
                        self.signals_buy += 1
                        self._score_cache.pop(pid, None)   # consumed
                        reasoning = (
                            f"MOMENTUM BUY [TICK]: {'; '.join(sc['buy_reasons'])} "
                            f"score={sc['buy_score']:.2f}"
                        )
                        decision = {
                            "agent": "MOMENTUM", "product_id": pid, "side": "BUY",
                            "confidence": round(sc["buy_score"], 3),
                            "price": round(price, 6), "score": round(sc["buy_score"], 3),
                            "reasoning": reasoning,
                            "balance": round(self.book.balance, 2), "pnl": None,
                        }
                        await database.save_agent_decision(decision)
                        logger.info(
                            f"MomentumAgentCB TICK BUY {pid} @{price:.4f} "
                            f"score={sc['buy_score']:.2f} spent=${spent:.2f}"
                        )
            return

        # Have a position — evaluate exit conditions
        pos = self.book.positions.get(pid)
        if not pos:
            return

        lock = self._get_lock(pid)
        if lock.locked():
            return
        async with lock:
            if not self.book.has_position(pid):  # re-check inside lock
                return

            pos = self.book.positions.get(pid)
            if not pos:
                return

            avg_price = pos["avg_price"]
            pct       = (price - avg_price) / avg_price

            exit_reason  = None
            exit_trigger = "TICK_SIGNAL"

            # 1. Trailing stop (3% from high-water)
            if self._check_trailing_stop(pid, price):
                hw = self._high_water.get(pid, avg_price)
                exit_reason  = f"MOMENTUM TRAIL STOP [TICK]: fell from ${hw:.4f} ({pct*100:+.1f}%)"
                exit_trigger = "TICK_STOP"

            # 2. Hard stop-loss (5% below entry)
            elif pct < -_HARD_STOP_LOSS:
                exit_reason  = f"MOMENTUM HARD STOP [TICK]: {pct*100:.1f}% below entry ${avg_price:.4f}"
                exit_trigger = "TICK_STOP"

            # 3. Take-profit (20% above entry)
            elif pct > _TAKE_PROFIT:
                exit_reason  = f"MOMENTUM TAKE PROFIT [TICK]: +{pct*100:.1f}% above entry ${avg_price:.4f}"
                exit_trigger = "TICK_PROFIT"

            if exit_reason:
                pnl = await self.book.sell(pid, price, self._high_water, trigger=exit_trigger)
                self._high_water.pop(pid, None)
                self._score_cache.pop(pid, None)
                self.signals_sell += 1
                decision = {
                    "agent": "MOMENTUM", "product_id": pid, "side": "SELL",
                    "confidence": 0.95,
                    "price": round(price, 6), "score": 0.95,
                    "reasoning": f"{exit_reason} pnl=${pnl:+.2f}",
                    "balance": round(self.book.balance, 2), "pnl": round(pnl, 4),
                }
                await database.save_agent_decision(decision)
                logger.info(
                    f"MomentumAgentCB TICK SELL {pid} @{price:.4f} "
                    f"pnl={pnl:+.2f} | {exit_reason}"
                )
            else:
                # Update high-water mark even when not exiting
                if price > self._high_water.get(pid, 0):
                    self._high_water[pid] = price

    def _live_price(self, pid: str, fallback: float) -> float:
        if self.ws:
            p = self.ws.get_price(pid)
            if p and p > 0:
                return p
        return fallback

    # ── Trailing stop check ────────────────────────────────────────────────────

    def _check_trailing_stop(self, pid: str, price: float) -> bool:
        """Update high-water mark; return True if trailing stop triggered."""
        if not self.book.has_position(pid):
            return False
        if pid not in self._high_water or price > self._high_water[pid]:
            self._high_water[pid] = price
            return False
        drawdown = (self._high_water[pid] - price) / self._high_water[pid]
        if drawdown > _TRAILING_STOP:
            logger.info(
                f"MomentumAgentCB trailing stop {pid}: "
                f"fell {drawdown*100:.1f}% from ${self._high_water[pid]:.4f}"
            )
            return True
        return False

    # ── Scoring ────────────────────────────────────────────────────────────────

    def _score(self, closes: List[float], volumes: List[float],
               highs: List[float], lows: List[float]) -> Dict:
        mom_s = _roc(closes, _MOMENTUM_SHORT)
        mom_m = _roc(closes, _MOMENTUM_MID)
        mom_l = _roc(closes, _MOMENTUM_LONG)

        # Momentum acceleration: current 5d ROC vs previous 5d ROC
        prev_s = _roc(closes[:-1], _MOMENTUM_SHORT) if len(closes) > _MOMENTUM_SHORT + 1 else mom_s
        accel  = mom_s - prev_s

        consistency  = _trend_consistency(closes, _MOMENTUM_LONG)
        vw_mom       = _vw_momentum(closes, volumes, _MOMENTUM_SHORT)
        vol_ratio    = _volume_ratio(volumes)
        sma_s        = _sma(closes, _MOMENTUM_SHORT)
        sma_l        = _sma(closes, _MOMENTUM_LONG)
        ma_trend     = (sma_s - sma_l) / max(sma_l, 1e-9)

        # Regime and volume-direction indicators
        adx_val, _, _ = _adx(highs, lows, closes, period=14)
        mfi_val       = _mfi(highs, lows, closes, volumes, period=14)

        # BUY score
        buy_score   = 0.0
        buy_reasons = []

        if mom_s > _MOMENTUM_THRESH:
            w = min(0.30, mom_s * 5)
            buy_score += w; buy_reasons.append(f"5d mom={mom_s*100:.1f}%")

        if mom_m > _MOMENTUM_THRESH * 0.5:
            w = min(0.25, mom_m * 3)
            buy_score += w; buy_reasons.append(f"10d mom={mom_m*100:.1f}%")

        if mom_l > 0:
            w = min(0.15, mom_l * 2)
            buy_score += w; buy_reasons.append(f"20d mom={mom_l*100:.1f}%")

        if consistency > 0.6:
            buy_score += 0.15; buy_reasons.append(f"trend={consistency*100:.0f}%")

        if accel > 0 and mom_s > 0:
            buy_score += 0.10; buy_reasons.append("accelerating")

        if vol_ratio > 1.2:
            buy_score += 0.05; buy_reasons.append(f"vol ratio {vol_ratio:.1f}×")

        if mfi_val > 60:
            buy_score += 0.05; buy_reasons.append(f"MFI={mfi_val:.0f} buying pressure")

        # SELL score
        sell_score   = 0.0
        sell_reasons = []

        if mom_s < -_MOMENTUM_THRESH:
            sell_score += min(0.40, abs(mom_s) * 5)
            sell_reasons.append(f"5d mom={mom_s*100:.1f}%")

        if mom_m < -_MOMENTUM_THRESH * 0.5:
            sell_score += min(0.30, abs(mom_m) * 3)
            sell_reasons.append("10d reversal")

        if accel < -0.01:
            sell_score += 0.15; sell_reasons.append("decelerating")

        if consistency < 0.4:
            sell_score += 0.15; sell_reasons.append(f"weak trend={consistency*100:.0f}%")

        if mfi_val < 40:
            sell_score += 0.05; sell_reasons.append(f"MFI={mfi_val:.0f} selling pressure")

        return {
            "buy_score":    round(buy_score,  3),
            "sell_score":   round(sell_score, 3),
            "buy_reasons":  buy_reasons,
            "sell_reasons": sell_reasons,
            "vw_mom":       round(vw_mom, 6),
            "mom_s":        round(mom_s, 4),
            "mom_m":        round(mom_m, 4),
            "consistency":  round(consistency, 3),
            "adx":          round(adx_val, 1),
            "mfi":          round(mfi_val, 1),
        }

    # ── Single-product analyze ────────────────────────────────────────────────

    async def analyze_product(self, product: Dict) -> Optional[Dict]:
        pid   = product["product_id"]
        price = self._live_price(pid, product.get("price", 0))
        if not price or price <= 0:
            return None

        candles = await database.get_candles(pid, limit=80)
        if len(candles) < _MOMENTUM_LONG + 6:
            return None

        closes  = [c["close"]  for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]
        sc      = self._score(closes, volumes, highs, lows)

        # Cache for real-time tick handler to reference between scans
        self._score_cache[pid] = sc

        buy_score  = sc["buy_score"]
        sell_score = sc["sell_score"]
        has_pos    = self.book.has_position(pid)

        side       = "HOLD"
        confidence = 0.0
        pnl        = None
        reasoning  = ""

        # ── Trailing stop (takes priority) ────────────────────────────────────
        if has_pos and self._check_trailing_stop(pid, price):
            hw_price = self._high_water.get(pid, price)
            pnl = await self.book.sell(pid, price, self._high_water, trigger="SCAN_STOP")
            self._high_water.pop(pid, None)
            side       = "SELL"
            confidence = 0.90
            reasoning  = (
                f"MOMENTUM TRAIL STOP: fell from ${hw_price:.4f} "
                f"pnl=${pnl:+.2f}"
            )
            self.signals_sell += 1

        elif buy_score >= _BUY_THRESHOLD and not has_pos and sc["vw_mom"] > 0 and sc["adx"] >= 20:
            frac = _MAX_POSITION_FRAC * buy_score
            self._high_water[pid] = price
            spent, _ = await self.book.buy(pid, price, frac, self._high_water, trigger="SCAN")
            if spent > 0:
                side       = "BUY"
                confidence = buy_score
                reasoning  = f"MOMENTUM BUY: {'; '.join(sc['buy_reasons'])} score={buy_score:.2f}"
                self.signals_buy += 1
            else:
                self._high_water.pop(pid, None)

        elif (sell_score >= _SELL_THRESHOLD or sc["mom_s"] < -_MOMENTUM_THRESH) and has_pos:
            pnl = await self.book.sell(pid, price, self._high_water, trigger="SCAN")
            self._high_water.pop(pid, None)
            side       = "SELL"
            confidence = sell_score
            reasoning  = f"MOMENTUM SELL: {'; '.join(sc['sell_reasons'])} pnl=${pnl:+.2f}"
            self.signals_sell += 1

        else:
            confidence = max(buy_score, sell_score)
            reasoning  = (
                f"MOMENTUM HOLD mom5={sc['mom_s']*100:.1f}% "
                f"mom10={sc['mom_m']*100:.1f}% trend={sc['consistency']*100:.0f}% "
                f"ADX={sc['adx']:.0f} MFI={sc['mfi']:.0f} "
                f"buy={buy_score:.2f} sell={sell_score:.2f}"
            )

        decision = {
            "agent":      "MOMENTUM",
            "product_id": pid,
            "side":       side,
            "confidence": round(confidence, 3),
            "price":      round(price, 6),
            "score":      round(max(buy_score, sell_score), 3),
            "reasoning":  reasoning,
            "balance":    round(self.book.balance, 2),
            "pnl":        round(pnl, 4) if pnl is not None else None,
        }
        await database.save_agent_decision(decision)

        # ── Outcome tracker: record + Ollama validation for BUY/SELL ────────────
        if side in ("BUY", "SELL"):
            indicators = {
                "mom_s":       sc["mom_s"],
                "mom_m":       sc["mom_m"],
                "consistency": sc["consistency"],
                "vw_mom":      sc["vw_mom"],
                "adx":         sc["adx"],
                "mfi":         sc["mfi"],
            }
            tracker = get_tracker()
            await tracker.record(
                source="MOMENTUM", product_id=pid, side=side,
                confidence=confidence, price=price, indicators=indicators,
            )
            val_prob = await tracker.validate_with_ollama(
                source="MOMENTUM", product_id=pid, side=side,
                confidence=confidence, price=price, indicators=indicators,
            )
            if val_prob is not None:
                logger.info(
                    f"MomentumAgent Ollama validation {pid} {side}: "
                    f"p={val_prob:.3f} (mom5d={sc['mom_s']*100:.1f}% "
                    f"trend={sc['consistency']*100:.0f}% ADX={sc['adx']:.0f} MFI={sc['mfi']:.0f})"
                )

        return decision if side != "HOLD" else None

    # ── Scan loop ─────────────────────────────────────────────────────────────

    async def scan_all(self) -> List[Dict]:
        products = await database.get_products(tracked_only=True)
        signals  = []
        for p in products:
            try:
                sig = await self.analyze_product(p)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"MomentumAgentCB error [{p.get('product_id')}]: {e}")
            await asyncio.sleep(0.05)
        return signals

    async def run_loop(self, interval: int = _SCAN_INTERVAL) -> None:
        await self.start()   # restore persisted state before first scan
        logger.info(
            f"MomentumAgentCB loop started | interval={interval}s | "
            f"balance=${self.book.balance:.2f} | trailing_stop={_TRAILING_STOP*100:.0f}%"
        )
        while True:
            try:
                sigs = await self.scan_all()
                self.last_scan_at = time.time()
                self.scan_count  += 1
                logger.info(
                    f"MomentumAgentCB scan #{self.scan_count} done | "
                    f"signals={len(sigs)} | balance=${self.book.balance:.2f} | "
                    f"pnl=${self.book.realized_pnl:+.2f}"
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"MomentumAgentCB loop error: {e}")
            await asyncio.sleep(interval)

    @property
    def status(self) -> Dict:
        return {
            "agent":          "MOMENTUM",
            "balance":        round(self.book.balance, 2),
            "realized_pnl":   round(self.book.realized_pnl, 2),
            "open_positions":  len(self.book.positions),
            "positions":      {
                pid: {
                    "size":        round(p["size"], 6),
                    "avg_price":   round(p["avg_price"], 6),
                    "high_water":  round(self._high_water.get(pid, p["avg_price"]), 6),
                }
                for pid, p in self.book.positions.items()
            },
            "trailing_stops":  len(self._high_water),
            "scan_count":     self.scan_count,
            "signals_buy":    self.signals_buy,
            "signals_sell":   self.signals_sell,
            "last_scan_at":   self.last_scan_at,
        }
