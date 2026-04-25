"""
TechAgent — Coinbase Trader
──────────────────────────────────────────────────────────────────────────────
Ported from trading_app/backend/agents/tech_agent.py.
No pandas / pandas-ta dependency — uses signal_generator indicator functions.

Scoring:
  RSI          0.35  (oversold BUY / overbought SELL)
  Bollinger    0.30  (price at lower/upper band)
  MACD         0.25  (crossover 0.25 / direction 0.10)
  Volume       0.10  (spike confirmation)
  Stochastic   0.15  (zone entry/exit timing)
  OBV          0.10-0.12 (divergence / confirmation)

BUY  threshold : score >= 0.55
SELL threshold : score >= 0.55

Dry-run balance: $1,000  (mirrors CNN/OrderExecutor)
Decisions saved to `agent_decisions` table so the CNN can read them.
"""
import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import database
from clients import coinbase_client
from agents.signal_generator import _rsi, _macd, _bollinger, _atr, _kelly_fraction
from services.outcome_tracker import get_tracker
from services.macro_signals import get_macro_service, MacroContext

logger = logging.getLogger(__name__)

_DRY_RUN_BALANCE   = 1_000.0
_BUY_THRESHOLD     = 0.55
_SELL_THRESHOLD    = 0.55
_MAX_POSITION_FRAC = 0.15    # max 15 % of portfolio per product (fallback cap)
MIN_PRICE          = 0.01    # skip micro-priced tokens (unprofitable spreads)
_TAKE_PROFIT       = 0.06    # 6 % take-profit above avg entry price (real-time)
_SCAN_INTERVAL     = 120     # 2 min — pure math, no Ollama dependency

# ATR-based trailing stop (replaces fixed _HARD_STOP_LOSS)
_ATR_MULTIPLIER = 3.0    # stop distance = ATR × multiplier
_ATR_STOP_MIN   = 0.015  # floor: never tighter than 1.5 % (prevent stop-hunting)
_ATR_STOP_MAX   = 0.12   # ceiling: never wider than 12 % (limit max drawdown)

# Trailing $-PnL take-profit (per-position)
_TRAIL_ARM_USD      = 1.00   # peak unrealized PnL must reach >= $1.00 to arm
_TRAIL_GIVEBACK_USD = 0.25   # sell when peak - current >= $0.25


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vol_sma(volumes: List[float], period: int = 20) -> float:
    w = volumes[-period:] if len(volumes) >= period else volumes
    return sum(w) / max(len(w), 1)


def _stoch_price(highs: List[float], lows: List[float], closes: List[float],
                 k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
    """Price-based Stochastic %K (and simple %D = mean of last d_period %K)."""
    if len(closes) < k_period:
        return 50.0, 50.0
    k_vals = []
    for lag in range(d_period):
        end  = len(closes) - lag
        if end < k_period:
            k_vals.append(50.0)
            continue
        lo = min(lows[end - k_period: end])
        hi = max(highs[end - k_period: end])
        k_vals.append(100.0 * (closes[end - 1] - lo) / max(hi - lo, 1e-9))
    k = k_vals[0]
    d = sum(k_vals) / len(k_vals)
    return k, d


def _obv_state(closes: List[float], volumes: List[float],
               lookback: int = 6) -> Tuple[bool, bool]:
    """Returns (obv_rising, price_rising_over_lookback)."""
    if len(closes) < lookback + 2:
        return False, False
    obv = 0.0
    mark_obv = None
    for i in range(1, len(closes)):
        obv += volumes[i] if closes[i] > closes[i - 1] else (
               -volumes[i] if closes[i] < closes[i - 1] else 0.0)
        if i == len(closes) - lookback - 1:
            mark_obv = obv
    obv_rising = (mark_obv is not None) and (obv > mark_obv)
    price_rising = closes[-1] > closes[-(lookback + 1)]
    return obv_rising, price_rising


# ── Dry-run portfolio book ────────────────────────────────────────────────────

class _Book:
    def __init__(self, agent_name: str, balance: float = _DRY_RUN_BALANCE):
        self._agent      = agent_name
        self.balance     = balance
        self.positions: Dict[str, Dict] = {}  # pid → {size, avg_price}
        self.realized_pnl = 0.0
        # Per-trigger exit stats — diagnostic only, not persisted (trades ledger is durable)
        self._stats: Dict[str, Dict] = {
            trigger: {"wins": 0, "losses": 0, "total_pnl": 0.0}
            for trigger in ("TICK_SIGNAL", "TICK_STOP", "TICK_TRAIL", "TICK_PROFIT", "SCAN")
        }

    async def load(self) -> None:
        """Restore balance, positions, and PnL from the database."""
        state = await database.load_agent_state(self._agent)
        if state:
            self.balance      = state["balance"]
            self.realized_pnl = state["realized_pnl"]
            self.positions    = state["positions"]

            corrupt = [pid for pid, pos in self.positions.items()
                       if pos.get("avg_price", 0) == 0]
            for pid in corrupt:
                del self.positions[pid]
                logger.warning(f"{self._agent}: dropped corrupt position {pid} (avg_price=0)")

            logger.info(
                f"{self._agent} state restored | balance=${self.balance:.2f} | "
                f"pnl=${self.realized_pnl:+.2f} | positions={len(self.positions)}"
                + (f" | dropped {len(corrupt)} corrupt" if corrupt else "")
            )
        else:
            logger.info(f"{self._agent} no saved state — starting fresh at ${self.balance:.2f}")

    async def _save(self) -> None:
        await database.save_agent_state(
            self._agent, self.balance, self.realized_pnl, self.positions, {}
        )

    async def buy(self, pid: str, price: float, frac: float,
                  trigger: str = "SCAN") -> Tuple[float, float]:
        """Spend `frac` of balance. Returns (spent, size). Writes to trades ledger."""
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
            self.positions[pid] = {"size": size, "avg_price": price, "peak_pnl_usd": 0.0}
        self.balance -= spend
        await self._save()
        await database.open_trade(
            agent=self._agent, product_id=pid, entry_price=price,
            size=size, usd_open=spend, trigger_open=trigger,
            balance_after=self.balance,
        )
        return spend, size

    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        """Close position. Returns realized PnL. Writes to trades ledger."""
        if pid not in self.positions:
            return 0.0
        pos = self.positions.pop(pid)
        proceeds = pos["size"] * price
        pnl = proceeds - pos["size"] * pos["avg_price"]
        self.balance += proceeds
        self.realized_pnl += pnl

        # Per-trigger stats (diagnostic — trades table is durable)
        bucket = self._stats.setdefault(
            trigger, {"wins": 0, "losses": 0, "total_pnl": 0.0}
        )
        if pnl > 0:
            bucket["wins"] += 1
        else:
            bucket["losses"] += 1
        bucket["total_pnl"] += pnl

        await self._save()
        await database.close_trade(
            agent=self._agent, product_id=pid, exit_price=price,
            size=pos["size"], pnl=pnl, trigger_close=trigger,
            balance_after=self.balance,
        )
        return pnl

    def has_position(self, pid: str) -> bool:
        return pid in self.positions

    @property
    def total_value(self) -> float:
        return self.balance


# ── TechAgent ─────────────────────────────────────────────────────────────────

class TechAgentCB:
    """
    Technical analysis agent for the Coinbase Trader.
    Runs as an independent asyncio loop; saves every decision to
    `agent_decisions` so the CNN/Ollama model can see them.
    Accepts an optional ws_subscriber to read live tick prices.
    """

    def __init__(self, ws_subscriber=None):
        self.book = _Book("TECH")
        self.ws   = ws_subscriber   # CoinbaseWSSubscriber — live tick prices
        self._tick_locks:  Dict[str, asyncio.Lock] = {}   # per-product tick lock
        self._score_cache: Dict[str, Dict] = {}           # pid → last scan scores
        self.scan_count    = 0
        self.signals_buy   = 0
        self.signals_sell  = 0
        self.last_scan_at: Optional[float] = None
        logger.info(f"TechAgentCB ready | live_prices={'yes' if ws_subscriber else 'no (DB only)'}")

    async def start(self) -> None:
        """Load persisted state and register real-time price handler."""
        await self.book.load()
        if self.ws:
            self.ws.register_price_handler(self.on_price_tick)
            logger.info("TechAgentCB registered real-time price handler")

    def _get_lock(self, pid: str) -> asyncio.Lock:
        if pid not in self._tick_locks:
            self._tick_locks[pid] = asyncio.Lock()
        return self._tick_locks[pid]

    # ── Macro adjustment helpers ──────────────────────────────────────────────

    def _macro_adjusted_buy_score(self, sc: Dict, macro: MacroContext) -> float:
        """Apply market-structure buy multiplier (funding rate, L/S ratio, OI)."""
        return min(1.0, sc["buy_score"] * macro.buy_gate_multiplier())

    def _macro_adjusted_sell_score(self, sc: Dict, macro: MacroContext) -> float:
        """Apply market-structure sell multiplier. SELL is never strongly suppressed."""
        return min(1.0, sc["sell_score"] * macro.sell_gate_multiplier())

    def _compute_atr_stop(self, candles: List[Dict], entry_price: float) -> float:
        """
        Compute an ATR-based stop distance (as a fraction of price).
        stop = ATR(14) × _ATR_MULTIPLIER / entry_price
        Clamped to [_ATR_STOP_MIN, _ATR_STOP_MAX].
        Returns _ATR_STOP_MIN when there is insufficient data.
        """
        if len(candles) < 15:
            return _ATR_STOP_MIN
        highs  = [c["high"]  for c in candles]
        lows   = [c["low"]   for c in candles]
        closes = [c["close"] for c in candles]
        atr    = _atr(highs, lows, closes)
        if atr <= 0 or entry_price <= 0:
            return _ATR_STOP_MIN
        raw = atr * _ATR_MULTIPLIER / entry_price
        return max(_ATR_STOP_MIN, min(raw, _ATR_STOP_MAX))

    async def on_price_tick(self, pid: str, price: float) -> None:
        """
        Fires on every WS tick for pid.
        Uses cached scan scores for entry signals; hard stop-loss and take-profit
        execute immediately without waiting for the next 2-min scan cycle.
        """
        sc = self._score_cache.get(pid)

        if not self.book.has_position(pid):
            # Check cached buy signal from last scan
            if sc and sc["buy_score"] >= _BUY_THRESHOLD:
                lock = self._get_lock(pid)
                if lock.locked():
                    return
                async with lock:
                    if self.book.has_position(pid):
                        return
                    frac = min(_kelly_fraction(sc["buy_score"]), _MAX_POSITION_FRAC)
                    spent, _ = await self.book.buy(pid, price, frac, trigger="TICK_SIGNAL")
                    if spent > 0:
                        self.signals_buy += 1
                        self._score_cache.pop(pid, None)  # consumed — don't re-trigger
                        reasoning = (
                            f"TECH BUY [TICK]: {'; '.join(sc['buy_reasons'])} "
                            f"score={sc['buy_score']:.2f}"
                        )
                        decision = {
                            "agent": "TECH", "product_id": pid, "side": "BUY",
                            "confidence": round(sc["buy_score"], 3),
                            "price": round(price, 6), "score": round(sc["buy_score"], 3),
                            "reasoning": reasoning,
                            "balance": round(self.book.balance, 2), "pnl": None,
                        }
                        await database.save_agent_decision(decision)
                        logger.info(
                            f"TechAgentCB TICK BUY {pid} @{price:.4f} "
                            f"score={sc['buy_score']:.2f} spent=${spent:.2f}"
                        )
            return

        # Have a position — check exits
        pos = self.book.positions.get(pid)
        if not pos:
            return

        lock = self._get_lock(pid)
        if lock.locked():
            return
        async with lock:
            if not self.book.has_position(pid):
                return
            pos = self.book.positions.get(pid)
            if not pos:
                return

            avg_price = pos["avg_price"]
            pct       = (price - avg_price) / avg_price

            current_pnl_usd = (price - avg_price) * pos["size"]
            prev_peak       = pos.get("peak_pnl_usd", 0.0)
            pos["peak_pnl_usd"] = max(prev_peak, current_pnl_usd)
            peak_pnl_usd    = pos["peak_pnl_usd"]

            exit_reason = None
            exit_trigger = "TICK_SIGNAL"

            # 1. Cached sell signal from last scan
            if sc and sc["sell_score"] >= _SELL_THRESHOLD:
                exit_reason = (
                    f"TECH SELL [TICK]: {'; '.join(sc['sell_reasons'])} "
                    f"score={sc['sell_score']:.2f}"
                )
                confidence   = sc["sell_score"]
                exit_trigger = "TICK_SIGNAL"

            # 2. ATR-based stop-loss (adapts to volatility)
            elif pct < -pos.get("atr_stop", _ATR_STOP_MIN):
                atr_stop_pct = pos.get("atr_stop", _ATR_STOP_MIN) * 100
                exit_reason  = (
                    f"TECH ATR STOP [TICK]: {pct*100:.1f}% below entry "
                    f"${avg_price:.4f} (stop={atr_stop_pct:.1f}%)"
                )
                confidence   = 0.95
                exit_trigger = "TICK_STOP"

            # 3. Trailing $-PnL take-profit: armed at peak >= _TRAIL_ARM_USD,
            #    fires when peak - current >= _TRAIL_GIVEBACK_USD.
            elif (peak_pnl_usd >= _TRAIL_ARM_USD
                  and (peak_pnl_usd - current_pnl_usd) >= _TRAIL_GIVEBACK_USD):
                exit_reason  = (
                    f"TECH TRAIL [TICK]: peak=+${peak_pnl_usd:.2f} "
                    f"current=+${current_pnl_usd:.2f} "
                    f"giveback=${peak_pnl_usd - current_pnl_usd:.2f}"
                )
                confidence   = 0.95
                exit_trigger = "TICK_TRAIL"

            # 4. Take-profit (legacy %-based backstop)
            elif pct > _TAKE_PROFIT:
                exit_reason  = f"TECH TAKE PROFIT [TICK]: +{pct*100:.1f}% above entry ${avg_price:.4f}"
                confidence   = 0.95
                exit_trigger = "TICK_PROFIT"

            if exit_reason:
                pnl = await self.book.sell(pid, price, trigger=exit_trigger)
                self._score_cache.pop(pid, None)
                self.signals_sell += 1
                decision = {
                    "agent": "TECH", "product_id": pid, "side": "SELL",
                    "confidence": round(confidence, 3),
                    "price": round(price, 6), "score": round(confidence, 3),
                    "reasoning": f"{exit_reason} pnl=${pnl:+.2f}",
                    "balance": round(self.book.balance, 2), "pnl": round(pnl, 4),
                }
                await database.save_agent_decision(decision)
                logger.info(
                    f"TechAgentCB TICK SELL {pid} @{price:.4f} "
                    f"pnl={pnl:+.2f} | {exit_reason}"
                )

    def _live_price(self, pid: str, fallback: float) -> float:
        """Return live WS tick price if available, else the DB-stored fallback."""
        if self.ws:
            p = self.ws.get_price(pid)
            if p and p > 0:
                return p
        return fallback

    # ── Core scoring ──────────────────────────────────────────────────────────

    def _score(self, candles: List[Dict]) -> Dict:
        closes  = [c["close"]  for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]

        rsi_val        = _rsi(closes)
        _, _, macd_h   = _macd(closes)
        _, _, macd_h_p = _macd(closes[:-1])   # previous bar
        _, _, _, bb_pos = _bollinger(closes)   # 0=lower,1=upper
        bb_upper = closes[-1] * (1 + 0.02)    # approx — we use bb_pos directly
        bb_lower = closes[-1] * (1 - 0.02)

        # Actual band prices from bollinger
        if len(closes) >= 22:
            mn    = sum(closes[-20:]) / 20
            std   = math.sqrt(sum((v - mn) ** 2 for v in closes[-20:]) / 20)
            bb_upper = mn + 2 * std
            bb_lower = mn - 2 * std

        vol_sma    = _vol_sma(volumes)
        vol_spike  = volumes[-1] > vol_sma * 1.2 if vol_sma > 0 else False
        stoch_k, stoch_d = _stoch_price(highs, lows, closes)
        prev_stoch_k, _  = _stoch_price(highs[:-1], lows[:-1], closes[:-1]) if len(closes) > 14 else (50.0, 50.0)
        obv_rising, price_rising = _obv_state(closes, volumes)

        macd_bull_cross = macd_h > 0 and macd_h_p <= 0
        macd_bear_cross = macd_h < 0 and macd_h_p >= 0

        cur = closes[-1]

        # ── BUY score ─────────────────────────────────────────────────────────
        buy_score   = 0.0
        buy_reasons = []

        if rsi_val < 30:
            buy_score += 0.35
            buy_reasons.append(f"RSI={rsi_val:.1f} oversold")

        if cur <= bb_lower:
            buy_score += 0.30
            buy_reasons.append(f"Price at lower BB")

        if macd_bull_cross:
            buy_score += 0.25; buy_reasons.append("MACD bull cross")
        elif macd_h > 0:
            buy_score += 0.10; buy_reasons.append("MACD positive")

        if vol_spike:
            buy_score += 0.10; buy_reasons.append(f"Vol spike {volumes[-1]/max(vol_sma,1):.1f}×")

        if stoch_k < 20 and stoch_k > prev_stoch_k:
            buy_score += 0.15; buy_reasons.append(f"Stoch {stoch_k:.0f} oversold+rising")
        elif stoch_k < 20:
            buy_score += 0.08; buy_reasons.append(f"Stoch {stoch_k:.0f} oversold")

        if price_rising and obv_rising:
            buy_score += 0.10; buy_reasons.append("OBV confirms up")
        elif not price_rising and obv_rising:
            buy_score += 0.08; buy_reasons.append("OBV divergence: accumulation")

        # ── SELL score ────────────────────────────────────────────────────────
        sell_score   = 0.0
        sell_reasons = []

        if rsi_val > 70:
            sell_score += 0.35; sell_reasons.append(f"RSI={rsi_val:.1f} overbought")

        if cur >= bb_upper:
            sell_score += 0.30; sell_reasons.append("Price at upper BB")

        if macd_bear_cross:
            sell_score += 0.25; sell_reasons.append("MACD bear cross")
        elif macd_h < 0:
            sell_score += 0.10; sell_reasons.append("MACD negative")

        if vol_spike and sell_score > 0:
            sell_score += 0.10; sell_reasons.append("Vol confirmation")

        if stoch_k > 80 and stoch_k < prev_stoch_k:
            sell_score += 0.15; sell_reasons.append(f"Stoch {stoch_k:.0f} overbought+falling")
        elif stoch_k > 80:
            sell_score += 0.08; sell_reasons.append(f"Stoch {stoch_k:.0f} overbought")

        if price_rising and not obv_rising:
            sell_score += 0.12; sell_reasons.append("OBV divergence: distribution")
        elif not price_rising and not obv_rising:
            sell_score += 0.08; sell_reasons.append("OBV confirms down")

        return {
            "buy_score":   round(buy_score,  3),
            "sell_score":  round(sell_score, 3),
            "buy_reasons": buy_reasons,
            "sell_reasons": sell_reasons,
            "rsi":    round(rsi_val, 1),
            "macd_h": round(macd_h, 6),
            "bb_pos": round(bb_pos, 3),
            "stoch_k": round(stoch_k, 1),
        }

    # ── Single-product scan ───────────────────────────────────────────────────

    async def analyze_product(self, product: Dict) -> Optional[Dict]:
        pid   = product["product_id"]
        price = self._live_price(pid, product.get("price", 0))
        if not price or price <= 0:
            return None
        if price < MIN_PRICE:
            return None

        candles = await database.get_candles(pid, limit=80)
        if len(candles) < 35:   # need enough for MACD(26) + buffer
            return None

        sc = self._score(candles)
        # Cache for real-time tick handler to reference between scans
        self._score_cache[pid] = sc

        # Fetch crypto market-structure macro (cached — negligible overhead)
        macro = await get_macro_service().get_macro_context()

        raw_buy    = sc["buy_score"]
        raw_sell   = sc["sell_score"]
        buy_score  = self._macro_adjusted_buy_score(sc, macro)
        sell_score = self._macro_adjusted_sell_score(sc, macro)
        has_pos    = self.book.has_position(pid)

        side       = "HOLD"
        confidence = 0.0
        pnl        = None
        reasoning  = ""

        indicators = {
            "rsi":    sc["rsi"],
            "bb_pos": sc["bb_pos"],
            "macd_h": sc["macd_h"],
            "stoch_k": sc["stoch_k"],
        }

        macro_mult = macro.buy_gate_multiplier()

        if buy_score >= _BUY_THRESHOLD and not has_pos:
            frac  = min(_kelly_fraction(raw_buy), _MAX_POSITION_FRAC)
            spent, _ = await self.book.buy(pid, price, frac, trigger="SCAN")
            if spent > 0:
                side       = "BUY"
                confidence = buy_score
                atr_stop   = self._compute_atr_stop(candles, price)
                self.book.positions[pid]["atr_stop"] = atr_stop
                reasoning  = (
                    f"TECH BUY: {'; '.join(sc['buy_reasons'])} score={raw_buy:.2f} "
                    f"macro_adj={buy_score:.2f} frac={frac:.2f} "
                    f"atr_stop={atr_stop*100:.1f}% macro={macro.regime_label()}"
                )
                self.signals_buy += 1

        elif sell_score >= _SELL_THRESHOLD and has_pos:
            pnl  = await self.book.sell(pid, price, trigger="SCAN")
            side       = "SELL"
            confidence = sell_score
            reasoning  = (
                f"TECH SELL: {'; '.join(sc['sell_reasons'])} score={raw_sell:.2f} "
                f"macro_adj={sell_score:.2f} pnl=${pnl:+.2f}"
            )
            self.signals_sell += 1

        else:
            confidence = max(buy_score, sell_score)
            reasoning  = (
                f"TECH HOLD RSI={sc['rsi']:.1f} MACD={sc['macd_h']:+.4f} "
                f"Stoch={sc['stoch_k']:.0f} BB={sc['bb_pos']:.2f} "
                f"raw_buy={raw_buy:.2f} adj_buy={buy_score:.2f} macro_mult={macro_mult:.2f}"
            )

        decision = {
            "agent":      "TECH",
            "product_id": pid,
            "side":       side,
            "confidence": round(confidence, 3),
            "price":      round(price, 6),
            "score":      round(max(raw_buy, raw_sell), 3),
            "reasoning":  reasoning,
            "balance":    round(self.book.balance, 2),
            "pnl":        round(pnl, 4) if pnl is not None else None,
        }
        await database.save_agent_decision(decision)

        # ── Outcome tracker: record + Ollama validation for BUY/SELL ────────────
        if side in ("BUY", "SELL"):
            tracker = get_tracker()
            await tracker.record(
                source="TECH", product_id=pid, side=side,
                confidence=confidence, price=price, indicators=indicators,
            )
            val_prob = await tracker.validate_with_ollama(
                source="TECH", product_id=pid, side=side,
                confidence=confidence, price=price, indicators=indicators,
            )
            if val_prob is not None:
                logger.info(
                    f"TechAgent Ollama validation {pid} {side}: "
                    f"p={val_prob:.3f} (score={confidence:.2f} "
                    f"RSI={sc['rsi']:.0f} Stoch={sc['stoch_k']:.0f} BB={sc['bb_pos']:.2f})"
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
                logger.error(f"TechAgentCB error [{p.get('product_id')}]: {e}")
            await asyncio.sleep(0.05)
        return signals

    async def run_loop(self, interval: int = _SCAN_INTERVAL, is_trading_fn=None) -> None:
        await self.start()   # restore persisted state before first scan
        logger.info(f"TechAgentCB loop started | interval={interval}s | balance=${self.book.balance:.2f}")
        while True:
            try:
                if is_trading_fn is None or is_trading_fn():
                    sigs = await self.scan_all()
                    self.last_scan_at = time.time()
                    self.scan_count  += 1
                    logger.info(
                        f"TechAgentCB scan #{self.scan_count} done | "
                        f"signals={len(sigs)} | balance=${self.book.balance:.2f} | "
                        f"pnl=${self.book.realized_pnl:+.2f}"
                    )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"TechAgentCB loop error: {e}")
            await asyncio.sleep(interval)

    @property
    def status(self) -> Dict:
        return {
            "agent":         "TECH",
            "balance":       round(self.book.balance, 2),
            "realized_pnl":  round(self.book.realized_pnl, 2),
            "open_positions": len(self.book.positions),
            "positions":     {
                pid: {"size": round(p["size"], 6), "avg_price": round(p["avg_price"], 6)}
                for pid, p in self.book.positions.items()
            },
            "scan_count":    self.scan_count,
            "signals_buy":   self.signals_buy,
            "signals_sell":  self.signals_sell,
            "last_scan_at":  self.last_scan_at,
            "exit_stats":    {
                trigger: {
                    "wins":      s["wins"],
                    "losses":    s["losses"],
                    "win_rate":  round(s["wins"] / max(s["wins"] + s["losses"], 1) * 100, 1),
                    "total_pnl": round(s["total_pnl"], 2),
                }
                for trigger, s in self.book._stats.items()
                if s["wins"] + s["losses"] > 0
            },
        }
