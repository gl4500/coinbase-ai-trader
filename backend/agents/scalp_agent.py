"""
ScalpAgent -- Coinbase Trader
──────────────────────────────────────────────────────────────────────────────
Fast-cycle scalping agent based on multi-indicator confluence scoring.

Strategy (per research):
  Entry : RSI(7) + BB(20,2) + VWAP confluence scoring (min 5 points)
  Regime: ADX(10) filter -- ADX>25 enables momentum signals, <20 mean-revert
  Exit  : +0.30% fixed take-profit  OR  -0.25% hard stop-loss  OR  15-min time exit
  Trailing stop: 1.5x ATR(7) from high-water mark

Scoring system (max 10 pts, need >= 5 to enter):
  RSI(7) < 25            +2 pts (deeply oversold)
  RSI(7) < 35            +1 pt
  Price <= BB lower band +2 pts
  Price <= BB mid        +1 pt
  VWAP distance < -0.5%  +2 pts (below VWAP, institutional support)
  Stoch(5,3,3) < 20      +1 pt
  OBV slope > 0.15       +1 pt
  MFI(7) < 25            +1 pt

Products: BTC-USD and ETH-USD initially (tightest spreads)
Fee math: Coinbase maker 0.006% per side = 0.012% round-trip
          Min viable TP = 0.15%; we use 0.30% for 2.5x cushion

Dry-run balance: $1,000 (mirrors Tech/Momentum/CNN)
Max position: 20% per slot (max 2 concurrent)
Exit reason logging: every closed trade records trigger, PnL, hold time,
and entry signal pattern so parameter tuning is data-driven.
"""
import asyncio
import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import database
from agents.signal_generator import (
    _rsi, _bollinger, _atr, _adx, _mfi, _obv_slope, _stoch_rsi, _vwap,
)
from services.outcome_tracker import get_tracker

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
_SCALP_BALANCE       = 1_000.0
_MAX_FRAC            = 0.20     # 20% per position
_MAX_CONCURRENT      = 2        # max open scalp positions
_TAKE_PROFIT         = 0.0030   # 0.30% TP
_HARD_STOP           = 0.0025   # 0.25% SL
_ATR_TRAIL_MULT      = 1.5      # trailing stop = 1.5 * ATR(7)
_TIME_EXIT_SEC       = 900      # 15 min hard time exit
_MIN_SCORE           = 5        # minimum confluence score to enter
_ADX_TREND           = 25       # ADX > this = trending (enables momentum signals)
_ADX_RANGE           = 20       # ADX < this = ranging (enables mean-revert signals)
_TICK_SLEEP          = 1.0      # check open positions every 1 second
_CANDLE_SLEEP        = 60.0     # re-evaluate entry every 60 seconds
_MIN_CANDLES         = 40               # minimum candle bars required for indicators
_MAX_SCALP_PRODUCTS  = 10              # cap — don't scan the entire universe
_MIN_PRICE           = 0.01            # skip micro-priced tokens (unprofitable spreads)
_SL_COOLDOWN_SEC     = 300             # 5-min re-entry block after a stop-loss exit


# ── Book ───────────────────────────────────────────────────────────────────────

class _ScalpBook:
    """Dry-run portfolio book for the ScalpAgent."""

    def __init__(self):
        self._agent       = "SCALP"
        self.balance      = _SCALP_BALANCE
        self.positions: Dict[str, Dict] = {}   # pid -> {size, avg_price, entry_time, high_water}
        self.realized_pnl = 0.0
        # Per-trigger outcome counters — used to diagnose which exit parameter loses money
        self._stats: Dict[str, Dict] = {
            trigger: {"wins": 0, "losses": 0, "total_pnl": 0.0}
            for trigger in ("TP", "SL", "TRAIL", "TIME")
        }

    async def load(self) -> None:
        state = await database.load_agent_state(self._agent)
        if state:
            self.balance      = state["balance"]
            self.realized_pnl = state["realized_pnl"]
            self.positions    = state["positions"]
            self._start_balance = self.balance

            corrupt = [pid for pid, pos in self.positions.items()
                       if pos.get("avg_price", 0) == 0]
            for pid in corrupt:
                del self.positions[pid]
                logger.warning(f"SCALP book: dropped corrupt position {pid} (avg_price=0)")

            logger.info(
                f"SCALP book restored | balance=${self.balance:.2f} | "
                f"pnl=${self.realized_pnl:+.2f} | positions={len(self.positions)}"
                + (f" | dropped {len(corrupt)} corrupt" if corrupt else "")
            )
        else:
            logger.info(f"SCALP book: starting fresh at ${self.balance:.2f}")

    async def _save(self) -> None:
        await database.save_agent_state(
            self._agent, self.balance, self.realized_pnl, self.positions, {}
        )

    def has_position(self, pid: str) -> bool:
        return pid in self.positions

    def position_count(self) -> int:
        return len(self.positions)

    async def buy(self, pid: str, price: float, atr: float,
                  trigger: str = "SCALP", entry_reasons: Optional[List[str]] = None) -> Tuple[float, float]:
        spend = min(self.balance * _MAX_FRAC, self.balance * 0.95)
        if spend < 1.0 or price <= 0 or price < _MIN_PRICE:
            return 0.0, 0.0
        size  = spend / price
        trail = max(atr * _ATR_TRAIL_MULT, price * _HARD_STOP)
        self.positions[pid] = {
            "size":          size,
            "avg_price":     price,
            "entry_time":    time.time(),
            "high_water":    price,
            "trail_dist":    trail,
            "entry_reasons": entry_reasons or [],  # confluence signals that fired
        }
        self.balance -= spend
        await self._save()
        await database.open_trade(
            agent=self._agent, product_id=pid, entry_price=price,
            size=size, usd_open=spend, trigger_open=trigger,
            balance_after=self.balance,
        )
        logger.info(f"SCALP BUY {pid} @ ${price:,.4f} | size={size:.6f} | spend=${spend:.2f}")
        return spend, size

    async def sell(self, pid: str, price: float, trigger: str = "SCALP") -> float:
        if pid not in self.positions:
            return 0.0
        pos       = self.positions.pop(pid)
        proceeds  = pos["size"] * price
        pnl       = proceeds - pos["size"] * pos["avg_price"]
        self.balance      += proceeds
        self.realized_pnl += pnl

        # ── Update per-trigger stats ──────────────────────────────────────────
        bucket = self._stats.get(trigger, self._stats.setdefault(
            trigger, {"wins": 0, "losses": 0, "total_pnl": 0.0}
        ))
        bucket["total_pnl"] += pnl
        if pnl >= 0:
            bucket["wins"]   += 1
        else:
            bucket["losses"] += 1

        # ── Structured exit log ───────────────────────────────────────────────
        hold_sec  = time.time() - pos["entry_time"]
        pct_move  = (price - pos["avg_price"]) / pos["avg_price"] * 100
        outcome   = "WIN" if pnl >= 0 else "LOSS"
        st        = self._stats[trigger] if trigger in self._stats else bucket
        win_rate  = st["wins"] / max(st["wins"] + st["losses"], 1) * 100
        log_fn    = logger.info if pnl >= 0 else logger.warning
        entry_sig = ", ".join(pos.get("entry_reasons", [])[:3]) or "n/a"
        log_fn(
            f"SCALP EXIT [{trigger}] {outcome} | {pid} | "
            f"entry=${pos['avg_price']:.4f} exit=${price:.4f} | "
            f"move={pct_move:+.3f}% pnl=${pnl:+.4f} | "
            f"hold={hold_sec:.0f}s | "
            f"entry signals: [{entry_sig}] | "
            f"{trigger} lifetime: {st['wins']}W/{st['losses']}L "
            f"wr={win_rate:.0f}% totpnl=${st['total_pnl']:+.2f}"
        )

        await self._save()
        await database.close_trade(
            agent=self._agent, product_id=pid, exit_price=price,
            size=pos["size"], pnl=pnl, trigger_close=trigger,
            balance_after=self.balance,
        )
        return pnl

    @property
    def status(self) -> Dict:
        return {
            "agent":          "SCALP",
            "balance":        round(self.balance, 2),
            "realized_pnl":   round(self.realized_pnl, 2),
            "open_positions": len(self.positions),
            "positions": {
                pid: {
                    "size":       round(p["size"], 6),
                    "avg_price":  round(p["avg_price"], 6),
                    "entry_time": p["entry_time"],
                }
                for pid, p in self.positions.items()
            },
            "exit_stats": {
                trigger: {
                    "wins":      s["wins"],
                    "losses":    s["losses"],
                    "win_rate":  round(s["wins"] / max(s["wins"] + s["losses"], 1) * 100, 1),
                    "total_pnl": round(s["total_pnl"], 2),
                }
                for trigger, s in self._stats.items()
                if s["wins"] + s["losses"] > 0
            },
        }


# ── Indicators ─────────────────────────────────────────────────────────────────

def _confluence_score(
    closes: List[float],
    highs:  List[float],
    lows:   List[float],
    vols:   List[float],
) -> Tuple[int, List[str]]:
    """
    Returns (score, reasons).
    Score >= _MIN_SCORE triggers an entry signal.
    """
    score   = 0
    reasons = []
    price   = closes[-1]

    rsi7 = _rsi(closes, period=7)
    if rsi7 < 25:
        score += 2
        reasons.append(f"RSI7={rsi7:.1f} deeply oversold (+2)")
    elif rsi7 < 35:
        score += 1
        reasons.append(f"RSI7={rsi7:.1f} oversold (+1)")

    _, bb_mid, bb_low, bb_pos = _bollinger(closes, period=20, mult=2.0)
    if price <= bb_low:
        score += 2
        reasons.append(f"BB lower band touch (+2)")
    elif price <= bb_mid:
        score += 1
        reasons.append(f"Price below BB mid (+1)")

    _, vwap_dist = _vwap(highs, lows, closes, vols, period=20)
    # vwap_dist is normalised to [-1,1]; -0.5 corresponds to raw -0.025% (0.5 * 0.05)
    if vwap_dist < -0.5:
        score += 2
        reasons.append(f"VWAP dist={vwap_dist:.2f} below VWAP (+2)")
    elif vwap_dist < 0:
        score += 1
        reasons.append(f"VWAP dist={vwap_dist:.2f} slight below VWAP (+1)")

    stoch_k, _ = _stoch_rsi(closes, period=14, k_smooth=3, d_smooth=3)
    if stoch_k < 20:
        score += 1
        reasons.append(f"StochRSI K={stoch_k:.1f} oversold (+1)")

    obv_sl = _obv_slope(closes, vols, period=10)
    if obv_sl > 0.15:
        score += 1
        reasons.append(f"OBV slope={obv_sl:+.2f} accumulating (+1)")

    mfi7 = _mfi(highs, lows, closes, vols, period=7)
    if mfi7 < 25:
        score += 1
        reasons.append(f"MFI7={mfi7:.1f} oversold (+1)")

    return score, reasons


def _exit_reason(pos: Dict, price: float) -> Optional[str]:
    """Check all exit conditions; return reason string or None."""
    entry = pos["avg_price"]
    pct   = (price - entry) / entry

    # Take-profit
    if pct >= _TAKE_PROFIT:
        return "TP"

    # Hard stop-loss
    if pct <= -_HARD_STOP:
        return "SL"

    # Trailing stop from high-water
    hw = pos["high_water"]
    trail = pos["trail_dist"]
    if price < hw - trail:
        return "TRAIL"

    # Time exit
    if time.time() - pos["entry_time"] >= _TIME_EXIT_SEC:
        return "TIME"

    return None


# ── Agent ──────────────────────────────────────────────────────────────────────

class ScalpAgent:
    """
    Fast scalping agent.  Two parallel loops:
      - entry_loop : scans candles every 60 s for entry signals
      - exit_loop  : monitors open positions every 1 s for exit conditions
    """

    def __init__(self, ws_subscriber=None, is_trading_fn=None):
        self.book             = _ScalpBook()
        self._ws              = ws_subscriber
        self._running         = False
        self._is_trading_fn   = is_trading_fn
        self._entry_task: Optional[asyncio.Task] = None
        self._exit_task:  Optional[asyncio.Task] = None
        self.scan_count       = 0
        self.last_scan_at: Optional[float] = None
        self._sl_cooldown: Dict[str, float] = {}  # pid -> timestamp of last SL exit

    # -- helpers -----------------------------------------------------------------

    def _live_price(self, pid: str) -> Optional[float]:
        if self._ws:
            return self._ws.get_price(pid)
        return None

    async def _get_scalp_products(self) -> List[str]:
        """Return tracked products that have enough candle data and a tradeable price."""
        tracked = await database.get_products(tracked_only=True, limit=200)
        eligible = []
        for p in tracked:
            pid   = p["product_id"]
            price = float(p.get("price") or 0)
            if price < _MIN_PRICE:
                continue   # skip micro-priced tokens
            candles = await database.get_candles(pid, limit=_MIN_CANDLES)
            if len(candles) >= _MIN_CANDLES:
                eligible.append(pid)
            if len(eligible) >= _MAX_SCALP_PRODUCTS:
                break
        return eligible

    async def _get_candles(self, pid: str) -> Optional[Dict]:
        """Fetch 1-min candle data; return None if not enough bars."""
        candles = await database.get_candles(pid, limit=120)
        if len(candles) < _MIN_CANDLES:
            return None
        return {
            "closes": [c["close"]  for c in candles],
            "highs":  [c["high"]   for c in candles],
            "lows":   [c["low"]    for c in candles],
            "vols":   [c["volume"] for c in candles],
        }

    # -- entry loop --------------------------------------------------------------

    async def _entry_loop(self) -> None:
        while self._running:
            try:
                if self._is_trading_fn is None or self._is_trading_fn():
                    await self._scan_entries()
                self.scan_count  += 1
                self.last_scan_at = time.time()
                logger.info(
                    f"SCALP scan #{self.scan_count} done | "
                    f"open={self.book.position_count()} | "
                    f"balance=${self.book.balance:.2f} | "
                    f"pnl=${self.book.realized_pnl:+.2f}"
                )
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"SCALP entry_loop error: {e}")
            await asyncio.sleep(_CANDLE_SLEEP)

    async def _scan_entries(self) -> None:
        if self.book.position_count() >= _MAX_CONCURRENT:
            return

        products = await self._get_scalp_products()
        if not products:
            logger.warning("SCALP: no eligible products with candle data — skipping scan")
            return

        for pid in products:
            if self.book.has_position(pid):
                continue
            if self.book.position_count() >= _MAX_CONCURRENT:
                break

            # Cooldown: block re-entry for 5 min after a stop-loss exit
            sl_ts = self._sl_cooldown.get(pid, 0)
            if time.time() - sl_ts < _SL_COOLDOWN_SEC:
                remaining = int(_SL_COOLDOWN_SEC - (time.time() - sl_ts))
                logger.debug(f"SCALP {pid}: SL cooldown active — {remaining}s remaining")
                continue

            data = await self._get_candles(pid)
            if not data:
                continue

            closes = data["closes"]
            highs  = data["highs"]
            lows   = data["lows"]
            vols   = data["vols"]

            # ADX regime gate
            adx, di_p, di_m = _adx(highs, lows, closes, period=10)
            trending = adx > _ADX_TREND
            ranging  = adx < _ADX_RANGE

            if not trending and not ranging:
                logger.debug(f"SCALP {pid}: ADX={adx:.1f} in dead zone — skip")
                continue

            score, reasons = _confluence_score(closes, highs, lows, vols)

            # Require more conviction in ranging market (noise risk)
            min_score = _MIN_SCORE if trending else _MIN_SCORE + 1

            if score < min_score:
                logger.debug(
                    f"SCALP {pid}: score={score}/{min_score} ADX={adx:.1f} — no entry"
                )
                continue

            # Require a live WS price — never enter on stale candle close.
            # If WS is disconnected, _live_price returns None and _check_exits
            # already skips all positions (can't monitor them), so opening new
            # ones would create unmonitored exposure.
            price = self._live_price(pid)
            if not price or price < _MIN_PRICE:
                logger.debug(f"SCALP {pid}: no live WS price (${price}) — skip entry")
                continue
            atr7  = _atr(highs, lows, closes, period=7)

            regime = 'TREND' if trending else 'RANGE'
            logger.info(
                f"SCALP ENTRY {pid} | score={score} | ADX={adx:.1f} "
                f"| {regime} | {', '.join(reasons[:3])}"
            )
            await self.book.buy(pid, price, atr7, trigger="SCALP", entry_reasons=reasons)
            await database.save_agent_decision({
                "agent":      "SCALP",
                "product_id": pid,
                "side":       "BUY",
                "confidence": round(score / 10.0, 3),
                "price":      round(price, 6),
                "score":      round(score / 10.0, 3),
                "reasoning":  (
                    f"SCALP BUY: score={score}/{min_score} ADX={adx:.1f} {regime} "
                    f"| {', '.join(reasons[:4])}"
                ),
                "balance":    round(self.book.balance, 2),
                "pnl":        None,
            })

            await asyncio.sleep(0.5)   # small gap between buys

    # -- exit loop ---------------------------------------------------------------

    async def _exit_loop(self) -> None:
        while self._running:
            try:
                await self._check_exits()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"SCALP exit_loop error: {e}")
            await asyncio.sleep(_TICK_SLEEP)

    async def _check_exits(self) -> None:
        for pid, pos in list(self.book.positions.items()):
            price = self._live_price(pid)
            if price is None or price <= 0:
                continue

            # Update high-water mark
            if price > pos["high_water"]:
                pos["high_water"] = price

            reason = _exit_reason(pos, price)
            if reason:
                if reason == "SL":
                    self._sl_cooldown[pid] = time.time()  # block re-entry for _SL_COOLDOWN_SEC
                pnl = await self.book.sell(pid, price, trigger=reason)
                confidence = 0.95 if reason in ("TP", "SL") else 0.80 if reason == "TRAIL" else 0.60
                await database.save_agent_decision({
                    "agent":      "SCALP",
                    "product_id": pid,
                    "side":       "SELL",
                    "confidence": confidence,
                    "price":      round(price, 6),
                    "score":      confidence,
                    "reasoning":  (
                        f"SCALP SELL [{reason}]: "
                        f"entry=${pos['avg_price']:.4f} exit=${price:.4f} "
                        f"pnl=${pnl:+.4f}"
                    ),
                    "balance":    round(self.book.balance, 2),
                    "pnl":        round(pnl, 4),
                })

    # -- public API --------------------------------------------------------------

    async def start(self) -> None:
        await self.book.load()
        self._running  = True
        self._entry_task = asyncio.create_task(self._entry_loop())
        self._exit_task  = asyncio.create_task(self._exit_loop())
        logger.info("ScalpAgent started")

    async def stop(self) -> None:
        self._running = False
        for t in (self._entry_task, self._exit_task):
            if t:
                t.cancel()
        logger.info("ScalpAgent stopped")

    @property
    def status(self) -> Dict:
        return {
            **self.book.status,
            "scan_count":   self.scan_count,
            "last_scan_at": self.last_scan_at,
        }
