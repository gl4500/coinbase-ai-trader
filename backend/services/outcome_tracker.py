"""
Outcome Tracker
───────────────────────────────────────────────────────────────────────────────
Records every Tech/Momentum/CNN signal, checks 4 hours later whether the
price moved in the predicted direction, and builds a compact lesson string.

Two roles:
  1. Long-running loop  — checks pending outcomes every 30 min, resolves WIN/LOSS
  2. Immediate validator — called right after a static agent (Tech/Momentum) fires;
                           asks Ollama "given past outcomes, do you confirm this signal?"

Lessons are injected into the CNN's Ollama prompt so the LLM sees a track record
of what actually happened to previous signals on this product.
"""
import asyncio
import json
import logging
import re
import time
from typing import Dict, List, Optional

import httpx

import database
from clients import coinbase_client

logger = logging.getLogger(__name__)

OLLAMA_URL       = "http://localhost:11434"
_WIN_THRESHOLD   = 0.005   # +0.5% = WIN for a BUY
_LOSS_THRESHOLD  = 0.005   # -0.5% adverse = LOSS
_CHECK_HORIZON   = 4 * 3600   # check 4 hours after signal


# ── Outcome Tracker ────────────────────────────────────────────────────────────

class OutcomeTracker:

    # ── Record a pending outcome ───────────────────────────────────────────────

    async def record(
        self,
        source:     str,    # TECH | MOMENTUM | CNN
        product_id: str,
        side:       str,    # BUY | SELL
        confidence: float,
        price:      float,
        indicators: Dict,
    ) -> None:
        """Save a pending signal. Outcome checked 4 h later by check_pending()."""
        try:
            await database.insert_signal_outcome({
                "source":          source,
                "product_id":      product_id,
                "side":            side,
                "confidence":      round(confidence, 4),
                "entry_price":     round(price, 6),
                "indicators_json": json.dumps(indicators),
                "check_after":     time.time() + _CHECK_HORIZON,
            })
            logger.debug(f"OutcomeTracker recorded {source} {side} {product_id} @ ${price:.4f}")
        except Exception as e:
            logger.warning(f"OutcomeTracker.record failed: {e}")

    # ── Validate immediately via Ollama ────────────────────────────────────────

    async def validate_with_ollama(
        self,
        source:     str,
        product_id: str,
        side:       str,
        confidence: float,
        price:      float,
        indicators: Dict,
    ) -> Optional[float]:
        """
        Called immediately after Tech or Momentum fires a BUY/SELL.
        Fetches past lessons for this product and asks Ollama to confirm
        or reject the signal in light of historical outcomes.
        Returns probability (0-1) or None if Ollama unavailable.
        """
        lessons = await self.get_lessons(product_id, limit=5)

        # Compact indicator summary by source
        ind_str = _format_indicators(source, indicators)

        lesson_block = ""
        if lessons:
            lesson_block = (
                "\n\nPast 4-hour outcomes for this asset:\n"
                + "\n".join(f"  • {l}" for l in lessons)
            )
        else:
            lesson_block = "\n\nNo past outcomes recorded yet for this asset."

        model  = "qwen2.5:7b"
        prompt = (
            f"{source} agent just signaled {side} for {product_id} "
            f"at ${price:,.4f}\n"
            f"Confidence: {confidence:.2f} | {ind_str}"
            f"{lesson_block}\n\n"
            f"Given this signal and the historical outcomes above, "
            f"what is the probability this {side} leads to a favorable "
            f"price move in the next 4 hours?\n"
            f'Respond with ONLY valid JSON: {{"probability": <0.00-1.00>}}'
        )

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": model, "prompt": prompt,
                          "stream": False, "format": "json"},
                )
                resp.raise_for_status()
                text = resp.json().get("response", "")
            prob = float(json.loads(text).get("probability", -1))
            if 0 <= prob <= 1:
                logger.info(
                    f"OutcomeTracker Ollama validation {source} {side} "
                    f"{product_id}: p={prob:.3f}"
                )
                return prob
        except Exception:
            try:
                m = re.search(r"\b0\.\d{2,4}\b", text)
                if m:
                    return float(m.group())
            except Exception:
                pass
        return None

    # ── Check pending outcomes ─────────────────────────────────────────────────

    async def check_pending(self) -> int:
        """
        Resolve all outcomes whose 4-hour window has passed.
        Fetches the current price, computes WIN/LOSS/NEUTRAL, writes lesson_text.
        Returns count of outcomes resolved.
        """
        rows = await database.get_pending_outcomes()
        resolved = 0
        for row in rows:
            pid        = row["product_id"]
            side       = row["side"]
            entry      = row["entry_price"]
            source     = row["source"]
            confidence = row["confidence"]

            # Get current price — prefer fresh candle close, fallback to DB
            exit_price = None
            try:
                candles = await coinbase_client.get_candles(pid, "ONE_HOUR", limit=1)
                if candles:
                    exit_price = candles[-1]["close"]
            except Exception:
                pass
            if not exit_price:
                product = await database.get_product(pid)
                if product:
                    exit_price = product.get("price")
            if not exit_price:
                continue   # can't resolve — skip until next run

            # pct_change from the signal's perspective:
            # BUY: positive = good, SELL: negative entry→exit = good
            raw_chg = (exit_price - entry) / max(entry, 1e-9)
            if side == "SELL":
                pct_change = -raw_chg   # SELL wins when price drops
            else:
                pct_change = raw_chg

            if pct_change > _WIN_THRESHOLD:
                outcome = "WIN"
            elif pct_change < -_LOSS_THRESHOLD:
                outcome = "LOSS"
            else:
                outcome = "NEUTRAL"

            # Build compact lesson text
            try:
                ind = json.loads(row.get("indicators_json") or "{}")
            except Exception:
                ind = {}
            ind_str = _format_indicators(source, ind)
            lesson_text = (
                f"{source} {side} conf={confidence:.2f} {ind_str} "
                f"→ {raw_chg:+.1%} after 4h [{outcome}]"
            )

            await database.resolve_signal_outcome(
                row_id      = row["id"],
                exit_price  = round(exit_price, 6),
                pct_change  = round(pct_change, 6),
                outcome     = outcome,
                lesson_text = lesson_text,
            )
            resolved += 1
            logger.info(f"Outcome resolved: {lesson_text}")

        return resolved

    # ── Get lessons for Ollama injection ──────────────────────────────────────

    async def get_lessons(self, product_id: str, limit: int = 5) -> List[str]:
        """Return up to `limit` recent lesson strings for this product."""
        return await database.get_recent_lessons(product_id, limit)

    # ── Background loop ───────────────────────────────────────────────────────

    async def run_loop(self, interval: int = 1800) -> None:
        logger.info(f"OutcomeTracker loop started | check_interval={interval}s | horizon=4h")
        while True:
            try:
                resolved = await self.check_pending()
                if resolved:
                    logger.info(f"OutcomeTracker resolved {resolved} outcome(s) this cycle")
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"OutcomeTracker loop error: {e}")
            await asyncio.sleep(interval)


# ── Indicator summary helpers ─────────────────────────────────────────────────

def _format_indicators(source: str, ind: Dict) -> str:
    if source == "TECH":
        parts = []
        if "rsi"    in ind: parts.append(f"RSI={ind['rsi']:.0f}")
        if "bb_pos" in ind: parts.append(f"BB={ind['bb_pos']:.2f}")
        if "macd_h" in ind: parts.append(f"MACD={'bull' if ind['macd_h'] > 0 else 'bear'}")
        if "stoch_k" in ind: parts.append(f"Stoch={ind['stoch_k']:.0f}")
        return " ".join(parts)
    elif source == "MOMENTUM":
        parts = []
        if "mom_s"       in ind: parts.append(f"mom5d={ind['mom_s']*100:+.1f}%")
        if "mom_m"       in ind: parts.append(f"mom10d={ind['mom_m']*100:+.1f}%")
        if "consistency" in ind: parts.append(f"trend={ind['consistency']*100:.0f}%")
        return " ".join(parts)
    elif source == "CNN":
        parts = []
        if "cnn_prob" in ind: parts.append(f"cnn={ind['cnn_prob']:.2f}")
        if "adx"      in ind: parts.append(f"ADX={ind['adx']:.0f}")
        if "regime"   in ind: parts.append(f"regime={ind['regime']}")
        if "rsi"      in ind: parts.append(f"RSI={ind['rsi']:.0f}")
        return " ".join(parts)
    return ""


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracker: Optional[OutcomeTracker] = None


def get_tracker() -> OutcomeTracker:
    global _tracker
    if _tracker is None:
        _tracker = OutcomeTracker()
    return _tracker
