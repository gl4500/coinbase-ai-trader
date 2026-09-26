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
from config import config
from services import outcome_labels as ol

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"
_WIN_THRESHOLD = 0.005  # +0.5% = WIN for a BUY
_LOSS_THRESHOLD = 0.005  # -0.5% adverse = LOSS
_CHECK_HORIZON = 4 * 3600  # check 4 hours after signal


# ── Outcome Tracker ────────────────────────────────────────────────────────────


class OutcomeTracker:
    # ── Record a pending outcome ───────────────────────────────────────────────

    async def record(
        self,
        source: str,  # CNN (TECH/MOMENTUM retired; historical rows remain)
        product_id: str,
        side: str,  # BUY | SELL
        confidence: float,
        price: float,
        indicators: Dict,
    ) -> None:
        """Save a pending signal. Outcome checked 4 h later by check_pending()."""
        try:
            signal_time = time.time()
            await database.insert_signal_outcome(
                {
                    "source": source,
                    "product_id": product_id,
                    "side": side,
                    "confidence": round(confidence, 4),
                    "entry_price": round(price, 6),
                    "indicators_json": json.dumps(indicators),
                    "check_after": signal_time + _CHECK_HORIZON,
                    "signal_time": signal_time,
                }
            )
            logger.debug(f"OutcomeTracker recorded {source} {side} {product_id} @ ${price:.4f}")
        except Exception as e:
            logger.warning(f"OutcomeTracker.record failed: {e}")

    # ── Validate immediately via Ollama ────────────────────────────────────────

    async def validate_with_ollama(
        self,
        source: str,
        product_id: str,
        side: str,
        confidence: float,
        price: float,
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
            lesson_block = "\n\nPast 4-hour outcomes for this asset:\n" + "\n".join(
                f"  • {lesson}" for lesson in lessons
            )
        else:
            lesson_block = "\n\nNo past outcomes recorded yet for this asset."

        model = config.ollama_model
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
            _t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False, "format": "json"},
                )
                resp.raise_for_status()
                text = resp.json().get("response", "")
            _elapsed = time.perf_counter() - _t0
            if _elapsed > 15:
                logger.warning(
                    f"[OLLAMA_LATENCY] app=polymarket caller=validate_with_ollama model={model} elapsed={_elapsed:.2f}s (SLOW)"
                )
            else:
                logger.info(
                    f"[OLLAMA_LATENCY] app=polymarket caller=validate_with_ollama model={model} elapsed={_elapsed:.2f}s"
                )
            prob = float(json.loads(text).get("probability", -1))
            if 0 <= prob <= 1:
                logger.info(
                    f"OutcomeTracker Ollama validation {source} {side} {product_id}: p={prob:.3f}"
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

    async def check_pending(self, now: Optional[float] = None) -> int:
        """Resolve matured outcomes at their defined target time.

        Label version 2 (docs/specs/2026-09-26-outcome-label-contract.md): the
        label is the endpoint return between two completed hourly bars fixed at
        record time. This method never reads a live price — version 1 resolved
        overdue rows with whatever price was current when it ran, which the
        2026-09-26 audit measured at a mean 45.75 h after the nominal horizon.

        A row whose bars are missing stays unresolved and its attempt count
        rises; once the contract's retry budget or grace window is spent it is
        marked UNAVAILABLE, which is terminal and never scored.
        """
        now = time.time() if now is None else now
        rows = await database.get_pending_outcomes()
        resolved = 0

        for row in rows:
            pid = row["product_id"]
            signal_time = _signal_time_of(row)
            plan_entry = ol.entry_candle_start(signal_time)
            plan_exit = ol.exit_candle_start(plan_entry)

            candles = await database.get_candles_at(pid, [plan_entry, plan_exit])
            result = ol.resolve(
                signal_time=signal_time,
                side=row["side"],
                candles=candles,
                now=now,
                attempts=row.get("resolve_attempts") or 0,
            )

            if result.status == "RESOLVED":
                changed = await database.resolve_signal_outcome_v2(
                    row_id=row["id"],
                    outcome=result.outcome,
                    signed_return=result.signed_return,
                    entry_price_v2=result.entry_price,
                    target_price=result.target_price,
                    price_observed_at=result.price_observed_at,
                    price_source=result.price_source,
                    lesson_text=_lesson(row, result),
                )
                if changed:
                    resolved += 1
                    logger.info(f"Outcome resolved: {_lesson(row, result)}")
            elif result.status == "UNAVAILABLE":
                await database.mark_signal_outcome_unavailable(row["id"], result.reason)
                logger.info(f"Outcome unavailable: {pid} id={row['id']} reason={result.reason}")
            elif result.reason != "not_matured":
                await database.bump_signal_outcome_attempts(row["id"])

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


# ── Label-version-2 helpers ───────────────────────────────────────────────────


def _signal_time_of(row: Dict) -> float:
    """Recover the signal time a row's schedule was derived from.

    Version-2 rows store `entry_candle_start`; any instant inside the preceding
    bar maps back to it, so `entry_candle_start - 1` reproduces the schedule
    exactly. Legacy pending rows have no schedule, so it is derived from
    `check_after`, which version 1 set to signal time + the horizon.
    """
    entry_start = row.get("entry_candle_start")
    if entry_start:
        return float(entry_start) - 1
    return float(row["check_after"]) - ol.H_BARS * ol.BAR_SECS


def _lesson(row: Dict, result: "ol.Resolution") -> str:
    """Compact lesson string describing the measured window, not the delay."""
    try:
        ind = json.loads(row.get("indicators_json") or "{}")
    except Exception:
        ind = {}
    ind_str = _format_indicators(row["source"], ind)
    return (
        f"{row['source']} {row['side']} conf={row['confidence']:.2f} {ind_str} "
        f"-> {result.signed_return:+.1%} over {ol.H_BARS}h [{result.outcome}]"
    )


# ── Indicator summary helpers ─────────────────────────────────────────────────


def _format_indicators(source: str, ind: Dict) -> str:
    # TECH branch removed #311-refactor-c (TechAgent retired). Historical
    # source='TECH' rows remain in signal_outcomes but are never re-formatted.
    if source == "MOMENTUM":
        parts = []
        if "mom_s" in ind:
            parts.append(f"mom5d={ind['mom_s'] * 100:+.1f}%")
        if "mom_m" in ind:
            parts.append(f"mom10d={ind['mom_m'] * 100:+.1f}%")
        if "consistency" in ind:
            parts.append(f"trend={ind['consistency'] * 100:.0f}%")
        return " ".join(parts)
    elif source == "CNN":
        parts = []
        if "cnn_prob" in ind:
            parts.append(f"cnn={ind['cnn_prob']:.2f}")
        if "adx" in ind:
            parts.append(f"ADX={ind['adx']:.0f}")
        if "regime" in ind:
            parts.append(f"regime={ind['regime']}")
        if "rsi" in ind:
            parts.append(f"RSI={ind['rsi']:.0f}")
        return " ".join(parts)
    return ""


# ── Singleton ─────────────────────────────────────────────────────────────────

_tracker: Optional[OutcomeTracker] = None


def get_tracker() -> OutcomeTracker:
    global _tracker
    if _tracker is None:
        _tracker = OutcomeTracker()
    return _tracker
