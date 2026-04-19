"""
Signal Generator — Technical Analysis
──────────────────────────────────────
Generates BUY/SELL signals from TA indicators computed on hourly candle data.

Indicators:
  • RSI(14)              — oversold / overbought
  • EMA cross(9/21)      — golden / death cross
  • MACD(5,13,3)         — histogram sign (fast params for 1h crypto)
  • Bollinger(20,2)      — band position
  • ADX(14)              — trend regime gate (< threshold → suppress momentum signals)
  • MFI(14)              — volume-weighted RSI (accumulation/distribution)
  • OBV slope            — on-balance volume trend
  • Stochastic RSI(14,3,3) — fast overbought/oversold
  • ATR(14)              — volatility for position sizing

Signals below min_signal_strength are suppressed.
ADX below adx_trend_threshold suppresses momentum signals (chop filter).
"""
import asyncio
import json
import logging
import math
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

import database
from config import config

logger    = logging.getLogger(__name__)
OLLAMA_URL = "http://localhost:11434"
_CACHE_TTL = 300


# ── Core Indicators ───────────────────────────────────────────────────────────

def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains  = [max(d, 0) for d in deltas]
    losses = [max(-d, 0) for d in deltas]
    ag = sum(gains[:period])  / period
    al = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        ag = (ag * (period - 1) + gains[i])  / period
        al = (al * (period - 1) + losses[i]) / period
    return 100.0 if al == 0 else 100 - (100 / (1 + ag / al))


def _ema(prices: List[float], period: int) -> List[float]:
    if len(prices) < period:
        return [prices[-1]] * len(prices) if prices else []
    k   = 2.0 / (period + 1)
    ema = [sum(prices[:period]) / period]
    for p in prices[period:]:
        ema.append(p * k + ema[-1] * (1 - k))
    return ema


def _macd(closes: List[float],
          fast: int = 5, slow: int = 13,
          signal: int = 3) -> Tuple[float, float, float]:
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast  = _ema(closes, fast)
    ema_slow  = _ema(closes, slow)
    diff      = len(ema_fast) - len(ema_slow)
    macd_line = [f - s for f, s in zip(ema_fast[diff:], ema_slow)]
    if len(macd_line) < signal:
        return macd_line[-1], macd_line[-1], 0.0
    sig_line = _ema(macd_line, signal)
    hist     = macd_line[-1] - sig_line[-1]
    return macd_line[-1], sig_line[-1], hist


def _bollinger(closes: List[float],
               period: int = 20, mult: float = 2.0) -> Tuple[float, float, float, float]:
    if len(closes) < period:
        p = closes[-1] if closes else 0
        return p, p, p, 0.5
    window = closes[-period:]
    mean   = sum(window) / period
    std    = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
    upper  = mean + mult * std
    lower  = mean - mult * std
    bw     = upper - lower
    pos    = (closes[-1] - lower) / bw if bw > 0 else 0.5
    return upper, mean, lower, max(0.0, min(1.0, pos))


def _ema_cross(closes: List[float]) -> float:
    if len(closes) < 22:
        return 0.0
    ema9  = _ema(closes, 9)[-1]
    ema21 = _ema(closes, 21)[-1]
    mid   = (ema9 + ema21) / 2
    return (ema9 - ema21) / mid if mid else 0.0


def _atr(highs: List[float], lows: List[float],
         closes: List[float], period: int = 14) -> float:
    """Average True Range — Wilder smoothing."""
    if len(closes) < period + 1:
        return 0.0
    trs = [max(highs[i] - lows[i],
               abs(highs[i]  - closes[i - 1]),
               abs(lows[i]   - closes[i - 1]))
           for i in range(1, len(closes))]
    atr = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


def _adx(highs: List[float], lows: List[float],
         closes: List[float], period: int = 14) -> Tuple[float, float, float]:
    """
    Returns (ADX, +DI, -DI).
    ADX > 25 → trending; < 25 → ranging/choppy.
    Uses Wilder smoothing (same as ATR).
    """
    if len(closes) < period * 2 + 1:
        return 20.0, 0.0, 0.0

    def _wilder(data: List[float], n: int) -> List[float]:
        if len(data) < n:
            return [0.0]
        s = [sum(data[:n]) / n]   # mean init (Wilder smoothing)
        for v in data[n:]:
            s.append((s[-1] * (n - 1) + v) / n)
        return s

    dm_plus, dm_minus, trs = [], [], []
    for i in range(1, len(closes)):
        up   = highs[i]  - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_plus.append(max(up, 0)   if up > down  else 0.0)
        dm_minus.append(max(down, 0) if down > up  else 0.0)
        trs.append(max(highs[i] - lows[i],
                       abs(highs[i]  - closes[i - 1]),
                       abs(lows[i]   - closes[i - 1])))

    atr_s  = _wilder(trs,      period)
    dmp_s  = _wilder(dm_plus,  period)
    dmm_s  = _wilder(dm_minus, period)

    di_plus  = [100 * p / a if a > 0 else 0 for p, a in zip(dmp_s, atr_s)]
    di_minus = [100 * m / a if a > 0 else 0 for m, a in zip(dmm_s, atr_s)]
    dx       = [abs(p - m) / (p + m) * 100 if (p + m) > 0 else 0
                for p, m in zip(di_plus, di_minus)]

    if len(dx) < period:
        return 20.0, di_plus[-1] if di_plus else 0.0, di_minus[-1] if di_minus else 0.0

    adx_s = _wilder(dx, period)
    return adx_s[-1], di_plus[-1], di_minus[-1]


def _mfi(highs: List[float], lows: List[float],
         closes: List[float], volumes: List[float],
         period: int = 14) -> float:
    """Money Flow Index — volume-weighted RSI."""
    if len(closes) < period + 1:
        return 50.0
    tp  = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    rmf = [t * v for t, v in zip(tp, volumes)]
    pos = sum(rmf[i] for i in range(len(tp) - period, len(tp))
              if i > 0 and tp[i] > tp[i - 1])
    neg = sum(rmf[i] for i in range(len(tp) - period, len(tp))
              if i > 0 and tp[i] <= tp[i - 1])
    return 100.0 if neg == 0 else 100 - (100 / (1 + pos / neg))


def _obv_slope(closes: List[float], volumes: List[float],
               period: int = 10) -> float:
    """
    On-Balance Volume slope over last `period` bars, normalised to [-1, 1].
    Positive = accumulation; negative = distribution.
    """
    if len(closes) < period + 1:
        return 0.0
    obv = [0.0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i - 1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    window = obv[-period:]
    n      = len(window)
    xm     = (n - 1) / 2
    ym     = sum(window) / n
    num    = sum((i - xm) * (window[i] - ym) for i in range(n))
    den    = sum((i - xm) ** 2 for i in range(n))
    slope  = num / den if den else 0.0
    mag    = max(abs(ym), 1.0)
    return max(-1.0, min(1.0, slope * period / mag))


def _vwap(highs: List[float], lows: List[float],
          closes: List[float], volumes: List[float],
          period: int = 20) -> Tuple[float, float]:
    """
    Rolling VWAP over last `period` bars.
    Returns (vwap_price, distance) where distance = (close - vwap) / vwap
      > 0 → price above VWAP (bullish momentum / potential overbought)
      < 0 → price below VWAP (bearish momentum / potential oversold)
    Distance is clipped to [-0.05, 0.05] and normalised to [-1, 1].
    """
    if len(closes) < period:
        return closes[-1] if closes else 0.0, 0.0
    h = highs[-period:];  l = lows[-period:]
    c = closes[-period:]; v = volumes[-period:]
    tp       = [(hi + lo + cl) / 3 for hi, lo, cl in zip(h, l, c)]
    vol_sum  = sum(v)
    if vol_sum == 0:
        return c[-1], 0.0
    vwap     = sum(t * vi for t, vi in zip(tp, v)) / vol_sum
    dist     = (c[-1] - vwap) / vwap if vwap else 0.0
    norm     = max(-1.0, min(1.0, dist / 0.05))
    return vwap, norm


def _hurst_exponent(closes: List[float], min_len: int = 50) -> float:
    """
    Estimate Hurst Exponent via R/S analysis (rescaled range).
    H > 0.6 → persistent trend (momentum works).
    H < 0.4 → mean-reverting (oscillator works).
    H ≈ 0.5 → random walk / unknown regime.
    Returns 0.5 (neutral) if fewer than min_len data points.
    """
    n = len(closes)
    if n < min_len:
        return 0.5
    lags = [max(2, n // 8), max(4, n // 4), max(8, n // 2), n]
    rs_vals, lag_vals = [], []
    for lag in lags:
        sub = closes[-lag:]
        if len(sub) < 4:
            continue
        mean   = sum(sub) / len(sub)
        dev    = [x - mean for x in sub]
        cum    = [sum(dev[:i + 1]) for i in range(len(dev))]
        r      = max(cum) - min(cum)
        s      = math.sqrt(sum(d ** 2 for d in dev) / len(dev))
        if s > 0:
            rs_vals.append(math.log(r / s))
            lag_vals.append(math.log(lag))
    if len(rs_vals) < 2:
        return 0.5
    # Linear regression of log(R/S) vs log(lag)
    n_pts  = len(lag_vals)
    xm     = sum(lag_vals) / n_pts
    ym     = sum(rs_vals)  / n_pts
    num    = sum((lag_vals[i] - xm) * (rs_vals[i] - ym) for i in range(n_pts))
    den    = sum((lag_vals[i] - xm) ** 2              for i in range(n_pts))
    h      = num / den if den else 0.5
    return max(0.0, min(1.0, h))


def _multi_rsi(closes: List[float],
               periods: Tuple[int, int, int] = (6, 12, 24),
               oversold: float = 30.0,
               overbought: float = 70.0) -> dict:
    """
    Compute RSI for three periods and return a vote count.
    buy_votes  = number of periods where RSI is oversold (< oversold threshold).
    sell_votes = number of periods where RSI is overbought (> overbought threshold).
    Max 3 votes each. Used to confirm signal strength across timeframes.
    """
    min_needed = max(periods) + 1
    if len(closes) < min_needed:
        return {"rsi6": 50.0, "rsi12": 50.0, "rsi24": 50.0,
                "buy_votes": 0, "sell_votes": 0}
    rsi6  = _rsi(closes, period=periods[0])
    rsi12 = _rsi(closes, period=periods[1])
    rsi24 = _rsi(closes, period=periods[2])
    buy_votes  = sum(1 for r in (rsi6, rsi12, rsi24) if r < oversold)
    sell_votes = sum(1 for r in (rsi6, rsi12, rsi24) if r > overbought)
    return {
        "rsi6":  round(rsi6,  1),
        "rsi12": round(rsi12, 1),
        "rsi24": round(rsi24, 1),
        "buy_votes":  buy_votes,
        "sell_votes": sell_votes,
    }


def _dissimilarity_index(closes: List[float], period: int = 14) -> float:
    """
    Dissimilarity Index (DI): percentage distance of current price from its SMA.
    DI = |close[-1] - SMA(close, period)| / SMA × 100

    When DI > ~3%, CNN/LSTM features become unreliable because the price has
    strayed far from the distribution seen during training.
    Returns 0.0 when there are insufficient data points.
    """
    if len(closes) < period:
        return 0.0
    sma = sum(closes[-period:]) / period
    if sma == 0:
        return 0.0
    return abs(closes[-1] - sma) / sma * 100.0


def _kelly_fraction(confidence: float, max_frac: float = 0.25) -> float:
    """
    Fractional Kelly Criterion (25% of full Kelly) for position sizing.
    For a binary win/loss bet where win_prob = confidence:
      full Kelly  = 2p - 1
      capped Kelly = min(max(full_kelly, 0), max_frac)
    Returns 0.0 when confidence ≤ 0.5 (no edge → don't trade).
    """
    full_kelly = 2.0 * confidence - 1.0
    return max(0.0, min(full_kelly, max_frac))


def _realized_vol(closes: List[float], window: int = 20,
                  annualize_days: int = 365) -> float:
    """
    Annualized realized volatility from log returns.
    annualize_days=365 for crypto (trades 24/7).
    Returns 0.0 if insufficient data.
    """
    if len(closes) < window + 1:
        return 0.0
    log_returns = [math.log(closes[i] / closes[i - 1])
                   for i in range(len(closes) - window, len(closes))
                   if closes[i - 1] > 0 and closes[i] > 0]
    if len(log_returns) < 2:
        return 0.0
    mean = sum(log_returns) / len(log_returns)
    variance = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
    return math.sqrt(variance * annualize_days)


def _shannon_entropy(closes: List[float], window: int = 20, n_bins: int = 10) -> float:
    """
    Shannon entropy of the log-return distribution over the last `window` bars.
    High entropy (~log2(n_bins)) = returns are uniformly distributed = noise.
    Low entropy = returns are concentrated = information / structure present.
    Normalised to [0, 1]: 0 = pure structure, 1 = pure noise.
    Returns 0.5 (neutral) if insufficient data.
    """
    if len(closes) < window + 1:
        return 0.5
    log_returns = [math.log(closes[i] / closes[i - 1])
                   for i in range(len(closes) - window, len(closes))
                   if closes[i - 1] > 0 and closes[i] > 0]
    if len(log_returns) < 4:
        return 0.5
    mn, mx = min(log_returns), max(log_returns)
    if mx == mn:
        return 0.0   # all returns identical = zero entropy
    bin_width = (mx - mn) / n_bins
    counts = [0] * n_bins
    for r in log_returns:
        idx = min(int((r - mn) / bin_width), n_bins - 1)
        counts[idx] += 1
    n = len(log_returns)
    entropy = -sum((c / n) * math.log2(c / n) for c in counts if c > 0)
    max_entropy = math.log2(n_bins)
    return round(entropy / max_entropy, 4) if max_entropy > 0 else 0.5


def _stoch_rsi(closes: List[float],
               period: int = 14, k_smooth: int = 3, d_smooth: int = 3
               ) -> Tuple[float, float]:
    """
    Stochastic RSI.  Returns (K, D) in range [0, 100].
    K < 20 → oversold;  K > 80 → overbought.
    """
    if len(closes) < period * 2:
        return 50.0, 50.0
    rsi_series = [_rsi(closes[:i + 1], period) for i in range(period, len(closes))]
    if len(rsi_series) < period:
        return 50.0, 50.0
    stoch = []
    for i in range(period - 1, len(rsi_series)):
        w  = rsi_series[i - period + 1: i + 1]
        mn, mx = min(w), max(w)
        stoch.append((rsi_series[i] - mn) / (mx - mn) * 100 if mx > mn else 50.0)
    k = _ema(stoch, k_smooth) if len(stoch) >= k_smooth else stoch
    d = _ema(k,     d_smooth) if len(k)     >= d_smooth else k
    return (k[-1] if k else 50.0), (d[-1] if d else 50.0)


# ── Ollama LLM ────────────────────────────────────────────────────────────────

async def _llm_confirm(product_id: str, side: str, context: str) -> Optional[str]:
    model = config.ollama_model
    prompt = (
        f"You are a crypto trading analyst. A {side} signal was detected for {product_id}.\n"
        f"Context:\n{context}\n\n"
        f"In ONE sentence, confirm or challenge this signal. Be concise."
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()[:200]
    except Exception as e:
        logger.debug(f"Ollama confirmation failed: {e}")
        return None


# ── Signal Generator ──────────────────────────────────────────────────────────

class SignalGenerator:
    def __init__(self):
        self._cache: Dict[str, Tuple[float, str]] = {}

    async def generate_signal(self, product: Dict) -> Optional[Dict]:
        pid   = product["product_id"]
        price = product.get("price")
        if not price or price <= 0:
            return None

        candles = await database.get_candles(pid, limit=100)
        if len(candles) < 30:
            return None

        closes  = [c["close"]  for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        volumes = [c["volume"] for c in candles]

        # ── Indicators ────────────────────────────────────────────────────────
        rsi_val              = _rsi(closes)
        macd_l, macd_s, macd_h = _macd(closes)
        _, _, _, bb_pos      = _bollinger(closes)
        ema_cross_val        = _ema_cross(closes)
        adx_val, di_p, di_m = _adx(highs, lows, closes)
        mfi_val              = _mfi(highs, lows, closes, volumes)
        obv_sl               = _obv_slope(closes, volumes)
        stoch_k, stoch_d     = _stoch_rsi(closes)
        atr_val              = _atr(highs, lows, closes)

        trending = adx_val >= config.adx_trend_threshold

        # ── Score signals ─────────────────────────────────────────────────────
        buy_score = sell_score = 0.0
        reasons = []

        # RSI — only weight heavily in trending markets
        rsi_weight = 1.0 if trending else 0.4
        if rsi_val < config.rsi_oversold:
            buy_score  += rsi_weight * (config.rsi_oversold - rsi_val) / config.rsi_oversold
            reasons.append(f"RSI={rsi_val:.1f} oversold")
        elif rsi_val > config.rsi_overbought:
            sell_score += rsi_weight * (rsi_val - config.rsi_overbought) / (100 - config.rsi_overbought)
            reasons.append(f"RSI={rsi_val:.1f} overbought")

        # EMA cross — suppress in ranging markets
        if trending:
            if ema_cross_val > 0.005:
                buy_score  += min(ema_cross_val * 10, 0.4)
                reasons.append(f"EMA9>EMA21 ({ema_cross_val:+.3f})")
            elif ema_cross_val < -0.005:
                sell_score += min(-ema_cross_val * 10, 0.4)
                reasons.append(f"EMA9<EMA21 ({ema_cross_val:+.3f})")

        # MACD — suppress in chop
        macd_weight = 1.0 if trending else 0.5
        bb_mid = sum(closes[-20:]) / min(20, len(closes))
        if macd_h > 0:
            buy_score  += macd_weight * min(abs(macd_h) / max(abs(bb_mid), 1) * 100, 0.3)
            reasons.append(f"MACD={macd_h:+.4f} bullish")
        elif macd_h < 0:
            sell_score += macd_weight * min(abs(macd_h) / max(abs(bb_mid), 1) * 100, 0.3)
            reasons.append(f"MACD={macd_h:+.4f} bearish")

        # Bollinger Bands
        if bb_pos < 0.1:
            buy_score  += (0.1 - bb_pos) * 3
            reasons.append(f"BB lower band (pos={bb_pos:.2f})")
        elif bb_pos > 0.9:
            sell_score += (bb_pos - 0.9) * 3
            reasons.append(f"BB upper band (pos={bb_pos:.2f})")

        # MFI — accumulation/distribution (works in both regimes)
        if mfi_val < 20:
            buy_score  += 0.25
            reasons.append(f"MFI={mfi_val:.1f} oversold (volume)")
        elif mfi_val > 80:
            sell_score += 0.25
            reasons.append(f"MFI={mfi_val:.1f} overbought (volume)")

        # OBV slope — confirm direction
        if obv_sl > 0.2:
            buy_score  += 0.15
            reasons.append(f"OBV accumulating ({obv_sl:+.2f})")
        elif obv_sl < -0.2:
            sell_score += 0.15
            reasons.append(f"OBV distributing ({obv_sl:+.2f})")

        # Stochastic RSI — extra weight in ranging markets
        stoch_weight = 0.4 if trending else 0.7
        if stoch_k < 20 and stoch_d < 20:
            buy_score  += stoch_weight * 0.3
            reasons.append(f"StochRSI K={stoch_k:.1f} oversold")
        elif stoch_k > 80 and stoch_d > 80:
            sell_score += stoch_weight * 0.3
            reasons.append(f"StochRSI K={stoch_k:.1f} overbought")

        # ADX confirms trend strength — bonus for strong trends
        if trending and adx_val > 40:
            if buy_score > sell_score:
                buy_score  *= 1.15
            else:
                sell_score *= 1.15
            reasons.append(f"ADX={adx_val:.1f} strong trend")
        elif not trending:
            reasons.append(f"ADX={adx_val:.1f} ranging (momentum suppressed)")

        buy_score  = min(buy_score,  1.0)
        sell_score = min(sell_score, 1.0)

        if buy_score >= sell_score:
            if buy_score < config.min_signal_strength:
                return None
            side, strength = "BUY", buy_score
        else:
            if sell_score < config.min_signal_strength:
                return None
            side, strength = "SELL", sell_score

        # Duplicate suppression
        cached = self._cache.get(pid)
        if cached and time.time() - cached[0] < _CACHE_TTL and cached[1] == side:
            return None
        self._cache[pid] = (time.time(), side)

        context = (
            f"Price: ${price:,.4f} | Regime: {'TRENDING' if trending else 'RANGING'}\n"
            f"ADX(14): {adx_val:.1f} | RSI(14): {rsi_val:.1f} | MFI(14): {mfi_val:.1f}\n"
            f"MACD hist: {macd_h:+.4f} | EMA cross: {ema_cross_val:+.4f}\n"
            f"Bollinger pos: {bb_pos:.2f} | StochRSI K/D: {stoch_k:.1f}/{stoch_d:.1f}\n"
            f"OBV slope: {obv_sl:+.2f} | ATR(14): {atr_val:.4f}\n"
            f"Signals: {', '.join(reasons)}"
        )

        reasoning = await _llm_confirm(pid, side, context)

        signal = {
            "product_id":  pid,
            "signal_type": f"TA_{'BUY' if side == 'BUY' else 'SELL'}",
            "side":        side,
            "price":       round(price, 6),
            "strength":    round(strength, 3),
            "rsi":         round(rsi_val, 2),
            "macd":        round(macd_h, 6),
            "ema_cross":   round(ema_cross_val, 4),
            "bb_position": round(bb_pos, 3),
            "reasoning":   reasoning or context,
            "atr":         round(atr_val, 6),
            "adx":         round(adx_val, 1),
        }

        signal_id    = await database.save_signal(signal)
        signal["id"] = signal_id

        logger.info(
            f"SIGNAL [{side}] {pid} | strength={strength:.2f} "
            f"RSI={rsi_val:.1f} ADX={adx_val:.1f} MFI={mfi_val:.1f} regime={'trend' if trending else 'range'}"
        )
        return signal

    async def scan_all(self) -> List[Dict]:
        products = await database.get_products(tracked_only=True)
        signals  = []
        for p in products:
            try:
                sig = await self.generate_signal(p)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"Signal error [{p.get('product_id')}]: {e}")
            await asyncio.sleep(0.2)
        signals.sort(key=lambda s: s["strength"], reverse=True)
        logger.info(f"Signal scan: {len(signals)} signal(s)")
        return signals
