"""
Macro Signal Service — Crypto market structure indicators.

Fetches real-time/near-real-time signals from free, no-auth APIs:
  • Funding rate         — Binance perpetual futures (8-hour resets)
  • Long/Short ratio     — Binance global account ratio (hourly)
  • Open Interest        — Binance futures OI in USD (real-time)
  • BTC Dominance        — CoinGecko global market data (daily)
  • Coinbase Premium     — CB vs Binance price spread (calculated)

All signals collapse into a MacroContext dataclass that exposes:
  buy_gate_multiplier()  → float [0, 1.5]  applied to buy scores
  sell_gate_multiplier() → float [0.5, 1.2] applied to sell scores
  regime_label()         → str  human-readable market regime

Usage in agents:
    from services.macro_signals import get_macro_service
    macro = await get_macro_service().get_macro_context()
    adjusted_buy_score = buy_score * macro.buy_gate_multiplier()
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ── Endpoints (all free, no auth required) ─────────────────────────────────────
_BINANCE_FUTURES  = "https://fapi.binance.com"
_BINANCE_SPOT     = "https://api.binance.com"
_HTTP_TIMEOUT     = 8  # seconds

# ── Thresholds ─────────────────────────────────────────────────────────────────
_FUNDING_OVERHEATED     =  0.001    # >+0.10% per 8h → longs overextended, suppress BUY
_FUNDING_OVERSOLD       = -0.001    # <-0.10% per 8h → shorts crowded, boost BUY (contrarian)
_LS_RATIO_LONG_HEAVY    =  1.8      # >1.8 (64%+ long) → overheated, reduce BUY
_LS_RATIO_SHORT_HEAVY   =  0.8      # <0.8 (44%+ short) → contrarian BUY boost
_COINBASE_PREMIUM_BEARISH = -0.002  # <-0.20% → US institutions selling, reduce BUY
_OI_SURGE_THRESHOLD     =  0.15     # OI rose >15% since last check → caution
_DEFAULT_CACHE_TTL      =  3_600    # 1 hour


# ── MacroContext ───────────────────────────────────────────────────────────────

@dataclass
class MacroContext:
    funding_rate:      float        # 8-hour Binance BTC perp funding rate
    ls_ratio:          float        # global long/short account ratio (>1 = more longs)
    oi_usd:            float        # BTC open interest in USD
    oi_trend:          float        # (current - prev) / prev — positive = OI growing
    btc_dominance:     float        # BTC market cap % of total crypto
    coinbase_premium:  float        # (CB_price - Binance_price) / Binance_price
    fetch_ok:          bool = True  # False when network errors occurred
    fetched_at:        float = field(default_factory=time.time)

    # ── Buy gate ──────────────────────────────────────────────────────────────

    def buy_gate_multiplier(self) -> float:
        """
        Returns a multiplier [0.0, 1.5] applied to agent buy scores.
        1.0 = neutral (no change). <1.0 = suppress BUY. >1.0 = boost BUY.
        When fetch_ok=False, returns a conservative 0.7 to avoid FOMO
        during uncertain market data.
        """
        if not self.fetch_ok:
            return 1.0  # neutral — don't suppress when data is unavailable

        mult = 1.0

        # ── Funding rate ──────────────────────────────────────────────────────
        if self.funding_rate > _FUNDING_OVERHEATED * 1.5:   # very hot (>0.15%)
            mult *= 0.45
        elif self.funding_rate > _FUNDING_OVERHEATED:        # hot (>0.10%)
            mult *= 0.70
        elif self.funding_rate < _FUNDING_OVERSOLD * 1.5:   # very negative (<-0.15%)
            mult *= 1.40   # strong contrarian BUY — shorts will be squeezed
        elif self.funding_rate < _FUNDING_OVERSOLD:          # negative (<-0.10%)
            mult *= 1.20   # moderate contrarian boost

        # ── Long/Short ratio ──────────────────────────────────────────────────
        if self.ls_ratio > _LS_RATIO_LONG_HEAVY * 1.2:     # very long-heavy (>2.2)
            mult *= 0.55
        elif self.ls_ratio > _LS_RATIO_LONG_HEAVY:          # long-heavy (>1.8)
            mult *= 0.80
        elif self.ls_ratio < _LS_RATIO_SHORT_HEAVY * 0.8:  # very short-heavy (<0.64)
            mult *= 1.25
        elif self.ls_ratio < _LS_RATIO_SHORT_HEAVY:         # short-heavy (<0.80)
            mult *= 1.10

        # ── Coinbase premium ──────────────────────────────────────────────────
        if self.coinbase_premium < _COINBASE_PREMIUM_BEARISH:
            mult *= 0.85   # US selling into rally — caution

        # ── OI trend ──────────────────────────────────────────────────────────
        if self.oi_trend > _OI_SURGE_THRESHOLD and self.funding_rate > 0:
            # OI surging + positive funding = leveraged longs piling in — risky
            mult *= 0.90

        return max(0.0, min(1.5, mult))

    # ── Sell gate ─────────────────────────────────────────────────────────────

    def sell_gate_multiplier(self) -> float:
        """
        Returns a multiplier [0.5, 1.2] applied to agent sell scores.
        SELL is never strongly suppressed — we must always be able to exit.
        Reduce sell strength only when shorts are massively crowded (squeeze risk).
        """
        if not self.fetch_ok:
            return 1.0   # neutral on failure — don't suppress exits

        mult = 1.0

        # Shorts very crowded → potential squeeze → don't sell into the lows
        if self.funding_rate < _FUNDING_OVERSOLD * 1.5:    # <-0.15%
            mult *= 0.75
        elif self.funding_rate < _FUNDING_OVERSOLD:         # <-0.10%
            mult *= 0.90

        # Overheated longs → amplify SELL signal slightly
        if self.funding_rate > _FUNDING_OVERHEATED * 1.5:
            mult *= 1.15

        return max(0.5, min(1.2, mult))

    # ── Regime label ──────────────────────────────────────────────────────────

    def regime_label(self) -> str:
        """Human-readable description of the current macro regime."""
        if not self.fetch_ok:
            return "UNKNOWN (fetch failed)"

        funding_pct = self.funding_rate * 100

        if self.funding_rate > _FUNDING_OVERHEATED and self.ls_ratio > _LS_RATIO_LONG_HEAVY:
            return f"OVERHEATED LONGS (funding={funding_pct:+.3f}% L/S={self.ls_ratio:.2f})"

        if self.funding_rate < _FUNDING_OVERSOLD and self.ls_ratio < _LS_RATIO_SHORT_HEAVY:
            return f"SHORT SQUEEZE SETUP (funding={funding_pct:+.3f}% L/S={self.ls_ratio:.2f})"

        if self.funding_rate > _FUNDING_OVERHEATED:
            return f"CAUTION: funding hot ({funding_pct:+.3f}%)"

        if self.funding_rate < _FUNDING_OVERSOLD:
            return f"CONTRARIAN BULLISH: shorts crowded (funding={funding_pct:+.3f}%)"

        return f"NEUTRAL (funding={funding_pct:+.3f}% L/S={self.ls_ratio:.2f})"

    # ── Summary dict for logging ──────────────────────────────────────────────

    def as_log_str(self) -> str:
        return (
            f"regime={self.regime_label()} | "
            f"funding={self.funding_rate*100:+.4f}%/8h | "
            f"L/S={self.ls_ratio:.2f} | "
            f"OI_trend={self.oi_trend:+.2%} | "
            f"BTC_dom={self.btc_dominance:.1f}% | "
            f"CB_premium={self.coinbase_premium*100:+.3f}% | "
            f"buy_mult={self.buy_gate_multiplier():.2f} | "
            f"sell_mult={self.sell_gate_multiplier():.2f}"
        )


# ── MacroSignalService ─────────────────────────────────────────────────────────

class MacroSignalService:
    """
    Fetches and caches macro signals from free APIs.
    All fetch methods catch exceptions and return safe defaults
    so a network outage never crashes an agent.
    """

    def __init__(self, cache_ttl: int = _DEFAULT_CACHE_TTL):
        self._cache_ttl = cache_ttl
        self._cache: Optional[MacroContext] = None
        self._cache_ts: float = 0.0
        self._prev_oi: Optional[float] = None  # for OI trend calculation

    # ── Individual fetchers ────────────────────────────────────────────────────

    async def _fetch_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """Binance perpetual funding rate for `symbol`. Returns 0.0 on error."""
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BINANCE_FUTURES}/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data[0]["fundingRate"])
        except Exception as e:
            logger.debug(f"Funding rate fetch failed ({symbol}): {e}")
            return 0.0

    async def _fetch_ls_ratio(self, symbol: str = "BTCUSDT") -> float:
        """
        Binance global long/short account ratio.
        Returns 1.0 (neutral) on error.
        """
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BINANCE_FUTURES}/futures/data/globalLongShortAccountRatio",
                    params={"symbol": symbol, "period": "1h", "limit": 1},
                )
                resp.raise_for_status()
                data = resp.json()
                return float(data[0]["longShortRatio"])
        except Exception as e:
            logger.debug(f"L/S ratio fetch failed ({symbol}): {e}")
            return 1.0

    async def _fetch_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """Binance futures open interest in contracts. Returns 0.0 on error."""
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BINANCE_FUTURES}/fapi/v1/openInterest",
                    params={"symbol": symbol},
                )
                resp.raise_for_status()
                return float(resp.json()["openInterest"])
        except Exception as e:
            logger.debug(f"Open interest fetch failed ({symbol}): {e}")
            return 0.0

    async def _fetch_btc_dominance(self) -> float:
        """
        BTC dominance estimated from Binance 24h ticker data.
        Sums market cap proxy (price × 24h volume) for BTC vs top alts.
        Returns 50.0 on error (neutral).
        Note: this is an approximation — Binance only lists its own pairs,
        but BTC dominance relative to listed alts is still a useful signal.
        """
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                resp = await client.get(
                    f"{_BINANCE_FUTURES}/fapi/v1/ticker/24hr",
                )
                resp.raise_for_status()
                tickers = resp.json()

            btc_vol  = 0.0
            total_vol = 0.0
            for t in tickers:
                sym    = t.get("symbol", "")
                vol    = float(t.get("quoteVolume", 0) or 0)
                total_vol += vol
                if sym.startswith("BTC"):
                    btc_vol += vol

            if total_vol <= 0:
                return 50.0
            return round(btc_vol / total_vol * 100, 2)
        except Exception as e:
            logger.debug(f"BTC dominance (Binance) fetch failed: {e}")
            return 50.0

    async def _fetch_coinbase_premium(self) -> float:
        """
        Coinbase BTC price vs Binance BTC price, expressed as a fraction.
        premium > 0  → CB more expensive → US buyers aggressive (bullish)
        premium < 0  → CB cheaper       → US selling (bearish)
        Returns 0.0 on error (neutral).
        """
        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT) as client:
                cb_resp  = await client.get(
                    "https://api.coinbase.com/v2/prices/BTC-USD/spot"
                )
                bn_resp  = await client.get(
                    f"{_BINANCE_SPOT}/api/v3/ticker/price",
                    params={"symbol": "BTCUSDT"},
                )
                cb_price = float(cb_resp.json()["data"]["amount"])
                bn_price = float(bn_resp.json()["price"])
                return (cb_price - bn_price) / bn_price if bn_price > 0 else 0.0
        except Exception as e:
            logger.debug(f"Coinbase premium fetch failed: {e}")
            return 0.0

    # ── Main entry point ───────────────────────────────────────────────────────

    async def get_macro_context(self, symbol: str = "BTCUSDT") -> MacroContext:
        """
        Returns a MacroContext (cached for cache_ttl seconds).
        Never raises — on total failure returns a conservative context
        with fetch_ok=False.
        """
        now = time.time()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            return self._cache

        fetch_ok = True
        try:
            funding  = await self._fetch_funding_rate(symbol)
            ls       = await self._fetch_ls_ratio(symbol)
            oi       = await self._fetch_open_interest(symbol)
            dom      = await self._fetch_btc_dominance()
            premium  = await self._fetch_coinbase_premium()

            # OI trend: compare with previous cached OI
            oi_trend = 0.0
            if self._prev_oi and self._prev_oi > 0:
                oi_trend = (oi - self._prev_oi) / self._prev_oi
            self._prev_oi = oi if oi > 0 else self._prev_oi

            ctx = MacroContext(
                funding_rate     = funding,
                ls_ratio         = ls,
                oi_usd           = oi,
                oi_trend         = oi_trend,
                btc_dominance    = dom,
                coinbase_premium = premium,
                fetch_ok         = True,
                fetched_at       = now,
            )
            logger.info(f"MacroSignals: {ctx.as_log_str()}")

        except Exception as e:
            logger.warning(f"MacroSignalService fetch error: {e}")
            fetch_ok = False
            ctx = MacroContext(
                funding_rate=0.0, ls_ratio=1.0, oi_usd=0.0,
                oi_trend=0.0, btc_dominance=50.0, coinbase_premium=0.0,
                fetch_ok=False, fetched_at=now,
            )

        self._cache    = ctx
        self._cache_ts = now
        return ctx


# ── Module-level singleton ─────────────────────────────────────────────────────

_instance: Optional[MacroSignalService] = None


def get_macro_service() -> MacroSignalService:
    global _instance
    if _instance is None:
        _instance = MacroSignalService()
    return _instance
