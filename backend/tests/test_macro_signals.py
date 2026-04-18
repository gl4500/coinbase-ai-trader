"""
Tests for services/macro_signals.py — Crypto macro indicator service.

Covers:
  - BinanceFundingRate fetch and interpretation
  - BinanceLongShortRatio fetch and interpretation
  - BinanceOpenInterest fetch
  - BTCDominance fetch (CoinGecko)
  - CoinbasePremium calculation
  - MacroContext multiplier logic
  - Caching behaviour
  - Graceful degradation on network failure

All external HTTP calls are mocked.
"""
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from services.macro_signals import (
    MacroContext,
    MacroSignalService,
    _FUNDING_OVERHEATED,
    _FUNDING_OVERSOLD,
    _LS_RATIO_LONG_HEAVY,
    _LS_RATIO_SHORT_HEAVY,
    _COINBASE_PREMIUM_BEARISH,
    get_macro_service,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _mock_http(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _neutral_context() -> MacroContext:
    """A MacroContext with all signals at neutral values."""
    return MacroContext(
        funding_rate=0.0001,       # 0.01% — healthy
        ls_ratio=1.0,              # balanced
        oi_usd=10_000_000_000,
        oi_trend=0.0,              # flat
        btc_dominance=52.0,        # typical
        coinbase_premium=0.0,      # parity
        fetch_ok=True,
    )


def _extreme_long_context() -> MacroContext:
    """Over-leveraged longs — classic before-crash setup."""
    return MacroContext(
        funding_rate=0.0015,       # 0.15% per 8h → very hot
        ls_ratio=2.5,              # 71% long
        oi_usd=15_000_000_000,
        oi_trend=0.25,             # OI rising
        btc_dominance=52.0,
        coinbase_premium=0.05,
        fetch_ok=True,
    )


def _extreme_short_context() -> MacroContext:
    """Short squeeze setup — shorts massively crowded."""
    return MacroContext(
        funding_rate=-0.0012,      # -0.12% per 8h → very negative
        ls_ratio=0.65,             # 39% long (61% short)
        oi_usd=12_000_000_000,
        oi_trend=0.1,
        btc_dominance=52.0,
        coinbase_premium=0.05,
        fetch_ok=True,
    )


# ── MacroContext.buy_gate_multiplier ───────────────────────────────────────────

class TestBuyGateMultiplier:

    def test_neutral_context_multiplier_is_one(self):
        ctx = _neutral_context()
        assert ctx.buy_gate_multiplier() == pytest.approx(1.0, abs=0.01)

    def test_overheated_funding_reduces_buy(self):
        """Funding > threshold → multiplier < 1 (suppress BUY)."""
        ctx = _neutral_context()
        ctx.funding_rate = _FUNDING_OVERHEATED + 0.0005
        assert ctx.buy_gate_multiplier() < 1.0

    def test_extreme_overheated_funding_strongly_reduces_buy(self):
        ctx = _extreme_long_context()
        mult = ctx.buy_gate_multiplier()
        assert mult < 0.6, f"Expected < 0.6 for overheated market, got {mult:.3f}"

    def test_negative_funding_boosts_buy(self):
        """Funding < threshold (shorts crowded) → multiplier > 1.0 (contrarian BUY boost)."""
        ctx = _neutral_context()
        ctx.funding_rate = _FUNDING_OVERSOLD - 0.0005
        assert ctx.buy_gate_multiplier() > 1.0

    def test_multiplier_capped_at_1_5(self):
        """Even in extreme short squeeze setup, multiplier must not exceed 1.5."""
        ctx = _extreme_short_context()
        assert ctx.buy_gate_multiplier() <= 1.5

    def test_long_heavy_ls_ratio_reduces_buy(self):
        """L/S ratio > threshold → reduce BUY strength."""
        ctx = _neutral_context()
        ctx.ls_ratio = _LS_RATIO_LONG_HEAVY + 0.5
        assert ctx.buy_gate_multiplier() < 1.0

    def test_short_heavy_ls_ratio_boosts_buy(self):
        """L/S ratio < threshold → contrarian BUY boost."""
        ctx = _neutral_context()
        ctx.ls_ratio = _LS_RATIO_SHORT_HEAVY - 0.1
        assert ctx.buy_gate_multiplier() > 1.0

    def test_negative_coinbase_premium_reduces_buy(self):
        """Negative CB premium = US selling → reduce BUY."""
        ctx = _neutral_context()
        ctx.coinbase_premium = _COINBASE_PREMIUM_BEARISH - 0.001
        assert ctx.buy_gate_multiplier() < 1.0

    def test_combined_extreme_longs_significantly_reduces_buy(self):
        """All signals pointing to overheated longs → multiplier well below 0.5."""
        ctx = _extreme_long_context()
        ctx.coinbase_premium = -0.003  # also US selling
        mult = ctx.buy_gate_multiplier()
        assert mult < 0.5, f"Extreme long setup should cut BUY to < 50%, got {mult:.3f}"

    def test_multiplier_never_negative(self):
        """Gate multiplier must be ≥ 0 in all circumstances."""
        ctx = _extreme_long_context()
        ctx.funding_rate = 0.005       # absurdly high
        ctx.ls_ratio     = 5.0
        ctx.coinbase_premium = -0.01
        assert ctx.buy_gate_multiplier() >= 0.0

    def test_fetch_failed_returns_conservative_multiplier(self):
        """When fetch_ok=False (network failure), return 1.0 — neutral, don't suppress buys."""
        ctx = _neutral_context()
        ctx.fetch_ok = False
        assert ctx.buy_gate_multiplier() == pytest.approx(1.0, abs=0.01)


class TestSellGateMultiplier:

    def test_neutral_sell_multiplier_is_one(self):
        ctx = _neutral_context()
        assert ctx.sell_gate_multiplier() == pytest.approx(1.0, abs=0.01)

    def test_extreme_short_context_reduces_sell_strength(self):
        """When shorts are massively crowded (squeeze imminent), don't amplify selling."""
        ctx = _extreme_short_context()
        assert ctx.sell_gate_multiplier() <= 1.0

    def test_sell_always_at_least_half(self):
        """SELL gate must never reduce signal below 50% — we must always be able to exit."""
        ctx = _extreme_short_context()
        assert ctx.sell_gate_multiplier() >= 0.5

    def test_overheated_longs_does_not_suppress_sell(self):
        """When market is overheated (extreme long), selling should not be suppressed."""
        ctx = _extreme_long_context()
        assert ctx.sell_gate_multiplier() >= 1.0


# ── MacroContext.regime_label ──────────────────────────────────────────────────

class TestRegimeLabel:

    def test_neutral_label(self):
        ctx = _neutral_context()
        label = ctx.regime_label()
        assert isinstance(label, str)
        assert len(label) > 0

    def test_overheated_label_contains_warning(self):
        ctx = _extreme_long_context()
        label = ctx.regime_label().upper()
        assert any(w in label for w in ("OVERHEATED", "CAUTION", "LONG", "HOT")), (
            f"Expected warning label for overheated market, got: {label}"
        )

    def test_short_squeeze_label(self):
        ctx = _extreme_short_context()
        label = ctx.regime_label().upper()
        assert any(w in label for w in ("SQUEEZE", "SHORT", "CONTRARIAN", "BEARISH")), (
            f"Expected squeeze label, got: {label}"
        )


# ── MacroSignalService fetch methods ───────────────────────────────────────────

class TestFundingRateFetch:

    @pytest.mark.asyncio
    async def test_fetch_returns_float(self):
        svc = MacroSignalService()
        mock_resp = _mock_http([{"fundingRate": "0.0001"}])

        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            rate = await svc._fetch_funding_rate("BTCUSDT")

        assert isinstance(rate, float)
        assert rate == pytest.approx(0.0001, abs=1e-7)

    @pytest.mark.asyncio
    async def test_fetch_network_error_returns_zero(self):
        svc = MacroSignalService()
        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("timeout")
            )
            rate = await svc._fetch_funding_rate("BTCUSDT")

        assert rate == 0.0

    @pytest.mark.asyncio
    async def test_fetch_long_short_ratio(self):
        svc = MacroSignalService()
        mock_resp = _mock_http([{"longShortRatio": "1.35"}])

        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            ratio = await svc._fetch_ls_ratio("BTCUSDT")

        assert ratio == pytest.approx(1.35, abs=0.001)

    @pytest.mark.asyncio
    async def test_fetch_ls_ratio_network_error_returns_one(self):
        """On failure, default to neutral L/S ratio of 1.0."""
        svc = MacroSignalService()
        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("timeout")
            )
            ratio = await svc._fetch_ls_ratio("BTCUSDT")

        assert ratio == pytest.approx(1.0, abs=0.001)

    @pytest.mark.asyncio
    async def test_fetch_btc_dominance(self):
        svc = MacroSignalService()
        # Binance futures ticker list: BTC vol=500, ETH=200, BNB=100 → BTC dom ≈ 62.5%
        mock_resp = _mock_http([
            {"symbol": "BTCUSDT",  "quoteVolume": "500000000"},
            {"symbol": "BTCBUSD",  "quoteVolume": "0"},
            {"symbol": "ETHUSDT",  "quoteVolume": "200000000"},
            {"symbol": "BNBUSDT",  "quoteVolume": "100000000"},
        ])

        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            dom = await svc._fetch_btc_dominance()

        # BTC vol = 500M, total = 800M → 62.5%
        assert dom == pytest.approx(62.5, abs=0.1)

    @pytest.mark.asyncio
    async def test_fetch_btc_dominance_network_error_returns_50(self):
        """On failure, return neutral 50% dominance."""
        svc = MacroSignalService()
        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("timeout")
            )
            dom = await svc._fetch_btc_dominance()

        assert dom == pytest.approx(50.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_fetch_open_interest(self):
        svc = MacroSignalService()
        mock_resp = _mock_http({"openInterest": "12345678.89"})

        with patch("services.macro_signals.httpx.AsyncClient") as mc:
            mc.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_resp)
            oi = await svc._fetch_open_interest("BTCUSDT")

        assert oi > 0


# ── Caching ────────────────────────────────────────────────────────────────────

class TestMacroSignalCache:

    @pytest.mark.asyncio
    async def test_second_call_within_ttl_uses_cache(self):
        """get_macro_context() should only make one set of HTTP calls within TTL."""
        svc = MacroSignalService(cache_ttl=600)

        # Pre-populate cache
        svc._cache = _neutral_context()
        svc._cache_ts = time.time()  # fresh

        with patch.object(svc, "_fetch_funding_rate", new=AsyncMock()) as mock_fr:
            ctx = await svc.get_macro_context()

        mock_fr.assert_not_called()
        assert ctx.funding_rate == _neutral_context().funding_rate

    @pytest.mark.asyncio
    async def test_expired_cache_triggers_fresh_fetch(self):
        """When cache is expired, a fresh fetch must be performed."""
        svc = MacroSignalService(cache_ttl=0)  # always expired

        with (
            patch.object(svc, "_fetch_funding_rate",  new=AsyncMock(return_value=0.0002)),
            patch.object(svc, "_fetch_ls_ratio",      new=AsyncMock(return_value=1.1)),
            patch.object(svc, "_fetch_open_interest", new=AsyncMock(return_value=1e10)),
            patch.object(svc, "_fetch_btc_dominance", new=AsyncMock(return_value=51.0)),
            patch.object(svc, "_fetch_coinbase_premium", new=AsyncMock(return_value=0.01)),
        ):
            ctx = await svc.get_macro_context()

        assert ctx.fetch_ok is True
        assert ctx.funding_rate == pytest.approx(0.0002, abs=1e-7)


# ── Constants sanity ───────────────────────────────────────────────────────────

class TestMacroConstants:

    def test_funding_overheated_is_positive(self):
        assert _FUNDING_OVERHEATED > 0

    def test_funding_overheated_reasonable_range(self):
        """Standard threshold is 0.05%–0.15% per 8h."""
        assert 0.0005 <= _FUNDING_OVERHEATED <= 0.002

    def test_funding_oversold_is_negative(self):
        assert _FUNDING_OVERSOLD < 0

    def test_ls_ratio_long_heavy_above_1(self):
        assert _LS_RATIO_LONG_HEAVY > 1.0

    def test_ls_ratio_short_heavy_below_1(self):
        assert _LS_RATIO_SHORT_HEAVY < 1.0

    def test_coinbase_premium_bearish_is_negative(self):
        assert _COINBASE_PREMIUM_BEARISH < 0


# ── get_macro_service singleton ────────────────────────────────────────────────

class TestGetMacroService:

    def test_returns_same_instance(self):
        s1 = get_macro_service()
        s2 = get_macro_service()
        assert s1 is s2
