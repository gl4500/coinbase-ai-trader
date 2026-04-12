"""
Coinbase CNN-LSTM Agent
────────────────────────────────────────────────
16-channel × 60-timestep feature tensor fed into a CNN-LSTM hybrid.

Channels:
  Ch 0   normalised close price
  Ch 1   log10 volume (/ 10)
  Ch 2   high-low range / close       (intrabar volatility)
  Ch 3   (close - open) / open        (candle body direction)
  Ch 4   RSI(14) / 100
  Ch 5   MACD histogram (normalised)
  Ch 6   EMA9 distance from close (normalised)
  Ch 7   EMA21 distance from close (normalised)
  Ch 8   Bollinger Band position (0=lower, 1=upper)
  Ch 9   1-bar price change (%)
  Ch 10  bid depth  (log, / 8)
  Ch 11  ask depth  (log, / 8)
  Ch 12  MFI(14) / 100                (volume-weighted RSI)
  Ch 13  OBV slope (normalised)       (accumulation/distribution)
  Ch 14  Stochastic RSI K / 100       (fast overbought/oversold)
  Ch 15  ADX(14) / 100                (trend regime strength)
  Ch 16  VWAP distance (normalised)   (price vs. volume-weighted avg — institutional level)
  Ch 17  5-min RSI(12) / 100          (fast momentum — current hour in 5-min bars)
  Ch 18  5-min price velocity         (rate of change per 5-min bar, normalised)
  Ch 19  5-min volume z-score         (volume spike vs. 60-bar mean, normalised)

Architecture: Conv1D × 4 → MaxPool × 2 → LSTM(2-layer) → FC → sigmoid
Blend: trending market → CNN 75% / LLM 25%
       ranging market  → CNN 40% / LLM 60%

Install PyTorch (CUDA 12.x):
  pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
import asyncio
import json
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

import database
from clients import coinbase_client
from agents.signal_generator import (
    _rsi, _ema, _macd, _bollinger, _ema_cross,
    _atr, _adx, _mfi, _obv_slope, _stoch_rsi, _vwap,
)
from config import config
from services.outcome_tracker import get_tracker

_CNN_DRY_RUN_BALANCE = 1_000.0
_CNN_MAX_FRAC        = 0.15    # max 15% of portfolio per position


class _CNNBook:
    """Lightweight dry-run portfolio book for the CNN agent.
    Mirrors the _Book classes in tech_agent_cb / momentum_agent_cb so that
    CNN trades appear in the `trades` table and per-product positions are tracked
    (prevents buying the same asset repeatedly).
    """
    def __init__(self):
        self._agent      = "CNN"
        self.balance     = _CNN_DRY_RUN_BALANCE
        self.positions: Dict[str, Dict] = {}   # pid → {size, avg_price}
        self.realized_pnl = 0.0

    async def load(self) -> None:
        state = await database.load_agent_state(self._agent)
        if state:
            self.balance      = state["balance"]
            self.realized_pnl = state["realized_pnl"]
            self.positions    = state["positions"]
            logger.info(
                f"CNN book restored | balance=${self.balance:.2f} | "
                f"pnl=${self.realized_pnl:+.2f} | positions={len(self.positions)}"
            )
        else:
            logger.info(f"CNN book: no saved state — starting fresh at ${self.balance:.2f}")

    async def _save(self) -> None:
        await database.save_agent_state(
            self._agent, self.balance, self.realized_pnl, self.positions, {}
        )

    def has_position(self, pid: str) -> bool:
        return pid in self.positions

    async def buy(self, pid: str, price: float, frac: float,
                  trigger: str = "SCAN") -> Tuple[float, float]:
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
        await self._save()
        await database.open_trade(
            agent=self._agent, product_id=pid, entry_price=price,
            size=size, usd_open=spend, trigger_open=trigger,
            balance_after=self.balance,
        )
        return spend, size

    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        if pid not in self.positions:
            return 0.0
        pos = self.positions.pop(pid)
        proceeds = pos["size"] * price
        pnl = proceeds - pos["size"] * pos["avg_price"]
        self.balance += proceeds
        self.realized_pnl += pnl
        await self._save()
        await database.close_trade(
            agent=self._agent, product_id=pid, exit_price=price,
            size=pos["size"], pnl=pnl, trigger_close=trigger,
            balance_after=self.balance,
        )
        return pnl

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
except ImportError:
    _TORCH = False
    logger.warning("PyTorch not found — CNN agent uses linear fallback")

N_CHANNELS = 20
SEQ_LEN    = 60
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "cnn_model.pt")
OLLAMA_URL = "http://localhost:11434"
_CACHE_TTL = 300


# ── CNN-LSTM Model ────────────────────────────────────────────────────────────

if _TORCH:
    class SignalCNN(nn.Module):
        """
        CNN-LSTM hybrid:
          1. 4× Conv1D layers extract local temporal patterns
          2. 2-layer LSTM captures long-range dependencies
          3. FC head → sigmoid probability
        """
        def __init__(self, n_ch: int = N_CHANNELS):
            super().__init__()
            # Convolutional feature extractor
            self.c1  = nn.Conv1d(n_ch, 32,  3, padding=1)
            self.c2  = nn.Conv1d(32,  64,  3, padding=1)
            self.p2  = nn.MaxPool1d(2)           # 60 → 30
            self.c3  = nn.Conv1d(64,  128, 3, padding=1)
            self.p3  = nn.MaxPool1d(2)           # 30 → 15
            self.c4  = nn.Conv1d(128, 128, 3, padding=1)
            # LSTM temporal reasoning
            self.lstm = nn.LSTM(128, 64, num_layers=2,
                                batch_first=True, dropout=0.2)
            self.drop = nn.Dropout(0.3)
            self.fc   = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.c1(x))
            x = F.relu(self.c2(x)); x = self.p2(x)
            x = F.relu(self.c3(x)); x = self.p3(x)
            x = F.relu(self.c4(x))
            x = x.permute(0, 2, 1)          # (B, seq, 128)
            x, _ = self.lstm(x)
            x = x[:, -1, :]                 # last timestep
            x = self.drop(x)
            return torch.sigmoid(self.fc(x))

        def predict(self, tensor: "torch.Tensor") -> float:
            self.eval()
            with torch.no_grad():
                return float(self.forward(tensor.unsqueeze(0)).item())


# ── Feature Builder ───────────────────────────────────────────────────────────

class FeatureBuilder:
    @staticmethod
    def _pad(series: List[float], n: int, v: float = 0.0) -> List[float]:
        return list(series[-n:]) if len(series) >= n else [v] * (n - len(series)) + list(series)

    @staticmethod
    def _slog(v: float) -> float:
        return math.log10(max(v, 1.0))

    def build(self, candles: List[Dict], ob: Dict,
              candles_5m: Optional[List[Dict]] = None,
              T: int = SEQ_LEN) -> List[List[float]]:
        if not candles:
            return [[0.0] * T] * N_CHANNELS

        opens   = [c["open"]   for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        closes  = [c["close"]  for c in candles]
        volumes = [c["volume"] for c in candles]

        # ── Ch 0: Normalised close ────────────────────────────────────────────
        mn, mx = min(closes), max(closes)
        rng    = mx - mn if mx != mn else 1.0
        norm_c = [(v - mn) / rng for v in closes]

        # ── Ch 2: High-low range / close ──────────────────────────────────────
        hl_range = [(h - l) / max(c, 1e-9) for h, l, c in zip(highs, lows, closes)]

        # ── Ch 3: Candle body direction ───────────────────────────────────────
        body = [max(-1.0, min(1.0, (c - o) / max(abs(o), 1e-9)))
                for o, c in zip(opens, closes)]

        # ── Ch 4: RSI(14) per bar ─────────────────────────────────────────────
        rsi_ch = [_rsi(closes[max(0, i - 20): i + 1]) / 100.0
                  for i in range(len(closes))]

        # ── Ch 5: MACD histogram ──────────────────────────────────────────────
        macd_ch = [0.0] * len(closes)
        if len(closes) >= 35:
            for i in range(35, len(closes) + 1):
                _, _, h = _macd(closes[:i])
                scale   = max(abs(closes[i - 1]), 1)
                macd_ch[i - 1] = max(-1.0, min(1.0, h / scale * 1000))

        # ── Ch 6 & 7: EMA distance ────────────────────────────────────────────
        ema9_ch  = [0.0] * len(closes)
        ema21_ch = [0.0] * len(closes)
        if len(closes) >= 21:
            e9   = _ema(closes, 9);  off9  = len(closes) - len(e9)
            e21  = _ema(closes, 21); off21 = len(closes) - len(e21)
            for i, v in enumerate(e9):
                ema9_ch[i + off9]  = max(-0.1, min(0.1, (v - closes[i + off9])  / max(closes[i + off9],  1e-9))) / 0.1
            for i, v in enumerate(e21):
                ema21_ch[i + off21] = max(-0.1, min(0.1, (v - closes[i + off21]) / max(closes[i + off21], 1e-9))) / 0.1

        # ── Ch 8: Bollinger position ──────────────────────────────────────────
        bb_ch = [_bollinger(closes[max(0, i - 22): i + 1])[3]
                 for i in range(len(closes))]

        # ── Ch 9: 1-bar price change ──────────────────────────────────────────
        chg_ch = [0.0]
        for i in range(1, len(closes)):
            chg = (closes[i] - closes[i - 1]) / max(closes[i - 1], 1e-9)
            chg_ch.append(max(-0.1, min(0.1, chg)) / 0.1)

        # ── Ch 10 & 11: Order book depth ─────────────────────────────────────
        bv   = float(ob.get("bid_depth", 0) or 0)
        av   = float(ob.get("ask_depth", 0) or 0)
        bid_ch = [self._slog(bv) / 8.0] * len(closes)
        ask_ch = [self._slog(av) / 8.0] * len(closes)

        # ── Ch 12: MFI(14) per bar ────────────────────────────────────────────
        mfi_ch = [_mfi(highs[max(0, i - 20): i + 1],
                       lows[max(0, i - 20): i + 1],
                       closes[max(0, i - 20): i + 1],
                       volumes[max(0, i - 20): i + 1]) / 100.0
                  for i in range(len(closes))]

        # ── Ch 13: OBV slope ─────────────────────────────────────────────────
        obv_raw = [0.0]
        for i in range(1, len(closes)):
            if closes[i] > closes[i - 1]:
                obv_raw.append(obv_raw[-1] + volumes[i])
            elif closes[i] < closes[i - 1]:
                obv_raw.append(obv_raw[-1] - volumes[i])
            else:
                obv_raw.append(obv_raw[-1])
        obv_ch = [0.0] * len(closes)
        sp = 10
        for i in range(sp, len(obv_raw)):
            w  = obv_raw[i - sp: i + 1]
            n  = len(w); xm = (n - 1) / 2; ym = sum(w) / n
            num = sum((j - xm) * (w[j] - ym) for j in range(n))
            den = sum((j - xm) ** 2 for j in range(n))
            slp = num / den if den else 0.0
            obv_ch[i] = max(-1.0, min(1.0, slp * sp / max(abs(ym), 1.0)))

        # ── Ch 14: Stochastic RSI K ───────────────────────────────────────────
        stoch_ch = [0.5] * len(closes)
        stoch_p  = 14
        for i in range(stoch_p * 2, len(closes)):
            k, _ = _stoch_rsi(closes[: i + 1], stoch_p)
            stoch_ch[i] = k / 100.0

        # ── Ch 15: ADX(14) regime ─────────────────────────────────────────────
        adx_val, _, _ = _adx(highs, lows, closes)
        adx_ch = [adx_val / 100.0] * len(closes)

        # ── Ch 16: VWAP distance per bar ──────────────────────────────────────
        # Rolling 20-bar VWAP; distance = (close - vwap) / vwap, norm to [-1,1]
        vwap_ch = [0.0] * len(closes)
        vwap_p  = 20
        for i in range(vwap_p, len(closes) + 1):
            _, dist = _vwap(highs[:i], lows[:i], closes[:i], volumes[:i], vwap_p)
            vwap_ch[i - 1] = dist

        # ── Ch 17-19: 5-minute fast channels ─────────────────────────────────
        # Broadcast current fast-timeframe state across all hourly timesteps.
        # CNN learns these as "context" channels representing right-now momentum.
        c5m = candles_5m or []
        if len(c5m) >= 14:
            c5    = [c["close"]  for c in c5m]
            v5    = [c["volume"] for c in c5m]

            # Ch 17: Fast RSI from last 12 × 5-min bars (= 1 hour)
            fast_rsi_val = _rsi(c5[-14:]) / 100.0

            # Ch 18: Price velocity — % change per bar over last 12 bars (1 hour)
            n_vel   = min(12, len(c5) - 1)
            p_start = c5[-(n_vel + 1)]
            p_end   = c5[-1]
            vel     = (p_end - p_start) / max(p_start, 1e-9) / max(n_vel, 1)
            # Normalise: 0.5% per 5-min bar = extreme → clip to [-1,1]
            vel_norm = max(-1.0, min(1.0, vel / 0.005))

            # Ch 19: Volume z-score — spike vs. rolling 60-bar mean
            v5_win  = v5[-60:] if len(v5) >= 60 else v5
            v_mean  = sum(v5_win) / len(v5_win)
            v_std   = math.sqrt(sum((x - v_mean) ** 2 for x in v5_win) / len(v5_win))
            v_z     = (v5[-1] - v_mean) / max(v_std, 1e-9)
            vol_z_norm = max(-1.0, min(1.0, v_z / 3.0))   # 3 σ = 1.0
        else:
            fast_rsi_val = 0.5
            vel_norm     = 0.0
            vol_z_norm   = 0.0

        fast_rsi_ch = [fast_rsi_val] * len(closes)
        vel_ch      = [vel_norm]     * len(closes)
        vol_z_ch    = [vol_z_norm]   * len(closes)

        P = self._pad
        channels = [
            P(norm_c,                                  T),   # 0
            P([self._slog(v) / 10.0 for v in volumes], T),  # 1
            P(hl_range,                                T),   # 2
            P(body,                                    T),   # 3
            P(rsi_ch,                                  T),   # 4
            P(macd_ch,                                 T),   # 5
            P(ema9_ch,                                 T),   # 6
            P(ema21_ch,                                T),   # 7
            P(bb_ch,                                   T),   # 8
            P(chg_ch,                                  T),   # 9
            P(bid_ch,                                  T),   # 10
            P(ask_ch,                                  T),   # 11
            P(mfi_ch,                                  T),   # 12
            P(obv_ch,                                  T),   # 13
            P(stoch_ch,                                T),   # 14
            P(adx_ch,                                  T),   # 15
            P(vwap_ch,                                 T),   # 16
            P(fast_rsi_ch,                             T),   # 17
            P(vel_ch,                                  T),   # 18
            P(vol_z_ch,                                T),   # 19
        ]
        assert len(channels) == N_CHANNELS
        return channels

    def to_tensor(self, channels):
        if not _TORCH:
            return channels
        return torch.tensor(channels, dtype=torch.float32)


# ── Ollama ────────────────────────────────────────────────────────────────────

async def _ollama_prob(product_id: str, context: str,
                       adx_val: float, rsi_val: float,
                       macd_h: float, bb_pos: float,
                       mfi_val: float, stoch_k: float,
                       cnn_prob: float,
                       lessons: Optional[List[str]] = None) -> Optional[float]:
    model  = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    regime = "TRENDING" if adx_val >= config.adx_trend_threshold else "RANGING"
    lesson_block = ""
    if lessons:
        lesson_block = (
            "\n\nPast 4-hour outcomes for this asset:\n"
            + "\n".join(f"  • {l}" for l in lessons)
            + "\n"
        )
    prompt = (
        f"You are a quantitative crypto trading analyst. "
        f"Estimate the probability (0.00-1.00) that {product_id} closes HIGHER in 4 hours.\n\n"
        f"Market regime: {regime} (ADX={adx_val:.1f})\n"
        f"RSI(14): {rsi_val:.1f} | MACD: {'bullish' if macd_h > 0 else 'bearish'} ({macd_h:+.5f})\n"
        f"Bollinger: {bb_pos:.0%} of band | MFI(14): {mfi_val:.1f} | StochRSI-K: {stoch_k:.1f}\n"
        f"CNN model probability: {cnn_prob:.3f}"
        f"{lesson_block}\n"
        f'Respond with ONLY valid JSON: {{"probability": <0.00-1.00>}}'
    )
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            resp = await client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model, "prompt": prompt,
                      "stream": False, "format": "json"},
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
        prob = float(json.loads(text).get("probability", -1))
        if 0 <= prob <= 1:
            return prob
    except Exception:
        try:
            m = re.search(r"\b0\.\d{2,4}\b", text)
            if m:
                return float(m.group())
        except Exception:
            pass
    return None


# ── CNN-LSTM Agent ─────────────────────────────────────────────────────────────

class CoinbaseCNNAgent:
    def __init__(self, ws_subscriber=None):
        self.ws     = ws_subscriber
        self.fb     = FeatureBuilder()
        self._cache: Dict[str, Tuple[float, float]] = {}
        self.model: Optional["SignalCNN"] = None
        self.book   = _CNNBook()          # dry-run portfolio — tracks positions + trades table
        # ── Runtime stats ──────────────────────────────────────────────────
        self.last_scan_at:     Optional[float] = None
        self.next_scan_at:     Optional[float] = None
        self.scan_count:       int = 0
        self.signals_total:    int = 0
        self.signals_buy:      int = 0
        self.signals_sell:     int = 0
        self.signals_executed: int = 0
        self.last_trained_at:  Optional[float] = None
        self.train_count:      int = 0
        if _TORCH:
            self.model = SignalCNN()
            self._load()
        logger.info(
            f"CoinbaseCNNAgent ready | torch={'yes' if _TORCH else 'linear'} | "
            f"model={'loaded' if self._exists() else 'random (untrained)'} | "
            f"channels={N_CHANNELS}"
        )

    async def start(self) -> None:
        """Load persisted book state before first scan."""
        await self.book.load()

    def _exists(self) -> bool:
        return os.path.exists(MODEL_PATH)

    def _load(self):
        if not _TORCH or not self.model or not self._exists():
            return
        try:
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
            )
            logger.info("CNN-LSTM model weights loaded from disk")
        except Exception as e:
            logger.warning(f"CNN model load failed (shape mismatch likely — retrain): {e}")

    def save_model(self):
        if _TORCH and self.model:
            torch.save(self.model.state_dict(), MODEL_PATH)

    def _cnn_prob(self, channels) -> float:
        if _TORCH and self.model:
            return self.model.predict(self.fb.to_tensor(channels))
        return self._linear(channels)

    @staticmethod
    def _linear(channels) -> float:
        """Fallback when PyTorch is unavailable — uses 6 of 16 channels."""
        try:
            score = (
                0.50
                + 0.20 * (channels[0][-1] - 0.5)      # normalised price
                + 0.15 * (0.5 - channels[4][-1])       # inverted RSI
                + 0.10 * channels[5][-1]                # MACD histogram
                + 0.10 * (0.5 - channels[12][-1])      # inverted MFI
                + 0.08 * channels[13][-1]               # OBV slope
                + 0.07 * (1.0 - channels[15][-1])      # low ADX = mean revert
            )
            return max(0.01, min(0.99, score))
        except Exception:
            return 0.5

    async def _ob(self, product_id: str) -> Dict:
        try:
            book = await coinbase_client.get_orderbook(product_id, limit=5)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if not bids or not asks:
                return {}
            bv = sum(b["size"] for b in bids)
            av = sum(a["size"] for a in asks)
            t  = bv + av
            return {"bid_depth": bv, "ask_depth": av,
                    "imbalance": (bv - av) / t if t else 0}
        except Exception:
            return {}

    def _live_price(self, pid: str, fallback: float) -> float:
        if self.ws:
            p = self.ws.get_price(pid)
            if p and p > 0:
                return p
        return fallback

    async def generate_signal(
        self,
        product: Dict,
        execute: bool = False,
        order_executor=None,
    ) -> Optional[Dict]:
        pid   = product["product_id"]
        price = self._live_price(pid, product.get("price") or 0)
        if not price or price <= 0:
            return None

        cached = self._cache.get(pid)
        if cached and time.time() - cached[1] < _CACHE_TTL:
            cnn_prob = cached[0]
            ob       = {}
            # Use neutral indicator values for cached result
            adx_val = config.adx_trend_threshold
            rsi_val = 50.0; macd_h = 0.0; bb_pos = 0.5
            mfi_val = 50.0; stoch_k = 50.0; atr_val = 0.0
            vwap_price = price; vwap_d = 0.0
            fast_rsi_val = vel_norm = vol_z_norm = 0.5
        else:
            candles = await database.get_candles(pid, limit=80)
            if len(candles) < 30:
                return None

            closes  = [c["close"]  for c in candles]
            highs   = [c["high"]   for c in candles]
            lows    = [c["low"]    for c in candles]
            volumes = [c["volume"] for c in candles]

            ob = await self._ob(pid)

            # Fetch 5-min candles for fast-timeframe channels (last 6 hours)
            candles_5m = []
            try:
                candles_5m = await coinbase_client.get_candles(
                    pid, granularity="FIVE_MINUTE", limit=72
                )
            except Exception as e5:
                logger.debug(f"5m candles unavailable for {pid}: {e5}")

            # Compute fast scalars here (mirrors FeatureBuilder Ch 17-19)
            # so we can save them independently to the scan record.
            fast_rsi_val = vel_norm = vol_z_norm = 0.0
            c5m = candles_5m or []
            if len(c5m) >= 14:
                c5    = [c["close"]  for c in c5m]
                v5    = [c["volume"] for c in c5m]
                fast_rsi_val = _rsi(c5[-14:]) / 100.0
                n_vel   = min(12, len(c5) - 1)
                p_start = c5[-(n_vel + 1)]
                p_end   = c5[-1]
                vel     = (p_end - p_start) / max(p_start, 1e-9) / max(n_vel, 1)
                vel_norm = max(-1.0, min(1.0, vel / 0.005))
                v5_win  = v5[-60:] if len(v5) >= 60 else v5
                v_mean  = sum(v5_win) / len(v5_win)
                v_std   = math.sqrt(sum((x - v_mean) ** 2 for x in v5_win) / max(len(v5_win), 1))
                v_z     = (v5[-1] - v_mean) / max(v_std, 1e-9)
                vol_z_norm = max(-1.0, min(1.0, v_z / 3.0))

            channels = self.fb.build(candles, ob, candles_5m=candles_5m)
            cnn_prob = self._cnn_prob(channels)

            rsi_val            = _rsi(closes)
            _, _, macd_h       = _macd(closes)
            _, _, _, bb_pos    = _bollinger(closes)
            adx_val, _, _      = _adx(highs, lows, closes)
            mfi_val            = _mfi(highs, lows, closes, volumes)
            stoch_k, _         = _stoch_rsi(closes)
            atr_val            = _atr(highs, lows, closes)
            vwap_price, vwap_d = _vwap(highs, lows, closes, volumes)

            self._cache[pid] = (cnn_prob, time.time())

        # ── Dynamic LLM/CNN blend based on ADX regime ─────────────────────────
        trending = adx_val >= config.adx_trend_threshold
        if trending:
            cnn_w, llm_w = config.cnn_trending_cnn_w, config.cnn_trending_llm_w
        else:
            cnn_w, llm_w = config.cnn_ranging_cnn_w, config.cnn_ranging_llm_w

        # Fetch most-recent Tech & Momentum decisions for this product
        # so the Ollama model can incorporate their votes into its reasoning.
        agent_votes = await database.get_agent_decisions(pid, limit=2)
        agent_ctx = ""
        for av in agent_votes:
            agent_ctx += (
                f"\n  {av['agent']:8s}: {av['side']:4s} "
                f"conf={av['confidence']:.2f} score={av.get('score', 0):.2f} "
                f"— {(av.get('reasoning') or '')[:70]}"
            )

        vwap_side = "above" if vwap_d > 0 else "below"
        context = (
            f"Price: ${price:,.4f} | Regime: {'TRENDING' if trending else 'RANGING'}\n"
            f"ADX(14): {adx_val:.1f} | RSI(14): {rsi_val:.1f} | MFI(14): {mfi_val:.1f}\n"
            f"MACD hist: {macd_h:+.6f} | Bollinger: {bb_pos:.2f} | StochRSI K: {stoch_k:.1f}\n"
            f"VWAP(20): ${vwap_price:,.4f} | Price {vwap_side} VWAP by {abs(vwap_d)*100:.2f}%\n"
            f"OB imbalance: {ob.get('imbalance', 0):+.2f} | ATR(14): {atr_val:.4f}\n"
            f"CNN raw: {cnn_prob:.4f} | weights CNN={cnn_w} LLM={llm_w}"
            + (f"\nSub-agent votes:{agent_ctx}" if agent_ctx else "")
        )

        # ── Option 2: skip LLM when CNN is already decisive ───────────────────
        # If cnn_prob is far from 0.5 (beyond llm_skip_threshold), the LLM
        # cannot flip the direction — skip the 10–30s Ollama call entirely.
        # Fetch outcome lessons so Ollama can learn from past signals
        lessons = await get_tracker().get_lessons(pid, limit=5)

        cnn_dist = abs(cnn_prob - 0.5)
        skip_llm = cnn_dist >= (config.llm_skip_threshold - 0.5)
        if skip_llm:
            llm_prob = None
            logger.debug(
                f"LLM skipped for {pid}: cnn_prob={cnn_prob:.3f} is decisive "
                f"(|{cnn_dist:.3f}| >= {config.llm_skip_threshold - 0.5:.3f})"
            )
        else:
            llm_prob = await _ollama_prob(pid, context, adx_val, rsi_val,
                                          macd_h, bb_pos, mfi_val, stoch_k, cnn_prob,
                                          lessons=lessons)

        if llm_prob is not None:
            model_prob = cnn_w * cnn_prob + llm_w * llm_prob
        else:
            model_prob = cnn_prob

        model_prob = max(0.01, min(0.99, model_prob))

        # ── Signal direction ──────────────────────────────────────────────────
        if model_prob > config.cnn_buy_threshold:
            side     = "BUY"
            strength = round((model_prob - 0.5) * 2, 3)
        elif model_prob < config.cnn_sell_threshold:
            side     = "SELL"
            strength = round((0.5 - model_prob) * 2, 3)
        else:
            side     = "HOLD"
            strength = 0.0

        passes = side != "HOLD"

        # ── Save every scan result for the confidence table ───────────────────
        await database.save_cnn_scan({
            "product_id":  pid,
            "price":       round(price, 6),
            "cnn_prob":    round(cnn_prob, 4),
            "llm_prob":    round(llm_prob, 4) if llm_prob is not None else None,
            "model_prob":  round(model_prob, 4),
            "cnn_weight":  cnn_w,
            "llm_weight":  llm_w,
            "side":        side,
            "strength":    round(strength, 3),
            "signal_gen":  passes,
            "regime":      "TRENDING" if trending else "RANGING",
            "adx":         round(adx_val, 1),
            "rsi":         round(rsi_val, 1),
            "macd":        round(macd_h, 6),
            "mfi":         round(mfi_val, 1),
            "stoch_k":     round(stoch_k, 1),
            "atr":         round(atr_val, 6),
            "vwap_dist":   round(vwap_d, 4),
            "fast_rsi":    round(fast_rsi_val, 4),
            "velocity":    round(vel_norm, 4),
            "vol_z":       round(vol_z_norm, 4),
        })

        if not passes:
            return None

        # Record CNN signal to outcome tracker (resolved 4h later)
        await get_tracker().record(
            source="CNN", product_id=pid, side=side,
            confidence=round(strength, 3), price=price,
            indicators={
                "cnn_prob": round(cnn_prob, 4),
                "adx":      round(adx_val, 1),
                "regime":   "TRENDING" if trending else "RANGING",
                "rsi":      round(rsi_val, 1),
            },
        )

        quote_size = min(strength * config.max_position_usd, config.max_position_usd)

        prob_line = (
            f"\ncnn_prob={cnn_prob:.4f} "
            f"llm_prob={llm_prob:.4f} "
            f"model_prob={model_prob:.4f} "
            f"cnn_w={cnn_w} llm_w={llm_w}"
            if llm_prob is not None
            else f"\ncnn_prob={cnn_prob:.4f} llm_prob=None model_prob={model_prob:.4f}"
        )

        signal = {
            "product_id":  pid,
            "signal_type": f"CNN_{'LONG' if side == 'BUY' else 'SHORT'}",
            "side":        side,
            "price":       round(price, 6),
            "strength":    strength,
            "rsi":         round(rsi_val, 2),
            "macd":        round(macd_h, 6),
            "bb_position": round(bb_pos, 3),
            "reasoning":   context + prob_line,
            "quote_size":  round(quote_size, 2),
            "atr":       round(atr_val, 6),
            "adx":       round(adx_val, 1),
            "vwap_dist": round(vwap_d, 4),
        }

        signal_id    = await database.save_signal(signal)
        signal["id"] = signal_id

        self.signals_total += 1
        if side == "BUY":
            self.signals_buy  += 1
        else:
            self.signals_sell += 1

        llm_str = f"{llm_prob:.2%}" if llm_prob is not None else "n/a"
        logger.info(
            f"CNN [{side}] {pid} | cnn={cnn_prob:.2%} llm={llm_str} "
            f"blend={model_prob:.2%} adx={adx_val:.1f}({'trend' if trending else 'range'}) "
            f"strength={strength:.2f} size=${quote_size:.2f}"
        )

        # ── Dry-run execution via _CNNBook (tracks positions + writes to trades table) ──
        if execute:
            if side == "BUY" and not self.book.has_position(pid):
                frac = _CNN_MAX_FRAC * strength
                spent, _ = await self.book.buy(pid, price, frac, trigger="SCAN")
                if spent > 0:
                    self.signals_executed += 1
                    signal["execution"] = {"success": True, "spent": round(spent, 2)}
                    logger.info(
                        f"CNN BOOK BUY {pid} @{price:.4f} strength={strength:.2f} "
                        f"spent=${spent:.2f} balance=${self.book.balance:.2f}"
                    )
                else:
                    signal["execution"] = {"success": False, "reason": "Insufficient balance"}
                    logger.warning(
                        f"CNN BOOK BUY skipped {pid}: balance=${self.book.balance:.2f} "
                        f"too low for frac={frac:.2f}"
                    )
            elif side == "SELL" and self.book.has_position(pid):
                pnl = await self.book.sell(pid, price, trigger="SCAN")
                self.signals_executed += 1
                signal["execution"] = {"success": True, "pnl": round(pnl, 4)}
                logger.info(
                    f"CNN BOOK SELL {pid} @{price:.4f} pnl=${pnl:+.2f} "
                    f"balance=${self.book.balance:.2f}"
                )
            elif side == "BUY" and self.book.has_position(pid):
                signal["execution"] = {"success": False, "reason": "Already holding position"}
            else:
                signal["execution"] = {"success": False, "reason": "No position to sell"}

            # Live order execution (only when order_executor provided and not dry-run)
            if order_executor and not order_executor.dry_run and quote_size >= 1.0:
                result = await order_executor.execute_signal(signal)
                signal["live_execution"] = result

        return signal

    async def scan_all(self, execute: bool = False,
                       order_executor=None) -> List[Dict]:
        products = await database.get_products(tracked_only=True)
        if not products:
            return []
        signals = []
        for p in products:
            try:
                sig = await self.generate_signal(p, execute, order_executor)
                if sig:
                    signals.append(sig)
            except Exception as e:
                logger.error(f"CNN error [{p.get('product_id')}]: {e}")
            await asyncio.sleep(config.scan_sleep_secs)
        signals.sort(key=lambda s: s["strength"], reverse=True)
        return signals

    async def run_loop(self, interval: int = 900,
                       train_every_n_scans: int = 4,
                       order_executor=None,
                       is_trading_fn=None) -> None:
        """
        Scan every `interval` seconds (default 15 min).
        Auto-train every `train_every_n_scans` scans (default 4 = ~1 hour).
        Pass order_executor + is_trading_fn to enable live trade execution
        on each auto-scan (mirrors the /api/cnn/scan?execute=true endpoint).

        Why 1 hour: candles are hourly so no new training data arrives faster
        than that. Training more often just re-learns the same data → overfitting.
        """
        await self.start()   # restore persisted book state before first scan
        logger.info(
            f"CNN-LSTM loop started | scan={interval}s | "
            f"auto-train every {train_every_n_scans} scans "
            f"(~{interval * train_every_n_scans // 60} min) | "
            f"channels={N_CHANNELS} | balance=${self.book.balance:.2f}"
        )
        while True:
            try:
                self.next_scan_at = time.time() + interval
                should_execute = is_trading_fn() if is_trading_fn else False
                await self.scan_all(
                    execute        = should_execute,
                    order_executor = order_executor if should_execute else None,
                )
                self.last_scan_at = time.time()
                self.scan_count  += 1
                self.next_scan_at = time.time() + interval

                # Auto-train after every N scans (aligned with new candle data)
                if self.scan_count % train_every_n_scans == 0:
                    logger.info(
                        f"CNN auto-train triggered (scan #{self.scan_count}, "
                        f"every {train_every_n_scans} scans)"
                    )
                    try:
                        result = await self.train_on_history(epochs=20)
                        if "error" in result:
                            logger.warning(f"CNN auto-train skipped: {result['error']}")
                        else:
                            self.last_trained_at = time.time()
                            self.train_count    += 1
                            logger.info(
                                f"CNN auto-train done — {result['samples']} samples | "
                                f"loss {result['initial_loss']:.4f} → {result['final_loss']:.4f}"
                            )
                    except Exception as te:
                        logger.error(f"CNN auto-train error: {te}")

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"CNN loop error: {e}")
            await asyncio.sleep(interval)

    async def train_on_history(self, epochs: int = 20) -> Dict:
        if not _TORCH or not self.model:
            return {"error": "PyTorch not available"}
        import torch.optim as optim
        products = await database.get_products()
        X_list, y_list = [], []
        for p in products:
            candles = await database.get_candles(p["product_id"], limit=80)
            if len(candles) < 30:
                continue
            channels = self.fb.build(candles, {})
            X_list.append(self.fb.to_tensor(channels))
            closes = [c["close"] for c in candles]
            y_list.append(min(1.0, max(0.0, _rsi(closes) / 100.0)))
        if len(X_list) < 3:
            return {"error": f"Not enough data ({len(X_list)} samples)"}
        X = torch.stack(X_list)
        y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
        opt, losses = optim.Adam(self.model.parameters(), lr=1e-3), []
        self.model.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy(self.model(X), y)
            loss.backward(); opt.step()
            losses.append(float(loss.item()))
        self.save_model()
        return {"epochs": epochs, "samples": len(X_list),
                "channels": N_CHANNELS,
                "initial_loss": round(losses[0], 6),
                "final_loss":   round(losses[-1], 6)}
