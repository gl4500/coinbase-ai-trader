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
  Ch 20  funding rate (normalised)    (perp market sentiment: positive = longs pay)
  Ch 21  BTC return correlation       (rolling 20-bar Pearson vs BTC returns)
  Ch 22  sin(hour-of-day × 2π/24)    (time-of-day cyclical encoding)
  Ch 23  cos(hour-of-day × 2π/24)    (time-of-day cyclical encoding)
  Ch 24  IV/RV20 spread (clipped [-1,1]) (Deribit IV minus 20d realized vol)
  Ch 25  IV/RV60 spread (clipped [-1,1]) (Deribit IV minus 60d realized vol)
  Ch 26  L/S sentiment ([-1,1])          (Binance top-trader long minus short ratio)

Architecture: Conv1D × 4 → MaxPool × 2 → LSTM(2-layer) → FC → sigmoid
Blend: trending market → CNN 75% / LLM 25%
       ranging market  → CNN 40% / LLM 60%

Install PyTorch (CUDA 12.x):
  pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
import asyncio
import bisect
import copy
import datetime as _dt
import json
import logging
import math
import os
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

import database
from clients import coinbase_client
from agents.exit_thresholds import _compute_exit_threshold  # task #46
from agents import exit_execution
from agents.signal_generator import (
    _rsi, _ema, _macd, _bollinger, _ema_cross,
    _atr, _adx, _mfi, _obv_slope, _stoch_rsi, _vwap,
    _kelly_fraction, _realized_vol,
)
from config import config
from services.outcome_tracker import get_tracker
from services.history_backfill import load_history, load_5m_history
from services.deribit_iv import get_iv, compute_iv_rv_spreads
from services.binance_sentiment import get_ls_sentiment
from services.okx_funding_history import fetch_funding_history
from services.okx_oi_history import fetch_oi_history
from services.hmm_regime import get_detector, regime_blend
from services import product_status

_CNN_DRY_RUN_BALANCE = 1_000.0
_CNN_MAX_FRAC        = 0.15    # max 15% of portfolio per position
_CNN_STOP_LOSS_PCT    = 0.08    # 8% hard stop-loss below avg entry price
_CNN_ATR_TRAIL_MULT   = 2.0     # trailing stop = ATR(14) × multiplier below peak
_CNN_ATR_TRAIL_MIN    = 0.06    # floor: never tighter than 6% (Session 57 cash-flow lever 3 — was 3%, exited too early in low-ATR regimes)
_CNN_ATR_TRAIL_MAX    = 0.15    # ceiling: never wider than 15% (limits max give-back)
_CNN_MAX_HOLD_SECS    = 7 * 24 * 3600  # 7-day max-hold — last resort for flat/forgotten positions
# Positions missing entry_time (opened before that field existed) are treated
# as if they were opened _CNN_LEGACY_HOLD_SECS ago — triggers max-hold exit promptly.
_CNN_LEGACY_HOLD_SECS = _CNN_MAX_HOLD_SECS + 1  # exceeds max-hold, forces exit on next check
MIN_PRICE            = 0.01    # skip micro-priced tokens (0.0000 display = unprofitable spreads)

# Task #46 B2 — model-driven exit: force-exit a held position when v4.5 says
# DOWN is the most likely 72-hour outcome with confidence > threshold.
# generate_signal caches p_down on pos['p_down'] (~5min staleness max);
# _check_risk_exits + exit_watcher.on_price_tick read it.
_P_DOWN_EXIT_THRESHOLD = 0.55

# Task #80 — staleness gate. Cached p_down expires after _P_DOWN_STALE_MS
# because the WS handler reads it on every tick (~5-10/sec/pid) but the scan
# that refreshes it runs every SCAN_INTERVAL_SECS (60s). If the market
# reverses between scans, a cached high p_down would fire MODEL_DOWN/
# WS_MODEL_DOWN against a position that is now genuinely climbing.
# generate_signal writes pos['p_down_ts_ms'] alongside pos['p_down'];
# both exit paths must check freshness before reading the value. Missing
# ts_ms (legacy position) counts as stale to fail closed.
_P_DOWN_STALE_MS = 90_000


# LightGBM filter / Hurst / DI / entropy / regime-gate constants + helper
# deleted #311-refactor-f. They only ever fired under MODEL_BACKEND=cnn
# (Phase-1 CNN-tuned suppressions); under the live xgb backend they were
# already skipped via the backend-only guards. The whole block was dead.


def _backend_label() -> str:
    """#267 — User-facing log prefix that reflects the active decider.

    Both supported MODEL_BACKEND values ("xgb" and "xgb_v45") route through
    xgb_signal, so signals are always XGB-driven. The xgb_v45 driver uses
    the 3-class model + indep_thresholds decision; xgb (default) uses v3.
    """
    if config.model_backend == "xgb_v45":
        return "XGB_V45"
    return "XGB"


class _CNNBook:
    """Lightweight dry-run portfolio book for the CNN agent.
    Previously mirrored the _Book class in the retired tech_agent_cb
    (#311-refactor-c); now the only paper book in the system. Ensures CNN
    trades appear in the `trades` table and per-product positions are
    tracked (prevents buying the same asset repeatedly).
    """
    def __init__(self):
        self._agent      = "CNN"
        self.balance     = _CNN_DRY_RUN_BALANCE
        self.positions: Dict[str, Dict] = {}   # pid → {size, avg_price}
        self.realized_pnl = 0.0
        # Per-pid lock so WS exit handler and scan-loop _check_risk_exits
        # cannot race a duplicate database.close_trade write.
        self._sell_locks: Dict[str, asyncio.Lock] = {}

    def _lock_for(self, pid: str) -> asyncio.Lock:
        if pid not in self._sell_locks:
            self._sell_locks[pid] = asyncio.Lock()
        return self._sell_locks[pid]

    @staticmethod
    def _migrate_position_state(pos: Dict[str, Any]) -> Dict[str, Any]:
        """Add peak_pnl_pct + position_dollars to legacy position dicts (#46).

        peak_pnl_pct = (peak_price - avg_price) / avg_price, clamped to 0 if
        avg_price is zero (corrupt-entry case).
        position_dollars = size * avg_price as an initial value; next scan
        updates with the live current_price.

        Idempotent — preserves existing values if already present.
        """
        if "peak_pnl_pct" not in pos:
            avg = pos.get("avg_price", 0.0)
            peak = pos.get("peak_price", avg)
            pos["peak_pnl_pct"] = ((peak - avg) / avg) if avg > 0 else 0.0
        if "position_dollars" not in pos:
            size = pos.get("size", 0.0)
            avg = pos.get("avg_price", 0.0)
            pos["position_dollars"] = float(size * avg)
        return pos

    async def load(self) -> None:
        state = await database.load_agent_state(self._agent)
        if state:
            self.balance      = state["balance"]
            self.realized_pnl = state["realized_pnl"]
            self.positions    = state["positions"]

            # Drop corrupt positions (avg_price == 0) — these were opened before
            # the price filter existed and can never be exited meaningfully
            corrupt = [pid for pid, pos in self.positions.items()
                       if pos.get("avg_price", 0) == 0]
            for pid in corrupt:
                del self.positions[pid]
                logger.warning(
                    f"CNN book: dropped corrupt position {pid} (avg_price=0) — "
                    f"will reconcile open trade row below"
                )

            # Migrate legacy positions to include peak_pnl_pct + position_dollars
            # (task #46). Idempotent — preserves any values already present.
            for pid in list(self.positions.keys()):
                self.positions[pid] = self._migrate_position_state(self.positions[pid])

            logger.info(
                f"CNN book restored | balance=${self.balance:.2f} | "
                f"pnl=${self.realized_pnl:+.2f} | positions={len(self.positions)}"
                + (f" | dropped {len(corrupt)} corrupt" if corrupt else "")
            )
        else:
            logger.info(f"CNN book: no saved state — starting fresh at ${self.balance:.2f}")

        # Reconcile: close ghost open trades in the DB that don't match the current book.
        # Two cases:
        #   1. Product not in current positions at all → close every open row
        #   2. Product IS in current positions but has multiple open rows → keep only newest (highest id)
        open_trades = await database.get_trades(agent=self._agent, open_only=True, limit=500)

        # Group open rows by product, sorted newest-first (get_trades returns newest-first)
        by_product: Dict[str, list] = {}
        for t in open_trades:
            by_product.setdefault(t["product_id"], []).append(t)

        ghost_count = 0
        for pid, rows in by_product.items():
            if pid not in self.positions:
                # Orphan product — close all rows
                for t in rows:
                    await database.close_trade_by_id(t["id"], trigger_close="STARTUP_CLEANUP")
                    ghost_count += 1
            elif len(rows) > 1:
                # Duplicate rows — keep newest (rows[0], highest id first), close the rest
                for t in rows[1:]:
                    await database.close_trade_by_id(t["id"], trigger_close="STARTUP_CLEANUP")
                    ghost_count += 1

        if ghost_count:
            logger.info(f"CNN book: reconciled {ghost_count} ghost open trade row(s) from prior sessions")

    async def _save(self) -> None:
        await database.save_agent_state(
            self._agent, self.balance, self.realized_pnl, self.positions, {}
        )

    def has_position(self, pid: str) -> bool:
        return pid in self.positions

    # ── Win/loss performance tracking ─────────────────────────────────────────
    wins:          int   = 0
    losses:        int   = 0
    _sum_win_pct:  float = 0.0   # cumulative % gain on winning trades
    _sum_loss_pct: float = 0.0   # cumulative % loss on losing trades (positive number)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def expectancy(self) -> float:
        """Expected % gain per trade = win_rate * avg_win_pct - loss_rate * avg_loss_pct."""
        if self.wins + self.losses == 0:
            return 0.0
        avg_win  = self._sum_win_pct  / self.wins   if self.wins   > 0 else 0.0
        avg_loss = self._sum_loss_pct / self.losses if self.losses > 0 else 0.0
        return self.win_rate * avg_win - (1 - self.win_rate) * avg_loss

    async def buy(self, pid: str, price: float, frac: float,
                  trigger: str = "SCAN") -> Tuple[float, float]:
        # #125a: tiered blacklist gating. Active (or no row) → full frac.
        # Probation → half size (real money, reduced risk). Suspended →
        # paper-trade only: log the signal but do not spend or open a trade.
        status_row = await database.get_product_status(pid)
        status = status_row["status"] if status_row else "active"
        if status == "suspended":
            logger.info(
                "CNN PAPER %s @ %.6f frac=%.4f — suspended tier (no execution)",
                pid, price, frac,
            )
            return 0.0, 0.0
        if status == "probation":
            frac *= 0.5

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
            self.positions[pid] = {
                "size":       size,
                "avg_price":  price,
                "entry_time": time.time(),
                "peak_price": price,
            }
        self.balance -= spend
        await self._save()
        await database.open_trade(
            agent=self._agent, product_id=pid, entry_price=price,
            size=size, usd_open=spend, trigger_open=trigger,
            balance_after=self.balance,
        )
        return spend, size

    async def sell(self, pid: str, price: float, trigger: str = "SCAN") -> float:
        # Per-pid lock prevents WS exit handler and scan-loop _check_risk_exits
        # from racing a duplicate database.close_trade write on the same pid.
        # In-memory guard inside the lock makes the loser a clean no-op.
        async with self._lock_for(pid):
            if pid not in self.positions:
                return 0.0
            pos = self.positions[pid]
            proceeds = pos["size"] * price
            pnl      = proceeds - pos["size"] * pos["avg_price"]
            pct_pnl  = (price - pos["avg_price"]) / pos["avg_price"] * 100.0

            # #109: persist trade close to DB FIRST — trades table is the source of
            # truth for realized PnL. If this raises, leave in-memory state intact
            # so the caller (or a retry) can try again. Updating agent_state before
            # close_trade caused the divergence seen in Session 54: agent_state
            # captured the gain but no closed-trade row existed; on restart the
            # reconcile path force-closed the orphan with pnl=0, locking in skew.
            await database.close_trade(
                agent=self._agent, product_id=pid, exit_price=price,
                size=pos["size"], pnl=pnl, trigger_close=trigger,
                balance_after=self.balance + proceeds,
            )

            # close_trade succeeded — now mutate in-memory state and persist it
            self.positions.pop(pid)
            self.balance      += proceeds
            self.realized_pnl += pnl

            if pnl > 0:
                self.wins         += 1
                self._sum_win_pct += pct_pnl
            elif pnl < 0:
                self.losses         += 1
                self._sum_loss_pct  += abs(pct_pnl)

            await self._save()

            # #125b: re-evaluate product status on every successful close so a
            # fresh trade can immediately tip a product into Probation/Suspended
            # (or recover it). Evaluator failures must not block the trade.
            try:
                await product_status.evaluate_and_persist(pid, agent=self._agent)
            except Exception:
                logger.exception("product_status evaluator failed for %s", pid)

            return pnl

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH = True
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(
            "CUDA available — CNN will train/infer on GPU: %s",
            torch.cuda.get_device_name(0),
        )
    else:
        logger.info("CUDA not available — CNN will use CPU")
except ImportError:
    _TORCH  = False
    _DEVICE = None
    logger.warning("PyTorch not found — CNN agent uses linear fallback")

N_CHANNELS      = 28
SEQ_LEN         = 60

# Channels whose values are still constant-zero during training because the
# training loop (train_on_history) doesn't yet pass the upstream inputs that
# populate them. After #86 wired OKX funding history (replacing the geo-
# blocked Binance source from #80/#81), Ch 20 is now populated end-to-end and
# no longer masked. OKX retention is ~90 days; older training samples will
# still see zeros, but inference and recent windows carry real values.
# Path A (#46): Ch 10/11/24/25/26 swapped to candle-derived proxies — no
# external data needed, real per-bar variance instead of constants.
# #99: Ch 17/18/19 (fast_rsi_1h, velocity_1h, vol_zscore) emit constants
# (0.5, 0.0, 0.0) from FeatureBuilder fallback — the 1h-derived sources
# never feed real values. Permutation importance confirms delta=0 in both
# glu1 and glu2. Mask zeros them at both training and inference so they
# don't waste conv capacity on a constant input.
# At inference we zero any masked channels so the input distribution matches
# what the model actually saw during training (P3b). _train_arch applies
# the same mask to training tensors via _zero_mask_channels.
_TRAINING_CONSTANT_CHANNELS: frozenset = frozenset({17, 18, 19})


def _close_position(closes, highs, lows):
    """Ch 10: where close sits in the bar's high-low range, [0,1].
    1.0 = closed at high (buy pressure), 0.0 = closed at low.
    Returns 0.0 for zero-range bars (no div-by-zero)."""
    out = []
    for c, h, l in zip(closes, highs, lows):
        rng = h - l
        out.append((c - l) / rng if rng > 0 else 0.0)
    return out


def _bar_range(closes, highs, lows):
    """Ch 11: relative bar size (high-low)/close, scaled ×10 and clipped [0,1].
    A 1% bar maps to 0.1; a 10%+ bar saturates at 1.0."""
    out = []
    for c, h, l in zip(closes, highs, lows):
        if c <= 0:
            out.append(0.0)
            continue
        v = (h - l) / c * 10.0
        out.append(min(1.0, max(0.0, v)))
    return out


def _rv_series(closes, window):
    """Ch 24/25: rolling realized volatility (std of log returns) per bar.
    Returns a per-bar list, length == len(closes). Pre-window bars are 0.0.
    Output scaled ×50 and clipped [0,1] (a 2% std/bar saturates at 1.0)."""
    out = [0.0] * len(closes)
    for i in range(window, len(closes)):
        rets = []
        for j in range(i - window + 1, i + 1):
            if closes[j - 1] > 0 and closes[j] > 0:
                rets.append(math.log(closes[j] / closes[j - 1]))
        if not rets:
            continue
        mu = sum(rets) / len(rets)
        var = sum((r - mu) ** 2 for r in rets) / len(rets)
        out[i] = min(1.0, math.sqrt(var) * 50.0)
    return out


def _volume_sentiment(closes, volumes, window):
    """Ch 26: rolling up-volume / total-volume, mapped to [-1,1].
    +1.0 = all bars in window were up (close > prev close), -1.0 = all down.
    First bar is 0.0 (no prior close)."""
    out = [0.0] * len(closes)
    for i in range(1, len(closes)):
        start = max(1, i - window + 1)
        up = 0.0
        tot = 0.0
        for j in range(start, i + 1):
            v = volumes[j]
            tot += v
            if closes[j] > closes[j - 1]:
                up += v
        if tot > 0:
            out[i] = 2.0 * (up / tot) - 1.0
    return out


def _indep_thresholds_decision(
    p_down: float, p_neutral: float, p_up: float,
    thresh_up: float, thresh_down: float,
) -> Tuple[str, float]:
    """v4.5 indep_thresholds rule. Mirrors tools/v4_5_horizon_compare.py:138.

    BUY  when p_up   > thresh_up   AND p_up   >= p_down (tie -> BUY).
    SELL when p_down > thresh_down AND p_down >  p_up  (strict).
    Else HOLD.

    Returns (side, strength) where strength is the winning class probability
    rounded to 3 decimal places, or 0.0 for HOLD.
    """
    if p_up > thresh_up and p_up >= p_down:
        return "BUY", round(p_up, 3)
    if p_down > thresh_down and p_down > p_up:
        return "SELL", round(p_down, 3)
    return "HOLD", 0.0


def _mask_training_constant_channels(channels):
    """Return a new channel list with channels in _TRAINING_CONSTANT_CHANNELS
    replaced by all-zero sequences of the same length. Input is not mutated.
    """
    if not channels:
        return channels
    T = len(channels[0]) if channels[0] else 0
    zero = [0.0] * T
    return [
        list(zero) if idx in _TRAINING_CONSTANT_CHANNELS else ch
        for idx, ch in enumerate(channels)
    ]


def _zero_mask_channels(x):
    """#99: zero `_TRAINING_CONSTANT_CHANNELS` on an [N, C, T] tensor.

    Tensor analogue of `_mask_training_constant_channels` for the training
    pipeline. _train_arch calls this on `X_all` before the train/val split
    so the model sees zeros for masked channels at training, matching what
    `_cnn_prob` produces at inference. Without this, ch 17 (constant 0.5
    in cache) would train as 0.5 but be zeroed at inference — real
    train/serve skew. Input is not mutated.
    """
    if not _TRAINING_CONSTANT_CHANNELS:
        return x
    out = x.clone()
    for ch in _TRAINING_CONSTANT_CHANNELS:
        out[:, ch, :] = 0.0
    return out

_PHASE2_LOG_EVERY = 5  # log dataset-build progress every N products (watchdog)
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "..", "cnn_model.pt")
_DATASET_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "cnn_dataset_cache.pt")


def _dataset_fingerprint(all_candle_sets, seq_len, forward_hours,
                         label_thresh, n_channels) -> str:
    """Stable SHA-256 over inputs that affect the built X/y tensors.

    Two runs with the same fingerprint are guaranteed to produce identical
    datasets; any change (new candles, different horizon, new label
    threshold, different channel count) produces a different hex digest.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(f"{seq_len}|{forward_hours}|{label_thresh}|{n_channels}".encode())
    for candles in all_candle_sets:
        if not candles:
            h.update(b"|empty")
            continue
        first, last = candles[0], candles[-1]
        h.update(
            f"|{len(candles)}|{first.get('time')}|{last.get('time')}|"
            f"{last.get('close')}".encode()
        )
    return h.hexdigest()


def _load_dataset_cache(path: str, fingerprint: str):
    """Return (X_list, y_list) on fingerprint hit; None on miss or error."""
    if not os.path.exists(path):
        return None
    try:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as _le:
        logger.warning(f"CNN dataset cache load failed: {_le}")
        return None
    if not isinstance(blob, dict) or blob.get("fingerprint") != fingerprint:
        return None
    return blob.get("X"), blob.get("y")


def _save_dataset_cache(path: str, fingerprint: str, X_list, y_list) -> None:
    """Atomically save X/y lists + fingerprint to disk."""
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"fingerprint": fingerprint, "X": X_list, "y": y_list}, tmp)
    os.replace(tmp, path)


# ── Per-product append-only dataset cache (P2) ────────────────────────────────
# The fingerprint cache above treats the whole universe as one blob: a single
# new candle on any product forces the full 103-minute phase-2 rebuild. The
# per-product layer below stores one entry per product and appends only the
# samples that rely on candles beyond the previous `last_n`. Schema changes
# (seq_len / forward_hours / label_thresh / n_channels) still invalidate the
# whole cache, but day-to-day ticking touches at most a handful of samples
# per product.
# Version 4 = triple-barrier labels (P3a) + per-sample index tracking for
# López-de-Prado sample-uniqueness weighting (P3c).
_DATASET_CACHE_VERSION = 12  # bumped for #143-B: Ch 27 (OKX OI) added; channel count 27→28
                            # a single value computed on the full candle
                            # window across all 60 timesteps — every cached
                            # sample's Ch 15 series leaked future info.
                            # Now per-bar causal expanding window. v10 caches
                            # must rebuild to pick up the corrected channel.
                            # (Prior: v10 = #98 OKX rv20/rv60.)

# Triple-barrier parameters (P3a). López de Prado 2018: label a sample by
# whichever of {upper barrier hit, lower barrier hit, time barrier} fires
# first inside the forward window — cuts label noise vs. sign-of-final-return.
_TB_UP_MULT = 0.01   # +1% upper barrier
_TB_DN_MULT = 0.01   # -1% lower barrier

# Label smoothing (P3d). Szegedy et al. 2016 Inception-v3: replace hard
# targets {0,1} with soft {ε, 1-ε} so BCE stops pushing logits to ±∞ and
# the model doesn't become over-confident on noisy financial labels.
_LABEL_SMOOTH = 0.05


def _smooth_labels(y, eps: float):
    """Map hard binary labels to soft: y' = y*(1-2ε)+ε. y=0→ε, y=1→1-ε, y=0.5→0.5."""
    return y * (1.0 - 2.0 * eps) + eps


def _purged_walkforward_splits(
    sample_indices,
    n_splits: int,
    forward_hours: int,
    embargo_bars: int,
):
    """Walk-forward CV with purging and embargo (López de Prado 2018 ch. 7).

    Time-series CV must go forward-only (never train on the future) AND purge
    training samples whose forward label window overlaps the validation fold,
    plus embargo a gap of bars immediately after the val block so nothing
    leaks via serial correlation.

    Args:
      sample_indices: candle indices of samples in time order.
      n_splits: number of folds (>= 2). Fold k trains on all samples strictly
        before the purge boundary of val block k, validates on val block k.
      forward_hours: length of the forward label window — any training sample
        at candle `i` where `i + forward_hours >= first_val_candle` is purged.
      embargo_bars: bars after each val block reserved from training in
        subsequent folds (serial-correlation guard).

    Returns:
      List of (train_positions, val_positions) tuples, length n_splits.
      Positions index into `sample_indices`, not raw candle indices.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2 for walk-forward CV")
    n = len(sample_indices)
    if n == 0:
        return [([], []) for _ in range(n_splits)]
    # Carve the tail into n_splits contiguous val blocks; earlier positions
    # serve as the (purged, embargoed) training pool.
    start = int(n * 0.5)  # first half is pure training seed, tail split K ways
    block = max(1, (n - start) // n_splits)
    folds = []
    for k in range(n_splits):
        v_lo = start + k * block
        v_hi = start + (k + 1) * block if k < n_splits - 1 else n
        val_pos = list(range(v_lo, v_hi))
        if not val_pos:
            folds.append(([], []))
            continue
        first_val_candle = sample_indices[v_lo]
        # Collect training positions: strictly earlier than val, no forward-window
        # overlap with val start, and outside the embargo of any prior val block.
        train_pos = []
        for pos in range(v_lo):
            cand = sample_indices[pos]
            # Purge: forward window of this sample must end before first val candle
            if cand + forward_hours >= first_val_candle:
                continue
            # Embargo: skip bars in the band (prev_val_end, prev_val_end+embargo]
            embargoed = False
            for j in range(k):
                pv_lo = start + j * block
                pv_hi = start + (j + 1) * block if j < n_splits - 1 else n
                if pv_lo >= v_lo:
                    break
                prev_val_end = sample_indices[pv_hi - 1]
                if prev_val_end < cand <= prev_val_end + embargo_bars:
                    embargoed = True
                    break
            if embargoed:
                continue
            train_pos.append(pos)
        folds.append((train_pos, val_pos))
    return folds


# Walk-forward CV folds (P3e). n=1 disables CV (keeps legacy 80/20 split).
# n=3 = three chronological folds covering the last 50% of samples; each
# fold purges train samples whose forward window overlaps val and embargoes
# _WALKFORWARD_EMBARGO bars after each val block.
_WALKFORWARD_FOLDS = 3
_WALKFORWARD_EMBARGO = 4  # bars (= forward_hours, conservative default)


def _per_regime_metrics(y_true, y_pred, regimes):
    """Group (y_true, y_pred) pairs by regime label and compute per-group
    n, accuracy (0.5 threshold), mean BCE loss, and positive rate (P4).

    Overall aggregate val metrics hide regime-dependent behaviour — a model
    may be accurate in TRENDING regimes and worse-than-random in CHAOTIC
    regimes while showing a respectable mean. Breaking out by HMM regime
    (see services/hmm_regime.py) surfaces that asymmetry.

    Args:
      y_true: iterable of 0.0/1.0 labels.
      y_pred: iterable of probabilities in [0, 1] (post-sigmoid).
      regimes: iterable of regime strings (e.g. "TRENDING"/"RANGING"/"CHAOTIC").

    Returns:
      dict[regime_str, {"n", "acc", "loss", "pos_rate"}]
    """
    import math
    y_true = list(y_true)
    y_pred = list(y_pred)
    regimes = list(regimes)
    if not (len(y_true) == len(y_pred) == len(regimes)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"regimes={len(regimes)}"
        )
    if not y_true:
        return {}
    buckets: dict = {}
    for yt, yp, r in zip(y_true, y_pred, regimes):
        buckets.setdefault(r, {"yt": [], "yp": []})
        buckets[r]["yt"].append(float(yt))
        buckets[r]["yp"].append(float(yp))
    out: dict = {}
    EPS = 1e-12  # log clamp to avoid log(0) when prediction is exactly 0 or 1
    for r, b in buckets.items():
        yt_list = b["yt"]
        yp_list = b["yp"]
        n = len(yt_list)
        correct = sum(1 for yt, yp in zip(yt_list, yp_list) if (yp >= 0.5) == (yt >= 0.5))
        loss = -sum(
            yt * math.log(max(min(yp, 1.0 - EPS), EPS))
            + (1.0 - yt) * math.log(max(min(1.0 - yp, 1.0 - EPS), EPS))
            for yt, yp in zip(yt_list, yp_list)
        ) / n
        pos_rate = sum(1 for yt in yt_list if yt >= 0.5) / n
        out[r] = {
            "n": n,
            "acc": correct / n,
            "loss": loss,
            "pos_rate": pos_rate,
        }
    return out


def _precision_recall_at_threshold(probs, labels, threshold):
    """Precision and recall at a fixed probability threshold on the val set.

    Mirrors the production BUY gate: `model_prob > config.cnn_buy_threshold`
    (cnn_agent.py:1637 — strict >). A training session that scores high here
    would generate high-quality BUY signals at live inference time, which is
    what we actually care about — more than BCE (calibration) or AUC
    (all-threshold ranking).

    Args:
      probs: iterable of probabilities in [0, 1] (post-sigmoid).
      labels: iterable of 0.0/1.0 truth labels.
      threshold: gate threshold; a prediction counts as positive when prob > threshold.

    Returns:
      (precision, recall) where each is a float in [0, 1] or None.
      precision is None when no predictions exceed the threshold.
      recall is None when the val set contains no positive labels.
    """
    probs  = list(probs)
    labels = list(labels)
    if len(probs) != len(labels):
        raise ValueError(
            f"length mismatch: probs={len(probs)}, labels={len(labels)}"
        )
    if not probs:
        return (None, None)
    tp = fp = fn = 0
    total_pos = 0
    for p, y in zip(probs, labels):
        is_pos = float(y) >= 0.5
        predicted = float(p) > threshold
        if is_pos:
            total_pos += 1
        if predicted and is_pos:
            tp += 1
        elif predicted and not is_pos:
            fp += 1
        elif not predicted and is_pos:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall    = tp / total_pos if total_pos > 0 else None
    return (precision, recall)


def _compute_uniqueness(sample_indices, forward_hours: int, n_candles: int):
    """Per-sample weight = mean(1/N_t) over the forward window [i+1..i+h].

    N_t counts how many samples have a forward window that includes time t;
    time points beyond n_candles are dropped from the average rather than
    counted as N_t=0. Isolated samples get weight 1.0; densely overlapping
    samples approach 1/forward_hours. See López de Prado 2018 ch. 4.
    """
    if not sample_indices:
        return []
    N = [0] * n_candles
    for i in sample_indices:
        for t in range(i + 1, i + forward_hours + 1):
            if 0 <= t < n_candles:
                N[t] += 1
    weights = []
    for i in sample_indices:
        terms = 0.0
        count = 0
        for t in range(i + 1, i + forward_hours + 1):
            if 0 <= t < n_candles and N[t] > 0:
                terms += 1.0 / N[t]
                count += 1
        weights.append(terms / count if count else 0.0)
    return weights


def _label_triple_barrier(candles, i: int, max_bars: int,
                          up_mult: float, dn_mult: float,
                          label_thresh: float):
    """Return 1.0, 0.0, or None for the triple-barrier label at index i.

    Scans candles[i+1 .. i+max_bars] and returns:
      - 1.0  if the upper barrier (high >= entry*(1+up_mult)) hits first
      - 0.0  if the lower barrier (low  <= entry*(1-dn_mult)) hits first
      - sign of close move at the time-barrier bar, if neither fired
      - None if entry price invalid or final move is inside ±label_thresh
    When both barriers are touched in the same bar the close direction
    breaks the tie.
    """
    n = len(candles)
    if i < 0 or i >= n:
        return None
    entry = candles[i]["close"]
    if entry <= 0:
        return None
    upper = entry * (1.0 + up_mult)
    lower = entry * (1.0 - dn_mult)
    end   = min(n - 1, i + max_bars)
    for k in range(i + 1, end + 1):
        c  = candles[k]
        hi = c["high"]; lo = c["low"]
        hit_up = hi >= upper
        hit_dn = lo <= lower
        if hit_up and hit_dn:
            cls = c["close"]
            if cls > entry:
                return 1.0
            if cls < entry:
                return 0.0
            return None
        if hit_up:
            return 1.0
        if hit_dn:
            return 0.0
    # Time barrier — use sign of close move vs. dead-zone threshold.
    exit_px = candles[end]["close"]
    ret     = (exit_px - entry) / entry
    if abs(ret) < label_thresh:
        return None
    return 1.0 if ret > 0 else 0.0


def _dataset_schema(seq_len: int, forward_hours: int,
                    label_thresh: float, n_channels: int) -> dict:
    return {
        "version":       _DATASET_CACHE_VERSION,
        "seq_len":       seq_len,
        "forward_hours": forward_hours,
        "label_thresh":  label_thresh,
        "n_channels":    n_channels,
    }


_TRAIN_5M_LIMIT = 72   # last 6h of 5m bars — mirrors inference fetch in scan_all
_RV_PREFIX_BARS = 60   # #98: prior closes prepended for RV60/RV20 lookback


def _aligned_btc_closes(target_candles: List[Dict],
                        btc_candles: Optional[List[Dict]]) -> Optional[List[float]]:
    """Return BTC closes aligned 1:1 with `target_candles` (forward-fill on gaps).

    Both candle series are assumed to use a 'start' (or 'time') unix-timestamp
    key. For each target bar we use the most recent BTC close whose start_ts
    is <= target start_ts; pre-target BTC bars seed the initial value so the
    series never starts None. Returns None when btc_candles is empty/None
    (caller treats that as "no BTC reference, leave Ch 21 at zero").
    """
    if not btc_candles:
        return None
    btc_sorted = sorted(
        btc_candles,
        key=lambda c: c.get("start") or c.get("time") or 0,
    )
    btc_starts = [int(c.get("start") or c.get("time") or 0) for c in btc_sorted]
    btc_closes = [float(c["close"]) for c in btc_sorted]

    out: List[float] = []
    last: Optional[float] = None
    j = 0
    n_btc = len(btc_starts)
    for c in target_candles:
        ts = int(c.get("start") or c.get("time") or 0)
        while j < n_btc and btc_starts[j] <= ts:
            last = btc_closes[j]
            j += 1
        if last is None:
            # No pre/at-target BTC bar yet — use first available close as seed.
            last = btc_closes[0]
        out.append(last)
    return out


def _aligned_funding_rates(target_candles: List[Dict],
                           funding_history: Optional[List[Tuple[int, float]]]) -> Optional[List[float]]:
    """Return funding rates aligned 1:1 with `target_candles` (forward-fill).

    `funding_history` is a list of `(funding_time_ms, rate)` tuples sorted
    ascending — the format returned by
    `services.binance_funding_history.fetch_funding_history`. Funding events
    fire every 8h on Binance futures, so the same rate stays "active" for
    every candle bar between events. For each target bar we use the most
    recent funding event whose `funding_time_ms <= target_start_seconds *
    1000`. Pre-target events seed the initial value so the series never
    starts None; if every event is *after* the first target bar, we seed
    the early bars with the first available rate (matches
    `_aligned_btc_closes` behaviour). Returns None when `funding_history`
    is empty/None (caller treats that as "no rate, leave Ch 20 at zero").
    """
    if not funding_history:
        return None
    fh_sorted = sorted(funding_history, key=lambda tr: tr[0])
    fh_times  = [int(t) for t, _ in fh_sorted]
    fh_rates  = [float(r) for _, r in fh_sorted]

    out: List[float] = []
    last: Optional[float] = None
    j = 0
    n_fh = len(fh_times)
    for c in target_candles:
        ts_ms = int(c.get("start") or c.get("time") or 0) * 1000
        while j < n_fh and fh_times[j] <= ts_ms:
            last = fh_rates[j]
            j += 1
        if last is None:
            last = fh_rates[0]
        out.append(last)
    return out


def _aligned_oi_history(target_candles: List[Dict],
                        oi_history: Optional[List[Tuple[int, float]]]) -> Optional[List[float]]:
    """Return open-interest values aligned 1:1 with `target_candles` (forward-fill).

    `oi_history` is a list of `(oi_time_ms, value)` tuples sorted ascending —
    the format returned by `services.okx_oi_history.fetch_oi_history`. Each
    target bar gets the most-recent OI snapshot whose `oi_time_ms <=
    target_start_seconds * 1000`. Pre-target snapshots seed the initial
    value so the series never starts None; if every snapshot is *after* the
    first target bar, early bars are seeded with the first available value
    (mirrors `_aligned_funding_rates` / `_aligned_btc_closes`). Returns None
    when `oi_history` is empty/None (caller treats that as "no OI signal,
    leave Ch 27 at zero").
    """
    if not oi_history:
        return None
    oi_sorted = sorted(oi_history, key=lambda tr: tr[0])
    oi_times  = [int(t) for t, _ in oi_sorted]
    oi_vals   = [float(v) for _, v in oi_sorted]

    out: List[float] = []
    last: Optional[float] = None
    j = 0
    n_oi = len(oi_times)
    for c in target_candles:
        ts_ms = int(c.get("start") or c.get("time") or 0) * 1000
        while j < n_oi and oi_times[j] <= ts_ms:
            last = oi_vals[j]
            j += 1
        if last is None:
            last = oi_vals[0]
        out.append(last)
    return out


def _build_samples_range(candles, i_start: int, i_end: int,
                         fb, seq_len: int, forward_hours: int,
                         label_thresh: float,
                         btc_closes: Optional[List[float]] = None,
                         funding_rates: Optional[List[float]] = None,
                         oi_rates: Optional[List[float]] = None,
                         candles_5m: Optional[List[Dict]] = None):
    """Build (X, y, indices) for sliding-window indices i in [i_start, i_end).

    Labels use triple-barrier (P3a): upper barrier (+_TB_UP_MULT), lower
    barrier (-_TB_DN_MULT), or time barrier at i+forward_hours. Dead-zone
    samples (time barrier && |final_ret| < label_thresh) are skipped.
    The returned `indices` list records the candle index i that produced
    each retained sample — needed for uniqueness weighting (P3c).

    When `btc_closes` is provided, it must be the same length as `candles`
    and time-aligned 1:1. Each sample at candle index i receives the slice
    btc_closes[i - seq_len + 1 : i + 1] so FeatureBuilder.build can populate
    Ch 21 (rolling BTC-return correlation) during training (#53).

    When `funding_rates` is provided (same length, time-aligned), each sample
    at candle index i forwards the scalar funding_rates[i] to fb.build so
    Ch 20 (funding rate) is populated with the rate active at that sample's
    timestamp (#54). Rates should be the raw per-period rate (e.g. 0.0001
    for +0.01%); normalisation/clipping happens inside fb.build.

    When `candles_5m` is provided, each sample at hourly candle index i is
    given the 5m bars whose `start` falls strictly before the close of hour
    i (i.e. start < candles[i].start + 3600), capped at the last 72 bars to
    mirror the inference fetch in `scan_all`. Replaces the previous synthetic
    proxy `candles[max(0, i-11): i+1]` so Ch 17/18/19 see real fast-timeframe
    data during training (#56). When None, falls back to that synthetic proxy.
    """
    X_list, y_list, idx_list = [], [], []
    if i_end <= i_start:
        return X_list, y_list, idx_list

    # Pre-extract 5m start timestamps once for O(log n) per-sample lookup.
    c5m_starts: Optional[List[int]] = None
    if candles_5m:
        c5m_starts = [int(c["start"]) for c in candles_5m]

    # #143-B: z-score OI over the full per-product series so each sample's
    # scalar OI is in standard-deviation units before fb.build clips at 3σ.
    oi_mean: float = 0.0
    oi_std: float = 1.0
    if oi_rates is not None and len(oi_rates) > 0:
        _oi_n = len(oi_rates)
        oi_mean = sum(oi_rates) / _oi_n
        _var = sum((v - oi_mean) ** 2 for v in oi_rates) / _oi_n
        oi_std = math.sqrt(_var) if _var > 0 else 1.0

    for i in range(i_start, i_end):
        label = _label_triple_barrier(
            candles, i, forward_hours, _TB_UP_MULT, _TB_DN_MULT, label_thresh
        )
        if label is None:
            continue
        window  = candles[i - seq_len + 1: i + 1]
        if c5m_starts is not None:
            t_close = int(candles[i]["start"]) + 3600  # exclusive end of hour i
            cutoff  = bisect.bisect_left(c5m_starts, t_close)
            five_m  = candles_5m[max(0, cutoff - _TRAIN_5M_LIMIT): cutoff]
        else:
            five_m = candles[max(0, i - 11): i + 1]
        btc_win = (btc_closes[i - seq_len + 1: i + 1]
                   if btc_closes is not None else None)
        fr_val  = funding_rates[i] if funding_rates is not None else None
        if oi_rates is not None:
            oi_val_z = (oi_rates[i] - oi_mean) / oi_std
        else:
            oi_val_z = None
        # #98: 60-bar prefix lookback for RV20/RV60 — without it _rv_series
        # returns 0.0 for the first `window` bars, leaving Ch 25 (rv60)
        # constant-zero across the entire SEQ_LEN=60 window.
        ext_start = max(0, i - seq_len + 1 - _RV_PREFIX_BARS)
        closes_ext = [c["close"] for c in candles[ext_start: i + 1]]
        channels = fb.build(window, {}, candles_5m=five_m,
                            btc_closes=btc_win, funding_rate=fr_val,
                            closes_ext=closes_ext, oi_rate=oi_val_z)
        X_list.append(fb.to_tensor(channels))
        y_list.append(label)
        idx_list.append(i)
    return X_list, y_list, idx_list


def _extend_or_rebuild_product(entry, candles, fb, seq_len: int,
                               forward_hours: int, label_thresh: float,
                               btc_closes: Optional[List[float]] = None,
                               funding_rates: Optional[List[float]] = None,
                               oi_rates: Optional[List[float]] = None,
                               candles_5m: Optional[List[Dict]] = None):
    """Return (new_entry, status) for one product's per-product cache slot.

    status ∈ {"skip", "hit", "append", "rebuild"}
      • skip    → candles too short for even one sample; entry is None
      • hit     → cache was already up to date
      • append  → new candles extended the entry with incremental samples
      • rebuild → entry was missing or misaligned; full rebuild

    Append path runs only when entry["first_ts"] matches AND last_n fits
    inside the new candle series (last_n <= len(candles)).

    `btc_closes` is forwarded verbatim (aligned 1:1 with `candles`) to
    `_build_samples_range` for both rebuild and append paths (#53).
    `funding_rates` is forwarded verbatim (aligned 1:1 with `candles`) to
    `_build_samples_range` for both rebuild and append paths (#54).
    `candles_5m` (per-product 5m parquet, time-aligned by timestamp) is
    forwarded verbatim to both paths (#56).
    """
    n = len(candles)
    if n < seq_len + forward_hours + 1:
        return None, "skip"

    first_ts = candles[0].get("start") or candles[0].get("time")
    last_ts  = candles[-1].get("start") or candles[-1].get("time")

    def _full_rebuild():
        X, y, idx = _build_samples_range(
            candles, seq_len - 1, n - forward_hours,
            fb, seq_len, forward_hours, label_thresh,
            btc_closes=btc_closes,
            funding_rates=funding_rates,
            oi_rates=oi_rates,
            candles_5m=candles_5m,
        )
        return {
            "first_ts": first_ts, "last_ts": last_ts, "last_n": n,
            "X": X, "y": y, "indices": idx,
        }

    if (entry is None
            or entry.get("first_ts") != first_ts
            or int(entry.get("last_n", 0)) > n
            or "indices" not in entry):
        # Missing "indices" ⇒ v<4 cache; full rebuild reseeds it.
        return _full_rebuild(), "rebuild"

    last_n = int(entry["last_n"])
    if last_n == n:
        return entry, "hit"

    i_start = max(seq_len - 1, last_n - forward_hours)
    i_end   = n - forward_hours
    X_new, y_new, idx_new = _build_samples_range(
        candles, i_start, i_end,
        fb, seq_len, forward_hours, label_thresh,
        btc_closes=btc_closes,
        funding_rates=funding_rates,
        oi_rates=oi_rates,
        candles_5m=candles_5m,
    )
    return {
        "first_ts": first_ts,
        "last_ts":  last_ts,
        "last_n":   n,
        "X":        list(entry.get("X", [])) + X_new,
        "y":        list(entry.get("y", [])) + y_new,
        "indices":  list(entry.get("indices", [])) + idx_new,
    }, "append"


def _load_pp_cache(path: str, schema: dict):
    """Return cached {pid: entry} dict when schema matches; None otherwise."""
    if not os.path.exists(path):
        return None
    try:
        import torch
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as _le:
        logger.warning(f"CNN per-product cache load failed: {_le}")
        return None
    if not isinstance(blob, dict) or blob.get("schema") != schema:
        return None
    products = blob.get("products")
    return products if isinstance(products, dict) else None


def _save_pp_cache(path: str, schema: dict, products: dict) -> None:
    """Atomically save {schema, products} to disk.

    On Windows, os.replace raises PermissionError (WinError 5) when the
    destination is held open by another process — most commonly the live
    backend keeping cnn_dataset_cache.pt mapped during inference. The lock
    window is brief, so retry a few times before giving up (#108).
    """
    import torch, time as _t
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"schema": schema, "products": products}, tmp)
    last_exc = None
    for delay in (0.0, 0.5, 1.5, 3.0):
        if delay:
            _t.sleep(delay)
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_exc = e
    raise last_exc

_MODEL_BAK_PATH = MODEL_PATH + ".bak"
_BEST_LOSS_PATH = os.path.join(os.path.dirname(__file__), "..", "cnn_best_loss.txt")
_CKPT_PATH      = os.path.join(os.path.dirname(__file__), "..", "cnn_checkpoint_resume.pt")


def _model_path_for() -> str:
    """Path to the active CNN checkpoint. Multi-arch registry deleted
    #311-refactor-e; glu1 is the only surviving arch."""
    return os.path.join(os.path.dirname(__file__), "..", "cnn_model_glu1.pt")


def _best_loss_path_for() -> str:
    """Path to the active best-loss baseline (glu1)."""
    return os.path.join(os.path.dirname(__file__), "..", "cnn_best_loss_glu1.txt")
_CKPT_EVERY     = 10   # save resume checkpoint every N epochs
_HEARTBEAT_EVERY = 1   # #106: every epoch — under GPU contention 5+ min epochs caused
                       # check (1800s) doesn't kill healthy training on slow archs
# OLLAMA_URL deleted #311-refactor-f — Ollama LLM blend was CNN-backend-only
# and is now removed; cnn_agent makes no HTTP calls to Ollama.
_CACHE_TTL      = 300
_EARLY_STOP_PATIENCE = 15   # stop if val_loss doesn't improve for this many epochs


# ── CNN-LSTM Model ────────────────────────────────────────────────────────────

if _TORCH:
    class GatedConv1d(nn.Module):
        """
        GLU conv block with BatchNorm. BatchNorm normalises activations so
        gradients don't vanish through stacked gated layers.
        """
        def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, padding: int = 1):
            super().__init__()
            self.conv_main = nn.Conv1d(in_ch, out_ch, kernel, padding=padding)
            self.conv_gate = nn.Conv1d(in_ch, out_ch, kernel, padding=padding)
            self.bn        = nn.BatchNorm1d(out_ch)

        def forward(self, x):
            out = self.conv_main(x) * torch.sigmoid(self.conv_gate(x))
            return self.bn(out)

    # SignalCNN (arch="glu2", ~280k params) DELETED #311-refactor-e —
    # multi-arch registry collapsed to glu1 only. Glu2 checkpoint preserved
    # at backend/retired/cnn_model_glu2.pt (host-side, gitignored).

    class SignalCNNGlu1(nn.Module):
        """Capacity-reduced sibling of SignalCNN — half-depth, half-width.

        Built to combat the epoch-1 memorization seen on glu2 in production
        retrains: ~20× fewer params trades expressivity for generalization.
        """
        arch = "glu1"

        def __init__(self, n_ch: int = N_CHANNELS):
            super().__init__()
            self.c1   = GatedConv1d(n_ch, 16)
            self.c2   = GatedConv1d(16,   32)
            self.p2   = nn.MaxPool1d(2)        # 60 → 30
            self.lstm = nn.LSTM(32, 16, num_layers=1, batch_first=True)
            self.drop = nn.Dropout(0.3)
            self.fc   = nn.Linear(16, 1)

        def forward(self, x):
            x = self.c1(x)
            x = self.c2(x); x = self.p2(x)
            x = x.permute(0, 2, 1)             # (B, seq, 32)
            self.lstm.flatten_parameters()
            x, _ = self.lstm(x)
            x = x[:, -1, :]
            x = self.drop(x)
            return self.fc(x)

        def predict(self, tensor: "torch.Tensor") -> float:
            self.eval()
            with torch.no_grad():
                device = next(self.parameters()).device
                return float(torch.sigmoid(self.forward(tensor.unsqueeze(0).to(device))).item())

    # SignalCNNGluM (arch="glum", ~50k params) DELETED #311-refactor-e —
    # never trained to production checkpoint; was a mid-size experiment.


def _build_cnn():
    """Construct the active CNN arch (glu1). Multi-arch registry deleted
    #311-refactor-e; glu1 is the only surviving arch."""
    return SignalCNNGlu1()


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
              btc_closes: Optional[List[float]] = None,
              funding_rate: Optional[float] = None,
              iv_rv20_spread: Optional[float] = None,
              iv_rv60_spread: Optional[float] = None,
              ls_sentiment: Optional[float] = None,
              closes_ext: Optional[List[float]] = None,
              oi_rate: Optional[float] = None,
              T: int = SEQ_LEN) -> List[List[float]]:
        if not candles:
            return [[0.0] * T] * N_CHANNELS

        # Pre-extract timestamps for time-of-day channels (Ch 22/23)
        _hours = []
        for c in candles:
            ts = c.get("start") or c.get("time") or 0
            try:
                if isinstance(ts, str):
                    # ISO format: 2025-01-06T10:00:00Z or 2025-01-06T10:00:00
                    _hours.append(int(ts[11:13]))
                else:
                    # Unix timestamp (int or float)
                    _hours.append(_dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).hour)
            except Exception:
                _hours.append(0)

        opens   = [c["open"]   for c in candles]
        highs   = [c["high"]   for c in candles]
        lows    = [c["low"]    for c in candles]
        closes  = [c["close"]  for c in candles]
        volumes = [c["volume"] for c in candles]

        # ── Ch 0: Normalised close ────────────────────────────────────────────
        # Within-window min/max normalization. Strict-causality test in
        # test_feature_builder_causality.py::TestAllChannelsLookahead xfails
        # this channel: every value depends on global min/max across the
        # whole input. Per #171 decision: accept-as-design — normalization
        # is anchored at the terminal of the SEQ_LEN window the CNN sees,
        # so the backward dependency does not cross the prediction
        # boundary. Switching to per-bar expanding min/max would change
        # the normalization scale and require a full retrain.
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
        if len(closes) >= 16:   # fast MACD(5,13,3) needs slow+signal=16 bars
            for i in range(16, len(closes) + 1):
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

        # ── Ch 10 & 11: Candle-derived intra-bar pressure (#46-A) ────────────
        # Replaced order-book depth (no historical L1 source) with candle proxies.
        bid_ch = _close_position(closes, highs, lows)   # buy pressure 0..1
        ask_ch = _bar_range(closes, highs, lows)        # bar volatility 0..1

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
        # Per-bar causal: ADX at bar i uses only highs/lows/closes[:i+1].
        # Earlier impl computed adx_val once on the full window and broadcast
        # it across all timesteps, which leaked future info into every bar
        # of every cached sample.
        adx_ch = [
            _adx(highs[: i + 1], lows[: i + 1], closes[: i + 1])[0] / 100.0
            for i in range(len(closes))
        ]

        # ── Ch 16: VWAP distance per bar ──────────────────────────────────────
        # Rolling 20-bar VWAP; distance = (close - vwap) / vwap, norm to [-1,1]
        vwap_ch = [0.0] * len(closes)
        vwap_p  = 20
        for i in range(vwap_p, len(closes) + 1):
            _, dist = _vwap(highs[:i], lows[:i], closes[:i], volumes[:i], vwap_p)
            vwap_ch[i - 1] = dist

        # ── Ch 17-19: 5-minute fast channels — SKIPPED (Tier A, Session 58.71w)
        # _TRAINING_CONSTANT_CHANNELS = {17, 18, 19} masks these channels to
        # zero before the model forward pass (per invariant #11), so the
        # builder values were always overwritten with zeros downstream.
        # Skipping the computation here is mathematically equivalent and
        # avoids the wasted work each scan. The 5m candle data
        # (candles_5m) is still accepted in the signature for backward
        # compatibility, but is no longer consumed by the build path.
        fast_rsi_ch = [0.0] * len(closes)
        vel_ch      = [0.0] * len(closes)
        vol_z_ch    = [0.0] * len(closes)

        # ── Ch 20: Funding rate ───────────────────────────────────────────────
        # Positive = longs pay shorts (bullish crowding); normalised to [-1,1]
        # Typical range ±0.1%; clip at ±1% → / 0.01 gives [-1,1]
        fr_val  = float(funding_rate) if funding_rate is not None else 0.0
        fr_norm = max(-1.0, min(1.0, fr_val / 0.01))
        fund_ch = [fr_norm] * len(closes)

        # ── Ch 21: BTC return correlation (rolling 20 bars) ───────────────────
        corr_ch = [0.0] * len(closes)
        if btc_closes and len(btc_closes) >= 2 and len(closes) >= 2:
            n_btc = min(len(btc_closes), len(closes))
            asset_ret = [
                (closes[i]     - closes[i - 1])    / max(closes[i - 1],    1e-9)
                for i in range(1, n_btc)
            ]
            btc_ret = [
                (btc_closes[i] - btc_closes[i - 1]) / max(btc_closes[i - 1], 1e-9)
                for i in range(len(btc_closes) - n_btc + 1, len(btc_closes))
            ]
            roll = 20
            for i in range(roll - 1, len(asset_ret)):
                a = asset_ret[max(0, i - roll + 1): i + 1]
                b = btc_ret[max(0, i - roll + 1):  i + 1]
                n  = len(a)
                if n < 5:
                    continue
                ma = sum(a) / n; mb = sum(b) / n
                num = sum((a[j] - ma) * (b[j] - mb) for j in range(n))
                da  = math.sqrt(sum((v - ma) ** 2 for v in a))
                db  = math.sqrt(sum((v - mb) ** 2 for v in b))
                corr = num / (da * db) if da > 0 and db > 0 else 0.0
                corr_ch[i + 1] = max(-1.0, min(1.0, corr))

        # ── Ch 22 & 23: Time-of-day sin/cos encoding ──────────────────────────
        _2pi_24 = 2.0 * math.pi / 24.0
        sin_ch  = [math.sin(_hours[i] * _2pi_24) for i in range(len(closes))]
        cos_ch  = [math.cos(_hours[i] * _2pi_24) for i in range(len(closes))]

        # ── Ch 24 & 25: Realized-vol windows (#46-A, #98) ────────────────────
        # Replaced IV/RV spread (Deribit only had BTC/ETH) with pure RV — no
        # external dep, real per-bar variance for all products. #98 adds
        # closes_ext: when caller provides ≥len(closes) prior closes, RV is
        # computed over the extended series so the first 20/60 bars of the
        # in-window region are non-zero (without prefix, _rv_series returns
        # 0.0 for the first `window` bars and rv60 was zero across SEQ_LEN=60).
        if closes_ext is not None and len(closes_ext) >= len(closes):
            _rv_src = closes_ext
        else:
            _rv_src = closes
        ivrv20_ch = _rv_series(_rv_src, window=20)[-len(closes):]
        ivrv60_ch = _rv_series(_rv_src, window=60)[-len(closes):]

        # ── Ch 26: Volume sentiment (#46-A) ──────────────────────────────────
        # Replaced Binance L/S (geo-blocked) with rolling up-vol / total-vol.
        ls_ch = _volume_sentiment(closes, volumes, window=20)

        # ── Ch 27: Open Interest (#143-B) ─────────────────────────────────────
        # Z-scored upstream by `_build_samples_range` over full per-product series.
        # Clip to [-1, 1] via /3.0 (3σ → 1.0). Broadcast scalar across window
        # like Ch 20 funding pattern.
        oi_val  = float(oi_rate) if oi_rate is not None else 0.0
        oi_norm = max(-1.0, min(1.0, oi_val / 3.0))
        oi_ch   = [oi_norm] * len(closes)

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
            P(fund_ch,                                 T),   # 20
            P(corr_ch,                                 T),   # 21
            P(sin_ch,                                  T),   # 22
            P(cos_ch,                                  T),   # 23
            P(ivrv20_ch,                               T),   # 24
            P(ivrv60_ch,                               T),   # 25
            P(ls_ch,                                   T),   # 26
            P(oi_ch,                                   T),   # 27
        ]
        assert len(channels) == N_CHANNELS
        return channels

    def to_tensor(self, channels):
        if not _TORCH:
            return channels
        return torch.tensor(channels, dtype=torch.float32)

    @staticmethod
    def get_learned_weights(model) -> Optional[List[float]]:
        """
        Extract per-channel importance from the first conv block's weights.
        Works for both GatedConv1d (reads conv_main) and legacy Conv1d blocks.
        Returns a list of N_CHANNELS floats that sum to 1.0.
        """
        if not _TORCH or model is None:
            return None
        try:
            first = model.c1
            # GatedConv1d: use main path weights
            if isinstance(first, GatedConv1d):
                w = first.conv_main.weight
            else:
                w = first.weight
            # Mean absolute weight per input channel across all output filters and kernel positions
            importance = w.abs().mean(dim=(0, 2)).detach().cpu().tolist()
            total = sum(importance) or 1.0
            return [v / total for v in importance]
        except Exception:
            return None


# ── Ollama LLM blend deleted #311-refactor-f ─────────────────────────────────
# The Ollama probability call and the associated decisive-skip decision tree
# were CNN-backend-only (per #250 the LLM was already skipped wholesale under
# MODEL_BACKEND=xgb). Removed alongside Hurst / LightGBM / regime gates as
# part of the dead CNN-only branch cleanup. Re-introducing an LLM blend would
# need its own brainstorm + spec.


# ── CNN-LSTM Agent ─────────────────────────────────────────────────────────────

class CoinbaseCNNAgent:
    def __init__(self, ws_subscriber=None):
        self.ws     = ws_subscriber
        self.fb     = FeatureBuilder()
        self._cache: Dict[str, Tuple[float, float, Dict[str, float]]] = {}
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
        # ── LLM token tracking ─────────────────────────────────────────────
        # Kept for main.py /api/cnn/status payload back-compat; always zero
        # after #311-refactor-f removed the Ollama blend.
        self.llm_calls:          int = 0
        self.llm_prompt_tokens:  int = 0
        self.llm_response_tokens: int = 0
        # ── Model health flags ──────────────────────────────────────────────
        # True when the on-disk checkpoint is incompatible with the current
        # architecture (channel count changed).  Signals are suppressed until
        # a fresh train run completes and saves a compatible checkpoint.
        self._needs_retrain: bool = False
        # Set by main.py when a training subprocess is running.  Read by
        # /api/cnn/status; no longer consulted inside generate_signal after
        # #311-refactor-f deleted the Ollama-skip tree.
        self.training_active: bool = False
        # Multi-arch registry deleted #311-refactor-e — glu1 is the only arch.
        if _TORCH:
            self.model = _build_cnn().to(_DEVICE)
            self._load()
        _status = "incompatible — signals suppressed until retrained" if self._needs_retrain else \
                  ("loaded" if self._exists() else "random (untrained)")
        _device_str = str(_DEVICE) if _TORCH else "n/a"
        logger.info(
            f"CoinbaseCNNAgent ready | torch={'yes' if _TORCH else 'linear'} | "
            f"device={_device_str} | model={_status} | arch=glu1 | "
            f"channels={N_CHANNELS}"
        )

    async def start(self) -> None:
        """Load persisted book state before first scan."""
        await self.book.load()

    def _exists(self) -> bool:
        return os.path.exists(_model_path_for())

    def _load(self):
        if not _TORCH or not self.model or not self._exists():
            return
        try:
            ckpt = torch.load(_model_path_for(), map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                arch          = ckpt.get("arch", "legacy")
                ckpt_channels = ckpt.get("n_channels", N_CHANNELS)

                if ckpt_channels != N_CHANNELS:
                    logger.warning(
                        f"CNN checkpoint has {ckpt_channels} channels, model expects {N_CHANNELS} "
                        f"— checkpoint incompatible. Signals suppressed until retrain completes."
                    )
                    self._needs_retrain = True
                    return

                if arch != getattr(self.model, "arch", "glu2"):
                    logger.warning(
                        f"CNN checkpoint arch='{arch}' incompatible with current arch="
                        f"'{getattr(self.model, 'arch', 'glu2')}' — signals suppressed until retrain."
                    )
                    self._needs_retrain = True
                    return

                self.model.load_state_dict(ckpt["state_dict"])
            else:
                # Legacy plain state_dict — attempt load
                self.model.load_state_dict(ckpt)
            self.model.to(_DEVICE)
            logger.info(
                f"CNN model loaded from disk "
                f"(arch={getattr(self.model, 'arch', 'unknown')}, device={_DEVICE})"
            )
        except Exception as e:
            logger.warning(
                f"CNN model load failed: {e} — signals suppressed until retrain completes"
            )
            self._needs_retrain = True

    def save_model(self, backup: bool = False):
        if not (_TORCH and self.model):
            return
        path = _model_path_for()
        if backup and os.path.exists(path):
            shutil.copy2(path, path + ".bak")
        torch.save(
            {
                "arch":       getattr(self.model, "arch", "glu"),
                "n_channels": N_CHANNELS,
                "state_dict": self.model.state_dict(),
            },
            path,
        )

    def _read_best_loss(self) -> float:
        try:
            v = float(open(_best_loss_path_for()).read().strip())
            # BCE at chance ≈ 0.693; anything below 0.1 is a stale/corrupted
            # sentinel (historic bug: content=1e-06 rejected every trained model).
            return v if v >= 0.1 else float("inf")
        except Exception:
            return float("inf")

    def _write_best_loss(self, loss: float) -> None:
        try:
            with open(_best_loss_path_for(), "w") as f:
                f.write(str(loss))
        except Exception:
            pass

    def _cnn_prob(self, channels, pid: Optional[str] = None) -> float:
        # Align inference input with the training distribution — zero out the
        # channels that were constant-zero at training (P3b).
        channels = _mask_training_constant_channels(channels)
        if config.model_backend in ("xgb", "xgb_v45"):
            from agents import xgb_signal
            return xgb_signal.xgb_prob(channels, pid=pid)
        # Unreachable under config._validate_backend; defensive only.
        raise RuntimeError(f"unsupported model_backend={config.model_backend!r}")

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

    async def _check_risk_exits(self, order_executor=None) -> None:
        """Check all open CNN positions for stop-loss, trailing stop, or max hold time.

        Runs at the top of every scan loop iteration so exits happen every 15 min
        even if no new signal is generated.

        Exit priority:
          1. Hard stop-loss   : price dropped ≥ _CNN_STOP_LOSS_PCT (8%) below avg entry
          2. Trailing stop    : price fell ≥ _CNN_TRAIL_PCT (6%) below the session peak
          3. Max hold time    : position held longer than _CNN_MAX_HOLD_SECS (7 days)
          4. Legacy position  : no entry_time recorded → exit on next check
        """
        for pid in list(self.book.positions.keys()):
            pos = self.book.positions.get(pid)
            if not pos:
                continue

            # Resolve current price — WS first, REST fallback
            price = self._live_price(pid, 0.0)
            if not price:
                try:
                    data  = await coinbase_client.get_product(pid)
                    price = float(data.get("price", 0) or 0)
                except Exception:
                    pass
            if not price:
                continue   # no price available — skip safely

            avg_price  = pos["avg_price"]
            pct_entry  = (price - avg_price) / avg_price

            # Update peak price — ratchets up, never down
            peak_price = pos.get("peak_price") or avg_price
            if price > peak_price:
                peak_price = price
                pos["peak_price"] = peak_price

            pct_from_peak = (price - peak_price) / peak_price

            # Compute ATR-based trail distance for this product
            trail_pct = _CNN_ATR_TRAIL_MIN  # fallback if candles unavailable
            try:
                candles = await database.get_candles(pid, limit=20)
                if len(candles) >= 15:
                    highs  = [c["high"]  for c in candles]
                    lows   = [c["low"]   for c in candles]
                    closes = [c["close"] for c in candles]
                    atr    = _atr(highs, lows, closes)
                    if atr > 0 and peak_price > 0:
                        raw = atr * _CNN_ATR_TRAIL_MULT / peak_price
                        trail_pct = max(_CNN_ATR_TRAIL_MIN, min(raw, _CNN_ATR_TRAIL_MAX))
            except Exception:
                pass

            # Cache for WS exit handler — read by agents/exit_watcher.on_price_tick
            # so it doesn't recompute ATR per tick (~580 ticks/sec aggregate).
            pos["trail_pct"] = trail_pct

            # Task #46: PnL-anchored trail. Ratchet peak_pnl_pct from the
            # ratcheted peak_price (handles positions whose peak existed before
            # the field was added; migration also seeds this on load).
            peak_pnl_from_peak_price = (peak_price - avg_price) / avg_price if avg_price > 0 else 0.0
            pos["peak_pnl_pct"] = max(pos.get("peak_pnl_pct", 0.0), peak_pnl_from_peak_price)
            pos["position_dollars"] = float(pos["size"]) * price

            total_capital = self.book.balance + sum(
                p.get("position_dollars", 0.0) for p in self.book.positions.values()
            )
            exit_threshold = _compute_exit_threshold(
                peak_pnl_pct=pos["peak_pnl_pct"],
                atr_pct=trail_pct,
                position_dollars=pos["position_dollars"],
                total_capital=total_capital,
            )

            # Positions without entry_time are legacy — treat as already overdue
            entry_time = pos.get("entry_time")
            hold_secs  = _CNN_LEGACY_HOLD_SECS if entry_time is None else time.time() - entry_time

            # Task #46 B2: MODEL_DOWN — force-exit when v4.5 says drop is
            # likely (cached by generate_signal). Sits between STOP_LOSS
            # (capital protection) and TRAIL_STOP (profit protection).
            # Task #80: ignore cached p_down older than _P_DOWN_STALE_MS;
            # missing ts means legacy position → fail closed (no fire).
            now_ms = int(time.time() * 1000)
            p_down_ts = pos.get("p_down_ts_ms")
            p_down = pos.get("p_down", 0.0) if (
                p_down_ts is not None and (now_ms - p_down_ts) <= _P_DOWN_STALE_MS
            ) else 0.0

            trigger = None
            if pct_entry <= -_CNN_STOP_LOSS_PCT:
                trigger = "STOP_LOSS"
            elif p_down > _P_DOWN_EXIT_THRESHOLD:
                trigger = "MODEL_DOWN"
            elif pct_entry < exit_threshold:
                trigger = "TRAIL_STOP"
            elif hold_secs >= _CNN_MAX_HOLD_SECS:
                trigger = "MAX_HOLD" if entry_time else "LEGACY_EXIT"

            if trigger:
                size = pos["size"]
                pnl  = await self.book.sell(pid, price, trigger=trigger)
                try:
                    await exit_execution.execute_live_exit(
                        order_executor, pid=pid, price=price,
                        size=size, trigger=trigger,
                    )
                except Exception:
                    logger.exception(
                        "live exit execution failed for %s (%s)", pid, trigger,
                    )
                logger.info(
                    f"CNN RISK EXIT {pid} @{price:.6f} | {trigger} | "
                    f"entry={pct_entry*100:+.2f}% peak={peak_price:.6f} "
                    f"peak_pnl={pos['peak_pnl_pct']*100:+.2f}% "
                    f"exit_thr={exit_threshold*100:+.2f}% (atr_trail={trail_pct*100:.1f}%) "
                    f"hold={hold_secs/3600:.1f}h | "
                    f"pnl=${pnl:+.2f} | balance=${self.book.balance:.2f}"
                )

    async def generate_signal(
        self,
        product: Dict,
        execute: bool = False,
        order_executor=None,
    ) -> Optional[Dict]:
        # #223 — XGB backend doesn't use the PyTorch CNN checkpoint, so an
        # incompatible CNN model must not block XGB shadow logging or signal
        # generation. Without this guard, the 28-channel migration left the
        # 27-channel CNN checkpoint flagged incompatible and silently killed
        # all save_cnn_scan writes for ~30 hours starting 2026-05-07 02:17 UTC.
        if self._needs_retrain and config.model_backend != "xgb":
            logger.debug(
                "CNN signal suppressed — checkpoint incompatible with current architecture. "
                "Run retrain.py or trigger /api/cnn/train to generate a compatible model."
            )
            return None

        pid   = product["product_id"]
        price = self._live_price(pid, product.get("price") or 0)
        if not price or price <= 0:
            return None
        if price < MIN_PRICE:
            logger.debug(f"CNN skip {pid}: price ${price:.8f} < MIN_PRICE ${MIN_PRICE}")
            return None

        # Default all indicator scalars so they're always defined regardless of
        # which branch runs or whether an early return/exception occurs mid-branch.
        cnn_prob = 0.5
        adx_val  = float(config.adx_trend_threshold)
        rsi_val  = 50.0;  macd_h = 0.0;  bb_pos = 0.5
        mfi_val  = 50.0;  stoch_k = 50.0; atr_val = 0.0
        vwap_price = price; vwap_d = 0.0
        fast_rsi_val = vel_norm = vol_z_norm = 0.5
        hurst = 0.5; di = 0.0; entropy = 0.5
        closes: List[float] = []
        ob = {}

        xgb_shadow: Optional[float] = None
        xgb_shadow_v45: Optional[Tuple[float, float, float]] = None
        channels = None  # initialized to None for cache-hit path; MC chain accepts None
        cached = self._cache.get(pid)
        if cached and time.time() - cached[1] < _CACHE_TTL:
            cnn_prob, _, _cached_ind = cached
            adx_val      = _cached_ind.get("adx_val",      adx_val)
            rsi_val      = _cached_ind.get("rsi_val",      rsi_val)
            macd_h       = _cached_ind.get("macd_h",       macd_h)
            mfi_val      = _cached_ind.get("mfi_val",      mfi_val)
            stoch_k      = _cached_ind.get("stoch_k",      stoch_k)
            atr_val      = _cached_ind.get("atr_val",      atr_val)
            vwap_d       = _cached_ind.get("vwap_d",       vwap_d)
            fast_rsi_val = _cached_ind.get("fast_rsi_val", fast_rsi_val)
            vel_norm     = _cached_ind.get("vel_norm",     vel_norm)
            vol_z_norm   = _cached_ind.get("vol_z_norm",   vol_z_norm)
            xgb_shadow   = _cached_ind.get("xgb_shadow",   None)
        else:
            # #98: fetch SEQ_LEN(60) + RV_PREFIX_BARS(60) + 20 buffer = 140
            # so closes_ext can satisfy the rv60 lookback at the first
            # in-window bar; build() pads/truncates each channel to SEQ_LEN.
            candles = await database.get_candles(pid, limit=140)
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
            fast_rsi_val = vel_norm = vol_z_norm = 0.0  # overwritten below if 5m data available
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

            # ── Fetch funding rate (Binance perp) ─────────────────────────────
            funding_rate: Optional[float] = None
            try:
                # Convert Coinbase product_id (e.g. BTC-USD) → Binance symbol (BTCUSDT)
                _bn_sym = pid.replace("-", "").replace("USD", "USDT")
                async with httpx.AsyncClient(timeout=3.0) as _hx:
                    _fr_resp = await _hx.get(
                        "https://fapi.binance.com/fapi/v1/premiumIndex",
                        params={"symbol": _bn_sym},
                    )
                if _fr_resp.status_code == 200:
                    funding_rate = float(_fr_resp.json().get("lastFundingRate", 0) or 0)
            except Exception as _fe:
                logger.debug(f"Funding rate unavailable for {pid}: {_fe}")

            # ── Fetch BTC hourly closes for correlation channel ────────────────
            btc_closes: Optional[List[float]] = None
            if pid != "BTC-USD":
                try:
                    btc_candles = await database.get_candles("BTC-USD", limit=140)
                    if btc_candles:
                        btc_closes = [c["close"] for c in btc_candles]
                except Exception as _be:
                    logger.debug(f"BTC closes unavailable: {_be}")

            # ── Fetch Deribit IV + compute IV/RV spread (BTC/ETH only) ───────
            iv_rv20_spread: Optional[float] = None
            iv_rv60_spread: Optional[float] = None
            try:
                spot = closes[-1] if closes else 0.0
                iv = await get_iv(pid, spot)
                if iv is not None:
                    rv20 = _realized_vol(closes, window=20)
                    rv60 = _realized_vol(closes, window=60)
                    spreads = compute_iv_rv_spreads(iv, rv20, rv60)
                    iv_rv20_spread = spreads["iv_rv20_spread"]
                    iv_rv60_spread = spreads["iv_rv60_spread"]
            except Exception as _ive:
                logger.debug(f"IV/RV spread unavailable for {pid}: {_ive}")

            # ── Fetch top-trader L/S sentiment (Binance futures) ────────────
            ls_sentiment: Optional[float] = None
            try:
                ls_sentiment = await get_ls_sentiment(pid)
            except Exception as _lse:
                logger.debug(f"L/S sentiment unavailable for {pid}: {_lse}")

            # #98: pass closes_ext so rv20/rv60 channels see prior history
            # — without it _rv_series returns 0.0 for the first `window` bars.
            # `candles` (limit=80) gives 80-len closes; the in-window slice is
            # the last SEQ_LEN of that, and closes_ext is the full series so
            # the rv60 lookback at the first in-window bar is satisfied.
            channels = self.fb.build(
                candles, ob, candles_5m=candles_5m,
                btc_closes=btc_closes, funding_rate=funding_rate,
                iv_rv20_spread=iv_rv20_spread, iv_rv60_spread=iv_rv60_spread,
                ls_sentiment=ls_sentiment,
                closes_ext=closes,
            )
            cnn_prob = self._cnn_prob(channels, pid=pid)

            # Always log v3 + v4.5 shadow probabilities, regardless of driver
            # (#181 + 2026-05-23 CNN deprecation). xgb_prob_shadow_v4_5 has
            # built-in isolation: v4.5 failure -> v45=None, never affects v3.
            try:
                from agents import xgb_signal as _xgb
                xgb_shadow, xgb_shadow_v45 = _xgb.xgb_prob_shadow_v4_5(
                    _mask_training_constant_channels(channels),
                    pid=pid,
                )
            except Exception:
                xgb_shadow     = None
                xgb_shadow_v45 = None

            # Task #46 B2: cache v4.5 p_down on the held position (if any) so
            # both _check_risk_exits + exit_watcher.on_price_tick can fire
            # MODEL_DOWN exits without re-running inference. Skip if no
            # position or no v4.5 result (failure isolation per invariant #17).
            # Task #80: also stamp ts_ms so readers can detect stale caches.
            if xgb_shadow_v45 is not None and pid in self.book.positions:
                self.book.positions[pid]["p_down"] = float(xgb_shadow_v45[0])
                self.book.positions[pid]["p_down_ts_ms"] = int(time.time() * 1000)

            rsi_val            = _rsi(closes)
            _, _, macd_h       = _macd(closes)
            _, _, _, bb_pos    = _bollinger(closes)
            adx_val, _, _      = _adx(highs, lows, closes)
            mfi_val            = _mfi(highs, lows, closes, volumes)
            stoch_k, _         = _stoch_rsi(closes)
            atr_val            = _atr(highs, lows, closes)
            vwap_price, vwap_d = _vwap(highs, lows, closes, volumes)
            # Hurst / DI / entropy computations dropped #311-refactor-f —
            # their only consumers were the CNN-backend-only suppression gates
            # and the Ollama-skip decision tree, both now removed.

            self._cache[pid] = (cnn_prob, time.time(), {
                "adx_val": adx_val, "rsi_val": rsi_val, "macd_h": macd_h,
                "mfi_val": mfi_val, "stoch_k": stoch_k, "atr_val": atr_val,
                "vwap_d": vwap_d, "fast_rsi_val": fast_rsi_val,
                "vel_norm": vel_norm, "vol_z_norm": vol_z_norm,
                "xgb_shadow": xgb_shadow,
            })

        # ── HMM regime detection ─────────────────────────────────────────────
        # Regime label is still persisted to cnn_scans for downstream analysis;
        # the historical CNN/LLM blend weights are kept here for DB column
        # back-compat even though the Ollama blend was deleted #311-refactor-f
        # and llm_prob is now always None.
        hmm_regime, hmm_conf, _hmm_state = get_detector().predict(closes)
        if hmm_regime == "UNKNOWN":
            # Fallback to binary ADX gate when HMM not fitted yet
            trending   = adx_val >= config.adx_trend_threshold
            hmm_regime = "TRENDING" if trending else "RANGING"
            hmm_conf   = 1.0
        else:
            trending = hmm_regime == "TRENDING"
        cnn_w, llm_w = regime_blend(hmm_regime, hmm_conf)

        vwap_pct_delta = ((price - vwap_price) / vwap_price * 100) if vwap_price else 0.0
        vwap_side = "above" if vwap_pct_delta > 0 else "below"
        context = (
            f"Price: ${price:,.4f} | Regime: {hmm_regime} (conf={hmm_conf:.2f})\n"
            f"ADX(14): {adx_val:.1f} | RSI(14): {rsi_val:.1f} | MFI(14): {mfi_val:.1f}\n"
            f"MACD hist: {macd_h:+.6f} | Bollinger: {bb_pos:.2f} | StochRSI K: {stoch_k:.1f}\n"
            f"VWAP(20): ${vwap_price:,.4f} | Price {vwap_side} VWAP by {abs(vwap_pct_delta):.2f}%\n"
            f"OB imbalance: {ob.get('imbalance', 0):+.2f} | ATR(14): {atr_val:.4f}\n"
            f"CNN raw: {cnn_prob:.4f}"
        )

        # Ollama LLM blend deleted #311-refactor-f — llm_prob is permanently
        # None, so model_prob equals the backend's own probability.
        llm_prob: Optional[float] = None
        model_prob = max(0.01, min(0.99, cnn_prob))

        # ── Signal direction ──────────────────────────────────────────────────
        if config.model_backend == "xgb_v45":
            # v4.5 3-class driver — indep_thresholds rule on (p_down, p_neutral, p_up).
            if xgb_shadow_v45 is None:
                # v4.5 inference failed — HOLD to avoid trading on garbage signal.
                side, strength = "HOLD", 0.0
            else:
                p_down, p_neutral, p_up = xgb_shadow_v45
                side, strength = _indep_thresholds_decision(
                    p_down, p_neutral, p_up,
                    thresh_up   = config.xgb_v45_thresh_up,
                    thresh_down = config.xgb_v45_thresh_down,
                )
        else:
            # MODEL_BACKEND=xgb — existing 2-class gate on v3 prob.
            if model_prob > config.cnn_buy_threshold:
                side     = "BUY"
                strength = round((model_prob - 0.5) * 2, 3)
            elif model_prob < config.cnn_sell_threshold:
                side     = "SELL"
                strength = round((0.5 - model_prob) * 2, 3)
            else:
                side     = "HOLD"
                strength = 0.0

        # MC filter chain (off by default; MC_FILTERS env-gated). Returns the
        # side unchanged and {} telemetry when MC_FILTERS is empty.
        from agents.mc import registry as _mc
        try:
            from agents.mc import ci_filter as _ci_filter  # registers ci into _FILTER_CLASSES
        except Exception:
            pass
        side, mc_telemetry = _mc.apply_buy_filters(
            side=side, model_prob=model_prob, pid=pid,
            channels=channels, context={"strength": strength},
        )
        if side == "HOLD":
            strength = 0.0  # re-zero if MC down-graded BUY to HOLD

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
            "regime":      hmm_regime,
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
            "xgb_prob":    round(xgb_shadow, 4) if xgb_shadow is not None else None,
            "xgb_prob_v4_5_down":    round(xgb_shadow_v45[0], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_v4_5_neutral": round(xgb_shadow_v45[1], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_v4_5_up":      round(xgb_shadow_v45[2], 4) if xgb_shadow_v45 is not None else None,
            "xgb_prob_stdev": mc_telemetry.get("ci", {}).get("stdev") if mc_telemetry else None,
            "mc_telemetry":   json.dumps(mc_telemetry) if mc_telemetry else None,
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
                "regime":   hmm_regime,
                "rsi":      round(rsi_val, 1),
            },
        )

        quote_size = min(strength * config.max_position_usd, config.max_position_usd)

        # Ollama blend deleted #311-refactor-f — llm_prob is always None.
        prob_line = f"\ncnn_prob={cnn_prob:.4f} llm_prob=None model_prob={model_prob:.4f}"

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

        _lbl = _backend_label()
        logger.info(
            f"{_lbl} [{side}] {pid} | {_lbl.lower()}={cnn_prob:.2%} "
            f"model={model_prob:.2%} adx={adx_val:.1f}({'trend' if trending else 'range'}) "
            f"strength={strength:.2f} size=${quote_size:.2f}"
        )

        # ── Dry-run execution via _CNNBook (tracks positions + writes to trades table) ──
        # Hurst / regime / LGBMFilter suppression gates deleted #311-refactor-f
        # — all three were CNN-backend-only (Phase-1 CNN-tuned), already
        # bypassed under the live xgb backend. BUYs now go straight to Kelly
        # sizing + book.buy.
        if execute:
            if side == "BUY" and not self.book.has_position(pid):
                # Kelly Criterion sizing: use model_prob as win probability.
                # strength = (model_prob - 0.5)*2 is NOT a probability — passing
                # it to Kelly gave frac=0 for all signals below model_prob=0.75.
                frac = min(_kelly_fraction(model_prob), _CNN_MAX_FRAC)
                spent, _ = await self.book.buy(pid, price, frac, trigger="SCAN")
                if spent > 0:
                    self.signals_executed += 1
                    signal["execution"] = {"success": True, "spent": round(spent, 2)}
                    logger.info(
                        f"{_backend_label()} BOOK BUY {pid} @{price:.4f} strength={strength:.2f} "
                        f"kelly_frac={frac:.2f} spent=${spent:.2f} "
                        f"balance=${self.book.balance:.2f}"
                    )
                else:
                    signal["execution"] = {"success": False, "reason": "Insufficient balance"}
                    logger.warning(
                        f"{_backend_label()} BOOK BUY skipped {pid}: balance=${self.book.balance:.2f} "
                        f"too low for kelly_frac={frac:.2f}"
                    )
            elif side == "SELL" and self.book.has_position(pid):
                pnl = await self.book.sell(pid, price, trigger="SCAN")
                self.signals_executed += 1
                signal["execution"] = {"success": True, "pnl": round(pnl, 4)}
                logger.info(
                    f"{_backend_label()} BOOK SELL {pid} @{price:.4f} pnl=${pnl:+.2f} "
                    f"balance=${self.book.balance:.2f}"
                )
            elif side == "BUY" and self.book.has_position(pid):
                signal["execution"] = {"success": False, "reason": "Already holding position"}
            else:
                signal["execution"] = {"success": False, "reason": "No position to sell"}

            # Live order execution (only when order_executor provided and not dry-run)
            if order_executor and not order_executor.dry_run and quote_size >= 1.0:
                result = await self._execute_live_order(order_executor, signal)
                signal["live_execution"] = result

        return signal

    async def _execute_live_order(self, order_executor, signal: Dict) -> Dict:
        """Route a live order through the maker (post-only) or taker path.

        Under ``config.use_maker_execution`` the entry is posted as a maker
        LIMIT (cheaper fee; fills are not guaranteed — ``execute_maker_signal``
        owns the market fallback). The best bid/ask are sourced from the live
        book and attached to ``signal`` only on this path, so the default taker
        path stays byte-for-byte unchanged.
        """
        if config.use_maker_execution:
            quotes = await coinbase_client.get_best_bid_ask([signal["product_id"]])
            quote  = quotes.get(signal["product_id"], {})
            signal["bid"] = quote.get("bid") or 0.0
            signal["ask"] = quote.get("ask") or 0.0
            return await order_executor.execute_maker_signal(signal)
        return await order_executor.execute_signal(signal)

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
                       is_trading_fn=None,
                       broadcast_fn=None,
                       auto_train_fn=None) -> None:
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
                self.last_scan_at = time.time()
                self.next_scan_at = time.time() + interval

                should_execute = is_trading_fn() if is_trading_fn else False

                # Primary exit: CNN's own SCAN-SELL fires first via scan_all
                # so the model can close positions before risk fallbacks
                # pre-empt it. Live data showed risk-first ordering caused
                # TRAIL_STOP/STOP_LOSS to lose -$92 net while SCAN exits
                # earned +$59 on the same window.
                await self.scan_all(
                    execute        = should_execute,
                    order_executor = order_executor if should_execute else None,
                )

                # Secondary/tertiary fallbacks (TRAIL_STOP → STOP_LOSS →
                # MAX_HOLD) run every loop regardless of is_trading gate —
                # stops must fire even when scanning is paused — but only
                # close positions CNN did not already exit via SCAN above.
                await self._check_risk_exits(order_executor)
                self.scan_count  += 1
                self.next_scan_at = time.time() + interval

                # Push fresh state to all connected browser clients immediately
                if broadcast_fn:
                    try:
                        await broadcast_fn()
                    except Exception:
                        pass

                # LightGBM-retrain hook + auto-train scheduler hook deleted
                # #311-refactor-f. The lgbm filter was CNN-tuned; auto-train
                # was a no-op under MODEL_BACKEND=xgb (#300 gated it off).
                # The train_every_n_scans and auto_train_fn parameters on
                # run_loop are kept for caller compat (main.py still passes
                # them); they are now ignored here. Auto-train infrastructure
                # cleanup (subprocess wiring, train_worker.py) is module 4c.

            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"CNN loop error: {e}")
            await asyncio.sleep(interval)

    async def train_on_history(self, epochs: int = 50) -> Dict:  # noqa: C901
        """
        Train the CNN on historical candle data with an 80/20 time-based train/val split.

        Labels are true 4-hour forward returns: 1.0 if close[t+4] > close[t], else 0.0.
        For each product, a sliding window of SEQ_LEN candles is stepped across all
        available history, generating multiple samples per product.

        Returns per-epoch train_loss and val_loss so the caller can detect:
          - Underfitting : both losses stay high (> 0.65) after training
          - Overfitting  : val_loss > train_loss by more than 20% in the final epoch
        Samples are ordered chronologically — the last 20% (most recent) become
        validation so there is no lookahead bias.
        """
        if not _TORCH or not self.model:
            return {"error": "PyTorch not available"}
        import torch.optim as optim
        import time as _t

        _FORWARD_HOURS = 4   # predict whether close is higher 4 hours ahead
        _train_start   = _t.time()

        loop     = asyncio.get_event_loop()
        products = await database.get_products()

        # ── Phase 1: collect candles (async DB calls stay on event loop) ──────
        # load_history() is sync parquet I/O — run each in the thread pool so
        # the event loop stays free to serve /api/cnn/train/status polls.
        _phase1_start = _t.time()
        logger.info(f"CNN training phase 1/3: loading candles for {len(products)} products")

        # Load BTC hourly history once — used to build a per-product aligned
        # series for Ch 21 (BTC return correlation). #57 stage (a).
        btc_hist_for_align = await loop.run_in_executor(None, load_history, "BTC-USD")
        if not btc_hist_for_align:
            btc_hist_for_align = await database.get_candles("BTC-USD", limit=500)
            for c in btc_hist_for_align or []:
                if "start_time" in c and "start" not in c:
                    c["start"] = c["start_time"]

        # [(pid, candles, btc_aligned, candles_5m, funding_aligned, oi_aligned)]
        all_candle_sets: list = []
        for p in products:
            pid  = p["product_id"]
            hist = await loop.run_in_executor(None, load_history, pid)
            if hist:
                live = await database.get_candles(pid, limit=200)
                for c in live:
                    if "start_time" in c and "start" not in c:
                        c["start"] = c["start_time"]
                live_starts = {c["start"] for c in hist}
                new_live    = [c for c in live if c["start"] not in live_starts]
                candles     = sorted(hist + new_live, key=lambda c: c["start"])
                logger.debug(
                    f"Training {pid}: {len(hist)} parquet + {len(new_live)} "
                    f"live = {len(candles)} bars"
                )
            else:
                candles = await database.get_candles(pid, limit=200)
                for c in candles:
                    if "start_time" in c and "start" not in c:
                        c["start"] = c["start_time"]

            if len(candles) >= SEQ_LEN + _FORWARD_HOURS + 1:
                # Per-product 5m parquet (#55, #56) — empty list when product
                # has no 5m backfill yet; _build_samples_range falls back to
                # synthetic hourly proxy in that case.
                c5m = await loop.run_in_executor(None, load_5m_history, pid) or []
                btc_aligned = (
                    _aligned_btc_closes(candles, btc_hist_for_align)
                    if pid != "BTC-USD" else None
                )
                # Per-product Binance funding history (#57 stage b). The
                # fetcher returns [] for unsupported symbols, fetch errors,
                # or non-200 responses — `_aligned_funding_rates` then
                # returns None and Ch 20 stays at zero (caller behaviour
                # matches the prior constant-zero state).
                fr_start_ms = int(candles[0]["start"]) * 1000
                fr_end_ms   = int(candles[-1]["start"]) * 1000
                funding_hist = await fetch_funding_history(
                    pid, fr_start_ms, fr_end_ms
                )
                funding_aligned = _aligned_funding_rates(candles, funding_hist)
                # Per-product OKX OI history (#143-A). Same forward-fill
                # alignment as funding; `_aligned_oi_history` returns None
                # when the fetcher yields no data, leaving Ch 27 at zero.
                oi_hist = await fetch_oi_history(
                    pid, fr_start_ms, fr_end_ms
                )
                oi_aligned = _aligned_oi_history(candles, oi_hist)
                all_candle_sets.append(
                    (pid, candles, btc_aligned, c5m, funding_aligned, oi_aligned)
                )

        _phase1_secs = _t.time() - _phase1_start
        logger.info(
            f"CNN training phase 1/3 done: {len(all_candle_sets)} products with data "
            f"in {_phase1_secs:.1f}s"
        )

        # ── Phase 2: build tensors in thread pool ─────────────────────────────
        # Sliding-window feature extraction + tensor stacking + GPU transfer are
        # all CPU/GPU-bound and must NOT run on the event loop.
        _phase2_start = _t.time()
        logger.info(f"CNN training phase 2/3: building feature tensors (sliding window)")
        fb = self.fb  # capture FeatureBuilder for closure

        _LABEL_THRESH = 0.003   # require ≥0.3% move to avoid noisy near-zero labels

        def _build_dataset():
            # Per-product append-only cache (P2). Schema change on any of
            # seq_len / forward_hours / label_thresh / n_channels invalidates
            # everything; otherwise only new candles per product cost work.
            schema = _dataset_schema(SEQ_LEN, _FORWARD_HOURS, _LABEL_THRESH, N_CHANNELS)
            cache  = _load_pp_cache(_DATASET_CACHE_PATH, schema) or {}
            n_products = len(all_candle_sets)
            if cache:
                logger.info(
                    f"CNN dataset per-product cache loaded: "
                    f"{len(cache)} products in cache, {n_products} to process"
                )
            else:
                logger.info(
                    f"CNN dataset per-product cache empty — building from scratch"
                )
            X_list, y_list, w_list = [], [], []
            hits = appends = rebuilds = skips = 0
            for prod_idx, (pid, candles, btc_aligned, c5m, funding_aligned, oi_aligned) in enumerate(all_candle_sets, 1):
                entry, status = _extend_or_rebuild_product(
                    cache.get(pid), candles, fb,
                    SEQ_LEN, _FORWARD_HOURS, _LABEL_THRESH,
                    btc_closes=btc_aligned,
                    funding_rates=funding_aligned,
                    oi_rates=oi_aligned,
                    candles_5m=c5m,
                )
                if status == "skip":
                    skips += 1
                elif status == "hit":
                    hits += 1
                elif status == "append":
                    appends += 1
                else:
                    rebuilds += 1
                if entry is None:
                    continue
                cache[pid] = entry
                X_list.extend(entry["X"])
                y_list.extend(entry["y"])
                # Per-product uniqueness weight = mean(1/N_t) over forward
                # window (P3c). Compute fresh each dataset assembly so cache
                # updates don't drift.
                w_list.extend(_compute_uniqueness(
                    entry.get("indices", []), _FORWARD_HOURS, int(entry["last_n"])
                ))
                if prod_idx % _PHASE2_LOG_EVERY == 0 or prod_idx == n_products:
                    logger.info(
                        f"CNN dataset build: {prod_idx}/{n_products} products, "
                        f"{len(X_list):,} samples so far "
                        f"(hits={hits}, appends={appends}, rebuilds={rebuilds})"
                    )
            try:
                _save_pp_cache(_DATASET_CACHE_PATH, schema, cache)
                logger.info(
                    f"CNN dataset cached to disk: {len(X_list):,} samples across "
                    f"{len(cache)} products "
                    f"(hits={hits}, appends={appends}, rebuilds={rebuilds}, "
                    f"skipped={skips})"
                )
            except Exception as _se:
                logger.warning(f"CNN dataset cache save failed (non-fatal): {_se}")
            return X_list, y_list, w_list

        X_list, y_list, w_list = await loop.run_in_executor(None, _build_dataset)
        _phase2_secs = _t.time() - _phase2_start
        logger.info(
            f"CNN training phase 2/3 done: {len(X_list):,} samples built "
            f"in {_phase2_secs:.1f}s ({_phase2_secs/60:.1f} min)"
        )

        if len(X_list) < 4:
            return {"error": f"Not enough data ({len(X_list)} samples, need ≥ 4)"}

        # ── Class balance check ───────────────────────────────────────────────
        n_pos = sum(1 for y in y_list if y == 1.0)
        n_neg = len(y_list) - n_pos
        if n_pos == 0 or n_neg == 0:
            # Single-class dataset — weighting is undefined; fall back to balanced
            logger.warning(
                f"CNN dataset: only one class present (up={n_pos}, down={n_neg}) — "
                "disabling class weighting (pos_weight=1.0)"
            )
            pos_weight_val = 1.0
        else:
            pos_weight_val = n_neg / n_pos   # >1 = more negatives (up-weight positives)
        logger.info(
            f"CNN dataset: {len(y_list)} samples | up={n_pos} ({100*n_pos/len(y_list):.1f}%) "
            f"down={n_neg} ({100*n_neg/len(y_list):.1f}%) | pos_weight={pos_weight_val:.3f}"
        )

        # ── 80/20 time-ordered split (no shuffle — preserves temporal order) ──
        split      = max(1, int(len(X_list) * 0.8))
        X_all      = torch.stack(X_list).to(_DEVICE)
        # #99: zero training-constant channels before fit so train and
        # inference both see zeros for the same channels. Without this,
        # ch 17 (=0.5 in cache) trains as 0.5 but inference zeros it.
        X_all      = _zero_mask_channels(X_all)
        y_all      = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)
        # Per-sample uniqueness weight (P3c). Fallback to 1.0 if upstream
        # couldn't produce a weight list (e.g. legacy code path).
        if not w_list or len(w_list) != len(X_list):
            w_list = [1.0] * len(X_list)
        w_all      = torch.tensor(w_list, dtype=torch.float32).unsqueeze(1).to(_DEVICE)
        X_train, X_val = X_all[:split], X_all[split:]
        y_train, y_val = y_all[:split], y_all[split:]
        w_train, w_val = w_all[:split], w_all[split:]

        model      = self.model  # capture for thread closure
        epoch_log: list = []    # [{epoch, train_loss, val_loss}]
        fit_box:   list = [{}]  # mutable container: {best_val_loss, stopped_epoch}
        _pos_w     = torch.tensor([pos_weight_val], dtype=torch.float32).to(_DEVICE)

        def _sync_fit() -> None:
            """Run the blocking PyTorch fit loop in a thread pool executor.

            - Batch size scales with dataset so gradient quality stays consistent
            - ReduceLROnPlateau adapts to actual learning, not a fixed schedule
            - Early stopping (patience=_EARLY_STOP_PATIENCE) prevents overtraining
            - Best-model restore rolls weights back to lowest val_loss epoch
            - Checkpoint saved every _CKPT_EVERY epochs for crash recovery
            """
            n_train = X_train.shape[0]

            # Smaller batches → more gradient noise → better generalization on
            # noisy financial signal (best practices: start at 32, scale with data)
            if n_train >= 200_000:
                BATCH = 256
            elif n_train >= 50_000:
                BATCH = 128
            else:
                BATCH = 64

            # Scale initial LR with batch size (linear scaling rule, anchor = 64)
            base_lr   = 1e-3 * (BATCH / 64) ** 0.5
            patience  = _EARLY_STOP_PATIENCE
            opt       = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=5, min_lr=1e-6,
            )

            best_val   = float("inf")
            best_sd    = None
            no_improve = 0
            start_ep   = 1

            # Resume from checkpoint if one exists from a previous interrupted run
            if os.path.exists(_CKPT_PATH):
                try:
                    ckpt = torch.load(_CKPT_PATH, map_location=_DEVICE, weights_only=False)
                    model.load_state_dict(ckpt["model_sd"])
                    opt.load_state_dict(ckpt["opt_sd"])
                    best_val   = ckpt.get("best_val", float("inf"))
                    best_sd    = ckpt.get("best_sd")
                    no_improve = ckpt.get("no_improve", 0)
                    start_ep   = ckpt.get("epoch", 0) + 1
                    epoch_log.extend(ckpt.get("epoch_log", []))
                    logger.info(
                        f"CNN fit resuming from checkpoint at epoch {start_ep} "
                        f"(best_val={best_val:.4f})"
                    )
                except Exception as _ce:
                    logger.warning(f"CNN checkpoint load failed, starting fresh: {_ce}")
                    start_ep = 1

            logger.info(
                f"CNN fit started: n_train={n_train} batch={BATCH} "
                f"lr={base_lr:.2e} epochs={epochs} start_ep={start_ep}"
            )

            model.train()
            for ep in range(start_ep, epochs + 1):
                # ── Mini-batch training ───────────────────────────────────────
                perm = torch.randperm(n_train)
                batch_losses = []
                for start in range(0, n_train, BATCH):
                    idx = perm[start: start + BATCH]
                    opt.zero_grad()
                    # Per-sample BCE with uniqueness weights (P3c) and label
                    # smoothing (P3d). reduction='none' gives a [B,1] tensor;
                    # smooth the training targets so BCE stops pushing logits
                    # to ±∞, then take the uniqueness-weighted mean.
                    per_sample = F.binary_cross_entropy_with_logits(
                        model(X_train[idx]),
                        _smooth_labels(y_train[idx], _LABEL_SMOOTH),
                        pos_weight=_pos_w, reduction="none",
                    )
                    wi   = w_train[idx]
                    loss = (per_sample * wi).sum() / wi.sum().clamp(min=1e-9)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()
                    batch_losses.append(float(loss.item()))
                train_loss_val = sum(batch_losses) / len(batch_losses)

                # ── Validation ────────────────────────────────────────────────
                model.eval()
                with torch.no_grad():
                    val_per_sample = F.binary_cross_entropy_with_logits(
                        model(X_val), y_val, pos_weight=_pos_w, reduction="none",
                    )
                    vl_t = float(
                        (val_per_sample * w_val).sum()
                        / w_val.sum().clamp(min=1e-9)
                    )
                model.train()

                scheduler.step(vl_t)   # ReduceLROnPlateau monitors val_loss

                tl = round(train_loss_val, 6)
                vl = round(vl_t, 6)
                current_lr = opt.param_groups[0]["lr"]
                epoch_log.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "lr": current_lr})
                logger.debug(
                    f"CNN train epoch {ep}/{epochs} | "
                    f"train={tl:.4f} val={vl:.4f} lr={current_lr:.2e}"
                )
                if ep == start_ep or ep % _HEARTBEAT_EVERY == 0:
                    logger.info(
                        f"CNN train heartbeat epoch {ep}/{epochs} | "
                        f"train={tl:.4f} val={vl:.4f} lr={current_lr:.2e} "
                        f"best_val={best_val:.4f}"
                    )

                # ── Best-model tracking ───────────────────────────────────────
                if vl < best_val:
                    best_val   = vl
                    best_sd    = {k: v.cpu() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        logger.info(
                            f"CNN early stop at epoch {ep}/{epochs} "
                            f"(no val improvement for {patience} epochs)"
                        )
                        break

                # ── Crash-recovery checkpoint every N epochs ──────────────────
                if ep % _CKPT_EVERY == 0:
                    try:
                        torch.save({
                            "epoch":      ep,
                            "model_sd":   {k: v.cpu() for k, v in model.state_dict().items()},
                            "opt_sd":     opt.state_dict(),
                            "best_val":   best_val,
                            "best_sd":    best_sd,
                            "no_improve": no_improve,
                            "epoch_log":  epoch_log,
                        }, _CKPT_PATH)
                    except Exception as _sce:
                        logger.warning(f"CNN checkpoint save failed: {_sce}")

            # Restore best weights and ensure model stays on the right device
            if best_sd is not None:
                model.load_state_dict(best_sd)
                model.to(_DEVICE)

            # Clean up resume checkpoint — training finished normally
            try:
                if os.path.exists(_CKPT_PATH):
                    os.remove(_CKPT_PATH)
            except Exception:
                pass

            fit_box[0] = {"best_val_loss": best_val, "stopped_epoch": len(epoch_log)}

        # Offload blocking compute to thread pool — keeps event loop free
        await loop.run_in_executor(None, _sync_fit)

        best_val_loss  = fit_box[0].get("best_val_loss", float("inf"))
        stopped_epoch  = fit_box[0].get("stopped_epoch", epochs)

        final      = epoch_log[-1]
        initial_tl = epoch_log[0]["train_loss"]
        final_tl   = final["train_loss"]
        final_vl   = final["val_loss"]

        # ── Validation probabilities (used for AUC, precision, recall) ───────
        val_auc = None
        val_precision_at_thresh = None
        val_recall_at_thresh    = None
        val_threshold = float(config.cnn_buy_threshold)
        _probs_list = None
        _labels_list = None
        try:
            model.eval()
            with torch.no_grad():
                _probs_list = torch.sigmoid(model(X_val)).cpu().numpy().flatten().tolist()
            _labels_list = y_val.cpu().numpy().flatten().tolist()
        except Exception as _pe:
            logger.debug(f"val-set probability pass skipped: {_pe}")

        # ── AUC-ROC on validation set ─────────────────────────────────────────
        if _probs_list is not None and _labels_list is not None:
            try:
                from sklearn.metrics import roc_auc_score as _auc
                val_auc = round(float(_auc(_labels_list, _probs_list)), 4)
                logger.info(f"CNN val AUC-ROC: {val_auc:.4f}")
            except Exception as _ae:
                logger.debug(f"AUC-ROC skipped: {_ae}")

        # ── Precision/Recall at production BUY threshold ──────────────────────
        # Mirrors `model_prob > config.cnn_buy_threshold` gate in _fire_signal.
        # Persisted so gate-choice alternatives can be compared empirically over
        # multiple training runs before any change to the save gate itself.
        if _probs_list is not None and _labels_list is not None:
            try:
                _p, _r = _precision_recall_at_threshold(
                    _probs_list, _labels_list, val_threshold,
                )
                val_precision_at_thresh = round(_p, 4) if _p is not None else None
                val_recall_at_thresh    = round(_r, 4) if _r is not None else None
                logger.info(
                    f"CNN val P/R @ {val_threshold:.2f}: "
                    f"precision={val_precision_at_thresh} "
                    f"recall={val_recall_at_thresh}"
                )
            except Exception as _pre:
                logger.debug(f"precision/recall skipped: {_pre}")

        # ── Fit diagnosis ─────────────────────────────────────────────────────
        overfit_gap = (final_vl - final_tl) / max(final_tl, 1e-9)
        loss_drop   = initial_tl - final_tl
        if final_tl > 0.65 and loss_drop < 0.02:
            fit_status = "UNDERFIT"
            fit_advice = "Loss barely moved — try more epochs or more training data"
        elif overfit_gap > 0.20:
            fit_status = "OVERFIT"
            fit_advice = (
                f"Val loss {final_vl:.4f} is {overfit_gap*100:.0f}% above train loss "
                f"{final_tl:.4f} — reduce epochs or collect more diverse candles"
            )
        else:
            fit_status = "OK"
            fit_advice = "Train and val loss are close — model is generalising well"

        # ── Conditional save: only overwrite when new model is better ─────────
        prev_best = self._read_best_loss()
        if best_val_loss < prev_best:
            self.save_model(backup=True)   # backs up existing .pt before overwriting
            self._write_best_loss(best_val_loss)
            self._needs_retrain = False
            save_note = f"saved (val {prev_best:.4f} → {best_val_loss:.4f})"
        else:
            save_note = (
                f"NOT saved — new val_loss {best_val_loss:.4f} >= "
                f"previous best {prev_best:.4f}; keeping prior checkpoint"
            )
            fit_status = fit_status if fit_status != "OK" else "REJECTED"
            fit_advice = save_note

        logger.info(
            f"CNN training complete | {len(X_train)} train / {len(X_val)} val samples | "
            f"train {initial_tl:.4f}→{final_tl:.4f} | val best={best_val_loss:.4f} | "
            f"stopped_epoch={stopped_epoch}/{epochs} | fit={fit_status} | {save_note}"
        )

        # ── Fit HMM regime detector on BTC history ───────────────────────────
        try:
            btc_hist = load_history("BTC-USD") or await database.get_candles("BTC-USD", limit=500)
            if btc_hist:
                btc_closes_all = [c["close"] for c in btc_hist]
                get_detector().fit(btc_closes_all)
        except Exception as _he:
            logger.warning(f"HMM fit during training failed: {_he}")

        # True if the model file was written during this training run
        _active_model_path = _model_path_for()
        _model_saved = os.path.exists(_active_model_path) and os.path.getmtime(_active_model_path) > _train_start

        result = {
            "epochs":           epochs,
            "stopped_epoch":    stopped_epoch,
            "samples":          len(X_list),
            "train_samples":    len(X_train),
            "val_samples":      len(X_val),
            "label_pos_pct":    round(100 * n_pos / max(len(y_list), 1), 1),
            "channels":         N_CHANNELS,
            "arch":             getattr(self.model, "arch", "glu2"),
            "initial_loss":     initial_tl,
            "final_train_loss": final_tl,
            "final_val_loss":   final_vl,
            "best_val_loss":    round(best_val_loss, 6),
            "val_auc":          val_auc,
            "val_precision_at_thresh": val_precision_at_thresh,
            "val_recall_at_thresh":    val_recall_at_thresh,
            "val_threshold":           val_threshold,
            "overfit_gap_pct":  round(overfit_gap * 100, 1),
            "fit_status":       fit_status,
            "fit_advice":       fit_advice,
            "duration_secs":    int(_t.time() - _train_start),
            "phase1_secs":      int(_phase1_secs),
            "phase2_secs":      int(_phase2_secs),
            "phase3_secs":      int(_t.time() - _train_start - _phase1_secs - _phase2_secs),
            "saved":            _model_saved,   # True = model file overwritten this run
            "epoch_log":        epoch_log,
        }

        try:
            await database.save_training_session(result)
        except Exception as _dbe:
            logger.warning(f"Failed to save training session to DB: {_dbe}")

        return result
