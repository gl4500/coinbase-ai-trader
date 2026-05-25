"""Dynamic-exit PnL labeling for the strategy-discovery rebuild (Phase 2).

For each candidate row (pid, t) and each horizon h, simulate the deployed
exit policy and report net PnL fraction after fees:

  - stop-loss at 8% drawdown vs entry (priority)
  - trail-stop at max(atr14_pct_t, 6%) drawdown vs running peak
  - max-hold cap at 168 bars (7d)
  - 1.2% retail round-trip fee subtracted from gross PnL

Mirrors agents/exit_watcher.on_price_tick and cnn_agent._check_risk_exits.
Pure functions. No I/O.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_DEFAULT_HORIZONS: Tuple[int, ...] = (1, 4, 24, 72, 168)
_DEFAULT_STOP_LOSS_PCT  = 0.08
_DEFAULT_ATR_TRAIL_FLOOR = 0.06
_DEFAULT_MAX_HOLD_BARS  = 168
_DEFAULT_ROUND_TRIP_FEE = 0.012


def _simulate_one(
    entry_idx: int,
    horizon: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_pcts: np.ndarray,
    stop_loss_pct: float,
    atr_trail_floor: float,
    max_hold_bars: int,
    round_trip_fee: float,
) -> float:
    """Simulate one (entry, horizon) trade. Returns net PnL fraction or NaN."""
    n = len(closes)
    entry_price = float(closes[entry_idx])
    horizon_cap = min(horizon, max_hold_bars)
    last_idx = entry_idx + horizon_cap
    if last_idx >= n:
        return float("nan")
    peak = entry_price
    for s in range(1, horizon_cap + 1):
        i = entry_idx + s
        bar_low  = float(lows[i])
        bar_high = float(highs[i])
        # 1. Stop-loss check (priority — matches cnn_agent._check_risk_exits)
        if bar_low / entry_price - 1.0 <= -stop_loss_pct:
            exit_price = entry_price * (1.0 - stop_loss_pct)
            return (exit_price / entry_price - 1.0) - round_trip_fee
        # 2. Trail-stop check (ATR-based with floor)
        if bar_high > peak:
            peak = bar_high
        atr_now = float(atr_pcts[i])
        if not np.isfinite(atr_now):
            atr_now = atr_trail_floor
        atr_pct = max(atr_now, atr_trail_floor)
        if bar_low / peak - 1.0 <= -atr_pct:
            exit_price = peak * (1.0 - atr_pct)
            return (exit_price / entry_price - 1.0) - round_trip_fee
    # 3. Horizon reached without trigger — exit at last bar's close
    exit_price = float(closes[last_idx])
    return (exit_price / entry_price - 1.0) - round_trip_fee


def simulate_dynamic_exit_labels(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    stop_loss_pct: float = _DEFAULT_STOP_LOSS_PCT,
    atr_trail_floor: float = _DEFAULT_ATR_TRAIL_FLOOR,
    max_hold_bars: int = _DEFAULT_MAX_HOLD_BARS,
    round_trip_fee: float = _DEFAULT_ROUND_TRIP_FEE,
) -> pd.DataFrame:
    """Add label_h{h} columns per horizon by simulating the deployed exit policy.

    Requires columns: ts, open, high, low, close, atr14_pct. NaN labels when
    horizon would extend past the end of df.
    """
    horizons_list = list(horizons) if horizons is not None else list(_DEFAULT_HORIZONS)
    out = df.copy()
    closes   = out["close"].to_numpy(dtype="float64")
    highs    = out["high"].to_numpy(dtype="float64")
    lows     = out["low"].to_numpy(dtype="float64")
    atr_pcts = out["atr14_pct"].to_numpy(dtype="float64")
    n = len(out)
    for h in horizons_list:
        col = np.empty(n, dtype="float64")
        for i in range(n):
            col[i] = _simulate_one(
                i, h, closes, highs, lows, atr_pcts,
                stop_loss_pct, atr_trail_floor, max_hold_bars, round_trip_fee,
            )
        out[f"label_h{h}"] = col
    return out
