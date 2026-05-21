"""Off-the-clock XGB track: sample building + OOF prediction (SP2).

Builds XGB training samples on either dollar bars (data/history/dollar/) or
1h time bars (data/history/), with two label variants (direction,
triple-barrier) across a horizon sweep, and produces out-of-fold predictions
for the deployment scorecard. See 2026-05-21-offclock-xgb-track-design.md.
"""
from __future__ import annotations

import os

from services.history_backfill import load_history

_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "history")

_TB_BARRIER = 0.01  # +/-1% triple-barrier


def load_dollar_bars(pid: str) -> list[dict]:
    """Load a product's dollar bars from data/history/dollar/<pid>.parquet.

    Returns OHLCV+start bar dicts sorted by start; [] if the file is missing.
    """
    import pyarrow.parquet as pq

    safe = pid.replace("/", "_")
    path = os.path.join(_HISTORY_DIR, "dollar", f"{safe}.parquet")
    if not os.path.exists(path):
        return []
    rows = pq.read_table(path).to_pydict()
    n = len(rows["start"])
    bars = [
        {
            "start": int(rows["start"][i]),
            "open": float(rows["open"][i]),
            "high": float(rows["high"][i]),
            "low": float(rows["low"][i]),
            "close": float(rows["close"][i]),
            "volume": float(rows["volume"][i]),
        }
        for i in range(n)
    ]
    bars.sort(key=lambda b: b["start"])
    return bars


def load_bars(substrate: str, pid: str) -> list[dict]:
    """Load a product's bars for the substrate: 'dollar' or 'time'."""
    if substrate == "dollar":
        return load_dollar_bars(pid)
    if substrate == "time":
        return load_history(pid)
    raise ValueError(f"unknown substrate {substrate!r}; expected 'dollar' or 'time'")


def direction_label(closes, t: int, k: int) -> tuple[int, float]:
    """k-bars-ahead direction label for entry bar t.

    Returns (label, exit_close): label is 1 if close[t+k] > close[t] else 0;
    exit_close is close[t+k]. The caller guarantees t + k < len(closes).
    """
    entry = closes[t]
    exit_close = float(closes[t + k])
    return (1 if exit_close > entry else 0), exit_close


def triple_barrier_label(bars: list[dict], t: int, k: int) -> tuple[int, float]:
    """Triple-barrier label for entry bar t with a k-bar vertical timeout.

    Upper barrier = close[t] * 1.01, lower = close[t] * 0.99. Scans bars
    t+1 .. t+k. Returns (label, exit_close):
      - upper hit first   -> (1, upper)
      - lower hit first   -> (0, lower)
      - both in one bar   -> close-direction breaks the tie (close >= entry -> UP)
      - neither (timeout) -> (1 if close[t+k] > close[t] else 0, close[t+k])
    The caller guarantees t + k < len(bars).
    """
    entry = bars[t]["close"]
    upper = entry * (1.0 + _TB_BARRIER)
    lower = entry * (1.0 - _TB_BARRIER)
    for i in range(t + 1, t + k + 1):
        b = bars[i]
        hit_up = b["high"] >= upper
        hit_dn = b["low"] <= lower
        if hit_up and hit_dn:
            return (1, upper) if b["close"] >= entry else (0, lower)
        if hit_up:
            return 1, upper
        if hit_dn:
            return 0, lower
    exit_close = float(bars[t + k]["close"])
    return (1 if exit_close > entry else 0), exit_close


_MACRO_WINDOW = 336  # macro tier lookback (= TIER_WINDOWS_V4["macro"])


def build_product_samples(
    bars: list[dict],
    label_variant: str,
    k: int,
    sample_step: int,
) -> dict:
    """Build samples for one product's bar list.

    Rolls one sample every `sample_step` bars from index 336 (macro lookback)
    up to len(bars) - k. Each sample: extract_v4 features over the micro/meso/
    macro tiers, a label, and entry/exit close prices.

    Returns a dict of numpy arrays: X (N,150), y (N,), entry_close (N,),
    exit_close (N,), entry_ts (N,). Empty arrays if the product is too short.

    Raises:
        ValueError: if label_variant is not 'direction' or 'triple_barrier'.
    """
    import numpy as np
    from tools.xgb_v4_features import N_FEATURES_V4, extract_v4

    if label_variant not in ("direction", "triple_barrier"):
        raise ValueError(
            f"unknown label_variant {label_variant!r}; "
            "expected 'direction' or 'triple_barrier'"
        )

    empty = {
        "X": np.zeros((0, N_FEATURES_V4), dtype=np.float64),
        "y": np.zeros(0, dtype=np.int64),
        "entry_close": np.zeros(0, dtype=np.float64),
        "exit_close": np.zeros(0, dtype=np.float64),
        "entry_ts": np.zeros(0, dtype=np.int64),
    }
    n = len(bars)
    last_t = n - k
    if last_t <= _MACRO_WINDOW:
        return empty

    closes = [b["close"] for b in bars]
    feats, ys, ec, xc, ts = [], [], [], [], []
    for t in range(_MACRO_WINDOW, last_t, sample_step):
        tier_slices = {
            "micro": bars[t - 60:t],
            "meso": bars[t - 168:t],
            "macro": bars[t - 336:t],
        }
        f, _ = extract_v4(tier_slices)
        if label_variant == "direction":
            label, exit_close = direction_label(closes, t, k)
        else:
            label, exit_close = triple_barrier_label(bars, t, k)
        feats.append(f[0])
        ys.append(label)
        ec.append(float(closes[t]))
        xc.append(exit_close)
        ts.append(int(bars[t]["start"]))

    if not feats:
        return empty
    return {
        "X": np.stack(feats, axis=0),
        "y": np.array(ys, dtype=np.int64),
        "entry_close": np.array(ec, dtype=np.float64),
        "exit_close": np.array(xc, dtype=np.float64),
        "entry_ts": np.array(ts, dtype=np.int64),
    }


def pool_samples(
    substrate: str,
    label_variant: str,
    k: int,
    pids: list[str],
    sample_step: int,
) -> dict:
    """Build and pool samples across products, sorted by entry timestamp.

    Returns the same dict shape as build_product_samples, concatenated over
    all products that yielded at least one sample.

    Raises:
        RuntimeError: if no product yields any sample.
    """
    import numpy as np

    parts = []
    for pid in pids:
        bars = load_bars(substrate, pid)
        s = build_product_samples(bars, label_variant, k, sample_step)
        if len(s["y"]) > 0:
            parts.append(s)

    if not parts:
        raise RuntimeError(
            f"no samples for substrate={substrate!r} label_variant="
            f"{label_variant!r} k={k} — check data/history/ inputs exist"
        )

    pooled = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    order = np.argsort(pooled["entry_ts"], kind="stable")
    return {key: val[order] for key, val in pooled.items()}
