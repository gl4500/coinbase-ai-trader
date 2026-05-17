"""Single-add probe (#156): replace ch13 (obv_slope, the most marginal noise
channel per #146 ablation) with a BTC USD-volume share series, then measure
Δ mean_auc against the +0.01 gate.

Why volume-share, not market-cap dominance:
    Earlier scopings of #156 stalled looking for hourly historical BTC market
    cap from CoinGecko/CMC. Both require paid plans and are exchange-agnostic
    — neither reflects the deployment universe (Coinbase USD pairs).
    Coinbase USD-volume share is locally computable from data we already
    backfill, and it captures the same "how much of activity is in BTC vs
    alts" signal that drives BTC-dominance trading wisdom. It's a faithful
    proxy for our universe.

Decision rule: Δ ≥ +0.01 mean AUC → integrate as a real channel.
Otherwise abandon the BTC-dominance path.

Run:
    cd backend && python tools/btc_dominance_probe.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.channel_replace import run_replace  # noqa: E402
from tools.feature_set_compare import _entry_to_arrays  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_HISTORY_DIR = os.path.join(BACKEND, "data", "history")
_BAR_SECS = 3600
_SEQ_LEN = 60
_TARGET_CHANNEL = 13  # obv_slope — most marginal per #146

# Fixed liquid basket used to compute the dominance series. Decoupled from
# the sample universe (top-N by cache sample count) so the proxy stays
# stable when alts/memes happen to dominate the cache. First-run #156 bug:
# pooled top-20 was all alts, BTC absent → 0% coverage. The basket below is
# the ten highest-liquidity USD-quoted Coinbase pairs by long-term volume.
_DOMINANCE_BASKET: Tuple[str, ...] = (
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD",
    "ADA-USD", "LTC-USD", "AVAX-USD", "LINK-USD", "DOT-USD",
)


# ---------------------------------------------------------------------------
# Pure functions (tested in tests/test_btc_dominance_probe.py)
# ---------------------------------------------------------------------------

def compute_btc_usd_volume_share(
    pid_history: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[int, float]:
    """Per-hour BTC USD-volume share among the supplied pid universe.

    pid_history values are (ts_secs, close, native_volume) parallel arrays.
    USD volume per pid per hour = close * native_volume.
    Share at hour h = btc_usd_vol[h] / sum(usd_vol[h] for pid in universe).
    Output only contains hours where BTC has data; pids missing from a given
    hour contribute 0 to the denominator.
    """
    if "BTC-USD" not in pid_history:
        return {}

    btc_ts, btc_close, btc_vol = pid_history["BTC-USD"]
    btc_usd = {int(t): float(c) * float(v) for t, c, v in zip(btc_ts, btc_close, btc_vol)}
    if not btc_usd:
        return {}

    # Sum total USD volume per hour across the universe.
    totals: Dict[int, float] = {}
    for pid, (ts, close, vol) in pid_history.items():
        for t, c, v in zip(ts, close, vol):
            totals[int(t)] = totals.get(int(t), 0.0) + float(c) * float(v)

    out: Dict[int, float] = {}
    for h, btc_v in btc_usd.items():
        denom = totals.get(h, 0.0)
        if denom > 0.0:
            out[h] = btc_v / denom
    return out


def build_btc_dom_signal(
    sample_end_ts: np.ndarray,
    share_by_hour: Mapping[int, float],
    seq_len: int = _SEQ_LEN,
) -> np.ndarray:
    """Build [N, seq_len] z-scored BTC-dominance signal aligned to per-sample
    bar-end timestamps.

    Forward-fill missing hours from the earliest known share value; bars
    earlier than the first known hour use the mean (yielding z=0 there). The
    full per-bar series is z-scored once over the entire share series so the
    channel has unit variance like the other channels in the cache.
    """
    n = len(sample_end_ts)
    out = np.zeros((n, seq_len), dtype=np.float32)
    if not share_by_hour:
        return out

    sorted_hours = sorted(share_by_hour.keys())
    vals = np.array([share_by_hour[h] for h in sorted_hours], dtype=np.float64)
    mu = float(vals.mean())
    sigma = float(vals.std()) or 1.0

    # Build a forward-fill lookup over the full hour range.
    first_h = sorted_hours[0]
    last_h = sorted_hours[-1]
    filled: Dict[int, float] = {}
    last_val = mu
    for h in range(first_h, last_h + _BAR_SECS, _BAR_SECS):
        if h in share_by_hour:
            last_val = share_by_hour[h]
        filled[h] = last_val

    for i, end_ts in enumerate(sample_end_ts):
        for j in range(seq_len):
            h = int(end_ts) - (seq_len - 1 - j) * _BAR_SECS
            h = (h // _BAR_SECS) * _BAR_SECS
            if h < first_h:
                v = mu                 # pre-history → neutral (z=0)
            else:
                v = filled.get(h, mu)
            out[i, j] = (v - mu) / sigma

    return out


# ---------------------------------------------------------------------------
# IO + probe runner
# ---------------------------------------------------------------------------

def _load_pid_history(
    pid: str,
    history_dir: str = _HISTORY_DIR,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read hourly parquet for `pid` and return (ts_secs, close, native_vol).
    Returns three empty arrays if the file is missing or unreadable.
    """
    path = os.path.join(history_dir, f"{pid}.parquet")
    if not os.path.exists(path):
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(path)
    except Exception:
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64))
    return (
        df["start"].to_numpy(dtype=np.int64),
        df["close"].to_numpy(dtype=np.float64),
        df["volume"].to_numpy(dtype=np.float64),
    )


def build_pid_history_from_basket(
    basket: Iterable[str],
    history_dir: str = _HISTORY_DIR,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load (ts, close, native_vol) for each pid in basket; skip missing files
    silently. Used to assemble the dominance computation universe independent
    of which pids happen to be in the sample cache."""
    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid in basket:
        ts, close, vol = _load_pid_history(pid, history_dir)
        if len(ts) > 0:
            out[pid] = (ts, close, vol)
    return out


def _load_pooled_with_btc_dom(
    n: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load pooled top-N samples + aligned market-wide BTC-dominance signal.

    BTC-dominance is a single market-wide series; the same signal is broadcast
    to every product's samples (unlike OI which is per-product).
    """
    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]

    sized = sorted(
        ((pid, len(e.get("X", []))) for pid, e in prods.items()),
        key=lambda x: -x[1],
    )[:n]
    pids_used = [pid for pid, _ in sized]
    print(f"  pooled top-{n} products: {pids_used}", flush=True)

    print(f"Loading parquet history for dominance basket: {list(_DOMINANCE_BASKET)}",
          flush=True)
    t0 = time.time()
    pid_history = build_pid_history_from_basket(_DOMINANCE_BASKET)
    elapsed = time.time() - t0
    print(f"  loaded {len(pid_history)}/{len(_DOMINANCE_BASKET)} basket parquets "
          f"in {elapsed:.1f}s", flush=True)

    print("Computing per-hour BTC USD-volume share...", flush=True)
    share_by_hour = compute_btc_usd_volume_share(pid_history)
    print(f"  hours covered: {len(share_by_hour):,}", flush=True)
    if share_by_hour:
        vals = np.array(list(share_by_hour.values()))
        print(f"  share stats: mean={vals.mean():.3f}  std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}", flush=True)

    Xs, ys, tss, sigs = [], [], [], []
    for pid, _ in sized:
        X, y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        ys.append(y)
        tss.append(ts)
        sigs.append(build_btc_dom_signal(ts, share_by_hour, seq_len=_SEQ_LEN))

    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    sig_all = np.concatenate(sigs, axis=0)

    order = np.argsort(ts_all, kind="stable")
    return (
        X_all[order],
        y_all[order],
        ts_all[order],
        sig_all[order],
        pids_used,
    )


def main():
    X, y, ts, sig, pids_used = _load_pooled_with_btc_dom(n=20)
    print(f"\npooled samples: n={len(y):,}", flush=True)
    coverage_pct = float((sig != 0).any(axis=1).mean())
    print(f"per-sample btc-dom coverage (any nonzero): {coverage_pct:.1%}", flush=True)

    print(f"\nReplacing ch{_TARGET_CHANNEL} (obv_slope) with BTC-dominance "
          f"z-score; running 5-fold purged CV (4h embargo)...", flush=True)
    result = run_replace(
        X, y, ts,
        channel_idx=_TARGET_CHANNEL,
        replacement=sig,
        n_folds=5, embargo_hours=4, n_estimators=200,
    )
    print(f"\n=== single-add probe: btc-dominance -> ch{_TARGET_CHANNEL} ===")
    print(f"  baseline mean_auc = {result['baseline_auc']:.4f}")
    print(f"  replaced mean_auc = {result['replaced_auc']:.4f}")
    print(f"  delta             = {result['delta']:+.4f}")
    gate = "PASS" if result['delta'] >= 0.01 else "FAIL"
    print(f"  +0.01 gate: {gate}")


if __name__ == "__main__":
    main()
