"""Per-channel drift diagnostic (#208).

Decomposes the scalar PSI from the #170 drift monitor for any channel
into actionable signals so we can classify drift before the next retrain:

  - decompose_psi:        per-bin contributions — *which* mass moved
  - summary_stats:        mean / var / skew / min / max — shape sketch
  - per_product_drift:    PSI per pid sorted desc — concentrated or broad?
  - bin_count_sensitivity: PSI vs n_bins — robust or normalization-fragile?

Pure numpy. CLI hydrates the live cnn_dataset_cache and runs all four
against the chronologically-sorted terminal-value series for the channel
selected via `--channel N` (defaults to 5 = macd_hist, the original
investigation that motivated this tool).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

_PSI_MINOR = 0.1
_PSI_SIGNIFICANT = 0.25
_DEFAULT_BINS = 10
_EPS = 1e-6


def _quantile_edges(values: np.ndarray, n_bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_probs(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(np.float64)


def _flag(psi: float) -> str:
    if psi < _PSI_MINOR:
        return "stable"
    if psi < _PSI_SIGNIFICANT:
        return "minor"
    return "significant"


def decompose_psi(
    a: np.ndarray,
    b: np.ndarray,
    n_bins: int = _DEFAULT_BINS,
) -> dict:
    """Per-bin PSI contribution breakdown.

    Edges are quantile-based on `a` (the reference half). For each bin
    we report the bin range, p (frac of `a`), q (frac of `b`), and the
    contribution (q-p) * log(q/p) with eps regularisation.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    edges = _quantile_edges(a, n_bins)
    p = _bin_probs(a, edges)
    q = _bin_probs(b, edges)
    p_safe = np.clip(p, _EPS, None)
    q_safe = np.clip(q, _EPS, None)
    contributions = (q_safe - p_safe) * np.log(q_safe / p_safe)
    per_bin: List[dict] = []
    for i in range(n_bins):
        per_bin.append(
            {
                "bin_idx": i,
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "p": float(p[i]),
                "q": float(q[i]),
                "contribution": float(contributions[i]),
            }
        )
    return {
        "total_psi": float(contributions.sum()),
        "flag": _flag(float(contributions.sum())),
        "n_bins": n_bins,
        "per_bin": per_bin,
    }


def summary_stats(values: np.ndarray) -> dict:
    """Mean / var (population) / skew / min / max / n. Empty input returns
    n=0 and NaN/0 for the rest — safe for callers to render."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "var": float("nan"),
            "skew": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    mean = float(arr.mean())
    var = float(arr.var())  # population
    if var > 0:
        skew = float(((arr - mean) ** 3).mean() / (var**1.5))
    else:
        skew = 0.0
    return {
        "n": n,
        "mean": mean,
        "var": var,
        "skew": skew,
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _channel_psi(series: np.ndarray, n_bins: int = _DEFAULT_BINS) -> float:
    arr = np.asarray(series, dtype=np.float64).ravel()
    n = arr.size
    if n < 4:
        return 0.0
    h = n // 2
    return decompose_psi(arr[:h], arr[h:], n_bins=n_bins)["total_psi"]


def per_product_drift(
    prods: Dict[str, dict],
    n_bins: int = _DEFAULT_BINS,
) -> List[dict]:
    """One PSI per product. Each product entry must carry `channel`
    (1-D values, chronological) and `ts` (used only to verify length)."""
    out: List[dict] = []
    for pid, entry in prods.items():
        ch = np.asarray(entry["channel"], dtype=np.float64).ravel()
        n = ch.size
        psi = _channel_psi(ch, n_bins=n_bins)
        out.append({"pid": pid, "n": int(n), "psi": float(psi), "flag": _flag(float(psi))})
    out.sort(key=lambda r: r["psi"], reverse=True)
    return out


def bin_count_sensitivity(
    a: np.ndarray,
    b: np.ndarray,
    n_bins_list: Iterable[int] = (5, 10, 20, 40),
) -> Dict[int, float]:
    """PSI for each n_bins in the input list. A drift that shrinks
    sharply with finer/coarser binning is normalization-fragile."""
    return {int(n): float(decompose_psi(a, b, n_bins=int(n))["total_psi"]) for n in n_bins_list}


_CHANNEL_NAMES = (
    "norm_close",
    "log_volume",
    "hl_range",
    "body",
    "rsi14",
    "macd_hist",
    "ema9_dist",
    "ema21_dist",
    "bb_pos",
    "ret_1",
    "bid",
    "ask",
    "mfi14",
    "obv_slope",
    "stoch_rsi_k",
    "adx14",
    "vwap_dist",
    "fast_rsi_1h",
    "velocity_1h",
    "vol_zscore",
    "funding_rate",
    "btc_corr_20",
    "hour_sin",
    "hour_cos",
    "ivrv_20",
    "ivrv_60",
    "vol_sentiment",
    "okx_oi",
)


def _channel_label(ch: int) -> str:
    if 0 <= ch < len(_CHANNEL_NAMES):
        return f"Ch {ch} ({_CHANNEL_NAMES[ch]})"
    return f"Ch {ch}"


def _load_cache_and_extract_channel(
    channel: int,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Load cnn_dataset_cache and extract a channel's terminal-value series.

    Returns (sorted_global, per_pid_dict) where:
      - sorted_global: chronologically-sorted terminal values for `channel`
      - per_pid_dict: {pid: {"channel": np.ndarray, "ts": np.ndarray}}
    """
    import torch  # local import — CLI-only path

    from tools.feature_set_compare import _entry_to_arrays
    from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n

    cache_path = os.path.join(BACKEND, "cnn_dataset_cache.pt")
    blob = torch.load(cache_path, map_location="cpu", weights_only=False)
    prods = blob["products"]
    snap = recommended_snapshot_ts(prods)
    pids = survivorship_aware_top_n(prods, n=20, snapshot_ts=snap)
    pid_data: Dict[str, dict] = {}
    Xs, tss = [], []
    for pid in pids:
        X, _y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        tss.append(ts)
        order = np.argsort(ts, kind="stable")
        pid_data[pid] = {
            "channel": X[order, channel, -1],
            "ts": ts[order],
        }
    X_all = np.concatenate(Xs, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    return X_all[order, channel, -1], pid_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bins", type=int, default=_DEFAULT_BINS)
    parser.add_argument(
        "--channel",
        type=int,
        default=5,
        help="Channel index to diagnose (default: 5 = macd_hist).",
    )
    args = parser.parse_args()

    sorted_global, pid_data = _load_cache_and_extract_channel(args.channel)
    n = sorted_global.size
    h = n // 2
    a, b = sorted_global[:h], sorted_global[h:]

    label = _channel_label(args.channel)
    print(f"\n{label} drift diagnostic — N={n:,}", flush=True)
    print("=" * 60, flush=True)

    print("\n[1] Per-bin PSI decomposition", flush=True)
    decomp = decompose_psi(a, b, n_bins=args.n_bins)
    print(f"  total_psi={decomp['total_psi']:.4f}  flag={decomp['flag']}", flush=True)
    print(f"  {'bin':>3} {'lo':>10} {'hi':>10} {'p':>7} {'q':>7} {'contrib':>9}", flush=True)
    for r in decomp["per_bin"]:
        lo = "-inf" if r["lo"] == -np.inf else f"{r['lo']:.4f}"
        hi = "+inf" if r["hi"] == np.inf else f"{r['hi']:.4f}"
        print(
            f"  {r['bin_idx']:>3} {lo:>10} {hi:>10} "
            f"{r['p']:>7.4f} {r['q']:>7.4f} {r['contribution']:>9.4f}",
            flush=True,
        )

    print("\n[2] Half-vs-half summary stats", flush=True)
    sa, sb = summary_stats(a), summary_stats(b)
    print(f"  {'metric':<8} {'first':>14} {'second':>14} {'delta':>14}", flush=True)
    for k in ("n", "mean", "var", "skew", "min", "max"):
        delta = sb[k] - sa[k] if k != "n" else sb[k] - sa[k]
        print(f"  {k:<8} {sa[k]:>14.4f} {sb[k]:>14.4f} {delta:>14.4f}", flush=True)

    print("\n[3] Per-product PSI (sorted)", flush=True)
    rows = per_product_drift(pid_data, n_bins=args.n_bins)
    print(f"  {'pid':<14} {'n':>8} {'psi':>8} flag", flush=True)
    for r in rows:
        print(f"  {r['pid']:<14} {r['n']:>8,} {r['psi']:>8.4f} {r['flag']}", flush=True)

    print("\n[4] Bin-count sensitivity", flush=True)
    sens = bin_count_sensitivity(a, b, n_bins_list=(5, 10, 20, 40))
    for n_bins, psi in sens.items():
        print(f"  n_bins={n_bins:>3}: PSI={psi:.4f}  flag={_flag(psi)}", flush=True)


if __name__ == "__main__":
    main()
