"""Per-channel distribution drift monitor (#170).

Splits each channel's chronologically-ordered terminal-value series
into two halves and reports Population Stability Index (PSI). Standard
thresholds: <0.1 stable, 0.1-0.25 minor drift, >=0.25 significant.

Companion to the stationarity audit (#164) — that one targets mean/var
shifts and lag-1 autocorr; this one is distribution-shape aware. Heavy
left-skew vs heavy right-skew with the same mean and variance both
escape #164 but show up here.

Run:
    python tools/drift_monitor.py [--snapshot-ts auto|<int>] [--n-bins 10]
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import torch  # noqa: E402

from tools.feature_set_compare import _entry_to_arrays  # noqa: E402
from tools.pid_snapshot import recommended_snapshot_ts, survivorship_aware_top_n  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_PSI_MINOR = 0.1
_PSI_SIGNIFICANT = 0.25
_DEFAULT_BINS = 10


def _bin_edges(values: np.ndarray, n_bins: int = _DEFAULT_BINS) -> np.ndarray:
    """Quantile-based edges. Constant input collapses to repeated edges
    (allowed — `_psi` regularises empty bins via epsilon)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _bin_counts(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin values according to edges. Returns probability vector
    (counts / total) of length len(edges) - 1."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == 0:
        return np.zeros(len(edges) - 1)
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts / total if total > 0 else counts.astype(np.float64)


def _psi(p: np.ndarray, q: np.ndarray, eps: float = 1e-6) -> float:
    """Population Stability Index between two probability vectors.

    PSI = sum( (q - p) * log(q / p) ).
    eps regularises zero bins (log(0)) on either side.
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    p_safe = np.clip(p, eps, None)
    q_safe = np.clip(q, eps, None)
    return float(np.sum((q_safe - p_safe) * np.log(q_safe / p_safe)))


def _flag_psi(psi: float) -> str:
    if psi < _PSI_MINOR:
        return "stable"
    if psi < _PSI_SIGNIFICANT:
        return "minor"
    return "significant"


def _channel_drift(series: np.ndarray, n_bins: int = _DEFAULT_BINS) -> dict:
    """Split chronologically into halves; PSI between bin probabilities.

    Pathological inputs (very short, constant) return safe defaults
    without crashing."""
    series = np.asarray(series, dtype=np.float64).ravel()
    n = len(series)
    if n < 4:
        return {"n": n, "psi": 0.0, "flag": "stable", "n_bins": n_bins}
    h = n // 2
    s1, s2 = series[:h], series[h:]
    edges = _bin_edges(s1, n_bins=n_bins)
    p = _bin_counts(s1, edges)
    q = _bin_counts(s2, edges)
    psi = _psi(p, q)
    return {"n": n, "psi": psi, "flag": _flag_psi(psi), "n_bins": n_bins}


def _parse_snapshot_ts(arg: Optional[str], prods: dict) -> Optional[int]:
    if arg is None:
        return None
    if arg == "auto":
        ts = recommended_snapshot_ts(prods)
        return ts if ts > 0 else None
    return int(arg)


def _audit_channels_drift(
    prods: dict, n_pids: int = 20, snapshot_ts: Optional[int] = None,
    n_bins: int = _DEFAULT_BINS,
) -> list:
    """Run per-channel PSI on terminal value of each sample's window,
    chronologically ordered. Returns list of dicts (one per channel)
    with `channel` index added."""
    pids = survivorship_aware_top_n(prods, n=n_pids, snapshot_ts=snapshot_ts)
    Xs, tss = [], []
    for pid in pids:
        X, _y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        tss.append(ts)
    X_all = np.concatenate(Xs, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    X_sorted = X_all[order]

    n_channels = X_sorted.shape[1]
    out = []
    for ch in range(n_channels):
        terminal = X_sorted[:, ch, -1]
        m = _channel_drift(terminal, n_bins=n_bins)
        m["channel"] = ch
        out.append(m)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-ts", default=None,
        help="Survivorship-aware top-N cutoff (#163). 'auto' or Unix-seconds int.",
    )
    parser.add_argument("--n-pids", type=int, default=20)
    parser.add_argument("--n-bins", type=int, default=_DEFAULT_BINS)
    args = parser.parse_args()

    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    snapshot_ts = _parse_snapshot_ts(args.snapshot_ts, prods)
    if snapshot_ts is not None:
        print(f"  snapshot_ts: {snapshot_ts}", flush=True)

    rows = _audit_channels_drift(
        prods, n_pids=args.n_pids, snapshot_ts=snapshot_ts, n_bins=args.n_bins,
    )

    print(
        f"\n{'ch':>3} {'n':>8} {'PSI':>8} flag",
        flush=True,
    )
    print("-" * 40, flush=True)
    for r in rows:
        print(
            f"{r['channel']:>3} {r['n']:>8,} {r['psi']:>8.4f} {r['flag']}",
            flush=True,
        )

    drifters = [r["channel"] for r in rows if r["flag"] == "significant"]
    minors = [r["channel"] for r in rows if r["flag"] == "minor"]
    print(
        f"\nsignificant ({len(drifters)}): {drifters}",
        flush=True,
    )
    print(f"minor       ({len(minors)}): {minors}", flush=True)


if __name__ == "__main__":
    main()
