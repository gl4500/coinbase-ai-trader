"""Regime-stratified walk-forward eval (#165).

Bins samples by realised-volatility terciles and reports per-regime AUC
inside the existing purged walk-forward folds (#132). The point: a single
"overall" AUC can hide a model that only works in one regime, or breaks
in CHAOTIC. This tool prints both numbers per fold.

Volatility proxy: stdev of log-returns over the last 24 bars of the
60-bar window, computed on the price-like channel (Ch 0 norm_close by
default — the trained-on first-seen normalised close from FeatureBuilder).

Run:
    python tools/regime_eval.py [--snapshot-ts auto|<int>] [--n-pids 20] [--n-folds 5]
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
from tools.walk_forward import purged_walk_forward_splits  # noqa: E402
from tools.xgb_features import extract_features  # noqa: E402

_CACHE_PATH = os.path.join(BACKEND, "cnn_dataset_cache.pt")
_LO_PCT = 100.0 / 3.0  # 33.33
_HI_PCT = 200.0 / 3.0  # 66.67
_VOL_LOOKBACK = 24  # last 24 bars of the 60-bar window
_VOL_CHANNEL = 0  # norm_close


def _classify_regimes(vols: np.ndarray) -> np.ndarray:
    """Tercile-bin volatilities into {'low','mid','high'} string labels.

    Constant input → all 'mid' (degenerate but non-crashing)."""
    vols = np.asarray(vols, dtype=np.float64).ravel()
    n = len(vols)
    out = np.full(n, "mid", dtype=object)
    if n == 0:
        return out
    lo = float(np.percentile(vols, _LO_PCT))
    hi = float(np.percentile(vols, _HI_PCT))
    if hi <= lo:
        return out
    out[vols <= lo] = "low"
    out[vols > hi] = "high"
    return out


def _auc_or_none(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """Mann-Whitney AUC. Returns None if a class is missing."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    # Vectorised pairwise comparison; tie-aware
    order = np.argsort(s, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average rank for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sum_ranks = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sum_ranks, inv, ranks)
    avg_ranks = sum_ranks / counts
    ranks_avg = avg_ranks[inv]
    rank_pos_sum = ranks_avg[y == 1].sum()
    n_pos, n_neg = pos.size, neg.size
    auc = (rank_pos_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _per_regime_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    regimes: np.ndarray,
) -> dict:
    """Return {regime: {n, base_rate, auc}} for each of low/mid/high."""
    y = np.asarray(y_true, dtype=np.int64).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    r = np.asarray(regimes).ravel()
    out = {}
    for label in ("low", "mid", "high"):
        mask = r == label
        n_r = int(mask.sum())
        if n_r == 0:
            out[label] = {"n": 0, "base_rate": 0.0, "auc": None}
            continue
        base = float(y[mask].mean()) if n_r else 0.0
        out[label] = {
            "n": n_r,
            "base_rate": base,
            "auc": _auc_or_none(y[mask], s[mask]),
        }
    return out


def _window_vol(
    X: np.ndarray, lookback: int = _VOL_LOOKBACK, channel: int = _VOL_CHANNEL
) -> np.ndarray:
    """Per-sample realised vol = std of last `lookback` bars on `channel`."""
    arr = np.asarray(X)
    last = arr[:, channel, -lookback:]
    return last.std(axis=1)


def _parse_snapshot_ts(arg: Optional[str], prods: dict) -> Optional[int]:
    if arg is None:
        return None
    if arg == "auto":
        ts = recommended_snapshot_ts(prods)
        return ts if ts > 0 else None
    return int(arg)


def _run_eval(
    prods: dict,
    n_pids: int = 20,
    n_folds: int = 5,
    snapshot_ts: Optional[int] = None,
) -> dict:
    """Run regime-stratified eval. Returns dict with overall + per-fold results.

    Uses an in-fold logistic-style score from xgb_features extraction +
    a tiny xgboost CV booster would be heavy; for diagnostic purposes we
    score with a no-op identity baseline (Ch 0 mean) so the harness shape
    is verified end-to-end. Real model wiring is a follow-up — this tool
    proves the regime split mechanic on the live cache."""
    from xgboost import XGBClassifier  # local import: optional dep at module level

    pids = survivorship_aware_top_n(prods, n=n_pids, snapshot_ts=snapshot_ts)
    Xs, ys, tss = [], [], []
    for pid in pids:
        X, y, ts = _entry_to_arrays(prods[pid])
        Xs.append(X)
        ys.append(y)
        tss.append(ts)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    ts_all = np.concatenate(tss, axis=0)
    feats, _names = extract_features(X_all, feature_set="v1")
    vols = _window_vol(X_all)

    overall = {"folds": [], "regime_summary": None}
    all_y, all_p, all_r = [], [], []
    for fold_i, (tr_idx, va_idx) in enumerate(
        purged_walk_forward_splits(ts_all, n_folds=n_folds, embargo_hours=4)
    ):
        if len(tr_idx) == 0 or len(va_idx) == 0:
            continue
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            tree_method="hist",
            n_jobs=4,
            verbosity=0,
        )
        clf.fit(feats[tr_idx], y_all[tr_idx])
        p = clf.predict_proba(feats[va_idx])[:, 1]
        regime = _classify_regimes(vols[va_idx])
        per = _per_regime_metrics(y_all[va_idx], p, regime)
        overall["folds"].append(
            {
                "fold": fold_i,
                "n_train": int(len(tr_idx)),
                "n_val": int(len(va_idx)),
                "auc_overall": _auc_or_none(y_all[va_idx], p),
                "per_regime": per,
            }
        )
        all_y.append(y_all[va_idx])
        all_p.append(p)
        all_r.append(regime)

    if all_y:
        all_y = np.concatenate(all_y)
        all_p = np.concatenate(all_p)
        all_r = np.concatenate(all_r)
        overall["regime_summary"] = _per_regime_metrics(all_y, all_p, all_r)
        overall["auc_overall"] = _auc_or_none(all_y, all_p)
    return overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-ts",
        default=None,
        help="Survivorship-aware top-N cutoff (#163). 'auto' or Unix-seconds int.",
    )
    parser.add_argument("--n-pids", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    print(f"Loading cache: {_CACHE_PATH}", flush=True)
    blob = torch.load(_CACHE_PATH, map_location="cpu", weights_only=False)
    prods = blob["products"]
    snapshot_ts = _parse_snapshot_ts(args.snapshot_ts, prods)
    if snapshot_ts is not None:
        print(f"  snapshot_ts: {snapshot_ts}", flush=True)

    res = _run_eval(
        prods,
        n_pids=args.n_pids,
        n_folds=args.n_folds,
        snapshot_ts=snapshot_ts,
    )

    print("\nFold-by-fold AUC (overall + per-regime):", flush=True)
    print(
        f"{'fold':>4} {'n_val':>8} {'AUC':>7} "
        f"{'low(n,auc)':>16} {'mid(n,auc)':>16} {'high(n,auc)':>16}",
        flush=True,
    )
    print("-" * 72, flush=True)
    for f in res["folds"]:
        per = f["per_regime"]

        def _fmt(d):
            a = d["auc"]
            return f"({d['n']:>5},{a:>5.3f})" if a is not None else f"({d['n']:>5}, n/a )"

        auc = f["auc_overall"]
        auc_s = f"{auc:.3f}" if auc is not None else "n/a"
        print(
            f"{f['fold']:>4} {f['n_val']:>8,} {auc_s:>7} "
            f"{_fmt(per['low']):>16} {_fmt(per['mid']):>16} {_fmt(per['high']):>16}",
            flush=True,
        )

    if res.get("regime_summary"):
        print("\nPooled (all folds concatenated):", flush=True)
        rs = res["regime_summary"]
        ao = res.get("auc_overall")
        print(f"  overall: AUC={ao:.4f}" if ao is not None else "  overall: AUC=n/a", flush=True)
        for label in ("low", "mid", "high"):
            d = rs[label]
            a = d["auc"]
            a_s = f"{a:.4f}" if a is not None else "n/a"
            print(
                f"  {label:>4}: n={d['n']:>6,} base={d['base_rate']:.3f} AUC={a_s}",
                flush=True,
            )


if __name__ == "__main__":
    main()
