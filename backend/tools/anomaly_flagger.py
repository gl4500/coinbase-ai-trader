"""Silver-layer OHLCV anomaly flagger (#167).

Cheap, deterministic sanity rules over a list of bronze candles.
Each detector emits a list of anomaly records; `scan_bars` runs all
of them and packs the result into a single report. Pure numpy.

Detectors:
  - flag_ohlc_consistency: high < low, open or close outside [low, high]
  - flag_zero_volume_runs: ≥ `min_run` consecutive zero-volume bars
  - flag_return_z_outliers: |log-return| > k * rolling stdev
  - flag_volume_spikes: volume > k * rolling median (window-tail)

A future commit adds cross-exchange reconciliation as a separate
detector — kept out of scope here so the primitive ships standalone.
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np


def flag_ohlc_consistency(bars: Iterable[dict]) -> List[dict]:
    """Flag bars where high < low or open/close fall outside [low, high]."""
    out: List[dict] = []
    for b in bars:
        h, l = b["high"], b["low"]
        o, c = b["open"], b["close"]
        if h < l:
            out.append({
                "start": b["start"], "kind": "ohlc_inconsistent",
                "detail": f"high={h} < low={l}",
            })
            continue
        if not (l <= o <= h) or not (l <= c <= h):
            out.append({
                "start": b["start"], "kind": "ohlc_inconsistent",
                "detail": f"open={o} close={c} outside [{l},{h}]",
            })
    return out


def flag_zero_volume_runs(
    bars: Iterable[dict], min_run: int = 5,
) -> List[dict]:
    """Flag the run-of-zero-volume tail when length >= min_run."""
    bars = list(bars)
    out: List[dict] = []
    run_start = None
    run_len = 0
    for b in bars:
        if b["volume"] == 0.0:
            if run_start is None:
                run_start = b["start"]
            run_len += 1
        else:
            if run_len >= min_run:
                out.append({
                    "start": run_start, "kind": "zero_volume_run",
                    "detail": f"len={run_len}",
                })
            run_start, run_len = None, 0
    if run_len >= min_run:
        out.append({
            "start": run_start, "kind": "zero_volume_run",
            "detail": f"len={run_len}",
        })
    return out


def flag_return_z_outliers(
    bars: Iterable[dict], window: int = 30, k: float = 4.0,
) -> List[dict]:
    """Flag bars where |log-return| > k * trailing-window stdev.

    Stdev is over the prior `window` returns (excluding current). For
    the first `window` bars there is no baseline → skipped.
    """
    bars = list(bars)
    out: List[dict] = []
    closes = np.array([b["close"] for b in bars], dtype=np.float64)
    if len(closes) < 2:
        return out
    log_ret = np.diff(np.log(closes))  # len = N - 1, indexed at i for bar[i+1]
    for i in range(window, len(log_ret)):
        baseline = log_ret[i - window:i]
        sd = float(np.std(baseline, ddof=1))
        if sd <= 0:
            continue
        z = abs(log_ret[i]) / sd
        if z > k:
            out.append({
                "start": bars[i + 1]["start"],
                "kind": "return_z_outlier",
                "detail": f"z={z:.2f} k={k}",
            })
    return out


def flag_volume_spikes(
    bars: Iterable[dict], window: int = 20, k: float = 10.0,
) -> List[dict]:
    """Flag bars where volume > k * trailing-window median."""
    bars = list(bars)
    out: List[dict] = []
    vols = np.array([b["volume"] for b in bars], dtype=np.float64)
    for i in range(window, len(vols)):
        med = float(np.median(vols[i - window:i]))
        if med <= 0:
            continue
        if vols[i] > k * med:
            out.append({
                "start": bars[i]["start"],
                "kind": "volume_spike",
                "detail": f"v={vols[i]:.2f} med={med:.2f} ratio={vols[i] / med:.1f}",
            })
    return out


def scan_bars(bars: Iterable[dict]) -> dict:
    """Run all detectors and return a combined report.

    Returns:
      {
        "n_bars": int,
        "anomalies": list[dict],   # each has start, kind, detail
        "by_kind": dict[str, int], # counts per kind for quick triage
      }
    """
    bars = list(bars)
    anomalies: List[dict] = []
    anomalies.extend(flag_ohlc_consistency(bars))
    anomalies.extend(flag_zero_volume_runs(bars))
    anomalies.extend(flag_return_z_outliers(bars))
    anomalies.extend(flag_volume_spikes(bars))
    by_kind: dict = {}
    for a in anomalies:
        by_kind[a["kind"]] = by_kind.get(a["kind"], 0) + 1
    return {
        "n_bars": len(bars),
        "anomalies": anomalies,
        "by_kind": by_kind,
    }
