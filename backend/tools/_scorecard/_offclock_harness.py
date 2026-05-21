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
