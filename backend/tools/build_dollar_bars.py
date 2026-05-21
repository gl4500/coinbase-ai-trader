"""Construct per-product dollar bars from 1-minute candles.

Stage 2 of the dollar-bar data pipeline (SP1). A dollar bar closes when the
cumulative dollar volume (volume x typical price) of consecutive 1-minute
candles crosses a per-product threshold. The threshold is calibrated so each
product yields about the same number of bars as its existing 1h history.
"""
from __future__ import annotations

import os

import pyarrow as pa
import pyarrow.parquet as pq

_HISTORY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "history")


def candle_dollar_value(candle: dict) -> float:
    """Dollar volume of one candle: volume x typical price ((H+L+C)/3)."""
    typical = (candle["high"] + candle["low"] + candle["close"]) / 3.0
    return candle["volume"] * typical


def calibrate_threshold(one_min_candles: list[dict], n_1h_bars: int) -> float:
    """Per-product dollar threshold = total dollar volume / 1h bar count.

    Raises:
        ValueError: if n_1h_bars is not positive.
    """
    if n_1h_bars <= 0:
        raise ValueError(f"n_1h_bars must be positive, got {n_1h_bars}")
    total = sum(candle_dollar_value(c) for c in one_min_candles)
    return total / n_1h_bars
