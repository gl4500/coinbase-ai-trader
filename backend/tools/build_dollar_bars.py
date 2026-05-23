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

from services.history_backfill import load_1m_history, load_history

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


def dollar_bars_from_candles(candles: list[dict], threshold: float) -> list[dict]:
    """Walk time-ordered 1m candles into dollar bars.

    A bar closes on the candle whose inclusion makes cumulative dollar value
    reach `threshold`. A 1m candle is atomic — never split. The trailing
    partial bar (residual below threshold at series end) is dropped.

    Each output bar: start, end, open, high, low, close, volume,
    dollar_value, n_candles.
    """
    bars: list[dict] = []
    acc_dollar = 0.0
    acc_volume = 0.0
    bar_start = None
    bar_open = None
    bar_high = None
    bar_low = None
    n = 0

    for c in candles:
        if bar_start is None:
            bar_start = c["start"]
            bar_open = c["open"]
            bar_high = c["high"]
            bar_low = c["low"]
        else:
            bar_high = max(bar_high, c["high"])
            bar_low = min(bar_low, c["low"])
        acc_dollar += candle_dollar_value(c)
        acc_volume += c["volume"]
        n += 1

        if acc_dollar >= threshold:
            bars.append({
                "start": bar_start,
                "end": c["start"],
                "open": bar_open,
                "high": bar_high,
                "low": bar_low,
                "close": c["close"],
                "volume": acc_volume,
                "dollar_value": acc_dollar,
                "n_candles": n,
            })
            acc_dollar = 0.0
            acc_volume = 0.0
            bar_start = None
            bar_open = None
            bar_high = None
            bar_low = None
            n = 0

    return bars


def build_dollar_bars_for_candles(
    one_min_candles: list[dict],
    one_h_candles: list[dict],
) -> list[dict]:
    """Clip 1m candles to the 1h window, calibrate the threshold, build bars.

    `one_h_candles` and `one_min_candles` must be time-sorted (the
    history_backfill loaders return sorted lists). Returns [] when there is no
    1h history or no 1m candle falls inside its span.
    """
    if not one_h_candles:
        return []
    first_ts = int(one_h_candles[0]["start"])
    last_ts = int(one_h_candles[-1]["start"])
    clipped = [c for c in one_min_candles
               if first_ts <= int(c["start"]) <= last_ts]
    if not clipped:
        return []
    threshold = calibrate_threshold(clipped, len(one_h_candles))
    return dollar_bars_from_candles(clipped, threshold)


_DOLLAR_SCHEMA = pa.schema([
    pa.field("start",        pa.int64()),
    pa.field("end",          pa.int64()),
    pa.field("open",         pa.float64()),
    pa.field("high",         pa.float64()),
    pa.field("low",          pa.float64()),
    pa.field("close",        pa.float64()),
    pa.field("volume",       pa.float64()),
    pa.field("dollar_value", pa.float64()),
    pa.field("n_candles",    pa.int64()),
])


def _dollar_parquet_path(product_id: str) -> str:
    """Dollar-bar parquet path — separate namespace under history/dollar/."""
    safe = product_id.replace("/", "_")
    return os.path.join(_HISTORY_DIR, "dollar", f"{safe}.parquet")


def _save_dollar_bars(path: str, bars: list[dict]) -> None:
    """Write dollar bars to a parquet file (overwrites)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.table(
        {
            "start":        [b["start"]        for b in bars],
            "end":          [b["end"]          for b in bars],
            "open":         [b["open"]         for b in bars],
            "high":         [b["high"]         for b in bars],
            "low":          [b["low"]          for b in bars],
            "close":        [b["close"]        for b in bars],
            "volume":       [b["volume"]       for b in bars],
            "dollar_value": [b["dollar_value"] for b in bars],
            "n_candles":    [b["n_candles"]    for b in bars],
        },
        schema=_DOLLAR_SCHEMA,
    )
    pq.write_table(table, path, compression="snappy")


def build_for_pid(product_id: str) -> dict:
    """Build and persist dollar bars for one product. No parquet if no bars."""
    one_min = load_1m_history(product_id)
    one_h = load_history(product_id)
    bars = build_dollar_bars_for_candles(one_min, one_h)
    if bars:
        _save_dollar_bars(_dollar_parquet_path(product_id), bars)
    return {"pid": product_id, "n_bars": len(bars)}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Build dollar bars for the top-20 products from 1m candles"
    )
    parser.add_argument("--cache", default="cnn_dataset_cache.pt",
                        help="cache for the survivorship-aware top-20 ranking")
    parser.add_argument("--pids", default=None,
                        help="comma-separated product ids (overrides --cache)")
    args = parser.parse_args()

    if args.pids:
        pids = [p.strip() for p in args.pids.split(",") if p.strip()]
    else:
        from tools._scorecard._cv_harness import top_n_pids_from_cache
        pids = list(top_n_pids_from_cache(args.cache))

    print(f"build_dollar_bars: {len(pids)} products", flush=True)
    for i, pid in enumerate(pids, 1):
        result = build_for_pid(pid)
        print(f"[{i}/{len(pids)}] {pid}: {result['n_bars']} dollar bars",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
