"""Tests for tools.strategy_discovery.build_phase2 (Phase 2 orchestrator)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tools.strategy_discovery.build_phase2 import (
    build_phase2_for_pid,
    build_phase2_for_universe,
)

_HOUR_S = 3_600
_DAY_S = 86_400


def _write_history_parquet(path: Path, n_hours: int = 400, start_day_s: int = 1_000 * _DAY_S):
    rng = np.random.default_rng(11)
    close = 100.0 + rng.normal(0.0, 0.5, size=n_hours).cumsum()
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_hours, dtype="int64") * _HOUR_S,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(n_hours, 1_000.0),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_marketcap_parquet(path: Path, n_days: int = 20, start_day_s: int = 1_000 * _DAY_S):
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
            "market_cap": np.full(n_days, 100_000.0),
            "volume_24h": np.full(n_days, 5_000.0),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="snappy")


def _write_supply_snapshot(path: Path, pid: str = "FOO-USD"):
    schema = pa.schema(
        [
            pa.field("pid", pa.string()),
            pa.field("circulating", pa.float64()),
            pa.field("total", pa.float64()),
            pa.field("max_supply", pa.float64()),
            pa.field("ingest_ts", pa.int64()),
            pa.field("schema_version", pa.int32()),
        ]
    )
    tbl = pa.table(
        {
            "pid": [pid],
            "circulating": [1_000_000.0],
            "total": [2_000_000.0],
            "max_supply": [None],
            "ingest_ts": [1_700_000_000],
            "schema_version": [1],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="snappy")


def test_build_phase2_for_pid_writes_parquet(tmp_path: Path):
    pid = "FOO-USD"
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"
    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_marketcap_parquet(marketcap_dir / f"{pid}.parquet", n_days=20)
    _write_supply_snapshot(supply_path, pid=pid)

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)

    assert result.error is None, f"unexpected error: {result.error}"
    assert result.rows_written > 0
    assert (output_dir / f"{pid}.parquet").exists()

    out = pq.read_table(output_dir / f"{pid}.parquet").to_pandas()
    # Must have all 13 features + 5 labels + identifiers + schema_version
    for col in (
        "ts",
        "pid",
        "market_cap",
        "fdv",
        "fdv_over_mc",
        "circ_over_total",
        "vol_24h",
        "vol_over_mc",
        "price_over_ema20",
        "price_over_ema50",
        "price_over_ema200",
        "ret_1h_sign",
        "ret_24h_sign",
        "ret_7d_sign",
        "atr14_pct",
        "label_h1",
        "label_h4",
        "label_h24",
        "label_h72",
        "label_h168",
        "schema_version",
    ):
        assert col in out.columns, f"missing column {col}"
    assert (out["pid"] == pid).all()
    assert (out["schema_version"] == 1).all()


def test_build_phase2_for_universe_iterates_all_pids(tmp_path: Path):
    pids = ["FOO-USD", "BAR-USD", "BAZ-USD"]
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"

    for p in pids:
        _write_history_parquet(history_dir / f"{p}.parquet", n_hours=400)
        _write_marketcap_parquet(marketcap_dir / f"{p}.parquet", n_days=20)
    # Universe JSON uses Phase 1 cohort layout: {cohort: [pids]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(
        json.dumps(
            {
                "large": ["FOO-USD"],
                "mid": ["BAR-USD"],
                "high_fdv_ratio": ["BAZ-USD"],
                "low_turnover": [],
            }
        ),
        encoding="utf-8",
    )

    # Single supply snapshot parquet for all three pids
    schema = pa.schema(
        [
            pa.field("pid", pa.string()),
            pa.field("circulating", pa.float64()),
            pa.field("total", pa.float64()),
            pa.field("max_supply", pa.float64()),
            pa.field("ingest_ts", pa.int64()),
            pa.field("schema_version", pa.int32()),
        ]
    )
    tbl = pa.table(
        {
            "pid": pids,
            "circulating": [1_000_000.0] * 3,
            "total": [2_000_000.0] * 3,
            "max_supply": [None] * 3,
            "ingest_ts": [1_700_000_000] * 3,
            "schema_version": [1] * 3,
        },
        schema=schema,
    )
    supply_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, supply_path, compression="snappy")

    results = build_phase2_for_universe(
        universe_path,
        history_dir=history_dir,
        marketcap_dir=marketcap_dir,
        supply_path=supply_path,
        output_dir=output_dir,
    )
    assert len(results) == 3
    assert {r.pid for r in results} == set(pids)
    assert all(r.error is None for r in results)
    for p in pids:
        assert (output_dir / f"{p}.parquet").exists()


def test_build_result_reports_drop_counts(tmp_path: Path):
    # Build marketcap parquet with a NaN volume_24h on day D+5 — should drop
    # 24 hourly rows from the output and report it in BuildResult.
    pid = "FOO-USD"
    history_dir = tmp_path / "history"
    marketcap_dir = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    output_dir = tmp_path / "phase2"

    _write_history_parquet(history_dir / f"{pid}.parquet", n_hours=400)
    _write_supply_snapshot(supply_path, pid=pid)

    # Marketcap with a single NaN-volume day.
    # Warmup cut is at hour 200 = day 1008.33 from epoch.
    # NaN at day index 9 (day 1009): after T+1 shift it covers day 1010,
    # which lands inside the post-warmup hourly range (1008.33–1016.62).
    n_days = 20
    start_day_s = 1_000 * _DAY_S
    vols = np.full(n_days, 5_000.0)
    vols[9] = np.nan
    df = pd.DataFrame(
        {
            "start": start_day_s + np.arange(n_days, dtype="int64") * _DAY_S,
            "market_cap": np.full(n_days, 100_000.0),
            "volume_24h": vols,
        }
    )
    marketcap_path = marketcap_dir / f"{pid}.parquet"
    marketcap_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False), marketcap_path, compression="snappy"
    )

    result = build_phase2_for_pid(pid, history_dir, marketcap_dir, supply_path, output_dir)
    assert result.error is None
    assert result.rows_dropped_missing_volume > 0
    # nan_label_counts should be a dict over all 5 horizons
    assert set(result.nan_label_counts.keys()) == {
        "label_h1",
        "label_h4",
        "label_h24",
        "label_h72",
        "label_h168",
    }
