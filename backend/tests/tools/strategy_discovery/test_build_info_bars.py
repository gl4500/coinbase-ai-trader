"""Tests for the build_info_bars CLI orchestrator."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.build_info_bars import build_info_bars_for_pid


def _write_history_parquet(path: Path, n_rows: int = 100, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "start":  np.arange(n_rows, dtype="int64") * 3600,
        "open":   np.full(n_rows, 100.0),
        "high":   np.full(n_rows, 101.0),
        "low":    np.full(n_rows, 99.0),
        "close":  np.full(n_rows, 100.0),
        "volume": rng.uniform(1.0, 10.0, size=n_rows),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def test_build_for_pid_writes_parquet_matching_schema(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    _write_history_parquet(hist_dir / "TEST-USD.parquet", n_rows=80, seed=1)
    result = build_info_bars_for_pid("TEST-USD", hist_dir, out_dir)
    assert result["error"] is None
    out_path = out_dir / "TEST-USD.parquet"
    assert out_path.exists()
    cols = set(pq.read_table(out_path).column_names)
    assert {"start", "end", "open", "high", "low", "close",
            "volume", "dollar_value", "n_1h"} <= cols


def test_build_for_pid_missing_history_returns_error_no_file(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    hist_dir.mkdir()
    result = build_info_bars_for_pid("ABSENT-USD", hist_dir, out_dir)
    assert result["error"] == "missing history"
    assert not (out_dir / "ABSENT-USD.parquet").exists()


def test_build_for_pid_empty_aggregation_returns_error_no_file(tmp_path: Path):
    hist_dir = tmp_path / "history"
    out_dir = tmp_path / "out"
    # All-zero volume → degenerate; aggregate_dollar_bars returns empty.
    df = pd.DataFrame({
        "start":  np.arange(10, dtype="int64") * 3600,
        "open":   np.full(10, 100.0),
        "high":   np.full(10, 100.0),
        "low":    np.full(10, 100.0),
        "close":  np.full(10, 100.0),
        "volume": np.zeros(10),
    })
    (hist_dir).mkdir()
    df.to_parquet(hist_dir / "ZERO-USD.parquet", compression="snappy", index=False)
    result = build_info_bars_for_pid("ZERO-USD", hist_dir, out_dir)
    assert result["error"] == "empty aggregation"
    assert not (out_dir / "ZERO-USD.parquet").exists()
