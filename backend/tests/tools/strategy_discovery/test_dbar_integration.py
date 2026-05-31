"""Integration smoke: info-bars → build_phase2 on a tiny fixture."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tools.strategy_discovery.build_info_bars import build_info_bars_for_pid
from tools.strategy_discovery.build_phase2 import build_phase2_for_pid
from tools.strategy_discovery.tokenomic_stamp import _TOKENOMIC_COLUMNS


def _seed_1h_history(path: Path, n: int = 800, seed: int = 5) -> None:
    """Synthetic 1h OHLCV with realistic trend + volume variation."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    closes = 100.0 * np.exp(np.cumsum(rets))
    opens = np.concatenate([[100.0], closes[:-1]])
    highs = np.maximum(opens, closes) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=n)))
    lows = np.minimum(opens, closes) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=n)))
    vols = rng.uniform(50.0, 500.0, size=n)
    df = pd.DataFrame({
        "start":  np.arange(n, dtype="int64") * 3600 + 1_700_000_000,
        "open":   opens, "high": highs, "low": lows, "close": closes,
        "volume": vols,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def _seed_marketcap(path: Path, n_days: int = 60, seed: int = 11) -> None:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "start":       np.arange(n_days, dtype="int64") * 86400 + 1_700_000_000,
        "market_cap":  rng.uniform(1e9, 5e9, size=n_days),
        "volume_24h":  rng.uniform(1e7, 5e7, size=n_days),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def _seed_supply(path: Path, pid: str) -> None:
    df = pd.DataFrame({
        "pid":         [pid],
        "circulating": [1_000_000_000.0],
        "total":       [1_500_000_000.0],
        "max_supply":  [2_000_000_000.0],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression="snappy", index=False)


def test_info_bars_to_phase2_smoke(tmp_path: Path):
    pid = "TEST-USD"
    hist_1h     = tmp_path / "history"
    hist_dbar   = tmp_path / "history" / "dollar_1h"
    mc_dir      = tmp_path / "marketcap"
    supply_path = tmp_path / "supply" / "snapshot.parquet"
    phase2_dir  = tmp_path / "phase2_dbar"

    _seed_1h_history(hist_1h / f"{pid}.parquet", n=1600)
    _seed_marketcap(mc_dir / f"{pid}.parquet", n_days=90)
    _seed_supply(supply_path, pid)

    r1 = build_info_bars_for_pid(pid, hist_1h, hist_dbar)
    assert r1["error"] is None
    assert r1["n_bars"] > 0

    r2 = build_phase2_for_pid(pid, hist_dbar, mc_dir, supply_path, phase2_dir)
    assert r2.error is None
    assert r2.rows_written > 0

    out = pq.read_table(phase2_dir / f"{pid}.parquet").to_pandas()
    # OHLCV + trend features
    for col in ("ts", "open", "high", "low", "close",
                "price_over_ema200", "atr14_pct", "ret_24h_sign"):
        assert col in out.columns
    # Tokenomic stamp columns
    for col in _TOKENOMIC_COLUMNS:
        assert col in out.columns
    # Labels finite for at least h24/h72/h168
    for h in (24, 72, 168):
        col = f"label_h{h}"
        assert col in out.columns
        # Most rows should have a finite label (tail rows are NaN by horizon)
        assert int(np.isfinite(out[col]).sum()) >= int(0.5 * len(out))
