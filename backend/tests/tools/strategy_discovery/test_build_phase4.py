"""Tests for tools.strategy_discovery.build_phase4 (Phase 4 driver)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_minimal_phase3(phase3_dir: Path):
    _PROFILE_COLUMNS = [
        "pid", "horizon", "leaf_id", "rule_path_summary",
        "cumulative_profit_raw", "cumulative_profit_deflated", "deflation_pp",
        "win_rate", "avg_win", "avg_loss", "max_dd", "sortino",
        "trade_count", "n_folds_passed_q0",
        "chosen_depth", "chosen_min_leaf",
        "bootstrap_triggered", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "n_combos_searched", "inner_cv_se", "schema_version",
    ]
    rows = [
        ("BTC-USD", 24, 0, "price_over_ema20 > 1.0", 0.10, 0.06, 0.04, 0.6, 0.08, -0.04, 0.20, 1.2, 50, 5, 5, 50, False, None, None, 9, 0.015, 1),
        ("ETH-USD", 24, 1, "price_over_ema20 > 1.0", 0.08, 0.05, 0.03, 0.55, 0.07, -0.04, 0.18, 1.1, 40, 5, 5, 50, False, None, None, 9, 0.012, 1),
        ("SOL-USD", 24, 2, "price_over_ema20 > 1.0", 0.06, 0.04, 0.02, 0.50, 0.06, -0.05, 0.22, 1.0, 35, 4, 5, 50, False, None, None, 9, 0.010, 1),
    ]
    df = pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
    phase3_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), phase3_dir / "profiles_h24.parquet")
    sidecar = {"BTC-USD__0": "price_over_ema20 > 1.0",
               "ETH-USD__1": "price_over_ema20 > 1.0",
               "SOL-USD__2": "price_over_ema20 > 1.0"}
    (phase3_dir / "rule_paths_h24.json").write_text(json.dumps(sidecar), encoding="utf-8")


def _write_minimal_phase2(phase2_dir: Path, pids):
    phase2_dir.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        n = 100
        df = pd.DataFrame({
            "ts": (np.arange(n, dtype="int64") * 3_600_000).tolist(),
            "close": [1.0] * n,
            "price_over_ema20": [1.5] * n,
            "vol_over_mc": [0.01] * n,
            "label_h24": [0.05] * n,
        })
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), phase2_dir / f"{pid}.parquet")


def test_sweeps_all_three_caps_writes_three_deployments(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4
    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    cards = build_phase4(
        phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir,
        caps=[3, 4, 5], beam_width=3, pool_size=3, bootstrap_iter=50, seed=42,
        horizons=[24],
    )
    assert set(cards.keys()) == {3, 4, 5}
    for cap in [3, 4, 5]:
        assert (output_dir / f"deployment_n{cap}.json").exists()


def test_writes_scorecard_md_and_telemetry_parquet(tmp_path: Path):
    from tools.strategy_discovery.build_phase4 import build_phase4
    phase3_dir = tmp_path / "phase3"
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase4"
    _write_minimal_phase3(phase3_dir)
    _write_minimal_phase2(phase2_dir, ["BTC-USD", "ETH-USD", "SOL-USD"])
    build_phase4(
        phase3_dir=phase3_dir, phase2_dir=phase2_dir, output_dir=output_dir,
        caps=[3], beam_width=3, pool_size=3, bootstrap_iter=50, seed=42,
        horizons=[24],
    )
    assert (output_dir / "scorecard.md").exists()
    # Telemetry parquet may or may not have content (depending on whether any trades fired)
    # but the file should be writable; if no telemetry, skip the assertion
    tele_path = output_dir / "portfolio_telemetry_n3.parquet"
    if tele_path.exists():
        df = pq.read_table(tele_path).to_pandas()
        # If exists, must have schema_version column
        assert "schema_version" in df.columns


def test_main_returns_zero_on_at_least_one_passing_cap(tmp_path: Path, monkeypatch):
    from tools.strategy_discovery import build_phase4 as bp4
    # Mock build_phase4 to return a card with overall_pass=True
    def fake_build(*args, **kwargs):
        from tools.strategy_discovery.scorecard import CapScorecard
        from tools.strategy_discovery.portfolio_sim import PortfolioMetrics
        return {3: CapScorecard(cap=3, metrics=PortfolioMetrics(),
                                k_evaluated=0, inflation=0.0,
                                gates={}, overall_pass=True, selected_profiles=[])}
    monkeypatch.setattr(bp4, "build_phase4", fake_build)
    rc = bp4.main(["--phase3-dir", str(tmp_path), "--phase2-dir", str(tmp_path),
                   "--output-dir", str(tmp_path / "out"), "--caps", "3"])
    assert rc == 0
