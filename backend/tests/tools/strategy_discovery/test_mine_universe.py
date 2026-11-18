"""Tests for tools.strategy_discovery.mine_universe (Phase 3 CLI driver)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tools.strategy_discovery.mine_profiles import LeafProfile
from tools.strategy_discovery.mine_universe import (
    pids_from_universe_json,
    write_profile_parquet,
)


def test_pids_from_universe_json_flattens_cohorts(tmp_path: Path):
    universe = {
        "large": ["BTC-USD", "ETH-USD"],
        "mid": ["LINK-USD"],
        "high_fdv_ratio": ["NEAR-USD", "BTC-USD"],
        "low_turnover": [],
    }
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    pids = pids_from_universe_json(universe_path)
    assert pids == ["BTC-USD", "ETH-USD", "LINK-USD", "NEAR-USD"]


def test_write_profile_parquet_round_trips_all_columns(tmp_path: Path):
    profiles = [
        LeafProfile(
            leaf_id=0,
            rule_path_summary="vol_over_mc > 0.08",
            trade_count=42,
            win_rate=0.6,
            avg_win=0.07,
            avg_loss=-0.04,
            max_dd=0.22,
            cumulative_profit_raw=0.072,
            cumulative_profit_deflated=0.041,
            deflation_pp=0.031,
            n_combos_searched=9,
            inner_cv_se=0.015,
            sortino=1.34,
            n_folds_passed_q0=4,
            bootstrap_triggered=True,
            bootstrap_ci_lower=0.020,
            bootstrap_ci_upper=0.060,
            chosen_depth=5,
            chosen_min_leaf=50,
            n_folds_evaluated=5,
            validation_version="chronological_distinct_folds_v2",
        ),
        LeafProfile(
            leaf_id=1,
            rule_path_summary="ret_24h_sign == 1",
            trade_count=10,
            win_rate=0.5,
            avg_win=0.06,
            avg_loss=-0.05,
            max_dd=0.10,
            cumulative_profit_raw=0.010,
            cumulative_profit_deflated=-0.005,
            deflation_pp=0.015,
            n_combos_searched=9,
            inner_cv_se=0.007,
            sortino=0.5,
            n_folds_passed_q0=5,
            bootstrap_triggered=False,
            bootstrap_ci_lower=None,
            bootstrap_ci_upper=None,
            chosen_depth=3,
            chosen_min_leaf=20,
        ),
    ]
    out_path = tmp_path / "profiles_h24.parquet"
    write_profile_parquet(profiles, pid="BTC-USD", horizon=24, output_path=out_path)
    assert out_path.exists()
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 2
    expected_cols = {
        "pid",
        "horizon",
        "leaf_id",
        "rule_path_summary",
        "trade_count",
        "win_rate",
        "avg_win",
        "avg_loss",
        "max_dd",
        "cumulative_profit_raw",
        "cumulative_profit_deflated",
        "deflation_pp",
        "n_combos_searched",
        "inner_cv_se",
        "sortino",
        "n_folds_passed_q0",
        "bootstrap_triggered",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "chosen_depth",
        "chosen_min_leaf",
        "schema_version",
        "n_folds_evaluated",
        "validation_version",
        "rule_digest",
        "rule_version",
    }
    assert set(df.columns) == expected_cols
    assert (df["pid"] == "BTC-USD").all()
    assert (df["horizon"] == 24).all()
    assert (df["schema_version"] == 3).all()
    assert df["rule_digest"].isna().all()
    assert (df["rule_version"] == "legacy_unverified").all()
    assert df["n_folds_evaluated"].tolist() == [5, 0]
    assert df["validation_version"].tolist() == [
        "chronological_distinct_folds_v2",
        "legacy_unverified",
    ]


def test_iterates_all_pid_horizon_pairs(tmp_path, monkeypatch):
    universe = {"large": ["A-USD", "B-USD"], "mid": ["C-USD"]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase3"
    phase2_dir.mkdir()
    for pid in ["A-USD", "B-USD", "C-USD"]:
        pa_table = pa.table({"ts": [0], "close": [1.0]})
        pq.write_table(pa_table, phase2_dir / f"{pid}.parquet")
    calls = []
    from tools.strategy_discovery import mine_universe as mu

    def fake_mine(pid, horizon, parquet_path, device="cuda", seed=42):
        calls.append((pid, horizon))
        return []

    monkeypatch.setattr(mu, "mine_profiles_for_pid_horizon", fake_mine)
    mu.mine_universe(
        universe_path=universe_path,
        phase2_dir=phase2_dir,
        output_dir=output_dir,
        horizons=[1, 4, 24],
        device="cpu",
        seed=42,
    )
    assert len(calls) == 9
    assert set(calls) == {(p, h) for p in ["A-USD", "B-USD", "C-USD"] for h in [1, 4, 24]}


def test_universe_writes_exact_sidecar_bound_to_profile_row(tmp_path, monkeypatch):
    from tools.strategy_discovery import mine_universe as mu
    from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS
    from tools.strategy_discovery.profit_tree import TreeNode
    from tools.strategy_discovery.rule_contract import (
        bind_rule,
        encode_leaf_rule,
        tree_source_digest,
    )

    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    schema = list(_FEATURE_COLUMNS)
    binding = bind_rule(
        encode_leaf_rule(tree, 0, schema),
        pid="BTC-USD",
        horizon=24,
        profile_leaf_id=0,
        source_leaf_id=0,
        source_outer_fold=4,
        source_tree_digest=tree_source_digest(tree, schema),
        feature_schema=schema,
    )
    profile = LeafProfile(
        leaf_id=0,
        rule_path_summary="market_cap <= 1.02",
        trade_count=42,
        win_rate=0.6,
        avg_win=0.07,
        avg_loss=-0.04,
        max_dd=0.22,
        cumulative_profit_raw=0.072,
        cumulative_profit_deflated=0.041,
        deflation_pp=0.031,
        n_combos_searched=9,
        inner_cv_se=0.015,
        sortino=1.34,
        n_folds_passed_q0=4,
        bootstrap_triggered=False,
        n_folds_evaluated=5,
        validation_version="chronological_distinct_folds_v2",
    )
    profile.rule_binding = binding
    universe = tmp_path / "universe.json"
    universe.write_text(json.dumps({"large": ["BTC-USD"]}))
    phase2 = tmp_path / "phase2"
    phase2.mkdir()
    pq.write_table(pa.table({"ts": [0]}), phase2 / "BTC-USD.parquet")
    monkeypatch.setattr(mu, "mine_profiles_for_pid_horizon", lambda **kwargs: [profile])
    output = tmp_path / "output"
    mu.mine_universe(universe_path=universe, phase2_dir=phase2, output_dir=output, horizons=[24])
    row = pq.read_table(output / "profiles_h24.parquet").to_pandas().iloc[0]
    sidecar = json.loads((output / "rule_paths_h24.json").read_text())
    assert sidecar["BTC-USD__0"] == binding
    assert row["rule_digest"] == binding["digest"]
    assert row["rule_version"] == binding["version"]
    assert row["schema_version"] == 3
    assert "rule_binding" not in row.index
    from tools.strategy_discovery.profile_loader import load_all_profiles

    loaded = load_all_profiles(output, horizons=[24])
    assert len(loaded) == 1
    assert loaded[0].machine_rule == binding["rule"]
    assert loaded[0].rule_digest == binding["digest"]
    assert binding["digest"] in loaded[0].profile_id
