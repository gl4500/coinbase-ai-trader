"""Tests for tools.strategy_discovery.profile_loader (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.profile_loader import (
    LoadedProfile,
    load_all_profiles,
)

_PROFILE_COLUMNS = [
    "pid",
    "horizon",
    "leaf_id",
    "rule_path_summary",
    "cumulative_profit_raw",
    "cumulative_profit_deflated",
    "deflation_pp",
    "win_rate",
    "avg_win",
    "avg_loss",
    "max_dd",
    "sortino",
    "trade_count",
    "n_folds_passed_q0",
    "chosen_depth",
    "chosen_min_leaf",
    "bootstrap_triggered",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "n_combos_searched",
    "inner_cv_se",
    "schema_version",
]


def _write_profile_parquet(path: Path, rows):
    df = pd.DataFrame(rows, columns=_PROFILE_COLUMNS)
    df["schema_version"] = 2
    df["validation_version"] = "chronological_distinct_folds_v2"
    df["n_folds_evaluated"] = 5
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _write_rule_paths_json(path: Path, mapping: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(mapping), encoding="utf-8")


def test_loads_all_horizon_parquets(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    # h1 has 1 profile, h24 has 2 profiles, others empty (no file)
    _write_profile_parquet(
        phase3_dir / "profiles_h1.parquet",
        [
            (
                "BTC-USD",
                1,
                0,
                "vol_over_mc > 0.05",
                0.05,
                0.03,
                0.02,
                0.6,
                0.07,
                -0.04,
                0.20,
                1.2,
                30,
                5,
                3,
                20,
                False,
                None,
                None,
                9,
                0.01,
                1,
            ),
        ],
    )
    _write_profile_parquet(
        phase3_dir / "profiles_h24.parquet",
        [
            (
                "BTC-USD",
                24,
                0,
                "price_over_ema20 > 1.02",
                0.10,
                0.06,
                0.04,
                0.55,
                0.08,
                -0.05,
                0.18,
                1.3,
                50,
                5,
                5,
                50,
                False,
                None,
                None,
                9,
                0.015,
                1,
            ),
            (
                "ETH-USD",
                24,
                1,
                "ret_24h_sign > 0",
                0.08,
                0.05,
                0.03,
                0.58,
                0.07,
                -0.04,
                0.15,
                1.4,
                40,
                4,
                5,
                50,
                False,
                None,
                None,
                9,
                0.012,
                1,
            ),
        ],
    )
    _write_rule_paths_json(phase3_dir / "rule_paths_h1.json", {"BTC-USD__0": "vol_over_mc > 0.05"})
    _write_rule_paths_json(
        phase3_dir / "rule_paths_h24.json",
        {
            "BTC-USD__0": "price_over_ema20 > 1.02",
            "ETH-USD__1": "ret_24h_sign > 0",
        },
    )

    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[1, 4, 24, 72, 168])
    assert len(profiles) == 3
    # All carry their (pid, horizon, leaf_id) identifiers + their rule_path
    by_id = {p.profile_id: p for p in profiles}
    assert len(by_id) == 3
    assert by_id["BTC-USD__h1__0"].horizon == 1
    assert by_id["BTC-USD__h24__0"].horizon == 24
    assert by_id["BTC-USD__h1__0"].rule_path == "vol_over_mc > 0.05"
    assert by_id["BTC-USD__h24__0"].rule_path == "price_over_ema20 > 1.02"
    assert all(isinstance(p, LoadedProfile) for p in profiles)


def test_attaches_rule_paths_from_sidecar_json(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    _write_profile_parquet(
        phase3_dir / "profiles_h24.parquet",
        [
            (
                "BTC-USD",
                24,
                0,
                "short_summary",
                0.05,
                0.02,
                0.03,
                0.6,
                0.08,
                -0.04,
                0.15,
                1.2,
                30,
                5,
                5,
                50,
                False,
                None,
                None,
                9,
                0.01,
                1,
            ),
        ],
    )
    _write_rule_paths_json(
        phase3_dir / "rule_paths_h24.json",
        {"BTC-USD__0": "vol_over_mc > 0.08 AND price_over_ema20 > 1.02"},
    )
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[24])
    assert len(profiles) == 1
    # Full rule from JSON wins over the parquet's truncated rule_path_summary
    assert profiles[0].rule_path == "vol_over_mc > 0.08 AND price_over_ema20 > 1.02"


def test_filters_profiles_below_min_folds_passed(tmp_path: Path):
    phase3_dir = tmp_path / "phase3"
    _write_profile_parquet(
        phase3_dir / "profiles_h24.parquet",
        [
            # Profile A passes (5 folds)
            (
                "BTC-USD",
                24,
                0,
                "rule_a",
                0.05,
                0.02,
                0.03,
                0.6,
                0.08,
                -0.04,
                0.15,
                1.2,
                30,
                5,
                5,
                50,
                False,
                None,
                None,
                9,
                0.01,
                1,
            ),
            # Profile B is borderline (4 folds)
            (
                "ETH-USD",
                24,
                1,
                "rule_b",
                0.04,
                0.01,
                0.03,
                0.55,
                0.07,
                -0.04,
                0.18,
                1.0,
                25,
                4,
                5,
                50,
                False,
                None,
                None,
                9,
                0.01,
                1,
            ),
            # Profile C below threshold (3 folds) — should be dropped
            (
                "SOL-USD",
                24,
                2,
                "rule_c",
                0.03,
                0.01,
                0.02,
                0.50,
                0.06,
                -0.05,
                0.20,
                0.8,
                20,
                3,
                5,
                50,
                False,
                None,
                None,
                9,
                0.01,
                1,
            ),
        ],
    )
    _write_rule_paths_json(
        phase3_dir / "rule_paths_h24.json",
        {
            "BTC-USD__0": "rule_a",
            "ETH-USD__1": "rule_b",
            "SOL-USD__2": "rule_c",
        },
    )
    profiles = load_all_profiles(phase3_dir=phase3_dir, horizons=[24], min_folds_passed_q0=4)
    pids = sorted(p.pid for p in profiles)
    assert pids == ["BTC-USD", "ETH-USD"]  # SOL-USD dropped (3 < 4 folds)


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"validation_version": "legacy_unverified", "n_folds_evaluated": 5},
        {"validation_version": "chronological_distinct_folds_v2", "n_folds_evaluated": 4},
        {"validation_version": "chronological_distinct_folds_v2", "n_folds_evaluated": None},
        {"validation_version": "chronological_distinct_folds_v2", "n_folds_evaluated": "5"},
        {"validation_version": "future_unknown", "n_folds_evaluated": 5},
    ],
)
def test_rejects_unverified_profiles_and_reports_exclusion_count(tmp_path, caplog, metadata):
    row = {name: 0 for name in _PROFILE_COLUMNS}
    row.update(
        pid="BTC-USD",
        horizon=24,
        leaf_id=0,
        rule_path_summary="x > 1",
        schema_version=2,
        n_folds_passed_q0=5,
        **metadata,
    )
    pd.DataFrame([row, row]).to_parquet(tmp_path / "profiles_h24.parquet", index=False)
    assert load_all_profiles(tmp_path, horizons=[24]) == []
    assert "excluded 2" in caplog.text.lower()
    assert "provenance" in caplog.text.lower()


@pytest.mark.parametrize("passed", [None, float("nan"), 4.5, 6, -1])
def test_invalid_pass_counts_are_rejected_without_truncation(tmp_path, passed):
    row = {name: 0 for name in _PROFILE_COLUMNS}
    row.update(
        pid="BTC-USD",
        horizon=24,
        leaf_id=0,
        rule_path_summary="x > 1",
        schema_version=2,
        n_folds_passed_q0=passed,
        validation_version="chronological_distinct_folds_v2",
        n_folds_evaluated=5,
    )
    pd.DataFrame([row]).to_parquet(tmp_path / "profiles_h24.parquet", index=False)
    assert load_all_profiles(tmp_path, horizons=[24]) == []


def _write_verified_rule_profile(path, copies=1):
    row = {name: 0 for name in _PROFILE_COLUMNS}
    row.update(
        pid="BTC-USD",
        horizon=24,
        leaf_id=0,
        rule_path_summary="(root)",
        schema_version=2,
        n_folds_passed_q0=5,
        validation_version="chronological_distinct_folds_v2",
        n_folds_evaluated=5,
    )
    pd.DataFrame([row] * copies).to_parquet(path / "profiles_h24.parquet", index=False)


@pytest.mark.parametrize(
    "paths",
    [None, {}, {"BTC-USD__0": ""}, {"BTC-USD__0": "  "}, {"BTC-USD__0": None}, {"BTC-USD__0": 42}],
)
def test_unresolved_sidecar_rule_never_uses_summary(tmp_path, caplog, paths):
    _write_verified_rule_profile(tmp_path)
    if paths is not None:
        _write_rule_paths_json(tmp_path / "rule_paths_h24.json", paths)
    assert load_all_profiles(tmp_path, horizons=[24]) == []
    assert "rule" in caplog.text.lower()


def test_explicit_root_rule_remains_a_valid_profile(tmp_path):
    _write_verified_rule_profile(tmp_path)
    _write_rule_paths_json(tmp_path / "rule_paths_h24.json", {"BTC-USD__0": "(root)"})
    assert load_all_profiles(tmp_path, horizons=[24])[0].rule_path == "(root)"


def test_previous_leaf_count_provenance_is_excluded(tmp_path, caplog):
    _write_verified_rule_profile(tmp_path)
    artifact = tmp_path / "profiles_h24.parquet"
    frame = pd.read_parquet(artifact)
    frame["validation_version"] = "chronological_v1"
    frame.to_parquet(artifact, index=False)
    _write_rule_paths_json(tmp_path / "rule_paths_h24.json", {"BTC-USD__0": "(root)"})
    assert load_all_profiles(tmp_path, horizons=[24]) == []
    assert "provenance" in caplog.text.lower()


def test_duplicate_complete_identity_is_rejected(tmp_path):
    _write_verified_rule_profile(tmp_path, copies=2)
    _write_rule_paths_json(tmp_path / "rule_paths_h24.json", {"BTC-USD__0": "(root)"})
    with pytest.raises(ValueError, match="duplicate"):
        load_all_profiles(tmp_path, horizons=[24])


@pytest.mark.parametrize("row_horizon", [1, 24.5, None, float("nan"), "24", True])
def test_row_horizon_must_match_artifact_without_coercion(tmp_path, caplog, row_horizon):
    _write_verified_rule_profile(tmp_path)
    artifact = tmp_path / "profiles_h24.parquet"
    frame = pd.read_parquet(artifact)
    frame["horizon"] = row_horizon
    frame.to_parquet(artifact, index=False)
    _write_rule_paths_json(tmp_path / "rule_paths_h24.json", {"BTC-USD__0": "(root)"})
    assert load_all_profiles(tmp_path, horizons=[24]) == []
    assert "horizon" in caplog.text.lower()


def test_mismatched_horizon_exclusion_keeps_valid_rows(tmp_path, caplog):
    _write_verified_rule_profile(tmp_path)
    artifact = tmp_path / "profiles_h24.parquet"
    frame = pd.read_parquet(artifact)
    wrong = frame.copy()
    wrong["horizon"] = 1
    pd.concat([wrong, frame]).to_parquet(artifact, index=False)
    _write_rule_paths_json(tmp_path / "rule_paths_h24.json", {"BTC-USD__0": "(root)"})
    profiles = load_all_profiles(tmp_path, horizons=[24])
    assert len(profiles) == 1
    assert profiles[0].horizon == 24
