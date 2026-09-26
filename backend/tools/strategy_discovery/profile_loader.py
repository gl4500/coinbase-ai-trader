"""Phase 4 profile loader — Phase 3 parquets + sidecars + Phase 2 features.

Pure I/O. No simulation, no selection.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS
from tools.strategy_discovery.rule_contract import BINDING_VERSION, validate_bound_rule

logger = logging.getLogger(__name__)

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@dataclass
class LoadedProfile:
    pid: str
    horizon: int
    leaf_id: int
    rule_path: str
    cumulative_profit_raw: float
    cumulative_profit_deflated: float
    deflation_pp: float
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd: float
    sortino: float
    trade_count: int
    n_folds_passed_q0: int
    chosen_depth: int
    chosen_min_leaf: int
    machine_rule: dict | None = None
    rule_digest: str = ""

    @property
    def profile_id(self) -> str:
        identity = f"{self.pid}__h{self.horizon}__{self.leaf_id}"
        return f"{identity}__r{self.rule_digest}" if self.rule_digest else identity


def load_all_profiles(
    phase3_dir: Path = Path(BACKEND) / "data" / "phase3",
    horizons: List[int] = None,
    min_folds_passed_q0: int = 4,
) -> List[LoadedProfile]:
    """Load all per-horizon profile parquets + rule-path sidecars.

    Require schema 3, a bound exact rule, chronological_distinct_folds_v2 and five folds. Re-enforce
    the four-pass minimum (or a stricter caller threshold) at the Phase 4 input
    boundary; missing, fractional and impossible pass counts are rejected.

    This version fixes period double-counting only. Metrics still aggregate
    qualifying leaves within a root-direction group, while the attached rule
    is one representative leaf. These are not fixed-policy validation results.
    """
    if horizons is None:
        horizons = [1, 4, 24, 72, 168]
    phase3_dir = Path(phase3_dir)
    out: List[LoadedProfile] = []
    seen_identities = set()
    for h in horizons:
        parquet_path = phase3_dir / f"profiles_h{int(h)}.parquet"
        sidecar_path = phase3_dir / f"rule_paths_h{int(h)}.json"
        if not parquet_path.exists():
            continue
        df = pq.read_table(parquet_path).to_pandas()
        required = {
            "schema_version",
            "validation_version",
            "n_folds_evaluated",
            "rule_version",
            "rule_digest",
        }
        if not required.issubset(df.columns):
            logger.warning(
                "%s: excluded %d profiles with missing provenance", parquet_path.name, len(df)
            )
            continue
        verified = (
            df["schema_version"].eq(3)
            & df["validation_version"].eq("chronological_distinct_folds_v2")
            & df["n_folds_evaluated"].eq(5)
            & df["rule_version"].eq(BINDING_VERSION)
        ).fillna(False)
        excluded = int((~verified).sum())
        if excluded:
            logger.warning(
                "%s: excluded %d profiles with invalid provenance", parquet_path.name, excluded
            )
        df = df.loc[verified]
        # Membership rejects missing, fractional, string and impossible counts
        # before integer conversion; passing folds cannot exceed evaluated folds.
        accepted = df["n_folds_passed_q0"].isin(
            [count for count in range(4, 6) if count >= min_folds_passed_q0]
        )
        df = df.loc[accepted]
        rule_paths: Dict[str, dict] = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                rule_paths = json.load(f)
        if not isinstance(rule_paths, dict):
            logger.warning("%s: excluded malformed rule sidecar", sidecar_path.name)
            continue
        for _, row in df.iterrows():
            row_horizon = row.get("horizon")
            if (
                not isinstance(row_horizon, Real)
                or isinstance(row_horizon, bool)
                or row_horizon != int(h)
            ):
                logger.warning(
                    "%s: excluded profile with invalid or mismatched horizon %r",
                    parquet_path.name,
                    row_horizon,
                )
                continue
            pid = str(row["pid"])
            raw_leaf = row["leaf_id"]
            if (
                not isinstance(raw_leaf, Real)
                or isinstance(raw_leaf, bool)
                or not math.isfinite(raw_leaf)
                or raw_leaf < 0
                or int(raw_leaf) != raw_leaf
            ):
                logger.warning("%s: excluded invalid profile leaf identity", parquet_path.name)
                continue
            leaf_id = int(raw_leaf)
            identity = (pid, int(row["horizon"]), leaf_id)
            if identity in seen_identities:
                raise ValueError(f"duplicate research profile identity: {identity}")
            seen_identities.add(identity)
            # Sidecars are per-horizon; their keys deliberately omit the horizon.
            sidecar_key = f"{pid}__{leaf_id}"
            try:
                machine_rule = validate_bound_rule(
                    rule_paths.get(sidecar_key),
                    pid=pid,
                    horizon=int(h),
                    profile_leaf_id=leaf_id,
                    expected_digest=row["rule_digest"],
                    feature_schema=list(_FEATURE_COLUMNS),
                )
            except ValueError as exc:
                logger.warning(
                    "%s: excluded profile %s with invalid rule: %s",
                    parquet_path.name,
                    identity,
                    exc,
                )
                continue
            out.append(
                LoadedProfile(
                    pid=pid,
                    horizon=int(row["horizon"]),
                    leaf_id=leaf_id,
                    rule_path=str(row.get("rule_path_summary", "")),
                    machine_rule=machine_rule,
                    rule_digest=row["rule_digest"],
                    cumulative_profit_raw=float(row["cumulative_profit_raw"]),
                    cumulative_profit_deflated=float(row["cumulative_profit_deflated"]),
                    deflation_pp=float(row["deflation_pp"]),
                    win_rate=float(row["win_rate"]),
                    avg_win=float(row["avg_win"]),
                    avg_loss=float(row["avg_loss"]),
                    max_dd=float(row["max_dd"]),
                    sortino=float(row["sortino"]),
                    trade_count=int(row["trade_count"]),
                    n_folds_passed_q0=int(row["n_folds_passed_q0"]),
                    chosen_depth=int(row["chosen_depth"]),
                    chosen_min_leaf=int(row["chosen_min_leaf"]),
                )
            )
    return out


def load_pid_features(
    pid: str,
    phase2_dir: Path = Path(BACKEND) / "data" / "phase2",
) -> pd.DataFrame:
    """Load Phase 2 parquet for one pid. Returns empty DataFrame if missing."""
    path = Path(phase2_dir) / f"{pid}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pq.read_table(path).to_pandas()
