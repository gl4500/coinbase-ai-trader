"""Phase 4 profile loader — Phase 3 parquets + sidecars + Phase 2 features.

Pure I/O. No simulation, no selection.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

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

    @property
    def profile_id(self) -> str:
        return f"{self.pid}__h{self.horizon}__{self.leaf_id}"


def load_all_profiles(
    phase3_dir: Path = Path(BACKEND) / "data" / "phase3",
    horizons: List[int] = None,
    min_folds_passed_q0: int = 4,
) -> List[LoadedProfile]:
    """Load all per-horizon profile parquets + rule-path sidecars.

    Require schema 2, chronological_v1 and five evaluated folds. Re-enforce
    the four-pass minimum (or a stricter caller threshold) at the Phase 4 input
    boundary; missing, fractional and impossible pass counts are rejected.
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
        required = {"schema_version", "validation_version", "n_folds_evaluated"}
        if not required.issubset(df.columns):
            logger.warning(
                "%s: excluded %d profiles with missing provenance", parquet_path.name, len(df)
            )
            continue
        verified = (
            df["schema_version"].eq(2)
            & df["validation_version"].eq("chronological_v1")
            & df["n_folds_evaluated"].eq(5)
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
        rule_paths: Dict[str, str] = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                rule_paths = json.load(f)
        for _, row in df.iterrows():
            pid = str(row["pid"])
            leaf_id = int(row["leaf_id"])
            identity = (pid, int(row["horizon"]), leaf_id)
            if identity in seen_identities:
                raise ValueError(f"duplicate research profile identity: {identity}")
            seen_identities.add(identity)
            # Sidecars are per-horizon; their keys deliberately omit the horizon.
            sidecar_key = f"{pid}__{leaf_id}"
            rule_str = rule_paths.get(sidecar_key)
            if not isinstance(rule_str, str) or not rule_str.strip():
                logger.warning(
                    "%s: excluded profile %s with unresolved rule", parquet_path.name, identity
                )
                continue
            out.append(
                LoadedProfile(
                    pid=pid,
                    horizon=int(row["horizon"]),
                    leaf_id=leaf_id,
                    rule_path=rule_str,
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
