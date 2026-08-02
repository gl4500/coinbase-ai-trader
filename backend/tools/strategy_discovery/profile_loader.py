"""Phase 4 profile loader — Phase 3 parquets + sidecars + Phase 2 features.

Pure I/O. No simulation, no selection.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow.parquet as pq

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
        return f"{self.pid}__{self.leaf_id}"


def load_all_profiles(
    phase3_dir: Path = Path(BACKEND) / "data" / "phase3",
    horizons: List[int] = None,
    min_folds_passed_q0: int = 4,
) -> List[LoadedProfile]:
    """Load all per-horizon profile parquets + rule-path sidecars.

    Filters profiles with n_folds_passed_q0 < min_folds_passed_q0 (re-enforces
    Phase 3 gate at the Phase 4 input boundary).
    """
    if horizons is None:
        horizons = [1, 4, 24, 72, 168]
    phase3_dir = Path(phase3_dir)
    out: List[LoadedProfile] = []
    for h in horizons:
        parquet_path = phase3_dir / f"profiles_h{int(h)}.parquet"
        sidecar_path = phase3_dir / f"rule_paths_h{int(h)}.json"
        if not parquet_path.exists():
            continue
        df = pq.read_table(parquet_path).to_pandas()
        rule_paths: Dict[str, str] = {}
        if sidecar_path.exists():
            with open(sidecar_path, "r", encoding="utf-8") as f:
                rule_paths = json.load(f)
        for _, row in df.iterrows():
            if int(row["n_folds_passed_q0"]) < min_folds_passed_q0:
                continue
            pid = str(row["pid"])
            leaf_id = int(row["leaf_id"])
            profile_id = f"{pid}__{leaf_id}"
            rule_str = rule_paths.get(profile_id, str(row.get("rule_path_summary", "")))
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
