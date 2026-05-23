"""Migration: add XGB v4 shadow telemetry column to cnn_scans (#xgb-v4 / Step B.1).

Adds `xgb_prob_v4 REAL` for shadow-mode telemetry — captures the v4 model's
prediction alongside v3's during the 1-week shadow validation period.

Idempotent — safe to re-run. Matches the pattern of
mc_telemetry_20260516.py.
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_v4 REAL to cnn_scans if not present.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [("xgb_prob_v4", "REAL")]
    c = sqlite3.connect(db_path)
    try:
        existing = {row[1] for row in c.execute("PRAGMA table_info(cnn_scans)")}
        added: List[str] = []
        already: List[str] = []
        for name, dtype in new_cols:
            if name in existing:
                already.append(name)
                continue
            c.execute(f"ALTER TABLE cnn_scans ADD COLUMN {name} {dtype}")
            added.append(name)
        c.commit()
    finally:
        c.close()
    return {"added": added, "already_present": already}
