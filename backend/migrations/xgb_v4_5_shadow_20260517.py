"""Migration: add XGB v4.5 3-class shadow telemetry columns (#xgb-v4.5 / Step B.1.5).

Adds three nullable REAL columns to cnn_scans:
  xgb_prob_v4_5_down REAL
  xgb_prob_v4_5_neutral REAL
  xgb_prob_v4_5_up REAL

All three must be written together or all NULL (per CLAUDE.md invariant #17).

Idempotent — safe to re-run. Matches the pattern of mc_telemetry_20260516.py
and xgb_v4_shadow_20260517.py.
"""
from __future__ import annotations
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_v4_5_{down,neutral,up} REAL columns to cnn_scans if absent.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [
        ("xgb_prob_v4_5_down",    "REAL"),
        ("xgb_prob_v4_5_neutral", "REAL"),
        ("xgb_prob_v4_5_up",      "REAL"),
    ]
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
