"""Migration: add MC telemetry columns to cnn_scans (#311-mc-schema).

Idempotent — safe to re-run.
"""
import sqlite3
from typing import Dict, List


def run(db_path: str) -> Dict[str, List[str]]:
    """Add xgb_prob_stdev REAL and mc_telemetry TEXT columns to cnn_scans.

    Returns {"added": [cols added this run], "already_present": [cols skipped]}.
    """
    new_cols = [
        ("xgb_prob_stdev", "REAL"),
        ("mc_telemetry",   "TEXT"),
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
