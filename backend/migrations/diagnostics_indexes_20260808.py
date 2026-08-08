"""Additive indexes for the diagnostics dashboard. Idempotent. Operator-applied
(mirrors mc_telemetry_20260516): run(db_path) opens its own sqlite3 connection."""

import sqlite3
from typing import Dict, List

_INDEXES = {
    "idx_cnn_scans_pid_scanned": "cnn_scans(product_id, scanned_at)",
    "idx_trades_agent_closed": "trades(agent, closed_at)",
}


def run(db_path: str) -> Dict[str, List[str]]:
    """Create indexes for diagnostics queries. Idempotent.

    Returns {"created": [indexes created], "already_present": [skipped]}.
    """
    conn = sqlite3.connect(db_path)
    try:
        existing = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        created, already = [], []
        for name, target in _INDEXES.items():
            if name in existing:
                already.append(name)
                continue
            conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {target}")
            created.append(name)
        conn.commit()
        return {"created": created, "already_present": already}
    finally:
        conn.close()
