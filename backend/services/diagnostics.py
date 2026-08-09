"""Read-only diagnostics aggregations for the Diagnostics dashboard tab.

Never writes; opens its own mode=ro connection; no coupling to the trading loop.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

_WINDOW_DAYS = {"30d": 30, "90d": 90}


def window_cutoff(window: str, now: float) -> Optional[str]:
    """Compute ISO cutoff timestamp for a window.

    Args:
        window: One of "all" (no cutoff), "30d", or "90d"
        now: Unix timestamp (seconds since epoch, float)

    Returns:
        ISO8601 timestamp string (e.g. "2023-11-01T12:00:00+00:00") or None for "all"

    Raises:
        ValueError: If window is not recognized
    """
    if window == "all":
        return None
    if window not in _WINDOW_DAYS:
        raise ValueError(f"unknown window: {window!r}")
    dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(
        days=_WINDOW_DAYS[window]
    )
    return dt.isoformat()


def _where_since(col: str, cutoff: Optional[str]) -> tuple[str, list[Any]]:
    """Build WHERE clause and params for created_at >= cutoff filter.

    Args:
        col: Column name to filter on (e.g. "created_at")
        cutoff: ISO8601 timestamp or None

    Returns:
        Tuple of (clause string, params list)
    """
    return (f" AND {col} >= ?", [cutoff]) if cutoff else ("", [])


def signal_edge(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict[str, Any]:
    """Compute signal edge metrics and calibration buckets.

    Args:
        conn: SQLite connection
        cutoff: ISO8601 timestamp or None for no cutoff

    Returns:
        Dict with keys: n, wins, losses, neutrals, precision, e_return, calibration
        precision = wins/n; calibration win_rate excludes NEUTRAL (wins/(wins+losses))
    """
    clause, params = _where_since("created_at", cutoff)
    base = (
        "FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        "AND outcome IN ('WIN','LOSS','NEUTRAL')" + clause
    )
    n, wins, losses, neutrals, e_return = conn.execute(
        "SELECT COUNT(*), "
        "SUM(outcome='WIN'), SUM(outcome='LOSS'), SUM(outcome='NEUTRAL'), "
        "AVG(pct_change) " + base,
        params,
    ).fetchone()
    n = n or 0
    calibration = []
    for r in conn.execute(
        "SELECT CAST(confidence*10 AS INT) AS b, COUNT(*), "
        "SUM(outcome='WIN'), SUM(outcome IN ('WIN','LOSS')), AVG(pct_change) "
        + base + " GROUP BY b ORDER BY b",
        params,
    ):
        bucket, cnt, w, wl, avg_ret = r
        calibration.append({
            "bucket": round(bucket / 10.0, 1),
            "n": cnt,
            "win_rate": (w / wl) if wl else 0.0,
            "avg_ret": avg_ret or 0.0,
        })
    return {
        "n": n,
        "wins": wins or 0,
        "losses": losses or 0,
        "neutrals": neutrals or 0,
        "precision": (wins / n) if n else 0.0,
        "e_return": e_return or 0.0,
        "calibration": calibration,
    }
