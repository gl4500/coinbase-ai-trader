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


def exit_attribution(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict[str, Any]:
    """Compute exit attribution by trigger and SCAN share of closes.

    Args:
        conn: SQLite connection
        cutoff: ISO8601 timestamp or None for no cutoff

    Returns:
        Dict with keys: by_trigger, scan_sell_share
        by_trigger: list of dicts with trigger, n, sum_pnl, avg_pct, win_rate
        scan_sell_share: fraction of closes triggered by SCAN
    """
    clause, params = _where_since("closed_at", cutoff)
    base = "FROM trades WHERE agent='CNN' AND closed_at IS NOT NULL" + clause
    by_trigger = []
    total = 0
    scan = 0
    for r in conn.execute(
        "SELECT trigger_close, COUNT(*), SUM(pnl), AVG(pct_pnl), "
        "SUM(pnl>0)*1.0/COUNT(*) " + base + " GROUP BY trigger_close ORDER BY SUM(pnl)",
        params,
    ):
        trig, cnt, sum_pnl, avg_pct, wr = r
        by_trigger.append({
            "trigger": trig,
            "n": cnt,
            "sum_pnl": sum_pnl or 0.0,
            "avg_pct": avg_pct or 0.0,
            "win_rate": wr or 0.0,
        })
        total += cnt
        if trig == "SCAN":
            scan += cnt
    return {
        "by_trigger": by_trigger,
        "scan_sell_share": (scan / total) if total else 0.0,
    }


def regime_and_asset(conn: sqlite3.Connection, cutoff: Optional[str]) -> Dict[str, Any]:
    """Compute per-asset PnL and per-regime PnL with nearest-scan regime join.

    Args:
        conn: SQLite connection
        cutoff: ISO8601 timestamp or None for no cutoff

    Returns:
        Dict with keys: by_asset, by_regime
        by_asset: list of dicts with product_id, n, sum_pnl, win_rate
        by_regime: list of dicts with regime, n, sum_pnl
    """
    clause, params = _where_since("closed_at", cutoff)
    base = "FROM trades t WHERE t.agent='CNN' AND t.closed_at IS NOT NULL" + clause
    by_asset = [
        {
            "product_id": r[0],
            "n": r[1],
            "sum_pnl": r[2] or 0.0,
            "win_rate": r[3] or 0.0,
        }
        for r in conn.execute(
            "SELECT product_id, COUNT(*), SUM(pnl), SUM(pnl>0)*1.0/COUNT(*) "
            + base + " GROUP BY product_id ORDER BY SUM(pnl)",
            params,
        )
    ]
    regime_agg: Dict[str, list] = {}
    for pnl, regime in conn.execute(
        "SELECT t.pnl, COALESCE((SELECT s.regime FROM cnn_scans s "
        "WHERE s.product_id=t.product_id AND s.scanned_at<=t.opened_at "
        "ORDER BY s.scanned_at DESC LIMIT 1), 'UNKNOWN') " + base,
        params,
    ):
        agg = regime_agg.setdefault(regime, [0, 0.0])
        agg[0] += 1
        agg[1] += pnl or 0.0
    by_regime = [
        {"regime": k, "n": v[0], "sum_pnl": v[1]}
        for k, v in sorted(regime_agg.items(), key=lambda kv: kv[1][1])
    ]
    return {"by_asset": by_asset, "by_regime": by_regime}
