"""Read-only diagnostics aggregations for the Diagnostics dashboard tab.

Never writes; opens its own mode=ro connection; no coupling to the trading loop.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from services import outcome_labels as ol

_WINDOW_DAYS = {"30d": 30, "90d": 90}
_CACHE: Dict[str, tuple] = {}  # window -> (expires_at, payload)
_TTL_SECS = 60.0


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
    dt = datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(days=_WINDOW_DAYS[window])
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


_CALIBRATION_SUPPRESSED = (
    "Suppressed: this label is a path-independent endpoint return over 4 bars, "
    "while the scoring models train on a path-dependent triple-barrier target "
    "(first touch of +/-1% on intrabar extremes for v3/v4, +/-label_thresh on "
    "closes for v4.5). Win rate by confidence bucket is therefore descriptive "
    "only, not a calibration curve. See "
    "docs/specs/2026-09-26-outcome-label-contract.md section 7."
)


def _scoring_clause() -> str:
    quoted = ",".join(f"'{o}'" for o in ol.SCORING_OUTCOMES)
    return f"outcome IN ({quoted})"


def signal_edge(
    conn: sqlite3.Connection,
    cutoff: Optional[str],
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Signal-label accuracy for the current label version only.

    Version-1 rows are reported separately under "legacy" rather than pooled:
    they were resolved with whatever price was current when the resolver ran
    (mean 45.75 h late per the 2026-09-26 audit), so pooling them would restate
    a stale definition as a current score.

    Denominators count only rows that are eligible (target time reached) and
    carry a scoring outcome. UNAVAILABLE and still-unresolved rows are counted
    and reported, never scored.

    This is label accuracy, not profitability: it ignores fees, position sizing
    and whether a trade was executed at all. Executed-trade P&L lives in
    `exit_attribution`.
    """
    now = time.time() if now is None else now
    clause, params = _where_since("created_at", cutoff)
    scoring = _scoring_clause() + " AND target_time <= ?"

    current = (
        f"FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        f"AND label_version = {ol.LABEL_VERSION}{clause}"
    )
    legacy = (
        "FROM signal_outcomes WHERE source='CNN' AND side='BUY' AND label_version IS NULL" + clause
    )

    n, wins, losses, neutrals, e_return = conn.execute(
        "SELECT COUNT(*), SUM(outcome='WIN'), SUM(outcome='LOSS'), "
        "SUM(outcome='NEUTRAL'), AVG(signed_return) " + current + f" AND {scoring}",
        params + [now],
    ).fetchone()
    n = n or 0

    eligible = (
        conn.execute(
            "SELECT COUNT(*) " + current + " AND COALESCE(target_time, check_after) <= ?",
            params + [now],
        ).fetchone()[0]
        or 0
    )
    unresolved = (
        conn.execute(
            "SELECT COUNT(*) "
            + current
            + " AND outcome IS NULL AND COALESCE(target_time, check_after) <= ?",
            params + [now],
        ).fetchone()[0]
        or 0
    )
    unavailable = (
        conn.execute(
            "SELECT COUNT(*) " + current + " AND outcome='UNAVAILABLE'", params
        ).fetchone()[0]
        or 0
    )

    buckets = []
    for bucket, cnt, w, wl, avg_ret in conn.execute(
        "SELECT CAST(confidence*10 AS INT) AS b, COUNT(*), "
        "SUM(outcome='WIN'), SUM(outcome IN ('WIN','LOSS')), AVG(signed_return) "
        + current
        + f" AND {scoring} GROUP BY b ORDER BY b",
        params + [now],
    ):
        buckets.append(
            {
                "bucket": round(bucket / 10.0, 1),
                "n": cnt,
                "win_rate": (w / wl) if wl else 0.0,
                "avg_ret": avg_ret or 0.0,
            }
        )

    ln, lwins, llosses, lneutrals, le_return = conn.execute(
        "SELECT COUNT(*), SUM(outcome='WIN'), SUM(outcome='LOSS'), "
        "SUM(outcome='NEUTRAL'), AVG(pct_change) " + legacy + f" AND {_scoring_clause()}",
        params,
    ).fetchone()
    ln = ln or 0

    return {
        "label_version": ol.LABEL_VERSION,
        "n": n,
        "wins": wins or 0,
        "losses": losses or 0,
        "neutrals": neutrals or 0,
        "precision": (wins / n) if n else 0.0,
        "e_return": e_return or 0.0,
        "return_units": "fraction",
        "is_profitability": False,
        "counts": {
            "eligible": eligible,
            "matured": n,
            "unresolved": unresolved,
            "unavailable": unavailable,
        },
        # Deliberately empty: see _CALIBRATION_SUPPRESSED.
        "calibration": [],
        "calibration_available": False,
        "calibration_suppressed_reason": _CALIBRATION_SUPPRESSED,
        "confidence_buckets": buckets,
        "legacy": {
            "label_version": 1,
            "n": ln,
            "wins": lwins or 0,
            "losses": llosses or 0,
            "neutrals": lneutrals or 0,
            "precision": (lwins / ln) if ln else 0.0,
            "e_return": le_return or 0.0,
            "note": (
                "Resolved under label version 1, which used the price available "
                "when the resolver ran rather than the price at the target time. "
                "Not comparable with the current version; shown for continuity only."
            ),
        },
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
        by_trigger.append(
            {
                "trigger": trig,
                "n": cnt,
                "sum_pnl": sum_pnl or 0.0,
                "avg_pct": avg_pct or 0.0,
                "win_rate": wr or 0.0,
            }
        )
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
            + base
            + " GROUP BY product_id ORDER BY SUM(pnl)",
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


def signal_funnel(
    conn: sqlite3.Connection, cutoff: Optional[str], now: Optional[float] = None
) -> Dict:
    """Compute signal funnel counts: scans, buy signals, executed trades, matured
    outcomes.

    Args:
        conn: SQLite connection
        cutoff: ISO8601 timestamp or None for no cutoff

    Returns:
        Dict with keys: scans, buy_signals, executed, matured
        scans: total cnn_scans (cutoff on scanned_at)
        buy_signals: cnn_scans with side='BUY' (cutoff on scanned_at)
        executed: trades opened (cutoff on opened_at)
        matured: signal_outcomes with outcome set (cutoff on created_at)
    """
    now = time.time() if now is None else now
    sc_cl, sc_p = _where_since("scanned_at", cutoff)
    op_cl, op_p = _where_since("opened_at", cutoff)
    cr_cl, cr_p = _where_since("created_at", cutoff)
    scans = conn.execute("SELECT COUNT(*) FROM cnn_scans WHERE 1=1" + sc_cl, sc_p).fetchone()[0]
    buys = conn.execute("SELECT COUNT(*) FROM cnn_scans WHERE side='BUY'" + sc_cl, sc_p).fetchone()[
        0
    ]
    executed = conn.execute(
        "SELECT COUNT(*) FROM trades WHERE agent='CNN'" + op_cl, op_p
    ).fetchone()[0]
    # Scoped to the current label version: pooling version-1 rows here would
    # restate delayed labels as matured current-version signals.
    matured = conn.execute(
        "SELECT COUNT(*) FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        f"AND label_version = {ol.LABEL_VERSION} AND {_scoring_clause()} AND target_time <= ?"
        + cr_cl,
        [now] + cr_p,
    ).fetchone()[0]
    legacy_matured = conn.execute(
        "SELECT COUNT(*) FROM signal_outcomes WHERE source='CNN' AND side='BUY' "
        f"AND label_version IS NULL AND {_scoring_clause()}" + cr_cl,
        cr_p,
    ).fetchone()[0]
    return {
        "scans": scans,
        "buy_signals": buys,
        "executed": executed,
        "matured": matured,
        "legacy_matured": legacy_matured,
    }


def compute_diagnostics(window: str, db_path: str, now: Optional[float] = None) -> Dict:
    """Orchestrate diagnostics computation with 60s TTL cache.

    Args:
        window: One of "all", "30d", or "90d"
        db_path: Path to SQLite database
        now: Unix timestamp (seconds since epoch, float); defaults to current time

    Returns:
        Dict with keys: window, generated_at, signal_edge, exit_attribution,
        regime_and_asset, signal_funnel
    """
    now = time.time() if now is None else now
    hit = _CACHE.get(window)
    if hit and hit[0] > now:
        return hit[1]
    cutoff = window_cutoff(window, now)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        payload = {
            "window": window,
            "generated_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "signal_edge": signal_edge(conn, cutoff, now=now),
            "exit_attribution": exit_attribution(conn, cutoff),
            "regime_and_asset": regime_and_asset(conn, cutoff),
            "signal_funnel": signal_funnel(conn, cutoff, now=now),
        }
    finally:
        conn.close()
    _CACHE[window] = (now + _TTL_SECS, payload)
    return payload
