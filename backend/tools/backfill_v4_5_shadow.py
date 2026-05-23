"""Backfill v4.5 shadow telemetry onto historical cnn_scans rows.

For each NULL row in scope, fetch tiers AS THEY WOULD HAVE BEEN at
scanned_at, run v4.5 inference, UPDATE the row's three v4.5 columns.

Safe to run while live backend (port 8001) is trading: SQLite WAL +
100-row batched transactions hold writer lock <50ms per batch.

USAGE
  cd backend
  ../.venv/Scripts/python.exe -m tools.backfill_v4_5_shadow [--days N]
       [--batch-size 100] [--db backend/coinbase.db] [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import List, Tuple

# Make backend/ importable when run via `python -m tools.backfill_v4_5_shadow`.
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from agents import xgb_signal  # noqa: E402


logger = logging.getLogger(__name__)


_NEUTRAL_FALLBACK = (0.33, 0.34, 0.33)


def _parse_iso_to_unix(s: str) -> float:
    """Parse '2026-05-23T21:18:19.313951+00:00' to unix seconds."""
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _hhmm() -> str:
    return datetime.now().strftime("%H:%M")


def _select_null_rows(conn: sqlite3.Connection, days: int) -> List[Tuple[int, str, str]]:
    """Select (id, product_id, scanned_at) for rows missing v4.5 in the window."""
    cutoff_unix = time.time() - days * 86400.0
    cutoff_iso  = datetime.fromtimestamp(cutoff_unix, tz=timezone.utc).isoformat()
    cur = conn.execute(
        """
        SELECT id, product_id, scanned_at
        FROM cnn_scans
        WHERE xgb_prob_v4_5_up IS NULL
          AND scanned_at > ?
        ORDER BY id
        """,
        (cutoff_iso,),
    )
    return cur.fetchall()


def _infer_one(pid: str, scanned_at: str) -> Tuple[float, float, float]:
    """Run v4.5 inference at historical timestamp. Raises on any error."""
    now_ts = _parse_iso_to_unix(scanned_at)
    probs = xgb_signal.xgb_prob_v4_5(channels=None, pid=pid, now_ts=now_ts)
    if probs == _NEUTRAL_FALLBACK:
        raise RuntimeError("xgb_prob_v4_5 returned neutral fallback")
    return probs


def backfill(
    db_path: str, days: int, batch_size: int, dry_run: bool,
) -> Tuple[int, int, int]:
    """Returns (total_scope, processed, skipped)."""
    logger.info("[%s] connecting to %s", _hhmm(), db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL")

    rows = _select_null_rows(conn, days)
    total = len(rows)
    logger.info("[%s] scope: %d NULL rows in last %d days", _hhmm(), total, days)

    if total == 0:
        logger.info("[%s] nothing to do — all rows in scope already populated",
                    _hhmm())
        conn.close()
        return total, 0, 0

    if dry_run:
        logger.info("[%s] DRY RUN — first 5 sample rows:", _hhmm())
        for r in rows[:5]:
            logger.info("  id=%d pid=%s scanned_at=%s", r[0], r[1], r[2])
        conn.close()
        return total, 0, 0

    started_at_iso = datetime.now(timezone.utc).isoformat()
    logger.info("[%s] backfill started_at=%s  (use for rollback scope)",
                _hhmm(), started_at_iso)

    processed = 0
    skipped   = 0
    t0        = time.time()

    for i in range(0, total, batch_size):
        batch = rows[i : i + batch_size]
        updates: List[Tuple[float, float, float, int]] = []

        for row_id, pid, scanned_at in batch:
            try:
                p_down, p_neutral, p_up = _infer_one(pid, scanned_at)
                updates.append((
                    round(p_down, 4), round(p_neutral, 4), round(p_up, 4), row_id,
                ))
            except Exception as e:
                logger.debug("skip row %d (%s @ %s): %s", row_id, pid, scanned_at, e)
                skipped += 1

        if updates:
            conn.executemany(
                """
                UPDATE cnn_scans
                SET xgb_prob_v4_5_down    = ?,
                    xgb_prob_v4_5_neutral = ?,
                    xgb_prob_v4_5_up      = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.commit()

        processed += len(batch)
        elapsed = time.time() - t0
        rate    = processed / elapsed if elapsed > 0 else 0.0
        remain  = total - processed
        eta_sec = remain / rate if rate > 0 else 0
        eta_mm  = int(eta_sec // 60)
        eta_ss  = int(eta_sec % 60)
        pct     = 100.0 * processed / total
        logger.info(
            "[%s] progress: %d/%d (%.1f%%) rate=%.1f/s eta=%d:%02d skipped=%d",
            _hhmm(), processed, total, pct, rate, eta_mm, eta_ss, skipped,
        )

    conn.close()
    succeeded = processed - skipped
    logger.info(
        "[%s] done. processed=%d succeeded=%d skipped=%d total_elapsed=%.1fs",
        _hhmm(), processed, succeeded, skipped, time.time() - t0,
    )
    return total, processed, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill v4.5 shadow probs.")
    parser.add_argument("--days", type=int, default=7,
                        help="Look back N days (default 7)")
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Rows per UPDATE transaction (default 100)")
    parser.add_argument(
        "--db",
        default=os.path.join(_BACKEND, "coinbase.db"),
        help="SQLite path (default backend/coinbase.db)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print scope + sample, no UPDATEs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    backfill(
        db_path=args.db,
        days=args.days,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
