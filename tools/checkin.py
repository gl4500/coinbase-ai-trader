"""
Operator check-in snapshot for the Coinbase AI Trader.

Run from anywhere — paths are anchored to the repo root next to this file.
Prints a compact, human-readable summary intended for periodic monitoring
during long-running CNN retrains and live trading sessions.

Usage:
    python tools/checkin.py
    python tools/checkin.py --window-min 20 --outcome-min 60

Sections:
    1. Training status        (backend/cnn_train_progress.json)
    2. Recent scans           (cnn_scans last N min: count, side mix, LLM hit rate, regimes)
    3. Recent trade outcomes  (signal_outcomes last M min: outcome bucket counts)
    4. Win/loss streak        (trades closed last 24h, ordered by closed_at)
    5. Open positions         (positions table snapshot)

Read-only — never writes to the database.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from typing import Optional


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(REPO_ROOT, "backend", "coinbase.db")
PROGRESS_PATH = os.path.join(REPO_ROOT, "backend", "cnn_train_progress.json")


def _section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def show_training(progress_path: str) -> None:
    _section("Training")
    if not os.path.exists(progress_path):
        print("(no progress file)")
        return
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            p = json.load(f)
    except Exception as e:
        print(f"(unreadable: {e})")
        return
    status = p.get("status", "?")
    pid = p.get("pid")
    started = p.get("started_at")
    elapsed = ""
    if isinstance(started, (int, float)):
        elapsed = f" ({(time.time() - started) / 60:.1f} min)"
    print(f"status={status} pid={pid}{elapsed}")
    result = p.get("result") or {}
    if result:
        keys = ("best_val_loss", "fit_status", "overfit_gap_pct",
                "val_auc", "precision_at_0p6", "recall_at_0p6", "saved")
        kv = " ".join(f"{k}={result[k]}" for k in keys if k in result)
        if kv:
            print(kv)


def show_scans(conn: sqlite3.Connection, window_min: int) -> None:
    _section(f"Scans (last {window_min} min)")
    cutoff = time.time() - window_min * 60
    rows = conn.execute(
        "SELECT side, regime, cnn_prob, llm_prob "
        "FROM cnn_scans WHERE scanned_at >= ?",
        (cutoff,),
    ).fetchall()
    if not rows:
        print("(none)")
        return
    sides = Counter(r[0] for r in rows)
    regimes = Counter(r[1] or "?" for r in rows)
    llm_used = sum(1 for r in rows if _safe_float(r[3]) > 0.0)
    cnn_buy = sum(1 for r in rows if _safe_float(r[2]) >= 0.5)
    print(f"total={len(rows)} cnn_prob>=0.5={cnn_buy} llm_used={llm_used}")
    print("sides=" + ", ".join(f"{k}:{v}" for k, v in sides.most_common()))
    print("regimes=" + ", ".join(f"{k}:{v}" for k, v in regimes.most_common(5)))


def show_outcomes(conn: sqlite3.Connection, window_min: int) -> None:
    _section(f"Outcomes (last {window_min} min)")
    cutoff = time.time() - window_min * 60
    rows = conn.execute(
        "SELECT outcome FROM signal_outcomes "
        "WHERE checked_at IS NOT NULL AND checked_at >= ?",
        (cutoff,),
    ).fetchall()
    if not rows:
        print("(none)")
        return
    counts = Counter(r[0] or "?" for r in rows)
    print(f"total={len(rows)}")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")


def show_streak(conn: sqlite3.Connection) -> None:
    _section("Recent closed trades (24h, newest first)")
    cutoff = time.time() - 24 * 3600
    rows = conn.execute(
        "SELECT agent, product_id, pnl, pct_pnl, trigger_close, closed_at "
        "FROM trades WHERE closed_at IS NOT NULL AND closed_at >= ? "
        "ORDER BY closed_at DESC LIMIT 10",
        (cutoff,),
    ).fetchall()
    if not rows:
        print("(none)")
        return
    wins = sum(1 for r in rows if _safe_float(r[2]) > 0)
    losses = sum(1 for r in rows if _safe_float(r[2]) < 0)
    pnl_total = sum(_safe_float(r[2]) for r in rows)
    print(f"shown={len(rows)} wins={wins} losses={losses} sum_pnl=${pnl_total:.2f}")
    for agent, pid, pnl, pct, trig, _ts in rows[:5]:
        print(f"  {agent:>8s} {pid:<10s} pnl=${_safe_float(pnl):+.2f} "
              f"({_safe_float(pct):+.2f}%) close={trig}")


def show_positions(conn: sqlite3.Connection) -> None:
    _section("Open positions")
    rows = conn.execute(
        "SELECT product_id, side, size, avg_price, current_price, "
        "cash_pnl, pct_pnl FROM positions"
    ).fetchall()
    if not rows:
        print("(none)")
        return
    for pid, side, size, avg, cur, cash, pct in rows:
        print(f"  {pid:<10s} {side:<5s} size={_safe_float(size):.6g} "
              f"avg={_safe_float(avg):.4f} cur={_safe_float(cur):.4f} "
              f"pnl=${_safe_float(cash):+.2f} ({_safe_float(pct):+.2f}%)")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Coinbase AI Trader check-in snapshot")
    parser.add_argument("--db", default=DB_PATH, help="path to coinbase.db")
    parser.add_argument("--progress", default=PROGRESS_PATH,
                        help="path to cnn_train_progress.json")
    parser.add_argument("--window-min", type=int, default=20,
                        help="lookback window for scans (minutes)")
    parser.add_argument("--outcome-min", type=int, default=60,
                        help="lookback window for outcomes (minutes)")
    args = parser.parse_args(argv)

    show_training(args.progress)

    if not os.path.exists(args.db):
        print(f"\n(database not found at {args.db})", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    try:
        show_scans(conn, args.window_min)
        show_outcomes(conn, args.outcome_min)
        show_streak(conn)
        show_positions(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
