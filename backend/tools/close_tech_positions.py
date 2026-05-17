"""One-shot preflight before TechAgent removal (#311-refactor-c).

Closes all open TECH paper positions at current Coinbase price, writes
each close to the trades table with trigger_close='MANUAL_TECH_RETIREMENT',
zeros out the agent_state.positions_json for TECH. Idempotent — re-run
with no open positions is a no-op.

Run from repo root:
    .venv/Scripts/python.exe -m backend.tools.close_tech_positions

Run from backend/:
    ../.venv/Scripts/python.exe -m tools.close_tech_positions
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import time
from typing import Optional

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


async def _get_price(pid: str, fallback: float) -> float:
    """Fetch current Coinbase price; fall back to caller-provided value."""
    try:
        from clients import coinbase_client
        data = await coinbase_client.get_product(pid)
        price = float(data.get("price") or 0)
        return price if price > 0 else fallback
    except Exception as exc:
        print(f"  warn: live price fetch failed for {pid}: {exc} — using fallback {fallback}")
        return fallback


async def close_tech_positions(db_path: Optional[str] = None) -> dict:
    """Close every open TECH paper position; zero out agent_state.positions_json.

    Returns dict {n_closed, final_balance, final_realized_pnl, fallback_used}.
    Idempotent: re-running with no open positions returns {n_closed: 0}.
    """
    db_path = db_path or os.path.join(BACKEND, "coinbase.db")
    db = sqlite3.connect(db_path)
    try:
        row = db.execute(
            "SELECT positions_json, balance, realized_pnl FROM agent_state "
            "WHERE agent='TECH'"
        ).fetchone()
        if not row or not row[0]:
            print("No TECH agent_state row or empty positions_json. Nothing to do.")
            return {"n_closed": 0, "final_balance": 0.0, "final_realized_pnl": 0.0,
                    "fallback_used": 0}
        positions = json.loads(row[0])
        balance = float(row[1])
        realized = float(row[2])
        if not positions:
            print("TECH has zero open positions. Nothing to do.")
            return {"n_closed": 0, "final_balance": balance,
                    "final_realized_pnl": realized, "fallback_used": 0}

        print(f"Closing {len(positions)} TECH positions; "
              f"starting balance ${balance:.2f}, realized ${realized:.2f}")
        closed_at_iso = time.strftime(
            "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
        )
        new_realized = realized
        new_balance = balance
        fallback_used = 0

        for pid, p in positions.items():
            size = float(p["size"])
            avg = float(p["avg_price"])
            price = await _get_price(pid, fallback=avg)
            if price == avg and avg > 0:
                fallback_used += 1
            usd_open = size * avg
            usd_close = size * price
            pnl = usd_close - usd_open
            pct = (price / avg - 1) * 100 if avg else 0.0
            opened_at = p.get("entry_time_iso") or closed_at_iso
            db.execute(
                """
                INSERT INTO trades (
                    agent, product_id, entry_price, exit_price, size,
                    usd_open, usd_close, pnl, pct_pnl, hold_secs,
                    trigger_open, trigger_close, balance_after,
                    opened_at, closed_at
                ) VALUES ('TECH', ?, ?, ?, ?, ?, ?, ?, ?, 0,
                          'PRE_RETIREMENT', 'MANUAL_TECH_RETIREMENT',
                          ?, ?, ?)
                """,
                (pid, avg, price, size, usd_open, usd_close, pnl, pct,
                 balance + pnl, opened_at, closed_at_iso),
            )
            new_realized += pnl
            new_balance += usd_close
            print(f"  {pid:<14} sz={size:.6f} avg={avg:.4f} "
                  f"now={price:.4f} pnl=${pnl:+.2f}")

        db.execute(
            "UPDATE agent_state SET positions_json=?, balance=?, realized_pnl=? "
            "WHERE agent='TECH'",
            ("{}", new_balance, new_realized),
        )
        db.commit()
        print(f"Done. Final TECH balance ${new_balance:.2f}  "
              f"realized_pnl ${new_realized:+.2f}  "
              f"(fallback price used for {fallback_used} positions)")
        return {
            "n_closed": len(positions),
            "final_balance": new_balance,
            "final_realized_pnl": new_realized,
            "fallback_used": fallback_used,
        }
    finally:
        db.close()


def main() -> int:
    result = asyncio.run(close_tech_positions())
    return 0 if result["n_closed"] >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
