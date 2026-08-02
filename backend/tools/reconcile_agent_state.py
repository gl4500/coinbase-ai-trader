"""
One-shot reconcile (#110): close orphan open trade rows and rewrite
agent_state.realized_pnl to match SUM(trades.pnl) for each agent.

Background — Session 54 introduced a divergence between
`agent_state.realized_pnl` (in-memory accumulator) and `SUM(trades.pnl)`
(persisted ledger) because `_CNNBook.sell()` updated agent_state BEFORE
calling `database.close_trade`. When close_trade failed (Windows file lock,
DB transient error, etc.), agent_state captured the gain but no closed
trade row existed; on next restart the reconcile path force-closed the
orphan row with `pnl=0`, locking the divergence in.

Session 55 patched the ordering (#109). This script repairs the historical
state already on disk for both agents (CNN, TECH).

Usage::

    cd backend
    .venv/Scripts/python.exe -m tools.reconcile_agent_state            # apply
    .venv/Scripts/python.exe -m tools.reconcile_agent_state --dry-run  # preview
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Dict

import database

_AGENTS = ("CNN", "TECH")


async def _orphan_open_trade_ids(agent: str) -> list[int]:
    """Return open trade row IDs for `agent` whose product_id is NOT in the
    agent's saved positions."""
    state = await database.load_agent_state(agent)
    held = set(state["positions"].keys()) if state else set()

    open_rows = await database.get_trades(agent=agent, open_only=True, limit=10_000)
    return [t["id"] for t in open_rows if t["product_id"] not in held]


async def _sum_closed_pnl(agent: str) -> float:
    """SUM(trades.pnl) for closed trades belonging to `agent`."""
    closed = await database.get_trades(agent=agent, closed_only=True, limit=100_000)
    return float(sum((t["pnl"] or 0.0) for t in closed))


async def reconcile(dry_run: bool = False) -> Dict[str, Dict]:
    """Close orphan open trades and rewrite realized_pnl from the trades table.

    Returns a per-agent summary::

        {
          "CNN":  {"orphans_closed": 0,  "realized_pnl_before": -5.55,
                   "realized_pnl_after": -11.53},
          "TECH": {"orphans_closed": 13, "realized_pnl_before": 39.07,
                   "realized_pnl_after": 23.64},
        }
    """
    summary: Dict[str, Dict] = {}

    for agent in _AGENTS:
        state = await database.load_agent_state(agent)
        orphan_ids = await _orphan_open_trade_ids(agent)

        # If dry-run, do NOT close orphans first — sum reflects current state
        if not dry_run:
            for tid in orphan_ids:
                await database.close_trade_by_id(tid, trigger_close="RECONCILE")

        new_pnl = await _sum_closed_pnl(agent)
        old_pnl = state["realized_pnl"] if state else 0.0

        if state and not dry_run and abs(old_pnl - new_pnl) > 1e-9:
            await database.save_agent_state(
                agent,
                balance=state["balance"],
                realized_pnl=new_pnl,
                positions=state["positions"],
                high_water=state["high_water"],
            )

        summary[agent] = {
            "orphans_closed": len(orphan_ids),
            "realized_pnl_before": old_pnl,
            "realized_pnl_after": new_pnl if not dry_run else new_pnl,
        }

    return summary


async def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change without writing"
    )
    args = ap.parse_args()

    await database.init_db()
    summary = await reconcile(dry_run=args.dry_run)

    label = "DRY-RUN — no changes written" if args.dry_run else "APPLIED"
    print(f"\n=== reconcile_agent_state ({label}) ===")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    asyncio.run(_main())
