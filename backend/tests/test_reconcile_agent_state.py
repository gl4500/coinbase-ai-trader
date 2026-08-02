"""
Tests for tools/reconcile_agent_state.py (#110).

The reconcile tool is a one-shot script that:
1. Closes orphan open trade rows (open in DB but not in agent_state.positions)
2. Rewrites agent_state.realized_pnl to match SUM(trades.pnl) for closed trades

This fixes the divergence created by Session 54 bug where _CNNBook.sell()
updated agent_state BEFORE close_trade — a close_trade failure left
agent_state with the gain captured but no closed trade row, so on restart
the reconcile path force-closed the orphan with pnl=0.
"""

import importlib
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


@pytest.fixture
async def init_db(tmp_path):
    """Per-test fresh SQLite DB.

    The shared `init_db` fixture in conftest.py sets DATABASE_URL via the
    env var, but `config.database_url` is frozen at first import — so
    reloading the database module still resolves to the production DB.
    Patching `database.DB_PATH` directly after reload is the working pattern
    (used in tests/test_database.py)."""
    import database

    importlib.reload(database)
    database.DB_PATH = str(tmp_path / "test.db")
    await database.init_db()
    yield database


@pytest.mark.asyncio
async def test_reconcile_closes_orphan_open_trades(init_db):
    """Open trade rows whose product_id isn't in agent_state.positions get closed."""
    import database
    from tools.reconcile_agent_state import reconcile

    # TECH agent has 2 open trades — only 1 is in saved positions
    await database.open_trade(
        agent="TECH",
        product_id="XRP-USD",
        entry_price=1.30,
        size=100.0,
        usd_open=130.0,
        trigger_open="SCAN",
        balance_after=870.0,
    )
    orphan_id = await database.open_trade(
        agent="TECH",
        product_id="BIO-USD",
        entry_price=0.50,
        size=200.0,
        usd_open=100.0,
        trigger_open="SCAN",
        balance_after=770.0,
    )

    await database.save_agent_state(
        "TECH",
        balance=770.0,
        realized_pnl=0.0,
        positions={
            "XRP-USD": {"size": 100.0, "avg_price": 1.30, "entry_time": 0.0, "peak_price": 1.30}
        },
        high_water={},
    )

    summary = await reconcile(dry_run=False)

    assert summary["TECH"]["orphans_closed"] == 1

    closed = await database.get_trades(agent="TECH", closed_only=True, limit=10)
    closed_ids = {t["id"] for t in closed}
    assert orphan_id in closed_ids

    open_remaining = await database.get_trades(agent="TECH", open_only=True, limit=10)
    assert len(open_remaining) == 1
    assert open_remaining[0]["product_id"] == "XRP-USD"


@pytest.mark.asyncio
async def test_reconcile_rewrites_realized_pnl_to_match_trade_sum(init_db):
    """agent_state.realized_pnl is overwritten with SUM(trades.pnl) for that agent."""
    import database
    from tools.reconcile_agent_state import reconcile

    # Open + close two CNN trades with different pnl
    await database.open_trade(
        agent="CNN",
        product_id="XRP-USD",
        entry_price=1.00,
        size=100.0,
        usd_open=100.0,
        trigger_open="SCAN",
        balance_after=900.0,
    )
    await database.close_trade(
        agent="CNN",
        product_id="XRP-USD",
        exit_price=1.10,
        size=100.0,
        pnl=10.0,
        trigger_close="SCAN",
        balance_after=1010.0,
    )
    await database.open_trade(
        agent="CNN",
        product_id="XRP-USD",
        entry_price=1.20,
        size=50.0,
        usd_open=60.0,
        trigger_open="SCAN",
        balance_after=950.0,
    )
    await database.close_trade(
        agent="CNN",
        product_id="XRP-USD",
        exit_price=1.10,
        size=50.0,
        pnl=-5.0,
        trigger_close="SCAN",
        balance_after=1005.0,
    )

    # agent_state has a wrong, divergent realized_pnl value
    await database.save_agent_state(
        "CNN",
        balance=1005.0,
        realized_pnl=999.99,
        positions={},
        high_water={},
    )

    summary = await reconcile(dry_run=False)

    state = await database.load_agent_state("CNN")
    assert state["realized_pnl"] == pytest.approx(5.0)
    assert summary["CNN"]["realized_pnl_before"] == pytest.approx(999.99)
    assert summary["CNN"]["realized_pnl_after"] == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_reconcile_dry_run_makes_no_changes(init_db):
    """dry_run=True returns the summary without mutating DB."""
    import database
    from tools.reconcile_agent_state import reconcile

    orphan_id = await database.open_trade(
        agent="TECH",
        product_id="BIO-USD",
        entry_price=0.50,
        size=100.0,
        usd_open=50.0,
        trigger_open="SCAN",
        balance_after=950.0,
    )
    await database.save_agent_state(
        "TECH",
        balance=950.0,
        realized_pnl=42.0,
        positions={},
        high_water={},
    )

    summary = await reconcile(dry_run=True)

    assert summary["TECH"]["orphans_closed"] == 1  # would close 1

    open_after = await database.get_trades(agent="TECH", open_only=True, limit=10)
    assert any(t["id"] == orphan_id for t in open_after), "dry-run must not close"

    state = await database.load_agent_state("TECH")
    assert state["realized_pnl"] == 42.0, "dry-run must not rewrite agent_state"


@pytest.mark.asyncio
async def test_reconcile_handles_agent_with_no_state_row(init_db):
    """Agents without a saved agent_state row are skipped without error."""
    from tools.reconcile_agent_state import reconcile

    summary = await reconcile(dry_run=False)
    assert summary["CNN"]["orphans_closed"] == 0
    assert summary["TECH"]["orphans_closed"] == 0
