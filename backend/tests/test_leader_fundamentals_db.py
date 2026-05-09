"""DDL + persistence helpers for coin_fundamentals."""
from __future__ import annotations

import pytest

from services.leader_fundamentals_db import FundamentalsDB, PersistRecord
from services.leader_ranker import FundamentalsRow


def _record(pid="BTC-USD", ts=1_700_000_000) -> PersistRecord:
    row = FundamentalsRow(
        product_id=pid, price=100.0, market_cap=1e9, fdv=1.5e9,
        volume_24h=5e7, circulating_supply=9e8, total_supply=1.5e9,
        price_chg_24h_pct=5.0, divergence_flags=[],
    )
    return PersistRecord(
        snapshot_ts=ts, product_id=pid, row=row, merge_quality="BOTH",
        cg_market_cap=1e9, cmc_market_cap=1e9,
        cg_circ_supply=9e8, cmc_circ_supply=9e8,
        cg_total_supply=1.5e9, cmc_total_supply=1.5e9,
        cg_volume_24h=5e7, cmc_volume_24h=5e7,
        source="forward", ingest_ts=ts, schema_version=1,
    )


@pytest.mark.asyncio
async def test_ensure_schema_idempotent(tmp_path):
    db = FundamentalsDB(str(tmp_path / "t.db"))
    await db.ensure_schema()
    await db.ensure_schema()


@pytest.mark.asyncio
async def test_insert_and_query_roundtrip(tmp_path):
    db = FundamentalsDB(str(tmp_path / "rt.db"))
    await db.ensure_schema()
    await db.insert([_record()])
    rows = await db.query_by_pid("BTC-USD", limit=10)
    assert len(rows) == 1
    assert rows[0].snapshot_ts == 1_700_000_000
    assert rows[0].row.price == 100.0


@pytest.mark.asyncio
async def test_pk_collision_replaces(tmp_path):
    db = FundamentalsDB(str(tmp_path / "pk.db"))
    await db.ensure_schema()
    await db.insert([_record()])
    await db.insert([_record()])
    rows = await db.query_by_pid("BTC-USD")
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_divergence_flags_json_roundtrip(tmp_path):
    db = FundamentalsDB(str(tmp_path / "div.db"))
    await db.ensure_schema()
    row = FundamentalsRow(
        product_id="ONDO-USD", price=1.0, market_cap=1e9, fdv=2e9,
        volume_24h=1e7, circulating_supply=1e9, total_supply=2e9,
        price_chg_24h_pct=8.0, divergence_flags=["circ_supply", "price"],
    )
    rec = PersistRecord(
        snapshot_ts=1, product_id="ONDO-USD", row=row, merge_quality="BOTH",
        cg_market_cap=1e9, cmc_market_cap=1e9,
        cg_circ_supply=9e8, cmc_circ_supply=1.1e9,
        cg_total_supply=2e9, cmc_total_supply=2e9,
        cg_volume_24h=5e6, cmc_volume_24h=1.5e7,
        source="forward", ingest_ts=1, schema_version=1,
    )
    await db.insert([rec])
    rows = await db.query_by_pid("ONDO-USD")
    assert rows[0].row.divergence_flags == ["circ_supply", "price"]


@pytest.mark.asyncio
async def test_pit_columns_persisted(tmp_path):
    db = FundamentalsDB(str(tmp_path / "pit.db"))
    await db.ensure_schema()
    rec = _record()
    await db.insert([rec])
    rows = await db.query_by_pid("BTC-USD")
    assert rows[0].ingest_ts == 1_700_000_000
    assert rows[0].schema_version == 1


@pytest.mark.asyncio
async def test_query_latest_per_pid(tmp_path):
    db = FundamentalsDB(str(tmp_path / "latest.db"))
    await db.ensure_schema()
    await db.insert([_record(ts=1_000), _record(ts=2_000), _record(ts=1_500)])
    latest = await db.query_latest_per_pid()
    assert latest["BTC-USD"].snapshot_ts == 2_000


@pytest.mark.asyncio
async def test_merge_quality_persisted(tmp_path):
    db = FundamentalsDB(str(tmp_path / "mq.db"))
    await db.ensure_schema()
    rec = _record()
    rec_cg = PersistRecord(**{**rec.__dict__, "product_id": "ETH-USD",
                              "merge_quality": "CG_ONLY",
                              "cmc_market_cap": None, "cmc_circ_supply": None,
                              "cmc_total_supply": None, "cmc_volume_24h": None})
    await db.insert([rec, rec_cg])
    eth = (await db.query_by_pid("ETH-USD"))[0]
    assert eth.merge_quality == "CG_ONLY"
    assert eth.cmc_market_cap is None
