"""SQLite persistence for #leader-filter coin_fundamentals table.

DDL itself lives in database.py:init_db() with the rest of the schema. This
module provides only insert/query helpers so it can be unit-tested against a
temp DB without engaging the full app init path. The DDL is also embedded as
_DDL constant here for the standalone ensure_schema() helper used by tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import aiosqlite

from services.leader_ranker import FundamentalsRow

MergeQuality = Literal["BOTH", "CG_ONLY", "CMC_ONLY"]
Source = Literal["forward", "backfill"]


_DDL = """
CREATE TABLE IF NOT EXISTS coin_fundamentals (
    snapshot_ts        INTEGER NOT NULL,
    product_id         TEXT    NOT NULL,
    price              REAL,
    market_cap         REAL,
    fdv                REAL,
    volume_24h         REAL,
    circulating_supply REAL,
    total_supply       REAL,
    price_chg_24h_pct  REAL,
    cg_market_cap      REAL,
    cmc_market_cap     REAL,
    cg_circ_supply     REAL,
    cmc_circ_supply    REAL,
    cg_total_supply    REAL,
    cmc_total_supply   REAL,
    cg_volume_24h      REAL,
    cmc_volume_24h     REAL,
    merge_quality      TEXT,
    divergence_flags   TEXT,
    source             TEXT,
    ingest_ts          INTEGER NOT NULL,
    schema_version     INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (snapshot_ts, product_id)
);

CREATE INDEX IF NOT EXISTS idx_coin_fundamentals_pid_ts
    ON coin_fundamentals(product_id, snapshot_ts);
"""


@dataclass(frozen=True)
class PersistRecord:
    snapshot_ts: int
    product_id: str
    row: FundamentalsRow
    merge_quality: MergeQuality
    cg_market_cap: float | None
    cmc_market_cap: float | None
    cg_circ_supply: float | None
    cmc_circ_supply: float | None
    cg_total_supply: float | None
    cmc_total_supply: float | None
    cg_volume_24h: float | None
    cmc_volume_24h: float | None
    source: Source
    ingest_ts: int
    schema_version: int


class FundamentalsDB:
    def __init__(self, db_path: str):
        self._path = db_path

    async def ensure_schema(self) -> None:
        async with aiosqlite.connect(self._path) as db:
            await db.executescript(_DDL)
            await db.commit()

    async def insert(self, records: list[PersistRecord]) -> None:
        if not records:
            return
        async with aiosqlite.connect(self._path) as db:
            await db.executemany(
                """
                INSERT OR REPLACE INTO coin_fundamentals (
                    snapshot_ts, product_id, price, market_cap, fdv, volume_24h,
                    circulating_supply, total_supply, price_chg_24h_pct,
                    cg_market_cap, cmc_market_cap, cg_circ_supply, cmc_circ_supply,
                    cg_total_supply, cmc_total_supply, cg_volume_24h, cmc_volume_24h,
                    merge_quality, divergence_flags, source, ingest_ts, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        r.snapshot_ts, r.product_id, r.row.price, r.row.market_cap,
                        r.row.fdv, r.row.volume_24h, r.row.circulating_supply,
                        r.row.total_supply, r.row.price_chg_24h_pct,
                        r.cg_market_cap, r.cmc_market_cap,
                        r.cg_circ_supply, r.cmc_circ_supply,
                        r.cg_total_supply, r.cmc_total_supply,
                        r.cg_volume_24h, r.cmc_volume_24h,
                        r.merge_quality, json.dumps(r.row.divergence_flags),
                        r.source, r.ingest_ts, r.schema_version,
                    )
                    for r in records
                ],
            )
            await db.commit()

    async def query_by_pid(self, pid: str, limit: int = 100) -> list[PersistRecord]:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM coin_fundamentals WHERE product_id = ? "
                "ORDER BY snapshot_ts DESC LIMIT ?",
                (pid, limit),
            ) as cur:
                return [_to_record(r) async for r in cur]

    async def query_latest_per_pid(self) -> dict[str, PersistRecord]:
        async with aiosqlite.connect(self._path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """
                SELECT cf.* FROM coin_fundamentals cf
                INNER JOIN (
                    SELECT product_id, MAX(snapshot_ts) AS ts
                    FROM coin_fundamentals GROUP BY product_id
                ) latest ON cf.product_id = latest.product_id
                       AND cf.snapshot_ts = latest.ts
                """
            ) as cur:
                return {r["product_id"]: _to_record(r) async for r in cur}


def _to_record(r) -> PersistRecord:
    flags = json.loads(r["divergence_flags"]) if r["divergence_flags"] else []
    row = FundamentalsRow(
        product_id=r["product_id"],
        price=r["price"], market_cap=r["market_cap"], fdv=r["fdv"],
        volume_24h=r["volume_24h"],
        circulating_supply=r["circulating_supply"],
        total_supply=r["total_supply"],
        price_chg_24h_pct=r["price_chg_24h_pct"],
        divergence_flags=flags,
    )
    return PersistRecord(
        snapshot_ts=r["snapshot_ts"], product_id=r["product_id"], row=row,
        merge_quality=r["merge_quality"],
        cg_market_cap=r["cg_market_cap"], cmc_market_cap=r["cmc_market_cap"],
        cg_circ_supply=r["cg_circ_supply"], cmc_circ_supply=r["cmc_circ_supply"],
        cg_total_supply=r["cg_total_supply"], cmc_total_supply=r["cmc_total_supply"],
        cg_volume_24h=r["cg_volume_24h"], cmc_volume_24h=r["cmc_volume_24h"],
        source=r["source"], ingest_ts=r["ingest_ts"],
        schema_version=r["schema_version"],
    )
