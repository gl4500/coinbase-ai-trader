"""Tests for the universe marketcap backfill orchestrator."""

from __future__ import annotations

import json

import pyarrow as pa
import pyarrow.parquet as pq

from tools.strategy_discovery import build_universe_marketcap as bum


def _write_mc_parquet(path: str, starts: list[int]) -> None:
    n = len(starts)
    table = pa.table(
        {
            "start": starts,
            "market_cap": [1e9] * n,
            "fdv": [1e9] * n,
            "volume_24h": [1e7] * n,
            "ingest_ts": [1_700_000_000] * n,
            "schema_version": [2] * n,
        }
    )
    pq.write_table(table, path, compression="snappy")


def test_pids_needing_backfill_filters_already_complete(tmp_path):
    """A pid with a parquet covering >= min_days is skipped."""
    mcdir = tmp_path / "marketcap"
    mcdir.mkdir()
    # BTC: covers 199 days (200 rows at 1-day stride), ETH: 99 days, SOL: no parquet.
    _write_mc_parquet(
        str(mcdir / "BTC-USD.parquet"),
        starts=[1_700_000_000 + i * 86400 for i in range(0, 200)],
    )
    _write_mc_parquet(
        str(mcdir / "ETH-USD.parquet"),
        starts=[1_700_000_000 + i * 86400 for i in range(0, 100)],
    )

    needs = bum.pids_needing_backfill(
        pids=["BTC-USD", "ETH-USD", "SOL-USD"],
        marketcap_dir=str(mcdir),
        min_days=180,
    )
    assert sorted(needs) == ["ETH-USD", "SOL-USD"]


def test_pids_needing_backfill_empty_dir(tmp_path):
    """All pids need backfill when the marketcap dir is empty."""
    mcdir = tmp_path / "marketcap"
    mcdir.mkdir()
    needs = bum.pids_needing_backfill(pids=["A", "B"], marketcap_dir=str(mcdir), min_days=180)
    assert sorted(needs) == ["A", "B"]


def test_universe_pids_from_curation_json(tmp_path):
    """universe_pids_from_curation flattens cohort dict into a deduplicated sorted list."""
    p = tmp_path / "universe.json"
    p.write_text(
        json.dumps(
            {
                "large": ["BTC-USD", "ETH-USD"],
                "mid": ["LINK-USD"],
                "high_fdv_ratio": ["AAA-USD"],
                "low_turnover": ["BBB-USD"],
            }
        )
    )
    pids = bum.universe_pids_from_curation(str(p))
    assert sorted(pids) == ["AAA-USD", "BBB-USD", "BTC-USD", "ETH-USD", "LINK-USD"]
