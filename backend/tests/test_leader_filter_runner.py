"""CLI: argument parsing, mode dispatch, smoke for forward run."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_args_forward():
    from tools.leader_filter_runner import parse_args
    args = parse_args(["--mode=forward"])
    assert args.mode == "forward"


def test_parse_args_inspect_with_pid_and_days():
    from tools.leader_filter_runner import parse_args
    args = parse_args(["--mode=inspect", "--pid=BTC-USD", "--days=7"])
    assert args.mode == "inspect"
    assert args.pid == "BTC-USD"
    assert args.days == 7


def test_parse_args_rejects_unknown_mode():
    from tools.leader_filter_runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--mode=backfill"])


def test_main_missing_cmc_key_friendly_error(capsys, monkeypatch):
    from tools.leader_filter_runner import main
    monkeypatch.delenv("COINMARKETCAP_API_KEY", raising=False)
    rc = main(["--mode=forward"])
    out = capsys.readouterr()
    assert rc != 0
    assert "COINMARKETCAP_API_KEY" in (out.out + out.err)


@pytest.mark.asyncio
async def test_run_forward_writes_top_leaders_json(tmp_path, monkeypatch):
    """Smoke: with mocked services, forward writes a well-formed top_leaders.json."""
    from services.coingecko_marketcap import MarketcapRow
    from tools import leader_filter_runner as cli

    cg_rows = {
        "BTC-USD": MarketcapRow(
            market_cap=1.3e12, fdv=1.4e12, circ_supply=1.97e7, total_supply=2.1e7,
            ingest_ts=1, schema_version=1,
            price=67000, volume_24h=2.5e10, price_chg_24h_pct=2.5,
        ),
        "ONDO-USD": MarketcapRow(
            market_cap=1.3e9, fdv=8.5e9, circ_supply=1.43e9, total_supply=1e10,
            ingest_ts=1, schema_version=1,
            price=0.85, volume_24h=9.5e7, price_chg_24h_pct=8.7,
        ),
    }
    cmc_rows = {
        "BTC-USD": MarketcapRow(
            market_cap=1.32e12, fdv=1.41e12, circ_supply=1.97e7, total_supply=2.1e7,
            ingest_ts=1, schema_version=1,
            price=67100, volume_24h=2.55e10, price_chg_24h_pct=2.6,
        ),
        "ONDO-USD": MarketcapRow(
            market_cap=1.31e9, fdv=8.6e9, circ_supply=1.43e9, total_supply=1e10,
            ingest_ts=1, schema_version=1,
            price=0.86, volume_24h=1e8, price_chg_24h_pct=8.8,
        ),
    }

    async def fake_cg(pids): return cg_rows
    async def fake_cmc(pids): return cmc_rows

    monkeypatch.setattr(
        "services.coingecko_marketcap.fetch_marketcap_snapshot", fake_cg
    )
    monkeypatch.setattr(
        "services.coinmarketcap_marketcap.fetch_marketcap_snapshot", fake_cmc
    )
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "k")
    monkeypatch.setenv("LEADER_MIN_MARKET_CAP_USD", "1000000")  # so ONDO passes

    db_path = tmp_path / "coinbase.db"
    out_dir = tmp_path / "out"; out_dir.mkdir()

    await cli.run_forward(
        db_path=str(db_path), data_dir=str(out_dir),
        tracked_pids=["BTC-USD", "ONDO-USD"],
    )

    leaders_path = out_dir / "top_leaders.json"
    assert leaders_path.exists()
    data = json.loads(leaders_path.read_text())
    assert "snapshot_ts" in data
    assert "thresholds_used" in data
    assert "top_leaders" in data
    assert "exclusions" in data
    assert "exclusion_counts" in data
    assert sum(data["exclusion_counts"].values()) == 2


@pytest.mark.asyncio
async def test_run_forward_inserts_db_rows(tmp_path, monkeypatch):
    from services.coingecko_marketcap import MarketcapRow
    from services.leader_fundamentals_db import FundamentalsDB
    from tools import leader_filter_runner as cli

    row = MarketcapRow(
        market_cap=1.3e12, fdv=1.4e12, circ_supply=1.97e7, total_supply=2.1e7,
        ingest_ts=1, schema_version=1,
        price=67000, volume_24h=2.5e10, price_chg_24h_pct=2.5,
    )
    async def fake_cg(pids): return {"BTC-USD": row}
    async def fake_cmc(pids): return {"BTC-USD": row}

    monkeypatch.setattr("services.coingecko_marketcap.fetch_marketcap_snapshot", fake_cg)
    monkeypatch.setattr("services.coinmarketcap_marketcap.fetch_marketcap_snapshot", fake_cmc)
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "k")

    db_path = tmp_path / "coinbase.db"
    out_dir = tmp_path / "out"; out_dir.mkdir()
    await cli.run_forward(str(db_path), str(out_dir), tracked_pids=["BTC-USD"])

    db = FundamentalsDB(str(db_path))
    rows = await db.query_by_pid("BTC-USD")
    assert len(rows) == 1
    assert rows[0].source == "forward"
    assert rows[0].schema_version == 1


@pytest.mark.asyncio
async def test_run_forward_handles_cg_only(tmp_path, monkeypatch):
    from services.coingecko_marketcap import MarketcapRow
    from services.leader_fundamentals_db import FundamentalsDB
    from tools import leader_filter_runner as cli

    row = MarketcapRow(
        market_cap=1.3e12, fdv=1.4e12, circ_supply=1.97e7, total_supply=2.1e7,
        ingest_ts=1, schema_version=1,
        price=67000, volume_24h=2.5e10, price_chg_24h_pct=2.5,
    )
    async def fake_cg(pids): return {"BTC-USD": row}
    async def fake_cmc(pids): return {}

    monkeypatch.setattr("services.coingecko_marketcap.fetch_marketcap_snapshot", fake_cg)
    monkeypatch.setattr("services.coinmarketcap_marketcap.fetch_marketcap_snapshot", fake_cmc)
    monkeypatch.setenv("COINMARKETCAP_API_KEY", "k")

    db_path = tmp_path / "coinbase.db"
    out_dir = tmp_path / "out"; out_dir.mkdir()
    await cli.run_forward(str(db_path), str(out_dir), tracked_pids=["BTC-USD"])

    db = FundamentalsDB(str(db_path))
    rows = await db.query_by_pid("BTC-USD")
    assert rows[0].merge_quality == "CG_ONLY"
