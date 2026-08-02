"""Unit tests for tools/strategy_discovery/inventory.py."""

from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from tools.strategy_discovery import inventory


def _write_ohlcv_parquet(path: str, starts: list[int]) -> None:
    """Helper: write an OHLCV parquet matching the history schema."""
    n = len(starts)
    table = pa.table(
        {
            "start": starts,
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [10.0] * n,
        }
    )
    pq.write_table(table, path, compression="snappy")


def test_scan_history_parquets_returns_per_pid_coverage(tmp_path):
    """scan_history_parquets returns {pid: (first_ts, last_ts, n_rows)} for *.parquet files."""
    hdir = tmp_path / "history"
    hdir.mkdir()
    _write_ohlcv_parquet(
        str(hdir / "BTC-USD.parquet"), starts=[1_700_000_000, 1_700_003_600, 1_700_007_200]
    )
    _write_ohlcv_parquet(str(hdir / "ETH-USD.parquet"), starts=[1_700_000_000, 1_700_003_600])

    out = inventory.scan_history_parquets(str(hdir))

    assert set(out.keys()) == {"BTC-USD", "ETH-USD"}
    btc = out["BTC-USD"]
    assert btc["first_ts"] == 1_700_000_000
    assert btc["last_ts"] == 1_700_007_200
    assert btc["n_rows"] == 3
    eth = out["ETH-USD"]
    assert eth["n_rows"] == 2


def test_scan_history_parquets_skips_macro_prefixed_files(tmp_path):
    """`__`-prefixed parquets (e.g. __MACRO__.parquet) are filtered out per CLAUDE.md invariant #8."""
    hdir = tmp_path / "history"
    hdir.mkdir()
    _write_ohlcv_parquet(str(hdir / "BTC-USD.parquet"), starts=[1_700_000_000])
    _write_ohlcv_parquet(str(hdir / "__MACRO__.parquet"), starts=[1_700_000_000])

    out = inventory.scan_history_parquets(str(hdir))

    assert set(out.keys()) == {"BTC-USD"}
    assert "__MACRO__" not in out


def test_scan_history_parquets_handles_missing_dir(tmp_path):
    """Missing directory returns empty dict, no exception."""
    out = inventory.scan_history_parquets(str(tmp_path / "does_not_exist"))
    assert out == {}


def _write_marketcap_parquet(path: str, starts: list[int]) -> None:
    """Helper: write a marketcap parquet matching the bronze schema."""
    n = len(starts)
    table = pa.table(
        {
            "start": starts,
            "market_cap": [1e9] * n,
            "fdv": [1.5e9] * n,
            "volume_24h": [1e7] * n,
            "ingest_ts": [1_700_000_000] * n,
            "schema_version": [2] * n,
        }
    )
    pq.write_table(table, path, compression="snappy")


def test_scan_marketcap_parquets_returns_per_pid_coverage(tmp_path):
    mdir = tmp_path / "marketcap"
    mdir.mkdir()
    _write_marketcap_parquet(str(mdir / "BTC-USD.parquet"), starts=[1_700_000_000, 1_700_086_400])

    out = inventory.scan_marketcap_parquets(str(mdir))

    assert set(out.keys()) == {"BTC-USD"}
    assert out["BTC-USD"]["n_rows"] == 2
    assert out["BTC-USD"]["first_ts"] == 1_700_000_000
    assert out["BTC-USD"]["last_ts"] == 1_700_086_400


def test_scan_1m_dir_counts_files_per_pid_subdir(tmp_path):
    """1m candles live under <1m_dir>/<pid>/*.parquet. Returns {pid: n_files}."""
    m1 = tmp_path / "1m"
    m1.mkdir()
    btc_dir = m1 / "BTC-USD"
    btc_dir.mkdir()
    (btc_dir / "2026-05.parquet").write_bytes(b"x")
    (btc_dir / "2026-04.parquet").write_bytes(b"x")
    eth_dir = m1 / "ETH-USD"
    eth_dir.mkdir()
    # eth_dir empty

    out = inventory.scan_1m_dir(str(m1))

    assert out == {"BTC-USD": 2, "ETH-USD": 0}


def test_scan_1m_dir_missing_returns_empty(tmp_path):
    """Missing 1m dir returns {}, not exception."""
    assert inventory.scan_1m_dir(str(tmp_path / "no_such_dir")) == {}


def test_inventory_report_contains_section_headers_and_pid_counts():
    """inventory_report renders Markdown with required sections and the totals."""
    history = {
        "BTC-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 4200},
        "ETH-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 4200},
    }
    marketcap = {"BTC-USD": {"first_ts": 1_700_000_000, "last_ts": 1_715_000_000, "n_rows": 175}}
    minute1 = {"BTC-USD": 12, "ETH-USD": 0}

    md = inventory.inventory_report(history=history, marketcap=marketcap, minute1=minute1)

    assert "# Local Data Inventory" in md
    assert "## 1-hour OHLCV (`backend/data/history/`)" in md
    assert "## CoinPaprika tokenomic (`backend/data/marketcap/`)" in md
    assert "## 1-minute OHLCV (`backend/data/history/1m/`)" in md
    # Pids appear in the tables
    assert "BTC-USD" in md
    assert "ETH-USD" in md
    # Counts surface in the summary line
    assert "2 pids" in md or "2  pids" in md or "Total: 2" in md  # accept any reasonable phrasing
