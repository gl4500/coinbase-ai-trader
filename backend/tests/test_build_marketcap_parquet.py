"""Tests for tools/build_marketcap_parquet.py — bronze marketcap parquet
writer (#295) following the bronze schema convention from
services/history_backfill.py (#168).

Schema (mirrors history_backfill but with marketcap columns):
    start          int64    epoch seconds, hourly bar-aligned
    market_cap     float64
    fdv            float64  (CoinGecko-only; CoinPaprika rows fdv == market_cap)
    ingest_ts      int64    PIT epoch seconds
    schema_version int32

Per-pid path: backend/data/marketcap/<pid>.parquet
"""
import os
import sys
import tempfile

import pyarrow.parquet as pq
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from tools import build_marketcap_parquet as bmp  # noqa: E402


# ── Schema match ────────────────────────────────────────────────────────────

class TestParquetSchema:

    def test_schema_columns_match_bronze_convention(self):
        names = [f.name for f in bmp._SCHEMA]
        assert names == [
            "start", "market_cap", "fdv", "ingest_ts", "schema_version"
        ]

    def test_schema_dtypes_match_bronze_convention(self):
        types = {f.name: str(f.type) for f in bmp._SCHEMA}
        assert types["start"] == "int64"
        assert types["market_cap"] == "double"
        assert types["fdv"] == "double"
        assert types["ingest_ts"] == "int64"
        assert types["schema_version"] == "int32"

    def test_schema_version_is_one(self):
        assert bmp._SCHEMA_VERSION == 1


# ── Round-trip + ordering ───────────────────────────────────────────────────

class TestRoundTrip:

    def test_save_then_load_returns_same_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [
                {"start": 1735689600, "market_cap": 1.0e12, "fdv": 1.05e12},
                {"start": 1735776000, "market_cap": 1.01e12, "fdv": 1.06e12},
            ]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            assert len(loaded) == 2
            assert loaded[0]["start"] == 1735689600
            assert loaded[0]["market_cap"] == pytest.approx(1.0e12)
            assert loaded[1]["start"] == 1735776000

    def test_save_sorts_rows_by_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [
                {"start": 200, "market_cap": 2.0, "fdv": 2.1},
                {"start": 100, "market_cap": 1.0, "fdv": 1.1},
                {"start": 300, "market_cap": 3.0, "fdv": 3.1},
            ]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            starts = [r["start"] for r in loaded]
            assert starts == [100, 200, 300]

    def test_save_dedupes_on_start_collision_keeping_latest_in_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [
                {"start": 100, "market_cap": 1.0, "fdv": 1.1},
                {"start": 100, "market_cap": 1.5, "fdv": 1.6},
            ]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            assert len(loaded) == 1
            assert loaded[0]["market_cap"] == pytest.approx(1.5)

    def test_fdv_defaults_to_market_cap_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [{"start": 100, "market_cap": 9.99e11}]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            assert loaded[0]["fdv"] == pytest.approx(9.99e11)


# ── PIT semantics (#168) ────────────────────────────────────────────────────

class TestPITSemantics:

    def test_save_stamps_ingest_ts_when_not_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [{"start": 100, "market_cap": 1.0e12, "fdv": 1.0e12}]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            assert loaded[0]["ingest_ts"] == 1778500000
            assert loaded[0]["schema_version"] == 1

    def test_save_preserves_existing_ingest_ts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            rows = [{
                "start": 100, "market_cap": 1.0e12, "fdv": 1.0e12,
                "ingest_ts": 1700000000, "schema_version": 1,
            }]
            bmp._save_marketcap_history(path, rows, now_ts=1778500000)
            loaded = bmp._load_marketcap_history(path)
            assert loaded[0]["ingest_ts"] == 1700000000

    def test_load_missing_path_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "DOES-NOT-EXIST.parquet")
            assert bmp._load_marketcap_history(path) == []

    def test_save_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "data", "marketcap", "BTC-USD.parquet")
            assert not os.path.exists(os.path.dirname(nested))
            bmp._save_marketcap_history(
                nested,
                [{"start": 1, "market_cap": 1.0, "fdv": 1.0}],
                now_ts=1778500000,
            )
            assert os.path.exists(nested)


# ── Integration: written file is readable by pyarrow with declared schema ───

class TestParquetFileShape:

    def test_written_file_has_declared_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "BTC-USD.parquet")
            bmp._save_marketcap_history(
                path,
                [{"start": 1, "market_cap": 1.0e12, "fdv": 1.0e12}],
                now_ts=1778500000,
            )
            tbl = pq.read_table(path)
            assert tbl.schema.equals(bmp._SCHEMA)

    def test_path_helper_uses_marketcap_subdir(self):
        path = bmp._parquet_path("BTC-USD")
        assert path.replace("\\", "/").endswith("data/marketcap/BTC-USD.parquet")


# ── Convert (ts_ms, mc) rows to bar-aligned save rows ───────────────────────

class TestRowsFromHistory:

    def test_converts_ms_to_bar_aligned_secs(self):
        # 2025-01-01 00:00:00 UTC = 1735689600 sec = 1735689600000 ms
        history = [(1735689600000, 1.0e12), (1735776000000, 1.01e12)]
        rows = bmp.rows_from_history(history)
        assert rows[0]["start"] == 1735689600
        assert rows[1]["start"] == 1735776000

    def test_floors_non_aligned_ms_to_hour_bar(self):
        # 2025-01-01 00:30:00 UTC -> floored to 2025-01-01 00:00:00
        history = [(1735691400000, 1.0e12)]
        rows = bmp.rows_from_history(history)
        assert rows[0]["start"] == 1735689600

    def test_fdv_defaults_to_market_cap_when_no_fdv_provided(self):
        history = [(1735689600000, 1.0e12)]
        rows = bmp.rows_from_history(history)
        assert rows[0]["fdv"] == pytest.approx(1.0e12)
