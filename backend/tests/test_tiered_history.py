"""TDD tests for services/tiered_history.py — XGB feature_set v3 data layer.

Contract:
    fetch_tiered(pid, source, now_ts) -> {"micro": List[Candle],
                                           "meso":  List[Candle],
                                           "macro": List[Candle]}

micro = last 60 hourly bars
meso  = last 168 hourly bars
macro = last 336 hourly bars

Short-history return: any tier whose underlying series is shorter than its
required length is returned as []. Caller (_extract_v3) interprets [] as
"fill that tier's slots with 0.0".
"""
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


def _candle(start_ts, close=100.0):
    return {"start": start_ts, "open": close, "high": close * 1.01,
            "low": close * 0.99, "close": close, "volume": 1000.0}


@pytest.fixture
def parquet_dir(tmp_path):
    d = tmp_path / "history"
    d.mkdir()
    return d


def _write_parquet(parquet_dir, pid, n_bars, start_ts=1_700_000_000):
    rows = [_candle(start_ts + i * 3600, close=100.0 + i * 0.1) for i in range(n_bars)]
    df = pd.DataFrame(rows)
    df["ingest_ts"] = 1_700_000_000
    df["schema_version"] = 1
    df.to_parquet(parquet_dir / f"{pid}.parquet")


@pytest.fixture
def sqlite_db(tmp_path):
    path = tmp_path / "coinbase.db"
    c = sqlite3.connect(path)
    c.execute("""
        CREATE TABLE candles (
            id INTEGER PRIMARY KEY, product_id TEXT, start REAL,
            open REAL, high REAL, low REAL, close REAL, volume REAL
        )""")
    c.commit()
    c.close()
    return path


def _seed_sqlite(db_path, pid, n_bars, start_ts=1_700_000_000):
    c = sqlite3.connect(db_path)
    for i in range(n_bars):
        ts = start_ts + i * 3600
        c.execute(
            "INSERT INTO candles (product_id, start, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (pid, ts, 100.0, 101.0, 99.0, 100.0 + i * 0.1, 1000.0),
        )
    c.commit()
    c.close()


# ──────────────────────────────────────────────────────────────────────
class TestSliceContracts:
    def test_returns_three_keys(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert set(result.keys()) == {"micro", "meso", "macro"}

    def test_micro_returns_last_60_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["micro"]) == 60

    def test_meso_returns_last_168_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["meso"]) == 168

    def test_macro_returns_last_336_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert len(result["macro"]) == 336

    def test_chronological_order_ascending(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="parquet", parquet_dir=str(parquet_dir))
        for tier in ("micro", "meso", "macro"):
            starts = [c["start"] for c in result[tier]]
            assert starts == sorted(starts), f"{tier} not sorted ascending"


class TestShortHistory:
    def test_short_history_returns_empty_list_for_macro(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "NEW-USD", 200)  # < 336
        result = fetch_tiered("NEW-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert len(result["meso"]) == 168
        assert len(result["micro"]) == 60

    def test_short_history_meso_empty_macro_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "TINY-USD", 100)  # < 168
        result = fetch_tiered("TINY-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_only_micro_history(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "FRESH-USD", 70)
        result = fetch_tiered("FRESH-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result["macro"] == []
        assert result["meso"] == []
        assert len(result["micro"]) == 60

    def test_parquet_missing_returns_all_empty(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        result = fetch_tiered("MISSING-USD", source="parquet", parquet_dir=str(parquet_dir))
        assert result == {"micro": [], "meso": [], "macro": []}


class TestSourceDispatch:
    def test_source_live_reads_sqlite_first(self, sqlite_db):
        from services.tiered_history import fetch_tiered
        _seed_sqlite(sqlite_db, "BTC-USD", 400)
        result = fetch_tiered("BTC-USD", source="live", db_path=str(sqlite_db))
        assert len(result["macro"]) == 336

    def test_source_live_falls_back_to_parquet_for_prefix(self, sqlite_db, parquet_dir):
        from services.tiered_history import fetch_tiered
        _seed_sqlite(sqlite_db, "BTC-USD", 100)  # SQLite has 100, < 336
        _write_parquet(parquet_dir, "BTC-USD", 400)  # parquet has 400
        result = fetch_tiered(
            "BTC-USD", source="live",
            db_path=str(sqlite_db), parquet_dir=str(parquet_dir),
        )
        assert len(result["macro"]) == 336

    def test_unknown_source_raises(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        with pytest.raises(ValueError, match="unknown source"):
            fetch_tiered("BTC-USD", source="bogus", parquet_dir=str(parquet_dir))


class TestNowTsFilter:
    def test_now_ts_excludes_future_bars(self, parquet_dir):
        from services.tiered_history import fetch_tiered
        _write_parquet(parquet_dir, "BTC-USD", 400, start_ts=1_700_000_000)
        cutoff = 1_700_000_000 + 100 * 3600
        result = fetch_tiered(
            "BTC-USD", source="parquet",
            parquet_dir=str(parquet_dir), now_ts=cutoff,
        )
        for tier in ("micro", "meso", "macro"):
            for c in result[tier]:
                assert c["start"] < cutoff, f"{tier} contains future bar"
