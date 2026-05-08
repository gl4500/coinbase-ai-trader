"""TDD tests for Bronze PIT tagging on parquet pulls (#168).

The bronze layer (raw candles in `backend/data/history/`) writes new
columns `ingest_ts` (Unix-seconds when the row was first persisted) and
`schema_version` (a constant int that bumps whenever the bronze schema
evolves). PIT semantics: existing rows preserve their original
ingest_ts across rewrites — only newly-introduced rows get stamped
with the current time.

No live API calls. `_HISTORY_DIR` is monkey-patched to a tmp path.
"""
from __future__ import annotations

import os
import sys

import pyarrow.parquet as pq
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from services import history_backfill as hb  # noqa: E402


def _candle(start_ts: int, close: float = 100.0) -> dict:
    return {
        "start": start_ts,
        "open": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 10.0,
    }


class TestSchemaConstant:

    def test_schema_version_defined_and_positive(self):
        assert isinstance(hb._SCHEMA_VERSION, int)
        assert hb._SCHEMA_VERSION >= 1


class TestPersistedColumns:

    def test_new_write_emits_ingest_ts_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        candles = [_candle(1_700_000_000 + i * 300) for i in range(3)]
        hb._save_to_path(hb._parquet_path_5m("BTC-USD"), candles, now_ts=1_700_009_999)

        table = pq.read_table(hb._parquet_path_5m("BTC-USD"))
        assert "ingest_ts" in table.column_names
        ingest = table.column("ingest_ts").to_pylist()
        assert all(t == 1_700_009_999 for t in ingest), ingest

    def test_new_write_emits_schema_version_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        candles = [_candle(1_700_000_000 + i * 300) for i in range(3)]
        hb._save_to_path(hb._parquet_path_5m("BTC-USD"), candles, now_ts=1_700_009_999)

        table = pq.read_table(hb._parquet_path_5m("BTC-USD"))
        assert "schema_version" in table.column_names
        sv = table.column("schema_version").to_pylist()
        assert all(v == hb._SCHEMA_VERSION for v in sv), sv


class TestPITPreservation:

    def test_existing_rows_keep_their_original_ingest_ts(self, tmp_path, monkeypatch):
        """Rewriting parquet must not re-stamp ingest_ts on rows that
        were already persisted. New rows added on the second write
        receive the second write's now_ts."""
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        path = hb._parquet_path_5m("BTC-USD")

        first_batch = [_candle(1_700_000_000 + i * 300) for i in range(3)]
        hb._save_to_path(path, first_batch, now_ts=1_000_001)

        # Reload, then add a new candle and save again with a later now_ts
        loaded = hb._load_from_path(path)
        new_candle = _candle(1_700_000_900 + 4 * 300)
        merged = loaded + [new_candle]
        hb._save_to_path(path, merged, now_ts=2_000_002)

        table = pq.read_table(path)
        rows = table.to_pylist()
        rows.sort(key=lambda r: r["start"])
        # First three rows were in the original batch → original ingest_ts
        for r in rows[:3]:
            assert r["ingest_ts"] == 1_000_001, r
        # Last row was newly added → second-write ingest_ts
        assert rows[-1]["ingest_ts"] == 2_000_002


class TestBackwardsCompat:

    def test_old_parquet_without_pit_columns_still_loads(
        self, tmp_path, monkeypatch,
    ):
        """A parquet file written before #168 has no ingest_ts /
        schema_version columns. Loading must not crash; the returned
        dicts may omit those keys (or carry them as None / 0)."""
        import pyarrow as pa

        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        path = hb._parquet_path_5m("OLD-USD")

        # Write the legacy 6-column schema directly (skipping _save_to_path)
        legacy_schema = pa.schema([
            pa.field("start", pa.int64()),
            pa.field("open", pa.float64()),
            pa.field("high", pa.float64()),
            pa.field("low", pa.float64()),
            pa.field("close", pa.float64()),
            pa.field("volume", pa.float64()),
        ])
        legacy_table = pa.table(
            {
                "start": [1, 2, 3],
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [10.0, 11.0, 12.0],
            },
            schema=legacy_schema,
        )
        pq.write_table(legacy_table, path, compression="snappy")

        loaded = hb._load_from_path(path)
        assert len(loaded) == 3
        assert [c["start"] for c in loaded] == [1, 2, 3]


class TestLoadHydratesPITFields:

    def test_loaded_candle_carries_ingest_ts(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        path = hb._parquet_path_5m("BTC-USD")

        candles = [_candle(1_700_000_000)]
        hb._save_to_path(path, candles, now_ts=12345)

        loaded = hb._load_from_path(path)
        assert loaded[0]["ingest_ts"] == 12345
        assert loaded[0]["schema_version"] == hb._SCHEMA_VERSION
