"""Tests for services/marketcap_history_cache.py — parquet-backed bronze cache
wrapping coingecko_marketcap.fetch_marketcap_history (#284 / #164b).

Cache semantics:
  * per-pid parquet file under <parquet_dir>/<pid>.parquet
  * schema mirrors tools/build_marketcap_parquet bronze schema:
        start int64 | market_cap f64 | fdv f64 | ingest_ts i64 | schema_version i32
  * cache hit: parquet covers [start_ms, end_ms] AND newest ingest_ts within
    refresh window -> skip API entirely
  * cache miss / stale: call fetch_marketcap_history, merge with existing rows,
    re-write parquet with PIT columns stamped at int(time.time()).
  * public coroutine returns list[(ts_ms, market_cap)] sorted ascending, to
    match coingecko_marketcap.fetch_marketcap_history.
"""
import os
import sys
import time
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)


# ── Public API surface ─────────────────────────────────────────────────────

class TestPublicApi:

    def test_module_exposes_async_get(self):
        from services import marketcap_history_cache as mhc
        assert hasattr(mhc, "fetch_marketcap_history_cached")
        import inspect
        assert inspect.iscoroutinefunction(mhc.fetch_marketcap_history_cached)


# ── Cache miss falls through to underlying fetcher ─────────────────────────

class TestCacheMiss:

    @pytest.mark.asyncio
    async def test_miss_calls_underlying_fetcher(self, tmp_path, monkeypatch):
        from services import marketcap_history_cache as mhc

        called = {"n": 0}

        async def _stub(pid, start_ms, end_ms):
            called["n"] += 1
            return [(1735689600000, 1.0e12), (1735776000000, 1.01e12)]

        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)

        rows = await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735776000000,
            parquet_dir=str(tmp_path),
        )
        assert called["n"] == 1
        assert len(rows) == 2
        assert rows[0][0] == 1735689600000
        assert rows[0][1] == pytest.approx(1.0e12)

    @pytest.mark.asyncio
    async def test_miss_writes_parquet_with_pit_columns(
        self, tmp_path, monkeypatch
    ):
        from services import marketcap_history_cache as mhc
        import pyarrow.parquet as pq

        async def _stub(pid, start_ms, end_ms):
            return [(1735689600000, 1.0e12)]

        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)
        monkeypatch.setattr(mhc.time, "time", lambda: 1778500000.0)

        await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735689600000,
            parquet_dir=str(tmp_path),
        )
        path = os.path.join(str(tmp_path), "BTC-USD.parquet")
        assert os.path.exists(path)
        tbl = pq.read_table(path)
        names = [f.name for f in tbl.schema]
        assert names == [
            "start", "market_cap", "fdv", "ingest_ts", "schema_version"
        ]
        d = tbl.to_pydict()
        assert d["ingest_ts"][0] == 1778500000
        assert d["schema_version"][0] == 1


# ── Cache hit short-circuits API ───────────────────────────────────────────

class TestCacheHit:

    @pytest.mark.asyncio
    async def test_hit_skips_underlying_fetcher(self, tmp_path, monkeypatch):
        from services import marketcap_history_cache as mhc

        # Seed parquet with two rows covering [start, end] and FRESH ingest_ts
        from tools.build_marketcap_parquet import _save_marketcap_history
        path = os.path.join(str(tmp_path), "BTC-USD.parquet")
        now_ts = 1778500000
        _save_marketcap_history(
            path,
            [
                {"start": 1735689600, "market_cap": 1.0e12, "fdv": 1.0e12,
                 "ingest_ts": now_ts, "schema_version": 1},
                {"start": 1735776000, "market_cap": 1.01e12, "fdv": 1.01e12,
                 "ingest_ts": now_ts, "schema_version": 1},
            ],
            now_ts=now_ts,
        )

        called = {"n": 0}
        async def _stub(pid, start_ms, end_ms):
            called["n"] += 1
            return []
        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)
        monkeypatch.setattr(mhc.time, "time", lambda: float(now_ts + 60))

        rows = await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735776000000,
            parquet_dir=str(tmp_path),
        )
        assert called["n"] == 0, "fresh cache must not call API"
        assert len(rows) == 2
        # output is (ts_ms, market_cap) — parquet stored start as epoch secs
        assert rows[0][0] == 1735689600000
        assert rows[1][0] == 1735776000000


# ── Refresh window: stale parquet forces re-fetch ──────────────────────────

class TestRefreshWindow:

    @pytest.mark.asyncio
    async def test_stale_ingest_ts_triggers_refetch(
        self, tmp_path, monkeypatch
    ):
        from services import marketcap_history_cache as mhc
        from tools.build_marketcap_parquet import _save_marketcap_history

        path = os.path.join(str(tmp_path), "BTC-USD.parquet")
        stale_ts = 1700000000  # > 24h before now_ts below
        _save_marketcap_history(
            path,
            [{"start": 1735689600, "market_cap": 1.0e12, "fdv": 1.0e12,
              "ingest_ts": stale_ts, "schema_version": 1}],
            now_ts=stale_ts,
        )

        called = {"n": 0}
        async def _stub(pid, start_ms, end_ms):
            called["n"] += 1
            return [(1735689600000, 1.0e12)]
        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)
        monkeypatch.setattr(mhc.time, "time", lambda: 1778500000.0)

        await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735689600000,
            parquet_dir=str(tmp_path),
            refresh_secs=86400,
        )
        assert called["n"] == 1, "stale cache must trigger refetch"


# ── Range coverage: missing rows trigger refetch even if fresh ─────────────

class TestRangeCoverage:

    @pytest.mark.asyncio
    async def test_partial_coverage_triggers_refetch(
        self, tmp_path, monkeypatch
    ):
        """Parquet has fresh rows but the requested range extends past the
        newest cached `start`. Must re-fetch to fill the gap."""
        from services import marketcap_history_cache as mhc
        from tools.build_marketcap_parquet import _save_marketcap_history

        path = os.path.join(str(tmp_path), "BTC-USD.parquet")
        now_ts = 1778500000
        _save_marketcap_history(
            path,
            [{"start": 1735689600, "market_cap": 1.0e12, "fdv": 1.0e12,
              "ingest_ts": now_ts, "schema_version": 1}],
            now_ts=now_ts,
        )

        called = {"n": 0}
        async def _stub(pid, start_ms, end_ms):
            called["n"] += 1
            return [(1735776000000, 1.01e12)]
        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)
        monkeypatch.setattr(mhc.time, "time", lambda: float(now_ts + 60))

        rows = await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735776000000,  # newer than cached
            parquet_dir=str(tmp_path),
        )
        assert called["n"] == 1, "missing range must trigger refetch"
        # merged rows should include both cached + freshly fetched
        starts = sorted(ts for ts, _ in rows)
        assert 1735689600000 in starts
        assert 1735776000000 in starts


# ── Output range filter ────────────────────────────────────────────────────

class TestRangeFilter:

    @pytest.mark.asyncio
    async def test_returned_rows_filtered_to_requested_range(
        self, tmp_path, monkeypatch
    ):
        from services import marketcap_history_cache as mhc
        from tools.build_marketcap_parquet import _save_marketcap_history

        path = os.path.join(str(tmp_path), "BTC-USD.parquet")
        now_ts = 1778500000
        _save_marketcap_history(
            path,
            [
                {"start": 1735603200, "market_cap": 0.99e12, "fdv": 0.99e12,
                 "ingest_ts": now_ts, "schema_version": 1},
                {"start": 1735689600, "market_cap": 1.0e12, "fdv": 1.0e12,
                 "ingest_ts": now_ts, "schema_version": 1},
                {"start": 1735776000, "market_cap": 1.01e12, "fdv": 1.01e12,
                 "ingest_ts": now_ts, "schema_version": 1},
            ],
            now_ts=now_ts,
        )

        async def _stub(pid, start_ms, end_ms):
            return []
        monkeypatch.setattr(mhc, "fetch_marketcap_history", _stub)
        monkeypatch.setattr(mhc.time, "time", lambda: float(now_ts + 60))

        rows = await mhc.fetch_marketcap_history_cached(
            "BTC-USD",
            start_ms=1735689600000,
            end_ms=1735689600000,
            parquet_dir=str(tmp_path),
        )
        # only one row inside [start_ms, end_ms]
        assert len(rows) == 1
        assert rows[0][0] == 1735689600000
