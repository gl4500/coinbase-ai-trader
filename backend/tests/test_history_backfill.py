"""
Tests for services/history_backfill.py — 5-minute candle support (#55).

Existing hourly backfill is untouched. These tests cover the new 5m path:
separate parquet location, FIVE_MINUTE granularity, correct pagination window,
and incremental re-fetch behaviour.

No live API calls — `_fetch_range` is mocked with AsyncMock in every async test.
No real file I/O — `_HISTORY_DIR` is monkey-patched to a tmp directory.
"""
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")

from services import history_backfill as hb  # noqa: E402


def _make_5m_candle(start_ts: int, close: float = 100.0) -> dict:
    return {
        "start":  start_ts,
        "open":   close,
        "high":   close + 0.5,
        "low":    close - 0.5,
        "close":  close,
        "volume": 10.0,
    }


# ── Parquet path separation ───────────────────────────────────────────────────

class TestFiveMinuteParquetPath:

    def test_5m_path_under_5m_subfolder(self):
        p = hb._parquet_path_5m("BTC-USD")
        assert os.path.basename(os.path.dirname(p)) == "5m", \
            f"5m parquet should live under .../history/5m/, got {p}"

    def test_5m_path_distinct_from_hourly(self):
        assert hb._parquet_path_5m("BTC-USD") != hb._parquet_path("BTC-USD"), \
            "5m and hourly parquet paths must not collide"

    def test_5m_path_encodes_product_id(self):
        assert "BTC-USD" in hb._parquet_path_5m("BTC-USD")


# ── load_5m_history ───────────────────────────────────────────────────────────

class TestLoad5mHistory:

    def test_returns_empty_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        assert hb.load_5m_history("BTC-USD") == []

    def test_roundtrip_5m_candles(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        candles = [_make_5m_candle(1_700_000_000 + i * 300, 100.0 + i) for i in range(5)]
        hb._save_to_path(hb._parquet_path_5m("BTC-USD"), candles)

        loaded = hb.load_5m_history("BTC-USD")
        assert len(loaded) == 5
        assert [c["start"] for c in loaded] == [c["start"] for c in candles]
        assert all(c["close"] == e["close"] for c, e in zip(loaded, candles))

    def test_5m_load_does_not_see_hourly_file(self, tmp_path, monkeypatch):
        """Writing the hourly parquet must not leak into load_5m_history."""
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        hourly = [_make_5m_candle(1_700_000_000 + i * 3600) for i in range(3)]
        hb._save_to_path(hb._parquet_path("BTC-USD"), hourly)

        assert hb.load_5m_history("BTC-USD") == []


# ── backfill_product_5m ───────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestBackfillProduct5m:

    async def test_uses_five_minute_granularity(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        # One page of candles, then empty — terminates the loop
        pages = [[_make_5m_candle(1_700_000_000 + i * 300) for i in range(10)], []]
        mock_fetch = AsyncMock(side_effect=pages)

        with patch.object(hb, "_fetch_range", mock_fetch):
            await hb.backfill_product_5m("BTC-USD", days=1)

        # First call's granularity kwarg must be FIVE_MINUTE
        first_call = mock_fetch.call_args_list[0]
        assert first_call.kwargs.get("granularity") == "FIVE_MINUTE", \
            f"backfill_product_5m must request FIVE_MINUTE, got {first_call}"

    async def test_writes_to_5m_path_not_hourly(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        pages = [[_make_5m_candle(1_700_000_000 + i * 300) for i in range(10)], []]
        mock_fetch = AsyncMock(side_effect=pages)

        with patch.object(hb, "_fetch_range", mock_fetch):
            result = await hb.backfill_product_5m("BTC-USD", days=1)

        assert os.path.exists(hb._parquet_path_5m("BTC-USD")), \
            "5m parquet file must exist after backfill"
        assert not os.path.exists(hb._parquet_path("BTC-USD")), \
            "hourly parquet file must NOT be written by 5m backfill"
        assert result["new_bars"] == 10

    async def test_uses_5m_pagination_window(self, tmp_path, monkeypatch):
        """Page window should be _MAX_PER_REQ * 300s = 90000s, not 1 080 000s."""
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        mock_fetch = AsyncMock(side_effect=[[], []])

        with patch.object(hb, "_fetch_range", mock_fetch):
            await hb.backfill_product_5m("BTC-USD", days=1)

        call = mock_fetch.call_args_list[0]
        # positional: (pid, start_ts, end_ts)
        start_ts = call.args[1]
        end_ts   = call.args[2]
        window   = end_ts - start_ts
        expected_max = hb._MAX_PER_REQ * 300
        assert window <= expected_max, \
            f"5m window {window}s exceeds expected max {expected_max}s (hourly math leaked)"

    async def test_incremental_skips_known_bars(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hb, "_HISTORY_DIR", str(tmp_path))
        os.makedirs(os.path.join(str(tmp_path), "5m"), exist_ok=True)
        existing = [_make_5m_candle(1_700_000_000 + i * 300) for i in range(3)]
        hb._save_to_path(hb._parquet_path_5m("BTC-USD"), existing)

        # Fetch returns the same three bars — all should be dedup-filtered
        mock_fetch = AsyncMock(side_effect=[list(existing), []])
        with patch.object(hb, "_fetch_range", mock_fetch):
            result = await hb.backfill_product_5m("BTC-USD", days=1)

        assert result["new_bars"] == 0, "known bars must not be counted as new"
        assert result["total_bars"] == 3


# ── _fetch_range granularity plumbing ─────────────────────────────────────────

@pytest.mark.asyncio
class TestFetchRangeGranularityParam:

    async def test_fetch_range_accepts_granularity_kwarg(self):
        """_fetch_range must accept an explicit granularity kwarg without TypeError.

        The body may fail (no network in tests), but we only need the signature
        to accept the kwarg so callers like backfill_product_5m can pass it.
        """
        # Should not raise TypeError: unexpected keyword argument
        result = await hb._fetch_range(
            "BTC-USD", 1_700_000_000, 1_700_000_900,
            granularity="FIVE_MINUTE",
        )
        # network failure returns [] from the existing except block; that's fine
        assert isinstance(result, list)
