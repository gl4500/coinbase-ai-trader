import pytest
from tools import backfill_1m_candles as b1m


def test_days_to_cover_computes_from_1h_first_ts(monkeypatch):
    # 1h history starting 10 days before now_ts
    now_ts = 10_000_000
    first_ts = now_ts - 10 * 86400
    monkeypatch.setattr(b1m, "load_history",
                        lambda pid: [{"start": first_ts}, {"start": now_ts}])
    assert b1m._days_to_cover("BTC-USD", now_ts) == 10


def test_days_to_cover_rounds_up_partial_day(monkeypatch):
    now_ts = 10_000_000
    first_ts = now_ts - (5 * 86400 + 100)  # 5 days + a bit
    monkeypatch.setattr(b1m, "load_history", lambda pid: [{"start": first_ts}])
    assert b1m._days_to_cover("BTC-USD", now_ts) == 6


def test_days_to_cover_no_parquet_returns_zero(monkeypatch):
    monkeypatch.setattr(b1m, "load_history", lambda pid: [])
    assert b1m._days_to_cover("BTC-USD", 10_000_000) == 0


def test_resolve_pids_from_explicit_arg():
    pids = b1m._resolve_pids("ignored.pt", "BTC-USD, ETH-USD ,SOL-USD")
    assert pids == ["BTC-USD", "ETH-USD", "SOL-USD"]
