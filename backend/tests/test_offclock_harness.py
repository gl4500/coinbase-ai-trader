import os
import pytest
from tools._scorecard import _offclock_harness as och


def test_load_bars_rejects_unknown_substrate():
    with pytest.raises(ValueError, match="substrate"):
        och.load_bars("hourly", "BTC-USD")


def test_load_dollar_bars_missing_file_returns_empty():
    assert och.load_dollar_bars("ZZZ-NONEXISTENT-USD") == []


def test_load_bars_time_delegates_to_history(monkeypatch):
    sentinel = [{"start": 1, "open": 1.0, "high": 1.0, "low": 1.0,
                 "close": 1.0, "volume": 1.0}]
    monkeypatch.setattr(och, "load_history", lambda pid: sentinel)
    assert och.load_bars("time", "BTC-USD") is sentinel


def test_load_dollar_bars_roundtrip(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({
        "start": [60, 0], "open": [2.0, 1.0], "high": [2.0, 1.0],
        "low": [2.0, 1.0], "close": [2.0, 1.0], "volume": [2.0, 1.0],
        "end": [119, 59], "dollar_value": [2.0, 1.0], "n_candles": [1, 1],
    })
    d = tmp_path / "dollar"
    d.mkdir()
    pq.write_table(table, str(d / "BTC-USD.parquet"))
    monkeypatch.setattr(och, "_HISTORY_DIR", str(tmp_path))
    bars = och.load_dollar_bars("BTC-USD")
    assert [b["start"] for b in bars] == [0, 60]   # sorted ascending
    assert bars[0]["close"] == 1.0
