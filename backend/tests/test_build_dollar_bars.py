import os

import pytest
from tools.build_dollar_bars import candle_dollar_value, calibrate_threshold


def test_candle_dollar_value_uses_typical_price():
    # typical price = (high + low + close) / 3 = (110 + 90 + 100) / 3 = 100
    c = {"start": 0, "open": 95.0, "high": 110.0, "low": 90.0,
         "close": 100.0, "volume": 4.0}
    assert candle_dollar_value(c) == pytest.approx(400.0)  # 4 * 100


def test_calibrate_threshold_is_total_over_1h_count():
    # 3 candles, each dollar value 300 => total 900; 1h count 3 => threshold 300
    candles = [
        {"start": i, "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 3.0}
        for i in range(3)
    ]
    assert calibrate_threshold(candles, n_1h_bars=3) == pytest.approx(300.0)


def test_calibrate_threshold_rejects_nonpositive_bar_count():
    with pytest.raises(ValueError, match="n_1h_bars"):
        calibrate_threshold([], n_1h_bars=0)


from tools.build_dollar_bars import dollar_bars_from_candles


def _flat_candle(start, price, vol):
    """A candle with open=high=low=close=price, so typical price == price."""
    return {"start": start, "open": price, "high": price, "low": price,
            "close": price, "volume": vol}


def test_dollar_bars_basic_boundaries():
    # 6 flat candles, dollar value 300 each; threshold 900 => a bar every 3.
    candles = [_flat_candle(i * 60, 100.0, 3.0) for i in range(6)]
    bars = dollar_bars_from_candles(candles, threshold=900.0)
    assert len(bars) == 2
    assert bars[0]["start"] == 0
    assert bars[0]["end"] == 120          # 3rd candle's start (i=2)
    assert bars[0]["n_candles"] == 3
    assert bars[0]["volume"] == pytest.approx(9.0)
    assert bars[0]["dollar_value"] == pytest.approx(900.0)
    assert bars[1]["start"] == 180


def test_dollar_bars_trailing_partial_dropped():
    # 7 candles => 2 full bars; the 7th (300 < 900) is an incomplete bar.
    candles = [_flat_candle(i * 60, 100.0, 3.0) for i in range(7)]
    bars = dollar_bars_from_candles(candles, threshold=900.0)
    assert len(bars) == 2


def test_dollar_bars_ohlc_aggregation():
    candles = [
        {"start": 0,   "open": 100.0, "high": 110.0, "low": 95.0,
         "close": 105.0, "volume": 10.0},
        {"start": 60,  "open": 105.0, "high": 120.0, "low": 100.0,
         "close": 115.0, "volume": 10.0},
        {"start": 120, "open": 115.0, "high": 118.0, "low": 90.0,
         "close": 92.0,  "volume": 10.0},
    ]
    # total dollar value ~3150; threshold 3000 => all 3 candles -> one bar.
    bars = dollar_bars_from_candles(candles, threshold=3000.0)
    assert len(bars) == 1
    assert bars[0]["open"] == 100.0
    assert bars[0]["high"] == 120.0
    assert bars[0]["low"] == 90.0
    assert bars[0]["close"] == 92.0
    assert bars[0]["n_candles"] == 3


def test_dollar_bars_empty_input():
    assert dollar_bars_from_candles([], threshold=100.0) == []


from tools.build_dollar_bars import build_dollar_bars_for_candles


def test_build_assembly_clips_to_1h_window():
    # 1h window covers starts 100..400; 1m candles include out-of-window ones
    # on both sides (50 before, 500 after) that must be clipped out.
    one_h = [{"start": 100}, {"start": 400}]  # n_1h_bars = 2
    one_min = (
        [_flat_candle(50, 100.0, 3.0)]                               # before window
        + [_flat_candle(s, 100.0, 3.0) for s in range(100, 460, 60)]  # 100..400, in window
        + [_flat_candle(500, 100.0, 3.0)]                            # after window
    )
    bars = build_dollar_bars_for_candles(one_min, one_h)
    # Only the 6 candles with 100 <= start <= 400 count: dollar value 6*300=1800;
    # threshold = 1800 / 2 = 900 => 2 bars.
    assert len(bars) == 2
    assert all(100 <= b["start"] <= 400 for b in bars)
    assert all(100 <= b["end"] <= 400 for b in bars)


def test_build_assembly_empty_1h_returns_empty():
    assert build_dollar_bars_for_candles([_flat_candle(0, 100.0, 3.0)], []) == []


def test_build_assembly_no_1m_in_window_returns_empty():
    one_h = [{"start": 1000}, {"start": 2000}]
    one_min = [_flat_candle(0, 100.0, 3.0), _flat_candle(60, 100.0, 3.0)]
    assert build_dollar_bars_for_candles(one_min, one_h) == []


from tools import build_dollar_bars as bdb


def test_dollar_parquet_path_uses_dollar_subdir():
    p = bdb._dollar_parquet_path("BTC-USD")
    assert p.endswith(os.path.join("dollar", "BTC-USD.parquet"))


def test_save_and_reload_dollar_bars_parquet_roundtrip(tmp_path):
    bars = [
        {"start": 0, "end": 120, "open": 100.0, "high": 110.0, "low": 90.0,
         "close": 105.0, "volume": 9.0, "dollar_value": 900.0, "n_candles": 3},
    ]
    path = str(tmp_path / "BTC-USD.parquet")
    bdb._save_dollar_bars(path, bars)
    import pyarrow.parquet as pq
    rows = pq.read_table(path).to_pydict()
    assert rows["start"] == [0]
    assert rows["n_candles"] == [3]
    assert rows["dollar_value"][0] == pytest.approx(900.0)


def test_build_for_pid_writes_parquet(tmp_path, monkeypatch):
    one_h = [{"start": 0}, {"start": 600}]  # n_1h_bars = 2
    one_min = [
        {"start": s, "open": 100.0, "high": 100.0, "low": 100.0,
         "close": 100.0, "volume": 3.0}
        for s in range(0, 660, 60)  # 11 candles in [0, 600]
    ]
    monkeypatch.setattr(bdb, "load_1m_history", lambda pid: one_min)
    monkeypatch.setattr(bdb, "load_history", lambda pid: one_h)
    monkeypatch.setattr(bdb, "_dollar_parquet_path",
                        lambda pid: str(tmp_path / f"{pid}.parquet"))
    result = bdb.build_for_pid("BTC-USD")
    assert result["pid"] == "BTC-USD"
    assert result["n_bars"] > 0
    assert (tmp_path / "BTC-USD.parquet").exists()


def test_build_for_pid_no_data_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bdb, "load_1m_history", lambda pid: [])
    monkeypatch.setattr(bdb, "load_history", lambda pid: [])
    monkeypatch.setattr(bdb, "_dollar_parquet_path",
                        lambda pid: str(tmp_path / f"{pid}.parquet"))
    result = bdb.build_for_pid("BTC-USD")
    assert result["n_bars"] == 0
    assert not (tmp_path / "BTC-USD.parquet").exists()
