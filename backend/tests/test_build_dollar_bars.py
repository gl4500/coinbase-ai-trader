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
