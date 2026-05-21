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
