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


from tools._scorecard._offclock_harness import direction_label


def test_direction_label_up():
    closes = [100.0, 101.0, 102.0, 103.0, 104.0]
    label, exit_close = direction_label(closes, t=0, k=4)
    assert label == 1
    assert exit_close == 104.0


def test_direction_label_down():
    closes = [100.0, 99.0, 98.0, 97.0, 96.0]
    label, exit_close = direction_label(closes, t=0, k=4)
    assert label == 0
    assert exit_close == 96.0


def test_direction_label_flat_is_zero():
    closes = [100.0, 100.0, 100.0]
    label, exit_close = direction_label(closes, t=0, k=2)
    assert label == 0          # not strictly greater
    assert exit_close == 100.0


from tools._scorecard._offclock_harness import triple_barrier_label


def _ohlc(start, o, h, l, c):
    return {"start": start, "open": o, "high": h, "low": l, "close": c,
            "volume": 1.0}


def test_triple_barrier_upper_hit():
    # entry close 100; upper barrier 101. Bar 2 highs to 101.5 -> UP.
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 100.5, 99.8, 100.2),
        _ohlc(2, 100.2, 101.5, 100.1, 101.0),
        _ohlc(3, 101.0, 101.2, 100.9, 101.0),
        _ohlc(4, 101.0, 101.1, 100.8, 100.9),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(101.0)   # entry * 1.01


def test_triple_barrier_lower_hit():
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 100.1, 98.5, 99.0),     # low 98.5 <= 99.0 barrier
        _ohlc(2, 99.0, 99.2, 98.8, 99.0),
        _ohlc(3, 99.0, 99.1, 98.9, 99.0),
        _ohlc(4, 99.0, 99.1, 98.9, 99.0),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 0
    assert exit_close == pytest.approx(99.0)    # entry * 0.99


def test_triple_barrier_timeout_uses_close_direction():
    # neither barrier hit within k; close[t+k]=100.4 > entry -> label 1
    bars = [_ohlc(i, 100.0, 100.3, 99.8, 100.0 + 0.1 * i) for i in range(5)]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(100.4)


def test_triple_barrier_both_hit_close_breaks_tie():
    # bar 1 hits both barriers; close 101.0 >= entry -> UP
    bars = [
        _ohlc(0, 100.0, 100.0, 100.0, 100.0),
        _ohlc(1, 100.0, 102.0, 98.0, 101.0),
        _ohlc(2, 101.0, 101.0, 101.0, 101.0),
        _ohlc(3, 101.0, 101.0, 101.0, 101.0),
        _ohlc(4, 101.0, 101.0, 101.0, 101.0),
    ]
    label, exit_close = triple_barrier_label(bars, t=0, k=4)
    assert label == 1
    assert exit_close == pytest.approx(101.0)


from tools._scorecard._offclock_harness import build_product_samples


def _rising_bars(n):
    """n bars with strictly rising close so direction labels are all 1."""
    return [
        {"start": i * 60, "open": 100.0 + 0.01 * i, "high": 100.0 + 0.01 * i,
         "low": 100.0 + 0.01 * i, "close": 100.0 + 0.01 * i, "volume": 1.0}
        for i in range(n)
    ]


def test_build_product_samples_shape_and_count():
    bars = _rising_bars(400)
    s = build_product_samples(bars, "direction", k=4, sample_step=24)
    # samples roll at t in range(336, 396, 24) -> 336, 360, 384 => 3 samples
    assert s["X"].shape == (3, 150)
    assert len(s["y"]) == 3
    assert list(s["y"]) == [1, 1, 1]            # rising closes
    assert s["entry_ts"][0] == 336 * 60


def test_build_product_samples_too_short_returns_empty():
    bars = _rising_bars(338)                    # < 336 + k + 1 for k=4
    s = build_product_samples(bars, "direction", k=4, sample_step=24)
    assert s["X"].shape == (0, 150)
    assert len(s["y"]) == 0


def test_build_product_samples_rejects_unknown_variant():
    bars = _rising_bars(400)
    with pytest.raises(ValueError, match="label_variant"):
        build_product_samples(bars, "regression", k=4, sample_step=24)
