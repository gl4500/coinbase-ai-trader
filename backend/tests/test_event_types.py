"""Tests for typed event payload dataclasses."""
import json

import pytest

from services import event_types as et


def test_price_tick_roundtrip():
    tick = et.PriceTickPayload(pid="BTC-USD", price=95123.45, bid=95120.0, ask=95125.0,
                               volume_24h=12345.67, source="ws")
    blob = json.dumps(tick.to_dict())
    restored = et.PriceTickPayload.from_dict(json.loads(blob))
    assert restored == tick


def test_candle_close_requires_tier():
    with pytest.raises(ValueError, match="tier"):
        et.CandleClosePayload(
            pid="BTC-USD", tier="bogus", open=1, high=2, low=0.5,
            close=1.5, volume=10, bar_ts_ms=1716595200000,
        )


def test_signal_scored_carries_input_event_ids():
    s = et.SignalScoredPayload(
        pid="BTC-USD", model="v4_5", model_version="h168_2026-05-23",
        feature_hash="abcdef", scores={"p_up": 0.6, "p_down": 0.2, "p_neutral": 0.2},
        side="BUY", strength=0.6, regime="TRENDING", deployment_profile_id=None,
        input_event_ids={"last_price_tick_id": 100, "last_candle_close_id": 99},
    )
    blob = s.to_dict()
    assert blob["input_event_ids"]["last_price_tick_id"] == 100


def test_trade_decided_requires_signal_event_id():
    with pytest.raises(ValueError, match="signal_event_id"):
        et.TradeDecidedPayload(
            pid="BTC-USD", model="v4_5", side="BUY", size=0.001, size_usd=95.0,
            intended_entry_price=95000.0, actual_entry_price=95000.0,
            fee_paid=0.57, trigger="SCAN", signal_event_id=None,
            deployment_profile_id=None, trade_uid="v4_5_BTC-USD_x",
        )


def test_exit_triggered_requires_trigger_price_event_id():
    with pytest.raises(ValueError, match="trigger_price_event_id"):
        et.ExitTriggeredPayload(
            pid="BTC-USD", trade_uid="x", trigger_type="WS_TRAIL_STOP",
            peak_pnl_pct=1.0, current_pnl_pct=0.5, exit_threshold=0.7,
            price_at_trigger=100.0, trigger_price_event_id=None,
        )


def test_event_type_enum_includes_all_seven():
    assert set(et.EVENT_TYPES) == {
        "price_tick", "candle_close", "marketcap_snapshot",
        "signal_scored", "trade_decided", "trade_closed", "exit_triggered",
    }
