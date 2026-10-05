from tools.alerts.moves import move_alerts
from tools.alerts.notify import should_notify


def test_move_alerts_fire_on_large_moves_in_either_direction():
    a = move_alerts("BTC-USD", now_px=95.0, px_1h=100.0, px_24h=100.0, th_1h=0.05, th_24h=0.10)
    assert [x["kind"] for x in a] == ["move_1h"] and a[0]["change"] < 0
    b = move_alerts("ETH-USD", now_px=112.0, px_1h=110.0, px_24h=100.0, th_1h=0.05, th_24h=0.10)
    assert [x["kind"] for x in b] == ["move_24h"]


def test_no_move_alert_below_thresholds_or_on_missing_prices():
    assert move_alerts("SOL-USD", 102.0, 100.0, 100.0, 0.05, 0.10) == []
    assert move_alerts("XRP-USD", 102.0, None, 0.0, 0.05, 0.10) == []


def test_cooldown_suppresses_repeat_notifications():
    state = {}
    assert should_notify("carry:BIT-30OCT26-CDE", 1000.0, state, cooldown_s=3600) is True
    assert should_notify("carry:BIT-30OCT26-CDE", 2000.0, state, cooldown_s=3600) is False
    assert should_notify("carry:BIT-30OCT26-CDE", 5000.0, state, cooldown_s=3600) is True
    assert should_notify("move_1h:BTC-USD", 2000.0, state, cooldown_s=3600) is True
