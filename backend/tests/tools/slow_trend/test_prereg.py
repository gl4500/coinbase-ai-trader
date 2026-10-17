"""Pins every preregistered value. Changing one must be a visible, reviewed diff."""

from tools.slow_trend import prereg as P


def test_frozen_values():
    assert P.PRODUCTS == ("BTC-USD", "ETH-USD")
    assert P.SMA_LEN == 100
    assert P.INITIAL_USD == 1000.0 and P.SLEEVE_USD == 500.0
    assert P.TAKER_FEE == 0.009 and P.MAKER_FEE == 0.005
    assert P.DEV_END == "2025-04-13"
    assert P.BLOCK_START == "2025-04-14" and P.BLOCK_END == "2026-10-02"
    assert P.MAX_MISSING_DAYS == 3
    assert P.DCA_TRANCHES == 52
    assert P.G2_DRAWDOWN_RATIO == 2 / 3
    assert P.MIN_BLOCK_ROUND_TRIPS == 4
    assert P.BOOT_BLOCK_WEEKS == 8 and P.BOOT_SENS_WEEKS == (4, 13)
    assert P.BOOT_RESAMPLES == 10_000 and P.BOOT_SEED == 20261003
    assert P.FREEZE_PATHS == ("backend/tools/slow_trend", "backend/clients/coinbase_client.py")


def test_scenarios():
    s = {x.name: x for x in P.SCENARIOS}
    assert set(s) == {"P0", "S10", "S25", "D1", "SM"}
    assert (s["P0"].entry_fee, s["P0"].exit_fee, s["P0"].slip, s["P0"].delay) == (
        0.009,
        0.009,
        0.0,
        0,
    )
    assert s["S10"].slip == 0.0010 and s["S25"].slip == 0.0025
    assert s["D1"].delay == 1 and s["D1"].slip == 0.0
    assert s["SM"].entry_fee == 0.005 and s["SM"].exit_fee == 0.009
    assert P.GATING_SENSITIVITIES == ("S10", "S25", "D1")
