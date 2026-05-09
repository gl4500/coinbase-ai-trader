"""Pure-function ranker — floor + multiplier + top-N."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from services.leader_ranker import (
    FundamentalsRow,
    LeaderRankerConfig,
    RankerResult,
    ScoreRow,
    rank_leaders,
    supply_multiplier,
    load_config_from_env,
)


def _row(pid, *, price=100.0, mcap=1e9, fdv=1.5e9, vol=5e7, circ=9e8,
         total=1.5e9, chg=5.0, flags=None):
    return FundamentalsRow(
        product_id=pid, price=price, market_cap=mcap, fdv=fdv,
        volume_24h=vol, circulating_supply=circ, total_supply=total,
        price_chg_24h_pct=chg, divergence_flags=flags or [],
    )


CFG = LeaderRankerConfig(
    min_market_cap_usd=50_000_000,
    min_volume_24h_usd=5_000_000,
    max_fdv_mcap_ratio=5.0,
    supply_divergence_threshold=0.10,
    top_n=3,
)


# ── supply_multiplier tier boundaries ─────────────────────────────────────

@pytest.mark.parametrize("ratio,expected", [
    (0.95, 1.00), (0.90, 1.00),
    (0.89, 0.85), (0.50, 0.85),
    (0.49, 0.70), (0.20, 0.70),
    (0.19, 0.50), (0.05, 0.50),
])
def test_supply_multiplier_tiers(ratio, expected):
    assert supply_multiplier(ratio) == expected


def test_supply_multiplier_zero_total_returns_lowest_tier():
    assert supply_multiplier(0.0) == 0.50


# ── floor enforcement ────────────────────────────────────────────────────

def test_floor_excludes_low_market_cap():
    snap = {"TINY-USD": _row("TINY-USD", mcap=10_000_000)}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["TINY-USD"].reason == "FLOOR_MCAP"
    assert "TINY-USD" not in r.top_pids


def test_floor_excludes_low_volume():
    snap = {"DRY-USD": _row("DRY-USD", vol=1_000_000)}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["DRY-USD"].reason == "FLOOR_VOL"


def test_floor_excludes_high_fdv_mcap_ratio():
    snap = {"LOCK-USD": _row("LOCK-USD", mcap=1e9, fdv=1e10)}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["LOCK-USD"].reason == "FLOOR_FDV"


def test_divergence_flag_excludes_row():
    snap = {"BAD-USD": _row("BAD-USD", flags=["circulating_supply"])}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["BAD-USD"].reason == "DIVERGENCE"


# ── score formula ────────────────────────────────────────────────────────

def test_score_uses_abs_price_change():
    pump = _row("PUMP-USD", chg=10.0)
    dump = _row("DUMP-USD", chg=-10.0)
    r = rank_leaders({"PUMP-USD": pump, "DUMP-USD": dump}, CFG,
                     snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["PUMP-USD"].raw_score == pytest.approx(r.scored["DUMP-USD"].raw_score)


def test_score_formula_is_abs_pct_times_vol_over_mcap():
    snap = {"X-USD": _row("X-USD", chg=8.0, vol=1e8, mcap=1e9)}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["X-USD"].raw_score == pytest.approx(0.8)


# ── top-N selection ──────────────────────────────────────────────────────

def test_top_n_returns_highest_scores_first():
    snap = {
        "A-USD": _row("A-USD", chg=2.0,  vol=1e7),   # 0.02
        "B-USD": _row("B-USD", chg=10.0, vol=2e8),   # 2.0
        "C-USD": _row("C-USD", chg=5.0,  vol=1e8),   # 0.5
        "D-USD": _row("D-USD", chg=8.0,  vol=5e7),   # 0.4
    }
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.top_pids == ["B-USD", "C-USD", "D-USD"]


def test_pids_outside_top_n_marked_ranked_out():
    snap = {f"X{i}-USD": _row(f"X{i}-USD", chg=float(i + 1)) for i in range(5)}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert len(r.top_pids) == 3
    ranked_out = [p for p, s in r.scored.items() if s.reason == "RANKED_OUT"]
    assert len(ranked_out) == 2


def test_top_n_respects_config_value():
    cfg5 = LeaderRankerConfig(**{**CFG.__dict__, "top_n": 5})
    snap = {f"X{i}-USD": _row(f"X{i}-USD", chg=float(i + 1)) for i in range(7)}
    r = rank_leaders(snap, cfg5, snapshot_ts=datetime.now(timezone.utc))
    assert len(r.top_pids) == 5


# ── supply multiplier integrated into final score ────────────────────────

def test_final_score_applies_supply_multiplier():
    row = _row("VEST-USD", circ=4e8, total=1e9, chg=10.0, vol=1e8)
    r = rank_leaders({"VEST-USD": row}, CFG, snapshot_ts=datetime.now(timezone.utc))
    s = r.scored["VEST-USD"]
    assert s.multiplier == 0.70
    assert s.final_score == pytest.approx(s.raw_score * 0.70)


# ── scored dict completeness ─────────────────────────────────────────────

def test_scored_dict_includes_every_input_pid():
    snap = {
        "PASS-USD":  _row("PASS-USD"),
        "FLOOR-USD": _row("FLOOR-USD", mcap=1_000_000),
        "DIV-USD":   _row("DIV-USD", flags=["price"]),
    }
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert set(r.scored.keys()) == {"PASS-USD", "FLOOR-USD", "DIV-USD"}


def test_passing_singleton_yields_pass_reason():
    snap = {"OK-USD": _row("OK-USD")}
    r = rank_leaders(snap, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.scored["OK-USD"].reason == "PASS"


def test_empty_snapshot_returns_empty_result():
    r = rank_leaders({}, CFG, snapshot_ts=datetime.now(timezone.utc))
    assert r.top_pids == []
    assert r.scored == {}


# ── env config ───────────────────────────────────────────────────────────

def test_load_config_from_env_uses_defaults_when_unset(monkeypatch):
    for k in ("LEADER_MIN_MARKET_CAP_USD", "LEADER_MIN_VOLUME_24H_USD",
              "LEADER_MAX_FDV_MCAP_RATIO", "LEADER_SUPPLY_DIVERGENCE_THRESHOLD",
              "LEADER_TOP_N"):
        monkeypatch.delenv(k, raising=False)
    cfg = load_config_from_env()
    assert cfg.min_market_cap_usd == 50_000_000
    assert cfg.top_n == 3


def test_load_config_from_env_overrides_each_field(monkeypatch):
    monkeypatch.setenv("LEADER_MIN_MARKET_CAP_USD", "100000000")
    monkeypatch.setenv("LEADER_TOP_N", "5")
    cfg = load_config_from_env()
    assert cfg.min_market_cap_usd == 100_000_000
    assert cfg.top_n == 5
