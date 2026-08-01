"""Tests for universe-curation logic."""

from __future__ import annotations

from tools.strategy_discovery import curate_universe as cu


def test_rank_by_market_cap_descending():
    """rank_by_market_cap returns pids sorted by descending mc."""
    candidates = {
        "BTC-USD": {
            "market_cap": 1.4e12,
            "volume_24h": 5e10,
            "circulating": 19.7e6,
            "total": 19.7e6,
        },
        "DOGE-USD": {"market_cap": 2e10, "volume_24h": 1e9, "circulating": 140e9, "total": 140e9},
        "ETH-USD": {"market_cap": 4e11, "volume_24h": 2e10, "circulating": 120e6, "total": 120e6},
    }
    ranked = cu.rank_by_market_cap(candidates)
    assert ranked == ["BTC-USD", "ETH-USD", "DOGE-USD"]


def test_rank_by_fdv_mc_ratio_descending_excludes_already_picked():
    """rank_by_fdv_mc_ratio excludes the already-picked pids."""
    candidates = {
        "A": {
            "market_cap": 1e9,
            "volume_24h": 1e7,
            "circulating": 100.0,
            "total": 1000.0,
        },  # ratio 10x
        "B": {
            "market_cap": 1e9,
            "volume_24h": 1e7,
            "circulating": 500.0,
            "total": 1000.0,
        },  # ratio 2x
        "C": {
            "market_cap": 1e9,
            "volume_24h": 1e7,
            "circulating": 200.0,
            "total": 1000.0,
        },  # ratio 5x
    }
    ranked = cu.rank_by_fdv_mc_ratio(candidates, exclude={"A"})
    assert ranked == ["C", "B"]


def test_rank_by_turnover_ascending_excludes_already_picked():
    """rank_by_turnover sorts ascending (LOW turnover wins)."""
    candidates = {
        "A": {
            "market_cap": 1e9,
            "volume_24h": 1e8,
            "circulating": 1.0,
            "total": 1.0,
        },  # turnover 0.1
        "B": {
            "market_cap": 1e9,
            "volume_24h": 1e6,
            "circulating": 1.0,
            "total": 1.0,
        },  # turnover 0.001
        "C": {
            "market_cap": 1e9,
            "volume_24h": 5e7,
            "circulating": 1.0,
            "total": 1.0,
        },  # turnover 0.05
    }
    ranked = cu.rank_by_turnover(candidates, exclude={"B"})
    assert ranked == ["C", "A"]


def test_curate_universe_picks_target_counts_with_overlap_handling():
    """End-to-end: given 100 candidates, picks 15+15+10+10 distinct pids."""
    candidates = {}
    for i in range(100):
        mc = (100 - i) * 1e9  # descending mc by i
        vol = 1e6 if i < 20 else 1e8  # first 20 have unusually low turnover
        ratio_circ = 100.0 if i >= 80 else 500.0  # last 20 have high fdv/mc ratio
        candidates[f"P{i:03d}"] = {
            "market_cap": mc,
            "volume_24h": vol,
            "circulating": ratio_circ,
            "total": 1000.0,
        }

    picked = cu.curate_universe(
        candidates,
        n_large=15,
        n_mid=15,
        n_high_fdv_ratio=10,
        n_low_turnover=10,
    )

    assert len(picked) == 4  # 4 cohorts
    total = sum(len(v) for v in picked.values())
    assert total == 15 + 15 + 10 + 10  # 50 expected, no overlap forces drop here
    large_mid = set(picked["large"]) | set(picked["mid"])
    assert large_mid == {f"P{i:03d}" for i in range(30)}
    assert all(int(p[1:]) >= 80 for p in picked["high_fdv_ratio"])
    # low_turnover drawn from remaining pool: large+mid+high_fdv excluded (i<30 and i>=80..89)
    # lowest-turnover after exclusion are P030..P039 (same vol=1e8 as rest, but highest mc)
    assert all(int(p[1:]) >= 30 and int(p[1:]) < 40 for p in picked["low_turnover"])


def test_curate_universe_respects_overlap_no_pid_listed_twice():
    """Same pid can't appear in two cohorts — overlap forces second cohort to skip."""
    candidates = {
        "A": {"market_cap": 5e9, "volume_24h": 1e6, "circulating": 100.0, "total": 1000.0},
        "B": {"market_cap": 4e9, "volume_24h": 1e7, "circulating": 200.0, "total": 1000.0},
        "C": {"market_cap": 3e9, "volume_24h": 1e8, "circulating": 500.0, "total": 1000.0},
        "D": {"market_cap": 2e9, "volume_24h": 1e9, "circulating": 700.0, "total": 1000.0},
        "E": {"market_cap": 1e9, "volume_24h": 1e10, "circulating": 900.0, "total": 1000.0},
    }
    picked = cu.curate_universe(
        candidates,
        n_large=1,
        n_mid=1,
        n_high_fdv_ratio=1,
        n_low_turnover=1,
    )
    all_picks = [p for cohort in picked.values() for p in cohort]
    assert len(all_picks) == len(set(all_picks))  # no duplicates
