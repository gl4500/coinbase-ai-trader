"""Per-field merge of CG + CMC MarketcapRow objects into FundamentalsRow."""
from __future__ import annotations

import pytest

from services.coingecko_marketcap import MarketcapRow
from services.leader_fundamentals_merge import (
    MergeOutcome,
    merge_marketcap_rows,
)


def _row(price=100.0, mcap=1e9, fdv=1.5e9, vol=5e7, circ=9e8, total=1.5e9, chg=5.0):
    return MarketcapRow(
        market_cap=mcap, fdv=fdv, circ_supply=circ, total_supply=total,
        ingest_ts=1, schema_version=1,
        price=price, volume_24h=vol, price_chg_24h_pct=chg,
    )


# ── median fields ─────────────────────────────────────────────────────────

def test_circulating_supply_uses_median():
    cg = _row(circ=900_000_000)
    cmc = _row(circ=910_000_000)
    out = merge_marketcap_rows("BTC-USD", cg, cmc, divergence_threshold=0.10)
    assert out.row.circulating_supply == 905_000_000.0


def test_total_supply_uses_median():
    out = merge_marketcap_rows("BTC-USD", _row(total=1.5e9), _row(total=1.51e9),
                               divergence_threshold=0.10)
    assert out.row.total_supply == pytest.approx(1.505e9)


def test_price_uses_median():
    out = merge_marketcap_rows("BTC-USD", _row(price=99.5), _row(price=100.5),
                               divergence_threshold=0.10)
    assert out.row.price == 100.0


def test_price_chg_pct_uses_median():
    out = merge_marketcap_rows("BTC-USD", _row(chg=5.0), _row(chg=6.0),
                               divergence_threshold=0.10)
    assert out.row.price_chg_24h_pct == 5.5


# ── average ──────────────────────────────────────────────────────────────

def test_volume_24h_uses_average():
    out = merge_marketcap_rows("BTC-USD", _row(vol=1e8), _row(vol=2e8),
                               divergence_threshold=0.10)
    assert out.row.volume_24h == 1.5e8


# ── recomputed ───────────────────────────────────────────────────────────

def test_market_cap_recomputed():
    cg = _row(price=100.0, circ=1e9)
    cmc = _row(price=110.0, circ=1.1e9)
    out = merge_marketcap_rows("BTC-USD", cg, cmc, divergence_threshold=0.10)
    assert out.row.market_cap == pytest.approx(105.0 * 1.05e9)


def test_fdv_recomputed():
    cg = _row(price=100.0, total=1.5e9)
    cmc = _row(price=110.0, total=1.6e9)
    out = merge_marketcap_rows("BTC-USD", cg, cmc, divergence_threshold=0.10)
    assert out.row.fdv == pytest.approx(105.0 * 1.55e9)


# ── divergence ───────────────────────────────────────────────────────────

def test_circ_supply_divergence_under_threshold_no_flag():
    out = merge_marketcap_rows("BTC-USD",
                               _row(circ=900_000_000), _row(circ=981_000_000),
                               divergence_threshold=0.10)
    assert "circ_supply" not in out.row.divergence_flags


def test_circ_supply_divergence_over_threshold_flagged():
    out = merge_marketcap_rows("BTC-USD",
                               _row(circ=900_000_000), _row(circ=1_100_000_000),
                               divergence_threshold=0.10)
    assert "circ_supply" in out.row.divergence_flags


def test_audit_entries_record_all_fields():
    out = merge_marketcap_rows("BTC-USD",
                               _row(circ=900_000_000), _row(circ=905_000_000),
                               divergence_threshold=0.10)
    fields = {a["field"] for a in out.audit_entries}
    assert "circ_supply" in fields
    assert "price" in fields


# ── single-source ────────────────────────────────────────────────────────

def test_cg_only_when_cmc_missing():
    out = merge_marketcap_rows("BTC-USD", _row(price=100.0), None,
                               divergence_threshold=0.10)
    assert out.merge_quality == "CG_ONLY"
    assert out.row.price == 100.0
    assert out.row.divergence_flags == []


def test_cmc_only_when_cg_missing():
    out = merge_marketcap_rows("BTC-USD", None, _row(price=100.0),
                               divergence_threshold=0.10)
    assert out.merge_quality == "CMC_ONLY"


def test_both_missing_raises():
    with pytest.raises(ValueError):
        merge_marketcap_rows("BTC-USD", None, None, divergence_threshold=0.10)


# ── raw provider values preserved ────────────────────────────────────────

def test_raw_provider_values_preserved():
    cg = _row(price=99.0, vol=1e8)
    cmc = _row(price=101.0, vol=2e8)
    out = merge_marketcap_rows("BTC-USD", cg, cmc, divergence_threshold=0.10)
    assert out.cg_market_cap == cg.market_cap
    assert out.cmc_market_cap == cmc.market_cap
    assert out.cg_volume_24h == 1e8
    assert out.cmc_volume_24h == 2e8
