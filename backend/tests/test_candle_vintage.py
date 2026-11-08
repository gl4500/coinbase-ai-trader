"""Candle vintage for version-2 labels (F113-1, repair proposal v2).

A stored hourly bar is only as good as the fetch that wrote it: a fetch that
lands before the exchange settles the bar stores a provisional snapshot, and
clock maturity alone cannot tell the two apart. Each stored bar now carries
`requested_at` -- the local request START of the fetch that wrote it -- and a
label may use a bar only when that request began at least FINALITY_MARGIN_SECS
after the bar closed.

This is a policy gate, not proof of finality: it assumes the response was a
fresh aggregate (pinned below), and every label records that basis.

Every test runs against a temp database or pure functions. Nothing here touches
a real database or the exchange.
"""

import json
import math
import os
import sqlite3
import sys
from pathlib import Path

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import inspect  # noqa: E402

import pytest  # noqa: E402

import database  # noqa: E402
from clients import coinbase_client as cb  # noqa: E402
from services import diagnostics  # noqa: E402
from services import outcome_labels as ol  # noqa: E402

HOUR = 3600
_BOUNDARY = 1_767_225_600
_ENTRY = ol.entry_candle_start(_BOUNDARY)
_EXIT = ol.exit_candle_start(_ENTRY)
_TARGET = ol.target_time(_ENTRY)
_MARGIN = ol.FINALITY_MARGIN_SECS


def _final_at(start):
    """Earliest request start the gate accepts for the bar stamped `start`."""
    return start + HOUR + _MARGIN


def _candle(start, open_, close, requested_at):
    return {
        "start_time": start,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "volume": 1.0,
        "requested_at": requested_at,
    }


def _bar(start, open_, close):
    """Exchange-shaped bar, as `save_candles` receives it."""
    return {
        "start": start,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "volume": 1.0,
    }


_SETTLED = object()


def _settled_book(exit_close=101.0, exit_requested_at=_SETTLED):
    return {
        _ENTRY: _candle(_ENTRY, 100.0, 100.0, _final_at(_ENTRY)),
        _EXIT: _candle(
            _EXIT,
            100.0,
            exit_close,
            _final_at(_EXIT) if exit_requested_at is _SETTLED else exit_requested_at,
        ),
    }


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    return database


# ── (i) Codex counterexample: clock maturity is not finality ──────────────────


_COUNTEREXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "pr113_provisional_label_counterexample_v1.json"
)


def test_codex_counterexample_snapshot_before_close_is_not_resolved():
    """Snapshot obtained at 15000 for an exit bar closing at 18000, resolved at
    18001. Version 2 as reviewed RESOLVED it as a WIN that the final bar would
    have scored a LOSS. With the vintage gate it must not resolve at all."""
    ce = json.loads(_COUNTEREXAMPLE.read_text())
    frame = ce["stored_snapshot_resolution"]
    entry, exit_ = frame["entry_candle_start"], frame["exit_candle_start"]
    book = {
        entry: _candle(entry, frame["entry_price"], frame["entry_price"], _final_at(entry)),
        exit_: _candle(
            exit_, frame["entry_price"], frame["target_price"], ce["exit_snapshot_obtained_at"]
        ),
    }
    r = ol.resolve(
        signal_time=entry - 1, side="BUY", candles=book, now=ce["resolution_now"], attempts=0
    )
    assert r.status == "PENDING"
    assert r.outcome is None
    assert r.reason == "unverified_exit_vintage"


# ── (ii) Delayed write of a provisional snapshot ──────────────────────────────


async def test_provisional_snapshot_written_after_close_is_still_not_final(db):
    """The fetch started before the close but its write landed long after.
    The vintage is the request start, never the write time."""
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0)], requested_at=_final_at(_ENTRY))
    await db.save_candles(
        "BTC-USD", [_bar(_EXIT, 100.0, 101.0)], requested_at=_TARGET - 30, now=_TARGET + 10 * HOUR
    )
    book = await db.get_candles_at("BTC-USD", [_ENTRY, _EXIT])
    assert book[_EXIT]["requested_at"] == _TARGET - 30
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_TARGET + 10 * HOUR)
    assert r.status == "PENDING"
    assert r.reason == "unverified_exit_vintage"


def test_entry_bar_is_gated_too():
    """The entry price is the entry bar's OPEN, which a late first trade can
    still revise, so the entry bar takes the same gate."""
    book = _settled_book()
    book[_ENTRY] = _candle(_ENTRY, 100.0, 100.0, _ENTRY + HOUR - 5)
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_TARGET + HOUR)
    assert r.status == "PENDING"
    assert r.reason == "unverified_entry_vintage"


# ── (iii) Stale overwrite refused / (iv) equal-timestamp conflict ─────────────


async def test_an_older_fetch_landing_late_cannot_overwrite_a_newer_vintage(db):
    await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=_final_at(_EXIT))
    await db.save_candles(
        "BTC-USD", [_bar(_EXIT, 100.0, 101.0)], requested_at=_TARGET - 30, now=_TARGET + HOUR
    )
    row = (await db.get_candles_at("BTC-USD", [_EXIT]))[_EXIT]
    assert row["close"] == 105.0
    assert row["requested_at"] == _final_at(_EXIT)


async def test_a_newer_fetch_replaces_the_provisional_bar(db):
    await db.save_candles(
        "BTC-USD", [_bar(_EXIT, 100.0, 101.0)], requested_at=_TARGET - 30, now=_TARGET
    )
    await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=_final_at(_EXIT))
    row = (await db.get_candles_at("BTC-USD", [_EXIT]))[_EXIT]
    assert row["close"] == 105.0
    assert row["requested_at"] == _final_at(_EXIT)


async def test_a_vintage_write_replaces_a_legacy_null_row(db):
    import aiosqlite

    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            "INSERT INTO candles (product_id,start_time,open,high,low,close,volume) "
            "VALUES ('BTC-USD',?,100,101,99,101,1)",
            (_EXIT,),
        )
        await conn.commit()
    await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=_final_at(_EXIT))
    row = (await db.get_candles_at("BTC-USD", [_EXIT]))[_EXIT]
    assert row["close"] == 105.0


async def test_equal_timestamp_conflict_keeps_the_first_value_and_is_reported(db):
    t = _final_at(_EXIT)
    assert await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=t) == 0
    conflicts = await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 99.0)], requested_at=t)
    assert conflicts == 1
    row = (await db.get_candles_at("BTC-USD", [_EXIT]))[_EXIT]
    assert row["close"] == 105.0


async def test_equal_timestamp_identical_values_is_not_a_conflict(db):
    t = _final_at(_EXIT)
    await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=t)
    assert await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 105.0)], requested_at=t) == 0


# ── (v) Invalid requested_at raises before any write ──────────────────────────


@pytest.mark.parametrize(
    "bad",
    [None, math.nan, math.inf, -math.inf, True, False, 0, -1.0, "1767240000"],
    ids=["none", "nan", "inf", "-inf", "true", "false", "zero", "negative", "string"],
)
async def test_invalid_requested_at_raises_and_writes_nothing(db, bad):
    with pytest.raises(ValueError):
        await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 101.0)], requested_at=bad)
    assert await db.get_candles_at("BTC-USD", [_EXIT]) == {}


async def test_requested_at_beyond_the_skew_allowance_raises(db):
    now = _TARGET + HOUR
    with pytest.raises(ValueError):
        await db.save_candles(
            "BTC-USD",
            [_bar(_EXIT, 100.0, 101.0)],
            requested_at=now + database.REQUESTED_AT_FUTURE_SKEW_SECS + 0.5,
            now=now,
        )
    assert await db.get_candles_at("BTC-USD", [_EXIT]) == {}


async def test_requested_at_within_the_skew_allowance_is_written(db):
    now = _TARGET + HOUR
    await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 101.0)], requested_at=now + 1.0, now=now)
    assert (await db.get_candles_at("BTC-USD", [_EXIT]))[_EXIT]["requested_at"] == now + 1.0


async def test_requested_at_is_required(db):
    with pytest.raises(TypeError):
        await db.save_candles("BTC-USD", [_bar(_EXIT, 100.0, 101.0)])


@pytest.mark.parametrize(
    "bad",
    [None, math.nan, math.inf, True, 0, -5.0, "x"],
    ids=["none", "nan", "inf", "bool", "zero", "negative", "string"],
)
def test_gate_rejects_an_invalid_stored_requested_at(bad):
    book = _settled_book(exit_requested_at=bad)
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_TARGET + 10 * HOUR)
    assert r.status == "PENDING"
    assert r.reason == "unverified_exit_vintage"


def test_strict_replay_rejects_a_requested_at_later_than_now():
    """As-of consistency: a resolver replayed at `now` cannot use a fetch that
    had not started yet. The write-side skew allowance does not apply here."""
    req = _final_at(_EXIT) + 2
    book = _settled_book(exit_requested_at=req)
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=req - 1)
    assert r.status == "PENDING"
    assert r.reason == "unverified_exit_vintage"


# ── (vi) Legacy NULL vintage ──────────────────────────────────────────────────


def test_legacy_null_vintage_is_pending_then_unavailable_at_give_up():
    book = _settled_book()
    book[_EXIT].pop("requested_at")
    early = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=book, now=_TARGET + HOUR)
    assert (early.status, early.reason) == ("PENDING", "unverified_exit_vintage")
    late = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=book,
        now=_TARGET + HOUR,
        attempts=ol.MAX_RESOLVE_ATTEMPTS - 1,
    )
    assert (late.status, late.reason) == ("UNAVAILABLE", "unverified_exit_vintage")
    assert late.outcome is None


# ── (vii) Margin boundary ─────────────────────────────────────────────────────


def test_request_starting_exactly_at_close_plus_margin_passes():
    req = _final_at(_EXIT)
    r = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=_settled_book(exit_requested_at=req),
        now=req,
    )
    assert r.status == "RESOLVED"


def test_request_starting_one_second_earlier_fails():
    req = _final_at(_EXIT) - 1
    r = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=_settled_book(exit_requested_at=req),
        now=req + HOUR,
    )
    assert r.status == "PENDING"
    assert r.reason == "unverified_exit_vintage"


def test_margin_is_a_named_policy_constant_not_a_measured_one():
    assert ol.FINALITY_MARGIN_SECS == 300
    assert "margin_unverified" in ol.FINALITY_BASIS
    assert "assumes_fresh_aggregate" in ol.FINALITY_BASIS
    assert f"close+{ol.FINALITY_MARGIN_SECS}s" in ol.FINALITY_BASIS
    assert "verified_final" not in ol.FINALITY_BASIS


# ── (viii) finality_basis persisted on every RESOLVED v2 label ────────────────


def test_resolution_carries_the_finality_basis():
    r = ol.resolve(signal_time=_BOUNDARY, side="BUY", candles=_settled_book(), now=_TARGET + HOUR)
    assert r.status == "RESOLVED"
    assert r.finality_basis == ol.FINALITY_BASIS


def test_non_resolved_results_carry_no_basis():
    r = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=_settled_book(exit_requested_at=_TARGET - 1),
        now=_TARGET + HOUR,
    )
    assert r.finality_basis is None


async def test_tracker_persists_the_finality_basis(db, monkeypatch):
    from services import outcome_tracker as ot

    await db.insert_signal_outcome(
        {
            "source": "CNN",
            "product_id": "BTC-USD",
            "side": "BUY",
            "confidence": 0.7,
            "entry_price": 100.0,
            "indicators_json": "{}",
            "check_after": _BOUNDARY + ol.H_BARS * HOUR,
            "signal_time": _BOUNDARY,
        }
    )
    row = (await db.get_pending_outcomes())[-1]
    await db.save_candles(
        "BTC-USD",
        [_bar(_ENTRY, 100.0, 100.0), _bar(_EXIT, 100.0, 102.0)],
        requested_at=_final_at(_EXIT),
    )
    assert await ot.get_tracker().check_pending(now=_final_at(_EXIT)) == 1
    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["finality_basis"] == ol.FINALITY_BASIS


def test_diagnostics_group_v2_counts_by_finality_basis(tmp_path):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE signal_outcomes (source TEXT, side TEXT, label_version INTEGER, "
        "outcome TEXT, signed_return REAL, pct_change REAL, confidence REAL, "
        "target_time REAL, check_after REAL, created_at TEXT, finality_basis TEXT)"
    )
    rows = [
        ("WIN", 0.02, ol.FINALITY_BASIS),
        ("LOSS", -0.02, ol.FINALITY_BASIS),
        ("WIN", 0.01, None),  # resolved before the gate existed
        ("UNAVAILABLE", None, None),
    ]
    for outcome, ret, basis in rows:
        conn.execute(
            "INSERT INTO signal_outcomes VALUES ('CNN','BUY',2,?,?,NULL,0.7,0,0,"
            "'2026-01-01T00:00:00+00:00',?)",
            (outcome, ret, basis),
        )
    edge = diagnostics.signal_edge(conn, None, now=1.0)
    assert edge["by_finality_basis"] == {ol.FINALITY_BASIS: 2, "none_recorded": 1}


# ── (ix) Parity: client and feature readers unchanged ─────────────────────────


def test_get_candles_signature_is_unchanged():
    params = list(inspect.signature(cb.get_candles).parameters)
    assert params == ["product_id", "granularity", "limit"]


async def test_get_candles_wraps_the_vintage_fetch_and_drops_the_field(monkeypatch):
    bars = [_bar(_ENTRY, 100.0, 100.0)]

    async def _fake(product_id, granularity="ONE_HOUR", limit=100):
        return bars, 123.0

    monkeypatch.setattr(cb, "get_candles_with_vintage", _fake)
    assert await cb.get_candles("BTC-USD", "ONE_HOUR", 100) == bars


async def test_vintage_fetch_stamps_the_request_start_before_sending(monkeypatch):
    clock = iter([1000.0, 2000.0, 3000.0])
    monkeypatch.setattr(cb.time, "time", lambda: next(clock))
    sent_at = {}

    async def _get(path, params):
        sent_at["end"] = params["end"]
        return {
            "candles": [
                {"start": "3600", "open": "1", "high": "1", "low": "1", "close": "1", "volume": "1"}
            ]
        }

    monkeypatch.setattr(cb, "_get", _get)
    candles, requested_at = await cb.get_candles_with_vintage("BTC-USD", "ONE_HOUR", 100)
    assert requested_at == 1000.0
    assert sent_at["end"] == "1000"
    assert candles[0]["start"] == 3600
    assert "requested_at" not in candles[0]


async def test_vintage_fetch_failure_returns_no_candles(monkeypatch):
    async def _boom(path, params):
        raise RuntimeError("down")

    monkeypatch.setattr(cb, "_get", _boom)
    candles, requested_at = await cb.get_candles_with_vintage("BTC-USD", "ONE_HOUR", 100)
    assert candles == []
    assert isinstance(requested_at, float)


async def test_feature_reader_only_gains_the_vintage_column(db):
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0)], requested_at=_final_at(_ENTRY))
    (row,) = await db.get_candles("BTC-USD")
    assert set(row) == {
        "id",
        "product_id",
        "start_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "requested_at",
    }


async def test_scanner_writes_candles_with_their_request_start():
    from unittest.mock import AsyncMock, patch

    from agents.market_scanner import MarketScanner

    product = {
        "product_id": "XRP-USD",
        "quote_currency_id": "USD",
        "base_currency_id": "XRP",
        "display_name": "XRP-USD",
        "status": "online",
        "trading_disabled": False,
        "volume_24h": "5000000",
        "price": "1.33",
        "price_percentage_change_24h": "0.5",
    }
    bars = [_bar(_ENTRY, 1.0, 1.0)]
    saver = AsyncMock()
    with (
        patch(
            "agents.market_scanner.coinbase_client.get_products",
            new=AsyncMock(return_value=[product]),
        ),
        patch(
            "agents.market_scanner.coinbase_client.get_best_bid_ask",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "agents.market_scanner.coinbase_client.get_candles_with_vintage",
            new=AsyncMock(return_value=(bars, 4242.0)),
        ),
        patch("agents.market_scanner.database.upsert_product", new=AsyncMock()),
        patch("agents.market_scanner.database.get_products", new=AsyncMock(return_value=[])),
        patch("agents.market_scanner.database.save_candles", new=saver),
        patch("agents.market_scanner.asyncio.sleep", new=AsyncMock()),
    ):
        await MarketScanner()._do_scan()
    saver.assert_awaited_once_with("XRP-USD", bars, requested_at=4242.0)


# ── Assumption pinned, not hidden ─────────────────────────────────────────────


def test_assumption_a_stale_cached_payload_with_a_late_request_is_accepted():
    """KNOWN LIMIT: the gate trusts that a response requested after
    close+margin is a fresh aggregate. A cache serving the provisional
    snapshot to a late request passes. This test documents that limit; it is
    why every label records `assumes_fresh_aggregate`."""
    stale_close = 101.0  # the provisional value, served late from a cache
    r = ol.resolve(
        signal_time=_BOUNDARY,
        side="BUY",
        candles=_settled_book(exit_close=stale_close, exit_requested_at=_final_at(_EXIT)),
        now=_final_at(_EXIT),
    )
    assert r.status == "RESOLVED"
    assert r.target_price == stale_close
    assert "assumes_fresh_aggregate" in r.finality_basis


def test_diagnostics_tolerate_a_db_without_the_basis_column():
    """Diagnostics read DB copies and snapshots read-only; one that predates the
    migration has no finality_basis column and must report, not crash."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE signal_outcomes (source TEXT, side TEXT, label_version INTEGER, "
        "outcome TEXT, signed_return REAL, pct_change REAL, confidence REAL, "
        "target_time REAL, check_after REAL, created_at TEXT)"
    )
    conn.execute(
        "INSERT INTO signal_outcomes VALUES ('CNN','BUY',2,'WIN',0.02,NULL,0.7,0,0,"
        "'2026-01-01T00:00:00+00:00')"
    )
    edge = diagnostics.signal_edge(conn, None, now=1.0)
    assert edge["n"] == 1
    assert edge["by_finality_basis"] == {"column_absent": 1}
