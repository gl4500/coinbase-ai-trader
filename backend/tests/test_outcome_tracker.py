"""OutcomeTracker resolution under the label-version-2 contract.

Spec: docs/specs/2026-09-26-outcome-label-contract.md

`outcome_tracker.py` had no test coverage before this file. These tests pin the
timing defect the 2026-09-26 audit found: resolution must use the price at the
defined target time, never the price that happens to be current when the
resolver runs, and must never reach the network for it.
"""

import os
import sys

BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import pytest  # noqa: E402

import database  # noqa: E402
from clients import coinbase_client as cb  # noqa: E402
from services import outcome_labels as ol  # noqa: E402
from services import outcome_tracker as ot  # noqa: E402

HOUR = 3600
_BOUNDARY = 1_767_225_600
_ENTRY = ol.entry_candle_start(_BOUNDARY)
_EXIT = ol.exit_candle_start(_ENTRY)
_TARGET = ol.target_time(_ENTRY)


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    return database


def _bar(start, open_, close):
    return {
        "start": start,
        "open": open_,
        "high": max(open_, close),
        "low": min(open_, close),
        "close": close,
        "volume": 1.0,
    }


async def _pending_row(db, side="BUY", signal_time=_BOUNDARY):
    await db.insert_signal_outcome(
        {
            "source": "CNN",
            "product_id": "BTC-USD",
            "side": side,
            "confidence": 0.7,
            "entry_price": 999.0,  # the scan quote, deliberately unlike the bar
            "indicators_json": "{}",
            "check_after": signal_time + ol.H_BARS * HOUR,
            "signal_time": signal_time,
        }
    )
    return (await db.get_pending_outcomes())[-1]


@pytest.fixture(autouse=True)
def no_live_price_fallback(monkeypatch):
    """Both version-1 price paths must be gone.

    Patched on the client and database modules themselves, so the assertion
    holds no matter which module reaches for them.
    """

    async def _boom(*a, **k):
        raise AssertionError("resolver used a live price instead of the target bar")

    monkeypatch.setattr(cb, "get_candles", _boom)
    monkeypatch.setattr(database, "get_product", _boom)
    return None


# ── The audit's defect ────────────────────────────────────────────────────────


async def test_delayed_resolution_uses_the_target_bar_not_the_latest_bar(db):
    """A row processed 200h late must be labelled from the target bar."""
    row = await _pending_row(db)
    await db.save_candles(
        "BTC-USD",
        [
            _bar(_ENTRY, 100.0, 100.0),
            _bar(_EXIT, 101.0, 102.0),  # target bar: +2% -> WIN
            _bar(_EXIT + 200 * HOUR, 50.0, 50.0),  # a much later, much worse bar
        ],
    )

    resolved = await ot.get_tracker().check_pending(now=_TARGET + 200 * HOUR)
    assert resolved == 1

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["target_price"] == 102.0
    assert stored["entry_price_v2"] == 100.0
    assert stored["signed_return"] == pytest.approx(0.02)
    assert stored["price_observed_at"] == _TARGET
    assert stored["label_version"] == ol.LABEL_VERSION
    # The scan quote is preserved, not overwritten.
    assert stored["entry_price"] == 999.0


async def test_sell_direction_is_scored_as_a_short(db):
    row = await _pending_row(db, side="SELL")
    await db.save_candles(
        "BTC-USD",
        [
            _bar(_ENTRY, 100.0, 100.0),
            _bar(_EXIT, 98.0, 97.0),  # price fell 3% -> SELL wins
        ],
    )
    await ot.get_tracker().check_pending(now=_TARGET)

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["signed_return"] == pytest.approx(0.03)


async def test_not_matured_rows_are_untouched(db):
    row = await _pending_row(db)
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0), _bar(_EXIT, 101.0, 101.0)])

    resolved = await ot.get_tracker().check_pending(now=_TARGET - 1)
    assert resolved == 0
    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] is None


# ── Missing data and retries ──────────────────────────────────────────────────


async def test_missing_target_bar_leaves_the_row_unresolved_and_counts_an_attempt(db):
    row = await _pending_row(db)
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0)])  # no exit bar

    resolved = await ot.get_tracker().check_pending(now=_TARGET)
    assert resolved == 0

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] is None
    assert stored["resolve_attempts"] == 1
    assert stored["target_price"] is None


async def test_repeated_failures_end_as_unavailable(db):
    row = await _pending_row(db)
    tracker = ot.get_tracker()
    for _ in range(ol.MAX_RESOLVE_ATTEMPTS):
        await tracker.check_pending(now=_TARGET)

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "UNAVAILABLE"
    assert stored["unresolved_reason"] == "missing_entry_candle"
    assert stored["target_price"] is None


async def test_a_backfilled_bar_still_resolves_correctly_after_earlier_failures(db):
    """Retry must produce the target-time label, not a now-price label."""
    row = await _pending_row(db)
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0)])
    await ot.get_tracker().check_pending(now=_TARGET)  # fails, exit bar absent

    await db.save_candles("BTC-USD", [_bar(_EXIT, 101.0, 90.0)])  # backfill: -10%
    resolved = await ot.get_tracker().check_pending(now=_TARGET + 5 * HOUR)
    assert resolved == 1

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "LOSS"
    assert stored["target_price"] == 90.0


# ── Idempotency ───────────────────────────────────────────────────────────────


async def test_reprocessing_does_not_duplicate_or_rewrite(db):
    row = await _pending_row(db)
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0), _bar(_EXIT, 101.0, 102.0)])

    import aiosqlite

    async def _count():
        async with aiosqlite.connect(db.DB_PATH) as conn:
            cur = await conn.execute("SELECT COUNT(*) FROM signal_outcomes")
            return (await cur.fetchone())[0]

    before = await _count()
    assert await ot.get_tracker().check_pending(now=_TARGET) == 1
    assert await ot.get_tracker().check_pending(now=_TARGET + 99 * HOUR) == 0
    assert await _count() == before

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["target_price"] == 102.0


# ── Honest lesson text ────────────────────────────────────────────────────────


async def test_lesson_text_states_the_real_horizon_not_a_false_4h_claim(db):
    row = await _pending_row(db)
    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0), _bar(_EXIT, 101.0, 102.0)])
    await ot.get_tracker().check_pending(now=_TARGET + 300 * HOUR)

    stored = await db.get_signal_outcome(row["id"])
    lesson = stored["lesson_text"]
    assert "4h" in lesson  # the horizon is 4 bars, and that is now true
    assert "WIN" in lesson
    # It must describe the measured window, not the processing delay.
    assert "300" not in lesson


# ── Legacy pending rows ───────────────────────────────────────────────────────


async def test_legacy_pending_row_without_target_time_is_still_resolvable(db):
    """v1 rows that never resolved have no target_time; derive it from
    check_after rather than stranding them."""
    import aiosqlite

    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            """INSERT INTO signal_outcomes
                 (source, product_id, side, confidence, entry_price,
                  indicators_json, check_after, created_at)
               VALUES ('CNN','BTC-USD','BUY',0.5,999.0,'{}',?,'2026-01-01T00:00:00Z')""",
            (_BOUNDARY + ol.H_BARS * HOUR,),
        )
        await conn.commit()

    await db.save_candles("BTC-USD", [_bar(_ENTRY, 100.0, 100.0), _bar(_EXIT, 101.0, 102.0)])
    resolved = await ot.get_tracker().check_pending(now=_TARGET + HOUR)
    assert resolved == 1

    rows = [r for r in await _all_rows(db) if r["lesson_text"]]
    assert rows[0]["outcome"] == "WIN"
    assert rows[0]["label_version"] == ol.LABEL_VERSION


async def _all_rows(db):
    import aiosqlite

    async with aiosqlite.connect(db.DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        cur = await conn.execute("SELECT * FROM signal_outcomes")
        return [dict(r) for r in await cur.fetchall()]
