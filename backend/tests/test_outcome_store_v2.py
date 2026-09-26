"""Persistence for label-version-2 outcomes: additive migration, provenance
columns, and idempotent resolution.

Spec: docs/specs/2026-09-26-outcome-label-contract.md
Every test runs against a temp database. Nothing here touches a real database.
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
from services import outcome_labels as ol  # noqa: E402

HOUR = 3600
_BOUNDARY = 1_767_225_600


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    return database


async def _columns(db):
    import aiosqlite

    async with aiosqlite.connect(db.DB_PATH) as conn:
        cur = await conn.execute("PRAGMA table_info(signal_outcomes)")
        return {r[1] for r in await cur.fetchall()}


# ── Additive migration ────────────────────────────────────────────────────────


async def test_migration_adds_provenance_columns(db):
    cols = await _columns(db)
    for expected in (
        "label_version",
        "target_time",
        "entry_candle_start",
        "exit_candle_start",
        "entry_price_v2",
        "target_price",
        "signed_return",
        "price_observed_at",
        "processed_at",
        "price_source",
        "resolve_attempts",
        "unresolved_reason",
    ):
        assert expected in cols, f"missing column {expected}"


async def test_migration_preserves_the_legacy_columns(db):
    """Additive only — v1 columns must survive untouched."""
    cols = await _columns(db)
    for legacy in (
        "entry_price",
        "exit_price",
        "pct_change",
        "outcome",
        "lesson_text",
        "check_after",
        "checked_at",
        "created_at",
    ):
        assert legacy in cols


async def test_init_db_is_idempotent(db):
    await db.init_db()
    await db.init_db()
    cols = await _columns(db)
    assert "label_version" in cols


# ── Recording ─────────────────────────────────────────────────────────────────


async def _insert(db, *, side="BUY", signal_time=_BOUNDARY):
    await db.insert_signal_outcome(
        {
            "source": "CNN",
            "product_id": "BTC-USD",
            "side": side,
            "confidence": 0.7,
            "entry_price": 100.0,
            "indicators_json": "{}",
            "check_after": signal_time + ol.H_BARS * HOUR,
            "signal_time": signal_time,
        }
    )
    rows = await db.get_pending_outcomes()
    return rows[-1]


async def test_new_rows_carry_the_label_version_and_target_schedule(db):
    row = await _insert(db)
    entry = ol.entry_candle_start(_BOUNDARY)
    assert row["label_version"] == ol.LABEL_VERSION
    assert row["entry_candle_start"] == entry
    assert row["exit_candle_start"] == entry + 3 * HOUR
    assert row["target_time"] == entry + 4 * HOUR
    assert row["resolve_attempts"] == 0
    assert row["outcome"] is None


async def test_insert_defaults_signal_time_to_now(db):
    """The production caller may omit signal_time; the default path must work.

    Regression: the first implementation referenced a module alias that was
    never imported, so this path raised NameError. Every other test passed
    signal_time explicitly and missed it; ruff's F821 caught it.
    """
    import time as _t

    before = _t.time()
    await db.insert_signal_outcome(
        {
            "source": "CNN",
            "product_id": "BTC-USD",
            "side": "BUY",
            "confidence": 0.5,
            "entry_price": 100.0,
            "indicators_json": "{}",
            "check_after": before + ol.H_BARS * HOUR,
        }
    )
    row = (await db.get_pending_outcomes())[-1] if await db.get_pending_outcomes() else None
    if row is None:
        import aiosqlite

        async with aiosqlite.connect(db.DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            cur = await conn.execute("SELECT * FROM signal_outcomes ORDER BY id DESC LIMIT 1")
            row = dict(await cur.fetchone())

    assert row["label_version"] == ol.LABEL_VERSION
    assert row["entry_candle_start"] == ol.entry_candle_start(before)
    assert row["target_time"] == ol.target_time(ol.entry_candle_start(before))


async def test_recorded_entry_price_is_left_as_the_scan_quote(db):
    """entry_price stays the quote the signal saw; the v2 reference is separate."""
    row = await _insert(db)
    assert row["entry_price"] == 100.0
    assert row["entry_price_v2"] is None


# ── Candle lookup ─────────────────────────────────────────────────────────────


async def test_get_candles_at_returns_only_requested_buckets(db):
    await db.save_candles(
        "BTC-USD",
        [
            {"start": _BOUNDARY, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            {
                "start": _BOUNDARY + HOUR,
                "open": 2.0,
                "high": 2.0,
                "low": 2.0,
                "close": 2.0,
                "volume": 1.0,
            },
            {
                "start": _BOUNDARY + 2 * HOUR,
                "open": 3.0,
                "high": 3.0,
                "low": 3.0,
                "close": 3.0,
                "volume": 1.0,
            },
        ],
    )
    got = await db.get_candles_at("BTC-USD", [_BOUNDARY, _BOUNDARY + 2 * HOUR])
    assert set(got) == {_BOUNDARY, _BOUNDARY + 2 * HOUR}
    assert got[_BOUNDARY + 2 * HOUR]["close"] == 3.0


async def test_get_candles_at_missing_bucket_is_simply_absent(db):
    got = await db.get_candles_at("BTC-USD", [_BOUNDARY])
    assert got == {}


# ── Idempotent resolution ─────────────────────────────────────────────────────


async def test_resolution_writes_all_provenance(db):
    row = await _insert(db)
    entry = ol.entry_candle_start(_BOUNDARY)
    changed = await db.resolve_signal_outcome_v2(
        row_id=row["id"],
        outcome="WIN",
        signed_return=0.02,
        entry_price_v2=100.0,
        target_price=102.0,
        price_observed_at=entry + 4 * HOUR,
        price_source=ol.PRICE_SOURCE_LOCAL_CANDLES,
        lesson_text="test lesson",
    )
    assert changed is True

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["signed_return"] == 0.02
    assert stored["entry_price_v2"] == 100.0
    assert stored["target_price"] == 102.0
    assert stored["price_observed_at"] == entry + 4 * HOUR
    assert stored["price_source"] == ol.PRICE_SOURCE_LOCAL_CANDLES
    assert stored["processed_at"] is not None
    assert stored["label_version"] == ol.LABEL_VERSION


async def test_a_second_resolution_cannot_overwrite_a_completed_label(db):
    """Retries must never rewrite a finished label."""
    row = await _insert(db)
    await db.resolve_signal_outcome_v2(
        row_id=row["id"],
        outcome="WIN",
        signed_return=0.02,
        entry_price_v2=100.0,
        target_price=102.0,
        price_observed_at=1,
        price_source="local_candles",
        lesson_text="first",
    )
    changed = await db.resolve_signal_outcome_v2(
        row_id=row["id"],
        outcome="LOSS",
        signed_return=-0.09,
        entry_price_v2=1.0,
        target_price=0.5,
        price_observed_at=2,
        price_source="local_candles",
        lesson_text="second",
    )
    assert changed is False

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"
    assert stored["signed_return"] == 0.02


async def test_resolved_rows_leave_the_pending_queue(db):
    row = await _insert(db)
    await db.resolve_signal_outcome_v2(
        row_id=row["id"],
        outcome="NEUTRAL",
        signed_return=0.0,
        entry_price_v2=100.0,
        target_price=100.0,
        price_observed_at=1,
        price_source="local_candles",
        lesson_text="x",
    )
    pending_ids = {r["id"] for r in await db.get_pending_outcomes()}
    assert row["id"] not in pending_ids


# ── Retries and the terminal unavailable state ────────────────────────────────


async def test_attempts_increment_without_resolving(db):
    row = await _insert(db)
    await db.bump_signal_outcome_attempts(row["id"])
    await db.bump_signal_outcome_attempts(row["id"])
    stored = await db.get_signal_outcome(row["id"])
    assert stored["resolve_attempts"] == 2
    assert stored["outcome"] is None


async def test_unavailable_is_terminal_and_records_the_reason(db):
    row = await _insert(db)
    changed = await db.mark_signal_outcome_unavailable(row["id"], "missing_exit_candle")
    assert changed is True

    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "UNAVAILABLE"
    assert stored["unresolved_reason"] == "missing_exit_candle"
    assert stored["processed_at"] is not None

    pending_ids = {r["id"] for r in await db.get_pending_outcomes()}
    assert row["id"] not in pending_ids


async def test_unavailable_cannot_clobber_an_already_resolved_label(db):
    row = await _insert(db)
    await db.resolve_signal_outcome_v2(
        row_id=row["id"],
        outcome="WIN",
        signed_return=0.02,
        entry_price_v2=100.0,
        target_price=102.0,
        price_observed_at=1,
        price_source="local_candles",
        lesson_text="x",
    )
    changed = await db.mark_signal_outcome_unavailable(row["id"], "missing_exit_candle")
    assert changed is False
    stored = await db.get_signal_outcome(row["id"])
    assert stored["outcome"] == "WIN"


# ── Legacy rows are left alone ────────────────────────────────────────────────


async def test_legacy_rows_keep_null_label_version_and_are_not_rewritten(db):
    """A v1 row (no label_version) must stay exactly as it was."""
    import aiosqlite

    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            """INSERT INTO signal_outcomes
                 (source, product_id, side, confidence, entry_price, exit_price,
                  pct_change, outcome, lesson_text, check_after, checked_at, created_at)
               VALUES ('CNN','BTC-USD','BUY',0.6,100.0,101.0,0.01,'WIN','legacy',
                       1.0,'2026-05-01T00:00:00Z','2026-05-01T00:00:00Z')""",
        )
        await conn.commit()
        cur = await conn.execute("SELECT id FROM signal_outcomes WHERE lesson_text='legacy'")
        legacy_id = (await cur.fetchone())[0]

    stored = await db.get_signal_outcome(legacy_id)
    assert stored["label_version"] is None
    assert stored["outcome"] == "WIN"
    assert stored["signed_return"] is None
    assert stored["target_time"] is None


@pytest.mark.parametrize("version", [None, 1, 3])
async def test_current_resolver_cannot_modify_other_versions(db, version):
    import aiosqlite

    row = await _insert(db)
    async with aiosqlite.connect(db.DB_PATH) as conn:
        await conn.execute(
            "UPDATE signal_outcomes SET label_version=? WHERE id=?", (version, row["id"])
        )
        await conn.commit()
    before = await db.get_signal_outcome(row["id"])
    assert row["id"] not in {r["id"] for r in await db.get_pending_outcomes()}
    assert not await db.resolve_signal_outcome_v2(
        row["id"], "WIN", 0.02, 100.0, 102.0, 1, "local_candles", "incorrect conversion"
    )
    assert not await db.mark_signal_outcome_unavailable(row["id"], "missing_exit_candle")
    await db.bump_signal_outcome_attempts(row["id"])
    assert await db.get_signal_outcome(row["id"]) == before
