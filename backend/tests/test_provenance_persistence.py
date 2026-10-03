"""Provenance part 2: model/config identity persisted on new scan and trade rows."""

import asyncio
import importlib
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_DIGEST = "sha256:" + "d" * 64


@pytest.fixture
def db(tmp_path):
    import database

    importlib.reload(database)
    database.DB_PATH = str(tmp_path / "t.db")
    asyncio.new_event_loop().run_until_complete(database.init_db())
    return database


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _scan(**kw):
    base = {
        "product_id": "ABC-USD",
        "price": 1.0,
        "model_prob": 0.7,
        "side": "BUY",
        "strength": 0.4,
    }
    base.update(kw)
    return base


def _col(db, sql):
    con = sqlite3.connect(db.DB_PATH)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# ── cnn_scans ─────────────────────────────────────────────────────────────────


def test_scan_stores_provenance(db):
    _run(db.save_cnn_scan(_scan(model_provenance=_DIGEST)))
    assert _col(db, "SELECT model_provenance FROM cnn_scans") == [(_DIGEST,)]


def test_scan_without_provenance_is_null_not_a_placeholder(db):
    _run(db.save_cnn_scan(_scan()))
    assert _col(db, "SELECT model_provenance FROM cnn_scans") == [(None,)]


@pytest.mark.parametrize("bad", ["unknown", "sha256:abc", ""])
def test_scan_rejects_malformed_provenance(db, bad):
    with pytest.raises(ValueError, match="model_provenance"):
        _run(db.save_cnn_scan(_scan(model_provenance=bad)))
    assert _col(db, "SELECT COUNT(*) FROM cnn_scans") == [(0,)]


# ── trades ────────────────────────────────────────────────────────────────────


def test_trade_entry_stores_provenance_and_close_keeps_it(db):
    _run(db.open_trade("CNN", "ABC-USD", 1.0, 2.0, 2.0, "SCAN", 98.0, model_provenance=_DIGEST))
    _run(db.close_trade("CNN", "ABC-USD", 1.1, 2.0, 0.2, "TRAIL_STOP", 100.2))
    assert _col(db, "SELECT model_provenance, closed_at IS NOT NULL FROM trades") == [(_DIGEST, 1)]


def test_trade_entry_without_provenance_is_null(db):
    _run(db.open_trade("CNN", "ABC-USD", 1.0, 2.0, 2.0, "SCAN", 98.0))
    assert _col(db, "SELECT model_provenance FROM trades") == [(None,)]


def test_unmatched_close_insert_stays_unattributed(db):
    _run(db.close_trade("CNN", "XYZ-USD", 1.1, 2.0, 0.2, "TRAIL_STOP", 100.2))
    assert _col(db, "SELECT trigger_open, model_provenance FROM trades") == [("UNKNOWN", None)]


def test_trade_rejects_malformed_provenance(db):
    with pytest.raises(ValueError, match="model_provenance"):
        _run(
            db.open_trade("CNN", "ABC-USD", 1.0, 2.0, 2.0, "SCAN", 98.0, model_provenance="unknown")
        )
    assert _col(db, "SELECT COUNT(*) FROM trades") == [(0,)]


# ── registry: digest -> detail ────────────────────────────────────────────────


def test_registry_round_trip_and_first_write_wins(db):
    detail = {"digest": _DIGEST, "components": {"v3": "sha256:" + "a" * 64}, "config": {"t": "x"}}
    _run(db.record_provenance(detail))
    _run(db.record_provenance(dict(detail, config={"t": "changed"})))
    got = _run(db.get_model_provenance(_DIGEST))
    assert got["detail"]["config"] == {"t": "x"}
    assert got["first_seen"]


def test_registry_rejects_malformed_digest(db):
    with pytest.raises(ValueError, match="model_provenance"):
        _run(db.record_provenance({"digest": "unknown"}))


def test_unknown_digest_lookup_is_none(db):
    assert _run(db.get_model_provenance(_DIGEST)) is None


def test_migration_adds_columns_to_an_existing_database(db):
    # Rebuild the pre-change schema: drop the new columns, then re-run init_db.
    con = sqlite3.connect(db.DB_PATH)
    con.execute("ALTER TABLE trades DROP COLUMN model_provenance")
    con.execute("ALTER TABLE cnn_scans DROP COLUMN model_provenance")
    con.commit()
    con.close()
    _run(db.init_db())
    con = sqlite3.connect(db.DB_PATH)
    cols = {
        t: [r[1] for r in con.execute(f"PRAGMA table_info({t})")] for t in ("trades", "cnn_scans")
    }
    con.close()
    assert "model_provenance" in cols["trades"] and "model_provenance" in cols["cnn_scans"]
