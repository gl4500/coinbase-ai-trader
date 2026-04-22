"""
Integration tests for the async SQLite database layer — Coinbase schema.
Uses a real (temporary) SQLite file — no mocking of DB calls.
"""
import os
import sys
import asyncio
import importlib
import pytest

# Make backend importable
BACKEND = os.path.join(os.path.dirname(__file__), "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

# Patch env before any imports touch config
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("APP_API_KEY",              "test-key")
os.environ.setdefault("DRY_RUN",                  "true")


@pytest.fixture
def db_module(tmp_path):
    """Reload the database module pointing at a fresh tmp SQLite file."""
    import database
    importlib.reload(database)
    database.DB_PATH = str(tmp_path / "test.db")
    return database


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def run(event_loop):
    def _run(coro):
        return event_loop.run_until_complete(coro)
    return _run


@pytest.fixture
def db(db_module, run):
    run(db_module.init_db())
    return db_module


# ── Timestamps ────────────────────────────────────────────────────────────────

class TestTimestamps:
    def test_now_returns_utc_offset(self, db_module):
        ts = db_module._now()
        assert ts.endswith("+00:00"), f"Expected UTC timestamp, got: {ts}"

    def test_product_timestamp_is_utc(self, db, run):
        product = {
            "product_id":    "BTC-USD",
            "base_currency": "BTC",
            "quote_currency": "USD",
        }
        run(db.upsert_product(product))
        products = run(db.get_products())
        assert any(p["last_updated"].endswith("+00:00") for p in products)

    def test_signal_timestamp_is_utc(self, db, run):
        # Need a product first (FK)
        run(db.upsert_product({
            "product_id": "ETH-USD", "base_currency": "ETH", "quote_currency": "USD"
        }))
        signal = {
            "product_id":  "ETH-USD",
            "signal_type": "RSI_OVERSOLD",
            "side":        "BUY",
            "price":       3000.0,
            "strength":    0.75,
        }
        run(db.save_signal(signal))
        signals = run(db.get_signals())
        assert signals[0]["created_at"].endswith("+00:00")

    def test_order_timestamp_is_utc(self, db, run):
        order = {
            "order_id":   "ord001",
            "product_id": "BTC-USD",
            "side":       "BUY",
            "order_type": "LIMIT",
            "price":      95000.0,
            "base_size":  0.001,
            "status":     "open",
        }
        run(db.save_order(order))
        orders = run(db.get_orders())
        assert orders[0]["created_at"].endswith("+00:00")


# ── Products CRUD ─────────────────────────────────────────────────────────────

class TestProducts:
    def _product(self, pid="BTC-USD", tracked=False):
        return {
            "product_id":    pid,
            "base_currency": pid.split("-")[0],
            "quote_currency": "USD",
            "price":         95_000.0,
            "volume_24h":    500_000_000,
            "is_tracked":    tracked,
        }

    def test_upsert_and_fetch(self, db, run):
        run(db.upsert_product(self._product()))
        products = run(db.get_products())
        assert len(products) == 1
        assert products[0]["product_id"] == "BTC-USD"

    def test_upsert_updates_price(self, db, run):
        run(db.upsert_product(self._product()))
        run(db.upsert_product({**self._product(), "price": 100_000.0}))
        p = run(db.get_product("BTC-USD"))
        assert p["price"] == 100_000.0

    def test_tracked_filter(self, db, run):
        run(db.upsert_product(self._product("SOL-USD", tracked=False)))
        run(db.upsert_product(self._product("ETH-USD", tracked=True)))
        tracked = run(db.get_products(tracked_only=True))
        assert len(tracked) == 1
        assert tracked[0]["product_id"] == "ETH-USD"

    def test_set_product_tracked(self, db, run):
        run(db.upsert_product(self._product("DOGE-USD")))
        run(db.set_product_tracked("DOGE-USD", True))
        p = run(db.get_product("DOGE-USD"))
        assert p["is_tracked"] == 1

    def test_get_nonexistent_returns_none(self, db, run):
        assert run(db.get_product("FAKE-USD")) is None

    def test_update_product_price(self, db, run):
        run(db.upsert_product(self._product("LINK-USD")))
        run(db.update_product_price("LINK-USD", 25.0, pct_change=5.5))
        p = run(db.get_product("LINK-USD"))
        assert p["price"] == 25.0
        assert p["price_pct_change_24h"] == 5.5


# ── Candles CRUD ──────────────────────────────────────────────────────────────

class TestCandles:
    def _candles(self, n=5, pid="BTC-USD"):
        return [
            {"start": 1_700_000_000 + i * 3600, "open": 95_000.0, "high": 96_000.0,
             "low": 94_000.0, "close": 95_500.0, "volume": 1000.0}
            for i in range(n)
        ]

    def test_save_and_fetch(self, db, run):
        run(db.upsert_product({
            "product_id": "BTC-USD", "base_currency": "BTC", "quote_currency": "USD"
        }))
        run(db.save_candles("BTC-USD", self._candles(5)))
        candles = run(db.get_candles("BTC-USD"))
        assert len(candles) == 5

    def test_candles_oldest_first(self, db, run):
        run(db.upsert_product({
            "product_id": "ETH-USD", "base_currency": "ETH", "quote_currency": "USD"
        }))
        run(db.save_candles("ETH-USD", self._candles(3, "ETH-USD")))
        candles = run(db.get_candles("ETH-USD"))
        assert candles[0]["start_time"] <= candles[-1]["start_time"]

    def test_duplicate_candles_ignored(self, db, run):
        run(db.upsert_product({
            "product_id": "SOL-USD", "base_currency": "SOL", "quote_currency": "USD"
        }))
        run(db.save_candles("SOL-USD", self._candles(3, "SOL-USD")))
        run(db.save_candles("SOL-USD", self._candles(3, "SOL-USD")))  # same timestamps
        candles = run(db.get_candles("SOL-USD"))
        assert len(candles) == 3


# ── Signals CRUD ──────────────────────────────────────────────────────────────

class TestSignals:
    def _signal(self, pid="BTC-USD"):
        return {
            "product_id":  pid,
            "signal_type": "CNN_LONG",
            "side":        "BUY",
            "price":       95_000.0,
            "strength":    0.80,
            "rsi":         28.5,
            "macd":        0.0012,
            "bb_position": 0.1,
            "reasoning":   "Oversold on RSI",
        }

    def test_save_and_retrieve(self, db, run):
        sig_id = run(db.save_signal(self._signal()))
        assert isinstance(sig_id, int)
        signals = run(db.get_signals())
        assert signals[0]["strength"] == 0.80
        assert signals[0]["signal_type"] == "CNN_LONG"

    def test_mark_acted(self, db, run):
        sig_id = run(db.save_signal(self._signal()))
        run(db.mark_signal_acted(sig_id, "order-xyz"))
        signals = run(db.get_signals())
        s = next(s for s in signals if s["id"] == sig_id)
        assert s["acted"] == 1
        assert s["order_id"] == "order-xyz"

    def test_limit(self, db, run):
        for _ in range(5):
            run(db.save_signal(self._signal()))
        signals = run(db.get_signals(limit=3))
        assert len(signals) <= 3


# ── Orders CRUD ───────────────────────────────────────────────────────────────

class TestOrders:
    def _order(self, oid="ord001"):
        return {
            "order_id":   oid,
            "product_id": "BTC-USD",
            "side":       "BUY",
            "order_type": "LIMIT",
            "price":      94_000.0,
            "base_size":  0.001,
            "status":     "open",
            "strategy":   "CNN_LONG",
        }

    def test_save_and_retrieve(self, db, run):
        run(db.save_order(self._order()))
        orders = run(db.get_orders())
        assert len(orders) == 1
        assert orders[0]["order_id"] == "ord001"

    def test_update_status(self, db, run):
        run(db.save_order(self._order("ord002")))
        run(db.update_order_status("ord002", "filled", filled_size=0.001, avg_fill_price=94_100.0))
        orders = run(db.get_orders(status="filled"))
        assert orders[0]["status"] == "filled"
        assert orders[0]["avg_fill_price"] == 94_100.0

    def test_filter_by_status(self, db, run):
        run(db.save_order(self._order("ord003")))
        run(db.save_order({**self._order("ord004"), "status": "cancelled"}))
        open_orders = run(db.get_orders(status="open"))
        assert all(o["status"] == "open" for o in open_orders)


# ── Positions & Portfolio ─────────────────────────────────────────────────────

class TestPortfolio:
    def _position(self, pid="BTC-USD"):
        return {
            "product_id":    pid,
            "base_currency": pid.split("-")[0],
            "side":          "BUY",
            "size":          0.1,
            "avg_price":     90_000.0,
            "current_price": 95_000.0,
            "initial_value": 9_000.0,
            "current_value": 9_500.0,
            "cash_pnl":      500.0,
            "pct_pnl":       5.56,
        }

    def test_empty_portfolio(self, db, run):
        summary = run(db.get_portfolio_summary())
        assert summary["open_positions"] == 0
        assert summary["total_value"] == 0.0

    def test_upsert_and_summary(self, db, run):
        run(db.upsert_position(self._position("BTC-USD")))
        summary = run(db.get_portfolio_summary())
        assert summary["open_positions"] == 1
        assert summary["total_value"] == 9_500.0
        assert summary["total_pnl"]   == 500.0

    def test_upsert_updates_existing(self, db, run):
        run(db.upsert_position(self._position("ETH-USD")))
        updated = {**self._position("ETH-USD"), "current_value": 10_000.0, "cash_pnl": 1_000.0}
        run(db.upsert_position(updated))
        positions = run(db.get_positions())
        eth = next(p for p in positions if p["product_id"] == "ETH-USD")
        assert eth["current_value"] == 10_000.0

    def test_delete_position(self, db, run):
        run(db.upsert_position(self._position("SOL-USD")))
        run(db.delete_position("SOL-USD"))
        positions = run(db.get_positions())
        assert all(p["product_id"] != "SOL-USD" for p in positions)


# ── CNN Training Sessions ────────────────────────────────────────────────────

class TestCNNTrainingSessions:
    """save_training_session must persist all training metrics, including the
    instrumentation added to compare gate choices (val_auc, precision/recall
    at the production BUY threshold). Without persistence we can't evaluate
    whether a different save gate would pick better checkpoints.
    """

    def _base_result(self) -> dict:
        return {
            "epochs":             50,
            "stopped_epoch":      17,
            "train_samples":      100_000,
            "val_samples":        25_000,
            "channels":           27,
            "arch":               "glu2",
            "initial_loss":       0.693,
            "final_train_loss":   0.42,
            "final_val_loss":     0.81,
            "best_val_loss":      0.66,
            "overfit_gap_pct":    92.0,
            "fit_status":         "OVERFIT",
            "fit_advice":         "train/val divergence",
            "duration_secs":      380,
            "saved":              False,
        }

    def test_persists_val_auc(self, db, run):
        r = self._base_result()
        r["val_auc"] = 0.6527
        run(db.save_training_session(r))
        import sqlite3
        cur = sqlite3.connect(db.DB_PATH).execute(
            "SELECT val_auc FROM cnn_training_sessions ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None
        assert abs(row[0] - 0.6527) < 1e-6

    def test_persists_precision_recall_and_threshold(self, db, run):
        r = self._base_result()
        r["val_precision_at_thresh"] = 0.58
        r["val_recall_at_thresh"]    = 0.31
        r["val_threshold"]           = 0.60
        run(db.save_training_session(r))
        import sqlite3
        cur = sqlite3.connect(db.DB_PATH).execute(
            "SELECT val_precision_at_thresh, val_recall_at_thresh, val_threshold "
            "FROM cnn_training_sessions ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row is not None
        assert abs(row[0] - 0.58) < 1e-6
        assert abs(row[1] - 0.31) < 1e-6
        assert abs(row[2] - 0.60) < 1e-6

    def test_persists_none_when_missing(self, db, run):
        # New fields must be nullable: old callers that don't pass them still work.
        r = self._base_result()
        run(db.save_training_session(r))
        import sqlite3
        cur = sqlite3.connect(db.DB_PATH).execute(
            "SELECT val_auc, val_precision_at_thresh, val_recall_at_thresh, val_threshold "
            "FROM cnn_training_sessions ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        assert row == (None, None, None, None)
