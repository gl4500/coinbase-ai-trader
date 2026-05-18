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


class TestMajorsAlwaysIncluded:
    """#105 — get_products must include core majors (BTC/ETH/SOL/...) even when
    their volume_24h is lower than memecoins. Without this pinning the CNN
    trains on a memecoin-dominated dataset and the OKX funding fetch silently
    skips BTC/ETH because they never enter the products iteration."""

    def _prod(self, pid, volume):
        return {
            "product_id":    pid,
            "base_currency": pid.split("-")[0],
            "quote_currency": "USD",
            "price":         1.0,
            "volume_24h":    volume,
            "is_tracked":    True,
        }

    def test_btc_eth_sol_returned_even_with_low_native_volume(self, db, run):
        # Memecoins with huge native-unit volume (typical real-world state)
        for i in range(100):
            run(db.upsert_product(self._prod(f"MEME{i}-USD", volume=1_000_000_000)))
        # Majors with lower native-unit volume (BTC trades in tiny coin counts)
        run(db.upsert_product(self._prod("BTC-USD", volume=50_000)))
        run(db.upsert_product(self._prod("ETH-USD", volume=200_000)))
        run(db.upsert_product(self._prod("SOL-USD", volume=500_000)))

        products = run(db.get_products(limit=100))
        pids = [p["product_id"] for p in products]
        assert "BTC-USD" in pids, "BTC-USD must be pinned regardless of volume rank"
        assert "ETH-USD" in pids, "ETH-USD must be pinned regardless of volume rank"
        assert "SOL-USD" in pids, "SOL-USD must be pinned regardless of volume rank"


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


# ── Product status persistence (#120c — task #124) ───────────────────────────

class TestProductStatusPersistence:
    """Per-product blacklist tier (#120c). The evaluator in
    `services.product_status` is pure and trade-driven; persistence lives in
    `database` so the scan loop and `_CNNBook.buy()` can read the live status
    without running the evaluator on every signal.

    Schema: product_status(product_id PK, status TEXT NOT NULL,
        reason TEXT, last_evaluated_at TEXT NOT NULL, demoted_at TEXT).
    """

    def test_table_created_by_init_db(self, db):
        import sqlite3
        cur = sqlite3.connect(db.DB_PATH).execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='product_status'"
        )
        assert cur.fetchone() is not None, (
            "init_db() must create product_status table — required for #120c persistence"
        )

    def test_init_db_is_idempotent(self, db, run):
        # Running init_db twice on the same path must not raise (production
        # path: backend reboots against an existing DB file).
        run(db.init_db())
        run(db.init_db())

    def test_get_returns_none_for_unknown_product(self, db, run):
        assert run(db.get_product_status("DOGE-USD")) is None

    def test_set_then_get_round_trips_status(self, db, run):
        run(db.set_product_status("BTC-USD", "active"))
        row = run(db.get_product_status("BTC-USD"))
        assert row is not None
        assert row["product_id"] == "BTC-USD"
        assert row["status"] == "active"
        assert row["last_evaluated_at"].endswith("+00:00"), (
            "last_evaluated_at must be a UTC ISO timestamp"
        )

    def test_set_persists_reason(self, db, run):
        run(db.set_product_status("ETH-USD", "probation",
                                  reason="demote: per-trade sharpe=-0.612"))
        row = run(db.get_product_status("ETH-USD"))
        assert row["status"] == "probation"
        assert row["reason"] == "demote: per-trade sharpe=-0.612"

    def test_set_is_upsert(self, db, run):
        run(db.set_product_status("SOL-USD", "active"))
        run(db.set_product_status("SOL-USD", "probation",
                                  reason="demote: sharpe=-0.7"))
        row = run(db.get_product_status("SOL-USD"))
        assert row["status"] == "probation"
        assert row["reason"] == "demote: sharpe=-0.7"

    def test_set_to_suspended_records_demoted_at(self, db, run):
        """First entry into 'suspended' must stamp demoted_at — used by the
        UI to show how long a pair has been blacklisted."""
        run(db.set_product_status("XRP-USD", "suspended",
                                  reason="demote: sharpe=-1.2"))
        row = run(db.get_product_status("XRP-USD"))
        assert row["status"] == "suspended"
        assert row["demoted_at"] is not None
        assert row["demoted_at"].endswith("+00:00")

    def test_promotion_clears_demoted_at(self, db, run):
        """Recovery (suspended -> probation) clears demoted_at so the next
        demotion stamps a fresh timestamp rather than reusing the stale one."""
        run(db.set_product_status("ADA-USD", "suspended",
                                  reason="demote: sharpe=-1.0"))
        run(db.set_product_status("ADA-USD", "probation",
                                  reason="promote: sharpe=+0.3"))
        row = run(db.get_product_status("ADA-USD"))
        assert row["status"] == "probation"
        assert row["demoted_at"] is None

    def test_active_does_not_set_demoted_at(self, db, run):
        run(db.set_product_status("LINK-USD", "active"))
        row = run(db.get_product_status("LINK-USD"))
        assert row["demoted_at"] is None

    def test_list_products_by_status(self, db, run):
        run(db.set_product_status("BTC-USD",  "active"))
        run(db.set_product_status("ETH-USD",  "probation"))
        run(db.set_product_status("DOGE-USD", "suspended"))
        run(db.set_product_status("XRP-USD",  "suspended"))

        suspended = run(db.list_products_by_status("suspended"))
        assert sorted(suspended) == ["DOGE-USD", "XRP-USD"]

        active = run(db.list_products_by_status("active"))
        assert active == ["BTC-USD"]

    def test_list_products_by_status_returns_empty_when_none(self, db, run):
        assert run(db.list_products_by_status("suspended")) == []


# ── XGB-Step3 (#181): xgb_prob column on cnn_scans ────────────────────────────

class TestCnnScansXgbProb:
    """Parallel xgb_prob logging in cnn_scans — XGB-Step3 (#181).

    Lets the shadow window collect side-by-side (cnn_prob, xgb_prob) per
    scan tick so we can compare ranking + win rate without flipping the
    live MODEL_BACKEND.
    """

    def _scan_payload(self, **overrides) -> dict:
        base = {
            "product_id": "BTC-USD",
            "price":      50_000.0,
            "cnn_prob":   0.55,
            "llm_prob":   0.40,
            "model_prob": 0.50,
            "cnn_weight": 0.6,
            "llm_weight": 0.4,
            "side":       "HOLD",
            "strength":   0.0,
            "signal_gen": False,
            "regime":     "RANGING",
            "adx":        18.0,
            "rsi":        55.0,
            "macd":       0.001,
            "mfi":        50.0,
            "stoch_k":    60.0,
            "atr":        100.0,
            "vwap_dist":  0.001,
            "fast_rsi":   0.50,
            "velocity":   0.0,
            "vol_z":      0.0,
        }
        base.update(overrides)
        return base

    def test_save_and_read_xgb_prob(self, db, run):
        """save_cnn_scan persists xgb_prob; get_cnn_scans returns it."""
        run(db.save_cnn_scan(self._scan_payload(xgb_prob=0.4242)))
        rows = run(db.get_cnn_scans(product_id="BTC-USD", limit=1))
        assert len(rows) == 1
        assert "xgb_prob" in rows[0]
        assert rows[0]["xgb_prob"] == pytest.approx(0.4242)

    def test_xgb_prob_defaults_to_null_when_omitted(self, db, run):
        """Backwards compat: callers that don't pass xgb_prob get NULL stored."""
        run(db.save_cnn_scan(self._scan_payload()))
        rows = run(db.get_cnn_scans(product_id="BTC-USD", limit=1))
        assert len(rows) == 1
        assert rows[0]["xgb_prob"] is None

    def test_xgb_prob_distinct_from_cnn_prob(self, db, run):
        """Stores cnn_prob and xgb_prob independently in a single row."""
        run(db.save_cnn_scan(
            self._scan_payload(cnn_prob=0.71, xgb_prob=0.33)
        ))
        rows = run(db.get_cnn_scans(product_id="BTC-USD", limit=1))
        assert rows[0]["cnn_prob"] == pytest.approx(0.71)
        assert rows[0]["xgb_prob"] == pytest.approx(0.33)


# ── MC telemetry column persistence (added 2026-05-16, #311-mc-wire) ──────


class TestSaveCnnScanMCColumns:
    def test_save_cnn_scan_persists_xgb_prob_stdev_when_present(self, db, run):
        from migrations.mc_telemetry_20260516 import run as mig_run
        mig_run(db.DB_PATH)
        run(db.upsert_product({
            "product_id": "BTC-USD", "base_currency": "BTC",
            "quote_currency": "USD",
        }))
        run(db.save_cnn_scan({
            "product_id": "BTC-USD", "price": 100.0,
            "cnn_prob": 0.60, "model_prob": 0.60, "side": "BUY",
            "strength": 0.2, "signal_gen": True,
            "xgb_prob_stdev": 0.0124,
            "mc_telemetry": '{"ci":{"decision":"keep"}}',
        }))
        import sqlite3
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_stdev, mc_telemetry FROM cnn_scans WHERE product_id='BTC-USD'"
        ).fetchone()
        c.close()
        assert row[0] == 0.0124
        assert row[1] == '{"ci":{"decision":"keep"}}'

    def test_save_cnn_scan_handles_missing_mc_keys_as_null(self, db, run):
        from migrations.mc_telemetry_20260516 import run as mig_run
        mig_run(db.DB_PATH)
        run(db.upsert_product({
            "product_id": "ETH-USD", "base_currency": "ETH",
            "quote_currency": "USD",
        }))
        run(db.save_cnn_scan({
            "product_id": "ETH-USD", "price": 200.0,
            "cnn_prob": 0.50, "model_prob": 0.50, "side": "HOLD",
            "strength": 0.0, "signal_gen": False,
        }))
        import sqlite3
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_stdev, mc_telemetry FROM cnn_scans WHERE product_id='ETH-USD'"
        ).fetchone()
        c.close()
        assert row[0] is None
        assert row[1] is None


# ── xgb_prob_v4 persistence (#xgb-v4 / Step B.1) ──────────────────────────


class TestSaveCnnScanXgbProbV4:
    def test_save_cnn_scan_persists_xgb_prob_v4_when_present(self, db, run):
        from migrations.xgb_v4_shadow_20260517 import run as mig_run
        mig_run(db.DB_PATH)
        run(db.upsert_product({
            "product_id": "BTC-USD", "base_currency": "BTC",
            "quote_currency": "USD",
        }))
        run(db.save_cnn_scan({
            "product_id": "BTC-USD", "price": 100.0,
            "cnn_prob": 0.60, "model_prob": 0.60, "side": "BUY",
            "strength": 0.2, "signal_gen": True,
            "xgb_prob": 0.55, "xgb_prob_v4": 0.42,
        }))
        import sqlite3
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob, xgb_prob_v4 FROM cnn_scans WHERE product_id=?"
            , ("BTC-USD",)
        ).fetchone()
        c.close()
        assert row is not None
        assert row[0] == 0.55
        assert row[1] == 0.42

    def test_save_cnn_scan_xgb_prob_v4_defaults_to_null(self, db, run):
        from migrations.xgb_v4_shadow_20260517 import run as mig_run
        mig_run(db.DB_PATH)
        run(db.upsert_product({
            "product_id": "ETH-USD", "base_currency": "ETH",
            "quote_currency": "USD",
        }))
        run(db.save_cnn_scan({
            "product_id": "ETH-USD", "price": 200.0,
            "cnn_prob": 0.50, "model_prob": 0.50, "side": "HOLD",
            "strength": 0.0, "signal_gen": False,
        }))
        import sqlite3
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_v4 FROM cnn_scans WHERE product_id=?"
            , ("ETH-USD",)
        ).fetchone()
        c.close()
        assert row[0] is None


# ── xgb_prob_v4_5_{down,neutral,up} persistence (#xgb-v4.5 / Step B.1.5) ──


class TestSaveCnnScanV4_5Cols:
    @pytest.mark.asyncio
    async def test_save_cnn_scan_persists_three_v45_probs(self, db, run):
        from migrations.xgb_v4_5_shadow_20260517 import run as mig_run
        import sqlite3

        mig_run(db.DB_PATH)
        await db.save_cnn_scan({
            "product_id": "BTC-USD",
            "price": 100.0,
            "model_prob": 0.6, "cnn_prob": 0.6, "llm_prob": None,
            "regime": "TRENDING", "side": "BUY",
            "cnn_weight": 1.0, "llm_weight": 0.0,
            "rsi": 50.0, "macd": 0.0, "mfi": 50.0, "adx": 20.0, "atr": 1.0,
            "stoch_k": 50.0,
            "vwap_dist": 0.0, "fast_rsi": 0.5, "velocity": 0.5, "vol_z": 0.5,
            "xgb_prob": 0.55, "strength": 0.5, "signal_gen": True,
            "xgb_prob_v4_5_down":    0.21,
            "xgb_prob_v4_5_neutral": 0.34,
            "xgb_prob_v4_5_up":      0.45,
        })
        # Note: connect via the same path
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up "
            "FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row is not None
        assert row[0] == pytest.approx(0.21)
        assert row[1] == pytest.approx(0.34)
        assert row[2] == pytest.approx(0.45)

    @pytest.mark.asyncio
    async def test_save_cnn_scan_v45_default_null(self, db, run):
        from migrations.xgb_v4_5_shadow_20260517 import run as mig_run
        import sqlite3

        mig_run(db.DB_PATH)
        await db.save_cnn_scan({
            "product_id": "BTC-USD",
            "price": 100.0,
            "model_prob": 0.6, "cnn_prob": 0.6, "llm_prob": None,
            "regime": "TRENDING", "side": "BUY",
            "cnn_weight": 1.0, "llm_weight": 0.0,
            "rsi": 50.0, "macd": 0.0, "mfi": 50.0, "adx": 20.0, "atr": 1.0,
            "stoch_k": 50.0,
            "vwap_dist": 0.0, "fast_rsi": 0.5, "velocity": 0.5, "vol_z": 0.5,
            "xgb_prob": 0.55, "strength": 0.5, "signal_gen": True,
        })
        c = sqlite3.connect(db.DB_PATH)
        row = c.execute(
            "SELECT xgb_prob_v4_5_down, xgb_prob_v4_5_neutral, xgb_prob_v4_5_up "
            "FROM cnn_scans WHERE product_id=?",
            ("BTC-USD",),
        ).fetchone()
        c.close()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None
