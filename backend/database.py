"""
Async SQLite database layer — Coinbase crypto trading app.
All timestamps stored as UTC ISO strings.
"""
import aiosqlite
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from config import config

logger  = logging.getLogger(__name__)
DB_PATH = config.database_url
_DB_TIMEOUT = 30   # seconds to wait for a locked DB before raising OperationalError


@asynccontextmanager
async def _db():
    """Open a DB connection with a C-level busy_timeout pragma.

    aiosqlite's Python-level timeout= applies only when opening the file;
    it does not retry individual write operations when the DB is locked by
    another concurrent writer.  PRAGMA busy_timeout is handled by the SQLite
    C library and applies to every subsequent lock acquisition on this
    connection — the correct fix for 'database is locked' under concurrency.
    """
    async with aiosqlite.connect(DB_PATH, timeout=_DB_TIMEOUT) as conn:
        await conn.execute("PRAGMA busy_timeout=30000")
        yield conn


async def init_db() -> None:
    async with _db() as db:
        # WAL mode: multiple readers + one writer can coexist; prevents most lock errors
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA busy_timeout=30000")   # 30s in milliseconds
        await db.commit()
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS products (
                product_id          TEXT PRIMARY KEY,
                base_currency       TEXT NOT NULL,
                quote_currency      TEXT NOT NULL,
                display_name        TEXT,
                price               REAL,
                price_pct_change_24h REAL DEFAULT 0,
                volume_24h          REAL DEFAULT 0,
                high_24h            REAL DEFAULT 0,
                low_24h             REAL DEFAULT 0,
                spread              REAL DEFAULT 0,
                is_tracked          INTEGER DEFAULT 0,
                last_updated        TEXT
            );

            CREATE TABLE IF NOT EXISTS candles (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id      TEXT NOT NULL,
                start_time      INTEGER NOT NULL,
                open            REAL NOT NULL,
                high            REAL NOT NULL,
                low             REAL NOT NULL,
                close           REAL NOT NULL,
                volume          REAL NOT NULL,
                UNIQUE(product_id, start_time),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            );

            CREATE TABLE IF NOT EXISTS signals (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id      TEXT NOT NULL,
                signal_type     TEXT NOT NULL,
                side            TEXT NOT NULL,
                price           REAL NOT NULL,
                strength        REAL NOT NULL,
                rsi             REAL,
                macd            REAL,
                ema_cross       REAL,
                bb_position     REAL,
                reasoning       TEXT,
                acted           INTEGER DEFAULT 0,
                order_id        TEXT,
                created_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id        TEXT PRIMARY KEY,
                product_id      TEXT NOT NULL,
                side            TEXT NOT NULL,
                order_type      TEXT NOT NULL,
                price           REAL,
                base_size       REAL,
                quote_size      REAL,
                status          TEXT DEFAULT 'pending',
                filled_size     REAL DEFAULT 0,
                avg_fill_price  REAL,
                fee_paid        REAL DEFAULT 0,
                strategy        TEXT,
                created_at      TEXT NOT NULL,
                updated_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS positions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id      TEXT NOT NULL UNIQUE,
                base_currency   TEXT NOT NULL,
                side            TEXT NOT NULL,
                size            REAL NOT NULL,
                avg_price       REAL NOT NULL,
                current_price   REAL,
                initial_value   REAL,
                current_value   REAL,
                cash_pnl        REAL,
                pct_pnl         REAL,
                updated_at      TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_candles_pid_time
                ON candles(product_id, start_time);
            CREATE INDEX IF NOT EXISTS idx_signals_acted
                ON signals(acted, created_at);
            CREATE INDEX IF NOT EXISTS idx_orders_status
                ON orders(status, created_at);
            CREATE INDEX IF NOT EXISTS idx_products_tracked
                ON products(is_tracked, volume_24h);

            CREATE TABLE IF NOT EXISTS cnn_scans (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id  TEXT NOT NULL,
                price       REAL NOT NULL,
                cnn_prob    REAL,
                llm_prob    REAL,
                model_prob  REAL NOT NULL,
                cnn_weight  REAL,
                llm_weight  REAL,
                side        TEXT NOT NULL,
                strength    REAL NOT NULL,
                signal_gen  INTEGER DEFAULT 0,
                regime      TEXT,
                adx         REAL,
                rsi         REAL,
                macd        REAL,
                mfi         REAL,
                stoch_k     REAL,
                atr         REAL,
                vwap_dist   REAL,
                fast_rsi    REAL,
                velocity    REAL,
                vol_z       REAL,
                scanned_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cnn_scans_time
                ON cnn_scans(scanned_at DESC);
            CREATE INDEX IF NOT EXISTS idx_cnn_scans_pid
                ON cnn_scans(product_id, scanned_at DESC);

            CREATE TABLE IF NOT EXISTS agent_state (
                agent           TEXT PRIMARY KEY,
                balance         REAL NOT NULL,
                realized_pnl    REAL NOT NULL DEFAULT 0,
                positions_json  TEXT NOT NULL DEFAULT '{}',
                high_water_json TEXT NOT NULL DEFAULT '{}',
                updated_at      TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_decisions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                agent       TEXT NOT NULL,      -- TECH | MOMENTUM
                product_id  TEXT NOT NULL,
                side        TEXT NOT NULL,      -- BUY | SELL | HOLD
                confidence  REAL NOT NULL,
                price       REAL NOT NULL,
                score       REAL,
                reasoning   TEXT,
                balance     REAL,              -- dry-run balance snapshot
                pnl         REAL,              -- realized PnL if SELL
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_agent_decisions_pid
                ON agent_decisions(product_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent
                ON agent_decisions(agent, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_decisions_created_at
                ON agent_decisions(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_decisions_side_created
                ON agent_decisions(side, created_at DESC);

            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT NOT NULL,    -- TECH | MOMENTUM | CNN
                product_id      TEXT NOT NULL,
                side            TEXT NOT NULL,    -- BUY | SELL
                confidence      REAL NOT NULL,
                entry_price     REAL NOT NULL,
                exit_price      REAL,
                pct_change      REAL,
                outcome         TEXT,             -- WIN | LOSS | NEUTRAL
                indicators_json TEXT,
                lesson_text     TEXT,
                check_after     REAL NOT NULL,    -- unix epoch when to resolve
                checked_at      TEXT,             -- ISO when resolved
                created_at      TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_pending
                ON signal_outcomes(check_after) WHERE outcome IS NULL;
            CREATE INDEX IF NOT EXISTS idx_signal_outcomes_lessons
                ON signal_outcomes(product_id, checked_at DESC) WHERE lesson_text IS NOT NULL;

            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                agent         TEXT NOT NULL,
                product_id    TEXT NOT NULL,
                entry_price   REAL NOT NULL,
                exit_price    REAL,
                size          REAL NOT NULL,
                usd_open      REAL NOT NULL,
                usd_close     REAL,
                pnl           REAL,
                pct_pnl       REAL,
                hold_secs     INTEGER,
                trigger_open  TEXT NOT NULL,
                trigger_close TEXT,
                balance_after REAL NOT NULL,
                opened_at     TEXT NOT NULL,
                closed_at     TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_trades_agent
                ON trades(agent, opened_at DESC);
            CREATE INDEX IF NOT EXISTS idx_trades_open
                ON trades(agent, product_id) WHERE closed_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_trades_closed_at
                ON trades(closed_at DESC);

            CREATE TABLE IF NOT EXISTS cnn_training_sessions (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                trained_at               TEXT NOT NULL,
                arch                     TEXT NOT NULL,
                channels                 INTEGER NOT NULL,
                epochs_requested         INTEGER NOT NULL,
                stopped_epoch            INTEGER NOT NULL,
                train_samples            INTEGER NOT NULL,
                val_samples              INTEGER NOT NULL,
                initial_train_loss       REAL NOT NULL,
                final_train_loss         REAL NOT NULL,
                final_val_loss           REAL NOT NULL,
                best_val_loss            REAL NOT NULL,
                overfit_gap_pct          REAL NOT NULL,
                fit_status               TEXT NOT NULL,
                fit_advice               TEXT NOT NULL,
                duration_secs            INTEGER NOT NULL,
                saved                    INTEGER NOT NULL DEFAULT 0,
                val_auc                  REAL,
                val_precision_at_thresh  REAL,
                val_recall_at_thresh     REAL,
                val_threshold            REAL
            );
        """)
        await db.commit()

        # ── Migrations: add columns to existing tables ────────────────────────
        for sql in [
            "ALTER TABLE cnn_scans ADD COLUMN vwap_dist REAL",
            "ALTER TABLE cnn_scans ADD COLUMN fast_rsi REAL",
            "ALTER TABLE cnn_scans ADD COLUMN velocity REAL",
            "ALTER TABLE cnn_scans ADD COLUMN vol_z REAL",
            "ALTER TABLE cnn_training_sessions ADD COLUMN val_auc REAL",
            "ALTER TABLE cnn_training_sessions ADD COLUMN val_precision_at_thresh REAL",
            "ALTER TABLE cnn_training_sessions ADD COLUMN val_recall_at_thresh REAL",
            "ALTER TABLE cnn_training_sessions ADD COLUMN val_threshold REAL",
        ]:
            try:
                await db.execute(sql)
                await db.commit()
            except Exception:
                pass   # column already exists — safe to ignore

    logger.info("Database initialised")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Products ──────────────────────────────────────────────────────────────────

async def upsert_product(p: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO products
               (product_id, base_currency, quote_currency, display_name,
                price, price_pct_change_24h, volume_24h, high_24h, low_24h,
                spread, is_tracked, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(product_id) DO UPDATE SET
                 price=excluded.price,
                 price_pct_change_24h=excluded.price_pct_change_24h,
                 volume_24h=excluded.volume_24h,
                 high_24h=excluded.high_24h,
                 low_24h=excluded.low_24h,
                 spread=excluded.spread,
                 is_tracked=excluded.is_tracked,
                 last_updated=excluded.last_updated""",
            (
                p["product_id"], p["base_currency"], p["quote_currency"],
                p.get("display_name", p["product_id"]),
                p.get("price"), p.get("price_pct_change_24h", 0),
                p.get("volume_24h", 0), p.get("high_24h", 0), p.get("low_24h", 0),
                p.get("spread", 0), 1 if p.get("is_tracked") else 0, _now(),
            )
        )
        await db.commit()


async def get_products(tracked_only: bool = False, limit: int = 100) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        where = "WHERE is_tracked = 1" if tracked_only else ""
        cursor = await db.execute(
            f"SELECT * FROM products {where} ORDER BY volume_24h DESC LIMIT ?", (limit,)
        )
        return [dict(r) for r in await cursor.fetchall()]


async def get_product(product_id: str) -> Optional[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def set_product_tracked(product_id: str, tracked: bool) -> None:
    async with _db() as db:
        await db.execute(
            "UPDATE products SET is_tracked = ? WHERE product_id = ?",
            (1 if tracked else 0, product_id)
        )
        await db.commit()


async def update_product_price(product_id: str, price: float,
                                pct_change: float = 0.0) -> None:
    async with _db() as db:
        await db.execute(
            "UPDATE products SET price=?, price_pct_change_24h=?, last_updated=? "
            "WHERE product_id=?",
            (price, pct_change, _now(), product_id)
        )
        await db.commit()


# ── Candles ───────────────────────────────────────────────────────────────────

async def save_candles(product_id: str, candles: List[Dict]) -> None:
    async with _db() as db:
        await db.executemany(
            """INSERT OR IGNORE INTO candles
               (product_id, start_time, open, high, low, close, volume)
               VALUES (?,?,?,?,?,?,?)""",
            [(product_id, c["start"], c["open"], c["high"],
              c["low"], c["close"], c["volume"]) for c in candles]
        )
        await db.commit()


async def candle_count(product_id: str) -> int:
    async with _db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM candles WHERE product_id=?", (product_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0


async def get_candles(product_id: str, limit: int = 100) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM candles WHERE product_id=? ORDER BY start_time DESC LIMIT ?",
            (product_id, limit)
        )
        rows = await cursor.fetchall()
        return list(reversed([dict(r) for r in rows]))  # oldest first


# ── Signals ───────────────────────────────────────────────────────────────────

async def save_signal(signal: Dict) -> int:
    async with _db() as db:
        cursor = await db.execute(
            """INSERT INTO signals
               (product_id, signal_type, side, price, strength, rsi, macd,
                ema_cross, bb_position, reasoning, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                signal["product_id"], signal["signal_type"], signal["side"],
                signal["price"], signal["strength"],
                signal.get("rsi"), signal.get("macd"),
                signal.get("ema_cross"), signal.get("bb_position"),
                signal.get("reasoning"), _now(),
            )
        )
        await db.commit()
        return cursor.lastrowid


async def mark_signal_acted(signal_id: int, order_id: str) -> None:
    async with _db() as db:
        await db.execute(
            "UPDATE signals SET acted=1, order_id=? WHERE id=?", (order_id, signal_id)
        )
        await db.commit()


async def get_signals(limit: int = 50, signal_type_prefix: Optional[str] = None) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        if signal_type_prefix:
            cursor = await db.execute(
                "SELECT * FROM signals WHERE signal_type LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"{signal_type_prefix}%", limit)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in await cursor.fetchall()]


# ── Orders ────────────────────────────────────────────────────────────────────

async def save_order(order: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT OR REPLACE INTO orders
               (order_id, product_id, side, order_type, price, base_size,
                quote_size, status, strategy, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                order["order_id"], order["product_id"], order["side"],
                order["order_type"], order.get("price"), order.get("base_size"),
                order.get("quote_size"), order.get("status", "pending"),
                order.get("strategy"), _now(),
            )
        )
        await db.commit()


async def update_order_status(order_id: str, status: str,
                               filled_size: float = 0,
                               avg_fill_price: Optional[float] = None) -> None:
    async with _db() as db:
        await db.execute(
            "UPDATE orders SET status=?, filled_size=?, avg_fill_price=?, updated_at=? "
            "WHERE order_id=?",
            (status, filled_size, avg_fill_price, _now(), order_id)
        )
        await db.commit()


async def get_orders(status: Optional[str] = None, limit: int = 100) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        if status:
            cursor = await db.execute(
                "SELECT * FROM orders WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit)
            )
        else:
            cursor = await db.execute(
                "SELECT * FROM orders ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        return [dict(r) for r in await cursor.fetchall()]


# ── Positions ─────────────────────────────────────────────────────────────────

async def upsert_position(pos: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO positions
               (product_id, base_currency, side, size, avg_price, current_price,
                initial_value, current_value, cash_pnl, pct_pnl, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(product_id) DO UPDATE SET
                 size=excluded.size, current_price=excluded.current_price,
                 current_value=excluded.current_value, cash_pnl=excluded.cash_pnl,
                 pct_pnl=excluded.pct_pnl, updated_at=excluded.updated_at""",
            (
                pos["product_id"], pos["base_currency"], pos.get("side", "BUY"),
                pos["size"], pos["avg_price"], pos.get("current_price"),
                pos.get("initial_value"), pos.get("current_value"),
                pos.get("cash_pnl"), pos.get("pct_pnl"), _now(),
            )
        )
        await db.commit()


async def get_positions() -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM positions WHERE size > 0 ORDER BY current_value DESC"
        )
        return [dict(r) for r in await cursor.fetchall()]


async def delete_position(product_id: str) -> None:
    async with _db() as db:
        await db.execute("DELETE FROM positions WHERE product_id=?", (product_id,))
        await db.commit()


# ── CNN Scans ─────────────────────────────────────────────────────────────────

async def save_cnn_scan(scan: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO cnn_scans
               (product_id, price, cnn_prob, llm_prob, model_prob,
                cnn_weight, llm_weight, side, strength, signal_gen,
                regime, adx, rsi, macd, mfi, stoch_k, atr, vwap_dist,
                fast_rsi, velocity, vol_z, scanned_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                scan["product_id"], scan["price"],
                scan.get("cnn_prob"), scan.get("llm_prob"), scan["model_prob"],
                scan.get("cnn_weight"), scan.get("llm_weight"),
                scan["side"], scan["strength"], 1 if scan.get("signal_gen") else 0,
                scan.get("regime"), scan.get("adx"), scan.get("rsi"),
                scan.get("macd"), scan.get("mfi"), scan.get("stoch_k"),
                scan.get("atr"), scan.get("vwap_dist"),
                scan.get("fast_rsi"), scan.get("velocity"), scan.get("vol_z"),
                _now(),
            )
        )
        await db.commit()


async def get_cnn_scans(
    limit: int = 200,
    product_id: Optional[str] = None,
    scan_run: Optional[str] = None,   # ISO prefix e.g. "2026-04-11T19"
) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        clauses, params = [], []
        if product_id:
            clauses.append("product_id = ?"); params.append(product_id)
        if scan_run:
            clauses.append("scanned_at LIKE ?"); params.append(f"{scan_run}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM cnn_scans {where} ORDER BY scanned_at DESC LIMIT ?",
            params
        )
        return [dict(r) for r in await cursor.fetchall()]


# ── Agent Decisions (Tech / Momentum) ────────────────────────────────────────

async def save_agent_decision(d: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO agent_decisions
               (agent, product_id, side, confidence, price, score,
                reasoning, balance, pnl, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                d["agent"], d["product_id"], d["side"],
                d["confidence"], d["price"], d.get("score"),
                d.get("reasoning"), d.get("balance"), d.get("pnl"),
                _now(),
            )
        )
        await db.commit()


async def save_agent_state(agent: str, balance: float, realized_pnl: float,
                            positions: Dict, high_water: Dict) -> None:
    import json
    async with _db() as db:
        await db.execute(
            """INSERT INTO agent_state
               (agent, balance, realized_pnl, positions_json, high_water_json, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(agent) DO UPDATE SET
                 balance=excluded.balance,
                 realized_pnl=excluded.realized_pnl,
                 positions_json=excluded.positions_json,
                 high_water_json=excluded.high_water_json,
                 updated_at=excluded.updated_at""",
            (agent, balance, realized_pnl,
             json.dumps(positions), json.dumps(high_water), _now())
        )
        await db.commit()


async def load_agent_state(agent: str) -> Optional[Dict]:
    import json
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM agent_state WHERE agent = ?", (agent,)
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["positions"]  = json.loads(d.pop("positions_json",  "{}"))
        d["high_water"] = json.loads(d.pop("high_water_json", "{}"))
        return d


async def purge_old_decisions(days: int = 7) -> int:
    """Delete agent_decisions rows older than `days` days. Returns deleted count."""
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    async with _db() as db:
        cur = await db.execute(
            "DELETE FROM agent_decisions WHERE created_at < ?", (cutoff,)
        )
        await db.commit()
        return cur.rowcount


async def get_agent_decisions(
    product_id: Optional[str] = None,
    agent: Optional[str] = None,
    limit: int = 20,
    signals_only: bool = False,
) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        clauses, params = [], []
        if product_id:
            clauses.append("product_id = ?"); params.append(product_id)
        if agent:
            clauses.append("agent = ?"); params.append(agent)
        if signals_only:
            clauses.append("side != 'HOLD'")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM agent_decisions {where} ORDER BY created_at DESC LIMIT ?",
            params,
        )
        return [dict(r) for r in await cursor.fetchall()]


# ── Trade ledger ──────────────────────────────────────────────────────────────

async def open_trade(
    agent: str, product_id: str, entry_price: float,
    size: float, usd_open: float, trigger_open: str, balance_after: float,
) -> int:
    """Insert an open trade row. Returns the new trade id."""
    async with _db() as db:
        cursor = await db.execute(
            """INSERT INTO trades
               (agent, product_id, entry_price, size, usd_open,
                trigger_open, balance_after, opened_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (agent, product_id, entry_price, size, usd_open,
             trigger_open, balance_after, _now()),
        )
        await db.commit()
        return cursor.lastrowid


async def close_trade(
    agent: str, product_id: str, exit_price: float, size: float,
    pnl: float, trigger_close: str, balance_after: float,
) -> None:
    """Update the most recent open trade for (agent, product_id) with close data."""
    usd_close = exit_price * size
    pct_pnl   = (pnl / (usd_close - pnl)) * 100 if (usd_close - pnl) > 0 else 0.0
    now       = _now()
    async with _db() as db:
        # Find the most recent open trade for this agent+product
        cursor = await db.execute(
            """SELECT id, opened_at FROM trades
               WHERE agent=? AND product_id=? AND closed_at IS NULL
               ORDER BY opened_at DESC LIMIT 1""",
            (agent, product_id),
        )
        row = await cursor.fetchone()
        if row:
            trade_id, opened_at = row[0], row[1]
            try:
                from datetime import datetime, timezone
                dt_open  = datetime.fromisoformat(opened_at.replace("Z", "+00:00"))
                dt_close = datetime.now(timezone.utc)
                hold_secs = int((dt_close - dt_open).total_seconds())
            except Exception:
                hold_secs = None
            await db.execute(
                """UPDATE trades SET
                   exit_price=?, usd_close=?, pnl=?, pct_pnl=?,
                   hold_secs=?, trigger_close=?, balance_after=?, closed_at=?
                   WHERE id=?""",
                (exit_price, usd_close, round(pnl, 4), round(pct_pnl, 2),
                 hold_secs, trigger_close, balance_after, now, trade_id),
            )
        else:
            # No open row found — insert a closed row directly (e.g. after restart)
            await db.execute(
                """INSERT INTO trades
                   (agent, product_id, entry_price, exit_price, size,
                    usd_open, usd_close, pnl, pct_pnl, trigger_open,
                    trigger_close, balance_after, opened_at, closed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (agent, product_id, exit_price, exit_price, size,
                 usd_close, usd_close, round(pnl, 4), round(pct_pnl, 2),
                 "UNKNOWN", trigger_close, balance_after, now, now),
            )
        await db.commit()


async def close_trade_by_id(trade_id: int, trigger_close: str = "RECONCILE") -> None:
    """Force-close a specific trade row by its primary key (used for reconciliation)."""
    from datetime import datetime, timezone
    now = _now()
    async with _db() as db:
        await db.execute(
            """UPDATE trades SET
               pnl=0, trigger_close=?, closed_at=?
               WHERE id=? AND closed_at IS NULL""",
            (trigger_close, now, trade_id),
        )
        await db.commit()


async def get_lgbm_training_rows() -> List[Dict]:
    """
    Return feature rows for LightGBM training: join closed CNN trades with the
    cnn_scans record captured at entry time (within 120s window).

    Returns list of dicts with keys matching data.lgbm_filter._FEATURES + 'pnl'.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("""
            SELECT
                s.cnn_prob,
                s.rsi,
                s.adx,
                s.strength,
                s.macd,
                s.mfi,
                s.stoch_k,
                CAST(strftime('%H', t.opened_at) AS INTEGER) AS hour_of_day,
                CAST(strftime('%w', t.opened_at) AS INTEGER) AS day_of_week,
                t.usd_open,
                t.pnl
            FROM trades t
            JOIN cnn_scans s ON s.product_id = t.product_id
                AND s.side = 'BUY'
                AND ABS(strftime('%s', t.opened_at) - strftime('%s', s.scanned_at)) < 120
            WHERE t.agent = 'CNN'
              AND t.closed_at IS NOT NULL
            ORDER BY t.opened_at ASC
        """)
        return [dict(r) for r in await cursor.fetchall()]


async def get_trades(
    agent: Optional[str] = None,
    product_id: Optional[str] = None,
    open_only: bool = False,
    closed_only: bool = False,
    limit: int = 100,
) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        clauses, params = [], []
        if agent:
            clauses.append("agent = ?"); params.append(agent)
        if product_id:
            clauses.append("product_id = ?"); params.append(product_id)
        if open_only:
            clauses.append("closed_at IS NULL")
        if closed_only:
            clauses.append("closed_at IS NOT NULL")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        cursor = await db.execute(
            f"SELECT * FROM trades {where} ORDER BY opened_at DESC LIMIT ?",
            params,
        )
        return [dict(r) for r in await cursor.fetchall()]


# ── Portfolio summary ─────────────────────────────────────────────────────────

async def get_portfolio_summary() -> Dict:
    async with _db() as db:
        cursor = await db.execute(
            "SELECT COUNT(*), SUM(current_value), SUM(initial_value), SUM(cash_pnl) "
            "FROM positions WHERE size > 0"
        )
        row = await cursor.fetchone()
        count, total_value, total_cost, total_pnl = row
        pct = ((total_pnl or 0) / max(total_cost or 1, 0.01)) * 100
        return {
            "open_positions": count or 0,
            "total_value":    round(total_value or 0, 2),
            "total_cost":     round(total_cost  or 0, 2),
            "total_pnl":      round(total_pnl   or 0, 2),
            "pct_pnl":        round(pct, 2),
        }


# ── Signal Outcomes (Outcome Tracker) ────────────────────────────────────────

async def insert_signal_outcome(d: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO signal_outcomes
               (source, product_id, side, confidence, entry_price,
                indicators_json, check_after, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                d["source"], d["product_id"], d["side"],
                d["confidence"], d["entry_price"],
                d.get("indicators_json", "{}"),
                d["check_after"], _now(),
            )
        )
        await db.commit()


async def get_pending_outcomes() -> List[Dict]:
    """Return all rows whose check_after has passed and are still unresolved."""
    import time as _time
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM signal_outcomes WHERE outcome IS NULL AND check_after <= ? "
            "ORDER BY check_after ASC LIMIT 100",
            (_time.time(),)
        )
        return [dict(r) for r in await cursor.fetchall()]


async def resolve_signal_outcome(
    row_id: int,
    exit_price: float,
    pct_change: float,
    outcome: str,
    lesson_text: str,
) -> None:
    async with _db() as db:
        await db.execute(
            """UPDATE signal_outcomes
               SET exit_price=?, pct_change=?, outcome=?,
                   lesson_text=?, checked_at=?
               WHERE id=?""",
            (exit_price, pct_change, outcome, lesson_text, _now(), row_id)
        )
        await db.commit()


async def get_recent_lessons(
    product_id: str,
    limit: int = 5,
    max_age_days: int = 30,
) -> List[str]:
    """Return up to `limit` recent lesson strings for this product.

    max_age_days: lessons older than this are excluded so stale regime-specific
    outcomes don't mislead Ollama in changed market conditions.
    """
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT lesson_text FROM signal_outcomes
               WHERE product_id=? AND lesson_text IS NOT NULL
                 AND checked_at >= datetime('now', ?)
               ORDER BY checked_at DESC LIMIT ?""",
            (product_id, f"-{max_age_days} days", limit)
        )
        rows = await cursor.fetchall()
        return [r["lesson_text"] for r in rows]


# ── CNN Training History ───────────────────────────────────────────────────────

async def save_training_session(r: Dict) -> None:
    async with _db() as db:
        await db.execute(
            """INSERT INTO cnn_training_sessions
               (trained_at, arch, channels, epochs_requested, stopped_epoch,
                train_samples, val_samples, initial_train_loss, final_train_loss,
                final_val_loss, best_val_loss, overfit_gap_pct, fit_status,
                fit_advice, duration_secs, saved,
                val_auc, val_precision_at_thresh, val_recall_at_thresh, val_threshold)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _now(),
                r.get("arch", "glu2"),
                r.get("channels", 0),
                r.get("epochs", 0),
                r.get("stopped_epoch", 0),
                r.get("train_samples", 0),
                r.get("val_samples", 0),
                r.get("initial_loss", 0.0),
                r.get("final_train_loss", 0.0),
                r.get("final_val_loss", 0.0),
                r.get("best_val_loss", 0.0),
                r.get("overfit_gap_pct", 0.0),
                r.get("fit_status", ""),
                r.get("fit_advice", ""),
                r.get("duration_secs", 0),
                1 if r.get("saved") else 0,
                r.get("val_auc"),
                r.get("val_precision_at_thresh"),
                r.get("val_recall_at_thresh"),
                r.get("val_threshold"),
            ),
        )
        await db.commit()


async def get_training_sessions(limit: int = 50) -> List[Dict]:
    async with _db() as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """SELECT * FROM cnn_training_sessions
               ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
