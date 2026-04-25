"""
Coinbase Advanced Trade — AI Trading Bot Backend
─────────────────────────────────────────────────
Signals: RSI · EMA Cross · MACD · Bollinger · CNN
Start:   uvicorn main:app --port 8001
"""
import sys
import os

_root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_venv_site = os.path.join(_root, ".venv", "Lib", "site-packages")
if os.path.isdir(_venv_site) and _venv_site not in sys.path:
    sys.path.insert(0, _venv_site)

import asyncio
import json
import re as _re
import time as _time
import logging
import subprocess
import threading
import webbrowser
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import database
from config import config
from clients import coinbase_client

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
    _CUDA_AVAILABLE  = _torch.cuda.is_available()
    _TORCH_DEVICE    = "cuda" if _CUDA_AVAILABLE else "cpu"
except ImportError:
    _TORCH_AVAILABLE = False
    _CUDA_AVAILABLE  = False
    _TORCH_DEVICE    = "cpu"
from agents.market_scanner import MarketScanner
from agents.signal_generator import SignalGenerator
from agents.order_executor import OrderExecutor
from agents.cnn_agent import CoinbaseCNNAgent
from agents.tech_agent_cb import TechAgentCB
from services.ws_subscriber import CoinbaseWSSubscriber
from services.portfolio_tracker import PortfolioTracker
from services.outcome_tracker import get_tracker
from services.history_backfill import get_backfill

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_level   = getattr(logging, config.log_level.upper(), logging.INFO)


class _WinSafeRotatingFileHandler(RotatingFileHandler):
    """
    RotatingFileHandler that survives WinError 32 (file-in-use) during rotation.
    On Windows, os.rename() fails if any handle still points to the source file.
    We close our own handle first, rename, then reopen.
    """
    def rotate(self, source: str, dest: str) -> None:
        if self.stream is not None:
            self.stream.close()      # release our handle before rename
            self.stream = None       # type: ignore[assignment]
        try:
            if os.path.exists(dest):
                os.remove(dest)
            os.rename(source, dest)
        except PermissionError:
            pass                     # another process holds it — skip silently
        self.stream = self._open()   # reopen fresh log file


_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file_all = _WinSafeRotatingFileHandler(
    os.path.join(_LOG_DIR, "backend.log"), maxBytes=5_000_000, backupCount=10, encoding="utf-8"
)
_file_all.setFormatter(_fmt)
_file_err = _WinSafeRotatingFileHandler(
    os.path.join(_LOG_DIR, "errors.log"), maxBytes=5_000_000, backupCount=10, encoding="utf-8"
)
_file_err.setLevel(logging.WARNING)
_file_err.setFormatter(_fmt)
logging.basicConfig(level=_level, handlers=[_console, _file_all, _file_err])

class _SuppressWinReset(logging.Filter):
    def filter(self, r: logging.LogRecord) -> bool:
        msg = r.getMessage()
        if "10054" in msg or "ConnectionResetError" in msg:
            return False
        # Also check exception info — asyncio logs the traceback as exc_info
        if r.exc_info and r.exc_info[1]:
            if "10054" in str(r.exc_info[1]) or "ConnectionResetError" in str(r.exc_info[1]):
                return False
        return True

_win_reset_filter = _SuppressWinReset()
# Apply to named loggers AND to all handlers so it never reaches the log files
logging.getLogger("asyncio").addFilter(_win_reset_filter)
logging.getLogger("uvicorn.error").addFilter(_win_reset_filter)
_file_err.addFilter(_win_reset_filter)
_file_all.addFilter(_win_reset_filter)

# Suppress httpx request-level chatter — our own code logs failures at WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_ERRORS_LOG = os.path.join(_LOG_DIR, "errors.log")
_ALL_LOG    = os.path.join(_LOG_DIR, "backend.log")
_LOG_RE     = _re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] ([\w.]+): (.+)$"
)


# ── Port cleanup ──────────────────────────────────────────────────────────────
_TRADING_APP_PORTS = {8000, 5173}

def _free_port(port: int) -> None:
    if port in _TRADING_APP_PORTS:
        return
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=5,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        for line in out.splitlines():
            if f":{port}" in line and "LISTEN" in line:
                parts = line.split()
                pid   = parts[-1]
                if pid.isdigit() and int(pid) != os.getpid():
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
                    logger.info("Freed port %d (killed PID %s)", port, pid)
    except Exception:
        pass


# ── Brave browser helpers ─────────────────────────────────────────────────────
_FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5174")

_BRAVE_PATHS = [
    r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    r"C:\Program Files (x86)\BraveSoftware\Brave-Browser\Application\brave.exe",
    os.path.expanduser(r"~\AppData\Local\BraveSoftware\Brave-Browser\Application\brave.exe"),
]

_brave_proc: Optional[subprocess.Popen] = None   # track the launched window


def _find_brave() -> Optional[str]:
    for p in _BRAVE_PATHS:
        if os.path.exists(p):
            return p
    return None


def _open_brave() -> None:
    global _brave_proc
    brave = _find_brave()
    if brave:
        # --new-window keeps it isolated so we can close just this window
        _brave_proc = subprocess.Popen([brave, "--new-window", _FRONTEND_URL])
        logger.info(f"Brave launched (PID {_brave_proc.pid}) → {_FRONTEND_URL}")
    else:
        logger.warning("Brave not found — falling back to default browser")
        webbrowser.open(_FRONTEND_URL)


def _close_brave() -> None:
    global _brave_proc
    if _brave_proc and _brave_proc.poll() is None:
        _brave_proc.terminate()
        logger.info(f"Brave window closed (PID {_brave_proc.pid})")
        _brave_proc = None


# ── App State ─────────────────────────────────────────────────────────────────
class AppState:
    scanner:         MarketScanner         = None
    signal_gen:      SignalGenerator       = None
    order_executor:  OrderExecutor         = None
    cnn_agent:       CoinbaseCNNAgent      = None
    tech_agent:      TechAgentCB           = None
    ws_subscriber:   CoinbaseWSSubscriber  = None
    portfolio:       PortfolioTracker      = None
    scanner_task:    asyncio.Task          = None
    portfolio_task:  asyncio.Task          = None
    cnn_task:        asyncio.Task          = None
    tech_task:       asyncio.Task          = None
    outcome_task:    asyncio.Task          = None
    backfill_task:   asyncio.Task          = None
    ws_connections:  List[WebSocket]       = []
    is_trading:      bool                  = False
    dry_run:         bool                  = True
    _balance_cache:  Optional[float]       = None   # cached live USD balance
    _balance_cache_ts: float               = 0.0    # epoch of last fetch
    # ── CNN background training state ─────────────────────────────────────
    train_status:       str                       = "idle"   # idle|running|completed|failed
    train_started_at:   Optional[float]           = None
    train_result:       Optional[Dict]            = None
    train_proc:         Optional[subprocess.Popen] = None   # subprocess handle
    train_watcher_task: Optional[asyncio.Task]    = None

app_state = AppState()


# ── Auth ──────────────────────────────────────────────────────────────────────
async def verify_api_key(x_api_key: str = Header(default="")) -> None:
    expected = config.app_api_key
    if not expected:
        logger.warning("APP_API_KEY not set — running without auth")
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ── WebSocket broadcast ───────────────────────────────────────────────────────
async def broadcast(payload: Dict) -> None:
    if not app_state.ws_connections:
        return
    text = json.dumps(payload)
    dead = []
    for ws in app_state.ws_connections:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        app_state.ws_connections.remove(ws)


async def broadcast_state() -> None:
    positions = await database.get_positions()
    signals   = await database.get_signals(limit=20)
    orders    = await database.get_orders(limit=30)
    products  = await database.get_products(tracked_only=True, limit=50)
    portfolio = app_state.portfolio.summary if app_state.portfolio else {}

    # Enrich products with live WS prices
    for p in products:
        live = app_state.ws_subscriber.state.get(p["product_id"]) if app_state.ws_subscriber else None
        if live and live.get("price"):
            p["price"] = live["price"]

    await broadcast({
        "type":       "state",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "is_trading": app_state.is_trading,
        "dry_run":    app_state.dry_run,
        "portfolio":  portfolio,
        "positions":  positions,
        "signals":    signals[:10],
        "orders":     orders,
        "products":   products[:20],
    })


# ── Training subprocess progress file ────────────────────────────────────────
_TRAIN_PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "cnn_train_progress.json")
_TRAIN_WORKER        = os.path.join(os.path.dirname(__file__), "train_worker.py")
_TRAIN_LOG_FILE      = os.path.join(os.path.dirname(__file__), "logs", "cnn_training.log")

# Watchdog thresholds for detecting a hung training subprocess.
# train_worker.py only writes the progress file at start and end, so its mtime
# is useless mid-run. We watch the training log mtime instead.
_TRAIN_STALE_START_SECS = 1800   # 30 min grace after start before staleness applies
_TRAIN_STALE_LOG_SECS   = 1800   # 30 min without log writes = stuck (phase-2 dataset
                                 # build logs every 10-13 min per 10 products, so the
                                 # old 15-min window tripped on normal cadence)


def _is_training_stale(data: dict, log_mtime, now: float) -> bool:
    """Return True when a 'running' training run should be considered hung.

    Stale iff status='running', ran past the startup grace, and the training
    log has not been updated for the log-staleness window. Missing log mtime
    is treated as 'unknown' → not stale, to avoid false positives on a fresh
    install where the log doesn't exist yet.
    """
    if data.get("status") != "running":
        return False
    started_at = data.get("started_at")
    if started_at is None:
        return False
    if now - started_at < _TRAIN_STALE_START_SECS:
        return False
    if log_mtime is None:
        return False
    return now - log_mtime >= _TRAIN_STALE_LOG_SECS


async def _train_progress_watcher() -> None:
    """Watch cnn_train_progress.json every 5 s and sync app_state from it.

    Detects when a training subprocess finishes and reloads the CNN model.
    """
    while True:
        await asyncio.sleep(5)
        try:
            if not os.path.exists(_TRAIN_PROGRESS_FILE):
                continue
            with open(_TRAIN_PROGRESS_FILE) as _f:
                _data = json.load(_f)
            _status = _data.get("status", "idle")

            if _status == "running" and app_state.train_status != "running":
                # Backend restarted while subprocess was training — re-sync,
                # but first verify the PID is still alive (guards against stale
                # "running" files left by a crashed subprocess).
                _pid = _data.get("pid")
                _pid_alive = False
                if _pid:
                    try:
                        import psutil as _psutil
                        _pid_alive = _psutil.pid_exists(int(_pid))
                    except ImportError:
                        import subprocess as _sp
                        _r = _sp.run(["tasklist", "/FI", f"PID eq {_pid}"], capture_output=True, text=True)
                        _pid_alive = str(_pid) in _r.stdout
                if not _pid_alive:
                    logger.warning(
                        f"Train progress file shows 'running' but PID {_pid} is gone — "
                        "marking failed and clearing training_active"
                    )
                    _data["status"] = "failed"
                    with open(_TRAIN_PROGRESS_FILE, "w") as _wf:
                        json.dump(_data, _wf)
                    if app_state.cnn_agent:
                        app_state.cnn_agent.training_active = False
                else:
                    app_state.train_status    = "running"
                    app_state.train_started_at = _data.get("started_at", _time.time())
                    if app_state.cnn_agent:
                        app_state.cnn_agent.training_active = True

            if _status == "running":
                _log_mtime = os.path.getmtime(_TRAIN_LOG_FILE) if os.path.exists(_TRAIN_LOG_FILE) else None
                if _is_training_stale(_data, _log_mtime, _time.time()):
                    _pid = _data.get("pid")
                    logger.warning(
                        f"Training PID {_pid} appears hung — "
                        f"log unchanged for >{_TRAIN_STALE_LOG_SECS//60} min. "
                        "Killing subprocess and marking failed."
                    )
                    if _pid:
                        try:
                            import subprocess as _sp
                            _sp.run(["taskkill", "/F", "/T", "/PID", str(_pid)],
                                    capture_output=True, text=True)
                        except Exception as _ke:
                            logger.warning(f"taskkill for PID {_pid} failed: {_ke}")
                    _data["status"] = "failed"
                    _data["result"] = {"error": "watchdog: log stale, subprocess killed"}
                    with open(_TRAIN_PROGRESS_FILE, "w") as _wf:
                        json.dump(_data, _wf)
                    _status = "failed"
                    # fall through to the status=failed transition branch below

            if _status in ("completed", "failed") and app_state.train_status == "running":
                # Training just finished — update state and reload model from disk
                app_state.train_status = _status
                app_state.train_result = _data.get("result")
                if app_state.cnn_agent:
                    app_state.cnn_agent.training_active = False
                if _status == "completed":
                    try:
                        app_state.cnn_agent._load()
                        logger.info("CNN model reloaded after subprocess training completed")
                    except Exception as _le:
                        logger.warning(f"CNN model reload failed: {_le}")
        except Exception as _we:
            logger.debug(f"Train progress watcher: {_we}")


# ── Agent startup stagger delays (seconds) ────────────────────────────────────
_TECH_START_DELAY     =  5   # CNN starts at 0; Tech after 5s


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()

    # Trim old decision history (keep 7 days — CNN only needs last 2 rows per product)
    deleted = await database.purge_old_decisions(days=7)
    if deleted:
        logger.info(f"Startup: purged {deleted} agent_decisions older than 7 days")

    # Instantiate all objects (no I/O, fast)
    app_state.ws_subscriber   = CoinbaseWSSubscriber(broadcast_fn=broadcast)
    app_state.scanner         = MarketScanner()
    app_state.signal_gen      = SignalGenerator()
    app_state.order_executor  = OrderExecutor(dry_run=app_state.dry_run)
    app_state.cnn_agent       = CoinbaseCNNAgent(ws_subscriber=app_state.ws_subscriber)
    _is_trading = lambda: app_state.is_trading
    app_state.tech_agent      = TechAgentCB(ws_subscriber=app_state.ws_subscriber)
    app_state.portfolio       = PortfolioTracker(ws_subscriber=app_state.ws_subscriber)

    # Seed WS subscriber from DB-cached tracked products immediately —
    # don't wait for the scan; the scan will refresh this list in the background.
    cached_products = await database.get_products(tracked_only=True, limit=200)
    cached_ids      = [p["product_id"] for p in cached_products]
    if cached_ids:
        app_state.ws_subscriber.set_products(cached_ids)
        logger.info(f"WS seeded from DB cache: {len(cached_ids)} products")
    await app_state.ws_subscriber.start()

    # Background scan — refreshes product list without blocking startup
    async def _background_scan():
        try:
            await app_state.scanner.scan()
            app_state.ws_subscriber.set_products(list(app_state.scanner.tracked_ids))
            logger.info(f"Background scan complete: {len(app_state.scanner.tracked_ids)} products")
        except Exception as e:
            logger.warning(f"Background scan failed: {e}")

    asyncio.create_task(_background_scan())

    app_state.scanner_task   = asyncio.create_task(app_state.scanner.run_loop())
    app_state.portfolio_task = asyncio.create_task(app_state.portfolio.run_loop())
    app_state.outcome_task   = asyncio.create_task(get_tracker().run_loop())
    app_state.backfill_task  = asyncio.create_task(get_backfill().run_loop())

    # CNN loop — starts immediately
    async def _auto_train_subprocess():
        """Spawn train_worker subprocess for auto-train — mirrors the UI Train button."""
        if app_state.train_status == "running":
            logger.info("CNN auto-train skipped — training subprocess already running")
            return
        app_state.train_status     = "running"
        app_state.train_started_at = _time.time()
        app_state.train_result     = None
        try:
            proc = subprocess.Popen(
                [sys.executable, _TRAIN_WORKER, "--epochs", "50"],
                cwd=os.path.dirname(__file__),
            )
            app_state.train_proc = proc
            app_state.cnn_agent.training_active = True
            logger.info(f"CNN auto-train subprocess started — PID {proc.pid}")
        except Exception as exc:
            app_state.train_status = "failed"
            logger.error(f"CNN auto-train subprocess failed to start: {exc}")

    app_state.cnn_task = asyncio.create_task(
        app_state.cnn_agent.run_loop(
            order_executor      = app_state.order_executor,
            is_trading_fn       = lambda: app_state.is_trading,
            train_every_n_scans = config.cnn_train_every_n_scans,
            broadcast_fn        = broadcast_state,
            auto_train_fn       = _auto_train_subprocess,
        )
    )

    # Sub-agents: short stagger so they don't all hammer the DB simultaneously
    async def _delayed_tech():
        await asyncio.sleep(_TECH_START_DELAY)
        await app_state.tech_agent.run_loop(is_trading_fn=_is_trading)

    app_state.tech_task     = asyncio.create_task(_delayed_tech())
    app_state.train_watcher_task  = asyncio.create_task(_train_progress_watcher())

    logger.info("Coinbase Trader ready — http://localhost:8001")
    logger.info(f"Credentials: {'OK' if config.has_credentials else 'MISSING'}")
    logger.info(f"Mode: {'DRY-RUN' if app_state.dry_run else 'LIVE'}")
    logger.info(f"WS tracking {len(cached_ids)} products (scan refreshing in background)")

    yield

    for task in [
        app_state.scanner_task, app_state.portfolio_task,
        app_state.cnn_task, app_state.tech_task,
        app_state.outcome_task, app_state.backfill_task,
        app_state.train_watcher_task,
    ]:
        if task:
            task.cancel()
    await app_state.ws_subscriber.stop()
    logger.info("App shut down")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Coinbase Trader", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:3000"],
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ── Status ────────────────────────────────────────────────────────────────────
_STATUS_BALANCE_TTL = 30.0   # refresh live balance at most every 30 s

@app.get("/api/status")
async def get_status():
    import time as _time
    balance = None
    if config.has_credentials:
        now = _time.time()
        if now - app_state._balance_cache_ts >= _STATUS_BALANCE_TTL:
            # Stamp BEFORE the await — prevents thundering herd when many
            # concurrent requests arrive while the TTL has expired.
            app_state._balance_cache_ts = now
            try:
                app_state._balance_cache = await asyncio.wait_for(
                    coinbase_client.get_usd_balance(), timeout=3.0
                )
            except Exception:
                pass
        balance = app_state._balance_cache
    dry_balance = (
        app_state.order_executor._dry_run_balance
        if app_state.order_executor and app_state.dry_run else None
    )
    return {
        "is_trading":      app_state.is_trading,
        "dry_run":         app_state.dry_run,
        "has_creds":       config.has_credentials,
        "usd_balance":     balance,
        "dry_run_balance": dry_balance,
        "tracked_pairs":   len(app_state.scanner.tracked_ids) if app_state.scanner else 0,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "app_api_key":     config.app_api_key,   # local app — key exposed for UI use
    }


# ── Products ──────────────────────────────────────────────────────────────────
@app.get("/api/products")
async def get_products(
    tracked: bool = Query(False),
    limit: int    = Query(50, le=200),
):
    products = await database.get_products(tracked_only=tracked, limit=limit)
    # Enrich with live WS price
    for p in products:
        live = app_state.ws_subscriber.state.get(p["product_id"]) if app_state.ws_subscriber else None
        if live and live.get("price"):
            p["price"] = live["price"]
    return products


@app.get("/api/products/{product_id}")
async def get_product(product_id: str):
    p = await database.get_product(product_id)
    if not p:
        try:
            raw = await coinbase_client.get_product(product_id)
            if raw:
                return raw
        except Exception:
            pass
        raise HTTPException(404, "Product not found")
    return p


@app.post("/api/products/{product_id}/track")
async def track_product(product_id: str, _: None = Depends(verify_api_key)):
    await database.set_product_tracked(product_id, True)
    app_state.scanner.tracked_ids.add(product_id)
    if app_state.ws_subscriber:
        app_state.ws_subscriber.set_products(list(app_state.scanner.tracked_ids))
    return {"success": True, "product_id": product_id}


# ── Prices / Candles / Orderbook ──────────────────────────────────────────────
@app.get("/api/price/{product_id}")
async def get_price(product_id: str):
    price = app_state.ws_subscriber.get_price(product_id) if app_state.ws_subscriber else None
    if price is None:
        try:
            bba   = await coinbase_client.get_best_bid_ask([product_id])
            price = bba.get(product_id, {}).get("price")
        except Exception:
            pass
    return {"product_id": product_id, "price": price}


@app.get("/api/candles/{product_id}")
async def get_candles(
    product_id:  str,
    granularity: str = Query("ONE_HOUR"),
    limit:       int = Query(100, le=300),
):
    candles = await database.get_candles(product_id, limit=limit)
    if not candles:
        candles = await coinbase_client.get_candles(product_id, granularity, limit)
    return candles


@app.get("/api/orderbook/{product_id}")
async def get_orderbook(product_id: str):
    book = await coinbase_client.get_orderbook(product_id, limit=10)
    bid_ask = app_state.ws_subscriber.get_bid_ask(product_id) if app_state.ws_subscriber else {}
    return {"product_id": product_id, **book, **bid_ask}


# ── Portfolio / Positions ─────────────────────────────────────────────────────
@app.get("/api/portfolio")
async def get_portfolio():
    return await database.get_portfolio_summary()


@app.get("/api/positions")
async def get_positions():
    return await database.get_positions()


@app.post("/api/positions/sync")
async def sync_positions(_: None = Depends(verify_api_key)):
    summary = await app_state.portfolio.sync()
    return {"success": True, "summary": summary}


# ── CNN Status ────────────────────────────────────────────────────────────────
@app.get("/api/cnn/status")
async def get_cnn_status():
    agent = app_state.cnn_agent
    dry_balance = (
        app_state.order_executor._dry_run_balance
        if app_state.order_executor and app_state.dry_run else None
    )
    dd_status = (
        app_state.order_executor.drawdown_status
        if app_state.order_executor else {}
    )
    return {
        "torch_available":  _TORCH_AVAILABLE,
        "cuda_available":   _CUDA_AVAILABLE,
        "device":           _TORCH_DEVICE,
        "model_loaded":     agent._exists() if agent else False,
        "last_scan_at":     agent.last_scan_at  if agent else None,
        "next_scan_at":     agent.next_scan_at  if agent else None,
        "scan_count":       agent.scan_count    if agent else 0,
        "signals_total":    agent.signals_total if agent else 0,
        "signals_buy":      agent.signals_buy   if agent else 0,
        "signals_sell":     agent.signals_sell  if agent else 0,
        "signals_executed": agent.signals_executed if agent else 0,
        "dry_run":          app_state.dry_run,
        "dry_run_balance":  dry_balance,
        "is_trading":       app_state.is_trading,
        "drawdown":         dd_status,
        "last_trained_at":  agent.last_trained_at if agent else None,
        "train_count":      agent.train_count     if agent else 0,
        # Win rate tracking
        "wins":             agent.book.wins        if agent else 0,
        "losses":           agent.book.losses      if agent else 0,
        "win_rate":         round(agent.book.win_rate * 100, 1) if agent else 0.0,
        "expectancy_pct":   round(agent.book.expectancy, 3)     if agent else 0.0,
        # LLM token usage
        "llm_calls":           agent.llm_calls           if agent else 0,
        "llm_prompt_tokens":   agent.llm_prompt_tokens   if agent else 0,
        "llm_response_tokens": agent.llm_response_tokens if agent else 0,
        "llm_total_tokens":    (agent.llm_prompt_tokens + agent.llm_response_tokens) if agent else 0,
        "training_active":     agent.training_active     if agent else False,
        "train_status":        app_state.train_status,
    }


# ── CNN Scans (all reviewed products) ─────────────────────────────────────────
@app.get("/api/cnn/scans")
async def get_cnn_scans(
    limit:      int           = Query(500, le=2000),
    product_id: Optional[str] = Query(None),
):
    return await database.get_cnn_scans(limit=limit, product_id=product_id)


# ── Signals ───────────────────────────────────────────────────────────────────
@app.get("/api/signals")
async def get_signals(
    limit:       int           = Query(50, le=500),
    signal_type: Optional[str] = Query(None),
):
    return await database.get_signals(limit=limit, signal_type_prefix=signal_type)


@app.post("/api/signals/scan")
async def trigger_signal_scan(_: None = Depends(verify_api_key)):
    signals = await app_state.signal_gen.scan_all()
    await broadcast_state()
    return {"signals_generated": len(signals), "signals": signals}


# ── CNN ───────────────────────────────────────────────────────────────────────
@app.post("/api/cnn/scan")
async def trigger_cnn_scan(
    execute: bool = Query(False),
    _: None = Depends(verify_api_key),
):
    executor = app_state.order_executor if (execute and app_state.is_trading) else None
    signals  = await app_state.cnn_agent.scan_all(
        execute=execute and app_state.is_trading,
        order_executor=executor,
    )
    await broadcast_state()
    return {"signals_generated": len(signals), "signals": signals}




@app.post("/api/cnn/train")
async def trigger_cnn_training(
    epochs: int = Query(50, ge=1, le=500),
    _: None = Depends(verify_api_key),
):
    """
    Spawns train_worker.py as a subprocess so training survives backend restarts.
    Poll GET /api/cnn/train/status for progress and results.
    Returns 409 if a training run is already in progress.
    """
    # Check in-memory state first
    if app_state.train_status == "running":
        elapsed = int(_time.time() - (app_state.train_started_at or _time.time()))
        return JSONResponse(
            status_code=409,
            content={"status": "running", "elapsed_secs": elapsed,
                     "detail": "Training already in progress"},
        )
    # Also check progress file — catches case where backend restarted mid-training
    if os.path.exists(_TRAIN_PROGRESS_FILE):
        try:
            with open(_TRAIN_PROGRESS_FILE) as _f:
                _pd = json.load(_f)
            if _pd.get("status") == "running":
                app_state.train_status    = "running"
                app_state.train_started_at = _pd.get("started_at", _time.time())
                return JSONResponse(
                    status_code=409,
                    content={"status": "running", "detail": "Training subprocess already running"},
                )
        except Exception:
            pass

    app_state.train_status    = "running"
    app_state.train_started_at = _time.time()
    app_state.train_result    = None

    try:
        proc = subprocess.Popen(
            [sys.executable, _TRAIN_WORKER, "--epochs", str(epochs)],
            cwd=os.path.dirname(__file__),
        )
        app_state.train_proc = proc
        if app_state.cnn_agent:
            app_state.cnn_agent.training_active = True
        logger.info(f"CNN training subprocess started — PID {proc.pid}, epochs={epochs}")
    except Exception as exc:
        app_state.train_status = "failed"
        app_state.train_result = {"error": str(exc)}
        raise HTTPException(status_code=500, detail=f"Failed to start training worker: {exc}")

    return {"status": "started", "epochs": epochs}


@app.get("/api/cnn/train/status")
async def cnn_train_status():
    """Poll this endpoint after POST /api/cnn/train to get progress and results."""
    if os.path.exists(_TRAIN_PROGRESS_FILE):
        try:
            with open(_TRAIN_PROGRESS_FILE) as _f:
                _data = json.load(_f)
            _status     = _data.get("status", "idle")
            _started_at = _data.get("started_at", app_state.train_started_at)
            _elapsed    = (
                int(_time.time() - _started_at) if _started_at and _status == "running" else 0
            )
            return {"status": _status, "elapsed_secs": _elapsed, "result": _data.get("result")}
        except Exception:
            pass
    # Fallback to in-memory state (no progress file yet)
    elapsed = int(_time.time() - app_state.train_started_at) if app_state.train_started_at else 0
    return {"status": app_state.train_status, "elapsed_secs": elapsed, "result": app_state.train_result}


@app.get("/api/cnn/training-history")
async def get_training_history(limit: int = Query(50, le=200)):
    """Return all persisted CNN training sessions, newest first."""
    return await database.get_training_sessions(limit=limit)


@app.post("/api/cnn/model/reload")
async def reload_cnn_model(_: None = Depends(verify_api_key)):
    """
    Hot-reload cnn_model.pt from disk into the running CNNAgent.

    Use after dropping new weights from a cloud training run (#69):
      1. Train on Colab via tools/colab_train.ipynb
      2. Download cnn_model.pt + cnn_best_loss.txt into backend/
      3. POST here — no backend restart needed

    Returns:
      - 409 if a local training run is currently active (would race the save)
      - 404 if cnn_model.pt is missing
      - 503 if the CNN agent isn't initialised yet (lifespan still starting)
      - 200 with file metadata on success
    """
    if app_state.train_status == "running":
        return JSONResponse(
            status_code=409,
            content={"detail": "Local training in progress; reload would race the on-disk save"},
        )
    from agents.cnn_agent import MODEL_PATH, N_CHANNELS
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail=f"Model file not found: {MODEL_PATH}")
    if not app_state.cnn_agent:
        raise HTTPException(status_code=503, detail="CNN agent not initialised")
    app_state.cnn_agent._load()
    st = os.stat(MODEL_PATH)
    logger.info(
        f"CNN model reloaded via /api/cnn/model/reload "
        f"(size={st.st_size} bytes, mtime={int(st.st_mtime)})"
    )
    return {
        "status": "reloaded",
        "model_path": MODEL_PATH,
        "size_bytes": st.st_size,
        "mtime_unix": st.st_mtime,
        "n_channels_expected": N_CHANNELS,
    }


@app.post("/api/cnn/lgbm/retrain")
async def force_lgbm_retrain(_: None = Depends(verify_api_key)):
    """
    Force a LightGBM filter retrain immediately, ignoring the trades-seen
    short-circuit (#70). Use after /api/cnn/model/reload so the gate is
    re-fit against the freshly loaded CNN's scan distribution rather than
    waiting for the next closed trade to trip auto-retrain.

    Returns:
      - 503 if the CNN agent isn't initialised
      - 200 {"status": "skipped"} if there are no rows or the fit returned None
      - 200 {"status": "retrained", "metrics": {...}} on success
    """
    if not app_state.cnn_agent:
        raise HTTPException(status_code=503, detail="CNN agent not initialised")
    metrics = await app_state.cnn_agent.force_lgbm_retrain()
    if metrics is None:
        return {"status": "skipped", "detail": "no training rows or fit returned None"}
    return {"status": "retrained", "metrics": metrics}


# ── History Backfill ──────────────────────────────────────────────────────────
@app.post("/api/history/backfill")
async def trigger_history_backfill(
    days:          int           = Query(365, ge=7, le=730),
    product_id:    Optional[str] = Query(None, description="Single product ID"),
    all_coinbase:  bool          = Query(False, description="Backfill ALL Coinbase USD pairs"),
    _: None = Depends(verify_api_key),
):
    """
    Fetch historical hourly candles from Coinbase and write to parquet files.
    Runs as a background task — never blocks agents.
    Incremental — safe to re-run; only downloads bars not already stored.
    Returns immediately with {"status": "started"}.
    """
    pids = [product_id] if product_id else None

    async def _run():
        try:
            result = await get_backfill().run(days=days, product_ids=pids, all_coinbase=all_coinbase)
            logger.info(f"Backfill complete: {result['total_products']} products | {result['total_new_bars']} new bars")
        except Exception as e:
            logger.error(f"Backfill error: {e}")

    asyncio.create_task(_run())
    scope = "all Coinbase USD pairs" if all_coinbase else (product_id or "tracked products")
    return {"status": "started", "days": days, "scope": scope}


@app.get("/api/history/status")
async def get_history_status():
    """List all products that have parquet history files and their bar counts."""
    import os
    from services.history_backfill import _HISTORY_DIR, load_history
    if not os.path.exists(_HISTORY_DIR):
        return {"files": []}
    files = []
    for fname in sorted(os.listdir(_HISTORY_DIR)):
        if not fname.endswith(".parquet"):
            continue
        pid    = fname.replace(".parquet", "").replace("_", "-")
        candles = load_history(pid)
        if candles:
            files.append({
                "product_id": pid,
                "bars":       len(candles),
                "oldest":     candles[0]["start"],
                "newest":     candles[-1]["start"],
            })
    return {"files": files}


# ── Orders ────────────────────────────────────────────────────────────────────
@app.get("/api/orders")
async def get_orders(
    status: Optional[str] = Query(None),
    limit:  int           = Query(100, le=500),
):
    return await database.get_orders(status=status, limit=limit)


class SideEnum(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"

class OrderTypeEnum(str, Enum):
    LIMIT  = "LIMIT"
    MARKET = "MARKET"

class PlaceOrderRequest(BaseModel):
    product_id: str
    side:       SideEnum
    order_type: OrderTypeEnum
    price:      Optional[float] = None   # required for LIMIT
    quote_size: float           = 100.0  # USD to spend
    strategy:   str             = "MANUAL"

    @field_validator("quote_size")
    @classmethod
    def size_positive(cls, v):
        if v <= 0:
            raise ValueError("quote_size must be positive")
        return v


@app.post("/api/orders")
async def place_order(req: PlaceOrderRequest, _: None = Depends(verify_api_key)):
    p = await database.get_product(req.product_id)
    if not p:
        raise HTTPException(404, "Product not tracked — add it first")

    price = req.price or p.get("price", 0)
    if not price:
        raise HTTPException(400, "No price available")

    if req.order_type == OrderTypeEnum.MARKET:
        result = await app_state.order_executor.execute_market_order(
            req.product_id, req.side.value, req.quote_size
        )
    else:
        signal = {
            "product_id":  req.product_id,
            "side":        req.side.value,
            "price":       price,
            "quote_size":  req.quote_size,
            "signal_type": req.strategy,
        }
        result = await app_state.order_executor.execute_signal(signal)

    await broadcast_state()
    return result


@app.delete("/api/orders/{order_id}")
async def cancel_order(order_id: str, _: None = Depends(verify_api_key)):
    return await app_state.order_executor.cancel_order(order_id)


# ── Trading control ───────────────────────────────────────────────────────────
@app.post("/api/trading/enable")
async def enable_trading(_: None = Depends(verify_api_key)):
    dry_run = config.dry_run
    app_state.dry_run    = dry_run
    app_state.is_trading = True
    app_state.order_executor = OrderExecutor(dry_run=dry_run)
    await broadcast({"type": "trading_status", "is_trading": True, "dry_run": dry_run})
    return {"success": True, "is_trading": True, "dry_run": dry_run}


@app.post("/api/trading/disable")
async def disable_trading(_: None = Depends(verify_api_key)):
    app_state.is_trading = False
    await broadcast({"type": "trading_status", "is_trading": False})
    # Close the Brave window 1.5 s after responding so the frontend sees the reply
    threading.Timer(1.5, _close_brave).start()
    return {"success": True, "is_trading": False}


# ── Scanner ───────────────────────────────────────────────────────────────────
@app.post("/api/scanner/run")
async def trigger_scan(_: None = Depends(verify_api_key)):
    if app_state.scanner.is_scanning:
        return {"success": False, "reason": "Scan in progress"}
    products = await app_state.scanner.scan()
    await broadcast_state()
    return {"success": True, "tracked_products": len(products)}


# ── Balance ───────────────────────────────────────────────────────────────────
@app.get("/api/balance")
async def get_balance():
    if not config.has_credentials:
        return {"usd_balance": None, "message": "No credentials configured"}
    try:
        import time as _time
        accounts = await coinbase_client.get_accounts()
        # Derive USD balance from already-fetched accounts — avoids a second get_accounts() call
        usd = next(
            (float(a["available"]) for a in accounts if a.get("currency") == "USD"),
            0.0,
        )
        # Refresh the shared status cache so /api/status doesn't need a separate call
        app_state._balance_cache    = usd
        app_state._balance_cache_ts = _time.time()
        return {"accounts": accounts, "usd_balance": usd}
    except Exception as e:
        return {"error": str(e)}


# ── Logs ─────────────────────────────────────────────────────────────────────
def _tail_lines(path: str, n: int) -> List[str]:
    """Return the last n lines of a file by seeking from the end — O(result) not O(file)."""
    BLOCK = 65536
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        buf = b""
        lines_found = 0
        pos = size
        while pos > 0 and lines_found <= n:
            read = min(BLOCK, pos)
            pos -= read
            f.seek(pos)
            chunk = f.read(read)
            buf = chunk + buf
            lines_found = buf.count(b"\n")
        raw = buf.decode("utf-8", errors="replace").splitlines()
    return raw[-n:] if len(raw) > n else raw


def _read_log_file(path: str, min_level: int, limit: int) -> List[Dict]:
    """Read and parse the rotating log file, return newest-first entries.
    Reads only the tail of the file — O(limit) not O(file size)."""
    entries: List[Dict] = []
    if not os.path.exists(path):
        return entries
    try:
        # Over-read by 10× to account for HOLD entries filtered by min_level
        lines = _tail_lines(path, limit * 10)
        current: Optional[Dict] = None
        for line in lines:
            line = line.rstrip("\n")
            m = _LOG_RE.match(line)
            if m:
                if current:
                    entries.append(current)
                ts, lvl, name, msg = m.groups()
                if logging.getLevelName(lvl) >= min_level:
                    current = {"time": ts, "level": lvl, "logger": name, "message": msg}
                else:
                    current = None
            elif current:
                current["message"] += "\n" + line
        if current:
            entries.append(current)
    except Exception as e:
        logger.warning(f"Log file read error: {e}")
    entries.reverse()   # newest first
    return entries[:limit]


@app.get("/api/logs")
async def get_logs(
    level:  str = Query("WARNING", description="Minimum level: WARNING or ERROR"),
    limit:  int = Query(200, le=500),
):
    min_level = logging.ERROR if level.upper() == "ERROR" else logging.WARNING
    # errors.log already contains WARNING+ so use it for WARNING filter;
    # for ERROR-only filter we still use errors.log and filter client-side
    entries = _read_log_file(_ERRORS_LOG, min_level, limit)
    return {"total": len(entries), "entries": entries}


@app.delete("/api/logs")
async def clear_logs(_: None = Depends(verify_api_key)):
    for path in (_ERRORS_LOG, _ALL_LOG):
        try:
            if os.path.exists(path):
                open(path, "w").close()
        except Exception:
            pass
    return {"success": True}


# ── Sub-agent status & decisions ─────────────────────────────────────────────

@app.get("/api/agents/status")
async def get_agent_status():
    tech_status = app_state.tech_agent.status     if app_state.tech_agent     else {}

    # CNN book status
    cnn_book = app_state.cnn_agent.book if app_state.cnn_agent else None
    cnn_status: Dict = {}
    if cnn_book:
        cnn_agent = app_state.cnn_agent
        cnn_status = {
            "agent":          "CNN",
            "balance":        round(cnn_book.balance, 2),
            "realized_pnl":   round(cnn_book.realized_pnl, 2),
            "open_positions": len(cnn_book.positions),
            "scan_count":     cnn_agent.scan_count     if cnn_agent else 0,
            "last_scan_at":   cnn_agent.last_scan_at   if cnn_agent else None,
            "signals_buy":    cnn_agent.signals_buy     if cnn_agent else 0,
            "signals_sell":   cnn_agent.signals_sell    if cnn_agent else 0,
            "positions":      {
                pid: {"size": round(p["size"], 6), "avg_price": round(p["avg_price"], 6)}
                for pid, p in cnn_book.positions.items()
            },
        }

    # Enrich positions with live price + unrealized PnL
    ws = app_state.ws_subscriber

    # Collect ALL held product IDs across all agents so we can ensure WS coverage
    all_held_pids: set = set()
    for st in (tech_status, cnn_status):
        all_held_pids.update(st.get("positions", {}).keys())

    # Subscribe WS to any held product not already in its subscription list
    # (catches positions opened before MIN_PRICE was raised)
    if ws and all_held_pids:
        current_subs = set(ws._products)
        missing      = all_held_pids - current_subs
        if missing:
            ws.set_products(list(current_subs | missing))
            logger.info(f"WS: added {len(missing)} held-but-untracked products — reconnecting: {missing}")
            asyncio.create_task(ws.reconnect())

    # REST fallback: for held products with no WS price, fetch via product endpoint and seed WS state
    if ws and all_held_pids:
        no_price = [pid for pid in all_held_pids if not ws.get_price(pid)]
        if no_price:
            async def _fetch_price(pid: str) -> None:
                try:
                    prod = await coinbase_client.get_product(pid)
                    price_str = prod.get("price") if prod else None
                    if price_str:
                        rest_price = float(price_str)
                        ws.state[pid] = {"product_id": pid, "price": rest_price,
                                         "bid": None, "ask": None, "volume_24h": 0.0, "pct_change": 0.0}
                        logger.debug(f"REST price fallback: {pid} = {rest_price}")
                except Exception as _e:
                    logger.debug(f"REST price fallback failed for {pid}: {_e}")
            await asyncio.gather(*[_fetch_price(pid) for pid in no_price])

    for st in (tech_status, cnn_status):
        positions = st.get("positions", {})
        for pid, pos in positions.items():
            live = ws.get_price(pid) if ws else None
            entry = pos.get("avg_price", 0)

            # Flag zero-entry positions — data is corrupt (opened before price was available)
            pos["entry_corrupt"] = entry == 0

            if live and entry > 0:
                pos["current_price"]  = round(live, 6)
                pos["unrealized_pnl"] = round((live - entry) * pos["size"], 4)
                pos["pct_pnl"]        = round((live - entry) / entry * 100, 2)
            else:
                pos["current_price"]  = None
                pos["unrealized_pnl"] = None
                pos["pct_pnl"]        = None

    return {"tech": tech_status, "cnn": cnn_status}


@app.get("/api/trades")
async def get_trades(
    agent:       Optional[str] = Query(None),
    product_id:  Optional[str] = Query(None),
    open_only:   bool          = Query(False),
    closed_only: bool          = Query(False),
    limit:       int           = Query(100, le=1000),
):
    return await database.get_trades(
        agent=agent, product_id=product_id,
        open_only=open_only, closed_only=closed_only, limit=limit,
    )


@app.get("/api/performance")
async def get_performance():
    """
    Monthly performance breakdown + rolling metrics for the dashboard.
    Returns monthly stats, cumulative balance curve, and $50k/yr projection.
    """
    import aiosqlite
    from database import DB_PATH, _DB_TIMEOUT
    async with aiosqlite.connect(DB_PATH, timeout=_DB_TIMEOUT) as db:
        db.row_factory = aiosqlite.Row

        # Monthly closed-trade aggregates — single-pass using window functions
        # (avoids 2 correlated subqueries per month that caused full-table rescans)
        cursor = await db.execute("""
            WITH ranked AS (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY strftime('%Y-%m', closed_at)
                        ORDER BY closed_at ASC
                    ) AS rn_asc,
                    ROW_NUMBER() OVER (
                        PARTITION BY strftime('%Y-%m', closed_at)
                        ORDER BY closed_at DESC
                    ) AS rn_desc
                FROM trades
                WHERE closed_at IS NOT NULL
            )
            SELECT
                strftime('%Y-%m', closed_at)                          AS month,
                COUNT(*)                                               AS trades,
                SUM(CASE WHEN pnl > 0  THEN 1 ELSE 0 END)             AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)             AS losses,
                ROUND(SUM(pnl), 2)                                     AS total_pnl,
                ROUND(AVG(pct_pnl), 2)                                 AS avg_pct_pnl,
                ROUND(AVG(CASE WHEN pnl > 0  THEN pnl END), 4)        AS avg_win,
                ROUND(AVG(CASE WHEN pnl <= 0 THEN pnl END), 4)        AS avg_loss,
                MIN(balance_after)                                     AS low_balance,
                MAX(balance_after)                                     AS high_balance,
                MAX(CASE WHEN rn_asc  = 1 THEN balance_after END)     AS open_balance,
                MAX(CASE WHEN rn_desc = 1 THEN balance_after END)     AS close_balance
            FROM ranked
            GROUP BY month
            ORDER BY month ASC
        """)
        months = [dict(r) for r in await cursor.fetchall()]

        # Add win_rate, expectancy, and monthly_return_pct per month
        for m in months:
            t = (m["wins"] or 0) + (m["losses"] or 0)
            wr = (m["wins"] or 0) / t if t else 0.0
            m["win_rate"] = round(wr * 100, 1)
            avg_win  = m["avg_win"]  or 0.0
            avg_loss = m["avg_loss"] or 0.0
            # Expectancy = expected $ per trade (negative = losing system)
            m["expectancy"] = round(wr * avg_win + (1 - wr) * avg_loss, 3)
            ob  = m["open_balance"] or 0
            pnl = m["total_pnl"]    or 0
            m["monthly_return_pct"] = round(pnl / ob * 100, 2) if ob > 0 else 0.0

        # Rolling 30-day stats (last 30 days of closed trades)
        cursor = await db.execute("""
            SELECT
                COUNT(*)                                   AS trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)  AS wins,
                ROUND(SUM(pnl), 2)                         AS total_pnl,
                (SELECT balance_after FROM trades
                 WHERE closed_at IS NOT NULL
                   AND closed_at >= datetime('now', '-30 days')
                 ORDER BY closed_at ASC  LIMIT 1)          AS first_balance,
                (SELECT balance_after FROM trades
                 WHERE closed_at IS NOT NULL
                   AND closed_at >= datetime('now', '-30 days')
                 ORDER BY closed_at DESC LIMIT 1)          AS last_balance
            FROM trades
            WHERE closed_at IS NOT NULL
              AND closed_at >= datetime('now', '-30 days')
        """)
        r30 = dict(await cursor.fetchone())
        t30 = r30["trades"] or 0
        w30 = r30["wins"] or 0
        fb30 = r30["first_balance"] or 0
        rolling_return_pct = round((r30["total_pnl"] or 0) / fb30 * 100, 2) if fb30 > 0 else 0.0
        rolling_win_rate   = round(w30 / t30 * 100, 1) if t30 else 0.0

        # All-time stats
        cursor = await db.execute("""
            SELECT
                COUNT(*)                                  AS total_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS total_wins,
                ROUND(SUM(pnl), 2)                        AS total_pnl,
                MIN(balance_after)                        AS first_balance,
                MAX(balance_after)                        AS peak_balance,
                (SELECT balance_after FROM trades
                 WHERE closed_at IS NOT NULL ORDER BY closed_at DESC LIMIT 1) AS current_balance
            FROM trades WHERE closed_at IS NOT NULL
        """)
        overall = dict(await cursor.fetchone())

    curr_bal = overall["current_balance"] or 0
    first_bal = overall["first_balance"] or curr_bal or 1
    total_t = overall["total_trades"] or 0
    total_w = overall["total_wins"] or 0
    all_time_win_rate = round(total_w / total_t * 100, 1) if total_t else 0.0

    # Projection: use trailing 30-day monthly return to project path to $50k/year
    # $50k/year means the account needs to generate $50k annually
    # We estimate months_to_goal based on trailing rate
    annual_goal_usd = 50_000
    projected_months: Optional[float] = None
    if rolling_return_pct > 0 and curr_bal > 0:
        monthly_rate = rolling_return_pct / 100
        # months until annual income (12 * monthly_pnl) >= $50k
        # 12 * bal * (1+r)^n * r >= 50000  → solve numerically
        bal = curr_bal
        for n in range(1, 600):
            bal *= (1 + monthly_rate)
            if 12 * bal * monthly_rate >= annual_goal_usd:
                projected_months = n
                break

    return {
        "months":               months,
        "rolling_30d": {
            "trades":         t30,
            "wins":           w30,
            "win_rate_pct":   rolling_win_rate,
            "return_pct":     rolling_return_pct,
            "total_pnl":      round(r30["total_pnl"] or 0, 2),
        },
        "overall": {
            "total_trades":   total_t,
            "total_wins":     total_w,
            "win_rate_pct":   all_time_win_rate,
            "total_pnl":      round(overall["total_pnl"] or 0, 2),
            "current_balance": round(curr_bal, 2),
            "peak_balance":    round(overall["peak_balance"] or 0, 2),
        },
        "projection": {
            "annual_goal_usd":    annual_goal_usd,
            "trailing_monthly_pct": rolling_return_pct,
            "months_to_goal":     projected_months,
        },
    }


@app.get("/api/agents/decisions")
async def get_agent_decisions(
    agent:        Optional[str] = Query(None, description="TECH or CNN"),
    product_id:   Optional[str] = Query(None),
    limit:        int           = Query(100, le=2000),
    signals_only: bool          = Query(False, description="If true, exclude HOLD decisions"),
):
    return await database.get_agent_decisions(
        product_id=product_id, agent=agent, limit=limit, signals_only=signals_only
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    app_state.ws_connections.append(ws)
    try:
        await broadcast_state()
    except Exception as _e:
        logger.debug(f"WS initial broadcast failed: {_e}")
    try:
        while True:
            await asyncio.sleep(5)
            try:
                await broadcast_state()
            except Exception as _be:
                # DB contention / transient error — log and keep the socket alive.
                # Do NOT let this propagate: an unhandled exception here closes the
                # WS connection, causing the frontend to show "Offline".
                logger.debug(f"WS broadcast error (socket kept alive): {_be}")
    except WebSocketDisconnect:
        pass
    finally:
        if ws in app_state.ws_connections:
            app_state.ws_connections.remove(ws)


if __name__ == "__main__":
    import uvicorn
    _free_port(8001)
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False,
                log_level=config.log_level.lower(),
                ws_ping_interval=20, ws_ping_timeout=20)
