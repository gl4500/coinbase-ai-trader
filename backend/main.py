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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import database
from config import config
from clients import coinbase_client

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
from agents.market_scanner import MarketScanner
from agents.signal_generator import SignalGenerator
from agents.order_executor import OrderExecutor
from agents.cnn_agent import CoinbaseCNNAgent
from agents.tech_agent_cb import TechAgentCB
from agents.momentum_agent_cb import MomentumAgentCB
from agents.scalp_agent import ScalpAgent
from services.ws_subscriber import CoinbaseWSSubscriber
from services.portfolio_tracker import PortfolioTracker
from services.outcome_tracker import get_tracker

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_fmt     = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
_level   = getattr(logging, config.log_level.upper(), logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(_fmt)
_file_all = RotatingFileHandler(
    os.path.join(_LOG_DIR, "backend.log"), maxBytes=5_000_000, backupCount=10, encoding="utf-8"
)
_file_all.setFormatter(_fmt)
_file_err = RotatingFileHandler(
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
    momentum_agent:  MomentumAgentCB       = None
    scalp_agent:     ScalpAgent            = None
    ws_subscriber:   CoinbaseWSSubscriber  = None
    portfolio:       PortfolioTracker      = None
    scanner_task:    asyncio.Task          = None
    portfolio_task:  asyncio.Task          = None
    cnn_task:        asyncio.Task          = None
    tech_task:       asyncio.Task          = None
    momentum_task:   asyncio.Task          = None
    scalp_task:      asyncio.Task          = None
    outcome_task:    asyncio.Task          = None
    ws_connections:  List[WebSocket]       = []
    is_trading:      bool                  = False
    dry_run:         bool                  = True

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


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.init_db()

    # WS subscriber created first so agents can reference it for live prices
    app_state.ws_subscriber   = CoinbaseWSSubscriber(broadcast_fn=broadcast)

    app_state.scanner         = MarketScanner()
    app_state.signal_gen      = SignalGenerator()
    app_state.order_executor  = OrderExecutor(dry_run=app_state.dry_run)
    app_state.cnn_agent       = CoinbaseCNNAgent(ws_subscriber=app_state.ws_subscriber)
    app_state.tech_agent      = TechAgentCB(ws_subscriber=app_state.ws_subscriber)
    app_state.momentum_agent  = MomentumAgentCB(ws_subscriber=app_state.ws_subscriber)
    app_state.scalp_agent     = ScalpAgent(ws_subscriber=app_state.ws_subscriber)
    app_state.portfolio       = PortfolioTracker(ws_subscriber=app_state.ws_subscriber)

    # Initial market scan — discovers all USD spot pairs dynamically
    try:
        await app_state.scanner.scan()
    except Exception as e:
        logger.warning(f"Initial scan failed: {e}")

    # Subscribe WS to whatever the scanner discovered
    app_state.ws_subscriber.set_products(list(app_state.scanner.tracked_ids))
    await app_state.ws_subscriber.start()

    app_state.scanner_task   = asyncio.create_task(app_state.scanner.run_loop())
    app_state.portfolio_task = asyncio.create_task(app_state.portfolio.run_loop())
    # CNN loop uses its own _CNNBook to track positions and write to the trades table.
    # Real live orders additionally go through order_executor when is_trading=True.
    app_state.cnn_task = asyncio.create_task(
        app_state.cnn_agent.run_loop(
            order_executor      = app_state.order_executor,
            is_trading_fn       = lambda: True,   # always run book simulation
            train_every_n_scans = config.cnn_train_every_n_scans,
        )
    )
    # Sub-agents: stagger start by 30 s each so they don't all hit the DB at startup
    async def _delayed_tech():
        await asyncio.sleep(30)
        await app_state.tech_agent.run_loop()
    async def _delayed_momentum():
        await asyncio.sleep(60)
        await app_state.momentum_agent.run_loop()
    async def _delayed_scalp():
        await asyncio.sleep(90)
        await app_state.scalp_agent.start()
    app_state.tech_task     = asyncio.create_task(_delayed_tech())
    app_state.momentum_task = asyncio.create_task(_delayed_momentum())
    app_state.scalp_task    = asyncio.create_task(_delayed_scalp())
    app_state.outcome_task  = asyncio.create_task(get_tracker().run_loop())

    logger.info("Coinbase Trader ready — http://localhost:8001")
    logger.info(f"Credentials: {'OK' if config.has_credentials else 'MISSING'}")
    logger.info(f"Mode: {'DRY-RUN' if app_state.dry_run else 'LIVE'}")
    logger.info(f"Tracking {len(app_state.scanner.tracked_ids)} USD spot pairs")

    yield

    if app_state.scalp_agent:
        await app_state.scalp_agent.stop()
    for task in [
        app_state.scanner_task, app_state.portfolio_task,
        app_state.cnn_task, app_state.tech_task, app_state.momentum_task,
        app_state.scalp_task, app_state.outcome_task,
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
@app.get("/api/status")
async def get_status():
    balance = None
    if config.has_credentials:
        try:
            balance = await coinbase_client.get_usd_balance()
        except Exception:
            pass
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
    epochs: int = Query(20, ge=1, le=200),
    _: None = Depends(verify_api_key),
):
    return await app_state.cnn_agent.train_on_history(epochs=epochs)


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
        accounts = await coinbase_client.get_accounts()
        return {"accounts": accounts, "usd_balance": await coinbase_client.get_usd_balance()}
    except Exception as e:
        return {"error": str(e)}


# ── Logs ─────────────────────────────────────────────────────────────────────
def _read_log_file(path: str, min_level: int, limit: int) -> List[Dict]:
    """Read and parse the rotating log file, return newest-first entries."""
    entries: List[Dict] = []
    if not os.path.exists(path):
        return entries
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
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
                # Continuation line (e.g. traceback)
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
    mom_status  = app_state.momentum_agent.status if app_state.momentum_agent else {}

    # CNN book status
    cnn_book = app_state.cnn_agent.book if app_state.cnn_agent else None
    cnn_status: Dict = {}
    if cnn_book:
        cnn_status = {
            "agent":          "CNN",
            "balance":        round(cnn_book.balance, 2),
            "realized_pnl":   round(cnn_book.realized_pnl, 2),
            "open_positions": len(cnn_book.positions),
            "positions":      {
                pid: {"size": round(p["size"], 6), "avg_price": round(p["avg_price"], 6)}
                for pid, p in cnn_book.positions.items()
            },
        }

    # SCALP book status
    scalp_status: Dict = {}
    if app_state.scalp_agent:
        scalp_status = app_state.scalp_agent.status

    # Enrich positions with live price + unrealized PnL
    ws = app_state.ws_subscriber
    for st in (tech_status, mom_status, cnn_status, scalp_status):
        positions = st.get("positions", {})
        for pid, pos in positions.items():
            live = ws.get_price(pid) if ws else None
            if live:
                pos["current_price"]  = round(live, 6)
                pos["unrealized_pnl"] = round((live - pos["avg_price"]) * pos["size"], 4)
                pos["pct_pnl"]        = round((live - pos["avg_price"]) / pos["avg_price"] * 100, 2)
            else:
                pos["current_price"]  = None
                pos["unrealized_pnl"] = None
                pos["pct_pnl"]        = None

    return {"tech": tech_status, "momentum": mom_status, "cnn": cnn_status, "scalp": scalp_status}


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


@app.get("/api/agents/decisions")
async def get_agent_decisions(
    agent:        Optional[str] = Query(None, description="TECH or MOMENTUM"),
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
    await broadcast_state()
    try:
        while True:
            await asyncio.sleep(5)
            await broadcast_state()
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
