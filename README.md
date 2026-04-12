# Coinbase AI Trader

AI-powered crypto trading system for [Coinbase Advanced Trade](https://www.coinbase.com/advanced-trade).

## Stack
- **Backend**: FastAPI + Python, SQLite via aiosqlite, JWT/ES256 CDP auth
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **AI**: 1D CNN (PyTorch) + Ollama (qwen2.5:7b) for signal generation
- **Indicators**: RSI(14), EMA cross(9/21), MACD(12,26,9), Bollinger Bands(20,2)
- **Data**: Coinbase Advanced Trade REST + WebSocket streams

## Quick Start

### 1. Configure credentials
```
cp .env .env.local
# Edit .env — add your COINBASE_API_KEY_NAME and COINBASE_API_PRIVATE_KEY
```

### 2. Start backend (new terminal)
```powershell
.\start_backend.ps1
# Opens http://localhost:8001
# API docs: http://localhost:8001/docs
```

### 3. Start frontend (new terminal)
```powershell
.\start_frontend.ps1
# Opens http://localhost:5174
```

Or use the GUI launcher:
```
python launcher.py
```

## Key .env Settings

| Variable | Required | Description |
|---|---|---|
| `COINBASE_API_KEY_NAME` | Yes | CDP Key Name (`organizations/.../apiKeys/...`) |
| `COINBASE_API_PRIVATE_KEY` | Yes | EC private key PEM from coinbase.com Developer Platform |
| `APP_API_KEY` | Yes | Internal API key for write endpoints |
| `TRADING_PAIRS` | No | Comma-separated pairs (default: BTC-USD,ETH-USD,SOL-USD,DOGE-USD,LINK-USD) |
| `MIN_SIGNAL_STRENGTH` | No | Minimum signal strength 0–1 (default 0.6) |
| `MAX_POSITION_USD` | No | Max single position in USD (default $500) |
| `DRY_RUN` | No | Set `false` only when ready for live trading |

## Trading Modes

| Mode | Setting | Description |
|---|---|---|
| **Read-only** | Default | Browse markets, view signals — no orders |
| **Dry-run** | `DRY_RUN=true` | Simulate signals without real money |
| **Live** | `DRY_RUN=false` | Real orders on Coinbase Advanced Trade |

Always start in dry-run mode to validate signal logic.

## Architecture

```
Coinbase REST API ──► MarketScanner ──► SQLite DB
                           │                │
  Coinbase WebSocket ──► WSSubscriber   SignalGenerator ──► OrderExecutor
                           │                │
                      PortfolioTracker   CoinbaseCNNAgent (CNN + Ollama)
                           │
             FastAPI (/api/*) ──► React Frontend (WebSocket)
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | App status, balance |
| `/api/products` | GET | Product list |
| `/api/candles/{product_id}` | GET | OHLCV candle data |
| `/api/orderbook/{product_id}` | GET | Live order book |
| `/api/positions` | GET | Open positions |
| `/api/signals` | GET | AI-generated trade signals |
| `/api/orders` | GET/POST | Order list / place order |
| `/api/orders/{id}` | DELETE | Cancel order |
| `/api/trading/enable` | POST | Enable trading |
| `/api/trading/disable` | POST | Pause trading |
| `/api/scanner/run` | POST | Trigger market scan |
| `/ws` | WS | Real-time state broadcast |
