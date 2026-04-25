# Coinbase AI Trader — Rebuild Standard
> Use this document to fully recreate the app if the codebase is lost.

---

## 1. Project Layout

```
polymarket_app/
├── backend/
│   ├── main.py                  # FastAPI app, lifespan, WS, endpoints
│   ├── config.py                # All config loaded from .env
│   ├── database.py              # Async SQLite (aiosqlite) — all DB functions
│   ├── coinbase.db              # SQLite database (auto-created on first run)
│   ├── cnn_model.pt             # Trained CNN weights (auto-saved after training)
│   ├── requirements.txt         # pip install -r requirements.txt
│   ├── agents/
│   │   ├── cnn_agent.py         # CNN-LSTM + Ollama blend, 27-channel tensor
│   │   ├── tech_agent_cb.py     # TechAgent — RSI/MACD/BB/Stoch/ADX/MFI
│   │   ├── signal_generator.py  # Shared indicator library (_rsi, _macd, _adx, _mfi, _vwap…)
│   │   ├── order_executor.py    # Dry-run + live order execution
│   │   └── market_scanner.py    # Discovers USD spot pairs from Coinbase
│   ├── services/
│   │   ├── outcome_tracker.py   # 4h WIN/LOSS outcome loop + Ollama validation
│   │   ├── ws_subscriber.py     # Coinbase WebSocket live tick prices
│   │   ├── portfolio_tracker.py # Real portfolio PnL tracker
│   │   └── history_backfill.py  # DELETED — CoinGecko was unreliable
│   ├── clients/
│   │   ├── coinbase_client.py   # Coinbase Advanced Trade REST client
│   │   └── coingecko_client.py  # DELETED
│   ├── logs/                    # backend.log, errors.log, launcher.log (auto-created)
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── App.tsx              # Root: tabs, WS connect, portfolio state
│   │   └── components/
│   │       ├── CNNDashboard.tsx     # CNN tab: stats, sub-agents, confidence table
│   │       ├── MarketBrowser.tsx    # Markets tab
│   │       ├── OrderBook.tsx        # Order Book tab
│   │       ├── PositionTracker.tsx  # Positions tab
│   │       ├── SignalDashboard.tsx  # Signals tab
│   │       └── LogViewer.tsx        # Logs tab
│   ├── package.json
│   └── vite.config.ts
├── launcher.py                  # tkinter GUI launcher (compile → .exe)
├── start_app.py                 # CLI launcher (no GUI, hidden processes)
├── build_exe.ps1                # Builds dist\Coinbase AI Trader.exe
├── launcher.ico                 # App icon
├── .venv/                       # Python virtual environment
└── REBUILD_STANDARD.md          # This file
```

---

## 2. Stack

| Layer | Tech |
|---|---|
| Backend | Python 3.11, FastAPI, uvicorn, aiosqlite |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS |
| ML | PyTorch (CNN-LSTM), falls back to linear scorer if no GPU |
| LLM | Ollama local (`llama3.1:8b`) at `http://localhost:11434` |
| Exchange | Coinbase Advanced Trade API (REST + WebSocket) |
| DB | SQLite (`coinbase.db`) |
| Launcher | PyInstaller + tkinter (windowed, no console) |

---

## 3. Environment (.env in `backend/`)

```
COINBASE_API_KEY_NAME=organizations/xxx/apiKeys/xxx
COINBASE_API_PRIVATE_KEY=-----BEGIN EC PRIVATE KEY-----\n...\n-----END EC PRIVATE KEY-----
APP_API_KEY=your_random_secret          # UI auth header
DRY_RUN=true                            # true = no real orders
OLLAMA_MODEL=llama3.1:8b
MAX_POSITION_USD=500
LOG_LEVEL=INFO
CNN_BUY_THRESHOLD=0.60
CNN_SELL_THRESHOLD=0.40
```

---

## 4. Database Schema

### `products`
```sql
CREATE TABLE IF NOT EXISTS products (
    product_id TEXT PRIMARY KEY, base_currency TEXT, quote_currency TEXT,
    display_name TEXT, price REAL, price_pct_change_24h REAL DEFAULT 0,
    volume_24h REAL DEFAULT 0, high_24h REAL DEFAULT 0, low_24h REAL DEFAULT 0,
    spread REAL DEFAULT 0, is_tracked INTEGER DEFAULT 0, last_updated TEXT
);
```

### `candles`
```sql
CREATE TABLE IF NOT EXISTS candles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT NOT NULL,
    start_time INTEGER NOT NULL, open REAL, high REAL, low REAL,
    close REAL, volume REAL,
    UNIQUE(product_id, start_time)
);
```

### `cnn_scans`
```sql
CREATE TABLE IF NOT EXISTS cnn_scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT, product_id TEXT, price REAL,
    cnn_prob REAL, llm_prob REAL, model_prob REAL, cnn_weight REAL, llm_weight REAL,
    side TEXT, strength REAL, signal_gen INTEGER DEFAULT 0, regime TEXT,
    adx REAL, rsi REAL, macd REAL, mfi REAL, stoch_k REAL, atr REAL,
    vwap_dist REAL, fast_rsi REAL, velocity REAL, vol_z REAL, scanned_at TEXT
);
```

### `agent_decisions`
```sql
CREATE TABLE IF NOT EXISTS agent_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, product_id TEXT,
    side TEXT, confidence REAL, price REAL, score REAL, reasoning TEXT,
    balance REAL, pnl REAL, created_at TEXT
);
```

### `agent_state`
```sql
CREATE TABLE IF NOT EXISTS agent_state (
    agent TEXT PRIMARY KEY, balance REAL, realized_pnl REAL DEFAULT 0,
    positions_json TEXT DEFAULT '{}', high_water_json TEXT DEFAULT '{}', updated_at TEXT
);
```

### `signal_outcomes`
```sql
CREATE TABLE IF NOT EXISTS signal_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, product_id TEXT,
    side TEXT, confidence REAL, entry_price REAL, exit_price REAL,
    pct_change REAL, outcome TEXT, indicators_json TEXT, lesson_text TEXT,
    check_after REAL, checked_at TEXT, created_at TEXT
);
```

---

## 5. Agent Architecture

### CoinbaseCNNAgent (`agents/cnn_agent.py`)
- **Scan interval**: 15 min auto-loop, or manual `/api/cnn/scan`
- **CNN**: 27-channel × 60-timestep input → 4× GatedConv1d (GLU, BatchNorm) → 2-layer LSTM → FC → sigmoid
- **Channels 0–16**: hourly candles (close, volume, range, body, RSI, MACD(5,13,3), EMA9, EMA21, Bollinger, 1-bar change, bid depth, ask depth, MFI, OBV slope, StochRSI, ADX, VWAP)
- **Channels 17–19**: 5-min candles (fast RSI(12), price velocity, volume z-score)
- **Channel 20**: Binance perpetual funding rate (crypto market sentiment)
- **Channel 21**: Rolling 20-bar BTC return correlation
- **Channels 22–23**: Time-of-day sin/cos encoding
- **Channels 24–25**: IV/RV20 and IV/RV60 spread (Deribit, BTC/ETH only)
- **Channel 26**: Binance top-trader long/short sentiment ratio
- **Blend**: `model_prob = cnn_w * cnn_prob + llm_w * llm_prob`
  - TRENDING (ADX≥25): CNN 75% / LLM 25%
  - RANGING: CNN 40% / LLM 60%
- **Risk exits**: hard stop-loss −8%, ATR trailing stop (2×ATR/peak, 3–15%), 7-day max-hold
- **Win/loss tracking**: `wins`, `losses`, `win_rate`, `expectancy` on `_CNNBook`
- **Ollama prompt includes**: regime, RSI, MACD, Bollinger, MFI, StochRSI, VWAP, CNN prob, sub-agent votes, past WIN/LOSS lessons
- **Signal**: BUY if model_prob > 0.60, SELL if < 0.40
- **Training**: subprocess (`train_worker.py`) every N scans or manual trigger; never blocks scan loop
- **Balance**: $1,000 dry-run via `OrderExecutor(dry_run=True)`

### TechAgentCB (`agents/tech_agent_cb.py`)
- **Scan interval**: 2 min (pure math, no Ollama)
- **Staggered start**: +30s after backend start
- **Indicators**: RSI(14), Bollinger Bands, MACD crossover, Volume spike, Stochastic %K/%D (price-based), OBV divergence, ADX(14), MFI(14)
- **Scoring** (BUY example): RSI<30 → +0.35, Price≤lower BB → +0.30, MACD bull cross → +0.25, Vol spike → +0.10, Stoch<20+rising → +0.15, OBV confirm → +0.10
- **Threshold**: score ≥ 0.55 for BUY or SELL
- **Portfolio**: `_Book("TECH")` — max 15% of portfolio per product, persisted to `agent_state`
- **After signal**: calls `OutcomeTracker.record()` + `OutcomeTracker.validate_with_ollama()`
- **Live prices**: reads from `ws_subscriber.get_price(pid)` first, falls back to DB

### OutcomeTracker (`services/outcome_tracker.py`)
- **Loop interval**: 30 min (resolves 4h-old outcomes)
- **WIN**: price moved +0.5% in signal direction after 4h
- **LOSS**: price moved -0.5% adverse after 4h
- **NEUTRAL**: in between
- **Lesson format**: `"TECH BUY conf=0.72 RSI=28 BB=0.05 → +0.8% after 4h [WIN]"`
- **Ollama validation**: fires immediately after each BUY/SELL signal; injects past lessons + fresh indicators
- **Singleton**: `get_tracker()` → `_tracker`

---

## 6. Signal Flow

```
Coinbase WS tick → ws_subscriber.state[pid]["price"]
         ↓
TechAgent (2 min scan)
  → score indicators (RSI, BB, MACD, Stoch, ADX, MFI, OBV)
  → BUY/SELL/HOLD → save to agent_decisions
  → if BUY/SELL: OutcomeTracker.record() + validate_with_ollama()
         ↓
MacroSignalService (1 hr cache)
  → fetch funding rate, L/S ratio, OI, BTC dominance, CB premium
  → buy_gate_multiplier / sell_gate_multiplier applied to all agent scores
         ↓
CNN Agent (15 min scan)
  → build 27-ch tensor from hourly + 5-min candles + macro channels
  → CNN prob → fetch sub-agent votes + outcome lessons
  → Ollama blend → model_prob → BUY/SELL/HOLD
  → _check_risk_exits: stop-loss / ATR trail / max-hold on every scan
  → save to cnn_scans + signals
  → if signal: OutcomeTracker.record()
         ↓
OutcomeTracker loop (30 min)
  → resolve pending 4h outcomes → WIN/LOSS/NEUTRAL → lesson_text
```

---

## 7. WebSocket Message Types

| type | Payload | When |
|---|---|---|
| `state` | positions, signals, orders, products, portfolio | Every 5s |
| `price_update` | product_id, price, pct_change | On WS tick |
| `trading_status` | is_trading, dry_run | On enable/disable |

---

## 8. Launcher Details

### `launcher.py` (tkinter GUI)
- Dark-themed control panel (BG=#0f0f1a, ACCENT=#6366f1)
- Auto-starts services 1s after opening via `self.after(1000, self._auto_start)`
- `_auto_start()` skips if services already running
- `_start_all()` guards against double-start by checking `proc.poll() is None`
- ALL subprocess calls use `_no_window()` → `CREATE_NO_WINDOW + STARTUPINFO(SW_HIDE)`
- Browser opened only by launcher (not by main.py) — prevents multi-browser bug
- Streams backend + frontend stdout to scrolled log panel in real time

### Build EXE
```powershell
.venv\Scripts\pyinstaller.exe launcher.py --onefile --windowed --name="Coinbase AI Trader" --clean --noconfirm --icon=launcher.ico
# Output: dist\Coinbase AI Trader.exe
```

### `_no_window()` helper (required on Windows for hidden child processes)
```python
def _no_window() -> dict:
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = subprocess.SW_HIDE
    return {"creationflags": subprocess.CREATE_NO_WINDOW, "startupinfo": si}
```

---

## 9. Frontend — CNN Confidence Table Columns

| Column | Source | Notes |
|---|---|---|
| Symbol | cnn_scans.product_id | ★ if signal generated |
| Blended% | model_prob | Green BUY / Red SELL |
| CNN% | cnn_prob | Raw model output |
| LLM% | llm_prob | Ollama output |
| Strength | strength bar | ≥60% fires signal |
| VWAP Δ | vwap_dist | +/- % from VWAP |
| Vel 5m | velocity | 5-min normalised velocity |
| ADX | adx | ≥25 = amber (trending) |
| RSI | rsi | <30 green / >70 red |
| MFI | mfi | <20 green / >80 red |
| StochK | stoch_k | <20 green / >80 red |
| Time | scanned_at | Scan timestamp |
| **Tech** | agent_decisions (TECH) | Side badge + conf% + score + PnL (purple) |
| Regime/Signal | regime + side | T=Trending / R=Ranging badge |

---

## 10. First-Time Setup

```bash
# 1. Create venv
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt

# 2. Install frontend deps
cd frontend && npm install

# 3. Create backend\.env with Coinbase API keys

# 4. Start
python start_app.py
# or double-click dist\Coinbase AI Trader.exe
```

---

## 12. TDD Pipeline — Regression Testing on Every Commit

### Overview
Every `git commit` automatically runs the full unit test suite via a pre-commit hook.
GitHub Actions runs the complete 8-stage DevSecOps pipeline on every push and PR.
No commit can land on `main` / `master` with failing tests.

---

### Git Pre-commit Hook (local, runs on every commit)
File: `.git/hooks/pre-commit` (bash, auto-invoked by git)

**What it does:**
1. Detects staged `.py` files — skips if no Python changed
2. Sets stub env vars (no real credentials needed)
3. Runs `pytest backend/tests -m "not slow and not integration"` (fast unit tests only)
4. Blocks commit if any test fails
5. To skip in emergencies only: `git commit --no-verify`

**Re-install after cloning:**
```bash
cp .git/hooks/pre-commit .git/hooks/pre-commit   # already present
chmod +x .git/hooks/pre-commit
```

---

### GitHub Actions CI (`/.github/workflows/ci.yml`)
Triggered on every push/PR to `main` or `master`.

| Stage | Tool | Blocks merge? |
|---|---|---|
| 1. Secret Detection | detect-secrets | Yes |
| 2. Python SAST | bandit | Yes |
| 3. Python Lint | ruff check | No |
| 4. Python Format | ruff format | No |
| 5. **Backend Tests** | pytest --cov ≥60% | **Yes** |
| 6. Python CVE Audit | pip-audit HIGH+ | Yes |
| 7. Frontend Lint | ESLint + tsc | No |
| 8. Node CVE Audit | npm audit HIGH+ | No |

Security gate (stage 8) fails the entire pipeline if stages 1, 2, 5, or 6 fail.

---

### Local CI Runner (Windows)
Mirrors GitHub Actions exactly. Run before pushing:
```powershell
.\run_checks.ps1
```
Prerequisites (one-time):
```powershell
pip install bandit[toml] ruff pytest pytest-asyncio pytest-cov httpx pip-audit detect-secrets
```

---

### Test Suite Structure
```
backend/tests/
├── conftest.py                    # Shared fixtures: tmp_db, init_db, stub env vars
├── test_bsm_integration.py        # BSM features: RV, entropy, Deribit IV, HMM, 27-ch builder
├── test_signal_improvements.py    # ADX norm, MACD fast, RSI thresholds, funding rate, BTC corr
├── test_cnn_agent.py              # CNN scan, FeatureBuilder, model load/save
├── test_lgbm_filter.py            # LGBMFilter train/predict/unseen-label fix
├── test_tech_agent_cb.py          # TechAgent scoring, ATR stop, Kelly sizing
├── test_signal_generator_new.py   # All indicator functions (_rsi, _adx, _macd, _vwap…)
├── test_database.py               # All DB CRUD functions
├── test_startup_sequence.py       # Lifespan order, agent boot, DB init
├── test_api_security.py           # API key auth, rate limits
├── test_fear_greed.py             # Fear & Greed fetch + cache
├── test_macro_signals.py          # Macro signal layer
└── ...                            # Others: atr_stops, position_sizer, market_scanner…
```

---

### TDD Workflow — Required for Every New Feature
```
1. Write test first  →  backend/tests/test_<feature>.py
2. Run tests         →  pytest backend/tests/test_<feature>.py  (should FAIL)
3. Implement feature →  backend/agents/ or backend/services/
4. Run tests again   →  pytest backend/tests/test_<feature>.py  (should PASS)
5. Run full suite    →  pytest backend/tests  (no regressions)
6. Commit            →  pre-commit hook runs tests automatically
```

Never skip step 1. Tests written after code are not TDD — they are documentation.

---

### Adding a New Feature — Checklist
- [ ] Write tests in `backend/tests/test_<feature>.py` covering: happy path, edge cases, failure modes
- [ ] Add `@pytest.mark.slow` to tests that take >2s (DB integration, network, CNN train)
- [ ] Run `pytest -m "not slow"` locally — all must pass before commit
- [ ] Pre-commit hook will block if any test fails
- [ ] CI pipeline will block PR if coverage drops below 60%

---

## 11. Key Design Decisions

| Decision | Reason |
|---|---|
| WS subscriber created before agents in lifespan | Agents need live prices at init |
| Tech +30s staggered start | Avoid DB stampede at startup |
| CNN always dry-run (`is_trading_fn=lambda: True`) | Never executes real orders in auto-loop |
| CoinGecko removed entirely | Unreliable, added latency, not needed |
| Browser opened only by launcher | main.py opening Brave caused 3 browser windows |
| CREATE_NO_WINDOW + STARTUPINFO(SW_HIDE) | CREATE_NO_WINDOW alone doesn't suppress cmd.exe from .cmd batch files |
| Outcome tracker fires immediately after signal | Ollama sees lesson history without waiting 15 min for CNN cycle |
| ADX + MFI computed in TechAgent (2 min) | Faster than waiting for CNN scan every 15 min; Ollama reads fresh values |
