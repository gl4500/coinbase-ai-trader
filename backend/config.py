"""
Central configuration loaded from .env.
All modules import the `config` singleton — never read os.environ directly.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


@dataclass
class Config:
    # ── Coinbase Advanced Trade API (CDP keys — jwt auth) ──────────────────────
    # Key Name:    organizations/{org_id}/apiKeys/{key_id}  (from Developer Platform)
    # Private Key: PEM-encoded EC key — use \\n for newlines in .env
    coinbase_api_key:    str  = field(default_factory=lambda: os.getenv("COINBASE_API_KEY_NAME",    ""))
    coinbase_api_secret: str  = field(default_factory=lambda: os.getenv("COINBASE_API_PRIVATE_KEY", ""))

    # ── App ────────────────────────────────────────────────────────────────────
    database_url:        str  = field(default_factory=lambda: os.getenv("DATABASE_URL",        "trading.db"))
    log_level:           str  = field(default_factory=lambda: os.getenv("LOG_LEVEL",           "INFO"))
    app_api_key:         str  = field(default_factory=lambda: os.getenv("APP_API_KEY",         ""))
    dry_run:             bool = field(default_factory=lambda: os.getenv("DRY_RUN", "true").lower() != "false")

    # ── Dynamic product discovery ──────────────────────────────────────────────
    max_tracked_products: int = field(default_factory=lambda: int(os.getenv("MAX_TRACKED_PRODUCTS", "100")))

    # ── Scan performance ───────────────────────────────────────────────────────
    # Skip Ollama LLM call when CNN prob is already this decisive (> threshold or < 1-threshold)
    # e.g. 0.75 means skip LLM if cnn_prob > 0.75 or < 0.25 — saves 10–30s per product
    llm_skip_threshold:   float = field(default_factory=lambda: float(os.getenv("LLM_SKIP_THRESHOLD",   "0.75")))
    # Seconds to sleep between products during a scan (reduce to speed up, raise to avoid rate limits)
    scan_sleep_secs:      float = field(default_factory=lambda: float(os.getenv("SCAN_SLEEP_SECS",      "0.1")))

    # ── Risk / sizing ──────────────────────────────────────────────────────────
    kelly_fraction:      float = field(default_factory=lambda: float(os.getenv("KELLY_FRACTION",      "0.25")))
    max_position_usd:    float = field(default_factory=lambda: float(os.getenv("MAX_POSITION_USD",    "500")))
    max_total_exposure:  float = field(default_factory=lambda: float(os.getenv("MAX_TOTAL_EXPOSURE",  "5000")))
    min_volume_24h:      float = field(default_factory=lambda: float(os.getenv("MIN_VOLUME_24H",      "1000000")))
    # ATR-based position sizing
    atr_risk_pct:        float = field(default_factory=lambda: float(os.getenv("ATR_RISK_PCT",        "0.01")))   # 1% account per trade
    atr_multiplier:      float = field(default_factory=lambda: float(os.getenv("ATR_MULTIPLIER",      "2.0")))    # stop = 2× ATR

    # ── Drawdown circuit breaker ───────────────────────────────────────────────
    daily_drawdown_limit:  float = field(default_factory=lambda: float(os.getenv("DAILY_DRAWDOWN_LIMIT",  "0.05")))  # 5%
    weekly_drawdown_limit: float = field(default_factory=lambda: float(os.getenv("WEEKLY_DRAWDOWN_LIMIT", "0.10")))  # 10%

    # ── TA signal thresholds ───────────────────────────────────────────────────
    rsi_oversold:        float = field(default_factory=lambda: float(os.getenv("RSI_OVERSOLD",        "30")))
    rsi_overbought:      float = field(default_factory=lambda: float(os.getenv("RSI_OVERBOUGHT",      "70")))
    min_signal_strength: float = field(default_factory=lambda: float(os.getenv("MIN_SIGNAL_STRENGTH", "0.20")))
    adx_trend_threshold: float = field(default_factory=lambda: float(os.getenv("ADX_TREND_THRESHOLD", "25.0")))

    # ── CNN signal gates ───────────────────────────────────────────────────────
    # model_prob must exceed cnn_buy_threshold to fire a BUY (symmetric: < 1 - threshold for SELL)
    cnn_buy_threshold:      float = field(default_factory=lambda: float(os.getenv("CNN_BUY_THRESHOLD",      "0.60")))
    cnn_sell_threshold:     float = field(default_factory=lambda: float(os.getenv("CNN_SELL_THRESHOLD",     "0.40")))
    # CNN/LLM blend weights per regime (must sum to 1.0 each pair)
    cnn_trending_cnn_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_CNN_W",     "0.75")))
    cnn_trending_llm_w:     float = field(default_factory=lambda: float(os.getenv("CNN_TRENDING_LLM_W",     "0.25")))
    cnn_ranging_cnn_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_CNN_W",      "0.40")))
    cnn_ranging_llm_w:      float = field(default_factory=lambda: float(os.getenv("CNN_RANGING_LLM_W",      "0.60")))
    # How often to auto-train (in number of scans; default 4 = ~1 hour at 15-min scan interval)
    cnn_train_every_n_scans: int  = field(default_factory=lambda: int(os.getenv("CNN_TRAIN_EVERY_N_SCANS",  "8")))

    # ── Ollama LLM ─────────────────────────────────────────────────────────────
    # Central default so every module reads the same model (CLAUDE.md invariant 7).
    ollama_model:         str  = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    # ── History backfill schedule ──────────────────────────────────────────────
    # How many hours between automatic incremental backfill runs (0 = disabled)
    backfill_interval_hours: int  = field(default_factory=lambda: int(os.getenv("BACKFILL_INTERVAL_HOURS", "24")))
    # How many days of history to fetch for a brand-new product
    backfill_new_product_days: int = field(default_factory=lambda: int(os.getenv("BACKFILL_NEW_PRODUCT_DAYS", "365")))

    # ── Coinbase API hosts (read-only) ─────────────────────────────────────────
    coinbase_rest_url: str = "https://api.coinbase.com/api/v3/brokerage"
    coinbase_ws_url:   str = "wss://advanced-trade-ws.coinbase.com"   # public ticker channel

    @property
    def has_credentials(self) -> bool:
        return bool(self.coinbase_api_key and self.coinbase_api_secret)


config = Config()
