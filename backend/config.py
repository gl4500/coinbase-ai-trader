"""
Central configuration loaded from .env.
All modules import the `config` singleton — never read os.environ directly.

Policy: every env var defined here MUST trace to a live consumer in backend/.
Dead entries are deleted on sight per refactor sweep policy (#311-refactor).
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))


_VALID_BACKENDS = {"xgb", "xgb_v45"}


def _validate_backend(value: str) -> str:
    if value == "cnn":
        raise ValueError(
            "MODEL_BACKEND=cnn is deprecated as of 2026-05-23. "
            "Use MODEL_BACKEND=xgb (default, v3 driver) or "
            "MODEL_BACKEND=xgb_v45 (v4.5 driver). See "
            "docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md"
        )
    if value not in _VALID_BACKENDS:
        raise ValueError(
            f"MODEL_BACKEND={value!r} invalid. Valid: {sorted(_VALID_BACKENDS)}"
        )
    return value


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
    # Auto-train cadence in scans. Active only under MODEL_BACKEND=cnn;
    # auto-train is gated off under MODEL_BACKEND=xgb per #300, so flipping
    # this knob has no effect while the xgb backend is live.
    # Default 4 = ~1 hour at the 15-min scan interval.
    cnn_train_every_n_scans: int  = field(default_factory=lambda: int(os.getenv("CNN_TRAIN_EVERY_N_SCANS",  "8")))
    # Scan-loop cadence in seconds (default 900 = 15 min). Lower to expedite
    # XGB shadow accumulation; raise to ease Coinbase REST pressure.
    scan_interval_secs:     int   = field(default_factory=lambda: int(os.getenv("SCAN_INTERVAL_SECS",       "900")))

    # ── Ollama LLM ─────────────────────────────────────────────────────────────
    # Central default so every module reads the same model (CLAUDE.md invariant 7).
    ollama_model:         str  = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.1:8b"))

    # ── Model backend selector ─────────────────────────────────────────────────
    # Valid values: "xgb" (v3 driver, default) | "xgb_v45" (v4.5 driver, dev).
    # Legacy "cnn" raises at startup — CNN driver was deprecated 2026-05-23.
    # See docs/superpowers/specs/2026-05-23-remove-cnn-driver-add-v45-driver-design.md.
    model_backend:       str  = field(default_factory=lambda: _validate_backend(os.getenv("MODEL_BACKEND", "xgb").lower()))

    # ── v4.5 indep_thresholds decision rule ────────────────────────────────────
    # BUY  when p_up   > xgb_v45_thresh_up   AND p_up   >= p_down.
    # SELL when p_down > xgb_v45_thresh_down AND p_down >  p_up.
    # Defaults 0.50/0.50 match tools/v4_5_horizon_compare.py:138.
    xgb_v45_thresh_up:   float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_UP",   "0.50")))
    xgb_v45_thresh_down: float = field(default_factory=lambda: float(os.getenv("XGB_V45_THRESH_DOWN", "0.50")))

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
