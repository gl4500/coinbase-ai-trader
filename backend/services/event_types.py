"""Typed payload dataclasses for the 7 event types in the spec.

Each dataclass:
  - Validates its own invariants in __post_init__
  - Round-trips via to_dict() / from_dict() (JSON-safe primitive dicts)
  - Mirrors the payload schema documented in 2026-05-25-event-sourced-architecture-design.md

We use frozen=True so payloads are immutable once constructed — events are
append-only and the dataclass instance carries that intent through the code.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Literal, Optional


EVENT_TYPES = (
    "price_tick",
    "candle_close",
    "marketcap_snapshot",
    "signal_scored",
    "trade_decided",
    "trade_closed",
    "exit_triggered",
)

_VALID_TIERS = {"1m", "5m", "15m", "1h", "4h", "1d"}
_VALID_SIDES = {"BUY", "SELL", "HOLD"}
_VALID_TRIGGERS = {"WS_TRAIL_STOP", "WS_STOP_LOSS", "STOP_LOSS",
                   "WS_MODEL_DOWN", "MAX_HOLD", "TRAIL_STOP", "RECONCILE"}


@dataclass(frozen=True)
class PriceTickPayload:
    pid: str
    price: float
    bid: Optional[float]
    ask: Optional[float]
    volume_24h: Optional[float]
    source: Literal["ws", "rest_fallback"]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "PriceTickPayload":
        return cls(**d)


@dataclass(frozen=True)
class CandleClosePayload:
    pid: str
    tier: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    bar_ts_ms: int

    def __post_init__(self):
        if self.tier not in _VALID_TIERS:
            raise ValueError(f"tier {self.tier!r} must be one of {_VALID_TIERS}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "CandleClosePayload":
        return cls(**d)


@dataclass(frozen=True)
class MarketcapSnapshotPayload:
    pid: str
    market_cap: Optional[float]
    fdv: Optional[float]
    circ_supply: Optional[float]
    total_supply: Optional[float]
    vol_24h: Optional[float]
    source: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "MarketcapSnapshotPayload":
        return cls(**d)


@dataclass(frozen=True)
class SignalScoredPayload:
    pid: str
    model: str
    model_version: str
    feature_hash: str
    scores: Dict[str, float]
    side: str
    strength: float
    regime: Optional[str]
    deployment_profile_id: Optional[str]
    input_event_ids: Dict[str, int]

    def __post_init__(self):
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side {self.side!r} must be one of {_VALID_SIDES}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SignalScoredPayload":
        return cls(**d)


@dataclass(frozen=True)
class TradeDecidedPayload:
    pid: str
    model: str
    side: str
    size: float
    size_usd: float
    intended_entry_price: float
    actual_entry_price: float
    fee_paid: float
    trigger: str
    signal_event_id: Optional[int]
    deployment_profile_id: Optional[str]
    trade_uid: str

    def __post_init__(self):
        if self.signal_event_id is None:
            raise ValueError("signal_event_id is required (no orphan trades)")
        if self.side not in _VALID_SIDES:
            raise ValueError(f"side {self.side!r} must be one of {_VALID_SIDES}")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeDecidedPayload":
        return cls(**d)


@dataclass(frozen=True)
class TradeClosedPayload:
    pid: str
    trade_uid: str
    exit_price: float
    exit_size: float
    pnl: float
    pct_pnl: float
    hold_secs: int
    trigger_close: str
    decision_event_id: int
    exit_signal_event_id: Optional[int]

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "TradeClosedPayload":
        return cls(**d)


@dataclass(frozen=True)
class ExitTriggeredPayload:
    pid: str
    trade_uid: str
    trigger_type: str
    peak_pnl_pct: float
    current_pnl_pct: float
    exit_threshold: float
    price_at_trigger: float
    trigger_price_event_id: Optional[int]

    def __post_init__(self):
        if self.trigger_price_event_id is None:
            raise ValueError("trigger_price_event_id is required for replay determinism")

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "ExitTriggeredPayload":
        return cls(**d)
