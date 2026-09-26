import os
import sys

_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME", "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("OLLAMA_MODEL", "llama3.1:8b")

import json

from services.regime.state import (
    EXPOSURE_CLAMP,
    MACRO_K,
    MVRV_ANCHORS,
    REGIME_STALE_DAYS,
    RegimeState,
)


def test_constants():
    assert MVRV_ANCHORS == [(0.8, 1.25), (1.5, 1.05), (3.0, 0.95), (3.5, 0.85)]
    assert MACRO_K == 0.3
    assert EXPOSURE_CLAMP == (0.4, 1.25)
    assert REGIME_STALE_DAYS == 3


def test_roundtrip_row():
    rs = RegimeState(
        date="2025-11-30",
        mvrv=1.0,
        mvrv_prior=1.1,
        corr_spx_90d=0.02,
        macro_risk_raw=0.3,
        macro_mult=1.0,
        exposure_scalar=1.1,
        confidence=1.0,
        components={"equity_trend": 0.5},
    )
    row = rs.to_row()
    assert row["date"] == "2025-11-30"
    assert row["exposure_scalar"] == 1.1
    assert json.loads(row["components"])["equity_trend"] == 0.5
    back = RegimeState.from_row(row)
    assert back.exposure_scalar == 1.1
    assert back.components["equity_trend"] == 0.5
    assert back.mvrv == 1.0
