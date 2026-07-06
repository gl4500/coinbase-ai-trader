import os, sys, math
_BACKEND = os.path.join(os.path.dirname(__file__), "..", "..")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)
os.environ.setdefault("COINBASE_API_KEY_NAME",    "organizations/test/apiKeys/test")
os.environ.setdefault("COINBASE_API_PRIVATE_KEY", "stub")
os.environ.setdefault("DRY_RUN",                  "true")
os.environ.setdefault("LOG_LEVEL",                "WARNING")
os.environ.setdefault("OLLAMA_MODEL",             "llama3.1:8b")

import pytest
from services.regime.features import corr_spx_90d, macro_risk_raw


def test_corr_perfectly_coupled():
    # identical return streams -> corr ~ +1
    base = [100.0]
    for i in range(120):
        base.append(base[-1] * (1.0 + 0.01 * math.sin(i)))
    assert corr_spx_90d(base, base) == pytest.approx(1.0, abs=1e-6)


def test_corr_insufficient_data_none():
    assert corr_spx_90d([100, 101, 102], [100, 101, 102]) is None


def test_macro_risk_riskon_when_equities_up_dollar_down():
    # SPX rising above its 50d MA, DXY falling, real yield falling -> risk-on (+)
    spx = [100 + i for i in range(60)]
    dxy = [100 - 0.05 * i for i in range(60)]
    ry = [2.0 - 0.01 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert v is not None and v > 0.5


def test_macro_risk_riskoff_when_equities_down_dollar_up():
    spx = [160 - i for i in range(60)]
    dxy = [100 + 0.05 * i for i in range(60)]
    ry = [1.0 + 0.01 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert v is not None and v < -0.5


def test_macro_risk_clipped():
    spx = [100 + 5 * i for i in range(60)]
    dxy = [100 - 1.0 * i for i in range(60)]
    ry = [3.0 - 0.1 * i for i in range(60)]
    v = macro_risk_raw(spx, dxy, ry)
    assert -1.0 <= v <= 1.0
