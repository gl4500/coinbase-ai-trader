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

import pytest

import database
from services.regime.state import RegimeState


@pytest.mark.asyncio
async def test_upsert_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    rs = RegimeState(
        date="2025-11-30",
        mvrv=1.0,
        mvrv_prior=1.1,
        corr_spx_90d=0.0,
        macro_risk_raw=-1.0,
        macro_mult=1.0,
        exposure_scalar=1.1,
        confidence=1.0,
        components={"a": 1.0},
    )
    await database.upsert_regime_state(rs.to_row())
    # upsert again (idempotent on date)
    rs2 = RegimeState(**{**rs.__dict__, "exposure_scalar": 0.9})
    await database.upsert_regime_state(rs2.to_row())
    latest = await database.get_latest_regime_state()
    assert latest["date"] == "2025-11-30"
    assert latest["exposure_scalar"] == 0.9  # overwritten


@pytest.mark.asyncio
async def test_series_range(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "t.db"))
    await database.init_db()
    for d, sc in [("2025-01-01", 1.0), ("2025-06-01", 1.1), ("2025-12-01", 0.8)]:
        await database.upsert_regime_state(
            RegimeState(
                date=d,
                mvrv=1,
                mvrv_prior=1,
                corr_spx_90d=0,
                macro_risk_raw=0,
                macro_mult=1,
                exposure_scalar=sc,
                confidence=1,
                components={},
            ).to_row()
        )
    rows = await database.get_regime_series("2025-03-01", "2025-12-31")
    assert [r["date"] for r in rows] == ["2025-06-01", "2025-12-01"]
