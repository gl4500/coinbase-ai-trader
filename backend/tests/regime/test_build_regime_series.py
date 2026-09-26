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

import numpy as np
import pandas as pd
import pytest

from tools.regime.build_regime_series import build_series


def _frame(n=200):
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    btc = 20000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    spx = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "btc": btc,
            "spx": spx,
            "dxy": np.linspace(103, 100, n),
            "real_yield": np.linspace(2.0, 1.5, n),
            "mvrv": np.linspace(0.9, 2.2, n),
        },
        index=idx,
    )


def test_build_series_produces_states_after_warmup():
    states = build_series(_frame(200))
    assert len(states) == 200 - 90  # first 90 days are warmup
    s = states[-1]
    assert 0.4 <= s.exposure_scalar <= 1.25
    assert s.date == "2023-07-19"  # 2023-01-01 + 199 days
    assert s.mvrv is not None


def test_build_series_empty_frame():
    assert build_series(pd.DataFrame(columns=["btc", "spx", "dxy", "real_yield", "mvrv"])) == []


@pytest.mark.asyncio
async def test_persist_creates_schema_on_a_fresh_db(tmp_path, monkeypatch):
    """The builder is an offline tool: it must work against a DB that has never
    been initialised by the backend, not only against the live one."""
    import database
    from tools.regime.build_regime_series import build_series, persist

    monkeypatch.setattr(database, "DB_PATH", str(tmp_path / "fresh.db"))
    states = build_series(_frame(120))
    written = await persist(states)

    assert written == 120 - 90
    latest = await database.get_latest_regime_state()
    assert latest["date"] == states[-1].date
