import json
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

import pandas as pd

from services.regime import sources


def _fake_get(fred_csv: bytes, cm_json: bytes):
    def _get(url, headers=None):
        if "stlouisfed" in url:
            return fred_csv
        return cm_json

    return _get


def test_fetch_fred_parses_csv():
    csv = b"observation_date,DFII10\n2024-01-02,2.10\n2024-01-03,.\n2024-01-04,2.20\n"
    s = sources.fetch_fred("DFII10", "2024-01-01", session_get=lambda u, headers=None: csv)
    assert len(s) == 2  # "." dropped
    assert s.iloc[-1] == 2.20


def test_fetch_fred_network_error_returns_empty():
    def boom(u, headers=None):
        raise OSError("net down")

    s = sources.fetch_fred("DFII10", "2024-01-01", session_get=boom)
    assert s.empty


def test_fetch_mvrv_parses_json():
    body = json.dumps(
        {
            "data": [
                {"time": "2024-01-02T00:00:00.000000000Z", "CapMVRVCur": "1.5"},
                {"time": "2024-01-03T00:00:00.000000000Z", "CapMVRVCur": "1.6"},
            ]
        }
    ).encode()
    s = sources.fetch_mvrv("2024-01-01", session_get=lambda u, headers=None: body)
    assert s.iloc[-1] == 1.6


def test_load_aligned_uses_cache_on_network_failure(tmp_path):
    # Seed a cache, then force network failure -> load returns cached frame.
    cache = tmp_path / "regime_sources.parquet"
    df = pd.DataFrame(
        {
            "btc": [1.0, 2.0],
            "spx": [1.0, 1.0],
            "dxy": [1.0, 1.0],
            "real_yield": [2.0, 2.0],
            "mvrv": [1.0, 1.1],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )
    df.to_parquet(cache)

    def boom(u, headers=None):
        raise OSError("net down")

    out = sources.load_aligned("2024-01-01", str(tmp_path), session_get=boom)
    assert list(out.columns) == ["btc", "spx", "dxy", "real_yield", "mvrv"]
    assert len(out) == 2
