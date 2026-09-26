"""Fetch + cache the daily series the regime layer needs.

FRED (no key): CBBTCUSD, SP500, DTWEXBGS, DFII10.
CoinMetrics community (needs browser UA): CapMVRVCur.
All fetches are injectable (session_get) so tests never hit the network, and
degrade to the local parquet cache on failure. See memory
btc_macro_drivers_findings for source quirks.
"""

from __future__ import annotations

import io
import json
import logging
import os
import urllib.request
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
_CM = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    "?assets=btc&metrics=CapMVRVCur&frequency=1d&page_size=10000&start_time={}"
)
_FRED_IDS = {"btc": "CBBTCUSD", "spx": "SP500", "dxy": "DTWEXBGS", "real_yield": "DFII10"}
_CACHE_NAME = "regime_sources.parquet"


def _urlopen(url: str, headers=None) -> bytes:
    req = urllib.request.Request(url, headers=headers or {})
    return urllib.request.urlopen(req, timeout=40).read()


def fetch_fred(series_id: str, start: str, session_get: Callable = _urlopen) -> pd.Series:
    try:
        raw = session_get(_FRED.format(series_id)).decode()
        df = pd.read_csv(io.StringIO(raw))
        df.columns = ["date", "val"]
        df["date"] = pd.to_datetime(df["date"])
        df["val"] = pd.to_numeric(df["val"], errors="coerce")
        df = df.dropna()
        s = df.set_index("date")["val"].sort_index()
        return s[s.index >= pd.Timestamp(start)]
    except Exception as e:
        logger.warning("fetch_fred(%s) failed: %s", series_id, e)
        return pd.Series(dtype=float)


def fetch_mvrv(start: str, session_get: Callable = _urlopen) -> pd.Series:
    try:
        raw = session_get(
            _CM.format(start), headers={"User-Agent": _UA, "Accept": "application/json"}
        )
        data = json.loads(raw.decode()).get("data", [])
        idx = pd.to_datetime([d["time"] for d in data]).tz_localize(None).normalize()
        vals = pd.to_numeric([d.get("CapMVRVCur") for d in data], errors="coerce")
        return pd.Series(vals, index=idx).dropna().sort_index()
    except Exception as e:
        logger.warning("fetch_mvrv failed: %s", e)
        return pd.Series(dtype=float)


def load_aligned(start: str, cache_dir: str, session_get: Callable = _urlopen) -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, _CACHE_NAME)
    cols = ["btc", "spx", "dxy", "real_yield", "mvrv"]
    series = {}
    for name, sid in _FRED_IDS.items():
        s = fetch_fred(sid, start, session_get)
        s.index = pd.to_datetime(s.index).normalize()
        series[name] = s
    series["mvrv"] = fetch_mvrv(start, session_get)

    if all(s.empty for s in series.values()):
        if os.path.exists(cache):
            return pd.read_parquet(cache)[cols]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(series).sort_index().asfreq("D").ffill(limit=4)
    df = df[df["btc"].notna()][cols]
    try:
        df.to_parquet(cache)
    except Exception as e:
        logger.warning("regime cache write failed: %s", e)
    return df
