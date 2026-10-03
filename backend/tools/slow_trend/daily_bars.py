"""Coinbase daily candles. Raw rows are evidence: fetched with page numbers, audited BEFORE
normalisation, and never silently deduplicated or repaired."""

from __future__ import annotations

import math
from typing import Any, Awaitable, Callable, Dict, Optional

import numpy as np
import pandas as pd

DAY = 86400
FIELDS = ("start", "open", "high", "low", "close", "volume")
VALUES = list(FIELDS[1:])
Getter = Callable[[str, Dict[str, str]], Awaitable[Any]]


async def fetch_daily(
    pid: str, start_ts: int, end_ts: int, getter: Getter, page_days: int = 300
) -> pd.DataFrame:
    frames, end, page = [], end_ts, 0
    while end > start_ts:
        start = max(start_ts, end - page_days * DAY)
        data = await getter(
            f"/products/{pid}/candles",
            {"start": str(start), "end": str(end), "granularity": "ONE_DAY"},
        )
        if not isinstance(data, dict) or not isinstance(data.get("candles"), list):
            raise ValueError(f"{pid}: malformed candles response on page {page}")
        for c in data["candles"]:
            missing = [k for k in FIELDS if k not in c]
            if missing:
                raise ValueError(f"{pid}: candle missing {missing} on page {page}")
        df = pd.DataFrame(data["candles"], columns=list(FIELDS))
        df["page"] = page
        frames.append(df)
        end, page = start, page + 1
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.astype({"start": "int64", "page": "int64", **{k: float for k in VALUES}})
    return raw[(raw["start"] >= start_ts) & (raw["start"] < end_ts)].reset_index(drop=True)


def window(raw: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    lo = pd.Timestamp(first).timestamp()
    hi = pd.Timestamp(last).timestamp() + DAY
    return raw[(raw["start"] >= lo) & (raw["start"] < hi)]


def _valid_rows(r: pd.DataFrame) -> np.ndarray:
    v = r[VALUES].to_numpy(dtype=float)
    finite = np.isfinite(v).all(axis=1)
    o, h, lo, c, vol = (r[k].to_numpy(dtype=float) for k in VALUES)
    with np.errstate(invalid="ignore"):
        ok = (
            (o > 0)
            & (h > 0)
            & (lo > 0)
            & (c > 0)
            & (vol >= 0)
            & (h >= np.maximum(o, c))
            & (lo <= np.minimum(o, c))
        )
    return finite & ok


def audit(raw: pd.DataFrame, first: str, last: str, max_missing: int) -> Dict[str, Any]:
    r = window(raw, first, last)
    aligned = (r["start"] % DAY == 0).to_numpy()
    valid = _valid_rows(r)
    content = r[list(FIELDS)]
    identical = int(content.duplicated().sum())
    distinct = content.drop_duplicates()
    conflicting = int(distinct["start"].duplicated().sum())
    present = set(pd.to_datetime(r["start"][aligned & valid], unit="s").dt.strftime("%Y-%m-%d"))
    days = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
    missing = [d for d in days if d not in present]
    invalid = int((~valid).sum())
    misaligned = int((~aligned).sum())
    adequate = invalid == 0 and misaligned == 0 and conflicting == 0 and len(missing) <= max_missing
    return {
        "missing_days": missing,
        "misaligned": misaligned,
        "invalid_rows": invalid,
        "conflicting_duplicates": conflicting,
        "identical_copies": identical,
        "adequate": adequate,
    }


def first_day(raw: pd.DataFrame) -> Optional[str]:
    aligned = raw["start"][raw["start"] % DAY == 0]
    if aligned.empty:
        return None
    return str(pd.to_datetime(aligned.min(), unit="s").date())


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    distinct = raw[list(FIELDS)].drop_duplicates()
    if distinct["start"].duplicated().any():
        raise ValueError("conflicting duplicate candles; audit must reject this data")
    return distinct.sort_values("start").reset_index(drop=True)


def to_calendar(norm: pd.DataFrame, first: str, last: str) -> pd.DataFrame:
    idx = pd.to_datetime(norm["start"], unit="s")
    out = pd.DataFrame(
        {"open": norm["open"].to_numpy(), "close": norm["close"].to_numpy()}, index=idx
    )
    return out.reindex(pd.date_range(first, last, freq="D"))


def product_constraints(product: Optional[dict]) -> Dict[str, float]:
    if not product:
        raise ValueError("product metadata missing")
    out = {}
    for src, dst in (
        ("base_increment", "base_increment"),
        ("base_min_size", "base_min"),
        ("quote_min_size", "quote_min"),
    ):
        try:
            val = float(product[src])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"product constraint {src} missing or non-numeric") from None
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"product constraint {src} must be finite and > 0, got {val}")
        out[dst] = val
    return out


def hourly_overlap(hourly: pd.DataFrame, daily_cal: pd.DataFrame) -> Dict[str, Any]:
    """Informational only: complete-UTC-day hourly closes vs daily candle closes."""
    h = hourly.assign(day=pd.to_datetime(hourly["start"].astype("int64"), unit="s").dt.floor("D"))
    g = h.sort_values("start").groupby("day").agg(n=("close", "size"), close=("close", "last"))
    g = g[g["n"] == 24]
    joined = g.join(daily_cal["close"].rename("daily"), how="inner").dropna()
    rel = (joined["close"] / joined["daily"] - 1).abs()
    return {
        "days_compared": int(len(joined)),
        "max_abs_rel_diff": float(rel.max()) if len(rel) else None,
        "days_over_10bps": int((rel > 0.001).sum()),
    }
