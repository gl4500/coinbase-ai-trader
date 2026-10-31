"""Public REST pollers.

`received_at_ns` is taken AFTER the response (or the exception); the request start is kept in
`meta.request_started_at_ns`. A failed, non-200 or application-error response is recorded as
such, and nothing is ever substituted for it.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from typing import Optional

from tools.recorder.store import envelope

OKX = "https://www.okx.com/api/v5"
DERIBIT = "https://www.deribit.com/api/v2/public"
INTX = "https://api.international.coinbase.com/api/v1"
COINBASE = "https://api.coinbase.com/api/v3/brokerage/market"
DAY_S, EIGHT_H_S = 86400, 28800


@dataclass(frozen=True)
class Poll:
    name: str
    url: str
    interval_s: int


def default_polls() -> tuple:
    polls = []
    for coin in ("BTC", "ETH"):
        swap = f"{coin}-USDT-SWAP"
        polls += [
            Poll(f"okx_funding_{coin}", f"{OKX}/public/funding-rate?instId={swap}", 300),
            Poll(
                f"okx_funding_history_{coin}",
                f"{OKX}/public/funding-rate-history?instId={swap}&limit=100",
                EIGHT_H_S,
            ),
            Poll(f"okx_oi_{coin}", f"{OKX}/public/open-interest?instType=SWAP&instId={swap}", 300),
            Poll(f"okx_mark_{coin}", f"{OKX}/public/mark-price?instType=SWAP&instId={swap}", 300),
            Poll(f"okx_index_{coin}", f"{OKX}/market/index-tickers?instId={coin}-USDT", 300),
            Poll(f"intx_quote_{coin}", f"{INTX}/instruments/{coin}-PERP/quote", 300),
            Poll(
                f"deribit_futures_{coin}",
                f"{DERIBIT}/get_book_summary_by_currency?currency={coin}&kind=future",
                300,
            ),
            Poll(
                f"deribit_options_{coin}",
                f"{DERIBIT}/get_book_summary_by_currency?currency={coin}&kind=option",
                900,
            ),
            Poll(
                f"deribit_instruments_{coin}", f"{DERIBIT}/get_instruments?currency={coin}", DAY_S
            ),
        ]
    polls += [
        Poll("okx_instruments_swap", f"{OKX}/public/instruments?instType=SWAP", DAY_S),
        Poll("intx_instruments", f"{INTX}/instruments", DAY_S),
        Poll("coinbase_spot_catalogue", f"{COINBASE}/products?limit=5000", DAY_S),
        Poll(
            "coinbase_futures_catalogue_all",
            f"{COINBASE}/products?limit=5000&product_type=FUTURE&get_all_products=true",
            DAY_S,
        ),
    ]
    return tuple(polls)


def next_due(now_s: float, interval_s: int) -> float:
    return (math.floor(now_s / interval_s) + 1) * interval_s


def app_status(status: Optional[int], text: Optional[str]) -> str:
    """Classify a response WITHOUT extracting values: transport / http / application / ok."""
    if status is None:
        return "transport_error"
    if status != 200:
        return "http_error"
    try:
        body = json.loads(text)
    except (TypeError, ValueError):
        return "unparseable"
    if isinstance(body, dict):
        if "error" in body and body["error"]:
            return "app_error"
        if "code" in body and str(body["code"]) != "0":
            return "app_error"
    return "ok"


async def poll_once(poll: Poll, store, http_get, clock) -> None:
    started = clock()
    status, text, error = None, None, None
    try:
        status, text = await http_get(poll.url)
    except Exception as exc:
        error = repr(exc)
    received = clock()
    verdict = app_status(status, text)
    store.write(
        f"poll/{poll.name}",
        envelope(
            "poll",
            poll.name,
            received,
            payload=text,
            status=status,
            error=error,
            meta={"url": poll.url, "request_started_at_ns": started, "app_status": verdict},
        ),
    )
    store.stats[f"poll/{poll.name}"]["last_app_status"] = verdict


async def run_poller(
    poll: Poll, store, http_get, stop: asyncio.Event, *, clock=time.time_ns, wall=time.time
) -> None:
    await poll_once(poll, store, http_get, clock)
    while not stop.is_set():
        delay = max(0.0, next_due(wall(), poll.interval_s) - wall())
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            await poll_once(poll, store, http_get, clock)


def make_http_get(client):
    async def get(url: str):
        resp = await client.get(url, timeout=20)
        return resp.status_code, resp.text

    return get
