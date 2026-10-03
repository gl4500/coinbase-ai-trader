"""Coinbase Advanced Trade public WebSocket capture (no auth).

Payloads are stored verbatim. Continuity is recorded as events, never inferred:
- connect, disconnect, stale;
- gap: the per-connection sequence_num is not previous+1;
- raw_discard_start and resubscribe_after_pause.

A gap or a resume after a raw pause forces a reconnect, so the next L2 data starts from a fresh
snapshot. A storage failure (StoreError) is fatal and is never reported as a network disconnect.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional, Tuple

from tools.recorder.store import StoreError, envelope

WS_URL = "wss://advanced-trade-ws.coinbase.com"
CHANNELS = ("level2", "market_trades", "heartbeats")
SOURCE = "coinbase_ws"
EVENTS = f"{SOURCE}/events"


class SeqTracker:
    """sequence_num is per connection, across channels and products (verified on a real capture)."""

    def __init__(self) -> None:
        self.last: Optional[int] = None

    def observe(self, seq: int) -> Optional[Tuple[int, int]]:
        gap = None
        if self.last is not None and seq != self.last + 1:
            gap = (self.last + 1, seq)
        self.last = seq
        return gap


def backoff_s(attempt: int) -> float:
    return float(min(60, 2**attempt))


def _default_connect(url: str):
    import websockets

    return websockets.connect(url, max_size=None, ping_interval=20)


async def run_coinbase_ws(
    store,
    products,
    stop: asyncio.Event,
    *,
    connect=None,
    clock=time.time_ns,
    max_silence_s: float = 30.0,
    url: str = WS_URL,
    sleep=asyncio.sleep,
) -> None:
    connect = connect or _default_connect
    attempt, conn_id = 0, 0
    while not stop.is_set():
        conn_id += 1

        def event(name, at=None, error=None, _cid=conn_id, **meta) -> None:
            store.write(
                EVENTS,
                envelope(
                    SOURCE,
                    "event",
                    at or clock(),
                    error=error,
                    meta={"event": name, "conn_id": _cid, **meta},
                ),
            )

        tracker, discarded = SeqTracker(), False
        try:
            async with connect(url) as ws:
                event("connect", products=list(products))
                for ch in CHANNELS:
                    await ws.send(
                        json.dumps(
                            {"type": "subscribe", "product_ids": list(products), "channel": ch}
                        )
                    )
                attempt = 0
                while not stop.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=max_silence_s)
                    except asyncio.TimeoutError:
                        event("stale", silence_s=max_silence_s)
                        break
                    received = clock()
                    try:
                        head = json.loads(msg)
                        channel, seq = str(head.get("channel", "unknown")), head.get("sequence_num")
                    except ValueError:
                        channel, seq = "unparsed", None
                    gap = tracker.observe(seq) if isinstance(seq, int) else None
                    if channel != "heartbeats" and not store.raw_enabled:
                        if not discarded:
                            event("raw_discard_start", at=received)
                            discarded = True
                        continue
                    if discarded and store.raw_enabled:
                        event("resubscribe_after_pause", at=received)
                        break
                    store.write(
                        f"{SOURCE}/{channel}", envelope(SOURCE, "message", received, payload=msg)
                    )
                    if gap:
                        event("gap", at=received, expected=gap[0], got=gap[1], channel=channel)
                        break
        except StoreError:
            raise
        except Exception as exc:
            event("disconnect", error=repr(exc))
        if stop.is_set():
            break
        await sleep(backoff_s(attempt))
        attempt += 1
