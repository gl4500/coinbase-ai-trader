"""Entry point:  python -m tools.recorder.run [--out DIR] [--products BTC-USD,ETH-USD]

Standalone and read-only. A storage failure is fatal: the run stops with a nonzero exit and a
stderr message rather than continuing as if data were being kept.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
import time
import uuid
from pathlib import Path

from tools.recorder.coinbase_ws import CHANNELS, run_coinbase_ws
from tools.recorder.health import MIN_FREE_BYTES, run_health
from tools.recorder.lock import WriterLock
from tools.recorder.pollers import default_polls, make_http_get, run_poller
from tools.recorder.store import SegmentStore, StoreError, envelope, recover_incomplete

DEFAULT_OUT = Path(r"C:\Users\gl450\market_recorder_data")
WS_STREAMS = {"level2": "l2_data", "market_trades": "market_trades", "heartbeats": "heartbeats"}


def package_digest() -> str:
    h = hashlib.sha256()
    for py in sorted(Path(__file__).parent.glob("*.py")):
        h.update(py.name.encode() + b"\0" + py.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return "sha256:" + h.hexdigest()


def expected_streams() -> list:
    return [f"coinbase_ws/{WS_STREAMS[c]}" for c in CHANNELS] + [
        f"poll/{p.name}" for p in default_polls()
    ]


async def main(out: Path, products: list) -> None:
    import httpx

    lock = WriterLock(out).acquire()
    stop = asyncio.Event()
    store = SegmentStore(out, uuid.uuid4().hex[:12])
    primary = None
    try:
        salvaged = recover_incomplete(out)
        store.write(
            "recorder/events",
            envelope(
                "recorder",
                "event",
                time.time_ns(),
                meta={
                    "event": "start",
                    "pid": os.getpid(),
                    "products": products,
                    "package_sha256": package_digest(),
                    "python": sys.version.split()[0],
                    "polls": [[p.name, p.url, p.interval_s] for p in default_polls()],
                    "min_free_bytes": MIN_FREE_BYTES,
                    "mono_ns": time.monotonic_ns(),
                    "salvaged_segments": salvaged,
                },
            ),
        )
        async with httpx.AsyncClient(headers={"User-Agent": "market-recorder/1"}) as client:
            get = make_http_get(client)
            tasks = [
                run_coinbase_ws(store, products, stop),
                run_health(store, out, stop, expected=expected_streams()),
            ]
            tasks += [run_poller(p, store, get, stop) for p in default_polls()]
            tasks.append(run_flusher(store, stop))
            await asyncio.gather(*tasks)
    except BaseException as exc:
        primary = exc
        raise
    finally:
        stop.set()
        finish(store, lock, primary=primary)


def _is_cancellation(exc) -> bool:
    return isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))


def finish(store, lock, primary) -> None:
    """Finalise and release. Each step is attempted independently (a failed stop event never
    skips closing the other streams). A finalisation failure is FATAL unless a GENUINE earlier
    error is already propagating, in which case it is reported on stderr instead of masking it.
    An operator stop (Ctrl+C / cancellation) is not a genuine error: a storage failure during
    it still surfaces as StoreError, so the CLI exits nonzero."""
    errors = []
    try:
        for step in (
            lambda: store.write(
                "recorder/events",
                envelope("recorder", "event", time.time_ns(), meta={"event": "stop"}),
            ),
            store.close,
        ):
            try:
                step()
            except StoreError as exc:
                errors.append(str(exc))
    finally:
        lock.release()
    if not errors:
        return
    message = "; ".join(errors)
    print(f"recorder: could not finalise segments: {message}", file=sys.stderr)
    if primary is None or _is_cancellation(primary):
        raise StoreError(message)


async def run_flusher(store, stop: asyncio.Event, interval_s: float = 1.0) -> None:
    """Independent timer so a quiet stream is still flushed within the declared bound."""
    while not stop.is_set():
        store.flush_due()
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


def cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Standalone read-only market data recorder")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--products", default="BTC-USD,ETH-USD")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    products = [p.strip() for p in args.products.split(",") if p.strip()]
    try:
        asyncio.run(main(args.out, products))
    except KeyboardInterrupt:
        return 0
    except StoreError as exc:
        print(f"recorder: FATAL storage failure, stopped: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"recorder: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(cli())
