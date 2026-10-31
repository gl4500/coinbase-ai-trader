import asyncio

from tools.recorder.pollers import (
    Poll,
    app_status,
    default_polls,
    next_due,
    poll_once,
    run_poller,
)
from tools.recorder.store import SegmentStore, read_records

T0 = 1_791_000_000 * 10**9
P = Poll("okx_funding_BTC", "https://example/f", 300)


def _recs(root, name="okx_funding_BTC"):
    recs = []
    for seg in sorted((root / "poll" / name).rglob("*.jsonl.gz")):
        recs += read_records(seg)
    return recs


def _clock(*values):
    it = iter(values)
    return lambda: next(it)


def test_default_polls_cover_every_source_and_both_coins():
    polls = {p.name: p for p in default_polls()}
    for coin in ("BTC", "ETH"):
        for kind in (
            "okx_funding",
            "okx_funding_history",
            "okx_oi",
            "okx_mark",
            "okx_index",
            "intx_quote",
            "deribit_futures",
            "deribit_options",
        ):
            assert f"{kind}_{coin}" in polls
    for name in (
        "coinbase_spot_catalogue",
        "coinbase_futures_catalogue_all",
        "okx_instruments_swap",
        "intx_instruments",
        "deribit_instruments_BTC",
        "deribit_instruments_ETH",
    ):
        assert name in polls
    assert polls["deribit_options_BTC"].interval_s == 900
    assert polls["okx_funding_history_BTC"].interval_s == 28800
    assert polls["coinbase_spot_catalogue"].interval_s == 86400
    assert all(p.url.startswith("https://") for p in polls.values())


def test_next_due_aligns_to_wall_clock_boundaries():
    assert next_due(1000.0, 300) == 1200.0
    assert next_due(1200.0, 300) == 1500.0


def test_app_status_classifies_transport_and_application_errors():
    assert app_status(200, '{"code":"0","data":[]}') == "ok"
    assert app_status(200, '{"code":"51001","msg":"bad inst"}') == "app_error"
    assert app_status(200, '{"jsonrpc":"2.0","error":{"code":10001}}') == "app_error"
    assert app_status(200, '{"jsonrpc":"2.0","result":[]}') == "ok"
    assert app_status(200, "<html>") == "unparseable"
    assert app_status(451, "restricted") == "http_error"
    assert app_status(None, None) == "transport_error"


def test_received_at_is_after_the_response_not_the_request(tmp_path):
    store = SegmentStore(tmp_path, "r")
    started = T0 + (20 * 3600 - 1) * 10**9  # 23:59:59 UTC
    received = started + 2 * 10**9  # 00:00:01 next day

    async def get(url):
        return 200, '{"code":"0","data":[1]}'

    asyncio.run(poll_once(P, store, get, _clock(started, received)))
    store.close()
    (r,) = _recs(tmp_path)
    assert r["received_at_ns"] == received
    assert r["meta"]["request_started_at_ns"] == started
    assert r["meta"]["app_status"] == "ok"
    seg = next((tmp_path / "poll" / "okx_funding_BTC").rglob("*.jsonl.gz"))
    assert seg.parent.name == "2026-10-04"


def test_errors_are_recorded_not_substituted(tmp_path):
    store = SegmentStore(tmp_path, "r")

    async def get_451(url):
        return 451, "restricted location"

    async def get_raise(url):
        raise TimeoutError("read timeout")

    asyncio.run(poll_once(P, store, get_451, _clock(T0, T0 + 1)))
    asyncio.run(poll_once(P, store, get_raise, _clock(T0 + 2, T0 + 3)))
    store.close()
    a, b = _recs(tmp_path)
    assert (a["status"], a["payload"], a["meta"]["app_status"]) == (
        451,
        "restricted location",
        "http_error",
    )
    assert b["status"] is None and b["payload"] is None and "read timeout" in b["error"]
    assert b["received_at_ns"] == T0 + 3 and b["meta"]["app_status"] == "transport_error"
    assert store.stats["poll/okx_funding_BTC"]["last_app_status"] == "transport_error"


def test_run_poller_polls_immediately_and_stops(tmp_path):
    store = SegmentStore(tmp_path, "r")
    calls = []
    stop = asyncio.Event()

    async def get(url):
        calls.append(url)
        stop.set()
        return 200, "{}"

    async def go():
        await asyncio.wait_for(run_poller(P, store, get, stop, clock=lambda: T0), timeout=2)

    asyncio.run(go())
    store.close()
    assert calls == ["https://example/f"] and len(_recs(tmp_path)) == 1
