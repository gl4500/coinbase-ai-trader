import json
import math

import pytest

from tools.alerts import run as R


def _future(
    pid, unit, size, expiry, margin, *, venue="FCM", fvenue="cde", etype="EXPIRING", funding=None
):
    return {
        "product_id": pid,
        "display_name": pid,
        "product_venue": venue,
        "future_product_details": {
            "venue": fvenue,
            "contract_expiry_type": etype,
            "contract_root_unit": unit,
            "contract_size": str(size),
            "contract_expiry": expiry,
            "funding_interval": funding,
            "funding_rate": "" if funding is None else "-0.000005",
            "overnight_margin_rate": {"short_margin_rate": str(margin)},
        },
    }


PRODUCTS = [
    _future("BIT-30OCT26-CDE", "BTC", 0.01, "2026-10-30T16:00:00Z", 0.2535),
    # perp-style CDE product: labelled EXPIRING with a placeholder expiry, but it carries funding
    _future("BIP-20DEC30-CDE", "BTC", 0.01, "2089-12-30T16:00:00Z", 0.2566, funding="3600s"),
    _future("DOGE-30OCT26-CDE", "DOGE", 1000, "2026-10-30T16:00:00Z", 0.5),  # not in scope
    _future("ET-30OCT26-CDE", "ETH", 0.1, "2026-10-30T16:00:00Z", 0.2995),
]
BOOKS = {
    "BIT-30OCT26-CDE": (90000.0, 90010.0),
    "BTC-USD": (86990.0, 87000.0),
    "ET-30OCT26-CDE": (2701.0, 2702.0),
    "ETH-USD": (2699.0, 2700.0),
    "SOL-USD": (200.0, 200.1),
    "XRP-USD": (2.50, 2.51),
}
PRICES = {
    "BTC-USD": (87000.0, 87100.0, 87500.0),
    "ETH-USD": (2700.0, 2550.0, 2650.0),
    "SOL-USD": (200.0, 199.0, 200.0),
    "XRP-USD": (2.5, 2.5, 2.5),
}
NOW = 1_791_200_000.0  # 2026-10-05


class Fake:
    def __init__(self, products=None):
        self.toasts = []
        self._products = PRODUCTS if products is None else products

    def products(self):
        return {"products": self._products, "complete": True, "pages": 1}

    def top(self, pid):
        return BOOKS[pid]

    def prices(self, pid, now):
        cur, h1, h24 = PRICES[pid]
        return {"now": cur, "px_1h": h1, "px_24h": h24, "asof": now, "missing": []}

    def toast(self, title, msg):
        self.toasts.append((title, msg))


def test_run_once_scopes_to_four_coins_logs_and_notifies_once(tmp_path):
    fake = Fake()
    out = R.run_once(tmp_path, fake, now=NOW)
    checked = {c["product_id"] for c in out["carry"]}
    assert checked == {"BIT-30OCT26-CDE", "ET-30OCT26-CDE"}  # perp and DOGE excluded
    assert out["carry_coverage"] == "COMPLETE"
    assert {m["product_id"] for m in out["moves_checked"]} == {
        "BTC-USD",
        "ETH-USD",
        "SOL-USD",
        "XRP-USD",
    }
    # ETH +5.9% in 1h -> move alert; BTC carry premium is large but one contract exceeds $1,000
    assert [(a["kind"], a["product_id"]) for a in out["alerts"]] == [("move_1h", "ETH-USD")]
    btc = next(c for c in out["carry"] if c["product_id"] == "BIT-30OCT26-CDE")
    assert btc["contracts_affordable"] == 0 and btc["alert"] is False
    lines = (tmp_path / "alerts.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["alerts"] == out["alerts"]
    n_toasts = len(fake.toasts)
    assert n_toasts == len(out["alerts"]) >= 1
    R.run_once(tmp_path, fake, now=NOW + 60)  # within cooldown
    assert len(fake.toasts) == n_toasts


def test_a_fetch_failure_for_one_product_is_logged_not_fatal(tmp_path):
    class Flaky(Fake):
        def top(self, pid):
            if pid == "ET-30OCT26-CDE":
                raise RuntimeError("book unavailable")
            return super().top(pid)

    out = R.run_once(tmp_path, Flaky(), now=NOW)
    errs = [c for c in out["carry"] if c.get("error")]
    assert [c["product_id"] for c in errs] == ["ET-30OCT26-CDE"]


def test_malformed_metadata_between_healthy_products_is_recorded_and_the_rest_still_run(tmp_path):
    bad_expiry = _future("BIT-BAD-CDE", "BTC", 0.01, "not-a-date", 0.25)
    no_id = _future("X", "ETH", 0.1, "2026-10-30T16:00:00Z", 0.3)
    del no_id["product_id"]
    products = [PRODUCTS[0], bad_expiry, "not-a-product", no_id, PRODUCTS[3]]
    out = R.run_once(tmp_path, Fake(products), now=NOW)
    healthy = {c["product_id"] for c in out["carry"] if not c.get("error")}
    assert healthy == {"BIT-30OCT26-CDE", "ET-30OCT26-CDE"}
    assert len([c for c in out["carry"] if c.get("error")]) == 3
    assert len(out["moves_checked"]) == 4
    assert (tmp_path / "latest.json").exists() and (tmp_path / "state.json").exists()


def test_eligibility_requires_cde_venue_expiring_type_and_no_funding(tmp_path):
    products = [
        _future("BIT-INTX", "BTC", 0.01, "2026-10-30T16:00:00Z", 0.25, venue="INTX", fvenue="intx"),
        _future("BIP-NEAR-CDE", "BTC", 0.01, "2026-11-27T16:00:00Z", 0.25, funding="3600s"),
        _future(
            "ET-UNK-CDE",
            "ETH",
            0.1,
            "2026-10-30T16:00:00Z",
            0.3,
            etype="UNKNOWN_CONTRACT_EXPIRY_TYPE",
        ),
        _future("ET-PERP-CDE", "ETH", 0.1, "2026-10-30T16:00:00Z", 0.3, etype="PERPETUAL"),
        PRODUCTS[0],
    ]
    out = R.run_once(tmp_path, Fake(products), now=NOW)
    assert [c["product_id"] for c in out["carry"]] == ["BIT-30OCT26-CDE"]
    excluded = {e["product_id"]: e["reason"] for e in out["excluded"]}
    assert set(excluded) == {"BIT-INTX", "BIP-NEAR-CDE", "ET-UNK-CDE", "ET-PERP-CDE"}
    assert "funding" in excluded["BIP-NEAR-CDE"]


def test_a_products_failure_marks_carry_coverage_failed(tmp_path):
    class Down(Fake):
        def products(self):
            raise RuntimeError("503")

    out = R.run_once(tmp_path, Down(), now=NOW)
    assert out["carry_coverage"] == "FAILED"


def test_an_incomplete_product_listing_is_reported_not_treated_as_complete(tmp_path):
    class Partial(Fake):
        def products(self):
            return {"products": PRODUCTS, "complete": False, "pages": R.MAX_PRODUCT_PAGES}

    out = R.run_once(tmp_path, Partial(), now=NOW)
    assert out["carry_coverage"] == "INCOMPLETE"


def test_emit_is_safe_without_a_console(monkeypatch):
    """Scheduled runs use pythonw (no console): sys.stdout is None and print would crash."""
    import sys

    monkeypatch.setattr(sys, "stdout", None)
    R._emit("ok")  # must not raise


def test_cli_rejects_a_non_positive_or_non_finite_capital(monkeypatch, tmp_path):
    def must_not_run(*a, **k):  # an accepted capital must never reach the network or a toast
        raise AssertionError("run_once reached with an invalid capital")

    monkeypatch.setattr(R, "run_once", must_not_run)
    for bad in ("0", "-5", "nan", "inf"):
        with pytest.raises(SystemExit):
            R.cli(["--capital", bad, "--out", str(tmp_path)])


# ── PublicSource against a mocked HTTP client (the real adapter, no network) ─────────────


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._p


class _Client:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return _Resp(self.handler(url, params or {}))


def _bars(end_start, n, close_fn=lambda i: 100.0 + i, skip=()):
    rows = []
    for i in range(n):
        start = end_start - (n - 1 - i) * 300
        if start in skip:
            continue
        c = close_fn(i)
        rows.append(
            {
                "start": str(start),
                "open": str(c),
                "high": str(c),
                "low": str(c),
                "close": str(c),
                "volume": "1",
            }
        )
    return rows


ANCHOR_NOW = 1_791_200_000  # multiple of 300
LAST = ANCHOR_NOW - 300  # latest CLOSED bar start


def _src(rows):
    return R.PublicSource(_Client(lambda url, p: {"candles": rows}))


def _close_at(rows, start):
    return next(float(r["close"]) for r in rows if int(r["start"]) == start)


def test_prices_anchor_by_timestamp_and_record_reference_times():
    rows = _bars(LAST, 300)
    m = _src(rows).prices("BTC-USD", ANCHOR_NOW)
    assert m["asof"] == LAST + 300
    assert m["now"] == _close_at(rows, LAST)
    assert m["px_1h"] == _close_at(rows, LAST - 3600) and m["ref_1h"] == LAST + 300 - 3600
    assert m["px_24h"] == _close_at(rows, LAST - 86400) and m["ref_24h"] == LAST + 300 - 86400


def test_a_gap_inside_the_window_does_not_shift_the_1h_anchor():
    rows = _bars(LAST, 300, skip={LAST - 1200})  # one missing bucket between anchor and now
    m = _src(rows).prices("BTC-USD", ANCHOR_NOW)
    assert m["px_1h"] == _close_at(rows, LAST - 3600)


def test_a_missing_anchor_bar_is_unavailable_with_no_fallback():
    rows = _bars(LAST, 300, skip={LAST - 3600})
    m = _src(rows).prices("BTC-USD", ANCHOR_NOW)
    assert m["px_1h"] is None and "px_1h" in m["missing"]


def test_the_in_progress_bar_is_never_used():
    rows = _bars(LAST + 300, 301)  # last row starts at ANCHOR_NOW: still open
    m = _src(rows).prices("BTC-USD", ANCHOR_NOW)
    assert m["asof"] == ANCHOR_NOW and m["now"] == _close_at(rows, LAST)


def test_identical_duplicates_collapse_but_conflicting_duplicates_are_unavailable():
    rows = _bars(LAST, 300)
    same = _src(rows + [dict(rows[-1])]).prices("BTC-USD", ANCHOR_NOW)
    assert same["now"] == _close_at(rows, LAST) and "unavailable" not in same
    clash = dict(rows[-1], close="999")
    bad = _src(rows + [clash]).prices("BTC-USD", ANCHOR_NOW)
    assert "unavailable" in bad and bad.get("now") is None


def test_a_stale_series_is_unavailable():
    rows = _bars(LAST - 3000, 300)  # latest closed bar ended ~50 min before now
    m = _src(rows).prices("BTC-USD", ANCHOR_NOW)
    assert "unavailable" in m and m.get("now") is None


def test_unavailable_prices_raise_no_move_alert(tmp_path):
    class Stale(Fake):
        def prices(self, pid, now):
            return {"unavailable": "stale", "asof": now - 9999}

    out = R.run_once(tmp_path, Stale(), now=NOW)
    assert not [a for a in out["alerts"] if a["kind"].startswith("move")]
    assert all("unavailable" in m for m in out["moves_checked"])


def test_products_follow_pagination_and_find_a_second_page_product():
    page1 = [PRODUCTS[0]] * 2
    page2 = [PRODUCTS[3]]

    def handler(url, p):
        off = int(p.get("offset", 0))
        if off == 0:
            return {"products": page1, "pagination": {"has_next": True}}
        return {"products": page2, "pagination": {"has_next": False}}

    res = R.PublicSource(_Client(handler)).products()
    assert [x["product_id"] for x in res["products"]][-1] == "ET-30OCT26-CDE"
    assert res["complete"] is True and res["pages"] == 2


def test_products_stop_at_the_page_bound_and_report_incomplete():
    def handler(url, p):
        return {"products": [PRODUCTS[0]], "pagination": {"has_next": True}}

    res = R.PublicSource(_Client(handler)).products()
    assert res["complete"] is False and res["pages"] == R.MAX_PRODUCT_PAGES
    assert math.isfinite(len(res["products"]))
