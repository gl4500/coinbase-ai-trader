import json

from tools.alerts import run as R


def _future(pid, unit, size, expiry, margin):
    return {
        "product_id": pid,
        "display_name": pid,
        "future_product_details": {
            "contract_root_unit": unit,
            "contract_size": str(size),
            "contract_expiry": expiry,
            "overnight_margin_rate": {"short_margin_rate": str(margin)},
        },
    }


PRODUCTS = [
    _future("BIT-30OCT26-CDE", "BTC", 0.01, "2026-10-30T16:00:00Z", 0.2535),
    _future("BIP-20DEC30-CDE", "BTC", 0.01, "2089-12-30T16:00:00Z", 0.2566),  # perp: excluded
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


class Fake:
    def __init__(self):
        self.toasts = []

    def products(self):
        return PRODUCTS

    def top(self, pid):
        return BOOKS[pid]

    def prices(self, pid):
        return PRICES[pid]

    def toast(self, title, msg):
        self.toasts.append((title, msg))


def test_run_once_scopes_to_four_coins_logs_and_notifies_once(tmp_path):
    fake = Fake()
    now = 1_791_200_000.0  # 2026-10-05
    out = R.run_once(tmp_path, fake, now=now)
    checked = {c["product_id"] for c in out["carry"]}
    assert checked == {"BIT-30OCT26-CDE", "ET-30OCT26-CDE"}  # perp and DOGE excluded
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
    R.run_once(tmp_path, fake, now=now + 60)  # within cooldown
    assert len(fake.toasts) == n_toasts


def test_a_fetch_failure_for_one_product_is_logged_not_fatal(tmp_path):
    class Flaky(Fake):
        def top(self, pid):
            if pid == "ET-30OCT26-CDE":
                raise RuntimeError("book unavailable")
            return super().top(pid)

    out = R.run_once(tmp_path, Flaky(), now=1_791_200_000.0)
    errs = [c for c in out["carry"] if c.get("error")]
    assert [c["product_id"] for c in errs] == ["ET-30OCT26-CDE"]


def test_emit_is_safe_without_a_console(monkeypatch):
    """Scheduled runs use pythonw (no console): sys.stdout is None and print would crash."""
    import sys

    monkeypatch.setattr(sys, "stdout", None)
    R._emit("ok")  # must not raise
