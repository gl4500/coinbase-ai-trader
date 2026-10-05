"""Read-only market alert for BTC, ETH, SOL and XRP. Places NO orders; reads public market data only.

  python -m tools.alerts.run [--out DIR] [--capital USD]

Each run checks (1) dated CDE futures carry for the four coins against the cost/cash hurdle and the
operator's capital, and (2) large 1h/24h spot moves. Every run appends one JSON line to
<out>/alerts.jsonl and rewrites <out>/latest.json; a NEW alert (outside its cooldown) also raises a
Windows notification. A failure for one product is recorded and never stops the others.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from tools.alerts.carry import CarryParams, evaluate_dated
from tools.alerts.moves import move_alerts
from tools.alerts.notify import should_notify, windows_toast

SCOPE = ("BTC", "ETH", "SOL", "XRP")
DEFAULT_OUT = Path(r"C:\Users\gl450\market_alerts")
PUBLIC = "https://api.coinbase.com/api/v3/brokerage/market"
COOLDOWN_S = 6 * 3600
PERP_EXPIRY_YEAR = 2080  # CDE perpetual-style products carry a far-future placeholder expiry


class PublicSource:
    def __init__(self, client):
        self.c = client

    def _get(self, path, **params):
        r = self.c.get(f"{PUBLIC}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def products(self):
        return self._get("/products", product_type="FUTURE", limit=250)["products"]

    def top(self, pid):
        book = self._get("/product_book", product_id=pid, limit=1)["pricebook"]
        return float(book["bids"][0]["price"]), float(book["asks"][0]["price"])

    def prices(self, pid):
        end = int(time.time())
        rows = self._get(
            f"/products/{pid}/candles",
            start=end - 300 * 300,
            end=end,
            granularity="FIVE_MINUTE",
            limit=300,
        )["candles"]
        closes = [float(r["close"]) for r in sorted(rows, key=lambda r: int(r["start"]))]
        if len(closes) < 13:
            return None, None, None
        return closes[-1], closes[-13], (closes[-289] if len(closes) >= 289 else None)

    def toast(self, title, msg):
        windows_toast(title, msg)


def _dated_in_scope(products):
    for p in products:
        f = p.get("future_product_details") or {}
        unit, expiry = f.get("contract_root_unit"), f.get("contract_expiry")
        if unit not in SCOPE or not expiry:
            continue
        exp = dt.datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        if exp.year >= PERP_EXPIRY_YEAR:
            continue
        yield p, unit, exp, f


def run_once(out: Path, src, *, now: float, params: CarryParams = CarryParams()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    carry, alerts, moves_checked = [], [], []
    when = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    try:
        products = list(src.products())
    except Exception as exc:
        products = []
        carry.append({"product_id": "*", "error": f"products: {exc!r}"})
    for p, unit, exp, f in _dated_in_scope(products):
        pid = p["product_id"]
        try:
            fut_bid, _ = src.top(pid)
            _, spot_ask = src.top(f"{unit}-USD")
            days = (exp - when).total_seconds() / 86400
            r = evaluate_dated(
                spot_ask=spot_ask,
                fut_bid=fut_bid,
                contract_size=float(f["contract_size"]),
                short_margin=float(f["overnight_margin_rate"]["short_margin_rate"]),
                days_to_expiry=days,
                p=params,
            )
            row = {
                "product_id": pid,
                "unit": unit,
                "days": round(days, 2),
                "fut_bid": fut_bid,
                "spot_ask": spot_ask,
                **r,
            }
        except Exception as exc:
            row = {"product_id": pid, "unit": unit, "error": repr(exc), "alert": False}
        carry.append(row)
        if row.get("alert"):
            alerts.append(
                {
                    "kind": "carry",
                    "product_id": pid,
                    "excess_annual": row["excess_annual"],
                    "contracts_affordable": row["contracts_affordable"],
                }
            )
    for unit in SCOPE:
        pid = f"{unit}-USD"
        try:
            now_px, px_1h, px_24h = src.prices(pid)
            moves_checked.append(
                {"product_id": pid, "now": now_px, "px_1h": px_1h, "px_24h": px_24h}
            )
            alerts.extend(move_alerts(pid, now_px, px_1h, px_24h))
        except Exception as exc:
            moves_checked.append({"product_id": pid, "error": repr(exc)})
    notified = []
    for a in alerts:
        key = f"{a['kind']}:{a['product_id']}"
        if should_notify(key, now, state, COOLDOWN_S):
            notified.append(key)
            if a["kind"] == "carry":
                msg = f"{a['product_id']}: ~{a['excess_annual']:.1%}/yr above cash on all capital ({a['contracts_affordable']} contract(s) affordable). Review only; nothing was traded."
            else:
                msg = f"{a['product_id']} moved {a['change']:+.1%} ({a['kind'][5:]}). Review only; nothing was traded."
            src.toast("Market alert", msg)
    record = {
        "time": when.isoformat(),
        "capital": params.capital,
        "alerts": alerts,
        "notified": notified,
        "carry": carry,
        "moves_checked": moves_checked,
    }
    with (out / "alerts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    tmp = out / "latest.json.tmp"
    tmp.write_text(json.dumps(record, indent=1))
    tmp.replace(out / "latest.json")
    state_path.write_text(json.dumps(state))
    return record


def _emit(text: str) -> None:
    if sys.stdout is not None:  # None under pythonw (scheduled, no console)
        print(text)


def cli(argv=None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description="Read-only market alert (BTC, ETH, SOL, XRP)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--capital", type=float, default=CarryParams.capital)
    args = ap.parse_args(argv)
    with httpx.Client(headers={"User-Agent": "market-alerts/1"}) as client:
        rec = run_once(
            args.out,
            PublicSource(client),
            now=time.time(),
            params=CarryParams(capital=args.capital),
        )
    _emit(json.dumps({"time": rec["time"], "alerts": rec["alerts"], "notified": rec["notified"]}))
    return 0


if __name__ == "__main__":
    sys.exit(cli())
