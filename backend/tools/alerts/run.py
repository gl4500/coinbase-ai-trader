"""Read-only market alert for BTC, ETH, SOL and XRP. Places NO orders; reads public market data only.

  python -m tools.alerts.run [--out DIR] [--capital USD]

Each run checks (1) dated CDE futures carry for the four coins against the cost/cash hurdle and the
operator's capital, and (2) large 1h/24h spot moves. Every run appends one JSON line to
<out>/alerts.jsonl and rewrites <out>/latest.json; a NEW alert (outside its cooldown) also raises a
Windows notification. A failure for one product is recorded and never stops the others.

Eligibility fails closed: venue FCM/cde, contract_expiry_type EXPIRING, no funding fields (CDE
perp-style products are labelled EXPIRING with a 2089 placeholder expiry but carry funding), and a
parseable timezone-aware expiry. Excluded in-scope products are listed with a reason. The product
listing is paged; `carry_coverage` is COMPLETE, INCOMPLETE (page bound hit) or FAILED, and only
COMPLETE means "every listed dated future was checked".

Moves use the latest CLOSED 5-minute bar (ended within FRESH_MAX_AGE_S of now) and the bars that
start exactly 1h and 24h earlier; `asof`, `ref_1h` and `ref_24h` are recorded. A missing anchor is
None (no fallback); conflicting duplicate bars or a stale series make the product unavailable.

Assumes ONE writer per --out directory: state.json (notification cooldowns) is read and rewritten
without locking, so concurrent runs against the same directory can lose a cooldown and notify twice.
A notification is best effort; the JSONL line is the record.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
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
PAGE_LIMIT = 250
MAX_PRODUCT_PAGES = 8  # bound on the product listing; hitting it marks carry coverage INCOMPLETE
BAR_S = 300  # FIVE_MINUTE candles
FRESH_MAX_AGE_S = 600  # the latest CLOSED bar must have ended within this long of `now`


class PublicSource:
    def __init__(self, client):
        self.c = client

    def _get(self, path, **params):
        r = self.c.get(f"{PUBLIC}{path}", params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def products(self) -> dict:
        """All FUTURE products, paged by limit/offset. `complete` is False if the page bound was hit."""
        out, pages, has_next = [], 0, True
        while has_next and pages < MAX_PRODUCT_PAGES:
            page = self._get("/products", product_type="FUTURE", limit=PAGE_LIMIT, offset=len(out))
            rows = page["products"]
            out.extend(rows)
            pages += 1
            pag = page.get("pagination") or {}
            has_next = bool(pag["has_next"]) if "has_next" in pag else len(rows) >= PAGE_LIMIT
            if not rows:
                has_next = False
        return {"products": out, "complete": not has_next, "pages": pages}

    def top(self, pid):
        book = self._get("/product_book", product_id=pid, limit=1)["pricebook"]
        return float(book["bids"][0]["price"]), float(book["asks"][0]["price"])

    def prices(self, pid, now: float) -> dict:
        end = int(now)
        rows = self._get(
            f"/products/{pid}/candles",
            start=end - 300 * BAR_S,
            end=end,
            granularity="FIVE_MINUTE",
            limit=300,
        )["candles"]
        return select_anchors(rows, now)

    def toast(self, title, msg):
        windows_toast(title, msg)


def select_anchors(rows, now: float) -> dict:
    """Pick the latest CLOSED 5-minute bar and the bars exactly 1h and 24h before it, by timestamp.

    Identical duplicate rows collapse; conflicting duplicates, no closed bar, or a stale series make
    the whole result unavailable. A missing anchor bar leaves that price None (no fallback).
    """
    bars = {}
    for r in rows:
        start = int(r["start"])
        vals = tuple(float(r[k]) for k in ("open", "high", "low", "close", "volume"))
        if start in bars and bars[start] != vals:
            return {"unavailable": f"conflicting duplicate bar at {start}"}
        bars[start] = vals
    closed = [s for s in bars if s + BAR_S <= now]
    if not closed:
        return {"unavailable": "no closed bar"}
    last = max(closed)
    asof = last + BAR_S
    if now - asof > FRESH_MAX_AGE_S:
        return {
            "unavailable": f"stale: latest closed bar ended {now - asof:.0f}s before now",
            "asof": asof,
        }
    out = {"now": bars[last][3], "asof": asof, "missing": []}
    for key, lag in (("1h", 3600), ("24h", 86400)):
        out[f"ref_{key}"] = asof - lag
        bar = bars.get(last - lag)
        out[f"px_{key}"] = bar[3] if bar else None
        if bar is None:
            out["missing"].append(f"px_{key}")
    return out


def _classify(p):
    """('out_of_scope', None) | ('excluded', reason) | ('eligible', (unit, expiry, details)).

    Fails closed: an unknown venue or expiry type is excluded. Raises on malformed eligible metadata.
    Perp-style CDE products are labelled EXPIRING with a placeholder expiry but carry funding.
    """
    f = p.get("future_product_details") or {}
    unit = f.get("contract_root_unit")
    if unit not in SCOPE:
        return "out_of_scope", None
    venue, fvenue = p.get("product_venue"), str(f.get("venue") or "").lower()
    if venue != "FCM" or fvenue != "cde":
        return "excluded", f"venue {venue!r}/{fvenue!r} is not CDE"
    etype = f.get("contract_expiry_type")
    if etype != "EXPIRING":
        return "excluded", f"contract_expiry_type {etype!r} is not EXPIRING"
    if f.get("funding_interval") or f.get("funding_rate") not in (None, ""):
        return "excluded", "carries funding (perpetual-style)"
    expiry = f.get("contract_expiry")
    if not expiry:
        raise ValueError("missing contract_expiry")
    exp = dt.datetime.fromisoformat(expiry.replace("Z", "+00:00"))
    if exp.tzinfo is None:
        raise ValueError(f"contract_expiry {expiry!r} has no timezone")
    if exp.year >= PERP_EXPIRY_YEAR:
        return "excluded", "placeholder far-future expiry"
    return "eligible", (unit, exp, f)


def run_once(out: Path, src, *, now: float, params: CarryParams = CarryParams()) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    carry, alerts, moves_checked, excluded = [], [], [], []
    when = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    try:
        listing = src.products()
        products = list(listing["products"])
        coverage = "COMPLETE" if listing["complete"] else "INCOMPLETE"
    except Exception as exc:
        products, coverage = [], "FAILED"
        carry.append({"product_id": "*", "error": f"products: {exc!r}"})
    for p in products:
        pid = p.get("product_id") if isinstance(p, dict) else None
        unit = None
        try:
            if not pid:
                raise ValueError(f"product without product_id: {str(p)[:80]}")
            kind, info = _classify(p)
            if kind == "out_of_scope":
                continue
            if kind == "excluded":
                excluded.append({"product_id": pid, "reason": info})
                continue
            unit, exp, f = info
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
            m = src.prices(pid, now)
            moves_checked.append({"product_id": pid, **m})
            if "unavailable" not in m:
                alerts.extend(move_alerts(pid, m["now"], m["px_1h"], m["px_24h"]))
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
        "carry_coverage": coverage,
        "alerts": alerts,
        "notified": notified,
        "carry": carry,
        "excluded": excluded,
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


def _positive_finite(text: str) -> float:
    v = float(text)
    if not (math.isfinite(v) and v > 0):
        raise argparse.ArgumentTypeError(f"must be a positive finite number, got {text!r}")
    return v


def cli(argv=None) -> int:
    import httpx

    ap = argparse.ArgumentParser(description="Read-only market alert (BTC, ETH, SOL, XRP)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--capital", type=_positive_finite, default=CarryParams.capital)
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
