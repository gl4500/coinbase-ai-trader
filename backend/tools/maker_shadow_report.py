"""Read-only summary of the paper maker-fill shadow (services/maker_shadow).

Rates use MEASURED intents only (filled + unfilled). no_quote / duplicate rows
are counted and reported, never silently folded into either side.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _med(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return median(xs) if xs else None


def summarize(rows: List[Dict]) -> Dict:
    measured = [r for r in rows if r["status"] in ("filled", "unfilled")]
    filled = [r for r in measured if r["status"] == "filled"]
    n = len(measured)
    return {
        "n_total": len(rows),
        "n_measured": n,
        "n_no_quote": sum(r["status"] == "no_quote" for r in rows),
        "n_duplicate": sum(r["status"] == "duplicate" for r in rows),
        "n_filled": len(filled),
        "n_touched": sum(bool(r["touched"]) for r in measured),
        "fill_rate": len(filled) / n if n else None,
        "touch_rate": sum(bool(r["touched"]) for r in measured) / n if n else None,
        "median_time_to_fill_s": _med([r["time_to_fill_s"] for r in filled]),
        "median_spread_bps": _med([r["spread_bps"] for r in measured]),
        "median_drift_bps_filled": _med([r["drift_bps"] for r in filled]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since-hours", type=float, default=None)
    args = ap.parse_args()
    import database

    since = time.time() - args.since_hours * 3600 if args.since_hours else None
    rows = asyncio.run(database.get_maker_shadow_rows(since_ts=since))
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()
