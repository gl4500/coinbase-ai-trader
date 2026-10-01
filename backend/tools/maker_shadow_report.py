"""Read-only summary of the paper maker-entry shadow (services/maker_shadow).

Every rate here is CONDITIONAL: over intents created after successful paper BUYs,
that had a usable quote, were not duplicates, and whose observation interval saw
no WS reconnect. ``coverage`` says what share of all rows that is. A cross rate is
a price-path proxy for a maker fill, not a fill rate — see the module docstring of
services/maker_shadow.py for the estimand.
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

LATE_S = 10.0


def _med(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    return median(vals) if vals else None


def summarize(rows: List[Dict]) -> Dict:
    observed = [r for r in rows if r["status"] in ("crossed", "not_crossed")]
    clean = [r for r in observed if r["feed_gap"] is False]
    crossed = [r for r in clean if r["status"] == "crossed"]
    n = len(clean)
    return {
        "n_total": len(rows),
        "n_measured": n,
        "n_no_quote": sum(r["status"] == "no_quote" for r in rows),
        "n_duplicate": sum(r["status"] == "duplicate" for r in rows),
        "n_feed_gap": sum(r["feed_gap"] is True for r in observed),
        "n_feed_unknown": sum(r["feed_gap"] is None for r in observed),
        "n_late": sum((r["finalised_late_s"] or 0.0) > LATE_S for r in observed),
        "coverage": n / len(rows) if rows else None,
        "cross_rate": len(crossed) / n if n else None,
        "touch_rate": sum(bool(r["touched"]) for r in clean) / n if n else None,
        "median_time_to_cross_s": _med([r["time_to_cross_s"] for r in crossed]),
        "median_spread_bps": _med([r["spread_bps"] for r in clean]),
        "median_markout_bps_crossed": _med([r["markout_bps"] for r in crossed]),
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
