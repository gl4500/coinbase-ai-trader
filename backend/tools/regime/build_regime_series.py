"""Offline builder: daily series -> per-day RegimeState -> regime_state table.

Run:  python -m tools.regime.build_regime_series [--start 2016-01-01]
Phase-1 offline utility; no live/scan-loop involvement.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import List

import pandas as pd

from services.regime.features import corr_spx_90d, macro_risk_raw
from services.regime.macro_regime import evaluate
from services.regime.state import RegimeState

_WARMUP = 90


def build_series(df: "pd.DataFrame") -> List[RegimeState]:
    """One RegimeState per day from index position `_WARMUP` onward.

    Each day is evaluated on the data available up to and including that day,
    so the series carries no forward-looking information.
    """
    if df is None or df.empty or len(df) <= _WARMUP:
        return []
    btc = df["btc"].tolist()
    spx = df["spx"].tolist()
    dxy = df["dxy"].tolist()
    real_yield = df["real_yield"].tolist()
    mvrv = df["mvrv"].tolist()
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    states: List[RegimeState] = []
    for i in range(_WARMUP, len(df)):
        upto = slice(0, i + 1)
        m = mvrv[i]
        states.append(
            evaluate(
                date=dates[i],
                mvrv=(None if pd.isna(m) else float(m)),
                corr_spx_90d=corr_spx_90d(btc[upto], spx[upto]),
                macro_risk_raw=macro_risk_raw(spx[upto], dxy[upto], real_yield[upto]),
            )
        )
    return states


async def persist(states: List[RegimeState]) -> int:
    """Upsert every state into `regime_state`; returns the row count written.

    Ensures the schema first: this is an offline tool that may run against a DB
    the backend has never initialised. `init_db` is idempotent.
    """
    import database

    await database.init_db()
    for state in states:
        await database.upsert_regime_state(state.to_row())
    return len(states)


async def _main(start: str, cache_dir: str) -> None:
    from services.regime import sources

    df = sources.load_aligned(start, cache_dir)
    states = build_series(df)
    written = await persist(states)
    span = f"{states[0].date} -> {states[-1].date}" if states else "-"
    print(f"regime_state upserted: {written} days ({span})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument(
        "--cache-dir", default=os.path.join(os.path.dirname(__file__), "..", "..", "data", "regime")
    )
    args = ap.parse_args()
    asyncio.run(_main(args.start, args.cache_dir))
