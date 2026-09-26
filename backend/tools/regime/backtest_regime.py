"""Offline backtest: overlay regime exposure_scalar on the historical trades
record and report whether it improves risk-adjusted outcomes.

Linear paper model: scaled_pnl = pnl * exposure_scalar(opened_at date). This is
the Phase-1 gate — Phase 2 (live wiring) proceeds only if the verdict HELPS and
is robust across sub-periods.

The verdict is a cheap kill-filter, NOT a deploy signal: the overlay is applied
in-sample to the same trades, with no purged walk-forward and no DSR/PBO
deflation. A HELPS here earns a place in the gauntlet; it does not earn live
capital.

Run:  python -m tools.regime.backtest_regime
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from typing import Dict, List

# A scalar that barely moves across the trade window cannot discriminate:
# scaled_pnl collapses to k * pnl, which leaves sharpe untouched and shrinks
# total by k regardless of whether the regime read was right. Below this
# dispersion the comparison is reported as INCONCLUSIVE rather than scored.
# 0.02 is ~2% of the [0.4, 1.25] clamp width — negligible variation.
MIN_SCALAR_DISPERSION = 0.02


def apply_scaling(trades: List[dict], scalar_by_date: Dict[str, float]) -> List[dict]:
    """Copy of `trades` with `scaled_pnl` = pnl * the scalar for its open date.

    Trades whose open date has no regime row keep a scalar of 1.0 (unscaled).
    """
    scaled = []
    for trade in trades:
        day = str(trade.get("opened_at", ""))[:10]
        scalar = scalar_by_date.get(day, 1.0)
        row = dict(trade)
        row["scalar"] = scalar
        row["scaled_pnl"] = float(trade.get("pnl") or 0.0) * scalar
        scaled.append(row)
    return scaled


def metrics(pnls: List[float]) -> dict:
    """Total, per-trade sharpe (mean/std) and max drawdown of a PnL series."""
    if not pnls:
        return {"total": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    total = sum(pnls)
    mean = total / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / len(pnls)
    std = math.sqrt(variance)
    sharpe = (mean / std) if std > 0 else 0.0
    cumulative, peak, max_dd = 0.0, 0.0, 0.0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        max_dd = min(max_dd, cumulative - peak)
    return {"total": total, "sharpe": sharpe, "max_drawdown": max_dd}


def scalar_stats(scaled_trades: List[dict], scalar_by_date: Dict[str, float]) -> dict:
    """Dispersion + coverage of the scalars actually applied to the trades.

    `matched` counts trades whose open date had a regime row; `coverage` is that
    as a fraction. `stdev` is the population stdev over every applied scalar,
    defaults included. `protective_days` counts trades the layer actually
    de-risked (scalar < 1.0) — the half of the design a leverage-only window
    never exercises.
    """
    applied = [t["scalar"] for t in scaled_trades]
    if not applied:
        return {
            "matched": 0,
            "coverage": 0.0,
            "stdev": 0.0,
            "min": 1.0,
            "max": 1.0,
            "unique": 0,
            "protective_days": 0,
        }
    mean = sum(applied) / len(applied)
    stdev = math.sqrt(sum((s - mean) ** 2 for s in applied) / len(applied))
    matched = sum(1 for t in scaled_trades if str(t.get("opened_at", ""))[:10] in scalar_by_date)
    return {
        "matched": matched,
        "coverage": matched / len(scaled_trades),
        "stdev": stdev,
        "min": min(applied),
        "max": max(applied),
        "unique": len(set(applied)),
        "protective_days": sum(1 for s in applied if s < 1.0),
    }


def compare(trades: List[dict], scalar_by_date: Dict[str, float]) -> dict:
    """Baseline vs regime-scaled metrics, per-year breakdown, and a verdict.

    Verdict is INCONCLUSIVE when the window cannot test the overlay — either the
    applied scalar barely varies (see MIN_SCALAR_DISPERSION) or it never drops
    below 1.0, which leaves the protective half of the layer unexercised. It is
    HELPS when scaling improves sharpe and drawdown without giving up more than
    10% of total PnL, else NO. `reason` explains which branch fired.
    """
    scaled = apply_scaling(trades, scalar_by_date)
    baseline_m = metrics([float(t.get("pnl") or 0.0) for t in trades])
    scaled_m = metrics([t["scaled_pnl"] for t in scaled])
    stats = scalar_stats(scaled, scalar_by_date)

    by_year: Dict[str, dict] = {}
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for trade in scaled:
        buckets[str(trade.get("opened_at", ""))[:4]].append(trade)
    for year, year_trades in sorted(buckets.items()):
        by_year[year] = {
            "baseline": metrics([float(t.get("pnl") or 0.0) for t in year_trades]),
            "scaled": metrics([t["scaled_pnl"] for t in year_trades]),
        }

    helps = (
        scaled_m["sharpe"] >= baseline_m["sharpe"]
        and scaled_m["max_drawdown"] >= baseline_m["max_drawdown"]  # less negative
        and scaled_m["total"] >= 0.9 * baseline_m["total"]
    )
    if stats["stdev"] < MIN_SCALAR_DISPERSION:
        verdict, reason = (
            "INCONCLUSIVE",
            (
                "exposure_scalar barely varies over the trade window "
                "(stdev %.4f) — scaling cannot be scored on this data" % stats["stdev"]
            ),
        )
    elif stats["protective_days"] == 0 and trades:
        verdict, reason = (
            "INCONCLUSIVE",
            (
                "the scalar never drops below 1.0 in this window, so the layer only "
                "levered up and its protective half was never exercised — this "
                "window tests regime leverage, not regime protection"
            ),
        )
    elif helps:
        verdict, reason = "HELPS", "scaling improved sharpe and drawdown without giving up total"
    else:
        verdict, reason = "NO", "scaling did not improve risk-adjusted outcomes"

    return {
        "baseline": baseline_m,
        "scaled": scaled_m,
        "delta": {k: scaled_m[k] - baseline_m[k] for k in baseline_m},
        "by_year": by_year,
        "scalar_stats": stats,
        "verdict": verdict,
        "reason": reason,
    }


async def _load_closed_trades() -> List[dict]:
    import aiosqlite

    import database

    async with aiosqlite.connect(database.DB_PATH, timeout=database._DB_TIMEOUT) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT pnl, usd_open, opened_at FROM trades WHERE closed_at IS NOT NULL"
        )
        return [dict(r) for r in await cursor.fetchall()]


async def _main() -> None:
    import database

    trades = await _load_closed_trades()
    series = await database.get_regime_series("2000-01-01", "2100-01-01")
    result = compare(trades, {r["date"]: r["exposure_scalar"] for r in series})

    print(f"TRADES:   {len(trades)} closed")
    print("BASELINE:", result["baseline"])
    print("SCALED:  ", result["scaled"])
    print("DELTA:   ", result["delta"])
    print("SCALAR:  ", result["scalar_stats"])
    print("BY YEAR: ")
    for year, m in result["by_year"].items():
        print(
            f"  {year}: base_total={m['baseline']['total']:.1f} "
            f"scaled_total={m['scaled']['total']:.1f}"
        )
    print("VERDICT: ", result["verdict"])
    print("REASON:  ", result["reason"])


if __name__ == "__main__":
    asyncio.run(_main())
