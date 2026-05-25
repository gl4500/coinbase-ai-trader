"""Phase 4 portfolio simulator — time-walk with concurrency cap.

Walks historical bars chronologically; at each bar, closes positions whose
horizon expired, evaluates which profiles fire on their pid, and enters
the highest-deflated-profit firing profiles up to the cap.

Per-pid cap of 1 carried from Phase 3. Exit PnL inherited from Phase 2
label_h{horizon} — no exit re-simulation.

Pure pandas + numpy. No I/O, no GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.strategy_discovery.profile_loader import LoadedProfile


@dataclass
class PortfolioMetrics:
    cumulative_profit_raw: float = 0.0
    cumulative_profit_deflated: float = 0.0
    max_dd: float = 0.0
    sortino: float = 0.0
    trade_count: int = 0
    pct_slots_full: float = 0.0
    mean_concurrent: float = 0.0


@dataclass
class TelemetryRow:
    ts: int
    equity: float
    n_open: int
    fired_profile_id: Optional[str] = None
    closed_profile_id: Optional[str] = None
    realized_pnl: Optional[float] = None


def parse_rule_path(rule_path: str) -> List[Tuple[str, str, float]]:
    """Parse 'feat_a > 1.02 AND feat_b <= 0.08' into [(feature, op, threshold), ...].

    Operators supported: >, <, >=, <=. The '(root)' or empty rule returns [] (always fires).
    """
    if rule_path.strip() in ("", "(root)"):
        return []
    conditions: List[Tuple[str, str, float]] = []
    for clause in rule_path.split(" AND "):
        clause = clause.strip()
        for op in (">=", "<=", ">", "<"):
            if f" {op} " in clause:
                feature, threshold_str = clause.split(f" {op} ", 1)
                conditions.append((feature.strip(), op, float(threshold_str.strip())))
                break
        else:
            raise ValueError(f"unparseable rule clause: {clause!r}")
    return conditions


def _rule_holds_at(conditions: List[Tuple[str, str, float]], row: pd.Series) -> bool:
    """Evaluate parsed conditions against a Phase 2 feature row."""
    if not conditions:
        return True
    for feature, op, threshold in conditions:
        if feature not in row.index:
            return False
        v = float(row[feature])
        if op == ">"  and not (v >  threshold): return False
        if op == ">=" and not (v >= threshold): return False
        if op == "<"  and not (v <  threshold): return False
        if op == "<=" and not (v <= threshold): return False
    return True


def _compute_max_dd(equity_series: List[float]) -> float:
    if not equity_series:
        return 0.0
    arr = np.asarray(equity_series, dtype="float64")
    running_max = np.maximum.accumulate(arr)
    drawdown = running_max - arr
    return float(drawdown.max())


def _compute_sortino(trade_pnls: List[float]) -> float:
    if not trade_pnls:
        return 0.0
    arr = np.asarray(trade_pnls, dtype="float64")
    mean = float(arr.mean())
    downside = arr[arr < 0]
    if downside.size == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(downside ** 2)))
    return mean / dd if dd > 0 else 0.0


def simulate_portfolio(
    subset: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
) -> Tuple[PortfolioMetrics, List[TelemetryRow]]:
    """Walk historical bars in the subset's union; enforce cap; return metrics + telemetry."""
    # Pre-parse rule paths for speed
    parsed_rules = {(p.pid, p.leaf_id): parse_rule_path(p.rule_path) for p in subset}
    label_cols = {p.profile_id: f"label_h{int(p.horizon)}" for p in subset}
    horizon_ms = {p.profile_id: int(p.horizon) * 3_600_000 for p in subset}

    # Build a master timestamp index across all pids in subset
    all_ts = set()
    for p in subset:
        f = pid_features.get(p.pid, pd.DataFrame())
        if not f.empty and "ts" in f.columns:
            all_ts.update(f["ts"].astype("int64").tolist())
    sorted_ts = sorted(all_ts)

    # Per-pid feature lookup by ts
    pid_ts_to_row: Dict[str, Dict[int, pd.Series]] = {}
    for pid, f in pid_features.items():
        if f.empty:
            continue
        pid_ts_to_row[pid] = {int(t): row for t, row in zip(f["ts"], (f.iloc[i] for i in range(len(f))))}

    open_positions: List[dict] = []   # {pid, profile_id, entry_ts, exit_ts, expected_pnl}
    trade_log: List[float] = []
    telemetry: List[TelemetryRow] = []
    equity = 0.0
    # Per-bar slot tracking: one entry per bar (ts) recording peak n_open for that bar
    bar_max_n_open: List[int] = []

    for ts in sorted_ts:
        # 1. Close positions whose exit_ts <= ts
        still_open: List[dict] = []
        closed_this_bar: List[dict] = []
        for p in open_positions:
            if p["exit_ts"] <= ts:
                closed_this_bar.append(p)
            else:
                still_open.append(p)
        for c in closed_this_bar:
            equity += c["expected_pnl"]
            trade_log.append(float(c["expected_pnl"]))
            telemetry.append(TelemetryRow(
                ts=ts, equity=equity, n_open=len(still_open),
                closed_profile_id=c["profile_id"], realized_pnl=float(c["expected_pnl"]),
            ))
        open_positions = still_open

        # 2. Evaluate firings (per-pid occupied set updated live during entries)
        occupied_pids = {p["pid"] for p in open_positions}
        firings: List[LoadedProfile] = []
        for profile in subset:
            if profile.pid in occupied_pids:
                continue
            row = pid_ts_to_row.get(profile.pid, {}).get(int(ts))
            if row is None:
                continue
            if _rule_holds_at(parsed_rules[(profile.pid, profile.leaf_id)], row):
                firings.append(profile)

        # 3. Enforce cap; tiebreaker = highest deflated profit
        available = cap - len(open_positions)
        if available <= 0:
            telemetry.append(TelemetryRow(ts=ts, equity=equity, n_open=len(open_positions)))
            bar_max_n_open.append(len(open_positions))
            continue
        firings.sort(key=lambda p: -p.cumulative_profit_deflated)
        for profile in firings[:available]:
            # Re-check per-pid max-1 since occupied_pids is updated live
            if profile.pid in occupied_pids:
                continue
            label_col = label_cols[profile.profile_id]
            row = pid_ts_to_row[profile.pid][int(ts)]
            if label_col not in row.index or pd.isna(row[label_col]):
                continue
            expected = float(row[label_col])
            open_positions.append({
                "pid": profile.pid,
                "profile_id": profile.profile_id,
                "entry_ts": int(ts),
                "exit_ts": int(ts) + horizon_ms[profile.profile_id],
                "expected_pnl": expected,
            })
            occupied_pids.add(profile.pid)
            telemetry.append(TelemetryRow(
                ts=ts, equity=equity, n_open=len(open_positions),
                fired_profile_id=profile.profile_id,
            ))
        # If no firings, emit a baseline telemetry row anyway
        if not firings:
            telemetry.append(TelemetryRow(ts=ts, equity=equity, n_open=len(open_positions)))
        bar_max_n_open.append(len(open_positions))

    equity_curve = [t.equity for t in telemetry]
    n_open_log = [t.n_open for t in telemetry]
    metrics = PortfolioMetrics(
        cumulative_profit_raw=equity,
        max_dd=_compute_max_dd(equity_curve),
        sortino=_compute_sortino(trade_log),
        trade_count=len(trade_log),
        pct_slots_full=float(sum(1 for n in bar_max_n_open if n >= cap) / max(len(bar_max_n_open), 1)),
        mean_concurrent=float(sum(bar_max_n_open) / max(len(bar_max_n_open), 1)),
    )
    return metrics, telemetry
