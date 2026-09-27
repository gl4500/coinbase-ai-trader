"""Phase 4 portfolio simulator — time-walk with concurrency cap.

Walks historical bars chronologically; at each bar, closes positions whose
horizon expired, evaluates which profiles fire on their pid, and enters
the highest-deflated-profit firing profiles up to the cap.

Per-pid cap of 1 carried from Phase 3. Exit PnL inherited from Phase 2
label_h{horizon} — no exit re-simulation.

Pure pandas + numpy. No I/O, no GPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tools.strategy_discovery.endpoint_consumers import (
    MissingEndpoint,
    ValidatedEndpoints,
    accounting_times,
    verify_frame_matches,
)
from tools.strategy_discovery.profile_loader import LoadedProfile
from tools.strategy_discovery.replay_timeline import (
    bar_availability_instants,
    close_checkpoints,
    decision_instants,
    ordered_instants,
)
from tools.strategy_discovery.rule_contract import rule_matches, validate_rule


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
    dd = float(np.sqrt(np.mean(downside**2)))
    return mean / dd if dd > 0 else 0.0


def simulate_portfolio(
    subset: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
    *,
    endpoints_by_pid: Dict[str, ValidatedEndpoints],
    bar_duration_ms: int,
) -> Tuple[PortfolioMetrics, List[TelemetryRow]]:
    """Replay the subset on ENDPOINT instants; enforce cap; return metrics + telemetry.

    Both new keywords are REQUIRED, never defaulted: a caller that omitted one would silently
    keep the wall-clock behaviour this replaces.

    Entries are decided at a bar's CLOSE, because the features a rule reads are close-derived
    -- the previous loop evaluated rules at bar STARTS, dating every entry one bar early. Exits
    are realized at the endpoint's `exit_observable_at`, not at `entry + horizon * 1h`, and the
    cap slot is released then. Contract §9.1: an eligibility boundary is a POSITION and an
    accounting time is an INSTANT; the single wall-clock `exit_ts` this replaces played both
    roles, which is why one defect produced errors in opposite directions.

    Each instant is processed as a BATCH -- close every due position, collect ALL eligible
    firings, then rank and cap them together. Opening one position at a time in product order
    would replace the `-cumulative_profit_deflated` ranking with alphabetical priority.
    """
    identities = [(profile.pid, profile.horizon, profile.leaf_id) for profile in subset]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate research profile identity in simulation subset")
    # Display summaries are never interpreted as executable rules.
    machine_rules = {}
    for profile in subset:
        validate_rule(profile.machine_rule)
        machine_rules[profile.profile_id] = profile.machine_rule
        frame = pid_features.get(profile.pid)
        if frame is not None and not frame.empty:
            # Legacy labels advance by rows; replay advances by hours. A gap can
            # realize a label before its exit and release its occupied slot early.
            if (
                "ts" not in frame
                or not pd.api.types.is_integer_dtype(frame["ts"].dtype)
                or frame["ts"].isna().any()
                or frame["ts"].duplicated().any()
                or (frame["ts"].sort_values().diff().dropna() != 3_600_000).any()
            ):
                raise ValueError(
                    "timestamps must be unique and contiguous hourly; "
                    "label exit provenance required for gaps"
                )
        if frame is not None:
            required = {clause["feature"] for clause in profile.machine_rule["conditions"]}
            missing = required.difference(frame.columns)
            if missing:
                raise ValueError(
                    f"missing feature columns for {profile.profile_id}: {sorted(missing)}"
                )
    # Accounting times per profile, from VALIDATED records only. A raw list would skip every
    # frame, config and candidate check the adapter performs.
    accounting: Dict[str, Dict[int, int]] = {}
    entry_instants: Dict[str, Dict[int, int]] = {}
    expected_pnl: Dict[str, Dict[int, float]] = {}
    for profile in subset:
        if profile.pid not in endpoints_by_pid:
            raise MissingEndpoint(
                f"no endpoints supplied for {profile.pid}; the replay will not invent exits "
                f"from horizon arithmetic"
            )
        validated = endpoints_by_pid[profile.pid]
        if not isinstance(validated, ValidatedEndpoints):
            raise TypeError(
                "expected ValidatedEndpoints from load_validated_endpoints for "
                f"{profile.pid}, got {type(validated).__name__}; raw endpoints have not been "
                f"checked against the frame, the config or the candidate values"
            )
        if int(validated.bar_duration_ms) != int(bar_duration_ms):
            raise MissingEndpoint(
                f"{profile.pid} endpoints declare bar_duration_ms "
                f"{validated.bar_duration_ms} but the replay was asked for {bar_duration_ms}; "
                f"the two describe different bars"
            )
        horizon = int(profile.horizon)
        accounting[profile.profile_id] = accounting_times(validated, horizon=horizon)
        # The record's own claim about when its entry became decidable, kept so it can be
        # cross-checked against the frame's bar grid rather than trusted.
        entry_instants[profile.profile_id] = {
            int(record.entry_row_id): int(record.entry_available_at)
            for record in validated.records
            if record.horizon == horizon
        }
        # The RECORD is the single source of the expected PnL. Reading it from the frame's
        # label column instead left the PnL and the exit coming from different places with
        # nothing rebinding them, so a validated record set paired with a changed frame scored
        # the wrong number. The adapter already validated each label_value against the frame's
        # independent candidate values, and it rejects non-finite labels -- which also closes
        # the pd.isna hole, since isna accepts +/-inf.
        expected_pnl[profile.profile_id] = {
            int(record.entry_row_id): float(record.label_value)
            for record in validated.records
            if record.horizon == horizon
        }

        # These records must describe THIS frame. A validated set proves some frame was
        # checked; only recomputing the identity proves it was this one, and a mismatched pair
        # would select trades on one frame and realize the other's outcomes.
        frame_for_binding = pid_features.get(profile.pid)
        if frame_for_binding is not None and not frame_for_binding.empty:
            verify_frame_matches(validated, frame_for_binding, product_id=profile.pid)

        # ARTIFACT-level coverage, checked ONCE here rather than row by row inside the loop.
        # Every finite-label row in the frame must have a record: the frame and the endpoints
        # were produced together, so a gap means they are not a matched pair, and no per-row
        # accounting can repair that (contract section 9.3). `isfinite`, not `notna`, because
        # notna accepts +/-inf.
        frame = pid_features.get(profile.pid)
        if frame is not None and not frame.empty:
            column = f"label_h{horizon}"
            if column not in frame.columns:
                raise MissingEndpoint(
                    f"{profile.pid} frame has no {column} for a horizon-{horizon} profile"
                )
            if "source_row_id" not in frame.columns:
                raise MissingEndpoint(
                    f"{profile.pid} frame has no source_row_id column; endpoint row identity "
                    f"cannot be resolved without it"
                )
            uncovered = [
                int(row_id)
                for row_id, value in zip(
                    frame["source_row_id"].tolist(),
                    frame[column].to_numpy(dtype="float64"),
                    strict=True,
                )
                if math.isfinite(value) and int(row_id) not in expected_pnl[profile.profile_id]
            ]
            if uncovered:
                raise MissingEndpoint(
                    f"{profile.profile_id} has {len(uncovered)} finite-label rows with no "
                    f"endpoint, first source row {uncovered[0]}; the frame and the endpoints "
                    f"are not a matched pair"
                )

    # PARTICIPATING products only: the subset's own pids. Including every supplied frame would
    # let an unrelated input change the occupancy denominator.
    participating = sorted(
        {p.pid for p in subset if not pid_features.get(p.pid, pd.DataFrame()).empty}
    )
    # `.tolist()` without astype: the timeline validates these strictly, and coercing here
    # first would destroy the evidence it checks for.
    pid_bar_starts = {pid: pid_features[pid]["ts"].tolist() for pid in participating}
    decisions = decision_instants(
        pid_bar_starts, bar_duration_ms=bar_duration_ms, participating=participating
    )

    # Every candidate accounting time becomes an inspection instant, so an exit later than the
    # last decision instant -- which is every exit on a final bar -- is still examined. These
    # create nothing and realize nothing on their own.
    checkpoints = close_checkpoints(
        instant for table in accounting.values() for instant in table.values()
    )

    # Rows are keyed by AVAILABILITY instant, not bar start: that is when the row's rule may
    # fire. Source row identity is passed explicitly rather than enumerated.
    pid_instant_to_row: Dict[str, Dict[int, pd.Series]] = {}
    pid_instant_to_source: Dict[str, Dict[int, int]] = {}
    for pid in participating:
        frame = pid_features[pid]
        if "source_row_id" not in frame.columns:
            raise MissingEndpoint(
                f"{pid} frame has no source_row_id column; endpoint row identity cannot be "
                f"resolved without it"
            )
        source_ids = frame["source_row_id"].tolist()
        instant_to_source = bar_availability_instants(
            frame["ts"].tolist(), source_ids, bar_duration_ms=bar_duration_ms
        )
        pid_instant_to_source[pid] = instant_to_source
        by_position = {int(row_id): frame.iloc[i] for i, row_id in enumerate(source_ids)}
        pid_instant_to_row[pid] = {
            instant: by_position[row_id] for instant, row_id in instant_to_source.items()
        }

    open_positions: List[dict] = []  # {pid, profile_id, entry_ts, exit_ts, expected_pnl}
    trade_log: List[float] = []
    telemetry: List[TelemetryRow] = []
    equity = 0.0
    # Per-bar slot tracking: one entry per bar (ts) recording peak n_open for that bar
    bar_max_n_open: List[int] = []

    for ts, is_decision in ordered_instants(decisions, checkpoints):
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
            telemetry.append(
                TelemetryRow(
                    ts=ts,
                    equity=equity,
                    n_open=len(still_open),
                    closed_profile_id=c["profile_id"],
                    realized_pnl=float(c["expected_pnl"]),
                )
            )
        open_positions = still_open

        # A checkpoint-only instant exists solely so a due position can be examined. Nothing
        # opens there and nothing is sampled, or the metric denominator would move.
        if not is_decision:
            continue

        # 2. Evaluate firings (per-pid occupied set updated live during entries)
        occupied_pids = {p["pid"] for p in open_positions}
        firings: List[LoadedProfile] = []
        for profile in subset:
            if profile.pid in occupied_pids:
                continue
            row = pid_instant_to_row.get(profile.pid, {}).get(int(ts))
            if row is None:
                continue
            if rule_matches(machine_rules[profile.profile_id], row):
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
            source_row_id = pid_instant_to_source[profile.pid][int(ts)]
            # A row is a candidate iff a RECORD exists for it. The frame's label column is no
            # longer consulted: it was the second, unvalidated copy of the same fact.
            expected = expected_pnl[profile.profile_id].get(source_row_id)
            if expected is None:
                continue
            # The record must agree with the frame about WHEN this row became decidable. A
            # disagreement means the endpoints and the frame describe different bars, and this
            # is the cheapest place to find that out. It is a necessary check, not a sufficient
            # one: full binding of a record set to a frame is the adapter's job, via the
            # recomputed data_id -- see load_validated_endpoints.
            claimed = entry_instants[profile.profile_id].get(source_row_id)
            if claimed is not None and claimed != int(ts):
                raise MissingEndpoint(
                    f"{profile.profile_id} row {source_row_id} declares entry_available_at "
                    f"{claimed} but the frame makes it decidable at {int(ts)}; the frame and "
                    f"the endpoints describe different bars"
                )
            exit_at = accounting[profile.profile_id].get(source_row_id)
            if exit_at is None:
                raise MissingEndpoint(
                    f"no endpoint for {profile.profile_id} at source row {source_row_id}; "
                    f"refusing to invent an exit time from horizon arithmetic"
                )
            open_positions.append(
                {
                    "pid": profile.pid,
                    "profile_id": profile.profile_id,
                    "entry_ts": int(ts),
                    # An ACCOUNTING TIME read from the endpoint, never re-derived.
                    "exit_ts": exit_at,
                    "expected_pnl": expected,
                }
            )
            occupied_pids.add(profile.pid)
            telemetry.append(
                TelemetryRow(
                    ts=ts,
                    equity=equity,
                    n_open=len(open_positions),
                    fired_profile_id=profile.profile_id,
                )
            )
        # If no firings, emit a baseline telemetry row anyway
        if not firings:
            telemetry.append(TelemetryRow(ts=ts, equity=equity, n_open=len(open_positions)))
        bar_max_n_open.append(len(open_positions))

    equity_curve = [t.equity for t in telemetry]
    [t.n_open for t in telemetry]
    metrics = PortfolioMetrics(
        cumulative_profit_raw=equity,
        max_dd=_compute_max_dd(equity_curve),
        sortino=_compute_sortino(trade_log),
        trade_count=len(trade_log),
        pct_slots_full=float(
            sum(1 for n in bar_max_n_open if n >= cap) / max(len(bar_max_n_open), 1)
        ),
        mean_concurrent=float(sum(bar_max_n_open) / max(len(bar_max_n_open), 1)),
    )
    return metrics, telemetry
