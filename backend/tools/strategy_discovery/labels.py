"""Dynamic-exit PnL labeling for the strategy-discovery rebuild (Phase 2).

For each candidate row (pid, t) and each horizon h, simulate the deployed
exit policy and report net PnL fraction after fees:

  - stop-loss at 8% drawdown vs entry (priority)
  - trail-stop at max(atr14_pct_t, 6%) drawdown vs running peak
  - max-hold cap at 168 bars (7d)
  - 1.2% retail round-trip fee subtracted from gross PnL

Mirrors agents/exit_watcher.on_price_tick and cnn_agent._check_risk_exits.
Pure functions. No I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_DEFAULT_HORIZONS: Tuple[int, ...] = (1, 4, 24, 72, 168)
_DEFAULT_STOP_LOSS_PCT = 0.08
_DEFAULT_ATR_TRAIL_FLOOR = 0.06
_DEFAULT_MAX_HOLD_BARS = 168
_DEFAULT_ROUND_TRIP_FEE = 0.012


@dataclass(frozen=True)
class _SimResult:
    """What one simulated trade produced.

    `exit_offset` and `exit_kind` are the facts `_simulate_one` used to compute `pnl`
    and then discarded, which is why three consumers each re-derived them from a
    different clock. Both readers -- the scalar labels and the endpoint records --
    take this one result, so they cannot diverge.

    An unavailable label (the horizon does not fit) carries `pnl = nan` and NO exit:
    `exit_offset is None`. It is never given a fabricated endpoint.
    """

    pnl: float
    exit_offset: Optional[int]
    exit_kind: Optional[str]

    @property
    def available(self) -> bool:
        """An exit alone is not availability: the PnL must also be finite.

        A NaN close produces a NaN PnL through the horizon branch while still
        reporting an exit offset, so checking the offset alone published an endpoint
        carrying label_value = NaN.
        """
        return self.exit_offset is not None and math.isfinite(self.pnl)


def _simulate_one(
    entry_idx: int,
    horizon: int,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_pcts: np.ndarray,
    stop_loss_pct: float,
    atr_trail_floor: float,
    max_hold_bars: int,
    round_trip_fee: float,
) -> _SimResult:
    """Simulate one (entry, horizon) trade.

    Returns the net PnL fraction together with the exit that produced it. The
    arithmetic is unchanged; only the discarded exit identity is now returned.
    """
    n = len(closes)
    entry_price = float(closes[entry_idx])
    horizon_cap = min(horizon, max_hold_bars)
    last_idx = entry_idx + horizon_cap
    if last_idx >= n:
        # Precheck preserved exactly: a row whose full horizon does not fit is
        # unavailable even when a stop would have fired on its first bar. Changing
        # this would change label values, which publication must not do.
        return _SimResult(float("nan"), None, None)
    peak = entry_price
    for s in range(1, horizon_cap + 1):
        i = entry_idx + s
        bar_low = float(lows[i])
        bar_high = float(highs[i])
        # 1. Stop-loss check (priority — matches cnn_agent._check_risk_exits)
        if bar_low / entry_price - 1.0 <= -stop_loss_pct:
            exit_price = entry_price * (1.0 - stop_loss_pct)
            return _SimResult((exit_price / entry_price - 1.0) - round_trip_fee, s, "stop")
        # 2. Trail-stop check (ATR-based with floor)
        if bar_high > peak:
            peak = bar_high
        atr_now = float(atr_pcts[i])
        if not np.isfinite(atr_now):
            atr_now = atr_trail_floor
        atr_pct = max(atr_now, atr_trail_floor)
        if bar_low / peak - 1.0 <= -atr_pct:
            exit_price = peak * (1.0 - atr_pct)
            return _SimResult((exit_price / entry_price - 1.0) - round_trip_fee, s, "trail")
    # 3. Horizon reached without trigger — exit at last bar's close
    exit_price = float(closes[last_idx])
    return _SimResult((exit_price / entry_price - 1.0) - round_trip_fee, horizon_cap, "horizon")


def simulate_dynamic_exit_labels(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    stop_loss_pct: float = _DEFAULT_STOP_LOSS_PCT,
    atr_trail_floor: float = _DEFAULT_ATR_TRAIL_FLOOR,
    max_hold_bars: int = _DEFAULT_MAX_HOLD_BARS,
    round_trip_fee: float = _DEFAULT_ROUND_TRIP_FEE,
) -> pd.DataFrame:
    """Add label_h{h} columns per horizon by simulating the deployed exit policy.

    Requires columns: ts, open, high, low, close, atr14_pct. NaN labels when
    horizon would extend past the end of df.
    """
    horizons_list = list(horizons) if horizons is not None else list(_DEFAULT_HORIZONS)
    out = df.copy()
    closes = out["close"].to_numpy(dtype="float64")
    highs = out["high"].to_numpy(dtype="float64")
    lows = out["low"].to_numpy(dtype="float64")
    atr_pcts = out["atr14_pct"].to_numpy(dtype="float64")
    n = len(out)
    for h in horizons_list:
        col = np.empty(n, dtype="float64")
        for i in range(n):
            col[i] = _simulate_one(
                i,
                h,
                closes,
                highs,
                lows,
                atr_pcts,
                stop_loss_pct,
                atr_trail_floor,
                max_hold_bars,
                round_trip_fee,
            ).pnl
        out[f"label_h{h}"] = col
    return out


_EXIT_BASIS = {
    "stop": "assumed_stop_level",
    "trail": "assumed_trail_level",
    "horizon": "bar_close",
}


def _validated_bar_starts(series) -> np.ndarray:
    """Bar starts as int64, validated rather than coerced.

    `to_numpy(dtype="int64")` truncates silently, so a fractional or boolean
    timestamp would be rewritten into something plausible and then published as the
    frame's identity. Chronology and uniqueness are checked here too: row ordinals
    and the clock must agree before either is used as an identity.
    """
    values = series.to_numpy()
    out = np.empty(len(values), dtype="int64")
    previous = None
    for position, value in enumerate(values):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"ts at row {position} must be an integer, not a bool")
        as_int = int(value)
        if as_int != value:
            raise ValueError(
                f"ts at row {position} is not a whole number ({value!r}); refusing to "
                f"truncate an input timestamp"
            )
        if previous is not None and as_int <= previous:
            raise ValueError(
                f"ts must be strictly increasing; row {position} ({as_int}) does not "
                f"follow {previous}"
            )
        previous = as_int
        out[position] = as_int
    return out


LABEL_VERSION = "label_endpoint_v1"
COST_VERSION = "round_trip_fee_v1"


def simulate_labels_with_endpoints(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    stop_loss_pct: float = _DEFAULT_STOP_LOSS_PCT,
    atr_trail_floor: float = _DEFAULT_ATR_TRAIL_FLOOR,
    max_hold_bars: int = _DEFAULT_MAX_HOLD_BARS,
    round_trip_fee: float = _DEFAULT_ROUND_TRIP_FEE,
    *,
    product_id: str = "UNKNOWN",
    bar_duration_ms: int = 3_600_000,
    feature_recipe: str = "atr14_pct_wilder_v1",
    with_dispositions: bool = False,
):
    """Labels plus the endpoint each label's exit produced.

    Returns `(labelled_df, endpoints)`, or `(labelled_df, endpoints, dispositions)`
    when `with_dispositions` is set. The labels are computed by the SAME shared
    result the endpoints come from, so the two cannot disagree.

    `bar_duration_ms` is **declared**, never inferred from row spacing: the next row
    may itself be missing, so spacing cannot establish a bar's duration.

    Publication only. It does not change any label value, does not address the
    contemporaneous-ATR causality defect (every record declares that blocker), and
    makes no claim about fills or intrabar timing.
    """
    from tools.strategy_discovery.endpoint_dataset import build_data_id
    from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER, LabelEndpoint

    if (
        not isinstance(product_id, str)
        or not product_id.strip()
        or product_id != product_id.strip()
        or product_id == "UNKNOWN"
    ):
        raise ValueError(
            "product_id must be a real product identity; a placeholder must never "
            "become part of a publishable artifact's identity"
        )
    if not isinstance(feature_recipe, str) or not feature_recipe.strip():
        raise ValueError("feature_recipe must name the recipe that produced atr14_pct")
    if type(bar_duration_ms) is not int or bar_duration_ms <= 0:
        raise ValueError("bar_duration_ms must be a positive int, declared not inferred")

    if type(max_hold_bars) is not int or max_hold_bars <= 0:
        raise ValueError("max_hold_bars must be a positive int without coercion")
    for name, value in (
        ("stop_loss_pct", stop_loss_pct),
        ("atr_trail_floor", atr_trail_floor),
        ("round_trip_fee", round_trip_fee),
    ):
        # Types before coercion: bool is an int subclass, so a True here would become
        # 1.0 silently and be bound into the artifact identity as a real parameter.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a real number, not {type(value).__name__}")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")

    horizons_list = list(horizons) if horizons is not None else list(_DEFAULT_HORIZONS)
    for horizon in horizons_list:
        if type(horizon) is not int or horizon <= 0:
            raise ValueError(
                f"each horizon must be a positive int without coercion, got "
                f"{horizon!r} ({type(horizon).__name__}); a bool would become 1"
            )
    labelled = df.copy()
    closes = labelled["close"].to_numpy(dtype="float64")
    highs = labelled["high"].to_numpy(dtype="float64")
    lows = labelled["low"].to_numpy(dtype="float64")
    atr_pcts = labelled["atr14_pct"].to_numpy(dtype="float64")
    # Timestamps are validated BEFORE conversion. to_numpy(dtype="int64") TRUNCATES,
    # so a fractional ts would publish a bar start that does not describe the frame
    # while every internal check still passed.
    starts = _validated_bar_starts(labelled["ts"])
    n = len(labelled)
    # The candidate frame carries its own row identity, so any later filtering has
    # something to preserve rather than having to reconstruct positions.
    labelled["source_row_id"] = np.arange(n, dtype="int64")

    config = {
        "stop_loss_pct": stop_loss_pct,
        "atr_trail_floor": atr_trail_floor,
        "max_hold_bars": max_hold_bars,
        "round_trip_fee": round_trip_fee,
    }
    data_id = build_data_id(
        product_id=product_id,
        bar_duration_ms=bar_duration_ms,
        timestamps=starts,
        closes=closes,
        highs=highs,
        lows=lows,
        atr_pcts=atr_pcts,
        feature_recipe=feature_recipe,
        config=config,
    )
    config_id = data_id  # the config is bound into data_id; kept explicit for readers

    endpoints: List[object] = []
    dispositions: Dict[str, int] = {
        "insufficient_horizon": 0,
        "nonfinite_label": 0,
    }
    for h in horizons_list:
        col = np.empty(n, dtype="float64")
        for i in range(n):
            result = _simulate_one(
                i,
                h,
                closes,
                highs,
                lows,
                atr_pcts,
                stop_loss_pct,
                atr_trail_floor,
                max_hold_bars,
                round_trip_fee,
            )
            col[i] = result.pnl
            if not result.available:
                # No endpoint is fabricated for an unavailable label; it is counted,
                # and the two reasons are counted SEPARATELY: a horizon that does not
                # fit is a boundary condition, while a non-finite PnL means the input
                # itself was unusable.
                reason = "insufficient_horizon" if result.exit_offset is None else "nonfinite_label"
                dispositions[reason] += 1
                continue
            exit_row = i + int(result.exit_offset)
            endpoints.append(
                LabelEndpoint(
                    product_id=product_id,
                    horizon=int(h),
                    data_id=data_id,
                    label_version=LABEL_VERSION,
                    cost_version=COST_VERSION,
                    config_id=config_id,
                    label_value=float(result.pnl),
                    entry_row_id=int(i),
                    exit_row_id=int(exit_row),
                    bars_held=int(result.exit_offset),
                    max_hold_bars=int(max_hold_bars),
                    entry_bar_start=int(starts[i]),
                    exit_bar_start=int(starts[exit_row]),
                    bar_duration_ms=int(bar_duration_ms),
                    entry_available_at=int(starts[i]) + int(bar_duration_ms),
                    exit_observable_at=int(starts[exit_row]) + int(bar_duration_ms),
                    exit_kind=result.exit_kind,
                    exit_price_basis=_EXIT_BASIS[result.exit_kind],
                    intrabar_timing_known=result.exit_kind == "horizon",
                    intrabar_order_assumption=(
                        "high_before_low" if result.exit_kind == "trail" else None
                    ),
                    blockers=(CAUSALITY_BLOCKER,),
                )
            )
        labelled[f"label_h{h}"] = col

    if with_dispositions:
        return labelled, endpoints, dispositions
    return labelled, endpoints
