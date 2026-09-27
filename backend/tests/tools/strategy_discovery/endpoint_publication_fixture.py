"""Publish a real endpoint dataset for a synthetic Phase 2 frame.

Shared because three test modules need it and a hand-built copy in each is how fixtures
drift apart. It goes through the producer's own `_publish_endpoints`, so the dataset, the
manifest and the sidecar are written by the code under test rather than imitated -- a
fixture that imitates the writer proves only that the imitation and the loader agree.

`exit_row_for` is the point of the helper: it lets a test publish endpoints whose exit is
EARLIER than `entry + horizon`, which is the case a wall-clock horizon gets wrong and the
one no default fixture produces.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

_BAR_MS = 3_600_000

DEFAULT_EXIT_CONFIG: Mapping[str, float] = {
    "stop_loss_pct": 0.08,
    "atr_trail_floor": 0.06,
    "max_hold_bars": 168,
    "round_trip_fee": 0.012,
}


def data_id_for_frame(
    pid: str,
    frame,
    *,
    exit_config: Mapping[str, float] = DEFAULT_EXIT_CONFIG,
    bar_ms: int = _BAR_MS,
) -> str:
    """The `data_id` the producer would compute for this frame, recomputed not quoted."""
    from tools.strategy_discovery.endpoint_dataset import build_data_id

    return build_data_id(
        product_id=pid,
        bar_duration_ms=bar_ms,
        timestamps=frame["ts"].tolist(),
        closes=frame["close"].tolist(),
        highs=frame["high"].tolist(),
        lows=frame["low"].tolist(),
        atr_pcts=frame["atr14_pct"].tolist(),
        feature_recipe="atr14_pct_wilder_v1",
        config=exit_config,
    )


def publish_endpoints_for_frame(
    output_dir,
    pid: str,
    frame,
    *,
    horizons: Sequence[int],
    exit_config: Mapping[str, float] = DEFAULT_EXIT_CONFIG,
    bar_ms: int = _BAR_MS,
    exit_row_for: Optional[Callable[[int, int], int]] = None,
) -> Optional[str]:
    """Write `{pid}.endpoints.json` + `endpoints/{pid}/` for every finite-label row.

    Retention is `math.isfinite`, NOT `notna`: `dropna` keeps +/-inf, so a notna-built
    expectation set and an isfinite-built one disagree on an inf-labelled row, and the
    disagreement surfaces later as a spurious missing endpoint.

    `exit_row_for(row, horizon)` overrides the exit row; the default is `row + horizon`,
    which is the horizon exit. Any earlier row is published as a `stop`, because a record
    claiming `horizon` while exiting early would be internally inconsistent and real
    validation rejects it.
    """
    from tools.strategy_discovery.build_phase2 import _publish_endpoints
    from tools.strategy_discovery.endpoint_records import (
        _EXIT_BASIS,
        CAUSALITY_BLOCKER,
        LabelEndpoint,
    )
    from tools.strategy_discovery.labels import COST_VERSION, LABEL_VERSION

    starts = [int(value) for value in frame["ts"].tolist()]
    row_ids = [int(value) for value in frame["source_row_id"].tolist()]
    data_id = data_id_for_frame(pid, frame, exit_config=exit_config, bar_ms=bar_ms)
    cap = int(exit_config["max_hold_bars"])

    records = []
    for horizon in horizons:
        column = f"label_h{int(horizon)}"
        if column not in frame.columns:
            continue
        values = frame[column].tolist()
        for row_id, value in zip(row_ids, values, strict=True):
            if not math.isfinite(float(value)):
                continue
            exit_row = (
                int(row_id) + int(horizon)
                if exit_row_for is None
                else int(exit_row_for(int(row_id), int(horizon)))
            )
            if exit_row <= int(row_id) or exit_row >= len(starts):
                raise AssertionError(
                    f"fixture would publish an impossible record: entry {row_id} exit "
                    f"{exit_row} in a frame of {len(starts)} rows. A retained row must have "
                    f"a real later exit row, or the fixture is testing nothing."
                )
            kind = "horizon" if exit_row == int(row_id) + int(horizon) else "stop"
            records.append(
                LabelEndpoint(
                    product_id=pid,
                    horizon=int(horizon),
                    data_id=data_id,
                    label_version=LABEL_VERSION,
                    cost_version=COST_VERSION,
                    config_id=data_id,
                    label_value=float(value),
                    entry_row_id=int(row_id),
                    exit_row_id=exit_row,
                    bars_held=exit_row - int(row_id),
                    max_hold_bars=cap,
                    entry_bar_start=starts[int(row_id)],
                    exit_bar_start=starts[exit_row],
                    bar_duration_ms=bar_ms,
                    entry_available_at=starts[int(row_id)] + bar_ms,
                    exit_observable_at=starts[exit_row] + bar_ms,
                    exit_kind=kind,
                    # Real validation ties these together and rejected the first version of
                    # this fixture: a horizon exit lands on the bar close so its within-bar
                    # instant is known, while a stop's is an assumed level that OHLC cannot
                    # time. Getting this wrong is not cosmetic -- it is the difference between
                    # a record that claims knowledge it does not have and one that does not.
                    exit_price_basis=_EXIT_BASIS[kind],
                    intrabar_timing_known=(kind == "horizon"),
                    intrabar_order_assumption=None,
                    blockers=(CAUSALITY_BLOCKER,),
                )
            )
    if not records:
        raise AssertionError(
            "fixture published no endpoints, so any test using it would assert against an "
            "empty universe -- the exact vacuity that made two Phase 4 tests worthless"
        )
    return _publish_endpoints(output_dir, pid, records, list(horizons))
