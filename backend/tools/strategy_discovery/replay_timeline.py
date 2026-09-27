"""The replay's ordered instants. Contract §9.1 and §9.5. Deliberately no decision logic.

Entry decisions become available at a bar's CLOSE, because the features a rule reads are
close-derived. Exits land at `exit_observable_at`, also a bar close. The pre-integration loop
iterated raw bar STARTS and did closing, entry evaluation and occupancy sampling at each one,
which dated every entry a bar early and left an exit on the final bar unreachable -- that
bar's close is later than every decision instant, so a condition tested only at decision
instants never fired and the position's PnL silently vanished.

What this module does NOT do is emit per-position events. An earlier design did, and it would
have:

  * replaced the existing `-cumulative_profit_deflated` ranking with ALPHABETICAL PRODUCT
    ORDER under a shared cap, because opening events one at a time in name order makes the
    name the tiebreaker; and
  * realized endpoints that were never entered, because seeding exits from every candidate
    creates a close for a position that never opened.

So ranking, the shared cap and the per-product constraint all stay in `portfolio_sim`, which
processes each instant as a BATCH. A checkpoint here only schedules an inspection; whether
anything closes depends entirely on what is actually open.
"""

from __future__ import annotations

import numbers
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np


def _whole(value, field: str) -> int:
    """An integral timestamp, VALIDATED rather than coerced.

    The property wanted is "this IS an integer", which is `numbers.Integral` -- checked
    BEFORE any conversion. An earlier version did `int(value)` and compared the round trip,
    which is still coercion: it happened to catch fractions but accepted integral FLOATS, so
    `bar_duration_ms=1.0` sailed through as 1. It also rejected only Python `bool`, and
    `np.bool_` is not a `bool` subclass, so a numpy boolean converted happily to 1.

    numpy integer scalars ARE accepted, because frames hand them out and rejecting them
    would make this module unusable on real input. Integral means integral, whatever the
    container.

    Every rejection is a `ValueError`. Without the explicit check the failure type depended
    on the input -- `int(nan)` raises `ValueError`, `int(inf)` raises `OverflowError`, and a
    string or `None` raises `TypeError` -- so a caller could not catch one thing.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{field} must not be a bool, got {value!r}")
    if not isinstance(value, numbers.Integral):
        raise ValueError(
            f"{field} must be an integer without coercion, got {type(value).__name__} {value!r}"
        )
    return int(value)


def _validated_duration(bar_duration_ms) -> int:
    duration = _whole(bar_duration_ms, "bar_duration_ms")
    if duration <= 0:
        raise ValueError(f"bar_duration_ms must be positive, got {duration}")
    return duration


def _validated_starts(starts: Sequence, *, pid: str) -> list:
    out = []
    previous = None
    for start in starts:
        current = _whole(start, f"{pid} bar start")
        if previous is not None and current <= previous:
            raise ValueError(f"{pid} bar starts must be strictly increasing and unique")
        previous = current
        out.append(current)
    return out


def bar_availability_instants(starts: Sequence, row_ids: Sequence, *, bar_duration_ms: int) -> dict:
    """availability instant -> the SOURCE row that becomes decidable at it.

    The per-product lookup a consumer needs to pair an instant with its row, and to
    cross-check each record's `entry_available_at` against `row.ts + bar_duration_ms`.

    `row_ids` is REQUIRED rather than derived by enumeration. On a filtered frame a position
    is not a source row id, and an endpoint's `entry_row_id` is a source row id -- so
    enumerating here would silently pair an instant with the wrong row the moment a caller
    passed a filtered sequence. That is the same identity confusion `reset_index(drop=True)`
    creates downstream, and it is cheaper to make the caller state the ids than to document
    the hazard.
    """
    duration = _validated_duration(bar_duration_ms)
    validated_starts = _validated_starts(starts, pid="frame")
    if len(row_ids) != len(validated_starts):
        raise ValueError(
            f"row_ids has {len(row_ids)} entries for {len(validated_starts)} bar starts; "
            f"the two describe the same rows or neither can be trusted"
        )
    validated_ids = []
    previous = None
    for row_id in row_ids:
        current = _whole(row_id, "source row id")
        if current < 0:
            raise ValueError(f"source row id must be non-negative, got {current}")
        if previous is not None and current <= previous:
            raise ValueError("source row ids must be strictly increasing and unique")
        previous = current
        validated_ids.append(current)
    return {
        start + duration: row_id
        for start, row_id in zip(validated_starts, validated_ids, strict=True)
    }


def decision_instants(
    pid_bar_starts: Mapping[str, Sequence],
    *,
    bar_duration_ms: int,
    participating: Optional[Iterable[str]] = None,
) -> list:
    """Unique union of bar closes: the entry opportunities AND the sampling basis (§9.5).

    One entry per unique master availability timestamp, matching the convention the
    pre-integration loop used, so `pct_slots_full` and `mean_concurrent` keep their
    denominators.

    `participating` names the products the subset actually trades. Without it every supplied
    frame contributes, and an unrelated input would change the denominator. A named product
    missing from the grid raises rather than contributing nothing: silently understating the
    denominator gives no signal at all.
    """
    duration = _validated_duration(bar_duration_ms)
    if participating is None:
        selected = list(pid_bar_starts)
    else:
        selected = list(participating)
        missing = [pid for pid in selected if pid not in pid_bar_starts]
        if missing:
            raise ValueError(
                f"participating products have no bar grid: {sorted(missing)}; contributing "
                f"nothing silently would understate the occupancy denominator"
            )

    instants = set()
    for pid in selected:
        for start in _validated_starts(pid_bar_starts[pid], pid=pid):
            instants.add(start + duration)
    return sorted(instants)


def close_checkpoints(candidate_exit_instants: Iterable) -> list:
    """Unique instants at which an OPEN position may become due.

    These SCHEDULE AN INSPECTION and nothing else: they create no positions and realize no
    PnL. Only an actually-open position closes, and only at its own `exit_ts`, so a
    checkpoint belonging to a candidate that never fired is a harmless no-op -- which is
    exactly why the whole list can be precomputed.

    Duplicates collapse. Two positions due at the same instant are both closed by one
    inspection, because closing is driven by the open positions rather than by a count of
    events.
    """
    return sorted({_whole(instant, "exit instant") for instant in candidate_exit_instants})


def ordered_instants(decisions: Iterable, checkpoints: Iterable) -> list:
    """`(instant, is_decision)` ascending, one entry per unique instant.

    A checkpoint coinciding with a decision instant yields ONE entry, marked as a decision:
    at a single instant the replay closes, then opens, then samples once. Two entries would
    sample occupancy twice and move the metric denominator.

    BOTH inputs are validated before any set is built. An earlier version validated neither,
    so `ordered_instants([0.5], [True])` returned `[(0.5, True), (True, False)]` -- a fraction
    survived into the output, and the bool is worse than useless: `True == 1`, so it either
    vanishes into instant 1 or masquerades as it, and the collision is invisible afterwards.
    """
    decision_set = {_whole(instant, "decision instant") for instant in decisions}
    checkpoint_set = {_whole(instant, "checkpoint instant") for instant in checkpoints}
    return [(instant, instant in decision_set) for instant in sorted(decision_set | checkpoint_set)]


__all__ = [
    "bar_availability_instants",
    "close_checkpoints",
    "decision_instants",
    "ordered_instants",
]
