"""Where published endpoints become usable consumer inputs. Contract §9.

Two quantities live here and they are NOT interchangeable (§9.1):

  * an ELIGIBILITY BOUNDARY is a position in a consumer's working frame, deciding when a
    slot reopens;
  * an ACCOUNTING TIME is a clock instant (`exit_observable_at`), deciding when realized
    PnL lands on an equity curve.

The single wall-clock `exit_ts` this replaces played both roles, which is why one defect
produced errors in opposite directions: an early exit released the slot late and realized
PnL late, while a gap realized PnL early and released the slot early, permitting two
positions in one leaf. A position is not a time.

The second thing this module exists to prevent is a consumer using endpoints nobody checked
against the frame. `ValidatedEndpoints` is gated on a module-private token, so it cannot be
constructed by accident with arbitrary records, and the boundary and accounting helpers
accept nothing else — "were these validated?" is answered by the type rather than by a
convention a caller can forget. The gate stops accidents and misplaced downstream trust; it
is not a security boundary against a caller who reads this file.

This module is also the single place where source-row identity becomes working-frame
position. Putting that conversion in either consumer would guarantee the other grew a
second copy, and two copies of one derivation drifting apart is the defect class this whole
effort has been unwinding.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

# Imported rather than reimplemented. Duplicating a validator is how the two halves of one
# rule drift apart, and this one already rejects coerced timestamps correctly.
from tools.strategy_discovery.labels import _validated_bar_starts

_REQUIRED_SIDECAR_FIELDS = (
    "manifest_digest",
    "data_id",
    "product_id",
    "horizons",
    "bar_duration_ms",
    "feature_recipe",
    "label_version",
    "cost_version",
    "exit_config",
    "record_digests",
)

_FRAME_COLUMNS = ("ts", "close", "high", "low", "atr14_pct", "source_row_id")

# A field separator for the frame fingerprint. Written as bytes([0]) rather than a
# \x00 literal because shell-mediated edits flattened that escape into a real NUL byte
# inside this file once already.
_SEP = bytes([0])


class MissingEndpoint(ValueError):
    """An artifact and a frame do not describe the same work, or a record is absent.

    Never a reason to fall back to horizon arithmetic: a silent fallback is what the
    endpoint contract removes, and it would reintroduce the disagreement while reporting
    success.
    """


# Held only by this module, and required to construct a ValidatedEndpoints. `init=False`
# on the flag below stops a caller CHOOSING False, but by itself it does not establish that
# validation ever ran -- so direct construction produced a trusted-looking object carrying
# arbitrary records, and every downstream isinstance check believed it. This token closes
# that. It guards against accident and against misplaced downstream trust; it is not a
# security boundary against a caller who reads this file.
_VALIDATED_BY_LOADER = object()


@dataclass(frozen=True)
class ValidatedEndpoints:
    """Records checked against the frame, the config and the independent candidate values.

    Construct only via `load_validated_endpoints`. `semantic_validation_performed` is
    `init=False` so a caller cannot claim otherwise, and `__post_init__` requires the
    module-private token so the flag cannot be true of records nobody validated.
    """

    records: tuple
    data_id: str
    bar_duration_ms: int
    exit_config: Mapping
    # Carried so a consumer can RECOMPUTE the frame's identity rather than trust that someone
    # validated it. Without this, a validated set from one frame pairs silently with another
    # that shares row ids but differs in the features a rule reads.
    feature_recipe: str
    # Digest of the FULL frame content this set was validated against, so a consumer can
    # confirm it holds THAT frame rather than another with the same row ids and clocks.
    frame_fingerprint: str
    token: Any = None
    semantic_validation_performed: bool = field(init=False, default=True)

    def __post_init__(self) -> None:
        if self.token is not _VALIDATED_BY_LOADER:
            raise TypeError(
                "ValidatedEndpoints must come from load_validated_endpoints; constructing "
                "one directly would mark unvalidated records as validated, and every "
                "downstream isinstance check would trust them"
            )
        # A frozen dataclass does not freeze a dict it holds. Mutating exit_config would
        # change the cap that every later record is validated against.
        object.__setattr__(self, "exit_config", MappingProxyType(dict(self.exit_config)))
        object.__setattr__(self, "token", None)


# ── identity: two id spaces, two rules (§9.3) ────────────────────────────────


def _validated_int_ids(values, *, field_name: str) -> np.ndarray:
    """int64 ids, VALIDATED rather than coerced.

    `to_numpy(dtype="int64")` and `int()` truncate silently. A truncated timestamp yields a
    bar start that does not describe the frame; a truncated row id is worse, because it
    still points at a real row -- just the wrong one.
    """
    # `np.asarray` on a mixed sequence coerces bools to ints BEFORE the check below can
    # see them, so `[False, 1]` validated cleanly as `[0, 1]`. An all-bool input WAS
    # rejected, which made the check look correct. Iterate the original objects instead.
    raw = list(values.to_numpy()) if hasattr(values, "to_numpy") else list(values)
    out = np.empty(len(raw), dtype="int64")
    for position, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{field_name} at position {position} must not be a bool")
        as_int = int(value)
        if as_int != value:
            raise ValueError(
                f"{field_name} at position {position} is not a whole number ({value!r}); "
                f"refusing to truncate a row reference"
            )
        if as_int < 0:
            raise ValueError(f"{field_name} at position {position} is negative ({as_int})")
        out[position] = as_int
    return out


def validated_source_ordinals(values) -> np.ndarray:
    """The UNFILTERED frame's ids: exactly `0..n-1`, no duplicates and no gaps.

    The producer writes `arange(n)`, so anything else means the frame was filtered,
    reordered or concatenated after labelling and its positions no longer mean what the
    endpoints reference. Uniqueness is checked HERE because the caller turns these into
    dictionary keys, where a duplicate silently overwrites rather than fails.
    """
    ids = _validated_int_ids(values, field_name="source_row_id")
    if not np.array_equal(ids, np.arange(len(ids), dtype="int64")):
        raise ValueError(
            "unfiltered source_row_id must be exactly 0..n-1 with no duplicates or gaps; "
            "anything else means the frame no longer matches the positions the endpoints "
            "reference"
        )
    return ids


def validated_retained_ids(values) -> np.ndarray:
    """A FILTERED frame's ids: strictly increasing and unique, gaps expected.

    Gaps are the whole purpose of a finite-label filter. Duplicates and reordering are not,
    and either would corrupt the position mapping silently.
    """
    ids = _validated_int_ids(values, field_name="source_row_id")
    if ids.size and (np.diff(ids) <= 0).any():
        raise ValueError("retained source_row_id must be strictly increasing and unique")
    return ids


def working_positions(values) -> dict:
    """source row id -> position in the working frame."""
    return {int(row_id): position for position, row_id in enumerate(validated_retained_ids(values))}


# ── candidate values: finite means isfinite, not "not null" (§9.3) ───────────


def finite_candidate_values(frame, *, product_id: str, horizons: Sequence[int]) -> dict:
    """Expected labels for every finite candidate, built from the FRAME.

    `isfinite`, not `notna`: `dropna` RETAINS +/-inf, so a consumer that retained rows with
    `notna` while building expectations here would disagree with itself -- an
    infinite-labelled row would be retained with no expectation and no endpoint, then fail
    as a spurious coverage error rather than as the data problem it is.

    Keys are built from the DECLARED product and horizons crossed with the frame's rows,
    never enumerated from records: deriving expected keys from the records under test makes
    a missing record undetectable, because removing one removes its own expectation.
    """
    ids = validated_retained_ids(frame["source_row_id"])
    values: dict = {}
    for horizon in horizons:
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise MissingEndpoint(f"declared horizon must be a non-bool int, got {horizon!r}")
        column = f"label_h{horizon}"
        if column not in frame.columns:
            raise MissingEndpoint(
                f"declared horizon {horizon} has no {column} in the frame; the frame and "
                f"the endpoints describe different work"
            )
        for row_id, value in zip(ids, frame[column].to_numpy(dtype="float64"), strict=True):
            if math.isfinite(value):
                values[(product_id, int(horizon), int(row_id))] = float(value)
    return values


# ── the adapter: load, verify identity, validate semantics ───────────────────


def _validated_sidecar(sidecar: Any) -> Mapping[str, Any]:
    if not isinstance(sidecar, Mapping):
        raise MissingEndpoint("sidecar must be a mapping")
    for name in _REQUIRED_SIDECAR_FIELDS:
        if name not in sidecar:
            raise MissingEndpoint(f"sidecar is missing required field {name!r}")
    return sidecar


def _validated_cap(exit_config: Any) -> int:
    """The CONFIGURED cap, which is never a horizon (§9.4).

    A record published at horizon 1 carries `max_hold_bars = 168` under the default
    configuration, because the simulation uses `min(horizon, cap)` internally while the
    record publishes the cap. Rebuilding this from `record.horizon` rejects every valid
    short-horizon record.
    """
    if not isinstance(exit_config, Mapping) or "max_hold_bars" not in exit_config:
        raise MissingEndpoint("sidecar exit_config must declare max_hold_bars")
    cap = exit_config["max_hold_bars"]
    # `True` passes isinstance(int) and `168.0` passes an equality check, so both are named.
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise MissingEndpoint(
            f"sidecar exit_config.max_hold_bars must be a positive non-bool int, got {cap!r}"
        )
    return cap


def load_validated_endpoints(
    directory, *, frame, sidecar: Mapping[str, Any], product_id: str
) -> ValidatedEndpoints:
    """Load a published dataset and validate every record against the frame. Or raise.

    The single path both consumers use. It establishes, in order:

    1. the frame's row identity is the one the endpoints reference (`0..n-1`);
    2. the frame IS the frame the endpoints describe, by RECOMPUTING `build_data_id` from
       the frame's own arrays and the declared `exit_config` -- quoting the sidecar's
       `data_id` into the loader would prove only that the manifest agrees with the
       sidecar, not that the frame matches (§9.2). Because `config_id == data_id` by
       construction, this one recompute verifies frame and config identity together;
    3. integrity, attribution and complete coverage, via `load_dataset` against
       expectations built independently from the frame;
    4. SEMANTICS, via `validate_endpoint` per record, with the clock map scoped to the two
       rows that record references and the cap taken from the declared config.

    Step 4 is what the integrity-only loader deliberately does not do. A verified digest
    says the record has not changed; it says nothing about whether its clocks agree with
    the frame.
    """
    from tools.strategy_discovery.endpoint_dataset import build_data_id, load_dataset
    from tools.strategy_discovery.endpoint_records import (
        ExpectedEndpointContext,
        validate_endpoint,
    )

    sidecar = _validated_sidecar(sidecar)
    cap = _validated_cap(sidecar["exit_config"])
    if not isinstance(product_id, str) or not product_id.strip():
        raise MissingEndpoint("product_id must be a non-empty str")
    for column in _FRAME_COLUMNS:
        if column not in frame.columns:
            raise MissingEndpoint(f"frame has no {column!r} column")

    ordinals = validated_source_ordinals(frame["source_row_id"])
    starts = _validated_bar_starts(frame["ts"])

    recomputed = build_data_id(
        product_id=product_id,
        bar_duration_ms=sidecar["bar_duration_ms"],
        timestamps=starts.tolist(),
        closes=frame["close"].to_numpy(dtype="float64").tolist(),
        highs=frame["high"].to_numpy(dtype="float64").tolist(),
        lows=frame["low"].to_numpy(dtype="float64").tolist(),
        atr_pcts=frame["atr14_pct"].to_numpy(dtype="float64").tolist(),
        feature_recipe=sidecar["feature_recipe"],
        config=dict(sidecar["exit_config"]),
    )
    if recomputed != sidecar["data_id"]:
        raise MissingEndpoint(
            f"data_id recomputed from the frame is {recomputed} but the sidecar declares "
            f"{sidecar['data_id']}; the frame is not the one these endpoints describe (or "
            f"the declared exit config is not the one that produced them -- config_id "
            f"equals data_id, so the id alone cannot say which)"
        )

    expected = finite_candidate_values(
        frame, product_id=product_id, horizons=list(sidecar["horizons"])
    )
    loaded = load_dataset(
        Path(directory),
        expected_manifest_digest=sidecar["manifest_digest"],
        expected_candidate_values=expected,
        expected_data_id=sidecar["data_id"],
        require_complete_coverage=True,
    )

    clock = {int(row_id): int(start) for row_id, start in zip(ordinals, starts, strict=True)}
    digests = sidecar["record_digests"]
    for record in loaded.records:
        key = f"{record.horizon}:{record.entry_row_id}"
        if key not in digests:
            raise MissingEndpoint(
                f"no stored digest for record {key}; recomputing it from the record under "
                f"test would attest nothing"
            )
        for row_id in (record.entry_row_id, record.exit_row_id):
            if row_id not in clock:
                raise MissingEndpoint(
                    f"record {key} references source row {row_id}, which the frame does not contain"
                )
        validate_endpoint(
            record,
            # Scoped to the two referenced rows: validate_endpoint type-validates every key
            # and value it is handed, so passing the whole frame per record is quadratic.
            source_bar_starts={
                record.entry_row_id: clock[record.entry_row_id],
                record.exit_row_id: clock[record.exit_row_id],
            },
            expected=ExpectedEndpointContext(
                product_id=product_id,
                horizon=record.horizon,
                data_id=sidecar["data_id"],
                label_version=sidecar["label_version"],
                cost_version=sidecar["cost_version"],
                # config_id == data_id by construction in the producer.
                config_id=sidecar["data_id"],
                label_value=expected[(product_id, record.horizon, record.entry_row_id)],
                bar_duration_ms=sidecar["bar_duration_ms"],
                # The CONFIG cap, never the horizon.
                max_hold_bars=cap,
                digest=digests[key],
            ),
        )

    return ValidatedEndpoints(
        records=tuple(loaded.records),
        data_id=sidecar["data_id"],
        bar_duration_ms=int(sidecar["bar_duration_ms"]),
        exit_config=dict(sidecar["exit_config"]),
        feature_recipe=sidecar["feature_recipe"],
        frame_fingerprint=frame_fingerprint(frame),
        token=_VALIDATED_BY_LOADER,
    )


# ── the two quantities, from validated records only ─────────────────────────


def frame_fingerprint(frame) -> str:
    """Digest of the frame's FULL content: every column and every value.

    Needed because `build_data_id` is NARROWER than it looks. It covers the arrays the exit
    simulation consumed -- ts, close, high, low, atr14_pct -- plus the config, and nothing
    else. It does NOT cover the feature columns a RULE reads, `source_row_id`, or the label
    columns, so two frames differing only in `price_over_ema20` share a `data_id`. A replay
    bound on `data_id` alone would happily select trades on one frame and realize the other's
    outcomes.

    This is an IN-PROCESS binding, not a persisted artifact digest: a `ValidatedEndpoints`
    lives in memory, so the fingerprint only has to be stable within one process, and the
    producer's `data_id` version is deliberately untouched.
    """
    import pandas as pd

    names = [str(column) for column in frame.columns]
    if len(set(names)) != len(names):
        raise ValueError(
            "frame has duplicate column names; a fingerprint cannot bind names to values"
        )

    # Each column's NAME is hashed together with ITS OWN values, in sorted name order. An
    # earlier version hashed sorted names and then all row values, which was wrong in BOTH
    # directions and both were reproduced: renaming two columns to swap their names without
    # moving any value produced an IDENTICAL fingerprint (the sorted header lost the
    # name-to-position binding while the row hashes followed physical order), and a harmless
    # column REORDER produced a different one. Binding per column fixes both: a swap changes
    # which values sit under a name, while a reorder does not.
    digest = hashlib.sha256()
    for name in sorted(names):
        column = frame[name]
        digest.update(name.encode("utf-8"))
        digest.update(_SEP)
        digest.update(str(column.dtype).encode("utf-8"))
        digest.update(_SEP)
        digest.update(pd.util.hash_pandas_object(column, index=True).to_numpy().tobytes())
        digest.update(_SEP)
    return "sha256:" + digest.hexdigest()


def verify_frame_matches(validated, frame, *, product_id: str) -> str:
    """Confirm these records were validated against THIS frame. Returns its `data_id`.

    Two checks, because one is not enough:

    1. the recomputed `build_data_id` must match -- that covers the arrays the exit simulation
       read, the declared duration, the recipe and the exit config; and
    2. the full-content `frame_fingerprint` must match -- because `data_id` does NOT cover the
       feature columns a rule reads, `source_row_id`, or the label columns, so a frame
       differing only in `price_over_ema20` passes check 1.

    A `ValidatedEndpoints` proves SOME frame was validated. Only these prove it was this one,
    and both inputs are ordinary public values, so no construction gate substitutes for them.
    Neither check can say WHICH field moved.
    """
    from tools.strategy_discovery.endpoint_dataset import build_data_id

    validated = _require_validated(validated)
    actual_fingerprint = frame_fingerprint(frame)
    if actual_fingerprint != validated.frame_fingerprint:
        raise MissingEndpoint(
            f"frame fingerprint {actual_fingerprint} does not match the {validated.frame_fingerprint} "
            f"these endpoints were validated against; the records were validated against a "
            f"different frame, so selecting trades on this one would realize the other's "
            f"outcomes (data_id alone would not catch a changed rule feature column)"
        )
    for column in _FRAME_COLUMNS:
        if column not in frame.columns:
            raise MissingEndpoint(f"frame has no {column!r} column")
    recomputed = build_data_id(
        product_id=product_id,
        bar_duration_ms=int(validated.bar_duration_ms),
        timestamps=_validated_bar_starts(frame["ts"]).tolist(),
        closes=frame["close"].to_numpy(dtype="float64").tolist(),
        highs=frame["high"].to_numpy(dtype="float64").tolist(),
        lows=frame["low"].to_numpy(dtype="float64").tolist(),
        atr_pcts=frame["atr14_pct"].to_numpy(dtype="float64").tolist(),
        feature_recipe=validated.feature_recipe,
        config=dict(validated.exit_config),
    )
    if recomputed != validated.data_id:
        raise MissingEndpoint(
            f"data_id recomputed from the supplied frame is {recomputed} but these endpoints "
            f"describe {validated.data_id}; the records were validated against a different "
            f"frame, so selecting trades on this one would realize the other's outcomes"
        )
    return recomputed


def _require_validated(value: Any) -> ValidatedEndpoints:
    """A consumer may not hand these helpers endpoints nobody checked.

    The type carries the guarantee. A raw list would look identical at the call site and
    silently skip every frame, config and candidate check.
    """
    if not isinstance(value, ValidatedEndpoints):
        raise TypeError(
            "expected ValidatedEndpoints from load_validated_endpoints, got "
            f"{type(value).__name__}; raw endpoints have not been checked against the "
            f"frame, the config or the candidate values"
        )
    return value


def _for_horizon(validated: ValidatedEndpoints, horizon: Any) -> dict:
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError("horizon must be a non-bool int")
    selected: dict = {}
    for record in validated.records:
        if record.horizon != horizon:
            continue
        key = int(record.entry_row_id)
        if key in selected:
            raise ValueError(f"two endpoints claim entry row {key} at horizon {horizon}")
        selected[key] = record
    return selected


def eligibility_boundaries(
    validated, retained_ids, *, horizon: int, require_all: bool = True
) -> np.ndarray:
    """Working-frame POSITION at which each retained candidate's slot reopens (§9.1).

    `len(retained_ids)` encodes terminal -- no retained row follows the exit -- the same
    convention `build_next_eligible` produces via `clamp_max(n)`, so `walk_and_sum` needs no
    change.

    Occupancy is half-open `[entry, exit)`: `searchsorted(..., side="left")` returns the
    position OF the exit row, which is itself eligible (§3a tie-order).
    """
    validated = _require_validated(validated)
    if require_all is not True and require_all is not False:
        raise ValueError("require_all must be an actual bool")
    ids = validated_retained_ids(retained_ids)
    selected = _for_horizon(validated, horizon)
    terminal = len(ids)
    out = np.full(terminal, terminal, dtype="int64")
    for position, row_id in enumerate(ids):
        record = selected.get(int(row_id))
        if record is None:
            if require_all:
                raise MissingEndpoint(
                    f"no endpoint for retained candidate source row {int(row_id)} at "
                    f"horizon {horizon}; a retained row has a finite label and a finite "
                    f"label produces an endpoint, so absence is a contradiction -- never a "
                    f"reason to fall back to a clock"
                )
            continue
        out[position] = int(np.searchsorted(ids, int(record.exit_row_id), side="left"))
    return out


def accounting_times(validated, *, horizon: int) -> dict:
    """entry source row id -> `exit_observable_at`. A clock INSTANT, never a position."""
    validated = _require_validated(validated)
    return {
        key: int(record.exit_observable_at)
        for key, record in _for_horizon(validated, horizon).items()
    }


__all__ = [
    "MissingEndpoint",
    "frame_fingerprint",
    "verify_frame_matches",
    "ValidatedEndpoints",
    "accounting_times",
    "eligibility_boundaries",
    "finite_candidate_values",
    "load_validated_endpoints",
    "validated_retained_ids",
    "validated_source_ordinals",
    "working_positions",
]
