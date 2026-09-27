"""Pure validator for shared label-endpoint records.

Contract: `docs/specs/2026-09-26-label-endpoint-contract.md`.

Three components used to derive *when a labelled trade ends* independently — mining
eligibility and portfolio replay from wall-clock, the label itself from a row offset
or an earlier triggered exit — and on gapped data they disagreed. The endpoint record
exists so the simulation PUBLISHES that fact and the others CONSUME it.

This module is pure: values in, validation out. It performs no I/O, wires no
consumer, reads no config, and changes no simulation semantics.

Three things it deliberately does not establish:

* **Not a fill.** `intrabar_timing_known` describes simulated within-bar timing
  under the declared bar model. For `stop` and `trail` the instant within the bar is
  unknowable from OHLC, so it is `False`; for `horizon` the exit is the bar's close,
  so it is `True`. Neither value is a claim about execution, and `True` never means a
  fill was observed.
* **Not causal.** The trail threshold reads the *current* bar's ATR, so a label can
  depend on information from after its own decision point. Every record of this
  simulation version therefore carries `CAUSALITY_BLOCKER`, which this validator
  requires and can never clear.
* **Not an artifact match by self-description.** A record agrees with itself by
  construction, so a self-declared `config_id` cannot attest its own embedded cap and
  a freshly recomputed self-hash proves nothing. Every binding is checked against an
  independently supplied `ExpectedEndpointContext`.

Row ids are **positional ordinals** into the original frame, which is why `bars_held`
is their difference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

CAUSALITY_BLOCKER = "label_atr_contemporaneous_causality"

_EXIT_BASIS = {
    "horizon": "bar_close",
    "stop": "assumed_stop_level",
    "trail": "assumed_trail_level",
}
# Only a trail exit rests on an unevidenced within-bar ordering, so only it may
# declare one.
_ORDER_ASSUMPTIONS = {"high_before_low"}


class _Terminal:
    """Sentinel: no retained row is eligible at or after the endpoint."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "TERMINAL_SENTINEL"


TERMINAL_SENTINEL = _Terminal()


@dataclass(frozen=True)
class LabelEndpoint:
    """One simulated trade's endpoint, as published by the label simulation."""

    product_id: str
    horizon: int
    data_id: str
    label_version: str
    cost_version: str
    config_id: str
    label_value: float
    entry_row_id: int
    exit_row_id: int
    bars_held: int
    max_hold_bars: int
    entry_bar_start: int
    exit_bar_start: int
    bar_duration_ms: int
    entry_available_at: int
    exit_observable_at: int
    exit_kind: str
    exit_price_basis: str
    intrabar_timing_known: bool
    intrabar_order_assumption: Optional[str]
    blockers: tuple


@dataclass(frozen=True)
class ExpectedEndpointContext:
    """What the caller independently expects, from the artifact rather than the record.

    Product, horizon and `data_id` alone cannot detect a swapped cached `label_value`
    or an altered cap or duration, because the record remains self-consistent. So the
    expected context carries every binding, plus a `digest` that must have been stored
    alongside the artifact rather than recomputed from the record under test.
    """

    product_id: str
    horizon: int
    data_id: str
    label_version: str
    cost_version: str
    config_id: str
    label_value: float
    bar_duration_ms: int
    max_hold_bars: int
    digest: str


def _int(value: Any, field: str, *, minimum: Optional[int] = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an int without coercion, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}, got {value}")
    return value


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be a nonempty unpadded string")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite real number, got {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return float(value)


def endpoint_digest(record: LabelEndpoint) -> str:
    """Deterministic content digest over every field of the record.

    Integrity only: it detects alteration of the record. It attests nothing about
    whether the simulation that produced it was correct, and recomputing it here
    proves nothing unless compared against a digest stored with the artifact.
    """
    if not isinstance(record, LabelEndpoint):
        raise ValueError("record must be a LabelEndpoint")
    payload = asdict(record)
    payload["blockers"] = list(payload["blockers"])
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_blockers(blockers: Any) -> None:
    if not isinstance(blockers, tuple):
        raise ValueError(
            "blockers must be a tuple of names; a mutable sequence could "
            "have its mandatory blocker removed after validation"
        )
    names = []
    for name in blockers:
        names.append(_name(name, "blocker name"))
    if len(set(names)) != len(names):
        raise ValueError("blocker names must be unique")
    if CAUSALITY_BLOCKER not in names:
        raise ValueError(
            f"every record of this simulation version must carry the {CAUSALITY_BLOCKER} "
            f"blocker: the trail threshold reads the current bar's ATR, so labels can "
            f"depend on information from after their own decision point"
        )


def _validate_exit_semantics(record: LabelEndpoint) -> None:
    kind = record.exit_kind
    if not isinstance(kind, str) or kind not in _EXIT_BASIS:
        raise ValueError(f"exit_kind must be one of {sorted(_EXIT_BASIS)}, got {kind!r}")
    if record.exit_price_basis != _EXIT_BASIS[kind]:
        raise ValueError(
            f"exit_price_basis {record.exit_price_basis!r} does not match exit_kind "
            f"{kind!r}; expected {_EXIT_BASIS[kind]!r}"
        )
    if type(record.intrabar_timing_known) is not bool:
        raise ValueError("intrabar_timing_known must be an actual bool")
    # A horizon exit lands on the bar's close, so its within-bar instant is known
    # under the declared model. A triggered exit's is not knowable from OHLC.
    expected_known = kind == "horizon"
    if record.intrabar_timing_known is not expected_known:
        raise ValueError(
            f"intrabar_timing_known must be {expected_known} for a {kind!r} exit; an OHLC "
            f"bar records four prices and no ordering, and this field never asserts a fill"
        )
    if kind == "trail":
        # Type first: an unhashable value would raise TypeError from set membership,
        # escaping this module's promise to signal every rejection as ValueError.
        if not isinstance(record.intrabar_order_assumption, str):
            raise ValueError(
                f"intrabar_order_assumption must be a string naming the assumption, got "
                f"{type(record.intrabar_order_assumption).__name__}"
            )
        if record.intrabar_order_assumption not in _ORDER_ASSUMPTIONS:
            raise ValueError(
                "a trail exit must record its intrabar_order_assumption; the trail "
                "raises the peak from this bar's high and then compares it against the "
                "same bar's low, which assumes an ordering the data does not establish"
            )
    elif record.intrabar_order_assumption is not None:
        raise ValueError(
            f"intrabar_order_assumption must be None for a {kind!r} exit, which rests on "
            f"no within-bar ordering"
        )


def _validate_rows_and_clocks(record: LabelEndpoint, source_bar_starts: Mapping[int, int]) -> None:
    _int(record.entry_row_id, "entry_row_id", minimum=0)
    _int(record.exit_row_id, "exit_row_id", minimum=0)
    _int(record.bars_held, "bars_held")
    _int(record.max_hold_bars, "max_hold_bars", minimum=1)
    _int(record.bar_duration_ms, "bar_duration_ms", minimum=1)
    _int(record.entry_bar_start, "entry_bar_start")
    _int(record.exit_bar_start, "exit_bar_start")
    _int(record.entry_available_at, "entry_available_at")
    _int(record.exit_observable_at, "exit_observable_at")

    if record.exit_row_id <= record.entry_row_id:
        raise ValueError(
            f"exit_row_id {record.exit_row_id} must be strictly after entry_row_id "
            f"{record.entry_row_id}; a zero-duration endpoint is not a trade"
        )

    if not isinstance(source_bar_starts, Mapping):
        raise ValueError("source_bar_starts must be a mapping of row_id to bar start")
    # The mapping's OWN types are validated rather than trusted. Membership and
    # equality are not type checks: `False` aliases the dict key `0`, and `0.0`
    # compares equal to `0`, so a malformed source would resolve silently. Validating
    # the record strictly while trusting its source is the same asymmetry as checking
    # a schema and merely asserting a dtype.
    for key, value in source_bar_starts.items():
        _int(key, "source row_id key", minimum=0)
        _int(value, "source bar start")
    for field, row_id in (
        ("entry_row_id", record.entry_row_id),
        ("exit_row_id", record.exit_row_id),
    ):
        if row_id not in source_bar_starts:
            raise ValueError(f"{field} {row_id} does not resolve in the declared source frame")

    entry_src = source_bar_starts[record.entry_row_id]
    exit_src = source_bar_starts[record.exit_row_id]
    if entry_src >= exit_src:
        raise ValueError(
            "source chronology disagrees with the row ordinals: the entry bar must "
            "start before the exit bar"
        )
    if record.entry_bar_start != entry_src:
        raise ValueError(
            f"entry_bar_start {record.entry_bar_start} does not equal the source bar "
            f"start {entry_src}; a plausible timestamp is not a correct one"
        )
    if record.exit_bar_start != exit_src:
        raise ValueError(
            f"exit_bar_start {record.exit_bar_start} does not equal the source bar start {exit_src}"
        )

    expected_entry_available = record.entry_bar_start + record.bar_duration_ms
    if record.entry_available_at != expected_entry_available:
        raise ValueError(
            f"entry_available_at must be entry_bar_start + bar_duration_ms "
            f"({expected_entry_available}), got {record.entry_available_at}"
        )
    expected_exit_observable = record.exit_bar_start + record.bar_duration_ms
    if record.exit_observable_at != expected_exit_observable:
        raise ValueError(
            f"exit_observable_at must be exit_bar_start + bar_duration_ms "
            f"({expected_exit_observable}), got {record.exit_observable_at}"
        )

    span = record.exit_row_id - record.entry_row_id
    if record.bars_held != span:
        raise ValueError(
            f"bars_held {record.bars_held} must equal exit_row_id - entry_row_id ({span})"
        )
    cap = min(record.horizon, record.max_hold_bars)
    if record.bars_held > cap:
        raise ValueError(
            f"bars_held {record.bars_held} exceeds min(horizon, max_hold_bars) ({cap})"
        )
    if record.exit_kind == "horizon" and record.bars_held != cap:
        raise ValueError(
            f"a horizon exit ran the full window, so bars_held must equal {cap}, got "
            f"{record.bars_held}; an earlier exit would have been a stop or a trail"
        )


def _validate_bindings(record: LabelEndpoint, expected: ExpectedEndpointContext) -> None:
    if not isinstance(expected, ExpectedEndpointContext):
        raise ValueError("expected must be an ExpectedEndpointContext")
    for field in ("product_id", "label_version", "cost_version", "config_id", "data_id"):
        _name(getattr(record, field), field)
        _name(getattr(expected, field), f"expected {field}")
        if getattr(record, field) != getattr(expected, field):
            raise ValueError(
                f"{field} {getattr(record, field)!r} does not match the independently "
                f"expected {getattr(expected, field)!r}"
            )
    for field in ("horizon", "bar_duration_ms", "max_hold_bars"):
        _int(getattr(record, field), field, minimum=1)
        _int(getattr(expected, field), f"expected {field}", minimum=1)
        if getattr(record, field) != getattr(expected, field):
            raise ValueError(
                f"{field} {getattr(record, field)} does not match the independently "
                f"expected {getattr(expected, field)}; a self-declared config_id cannot "
                f"attest an embedded value"
            )
    record_label = _finite(record.label_value, "label_value")
    expected_label = _finite(expected.label_value, "expected label_value")
    if record_label != expected_label:
        raise ValueError(
            f"label_value {record_label} does not match the independently expected "
            f"{expected_label}; a correct endpoint paired with a different cached PnL "
            f"is undetectable from the record alone"
        )
    digest = _name(expected.digest, "expected digest")
    actual = endpoint_digest(record)
    if actual != digest:
        raise ValueError(f"digest {actual} does not match the independently stored {digest}")


def validate_endpoint(
    record: LabelEndpoint,
    *,
    source_bar_starts: Mapping[int, int],
    expected: ExpectedEndpointContext,
) -> LabelEndpoint:
    """Validate one endpoint record. Returns it, or raises `ValueError`.

    Requires every invariant and rejects violations; it never repairs a record, and
    every rejection is a `ValueError` — no other exception type escapes.

    `source_bar_starts` is an **endpoint lookup subset**, not necessarily the whole
    frame: it need only contain the rows this record references, and every key and
    value in it is type-validated, so callers should scope it rather than passing a
    full frame.
    """
    if not isinstance(record, LabelEndpoint):
        raise ValueError("record must be a LabelEndpoint")
    _validate_blockers(record.blockers)
    _validate_exit_semantics(record)
    _validate_bindings(record, expected)
    _validate_rows_and_clocks(record, source_bar_starts)
    return record


def map_exit_to_first_retained_candidate(exit_row_id: int, retained_row_ids: Sequence[int]):
    """Map a source exit row to the first retained row at or after it.

    Returns an **original-frame row id**, or `TERMINAL_SENTINEL` when no retained row
    follows. The result is an **eligibility candidate boundary** — the earliest row at
    which a new position may be considered. It is **not** a portfolio accounting time,
    and it is **not** a position in any working array; callers must not confuse the
    returned original row id with an index into their filtered frame.

    A valid exit may legitimately land on a row the working frame dropped, because
    that row carries no label of its own. That is a mapping step, not a rejection.
    """
    _int(exit_row_id, "exit_row_id", minimum=0)
    if isinstance(retained_row_ids, (str, bytes)) or not isinstance(retained_row_ids, Sequence):
        raise ValueError("retained_row_ids must be a sequence of row ids")
    previous = None
    for row_id in retained_row_ids:
        _int(row_id, "retained row id", minimum=0)
        if previous is not None and row_id <= previous:
            raise ValueError("retained row ids must be strictly increasing and unique")
        previous = row_id
    for row_id in retained_row_ids:
        if row_id >= exit_row_id:
            return row_id
    return TERMINAL_SENTINEL
