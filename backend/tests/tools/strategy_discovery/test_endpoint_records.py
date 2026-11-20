"""Regression tests for the pure label-endpoint record validator.

The validator exists because three components independently derived when a labelled
trade ends — mining eligibility and portfolio replay from **wall-clock**, the label
itself from a **row offset** or an earlier triggered exit — and on gapped data they
disagreed. Contract: `docs/specs/2026-09-26-label-endpoint-contract.md`.

Written test-first; each case names the production behaviour that would make it fail.

**Row ids are positional ordinals** into the original frame, which is why `bars_held`
is their difference.

Three things this module deliberately does **not** establish:

* that an endpoint corresponds to a real fill — it does not. `intrabar_timing_known`
  describes *simulated within-bar timing under the declared bar model*, never an
  observed execution.
* that the labels are causal — they are not. The trail threshold reads the
  **current** bar's ATR, so a label depends on information from after its own
  decision point (verified: changing only a bar's close moves its label). Every
  record therefore carries a **version-wide** causality blocker this validator
  requires and can never clear.
* that a record matches a real artifact by self-description. A self-declared
  `config_id` cannot attest its own embedded cap, and a freshly recomputed self-hash
  proves nothing — so every binding is checked against an **independently supplied**
  expected context and digest.
"""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.endpoint_records import (  # noqa: E402
    CAUSALITY_BLOCKER,
    TERMINAL_SENTINEL,
    ExpectedEndpointContext,
    LabelEndpoint,
    endpoint_digest,
    map_exit_to_first_retained_candidate,
    validate_endpoint,
)

_H = 3_600_000
_SOURCE = {0: 0, 1: _H, 2: 2 * _H, 3: 3 * _H, 4: 4 * _H}
_DATA_ID = "sha256:frame-abc"


def _endpoint(**over):
    """Valid horizon exit: enter row 0, exit row 2, horizon 2, cap 168."""
    base = dict(
        product_id="BTC-USD",
        horizon=2,
        data_id=_DATA_ID,
        label_version="label_endpoint_v1",
        cost_version="round_trip_fee_v1",
        config_id="cfg-1",
        label_value=0.0125,
        entry_row_id=0,
        exit_row_id=2,
        bars_held=2,
        max_hold_bars=168,
        entry_bar_start=0,
        exit_bar_start=2 * _H,
        bar_duration_ms=_H,
        entry_available_at=_H,
        exit_observable_at=3 * _H,
        exit_kind="horizon",
        exit_price_basis="bar_close",
        intrabar_timing_known=True,
        intrabar_order_assumption=None,
        blockers=(CAUSALITY_BLOCKER,),
    )
    base.update(over)
    return LabelEndpoint(**base)


def _context(record=None, **over):
    """Expected context supplied INDEPENDENTLY of the record.

    Defaults mirror a correct record so that each test perturbs exactly one side.
    """
    src = record if record is not None else _endpoint()
    base = dict(
        product_id="BTC-USD",
        horizon=2,
        data_id=_DATA_ID,
        label_version="label_endpoint_v1",
        cost_version="round_trip_fee_v1",
        config_id="cfg-1",
        label_value=0.0125,
        bar_duration_ms=_H,
        max_hold_bars=168,
        digest=endpoint_digest(src),
    )
    base.update(over)
    return ExpectedEndpointContext(**base)


def _validate(record=None, *, source=None, context=None, **ctx_over):
    rec = _endpoint() if record is None else record
    ctx = context if context is not None else _context(rec, **ctx_over)
    return validate_endpoint(
        rec,
        source_bar_starts=_SOURCE if source is None else source,
        expected=ctx,
    )


# ── happy paths, so every rejection below means something ────────────────────


def test_a_consistent_horizon_endpoint_validates():
    _validate()


def test_a_consistent_trail_endpoint_validates():
    rec = _endpoint(
        exit_kind="trail",
        exit_price_basis="assumed_trail_level",
        intrabar_timing_known=False,
        intrabar_order_assumption="high_before_low",
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    _validate(rec)


def test_a_consistent_stop_endpoint_validates():
    rec = _endpoint(
        exit_kind="stop",
        exit_price_basis="assumed_stop_level",
        intrabar_timing_known=False,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    _validate(rec)


# ── expected-context mismatches: the record may be internally perfect ────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("product_id", "ETH-USD"),
        ("horizon", 24),
        ("data_id", "sha256:some-other-frame"),
        ("label_version", "label_endpoint_v2"),
        ("cost_version", "round_trip_fee_v2"),
        ("config_id", "cfg-2"),
        ("label_value", 0.9),
        ("bar_duration_ms", 1_800_000),
        ("max_hold_bars", 4),
    ],
)
def test_expected_context_mismatch_is_rejected(field, value):
    """A self-consistent record bound to the WRONG context must fail. A swapped
    cached `label_value`, an altered cap, or a changed duration cannot be caught by
    the record alone, because the record agrees with itself."""
    with pytest.raises(ValueError, match=field):
        _validate(**{field: value})


def test_digest_mismatch_is_rejected():
    """The expected digest comes from the artifact, not from recomputing the record."""
    with pytest.raises(ValueError, match="digest"):
        _validate(digest="sha256:not-this-record")


def test_a_tampered_record_fails_its_independently_stored_digest():
    original = _endpoint()
    stored = endpoint_digest(original)
    tampered = _endpoint(label_value=0.5)
    with pytest.raises(ValueError, match="digest|label_value"):
        _validate(tampered, context=_context(original, digest=stored))


def test_a_coherently_halved_duration_still_fails_the_independent_binding():
    """Halve the duration AND recompute both availability fields, so every internal
    arithmetic check passes. Only the independently declared duration can reject it
    — which is the binding this test exists for."""
    half = 1_800_000
    rec = _endpoint(
        bar_duration_ms=half,
        entry_available_at=0 + half,
        exit_observable_at=2 * _H + half,
    )
    with pytest.raises(ValueError, match="bar_duration_ms"):
        _validate(rec, context=_context(rec, bar_duration_ms=_H))


def test_a_coherently_altered_cap_still_fails_the_independent_binding():
    """The record lowers its own cap AND shortens its hold to match, so every
    internal invariant holds. A self-declared `config_id` cannot attest its own
    embedded cap, so only the independently supplied cap catches it."""
    rec = _endpoint(
        exit_kind="stop",
        exit_price_basis="assumed_stop_level",
        intrabar_timing_known=False,
        max_hold_bars=1,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    with pytest.raises(ValueError, match="max_hold_bars"):
        _validate(rec, context=_context(rec, max_hold_bars=168))


def test_a_swapped_but_finite_label_value_fails_the_independent_binding():
    """The dangerous case is not a NaN — it is a perfectly ordinary number from a
    DIFFERENT cached label. The record is flawless; only the expected value differs."""
    rec = _endpoint(label_value=0.5)
    with pytest.raises(ValueError, match="label_value"):
        _validate(rec, context=_context(rec, label_value=0.0125))


def test_a_coherently_altered_config_id_fails_the_independent_binding():
    rec = _endpoint(config_id="cfg-tampered")
    with pytest.raises(ValueError, match="config_id"):
        _validate(rec, context=_context(rec, config_id="cfg-1"))


def test_digest_is_deterministic_and_content_sensitive():
    assert endpoint_digest(_endpoint()) == endpoint_digest(_endpoint())
    assert endpoint_digest(_endpoint()) != endpoint_digest(
        _endpoint(exit_row_id=3, bars_held=3, exit_bar_start=3 * _H, exit_observable_at=4 * _H)
    )


# ── row and duration invariants ──────────────────────────────────────────────


def test_zero_duration_endpoint_is_rejected():
    rec = _endpoint(exit_row_id=0, bars_held=0, exit_bar_start=0, exit_observable_at=_H)
    with pytest.raises(ValueError, match="exit_row_id"):
        _validate(rec)


def test_reversed_rows_are_rejected():
    rec = _endpoint(
        entry_row_id=2,
        exit_row_id=1,
        bars_held=-1,
        entry_bar_start=2 * _H,
        exit_bar_start=_H,
        entry_available_at=3 * _H,
        exit_observable_at=2 * _H,
    )
    with pytest.raises(ValueError, match="exit_row_id"):
        _validate(rec)


def test_bars_held_must_equal_the_row_difference():
    with pytest.raises(ValueError, match="bars_held"):
        _validate(_endpoint(bars_held=1))


def test_a_horizon_exit_must_be_exactly_at_the_cap():
    """A `horizon` exit by definition ran the full `min(horizon, max_hold_bars)`.
    An earlier one would have been a stop or a trail."""
    rec = _endpoint(exit_row_id=1, bars_held=1, exit_bar_start=_H, exit_observable_at=2 * _H)
    with pytest.raises(ValueError, match="bars_held"):
        _validate(rec)


def test_a_horizon_exit_respects_the_cap_rather_than_the_horizon():
    """`horizon_cap = min(horizon, max_hold_bars)`, so the cap binds when smaller."""
    rec = _endpoint(horizon=5, max_hold_bars=2, exit_row_id=2, bars_held=2)
    _validate(rec, horizon=5, max_hold_bars=2)


@pytest.mark.parametrize(
    "kind,basis", [("stop", "assumed_stop_level"), ("trail", "assumed_trail_level")]
)
def test_a_triggered_exit_may_end_earlier_than_the_cap(kind, basis):
    rec = _endpoint(
        exit_kind=kind,
        exit_price_basis=basis,
        intrabar_timing_known=False,
        intrabar_order_assumption="high_before_low" if kind == "trail" else None,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    _validate(rec)


def test_bars_held_may_not_exceed_the_cap():
    rec = _endpoint(
        exit_kind="stop",
        exit_price_basis="assumed_stop_level",
        intrabar_timing_known=False,
        horizon=5,
        max_hold_bars=1,
        exit_row_id=3,
        bars_held=3,
        exit_bar_start=3 * _H,
        exit_observable_at=4 * _H,
    )
    with pytest.raises(ValueError, match="bars_held"):
        _validate(rec, horizon=5, max_hold_bars=1)


# ── timestamps must EQUAL the source, not merely look plausible ──────────────


def test_entry_bar_start_must_equal_the_source_bar():
    with pytest.raises(ValueError, match="entry_bar_start"):
        _validate(_endpoint(entry_bar_start=_H, entry_available_at=2 * _H))


def test_exit_bar_start_must_equal_the_source_bar():
    with pytest.raises(ValueError, match="exit_bar_start"):
        _validate(_endpoint(exit_bar_start=3 * _H, exit_observable_at=4 * _H))


@pytest.mark.parametrize("field", ["entry_available_at", "exit_observable_at"])
def test_availability_must_be_bar_start_plus_declared_duration(field):
    with pytest.raises(ValueError, match=field):
        _validate(_endpoint(**{field: 99}))


def test_source_chronology_must_place_entry_before_exit():
    """Row ordinals and timestamps must agree; a frame whose ids invert its clock
    is not a frame this endpoint can be validated against."""
    scrambled = {0: 5 * _H, 1: _H, 2: 0, 3: 3 * _H, 4: 4 * _H}
    with pytest.raises(ValueError, match="chronolog"):
        _validate(source=scrambled)


def test_an_unresolvable_row_id_is_malformed():
    """Isolated deliberately: an out-of-range exit row would ALSO fail the cap
    check, which would mask the defect under test. So the endpoint stays valid and
    the SOURCE loses that row instead — only the lookup can reject it."""
    truncated = {k: v for k, v in _SOURCE.items() if k != 2}
    with pytest.raises(ValueError, match="row_id"):
        _validate(source=truncated)


# ── exit kind and price basis must agree ─────────────────────────────────────


@pytest.mark.parametrize(
    "kind,basis",
    [
        ("horizon", "assumed_stop_level"),
        ("horizon", "assumed_trail_level"),
        ("stop", "bar_close"),
        ("stop", "assumed_trail_level"),
        ("trail", "bar_close"),
        ("trail", "assumed_stop_level"),
    ],
)
def test_mismatched_exit_kind_and_price_basis_are_rejected(kind, basis):
    rec = _endpoint(
        exit_kind=kind,
        exit_price_basis=basis,
        intrabar_timing_known=(kind == "horizon"),
        intrabar_order_assumption="high_before_low" if kind == "trail" else None,
    )
    with pytest.raises(ValueError, match="exit_price_basis"):
        _validate(rec)


@pytest.mark.parametrize("kind", ["", "   ", "STOP", "filled", None, 7])
def test_unknown_exit_kind_is_rejected(kind):
    with pytest.raises(ValueError, match="exit_kind"):
        _validate(_endpoint(exit_kind=kind))


# ── intrabar semantics ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind,basis", [("stop", "assumed_stop_level"), ("trail", "assumed_trail_level")]
)
def test_a_triggered_exit_may_not_claim_a_known_intrabar_time(kind, basis):
    """An OHLC bar records four prices and no ordering, so the instant is unknown."""
    rec = _endpoint(
        exit_kind=kind,
        exit_price_basis=basis,
        intrabar_timing_known=True,
        intrabar_order_assumption="high_before_low" if kind == "trail" else None,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    with pytest.raises(ValueError, match="intrabar_timing_known"):
        _validate(rec)


def test_a_horizon_exit_has_a_known_within_bar_instant():
    """Known under the DECLARED BAR MODEL — the bar's close. Not an observed fill."""
    with pytest.raises(ValueError, match="intrabar_timing_known"):
        _validate(_endpoint(intrabar_timing_known=False))


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_intrabar_timing_known_must_be_an_actual_bool(value):
    with pytest.raises(ValueError, match="intrabar_timing_known"):
        _validate(_endpoint(intrabar_timing_known=value))


def test_a_trail_exit_must_record_its_ordering_assumption():
    """The trail assumes the bar's high preceded its low, without evidence."""
    rec = _endpoint(
        exit_kind="trail",
        exit_price_basis="assumed_trail_level",
        intrabar_timing_known=False,
        intrabar_order_assumption=None,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    with pytest.raises(ValueError, match="intrabar_order_assumption"):
        _validate(rec)


@pytest.mark.parametrize("kind,basis", [("horizon", "bar_close"), ("stop", "assumed_stop_level")])
def test_a_non_trail_exit_may_not_carry_an_ordering_assumption(kind, basis):
    rec = _endpoint(
        exit_kind=kind,
        exit_price_basis=basis,
        intrabar_timing_known=(kind == "horizon"),
        intrabar_order_assumption="high_before_low",
        exit_row_id=2 if kind == "horizon" else 1,
        bars_held=2 if kind == "horizon" else 1,
        exit_bar_start=(2 if kind == "horizon" else 1) * _H,
        exit_observable_at=(3 if kind == "horizon" else 2) * _H,
    )
    with pytest.raises(ValueError, match="intrabar_order_assumption"):
        _validate(rec)


# ── strict types and ranges ──────────────────────────────────────────────────


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True, None, "0.1"])
def test_a_non_finite_or_non_numeric_label_value_is_rejected(value):
    """The context carries a placeholder digest here on purpose: a non-finite record
    cannot be digested at all (see the next test), so computing one from it would
    raise inside the harness before the validator was reached."""
    rec = _endpoint(label_value=value)
    ctx = ExpectedEndpointContext(
        product_id="BTC-USD",
        horizon=2,
        data_id=_DATA_ID,
        label_version="label_endpoint_v1",
        cost_version="round_trip_fee_v1",
        config_id="cfg-1",
        label_value=value,
        bar_duration_ms=_H,
        max_hold_bars=168,
        digest="sha256:placeholder",
    )
    with pytest.raises(ValueError, match="label_value"):
        _validate(rec, context=ctx)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_record_cannot_be_digested(value):
    """Strict JSON: a NaN or infinity must not escape into a digest payload as a
    non-standard literal that a strict parser would later reject."""
    with pytest.raises(ValueError):
        endpoint_digest(_endpoint(label_value=value))


@pytest.mark.parametrize(
    "field,value",
    [
        ("product_id", ""),
        ("product_id", " BTC-USD "),
        ("product_id", 7),
        ("horizon", 0),
        ("horizon", -2),
        ("horizon", True),
        ("horizon", 2.0),
        ("entry_row_id", -1),
        ("entry_row_id", True),
        ("exit_row_id", True),
        ("bar_duration_ms", 0),
        ("bar_duration_ms", -_H),
        ("max_hold_bars", 0),
        ("max_hold_bars", True),
        ("label_version", ""),
        ("cost_version", ""),
        ("config_id", ""),
        ("data_id", ""),
    ],
)
def test_strict_field_types_and_ranges(field, value):
    with pytest.raises(ValueError, match=field):
        _validate(
            _endpoint(**{field: value}),
            **{field: value}
            if field
            in {
                "horizon",
                "bar_duration_ms",
                "max_hold_bars",
                "product_id",
                "label_version",
                "cost_version",
                "config_id",
                "data_id",
            }
            else {},
        )


# ── the version-wide causality blocker is never clearable ────────────────────


def test_the_causality_blocker_is_mandatory():
    """Verified defect: the trail threshold uses the CURRENT bar's ATR, so labels
    depend on information from after their own decision point. This legacy
    simulation version carries the blocker on every record, and no endpoint
    record — however well formed — clears it."""
    with pytest.raises(ValueError, match="blocker"):
        _validate(_endpoint(blockers=()))


def test_the_causality_blocker_coexists_with_other_blockers():
    _validate(_endpoint(blockers=(CAUSALITY_BLOCKER, "some_other_named_blocker")))


@pytest.mark.parametrize(
    "value",
    [
        [CAUSALITY_BLOCKER],
        CAUSALITY_BLOCKER,
        (CAUSALITY_BLOCKER, 7),
        ("",),
        (CAUSALITY_BLOCKER, CAUSALITY_BLOCKER),
    ],
)
def test_blockers_must_be_a_tuple_of_unique_names(value):
    """A mutable list would let a caller pop the mandatory blocker after validation."""
    with pytest.raises(ValueError, match="blocker"):
        _validate(_endpoint(blockers=value))


# ── eligibility mapping: a dropped working row is not an invalid endpoint ────


def test_a_valid_exit_on_a_dropped_row_maps_to_the_next_retained_candidate():
    """The contract's worked example. Original [0,1,2,3,4], retained [0,1,3,4],
    a valid exit at source row 2 maps to source 3 — it is NOT a rejection."""
    assert map_exit_to_first_retained_candidate(2, (0, 1, 3, 4)) == 3


def test_an_exit_on_a_retained_row_maps_to_itself_exact_boundary():
    assert map_exit_to_first_retained_candidate(3, (0, 1, 3, 4)) == 3
    assert map_exit_to_first_retained_candidate(0, (0, 1, 3, 4)) == 0


def test_an_exit_beyond_every_retained_row_maps_to_the_terminal_sentinel():
    assert map_exit_to_first_retained_candidate(9, (0, 1, 3, 4)) is TERMINAL_SENTINEL
    assert map_exit_to_first_retained_candidate(5, (0, 1, 3, 4)) is TERMINAL_SENTINEL


def test_an_empty_retained_set_maps_to_the_terminal_sentinel():
    """A frame with no retained rows is legitimate: nothing is eligible."""
    assert map_exit_to_first_retained_candidate(0, ()) is TERMINAL_SENTINEL


def test_validation_does_not_require_the_exit_row_to_be_retained():
    """Validation resolves against the ORIGINAL frame only. Requiring the exit row
    to survive the label filter would reject correct endpoints."""
    _validate()  # exit row 2 is dropped in the retained set above; still valid


@pytest.mark.parametrize("retained", [(3, 1, 0), (0, 1, 1, 3), (0, -1), (0, True), (0, 1.0)])
def test_retained_rows_must_be_strictly_increasing_unique_nonnegative_ints(retained):
    with pytest.raises(ValueError, match="retained"):
        map_exit_to_first_retained_candidate(2, retained)


@pytest.mark.parametrize("bad", [-1, True, 1.0, "2", None])
def test_mapping_rejects_a_malformed_exit_row_id(bad):
    with pytest.raises(ValueError, match="exit_row_id"):
        map_exit_to_first_retained_candidate(bad, (0, 1, 3, 4))


# ─────────────────────────────────────────────────────────────────────────────
# Input-validation gaps found by Codex executing the uncommitted module.
#
# Both are the same shape: the RECORD was validated strictly while its INPUTS were
# trusted. Membership and equality are not type checks — `False` aliases the dict
# key `0`, and `0.0` compares equal to `0` — so a malformed source frame resolved
# silently. And an unhashable value reached a set membership test, raising
# TypeError and escaping this module's promise that every rejection is a ValueError.
# ─────────────────────────────────────────────────────────────────────────────


def test_a_bool_source_key_does_not_alias_row_zero():
    """`hash(False) == hash(0)`, so `{False: 0}` would satisfy a membership test for
    row 0 and the endpoint would validate against a frame that does not describe it."""
    with pytest.raises(ValueError, match="source row_id key"):
        _validate(source={False: 0, 2: 2 * _H})


def test_float_source_keys_are_rejected():
    with pytest.raises(ValueError, match="source row_id key"):
        _validate(source={0.0: 0, 2: 2 * _H})


def test_float_source_values_are_rejected():
    """`0 == 0.0` is True, so equality alone would accept a float-typed frame."""
    with pytest.raises(ValueError, match="source bar start"):
        _validate(source={0: 0.0, 2: float(2 * _H)})


@pytest.mark.parametrize("bad", [-1, True])
def test_malformed_source_keys_are_rejected(bad):
    with pytest.raises(ValueError, match="source row_id key"):
        _validate(source={bad: 0, 0: 0, 2: 2 * _H})


@pytest.mark.parametrize("value", [[], {}, set(), 7, 1.0])
def test_an_unhashable_or_non_string_order_assumption_raises_value_error(value):
    """A list reached `in _ORDER_ASSUMPTIONS` and raised TypeError, which is not the
    exception this module promises. Type is now checked before membership.

    The context comes from a VALID BASELINE rather than from `rec`: digesting a record
    holding a set raises during JSON encoding, so a fixture-derived context would fail
    before `validate_endpoint` was ever reached — and a fixture failure is not evidence
    of the validation defect under test.
    """
    rec = _endpoint(
        exit_kind="trail",
        exit_price_basis="assumed_trail_level",
        intrabar_timing_known=False,
        intrabar_order_assumption=value,
        exit_row_id=1,
        bars_held=1,
        exit_bar_start=_H,
        exit_observable_at=2 * _H,
    )
    with pytest.raises(ValueError, match="intrabar_order_assumption"):
        _validate(rec, context=_context())
