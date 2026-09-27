"""The shared adapter: where published endpoints become usable consumer inputs.

Contract §9. Two quantities live here and they are not interchangeable — an **eligibility
boundary** is a position in a consumer's working frame, an **accounting time** is a clock
instant. The wall-clock `exit_ts` this replaces played both roles, which is why one defect
produced errors in opposite directions.

The other thing this module exists to prevent is a consumer accepting endpoints it has not
validated. `load_validated_endpoints` is the only way to obtain a `ValidatedEndpoints`, and
the boundary and accounting helpers accept nothing else — so "did anyone check these against
the frame?" is answered by the type rather than by a convention someone can forget.
"""

import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import pandas as pd  # noqa: E402

from tools.strategy_discovery.endpoint_consumers import (  # noqa: E402
    _VALIDATED_BY_LOADER,
    MissingEndpoint,
    ValidatedEndpoints,
    accounting_times,
    eligibility_boundaries,
    finite_candidate_values,
    load_validated_endpoints,
    validated_retained_ids,
    validated_source_ordinals,
    working_positions,
)
from tools.strategy_discovery.endpoint_dataset import (  # noqa: E402
    DATASET_FILENAME,
    write_dataset,
)
from tools.strategy_discovery.endpoint_records import (  # noqa: E402
    TERMINAL_SENTINEL,
    endpoint_digest,
    map_exit_to_first_retained_candidate,
)
from tools.strategy_discovery.labels import (  # noqa: E402
    _DEFAULT_ATR_TRAIL_FLOOR,
    _DEFAULT_MAX_HOLD_BARS,
    _DEFAULT_ROUND_TRIP_FEE,
    _DEFAULT_STOP_LOSS_PCT,
    COST_VERSION,
    LABEL_VERSION,
    simulate_labels_with_endpoints,
)

_BAR = 3_600_000
_PRODUCT = "BTC-USD"
_HORIZONS = (1, 2)
_ROWS = [
    (100.0, 100.0, 100.0),
    (101.0, 99.5, 100.5),
    (102.0, 100.0, 101.5),
    (103.0, 101.0, 102.0),
    (104.0, 102.0, 103.0),
]


def _frame(bar_hours=None):
    """A stamped-frame-shaped input, optionally with gaps between bars."""
    hours = bar_hours or [1] * len(_ROWS)
    starts, clock = [], 0
    for gap in hours:
        starts.append(clock)
        clock += int(gap) * _BAR
    return pd.DataFrame(
        {
            "ts": starts,
            "open": [r[2] for r in _ROWS],
            "high": [r[0] for r in _ROWS],
            "low": [r[1] for r in _ROWS],
            "close": [r[2] for r in _ROWS],
            "atr14_pct": [0.06] * len(_ROWS),
        }
    )


def _exit_config():
    """The declared exit config, matching the producer's defaults.

    `max_hold_bars` is 168 -- the CONFIGURED cap, which is NOT any horizon. A consumer that
    rebuilt this from `record.horizon` would reject every valid short-horizon record.
    """
    # Imported, not restated. A fixture that hard-codes these drifts from the producer the
    # moment a default changes, and then the data_id recompute fails for a reason that has
    # nothing to do with the code under test -- which is exactly what happened when I wrote
    # 0.006 here and the real fee is 0.012.
    return {
        "stop_loss_pct": _DEFAULT_STOP_LOSS_PCT,
        "atr_trail_floor": _DEFAULT_ATR_TRAIL_FLOOR,
        "max_hold_bars": _DEFAULT_MAX_HOLD_BARS,
        "round_trip_fee": _DEFAULT_ROUND_TRIP_FEE,
    }


def _publish(tmp_path, bar_hours=None):
    """Publish a real dataset plus the sidecar the adapter consumes."""
    frame = _frame(bar_hours)
    labelled, endpoints = simulate_labels_with_endpoints(
        frame, horizons=list(_HORIZONS), product_id=_PRODUCT
    )
    directory = tmp_path / "endpoints" / _PRODUCT
    manifest_digest = write_dataset(directory, endpoints=endpoints, data_id=endpoints[0].data_id)
    sidecar = {
        "sidecar_version": 1,
        "manifest_digest": manifest_digest,
        "data_id": endpoints[0].data_id,
        "product_id": _PRODUCT,
        "horizons": list(_HORIZONS),
        "bar_duration_ms": _BAR,
        "feature_recipe": "atr14_pct_wilder_v1",
        "label_version": LABEL_VERSION,
        "cost_version": COST_VERSION,
        "exit_config": _exit_config(),
        "record_digests": {f"{e.horizon}:{e.entry_row_id}": endpoint_digest(e) for e in endpoints},
    }
    return labelled, endpoints, directory, sidecar


def _load(labelled, directory, sidecar):
    return load_validated_endpoints(directory, frame=labelled, sidecar=sidecar, product_id=_PRODUCT)


# ── the happy path, so every rejection below means something ─────────────────


def test_a_published_dataset_loads_and_reports_that_semantics_were_checked(tmp_path):
    labelled, endpoints, directory, sidecar = _publish(tmp_path)
    validated = _load(labelled, directory, sidecar)
    assert len(validated.records) == len(endpoints)
    assert validated.data_id == sidecar["data_id"]
    assert validated.bar_duration_ms == _BAR
    assert validated.exit_config["max_hold_bars"] == 168
    # Unlike the integrity-only loader, this path DID run validate_endpoint.
    assert validated.semantic_validation_performed is True


def test_the_semantics_flag_cannot_be_constructed_as_false(tmp_path):
    """A guarantee a caller can pass False to is documentation, not a guarantee."""
    with pytest.raises(TypeError):
        ValidatedEndpoints(
            records=(),
            data_id="sha256:x",
            bar_duration_ms=_BAR,
            exit_config=_exit_config(),
            feature_recipe="atr14_pct_wilder_v1",
            frame_fingerprint="sha256:fixture-frame",
            semantic_validation_performed=False,
        )


def test_a_short_horizon_record_validates_against_the_config_cap_not_its_horizon(tmp_path):
    """THE regression for the cap blocker.

    A horizon-1 record publishes `max_hold_bars=168`, because the simulation uses
    `min(horizon, cap)` internally while the record carries the configured cap. Rebuilding
    the expected cap from the horizon rejects every valid short-horizon record.
    """
    labelled, endpoints, directory, sidecar = _publish(tmp_path)
    h1 = [e for e in endpoints if e.horizon == 1]
    assert h1, "the fixture must actually contain horizon-1 records"
    assert {e.max_hold_bars for e in h1} == {168}
    assert any(r.horizon == 1 for r in _load(labelled, directory, sidecar).records)


# ── the frame is bound by RECOMPUTING its identity, not by quoting it ────────


def test_a_frame_that_does_not_match_the_declared_data_id_is_refused(tmp_path):
    """Quoting the sidecar's `data_id` proves only that the manifest agrees with the
    sidecar. Altering one close leaves every stored hash internally consistent; only
    recomputing `build_data_id` from the frame catches it."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    labelled = labelled.copy()
    labelled.loc[0, "close"] = labelled.loc[0, "close"] + 1.0
    with pytest.raises(MissingEndpoint, match="data_id"):
        _load(labelled, directory, sidecar)


def test_a_declared_exit_config_that_did_not_produce_the_records_is_refused(tmp_path):
    """`config_id == data_id` by construction, so the config is bound by the same recompute.
    A changed cap therefore surfaces as a data_id mismatch -- correct, though the id alone
    cannot say whether the frame or the config moved."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    sidecar = dict(sidecar, exit_config=dict(_exit_config(), max_hold_bars=169))
    with pytest.raises(MissingEndpoint, match="data_id"):
        _load(labelled, directory, sidecar)


def test_a_record_whose_stored_digest_disagrees_is_refused(tmp_path):
    labelled, endpoints, directory, sidecar = _publish(tmp_path)
    key = f"{endpoints[0].horizon}:{endpoints[0].entry_row_id}"
    sidecar = dict(sidecar, record_digests=dict(sidecar["record_digests"], **{key: "sha256:0"}))
    with pytest.raises(ValueError):
        _load(labelled, directory, sidecar)


def test_a_record_with_no_stored_digest_is_refused_rather_than_recomputed(tmp_path):
    """Recomputing the digest from the record under test would attest nothing -- the exact
    circularity the external anchor exists to break."""
    labelled, endpoints, directory, sidecar = _publish(tmp_path)
    digests = dict(sidecar["record_digests"])
    digests.pop(f"{endpoints[0].horizon}:{endpoints[0].entry_row_id}")
    with pytest.raises(MissingEndpoint, match="digest"):
        _load(labelled, directory, dict(sidecar, record_digests=digests))


def test_tampered_dataset_content_is_refused(tmp_path):
    labelled, _, directory, sidecar = _publish(tmp_path)
    content = (directory / DATASET_FILENAME).read_bytes()
    (directory / DATASET_FILENAME).write_bytes(content.replace(b"bar_close", b"xxxclose", 1))
    with pytest.raises(ValueError, match="checksum"):
        _load(labelled, directory, sidecar)


@pytest.mark.parametrize("field", ["manifest_digest", "data_id", "bar_duration_ms", "exit_config"])
def test_a_sidecar_missing_a_required_field_is_refused(tmp_path, field):
    labelled, _, directory, sidecar = _publish(tmp_path)
    reduced = {k: v for k, v in sidecar.items() if k != field}
    with pytest.raises(MissingEndpoint, match=field):
        _load(labelled, directory, reduced)


@pytest.mark.parametrize("cap", [0, -1, True, 168.0, "168", None])
def test_a_malformed_declared_cap_is_refused(tmp_path, cap):
    """`True` would pass an `isinstance(int)` check and `168.0` an equality one."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    sidecar = dict(sidecar, exit_config=dict(_exit_config(), max_hold_bars=cap))
    with pytest.raises(MissingEndpoint, match="max_hold_bars"):
        _load(labelled, directory, sidecar)


# ── identity: two id spaces, two rules ───────────────────────────────────────


def test_unfiltered_ordinals_must_be_exactly_zero_to_n_minus_one():
    """The producer writes `arange(n)`. Anything else means the frame was filtered,
    reordered or concatenated after labelling, so its positions no longer mean what the
    endpoints reference. Uniqueness matters here because these become dict keys, where a
    duplicate silently overwrites instead of failing."""
    assert validated_source_ordinals(pd.Series([0, 1, 2, 3])).tolist() == [0, 1, 2, 3]
    for bad in ([0, 1, 1, 3], [0, 1, 3, 4], [1, 2, 3, 4], [3, 2, 1, 0]):
        with pytest.raises(ValueError, match="0..n-1"):
            validated_source_ordinals(pd.Series(bad))


def test_retained_ids_allow_gaps_but_not_duplicates_or_reordering():
    """Gaps are the whole purpose of a finite-label filter; duplicates and reordering are
    not, and either corrupts the position mapping silently."""
    assert validated_retained_ids(pd.Series([0, 3, 7])).tolist() == [0, 3, 7]
    for bad in ([0, 3, 3], [7, 3, 0]):
        with pytest.raises(ValueError, match="strictly increasing"):
            validated_retained_ids(pd.Series(bad))


@pytest.mark.parametrize("bad", [[0, 1.5], [0, True], [-1, 0]])
def test_ids_are_validated_never_coerced(bad):
    """`to_numpy(dtype="int64")` truncates. A truncated timestamp yields a bar start that
    does not describe the frame; a truncated row id is worse, because it still points at a
    real row -- just the wrong one."""
    with pytest.raises(ValueError):
        validated_retained_ids(pd.Series(bad))


def test_working_positions_maps_source_ids_to_positions():
    assert working_positions([0, 3, 7]) == {0: 0, 3: 1, 7: 2}


# ── finite means isfinite, not "not null" ────────────────────────────────────


def test_finite_candidate_values_excludes_infinities_as_well_as_nan():
    """`dropna` RETAINS +/-inf. A consumer that retained rows with `notna` while building
    expectations with `isfinite` would disagree with itself, and an infinite-labelled row
    would be retained with no expectation and no endpoint -- failing as a spurious coverage
    error rather than as the data problem it is."""
    frame = pd.DataFrame(
        {
            "source_row_id": [0, 1, 2, 3],
            "label_h1": [0.1, float("nan"), float("inf"), -0.2],
            "label_h2": [0.3, 0.4, float("-inf"), float("nan")],
        }
    )
    values = finite_candidate_values(frame, product_id=_PRODUCT, horizons=[1, 2])
    assert values == {
        (_PRODUCT, 1, 0): 0.1,
        (_PRODUCT, 1, 3): -0.2,
        (_PRODUCT, 2, 0): 0.3,
        (_PRODUCT, 2, 1): 0.4,
    }
    # the contrast that makes the point
    assert frame["label_h1"].dropna().tolist() == [0.1, float("inf"), -0.2]


def test_finite_candidate_values_refuses_a_horizon_the_frame_lacks():
    frame = pd.DataFrame({"source_row_id": [0], "label_h1": [0.1]})
    with pytest.raises(MissingEndpoint, match="label_h9"):
        finite_candidate_values(frame, product_id=_PRODUCT, horizons=[1, 9])


# ── eligibility boundaries are POSITIONS, and only from validated records ────


def test_the_boundary_helpers_refuse_unvalidated_endpoints(tmp_path):
    """Codex's constraint made structural: a consumer cannot hand these a raw list it
    never checked against the frame. The type carries the guarantee, not a convention."""
    _, endpoints, _, _ = _publish(tmp_path)
    for call in (
        lambda: eligibility_boundaries(endpoints, [0, 1, 2], horizon=1),
        lambda: accounting_times(endpoints, horizon=1),
        lambda: eligibility_boundaries(list(endpoints), [0, 1, 2], horizon=1),
    ):
        with pytest.raises(TypeError, match="ValidatedEndpoints"):
            call()


def test_the_eligibility_boundary_is_a_working_position_not_a_row_id(tmp_path):
    """With source rows [0, 3, 7] retained, an exit at source row 3 is working POSITION 1.
    Returning the row id 3 would index the wrong row in every consumer array."""
    validated = _validated_with(tmp_path, [(0, 3)], horizon=1)
    assert eligibility_boundaries(validated, [0, 3, 7], horizon=1, require_all=False).tolist() == [
        1,
        3,
        3,
    ]


def test_an_exit_landing_on_a_dropped_row_maps_forward_not_to_failure(tmp_path):
    """§3: a valid exit may land on a row the working frame dropped, because that row
    carries no label of its own. A mapping step, not a rejection."""
    validated = _validated_with(tmp_path, [(0, 5)], horizon=1)
    assert eligibility_boundaries(validated, [0, 3, 7], horizon=1, require_all=False)[0] == 2


def test_an_exit_past_the_last_retained_row_is_terminal(tmp_path):
    validated = _validated_with(tmp_path, [(0, 99)], horizon=1)
    assert eligibility_boundaries(validated, [0, 3, 7], horizon=1, require_all=False)[0] == 3


def test_a_retained_candidate_with_no_endpoint_raises(tmp_path):
    """§9.3: inside a coverage-verified artifact this is a defensive assertion, and it must
    never fall back to horizon arithmetic."""
    validated = _validated_with(tmp_path, [(0, 1)], horizon=1)
    with pytest.raises(MissingEndpoint, match="no endpoint for retained candidate"):
        eligibility_boundaries(validated, [0, 3, 7], horizon=1)


def test_the_vectorized_mapping_agrees_with_the_pure_helper_on_every_row(tmp_path):
    """The pure helper is the ORACLE. The vectorized path exists only because the helper is
    O(n) per call and re-validates its whole input each time, so calling it per endpoint
    over a real frame is quadratic. They must never disagree."""
    retained = [0, 1, 4, 5, 9, 10, 17]
    validated = _validated_with(tmp_path, [(r, r + 3) for r in retained], horizon=1)
    boundaries = eligibility_boundaries(validated, retained, horizon=1, require_all=False)
    for position, record in enumerate(
        sorted((r for r in validated.records if r.horizon == 1), key=lambda r: r.entry_row_id)
    ):
        oracle = map_exit_to_first_retained_candidate(record.exit_row_id, retained)
        expected = len(retained) if oracle is TERMINAL_SENTINEL else retained.index(oracle)
        assert boundaries[position] == expected


def test_only_the_requested_horizon_is_used(tmp_path):
    """One frame carries several horizons; mixing them would pair one horizon's rule with
    another's exit timing."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    validated = _load(labelled, directory, sidecar)
    retained = labelled["source_row_id"].tolist()
    one = eligibility_boundaries(validated, retained, horizon=1, require_all=False)
    two = eligibility_boundaries(validated, retained, horizon=2, require_all=False)
    assert one.tolist() != two.tolist()


# ── accounting times are INSTANTS ────────────────────────────────────────────


def test_accounting_times_come_from_exit_observable_at(tmp_path):
    labelled, endpoints, directory, sidecar = _publish(tmp_path)
    validated = _load(labelled, directory, sidecar)
    times = accounting_times(validated, horizon=1)
    for record in (r for r in endpoints if r.horizon == 1):
        assert times[record.entry_row_id] == record.exit_observable_at
        # the distinguishing property: a bar CLOSE, never the bar's start
        assert times[record.entry_row_id] == record.exit_bar_start + _BAR


def test_an_accounting_time_is_never_a_position(tmp_path):
    """Sanity check on the type separation: the instants are milliseconds, far outside any
    plausible row index, so a substitution would be caught rather than silently plausible."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    validated = _load(labelled, directory, sidecar)
    assert all(t >= _BAR for t in accounting_times(validated, horizon=1).values())


# ── helper: a validated set built from hand-specified entry/exit pairs ───────


def _validated_with(tmp_path, pairs, *, horizon):
    """Publish records with chosen (entry_row_id, exit_row_id) pairs, then load them.

    Goes through `load_validated_endpoints` rather than constructing `ValidatedEndpoints`
    directly, so these tests exercise the same guarantee the consumers will.
    """
    import dataclasses

    labelled, endpoints, _, sidecar = _publish(tmp_path)
    template = next(e for e in endpoints if e.horizon == horizon)
    starts = labelled["ts"].tolist()

    def _bar_start(row_id):
        # rows beyond the frame are deliberately reachable: an exit may legitimately point
        # past the retained rows, and the clock map is scoped per record
        return (
            starts[row_id]
            if row_id < len(starts)
            else starts[-1] + (row_id - len(starts) + 1) * _BAR
        )

    records = tuple(
        dataclasses.replace(
            template,
            entry_row_id=entry,
            exit_row_id=exit_row,
            bars_held=exit_row - entry,
            entry_bar_start=_bar_start(entry),
            exit_bar_start=_bar_start(exit_row),
            entry_available_at=_bar_start(entry) + _BAR,
            exit_observable_at=_bar_start(exit_row) + _BAR,
        )
        for entry, exit_row in pairs
    )
    # Reaches for the private construction token DELIBERATELY. These tests need
    # hand-specified (entry, exit) pairs -- an exit past the end of the frame, an exit on a
    # dropped row -- that no real simulation would emit, so they cannot come through
    # load_validated_endpoints. The gate blocking this by default is the gate working;
    # test_validated_endpoints_cannot_be_forged_by_direct_construction covers that.
    return ValidatedEndpoints(
        records=records,
        data_id=sidecar["data_id"],
        bar_duration_ms=_BAR,
        exit_config=_exit_config(),
        feature_recipe="atr14_pct_wilder_v1",
        frame_fingerprint="sha256:fixture-frame",
        token=_VALIDATED_BY_LOADER,
    )


# ── two defects found by executed review, reproduced before fixing ───────────


@pytest.mark.parametrize("values", [[False, 1], (False, 1), [0, True], (0, True, 2)])
def test_a_bool_mixed_into_an_int_sequence_is_still_rejected(values):
    """Codex 461644d3: `validated_source_ordinals([False, 1])` returned `[0, 1]`.

    `np.asarray(list(values))` coerced the mixed sequence to int64 BEFORE the bool check
    ran, erasing the very evidence the check looks for. An all-bool array was rejected, so
    the check looked like it worked. Validation must see the ORIGINAL element types.
    """
    for validator in (validated_source_ordinals, validated_retained_ids):
        with pytest.raises(ValueError, match="bool"):
            validator(values)


def test_validated_endpoints_cannot_be_forged_by_direct_construction():
    """Codex 461644d3: `ValidatedEndpoints(records=(), data_id="unverified", ...)` built
    cleanly with `semantic_validation_performed=True`.

    `init=False` stops a caller CHOOSING False; it does not establish that validation
    happened. My docstring claimed `load_validated_endpoints` was the only route, and that
    was simply false — so every downstream `isinstance` check would have trusted fabricated
    records. The guard is against accident and against misplaced downstream trust, not
    against a determined caller reading the source.
    """
    with pytest.raises(TypeError, match="load_validated_endpoints"):
        ValidatedEndpoints(
            records=(),
            data_id="unverified",
            bar_duration_ms=1,
            exit_config={},
            feature_recipe="atr14_pct_wilder_v1",
            frame_fingerprint="sha256:fixture-frame",
        )


def test_a_validated_results_config_cannot_be_mutated_afterwards(tmp_path):
    """A frozen dataclass does not freeze a dict it holds. A consumer that mutated
    `exit_config` would change the cap every later record is validated against."""
    labelled, _, directory, sidecar = _publish(tmp_path)
    validated = _load(labelled, directory, sidecar)
    with pytest.raises(TypeError):
        validated.exit_config["max_hold_bars"] = 1
    assert validated.exit_config["max_hold_bars"] == _DEFAULT_MAX_HOLD_BARS


# ── the frame fingerprint, and the collision it had ──────────────────────────


def test_renaming_two_columns_to_swap_them_changes_the_fingerprint():
    """Codex 6dfdbd5c, reproduced before fixing.

    An earlier version hashed SORTED column names and then all row values together. Renaming
    two columns to swap their names -- without moving a single value -- produced an IDENTICAL
    fingerprint, because the sorted header lost the name-to-position binding while the row
    hashes followed physical order. `price_over_ema20` went 1.5 -> 0.01 undetected, which is
    exactly a changed rule feature.
    """
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint

    original = pd.DataFrame({"price_over_ema20": [1.5], "vol_over_mc": [0.01]})
    swapped = original.rename(
        columns={"price_over_ema20": "vol_over_mc", "vol_over_mc": "price_over_ema20"}
    )
    assert original["price_over_ema20"].tolist() == [1.5]
    assert swapped["price_over_ema20"].tolist() == [0.01], "the swap must change the feature"
    assert frame_fingerprint(original) != frame_fingerprint(swapped)


def test_reordering_columns_does_not_change_the_fingerprint():
    """The other direction, which the same earlier version got wrong too: a harmless column
    reorder changed the fingerprint. Rules look columns up by NAME, so order carries no
    meaning and flagging it would be a false positive."""
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint

    original = pd.DataFrame({"price_over_ema20": [1.5], "vol_over_mc": [0.01]})
    reordered = original[["vol_over_mc", "price_over_ema20"]]
    assert list(reordered.columns) != list(original.columns)
    assert frame_fingerprint(original) == frame_fingerprint(reordered)


def test_duplicate_column_names_cannot_be_fingerprinted():
    """With two columns of one name there is no name-to-values mapping to bind."""
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint

    with pytest.raises(ValueError, match="duplicate column names"):
        frame_fingerprint(pd.DataFrame([[1, 2]], columns=["x", "x"]))


def test_a_changed_value_changes_the_fingerprint():
    """The base case, so the two tests above are not the only evidence the digest responds to
    anything at all."""
    from tools.strategy_discovery.endpoint_consumers import frame_fingerprint

    original = pd.DataFrame({"price_over_ema20": [1.5], "vol_over_mc": [0.01]})
    changed = original.copy()
    changed.loc[0, "price_over_ema20"] = 1.6
    assert frame_fingerprint(original) != frame_fingerprint(changed)
