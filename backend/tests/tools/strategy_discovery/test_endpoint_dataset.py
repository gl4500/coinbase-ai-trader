"""Endpoint dataset and manifest: integrity and provenance binding.

**Not authentication.** These checks prove content has not changed since it was
recorded. They say nothing about who recorded it, or whether the recording was
correct.

The design point that earns its keep is the **external anchor**. An earlier draft had
the manifest carry the dataset's checksum and treated that as the binding — but a
manifest is freely replaceable, so swapping the dataset *and* regenerating its manifest
together produces a coherent pair whose every internal hash agrees. A checksum inside
the thing being checked detects accident, never substitution. So `load_dataset`
requires an **independently supplied expected manifest digest**, retained by whatever
artifact references this dataset, and the chain is:

    referencing artifact -> expected manifest digest -> manifest
                         -> dataset checksum -> dataset content

Every link verified against something *outside* itself.

Loading fails closed on more than hashes: unsupported versions, duplicate keys,
truncated or malformed content, and schema or config mismatches. And parsing
successfully clears nothing — records are checked against independent candidate values,
and the version-wide causality blocker survives every successful load.
"""

import json
import os
import sys

import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "..")
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import pandas as pd  # noqa: E402

from tools.strategy_discovery.endpoint_dataset import (  # noqa: E402
    DATASET_FILENAME,
    DATASET_VERSION,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    build_data_id,
    dataset_checksum,
    load_dataset,
    manifest_digest,
    serialize_dataset,
    write_dataset,
)
from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER  # noqa: E402
from tools.strategy_discovery.labels import simulate_labels_with_endpoints  # noqa: E402

_H = 3_600_000
_FLOOR = 0.06
_ROWS = [
    (100.0, 100.0, 100.0, _FLOOR),
    (101.0, 99.5, 100.5, _FLOOR),
    (102.0, 100.0, 101.5, _FLOOR),
    (103.0, 101.0, 102.0, _FLOOR),
]


def _frame():
    return pd.DataFrame(
        {
            "ts": [i * _H for i in range(len(_ROWS))],
            "open": [r[2] for r in _ROWS],
            "high": [r[0] for r in _ROWS],
            "low": [r[1] for r in _ROWS],
            "close": [r[2] for r in _ROWS],
            "atr14_pct": [r[3] for r in _ROWS],
        }
    )


def _produce():
    labelled, endpoints = simulate_labels_with_endpoints(
        _frame(), horizons=[1, 2], product_id="BTC-USD"
    )
    return labelled, endpoints


_PRODUCT = "BTC-USD"
_HORIZONS = (1, 2)


def _candidate_values(labelled, endpoints=None):
    """Expected labels built INDEPENDENTLY of the dataset.

    Keys come from the DECLARED product and horizons crossed with every finite
    candidate row, never from the records. Enumerating keys from the endpoints would
    make a missing record undetectable, because removing a record would remove its own
    expectation along with it. `endpoints` is accepted and ignored so callers read
    symmetrically.
    """
    import math as _math

    values = {}
    for horizon in _HORIZONS:
        column = labelled[f"label_h{horizon}"]
        for row_id, value in zip(labelled["source_row_id"], column):
            if not _math.isnan(value):
                values[(_PRODUCT, int(horizon), int(row_id))] = float(value)
    return values


def _published(tmp_path):
    labelled, endpoints = _produce()
    digest = write_dataset(tmp_path, endpoints=endpoints, data_id=endpoints[0].data_id)
    return labelled, endpoints, digest


def _load(tmp_path, labelled, endpoints, digest, **over):
    kwargs = dict(
        expected_manifest_digest=digest,
        expected_candidate_values=_candidate_values(labelled, endpoints),
        expected_data_id=endpoints[0].data_id,
    )
    kwargs.update(over)
    return load_dataset(tmp_path, **kwargs)


def _read_manifest(tmp_path):
    return json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))


def _write_manifest(tmp_path, manifest):
    (tmp_path / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


# ── the happy path, so every rejection below means something ─────────────────


def test_a_published_dataset_loads_and_returns_its_records(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    loaded = _load(tmp_path, labelled, endpoints, digest)
    assert len(loaded.records) == len(endpoints)
    assert {(e.product_id, e.horizon, e.entry_row_id) for e in loaded.records} == {
        (e.product_id, e.horizon, e.entry_row_id) for e in endpoints
    }
    assert loaded.coverage_complete is True
    # A clean load is integrity, attribution and coverage -- never semantics.
    assert loaded.semantic_validation_required is True


def test_publication_is_deterministic(tmp_path):
    """Same endpoints, same bytes, same digest — so a digest comparison means
    something across runs and machines."""
    _, endpoints = _produce()
    first = write_dataset(tmp_path / "a", endpoints=endpoints, data_id=endpoints[0].data_id)
    second = write_dataset(tmp_path / "b", endpoints=endpoints, data_id=endpoints[0].data_id)
    assert first == second
    assert (tmp_path / "a" / DATASET_FILENAME).read_bytes() == (
        tmp_path / "b" / DATASET_FILENAME
    ).read_bytes()


def test_a_successful_load_never_clears_the_causality_blocker(tmp_path):
    """Parsing and checksums prove integrity. They do not make a label causal."""
    labelled, endpoints, digest = _published(tmp_path)
    for record in _load(tmp_path, labelled, endpoints, digest).records:
        assert CAUSALITY_BLOCKER in record.blockers


# ── the external anchor: a coherent paired swap must fail ────────────────────


def test_a_coherently_swapped_dataset_and_manifest_pair_is_rejected(tmp_path):
    """THE headline case. Replace the dataset AND regenerate its manifest so every
    internal hash agrees. The pair is self-consistent; it is still not the artifact
    that was referenced, and only the independently retained digest can tell."""
    labelled, endpoints, digest = _published(tmp_path)

    other_frame = _frame()
    other_frame["close"] = [100.0, 100.5, 101.5, 90.0]  # different content
    _, other_endpoints = simulate_labels_with_endpoints(
        other_frame, horizons=[1, 2], product_id="BTC-USD"
    )
    # A complete, internally coherent republication over the top of the first.
    other_digest = write_dataset(
        tmp_path, endpoints=other_endpoints, data_id=other_endpoints[0].data_id
    )
    assert other_digest != digest, "the swap must actually change the content"

    with pytest.raises(ValueError, match="manifest"):
        _load(tmp_path, labelled, endpoints, digest)


def test_a_missing_expected_digest_is_not_a_pass(tmp_path):
    """Absence of an anchor is not permission to skip the check."""
    labelled, endpoints, digest = _published(tmp_path)
    for missing in (None, "", "   "):
        with pytest.raises(ValueError, match="expected_manifest_digest"):
            _load(tmp_path, labelled, endpoints, digest, expected_manifest_digest=missing)


def test_the_manifest_digest_is_content_sensitive(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    assert manifest_digest(manifest) == digest
    manifest["row_count"] = manifest["row_count"] + 1
    assert manifest_digest(manifest) != digest


# ── fail closed on more than hashes ──────────────────────────────────────────


def test_an_unsupported_dataset_version_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["dataset_version"] = "endpoint_dataset_v99"
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="version"):
        _load(tmp_path, labelled, endpoints, manifest_digest(manifest))


def test_an_unsupported_manifest_version_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["manifest_version"] = "endpoint_manifest_v99"
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="version"):
        _load(tmp_path, labelled, endpoints, manifest_digest(manifest))


def test_a_schema_version_mismatch_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["schema_version"] = manifest["schema_version"] + 1
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="schema"):
        _load(tmp_path, labelled, endpoints, manifest_digest(manifest))


def test_a_data_id_mismatch_against_the_expectation_is_rejected(tmp_path):
    """The dataset is bound to the input frame it was computed on."""
    labelled, endpoints, digest = _published(tmp_path)
    with pytest.raises(ValueError, match="data_id"):
        _load(tmp_path, labelled, endpoints, digest, expected_data_id="sha256:other-frame")


def test_truncated_dataset_content_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    path = tmp_path / DATASET_FILENAME
    raw = path.read_bytes()
    path.write_bytes(raw[: len(raw) // 2])
    with pytest.raises(ValueError, match="checksum|truncat|malformed"):
        _load(tmp_path, labelled, endpoints, digest)


def test_malformed_dataset_content_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    (tmp_path / DATASET_FILENAME).write_text("{not json at all", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum|malformed"):
        _load(tmp_path, labelled, endpoints, digest)


def test_a_dataset_checksum_mismatch_is_rejected(tmp_path):
    """Content edited and the manifest re-anchored, but the checksum inside it left
    stale — the accidental-corruption case the checksum does catch."""
    labelled, endpoints, digest = _published(tmp_path)
    path = tmp_path / DATASET_FILENAME
    raw = path.read_text(encoding="utf-8")
    path.write_text(raw.replace("BTC-USD", "ETH-USD", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        _load(tmp_path, labelled, endpoints, digest)


def test_duplicate_entry_keys_are_rejected(tmp_path):
    """One (product, horizon, entry) must appear once. A duplicate silently shadows."""
    labelled, endpoints, digest = _published(tmp_path)
    lines = (tmp_path / DATASET_FILENAME).read_text(encoding="utf-8").splitlines()
    duplicated = "\n".join(lines + [lines[0]]) + "\n"
    # write_bytes, not write_text: on Windows the latter translates newlines and would
    # change the very bytes the checksum covers, failing on the wrong reason.
    (tmp_path / DATASET_FILENAME).write_bytes(duplicated.encode("utf-8"))
    manifest = _read_manifest(tmp_path)
    # re-anchor so the failure is the DUPLICATE, not the checksum or the count
    from tools.strategy_discovery.endpoint_dataset import dataset_checksum

    manifest["dataset_checksum"] = dataset_checksum(duplicated.encode("utf-8"))
    manifest["row_count"] = len(lines) + 1
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="duplicate"):
        _load(tmp_path, labelled, endpoints, manifest_digest(manifest))


def test_a_row_count_mismatch_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["row_count"] = manifest["row_count"] + 5
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="row_count"):
        _load(tmp_path, labelled, endpoints, manifest_digest(manifest))


# ── records are checked against INDEPENDENT candidate values ────────────────


def test_a_record_whose_label_disagrees_with_the_candidate_frame_is_rejected(tmp_path):
    """Parsing and checksums cannot detect this: the dataset is internally perfect and
    its label simply is not the one the candidate frame recorded."""
    labelled, endpoints, digest = _published(tmp_path)
    values = _candidate_values(labelled, endpoints)
    key = next(iter(values))
    values[key] = values[key] + 0.5
    with pytest.raises(ValueError, match="label_value"):
        _load(tmp_path, labelled, endpoints, digest, expected_candidate_values=values)


def test_a_record_with_no_candidate_value_is_rejected(tmp_path):
    """An endpoint referring to a candidate that does not exist is unattributable."""
    labelled, endpoints, digest = _published(tmp_path)
    values = _candidate_values(labelled, endpoints)
    values.pop(next(iter(values)))
    with pytest.raises(ValueError, match="candidate"):
        _load(tmp_path, labelled, endpoints, digest, expected_candidate_values=values)


def test_missing_candidate_values_entirely_is_not_a_pass(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    for empty in (None, {}):
        with pytest.raises(ValueError, match="candidate"):
            _load(tmp_path, labelled, endpoints, digest, expected_candidate_values=empty)


# ── atomicity: an interrupted write must not look complete ───────────────────


def test_a_dataset_without_its_manifest_is_incomplete_not_empty(tmp_path):
    """The manifest is written LAST, so its absence marks an interrupted write. A
    missing manifest must never read as a successfully published empty dataset."""
    labelled, endpoints, digest = _published(tmp_path)
    (tmp_path / MANIFEST_FILENAME).unlink()
    with pytest.raises(ValueError, match="manifest"):
        _load(tmp_path, labelled, endpoints, digest)


def test_a_manifest_without_its_dataset_is_rejected(tmp_path):
    labelled, endpoints, digest = _published(tmp_path)
    (tmp_path / DATASET_FILENAME).unlink()
    with pytest.raises(ValueError, match="dataset"):
        _load(tmp_path, labelled, endpoints, digest)


def test_no_temporary_files_survive_a_successful_publication(tmp_path):
    """Atomic replace, not write-in-place: nothing partial is left behind."""
    _, endpoints = _produce()
    write_dataset(tmp_path, endpoints=endpoints, data_id=endpoints[0].data_id)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted([DATASET_FILENAME, MANIFEST_FILENAME])


def test_republication_replaces_both_files_atomically(tmp_path):
    labelled, endpoints, first = _published(tmp_path)
    second = write_dataset(tmp_path, endpoints=endpoints, data_id=endpoints[0].data_id)
    assert first == second  # same content, same digest
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [DATASET_FILENAME, MANIFEST_FILENAME]
    )


def test_publishing_an_empty_endpoint_set_is_refused(tmp_path):
    """A run that produced no endpoints is a disposition to report, not a dataset to
    publish — publishing it would make 'nothing survived' indistinguishable from
    'nothing was attempted'."""
    with pytest.raises(ValueError, match="empty"):
        write_dataset(tmp_path, endpoints=[], data_id="sha256:whatever")


# ── the identity helper still refuses what it should ─────────────────────────


def test_data_id_binds_the_atr_column(tmp_path):
    """`_simulate_one` reads the ATR directly, so an identity without it would not
    notice the input most likely to change — and the one carrying a known defect."""
    common = dict(
        product_id="BTC-USD",
        bar_duration_ms=_H,
        timestamps=[0, _H],
        closes=[100.0, 101.0],
        highs=[100.0, 101.0],
        lows=[100.0, 101.0],
        feature_recipe="atr14_pct_wilder_v1",
        config={"stop_loss_pct": 0.08},
    )
    first = build_data_id(atr_pcts=[0.06, 0.06], **common)
    second = build_data_id(atr_pcts=[0.06, 0.07], **common)
    assert first != second


def test_data_id_distinguishes_identical_prices_on_different_clocks():
    common = dict(
        product_id="BTC-USD",
        bar_duration_ms=_H,
        closes=[100.0, 101.0],
        highs=[100.0, 101.0],
        lows=[100.0, 101.0],
        atr_pcts=[0.06, 0.06],
        feature_recipe="atr14_pct_wilder_v1",
        config={"stop_loss_pct": 0.08},
    )
    assert build_data_id(timestamps=[0, _H], **common) != build_data_id(
        timestamps=[_H, 2 * _H], **common
    )


def test_data_id_hashes_a_non_finite_atr_consistently():
    """A NaN ATR is legitimate input — the simulation falls back to its floor — so it
    must hash deterministically rather than abort or vary."""
    common = dict(
        product_id="BTC-USD",
        bar_duration_ms=_H,
        timestamps=[0, _H],
        closes=[100.0, 101.0],
        highs=[100.0, 101.0],
        lows=[100.0, 101.0],
        feature_recipe="atr14_pct_wilder_v1",
        config={"stop_loss_pct": 0.08},
    )
    nan_atr = [float("nan"), 0.06]
    assert build_data_id(atr_pcts=nan_atr, **common) == build_data_id(atr_pcts=nan_atr, **common)
    assert build_data_id(atr_pcts=nan_atr, **common) != build_data_id(
        atr_pcts=[0.06, 0.06], **common
    )


# ── coverage is checked against independently built expectations ─────────────


def test_a_missing_record_is_detected(tmp_path):
    """Only possible because expected keys come from the candidate frame. Had they been
    enumerated from the records, deleting a record would delete its own expectation."""
    labelled, endpoints = _produce()
    fewer = [e for e in endpoints if not (e.horizon == 2 and e.entry_row_id == 0)]
    assert len(fewer) == len(endpoints) - 1
    digest = write_dataset(tmp_path, endpoints=fewer, data_id=endpoints[0].data_id)
    with pytest.raises(ValueError, match="no record|complete"):
        _load(tmp_path, labelled, endpoints, digest)


def test_an_incomplete_dataset_can_be_loaded_only_when_explicitly_allowed(tmp_path):
    """And it is still reported as incomplete, with a named disposition — an arbitrary
    subset must never present itself as the whole."""
    labelled, endpoints = _produce()
    fewer = [e for e in endpoints if not (e.horizon == 2 and e.entry_row_id == 0)]
    digest = write_dataset(tmp_path, endpoints=fewer, data_id=endpoints[0].data_id)
    loaded = _load(tmp_path, labelled, endpoints, digest, require_complete_coverage=False)
    assert loaded.coverage_complete is False
    assert loaded.dispositions["missing_record"] == 1
    assert len(loaded.records) == len(fewer)


def test_the_expected_key_set_is_independent_of_the_published_records(tmp_path):
    """Sanity check on the harness itself: the expectation is derived from the frame,
    so it does not shrink when records are removed."""
    labelled, endpoints = _produce()
    full = _candidate_values(labelled)
    fewer = [e for e in endpoints if not (e.horizon == 2 and e.entry_row_id == 0)]
    assert len(_candidate_values(labelled)) == len(full)
    assert len(full) > len(fewer)


# ── rows must bind to their own header, and keep the mandatory blocker ────────
#
# Found by executed review (Codex cf/cfc3a007), reproduced before fixing: replacing
# every record's data_id with "sha256:wrong-source" and emptying its blockers, then
# publishing under the ORIGINAL data_id, LOADED CLEANLY with coverage_complete=True.
# The manifest's data_id was anchored externally; each row's own copy never was. One
# fact in two places, only one of them checked -- the same defect class as every other
# finding this session.


def _republish_with(tmp_path, endpoints, declared, **replacements):
    """Hand-build a published pair whose ROWS were mutated, under a DECLARED header.

    Deliberately bypasses `write_dataset`: the writer now rejects these inputs (its own
    tests cover that), so routing through it would never reach the loader. This is what
    a tampered or mis-generated artifact looks like on disk.
    """
    import dataclasses

    mutated = [dataclasses.replace(e, **replacements) for e in endpoints]
    content = serialize_dataset(mutated)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_checksum": dataset_checksum(content),
        "row_count": content.decode("utf-8").count(chr(10)),
        "data_id": declared,
        "label_version": endpoints[0].label_version,
        "cost_version": endpoints[0].cost_version,
        "config_id": endpoints[0].config_id,
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / DATASET_FILENAME).write_bytes(content)
    _write_manifest(tmp_path, manifest)
    return mutated, manifest_digest(manifest)


def test_a_row_claiming_a_different_source_than_its_manifest_is_rejected(tmp_path):
    labelled, endpoints = _produce()
    _, digest = _republish_with(
        tmp_path, endpoints, endpoints[0].data_id, data_id="sha256:wrong-source"
    )
    with pytest.raises(ValueError, match="data_id"):
        _load(tmp_path, labelled, endpoints, digest)


def test_a_record_missing_the_mandatory_causality_blocker_is_rejected(tmp_path):
    """Tested SEPARATELY from the data_id case: a combined mutation would pass as soon
    as either check fired, proving only one of them exists."""
    labelled, endpoints = _produce()
    _, digest = _republish_with(tmp_path, endpoints, endpoints[0].data_id, blockers=())
    with pytest.raises(ValueError, match=CAUSALITY_BLOCKER):
        _load(tmp_path, labelled, endpoints, digest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_version", "label_endpoint_v0"),
        ("cost_version", "free_v1"),
        ("config_id", "sha256:other-config"),
    ],
)
def test_a_row_disagreeing_with_its_header_version_or_config_is_rejected(tmp_path, field, value):
    labelled, endpoints = _produce()
    _, digest = _republish_with(tmp_path, endpoints, endpoints[0].data_id, **{field: value})
    with pytest.raises(ValueError, match=field):
        _load(tmp_path, labelled, endpoints, digest)


def test_a_single_odd_row_among_good_ones_is_rejected(tmp_path):
    """Heterogeneity, not just wholesale replacement: one substituted row must not hide
    behind its well-formed neighbours."""
    import dataclasses

    labelled, endpoints = _produce()
    mixed = list(endpoints)
    mixed[2] = dataclasses.replace(mixed[2], cost_version="free_v1")
    with pytest.raises(ValueError, match="cost_version"):
        write_dataset(tmp_path, endpoints=mixed, data_id=endpoints[0].data_id)


# ── the writer must not bless what the loader would reject ───────────────────


def test_the_writer_refuses_records_that_disagree_with_the_declared_source(tmp_path):
    """Publication is where the inconsistency is cheapest to catch. A writer that emits
    an artifact its own loader rejects has merely moved the failure downstream."""
    import dataclasses

    labelled, endpoints = _produce()
    wrong = [dataclasses.replace(e, data_id="sha256:wrong-source") for e in endpoints]
    with pytest.raises(ValueError, match="data_id"):
        write_dataset(tmp_path, endpoints=wrong, data_id=endpoints[0].data_id)


def test_the_writer_refuses_records_without_the_mandatory_blocker(tmp_path):
    import dataclasses

    labelled, endpoints = _produce()
    cleared = [dataclasses.replace(e, blockers=()) for e in endpoints]
    with pytest.raises(ValueError, match=CAUSALITY_BLOCKER):
        write_dataset(tmp_path, endpoints=cleared, data_id=endpoints[0].data_id)


# ── types are checked, not coerced ───────────────────────────────────────────


def test_a_boolean_schema_version_is_not_a_valid_schema_version(tmp_path):
    """`True == 1` in Python, so an equality check alone accepts a bool where a version
    number belongs."""
    labelled, endpoints, digest = _published(tmp_path)
    manifest = _read_manifest(tmp_path)
    manifest["schema_version"] = True
    _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="schema version"):
        _load(
            tmp_path,
            labelled,
            endpoints,
            digest,
            expected_manifest_digest=manifest_digest(manifest),
        )


@pytest.mark.parametrize("value", [1, 0, "yes", None])
def test_require_complete_coverage_must_be_an_actual_bool(tmp_path, value):
    """A truthy string silently relaxing a coverage requirement is exactly the kind of
    accident this whole module exists to refuse."""
    labelled, endpoints, digest = _published(tmp_path)
    with pytest.raises(ValueError, match="require_complete_coverage"):
        _load(tmp_path, labelled, endpoints, digest, require_complete_coverage=value)


@pytest.mark.parametrize(
    "key",
    [
        ("BTC-USD", True, 99),  # a bool on a key that does not already exist
        ("BTC-USD", 9, False),  # horizon 9 is unused, so the bool survives to be seen
        ("BTC-USD", 1),  # wrong arity
        "BTC-USD/1/0",  # not a tuple at all
        ("", 1, 0),  # empty product
    ],
)
def test_malformed_expected_candidate_keys_are_rejected(tmp_path, key):
    labelled, endpoints, digest = _published(tmp_path)
    values = dict(_candidate_values(labelled))
    values[key] = 0.0
    with pytest.raises(ValueError, match="expected_candidate_values"):
        _load(tmp_path, labelled, endpoints, digest, expected_candidate_values=values)


def test_a_bool_that_aliases_an_existing_key_cannot_be_detected_but_fails_closed(
    tmp_path,
):
    """The boundary this module cannot police, stated honestly.

    `values[("BTC-USD", True, 0)] = 0.0` does not add a malformed key -- `True` hashes
    as `1`, so it OVERWRITES the expectation at `("BTC-USD", 1, 0)` and the dict keeps
    the original int key. No loader-side key check can see a bool that is already gone.

    What is guaranteed here is narrower than "fail-closed", and the narrowness matters.
    Rejection is possible only because the overwrite CHANGED the value, leaving the
    surviving mapping inconsistent with the published label. Had the aliased write stored
    an identical value, the insertion history would be unobservable and acceptance would
    be the correct outcome, not a missed defect. So: no detection of the bool, and no
    universal promise of rejection -- only that an expectation which no longer matches
    the artifact is refused.
    """
    labelled, endpoints, digest = _published(tmp_path)
    values = dict(_candidate_values(labelled))
    original = values[("BTC-USD", 1, 0)]
    values[("BTC-USD", True, 0)] = 0.0

    assert len(values) == len(_candidate_values(labelled)), "no key was added"
    assert values[("BTC-USD", 1, 0)] == 0.0, "the legitimate expectation was overwritten"
    assert original != 0.0, "the overwrite actually changed the expected value"
    assert all(not isinstance(part, bool) for key in values for part in key), (
        "the bool did not survive in the mapping at all"
    )

    with pytest.raises(ValueError, match="disagrees with the candidate frame"):
        _load(tmp_path, labelled, endpoints, digest, expected_candidate_values=values)


def test_a_load_result_cannot_be_constructed_claiming_semantics_are_settled(tmp_path):
    """The docstring said it was always True; the constructor let a caller say
    otherwise. A guarantee a caller can override is documentation, not a guarantee."""
    from tools.strategy_discovery.endpoint_dataset import LoadedEndpointDataset

    with pytest.raises(TypeError):
        LoadedEndpointDataset(
            records=(),
            coverage_complete=True,
            dispositions={},
            semantic_validation_required=False,
        )
    assert (
        LoadedEndpointDataset(
            records=(), coverage_complete=True, dispositions={}
        ).semantic_validation_required
        is True
    )


# ── the atomicity claim, stated as narrowly as it is actually true ───────────


def test_an_interrupted_overwrite_leaves_a_stale_manifest_that_is_still_rejected(
    tmp_path,
):
    """The honest boundary. Publication is per-file atomic, NOT pair-atomic: on an
    OVERWRITE, an interruption after the dataset is replaced leaves the OLD manifest
    behind -- not no manifest. Nothing silently passes, because the checksum no longer
    matches, but the containment is the checksum, not atomicity. Claiming otherwise
    would be the overclaim this module was written to avoid."""
    labelled, endpoints, digest = _published(tmp_path)

    other = _frame()
    other["close"] = [100.0, 100.5, 101.5, 90.0]
    _, other_endpoints = simulate_labels_with_endpoints(
        other, horizons=[1, 2], product_id="BTC-USD"
    )
    # Dataset replaced, manifest write "interrupted" -- the first file only.
    (tmp_path / DATASET_FILENAME).write_bytes(serialize_dataset(other_endpoints))

    manifest = _read_manifest(tmp_path)
    assert manifest_digest(manifest) == digest, "the stale manifest is still the old one"
    with pytest.raises(ValueError, match="checksum"):
        _load(tmp_path, labelled, endpoints, digest)
