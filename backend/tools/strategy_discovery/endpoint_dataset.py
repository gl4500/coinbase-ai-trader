"""Input identity for published label endpoints.

`data_id` binds **everything the exit decision reads**, not merely the prices. That
distinction is load-bearing: `labels._simulate_one` consumes `atr_pcts[i]` directly to
set its trail threshold, so an OHLC-only hash would let the ATR column — the very
column carrying the known contemporaneous-causality defect — be altered or recomputed
without invalidating a single binding.

Identical prices on different clocks must not share an identity either, so the ordered
timestamps, the row count, the product and the **declared** bar duration are bound too.
A declared duration rather than an inferred one, because the next row may itself be
missing and spacing therefore cannot establish a bar's length.

Encoding is deterministic by construction: every float goes in as `float.hex()`, which
round-trips exactly, and non-finite values become explicit tokens rather than reaching
a JSON encoder that would either fail or emit a non-standard literal. A NaN in the ATR
column is legitimate input — the simulation falls back to its floor — so it must hash
consistently rather than abort.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

DATA_ID_VERSION = "endpoint_data_id_v1"

_NON_FINITE = {"nan": "nan", "inf": "inf", "-inf": "-inf"}


def _encode_float(value: Any) -> str:
    """Exact, deterministic, and explicit about non-finite values."""
    number = float(value)
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return float.hex(number)


def _encode_floats(values: Sequence[Any], field: str) -> list:
    if values is None:
        raise ValueError(f"{field} is required for the input identity")
    return [_encode_float(v) for v in values]


def _encode_ints(values: Sequence[Any], field: str) -> list:
    out = []
    for value in values:
        number = int(value)
        if number != value:
            raise ValueError(f"{field} must contain whole numbers, got {value!r}")
        out.append(number)
    return out


def build_data_id(
    *,
    product_id: str,
    bar_duration_ms: int,
    timestamps: Sequence[int],
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    atr_pcts: Sequence[float],
    feature_recipe: str,
    config: Mapping[str, Any],
) -> str:
    """Digest the identity of the inputs a published endpoint depends on.

    Covers the product, the declared bar duration, the ordered timestamps and row
    count, every array the simulation consumes — **including the ATR** — the feature
    recipe that produced the ATR, and the exit config that shapes every exit.

    Two frames with identical prices but different timestamps, a different declared
    duration, a different ATR column, a different feature recipe, or a different exit
    config all receive different identities.
    """
    if not isinstance(product_id, str) or not product_id.strip():
        raise ValueError("product_id must be a nonempty string")
    if type(bar_duration_ms) is not int or bar_duration_ms <= 0:
        raise ValueError("bar_duration_ms must be a positive int, declared not inferred")
    if not isinstance(feature_recipe, str) or not feature_recipe.strip():
        raise ValueError("feature_recipe must be a nonempty string")
    if not isinstance(config, Mapping) or not config:
        raise ValueError("config is required: the exit parameters shape every endpoint")

    lengths = {len(timestamps), len(closes), len(highs), len(lows), len(atr_pcts)}
    if len(lengths) != 1:
        raise ValueError("every consumed array must have the same length as the frame")

    body = {
        "version": DATA_ID_VERSION,
        "product_id": product_id,
        "bar_duration_ms": bar_duration_ms,
        "row_count": len(closes),
        "feature_recipe": feature_recipe,
        "timestamps": _encode_ints(timestamps, "timestamps"),
        "closes": _encode_floats(closes, "closes"),
        "highs": _encode_floats(highs, "highs"),
        "lows": _encode_floats(lows, "lows"),
        # Bound deliberately: the trail threshold reads this column directly, so an
        # identity without it would not notice the input most likely to change.
        "atr_pcts": _encode_floats(atr_pcts, "atr_pcts"),
        "config": {key: _encode_float(config[key]) for key in sorted(config)},
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset and manifest: integrity and provenance binding, NOT authentication.
#
# These checks prove content has not changed since it was recorded. They say nothing
# about who recorded it, or whether the recording was correct.
#
# The binding rests on an EXTERNAL ANCHOR. A manifest is freely replaceable, so a
# checksum stored inside it detects accidental corruption but cannot reject a
# coherently swapped dataset-and-manifest pair whose every internal hash agrees.
# `load_dataset` therefore requires an independently supplied expected manifest
# digest, retained by whatever artifact references this dataset:
#
#     referencing artifact -> expected manifest digest -> manifest
#                          -> dataset checksum -> dataset content
#
# Each link is verified against something outside itself.
#
# SCOPE OF THIS LOADER, stated because the boundary matters more than the code:
# it establishes INTEGRITY (content unchanged), ATTRIBUTION (labels agree with the
# independent candidate frame) and COVERAGE (every expected candidate has exactly one
# record). It does NOT perform full semantic validation -- it does not check that a
# record's bar clocks, durations or cap agree with the original source frame, because
# doing so requires independently sourced clock and config context that this function
# is not given, and reconstructing those expectations from the very fields under check
# would prove nothing. Results therefore carry `semantic_validation_required = True`,
# and a caller must run `endpoint_records.validate_endpoint` against independently
# sourced context before treating any record as evidence.
# ─────────────────────────────────────────────────────────────────────────────

DATASET_VERSION = "endpoint_dataset_v1"
MANIFEST_VERSION = "endpoint_manifest_v1"
SCHEMA_VERSION = 1

DATASET_FILENAME = "endpoints.jsonl"
MANIFEST_FILENAME = "endpoints_manifest.json"

_KEY_FIELDS = ("product_id", "horizon", "entry_row_id")

# Fields that are properties of the PUBLICATION, not of an individual row. Each row
# carries its own copy, so each copy must be checked against the manifest -- otherwise
# the externally anchored header says one thing and the rows another, which is how a
# dataset came back clean while every record named "sha256:wrong-source" as its source.
_HEADER_FIELDS = ("data_id", "label_version", "cost_version", "config_id")

_RECORD_FIELDS = (
    "product_id",
    "horizon",
    "data_id",
    "label_version",
    "cost_version",
    "config_id",
    "label_value",
    "entry_row_id",
    "exit_row_id",
    "bars_held",
    "max_hold_bars",
    "entry_bar_start",
    "exit_bar_start",
    "bar_duration_ms",
    "entry_available_at",
    "exit_observable_at",
    "exit_kind",
    "exit_price_basis",
    "intrabar_timing_known",
    "intrabar_order_assumption",
    "blockers",
)


@dataclass(frozen=True)
class LoadedEndpointDataset:
    """What a verified load establishes, and what it explicitly does not.

    `semantic_validation_required` is always True: integrity and attribution are not
    semantics, and a caller must still validate each record against independently
    sourced clock and config context before using it as evidence.
    """

    records: tuple
    coverage_complete: bool
    dispositions: dict
    # init=False, not a default: a guarantee a caller can pass False to is
    # documentation, not a guarantee.
    semantic_validation_required: bool = field(init=False, default=True)


def _validate_expected_keys(values: Mapping[tuple, float]) -> None:
    """Expected keys must be well-formed before anything is compared against them.

    A bool key component silently aliases an integer (`True` IS dict key `1`), so a
    malformed expectation would quietly match the wrong row rather than fail.

    The limit of this check is worth stating: it sees only bools that SURVIVED into the
    mapping. A bool that aliased an already-present key was erased by `dict` before this
    function ran, and nothing here can recover that history -- such a mapping is caught
    later, and only if its overwritten value disagrees with the published label.
    """
    for key in values:
        if not isinstance(key, tuple) or len(key) != len(_KEY_FIELDS):
            raise ValueError(
                f"expected_candidate_values keys must be {_KEY_FIELDS} triples; got {key!r}"
            )
        for name, value in zip(_KEY_FIELDS, key, strict=True):
            if name == "product_id":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        f"expected_candidate_values key {key!r} has a malformed product_id"
                    )
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"expected_candidate_values key {key!r} has a malformed {name}: a "
                    f"bool aliases an integer row id and would match the wrong record"
                )


def dataset_checksum(content: bytes) -> str:
    """Checksum of the dataset's exact bytes."""
    if not isinstance(content, bytes):
        raise ValueError("dataset content must be bytes")
    return "sha256:" + hashlib.sha256(content).hexdigest()


def manifest_digest(manifest: Mapping[str, Any]) -> str:
    """Digest of the manifest's content.

    Compared against a digest held by the REFERENCING artifact. Recomputing it here
    and comparing it to itself would prove nothing, which is exactly why the expected
    value must arrive from outside.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest must be a mapping")
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _record_to_row(record: Any) -> dict:
    row = {field: getattr(record, field) for field in _RECORD_FIELDS}
    row["label_value"] = _encode_float(row["label_value"])
    row["blockers"] = list(row["blockers"])
    return row


def serialize_dataset(endpoints: Sequence[Any]) -> bytes:
    """Deterministic line-per-record encoding, sorted by key.

    Sorted so identical endpoints always produce identical bytes: comparing digests
    across runs or machines is only meaningful if the encoding is stable.
    """
    if not endpoints:
        raise ValueError(
            "refusing to publish an empty endpoint dataset: a run that produced none is "
            "a disposition to report, or nothing-survived becomes indistinguishable "
            "from nothing-was-attempted"
        )
    rows = sorted(
        (_record_to_row(e) for e in endpoints),
        key=lambda r: (r["product_id"], r["horizon"], r["entry_row_id"]),
    )
    lines = [
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) for row in rows
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _atomic_write_bytes(path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validated_header(endpoints: Sequence[Any], data_id: str) -> dict:
    """The publication-wide header, validated against the endpoints, not copied from them.

    `data_id` arrives from the caller and every row must match it; `label_version` must
    match the version this code produces. `cost_version` and `config_id` are required to
    be homogeneous, so a single substituted row cannot hide behind well-formed
    neighbours. What this does NOT establish is that `config_id` describes the RIGHT
    config -- that is semantics, and needs independently sourced config context.
    """
    from tools.strategy_discovery.endpoint_records import CAUSALITY_BLOCKER
    from tools.strategy_discovery.labels import LABEL_VERSION

    if not isinstance(data_id, str) or not data_id.strip():
        raise ValueError("data_id must be a non-empty str naming the source frame")

    header = {"data_id": data_id, "label_version": LABEL_VERSION}
    for position, endpoint in enumerate(endpoints):
        for name, expected in (
            ("data_id", data_id),
            ("label_version", LABEL_VERSION),
        ):
            actual = getattr(endpoint, name)
            if actual != expected:
                raise ValueError(
                    f"record {position} declares {name} {actual!r} but the publication "
                    f"declares {expected!r}; a row that disagrees with its own header is "
                    f"not publishable"
                )
        for name in ("cost_version", "config_id"):
            actual = getattr(endpoint, name)
            if not isinstance(actual, str) or not actual.strip():
                raise ValueError(f"record {position} has a malformed {name}")
            if name not in header:
                header[name] = actual
            elif actual != header[name]:
                raise ValueError(
                    f"record {position} declares {name} {actual!r} while another record "
                    f"declares {header[name]!r}; one publication carries one {name}"
                )
        if CAUSALITY_BLOCKER not in tuple(endpoint.blockers):
            raise ValueError(
                f"record {position} is missing the mandatory blocker "
                f"{CAUSALITY_BLOCKER}; this label version is not causal and no record "
                f"may be published as though it were"
            )
    return header


def write_dataset(directory, *, endpoints: Sequence[Any], data_id: str) -> str:
    """Publish the dataset and its manifest atomically. Returns the manifest digest.

    Each file is written under a temporary name and then `os.replace`d, so no reader
    ever observes a PARTIAL file. Publication is therefore per-file atomic but NOT
    pair-atomic, and the difference is worth stating precisely:

      * FIRST publication, interrupted after the dataset lands -> a dataset with no
        manifest, visibly incomplete.
      * OVERWRITE, interrupted after the dataset lands -> the dataset is new while the
        manifest is the OLD one. Not "no manifest". That pair is contained by the
        CHECKSUM, which no longer matches, not by atomicity.

    Nothing silently passes in either case, but the containment is the hash chain, and
    saying "atomic" without that distinction would overclaim.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    header = _validated_header(endpoints, data_id)
    content = serialize_dataset(endpoints)

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_version": DATASET_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_checksum": dataset_checksum(content),
        "row_count": content.decode("utf-8").count("\n"),
    }
    manifest.update(header)

    _atomic_write_bytes(directory / DATASET_FILENAME, content)
    _atomic_write_bytes(
        directory / MANIFEST_FILENAME,
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )
    return manifest_digest(manifest)


def load_dataset(
    directory,
    *,
    expected_manifest_digest: str,
    expected_candidate_values: Mapping[tuple, float],
    expected_data_id: str,
    require_complete_coverage: bool = True,
) -> LoadedEndpointDataset:
    """Load and verify a published endpoint dataset, or raise `ValueError`.

    Fails closed on unsupported versions, schema or `data_id` mismatch, a missing file
    on either side, truncated or malformed content, checksum mismatch, duplicate keys,
    a row-count mismatch, a record whose label disagrees with the independent candidate
    value, a record with no candidate at all, and -- unless
    `require_complete_coverage` is cleared -- any expected candidate with no record.

    `expected_candidate_values` must be built INDEPENDENTLY of the dataset, from the
    declared products and horizons against the candidate frame. Deriving its keys from
    the records would make a missing record undetectable, since removing a record
    would remove its own expectation.

    Returns a `LoadedEndpointDataset`. A clean load is evidence of integrity,
    attribution and coverage -- never of semantics: `semantic_validation_required`
    stays True and the records keep their blockers.
    """
    from tools.strategy_discovery.endpoint_records import (
        CAUSALITY_BLOCKER,
        LabelEndpoint,
    )

    directory = Path(directory)
    if not isinstance(expected_manifest_digest, str) or not expected_manifest_digest.strip():
        raise ValueError(
            "expected_manifest_digest is required and must come from the referencing "
            "artifact; its absence is not permission to skip the check"
        )
    if not expected_candidate_values:
        raise ValueError(
            "expected_candidate_values is required: records are checked against the "
            "candidate frame, never against themselves"
        )
    if not isinstance(expected_data_id, str) or not expected_data_id.strip():
        raise ValueError("expected_data_id is required")
    if require_complete_coverage is not True and require_complete_coverage is not False:
        raise ValueError(
            "require_complete_coverage must be an actual bool; a truthy value silently "
            "relaxing a coverage requirement is the accident this module refuses"
        )
    _validate_expected_keys(expected_candidate_values)

    manifest_path = directory / MANIFEST_FILENAME
    dataset_path = directory / DATASET_FILENAME
    if not manifest_path.exists():
        raise ValueError(
            "manifest is missing: an interrupted publication is incomplete, never an empty success"
        )
    if not dataset_path.exists():
        raise ValueError("dataset content is missing while its manifest is present")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"manifest is malformed: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest is malformed: expected an object")

    # EXTERNAL ANCHOR FIRST. A coherently swapped pair satisfies every internal check,
    # so nothing inside the artifact is trusted until the manifest matches the digest
    # the referencing artifact retained.
    actual_digest = manifest_digest(manifest)
    if actual_digest != expected_manifest_digest:
        raise ValueError(
            f"manifest digest {actual_digest} does not match the independently retained "
            f"{expected_manifest_digest}; a self-consistent dataset and manifest pair is "
            f"still not the artifact that was referenced"
        )

    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError(f"unsupported manifest version {manifest.get('manifest_version')!r}")
    if manifest.get("dataset_version") != DATASET_VERSION:
        raise ValueError(f"unsupported dataset version {manifest.get('dataset_version')!r}")
    schema_version = manifest.get("schema_version")
    # `True == 1`, so an equality check alone accepts a bool where a version belongs.
    if isinstance(schema_version, bool) or schema_version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {schema_version!r}")
    if manifest.get("data_id") != expected_data_id:
        raise ValueError(
            f"manifest data_id {manifest.get('data_id')!r} does not match the expected "
            f"{expected_data_id!r}"
        )

    content = dataset_path.read_bytes()
    actual_checksum = dataset_checksum(content)
    if actual_checksum != manifest.get("dataset_checksum"):
        raise ValueError(
            f"dataset checksum {actual_checksum} does not match the manifest's "
            f"{manifest.get('dataset_checksum')!r}"
        )

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"dataset content is malformed: {exc}") from exc
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != manifest.get("row_count"):
        raise ValueError(
            f"row_count {manifest.get('row_count')!r} does not match the {len(lines)} rows present"
        )

    records = []
    seen = set()
    for position, line in enumerate(lines):
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"dataset row {position} is malformed: {exc}") from exc
        if not isinstance(row, dict) or set(row) != set(_RECORD_FIELDS):
            raise ValueError(f"dataset row {position} is malformed: unexpected fields")
        for name in _HEADER_FIELDS:
            if row[name] != manifest.get(name):
                raise ValueError(
                    f"dataset row {position} declares {name} {row[name]!r} but the "
                    f"externally anchored manifest declares {manifest.get(name)!r}; the "
                    f"header being anchored does not make the rows anchored"
                )
        if CAUSALITY_BLOCKER not in tuple(row["blockers"]):
            raise ValueError(
                f"dataset row {position} is missing the mandatory blocker "
                f"{CAUSALITY_BLOCKER}; a load that accepts a record without it would "
                f"launder a non-causal label into usable evidence"
            )
        # Key field types are checked BEFORE a key is built from them: a JSON array
        # would make the tuple unhashable, and a bool would alias an integer row id.
        for name in _KEY_FIELDS:
            value = row[name]
            if name == "product_id":
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"dataset row {position} has a malformed product_id")
            elif isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"dataset row {position} has a malformed {name}")
        key = tuple(row[field] for field in _KEY_FIELDS)
        if key in seen:
            raise ValueError(
                f"duplicate entry key {key}; one product/horizon/entry appears once, or a "
                f"later row silently shadows an earlier one"
            )
        seen.add(key)

        encoded_label = row["label_value"]
        if not isinstance(encoded_label, str) or encoded_label in _NON_FINITE:
            raise ValueError(f"dataset row {position} has a non-finite label_value")
        row = dict(row)
        row["label_value"] = float.fromhex(encoded_label)
        row["blockers"] = tuple(row["blockers"])

        if key not in expected_candidate_values:
            raise ValueError(
                f"no candidate value for {key}; an endpoint referring to a candidate that "
                f"does not exist is unattributable"
            )
        expected_label = expected_candidate_values[key]
        if float(expected_label).hex() != row["label_value"].hex():
            raise ValueError(
                f"label_value for {key} disagrees with the candidate frame: "
                f"{row['label_value']!r} vs {expected_label!r}"
            )
        records.append(LabelEndpoint(**row))

    missing = sorted(set(expected_candidate_values) - seen)
    dispositions = {"missing_record": len(missing)}
    if missing and require_complete_coverage:
        raise ValueError(
            f"{len(missing)} expected candidate(s) have no record, first {missing[0]}; an "
            f"arbitrary subset is not a complete dataset"
        )

    return LoadedEndpointDataset(
        records=tuple(records),
        coverage_complete=not missing,
        dispositions=dispositions,
    )
