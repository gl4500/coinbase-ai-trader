"""Exact, JSON-safe tree paths. Display summaries are never executable evidence.

Hexadecimal float strings preserve the fitted binary64 thresholds. Right branches
mean the complement of <=, including NaN, matching the existing tree router.
Missing columns are unavailable inputs, distinct from a present NaN value.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Protocol, TypedDict, cast

from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves

RULE_VERSION = "tree_path_hex_v1"
BINDING_VERSION = "profile_rule_binding_v1"


class RuleCondition(TypedDict):
    """One split on the path: go left when ``value <= float.fromhex(threshold_hex)``."""

    feature: str
    go_left: bool
    threshold_hex: str


class MachineRule(TypedDict):
    """An executable leaf path. ``nan_policy`` is always ``"right"``."""

    version: str
    nan_policy: str
    conditions: list[RuleCondition]


class BoundRule(TypedDict):
    """A rule bound to its profile group and source tree, plus the binding digest."""

    version: str
    pid: str
    horizon: int
    profile_leaf_id: int
    source_leaf_id: int
    source_outer_fold: int
    source_tree_digest: str
    feature_schema: list[str]
    rule: MachineRule
    digest: str


class FeatureValues(Protocol):
    """Values looked up by column name: a dict or a pandas Series row both qualify."""

    def __contains__(self, key: object, /) -> bool: ...

    def __getitem__(self, key: str, /) -> object: ...


def validate_rule(rule: object) -> None:
    """Accept only a well-formed ``MachineRule``; return None, raise otherwise.

    Required: a dict with exactly ``version`` (== RULE_VERSION), ``nan_policy``
    (== "right") and ``conditions`` (a list, possibly empty). Each condition is a
    dict with exactly a non-blank str ``feature``, a bool ``go_left`` and a
    ``threshold_hex`` that parses to a finite binary64 written canonically
    (``float.hex`` round trip). An empty condition list is the unconditional root.

    Raises:
        ValueError: on any other shape, version, policy, type, unparseable,
            non-finite or non-canonical threshold.
    """
    if (
        not isinstance(rule, dict)
        or set(rule) != {"version", "nan_policy", "conditions"}
        or rule.get("version") != RULE_VERSION
        or rule.get("nan_policy") != "right"
        or not isinstance(rule.get("conditions"), list)
    ):
        raise ValueError("invalid machine rule version or routing policy")
    for clause in rule["conditions"]:
        if (
            not isinstance(clause, dict)
            or set(clause) != {"feature", "go_left", "threshold_hex"}
            or not isinstance(clause.get("feature"), str)
            or not clause["feature"].strip()
            or type(clause.get("go_left")) is not bool
            or not isinstance(clause.get("threshold_hex"), str)
        ):
            raise ValueError("invalid machine rule condition")
        try:
            threshold = float.fromhex(clause["threshold_hex"])
        except (ValueError, OverflowError) as exc:
            raise ValueError("invalid machine rule threshold") from exc
        if not math.isfinite(threshold) or threshold.hex() != clause["threshold_hex"]:
            raise ValueError("threshold must be finite canonical binary64")


def encode_leaf_rule(root: TreeNode, leaf_id: int, feature_names: Sequence[str]) -> MachineRule:
    """Return the exact root-to-leaf path for ``collect_leaves(root)[leaf_id]``.

    ``leaf_id`` is a non-bool integer index into ``collect_leaves(root)``.
    ``feature_names[i]`` names split feature ``i``. Each split contributes one
    condition with the threshold as canonical hex; the root leaf yields an empty
    condition list. The result passes ``validate_rule``.

    Raises:
        ValueError: if ``leaf_id`` is not a valid leaf index, a split feature is
            not a non-bool integer index into ``feature_names``, a split threshold
            is not convertible to a finite float, or the leaf is unreachable.
    """
    leaves = collect_leaves(root)
    if (
        not isinstance(leaf_id, Integral)
        or isinstance(leaf_id, bool)
        or not 0 <= leaf_id < len(leaves)
    ):
        raise ValueError("invalid leaf identity")
    target = leaves[leaf_id]
    path: list[RuleCondition] = []

    def visit(node: TreeNode) -> bool:
        if node is target:
            return True
        if node.is_leaf:
            return False
        if (
            not isinstance(node.feature, Integral)
            or isinstance(node.feature, bool)
            or not 0 <= node.feature < len(feature_names)
        ):
            raise ValueError("invalid split feature")
        try:
            threshold = float(node.threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid split threshold") from exc
        if not math.isfinite(threshold):
            raise ValueError("nonfinite split threshold")
        for go_left, child in ((True, node.left), (False, node.right)):
            path.append(
                {
                    "feature": feature_names[node.feature],
                    "go_left": go_left,
                    "threshold_hex": threshold.hex(),
                }
            )
            if child is not None and visit(child):
                return True
            path.pop()
        return False

    if not visit(root):
        raise ValueError("unresolved leaf identity")
    rule: MachineRule = {"version": RULE_VERSION, "nan_policy": "right", "conditions": path}
    validate_rule(rule)
    return rule


def rule_matches(rule: object, values: FeatureValues) -> bool:
    """Return True when ``values`` follows every condition of ``rule``.

    ``rule`` must pass ``validate_rule``. A condition goes left when
    ``float(value) <= threshold``, so NaN routes right, matching the tree router.
    A present value that cannot be converted to float makes the rule not match
    (returns False). The empty (root) rule matches every row.

    Raises:
        ValueError: if ``rule`` is invalid, or a condition's feature column is
            absent from ``values`` (a missing column is not a NaN).
    """
    validate_rule(rule)
    for clause in cast(MachineRule, rule)["conditions"]:
        feature = clause["feature"]
        if feature not in values:
            raise ValueError(f"missing feature column: {feature}")
        try:
            value = float(values[feature])  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return False
        goes_left = value <= float.fromhex(clause["threshold_hex"])
        if goes_left != clause["go_left"]:
            return False
    return True


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _binding_digest(body: Mapping[str, object]) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tree_source_digest(root: TreeNode, feature_schema: list[str]) -> str:
    """Fingerprint the tree's ordered leaf paths and the float64 feature schema.

    Required: ``feature_schema`` is a non-empty list of unique, non-blank str.
    Returns a 64-char lowercase hex SHA-256 over the rule version, dtype, schema
    and ``encode_leaf_rule`` for every leaf in ``collect_leaves`` order.

    Raises:
        ValueError: on an invalid schema, or anything ``encode_leaf_rule`` rejects.
    """
    if (
        not isinstance(feature_schema, list)
        or not feature_schema
        or any(not isinstance(name, str) or not name.strip() for name in feature_schema)
        or len(set(feature_schema)) != len(feature_schema)
    ):
        raise ValueError("invalid tree feature schema")
    return _binding_digest(
        {
            "version": RULE_VERSION,
            "feature_dtype": "float64",
            "feature_schema": feature_schema,
            "leaf_paths": [
                encode_leaf_rule(root, leaf, feature_schema)
                for leaf in range(len(collect_leaves(root)))
            ],
        }
    )


def _validate_binding_body(body: object) -> None:
    required = {
        "version",
        "pid",
        "horizon",
        "profile_leaf_id",
        "source_leaf_id",
        "source_outer_fold",
        "source_tree_digest",
        "feature_schema",
        "rule",
    }
    if not isinstance(body, dict) or set(body) != required or body["version"] != BINDING_VERSION:
        raise ValueError("invalid rule binding version or fields")
    if not isinstance(body["pid"], str) or not body["pid"] or body["pid"].strip() != body["pid"]:
        raise ValueError("invalid bound product identity")
    for field in ("horizon", "profile_leaf_id", "source_leaf_id", "source_outer_fold"):
        if type(body[field]) is not int or body[field] < 0:
            raise ValueError(f"invalid bound {field}")
    if body["horizon"] == 0 or body["source_outer_fold"] >= 5:
        raise ValueError("invalid bound horizon or source fold")
    if not _is_digest(body["source_tree_digest"]):
        raise ValueError("invalid source tree digest")
    validate_rule(body["rule"])
    schema = body["feature_schema"]
    if (
        not isinstance(schema, list)
        or not schema
        or any(not isinstance(name, str) or not name.strip() for name in schema)
        or len(set(schema)) != len(schema)
        or any(clause["feature"] not in schema for clause in body["rule"]["conditions"])
    ):
        raise ValueError("invalid bound feature schema")


def bind_rule(
    rule: MachineRule,
    *,
    pid: str,
    horizon: int,
    profile_leaf_id: int,
    source_leaf_id: int,
    source_outer_fold: int,
    source_tree_digest: str,
    feature_schema: list[str],
) -> BoundRule:
    """Bind a representative rule to its group and distinct source-tree identity.

    The digest detects stale/mixed artifacts; it is not a signature, proof of
    training history, or a claim that group metrics measure this rule's returns.

    Required: ``rule`` passes ``validate_rule``; ``pid`` is a non-empty str with
    no surrounding whitespace; ``horizon`` > 0 and the leaf/fold ids >= 0 are
    plain int (not bool); ``source_outer_fold`` < 5; ``source_tree_digest`` is
    64-char lowercase hex; ``feature_schema`` is a non-empty list of unique,
    non-blank str containing every rule feature. Returns a JSON-normalized copy
    with ``digest`` = SHA-256 of the canonical JSON body.

    Raises:
        ValueError: if any of the above does not hold.
    """
    body = dict(
        version=BINDING_VERSION,
        pid=pid,
        horizon=horizon,
        profile_leaf_id=profile_leaf_id,
        source_leaf_id=source_leaf_id,
        source_outer_fold=source_outer_fold,
        source_tree_digest=source_tree_digest,
        feature_schema=feature_schema,
        rule=rule,
    )
    _validate_binding_body(body)
    body = json.loads(json.dumps(body, allow_nan=False))
    return cast(BoundRule, {**body, "digest": _binding_digest(body)})


def validate_bound_rule(
    payload: object,
    *,
    pid: str,
    horizon: int,
    profile_leaf_id: int,
    expected_digest: str,
    feature_schema: list[str],
) -> MachineRule:
    """Verify sidecar content against the independently stored profile-row digest.

    Required: ``payload`` is a ``BoundRule``-shaped dict whose body satisfies the
    ``bind_rule`` contract; its ``digest``, the recomputed body digest and
    ``expected_digest`` (from the profile row) are all equal; and its pid,
    horizon, profile_leaf_id and feature_schema equal the arguments (``horizon``
    and ``profile_leaf_id`` must be plain int). Returns a fresh copy of the bound
    ``MachineRule``.

    Raises:
        ValueError: on a missing or malformed digest, an invalid body, a digest
            mismatch, or a profile identity mismatch.
    """
    if not isinstance(payload, dict) or not _is_digest(payload.get("digest")):
        raise ValueError("missing rule binding digest")
    body = {key: value for key, value in payload.items() if key != "digest"}
    _validate_binding_body(body)
    if (
        not _is_digest(expected_digest)
        or payload["digest"] != expected_digest
        or _binding_digest(body) != expected_digest
    ):
        raise ValueError("rule binding digest mismatch")
    if (
        body["pid"] != pid
        or type(horizon) is not int
        or body["horizon"] != horizon
        or type(profile_leaf_id) is not int
        or body["profile_leaf_id"] != profile_leaf_id
        or body["feature_schema"] != feature_schema
    ):
        raise ValueError("rule binding profile identity mismatch")
    return cast(MachineRule, json.loads(json.dumps(body["rule"])))
