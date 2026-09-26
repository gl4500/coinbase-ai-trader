"""Exact, JSON-safe tree paths. Display summaries are never executable evidence.

Hexadecimal float strings preserve the fitted binary64 thresholds. Right branches
mean the complement of <=, including NaN, matching the existing tree router.
Missing columns are unavailable inputs, distinct from a present NaN value.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral

from tools.strategy_discovery.profit_tree import collect_leaves

RULE_VERSION = "tree_path_hex_v1"
BINDING_VERSION = "profile_rule_binding_v1"


def validate_rule(rule: dict) -> None:
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


def encode_leaf_rule(root, leaf_id: int, feature_names) -> dict:
    leaves = collect_leaves(root)
    if (
        not isinstance(leaf_id, Integral)
        or isinstance(leaf_id, bool)
        or not 0 <= leaf_id < len(leaves)
    ):
        raise ValueError("invalid leaf identity")
    target = leaves[leaf_id]
    path = []

    def visit(node):
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
    rule = {"version": RULE_VERSION, "nan_policy": "right", "conditions": path}
    validate_rule(rule)
    return rule


def rule_matches(rule: dict, values) -> bool:
    validate_rule(rule)
    for clause in rule["conditions"]:
        feature = clause["feature"]
        if feature not in values:
            raise ValueError(f"missing feature column: {feature}")
        try:
            value = float(values[feature])
        except (TypeError, ValueError, OverflowError):
            return False
        goes_left = value <= float.fromhex(clause["threshold_hex"])
        if goes_left != clause["go_left"]:
            return False
    return True


def _is_digest(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _binding_digest(body) -> str:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def tree_source_digest(root, feature_schema) -> str:
    """Fingerprint the tree's ordered leaf paths and the float64 feature schema."""
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


def _validate_binding_body(body) -> None:
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
    rule,
    *,
    pid,
    horizon,
    profile_leaf_id,
    source_leaf_id,
    source_outer_fold,
    source_tree_digest,
    feature_schema,
) -> dict:
    """Bind a representative rule to its group and distinct source-tree identity.

    The digest detects stale/mixed artifacts; it is not a signature, proof of
    training history, or a claim that group metrics measure this rule's returns.
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
    return {**body, "digest": _binding_digest(body)}


def validate_bound_rule(
    payload, *, pid, horizon, profile_leaf_id, expected_digest, feature_schema
) -> dict:
    """Verify sidecar content against the independently stored profile-row digest."""
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
    return json.loads(json.dumps(body["rule"]))
