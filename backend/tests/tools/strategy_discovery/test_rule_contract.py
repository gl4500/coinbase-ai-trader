"""Machine rules must preserve tree routing, independently of display text."""

import json

import pytest
import torch

from tools.strategy_discovery.mine_profiles import _assign_leaves
from tools.strategy_discovery.profit_tree import TreeNode
from tools.strategy_discovery.rule_contract import (
    bind_rule,
    encode_leaf_rule,
    rule_matches,
    tree_source_digest,
    validate_bound_rule,
)


@pytest.mark.parametrize("threshold", [1.0249, 1e-12, -3.123456789012345, 0.0, -0.0])
def test_json_roundtrip_preserves_both_tree_branches(threshold):
    import numpy as np

    tree = TreeNode(feature=0, threshold=threshold, left=TreeNode(), right=TreeNode())
    values = [
        np.nextafter(threshold, -float("inf")),
        threshold,
        np.nextafter(threshold, float("inf")),
        float("nan"),
        -float("inf"),
        float("inf"),
    ]
    assigned = _assign_leaves(tree, torch.tensor([[x] for x in values], dtype=torch.float64))
    rules = [json.loads(json.dumps(encode_leaf_rule(tree, leaf, ["x"]))) for leaf in (0, 1)]
    for value, expected in zip(values, assigned, strict=True):
        assert [leaf for leaf, rule in enumerate(rules) if rule_matches(rule, {"x": value})] == [
            expected
        ]


def test_root_is_explicit_and_invalid_leaf_is_not_unconditional():
    root = TreeNode()
    rule = encode_leaf_rule(root, 0, ["x"])
    assert rule["conditions"] == []
    assert rule_matches(rule, {})
    for invalid in (-1, 1, True, 0.5):
        with pytest.raises(ValueError, match="leaf"):
            encode_leaf_rule(root, invalid, ["x"])


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_split_threshold_is_rejected(threshold):
    tree = TreeNode(feature=0, threshold=threshold, left=TreeNode(), right=TreeNode())
    with pytest.raises(ValueError, match="threshold"):
        encode_leaf_rule(tree, 0, ["x"])


def test_missing_column_does_not_impersonate_nan():
    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    right = encode_leaf_rule(tree, 1, ["x"])
    assert rule_matches(right, {"x": float("nan")})
    with pytest.raises(ValueError, match="missing feature"):
        rule_matches(right, {})


@pytest.mark.parametrize("mutation", ["version", "policy", "threshold", "direction", "feature"])
def test_malformed_machine_rule_cannot_execute(mutation):
    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    rule = encode_leaf_rule(tree, 0, ["x"])
    if mutation == "version":
        rule["version"] = "legacy_rounded"
    elif mutation == "policy":
        rule["nan_policy"] = "left"
    elif mutation == "threshold":
        rule["conditions"][0]["threshold_hex"] = "nan"
    elif mutation == "direction":
        rule["conditions"][0]["go_left"] = 1
    else:
        rule["conditions"][0]["feature"] = ""
    with pytest.raises(ValueError):
        rule_matches(rule, {"x": 1.0})


def _bound_rule():
    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    return bind_rule(
        encode_leaf_rule(tree, 1, ["x"]),
        pid="BTC-USD",
        horizon=24,
        profile_leaf_id=3,
        source_leaf_id=1,
        source_outer_fold=4,
        source_tree_digest="a" * 64,
        feature_schema=["x", "y"],
    )


def _validate_payload(payload, **overrides):
    expected = dict(
        pid="BTC-USD",
        horizon=24,
        profile_leaf_id=3,
        expected_digest=payload["digest"],
        feature_schema=["x", "y"],
    )
    expected.update(overrides)
    return validate_bound_rule(payload, **expected)


def test_bound_rule_roundtrip_preserves_source_leaf_distinct_from_group():
    payload = json.loads(json.dumps(_bound_rule()))
    assert payload["profile_leaf_id"] == 3
    assert payload["source_leaf_id"] == 1
    rule = _validate_payload(payload)
    assert rule_matches(rule, {"x": float("nan")})
    assert _bound_rule()["digest"] == payload["digest"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"pid": "ETH-USD"},
        {"horizon": 1},
        {"profile_leaf_id": 1},
        {"expected_digest": "b" * 64},
        {"expected_digest": None},
        {"feature_schema": ["y", "x"]},
        {"feature_schema": ["x"]},
    ],
)
def test_wrong_or_stale_sidecar_is_rejected(overrides):
    with pytest.raises(ValueError):
        _validate_payload(_bound_rule(), **overrides)


@pytest.mark.parametrize(
    "field,value",
    [
        ("horizon", True),
        ("horizon", 24.5),
        ("source_leaf_id", -1),
        ("source_outer_fold", 5),
        ("source_outer_fold", True),
        ("source_tree_digest", "unknown"),
        ("version", "legacy"),
    ],
)
def test_bound_metadata_corruption_is_rejected(field, value):
    payload = _bound_rule()
    payload[field] = value
    with pytest.raises(ValueError):
        _validate_payload(payload)


def test_changed_machine_rule_cannot_keep_old_digest():
    payload = _bound_rule()
    payload["rule"]["conditions"][0]["threshold_hex"] = float(1.02).hex()
    with pytest.raises(ValueError, match="digest"):
        _validate_payload(payload)


def test_tree_digest_binds_exact_thresholds_and_ordered_schema():
    tree = TreeNode(feature=0, threshold=1.0249, left=TreeNode(), right=TreeNode())
    original = tree_source_digest(tree, ["x", "y"])
    assert original == tree_source_digest(tree, ["x", "y"])
    assert original != tree_source_digest(tree, ["y", "x"])
    tree.threshold = 1.02
    assert original != tree_source_digest(tree, ["x", "y"])
    with pytest.raises(ValueError, match="schema"):
        tree_source_digest(tree, ["x", "x"])


def test_generated_tree_rules_partition_rows_like_actual_router():
    import numpy as np

    from tools.strategy_discovery.profit_tree import collect_leaves

    rng = np.random.default_rng(824)
    names = ["x", "y", "z"]
    for _ in range(10):
        boundaries = []

        def make_tree(depth):
            if depth == 0 or rng.random() < 0.2:
                return TreeNode()
            feature = int(rng.integers(3))
            threshold = float(rng.normal())
            boundaries.append((feature, threshold))
            return TreeNode(
                feature=feature,
                threshold=threshold,
                left=make_tree(depth - 1),
                right=make_tree(depth - 1),
            )

        tree = make_tree(4)
        rows = list(rng.normal(size=(20, 3)))
        for feature, threshold in boundaries:
            for value in [
                threshold,
                np.nextafter(threshold, -np.inf),
                np.nextafter(threshold, np.inf),
            ]:
                row = np.zeros(3)
                row[feature] = value
                rows.append(row)
        for feature in range(3):
            for value in [np.nan, np.inf, -np.inf]:
                row = np.zeros(3)
                row[feature] = value
                rows.append(row)
        assigned = _assign_leaves(tree, torch.tensor(np.asarray(rows), dtype=torch.float64))
        rules = [encode_leaf_rule(tree, leaf, names) for leaf in range(len(collect_leaves(tree)))]
        for row, expected in zip(rows, assigned, strict=True):
            values = dict(zip(names, row, strict=True))
            matches = [leaf for leaf, rule in enumerate(rules) if rule_matches(rule, values)]
            assert matches == [expected]
