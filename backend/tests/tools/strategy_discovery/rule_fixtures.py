"""Explicit machine-rule fixtures; never used to migrate real legacy artifacts."""

import json
from typing import List, Tuple

import pandas as pd

from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS
from tools.strategy_discovery.rule_contract import BINDING_VERSION, RULE_VERSION, bind_rule


def parse_rule_path(rule_path: str) -> List[Tuple[str, str, float]]:
    """Parse 'feat_a > 1.02 AND feat_b <= 0.08' into [(feature, op, threshold), ...].

    Operators supported: >, <, >=, <=. Only explicit '(root)' is unconditional.
    """
    if not isinstance(rule_path, str) or not rule_path.strip():
        raise ValueError("unresolved rule cannot be simulated")
    if rule_path.strip() == "(root)":
        return []
    conditions: List[Tuple[str, str, float]] = []
    for clause in rule_path.split(" AND "):
        clause = clause.strip()
        for op in (">=", "<=", ">", "<"):
            if f" {op} " in clause:
                feature, threshold_str = clause.split(f" {op} ", 1)
                conditions.append((feature.strip(), op, float(threshold_str.strip())))
                break
        else:
            raise ValueError(f"unparseable rule clause: {clause!r}")
    return conditions


def machine_rule_fixture(text):
    if not isinstance(text, str) or not text.strip():
        return None
    conditions = []
    for feature, op, threshold in parse_rule_path(text):
        if op not in ("<=", ">"):
            raise ValueError("fixture requires tree-compatible comparison")
        conditions.append(
            {"feature": feature, "go_left": op == "<=", "threshold_hex": threshold.hex()}
        )
    return {"version": RULE_VERSION, "nan_policy": "right", "conditions": conditions}


def write_bound_sidecar_fixture(path, mapping):
    horizon = int(path.stem.rsplit("h", 1)[1])
    parquet = path.with_name(f"profiles_h{horizon}.parquet")
    frame = pd.read_parquet(parquet)
    output = {}
    for key, text in mapping.items():
        if not isinstance(text, str) or not text.strip():
            output[key] = text
            continue
        pid, leaf = key.rsplit("__", 1)
        leaf = int(leaf)
        binding = bind_rule(
            machine_rule_fixture(text),
            pid=pid,
            horizon=horizon,
            profile_leaf_id=leaf,
            source_leaf_id=leaf,
            source_outer_fold=4,
            source_tree_digest="a" * 64,
            feature_schema=list(_FEATURE_COLUMNS),
        )
        output[key] = binding
        matching = frame["pid"].eq(pid) & frame["leaf_id"].eq(leaf)
        frame.loc[matching, "rule_digest"] = binding["digest"]
        frame.loc[matching, "rule_version"] = BINDING_VERSION
    frame.to_parquet(parquet, index=False)
    path.write_text(json.dumps(output), encoding="utf-8")
