"""Explicit machine-rule fixtures; never used to migrate real legacy artifacts."""

import json

import pandas as pd

from tools.strategy_discovery.mine_profiles import _FEATURE_COLUMNS
from tools.strategy_discovery.portfolio_sim import parse_rule_path
from tools.strategy_discovery.rule_contract import BINDING_VERSION, RULE_VERSION, bind_rule


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
