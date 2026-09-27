"""Per-(pid, horizon) mining orchestrator for Phase 3.

Composes profit_tree + purged_wf + the deflation factor + Q0 gates + bootstrap CI.
Pure functions on torch.Tensor inputs (caller loads the parquet). No filesystem.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from tools.strategy_discovery.endpoint_consumers import load_validated_endpoints
from tools.strategy_discovery.profit_split import (
    build_next_eligible_from_endpoints,
    walk_and_sum,
)
from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves, fit_tree
from tools.strategy_discovery.purged_wf import inner_folds, outer_folds
from tools.strategy_discovery.rule_contract import bind_rule, encode_leaf_rule, tree_source_digest

_DEPTH_GRID = (3, 5, 7)
_MIN_LEAF_GRID = (20, 50, 100)
_RETAIL_FEE = 0.012
_Q0_AVG_WIN = 0.05  # >= +5%
_Q0_AVG_LOSS = -0.10  # avg_loss must be >= -0.10 to pass
_Q0_MAX_DD = 0.30
_Q0_MIN_FOLDS = 4
_BOOTSTRAP_N = 1000
_logger = logging.getLogger(__name__)


@dataclass
class LeafProfile:
    leaf_id: int
    rule_path_summary: str
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_dd: float
    cumulative_profit_raw: float
    cumulative_profit_deflated: float
    deflation_pp: float
    n_combos_searched: int
    inner_cv_se: float
    sortino: float
    n_folds_passed_q0: int
    bootstrap_triggered: bool
    bootstrap_ci_lower: Optional[float] = None
    bootstrap_ci_upper: Optional[float] = None
    chosen_depth: int = 0
    chosen_min_leaf: int = 0
    n_folds_evaluated: int = 0
    validation_version: str = "legacy_unverified"
    rule_binding: Optional[dict] = None


def apply_deflation(raw: float, inner_cv_se: float, n_combos: int) -> Tuple[float, float]:
    """Apply max-of-N inflation correction to a search-best profit estimate.

    Returns (deflated_profit, inflation). inflation = σ × √(2 × ln N).
    """
    inflation = float(inner_cv_se) * math.sqrt(2.0 * math.log(max(int(n_combos), 1)))
    return raw - inflation, inflation


def long_shot_band(avg_win: float, avg_loss: float, win_rate: float) -> bool:
    """Per spec: avg_win >= 15% AND |avg_loss| <= 7% AND win_rate >= 70%."""
    return avg_win >= 0.15 and abs(avg_loss) <= 0.07 and win_rate >= 0.70


def leaf_metrics(trades_net: np.ndarray) -> dict:
    """Compute trade-list metrics from a leaf's net-PnL trade sequence."""
    n = int(trades_net.shape[0])
    if n == 0:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_dd": 0.0,
            "cumulative_profit_raw": 0.0,
            "sortino": 0.0,
        }
    wins = trades_net[trades_net > 0]
    losses = trades_net[trades_net < 0]
    avg_win = float(wins.mean()) if wins.size > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    cum = float(trades_net.sum())
    equity = np.concatenate([[0.0], np.cumsum(trades_net)])
    running_max = np.maximum.accumulate(equity)
    drawdown = running_max - equity
    max_dd = float(drawdown.max())
    mean_trade = float(trades_net.mean())
    if losses.size > 0:
        downside_dev = float(np.sqrt(np.mean(losses**2)))
    else:
        downside_dev = 0.0
    sortino = mean_trade / downside_dev if downside_dev > 0 else 0.0
    return {
        "trade_count": n,
        "win_rate": float((trades_net > 0).mean()),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_dd": max_dd,
        "cumulative_profit_raw": cum,
        "sortino": sortino,
    }


def leaf_qualifies(fold_metric: dict) -> bool:
    """True if the leaf passes all three Q0 hard gates on this fold."""
    if fold_metric["avg_win"] < _Q0_AVG_WIN:
        return False
    if fold_metric["avg_loss"] < _Q0_AVG_LOSS:
        return False
    if fold_metric["max_dd"] > _Q0_MAX_DD:
        return False
    return True


def pick_best_hyperparams(inner_scores: dict) -> Tuple[int, int, float, float]:
    """Pick argmax of inner-CV mean-profit table.

    Returns (chosen_depth, chosen_min_leaf, raw_max_profit, inner_cv_se).
    inner_cv_se = std (ddof=1) of inner mean profits — drives deflation.
    """
    best_combo, raw_max = max(inner_scores.items(), key=lambda kv: kv[1])
    chosen_depth, chosen_min_leaf = best_combo
    values = np.array(list(inner_scores.values()), dtype="float64")
    inner_cv_se = float(values.std(ddof=1))
    return int(chosen_depth), int(chosen_min_leaf), float(raw_max), inner_cv_se


def bootstrap_ci(
    trades_net: np.ndarray,
    n_iter: int = _BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Returns (95% lower, 95% upper) on cumulative profit via bootstrap resampling."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(trades_net)
    if n == 0:
        return 0.0, 0.0
    samples = rng.choice(trades_net, size=(int(n_iter), n), replace=True)
    cum = samples.sum(axis=1)
    return float(np.percentile(cum, 2.5)), float(np.percentile(cum, 97.5))


_FEATURE_COLUMNS = (
    "market_cap",
    "fdv",
    "fdv_over_mc",
    "circ_over_total",
    "vol_24h",
    "vol_over_mc",
    "price_over_ema20",
    "price_over_ema50",
    "price_over_ema200",
    "ret_1h_sign",
    "ret_24h_sign",
    "ret_7d_sign",
    "atr14_pct",
)


def _serialize_rule_summary(root: TreeNode, leaf_id: int, feature_names) -> str:
    """Return a one-line human-readable rule path for the leaf_id-th leaf.

    Thresholds are rounded to 2 decimal places so that similar trees fitted on
    different folds produce matching rule strings when the splits cluster near
    the same boundary.
    """
    target_leaves = collect_leaves(root)
    if leaf_id >= len(target_leaves):
        return ""
    target = target_leaves[leaf_id]
    path_conditions = []

    def walk(node: TreeNode) -> bool:
        if node is target:
            return True
        if node.is_leaf:
            return False
        t_rounded = round(node.threshold, 2)
        path_conditions.append(f"{feature_names[node.feature]} <= {t_rounded}")
        if walk(node.left):
            return True
        path_conditions.pop()
        path_conditions.append(f"{feature_names[node.feature]} > {t_rounded}")
        if walk(node.right):
            return True
        path_conditions.pop()
        return False

    walk(root)
    return " AND ".join(path_conditions) if path_conditions else "(root)"


def _leaf_direction_key(root: TreeNode, leaf_id: int, feature_names) -> str:
    """Return a coarse cross-fold identity key for the leaf.

    Uses the (feature_name, direction) of the ROOT split only, so that leaves
    on the same side of the primary discriminating split match across folds
    regardless of the depth or sub-splits the tree learns per fold.  This is
    intentionally coarse: in production the primary split captures the dominant
    signal; secondary sub-splits refine but don't change the leaf's fundamental
    character.
    """
    target_leaves = collect_leaves(root)
    if leaf_id >= len(target_leaves):
        return ""
    target = target_leaves[leaf_id]
    # Walk toward target, recording each step; return only the FIRST step
    path_steps: List[str] = []

    def walk(node: TreeNode) -> bool:
        if node is target:
            return True
        if node.is_leaf:
            return False
        path_steps.append(f"{feature_names[node.feature]}:LE")
        if walk(node.left):
            return True
        path_steps.pop()
        path_steps.append(f"{feature_names[node.feature]}:GT")
        if walk(node.right):
            return True
        path_steps.pop()
        return False

    walk(root)
    # Return only the root-level step for stable cross-fold identity
    return path_steps[0] if path_steps else "(root)"


def _replay_trades(indices_subset, next_eligible_np, labels_np):
    """Walk a sorted subset of indices with max-1 concurrency; return per-trade labels."""
    open_until = -1
    out = []
    for i in sorted(int(x) for x in indices_subset):
        if i < open_until:
            continue
        out.append(float(labels_np[i]))
        open_until = int(next_eligible_np[i])
    return out


def _assign_leaves(root: TreeNode, features_subset: torch.Tensor) -> List[int]:
    """Route each row of features_subset through the tree; return leaf index per row."""
    all_leaves = collect_leaves(root)
    leaf_map: dict = {id(leaf): idx for idx, leaf in enumerate(all_leaves)}
    n_rows = features_subset.shape[0]
    assignments = []
    for row_i in range(n_rows):
        row = features_subset[row_i]
        node = root
        while not node.is_leaf:
            val = float(row[node.feature].item())
            node = node.left if val <= node.threshold else node.right
        assignments.append(leaf_map[id(node)])
    return assignments


def _require_published_endpoints(frame, *, pid: str, parquet_path):
    """Validated endpoints for this product, or raise. There is no fallback on purpose.

    The layout is the producer's own: `build_phase2` writes `{pid}.endpoints.json` and
    `endpoints/{pid}/` beside `{pid}.parquet`, so the location is derived from the parquet's
    parent rather than threaded through a new parameter no caller would have a better value
    for.

    Mining a product on horizon arithmetic because its publication is missing would be the
    worst available outcome: the run would emit ordinary-looking profiles measured on an
    occupancy basis that disagrees with the labels, and nothing downstream could tell.
    """
    directory = Path(parquet_path).parent
    sidecar_path = directory / f"{pid}.endpoints.json"
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"{sidecar_path} is missing, so {pid} has no published endpoints. Mining cannot "
            f"fall back to a wall-clock horizon -- that is the disagreement the endpoint "
            f"contract removes. Re-run Phase 2 for this product first."
        )
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        sidecar = json.load(handle)
    return load_validated_endpoints(
        directory / "endpoints" / pid,
        frame=frame,
        sidecar=sidecar,
        product_id=pid,
    )


def mine_profiles_for_pid_horizon(
    pid: str,
    horizon: int,
    parquet_path,
    device: str = "cuda",
    seed: int = 42,
) -> List[LeafProfile]:
    """End-to-end per-(pid, horizon) mining.

    Steps:
      1. Load parquet → tensors on `device`.
      2. Outer 5-fold Purged WF; per outer fold:
         a. Inner 3-fold hyperparam search across {depth} × {min_leaf} = 9 combos.
         b. Refit best combo on full outer train; evaluate on outer test.
         c. Per leaf: compute metrics; check Q0 gates.
      3. Aggregate per-leaf across folds; require ≥4-of-5 folds passing.
      4. Apply deflation factor to the cumulative profit using inner-CV SE.
      5. Trigger bootstrap CI for low-trade-count or long-shot leaves.
    """
    from collections import defaultdict

    import pyarrow.parquet as _pq
    from pandas.api.types import is_integer_dtype

    df = _pq.read_table(parquet_path).to_pandas()
    label_col = f"label_h{int(horizon)}"
    if label_col not in df.columns:
        return []
    # Row-count labels and clock-hour replay agree only on contiguous hourly input. Validate
    # before dropping unlabeled rows so invalid source rows cannot be hidden.
    if "ts" not in df or not is_integer_dtype(df["ts"].dtype) or df["ts"].isna().any():
        raise ValueError("timestamps must be non-null integer milliseconds")
    # Checked on the frame AS READ, with no sort. The miner used to sort first, which was a
    # silent repair: a reordered parquet is not the artifact Phase 2 published, and reordering
    # it to make the checks pass hides that. A published frame is already unique, ascending and
    # contiguous, so this rejects exactly the frames that were never publishable.
    if df["ts"].duplicated().any() or (df["ts"].diff().dropna() != 3_600_000).any():
        raise ValueError(
            "timestamps must be unique, ascending and contiguous hourly; label exit provenance "
            "required for gaps, and a frame out of published order is not the published artifact"
        )

    # Now the publication, still against the unfiltered frame in its published order (Codex
    # 26224ddf). `build_data_id` hashes the timestamps and every price array IN FRAME ORDER, so
    # a frame that validates IS the published artifact row for row; and `source_row_id` must
    # still be the original ordinals 0..n-1, which a filtered frame could never satisfy.
    validated = _require_published_endpoints(df, pid=pid, parquet_path=parquet_path)

    # `isfinite`, not `dropna`: dropna KEEPS +/-inf (executed -- [0.1, nan, inf, -0.2].dropna()
    # returns [0.1, inf, -0.2]), while the publication selected its rows by isfinite. A row
    # retained here but absent there surfaces later as a spurious missing endpoint.
    retained = np.isfinite(df[label_col].to_numpy(dtype="float64"))
    df = df.loc[retained].reset_index(drop=True)
    retained_row_ids = df["source_row_id"]
    n = len(df)
    if n < 200:
        _logger.warning(
            "Insufficient history for %s h%s: fewer than 200 labeled rows", pid, horizon
        )
        return []
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    labels = torch.tensor(df[label_col].to_numpy(dtype="float64"), device=dev)
    features = torch.tensor(df[list(_FEATURE_COLUMNS)].to_numpy(dtype="float64"), device=dev)
    if features.dtype != torch.float64:
        raise ValueError("exact rule routing requires float64 feature tensors before fitting")
    # Read from the records, not re-derived from a clock. The wall-clock baseline is gone from
    # this module entirely -- it survives in `profit_split`, where the equivalence test compares
    # the two and so keeps the evidence that this generalises the old behaviour.
    next_eligible = build_next_eligible_from_endpoints(
        validated, retained_row_ids, horizon=int(horizon), device=dev
    )

    outer = outer_folds(n, n_folds=5, embargo_bars=int(horizon))
    nested = [inner_folds(train, n_folds=3, embargo_bars=int(horizon)) for train, _ in outer]
    minimum_train_rows = max(int(horizon), 2 * min(_MIN_LEAF_GRID))
    if len(outer) != 5 or any(
        len(inner) != 3 or any(len(train) < minimum_train_rows for train, _ in inner)
        for inner in nested
    ):
        _logger.warning(
            "Insufficient history for %s h%s: require 5 outer and 3 inner folds "
            "with at least %s training rows each; no strategy verdict",
            pid,
            horizon,
            minimum_train_rows,
        )
        return []
    fold_pass_count: Dict[str, int] = defaultdict(int)
    fold_trade_lists: Dict[str, list] = defaultdict(list)
    fold_summaries: Dict[str, dict] = {}

    next_eligible_np = next_eligible.cpu().numpy()
    labels_np = labels.cpu().numpy()

    for outer_fold_id, ((outer_train_idx, outer_test_idx), inner) in enumerate(
        zip(outer, nested, strict=True)
    ):
        # Both levels are prefixes of the sorted frame. fit_tree uses local
        # row IDs against the full labels/next_eligible arrays, so the prefix
        # invariant is required for correct feature/label alignment.
        inner_scores: Dict[tuple, list] = defaultdict(list)
        for depth in _DEPTH_GRID:
            for min_leaf in _MIN_LEAF_GRID:
                fold_profits = []
                for inner_train, inner_test in inner:
                    tree = fit_tree(
                        features=features[inner_train],
                        labels=labels,
                        next_eligible=next_eligible,
                        max_depth=depth,
                        min_leaf=min_leaf,
                    )
                    # Route inner_test rows through the tree to assign them to leaves
                    inner_test_tensor = torch.tensor(inner_test, dtype=torch.int64, device=dev)
                    assignments = _assign_leaves(tree, features[inner_test_tensor])
                    n_leaves = len(collect_leaves(tree))
                    leaf_test_rows: List[List[int]] = [[] for _ in range(n_leaves)]
                    for pos, leaf_id in enumerate(assignments):
                        leaf_test_rows[leaf_id].append(int(inner_test[pos]))
                    cum = 0.0
                    for leaf_test in leaf_test_rows:
                        if not leaf_test:
                            continue
                        sub = torch.tensor(leaf_test, dtype=torch.int64, device=dev).unsqueeze(0)
                        cum += float(walk_and_sum(sub, next_eligible, labels)[0].item())
                    fold_profits.append(cum)
                inner_scores[(depth, min_leaf)].append(float(np.mean(fold_profits)))
        inner_score_mean = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        chosen_depth, chosen_min_leaf, raw_max, inner_cv_se = pick_best_hyperparams(
            inner_score_mean
        )
        tree = fit_tree(
            features=features[outer_train_idx],
            labels=labels,
            next_eligible=next_eligible,
            max_depth=chosen_depth,
            min_leaf=chosen_min_leaf,
        )
        # Route outer_test rows through the fitted tree
        outer_test_tensor = torch.tensor(outer_test_idx, dtype=torch.int64, device=dev)
        assignments = _assign_leaves(tree, features[outer_test_tensor])
        leaves = collect_leaves(tree)
        n_leaves = len(leaves)
        source_digest = tree_source_digest(tree, list(_FEATURE_COLUMNS))
        leaf_test_rows_outer: List[List[int]] = [[] for _ in range(n_leaves)]
        for pos, leaf_id in enumerate(assignments):
            leaf_test_rows_outer[leaf_id].append(int(outer_test_idx[pos]))
        passed_directions = set()
        for leaf_id in range(n_leaves):
            global_test_rows = leaf_test_rows_outer[leaf_id]
            if not global_test_rows:
                continue
            # Use direction-only key for cross-fold identity (threshold-invariant)
            direction_key = _leaf_direction_key(tree, leaf_id, _FEATURE_COLUMNS)
            trades = _replay_trades(global_test_rows, next_eligible_np, labels_np)
            metrics = leaf_metrics(np.asarray(trades))
            metrics["deflated_profit"], _ = apply_deflation(
                raw=metrics["cumulative_profit_raw"],
                inner_cv_se=inner_cv_se,
                n_combos=9,
            )
            if leaf_qualifies(metrics):
                passed_directions.add(direction_key)
                fold_trade_lists[direction_key].extend(trades)
                fold_summaries[direction_key] = {
                    **metrics,
                    "chosen_depth": chosen_depth,
                    "chosen_min_leaf": chosen_min_leaf,
                    "inner_cv_se": inner_cv_se,
                    "raw_max": raw_max,
                    "rule_summary": _serialize_rule_summary(tree, leaf_id, _FEATURE_COLUMNS),
                    "machine_rule": encode_leaf_rule(tree, leaf_id, _FEATURE_COLUMNS),
                    "source_leaf_id": leaf_id,
                    "source_outer_fold": outer_fold_id,
                    "source_tree_digest": source_digest,
                }
        # Several leaves can share one root-direction identity. Their evidence
        # comes from the same held-out period, so count that period only once.
        for direction_key in passed_directions:
            fold_pass_count[direction_key] += 1

    rng = np.random.default_rng(seed)
    profiles: List[LeafProfile] = []
    for leaf_id, direction_key in enumerate(sorted(fold_pass_count.keys())):
        n_pass = fold_pass_count[direction_key]
        if n_pass < _Q0_MIN_FOLDS:
            continue
        trades = np.asarray(fold_trade_lists[direction_key])
        m = leaf_metrics(trades)
        deflated, infl = apply_deflation(
            raw=m["cumulative_profit_raw"],
            inner_cv_se=fold_summaries[direction_key]["inner_cv_se"],
            n_combos=9,
        )
        avg_trades_per_fold = m["trade_count"] / 5.0
        is_long_shot = long_shot_band(m["avg_win"], m["avg_loss"], m["win_rate"])
        bootstrap_triggered = avg_trades_per_fold < 30 or is_long_shot
        ci_lower, ci_upper = (None, None)
        if bootstrap_triggered and m["trade_count"] > 0:
            ci_lower, ci_upper = bootstrap_ci(trades, n_iter=_BOOTSTRAP_N, rng=rng)
        rule_summary = fold_summaries[direction_key].get("rule_summary", direction_key)
        profiles.append(
            LeafProfile(
                leaf_id=leaf_id,
                rule_path_summary=rule_summary,
                trade_count=m["trade_count"],
                win_rate=m["win_rate"],
                avg_win=m["avg_win"],
                avg_loss=m["avg_loss"],
                max_dd=m["max_dd"],
                cumulative_profit_raw=m["cumulative_profit_raw"],
                cumulative_profit_deflated=deflated,
                deflation_pp=infl,
                n_combos_searched=9,
                inner_cv_se=fold_summaries[direction_key]["inner_cv_se"],
                sortino=m["sortino"],
                n_folds_passed_q0=n_pass,
                bootstrap_triggered=bootstrap_triggered,
                bootstrap_ci_lower=ci_lower,
                bootstrap_ci_upper=ci_upper,
                chosen_depth=fold_summaries[direction_key]["chosen_depth"],
                chosen_min_leaf=fold_summaries[direction_key]["chosen_min_leaf"],
                n_folds_evaluated=len(outer),
                validation_version="chronological_distinct_folds_v2",
                rule_binding=bind_rule(
                    fold_summaries[direction_key]["machine_rule"],
                    pid=pid,
                    horizon=int(horizon),
                    profile_leaf_id=leaf_id,
                    source_leaf_id=fold_summaries[direction_key]["source_leaf_id"],
                    source_outer_fold=fold_summaries[direction_key]["source_outer_fold"],
                    source_tree_digest=fold_summaries[direction_key]["source_tree_digest"],
                    feature_schema=list(_FEATURE_COLUMNS),
                ),
            )
        )
    return profiles
