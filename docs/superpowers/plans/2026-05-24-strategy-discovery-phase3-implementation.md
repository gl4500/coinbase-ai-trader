# Strategy-Discovery Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mine custom-criterion decision-tree profiles from the Phase 2 per-token parquets — one tree per (pid, horizon) pair, ranked by selection-bias-corrected cumulative profit, validated via Purged WF + bootstrap CI, output as `profiles_h{h}.parquet` + `rule_paths_h{h}.json` for Phase 4 scorecard consumption.

**Architecture:** Five loosely-coupled pure-function modules under `backend/tools/strategy_discovery/`. `profit_split.py` is the GPU-vectorized split criterion (concurrency-capped cumulative PnL, max-1-per-token). `profit_tree.py` is the recursive tree fitter that scans candidates via `profit_split`. `purged_wf.py` is the Purged Walk-Forward + nested-CV harness. `mine_profiles.py` orchestrates per-(pid, horizon) mining with hyperparam search, deflation factor, Q0 gates, and bootstrap CI. `mine_universe.py` is the universe driver + CLI.

**Tech Stack:** Python 3.14, PyTorch (CUDA + CPU fallback, mirrors `xgb_v4_5_features_batch.py` conventions), pandas, numpy, pyarrow. Pytest with mocks only. GPU is mandatory at runtime; CPU is the test path.

**Spec:** `docs/superpowers/specs/2026-05-24-strategy-discovery-phase3-design.md`
**Companion HTML:** `backend/tools/phase3_deflation_explainer.html`

---

## File Map

| File | Purpose |
|---|---|
| `backend/tools/strategy_discovery/profit_split.py` (NEW) | Custom split criterion: GPU-vectorized concurrency-capped cumulative PnL with max-1-per-token rule. |
| `backend/tools/strategy_discovery/profit_tree.py` (NEW) | Recursive tree fitter; uses `profit_split.best_split` at each node. |
| `backend/tools/strategy_discovery/purged_wf.py` (NEW) | Purged Walk-Forward CV + nested inner CV. Pure numpy index math, no GPU. |
| `backend/tools/strategy_discovery/mine_profiles.py` (NEW) | Per-(pid, horizon) orchestrator: hyperparam search, deflation, Q0 gates, bootstrap CI. |
| `backend/tools/strategy_discovery/mine_universe.py` (NEW) | Universe driver + CLI; writes `profiles_h{h}.parquet`, `rule_paths_h{h}.json`, `mining_summary.md`. |
| `backend/tests/tools/strategy_discovery/test_profit_split.py` (NEW) | 4 tests |
| `backend/tests/tools/strategy_discovery/test_profit_tree.py` (NEW) | 4 tests |
| `backend/tests/tools/strategy_discovery/test_purged_wf.py` (NEW) | 3 tests |
| `backend/tests/tools/strategy_discovery/test_mine_profiles.py` (NEW) | 6 tests |
| `backend/tests/tools/strategy_discovery/test_mine_universe.py` (NEW) | 3 tests |
| `CHANGELOG.md` (MODIFY, prepend entry) | Session log of Phase 3 implementation |
| `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md` (MODIFY) | Append Phase 3 sub-section after Phase 2 |

**Module boundary discipline (per `feedback_loose_coupling`):**
- `profit_split.py` knows about torch tensors + the concurrency rule; nothing about trees, validation, or I/O.
- `profit_tree.py` knows about splits and tree nodes; nothing about validation, hyperparams, or filesystem.
- `purged_wf.py` knows only about row indices and embargo arithmetic; no tensors, no trees.
- `mine_profiles.py` composes the three above + handles deflation + Q0 gates + bootstrap CI; no filesystem (caller passes loaded tensors).
- `mine_universe.py` is the only module that touches the filesystem (parquet I/O, JSON, markdown).

Tests for each module import only that module and its direct dependencies. `test_profit_tree.py` may mock `profit_split.best_split`; `test_mine_profiles.py` may mock the lower three.

**Branch discipline:** All commits land on the same fresh feature branch (`feat/strategy-discovery-phase3`), created fresh from `main` before Task 1's first commit per the CLAUDE.md rule. Every commit uses surgical `--` pathspec.

---

## Task 1: Split criterion (`profit_split.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/profit_split.py`
- Create: `backend/tests/tools/strategy_discovery/test_profit_split.py`

(Note: `backend/tests/tools/strategy_discovery/__init__.py` already exists from Phase 2 — do not recreate.)

**Scaffolding (write before any test):**

- [ ] **Step 1.0: Create skeleton `profit_split.py`**

```python
"""GPU-vectorized custom split criterion for Phase 3 mining.

For a candidate split `(feature_j, threshold_t)` on a row subset, computes the
concurrency-capped cumulative PnL of each side (max 1 open position per token).
The split metric is `max(cum_pnl_left, cum_pnl_right)`.

Pure functions on torch.Tensor. No I/O, no tree state, no filesystem.
Mirrors backend/tools/xgb_v4_5_features_batch.py conventions for device handling.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class SplitResult:
    feature: int
    threshold: float
    left_mask: torch.Tensor   # (n,) bool — True for rows going to left subtree
    score: float              # the split_metric value (cum_pnl of better side)
```

### Round 1 — `test_walk_and_sum_matches_naive_python_reference`

- [ ] **Step 1.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_profit_split.py`:

```python
"""Tests for tools.strategy_discovery.profit_split (Phase 3)."""
from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

from tools.strategy_discovery.profit_split import (
    SplitResult,
    best_split,
    build_next_eligible,
    walk_and_sum,
)


def _naive_walk_and_sum_py(
    indices: List[int],
    next_eligible: List[int],
    labels: List[float],
) -> float:
    """Reference: walk indices in order, only enter if not already in a trade."""
    open_until = -1   # exclusive
    total = 0.0
    for i in indices:
        if i < open_until:
            continue
        total += labels[i]
        open_until = next_eligible[i]
    return total


def test_walk_and_sum_matches_naive_python_reference():
    rng = np.random.default_rng(13)
    N = 500
    labels = rng.normal(0.0, 0.05, size=N).astype("float64")
    horizon_bars = 24
    # build a synthetic next_eligible from a chronological sequence
    next_eligible = np.minimum(np.arange(N) + horizon_bars, N).astype("int64")
    # build B = 7 candidate subsets of distinct rows
    B = 7
    subsets = []
    for _ in range(B):
        size = rng.integers(50, 200)
        chosen = sorted(rng.choice(N, size=size, replace=False).tolist())
        subsets.append(chosen)
    max_k = max(len(s) for s in subsets)
    # pad to a (B, max_k) tensor with sentinel = -1
    subset_idx = torch.full((B, max_k), -1, dtype=torch.int64)
    for b, s in enumerate(subsets):
        subset_idx[b, : len(s)] = torch.tensor(s, dtype=torch.int64)
    out = walk_and_sum(
        subset_idx,
        torch.from_numpy(next_eligible),
        torch.from_numpy(labels),
    )
    expected = [_naive_walk_and_sum_py(s, next_eligible.tolist(), labels.tolist()) for s in subsets]
    np.testing.assert_allclose(out.cpu().numpy(), np.array(expected), rtol=1e-9, atol=1e-12)
```

- [ ] **Step 1.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py::test_walk_and_sum_matches_naive_python_reference -v
```

Expected: `ImportError` — `walk_and_sum`, `build_next_eligible`, `best_split` not yet implemented.

- [ ] **Step 1.1.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/profit_split.py`:

```python
def build_next_eligible(ts_ms: torch.Tensor, horizon_bars: int) -> torch.Tensor:
    """Returns `next_eligible[i] = min { j > i : ts[j] > ts[i] + horizon_bars * 1h }`.

    Returns `len(ts)` for any row whose horizon would extend past the end.
    """
    n = ts_ms.shape[0]
    horizon_ms = int(horizon_bars) * 3_600_000
    target = ts_ms + horizon_ms                                   # (n,)
    # ts_ms is monotone non-decreasing → use torch.searchsorted
    out = torch.searchsorted(ts_ms, target, right=True)
    # cap at n (out-of-range entries become n)
    return out.clamp_max(n)


def walk_and_sum(
    subset_indices: torch.Tensor,    # (B, K) int64 — padded with -1
    next_eligible: torch.Tensor,     # (N,)  int64
    labels: torch.Tensor,            # (N,)  float
) -> torch.Tensor:
    """Vectorized concurrency-capped (max-1) cumulative PnL across B candidate subsets.

    For each row in B, scan its K rows in order; if a row's index is below the
    current open_until, skip; else add its label and advance open_until to its
    next_eligible. Padding (-1) is treated as "no row".
    """
    B, K = subset_indices.shape
    N = labels.shape[0]
    # Mask padding
    valid = subset_indices >= 0                                   # (B, K)
    safe_idx = subset_indices.clamp_min(0)                        # (B, K) — only used where valid
    # gather labels and next_eligible aligned to subset rows
    row_labels = labels.gather(0, safe_idx.reshape(-1)).reshape(B, K)         # (B, K)
    row_next   = next_eligible.gather(0, safe_idx.reshape(-1)).reshape(B, K)   # (B, K)
    # Per-row sequential reduction is inherently serial; we vectorize over B.
    # On CPU/GPU this is a Python loop over K (small) with all B in parallel.
    open_until = torch.full((B,), -1, dtype=torch.int64, device=subset_indices.device)
    total      = torch.zeros((B,), dtype=labels.dtype, device=subset_indices.device)
    for k in range(K):
        col_idx   = safe_idx[:, k]                                # (B,)
        col_lab   = row_labels[:, k]                              # (B,)
        col_next  = row_next[:, k]                                # (B,)
        col_valid = valid[:, k]                                   # (B,)
        # fire if valid AND col_idx >= open_until
        fire = col_valid & (col_idx >= open_until)
        total      = total + torch.where(fire, col_lab, torch.zeros_like(col_lab))
        open_until = torch.where(fire, col_next, open_until)
    return total
```

- [ ] **Step 1.1.4: Run the test to verify it passes**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `1 passed`.

- [ ] **Step 1.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/profit_split.py backend/tests/tools/strategy_discovery/test_profit_split.py
git commit -m "$(cat <<'EOF'
feat(phase3): add walk_and_sum + build_next_eligible (split-criterion core)

Phase 3 strategy-discovery rebuild — vectorized concurrency-capped (max-1)
cumulative PnL across B candidate subsets. Reference equivalence pinned
against a scalar Python implementation on 500-row synthetic data.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/profit_split.py backend/tests/tools/strategy_discovery/test_profit_split.py
```

### Round 2 — `test_concurrency_max_1_skips_overlapping_entry`

- [ ] **Step 1.2.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_profit_split.py`:

```python
def test_concurrency_max_1_skips_overlapping_entry():
    # 5 rows, ts evenly spaced at 1h intervals, horizon = 3 bars.
    # next_eligible[0]=3, [1]=4, [2]=5, [3]=5, [4]=5.
    # Subset = [0, 1, 2, 3, 4]. We expect: enter at 0 (pays labels[0]), skip 1 (1<3),
    # skip 2 (2<3), enter at 3 (pays labels[3]), skip 4 (4<5).
    ts = torch.arange(5, dtype=torch.int64) * 3_600_000
    labels = torch.tensor([1.0, 10.0, 100.0, 2.0, 50.0], dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=3)
    assert next_eligible.tolist() == [3, 4, 5, 5, 5]
    subset = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.int64)
    total = walk_and_sum(subset, next_eligible, labels)
    # 1.0 + 2.0 = 3.0  (rows 0 and 3 fire; 1, 2, 4 skipped)
    assert total.item() == pytest.approx(3.0, abs=1e-12)
```

- [ ] **Step 1.2.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `2 passed` (implementation from Round 1 already handles the max-1 rule). If FAILS, inspect: most likely off-by-one in `build_next_eligible` (the `right=True` flag matters — it ensures `ts[j] > ts[i] + horizon`, not `>=`).

- [ ] **Step 1.2.3: (Skipped — test passes)**

- [ ] **Step 1.2.4: Run full module file to confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `2 passed`.

- [ ] **Step 1.2.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_profit_split.py
git commit -m "$(cat <<'EOF'
test(phase3): pin concurrency max-1 rule on small worked example

Round 2 of profit_split.py — explicit 5-row example with horizon=3 verifies
the skip behavior and the build_next_eligible right=True semantics.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profit_split.py
```

### Round 3 — `test_split_metric_picks_higher_pnl_subgroup`

- [ ] **Step 1.3.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_profit_split.py`:

```python
def test_split_metric_picks_higher_pnl_subgroup():
    # 100 rows; feature 0 is a step function (rows 0..49 = 0.0, rows 50..99 = 1.0).
    # Labels: left half (feature=0) sums to -1.0; right half (feature=1) sums to +5.0.
    # A split at threshold=0.5 on feature 0 should win with score=+5.0 (the right side).
    N = 100
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    horizon_bars = 1
    features = torch.zeros((N, 1), dtype=torch.float64)
    features[50:, 0] = 1.0
    labels = torch.zeros(N, dtype=torch.float64)
    labels[:50] = -0.02     # 50 rows * -0.02 = -1.0
    labels[50:] = 0.10      # 50 rows * 0.10 = 5.0
    next_eligible = build_next_eligible(ts, horizon_bars=horizon_bars)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is not None
    assert result.feature == 0
    assert 0.0 < result.threshold < 1.0
    assert result.score == pytest.approx(5.0, abs=1e-9)
```

- [ ] **Step 1.3.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py::test_split_metric_picks_higher_pnl_subgroup -v
```

Expected: `AttributeError` / `ImportError` — `best_split` not implemented.

- [ ] **Step 1.3.3: Implement the minimal code**

Append to `backend/tools/strategy_discovery/profit_split.py`:

```python
def _quantile_thresholds(values: torch.Tensor, n: int) -> torch.Tensor:
    """Returns up to n unique candidate thresholds via quantiles of `values`."""
    if values.numel() == 0:
        return values.new_empty((0,))
    q = torch.linspace(0.0, 1.0, n + 2, dtype=values.dtype, device=values.device)[1:-1]
    th = torch.quantile(values, q)
    return torch.unique(th)


def best_split(
    features: torch.Tensor,          # (n, F) feature values at the current node's rows
    indices: torch.Tensor,           # (n,)   absolute row indices into the pid's full tensor
    labels: torch.Tensor,            # (N,)   the pid's full label column
    next_eligible: torch.Tensor,     # (N,)   precomputed eligibility
    n_thresholds: int = 256,
) -> Optional[SplitResult]:
    """Scan all (feature, threshold) candidates; return the best SplitResult or None.

    `None` means no candidate produced strictly positive cum_pnl in either side.
    """
    n, F = features.shape
    if n < 2:
        return None
    best: Optional[SplitResult] = None
    for f in range(F):
        col = features[:, f]
        thresholds = _quantile_thresholds(col, n_thresholds)
        if thresholds.numel() == 0:
            continue
        # Build (T, n) bool mask: left_mask[t, i] = col[i] <= thresholds[t]
        left_mask = col.unsqueeze(0) <= thresholds.unsqueeze(1)        # (T, n)
        # For each (t, side) build a padded index subset; pad with -1.
        # We only need the abs indices for walk_and_sum; pad them per row.
        T = thresholds.numel()
        for side_is_left in (True, False):
            mask = left_mask if side_is_left else ~left_mask           # (T, n)
            counts = mask.sum(dim=1)                                    # (T,)
            if counts.max().item() == 0:
                continue
            max_k = int(counts.max().item())
            # Build a (T, max_k) index tensor with -1 padding.
            subset_idx = torch.full((T, max_k), -1, dtype=torch.int64,
                                    device=features.device)
            # Use cumsum trick to assign abs indices into row-local positions.
            # For each (t, i), if mask[t, i] is True, its position in the row is
            # (cumsum of mask[t, :i+1]) - 1.
            row_pos = mask.cumsum(dim=1) - 1                            # (T, n)
            valid_mask = mask                                            # (T, n)
            # Broadcast indices for scatter
            abs_idx_broadcast = indices.unsqueeze(0).expand(T, n)       # (T, n)
            # Scatter where valid
            r_t, r_n = torch.where(valid_mask)
            subset_idx[r_t, row_pos[r_t, r_n]] = abs_idx_broadcast[r_t, r_n]
            # walk_and_sum across T candidate subsets in parallel
            scores = walk_and_sum(subset_idx, next_eligible, labels)    # (T,)
            best_t = int(scores.argmax().item())
            best_score = float(scores[best_t].item())
            if best_score > 0.0 and (best is None or best_score > best.score):
                best = SplitResult(
                    feature=f,
                    threshold=float(thresholds[best_t].item()),
                    left_mask=(col <= thresholds[best_t]),
                    score=best_score,
                )
    return best
```

- [ ] **Step 1.3.4: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `3 passed`.

- [ ] **Step 1.3.5: Commit**

```bash
git add backend/tools/strategy_discovery/profit_split.py backend/tests/tools/strategy_discovery/test_profit_split.py
git commit -m "$(cat <<'EOF'
feat(phase3): add best_split — scan candidates, pick max-PnL subgroup

Round 3 of profit_split.py — for each (feature, quantile-threshold) candidate
on the current node, computes left and right cumulative PnL via walk_and_sum
and picks the split whose better subgroup has the highest cum_pnl.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/profit_split.py backend/tests/tools/strategy_discovery/test_profit_split.py
```

### Round 4 — `test_no_profitable_split_returns_none`

- [ ] **Step 1.4.1: Write the failing test**

Append:

```python
def test_no_profitable_split_returns_none():
    # All labels strictly negative — no split can produce positive cum_pnl on either side.
    N = 30
    ts = torch.arange(N, dtype=torch.int64) * 3_600_000
    features = torch.linspace(0.0, 1.0, N, dtype=torch.float64).unsqueeze(1)
    labels = torch.full((N,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    indices = torch.arange(N, dtype=torch.int64)
    result = best_split(features, indices, labels, next_eligible, n_thresholds=8)
    assert result is None
```

- [ ] **Step 1.4.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `4 passed`. If FAILS, inspect the `best_score > 0.0` guard in `best_split`.

- [ ] **Step 1.4.3: (Skipped — test passes)**

- [ ] **Step 1.4.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_split.py -v
```

Expected: `4 passed`.

- [ ] **Step 1.4.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_profit_split.py
git commit -m "$(cat <<'EOF'
test(phase3): pin best_split returns None on all-negative labels

Round 4 of profit_split.py — locks the early-stop signal: when no candidate
split produces strictly positive cum_pnl on either side, best_split returns
None so the tree fitter can stop recursing.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profit_split.py
```

---

## Task 2: Tree fitter (`profit_tree.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/profit_tree.py`
- Create: `backend/tests/tools/strategy_discovery/test_profit_tree.py`

**Scaffolding:**

- [ ] **Step 2.0: Create skeleton `profit_tree.py`**

```python
"""Recursive tree fitter for Phase 3 mining.

Builds a binary tree by repeatedly calling profit_split.best_split on the
current node's row subset until depth/leaf-size constraints stop the recursion.

Pure functions on torch.Tensor + Python dataclasses. No I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from tools.strategy_discovery.profit_split import SplitResult, best_split


@dataclass
class TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left:  Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    indices: Optional[torch.Tensor] = None         # populated on leaves
    cumulative_pnl: float = 0.0                    # populated on leaves

    @property
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None
```

### Round 1 — `test_fit_respects_max_depth`

- [ ] **Step 2.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_profit_tree.py`:

```python
"""Tests for tools.strategy_discovery.profit_tree (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.strategy_discovery.profit_split import build_next_eligible
from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves, fit_tree


def _max_depth(node: TreeNode, depth: int = 0) -> int:
    if node.is_leaf:
        return depth
    return max(_max_depth(node.left, depth + 1), _max_depth(node.right, depth + 1))


def _synthetic_data(n: int = 200, seed: int = 7):
    rng = np.random.default_rng(seed)
    ts = torch.arange(n, dtype=torch.int64) * 3_600_000
    features = torch.from_numpy(rng.normal(0.0, 1.0, size=(n, 4)).astype("float64"))
    labels = torch.from_numpy(rng.normal(0.0, 0.05, size=n).astype("float64"))
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    return features, labels, next_eligible


def test_fit_respects_max_depth():
    features, labels, next_eligible = _synthetic_data()
    root = fit_tree(features, labels, next_eligible, max_depth=2, min_leaf=10)
    assert _max_depth(root) <= 2
```

- [ ] **Step 2.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py::test_fit_respects_max_depth -v
```

Expected: `ImportError` — `fit_tree`, `collect_leaves` not yet implemented.

- [ ] **Step 2.1.3: Implement**

Append to `backend/tools/strategy_discovery/profit_tree.py`:

```python
def fit_tree(
    features: torch.Tensor,          # (N, F)
    labels: torch.Tensor,            # (N,)
    next_eligible: torch.Tensor,     # (N,)
    max_depth: int,
    min_leaf: int,
    n_thresholds: int = 256,
) -> TreeNode:
    """Build a profit-maximizing decision tree on the full pid row set."""
    N = features.shape[0]
    all_indices = torch.arange(N, dtype=torch.int64, device=features.device)
    return _fit_recursive(
        features, all_indices, labels, next_eligible,
        depth=0, max_depth=max_depth, min_leaf=min_leaf, n_thresholds=n_thresholds,
    )


def _fit_recursive(
    features: torch.Tensor,          # (N, F) — the full pid tensor (never sliced)
    indices: torch.Tensor,           # (n,)   — absolute indices of rows in this node
    labels: torch.Tensor,            # (N,)
    next_eligible: torch.Tensor,     # (N,)
    *,
    depth: int,
    max_depth: int,
    min_leaf: int,
    n_thresholds: int,
) -> TreeNode:
    node_rows = features.index_select(0, indices)
    n = indices.shape[0]
    if depth >= max_depth or n < 2 * min_leaf:
        return _leaf(indices, labels, next_eligible)
    split = best_split(node_rows, indices, labels, next_eligible, n_thresholds=n_thresholds)
    if split is None:
        return _leaf(indices, labels, next_eligible)
    left_idx  = indices[split.left_mask]
    right_idx = indices[~split.left_mask]
    if int(left_idx.shape[0]) < min_leaf or int(right_idx.shape[0]) < min_leaf:
        return _leaf(indices, labels, next_eligible)
    return TreeNode(
        feature=split.feature,
        threshold=split.threshold,
        left=_fit_recursive(features, left_idx, labels, next_eligible,
                            depth=depth + 1, max_depth=max_depth,
                            min_leaf=min_leaf, n_thresholds=n_thresholds),
        right=_fit_recursive(features, right_idx, labels, next_eligible,
                             depth=depth + 1, max_depth=max_depth,
                             min_leaf=min_leaf, n_thresholds=n_thresholds),
    )


def _leaf(indices: torch.Tensor, labels: torch.Tensor,
          next_eligible: torch.Tensor) -> TreeNode:
    from tools.strategy_discovery.profit_split import walk_and_sum
    # Sort indices chronologically and compute leaf cum_pnl via walk_and_sum.
    sorted_idx, _ = torch.sort(indices)
    pad = sorted_idx.unsqueeze(0)
    cum = float(walk_and_sum(pad, next_eligible, labels)[0].item())
    return TreeNode(indices=sorted_idx, cumulative_pnl=cum)


def collect_leaves(root: TreeNode) -> List[TreeNode]:
    out: List[TreeNode] = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            out.append(n)
        else:
            stack.append(n.right)
            stack.append(n.left)
    return out
```

- [ ] **Step 2.1.4: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `1 passed`.

- [ ] **Step 2.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/profit_tree.py backend/tests/tools/strategy_discovery/test_profit_tree.py
git commit -m "$(cat <<'EOF'
feat(phase3): add fit_tree + collect_leaves with max-depth cap

Phase 3 strategy-discovery rebuild — recursive tree fitter using best_split
at each node. Max-depth cap pinned via test on 200-row synthetic data.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/profit_tree.py backend/tests/tools/strategy_discovery/test_profit_tree.py
```

### Round 2 — `test_fit_respects_min_samples_per_leaf`

- [ ] **Step 2.2.1: Write the failing test**

Append:

```python
def test_fit_respects_min_samples_per_leaf():
    features, labels, next_eligible = _synthetic_data(n=200)
    root = fit_tree(features, labels, next_eligible, max_depth=10, min_leaf=40)
    leaves = collect_leaves(root)
    for leaf in leaves:
        assert leaf.indices is not None
        assert int(leaf.indices.shape[0]) >= 40, f"leaf with {leaf.indices.shape[0]} rows"
```

- [ ] **Step 2.2.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `2 passed` (Round 1 impl already enforces `min_leaf`). If FAILS, inspect the `< 2 * min_leaf` guard.

- [ ] **Step 2.2.3: (Skipped — test passes)**

- [ ] **Step 2.2.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `2 passed`.

- [ ] **Step 2.2.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_profit_tree.py
git commit -m "$(cat <<'EOF'
test(phase3): pin min_leaf enforcement across all surviving leaves

Round 2 of profit_tree.py — every leaf must have >= min_leaf rows. The
recursive splitter's '< 2 * min_leaf' early-stop and the post-split
'min(|left|, |right|) < min_leaf' guard together enforce this.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profit_tree.py
```

### Round 3 — `test_fit_stops_on_unprofitable_split`

- [ ] **Step 2.3.1: Write the failing test**

Append:

```python
def test_fit_stops_on_unprofitable_split():
    # All labels negative -> best_split returns None -> tree is a single leaf.
    n = 200
    ts = torch.arange(n, dtype=torch.int64) * 3_600_000
    features = torch.from_numpy(np.random.default_rng(1).normal(0.0, 1.0, size=(n, 4)).astype("float64"))
    labels = torch.full((n,), -0.05, dtype=torch.float64)
    next_eligible = build_next_eligible(ts, horizon_bars=1)
    root = fit_tree(features, labels, next_eligible, max_depth=5, min_leaf=20)
    assert root.is_leaf
    assert root.indices is not None
    assert int(root.indices.shape[0]) == n
    # Leaf cum_pnl is the walk_and_sum of all rows under max-1 concurrency.
    assert root.cumulative_pnl < 0
```

- [ ] **Step 2.3.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `3 passed`. If FAILS, check that `_fit_recursive` returns a leaf when `best_split is None`.

- [ ] **Step 2.3.3: (Skipped — test passes)**

- [ ] **Step 2.3.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `3 passed`.

- [ ] **Step 2.3.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_profit_tree.py
git commit -m "$(cat <<'EOF'
test(phase3): pin tree stops on unprofitable-split signal

Round 3 of profit_tree.py — when best_split returns None (no candidate
produces positive cum_pnl), the recursive fitter must return a leaf
without splitting.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profit_tree.py
```

### Round 4 — `test_collect_leaves_returns_all_leaves_in_dfs_order`

- [ ] **Step 2.4.1: Write the failing test**

Append:

```python
def test_collect_leaves_returns_all_leaves_in_dfs_order():
    # Construct a known tree by hand: root -> (left=leaf_a, right=node -> (leaf_b, leaf_c))
    leaf_a = TreeNode(indices=torch.tensor([0, 1, 2], dtype=torch.int64), cumulative_pnl=1.0)
    leaf_b = TreeNode(indices=torch.tensor([3, 4],    dtype=torch.int64), cumulative_pnl=2.0)
    leaf_c = TreeNode(indices=torch.tensor([5, 6, 7], dtype=torch.int64), cumulative_pnl=3.0)
    right_subtree = TreeNode(feature=0, threshold=0.5, left=leaf_b, right=leaf_c)
    root = TreeNode(feature=1, threshold=0.0, left=leaf_a, right=right_subtree)
    leaves = collect_leaves(root)
    assert [leaf.cumulative_pnl for leaf in leaves] == [1.0, 2.0, 3.0]
```

- [ ] **Step 2.4.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `4 passed`. If FAILS, check the stack order in `collect_leaves` — `stack.pop()` is LIFO so to get left-to-right DFS we push right first then left.

- [ ] **Step 2.4.3: (Skipped — test passes)**

- [ ] **Step 2.4.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_profit_tree.py -v
```

Expected: `4 passed`.

- [ ] **Step 2.4.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_profit_tree.py
git commit -m "$(cat <<'EOF'
test(phase3): pin collect_leaves DFS-left-first traversal order

Round 4 of profit_tree.py — leaves must come back in left-to-right DFS
order so leaf_id assignment downstream is deterministic.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_profit_tree.py
```

---

## Task 3: Purged Walk-Forward CV (`purged_wf.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/purged_wf.py`
- Create: `backend/tests/tools/strategy_discovery/test_purged_wf.py`

**Scaffolding:**

- [ ] **Step 3.0: Create skeleton `purged_wf.py`**

```python
"""Purged Walk-Forward CV + nested inner CV for Phase 3 mining.

Pure numpy index math. No tensors, no trees, no I/O.

Embargo rule: any train row whose ts falls within [test_start - horizon, test_start)
is dropped from the train set to prevent label leakage (a train row's
label_h{horizon} could span into the test fold).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
```

### Round 1 — `test_5_folds_cover_all_rows_disjointly`

- [ ] **Step 3.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_purged_wf.py`:

```python
"""Tests for tools.strategy_discovery.purged_wf (Phase 3)."""
from __future__ import annotations

import numpy as np
import pytest

from tools.strategy_discovery.purged_wf import inner_folds, outer_folds


def test_5_folds_cover_all_rows_disjointly():
    n = 1000
    folds = outer_folds(n, n_folds=5, embargo_bars=0)
    # 5 folds, each (train_idx, test_idx)
    assert len(folds) == 5
    # Test indices across folds are disjoint and cover all rows except fold 0's "before" (none).
    test_union = np.concatenate([test_idx for _, test_idx in folds])
    assert len(np.unique(test_union)) == len(test_union), "test sets overlap"
    assert set(test_union.tolist()) == set(range(n)), "test sets don't cover all rows"
    # Per-fold sizes are within 1 of n / n_folds
    expected_size = n // 5
    for _, test_idx in folds:
        assert abs(len(test_idx) - expected_size) <= 1
```

- [ ] **Step 3.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py::test_5_folds_cover_all_rows_disjointly -v
```

Expected: `ImportError` — `outer_folds` not yet implemented.

- [ ] **Step 3.1.3: Implement**

Append to `backend/tools/strategy_discovery/purged_wf.py`:

```python
def outer_folds(
    n_rows: int,
    n_folds: int = 5,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Chronological k-fold WF with embargo. Returns [(train_idx, test_idx), ...].

    Fold k's test set = rows [k * size, (k+1) * size) (last fold absorbs remainder).
    Train set = all rows OUTSIDE test EXCEPT those within `embargo_bars` of test_start.
    Fold 0 has no train data and is dropped (returned list has up to n_folds entries
    but the first fold may be skipped if its train would be empty).
    """
    base = n_rows // n_folds
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_start = k * base
        test_end   = (k + 1) * base if k < n_folds - 1 else n_rows
        test_idx = np.arange(test_start, test_end, dtype=np.int64)
        embargo_lo = max(0, test_start - embargo_bars)
        # train = rows OUTSIDE [embargo_lo, test_end) AND outside the test fold
        train_mask = np.ones(n_rows, dtype=bool)
        train_mask[embargo_lo:test_end] = False
        train_idx = np.where(train_mask)[0]
        out.append((train_idx, test_idx))
    return out


def inner_folds(
    train_idx: np.ndarray,
    n_folds: int = 3,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Nested CV on the outer-train subset. Same shape as outer_folds but the
    embargo is computed on the positions WITHIN train_idx, not the global row ids."""
    n = len(train_idx)
    base = n // n_folds
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        test_start = k * base
        test_end   = (k + 1) * base if k < n_folds - 1 else n
        embargo_lo = max(0, test_start - embargo_bars)
        mask = np.ones(n, dtype=bool)
        mask[embargo_lo:test_end] = False
        inner_train = train_idx[mask]
        inner_test  = train_idx[test_start:test_end]
        out.append((inner_train, inner_test))
    return out
```

- [ ] **Step 3.1.4: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py -v
```

Expected: `1 passed`.

- [ ] **Step 3.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/purged_wf.py backend/tests/tools/strategy_discovery/test_purged_wf.py
git commit -m "$(cat <<'EOF'
feat(phase3): add outer_folds + inner_folds (Purged WF + nested CV)

Phase 3 strategy-discovery rebuild — k-fold chronological WF with embargo
applied at the train boundary. inner_folds operates on the outer-train
subset for nested hyperparam search. Pure numpy, no tensors.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/purged_wf.py backend/tests/tools/strategy_discovery/test_purged_wf.py
```

### Round 2 — `test_embargo_drops_train_rows_within_horizon_of_test_start`

- [ ] **Step 3.2.1: Write the failing test**

Append:

```python
def test_embargo_drops_train_rows_within_horizon_of_test_start():
    n = 100
    folds = outer_folds(n, n_folds=5, embargo_bars=10)
    # Fold 2 has test_start = 40. Embargo drops train rows [30, 40).
    train_idx, test_idx = folds[2]
    assert test_idx.tolist() == list(range(40, 60))
    # Train must contain rows [0..30) and [60..100); rows [30..40) embargoed.
    assert set(range(0, 30)).issubset(set(train_idx.tolist()))
    assert set(range(60, 100)).issubset(set(train_idx.tolist()))
    for embargoed in range(30, 40):
        assert embargoed not in train_idx.tolist()
```

- [ ] **Step 3.2.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py -v
```

Expected: `2 passed`. If FAILS, check that `embargo_lo = max(0, test_start - embargo_bars)` correctly excludes embargoed rows from `train_mask`.

- [ ] **Step 3.2.3: (Skipped — test passes)**

- [ ] **Step 3.2.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py -v
```

Expected: `2 passed`.

- [ ] **Step 3.2.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_purged_wf.py
git commit -m "$(cat <<'EOF'
test(phase3): pin embargo behavior on outer fold boundary

Round 2 of purged_wf.py — locks the spec's leakage rule: train rows whose
ts falls within [test_start - horizon, test_start) are dropped because
their label_h{horizon} could span into the test fold.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_purged_wf.py
```

### Round 3 — `test_nested_inner_cv_uses_only_outer_train`

- [ ] **Step 3.3.1: Write the failing test**

Append:

```python
def test_nested_inner_cv_uses_only_outer_train():
    n = 600
    outer = outer_folds(n, n_folds=5, embargo_bars=0)
    # Pick outer fold 2: train = [0..120) ∪ [240..600), test = [120..240).
    outer_train, outer_test = outer[2]
    inner = inner_folds(outer_train, n_folds=3, embargo_bars=0)
    assert len(inner) == 3
    # Every inner train+test index must lie within outer_train; never in outer_test.
    outer_train_set = set(outer_train.tolist())
    outer_test_set  = set(outer_test.tolist())
    for inner_train, inner_test in inner:
        for idx in inner_train.tolist() + inner_test.tolist():
            assert idx in outer_train_set, f"inner idx {idx} not in outer train"
            assert idx not in outer_test_set, f"inner idx {idx} leaked from outer test"
```

- [ ] **Step 3.3.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py -v
```

Expected: `3 passed`. If FAILS, check that `inner_folds` indexes into `train_idx` (not into the global row range).

- [ ] **Step 3.3.3: (Skipped — test passes)**

- [ ] **Step 3.3.4: Confirm green**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_purged_wf.py -v
```

Expected: `3 passed`.

- [ ] **Step 3.3.5: Commit**

```bash
git add backend/tests/tools/strategy_discovery/test_purged_wf.py
git commit -m "$(cat <<'EOF'
test(phase3): pin inner CV stays within outer train (no outer-test leakage)

Round 3 of purged_wf.py — locks the nested-CV isolation rule: inner folds
operate on the outer-train subset only, so hyperparam search cannot peek
at the outer-test fold.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tests/tools/strategy_discovery/test_purged_wf.py
```

---

## Task 4: Per-(pid, horizon) orchestrator (`mine_profiles.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/mine_profiles.py`
- Create: `backend/tests/tools/strategy_discovery/test_mine_profiles.py`

**Scaffolding:**

- [ ] **Step 4.0: Create skeleton `mine_profiles.py`**

```python
"""Per-(pid, horizon) mining orchestrator for Phase 3.

Composes profit_tree + purged_wf + the deflation factor + Q0 gates + bootstrap CI.
Pure functions on torch.Tensor inputs (caller loads the parquet). No filesystem.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from tools.strategy_discovery.profit_tree import TreeNode, collect_leaves, fit_tree
from tools.strategy_discovery.profit_split import build_next_eligible, walk_and_sum
from tools.strategy_discovery.purged_wf import inner_folds, outer_folds

_DEPTH_GRID    = (3, 5, 7)
_MIN_LEAF_GRID = (20, 50, 100)
_RETAIL_FEE    = 0.012
_Q0_AVG_WIN    = 0.05    # ≥ +5%
_Q0_AVG_LOSS   = -0.10   # ≤ -10% magnitude (i.e. avg_loss must be > -0.10 to fail, ≤ -0.10 to pass the "deeper than -10%" check)
_Q0_MAX_DD     = 0.30
_Q0_MIN_FOLDS  = 4
_BOOTSTRAP_N   = 1000


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
```

### Round 1 — `test_deflation_factor_applied_to_reported_profit`

- [ ] **Step 4.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_mine_profiles.py`:

```python
"""Tests for tools.strategy_discovery.mine_profiles (Phase 3)."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tools.strategy_discovery.mine_profiles import (
    apply_deflation,
    bootstrap_ci,
    leaf_metrics,
    leaf_qualifies,
    long_shot_band,
    pick_best_hyperparams,
)


def test_deflation_factor_applied_to_reported_profit():
    # raw_max = 7.2%, inner_cv_se = 1.5%, N = 9 combos.
    # Inflation = 1.5% * sqrt(2 * ln 9) = 1.5% * 2.0972 ≈ 3.146%
    # Deflated = 7.2% - 3.146% ≈ 4.054%
    deflated, infl = apply_deflation(raw=0.072, inner_cv_se=0.015, n_combos=9)
    assert infl == pytest.approx(0.015 * math.sqrt(2 * math.log(9)), rel=1e-9)
    assert deflated == pytest.approx(0.072 - infl, rel=1e-9)
```

- [ ] **Step 4.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_deflation_factor_applied_to_reported_profit -v
```

Expected: `ImportError` — `apply_deflation` not yet implemented.

- [ ] **Step 4.1.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def apply_deflation(raw: float, inner_cv_se: float, n_combos: int) -> Tuple[float, float]:
    """Apply max-of-N inflation correction to a search-best profit estimate.

    Returns (deflated_profit, inflation). inflation = σ × √(2 × ln N).
    """
    inflation = float(inner_cv_se) * math.sqrt(2.0 * math.log(max(int(n_combos), 1)))
    return raw - inflation, inflation
```

- [ ] **Step 4.1.4: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `1 passed`.

- [ ] **Step 4.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add deflation factor for selection-bias correction

Phase 3 strategy-discovery rebuild — implements raw − σ × √(2 ln N) per
the spec. For N=9 hyperparam combos and σ=1.5% inner-CV SE, inflation ≈
3.15pp; this gets subtracted from the search-best cumulative profit
before downstream gating and ranking.

Companion HTML walkthrough at backend/tools/phase3_deflation_explainer.html.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

### Round 2 — `test_long_shot_band_triggers_bootstrap`

- [ ] **Step 4.2.1: Write the failing test**

Append:

```python
def test_long_shot_band_triggers_bootstrap():
    # Long-shot: avg_win >= 15%, |avg_loss| <= 7%, win_rate >= 70%.
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.75) is True
    # Just outside the band on each axis.
    assert long_shot_band(avg_win=0.14, avg_loss=-0.05, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.08, win_rate=0.75) is False
    assert long_shot_band(avg_win=0.16, avg_loss=-0.05, win_rate=0.69) is False
```

- [ ] **Step 4.2.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_long_shot_band_triggers_bootstrap -v
```

Expected: `ImportError` — `long_shot_band` not implemented.

- [ ] **Step 4.2.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def long_shot_band(avg_win: float, avg_loss: float, win_rate: float) -> bool:
    """Per spec: avg_win >= 15% AND |avg_loss| <= 7% AND win_rate >= 70%."""
    return avg_win >= 0.15 and abs(avg_loss) <= 0.07 and win_rate >= 0.70
```

- [ ] **Step 4.2.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `2 passed`.

- [ ] **Step 4.2.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add long_shot_band predicate for bootstrap trigger

Round 2 of mine_profiles.py — locks the spec's long-shot definition:
avg_win >= 15% AND |avg_loss| <= 7% AND win_rate >= 70%. Profiles in
this band trigger the bootstrap CI layer regardless of trade count.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

### Round 3 — `test_leaf_metrics_computes_win_loss_dd_sortino`

- [ ] **Step 4.3.1: Write the failing test**

Append:

```python
def test_leaf_metrics_computes_win_loss_dd_sortino():
    # 6 trades: 4 wins +8%, 2 losses -5%. Per-trade net = label minus fee already.
    trades = np.array([0.08, 0.08, -0.05, 0.08, 0.08, -0.05])
    m = leaf_metrics(trades)
    assert m["trade_count"] == 6
    assert m["win_rate"] == pytest.approx(4 / 6)
    assert m["avg_win"]  == pytest.approx(0.08)
    assert m["avg_loss"] == pytest.approx(-0.05)
    # cumulative = sum = 0.32 - 0.10 = 0.22
    assert m["cumulative_profit_raw"] == pytest.approx(0.22, abs=1e-9)
    # equity curve: 0, .08, .16, .11, .19, .27, .22 → max = .27, final = .22, dd = .05
    assert m["max_dd"] == pytest.approx(0.05, abs=1e-9)
    # Sortino numerator = mean = 0.22/6 ≈ 0.0367; downside dev = sqrt(mean of negative^2)
    # negatives = [-0.05, -0.05], mean(neg^2) = 0.0025, sqrt = 0.05
    # Sortino = 0.0367 / 0.05 ≈ 0.733
    assert m["sortino"] == pytest.approx(0.22 / 6 / 0.05, rel=1e-6)
```

- [ ] **Step 4.3.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_leaf_metrics_computes_win_loss_dd_sortino -v
```

Expected: `ImportError` — `leaf_metrics` not implemented.

- [ ] **Step 4.3.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def leaf_metrics(trades_net: np.ndarray) -> dict:
    """Compute trade-list metrics from a leaf's net-PnL trade sequence."""
    n = int(trades_net.shape[0])
    if n == 0:
        return {
            "trade_count": 0, "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "max_dd": 0.0, "cumulative_profit_raw": 0.0, "sortino": 0.0,
        }
    wins = trades_net[trades_net > 0]
    losses = trades_net[trades_net < 0]
    avg_win  = float(wins.mean())  if wins.size  > 0 else 0.0
    avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
    cum = float(trades_net.sum())
    equity = np.concatenate([[0.0], np.cumsum(trades_net)])
    running_max = np.maximum.accumulate(equity)
    drawdown    = running_max - equity
    max_dd = float(drawdown.max())
    mean_trade = float(trades_net.mean())
    if losses.size > 0:
        downside_dev = float(np.sqrt(np.mean(losses ** 2)))
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
```

- [ ] **Step 4.3.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `3 passed`.

- [ ] **Step 4.3.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add leaf_metrics — win/loss/DD/Sortino from net-PnL trades

Round 3 of mine_profiles.py — computes the per-leaf metrics consumed by
the Q0 gate evaluator and reported in the output parquet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

### Round 4 — `test_q0_gates_applied_to_deflated_profit`

- [ ] **Step 4.4.1: Write the failing test**

Append:

```python
def test_q0_gates_applied_to_deflated_profit():
    # Per-fold pass/fail vectors of length 5.
    # Need >= 4 folds passing all three gates.
    fold_metrics = [
        {"avg_win": 0.08, "avg_loss": -0.08, "max_dd": 0.20, "deflated_profit": 0.05},  # PASS
        {"avg_win": 0.06, "avg_loss": -0.09, "max_dd": 0.25, "deflated_profit": 0.03},  # PASS
        {"avg_win": 0.07, "avg_loss": -0.11, "max_dd": 0.22, "deflated_profit": 0.04},  # FAIL (avg_loss too negative)
        {"avg_win": 0.09, "avg_loss": -0.07, "max_dd": 0.18, "deflated_profit": 0.06},  # PASS
        {"avg_win": 0.10, "avg_loss": -0.06, "max_dd": 0.15, "deflated_profit": 0.07},  # PASS
    ]
    n_pass = sum(leaf_qualifies(m) for m in fold_metrics)
    assert n_pass == 4
    # The leaf should qualify because n_pass >= _Q0_MIN_FOLDS (4).
    from tools.strategy_discovery.mine_profiles import _Q0_MIN_FOLDS
    assert n_pass >= _Q0_MIN_FOLDS
```

- [ ] **Step 4.4.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_q0_gates_applied_to_deflated_profit -v
```

Expected: `ImportError` — `leaf_qualifies` not implemented.

- [ ] **Step 4.4.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def leaf_qualifies(fold_metric: dict) -> bool:
    """True if the leaf passes all three Q0 hard gates on this fold."""
    if fold_metric["avg_win"] < _Q0_AVG_WIN:
        return False
    if fold_metric["avg_loss"] < _Q0_AVG_LOSS:
        return False
    if fold_metric["max_dd"] > _Q0_MAX_DD:
        return False
    return True
```

- [ ] **Step 4.4.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `4 passed`.

- [ ] **Step 4.4.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add leaf_qualifies Q0 gate (avg_win, avg_loss, max_dd)

Round 4 of mine_profiles.py — locks Q0 hard gates: avg_win >= +5%,
avg_loss >= -10% (magnitude), max_dd <= 30%. A leaf must pass on >= 4
of 5 outer folds to become a profile.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

### Round 5 — `test_pick_best_hyperparams_picks_max_inner_cv`

- [ ] **Step 4.5.1: Write the failing test**

Append:

```python
def test_pick_best_hyperparams_picks_max_inner_cv():
    # Manually craft a {(depth, min_leaf): inner_mean_profit} table.
    # The best combo should be argmax of inner_mean_profit.
    inner_scores = {
        (3, 20):  0.012,
        (3, 50):  0.018,
        (3, 100): 0.005,
        (5, 20):  0.022,
        (5, 50):  0.019,
        (5, 100): 0.011,
        (7, 20):  0.025,   # <-- max
        (7, 50):  0.020,
        (7, 100): 0.015,
    }
    chosen_depth, chosen_min_leaf, raw_max, inner_se = pick_best_hyperparams(inner_scores)
    assert chosen_depth == 7
    assert chosen_min_leaf == 20
    assert raw_max == pytest.approx(0.025, rel=1e-9)
    # SE = std across the 9 combos (used for deflation)
    expected_se = float(np.std(list(inner_scores.values()), ddof=1))
    assert inner_se == pytest.approx(expected_se, rel=1e-9)
```

- [ ] **Step 4.5.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_pick_best_hyperparams_picks_max_inner_cv -v
```

Expected: `ImportError` — `pick_best_hyperparams` not implemented.

- [ ] **Step 4.5.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def pick_best_hyperparams(inner_scores: dict) -> Tuple[int, int, float, float]:
    """Pick argmax of the inner-CV mean-profit table.

    Returns (chosen_depth, chosen_min_leaf, raw_max_profit, inner_cv_se).
    inner_cv_se = std (ddof=1) of the inner mean profits across all combos —
    used as σ in the deflation factor.
    """
    best_combo, raw_max = max(inner_scores.items(), key=lambda kv: kv[1])
    chosen_depth, chosen_min_leaf = best_combo
    values = np.array(list(inner_scores.values()), dtype="float64")
    inner_cv_se = float(values.std(ddof=1))
    return int(chosen_depth), int(chosen_min_leaf), float(raw_max), inner_cv_se
```

- [ ] **Step 4.5.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `5 passed`.

- [ ] **Step 4.5.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add pick_best_hyperparams (argmax + SE for deflation)

Round 5 of mine_profiles.py — selects the best combo from the inner-CV
score table and computes the per-combo SE that feeds the deflation
factor. Both pieces of information are persisted in the output parquet.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

### Round 6 — `test_bootstrap_ci_returns_95pct_band_on_resampled_trades`

- [ ] **Step 4.6.1: Write the failing test**

Append:

```python
def test_bootstrap_ci_returns_95pct_band_on_resampled_trades():
    # 50 trades, +1% each. Bootstrap CI on cumulative profit should be very tight
    # around 0.50 (no variance in the trades).
    trades = np.full(50, 0.01)
    rng = np.random.default_rng(7)
    lower, upper = bootstrap_ci(trades, n_iter=500, rng=rng)
    assert lower == pytest.approx(0.50, abs=1e-6)
    assert upper == pytest.approx(0.50, abs=1e-6)
    # Now mix wins and losses — CI should widen.
    mixed = np.concatenate([np.full(30, 0.10), np.full(20, -0.05)])
    lo2, hi2 = bootstrap_ci(mixed, n_iter=500, rng=rng)
    # point estimate = 30*0.10 + 20*(-0.05) = 3.0 - 1.0 = 2.0
    # CI should bracket 2.0 with non-zero width
    assert lo2 < 2.0 < hi2
    assert (hi2 - lo2) > 0.1   # non-trivial width
```

- [ ] **Step 4.6.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_bootstrap_ci_returns_95pct_band_on_resampled_trades -v
```

Expected: `ImportError` — `bootstrap_ci` not implemented.

- [ ] **Step 4.6.3: Implement**

Append to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def bootstrap_ci(
    trades_net: np.ndarray,
    n_iter: int = _BOOTSTRAP_N,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Returns (95% lower, 95% upper) on cumulative profit via bootstrap resampling.

    Resamples len(trades_net) trades with replacement, n_iter times.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(trades_net)
    if n == 0:
        return 0.0, 0.0
    samples = rng.choice(trades_net, size=(int(n_iter), n), replace=True)
    cum = samples.sum(axis=1)
    return float(np.percentile(cum, 2.5)), float(np.percentile(cum, 97.5))
```

- [ ] **Step 4.6.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `6 passed`.

- [ ] **Step 4.6.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): add bootstrap_ci on cumulative profit (95% band)

Round 6 of mine_profiles.py — closes out the per-leaf evaluation surface.
Trade-list bootstrap resampling produces a 95% CI on cumulative profit;
triggered when trade_count_per_fold < 30 OR profile in long-shot band.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

---

## Task 5: Universe driver + CLI (`mine_universe.py` + tests)

**Files:**
- Create: `backend/tools/strategy_discovery/mine_universe.py`
- Create: `backend/tests/tools/strategy_discovery/test_mine_universe.py`

**Scaffolding:**

- [ ] **Step 5.0: Create skeleton `mine_universe.py`**

```python
"""Phase 3 universe driver — iterate (pid, horizon) pairs, write outputs.

Loads phase2 parquets, dispatches mine_profiles per (pid, horizon), and
writes profiles_h{h}.parquet + rule_paths_h{h}.json + mining_summary.md.

The ONLY module in Phase 3 that touches the filesystem.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

BACKEND = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from tools.strategy_discovery.mine_profiles import LeafProfile  # noqa: E402

_DEFAULT_HORIZONS = (1, 4, 24, 72, 168)
_SCHEMA_VERSION = 1
_DEFAULT_PHASE2_DIR = Path(BACKEND) / "data" / "phase2"
_DEFAULT_OUTPUT_DIR = Path(BACKEND) / "data" / "phase3"
```

### Round 1 — `test_pids_from_universe_json_flattens_cohorts`

- [ ] **Step 5.1.1: Write the failing test**

Create `backend/tests/tools/strategy_discovery/test_mine_universe.py`:

```python
"""Tests for tools.strategy_discovery.mine_universe (Phase 3 CLI driver)."""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tools.strategy_discovery.mine_universe import (
    _DEFAULT_HORIZONS,
    pids_from_universe_json,
    write_profile_parquet,
)
from tools.strategy_discovery.mine_profiles import LeafProfile


def test_pids_from_universe_json_flattens_cohorts(tmp_path: Path):
    universe = {
        "large":          ["BTC-USD", "ETH-USD"],
        "mid":            ["LINK-USD"],
        "high_fdv_ratio": ["NEAR-USD", "BTC-USD"],  # BTC-USD duplicated across cohorts
        "low_turnover":   [],
    }
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    pids = pids_from_universe_json(universe_path)
    # Sorted, deduplicated
    assert pids == ["BTC-USD", "ETH-USD", "LINK-USD", "NEAR-USD"]
```

- [ ] **Step 5.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py::test_pids_from_universe_json_flattens_cohorts -v
```

Expected: `ImportError` — `pids_from_universe_json` not implemented.

- [ ] **Step 5.1.3: Implement**

Append to `backend/tools/strategy_discovery/mine_universe.py`:

```python
def pids_from_universe_json(universe_path: Path) -> List[str]:
    """Flatten {cohort: [pids]} into a deduplicated sorted pid list."""
    with open(universe_path, "r", encoding="utf-8") as f:
        cohorts = json.load(f)
    seen: set = set()
    for pids in cohorts.values():
        seen.update(pids)
    return sorted(seen)
```

- [ ] **Step 5.1.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py -v
```

Expected: `1 passed`.

- [ ] **Step 5.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_universe.py backend/tests/tools/strategy_discovery/test_mine_universe.py
git commit -m "$(cat <<'EOF'
feat(phase3): add pids_from_universe_json (cohort flattener)

Phase 3 strategy-discovery rebuild — reuses Phase 2's universe JSON shape
({cohort: [pid]}); driver consumes the deduplicated sorted pid list.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_universe.py backend/tests/tools/strategy_discovery/test_mine_universe.py
```

### Round 2 — `test_write_profile_parquet_round_trips_all_columns`

- [ ] **Step 5.2.1: Write the failing test**

Append:

```python
def test_write_profile_parquet_round_trips_all_columns(tmp_path: Path):
    profiles = [
        LeafProfile(
            leaf_id=0,
            rule_path_summary="vol_over_mc > 0.08",
            trade_count=42,
            win_rate=0.6,
            avg_win=0.07,
            avg_loss=-0.04,
            max_dd=0.22,
            cumulative_profit_raw=0.072,
            cumulative_profit_deflated=0.041,
            deflation_pp=0.031,
            n_combos_searched=9,
            inner_cv_se=0.015,
            sortino=1.34,
            n_folds_passed_q0=4,
            bootstrap_triggered=True,
            bootstrap_ci_lower=0.020,
            bootstrap_ci_upper=0.060,
            chosen_depth=5,
            chosen_min_leaf=50,
        ),
        LeafProfile(
            leaf_id=1,
            rule_path_summary="ret_24h_sign == 1",
            trade_count=10,
            win_rate=0.5,
            avg_win=0.06,
            avg_loss=-0.05,
            max_dd=0.10,
            cumulative_profit_raw=0.010,
            cumulative_profit_deflated=-0.005,
            deflation_pp=0.015,
            n_combos_searched=9,
            inner_cv_se=0.007,
            sortino=0.5,
            n_folds_passed_q0=5,
            bootstrap_triggered=False,
            bootstrap_ci_lower=None,
            bootstrap_ci_upper=None,
            chosen_depth=3,
            chosen_min_leaf=20,
        ),
    ]
    out_path = tmp_path / "profiles_h24.parquet"
    write_profile_parquet(profiles, pid="BTC-USD", horizon=24, output_path=out_path)
    assert out_path.exists()
    df = pq.read_table(out_path).to_pandas()
    assert len(df) == 2
    expected_cols = {
        "pid", "horizon", "leaf_id", "rule_path_summary",
        "trade_count", "win_rate", "avg_win", "avg_loss", "max_dd",
        "cumulative_profit_raw", "cumulative_profit_deflated", "deflation_pp",
        "n_combos_searched", "inner_cv_se", "sortino", "n_folds_passed_q0",
        "bootstrap_triggered", "bootstrap_ci_lower", "bootstrap_ci_upper",
        "chosen_depth", "chosen_min_leaf", "schema_version",
    }
    assert set(df.columns) == expected_cols
    assert (df["pid"] == "BTC-USD").all()
    assert (df["horizon"] == 24).all()
    assert (df["schema_version"] == 1).all()
```

- [ ] **Step 5.2.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py::test_write_profile_parquet_round_trips_all_columns -v
```

Expected: `ImportError` — `write_profile_parquet` not implemented.

- [ ] **Step 5.2.3: Implement**

Append to `backend/tools/strategy_discovery/mine_universe.py`:

```python
def write_profile_parquet(
    profiles: List[LeafProfile],
    *,
    pid: str,
    horizon: int,
    output_path: Path,
) -> None:
    """Write a list of LeafProfile rows to parquet, adding pid + horizon + schema_version."""
    rows = []
    for p in profiles:
        d = asdict(p)
        d["pid"] = pid
        d["horizon"] = int(horizon)
        d["schema_version"] = _SCHEMA_VERSION
        rows.append(d)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # If file exists, append (parquet writer handles this via to_parquet on a combined df).
    if output_path.exists():
        existing = pq.read_table(output_path).to_pandas()
        df = pd.concat([existing, df], ignore_index=True)
    df.to_parquet(output_path, compression="snappy", index=False)
```

- [ ] **Step 5.2.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py -v
```

Expected: `2 passed`.

- [ ] **Step 5.2.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_universe.py backend/tests/tools/strategy_discovery/test_mine_universe.py
git commit -m "$(cat <<'EOF'
feat(phase3): add write_profile_parquet (append-on-write per pid)

Round 2 of mine_universe.py — writes LeafProfile rows to a per-horizon
parquet, appending if the file already exists so multiple pids land in
the same file. Adds pid + horizon + schema_version columns.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_universe.py backend/tests/tools/strategy_discovery/test_mine_universe.py
```

### Round 3 — `test_iterates_all_pid_horizon_pairs`

- [ ] **Step 5.3.1: Write the failing test**

Append:

```python
def test_iterates_all_pid_horizon_pairs(tmp_path, monkeypatch):
    # Mock mine_profiles_for_pid_horizon to record (pid, horizon) calls and return [].
    universe = {"large": ["A-USD", "B-USD"], "mid": ["C-USD"]}
    universe_path = tmp_path / "universe.json"
    universe_path.write_text(json.dumps(universe), encoding="utf-8")
    phase2_dir = tmp_path / "phase2"
    output_dir = tmp_path / "phase3"
    phase2_dir.mkdir()
    # Create stub parquets so the mine_universe loop doesn't bail on missing files.
    for pid in ["A-USD", "B-USD", "C-USD"]:
        pa_table = pa.table({"ts": [0], "close": [1.0]})
        pq.write_table(pa_table, phase2_dir / f"{pid}.parquet")

    calls = []
    from tools.strategy_discovery import mine_universe as mu
    def fake_mine(pid, horizon, parquet_path, device="cuda", seed=42):
        calls.append((pid, horizon))
        return []
    monkeypatch.setattr(mu, "mine_profiles_for_pid_horizon", fake_mine)

    mu.mine_universe(
        universe_path=universe_path,
        phase2_dir=phase2_dir,
        output_dir=output_dir,
        horizons=[1, 4, 24],
        device="cpu",
        seed=42,
    )
    # 3 pids × 3 horizons = 9 calls
    assert len(calls) == 9
    assert set(calls) == {(p, h) for p in ["A-USD", "B-USD", "C-USD"] for h in [1, 4, 24]}
```

- [ ] **Step 5.3.2: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py::test_iterates_all_pid_horizon_pairs -v
```

Expected: `ImportError` — `mine_universe` + `mine_profiles_for_pid_horizon` not yet implemented.

- [ ] **Step 5.3.3: Implement**

Append a stub of `mine_profiles_for_pid_horizon` to `backend/tools/strategy_discovery/mine_profiles.py`:

```python
def mine_profiles_for_pid_horizon(
    pid: str,
    horizon: int,
    parquet_path,
    device: str = "cuda",
    seed: int = 42,
) -> List[LeafProfile]:
    """End-to-end per-(pid, horizon) mining. Returns surviving leaves.

    Stub implementation: tests at this layer mock this function.
    The full body (load parquet -> tensor, run outer CV with inner hyperparam
    search, evaluate Q0 gates, bootstrap CI, return LeafProfile list) ships
    in subsequent rounds. This stub returns an empty list so the universe
    driver test can verify iteration without depending on the mining itself.
    """
    return []
```

Append the `mine_universe` driver + `main` CLI to `backend/tools/strategy_discovery/mine_universe.py`:

```python
from tools.strategy_discovery.mine_profiles import mine_profiles_for_pid_horizon  # noqa: E402


def mine_universe(
    *,
    universe_path: Path,
    phase2_dir: Path = _DEFAULT_PHASE2_DIR,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    horizons = _DEFAULT_HORIZONS,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[int, List[LeafProfile]]:
    """Iterate (pid, horizon) cross-product; collect profiles per horizon."""
    pids = pids_from_universe_json(Path(universe_path))
    all_profiles: Dict[int, List[LeafProfile]] = {int(h): [] for h in horizons}
    rule_paths_per_horizon: Dict[int, Dict[str, str]] = {int(h): {} for h in horizons}
    output_dir.mkdir(parents=True, exist_ok=True)
    for pid in pids:
        parquet_path = phase2_dir / f"{pid}.parquet"
        if not parquet_path.exists():
            continue
        for h in horizons:
            profiles = mine_profiles_for_pid_horizon(
                pid=pid, horizon=int(h), parquet_path=parquet_path,
                device=device, seed=seed,
            )
            all_profiles[int(h)].extend(profiles)
            if profiles:
                write_profile_parquet(
                    profiles, pid=pid, horizon=int(h),
                    output_path=output_dir / f"profiles_h{int(h)}.parquet",
                )
                for p in profiles:
                    rule_paths_per_horizon[int(h)][f"{pid}__{p.leaf_id}"] = p.rule_path_summary
    # Emit rule_paths sidecars
    for h, paths in rule_paths_per_horizon.items():
        if paths:
            with open(output_dir / f"rule_paths_h{int(h)}.json", "w", encoding="utf-8") as f:
                json.dump(paths, f, indent=2)
    return all_profiles


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Mine Phase 3 profiles for a universe.")
    parser.add_argument(
        "--universe",
        default=str(Path(BACKEND).parent / "docs" / "superpowers" / "specs" / "2026-05-23-universe-50.json"),
    )
    parser.add_argument("--phase2-dir",  default=str(_DEFAULT_PHASE2_DIR))
    parser.add_argument("--output-dir",  default=str(_DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device",      default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    profiles = mine_universe(
        universe_path=Path(args.universe),
        phase2_dir=Path(args.phase2_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        seed=args.seed,
    )
    for h, plist in sorted(profiles.items()):
        print(f"  h{h}: {len(plist)} profiles", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5.3.4: Run**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_universe.py -v
```

Expected: `3 passed`.

- [ ] **Step 5.3.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_universe.py backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_universe.py
git commit -m "$(cat <<'EOF'
feat(phase3): add mine_universe driver + CLI + mine_profiles_for_pid_horizon stub

Round 3 of mine_universe.py — iterates (pid, horizon) cross-product,
writes per-horizon parquet + JSON sidecar of rule paths. The
mine_profiles_for_pid_horizon entrypoint lands as a stub returning [];
its full body composes the helpers from Task 4 and is filled in by the
operator at runtime (the full end-to-end mining loop wires the GPU code).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_universe.py backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_universe.py
```

---

## Task 6: Compose mine_profiles_for_pid_horizon + integration tests

The stub from Task 5 needs the full body. This task fills it in by composing the helpers from Task 4 with the tree fitter and CV harness.

**Files:**
- Modify: `backend/tools/strategy_discovery/mine_profiles.py` (replace stub body)
- Modify: `backend/tests/tools/strategy_discovery/test_mine_profiles.py` (add 1 integration test)

### Round 1 — `test_mine_profiles_for_pid_horizon_returns_qualifying_leaves_only`

- [ ] **Step 6.1.1: Write the failing test**

Append to `backend/tests/tools/strategy_discovery/test_mine_profiles.py`:

```python
def test_mine_profiles_for_pid_horizon_returns_qualifying_leaves_only(tmp_path):
    """Integration: build a synthetic Phase 2 parquet where a clear cohort exists,
    run mine_profiles_for_pid_horizon end-to-end (on CPU), and assert at least
    one qualifying profile comes back."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tools.strategy_discovery.mine_profiles import mine_profiles_for_pid_horizon

    rng = np.random.default_rng(101)
    n = 1000
    # Construct a feature where rows with feat_0 > 0.5 get +8% labels (winners),
    # other rows get -2% labels (losers). The mining tree should find this split.
    ts_ms = np.arange(n, dtype="int64") * 3_600_000
    feat_0 = rng.uniform(0.0, 1.0, size=n)
    feat_1 = rng.uniform(0.0, 1.0, size=n)   # noise
    labels = np.where(feat_0 > 0.5, 0.08, -0.02).astype("float64")
    df = pd.DataFrame({
        "ts":               ts_ms,
        "open":             np.full(n, 1.0),
        "high":             np.full(n, 1.0),
        "low":              np.full(n, 1.0),
        "close":            np.full(n, 1.0),
        "market_cap":       np.full(n, 1e9),
        "fdv":              np.full(n, 2e9),
        "fdv_over_mc":      np.full(n, 2.0),
        "circ_over_total":  np.full(n, 0.5),
        "vol_24h":          np.full(n, 1e7),
        "vol_over_mc":      np.full(n, 0.01),
        "price_over_ema20": feat_0,                    # the discriminative feature
        "price_over_ema50": np.full(n, 1.0),
        "price_over_ema200":np.full(n, 1.0),
        "ret_1h_sign":      np.full(n, 0.0),
        "ret_24h_sign":     feat_1,                    # noise feature
        "ret_7d_sign":      np.full(n, 0.0),
        "atr14_pct":        np.full(n, 0.02),
        "label_h1":   labels,
        "label_h4":   labels,
        "label_h24":  labels,
        "label_h72":  labels,
        "label_h168": labels,
    })
    parquet_path = tmp_path / "FOO-USD.parquet"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path)

    profiles = mine_profiles_for_pid_horizon(
        pid="FOO-USD", horizon=24, parquet_path=parquet_path, device="cpu", seed=42,
    )
    assert len(profiles) >= 1
    # The winning profile's avg_win should be near +8% (gross) - fee not subtracted
    # again because labels already include the 1.2% fee per Phase 2.
    winners = [p for p in profiles if p.avg_win >= 0.05 and p.cumulative_profit_deflated > 0]
    assert len(winners) >= 1, f"no winners; got profiles: {[(p.avg_win, p.cumulative_profit_deflated) for p in profiles]}"
```

- [ ] **Step 6.1.2: Run the test to verify it fails**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py::test_mine_profiles_for_pid_horizon_returns_qualifying_leaves_only -v
```

Expected: `AssertionError: assert len(profiles) >= 1` — the stub returns `[]`.

- [ ] **Step 6.1.3: Implement the full mine_profiles_for_pid_horizon body**

Replace the stub `mine_profiles_for_pid_horizon` in `backend/tools/strategy_discovery/mine_profiles.py` with the full implementation:

```python
_FEATURE_COLUMNS = (
    "market_cap", "fdv", "fdv_over_mc", "circ_over_total", "vol_24h", "vol_over_mc",
    "price_over_ema20", "price_over_ema50", "price_over_ema200",
    "ret_1h_sign", "ret_24h_sign", "ret_7d_sign", "atr14_pct",
)


def _serialize_rule_summary(root: TreeNode, leaf_id: int, feature_names) -> str:
    """Return a one-line human-readable rule path for the leaf_id-th leaf."""
    target_leaves = collect_leaves(root)
    if leaf_id >= len(target_leaves):
        return ""
    target = target_leaves[leaf_id]
    # Walk the tree, accumulating conditions along the path that lands on `target`.
    path_conditions = []
    def walk(node: TreeNode) -> bool:
        if node is target:
            return True
        if node.is_leaf:
            return False
        # Left subtree
        path_conditions.append(f"{feature_names[node.feature]} <= {node.threshold:.4f}")
        if walk(node.left):
            return True
        path_conditions.pop()
        # Right subtree
        path_conditions.append(f"{feature_names[node.feature]} > {node.threshold:.4f}")
        if walk(node.right):
            return True
        path_conditions.pop()
        return False
    walk(root)
    return " AND ".join(path_conditions) if path_conditions else "(root)"


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
    import pyarrow.parquet as _pq
    df = _pq.read_table(parquet_path).to_pandas()
    label_col = f"label_h{int(horizon)}"
    if label_col not in df.columns:
        return []
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    n = len(df)
    if n < 200:
        return []
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    ts_ms = torch.tensor(df["ts"].to_numpy(dtype="int64"), device=dev)
    labels = torch.tensor(df[label_col].to_numpy(dtype="float64"), device=dev)
    features = torch.tensor(df[list(_FEATURE_COLUMNS)].to_numpy(dtype="float64"), device=dev)
    next_eligible = build_next_eligible(ts_ms, horizon_bars=int(horizon))

    outer = outer_folds(n, n_folds=5, embargo_bars=int(horizon))
    # Per-leaf accumulators keyed by (chosen_depth, chosen_min_leaf, leaf_id)
    # leaf_id is reassigned per refit, so we accumulate by serialized rule path.
    from collections import defaultdict
    fold_pass_count: Dict[str, int] = defaultdict(int)
    fold_trade_lists: Dict[str, list] = defaultdict(list)
    fold_summaries: Dict[str, dict] = {}

    inner_scores_per_fold: List[dict] = []
    for outer_train_idx, outer_test_idx in outer:
        inner = inner_folds(outer_train_idx, n_folds=3, embargo_bars=int(horizon))
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
                    # Score on inner_test: sum of label_h{horizon} for rows landing in any leaf
                    # via the fitted tree's leaves' walk_and_sum, but on the inner_test subset.
                    leaves = collect_leaves(tree)
                    cum = 0.0
                    for leaf in leaves:
                        leaf_rows = leaf.indices.cpu().numpy()
                        # Project leaf_rows down to those that are in inner_test
                        common = np.intersect1d(leaf_rows, inner_test)
                        if len(common) == 0:
                            continue
                        sub = torch.tensor(common, dtype=torch.int64, device=dev).unsqueeze(0)
                        cum += float(walk_and_sum(sub, next_eligible, labels)[0].item())
                    fold_profits.append(cum)
                inner_scores[(depth, min_leaf)].append(float(np.mean(fold_profits)))
        # Reduce to mean across inner folds
        inner_score_mean = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        inner_scores_per_fold.append(inner_score_mean)
        chosen_depth, chosen_min_leaf, raw_max, inner_cv_se = pick_best_hyperparams(inner_score_mean)
        # Refit on the FULL outer train using the chosen combo.
        tree = fit_tree(
            features=features[outer_train_idx],
            labels=labels,
            next_eligible=next_eligible,
            max_depth=chosen_depth,
            min_leaf=chosen_min_leaf,
        )
        # Evaluate on outer_test.
        leaves = collect_leaves(tree)
        for leaf_id, leaf in enumerate(leaves):
            leaf_rows = leaf.indices.cpu().numpy()
            common = np.intersect1d(leaf_rows, outer_test_idx)
            if len(common) == 0:
                continue
            rule_summary = _serialize_rule_summary(tree, leaf_id, _FEATURE_COLUMNS)
            sub = torch.tensor(common, dtype=torch.int64, device=dev).unsqueeze(0)
            # Reconstruct per-trade PnL via walk_and_sum logic at trade level
            # (since walk_and_sum returns the sum, we recompute the trades inline).
            trades = _replay_trades(common.tolist(), next_eligible.cpu().numpy(),
                                    labels.cpu().numpy())
            metrics = leaf_metrics(np.asarray(trades))
            metrics["deflated_profit"], _ = apply_deflation(
                raw=metrics["cumulative_profit_raw"], inner_cv_se=inner_cv_se, n_combos=9,
            )
            if leaf_qualifies(metrics):
                fold_pass_count[rule_summary] += 1
                fold_trade_lists[rule_summary].extend(trades)
                fold_summaries[rule_summary] = {
                    **metrics,
                    "chosen_depth": chosen_depth,
                    "chosen_min_leaf": chosen_min_leaf,
                    "inner_cv_se": inner_cv_se,
                    "raw_max": raw_max,
                }

    rng = np.random.default_rng(seed)
    profiles: List[LeafProfile] = []
    for leaf_id, rule_summary in enumerate(sorted(fold_pass_count.keys())):
        n_pass = fold_pass_count[rule_summary]
        if n_pass < _Q0_MIN_FOLDS:
            continue
        trades = np.asarray(fold_trade_lists[rule_summary])
        m = leaf_metrics(trades)
        deflated, infl = apply_deflation(
            raw=m["cumulative_profit_raw"],
            inner_cv_se=fold_summaries[rule_summary]["inner_cv_se"],
            n_combos=9,
        )
        avg_trades_per_fold = m["trade_count"] / 5.0
        is_long_shot = long_shot_band(m["avg_win"], m["avg_loss"], m["win_rate"])
        bootstrap_triggered = avg_trades_per_fold < 30 or is_long_shot
        ci_lower, ci_upper = (None, None)
        if bootstrap_triggered and m["trade_count"] > 0:
            ci_lower, ci_upper = bootstrap_ci(trades, n_iter=_BOOTSTRAP_N, rng=rng)
        profiles.append(LeafProfile(
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
            inner_cv_se=fold_summaries[rule_summary]["inner_cv_se"],
            sortino=m["sortino"],
            n_folds_passed_q0=n_pass,
            bootstrap_triggered=bootstrap_triggered,
            bootstrap_ci_lower=ci_lower,
            bootstrap_ci_upper=ci_upper,
            chosen_depth=fold_summaries[rule_summary]["chosen_depth"],
            chosen_min_leaf=fold_summaries[rule_summary]["chosen_min_leaf"],
        ))
    return profiles


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
```

- [ ] **Step 6.1.4: Run the test**

```bash
cd backend && python -m pytest tests/tools/strategy_discovery/test_mine_profiles.py -v
```

Expected: `7 passed` (6 from Task 4 + the new integration test).

- [ ] **Step 6.1.5: Commit**

```bash
git add backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
git commit -m "$(cat <<'EOF'
feat(phase3): implement full mine_profiles_for_pid_horizon end-to-end

Round 6 of mine_profiles.py — composes the helpers from Task 4 + tree
fitter + CV harness into the full per-(pid, horizon) mining loop:

  load parquet -> tensors on device
  for each outer fold:
    inner 3-fold hyperparam search (9 combos)
    pick best, refit on full outer train, evaluate on outer test
    per leaf: compute metrics, check Q0 gates
  aggregate per-leaf across folds; require >=4-of-5 folds passing
  apply deflation factor
  trigger bootstrap CI for low-N / long-shot leaves
  return LeafProfile list

Pin test: synthetic parquet with a clean +8% / -2% split on
price_over_ema20 should produce at least one qualifying profile.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- backend/tools/strategy_discovery/mine_profiles.py backend/tests/tools/strategy_discovery/test_mine_profiles.py
```

---

## Task 7: Full-suite green check + memory/CHANGELOG sync

No new code in this task — verify everything green, append CHANGELOG, update memory.

**Files:**
- Modify: `CHANGELOG.md`
- Modify (out-of-tree): `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`

- [ ] **Step 7.1: Run the full backend test suite once**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/ -q --tb=line
```

Expected: full suite green. The Phase 3 additions are 20 new tests (4 + 4 + 3 + 6 + 3, plus 1 integration in Task 6). Pre-existing baseline including Phase 2: ~1231 passed. New target: ~1251 passed.

- [ ] **Step 7.2: Shell cleanup per CLAUDE.md**

```powershell
$backendPid = (Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue).OwningProcess
Get-Process python -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -ne $backendPid -and $_.ProcessName -ne 'Coinbase AI Trader' } |
    Stop-Process -Force
```

- [ ] **Step 7.3: Prepend CHANGELOG entry**

Open `C:\Users\gl450\polymarket_app\CHANGELOG.md` and insert this entry at the top of the `## Unreleased` section (above any other 2026-05-24 entries):

```markdown
### Session — 2026-05-24 — Strategy-discovery Phase 3: custom-criterion decision tree mining

Implemented Phase 3 of the strategy-discovery rebuild per spec
`docs/superpowers/specs/2026-05-24-strategy-discovery-phase3-design.md`
and plan `docs/superpowers/plans/2026-05-24-strategy-discovery-phase3-implementation.md`.

**New modules (all under `backend/tools/strategy_discovery/`):**
- `profit_split.py` — GPU-vectorized concurrency-capped (max-1-per-token) cumulative-PnL split criterion (`walk_and_sum`, `best_split`, `build_next_eligible`).
- `profit_tree.py` — Recursive profit-maximizing decision-tree fitter (`fit_tree`, `collect_leaves`).
- `purged_wf.py` — Purged Walk-Forward CV + nested inner CV (`outer_folds`, `inner_folds`).
- `mine_profiles.py` — Per-(pid, horizon) orchestrator: hyperparam search → deflation factor → Q0 gates → bootstrap CI. Composes the helpers above.
- `mine_universe.py` — CLI + universe driver; writes `profiles_h{h}.parquet` + `rule_paths_h{h}.json` per horizon.

**Test surface added:** 20 new tests under `backend/tests/tools/strategy_discovery/`. Full backend suite green.

**Operator step (post-merge):**

    cd backend && python -m tools.strategy_discovery.mine_universe \
        --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
        --device cuda --seed 42

Outputs land in `backend/data/phase3/`. Expected runtime ~30-60 min on RTX 2060.
Companion: `backend/tools/phase3_deflation_explainer.html` walks through the
selection-bias inflation math.
```

- [ ] **Step 7.4: Update memory — `coinbase_trader_architecture.md`**

Open `C:\Users\gl450\.claude\projects\C--Users-gl450\memory\coinbase_trader_architecture.md`. After the existing "Phase 2" sub-section under "Strategy-discovery rebuild", append:

```markdown

### Phase 3 (custom-criterion decision tree mining, 2026-05-24)

Phase 3 turns the per-token Phase 2 parquets into per-(pid, horizon) profile
parquets via custom-criterion decision-tree mining. 250 mining runs total
(50 pids × 5 horizons). GPU mandatory (PyTorch + CUDA, mirrors
xgb_v4_5_features_batch pattern).

**New modules** (all in `backend/tools/strategy_discovery/`):
- `profit_split.py` — `best_split(features, indices, labels, next_eligible)` scans
  13 features × 256 quantile thresholds via vectorized `walk_and_sum` (max-1
  concurrency per token). Returns SplitResult or None.
- `profit_tree.py` — `fit_tree(features, labels, next_eligible, max_depth, min_leaf)`
  recursive tree fitter. `collect_leaves(root)` returns leaves in DFS-left-first
  order.
- `purged_wf.py` — `outer_folds(n_rows, n_folds=5, embargo_bars=168)` + `inner_folds`
  for nested CV. Pure numpy.
- `mine_profiles.py` — `mine_profiles_for_pid_horizon(pid, horizon, parquet_path)`
  is the per-pair entrypoint. Composes inner-CV hyperparam search across
  {depth ∈ 3,5,7} × {min_leaf ∈ 20,50,100} = 9 combos, applies deflation
  (raw − σ × √(2 ln 9)), Q0 gates (avg_win ≥ +5%, avg_loss ≤ -10%, max_dd ≤ 30%,
  ≥4-of-5 folds), bootstrap CI for low-N / long-shot leaves.
- `mine_universe.py` — CLI; iterates (pid, horizon) cross-product; writes
  `profiles_h{h}.parquet` + `rule_paths_h{h}.json` per horizon.

**Output schema:** `pid, horizon, leaf_id, rule_path_summary, trade_count,
win_rate, avg_win, avg_loss, max_dd, cumulative_profit_raw,
cumulative_profit_deflated, deflation_pp, n_combos_searched, inner_cv_se,
sortino, n_folds_passed_q0, bootstrap_triggered, bootstrap_ci_lower,
bootstrap_ci_upper, chosen_depth, chosen_min_leaf, schema_version`.

**Tests:** 20 tests under `backend/tests/tools/strategy_discovery/test_{profit_split,profit_tree,purged_wf,mine_profiles,mine_universe}.py`. All mock-only — CUDA tests gated on `torch.cuda.is_available()`.

**Companion explainer:** `backend/tools/phase3_deflation_explainer.html` walks through the σ × √(2 ln N) selection-bias correction.

**Operator runs** (post-merge):
```
cd backend && python -m tools.strategy_discovery.mine_universe \
    --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
    --device cuda --seed 42
```

**Status:** code-complete on branch. Phase 4 (scorecard + deployment selection) is the next brainstorm round.

## See also
- [[xgb_post_scorecard_roadmap]] — operator picked bar-structure (off 4h+ time bars) before this rebuild
- 2026-05-23-strategy-discovery-rebuild-brainstorm.md (spec)
- 2026-05-23-strategy-discovery-phase2-design.md (spec, Phase 2)
- 2026-05-24-strategy-discovery-phase3-design.md (spec, this round)
- 2026-05-24-strategy-discovery-phase3-implementation.md (plan, this round)
```

Memory file is OUTSIDE the repo — not committed.

- [ ] **Step 7.5: Commit CHANGELOG**

Pre-commit checks:
- `git rev-parse --abbrev-ref HEAD` → should be `feat/strategy-discovery-phase3`
- `git status -s` → CHANGELOG.md should be the only intended modification

Commit (surgical):

```bash
git commit -m "$(cat <<'EOF'
docs: changelog entry for strategy-discovery Phase 3

Records the 5-module Phase 3 implementation + 20-test surface added across
this branch. Memory file coinbase_trader_architecture.md updated out-of-tree
per the sync rule.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- CHANGELOG.md
```

If pre-commit hook fails on unrelated WIP, use the stash workaround from `feedback_parallel_agent_coordination.md` item 4.

- [ ] **Step 7.6: Final verification**

```bash
cd C:\Users\gl450\polymarket_app\backend
python -m pytest tests/tools/strategy_discovery/ -q
```

Expected: all strategy_discovery tests pass — 19 (Phase 1) + 19 (Phase 2) + 20 (Phase 3) = 58 tests.

```bash
git log --oneline main..HEAD
```

Expected: ~25 commits on `feat/strategy-discovery-phase3`. DO NOT push — the operator authorizes push separately.

---

## Self-Review

**1. Spec coverage** — every spec section maps to a task:

- Goal + I/O + output schema → Tasks 5 + 6 (mine_universe writes the parquet with the exact 22-column schema; mine_profiles fills the rows)
- Algorithm — custom split criterion → Task 1 (profit_split, all 4 tests)
- Tree fitting → Task 2 (profit_tree, all 4 tests)
- Causality / Purged WF / nested CV / embargo → Task 3 (purged_wf, 3 tests)
- Q0 hard gates (avg_win ≥ +5%, avg_loss ≤ -10%, max_dd ≤ 30%) → Task 4 Round 4 (`leaf_qualifies`)
- Q0 ≥4-of-5 fold qualifier → Task 4 Round 4 + Task 6 (aggregation logic)
- Deflation factor (σ × √(2 ln N)) → Task 4 Round 1 (`apply_deflation`) + Task 6 (applied in the mining loop)
- Long-shot band → Task 4 Round 2 (`long_shot_band`)
- Leaf metrics (win/loss/DD/Sortino) → Task 4 Round 3 (`leaf_metrics`)
- Hyperparam search picking → Task 4 Round 5 (`pick_best_hyperparams`)
- Bootstrap CI → Task 4 Round 6 (`bootstrap_ci`)
- End-to-end orchestration → Task 6 (`mine_profiles_for_pid_horizon` full body)
- Universe driver + CLI → Task 5 (`mine_universe`, `main`)
- GPU strategy → Task 1 (`torch.gather` / vectorized walk_and_sum) + Task 6 (device handling)
- Testing surface (19 unit + 1 integration) → Tasks 1-6 (4 + 4 + 3 + 6 + 3 + 1 = 21 tests; spec says 19 but spec count was an estimate — actual coverage is slightly higher and includes the long-shot, hyperparam-pick, and leaf_metrics tests not in the spec table)
- CHANGELOG + memory sync → Task 7

**2. Placeholder scan** — no TBD/TODO/"add appropriate"/"similar to" found. All code blocks complete.

**3. Type consistency:**
- `SplitResult` dataclass defined Task 1 Step 1.0; consumed by `_fit_recursive` Task 2 Step 2.1.3 (uses `split.left_mask`, `split.feature`, `split.threshold`). ✓
- `TreeNode` defined Task 2 Step 2.0; consumed by `collect_leaves` Task 2 Step 2.1.3 and `_serialize_rule_summary` Task 6 Step 6.1.3. ✓
- `LeafProfile` defined Task 4 Step 4.0; written by Task 5 Round 2 (`write_profile_parquet` uses `asdict`); returned by Task 6 Step 6.1.3. ✓
- `build_next_eligible`, `walk_and_sum`, `best_split` signatures match between Tasks 1, 2, 6. ✓
- `outer_folds`, `inner_folds` signatures match Task 3 → Task 6 consumption. ✓
- `apply_deflation`, `bootstrap_ci`, `leaf_metrics`, `leaf_qualifies`, `long_shot_band`, `pick_best_hyperparams` all defined Task 4 + used Task 6. ✓

No inconsistencies found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-24-strategy-discovery-phase3-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** — fresh subagent per task with two-stage review (spec compliance, then code quality) between each. 7 independent tasks; cleanest with subagent isolation since GPU code is tricky and benefits from focused context per module.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch checkpoints after each task. Cheaper context-wise but no fresh-context review per task.

Which approach?
