# Strategy Discovery Rebuild — Phase 3 Design Spec

**Date:** 2026-05-24
**Author:** Claude Opus 4.7 (post Phase 2 cutover)
**Status:** Approved (operator 2026-05-24)
**Predecessor brainstorms:**
- `2026-05-23-strategy-discovery-rebuild-brainstorm.md` (Phases 1–4, Q0–Q6 decisions)
- `2026-05-23-strategy-discovery-phase2-design.md` (Phase 2 spec, complete)

**Phase 2 plan:** `2026-05-23-strategy-discovery-phase2-implementation.md` (complete — PR #2 on `feat/strategy-discovery-phase2-clean`)

**Companion artifact:** `backend/tools/phase3_deflation_explainer.html` (worked-example walkthrough of the deflation factor)

---

## Goal

For each `(pid, horizon) ∈ universe × {1, 4, 24, 72, 168}`, mine a custom-criterion decision tree from the Phase 2 per-token parquet and emit a ranked list of leaf **profiles**. Each profile = `(rule_path, win_rate, avg_win, avg_loss, max_DD, cumulative_profit_raw, cumulative_profit_deflated, sortino, trade_count, bootstrap_CI)`. Q0 hard gates are applied per leaf at the Purged-WF fold level; only leaves passing on ≥4 of 5 outer folds become profiles. Phase 4 consumes the profile parquets for scorecard ranking and deployment selection.

Phase 3 owns: the custom split criterion, the per-token tree fitter, the Purged-WF + nested-CV harness, the bootstrap CI layer, the deflation factor, and the universe orchestrator + CLI. Phase 3 does NOT own: cross-profile scorecard ranking, deployment artifacts, runtime inference path — those live in Phase 4.

---

## Inputs (from Phase 2)

| Source | Path | Content |
|---|---|---|
| Per-token parquet | `backend/data/phase2/{pid}.parquet` (≈50 files) | 13 features (`market_cap`, `fdv`, `fdv_over_mc`, `circ_over_total`, `vol_24h`, `vol_over_mc`, `price_over_ema20`, `price_over_ema50`, `price_over_ema200`, `ret_1h_sign`, `ret_24h_sign`, `ret_7d_sign`, `atr14_pct`) + 5 horizon labels (`label_h1`, `label_h4`, `label_h24`, `label_h72`, `label_h168`) + identifiers (`ts`, `pid`, `schema_version`) |
| Curated universe | `docs/superpowers/specs/2026-05-23-universe-50.json` | `{cohort: [pid]}` for the 50-pid universe |

---

## Outputs

| Path | Content |
|---|---|
| `backend/data/phase3/profiles_h{h}.parquet` (5 files) | One row per surviving leaf across all pids for horizon `h`. Columns listed below. |
| `backend/data/phase3/rule_paths_h{h}.json` (5 sidecars) | `{leaf_uid: "vol_over_mc > 0.08 AND price_over_ema20 > 1.02"}` — human-readable rule strings keyed by `leaf_uid = "{pid}__{leaf_id}"`. |
| `backend/data/phase3/mining_summary.md` | Operator-facing audit log: per-pid CV scores, hyperparam picks, deflation factors, leaves pruned by which Q0 gate. |

### Profile-parquet row schema

| Column | Type | Source |
|---|---|---|
| `pid` | str | identifier |
| `horizon` | int | the bar count (1, 4, 24, 72, 168) |
| `leaf_id` | int | local to the (pid, horizon) tree (0-indexed leaf order) |
| `rule_path_summary` | str | one-line human-readable (full path lives in the sidecar JSON) |
| `trade_count` | int | over all 5 outer folds combined |
| `win_rate` | float | fraction of trades with positive net PnL |
| `avg_win` | float | mean net PnL of winning trades (after 1.2% round-trip fee) |
| `avg_loss` | float | mean net PnL of losing trades (negative; magnitude compared to -10% gate) |
| `max_dd` | float | worst peak-to-trough drawdown on the leaf's equity curve |
| `cumulative_profit_raw` | float | sum of net PnL across all leaf trades (search-best, inflated) |
| `cumulative_profit_deflated` | float | raw − σ × √(2·ln(9)); **the ranking metric** |
| `deflation_pp` | float | the σ × √(2·ln(9)) gap in percentage points |
| `n_combos_searched` | int | constant 9 in v1 (one per hyperparam combo) |
| `inner_cv_se` | float | per-combo standard error from inner 3-fold CV, drives deflation |
| `sortino` | float | secondary ranking; computed on leaf trades |
| `n_folds_passed_q0` | int | 0–5; profile must have ≥4 |
| `bootstrap_triggered` | bool | True if trade_count_per_fold < 30 OR profile in long-shot band |
| `bootstrap_ci_lower` | float | 95% CI lower bound on cumulative_profit_deflated (NaN if not triggered) |
| `bootstrap_ci_upper` | float | 95% CI upper bound (NaN if not triggered) |
| `chosen_depth` | int | hyperparam pick from inner CV (∈ {3,5,7}) |
| `chosen_min_leaf` | int | hyperparam pick (∈ {20,50,100}) |
| `schema_version` | int | constant `1` |

---

## Locked decisions (from brainstorm 2026-05-23 + Phase 3 questions 2026-05-24)

| # | Decision | Source |
|---|---|---|
| 1 | Mine **one tree per horizon** (5 mining runs, 5 ranked profile lists) | Phase 3 brainstorm 2026-05-24 |
| 2 | **Per-token mining** — 50 trees per horizon × 5 = 250 trees | Phase 3 brainstorm 2026-05-24 |
| 3 | **Split criterion = concurrency-capped cumulative PnL**, max-1-per-token | Phase 3 brainstorm 2026-05-24 |
| 4 | **Per-token concurrency rule = max 1 open position per token**; portfolio-level 5-cap deferred to Phase 4 | Phase 3 brainstorm 2026-05-24 |
| 5 | **Hyperparam search via nested CV per token**: depth ∈ {3,5,7} × min_leaf ∈ {20,50,100} = 9 combos, 3-fold inner WF | Phase 3 brainstorm 2026-05-24 |
| 6 | **Output = parquet of profile metrics + JSON of rule paths** | Phase 3 brainstorm 2026-05-24 |
| 7 | **GPU training mandatory** (PyTorch + CUDA), CPU fallback | Operator instruction 2026-05-24 |
| Q0 | Hard gates: avg_win ≥ +5% net, avg_loss ≤ −10% net magnitude, max_DD ≤ 30% | Brainstorm 2026-05-23 |
| Q0 | Primary ranking = cumulative profit (after retail 1.2% fee) | Brainstorm 2026-05-23 |
| Q0 | Secondary ranking = Sortino | Brainstorm 2026-05-23 |
| Q3 | 13-feature input set (Phase 2 expanded the original 11 with EMA50 + EMA200) | Phase 2 design |
| Q6 | Validation = Purged WF (5-fold, embargo = max horizon = 168h = 7d) + Bootstrap CI for low-N / long-shot | Brainstorm 2026-05-23 |

---

## Algorithm — Custom split criterion

For a candidate split `(feature_j, threshold_t)` on the current node's row subset `R`:

```
left_subset  = R[ R[feature_j] <= threshold_t ]
right_subset = R[ R[feature_j] >  threshold_t ]

for subset in (left, right):
    # Order rows chronologically by ts (these are candidate entry hours).
    # Walk forward: at each row, if no open position in this token, "enter"
    # (size = 1 unit per Q0 sizing rule); the trade's exit-bar is at
    # row.ts + horizon * 1h, with realized PnL = row.label_h{h}.
    # If a position is still open (entry_ts < row.ts <= entry_ts + horizon),
    # skip this row (no new entry).
    cumulative_pnl[subset] = walk_and_sum(subset, horizon)

split_metric = max(cumulative_pnl[left], cumulative_pnl[right])
```

The split that maximizes `split_metric` wins. The "rejected" subgroup is not orphaned — it becomes its own subtree and gets re-split until depth/leaf constraints stop it. Both leaves contribute to the final profile list independently.

### Candidate-threshold enumeration

Per node and per feature: 256 quantile thresholds from the feature's distribution within the current row subset. 13 features × 256 = **3,328 candidate splits per node**.

### GPU vectorization

- Per pid: load Phase 2 parquet → `torch.Tensor` on `cuda` once (small: ~4,000 × 18 × float32 ≈ 280 KB).
- Precompute `next_eligible_idx[i] = min { j > i : ts[j] > ts[i] + horizon × 3600s }` (one int64 vector per pid, ~32 KB; O(n) once).
- Per split-scan: vectorized `walk_and_sum` using `torch.gather` over the precomputed eligibility index. All 3,328 candidate splits processed in one kernel launch.
- Memory budget per node: 3,328 candidates × 4,000 rows × float32 ≈ 50 MB. Fits comfortably on RTX 2060 (6 GB).

### Tree fitting loop

```python
def fit(rows, depth=0):
    if depth >= max_depth:           return Leaf(rows)
    if len(rows) < 2 * min_leaf:     return Leaf(rows)
    split = best_split(rows)         # GPU-vectorized scan
    if split is None:                return Leaf(rows)    # no profitable split
    if min(|left|, |right|) < min_leaf: return Leaf(rows)
    return Node(split,
                fit(rows[left],  depth+1),
                fit(rows[right], depth+1))
```

`best_split` returns `None` if no candidate split produces a strictly positive `split_metric` (cumulative PnL ≤ 0 in both subgroups → no profitable cohort to extract → stop here).

### Cross-tree batching

Per-token trees are independent. Fit them in batches of ~10 by stacking 10 pids' row tensors into one padded `(10, max_rows, 18)` tensor and running the split scan as a batched op. Batch size auto-tuned via `torch.cuda.mem_get_info()` at startup.

---

## Validation

### Outer Purged Walk-Forward CV (5 folds, embargo = 168 bars / 7d)

Each pid's parquet rows are split chronologically into 5 contiguous folds. For fold `k`:
- Train rows = folds `1..k-1`
- Test rows = fold `k`
- **Embargo:** drop any train row whose `ts ∈ [test_start - 168h, test_start)` to prevent label leakage (a train-row `label_h168` could span into the test fold).

### Nested CV for hyperparameter search

The 9 hyperparam combos {depth ∈ 3,5,7} × {min_leaf ∈ 20,50,100} are scored via an inner WF on the training portion of each outer fold:

- Inner WF = 3 folds (not 5) within each outer training set.
- For each (combo, inner fold): fit tree → simulate on inner test → measure cumulative profit.
- Best combo = argmax of mean inner-test cumulative profit across the 3 inner folds.
- Refit the chosen combo on the FULL outer training set; evaluate on the outer test fold.

**Compute budget per (pid, horizon):** 5 outer × 3 inner × 9 combos = 135 hyperparam fits + 5 outer refits = 140 fits. Across 250 (pid, horizon) pairs = ≈35,000 tree fits. GPU target wall-time: **<60 min on RTX 2060**; CPU fallback: ~4–6 hours.

### Q0 hard gates applied at OUTER fold level

For each leaf in the outer-fold tree:
- `avg_win ≥ +5%` net (measured on outer test trades)
- `avg_loss ≤ -10%` net magnitude
- `max_DD ≤ 30%` (computed on the outer-test cumulative equity curve of just this leaf's trades)

A leaf must pass these on **≥4 of 5 outer folds** to qualify as a profile.

### Bootstrap CI (layered for low-N / long-shot)

A leaf triggers bootstrap CI if either:
- `trade_count_per_fold < 30` (low-frequency), OR
- Profile lands in long-shot band: `avg_win ≥ +15% AND |avg_loss| ≤ 7% AND win_rate ≥ 70%`

Procedure: resample the leaf's trade list (across all 5 outer folds) with replacement, 1,000 iterations. Compute 95% CI on `cumulative_profit_deflated` and `sortino`. Report CI alongside point estimate. **Does not gate** — operator-facing context only.

### Deflation factor (selection-bias correction)

The hyperparam search inspects 9 combos per (pid, horizon). Per the Deflated Sharpe Ratio diagnostic (task #16), the reported maximum is an upward-biased "best of N" pick. Apply:

```
inflation        = σ × √(2 × ln(N))    # N = 9 combos, σ = inner_cv_se
cumulative_profit_deflated = cumulative_profit_raw − inflation
```

Both raw and deflated cumulative profit are written to the output parquet. **Q0 gates and ranking both apply to the deflated number.** See `backend/tools/phase3_deflation_explainer.html` for the worked example.

---

## Module Structure

```
backend/tools/strategy_discovery/
  profit_split.py    (NEW)  Custom split criterion (concurrency-capped cumulative PnL, max 1/token)
  profit_tree.py     (NEW)  Recursive tree fitter
  purged_wf.py       (NEW)  Purged WF + nested-CV harness
  mine_profiles.py   (NEW)  Per-(pid, horizon) mining orchestrator + deflation + Q0 gate
  mine_universe.py   (NEW)  Universe driver + CLI
```

### `profit_split.py` — Public API

```python
def walk_and_sum(
    subset_indices: torch.Tensor,    # (B, K) row indices into the pid's full row tensor
    next_eligible: torch.Tensor,     # (N,) precomputed eligibility lookup
    labels: torch.Tensor,            # (N,) the chosen horizon's label column
) -> torch.Tensor                    # (B,) cumulative net PnL per candidate subset
    """Vectorized concurrency-capped (max-1) cumulative PnL across B candidate subsets."""

def best_split(
    rows: torch.Tensor,              # (n, 13) the current node's feature values
    indices: torch.Tensor,           # (n,) absolute row indices into the pid's full tensor
    labels: torch.Tensor,            # (N,) the pid's full label column
    next_eligible: torch.Tensor,     # (N,) precomputed
    n_thresholds: int = 256,
) -> Optional[SplitResult]
    """Returns the best (feature, threshold, left_idx, right_idx, score) or None
    if no candidate split produces strictly positive cumulative PnL in either side."""
```

### `profit_tree.py` — Public API

```python
@dataclass
class TreeNode:
    feature: Optional[int]           # None for leaf
    threshold: Optional[float]       # None for leaf
    left:  Optional["TreeNode"]
    right: Optional["TreeNode"]
    indices: torch.Tensor            # row indices in this node (used at leaf)
    cumulative_pnl: float            # leaf cumulative PnL

def fit_tree(
    features: torch.Tensor,          # (N, 13)
    labels: torch.Tensor,            # (N,)
    next_eligible: torch.Tensor,     # (N,)
    max_depth: int,
    min_leaf: int,
) -> TreeNode

def collect_leaves(root: TreeNode) -> List[TreeNode]
def serialize_rule_path(root: TreeNode, leaf: TreeNode, feature_names: List[str]) -> str
```

### `purged_wf.py` — Public API

```python
def outer_folds(
    n_rows: int,
    n_folds: int = 5,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]
    """Returns [(train_idx, test_idx), ...]. Train indices have the embargo applied."""

def inner_folds(
    train_idx: np.ndarray,
    n_folds: int = 3,
    embargo_bars: int = 168,
) -> List[Tuple[np.ndarray, np.ndarray]]
    """Same shape as outer_folds but on the outer-train subset only."""
```

### `mine_profiles.py` — Public API

```python
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
    bootstrap_ci_lower: Optional[float]
    bootstrap_ci_upper: Optional[float]
    chosen_depth: int
    chosen_min_leaf: int

def mine_profiles_for_pid_horizon(
    pid: str,
    horizon: int,
    parquet_path: Path,
    device: str = "cuda",
    seed: int = 42,
) -> List[LeafProfile]
    """End-to-end per-(pid, horizon) mining. Returns surviving leaves (Q0 gates,
    ≥4-of-5 outer folds). Bootstrap CI computed for triggered leaves."""
```

### `mine_universe.py` — Public API

```python
def mine_universe(
    universe_path: Path,
    phase2_dir: Path     = Path("backend/data/phase2"),
    output_dir: Path     = Path("backend/data/phase3"),
    horizons: List[int]  = [1, 4, 24, 72, 168],
    device: str          = "cuda",
    seed: int            = 42,
) -> Dict[int, List[LeafProfile]]
    """Iterate (pid, horizon) cross-product. Emit profiles_h{h}.parquet and
    rule_paths_h{h}.json per horizon, plus mining_summary.md."""

def main(argv: Optional[List[str]] = None) -> int
    """CLI entrypoint."""
```

Module boundary rationale: `profit_split` is a leaf-level pure function; `profit_tree` only knows about splits; `purged_wf` only knows about row indices; `mine_profiles` composes them and owns the deflation + Q0 gates; `mine_universe` is the only module that touches the filesystem.

---

## GPU Strategy

**Framework: PyTorch + CUDA** (mirrors `backend/tools/xgb_v4_5_features_batch.py` conventions — same `device={cpu,cuda}` CLI flag, same `torch.use_deterministic_algorithms(True, warn_only=True)` for reproducibility).

**What runs on GPU:**
- Parquet → tensor cast (per pid, once)
- `next_eligible_idx` lookup table (per pid, once)
- Per-node split scan (3,328 candidates per kernel launch)
- Bootstrap CI resampling (1,000 × 50 indices as a single `torch.gather`)

**What stays on CPU:**
- Tree-structure bookkeeping (Python objects: `(feature, threshold, left, right, leaf_stats)`)
- Hyperparameter search loop control (135 iterations per pid)
- Universe orchestration, parquet I/O, JSON serialization

**Cross-tree batching:** Fit up to 10 trees in parallel via padded `(10, max_rows, 18)` tensor. Batch size auto-tuned at startup based on `torch.cuda.mem_get_info()`.

**Determinism contract:** `--device cpu` and `--device cuda` produce results within `max_abs_diff < 1e-4` on `cumulative_profit` (drift from FP reduction order, matches the parallel agent's GPU contract). Same input + same seed + same `--device` → bit-identical profile parquet across runs.

**Compute estimates:**

| Path | Wall-time (full universe = 250 pairs × 140 fits = 35k fits) |
|---|---|
| `--device cuda` (RTX 2060 6GB) | ~30–60 min |
| `--device cpu` (8-core baseline) | ~4–6 hours |

---

## Testing

Mock-only per CLAUDE.md (no live GPU required for non-CUDA tests; `torch.cuda.is_available()` mockable).

| Test file | Test | What it pins |
|---|---|---|
| `test_profit_split.py` | `test_concurrency_max_1_skips_overlapping_entry` | Per-token cap of 1 enforced |
| | `test_split_metric_picks_higher_pnl_subgroup` | `max(left, right)` selection |
| | `test_walk_and_sum_matches_naive_python_reference` | GPU/vectorized vs scalar reference, rtol=1e-6 |
| | `test_no_profitable_split_returns_none` | None when both subgroups have cum_pnl ≤ 0 |
| `test_profit_tree.py` | `test_fit_respects_max_depth` | Depth cap honored |
| | `test_fit_respects_min_samples_per_leaf` | Min leaf size honored |
| | `test_fit_stops_on_unprofitable_split` | Early stop when `best_split` returns None |
| | `test_leaf_stats_are_pid_local` | No cross-pid contamination |
| `test_purged_wf.py` | `test_embargo_drops_train_rows_within_horizon_of_test_start` | Embargo correctness |
| | `test_5_folds_cover_all_rows_disjointly` | Fold partition correctness |
| | `test_nested_inner_cv_uses_only_outer_train` | No outer-test leakage to inner search |
| `test_mine_profiles.py` | `test_hyperparam_search_picks_best_inner_cv_combo` | Selection logic |
| | `test_deflation_factor_applied_to_reported_profit` | `raw − σ × √(2 ln N)` formula |
| | `test_q0_gates_applied_to_deflated_profit` | Gate uses deflated, not raw |
| | `test_bootstrap_ci_triggers_on_low_trade_count` | <30 trades/fold triggers |
| | `test_bootstrap_ci_triggers_on_long_shot_band` | `avg_win ≥ 15% AND |avg_loss| ≤ 7% AND wr ≥ 70%` triggers |
| | `test_4_of_5_folds_required_for_leaf_to_qualify` | Outer-fold qualification rule |
| `test_mine_universe.py` | `test_iterates_all_pid_horizon_pairs` | 50 pids × 5 horizons coverage |
| | `test_emits_profile_parquet_and_rule_json` | Output artifacts produced |
| | `test_cuda_and_cpu_paths_agree_within_1e_minus_4` | Device parity (skipif no CUDA) |

**Total new test surface:** 19 tests, mock-only.

---

## Operator Integration (post-implementation)

After Phase 3 code lands and tests pass:

```
cd backend && python -m tools.strategy_discovery.mine_universe \
    --universe ../docs/superpowers/specs/2026-05-23-universe-50.json \
    --device cuda --seed 42
```

Expected runtime: ~30–60 min on RTX 2060. Outputs land in `backend/data/phase3/`:
- `profiles_h{1,4,24,72,168}.parquet` (5 files)
- `rule_paths_h{1,4,24,72,168}.json` (5 sidecars)
- `mining_summary.md`

Operator reviews `mining_summary.md` for per-pid CV picks, deflation factors, leaves pruned by which gate. Then kicks off **Phase 4 brainstorm** (scorecard + deployment selection across the 5 ranked profile lists).

---

## What Phase 3 is NOT

- Not the scorecard / Q0-gate aggregation across the full profile set (Phase 4)
- Not cross-profile ranking or deployment artifact selection (Phase 4)
- Not the real-time inference path for deployed profiles (Phase 4)
- Not pooled cross-pid mining (per-token by Phase 3 brainstorm decision; sensitivity check is in the backlog)
- Not multi-horizon mixed-label mining (one tree per horizon by Phase 3 brainstorm decision)
- Not probe-style ablations ("what if we drop feature X?") — separate workflow

---

## Backlog (deferred)

| ID | Item | Trigger to revisit |
|---|---|---|
| (new) | Cross-pid pooled mining as sensitivity check | If per-token results show very few qualifying profiles |
| (new) | Cross-pid/cross-horizon deflation (Bonferroni across 250 pairs) | Phase 4, when ranking across the full pool |
| (new) | Per-leaf serialized tree object for runtime deployment | Phase 4 (when scorecard picks winners) |
| (new) | Ensemble of trees on different seeds | If single-tree variance proves too high |
| #18 | Sample-uniqueness weighting (overlapping forward-windows) | If per-fold trade counts are very low and label autocorrelation dominates |
| #17 | Combinatorial Purged CV (CPCV) upgrade | If 5-fold WF results are unstable across runs |
| #52 | Add 336h horizon | After max-hold redesign raises the deployed cap above 168 bars |

---

## See also

- `2026-05-23-strategy-discovery-rebuild-brainstorm.md` — original Phase 1–4 spec (Q0–Q6 decisions)
- `2026-05-23-strategy-discovery-phase2-design.md` — Phase 2 spec (input source for Phase 3)
- `2026-05-23-strategy-discovery-phase2-implementation.md` — Phase 2 plan (complete on `feat/strategy-discovery-phase2-clean`)
- `backend/tools/phase3_deflation_explainer.html` — companion worked-example for the deflation factor
- `coinbase_trader_architecture.md` (memory) — current backend architecture + strategy_discovery package overview
- Task #16 results (Deflated Sharpe Ratio diagnostic) — methodological ancestor of the deflation factor
