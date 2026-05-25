# Strategy Discovery Rebuild — Phase 4 Design Spec

**Date:** 2026-05-24
**Author:** Claude Opus 4.7 (post Phase 3 cutover)
**Status:** Approved (operator 2026-05-24)
**Predecessor brainstorms:**
- `2026-05-23-strategy-discovery-rebuild-brainstorm.md` (Phases 1–4, Q0–Q6 decisions)
- `2026-05-23-strategy-discovery-phase2-design.md` (Phase 2 spec, complete)
- `2026-05-24-strategy-discovery-phase3-design.md` (Phase 3 spec, complete)

**Phase 3 implementation:** PR #4 on `feat/strategy-discovery-phase3` (commits `41e4019..3c1a050`)

---

## Goal

Consume Phase 3's per-horizon profile parquets, run a portfolio-aware knapsack for each concurrency cap `N ∈ {3, 4, 5}`, and emit a scorecard + deployment artifact per cap. The operator reads the scorecard, picks the winning cap (or aborts if none qualify), and runs a separate integration step to wire the chosen deployment into the live agent.

Phase 4 owns: the portfolio simulator, the beam-search knapsack, the portfolio-level deflation factor, the per-cap scorecard with verdict, and the deployment-JSON emitter. Phase 4 does NOT own: the runtime inference path into `agents/cnn_agent.py` (that lands as a separate integration commit once a deployment is selected), live shadow telemetry, or cross-deployment ensembling.

---

## Inputs (from Phase 3)

| Source | Path | Content |
|---|---|---|
| Per-horizon profile parquets | `backend/data/phase3/profiles_h{1,4,24,72,168}.parquet` (5 files) | Up to ~50 leaves each (one per pid that survived Phase 3's ≥4-of-5 outer-fold Q0 gate). Schema includes `pid, horizon, leaf_id, rule_path_summary, cumulative_profit_raw, cumulative_profit_deflated, deflation_pp, win_rate, avg_win, avg_loss, max_dd, sortino, trade_count, n_folds_passed_q0, bootstrap_ci_lower/upper, chosen_depth, chosen_min_leaf, inner_cv_se`. |
| Rule-path sidecars | `backend/data/phase3/rule_paths_h{h}.json` (5 files) | `{leaf_uid: rule_path_string}` keyed by `"{pid}__{leaf_id}"`. |
| Phase 2 feature parquets | `backend/data/phase2/{pid}.parquet` (~50 files) | Re-loaded so the portfolio simulator can evaluate rule firings at each historical bar. Same 13-feature + 5-label schema Phase 3 consumed. |

---

## Outputs

| Path | Content |
|---|---|
| `backend/data/phase4/scorecard.md` | Operator-facing report: per-cap (N=3,4,5) portfolio metrics, gate pass/fail, comparison table, deploy/abort verdict. |
| `backend/data/phase4/deployment_n{3,4,5}.json` (3 files) | Per cap, the picked profile set with `{cap, profiles: [{pid, horizon, leaf_id, rule_path, expected_avg_win, expected_avg_loss, expected_max_dd, ...}]}`. What the live agent will load. |
| `backend/data/phase4/portfolio_telemetry_n{3,4,5}.parquet` (3 files) | Per-bar equity curve, slot utilization, fired-profile log. For post-deploy comparison + plotting. |

### deployment_n{N}.json schema

```json
{
  "cap": 3,
  "selected_at_utc": "2026-05-24T...",
  "k_subsets_evaluated": 12000,
  "portfolio_metrics": {
    "cumulative_profit_raw": 0.187,
    "cumulative_profit_deflated": 0.124,
    "deflation_pp": 0.063,
    "max_dd": 0.211,
    "sortino": 1.42,
    "trade_count": 142,
    "pct_slots_full": 0.34,
    "mean_concurrent": 1.8
  },
  "gates": {
    "max_dd_le_30": true,
    "deflated_profit_gt_0": true,
    "trade_count_ge_50": true,
    "sortino_ge_0": true,
    "overall": "pass"
  },
  "profiles": [
    {
      "pid": "BTC-USD",
      "horizon": 24,
      "leaf_id": 3,
      "rule_path": "vol_over_mc > 0.08 AND price_over_ema20 > 1.02",
      "expected_avg_win": 0.083,
      "expected_avg_loss": -0.054,
      "expected_max_dd": 0.219,
      "expected_trade_count": 47,
      "expected_sortino": 1.34,
      "phase3_cumulative_profit_deflated": 0.041
    }
    // ... up to cap entries
  ]
}
```

### portfolio_telemetry_n{N}.parquet row schema

| Column | Type | Source |
|---|---|---|
| `ts` | int64 (ms) | bar timestamp |
| `equity` | float64 | running equity including unrealized PnL of open positions |
| `n_open` | int | open positions at this bar (0..N) |
| `fired_profile_id` | str (nullable) | `"{pid}__{leaf_id}"` if a profile fired at this bar (entered), else null |
| `closed_profile_id` | str (nullable) | profile that closed at this bar (label realized), else null |
| `realized_pnl` | float64 (nullable) | net PnL of a closed trade at this bar, else null |
| `schema_version` | int | constant `1` |

---

## Locked decisions (from brainstorm 2026-05-24)

| # | Decision | Source |
|---|---|---|
| 1 | **Portfolio-aware knapsack selection** — search over subsets of size N, pick the subset maximizing deflated portfolio cumulative profit | Phase 4 brainstorm |
| 2 | **Cap sweep N ∈ {3, 4, 5}** — 3 separate knapsacks, 3 separate deployments, operator picks winner from scorecard | Phase 4 brainstorm |
| 3 | **Beam-search heuristic**: `BEAM_WIDTH = 20`, candidate pool = top-100 profiles by Phase 3 deflated profit, step count = N. K_evaluated ≈ N × 20 × 100 | Phase 4 brainstorm + spec |
| 4 | **Portfolio-level deflation factor** = `σ_portfolio × √(2 × ln K_evaluated)` where σ_portfolio is bootstrapped from the fired-trade list | Phase 4 brainstorm |
| 5 | **Tiebreaker on simultaneous fires** = highest `cumulative_profit_deflated`; never double-position the same pid (carries Phase 3 rule) | Phase 4 spec |
| 6 | **Portfolio Q0 gates**: max_dd ≤ 30%, deflated cum_profit > 0, trade_count ≥ 50, sortino ≥ 0. ALL must pass for cap to qualify. | Phase 4 spec |
| 7 | **Verdict** = highest deflated-profit passing cap → "deploy"; all 3 fail → "abort" | Phase 4 spec |
| 8 | **Exit PnL inherited from Phase 2 label_h{horizon}** — Phase 4 does NOT re-simulate exits | Phase 4 spec |

---

## Algorithm

### Portfolio simulator

For a candidate subset of profiles `S` and cap `N`, walk historical bars chronologically:

```
open_positions: List[(pid, profile_id, entry_ts, exit_ts, expected_pnl)] = []
trade_log: List[(ts, profile_id, pid, net_pnl)] = []
equity_curve: List[(ts, equity)] = []
slot_util_log: List[(ts, n_open)] = []
equity = 0.0

for ts in sorted_bars:
    # 1. Close positions whose exit_ts == ts
    closed = [p for p in open_positions if p.exit_ts == ts]
    for p in closed:
        equity += p.expected_pnl    # label_h{horizon} already net of 1.2% fee (Phase 2)
        trade_log.append((ts, p.profile_id, p.pid, p.expected_pnl))
        open_positions.remove(p)

    # 2. Evaluate firings: which profiles in S have their rule_path hold at (pid, ts)?
    firings = [p for p in S if _profile_fires_at(p, ts)]

    # 3. Enforce per-pid cap (max 1, carried from Phase 3)
    occupied_pids = {p.pid for p in open_positions}
    firings = [f for f in firings if f.pid not in occupied_pids]

    # 4. Enforce portfolio cap N
    available_slots = N - len(open_positions)
    if available_slots <= 0:
        slot_util_log.append((ts, len(open_positions)))
        continue

    # 5. Tiebreaker: highest cumulative_profit_deflated wins when slots < firings
    firings.sort(key=lambda f: -f.cumulative_profit_deflated)
    for profile in firings[:available_slots]:
        exit_ts  = ts + profile.horizon * 3_600_000
        expected = _lookup_label(profile.pid, ts, profile.horizon)
        open_positions.append((profile.pid, profile.profile_id, ts, exit_ts, expected))

    equity_curve.append((ts, equity + sum(p.expected_pnl for p in open_positions)))
    slot_util_log.append((ts, len(open_positions)))

return PortfolioMetrics(
    cumulative_profit_raw=equity,
    max_dd=_compute_max_dd(equity_curve),
    sortino=_compute_sortino(trade_log),
    trade_count=len(trade_log),
    pct_slots_full=sum(1 for _, n in slot_util_log if n == N) / len(slot_util_log),
    mean_concurrent=sum(n for _, n in slot_util_log) / len(slot_util_log),
)
```

**Key invariants:**
- A profile's rule path holds at `(pid, ts)` iff that pid's Phase 2 row at `ts` satisfies all feature thresholds in the rule. So `_profile_fires_at` re-evaluates the rule against live feature values.
- Exit PnL comes from Phase 2's `label_h{horizon}` at the entry row — Phase 4 does NOT re-simulate exits. This keeps Phase 4 honest: it inherits Phase 2's labels exactly, matching what Phase 3 used for ranking.
- Per-pid cap of 1 carried from Phase 3 — even if a profile fires on a different horizon for an already-open pid, skip.
- Tiebreaker key = `cumulative_profit_deflated` so the portfolio prioritizes profiles Phase 3 ranked highest after selection-bias correction.

### Beam-search knapsack

```
all_qualifying = top_100_by(profiles, key=cumulative_profit_deflated)
beam = [[]]                       # start with the empty subset
K_evaluated = 0

for step in 1..N:                 # build up subsets of size 1, 2, ..., N
    candidates = []
    for subset in beam:
        for profile in all_qualifying:
            if profile in subset:
                continue
            new_subset = subset + [profile]
            metrics = portfolio_sim(new_subset, cap=N)
            K_evaluated += 1
            candidates.append((new_subset, metrics))
    candidates.sort(key=lambda x: -x[1].cumulative_profit_deflated)
    beam = [s for s, _ in candidates[:BEAM_WIDTH]]

best_subset, best_metrics = max(beam_with_metrics, key=lambda x: x[1].cumulative_profit_deflated)
return best_subset, best_metrics, K_evaluated
```

**Defaults (locked):**
- `BEAM_WIDTH = 20`
- Pool = top-100 profiles by Phase 3 `cumulative_profit_deflated`
- K_evaluated upper bound for N=5: `5 × 20 × 100 = 10,000` portfolio sims per cap

### Portfolio-level deflation

```python
σ_portfolio = bootstrap_std(trade_log_pnls, n_iter=1000)
inflation   = σ_portfolio × √(2 × ln K_evaluated)
cumulative_profit_deflated = cumulative_profit_raw − inflation
```

Both raw and deflated written to the scorecard + deployment JSON. Portfolio Q0 gates apply to the deflated number.

### Portfolio Q0 gates

| Gate | Threshold | Rationale |
|---|---|---|
| `max_dd` | ≤ 0.30 | Inherits Q0 hard gate from the original brainstorm |
| `cumulative_profit_deflated` | > 0 | Minimum bar: strategy must make money after selection bias |
| `trade_count` | ≥ 50 | Statistical validity (≥ 10 trades per profile × N=5) |
| `sortino` | ≥ 0 | Non-degenerate risk-adjusted return |

A cap **passes** iff ALL four gates pass. Verdict logic:
- If multiple caps pass: pick highest `cumulative_profit_deflated`. Verdict = `"deploy at N={chosen}"`.
- If no caps pass: verdict = `"abort — no qualifying portfolio at any cap"`.

---

## Module Structure

```
backend/tools/strategy_discovery/
  profile_loader.py     (NEW)  Load Phase 3 parquets + sidecars + Phase 2 features
  portfolio_sim.py      (NEW)  Time-walk simulator: cap enforcement, equity curve, metrics
  knapsack_search.py    (NEW)  Beam search over subsets, calls portfolio_sim, returns best
  scorecard.py          (NEW)  Per-cap gate evaluation, Markdown rendering, verdict
  build_phase4.py       (NEW)  Orchestrator + CLI: sweep N∈{3,4,5}, write all artifacts
```

### `profile_loader.py` — Public API

```python
@dataclass
class LoadedProfile:
    pid: str
    horizon: int
    leaf_id: int
    rule_path: str                       # full string from sidecar JSON
    cumulative_profit_deflated: float    # the Phase 3 ranking metric (tiebreaker key)
    # plus all other columns from the Phase 3 profile parquet
    # (avg_win, avg_loss, max_dd, sortino, n_folds_passed_q0, ...)

def load_all_profiles(
    phase3_dir: Path = Path("backend/data/phase3"),
    horizons: List[int] = [1, 4, 24, 72, 168],
    min_folds_passed_q0: int = 4,
) -> List[LoadedProfile]
    """Loads all 5 horizon parquets + sidecars, drops profiles with
    n_folds_passed_q0 < min_folds_passed_q0 (re-enforces Phase 3 gate),
    returns a flat list."""

def load_pid_features(pid: str, phase2_dir: Path) -> pd.DataFrame
    """Returns Phase 2 parquet for the pid (used by portfolio_sim for rule evaluation)."""
```

### `portfolio_sim.py` — Public API

```python
@dataclass
class PortfolioMetrics:
    cumulative_profit_raw: float
    cumulative_profit_deflated: float = 0.0   # filled in by knapsack_search after K_evaluated known
    max_dd: float
    sortino: float
    trade_count: int
    pct_slots_full: float
    mean_concurrent: float

@dataclass
class TelemetryRow:
    ts: int
    equity: float
    n_open: int
    fired_profile_id: Optional[str]
    closed_profile_id: Optional[str]
    realized_pnl: Optional[float]

def simulate_portfolio(
    subset: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],   # pre-loaded per-pid Phase 2 frames
) -> Tuple[PortfolioMetrics, List[TelemetryRow]]
    """Walk historical bars in subset's union; enforce cap; return metrics + telemetry."""

def parse_rule_path(rule_path: str) -> List[Tuple[str, str, float]]
    """Parse a rule string like 'price_over_ema20 > 1.02 AND vol_over_mc <= 0.08'
    into a list of (feature, op, threshold) conditions."""
```

### `knapsack_search.py` — Public API

```python
@dataclass
class KnapsackResult:
    best_subset: List[LoadedProfile]
    best_metrics: PortfolioMetrics
    k_evaluated: int
    inflation: float
    beam_history: List[List[float]]   # per-step beam scores, for diagnostics

def beam_search_knapsack(
    all_qualifying: List[LoadedProfile],
    cap: int,
    pid_features: Dict[str, pd.DataFrame],
    beam_width: int = 20,
    pool_size: int = 100,
    bootstrap_iter: int = 1000,
    seed: int = 42,
) -> KnapsackResult
    """Top-pool_size profiles → beam search over subsets of size `cap` →
    return KnapsackResult with best subset + deflation applied."""
```

### `scorecard.py` — Public API

```python
@dataclass
class CapScorecard:
    cap: int
    metrics: PortfolioMetrics
    k_evaluated: int
    inflation: float
    gates: Dict[str, bool]               # 4 gate names → pass/fail
    overall_pass: bool
    selected_profiles: List[LoadedProfile]

def evaluate_cap_gates(metrics: PortfolioMetrics) -> Tuple[Dict[str, bool], bool]
    """Apply the 4 portfolio Q0 gates; return (gate_pass_dict, overall_pass_bool)."""

def render_scorecard(per_cap: List[CapScorecard]) -> str
    """Render Markdown report with per-cap sections + comparison table + verdict."""

def pick_verdict(per_cap: List[CapScorecard]) -> Tuple[Optional[int], str]
    """Returns (chosen_cap_or_None, verdict_string)."""
```

### `build_phase4.py` — Public API

```python
def build_phase4(
    *,
    phase3_dir: Path = Path("backend/data/phase3"),
    phase2_dir: Path = Path("backend/data/phase2"),
    output_dir: Path = Path("backend/data/phase4"),
    caps: List[int] = [3, 4, 5],
    beam_width: int = 20,
    pool_size: int = 100,
    seed: int = 42,
) -> Dict[int, CapScorecard]
    """Sweep caps, write scorecard.md + deployment_n{N}.json + portfolio_telemetry_n{N}.parquet."""

def main(argv: Optional[List[str]] = None) -> int
    """CLI entrypoint."""
```

Module boundary rationale: `profile_loader` is pure I/O; `portfolio_sim` is the only module that simulates time; `knapsack_search` only depends on `portfolio_sim`; `scorecard` only consumes search results; `build_phase4` composes them and writes to disk.

---

## Testing

Mock-only per CLAUDE.md (no live GPU required — Phase 4 is pure pandas + numpy; no torch).

| Test file | Test | What it pins |
|---|---|---|
| `test_profile_loader.py` | `test_loads_all_horizon_parquets` | Loads + concatenates 5 horizon files |
| | `test_attaches_rule_paths_from_sidecar_json` | leaf_uid → rule_path lookup |
| | `test_filters_profiles_below_min_folds_passed` | Drops profiles with `n_folds_passed_q0 < 4` |
| `test_portfolio_sim.py` | `test_concurrency_cap_blocks_new_entries_when_full` | At cap=N, no entry after N open |
| | `test_max_1_position_per_pid_carried_over` | Phase 3 rule still holds at portfolio level |
| | `test_simultaneous_fires_resolved_by_deflated_profit_tiebreaker` | Higher-deflated-profit wins |
| | `test_exit_pnl_read_from_phase2_label` | Inherits Phase 2 labels exactly |
| | `test_max_dd_computed_on_equity_curve` | Drawdown math correct |
| | `test_slot_utilization_telemetry` | % time full, mean concurrent reported |
| `test_knapsack_search.py` | `test_beam_search_finds_known_optimal_on_toy_3_profile_pool` | Tiny pool (3 profiles, cap=2) — beam matches exhaustive |
| | `test_returns_k_evaluated_for_deflation` | Counter incremented per portfolio_sim call |
| | `test_beam_width_caps_branching` | beam_width=5 → ≤5 subsets per step |
| `test_scorecard.py` | `test_portfolio_gates_max_dd_30` | DD > 30% → cap fails |
| | `test_portfolio_gates_deflated_profit_positive` | Deflated ≤ 0 → cap fails |
| | `test_portfolio_gates_trade_count_50` | < 50 trades → cap fails |
| | `test_verdict_picks_highest_deflated_passing_cap` | When 4 + 5 pass, picks higher |
| | `test_verdict_abort_when_all_caps_fail` | All-fail → abort verdict |
| `test_build_phase4.py` | `test_sweeps_all_three_caps_writes_three_deployments` | n=3,4,5 → 3 JSONs |
| | `test_writes_scorecard_md_and_telemetry_parquet` | All artifact paths produced |
| | `test_main_returns_zero_on_at_least_one_passing_cap` | CLI exit code semantics |

**Total new test surface:** 19 tests, mock-only.

---

## Operator Integration (post-implementation)

After Phase 4 code lands and tests pass:

```
cd backend && python -m tools.strategy_discovery.build_phase4 \
    --phase3-dir data/phase3 --phase2-dir data/phase2 \
    --output-dir data/phase4 --seed 42
```

Expected runtime: ~60–90 min on CPU (3 caps × ~10k portfolio sims each). No GPU required (Phase 4 is pure pandas/numpy — the GPU was Phase 3's mining loop).

Operator reads `backend/data/phase4/scorecard.md`, sees the per-cap pass/fail + verdict. If verdict = `"deploy at N=X"`:

1. Inspect `deployment_n{X}.json` — confirm the selected profile set makes sense (no degenerate pids, no rule paths that look like noise).
2. Open a separate integration commit to wire `deployment_n{X}.json` into `agents/cnn_agent.py` (load the rules, check firings at each scan-loop tick, emit BUY signals).
3. Validate in shadow on port 8002 for the appropriate window (per `feedback_backend_port_isolation`).
4. Promote to 8001 if shadow telemetry beats v3.

If verdict = `"abort"`: investigate which gate failed; consider rerunning Phase 1-3 with adjusted parameters (e.g., wider universe, longer history, different horizon set), then re-run Phase 4.

---

## What Phase 4 is NOT

- Not the runtime inference path into `agents/cnn_agent.py` (separate integration commit once a deployment is picked)
- Not live shadow telemetry (deployment validation step, not selection step)
- Not cross-deployment ensembling (combining N=3 + N=5 portfolios)
- Not re-fitting profiles on newer data (that's a Phase 1-3 re-run, not Phase 4)
- Not held-out fold validation (deferred to backlog)

---

## Backlog (deferred)

| ID | Item | Trigger to revisit |
|---|---|---|
| (new) | Held-out fold validation of the selection (Phase 3 re-run with 6 folds, fold 6 = Phase 4 validation only) | If deflated portfolio profit looks suspicious vs raw or if first shadow week's realized profit diverges from expected |
| (new) | Hybrid tiebreaker (profile rotation, not always highest-deflated) | If telemetry shows one profile starves the others (e.g. one profile takes 80% of trades) |
| (new) | Walk-forward portfolio refit (re-pick deployment every M weeks) | After live trades validate the v1 picks |
| (new) | Per-pid weight optimization (not uniform 1/N capital per slot) | If post-deploy data shows pid-specific edge differences |
| (new) | Cross-deployment ensembling (e.g. 50% capital on N=3 portfolio + 50% on N=5) | If two caps both pass and operator wants to diversify |

---

## See also

- `2026-05-23-strategy-discovery-rebuild-brainstorm.md` — original Phase 1–4 spec (Q0–Q6 decisions)
- `2026-05-23-strategy-discovery-phase2-design.md` — Phase 2 spec (labels Phase 4 inherits as exits)
- `2026-05-24-strategy-discovery-phase3-design.md` — Phase 3 spec (profile parquets Phase 4 consumes)
- `2026-05-24-strategy-discovery-phase3-implementation.md` — Phase 3 plan (Phase 4 mirrors structure)
- `backend/tools/phase3_deflation_explainer.html` — companion walkthrough of the deflation math (same formula, applied at portfolio level here)
- `coinbase_trader_architecture.md` (memory) — current backend architecture + strategy_discovery package overview
- v3 scorecard at `2026-05-18-xgb-scorecard-baseline-results.md` — methodological ancestor of the Phase 4 scorecard format
