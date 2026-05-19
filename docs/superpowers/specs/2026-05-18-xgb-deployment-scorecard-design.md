# XGB Deployment-Aligned Scorecard — Design Spec

**Date:** 2026-05-18
**Status:** Draft — pending user review
**Scope:** polymarket_app XGB v3 driver + v4 + v4.5 shadow tracks

## Problem

Current promotion gate for XGB cutover is **AUC ≥ 0.55** on 5-fold purged-WF CV. Two issues:

1. **AUC measures wrong thing.** AUC ranks pairs across the entire score distribution; trading fires only at the gate threshold. A model with AUC 0.52 but tight precision at gate can outperform a model with AUC 0.55 but flat calibration.
2. **AUC ceiling has held at 0.5284 across 7+ exogenous probes** (see `xgb_probe_results_log.md`). Forcing the model to lift AUC by another 0.022 may not be reachable on price/orderflow features alone, but a deployment-aligned target may already be met or near-met.

Literature survey (4 parallel agents, 2026-05-18) confirms: serious crypto-XGB practitioners measure precision-at-gate, expected return per signal, paper-Sharpe, and calibration — not AUC alone. Sources: Salinas et al. 2025 (`Financial Innovation`), Hudson & Thames meta-labeling work, Lopez de Prado AFML Ch. 14, FinTSB 2025 benchmark.

## The Scorecard

Four metrics tracked per model variant per val fold. No single primary — each has its own promotion gate.

### 1. Precision-at-gate

**Formula:** `P_τ = TP_τ / (TP_τ + FP_τ)`, where TP_τ = samples with score > τ AND label = 1 (UP barrier hit).

**Threshold:** sweep τ ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}. Report per-τ table; flag τ giving max P_τ with at least 100 fired signals on val fold.

**Promotion gate:** at chosen operating τ, `P_τ ≥ pos_rate + 0.03` (must beat naive base rate by ≥3 percentage points).

### 2. Expected return per signal

**Formula:** `E[r_τ] = mean(r_i for i where score_i > τ) − 2 × fee`, where `r_i` is the realized log-return at the resolution of the triple-barrier (TP / SL / timeout) for sample i, **NOT** the binary label.

**Fee model:** Three tiers reported side-by-side per metric column:

| Tier label | Per-side fee | Round-trip deduction |
|---|---|---|
| `retail` | 0.6% | 0.012 |
| `mid` | 0.25% | 0.005 |
| `pro` | 0.05% | 0.001 |

Per-tier E[r] reported as separate columns in the per-τ table. Promotion gates evaluated against `retail` by default (conservative); `mid` and `pro` are informational.

**Position size:** fixed $1 per signal. Decouples scoring quality from sizing policy. See "Out of scope" below.

**Threshold:** same τ sweep as #1. Report E[r_τ] per τ.

**Promotion gate:** `E[r_τ] > 0` at chosen operating τ (must be net-positive after fees on val fold).

### 3. Paper-trade Sharpe

**Formula:** `Sharpe_τ = annualize(μ_r / σ_r)` over fired signals only, where `μ_r`, `σ_r` are mean/std of `r_i − 2 × fee` for samples with score_i > τ.

**Annualization:** **per-fold then averaged** (matches existing AUC fold-reporting convention). For each of the 5 purged-WF folds f:

1. Compute fold f's per-signal Sharpe: `S_f = mean(r_f) / std(r_f)` over fired signals in fold f.
2. Compute fold f's annualization factor: `N_f = F_f × 365 / T_f`, where `F_f` = signals fired in fold f and `T_f` = fold span in days.
3. Annual Sharpe per fold: `S_f_annual = S_f × √N_f`.

Report `mean(S_f_annual) ± std(S_f_annual)` across folds. Gives an honest variance estimate (e.g., "Sharpe 0.7 ± 0.4 across folds" warns of regime-dependent stability).

**Promotion gate:** TBD. Set after baseline measurement to avoid pre-committing to a number without knowing what's achievable on current cache. Initial target: `Sharpe_annual > 0` (mandatory) and `> 0.5` (stretch).

### 4. Expected Calibration Error (ECE)

**Formula:** `ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|`, where bins `B_b` are score deciles, `acc(B_b)` = empirical hit rate in bin, `conf(B_b)` = mean score in bin.

**Threshold:** none — distribution-wide metric.

**Promotion gate:** `ECE < 0.05` (standard "well-calibrated" cutoff in calibration literature).

## Val fold convention

Use **aggregated out-of-fold predictions across all 5 purged-WF folds** (5-fold CV with 4h embargo, same as existing probes). Each sample appears exactly once in OOF; total ≈ 167k samples on current 28-channel cache v12 with survivorship-aware top-20.

This matches how `tools/feature_set_compare.py`, `tools/rsi_rank_probe.py`, and existing probes evaluate AUC. Preserves comparability with the probe-results log.

## Scope: tracks measured

Three parallel scorecards, one per active XGB head:

| Track | Output | Decision rule | Used for |
|---|---|---|---|
| **v3 driver** | binary `xgb_prob` ∈ [0.01, 0.99] | `BUY if > 0.55; SELL if < 0.45` | live (current production) |
| **v4 shadow** | binary `xgb_prob_v4` | same | shadow telemetry, B.1 horizon-sweep |
| **v4.5 shadow** | 3-class softmax `(p_down, p_neutral, p_up)` × 3 horizons (h24/h72/h168) × 3 decision rules (argmax_margin/indep_thresholds/net_direction) | per decision rule | shadow telemetry, Session 58.71k onward |

For v4.5, scorecard computes **9 cells expanded** (3 horizons × 3 decision rules), reported as a 3×3 table per metric. No collapse — full grid stays visible so cross-horizon and cross-rule effects are inspectable. Use `score = p_up` for metrics #1–#3 (precision/return/Sharpe operate on UP barrier hits); use full softmax for ECE.

**v4.5 SELL-side measurement is deferred to v2 of the scorecard** (per O5 resolution). v1 measures LONG signals only across all three tracks. SELL-side scorecard will be specified in a follow-up design once the SELL-side CLI tooling is in place.

## Computation pipeline

New file: `backend/tools/scorecard.py` (with TDD test file `backend/tests/test_scorecard.py`, per `feedback_tdd_workflow`).

Public surface:

```python
FEE_TIERS: Mapping[str, float] = {"retail": 0.006, "mid": 0.0025, "pro": 0.0005}

def compute_scorecard(
    scores: np.ndarray,           # (N,) model output for the score-class
    labels: np.ndarray,           # (N,) binary 0/1 = UP-barrier-hit
    returns: np.ndarray,          # (N,) realized log-return per sample
    fold_ids: np.ndarray,         # (N,) integer fold index 0..4 for per-fold Sharpe annualization
    fold_spans_days: Mapping[int, float],  # fold_id -> span in days (for N_f)
    *,
    fee_tiers: Mapping[str, float] = FEE_TIERS,
    gate_tier: str = "retail",    # which tier evaluates hard gates
    tau_grid: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    n_ece_bins: int = 10,
) -> ScorecardReport:
    ...
```

`ScorecardReport` is a dataclass with: per-τ × per-tier table (precision, E[r] per tier, Sharpe mean ± std per tier, n_fired), ECE scalar, recommended_operating_tau (chosen on `gate_tier`), per-metric `gate_passed` bool (evaluated on `gate_tier`).

CLI: `python -m tools.scorecard --track v3 --cache cnn_dataset_cache.pt` runs the OOF aggregation and prints the report.

Reuses existing `_compute_realized_return_per_sample` helper (currently lives in `tools/feature_set_compare.py:_realized_returns`; if not extracted, refactor as part of this spec per `feedback_loose_coupling`).

## Out of scope (explicit non-goals)

- **Sizing policy** (Kelly, vol-target, risk-parity, prob-weighted). Scorecard uses fixed $1/signal. Sizing is a *separate downstream design* that consumes a calibrated model.
- **Maker-rebate modeling.** Three flat tiers reported (retail / mid / pro) per O1; rebate-tracking for maker fills not modeled.
- **Slippage modeling.** Assume fills at the close of the entry bar. Slippage estimation requires L2 data we don't ingest.
- **Replacing the AUC gate immediately.** This spec *adds* a scorecard to track in parallel; whether to retire the 0.55 AUC gate is a follow-up decision based on the scorecard's first baseline run.
- **Meta-labeling.** AFML survey (Agent 4) flagged meta-labeling won't help at primary AUC ≤ 0.53. Revisit if/when scorecard shows the primary clears its own gates.

## Per-metric promotion gates (summary)

| Metric | Gate | Hard or stretch |
|---|---|---|
| Precision-at-gate | `P_τ ≥ pos_rate + 0.03` at operating τ with ≥100 signals | hard |
| Expected return | `E[r_τ] > 0` after fees | hard |
| Paper-Sharpe (annualized) | `> 0` (hard), `> 0.5` (stretch) | hard / stretch |
| ECE | `< 0.05` | hard |

Promotion: model variant passes the scorecard if **all 4 hard gates** met. Stretch gates inform ranking among multiple passing variants.

## Open questions — resolved 2026-05-18

- **O1 → resolved:** Report all three fee tiers (retail/mid/pro) side-by-side. Retail used for hard-gate evaluation; mid/pro informational. See "Expected return per signal" section above for tier table.
- **O2 → resolved:** v4.5's 9-cell grid stays expanded (3 horizons × 3 decision rules per metric). No collapse.
- **O3 → resolved:** Per-fold annualization, then mean ± std across folds. Matches existing AUC fold-reporting convention. See "Paper-trade Sharpe" section above.
- **O4 → resolved:** Decile binning at 167k. Adaptive binning only revisited if scorecard ever runs on smaller subsets (per-product, per-regime).
- **O5 → resolved (v1 deferral):** v1 scorecard measures LONG signals only across all three tracks. SELL-side measurement deferred to v2 scorecard spec, contingent on user's separate CLI work that will produce SELL-side tooling. v2 will likely follow Option B (signed score `p_up − p_down`) but final choice pending the upstream tooling.

## Implementation steps (for writing-plans phase)

1. Extract `_realized_returns` helper into `backend/tools/_returns.py` (per loose-coupling rule).
2. Write `backend/tests/test_scorecard.py` first — TDD red. Test cases: (a) trivial precision computation on synthetic, (b) E[r] with known fee deduction, (c) Sharpe annualization, (d) ECE on synthetic well/poorly calibrated arrays, (e) end-to-end on a slice of cache.
3. Implement `backend/tools/scorecard.py` to green.
4. CLI integration: `python -m tools.scorecard --track {v3,v4,v4.5}`.
5. Run baseline scorecard on all 3 active tracks. Persist report as `docs/superpowers/specs/2026-05-18-xgb-scorecard-baseline-results.md`.
6. Compare baselines, decide gate-retirement question (whether AUC 0.55 is dropped or kept alongside).

## See also

- `xgb_feature_optimization_findings.md` — AUC 0.5284 ceiling that motivated this spec
- `xgb_probe_results_log.md` — 7+ failed exogenous probes confirming data-limited regime
- `coinbase_trader_architecture.md` — XGB v3/v4/v4.5 pipeline state
- `backlog_xgb_v5_three_class.md` — v4.5 shipped 2026-05-18, this scorecard will evaluate it
