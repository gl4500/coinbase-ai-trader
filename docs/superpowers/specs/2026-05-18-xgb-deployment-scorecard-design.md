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

**Fee model:** Coinbase Advanced Trade retail tier — **0.6% per side (1.2% round-trip)**, taken as `2 × fee = 0.012` deduction per signal. Single-tier in v1 of the scorecard; multi-tier extension is open question O1.

**Position size:** fixed $1 per signal. Decouples scoring quality from sizing policy. See "Out of scope" below.

**Threshold:** same τ sweep as #1. Report E[r_τ] per τ.

**Promotion gate:** `E[r_τ] > 0` at chosen operating τ (must be net-positive after fees on val fold).

### 3. Paper-trade Sharpe

**Formula:** `Sharpe_τ = annualize(μ_r / σ_r)` over fired signals only, where `μ_r`, `σ_r` are mean/std of `r_i − 2 × fee` for samples with score_i > τ.

**Annualization:** `√N` where `N` = signals fired per year on val fold (true sample frequency, NOT bars-per-year). If val fold spans T days and fires F signals, `N = F × 365 / T` and `Sharpe_annual = Sharpe_per_signal × √N`.

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

For v4.5, scorecard computes **9 cells per horizon** (3 horizons × 3 decision rules). Use `score = p_up` for metrics #1–#3 (precision/return/Sharpe operate on UP barrier hits); use full softmax for ECE.

## Computation pipeline

New file: `backend/tools/scorecard.py` (with TDD test file `backend/tests/test_scorecard.py`, per `feedback_tdd_workflow`).

Public surface:

```python
def compute_scorecard(
    scores: np.ndarray,           # (N,) model output for the score-class
    labels: np.ndarray,           # (N,) binary 0/1 = UP-barrier-hit
    returns: np.ndarray,          # (N,) realized log-return per sample
    *,
    fee: float = 0.006,           # one-side fee, default 0.6% Coinbase retail
    tau_grid: Sequence[float] = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    n_ece_bins: int = 10,
) -> ScorecardReport:
    ...
```

`ScorecardReport` is a dataclass with: per-τ table (precision, E[r], Sharpe, n_fired), ECE scalar, recommended_operating_tau, per-metric gate_passed bool.

CLI: `python -m tools.scorecard --track v3 --cache cnn_dataset_cache.pt` runs the OOF aggregation and prints the report.

Reuses existing `_compute_realized_return_per_sample` helper (currently lives in `tools/feature_set_compare.py:_realized_returns`; if not extracted, refactor as part of this spec per `feedback_loose_coupling`).

## Out of scope (explicit non-goals)

- **Sizing policy** (Kelly, vol-target, risk-parity, prob-weighted). Scorecard uses fixed $1/signal. Sizing is a *separate downstream design* that consumes a calibrated model.
- **Multi-fee-tier comparison.** v1 ships single retail tier (0.6%); side-by-side multi-tier is open question O1.
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

## Open questions

- **O1:** Should the scorecard report all three fee tiers (retail/mid/pro) side-by-side instead of just retail? Defer to user review.
- **O2:** Should v4.5's 9-cell sweep collapse to a single representative cell for cross-track comparison, or stay expanded? Defer.
- **O3:** Paper-Sharpe annualization formula assumes signal frequency stable across folds. If frequency varies wildly per fold (e.g., regime-dependent), need to choose between per-fold annualization or pooled. Decide after first run.
- **O4:** ECE binning by decile assumes enough mass per bin. With binary labels and 167k samples this is safe, but if scorecard runs on smaller subsets (per-product, per-regime) switch to adaptive binning (Bayesian Binning into Quantiles).
- **O5:** For v4.5 (3-class), the spec as drafted only scores LONG signals (`score = p_up` → did UP barrier hit?). SELL signals (`score = p_down` → did DOWN barrier hit?) are not measured — that omits half the model's action. Option A: double the cell count to 18 (9 UP × 9 DOWN), report each side separately. Option B: aggregate per-side metrics into a single signed score using `score = p_up − p_down`. Option C: stay UP-only in v1, ship SELL-side scorecard as a follow-up. Defer to user review.

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
