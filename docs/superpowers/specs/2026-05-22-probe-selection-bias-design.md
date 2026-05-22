# Probe Selection-Bias Meta-Analysis — Design Spec

**Date:** 2026-05-22
**Status:** Draft — pending user review
**Scope:** polymarket_app — post-scorecard XGB roadmap task #16

## Problem

The XGB feature search has run a large number of trials — ~17 single-add channel candidates, an 81-config hyperparameter grid, 5 label horizons, two feature sets, a top-N retention curve — and reports a best of AUC ≈ 0.5284 (top-40, tuned) with one confirmed channel lift (RSI-rank, Δ ≈ +0.0124 survivorship-aware). After that many trials, the apparent best is partly selection noise: the more configurations you try, the higher the best result a *true-null* search would still produce. This diagnostic deflates the recorded results for the size of the search and returns a go/no-go verdict: **is the XGB feature-search signal distinguishable from best-of-N noise, or is further feature work on this cache a dead end?**

It is the gate task for the post-scorecard roadmap — the 6 downstream paths (#17–#24, excluding the now-shipped bar-structure path) are worth pursuing only if there is a real edge to refine.

## Decisions locked (from brainstorming, 2026-05-22)

| Decision | Choice | Rationale |
|---|---|---|
| Scope | Meta-analysis of the recorded probe log only | The probe log records per-trial AUC/Δ, not return series or per-strategy fold matrices, so a full Bailey CSCV/PBO is infeasible without re-running every probe. A Deflated-Sharpe-style multiple-testing analysis works directly from the recorded numbers. PBO/CSCV is left to roadmap task #17. |
| Primary noise scale | Empirical fold-level SE | The purged-WF per-fold AUC spread (logged baseline folds `[0.516, 0.507, 0.527, 0.523, 0.529]`, SE ≈ 0.0086) is the honest noise unit. Triple-barrier 4h-horizon labels overlap ~75% with neighbours, so the effective sample size is far below 167k. |
| Contrast noise scale | iid DeLong/Mann-Whitney SE | Reported alongside (SE ≈ 0.0014 at 167k samples) purely to demonstrate why naive significance testing overstates the edge — it is the textbook mistake, not the verdict basis. |
| Trial count N | Tiered: N ≈ 17 / 100 / 200 | N is a judgment call (channel candidates only vs. + the 81-config grid + 5 horizons vs. conservative). Report the verdict at all three; headline on the full-search tier (~100). |

## The two deflation tracks

The diagnostic deflates two distinct claims, each with the same machinery but different inputs:

**Track A — v3's base AUC edge.** Observed: the best documented configuration's AUC (top-40 tuned, ≈ 0.5284) and the scorecard's v3 OOF AUC (≈ 0.512), against the null AUC = 0.50. N = the full search that produced the deployed v3 (channel set + 81-config grid + horizons). Question: is the *base ranking edge* above the best-of-N noise floor?

**Track B — the best channel-add lift.** Observed: the best probe Δ (RSI-rank, ≈ +0.0124), against the null Δ = 0. N = the ~17 single-add channel candidates. Noise scale: the empirical fold-level AUC SE (≈ 0.0086) used directly as the Δ noise unit. The probe log does not retain per-fold Δ values, so a true paired-difference SE cannot be computed; a paired Δ would have *lower* variance (baseline and replaced share folds and most channels), so using the full AUC fold-SE is the conservative substitute — it overstates Δ noise, making the "is it real" test harder to pass, not easier. Question: is the *one confirmed lift* real, or the max of 17 noise draws?

## Components

One module — `backend/tools/probe_selection_bias.py`:

- **Trial table** — the ~25 named trials transcribed from `xgb_probe_results_log.md` into a structured Python literal: each row `(name, baseline_auc, achieved_auc, delta, passed)`. The marketcap probe is excluded (null-coverage — never a real trial) and noted. The 81-config hyperparameter grid is not individually logged; it enters only as a count contributing to N. The transcription and the N-counting judgment are documented inline in the spec/module.
- **Noise model** — `fold_level_se(fold_aucs)` (the empirical SE, primary) and `iid_auc_se(n_pos, n_neg)` (the DeLong SE, contrast).
- **Deflation math** (pure, unit-tested functions):
  - `expected_max_under_null(n_trials, se, center)` — `center + se · E[max of N standard normals]`, using Bailey & López de Prado's expected-max approximation `E[Z_(N)] ≈ (1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))` (γ = Euler-Mascheroni).
  - `deflated_probability(observed, n_trials, se, center)` — a DSR-style `Φ((observed − E[max_under_null]) / se)`: the probability the observed edge exceeds the selection-noise floor.
- **Orchestration / CLI** — runs Track A and Track B across the three N tiers, prints and writes the report.

## Output

`docs/superpowers/specs/2026-05-22-probe-selection-bias-results.md`, written by the CLI:

- The transcribed trial table.
- The noise model: empirical fold SE vs. iid DeLong SE, side by side.
- For each track × N tier: `expected_max_under_null`, the observed value, and `deflated_probability`.
- A go/no-go verdict: under the honest (fold-level) noise scale and the full-search N, is the feature-search edge distinguishable from noise? An explicit recommendation on whether roadmap paths #17–#24 are worth pursuing.

## Testing

- The deflation math functions are pure and unit-tested: `expected_max_under_null` against hand-computed values for small N; `deflated_probability` bounds (→1 when observed ≫ floor, → ~0.5 when observed = floor); `fold_level_se` and `iid_auc_se` against known inputs.
- The trial table and the report rendering are tested for shape/structure (row count, required keys, the verdict line present).
- No live model runs — this is a pure computation over recorded numbers. TDD red-green-refactor.

## Out of scope

- Full Bailey CSCV / PBO — needs per-strategy fold matrices the probe log does not retain; left to roadmap task #17 (CPCV).
- Re-running any probe or re-training any model — the analysis consumes only documented results.
- Re-deriving the probe AUCs — the recorded numbers in `xgb_probe_results_log.md` are taken as given.
- Acting on the verdict — whether to drop roadmap paths is a follow-up decision the verdict informs, not part of this task.

## See also

- `xgb_probe_results_log.md` (memory) — the verbatim probe history this analysis transcribes
- `xgb_feature_optimization_findings.md` (memory) — the AUC-ceiling summary (top-40 tuned 0.5284)
- `xgb_post_scorecard_roadmap.md` (memory) — the 8-path roadmap; #16 is the gate before #17–#24
- `2026-05-18-xgb-scorecard-baseline-results.md` — the v3 deployment verdict (1 of 4 gates) this complements
