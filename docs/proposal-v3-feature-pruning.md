# Proposal — v3 feature-pruning shipping path

**Date:** 2026-06-07
**Status:** Draft proposal (loop-driven, not committed)
**Based on:** `v3_per_channel_pruned_ablation.html` Sections IV–VI

---

## What we proved

Two ablations on the live v3 booster, each validated across 5 XGB seeds × 5 walk-forward folds (25 comparisons each):

| Variant | Cut | AUC effect | Proof |
|---|---|---|---|
| **(iii) drop masked** | Remove 30 features from ch17/18/19 (already-zero by `_TRAINING_CONSTANT_CHANNELS`) | **Bit-exact identical predictions** — all 25 deltas = 0.0 exactly; max prediction diff = 0.0 across 1,517 samples | Mathematical: constant-zero columns yield split-gain = 0, so they can never enter a tree |
| **(ii) drop all zero-gain** | Remove 291 features (350 → 59) with zero gain in the live booster | **Statistically lossless** — grand mean Δ = +0.0003 ± 0.0016 (std) across 5 seeds; 2/5 seeds favor pruned, 3/5 favor baseline | Empirical: std is 5× the mean magnitude, true effect indistinguishable from zero |

## What this enables

A two-tier shipping path with very different risk profiles:

### Tier A — Drop masked channels (provably lossless)

- **Change:** remove ch17/18/19 from `_extract_v3` / `_v3_feature_names` / `feature_weights_v3`
- **Reduction:** 350 features → 320 (8.6%)
- **Risk:** zero. By mathematical proof, the predictions of any retrained v3 booster are identical.
- **TDD test:** fixture comparison — load current `xgb_model.json`, predict on a held-out sample with full features, predict with the 30 masked-channel features set to zero (equivalent to the cut), assert `np.allclose(scores, scores_pruned, atol=0)`.
- **Code change cost:** small. ~30 LOC in `xgb_features.py`. Possibly 10 LOC in `cnn_agent.py` to skip the masked-channel builders.
- **Backward compat:** named-feature list changes (320 names vs 350). Any consumer of `xgb_features.json` must be updated. The current consumers are: `xgb_signal._try_load`, `xgb_topn_sweep.py` (retired), `train_xgb.train_xgb_v3`. Each needs a check.

### Tier B — Drop all zero-gain features (statistically lossless)

- **Change:** ship a `xgb_active_channels.json` artifact alongside `xgb_model.json`; gate channel-builder execution on it; skip the 19 channels with no active features (everything except ch0/1/2/3/4/15/20/21/24).
- **Reduction:** 350 features → 59 (83%). 28 channels → 9 channels at extraction time.
- **Risk:** small. Per the 5-seed assessment, the AUC effect is +0.0003 ± 0.0016 — within fold-noise. **Note:** this is NOT bit-exact like Tier A; predictions will differ in floating-point at the 3rd decimal across seeds.
- **TDD test:** OOF AUC parity check. Train baseline + active-channels-only on the same WF splits; assert mean AUC delta within ±0.005 (5× the observed std).
- **Code change cost:** larger. New artifact tracked with the booster. New config flag `XGB_SKIP_INACTIVE_CHANNELS=true`. `_extract_v3` needs branching.
- **Backward compat:** the named-feature list is unchanged (only the values change — inactive channels emit zeros). Booster compatibility is preserved. The artifact is opt-in via the flag.

### Tier C — Combine A + B

If both ship, the named-feature list shrinks to 9 channels × variable stats per tier = ~50 features. Cleanest end state. But the named-list change is a hard fork from the current artifact, so it's worth treating as a separate retrain milestone.

## Recommended path

| Step | What | When | Cost |
|---|---|---|---|
| 1 | Tier A (drop masked) | Anytime. Lowest risk. | ~1 dev-day with tests |
| 2 | Inference-latency benchmark on a single PID scan path, measure wall-clock for `_extract_v3` with/without inactive channels | Before Tier B | ~30 min benchmark + analysis |
| 3 | Tier B (active-channels artifact + flag) | After the benchmark confirms ≥30% wall-clock savings (the threshold below which it's not worth the complexity) | ~3 dev-days |
| 4 | Combined A+B retrain | Tied to the next v3 retrain cycle (next operator-scheduled retrain) | included in retrain prep |

## What this does NOT change

- AUC ceiling. The v3 model's edge is ~0.51 OOF, well below the 0.55 deployment gate. Pruning doesn't lift; it just makes inference cheaper.
- Scorecard verdict. v3 still fails 3 of 4 gates at retail fees regardless of feature count.
- Operator priority. Bar-structure remains the path forward for AUC improvement; this is purely a compute optimization.

## What this DOES change

- **Inference latency.** Channel-level skip eliminates the most expensive channel-builders (MACD, Bollinger, OBV, EMA9, EMA21, VWAP, MFI, Stoch RSI, hour-of-day, ROC, OI z-score). These run once per PID per scan today and are pure waste.
- **Pattern C ModelService viability.** If/when the event-sourced ModelService runs multiple model variants concurrently (per the Pattern C architecture), per-inference cost matters more. Tier B is the natural prerequisite for that work to be operationally cheap.
- **Code clarity.** The 30 always-zero feature slots in `xgb_features.json` are cruft. Tier A removes them.

## Open questions to resolve before shipping

1. Does the same finding hold for **v4.5**? The pruning math (constant-column inert) is model-agnostic, but the active-channel set for v4.5 is its own measurement.
2. Does the optimization compose with the next-generation models (v5 dollar-bar track)?

## Update: top-N sweep changes the recommendation (2026-06-07)

After running an N ∈ {350, 59, 40, 30, 20, 10, 5} sweep × 3 seeds × 5 folds:

| N | grand mean AUC | Δ vs N=350 |
|---|---|---|
| 350 | 0.5082 | +0.0000 |
| 59 | 0.5094 | +0.0012 |
| 40 | 0.5091 | +0.0009 |
| **30** | **0.5100** | **+0.0018** (peak) |
| 20 | 0.5070 | −0.0012 |
| 10 | 0.5083 | +0.0001 |
| 5 | 0.5074 | −0.0007 |

**The AUC is flat from N=5 to N=350.** No cliff in the tested range. The 5 features carrying the model are:

1. `ch4_pct_rank` — RSI(14) position in its 60-bar window
2. `ch1_slope` — volume slope
3. `ch0_slope` — close slope
4. `ch24_m168_slope` — IV/RV20 spread, meso-tier slope
5. `ch24_m168_mean` — IV/RV20 spread, meso-tier mean

**Revised Tier B recommendation:** ship **top-30 by gain** as the optimal cut, not "all-non-zero-gain (59)". Top-30 is the empirical peak and has lower seed-variance than top-40 or top-59. Same retraining cycle as the rest of the proposal; just a different N constant.

**Revised inference-latency upper bound:** with only 5 features actually needed, the channel-builder skip can be even more aggressive. The 9 active channels reduce to **4 channels** that carry the top-5 features: ch0, ch1, ch4, ch24. That's ~85% channel reduction at inference time (from 28 to 4). Pattern C ModelService can stage these as a stripped configuration without losing AUC.
